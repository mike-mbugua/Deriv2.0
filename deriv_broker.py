"""
deriv_broker.py
═══════════════════════════════════════════════════════════════════
ExecutionBroker implementation for Deriv.

Two execution backends:
  DerivWebSocketBroker  — places trades via Deriv JSON WebSocket API
  DerivMT5Broker        — places trades via MT5 (recommended)

DerivMT5Broker extends MT5Broker from execution_broker.py and adds:
  • Deriv-specific symbol resolution (synthetics + frx forex)
  • Adjusted lot sizing (synthetics use different contract specs)
  • Price precision rounding for synthetics
  • Account info via MT5

DerivWebSocketBroker is a lighter option if you do not have MT5.
It uses Deriv's "buy" / "sell" contract API for CFD positions.
Note: Deriv's WebSocket trade API works differently from MT5 —
it uses "multiplier contracts" rather than raw CFD positions.
For a straightforward MT4/MT5-like experience, use DerivMT5Broker.

Usage in main.py
─────────────────
  from deriv_broker import DerivMT5Broker, DerivWebSocketBroker

  # Recommended: MT5 broker
  broker = DerivMT5Broker(
      login    = int(os.getenv("MT5_LOGIN")),
      password = os.getenv("MT5_PASSWORD"),
      server   = os.getenv("MT5_SERVER"),   # e.g. "Deriv-Demo"
  )
  broker.connect()

  # Alternative: WebSocket broker (no MT5 required)
  broker = DerivWebSocketBroker(api_token=os.getenv("DERIV_API_TOKEN"))

  # Pass to PaperTrader exactly as before:
  trader = PaperTrader(..., broker=broker)
"""

import logging
import os
from typing import Optional

from execution_broker import ExecutionBroker, TradeResult
from deriv_config import DERIV_PRICE_PRECISION, DERIV_MT5_SYMBOL_MAP
from deriv_feed import DerivWebSocketClient

logger = logging.getLogger(__name__)


# ================================================================== #
#  Helper: price rounding for synthetics                             #
# ================================================================== #

def _round_deriv_price(instrument: str, price: float) -> float:
    decimals = DERIV_PRICE_PRECISION.get(instrument, 3)
    return round(price, decimals)


# ================================================================== #
#  MT5 symbol map for Deriv                                          #
# ================================================================== #
# This is the same as DERIV_MT5_SYMBOL_MAP in deriv_feed.py.
# Duplicated here so deriv_broker.py has no circular import.

_MT5_SYMBOL_MAP = DERIV_MT5_SYMBOL_MAP


# ================================================================== #
#  DerivMT5Broker  (recommended)                                     #
# ================================================================== #

class DerivMT5Broker(ExecutionBroker):
    """
    Sends orders to Deriv via MetaTrader 5.

    Inherits the same open/close logic as MT5Broker in execution_broker.py
    but resolves symbols through the Deriv MT5 symbol map (which includes
    synthetics like "Volatility 75 Index").

    Steps to connect:
      1. Install MT5 from Deriv: https://deriv.com/platforms/mt5/
      2. Log in with your Deriv MT5 credentials
      3. Set in .env:
           MT5_LOGIN=12345678
           MT5_PASSWORD=your_password
           MT5_SERVER=Deriv-Demo       (or Deriv-Server for live)
      4. Optionally set MT5_PATH if MT5 is not in the default location
    """

    def __init__(
        self,
        login:     int = None,
        password:  str = None,
        server:    str = None,
        path:      str = None,
        deviation: int = 20,
        magic:     int = 20250101,
    ):
        self.login     = login    or int(os.getenv("MT5_LOGIN",    0))
        self.password  = password or os.getenv("MT5_PASSWORD",    "")
        self.server    = server   or os.getenv("MT5_SERVER",      "Deriv-Demo")
        self.path      = path     or os.getenv("MT5_PATH",        "")
        self.deviation = deviation
        self.magic     = magic
        self._mt5      = None
        self._connected = False

    @property
    def name(self) -> str:
        return "DerivMT5Broker"

    # ── Connection ────────────────────────────────────────────────── #

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error(
                "MetaTrader5 not installed. Run: pip install MetaTrader5"
            )
            return False

        init_kwargs = {}
        if self.path:
            init_kwargs["path"] = self.path

        if not self._mt5.initialize(**init_kwargs):
            logger.error(f"[DerivMT5] initialize() failed: {self._mt5.last_error()}")
            return False

        if self.login and self.password and self.server:
            ok = self._mt5.login(
                login    = self.login,
                password = self.password,
                server   = self.server,
            )
            if not ok:
                logger.error(f"[DerivMT5] login failed: {self._mt5.last_error()}")
                self._mt5.shutdown()
                return False

        info = self._mt5.account_info()
        if info:
            logger.info(
                f"[DerivMT5] connected | account={info.login} | "
                f"server={info.server} | balance={info.balance:.2f} {info.currency}"
            )
        self._connected = True
        return True

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("[DerivMT5] disconnected.")

    def _require_connection(self) -> bool:
        if self._connected:
            return True
        return self.connect()

    # ── Symbol resolution ─────────────────────────────────────────── #

    def _mt5_symbol(self, instrument: str) -> str:
        """
        Map an instrument name (OANDA-format or Deriv short code)
        to the exact MT5 symbol string for Deriv's terminal.
        """
        sym = _MT5_SYMBOL_MAP.get(instrument, instrument.replace("_", ""))
        if self._mt5:
            self._mt5.symbol_select(sym, True)   # ensure visible in MarketWatch
        return sym

    # ── Lot sizing for synthetics ─────────────────────────────────── #

    @staticmethod
    def _units_to_lots(instrument: str, units: int) -> float:
        """
        Convert signed unit count to MT5 lots.

        Synthetics on Deriv MT5 use smaller contract sizes than
        standard forex. Deriv typically uses:
          - Forex:      1 lot = 100,000 units  (standard)
          - Synthetics: 1 lot = 1 unit of the index  (varies)

        For safety we use the universal approach:
          lots = abs(units) / 100_000   (standard lot)
          Floored to 0.01 (MT5 minimum)

        If your Deriv MT5 account shows a different contract size
        for synthetics in the symbol specs, divide by that instead.
        You can check it with:
          mt5.symbol_info("Volatility 75 Index").trade_contract_size
        """
        lots = round(abs(units) / 100_000, 2)
        return max(lots, 0.01)

    # ── Core order methods ────────────────────────────────────────── #

    def open_trade(
        self,
        instrument: str,
        units:      int,
        sl_price:   float,
        tp_price:   float,
    ) -> TradeResult:
        if not self._require_connection():
            return TradeResult(success=False, error="MT5 not connected")

        mt5    = self._mt5
        symbol = self._mt5_symbol(instrument)

        # Round prices to Deriv-appropriate precision
        sl_price = _round_deriv_price(instrument, sl_price)
        tp_price = _round_deriv_price(instrument, tp_price)

        # Get contract size from symbol info for accurate lot calc
        sym_info = mt5.symbol_info(symbol)
        contract_size = (
            float(sym_info.trade_contract_size)
            if sym_info is not None
            else 100_000.0
        )
        lots = round(abs(units) / contract_size, 2)
        lots = max(lots, 0.01)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            err = f"No tick data for {symbol} — check MarketWatch"
            logger.error(f"[DerivMT5] {err}")
            return TradeResult(success=False, error=err)

        price      = tick.ask if units > 0 else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if units > 0 else mt5.ORDER_TYPE_SELL

        # Try all filling modes — Deriv brokers vary
        result = None
        for filling in [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN,
        ]:
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       lots,
                "type":         order_type,
                "price":        price,
                "sl":           sl_price,
                "tp":           tp_price,
                "deviation":    self.deviation,
                "magic":        self.magic,
                "comment":      "smc_deriv",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                break

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "no result"
            logger.error(
                f"[DerivMT5] open_trade failed {symbol}: "
                f"retcode={retcode} | {comment}"
            )
            return TradeResult(success=False, error=f"retcode={retcode} {comment}")

        logger.info(
            f"[DerivMT5] opened {symbol} {'BUY' if units > 0 else 'SELL'} "
            f"lots={lots} @ {result.price:.5f} | "
            f"sl={sl_price} tp={tp_price} | ticket={result.order}"
        )
        return TradeResult(
            success    = True,
            trade_id   = str(result.order),
            fill_price = float(result.price),
        )

    def close_trade(self, trade_id: str, instrument: str) -> TradeResult:
        if not self._require_connection():
            return TradeResult(success=False, error="MT5 not connected")

        mt5    = self._mt5
        symbol = self._mt5_symbol(instrument)
        ticket = int(trade_id)

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"[DerivMT5] no open position with ticket={ticket}")
            return TradeResult(success=False, error=f"ticket {ticket} not found")

        pos  = positions[0]
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(success=False, error=f"No tick for {symbol}")

        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    self.deviation,
            "magic":        self.magic,
            "comment":      "smc_deriv_close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "no result"
            logger.error(
                f"[DerivMT5] close_trade failed ticket={ticket}: "
                f"retcode={retcode} | {comment}"
            )
            return TradeResult(success=False, error=f"retcode={retcode} {comment}")

        logger.info(
            f"[DerivMT5] closed ticket={ticket} {symbol} @ {result.price}"
        )
        return TradeResult(
            success    = True,
            trade_id   = trade_id,
            fill_price = float(result.price),
        )

    def close_all(self, instrument: str) -> TradeResult:
        if not self._require_connection():
            return TradeResult(success=False, error="MT5 not connected")

        symbol    = self._mt5_symbol(instrument)
        positions = self._mt5.positions_get(symbol=symbol)

        if not positions:
            return TradeResult(success=True)

        errors = []
        for pos in positions:
            r = self.close_trade(str(pos.ticket), instrument)
            if not r.success:
                errors.append(r.error)

        if errors:
            return TradeResult(success=False, error="; ".join(errors))
        return TradeResult(success=True)

    def account_info(self) -> dict:
        if not self._require_connection():
            return {}
        info = self._mt5.account_info()
        if info is None:
            return {}
        return {
            "login":       info.login,
            "server":      info.server,
            "balance":     info.balance,
            "equity":      info.equity,
            "margin":      info.margin,
            "free_margin": info.margin_free,
            "currency":    info.currency,
        }


# ================================================================== #
#  DerivWebSocketBroker  (no MT5 required)                           #
# ================================================================== #

class DerivWebSocketBroker(ExecutionBroker):
    """
    Places trades on Deriv via the JSON WebSocket API using
    "multiplier contracts" — Deriv's CFD-like product.

    Requirements:
      - pip install websocket-client
      - DERIV_API_TOKEN in .env with "Trade" scope

    Multiplier contracts work as follows:
      - You specify a stake (USD amount) and a multiplier (e.g. x50)
      - P&L = stake × multiplier × % price change
      - Built-in take-profit, stop-loss, and deal cancellation

    This is simpler than MT5 for synthetics-only trading.
    For forex, use DerivMT5Broker.

    NOTE: Unit-to-stake conversion
      The `units` parameter from PaperTrader represents the contract
      size in currency units. We convert it to a Deriv stake using:
        stake = units × price × risk_fraction
      However this broker instead uses `units` directly as the dollar
      stake (i.e. treat `units` as if it means dollars, not currency units).
      PaperTrader's risk manager already computes units as a dollar-risk
      measure. Adjust the conversion below if you want different behaviour.
    """

    # Default multiplier per instrument
    DEFAULT_MULTIPLIERS = {
        "R_10":      20,
        "R_25":      20,
        "R_50":      20,
        "R_75":      20,
        "R_100":     20,
        "CRASH1000": 20,
        "BOOM1000":  20,
        "CRASH500":  20,
        "BOOM500":   20,
        "STPIDX":    100,
    }
    DEFAULT_MULTIPLIER = 20

    def __init__(
        self,
        api_token:  str = None,
        app_id:     str = "1089",
        multiplier: int = None,
    ):
        self.api_token  = api_token or os.getenv("DERIV_API_TOKEN", "")
        self.app_id     = app_id
        self._multiplier_override = multiplier
        self._client    = DerivWebSocketClient(api_token=self.api_token)
        # contract_id → instrument mapping for close_trade
        self._open_contracts: dict = {}

    @property
    def name(self) -> str:
        return "DerivWebSocketBroker"

    def _get_multiplier(self, instrument: str) -> int:
        if self._multiplier_override:
            return self._multiplier_override
        return self.DEFAULT_MULTIPLIERS.get(instrument, self.DEFAULT_MULTIPLIER)

    def _authorize(self) -> bool:
        """Authorize the WebSocket token. Required before trading."""
        if not self.api_token:
            logger.error("[DerivWS] No DERIV_API_TOKEN set.")
            return False
        try:
            resp = self._client._ws_request({"authorize": self.api_token})
            if "authorize" in resp:
                logger.info(
                    f"[DerivWS] authorized | "
                    f"balance={resp['authorize'].get('balance')} "
                    f"{resp['authorize'].get('currency')}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"[DerivWS] authorize failed: {e}")
            return False

    def open_trade(
        self,
        instrument: str,
        units:      int,
        sl_price:   float,
        tp_price:   float,
    ) -> TradeResult:
        """
        Place a multiplier contract on Deriv.
        `units` is interpreted as the dollar stake amount.
        SL and TP are passed as limit offsets.
        """
        if not self._authorize():
            return TradeResult(success=False, error="Authorization failed")

        # Convert units to stake; treat units as dollar stake directly
        # Minimum stake on Deriv is $1
        stake       = max(1.0, round(abs(units) / 100_000 * 100, 2))
        contract_type = "MULTUP" if units > 0 else "MULTDOWN"
        multiplier  = self._get_multiplier(instrument)

        # Derive the Deriv symbol name
        from deriv_feed import DERIV_SYMBOL_MAP
        symbol = DERIV_SYMBOL_MAP.get(instrument, instrument)

        payload: dict = {
            "buy": 1,
            "price": stake,
            "parameters": {
                "amount":        stake,
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "symbol":        symbol,
                "multiplier":    multiplier,
            }
        }

        # Add stop-loss barrier if provided
        if sl_price and sl_price > 0:
            payload["parameters"]["stop_loss"] = {
                "order_type":   "stop",
                "order_amount": stake,   # close for full loss of stake
            }

        # Add take-profit barrier if provided
        if tp_price and tp_price > 0:
            payload["parameters"]["take_profit"] = {
                "order_type":   "limit",
                "order_amount": stake,   # close at full profit of stake
            }

        try:
            resp        = self._client._ws_request(payload)
            buy_resp    = resp.get("buy", {})
            contract_id = buy_resp.get("contract_id", "")
            buy_price   = float(buy_resp.get("buy_price", 0))

            if not contract_id:
                err = resp.get("error", {}).get("message", "No contract_id returned")
                logger.error(f"[DerivWS] open_trade failed {symbol}: {err}")
                return TradeResult(success=False, error=err)

            self._open_contracts[str(contract_id)] = instrument
            logger.info(
                f"[DerivWS] opened {symbol} {contract_type} "
                f"stake={stake} x{multiplier} | contract_id={contract_id}"
            )
            return TradeResult(
                success    = True,
                trade_id   = str(contract_id),
                fill_price = buy_price,
            )
        except Exception as e:
            logger.error(f"[DerivWS] open_trade exception {symbol}: {e}")
            return TradeResult(success=False, error=str(e))

    def close_trade(self, trade_id: str, instrument: str) -> TradeResult:
        """Sell (close) an open multiplier contract."""
        if not self._authorize():
            return TradeResult(success=False, error="Authorization failed")

        payload = {"sell": int(trade_id), "price": 0}   # 0 = market price
        try:
            resp      = self._client._ws_request(payload)
            sell_resp = resp.get("sell", {})
            sold_for  = float(sell_resp.get("sold_for", 0))

            if "error" in resp:
                err = resp["error"].get("message", "close failed")
                logger.error(f"[DerivWS] close_trade failed id={trade_id}: {err}")
                return TradeResult(success=False, error=err)

            self._open_contracts.pop(trade_id, None)
            logger.info(
                f"[DerivWS] closed contract_id={trade_id} sold_for={sold_for}"
            )
            return TradeResult(
                success    = True,
                trade_id   = trade_id,
                fill_price = sold_for,
            )
        except Exception as e:
            logger.error(f"[DerivWS] close_trade exception id={trade_id}: {e}")
            return TradeResult(success=False, error=str(e))

    def close_all(self, instrument: str) -> TradeResult:
        """Close all open contracts for a given instrument."""
        errors = []
        to_close = [
            cid for cid, inst in self._open_contracts.items()
            if inst == instrument
        ]
        for cid in to_close:
            r = self.close_trade(cid, instrument)
            if not r.success:
                errors.append(r.error)

        if errors:
            return TradeResult(success=False, error="; ".join(errors))
        return TradeResult(success=True)
"""
execution_broker.py
────────────────────
Decouples order execution from the analysis engine.

The analysis engine (PaperTrader) now calls an ExecutionBroker instead of
calling OandaClient directly. Swap the broker at startup and the entire
analysis, ML, and risk pipeline stays untouched.

Classes
───────
  ExecutionBroker   — abstract base (interface)
  OandaBroker       — wraps the existing OandaClient (drop-in replacement)
  MT5Broker         — sends orders to MetaTrader 5 via the official Python lib
  SimBroker         — silent no-op used in sim_mode (replaces the old if/else)

Usage in main.py / paper_trader.py
────────────────────────────────────
  from execution_broker import OandaBroker, MT5Broker, SimBroker

  # Live execution via MetaTrader 5
  broker = MT5Broker(login=123456, password="pw", server="BrokerName-Live")

  # Live execution via OANDA (old behaviour, unchanged)
  broker = OandaBroker(oanda_client)

  # Simulate only — no real orders
  broker = SimBroker()

  # Pass into PaperTrader
  trader = PaperTrader(..., broker=broker)

TradeResult
────────────
All open/close calls return a TradeResult dataclass so PaperTrader can
store the broker-assigned trade ID (OANDA tradeID or MT5 ticket number)
without knowing which broker placed it.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ================================================================== #
#  Shared result dataclass                                            #
# ================================================================== #

@dataclass
class TradeResult:
    """Returned by every open_trade / close_trade call."""
    success:    bool
    trade_id:   str   = ""    # OANDA tradeID or MT5 ticket (as string)
    fill_price: float = 0.0   # actual fill price if available
    error:      str   = ""    # error message on failure


# ================================================================== #
#  Abstract base                                                      #
# ================================================================== #

class ExecutionBroker(ABC):
    """
    All execution brokers must implement these four methods.
    PaperTrader calls only these — never reaches into broker internals.
    """

    @abstractmethod
    def open_trade(
        self,
        instrument: str,
        units:      int,        # signed: positive = long, negative = short
        sl_price:   float,
        tp_price:   float,
    ) -> TradeResult:
        """Place a market order with SL and TP attached."""
        ...

    @abstractmethod
    def close_trade(self, trade_id: str, instrument: str) -> TradeResult:
        """Close a specific open trade by its broker trade ID."""
        ...

    @abstractmethod
    def close_all(self, instrument: str) -> TradeResult:
        """Close all positions for an instrument (fallback if no trade_id)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable broker name for logging."""
        ...


# ================================================================== #
#  SimBroker — silent no-op (sim_mode)                               #
# ================================================================== #

class SimBroker(ExecutionBroker):
    """
    Does nothing. Used when sim_mode=True.
    Replaces the scattered `if self.oanda and not self.sim_mode` guards —
    now PaperTrader always calls broker.open_trade(). In sim mode that
    call is a no-op returning success.
    """

    @property
    def name(self) -> str:
        return "SimBroker"

    def open_trade(self, instrument, units, sl_price, tp_price) -> TradeResult:
        logger.debug(f"[SIM] open  {instrument} units={units} sl={sl_price} tp={tp_price}")
        return TradeResult(success=True, trade_id="SIM", fill_price=0.0)

    def close_trade(self, trade_id, instrument) -> TradeResult:
        logger.debug(f"[SIM] close {instrument} id={trade_id}")
        return TradeResult(success=True, trade_id=trade_id)

    def close_all(self, instrument) -> TradeResult:
        logger.debug(f"[SIM] close_all {instrument}")
        return TradeResult(success=True)


# ================================================================== #
#  OandaBroker — wraps the existing OandaClient                      #
# ================================================================== #

class OandaBroker(ExecutionBroker):
    """
    Thin adapter around the existing OandaClient.
    Behaviour is identical to what paper_trader.py did before this
    refactor — nothing in the order logic has changed.
    """

    def __init__(self, oanda_client):
        self._client = oanda_client

    @property
    def name(self) -> str:
        return "OandaBroker"

    def open_trade(self, instrument, units, sl_price, tp_price) -> TradeResult:
        try:
            resp = self._client.market_order(instrument, units, sl_price, tp_price)
            trade_id = (
                resp.get("orderFillTransaction", {})
                    .get("tradeOpened", {}).get("tradeID", "")
                or resp.get("orderFillTransaction", {}).get("id", "")
            )
            fill = float(
                resp.get("orderFillTransaction", {}).get("price", 0) or 0
            )
            logger.info(
                f"[OANDA] opened {instrument} units={units} "
                f"sl={sl_price} tp={tp_price} → id={trade_id}"
            )
            return TradeResult(success=True, trade_id=str(trade_id), fill_price=fill)
        except Exception as e:
            body = getattr(getattr(e, "response", None), "text", str(e))
            logger.error(f"[OANDA] open_trade failed {instrument}: {e} | {body}")
            return TradeResult(success=False, error=str(e))

    def close_trade(self, trade_id: str, instrument: str) -> TradeResult:
        try:
            self._client.close_trade(trade_id)
            logger.info(f"[OANDA] closed trade id={trade_id} ({instrument})")
            return TradeResult(success=True, trade_id=trade_id)
        except Exception as e:
            logger.error(f"[OANDA] close_trade failed id={trade_id}: {e}")
            return TradeResult(success=False, error=str(e))

    def close_all(self, instrument: str) -> TradeResult:
        try:
            self._client.close_all_positions(instrument)
            logger.info(f"[OANDA] close_all {instrument}")
            return TradeResult(success=True)
        except Exception as e:
            logger.error(f"[OANDA] close_all failed {instrument}: {e}")
            return TradeResult(success=False, error=str(e))


# ================================================================== #
#  MT5 symbol map                                                     #
# ================================================================== #

# OANDA instrument names → MT5 symbol names.
# Symbol names vary by broker — edit these to match yours.
# Common variants: "GBPUSD", "GBPUSDm" (micro), "GBPUSD." (ECN suffix)
MT5_SYMBOL_MAP = {
    "GBP_USD": "GBPUSD",
    "EUR_USD": "EURUSD",
    "GBP_JPY": "GBPJPY",
    "XAU_USD": "XAUUSD",
    "USD_CAD": "USDCAD",
    "USD_CHF": "USDCHF",
    "XAG_USD": "XAGUSD",
    "AUD_JPY": "AUDJPY",
    "NZD_JPY": "NZDJPY",
    "AUD_USD": "AUDUSD",
    "AUD_NZD": "AUDNZD",
    "GBP_AUD": "GBPAUD",
    "EUR_JPY": "EURJPY",
    "EUR_AUD": "EURAUD",
    "USD_JPY": "USDJPY",
}


# ================================================================== #
#  MT5Broker                                                          #
# ================================================================== #

class MT5Broker(ExecutionBroker):
    """
    Executes orders via the MetaTrader 5 Python library.

    Requirements
    ────────────
    • Windows only (the MT5 terminal only runs on Windows)
    • pip install MetaTrader5
    • MT5 terminal must be running and logged into your broker account
    • Enable algo trading: Tools → Options → Expert Advisors
      → "Allow automated trading"

    Parameters
    ────────────
    login    : int   — MT5 account number (shown on login screen)
    password : str   — MT5 account password
    server   : str   — broker server name (shown on login screen)
    path     : str   — optional path to terminal64.exe if auto-detect fails
    deviation: int   — max allowed slippage in points (default 20)
    magic    : int   — magic number to tag bot orders (default 20250101)
    """

    def __init__(
        self,
        login:     int   = 0,
        password:  str   = "",
        server:    str   = "",
        path:      str   = None,
        deviation: int   = 20,
        magic:     int   = 20250101,
    ):
        self.login     = login
        self.password  = password
        self.server    = server
        self.path      = path
        self.deviation = deviation
        self.magic     = magic
        self._mt5      = None
        self._connected = False

    @property
    def name(self) -> str:
        return f"MT5Broker(login={self.login}@{self.server})"

    # ── Connection ────────────────────────────────────────────────── #
    def get_open_position_count(self) -> int:
        """Get actual open trade count directly from MT5."""
        if not self._require_connection():
            return 0
        positions = self._mt5.positions_get()
        return len(positions) if positions else 0
    
    def connect(self) -> bool:
        """
        Initialise and log in to MT5. Call once at bot startup.
        Returns True on success.
        """
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error(
                "MetaTrader5 package not installed. "
                "Run: pip install MetaTrader5"
            )
            return False

        init_kwargs = {}
        if self.path:
            init_kwargs["path"] = self.path

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        if self.login and self.password and self.server:
            ok = mt5.login(
                login    = self.login,
                password = self.password,
                server   = self.server,
            )
            if not ok:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info = mt5.account_info()
        if info:
            logger.info(
                f"MT5 connected | account={info.login} | "
                f"server={info.server} | balance={info.balance:.2f} | "
                f"currency={info.currency}"
            )
        self._connected = True
        return True

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def _require_connection(self) -> bool:
        if self._connected:
            return True
        return self.connect()

    # ── Symbol helpers ────────────────────────────────────────────── #

    def _mt5_symbol(self, instrument: str) -> str:
        sym = MT5_SYMBOL_MAP.get(instrument, instrument.replace("_", ""))
        if self._mt5:
            self._mt5.symbol_select(sym, True)   # ensure visible in MarketWatch
        return sym

    # ── Core order methods ────────────────────────────────────────── #

    def open_trade(self, instrument, units, sl_price, tp_price) -> TradeResult:
        print(f"[DEBUG] MT5Broker.open_trade called: {instrument} units={units} sl={sl_price} tp={tp_price}")  # remove after confirmed

        if not self._require_connection():
            return TradeResult(success=False, error="MT5 not connected")

        mt5    = self._mt5
        symbol = self._mt5_symbol(instrument)

        # Convert signed units to lots  (1 standard lot = 100,000 units)
        lots = round(abs(units) / 100_000, 2)
        lots = max(lots, 0.01) # MT5 minimum lot size

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            err = f"No tick data for {symbol} — check MarketWatch"
            logger.error(f"[MT5] {err}")
            return TradeResult(success=False, error=err)

        # Use ask for buys, bid for sells
        price      = tick.ask if units > 0 else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if units > 0 else mt5.ORDER_TYPE_SELL

        # Try each filling mode until one works (varies by broker)
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
                "comment":      "smc_bot",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                break  # this filling mode works

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "no result"
            logger.error(
                f"[MT5] open_trade failed {symbol}: "
                f"retcode={retcode} | {comment}"
            )
            return TradeResult(success=False, error=f"retcode={retcode} {comment}")

        logger.info(
            f"[MT5] opened {symbol} {'BUY' if units > 0 else 'SELL'} "
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
            logger.warning(f"[MT5] no open position with ticket={ticket}")
            return TradeResult(success=False, error=f"ticket {ticket} not found")

        pos  = positions[0]
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(success=False, error=f"No tick for {symbol}")

        # Closing direction is opposite to opening direction
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
            "comment":      "smc_bot_close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "no result"
            logger.error(
                f"[MT5] close_trade failed ticket={ticket}: "
                f"retcode={retcode} | {comment}"
            )
            return TradeResult(success=False, error=f"retcode={retcode} {comment}")

        logger.info(
            f"[MT5] closed ticket={ticket} {symbol} @ {result.price:.5f}"
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
        """Return a summary of the MT5 account (useful for logging)."""
        if not self._require_connection():
            return {}
        info = self._mt5.account_info()
        if info is None:
            return {}
        return {
            "login":        info.login,
            "server":       info.server,
            "balance":      info.balance,
            "equity":       info.equity,
            "margin":       info.margin,
            "free_margin":  info.margin_free,
            "currency":     info.currency,
        }
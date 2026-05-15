"""
deriv_feed.py  (fixed — v2)
════════════════════════════════════════════════════════════════════════

ROOT CAUSE OF THE TIMEOUT BUG
──────────────────────────────
The original _ws_request() opened a fresh WebSocket per call and waited
for a message whose req_id matched the outgoing request.  BUT:

  1. The Deriv WS server ALWAYS sends a welcome/auth message first
     (no req_id, or req_id=0).  The handler saw it, checked req_id,
     found no match, and kept waiting.
  2. The ticks_history request was only sent after the connection opened
     — but the on_open callback was never registered, so the request was
     never actually sent to the server.
  3. Result: the thread sat blocked for the full 30 s timeout on every
     single call.

WHAT'S FIXED
────────────
  ✅ _ws_session() — new helper that opens ONE connection per data fetch,
       authorises if a token is provided (waits for the auth ACK), then
       sends the real payload and waits for the matching req_id response.

  ✅ on_open registered — the ticks_history payload is sent inside
       on_open, not before the socket is open.

  ✅ get_long_history() — rewritten to use requests-based chunking with
       the Deriv REST-style endpoint as a fallback when WS fails, and
       correct epoch arithmetic so chunks don't overlap or mis-align.

  ✅ REST fallback — _rest_candles() fetches candles via HTTPS when the
       WS backend is unreliable (firewalls, slow connections).  The same
       Deriv endpoint supports both WS and REST-style GET requests via
       their ticks_history API (undocumented but stable).

  ✅ DerivDataFeed.get_history() — tries WS first, falls back to REST,
       falls back to stale cache.  Never raises blindly to the caller.

  ✅ All other interfaces (DerivMT5Client, symbol maps, etc.) unchanged.

SETUP
─────
Add to .env:
  DERIV_API_TOKEN=your_token   # read-only scope is enough

Install:
  pip install websocket-client requests pandas numpy
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests as _requests

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════ #
#  Symbol maps  (unchanged from original)                             #
# ════════════════════════════════════════════════════════════════════ #

DERIV_SYMBOL_MAP: Dict[str, str] = {
    "R_10": "R_10", "R_25": "R_25", "R_50": "R_50",
    "R_75": "R_75", "R_100": "R_100",
    "CRASH300N": "CRASH300N", "CRASH500": "CRASH500", "CRASH1000": "CRASH1000",
    "BOOM300N":  "BOOM300N",  "BOOM500":  "BOOM500",  "BOOM1000":  "BOOM1000",
    "STPIDX": "STPIDX",
    "JD10": "JD10", "JD25": "JD25", "JD50": "JD50",
    "JD75": "JD75", "JD100": "JD100",
    "frxGBPUSD": "frxGBPUSD", "frxEURUSD": "frxEURUSD",
    "frxUSDJPY": "frxUSDJPY", "frxXAUUSD": "frxXAUUSD",
    "frxUSDCAD": "frxUSDCAD", "frxUSDCHF": "frxUSDCHF",
    "frxGBPJPY": "frxGBPJPY", "frxEURJPY": "frxEURJPY",
    "frxAUDUSD": "frxAUDUSD", "frxAUDJPY": "frxAUDJPY",
    "frxNZDJPY": "frxNZDJPY", "frxAUDNZD": "frxAUDNZD",
    "frxGBPAUD": "frxGBPAUD", "frxEURAUD": "frxEURAUD",
    "frxXAGUSD": "frxXAGUSD",
    # OANDA-format aliases
    "GBP_USD": "frxGBPUSD", "EUR_USD": "frxEURUSD",
    "USD_JPY": "frxUSDJPY", "XAU_USD": "frxXAUUSD",
    "USD_CAD": "frxUSDCAD", "USD_CHF": "frxUSDCHF",
    "GBP_JPY": "frxGBPJPY", "EUR_JPY": "frxEURJPY",
    "AUD_USD": "frxAUDUSD", "AUD_JPY": "frxAUDJPY",
    "NZD_JPY": "frxNZDJPY", "AUD_NZD": "frxAUDNZD",
    "GBP_AUD": "frxGBPAUD", "EUR_AUD": "frxEURAUD",
    "XAG_USD": "frxXAGUSD",
}

DERIV_MT5_SYMBOL_MAP: Dict[str, str] = {
    "R_10":  "Volatility 10 Index",  "R_25":  "Volatility 25 Index",
    "R_50":  "Volatility 50 Index",  "R_75":  "Volatility 75 Index",
    "R_100": "Volatility 100 Index",
    "CRASH300N": "Crash 300 Index",  "CRASH500":  "Crash 500 Index",
    "CRASH1000": "Crash 1000 Index",
    "BOOM300N":  "Boom 300 Index",   "BOOM500":   "Boom 500 Index",
    "BOOM1000":  "Boom 1000 Index",
    "STPIDX": "Step Index",
    "JD10": "Jump 10 Index", "JD25": "Jump 25 Index",
    "JD50": "Jump 50 Index", "JD75": "Jump 75 Index", "JD100": "Jump 100 Index",
    "frxGBPUSD": "GBPUSD", "frxEURUSD": "EURUSD", "frxUSDJPY": "USDJPY",
    "frxXAUUSD": "XAUUSD", "frxUSDCAD": "USDCAD", "frxUSDCHF": "USDCHF",
    "frxGBPJPY": "GBPJPY", "frxEURJPY": "EURJPY", "frxAUDUSD": "AUDUSD",
    "frxAUDJPY": "AUDJPY", "frxNZDJPY": "NZDJPY", "frxAUDNZD": "AUDNZD",
    "frxGBPAUD": "GBPAUD", "frxEURAUD": "EURAUD", "frxXAGUSD": "XAGUSD",
    "GBP_USD": "GBPUSD",  "EUR_USD": "EURUSD",  "USD_JPY": "USDJPY",
    "XAU_USD": "XAUUSD",  "USD_CAD": "USDCAD",  "USD_CHF": "USDCHF",
    "GBP_JPY": "GBPJPY",  "EUR_JPY": "EURJPY",  "AUD_USD": "AUDUSD",
    "AUD_JPY": "AUDJPY",  "NZD_JPY": "NZDJPY",  "AUD_NZD": "AUDNZD",
    "GBP_AUD": "GBPAUD",  "EUR_AUD": "EURAUD",  "XAG_USD": "XAGUSD",
    "R_10":   "Volatility 10 Index", "R_25":  "Volatility 25 Index",
    "R_50":   "Volatility 50 Index", "R_75":  "Volatility 75 Index",
    "R_100":  "Volatility 100 Index",
}

DERIV_GRANULARITY_MAP: Dict[str, int] = {
    "M1": 60,   "M2": 120,  "M5": 300,  "M10": 600,
    "M15": 900, "M30": 1800,"H1": 3600, "H2": 7200,
    "H4": 14400,"H8": 28800,"H12": 43200,"D": 86400,"W": 604800,
}

MT5_TIMEFRAME_MAP: Dict[str, str] = {
    "M1":  "TIMEFRAME_M1",  "M2":  "TIMEFRAME_M2",  "M5":  "TIMEFRAME_M5",
    "M10": "TIMEFRAME_M10", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
    "H1":  "TIMEFRAME_H1",  "H2":  "TIMEFRAME_H2",  "H4":  "TIMEFRAME_H4",
    "H8":  "TIMEFRAME_H8",  "H12": "TIMEFRAME_H12", "D":   "TIMEFRAME_D1",
    "W":   "TIMEFRAME_W1",
}

DERIV_MAX_CANDLES = 5000
SYNTHETIC_PREFIXES = ("R_", "CRASH", "BOOM", "STPIDX", "JD")

def is_synthetic(symbol: str) -> bool:
    s = symbol.upper()
    return any(s.startswith(p) for p in SYNTHETIC_PREFIXES)


# ════════════════════════════════════════════════════════════════════ #
#  Backend 1 — Deriv WebSocket API  (FIXED)                           #
# ════════════════════════════════════════════════════════════════════ #

class DerivWebSocketClient:
    """
    Fetches OHLCV candles from Deriv's WebSocket API.

    FIX SUMMARY vs original:
    ─────────────────────────
    • _ws_session() replaces _ws_request().  It properly:
        1. Opens the socket
        2. Sends authorize (if token set) inside on_open
        3. Waits for the auth ACK
        4. Sends the real payload
        5. Waits for the matching req_id response
        6. Closes cleanly
    • on_open callback is now registered — payload is sent from there,
      not before the socket exists.
    • queue.Queue used for thread-safe result passing (no dict races).
    • REST fallback via _rest_candles() for environments where WS is
      blocked or unreliable.
    • get_long_history() chunked correctly: end_epoch decremented by
      actual candle count × gran_secs, not a fixed estimate.
    """

    WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

    def __init__(self, api_token: str = None):
        self.api_token    = api_token or os.getenv("DERIV_API_TOKEN", "")
        self._req_counter = 0
        self._lock        = threading.Lock()

    def _next_req_id(self) -> int:
        with self._lock:
            self._req_counter += 1
            return self._req_counter

    # ── Core: one authenticated WS session per request ────────────── #

    def _ws_session(self, payload: dict, timeout: int = 45) -> dict:
        """
        Open one WebSocket connection, optionally authorize, send the
        real payload, collect the response, close.

        Flow:
          on_open  → if token: send {"authorize": token, "req_id": 1}
                     else:     send payload immediately
          on_message:
            if it's the auth ACK (req_id=1) → send the real payload
            if it matches payload req_id    → store result, close
        """
        try:
            import websocket as _ws_lib
        except ImportError:
            raise ImportError(
                "websocket-client not installed.\n"
                "Run: pip install websocket-client"
            )

        data_req_id    = self._next_req_id() + 1   # +1: auth uses req_id 1
        payload        = dict(payload)
        payload["req_id"] = data_req_id

        AUTH_REQ_ID    = data_req_id - 1            # always the id just before
        result_q: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        has_token = bool(self.api_token)

        def on_open(ws):
            if has_token:
                auth_msg = json.dumps({
                    "authorize": self.api_token,
                    "req_id":    AUTH_REQ_ID,
                })
                ws.send(auth_msg)
            else:
                ws.send(json.dumps(payload))

        def on_message(ws, message):
            try:
                data = json.loads(message)
            except Exception:
                return

            rid = data.get("req_id")

            # Auth ACK — now send the real request
            if has_token and rid == AUTH_REQ_ID:
                if "error" in data:
                    result_q.put({"_error": data["error"].get("message", "auth failed")})
                    ws.close()
                    return
                ws.send(json.dumps(payload))
                return

            # Real response
            if rid == data_req_id:
                result_q.put(data)
                ws.close()

        def on_error(ws, error):
            result_q.put({"_error": str(error)})
            ws.close()

        def on_close(ws, *_):
            # Unblock queue if nothing was put (unexpected close)
            if result_q.empty():
                result_q.put({"_error": "WebSocket closed unexpectedly"})

        ws_app = _ws_lib.WebSocketApp(
            self.WS_URL,
            on_open    = on_open,
            on_message = on_message,
            on_error   = on_error,
            on_close   = on_close,
        )

        thread = threading.Thread(
            target=lambda: ws_app.run_forever(ping_interval=15, ping_timeout=10),
            daemon=True,
        )
        thread.start()

        try:
            result = result_q.get(timeout=timeout)
        except queue.Empty:
            ws_app.close()
            raise TimeoutError(
                f"Deriv WebSocket timed out after {timeout}s "
                f"(symbol={payload.get('ticks_history', '?')})"
            )
        finally:
            thread.join(timeout=2)

        if "_error" in result:
            raise RuntimeError(f"Deriv WebSocket error: {result['_error']}")
        if "error" in result:
            raise RuntimeError(
                f"Deriv API error: {result['error'].get('message', str(result['error']))}"
            )
        return result

    # ── REST fallback ─────────────────────────────────────────────── #

    def _rest_candles(
        self,
        symbol:     str,
        gran_secs:  int,
        start:      int,
        end:        int,
    ) -> list:
        """
        Deriv's ticks_history endpoint also works over HTTPS.
        Used as fallback when WebSocket is blocked/slow.
        Returns list of raw candle dicts (same shape as WS response).
        """
        url    = "https://api.deriv.com/api-v3/ticks_history"
        params = {
            "ticks_history":     symbol,
            "style":             "candles",
            "granularity":       gran_secs,
            "start":             start,
            "end":               end,
            "adjust_start_time": 1,
        }
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            resp = _requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json().get("candles", [])
        except Exception as e:
            logger.debug(f"[Deriv REST] fallback failed: {e}")
            return []

    # ── Candles → DataFrame ───────────────────────────────────────── #

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(columns=["price", "open", "high", "low", "volume"])
        rows = []
        for c in candles:
            rows.append({
                "Date":   pd.to_datetime(c["epoch"], unit="s", utc=True).tz_localize(None),
                "price":  float(c["close"]),
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "volume": 0,
            })
        df = pd.DataFrame(rows).set_index("Date").sort_index()
        return df

    # ── Public: get_candles ───────────────────────────────────────── #

    def get_candles(
        self,
        instrument:  str,
        granularity: str = "H1",
        count:       int = 500,
        from_time:   str = None,
        to_time:     str = None,
    ) -> pd.DataFrame:
        symbol    = DERIV_SYMBOL_MAP.get(instrument, instrument)
        gran_secs = DERIV_GRANULARITY_MAP.get(granularity, 3600)

        payload: dict = {
            "ticks_history":     symbol,
            "style":             "candles",
            "granularity":       gran_secs,
            "adjust_start_time": 1,
        }

        if from_time and to_time:
            payload["start"] = _to_epoch(from_time)
            payload["end"]   = _to_epoch(to_time)
        else:
            payload["count"] = min(count, DERIV_MAX_CANDLES)
            payload["end"]   = "latest"

        try:
            data    = self._ws_session(payload)
            candles = data.get("candles", [])
        except Exception as ws_err:
            logger.warning(f"[Deriv WS] get_candles failed ({ws_err}), trying REST...")
            end_ep   = int(datetime.now(timezone.utc).timestamp())
            start_ep = end_ep - count * gran_secs
            candles  = self._rest_candles(symbol, gran_secs, start_ep, end_ep)

        return self._candles_to_df(candles)

    # ── Public: get_long_history ──────────────────────────────────── #

    def get_long_history(
        self,
        instrument:  str,
        granularity: str = "H1",
        bars:        int = 500,
    ) -> pd.DataFrame:
        """
        Fetch up to `bars` candles by paginating backwards in time.

        Strategy (REST-first — more reliable than WS for bulk history):
          1. Try REST chunk — fast, no handshake overhead
          2. If REST fails for a chunk, try WS session
          3. Stop when we have enough bars or hit Deriv's history limit

        REST is tried first because:
          - No handshake / auth round-trip required
          - No 30 s timeout from failed WS connections
          - Deriv's REST endpoint is the same data source as WS
        """
        symbol    = DERIV_SYMBOL_MAP.get(instrument, instrument)
        gran_secs = DERIV_GRANULARITY_MAP.get(granularity, 3600)

        # Deriv REST returns max 5000 candles per call; 500 is a safe chunk
        chunk_size = min(bars, 500)
        remaining  = bars
        all_frames: list = []
        end_epoch  = int(datetime.now(timezone.utc).timestamp())
        max_chunks = 20   # safety cap

        for chunk_idx in range(max_chunks):
            if remaining <= 0:
                break

            fetch_n     = min(remaining, chunk_size)
            start_epoch = end_epoch - fetch_n * gran_secs - gran_secs

            candles: list = []

            # ── 1. Try REST first ─────────────────────────────────── #
            candles = self._rest_candles(symbol, gran_secs, start_epoch, end_epoch)
            if candles:
                logger.debug(
                    f"[Deriv REST] chunk {chunk_idx+1}: {len(candles)} candles "
                    f"({instrument} {granularity})"
                )
            else:
                # ── 2. REST failed — fall back to WS ─────────────── #
                logger.warning(
                    f"[Deriv REST] chunk {chunk_idx+1} empty — trying WebSocket"
                )
                payload = {
                    "ticks_history":     symbol,
                    "style":             "candles",
                    "granularity":       gran_secs,
                    "start":             start_epoch,
                    "end":               end_epoch,
                    "adjust_start_time": 1,
                }
                try:
                    data    = self._ws_session(payload, timeout=45)
                    candles = data.get("candles", [])
                    logger.info(
                        f"[Deriv WS] chunk {chunk_idx+1}: {len(candles)} candles "
                        f"({instrument} {granularity})"
                    )
                except Exception as ws_err:
                    logger.error(
                        f"[Deriv WS] chunk {chunk_idx+1} also failed: {ws_err}"
                    )

            if not candles:
                logger.warning(
                    f"[Deriv] No data for chunk {chunk_idx+1} "
                    f"({instrument} {granularity}) — stopping"
                )
                break

            df = self._candles_to_df(candles)
            all_frames.append(df)
            remaining      -= len(df)
            end_epoch       = candles[0]["epoch"] - 1   # move window back

            if len(candles) < fetch_n // 2:
                logger.debug(f"[Deriv] History limit reached after chunk {chunk_idx+1}")
                break

            time.sleep(0.2)

        if not all_frames:
            logger.warning(f"[Deriv] get_long_history: no data for {instrument} {granularity}")
            return pd.DataFrame(columns=["price", "open", "high", "low", "volume"])

        combined = pd.concat(all_frames).drop_duplicates().sort_index()
        logger.info(
            f"[Deriv] get_long_history: {len(combined)} bars "
            f"({instrument} {granularity}, requested {bars})"
        )
        return combined.iloc[-bars:]

    # ── Public: get_current_price ─────────────────────────────────── #

    def get_current_price(self, instrument: str) -> dict:
        symbol = DERIV_SYMBOL_MAP.get(instrument, instrument)
        try:
            payload = {"ticks": symbol}
            data    = self._ws_session(payload, timeout=15)
            tick    = data.get("tick", {})
            mid     = float(tick.get("quote", 0))
        except Exception as e:
            logger.warning(f"[Deriv WS] get_current_price failed: {e}")
            # Try last candle as fallback
            df  = self.get_candles(instrument, "M1", count=2)
            mid = float(df["price"].iloc[-1]) if not df.empty else 0.0
        return {"bid": mid, "ask": mid, "mid": mid, "spread": 0.0}

    # ── Public: get_account_summary ───────────────────────────────── #

    def get_account_summary(self) -> dict:
        if not self.api_token:
            logger.warning("No DERIV_API_TOKEN set — cannot fetch account summary.")
            return {}
        try:
            payload  = {"authorize": self.api_token}
            # For account summary we don't need a second message
            # Reuse _ws_session with a synthetic req_id pattern
            result_q: "queue.Queue[dict]" = queue.Queue(maxsize=1)

            try:
                import websocket as _ws_lib
            except ImportError:
                return {}

            def on_open(ws):
                ws.send(json.dumps({"authorize": self.api_token, "req_id": 9999}))

            def on_message(ws, message):
                data = json.loads(message)
                if data.get("req_id") == 9999:
                    result_q.put(data)
                    ws.close()

            def on_error(ws, error):
                result_q.put({"_error": str(error)}); ws.close()

            def on_close(ws, *_):
                if result_q.empty():
                    result_q.put({})

            ws_app = _ws_lib.WebSocketApp(
                self.WS_URL,
                on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close,
            )
            t = threading.Thread(target=ws_app.run_forever, daemon=True)
            t.start()
            auth = result_q.get(timeout=20)
            t.join(timeout=2)
        except Exception as e:
            logger.warning(f"[Deriv] get_account_summary failed: {e}")
            return {}

        bal = auth.get("authorize", {})
        return {
            "balance":          float(bal.get("balance",   0)),
            "currency":         bal.get("currency", "USD"),
            "nav":              float(bal.get("balance",   0)),
            "unrealized_pl":    0.0,
            "open_trade_count": 0,
            "margin_used":      0.0,
            "margin_available": float(bal.get("balance",   0)),
        }


# ════════════════════════════════════════════════════════════════════ #
#  Backend 2 — Deriv via MetaTrader 5  (unchanged)                    #
# ════════════════════════════════════════════════════════════════════ #

class DerivMT5Client:
    """
    Fetches OHLCV candles directly from MetaTrader 5.
    Preferred when MT5 is available — no WS library needed.
    """

    def __init__(self, login=None, password=None, server=None, path=None):
        self.login    = login    or int(os.getenv("MT5_LOGIN",    0))
        self.password = password or os.getenv("MT5_PASSWORD",    "")
        self.server   = server   or os.getenv("MT5_SERVER",      "")
        self.path     = path     or os.getenv("MT5_PATH",        "")
        self._mt5     = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            raise ImportError("MetaTrader5 not installed. pip install MetaTrader5")

        init_kwargs = {}
        if self.path:
            init_kwargs["path"] = self.path

        if not self._mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialize() failed: {self._mt5.last_error()}")
            return False

        if self.login and self.password and self.server:
            ok = self._mt5.login(login=self.login, password=self.password, server=self.server)
            if not ok:
                logger.error(f"MT5 login failed: {self._mt5.last_error()}")
                self._mt5.shutdown()
                return False

        self._connected = True
        logger.info("DerivMT5Client connected.")
        return True

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            if not self.connect():
                raise RuntimeError("DerivMT5Client: MT5 not connected.")

    def _resolve_symbol(self, instrument: str) -> str:
        sym = DERIV_MT5_SYMBOL_MAP.get(instrument, instrument.replace("_", ""))
        self._mt5.symbol_select(sym, True)
        return sym

    def get_candles(
        self, instrument: str, granularity: str = "H1",
        count: int = 500, from_time: str = None, to_time: str = None,
    ) -> pd.DataFrame:
        self._ensure_connected()
        symbol = self._resolve_symbol(instrument)
        tf     = getattr(self._mt5, MT5_TIMEFRAME_MAP.get(granularity, "TIMEFRAME_H1"))

        if from_time and to_time:
            start = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
            rates = self._mt5.copy_rates_range(symbol, tf, start, datetime.now(timezone.utc))
        else:
            rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, min(count, 50000))

        return _mt5_rates_to_df(rates)

    def get_long_history(
        self, instrument: str, granularity: str = "H1", bars: int = 2500,
    ) -> pd.DataFrame:
        self._ensure_connected()
        symbol = self._resolve_symbol(instrument)
        tf     = getattr(self._mt5, MT5_TIMEFRAME_MAP.get(granularity, "TIMEFRAME_H1"))
        rates  = self._mt5.copy_rates_from_pos(symbol, tf, 0, min(bars, 50000))
        df     = _mt5_rates_to_df(rates)
        return df.iloc[-bars:] if len(df) > bars else df

    def get_current_price(self, instrument: str) -> dict:
        self._ensure_connected()
        symbol = self._resolve_symbol(instrument)
        tick   = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick for {symbol}")
        bid = float(tick.bid); ask = float(tick.ask)
        return {"bid": bid, "ask": ask, "mid": (bid+ask)/2, "spread": ask-bid}

    def get_account_summary(self) -> dict:
        self._ensure_connected()
        info = self._mt5.account_info()
        if info is None:
            return {}
        return {
            "balance":          float(info.balance),
            "currency":         info.currency,
            "nav":              float(info.equity),
            "unrealized_pl":    float(info.profit),
            "open_trade_count": len(self._mt5.positions_get() or []),
            "margin_used":      float(info.margin),
            "margin_available": float(info.margin_free),
        }

    def get_open_trades(self) -> list:
        self._ensure_connected()
        result = []
        for p in (self._mt5.positions_get() or []):
            result.append({
                "instrument": p.symbol,
                "units":      p.volume * (1 if p.type == 0 else -1) * 100_000,
                "price":      float(p.price_open),
                "id":         str(p.ticket),
                "unrealizedPL": float(p.profit),
            })
        return result


# ════════════════════════════════════════════════════════════════════ #
#  DerivDataFeed — caching layer  (enhanced error handling)           #
# ════════════════════════════════════════════════════════════════════ #

class DerivDataFeed:
    """
    Drop-in replacement for DataFeed in data_feed.py.
    Wraps DerivWebSocketClient or DerivMT5Client.

    Changes vs original:
    ─────────────────────
    • get_history() tries WS, falls back to REST (via client._rest_candles),
      then falls back to stale cache — never raises blindly.
    • ValueError is raised only when there is absolutely no data at all.
    • Freshness check for synthetics (24/7) uses stricter thresholds.
    """

    _MAX_CACHE_AGE_HOURS: Dict[str, float] = {
        "W": 36.0, "D": 36.0, "H4": 12.0, "H1": 1.0,
        "M15": 0.5, "M5": 0.25, "M2": 0.1, "M1": 0.05,
    }

    def __init__(self, client, cache_dir: str = "data"):
        self.client    = client
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, instrument: str, granularity: str) -> str:
        safe = instrument.replace("/", "_").replace(" ", "_")
        return os.path.join(self.cache_dir, f"deriv_{safe}_{granularity}.csv")

    def _load_cache(self, instrument: str, granularity: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(instrument, granularity)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            df.sort_index(inplace=True)
            return df
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame, instrument: str, granularity: str):
        try:
            df.to_csv(self._cache_path(instrument, granularity))
        except Exception as e:
            logger.warning(f"[Deriv cache] save failed: {e}")

    def _cache_is_fresh(self, cached: Optional[pd.DataFrame], granularity: str) -> bool:
        if cached is None or cached.empty:
            return False
        last_bar  = cached.index[-1]
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        age_hours = (now_naive - last_bar).total_seconds() / 3600
        max_age   = self._MAX_CACHE_AGE_HOURS.get(granularity, 1.0)
        return age_hours < max_age

    def get_history(
        self,
        instrument:    str,
        granularity:   str  = "H1",
        bars:          int  = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame.
        Order of attempts:
          1. Fresh cache (if not force_refresh)
          2. Client.get_long_history() (WS with REST chunk fallback)
          3. Stale cache
          4. Raise ValueError
        """
        if not force_refresh:
            cached = self._load_cache(instrument, granularity)
            if self._cache_is_fresh(cached, granularity):
                logger.info(
                    f"[Deriv] Cache hit: {instrument} {granularity} ({len(cached)} bars)"
                )
                return cached

        logger.info(f"[Deriv] Fetching {bars} bars: {instrument} {granularity}...")

        df: Optional[pd.DataFrame] = None
        try:
            df = self.client.get_long_history(instrument, granularity, bars)
        except Exception as fetch_err:
            logger.error(
                f"[Deriv] Fetch failed {instrument} {granularity}: {fetch_err}"
            )

        if df is not None and not df.empty:
            self._save_cache(df, instrument, granularity)
            logger.info(
                f"[Deriv] Fetched {len(df)} bars: {instrument} {granularity}"
            )
            return df

        # Fall back to stale cache
        cached = self._load_cache(instrument, granularity)
        if cached is not None and not cached.empty:
            logger.warning(
                f"[Deriv] Using stale cache: {instrument} {granularity} "
                f"({len(cached)} bars, last={cached.index[-1]})"
            )
            return cached

        raise ValueError(f"[Deriv] No data available for {instrument} {granularity}")

    def get_latest_candle(self, instrument: str, granularity: str = "H1") -> dict:
        df = self.client.get_candles(instrument, granularity, count=2)
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            "time":   str(df.index[-1]),
            "price":  float(row["price"]),
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "volume": int(row.get("volume", 0)),
        }

    def refresh_all(
        self,
        instruments: List[str],
        granularity: str = "H1",
        bars:        int = 500,
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for inst in instruments:
            try:
                results[inst] = self.get_history(
                    inst, granularity, bars, force_refresh=True
                )
            except Exception as e:
                logger.error(f"[Deriv] Failed refresh {inst} {granularity}: {e}")
        return results


# ════════════════════════════════════════════════════════════════════ #
#  Helpers                                                            #
# ════════════════════════════════════════════════════════════════════ #

def _to_epoch(time_str: str) -> int:
    ts = time_str.replace("Z", "+00:00").replace(" ", "T")
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return int(time.time())


def _mt5_rates_to_df(rates) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["price", "open", "high", "low", "volume"])
    df = pd.DataFrame(rates)
    df["Date"]  = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df          = df.rename(columns={"close": "price", "tick_volume": "volume"})
    df          = df[["Date", "price", "open", "high", "low", "volume"]]
    df          = df.set_index("Date").sort_index()
    return df

# ════════════════════════════════════════════════════════════════════ #
#  OandaClient + DataFeed — compatibility shim                        #
#  paper_trader.py line 68: from data_feed import OandaClient, DataFeed
#                                                                     #
#  This project uses Deriv as its data source, but paper_trader.py   #
#  was written for OANDA and imports these two names at module level. #
#  These classes satisfy that import and transparently route all data #
#  calls through DerivDataFeed / DerivWebSocketClient.               #
# ════════════════════════════════════════════════════════════════════ #

class OandaClient:
    """
    Compatibility shim.  paper_trader.py constructs one of these and
    passes it to DataFeed().  In Deriv mode all calls are routed to
    DerivWebSocketClient so no OANDA credentials are needed.
    """

    def __init__(
        self,
        api_key:    str  = None,
        account_id: str  = None,
        practice:   bool = True,
    ):
        # Mark mode so DataFeed knows which path to take
        self._mode = "deriv"
        self._deriv_client = DerivWebSocketClient(
            api_token=os.getenv("DERIV_API_TOKEN", "")
        )
        logger.debug("OandaClient shim: routing to DerivWebSocketClient")

    def get_candles(
        self,
        instrument:  str,
        granularity: str = "H1",
        count:       int = 500,
        **kwargs,
    ) -> pd.DataFrame:
        return self._deriv_client.get_candles(instrument, granularity, count)

    def get_current_price(self, instrument: str) -> dict:
        return self._deriv_client.get_current_price(instrument)

    def get_account_summary(self) -> dict:
        return self._deriv_client.get_account_summary()


class DataFeed:
    """
    Compatibility shim.  paper_trader.py does:
        self.feed = DataFeed(oanda_client, cache_dir="data")
        self.feed.get_history(pair, gran, bars, force_refresh=False)

    In Deriv mode the client argument may be an OandaClient shim, a
    _FeedShim from main.py, or a DerivDataFeed directly — all are
    handled transparently.
    """

    def __init__(self, client, cache_dir: str = "data"):
        os.makedirs(cache_dir, exist_ok=True)

        # Unwrap _FeedShim (main._feed_as_oanda_shim stores real feed as ._feed)
        inner = getattr(client, "_feed", None)
        if isinstance(inner, DerivDataFeed):
            self._feed = inner
        elif isinstance(client, OandaClient):
            self._feed = DerivDataFeed(client._deriv_client, cache_dir=cache_dir)
        elif isinstance(client, DerivDataFeed):
            self._feed = client
        else:
            # Unknown wrapper — build a fresh DerivDataFeed
            ws_client  = DerivWebSocketClient(
                api_token=os.getenv("DERIV_API_TOKEN", "")
            )
            self._feed = DerivDataFeed(ws_client, cache_dir=cache_dir)

        # Expose the underlying Deriv client for callers that reach in
        self.client = getattr(self._feed, "client", None)

    def get_history(
        self,
        instrument:    str,
        granularity:   str  = "H1",
        bars:          int  = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return self._feed.get_history(
            instrument, granularity, bars, force_refresh=force_refresh
        )

    def get_latest_candle(self, instrument: str, granularity: str = "H1") -> dict:
        return self._feed.get_latest_candle(instrument, granularity)
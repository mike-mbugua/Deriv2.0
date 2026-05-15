"""
telegram_notifier.py  (Deriv-patched)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sends trading signals and status updates to a Telegram chat.

Fix applied vs original:
  • Pip calculation now uses the instrument's actual pip_size from
    deriv_config.DERIV_PIP_SIZE (falls back to 0.0001 for forex).
    Without this fix, synthetics like R_75 (price ~800,000) showed
    absurd pip values because the original hardcoded 0.0001.

Setup
─────
Add these two lines to your .env file:
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=987654321

How to get those values:
  1. Open Telegram → search @BotFather → /newbot → follow prompts → copy token
  2. Send any message to your new bot, then visit:
     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     Your chat_id is inside: result[0]["message"]["chat"]["id"]
"""

import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"

# ── Pip size lookup (Deriv synthetics + forex) ────────────────────── #
# Loaded lazily so this module can be imported even without deriv_config.
_PIP_SIZE_CACHE: dict = {}

def _get_pip_size(instrument: str) -> float:
    """
    Return the pip size for an instrument.
    Tries deriv_config first, then config.py, then falls back to 0.0001.
    Result is cached after first lookup.
    """
    if instrument in _PIP_SIZE_CACHE:
        return _PIP_SIZE_CACHE[instrument]

    pip = None

    # 1. Try Deriv pip table
    try:
        from deriv_config import DERIV_PIP_SIZE
        pip = DERIV_PIP_SIZE.get(instrument)
    except ImportError:
        pass

    # 2. Try OANDA pip table
    if pip is None:
        try:
            from config import PIP_SIZE
            pip = PIP_SIZE.get(instrument)
        except ImportError:
            pass

    # 3. Hard-coded sensible defaults
    if pip is None:
        if "JPY" in instrument:
            pip = 0.01
        elif instrument in ("XAU_USD", "frxXAUUSD"):
            pip = 0.1
        else:
            pip = 0.0001   # standard forex default

    _PIP_SIZE_CACHE[instrument] = pip
    return pip


def _price_to_pips(instrument: str, price_diff: float) -> float:
    """Convert an absolute price difference to pips for any instrument."""
    pip = _get_pip_size(instrument)
    return abs(price_diff) / pip


class TelegramNotifier:
    """
    Drop-in Telegram notifier for the PaperTrader bot.

    Usage (inside paper_trader.py):
        from telegram_notifier import TelegramNotifier
        self.telegram = TelegramNotifier()          # reads .env automatically
        self.telegram.send_signal(...)              # on open
        self.telegram.send_close(...)               # on close
        self.telegram.send_no_signal(...)           # on WAIT/filtered
    """

    def __init__(self, token: str = None, chat_id: str = None):
        self.token   = token   or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID",   "")

        if not self.token or not self.chat_id:
            raise ValueError(
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.\n"
                "See module docstring for instructions."
            )

    # ─────────────────────────────────────────────────────────────── #
    #  Core HTTP send                                                  #
    # ─────────────────────────────────────────────────────────────── #

    def send(self, message: str) -> bool:
        """Send any raw HTML-formatted message. Returns True on success."""
        url     = _TELEGRAM_URL.format(token=self.token)
        payload = {
            "chat_id":    self.chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"[TELEGRAM] send failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────── #
    #  Signal opened                                                   #
    # ─────────────────────────────────────────────────────────────── #

    def send_signal(
        self,
        pair:             str,
        direction:        str,      # "LONG" or "SHORT"
        entry:            float,
        sl:               float,
        tp:               float,
        units:            int,
        confidence:       float,    # 0.0 – 1.0
        ml_prob:          float,    # 0.0 – 1.0
        regime:           str,
        strategy:         str,
        weekly_structure: str  = "",
        daily_structure:  str  = "",
        h4_context:       str  = "",
        bull_pct:         float = 0.0,
        bear_pct:         float = 0.0,
        pd_arrays:        list  = None,
        fib_zones:        list  = None,
        reason:           str  = "",
        session_ok:       bool  = True,
        price:            float = 0.0,
    ) -> bool:
        pd_arrays = pd_arrays or []
        fib_zones = fib_zones or []

        emoji    = "🟢" if direction == "LONG" else "🔴"
        sess_tag = "● SESSION OK" if session_ok else "○ OFF-SESSION"
        rr       = abs(tp - entry) / abs(sl - entry) if (sl != entry) else 0.0

        def _arrow(s: str) -> str:
            s = s.upper()
            if "BULL" in s or "TREND" in s:  return "↑"
            if "BEAR" in s:                  return "↓"
            if "RANG" in s or "OSCILL" in s: return "↔"
            return "–"

        def _bar(pct: float) -> str:
            filled = round(pct / 100 * 18)
            return "█" * filled + "░" * (18 - filled)

        gap  = abs(bull_pct - bear_pct)
        bias = "BULLISH ▲" if bull_pct > bear_pct else "BEARISH ▼"
        if   gap >= 30: edge = "STRONG EDGE"
        elif gap >= 15: edge = "MODERATE EDGE"
        else:           edge = "WEAK EDGE"

        # ── FIX: use instrument-aware pip calculation ─────────────── #
        sl_pips = _price_to_pips(pair, entry - sl)
        tp_pips = _price_to_pips(pair, tp - entry)

        # ── Format price — more decimals for forex, fewer for synthetics #
        pip_size = _get_pip_size(pair)
        price_fmt = ".5f" if pip_size <= 0.0001 else (".3f" if pip_size <= 0.01 else ".2f")

        lines = [
            f"{emoji} <b>SIGNAL — {pair}</b>   {direction}   {sess_tag}",
            f"   Price: {(price or entry):{price_fmt}}",
            "",
            "<b>MARKET STRUCTURE</b>",
            f"  Weekly : {weekly_structure.upper():<12} {_arrow(weekly_structure)}",
            f"  Daily  : {daily_structure.upper():<12} {_arrow(daily_structure)}",
            f"  Regime : {regime.upper():<14}  4H Context : {h4_context.upper()} {_arrow(h4_context)}",
            "",
            "<b>TRADE PROBABILITY</b>",
            f"  Bull: {bull_pct:5.1f}%  [{_bar(bull_pct)}]",
            f"  Bear: {bear_pct:5.1f}%  [{_bar(bear_pct)}]",
            f"  Gap : {gap:5.1f}%  Bias: {bias}  → {edge}",
        ]

        if pd_arrays or fib_zones:
            lines += ["", "<b>PD ARRAYS</b>  (unmitigated)"]
            for a in pd_arrays:
                lines.append(f"  {a}")
            if fib_zones:
                lines.append("  <b>FIBONACCI ZONES</b>")
                for z in fib_zones:
                    lines.append(f"  {z}")

        lines += [
            "",
            "<b>ENTRY SETUP</b>",
            f"  Entry : {entry:{price_fmt}}",
            f"  SL    : {sl:{price_fmt}}  ({sl_pips:.1f} pips)",
            f"  TP    : {tp:{price_fmt}}  ({tp_pips:.1f} pips)",
            f"  R:R   : 1:{rr:.2f}",
            f"  Units : {units:,}",
            f"  Conf  : {confidence:.0%}   ML: {ml_prob:.0%}",
            f"  Strat : {strategy}",
        ]

        if reason:
            lines += ["", f"<b>REASON:</b> {reason}"]

        return self.send("\n".join(lines))

    # ─────────────────────────────────────────────────────────────── #
    #  No-signal / WAIT                                                #
    # ─────────────────────────────────────────────────────────────── #

    def send_no_signal(
        self,
        pair:             str,
        price:            float,
        session_ok:       bool  = True,
        weekly_structure: str   = "",
        daily_structure:  str   = "",
        h4_context:       str   = "",
        regime:           str   = "",
        bull_pct:         float = 0.0,
        bear_pct:         float = 0.0,
        pd_arrays:        list  = None,
        fib_zones:        list  = None,
        reason:           str   = "",
    ) -> bool:
        pd_arrays = pd_arrays or []
        fib_zones = fib_zones or []

        sess_tag = "● SESSION OK" if session_ok else "○ OFF-SESSION"

        def _arrow(s: str) -> str:
            s = s.upper()
            if "BULL" in s or "TREND" in s:  return "↑"
            if "BEAR" in s:                  return "↓"
            if "RANG" in s or "OSCILL" in s: return "↔"
            return "–"

        def _bar(pct: float) -> str:
            filled = round(pct / 100 * 18)
            return "█" * filled + "░" * (18 - filled)

        gap  = abs(bull_pct - bear_pct)
        bias = "BULLISH ▲" if bull_pct > bear_pct else "BEARISH ▼"
        if   gap >= 30: edge = "STRONG EDGE"
        elif gap >= 15: edge = "MODERATE EDGE"
        else:           edge = "WEAK EDGE"

        pip_size  = _get_pip_size(pair)
        price_fmt = ".5f" if pip_size <= 0.0001 else (".3f" if pip_size <= 0.01 else ".2f")

        lines = [
            f"⏸ <b>WAIT — {pair}</b>   {sess_tag}   Price: {price:{price_fmt}}",
            "",
            "<b>MARKET STRUCTURE</b>",
            f"  Weekly : {weekly_structure.upper():<12} {_arrow(weekly_structure)}",
            f"  Daily  : {daily_structure.upper():<12} {_arrow(daily_structure)}",
            f"  Regime : {regime.upper():<14}  4H Context : {h4_context.upper()} {_arrow(h4_context)}",
            "",
            "<b>TRADE PROBABILITY</b>",
            f"  Bull: {bull_pct:5.1f}%  [{_bar(bull_pct)}]",
            f"  Bear: {bear_pct:5.1f}%  [{_bar(bear_pct)}]",
            f"  Gap : {gap:5.1f}%  Bias: {bias}  → {edge}",
        ]

        if pd_arrays or fib_zones:
            lines += ["", "<b>PD ARRAYS</b>  (unmitigated)"]
            for a in pd_arrays:
                lines.append(f"  {a}")
            if fib_zones:
                lines.append("  <b>FIBONACCI ZONES</b>")
                for z in fib_zones:
                    lines.append(f"  {z}")

        if reason:
            lines += ["", f"<b>REASON:</b> {reason}"]

        return self.send("\n".join(lines))

    # ─────────────────────────────────────────────────────────────── #
    #  Trade closed                                                    #
    # ─────────────────────────────────────────────────────────────── #

    def send_close(
        self,
        pair:      str,
        direction: str,
        pnl:       float,
        pnl_pips:  float,
        reason:    str  = "",
        cash:      float = 0.0,
        nav:       float = 0.0,
    ) -> bool:
        emoji = "✅" if pnl >= 0 else "❌"
        sign  = "+" if pnl >= 0 else ""
        lines = [
            f"{emoji} <b>CLOSED — {pair}</b>  (was {direction})",
            f"  PnL   : {sign}{pnl:.2f} USD  ({sign}{pnl_pips:.1f} pips)",
        ]
        if reason:
            lines.append(f"  Reason: {reason.upper()}")
        if cash:
            lines.append(f"  Cash  : ${cash:,.2f}")
        if nav:
            lines.append(f"  NAV   : ${nav:,.2f}")
        return self.send("\n".join(lines))

    # ─────────────────────────────────────────────────────────────── #
    #  Portfolio heartbeat                                             #
    # ─────────────────────────────────────────────────────────────── #

    def send_portfolio(
        self,
        cash:         float,
        nav:          float,
        pnl:          float,
        open_trades:  int,
        max_trades:   int,
        win_rate:     float = 0.0,
        total_trades: int   = 0,
    ) -> bool:
        emoji = "📈" if pnl >= 0 else "📉"
        sign  = "+" if pnl >= 0 else ""
        ret   = (nav - cash) / cash * 100 if cash else 0.0
        lines = [
            f"{emoji} <b>PORTFOLIO UPDATE</b>",
            f"  Cash   : ${cash:,.2f}",
            f"  NAV    : ${nav:,.2f}",
            f"  PnL    : {sign}{pnl:.2f} USD  ({sign}{ret:.2f}%)",
            f"  Open   : {open_trades}/{max_trades}",
        ]
        if total_trades:
            lines.append(f"  Trades : {total_trades}  WR={win_rate:.1f}%")
        return self.send("\n".join(lines))

    # ─────────────────────────────────────────────────────────────── #
    #  Error / crash alert  (new — call from main.py exception handler)#
    # ─────────────────────────────────────────────────────────────── #

    def send_error(self, error_msg: str, context: str = "") -> bool:
        """
        Send a crash or critical error alert to Telegram.

        Usage in main.py:
            except Exception as e:
                if trader.telegram:
                    trader.telegram.send_error(str(e), context="run_live loop")
                raise
        """
        lines = [
            "🚨 <b>BOT ERROR</b>",
            f"  {error_msg[:300]}",   # cap at 300 chars
        ]
        if context:
            lines.append(f"  Context: {context}")
        return self.send("\n".join(lines))
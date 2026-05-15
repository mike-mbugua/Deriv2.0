"""
smc_logger.py  —  Rich terminal dashboard for the SMC forex bot
═══════════════════════════════════════════════════════════════

Prints a clean, structured panel every loop cycle showing:
  • Per-pair SMC analysis: market structure, bias, probabilities
  • All active PD arrays (FVGs, OBs, Breaker Blocks)
  • BOS / CHOCH events, Fibonacci zones, DOL target
  • Trade probability table (Bull% vs Bear% with visual bar)
  • Open positions with live P&L
  • Portfolio summary

No extra dependencies — pure Python + ANSI codes.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ── ANSI colour codes ───────────────────────────────────────────── #

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# Foreground colours
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"
MAGENTA= "\033[35m"
WHITE  = "\033[37m"
BRIGHT_WHITE = "\033[97m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_RED   = "\033[91m"
BRIGHT_YELLOW= "\033[93m"
BRIGHT_CYAN  = "\033[96m"

# Background colours (used for signal badges)
BG_GREEN  = "\033[42m"
BG_RED    = "\033[41m"
BG_YELLOW = "\033[43m"
BG_BLUE   = "\033[44m"
BG_GREY   = "\033[100m"

W = 78   # panel width


def _supports_color() -> bool:
    """Check if the terminal supports ANSI colour codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if USE_COLOR else text


def _line(char: str = "─", width: int = W) -> str:
    return char * width


def _bar(pct: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = round(pct / 100 * width)
    return fill * filled + empty * (width - filled)


def _signal_badge(signal: int) -> str:
    if signal == 1:
        return _c(BG_GREEN + BOLD, " ▲ LONG  ")
    if signal == -1:
        return _c(BG_RED + BOLD, " ▼ SHORT ")
    return _c(BG_GREY + DIM, "  WAIT  ")


def _bias_color(bias: str) -> str:
    if bias == "bullish":
        return _c(BRIGHT_GREEN + BOLD, "BULLISH ▲")
    if bias == "bearish":
        return _c(BRIGHT_RED + BOLD, "BEARISH ▼")
    return _c(YELLOW, "NEUTRAL  ")


def _struct_color(struct: str) -> str:
    if struct == "uptrend":
        return _c(GREEN, "UPTREND   ↑")
    if struct == "downtrend":
        return _c(RED, "DOWNTREND ↓")
    return _c(YELLOW, "RANGING   ↔")


def _prob_bar(bull: float, bear: float) -> str:
    """Draw a split probability bar: green for bull, red for bear."""
    w = 30
    b_len = round(bull / 100 * w)
    r_len = w - b_len
    bar = _c(GREEN, "█" * b_len) + _c(RED, "█" * r_len)
    return f"[{bar}]"


# ═══════════════════════════════════════════════════════════════════ #
#  Main dashboard print function                                      #
# ═══════════════════════════════════════════════════════════════════ #

def print_smc_dashboard(
    decisions:       Dict[str, object],    # pair → SwitchDecision (H1)
    h4_decisions:    Dict[str, object],    # pair → SwitchDecision (H4)
    prices:          Dict[str, float],
    portfolio:       dict,
    open_states:     Dict[str, object],    # pair → PairState
    loop_count:      int = 0,
    next_check_secs: int = 300,
    instruments:     List[str] = None,
):
    """
    Print the full SMC analysis dashboard to the terminal.

    Call once per live loop cycle after all pairs have been analysed.
    """
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
    pairs  = instruments or list(decisions.keys())

    print()
    _print_header(now, loop_count, portfolio)

    for pair in pairs:
        dec    = decisions.get(pair)
        h4_dec = h4_decisions.get(pair)
        price  = prices.get(pair, 0.0)
        state  = open_states.get(pair)
        _print_pair_panel(pair, dec, h4_dec, price, state)

    _print_portfolio_footer(portfolio, open_states, prices, next_check_secs)
    print()


# ─── Header ──────────────────────────────────────────────────────── #

def _print_header(now: str, loop_count: int, portfolio: dict):
    nav    = portfolio.get("nav", 0)
    cash   = portfolio.get("cash_balance", 0)
    pnl    = portfolio.get("total_pnl", 0)
    ret    = portfolio.get("return_pct", 0)
    trades = portfolio.get("total_trades", 0)
    opens  = len(portfolio.get("open_positions", {}))

    pnl_col  = BRIGHT_GREEN if pnl >= 0 else BRIGHT_RED
    ret_col  = BRIGHT_GREEN if ret >= 0 else BRIGHT_RED

    print(_c(BOLD + BRIGHT_CYAN, "╔" + _line("═") + "╗"))
    title = f"  SMC FOREX BOT  |  Loop #{loop_count}  |  {now}  "
    pad   = W - len(title)
    print(_c(BOLD + BRIGHT_CYAN, "║") + _c(BOLD + BRIGHT_WHITE, title) + " " * pad + _c(BOLD + BRIGHT_CYAN, "║"))
    print(_c(BOLD + BRIGHT_CYAN, "╠" + _line("═") + "╣"))

    nav_str = f"NAV: ${nav:>10,.2f}"
    pnl_str = f"P&L: {_c(pnl_col, f'${pnl:+,.2f}')}"
    ret_str = f"Return: {_c(ret_col, f'{ret:+.2f}%')}"
    tr_str  = f"Trades: {trades}  |  Open: {opens}"

    line = f"  {nav_str}   {ret_str}   {tr_str}"
    pad  = max(0, W - len(line) + (len(pnl_col) + len(ret_col) + len(RESET) * 2 if USE_COLOR else 0))
    print(_c(BRIGHT_CYAN, "║") + line + " " * (W - _visible_len(line)) + _c(BRIGHT_CYAN, "║"))
    print(_c(BRIGHT_CYAN, "╚" + _line("═") + "╝"))


# ─── Per-pair panel ───────────────────────────────────────────────── #

def _print_pair_panel(
    pair:    str,
    dec:     Optional[object],   # H1 SwitchDecision
    h4_dec:  Optional[object],   # H4 SwitchDecision
    price:   float,
    state:   Optional[object],   # PairState
):
    if dec is None:
        print(_c(DIM, f"  {pair:<10} — no analysis data"))
        return

    prob     = getattr(dec, "probability", None)
    bull_pct = prob.bullish    if prob else 50.0
    bear_pct = prob.bearish    if prob else 50.0
    conf     = prob.confidence if prob else 0.0
    bias_str = getattr(dec, "bias", "neutral")
    bias_val = bias_str.value if hasattr(bias_str, "value") else str(bias_str)

    w_struct = getattr(dec, "weekly_structure", "unknown")
    d_struct = getattr(dec, "daily_structure",  "unknown")
    regime   = dec.regime.value if hasattr(dec.regime, "value") else str(dec.regime)

    h4_struct = getattr(h4_dec, "weekly_structure", "—") if h4_dec else "—"
    signal    = getattr(dec, "signal", 0)
    session   = getattr(dec, "good_session", True)
    reason    = getattr(dec, "reason", "")

    w_sh = getattr(dec, "weekly_swing_high", 0.0)
    w_sl = getattr(dec, "weekly_swing_low",  0.0)
    d_sh = getattr(dec, "daily_swing_high",  0.0)
    d_sl = getattr(dec, "daily_swing_low",   0.0)

    fvg      = getattr(dec, "nearest_fvg", None)
    ob       = getattr(dec, "nearest_ob",  None)
    bos      = getattr(dec, "last_bos",    None)
    fibs     = getattr(dec, "fib_zones",   [])
    dol_tgt  = getattr(dec, "dol_target",  0.0)
    dol_kind = getattr(dec, "dol_kind",    "")

    # Position state
    pos_dir  = ""
    pos_info = ""
    if state and getattr(state, "position", 0) != 0:
        upnl      = getattr(state, "unrealized_pnl", 0.0)
        entry     = getattr(state, "entry_price", 0.0)
        sl        = getattr(state, "sl_price",    0.0)
        tp        = getattr(state, "tp_price",    0.0)
        units     = abs(getattr(state, "units",   0))
        pos_dir   = "LONG" if state.position == 1 else "SHORT"
        upnl_col  = BRIGHT_GREEN if upnl >= 0 else BRIGHT_RED
        pos_info  = (
            f"Entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  "
            f"Units={units:,}  uPnL={_c(upnl_col, f'${upnl:+.2f}')}"
        )

    # ── Panel border ─────────────────────────────────────────────── #
    pair_col = BRIGHT_WHITE + BOLD
    print(_c(CYAN, "┌" + _line("─") + "┐"))

    # Pair title row
    sig_badge  = _signal_badge(signal)
    sess_badge = _c(GREEN, "● SESSION OK") if session else _c(RED, "○ OFF-HOURS ")
    title_row  = f"  {_c(pair_col, f'{pair:<10}')}  {sig_badge}  {sess_badge}  Price: {_c(BRIGHT_WHITE, f'{price:.5f}')}"
    if pos_dir:
        title_row += f"  {_c(BG_BLUE + BOLD, f' {pos_dir} ')}"
    _print_panel_row(title_row)

    print(_c(CYAN, "├" + _line("─") + "┤"))

    # ── Market structure ─────────────────────────────────────────── #
    _print_panel_row(f"  {_c(BOLD, 'MARKET STRUCTURE')}")
    _print_panel_row(
        f"    Weekly : {_struct_color(w_struct):<30}  "
        f"SH={_fmt_price(w_sh)}  SL={_fmt_price(w_sl)}"
    )
    _print_panel_row(
        f"    Daily  : {_struct_color(d_struct):<30}  "
        f"SH={_fmt_price(d_sh)}  SL={_fmt_price(d_sl)}"
    )
    _print_panel_row(
        f"    Regime : {_c(MAGENTA, f'{regime.upper():<12}')}  "
        f"4H Context : {_struct_color(h4_struct)}"
    )

    print(_c(CYAN, "├" + _line("─") + "┤"))

    # ── Probability table ────────────────────────────────────────── #
    _print_panel_row(f"  {_c(BOLD, 'TRADE PROBABILITY')}")

    bull_bar = _c(GREEN, _bar(bull_pct, 18))
    bear_bar = _c(RED,   _bar(bear_pct, 18))
    _print_panel_row(
        f"    Bull: {_c(BRIGHT_GREEN + BOLD, f'{bull_pct:5.1f}%')}  [{bull_bar}]"
    )
    _print_panel_row(
        f"    Bear: {_c(BRIGHT_RED   + BOLD, f'{bear_pct:5.1f}%')}  [{bear_bar}]"
    )

    gap_col  = BRIGHT_GREEN if conf >= 30 else (YELLOW if conf >= 15 else RED)
    verdict  = (
        _c(BRIGHT_GREEN, "STRONG EDGE") if conf >= 30 else
        _c(YELLOW,       "MODERATE EDGE") if conf >= 15 else
        _c(RED,          "NO EDGE — SKIP")
    )
    _print_panel_row(
        f"    Gap  : {_c(gap_col + BOLD, f'{conf:5.1f}%')}  Bias: {_bias_color(bias_val)}  → {verdict}"
    )

    print(_c(CYAN, "├" + _line("─") + "┤"))

    # ── PD Arrays ────────────────────────────────────────────────── #
    _print_panel_row(f"  {_c(BOLD, 'PD ARRAYS  (unmitigated)')}")

    if fvg:
        mit_str = _c(RED, "MITIGATED") if fvg.mitigated else _c(GREEN, "ACTIVE")
        dir_col = GREEN if fvg.direction == "bullish" else RED
        _print_panel_row(
            f"    FVG  [{_c(dir_col, fvg.direction.upper()[:4])}] "
            f"{fvg.bottom:.5f} — {fvg.top:.5f}   {mit_str}"
        )
    else:
        _print_panel_row(f"    FVG  : {_c(DIM, 'none found')}")

    if ob:
        mit_str = _c(RED, "MITIGATED") if ob.mitigated else _c(GREEN, "ACTIVE")
        dir_col = GREEN if ob.direction == "bullish" else RED
        _print_panel_row(
            f"    OB   [{_c(dir_col, ob.direction.upper()[:4])}] "
            f"{ob.bottom:.5f} — {ob.top:.5f}   {mit_str}"
        )
    else:
        _print_panel_row(f"    OB   : {_c(DIM, 'none found')}")

    if bos:
        dir_col = GREEN if bos.direction == "bullish" else RED
        kind_str = _c(MAGENTA, bos.kind.upper())
        _print_panel_row(
            f"    BOS  [{_c(dir_col, bos.direction.upper()[:4])}] "
            f"@ {bos.price:.5f}  type={kind_str}"
        )
    else:
        _print_panel_row(f"    BOS  : {_c(DIM, 'none detected')}")

    # Fibonacci zones (show max 4)
    fib_lines = []
    seen_levels = set()
    for z in fibs:
        key = (z.level, z.direction)
        if key in seen_levels:
            continue
        seen_levels.add(key)
        z_col = GREEN if z.direction == "support" else RED
        fib_lines.append(
            f"    Fib {z.level:.3f} [{_c(z_col, z.direction[:4].upper())}] @ {z.price:.5f}"
        )
        if len(fib_lines) >= 4:
            break

    if fib_lines:
        _print_panel_row(f"  {_c(BOLD, 'FIBONACCI ZONES')}")
        for fl in fib_lines:
            _print_panel_row(fl)

    # DOL Target
    if dol_tgt:
        dol_dist = abs(dol_tgt - price)
        dol_col  = BRIGHT_CYAN
        _print_panel_row(
            f"  {_c(BOLD, 'DOL TARGET')}  "
            f"{_c(dol_col + BOLD, f'{dol_tgt:.5f}')}  "
            f"({dol_kind.upper()})  "
            f"dist={_c(YELLOW, f'{dol_dist:.5f}')}"
        )

    # Open position details
    if pos_info:
        print(_c(CYAN, "├" + _line("─") + "┤"))
        _print_panel_row(f"  {_c(BG_BLUE + BOLD, f' OPEN {pos_dir} ')}")
        _print_panel_row(f"    {pos_info}")

    # Reason / confirmation tags
    if reason:
        print(_c(CYAN, "├" + _line("─") + "┤"))
        _print_panel_row(f"  {_c(DIM, 'REASON:')} {_c(DIM, _truncate(reason, W - 12))}")

    print(_c(CYAN, "└" + _line("─") + "┘"))


# ─── Portfolio footer ─────────────────────────────────────────────── #

def _print_portfolio_footer(
    portfolio:   dict,
    open_states: Dict[str, object],
    prices:      Dict[str, float],
    next_secs:   int,
):
    nav        = portfolio.get("nav", 0)
    cash       = portfolio.get("cash_balance", 0)
    pnl        = portfolio.get("total_pnl", 0)
    ret        = portfolio.get("return_pct", 0)
    n_open     = len(portfolio.get("open_positions", {}))
    n_trades   = portfolio.get("total_trades", 0)
    mins, secs = divmod(next_secs, 60)

    pnl_col = BRIGHT_GREEN if pnl >= 0 else BRIGHT_RED
    ret_col = BRIGHT_GREEN if ret  >= 0 else BRIGHT_RED

    print(_c(CYAN, "╔" + _line("═") + "╗"))
    _print_panel_row(
        f"  {_c(BOLD, 'PORTFOLIO SUMMARY')}",
        border=_c(CYAN, "║")
    )
    _print_panel_row(
        f"    NAV       : {_c(BRIGHT_WHITE + BOLD, f'${nav:>12,.2f}')}   "
        f"Cash   : ${cash:>12,.2f}",
        border=_c(CYAN, "║")
    )
    _print_panel_row(
        f"    Total P&L : {_c(pnl_col + BOLD, f'${pnl:>+12,.2f}')}   "
        f"Return : {_c(ret_col + BOLD, f'{ret:>+8.2f}%')}",
        border=_c(CYAN, "║")
    )
    _print_panel_row(
        f"    Closed Trades: {n_trades}   "
        f"Open: {n_open}   "
        f"Next scan in {mins:02d}:{secs:02d}",
        border=_c(CYAN, "║")
    )

    # Open position quick-table
    open_pos = portfolio.get("open_positions", {})
    if open_pos:
        _print_panel_row(f"    {_c(BOLD, 'Open Positions:')}", border=_c(CYAN, "║"))
        for pair, pos in open_pos.items():
            upnl     = pos.get("upnl", 0)
            upnl_col = BRIGHT_GREEN if upnl >= 0 else BRIGHT_RED
            dir_str  = pos.get("direction", "?").upper()
            dir_col  = GREEN if dir_str == "LONG" else RED
            _print_panel_row(
                f"      {pair:<10} {_c(dir_col, dir_str):<6}  "
                f"Entry={pos.get('entry', 0):.5f}  "
                f"SL={pos.get('sl', 0):.5f}  "
                f"TP={pos.get('tp', 0):.5f}  "
                f"uPnL={_c(upnl_col, f'${upnl:+.2f}')}",
                border=_c(CYAN, "║")
            )

    print(_c(CYAN, "╚" + _line("═") + "╝"))


# ─── Trade event banners ──────────────────────────────────────────── #

def print_trade_opened(pair: str, direction: str, setup):
    units  = getattr(setup, "units",    0)
    entry  = getattr(setup, "entry",    0.0)
    sl     = getattr(setup, "sl_price", 0.0)
    tp     = getattr(setup, "tp_price", 0.0)
    rr     = getattr(setup, "rr_ratio", 0.0)
    risk   = getattr(setup, "risk_amount", 0.0)

    dir_col = GREEN if direction.upper() == "LONG" else RED
    print()
    print(_c(BG_GREEN + BOLD if direction.upper() == "LONG" else BG_RED + BOLD,
             f"  ▶ TRADE OPENED — {pair} {direction.upper()}  "))
    print(_c(CYAN, f"  Entry  : {entry:.5f}"))
    print(_c(RED,  f"  SL     : {sl:.5f}"))
    print(_c(GREEN,f"  TP     : {tp:.5f}"))
    print(        f"  Units  : {units:,}   Risk: ${risk:.2f}   RR: 1:{rr:.1f}")
    print()


def print_trade_closed(pair: str, reason: str, pnl: float, balance: float):
    pnl_col  = BRIGHT_GREEN if pnl >= 0 else BRIGHT_RED
    icon     = "✔" if pnl >= 0 else "✘"
    reason_u = reason.upper().replace("_", " ")

    print()
    print(_c(BG_GREEN + BOLD if pnl >= 0 else BG_RED + BOLD,
             f"  {icon} TRADE CLOSED — {pair}  [{reason_u}]  "))
    print(_c(pnl_col + BOLD, f"  PnL    : ${pnl:+.2f}"))
    print(                   f"  Balance: ${balance:,.2f}")
    print()


def print_no_signal_summary(pair_verdicts: Dict[str, str]):
    """
    Print a compact one-liner per pair when no trades are being taken.
    pair_verdicts: {pair: short reason string}
    """
    print(_c(DIM, "  ┄ Scan complete — no new entries ┄"))
    for pair, reason in pair_verdicts.items():
        print(_c(DIM, f"    {pair:<10} {reason}"))
    print()


# ─── Utility helpers ─────────────────────────────────────────────── #

def _print_panel_row(text: str, border: str = None):
    """Print a line padded to panel width with side borders."""
    border = border or _c(CYAN, "│")
    vlen   = _visible_len(text)
    pad    = max(0, W - vlen)
    print(f"{border}{text}{' ' * pad}{border}")


def _visible_len(text: str) -> int:
    """Length of a string ignoring ANSI escape codes."""
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def _fmt_price(p: float) -> str:
    if p == 0.0:
        return _c(DIM, "  —    ")
    return f"{p:.5f}"


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
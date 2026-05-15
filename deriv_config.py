"""
deriv_config.py
═══════════════════════════════════════════════════════════════════
Deriv-specific instrument definitions.

Merge strategy
──────────────
  This file does NOT replace config.py. Instead, import both and
  merge them in your .env / startup code using DERIV_MODE=true.

  In main.py:
      if os.getenv("DERIV_MODE", "false").lower() == "true":
          from deriv_config import DerivBotConfig as BotConfig
          from deriv_config import (
              DERIV_INSTRUMENTS as INSTRUMENTS,
              DERIV_PIP_SIZE    as PIP_SIZE,
              DERIV_SPREAD      as SPREAD,
              DERIV_SLIPPAGE    as SLIPPAGE,
              DERIV_LEVERAGE    as LEVERAGE,
              DERIV_SESSION_WINDOWS as SESSION_WINDOWS,
          )
      else:
          from config import BotConfig, INSTRUMENTS, PIP_SIZE, ...

  Or combine both configs in one run with the UnifiedBotConfig below.

Pip sizes for synthetics
─────────────────────────
  Synthetics don't have traditional "pips". We define pip_size as the
  minimum price increment so ATR-based calculations stay consistent:
    Vol 10  → 0.001   (moves in small increments)
    Vol 25  → 0.001
    Vol 50  → 0.001
    Vol 75  → 0.001
    Vol 100 → 0.001
    Crash/Boom → 0.001
    Step    → 0.10

  These are used only for position sizing and display — the actual
  SL/TP distances are set by ATR multiples, which self-calibrate.

Leverage
─────────
  Deriv multipliers vary by account type (standard/swap-free/STP).
  Defaults below are conservative; override in .env as needed.
  Synthetics typically allow up to x500 but we cap at x50 for safety.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()


# ================================================================== #
#  Deriv synthetic instruments                                        #
# ================================================================== #

DERIV_SYNTHETIC_INSTRUMENTS = [
    "R_75",       # Volatility 75 Index — most popular
    "R_100",      # Volatility 100 Index
    "R_50",       # Volatility 50 Index
    "R_25",       # Volatility 25 Index
    "R_10",       # Volatility 10 Index
    "CRASH1000",  # Crash 1000 Index
    "BOOM1000",   # Boom 1000 Index
    "CRASH500",   # Crash 500 Index
    "BOOM500",    # Boom 500 Index
    "STPIDX",    # Step Index
]

# Forex pairs available on Deriv (using OANDA-compatible naming)
DERIV_FOREX_INSTRUMENTS = [
    "GBP_USD",
    "EUR_USD",
    "XAU_USD",
    "USD_JPY",
    "GBP_JPY",
]

# Combined: run SMC on synthetics first (24/7), then forex
DERIV_INSTRUMENTS = DERIV_SYNTHETIC_INSTRUMENTS + DERIV_FOREX_INSTRUMENTS

# Primary pair for single-pair modes
DERIV_PRIMARY_PAIR = "R_75"


# ================================================================== #
#  Pip sizes                                                         #
# ================================================================== #

DERIV_PIP_SIZE: Dict[str, float] = {
    # ── Synthetics ─────────────────────────────────────────────────
    "R_10":      0.001,
    "R_25":      0.001,
    "R_50":      0.001,
    "R_75":      0.001,
    "R_100":     0.001,
    "CRASH300N": 0.001,
    "CRASH500":  0.001,
    "CRASH1000": 0.001,
    "BOOM300N":  0.001,
    "BOOM500":   0.001,
    "BOOM1000":  0.001,
    "STPIDX":    0.10,
    "JD10":      0.001,
    "JD25":      0.001,
    "JD50":      0.001,
    "JD75":      0.001,
    "JD100":     0.001,
    # ── Forex (same as config.py) ──────────────────────────────────
    "GBP_USD":   0.0001,
    "EUR_USD":   0.0001,
    "USD_JPY":   0.01,
    "GBP_JPY":   0.01,
    "EUR_JPY":   0.01,
    "AUD_JPY":   0.01,
    "NZD_JPY":   0.01,
    "XAU_USD":   0.1,
    "XAG_USD":   0.01,
    "USD_CAD":   0.0001,
    "USD_CHF":   0.0001,
    "AUD_USD":   0.0001,
    "AUD_NZD":   0.0001,
    "GBP_AUD":   0.0001,
    "EUR_AUD":   0.0001,
}


# ================================================================== #
#  Spread estimates (price units, not pips)                          #
# ================================================================== #
#  Synthetics on Deriv have very tight fixed spreads.
#  Values below are conservative estimates; the actual spread is
#  embedded in Deriv's pricing and does not change with volatility.

DERIV_SPREAD: Dict[str, float] = {
    # ── Synthetics ─────────────────────────────────────────────────
    "R_10":      0.001,
    "R_25":      0.001,
    "R_50":      0.002,
    "R_75":      0.003,
    "R_100":     0.004,
    "CRASH300N": 0.002,
    "CRASH500":  0.002,
    "CRASH1000": 0.003,
    "BOOM300N":  0.002,
    "BOOM500":   0.002,
    "BOOM1000":  0.003,
    "STPIDX":    0.10,
    "JD10":      0.001,
    "JD25":      0.001,
    "JD50":      0.002,
    "JD75":      0.003,
    "JD100":     0.004,
    # ── Forex on Deriv (slightly wider than raw OANDA) ─────────────
    "GBP_USD":   0.00018,
    "EUR_USD":   0.00013,
    "USD_JPY":   0.025,
    "GBP_JPY":   0.035,
    "EUR_JPY":   0.025,
    "XAU_USD":   0.60,
    "USD_CAD":   0.00018,
    "USD_CHF":   0.00018,
    "AUD_USD":   0.00015,
    "AUD_JPY":   0.025,
    "NZD_JPY":   0.025,
    "AUD_NZD":   0.00028,
    "GBP_AUD":   0.00040,
    "EUR_AUD":   0.00035,
    "XAG_USD":   0.03,
}


# ================================================================== #
#  Slippage estimates                                                 #
# ================================================================== #
#  Synthetics have near-zero slippage (algorithmic pricing).
#  Forex on Deriv has slightly more than synthetics.

DERIV_SLIPPAGE: Dict[str, float] = {
    # Synthetics — nearly zero
    "R_10":      0.0001,
    "R_25":      0.0001,
    "R_50":      0.0002,
    "R_75":      0.0002,
    "R_100":     0.0003,
    "CRASH300N": 0.0002,
    "CRASH500":  0.0002,
    "CRASH1000": 0.0003,
    "BOOM300N":  0.0002,
    "BOOM500":   0.0002,
    "BOOM1000":  0.0003,
    "STPIDX":    0.05,
    "JD10":      0.0001,
    "JD25":      0.0001,
    "JD50":      0.0002,
    "JD75":      0.0002,
    "JD100":     0.0003,
    # Forex
    "GBP_USD":   0.00006,
    "EUR_USD":   0.00006,
    "USD_JPY":   0.012,
    "GBP_JPY":   0.012,
    "EUR_JPY":   0.010,
    "XAU_USD":   0.25,
    "USD_CAD":   0.00009,
    "USD_CHF":   0.00009,
    "AUD_USD":   0.00006,
    "AUD_JPY":   0.010,
    "NZD_JPY":   0.010,
    "AUD_NZD":   0.00012,
    "GBP_AUD":   0.00018,
    "EUR_AUD":   0.00014,
    "XAG_USD":   0.012,
}


# ================================================================== #
#  Leverage                                                           #
# ================================================================== #
#  Deriv synthetic leverage caps per instrument type.
#  We default to conservative values. Override in .env if your account
#  allows higher leverage and you want to size positions larger.

DERIV_LEVERAGE: Dict[str, int] = {
    # Synthetics — high leverage available but cap conservatively
    "R_10":      100,
    "R_25":      100,
    "R_50":      100,
    "R_75":      50,    # Vol 75 is volatile — cap at 50x
    "R_100":     50,
    "CRASH300N": 50,
    "CRASH500":  50,
    "CRASH1000": 50,
    "BOOM300N":  50,
    "BOOM500":   50,
    "BOOM1000":  50,
    "STPIDX":    100,
    "JD10":      100,
    "JD25":      100,
    "JD50":      100,
    "JD75":      50,
    "JD100":     50,
    # Forex on Deriv
    "GBP_USD":   30,
    "EUR_USD":   30,
    "USD_JPY":   30,
    "GBP_JPY":   20,
    "EUR_JPY":   20,
    "XAU_USD":   20,
    "USD_CAD":   30,
    "USD_CHF":   30,
    "AUD_USD":   30,
    "AUD_JPY":   20,
    "NZD_JPY":   20,
    "AUD_NZD":   20,
    "GBP_AUD":   20,
    "EUR_AUD":   20,
    "XAG_USD":   20,
}


# ================================================================== #
#  Price precision (decimal places for SL/TP rounding)              #
# ================================================================== #

DERIV_PRICE_PRECISION: Dict[str, int] = {
    # Synthetics — prices range from ~100 to ~100,000+
    # Use 3dp for most; Step Index uses 2dp.
    "R_10":      3,
    "R_25":      3,
    "R_50":      3,
    "R_75":      3,
    "R_100":     3,
    "CRASH300N": 3,
    "CRASH500":  3,
    "CRASH1000": 3,
    "BOOM300N":  3,
    "BOOM500":   3,
    "BOOM1000":  3,
    "STPIDX":    2,
    "JD10":      3,
    "JD25":      3,
    "JD50":      3,
    "JD75":      3,
    "JD100":     3,
    # Forex same as OANDA
    "GBP_USD":   5,
    "EUR_USD":   5,
    "USD_JPY":   3,
    "GBP_JPY":   3,
    "EUR_JPY":   3,
    "AUD_JPY":   3,
    "NZD_JPY":   3,
    "XAU_USD":   2,
    "XAG_USD":   3,
    "USD_CAD":   5,
    "USD_CHF":   5,
    "AUD_USD":   5,
    "AUD_NZD":   5,
    "GBP_AUD":   5,
    "EUR_AUD":   5,
}


# ================================================================== #
#  Session windows                                                    #
# ================================================================== #
#  Synthetics trade 24/7 → allow all hours (0-23).
#  Forex pairs keep the same windows as in config.py.

DERIV_SESSION_WINDOWS: Dict[str, list] = {
    # Synthetics — all hours
    "R_10":      [(0, 24)],
    "R_25":      [(0, 24)],
    "R_50":      [(0, 24)],
    "R_75":      [(0, 24)],
    "R_100":     [(0, 24)],
    "CRASH300N": [(0, 24)],
    "CRASH500":  [(0, 24)],
    "CRASH1000": [(0, 24)],
    "BOOM300N":  [(0, 24)],
    "BOOM500":   [(0, 24)],
    "BOOM1000":  [(0, 24)],
    "STPIDX":    [(0, 24)],
    "JD10":      [(0, 24)],
    "JD25":      [(0, 24)],
    "JD50":      [(0, 24)],
    "JD75":      [(0, 24)],
    "JD100":     [(0, 24)],
    # Forex — same quality session windows as config.py
    "GBP_USD":   [(6, 16)],
    "EUR_USD":   [(6, 16)],
    "GBP_JPY":   [(6, 16)],
    "XAU_USD":   [(6, 20)],
    "USD_CAD":   [(12, 21)],
    "USD_CHF":   [(12, 21)],
    "AUD_USD":   [(0, 9), (12, 16)],
    "AUD_JPY":   [(0, 9)],
    "NZD_JPY":   [(0, 9)],
    "AUD_NZD":   [(0, 9)],
    "GBP_AUD":   [(6, 16)],
    "EUR_JPY":   [(0, 9), (6, 16)],
    "EUR_AUD":   [(6, 16)],
    "USD_JPY":   [(0, 9), (12, 21)],
    "XAG_USD":   [(12, 20)],
}


# ================================================================== #
#  Display names (for Telegram / logs)                               #
# ================================================================== #

DERIV_DISPLAY_NAMES: Dict[str, str] = {
    "R_10":      "Vol 10",
    "R_25":      "Vol 25",
    "R_50":      "Vol 50",
    "R_75":      "Vol 75",
    "R_100":     "Vol 100",
    "CRASH300N": "Crash 300",
    "CRASH500":  "Crash 500",
    "CRASH1000": "Crash 1000",
    "BOOM300N":  "Boom 300",
    "BOOM500":   "Boom 500",
    "BOOM1000":  "Boom 1000",
    "STPIDX":    "Step Index",
    "JD10":      "Jump 10",
    "JD25":      "Jump 25",
    "JD50":      "Jump 50",
    "JD75":      "Jump 75",
    "JD100":     "Jump 100",
}

# ================================================================== #
#  MT5 symbol map (canonical name → MT5 display name)               #
#  Also imported by deriv_broker.py                                  #
# ================================================================== #

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
    # Forex — OANDA-style keys → MT5 symbol names
    "GBP_USD": "GBPUSD",  "EUR_USD": "EURUSD",  "USD_JPY": "USDJPY",
    "XAU_USD": "XAUUSD",  "USD_CAD": "USDCAD",  "USD_CHF": "USDCHF",
    "GBP_JPY": "GBPJPY",  "EUR_JPY": "EURJPY",  "AUD_USD": "AUDUSD",
    "AUD_JPY": "AUDJPY",  "NZD_JPY": "NZDJPY",  "AUD_NZD": "AUDNZD",
    "GBP_AUD": "GBPAUD",  "EUR_AUD": "EURAUD",  "XAG_USD": "XAGUSD",
    # frx-prefixed aliases (Deriv WebSocket naming)
    "frxGBPUSD": "GBPUSD", "frxEURUSD": "EURUSD", "frxUSDJPY": "USDJPY",
    "frxXAUUSD": "XAUUSD", "frxUSDCAD": "USDCAD", "frxUSDCHF": "USDCHF",
    "frxGBPJPY": "GBPJPY", "frxEURJPY": "EURJPY", "frxAUDUSD": "AUDUSD",
    "frxAUDJPY": "AUDJPY", "frxNZDJPY": "NZDJPY", "frxAUDNZD": "AUDNZD",
    "frxGBPAUD": "GBPAUD", "frxEURAUD": "EURAUD", "frxXAGUSD": "XAGUSD",
}
# ================================================================== #
#  BotConfig for Deriv mode                                          #
# ================================================================== #

@dataclass
class DerivBotConfig:
    """
    Drop-in replacement for BotConfig (config.py) when running on Deriv.

    Key differences from BotConfig:
      - Default instruments are synthetics + 3 forex pairs
      - SL/TP ATR multiples are slightly wider (synthetics move faster)
      - Session filter is bypassed for synthetics
      - Deriv API token instead of OANDA credentials

    Usage:
        # In main.py, replace:
        #   from config import BotConfig
        # With:
        #   from deriv_config import DerivBotConfig as BotConfig
    """

    # ── Deriv credentials ─────────────────────────────────────────── #
    deriv_api_token: str = os.getenv("DERIV_API_TOKEN", "")
    deriv_app_id:    str = os.getenv("DERIV_APP_ID", "1089")   # public demo ID

    # ── MT5 credentials (for MT5 backend) ─────────────────────────── #
    mt5_login:    int = int(os.getenv("MT5_LOGIN",    0))
    mt5_password: str = os.getenv("MT5_PASSWORD",    "")
    mt5_server:   str = os.getenv("MT5_SERVER",      "Deriv-Demo")
    mt5_path:     str = os.getenv("MT5_PATH",        "")

    # ── Which backend to use ─────────────────────────────────────── #
    # "websocket" or "mt5"
    deriv_backend: str = os.getenv("DERIV_BACKEND", "mt5")

    # ── Active instruments ─────────────────────────────────────────── #
    instruments:  List[str] = field(
        default_factory=lambda: DERIV_INSTRUMENTS
    )
    primary_pair: str = DERIV_PRIMARY_PAIR

    # ── Strategy parameters (same as config.py) ────────────────────── #
    sma_s:         int   = int(os.getenv("SMA_S", 50))
    sma_l:         int   = int(os.getenv("SMA_L", 200))
    con_window:    int   = int(os.getenv("CON_WINDOW", 3))
    boll_sma:      int   = int(os.getenv("BOLL_SMA", 30))
    boll_dev:      float = float(os.getenv("BOLL_DEV", 2.0))
    adx_period:    int   = int(os.getenv("ADX_PERIOD", 14))
    adx_threshold: float = float(os.getenv("ADX_THRESHOLD", 25.0))
    hurst_lags:    int   = int(os.getenv("HURST_LAGS", 30))

    # ── Risk management ────────────────────────────────────────────── #
    # Slightly wider ATR multiples for synthetics (faster movement)
    risk_pct:       float = float(os.getenv("RISK_PCT",       1.0))
    rr_ratio:       float = float(os.getenv("RR_RATIO",       2.0))
    sl_atr_mult:    float = float(os.getenv("SL_ATR_MULT",    1.5))  # wider for synthetics
    tp_atr_mult:    float = float(os.getenv("TP_ATR_MULT",    3.0))  # wider for synthetics
    max_positions:  int   = int(os.getenv("MAX_POSITIONS",    3))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", 5.0))

    # ── Transaction costs ─────────────────────────────────────────── #
    tc:            float = float(os.getenv("TC", 0.00005))   # lower for synthetics
    spread_pips:   float = float(os.getenv("SPREAD_PIPS",   1.0))
    slippage_pips: float = float(os.getenv("SLIPPAGE_PIPS", 0.2))

    # ── Position sizing ────────────────────────────────────────────── #
    fixed_units: int = int(os.getenv("FIXED_UNITS", 0))

    # ── Timeframes ─────────────────────────────────────────────────── #
    context_granularity: str = os.getenv("CONTEXT_GRAN", "H4")
    signal_granularity:  str = os.getenv("SIGNAL_GRAN",  "H1")
    entry_granularity:   str = os.getenv("ENTRY_GRAN",   "M5")

    # ── Paper trader ───────────────────────────────────────────────── #
    initial_balance: float = float(os.getenv("INITIAL_BALANCE", 10_000.0))
    loop_interval:   int   = int(os.getenv("LOOP_INTERVAL", 120))   # 2 min for synthetics

    # ── Historical data ────────────────────────────────────────────── #
    history_bars: int = int(os.getenv("HISTORY_BARS", 2500))
    h4_bars:      int = int(os.getenv("H4_BARS", 500))
    h1_bars:      int = int(os.getenv("H1_BARS", 500))
    m5_bars:      int = int(os.getenv("M5_BARS", 288))

    # ── ML session model ───────────────────────────────────────────── #
    ml_feature_lags:  int   = int(os.getenv("ML_FEATURE_LAGS", 5))
    ml_min_hit_ratio: float = float(os.getenv("ML_MIN_HIT_RATIO", 0.52))
    ml_h1_bars:       int   = int(os.getenv("ML_H1_BARS", 3000))

    # ── Optimization ranges (same as config.py) ────────────────────── #
    sma_s_range:    Tuple[int, int, int] = (10,  60,  10)
    sma_l_range:    Tuple[int, int, int] = (100, 300, 50)
    con_range:      Tuple[int, int, int] = (1,   15,  1)
    boll_sma_range: Tuple[int, int, int] = (10,  60,  10)
    boll_dev_range: Tuple[int, int, int] = (2,   8,   1)
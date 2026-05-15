"""
config.py
Central configuration for the OANDA-only forex bot.
All tuneable parameters — override via .env file.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List
from dotenv import load_dotenv

load_dotenv()


# ================================================================== #
#  Supported instruments — ordered by priority (waterfall)           #
# ================================================================== #

PRIMARY_PAIR = "GBP_USD"

INSTRUMENTS = [
    "GBP_USD",
    "EUR_USD",
    "GBP_JPY",
    "XAU_USD",
    "USD_CAD",
    "USD_CHF",
    "XAG_USD",
    "AUD_JPY",
    "NZD_JPY",
    "AUD_USD",
    "AUD_NZD",
    "GBP_AUD",
    "EUR_JPY",
    "EUR_AUD",
    "USD_JPY",
]
GRANULARITY_H4 = "H4"
GRANULARITY_H1 = "H1"
GRANULARITY_M5 = "M5"

# ── Pip sizes per instrument ─────────────────────────────────────── #
PIP_SIZE = {
    "GBP_USD":    0.0001,
    "EUR_USD":    0.0001,
    "GBP_JPY":    0.01,
    "XAU_USD":    0.1,
    "USD_CAD":    0.0001,
    "USD_CHF":    0.0001,
    "XAG_USD":    0.01,
    "AUD_JPY":    0.01,
    "NZD_JPY":    0.01,
    "AUD_USD":    0.0001,
    "AUD_NZD":    0.0001,
    "GBP_AUD":    0.0001,
    "EUR_JPY":    0.01,
    "EUR_AUD":    0.0001,
    "USD_JPY":    0.01,
}

SPREAD = {
    "GBP_USD":    0.00015,
    "EUR_USD":    0.00010,
    "GBP_JPY":    0.030,
    "XAU_USD":    0.50,
    "USD_CAD":    0.00015,
    "USD_CHF":    0.00015,
    "XAG_USD":    0.03,
    "AUD_JPY":    0.020,
    "NZD_JPY":    0.020,
    "AUD_USD":    0.00013,
    "AUD_NZD":    0.00025,
    "GBP_AUD":    0.00035,
    "EUR_JPY":    0.020,
    "EUR_AUD":    0.00030,
    "USD_JPY":    0.020,
}

SLIPPAGE = {
    "GBP_USD":    0.00005,
    "EUR_USD":    0.00005,
    "GBP_JPY":    0.010,
    "XAU_USD":    0.20,
    "USD_CAD":    0.00008,
    "USD_CHF":    0.00008,
    "XAG_USD":    0.01,
    "AUD_JPY":    0.008,
    "NZD_JPY":    0.008,
    "AUD_USD":    0.00005,
    "AUD_NZD":    0.00010,
    "GBP_AUD":    0.00015,
    "EUR_JPY":    0.008,
    "EUR_AUD":    0.00012,
    "USD_JPY":    0.008,
}

LEVERAGE = {
    "GBP_USD":    30,
    "EUR_USD":    30,
    "GBP_JPY":    20,
    "XAU_USD":    20,
    "USD_CAD":    30,
    "USD_CHF":    30,
    "XAG_USD":    20,
    "AUD_JPY":    20,
    "NZD_JPY":    20,
    "AUD_USD":    30,
    "AUD_NZD":    20,
    "GBP_AUD":    20,
    "EUR_JPY":    20,
    "EUR_AUD":    20,
    "USD_JPY":    30,
}

SESSION_WINDOWS = {
    "GBP_USD":    [(6, 16)],
    "EUR_USD":    [(6, 16)],
    "GBP_JPY":    [(6, 16)],
    "XAU_USD":    [(6, 20)],
    "USD_CAD":    [(12, 21)],
    "USD_CHF":    [(12, 21)],
    "XAG_USD":    [(12, 20)],
    "AUD_JPY":    [(0, 9)],
    "NZD_JPY":    [(0, 9)],
    "AUD_USD":    [(0, 9), (12, 16)],
    "AUD_NZD":    [(0, 9)],
    "GBP_AUD":    [(6, 16)],
    "EUR_JPY":    [(0, 9), (6, 16)],
    "EUR_AUD":    [(6, 16)],
    "USD_JPY":    [(0, 9), (12, 21)],
}


@dataclass
class BotConfig:

    # ── OANDA credentials ────────────────────────────────────────── #
    oanda_api_key:  str  = os.getenv("OANDA_API_KEY", "")
    oanda_account:  str  = os.getenv("OANDA_ACCOUNT_ID", "")
    oanda_practice: bool = os.getenv("OANDA_PRACTICE", "true").lower() == "true"

    # ── Active instruments ────────────────────────────────────────── #
    instruments:  List[str] = field(default_factory=lambda: INSTRUMENTS)
    primary_pair: str       = PRIMARY_PAIR

    # ── Transaction costs ─────────────────────────────────────────── #
    tc: float = float(os.getenv("TC", 0.00007))

    # ── Strategy parameters ───────────────────────────────────────── #
    sma_s:         int   = int(os.getenv("SMA_S", 50))
    sma_l:         int   = int(os.getenv("SMA_L", 200))
    con_window:    int   = int(os.getenv("CON_WINDOW", 3))
    boll_sma:      int   = int(os.getenv("BOLL_SMA", 30))
    boll_dev:      float = float(os.getenv("BOLL_DEV", 2.0))
    adx_period:    int   = int(os.getenv("ADX_PERIOD", 14))
    adx_threshold: float = float(os.getenv("ADX_THRESHOLD", 25.0))
    hurst_lags:    int   = int(os.getenv("HURST_LAGS", 30))

    # ── Risk management ───────────────────────────────────────────── #
    risk_pct:       float = float(os.getenv("RISK_PCT", 1.0))
    rr_ratio:       float = float(os.getenv("RR_RATIO", 2.0))
    sl_atr_mult:    float = float(os.getenv("SL_ATR_MULT", 1.0))
    tp_atr_mult:    float = float(os.getenv("TP_ATR_MULT", 2.0))
    max_positions:  int   = int(os.getenv("MAX_POSITIONS", 5))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", 5.0))

    # ── Execution cost model (pips) ───────────────────────────────── #
    # Used by PaperTrader and _print_config. Expressed in pips so they
    # are instrument-agnostic at the config level; the risk manager
    # converts to price using PIP_SIZE when building a trade setup.
    spread_pips:   float = float(os.getenv("SPREAD_PIPS", 1.5))
    slippage_pips: float = float(os.getenv("SLIPPAGE_PIPS", 0.5))

    # ── Fixed unit override ───────────────────────────────────────── #
    # 0 = use ATR-based position sizing (recommended).
    # Set to e.g. 10000 to force a fixed lot size for testing.
    fixed_units: int = int(os.getenv("FIXED_UNITS", 0))

    # ── Timeframe granularities ───────────────────────────────────── #
    context_granularity: str = os.getenv("CONTEXT_GRAN", "H4")
    signal_granularity:  str = os.getenv("SIGNAL_GRAN",  "H1")
    entry_granularity:   str = os.getenv("ENTRY_GRAN",   "M5")

    # ── Paper trader ──────────────────────────────────────────────── #
    initial_balance: float = float(os.getenv("INITIAL_BALANCE", 10_000.0))
    loop_interval:   int   = int(os.getenv("LOOP_INTERVAL", 120))

    # ── Historical data bars per timeframe ────────────────────────── #
    history_bars: int = int(os.getenv("HISTORY_BARS", 2500))
    h4_bars:      int = int(os.getenv("H4_BARS", 500))
    h1_bars:      int = int(os.getenv("H1_BARS", 500))
    m5_bars:      int = int(os.getenv("M5_BARS", 288))

    # ── Optimization search ranges ────────────────────────────────── #
    sma_s_range:    Tuple[int, int, int] = (10,  60,  10)
    sma_l_range:    Tuple[int, int, int] = (100, 300, 50)
    con_range:      Tuple[int, int, int] = (1,   15,  1)
    boll_sma_range: Tuple[int, int, int] = (10,  60,  10)
    boll_dev_range: Tuple[int, int, int] = (2,   8,   1)

    # ── ML session model ──────────────────────────────────────────── #
    ml_feature_lags:  int   = int(os.getenv("ML_FEATURE_LAGS", 5))
    ml_min_hit_ratio: float = float(os.getenv("ML_MIN_HIT_RATIO", 0.52))
    ml_h1_bars:       int   = int(os.getenv("ML_H1_BARS", 3000))
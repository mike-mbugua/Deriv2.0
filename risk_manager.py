"""
risk_manager.py  (v4 — instrument-agnostic: forex + Deriv synthetics)
──────────────────────────────────────────────────────────────────────
CHANGES FROM v3
───────────────
  ✅ PIP_SIZE / SPREAD / SLIPPAGE / LEVERAGE now import from config
     but can be monkey-patched at runtime by main.py for Deriv mode
     (see _patch_risk_manager_tables in main.py)
  ✅ _round_price falls back to DERIV_PRICE_PRECISION for synthetics
  ✅ pip_value_per_unit handles synthetics (no cross-rate adjustment)
  ✅ All other v3 fixes retained (NAV, sizing formula, margin check etc.)

KEY MATHS (unchanged from v3)
──────────────────────────────
  risk_amount  = account_nav × 0.5%
  sl_dist      = sl_atr_mult × ATR(14)
  tp_dist      = sl_dist × rr_ratio  → 1:2 RR default
  units        = risk_amount / (sl_pips × pip_value_per_unit)
  margin_req   = (units × entry_price) / leverage
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# These are importable from config and can be monkey-patched by main.py
# for Deriv mode (see _patch_risk_manager_tables).
from config import PIP_SIZE, SPREAD, SLIPPAGE, LEVERAGE

logger = logging.getLogger(__name__)


# ── Swing point helpers (unchanged) ──────────────────────────────── #

def _find_swing_high(df: pd.DataFrame, lookback: int = 50) -> float:
    if "high" not in df.columns or len(df) < lookback:
        return 0.0
    highs  = df["high"].iloc[-lookback:]
    result = 0.0
    for i in range(3, len(highs) - 3):
        val = highs.iloc[i]
        if val == highs.iloc[i-3:i+4].max():
            result = float(val)
    return result


def _find_swing_low(df: pd.DataFrame, lookback: int = 50) -> float:
    if "low" not in df.columns or len(df) < lookback:
        return 0.0
    lows   = df["low"].iloc[-lookback:]
    result = float("inf")
    for i in range(3, len(lows) - 3):
        val = lows.iloc[i]
        if val == lows.iloc[i-3:i+4].min():
            result = float(val)
    return result if result != float("inf") else 0.0


def _is_near_ath(df: pd.DataFrame, threshold: float = 0.02) -> bool:
    if "high" not in df.columns or df.empty:
        return False
    ath           = float(df["high"].max())
    current_price = float(df["price"].iloc[-1])
    return (ath - current_price) / ath < threshold


def _is_near_atl(df: pd.DataFrame, threshold: float = 0.02) -> bool:
    if "low" not in df.columns or df.empty:
        return False
    atl           = float(df["low"].min())
    current_price = float(df["price"].iloc[-1])
    return (current_price - atl) / atl < threshold


# ── Price precision (instrument-aware rounding) ───────────────────── #
# OANDA forex precision (decimal places accepted without error)

PRICE_PRECISION = {
    "GBP_USD": 5, "EUR_USD": 5, "USD_JPY": 3, "NZD_JPY": 3,
    "XAU_USD": 2, "EUR_GBP": 5, "GBP_CHF": 5, "USD_CAD": 5,
    "USD_CHF": 5, "XAG_USD": 3, "AUD_JPY": 3, "AUD_USD": 5,
    "AUD_NZD": 5, "GBP_AUD": 5, "GBP_NZD": 5, "EUR_JPY": 3,
    "EUR_AUD": 5, "GBP_JPY": 3, "USD_JPY": 3,
}

# Deriv synthetic precision (also used for frx forex via Deriv)
_DERIV_PRECISION = {
    "R_10": 3, "R_25": 3, "R_50": 3, "R_75": 3, "R_100": 3,
    "CRASH300N": 3, "CRASH500": 3, "CRASH1000": 3,
    "BOOM300N": 3, "BOOM500": 3, "BOOM1000": 3,
    "STPIDX": 2,
    "JD10": 3, "JD25": 3, "JD50": 3, "JD75": 3, "JD100": 3,
    # frx forex on Deriv (same as OANDA but routed differently)
    "frxGBPUSD": 5, "frxEURUSD": 5, "frxUSDJPY": 3,
    "frxXAUUSD": 2, "frxGBPJPY": 3, "frxEURJPY": 3,
    "frxAUDUSD": 5, "frxAUDJPY": 3, "frxNZDJPY": 3,
    "frxUSDCAD": 5, "frxUSDCHF": 5,
}

# Merge Deriv into the combined lookup
PRICE_PRECISION.update(_DERIV_PRECISION)


def _round_price(instrument: str, price: float) -> float:
    decimals = PRICE_PRECISION.get(instrument, 5)
    return round(price, decimals)


# ── TradeSetup dataclass ──────────────────────────────────────────── #

@dataclass
class TradeSetup:
    instrument:    str
    direction:     int
    entry_price:   float
    sl_price:      float
    tp_price:      float
    units:         int
    risk_amount:   float
    sl_pips:       float
    tp_pips:       float
    rr_ratio:      float
    atr:           float
    spread_cost:   float
    slippage_cost: float
    margin_required: float
    confidence:    float = 0.5

    def __str__(self):
        side = "LONG" if self.direction == 1 else "SHORT"
        prec = PRICE_PRECISION.get(self.instrument, 5)
        fmt  = f".{prec}f"
        return (
            f"[{self.instrument}] {side} @ {self.entry_price:{fmt}} | "
            f"SL={self.sl_price:{fmt}} ({self.sl_pips:.1f}p) | "
            f"TP={self.tp_price:{fmt}} ({self.tp_pips:.1f}p) | "
            f"RR={self.rr_ratio:.2f} | Units={self.units:,} | "
            f"Risk=${self.risk_amount:.2f} | Margin=${self.margin_required:.2f}"
        )

    def verify(self) -> bool:
        ok = True
        if self.direction == 1:
            if self.sl_price >= self.entry_price:
                logger.error(f"SETUP ERROR {self.instrument}: SL {self.sl_price} >= entry {self.entry_price}")
                ok = False
            if self.tp_price <= self.entry_price:
                logger.error(f"SETUP ERROR {self.instrument}: TP {self.tp_price} <= entry {self.entry_price}")
                ok = False
        else:
            if self.sl_price <= self.entry_price:
                logger.error(f"SETUP ERROR {self.instrument}: SL {self.sl_price} <= entry {self.entry_price}")
                ok = False
            if self.tp_price >= self.entry_price:
                logger.error(f"SETUP ERROR {self.instrument}: TP {self.tp_price} >= entry {self.entry_price}")
                ok = False
        if self.rr_ratio < 1.9:
            logger.warning(f"SETUP WARN {self.instrument}: RR={self.rr_ratio:.2f} below 1:2 target")
        if self.units == 0:
            logger.error(f"SETUP ERROR {self.instrument}: units=0")
            ok = False
        return ok


# ── RiskManager ──────────────────────────────────────────────────── #

class RiskManager:
    """
    Instrument-agnostic risk manager.
    Works for OANDA forex, Deriv forex, and Deriv synthetic indices.

    The only instrument-specific data it uses:
      PIP_SIZE, SPREAD, SLIPPAGE, LEVERAGE
    These dicts live in config.py and are monkey-patched by main.py
    for Deriv mode — no code changes needed here.
    """

    def __init__(
        self,
        risk_pct:       float = 1.0,
        rr_ratio:       float = 2.0,
        sl_atr_mult:    float = 1.0,
        max_positions:  int   = 2,
        max_daily_loss: float = 5.0,
    ):
        self.risk_pct        = risk_pct
        self.rr_ratio        = rr_ratio
        self.sl_atr_mult     = sl_atr_mult
        self.max_positions   = max_positions
        self.max_daily_loss  = max_daily_loss

        self._daily_open_nav:     float = None
        self._daily_date:         str   = ""
        self._daily_loss_tripped: bool  = False

    # ── ATR ──────────────────────────────────────────────────────── #

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period + 1:
            return 0.0
        if not {"high", "low", "price"}.issubset(df.columns):
            closes = df["price"].dropna()
            pct    = closes.pct_change().abs().rolling(period).mean()
            val    = pct.dropna()
            return float(val.iloc[-1] * closes.iloc[-1]) if not val.empty else 0.0
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["price"].shift(1)).abs(),
            (df["low"]  - df["price"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period).mean()
        val = atr.dropna()
        return float(val.iloc[-1]) if not val.empty else 0.0

    @staticmethod
    def price_to_pips(price_diff: float, instrument: str) -> float:
        pip = PIP_SIZE.get(instrument, 0.0001)
        return abs(price_diff) / pip

    # ── Build trade setup ─────────────────────────────────────────── #

    def build_trade_setup(
        self,
        instrument:   str,
        direction:    int,
        entry_price:  float,
        account_nav:  float,
        price_df:     pd.DataFrame,
        h4_df:        pd.DataFrame = None,
        atr_period:   int   = 14,
        confidence:   float = 0.5,
        swing_tp:     float = 0.0,
        liq_zones:    list  = None,
        near_ath:     bool  = False,
        near_atl:     bool  = False,
    ) -> Optional[TradeSetup]:
        atr = self.compute_atr(price_df, atr_period)
        if atr <= 0:
            logger.warning(f"ATR is zero for {instrument} — skipping.")
            return None

        # Fetch instrument-specific parameters
        # (these dicts are monkey-patched for Deriv by main.py)
        pip      = PIP_SIZE.get(instrument, 0.0001)
        spread   = SPREAD.get(instrument, 0.0001)
        slippage = SLIPPAGE.get(instrument, 0.00005)
        leverage = LEVERAGE.get(instrument, 30)

        # ── SL / TP distances ─────────────────────────────────── #
        sl_dist_raw    = self.sl_atr_mult * atr
        tx_cost        = (spread / 2.0) + slippage
        actual_sl_dist = sl_dist_raw + tx_cost
        actual_tp_dist = actual_sl_dist * self.rr_ratio

        liq_zones = liq_zones or []

        # ATH/ATL: widen SL
        if (near_ath and direction == 1) or (near_atl and direction == -1):
            actual_sl_dist = (sl_dist_raw * 1.5) + tx_cost
            actual_tp_dist = actual_sl_dist * self.rr_ratio
            logger.info(f"[{instrument}] ATH/ATL context — SL widened to 1.5×ATR")

        # Swing TP override
        elif swing_tp > 0:
            tp_dist_from_swing = abs(swing_tp - entry_price)
            rr_from_swing      = tp_dist_from_swing / (actual_sl_dist + 1e-9)
            if rr_from_swing >= 1.5:
                actual_tp_dist = tp_dist_from_swing
                logger.info(f"[{instrument}] Using swing TP={swing_tp:.5f} RR=1:{rr_from_swing:.1f}")
            else:
                actual_tp_dist = actual_sl_dist * self.rr_ratio

        if actual_sl_dist <= 0:
            logger.error(f"SL distance is zero for {instrument}.")
            return None

        # ── Raw SL / TP prices ─────────────────────────────────── #
        if direction == 1:
            sl_price = entry_price - actual_sl_dist
            tp_price = entry_price + actual_tp_dist
        else:
            sl_price = entry_price + actual_sl_dist
            tp_price = entry_price - actual_tp_dist

        # ── Liquidity zone adjustments ─────────────────────────── #
        for z in liq_zones:
            if z.bottom <= entry_price <= z.top and not z.swept:
                logger.warning(f"[{instrument}] Entry inside active zone — skipping")
                return None

        for z in liq_zones:
            if z.bottom <= sl_price <= z.top:
                if direction == 1:
                    sl_price = round(z.bottom - atr * 0.2, 5)
                else:
                    sl_price = round(z.top + atr * 0.2, 5)
                break

        for z in sorted(liq_zones, key=lambda x: abs(x.level - entry_price)):
            if z.swept:
                continue
            if direction == 1 and entry_price < z.level < tp_price:
                candidate = round(z.bottom - atr * 0.1, 5)
                new_rr    = abs(candidate - entry_price) / (abs(sl_price - entry_price) + 1e-9)
                if new_rr >= 1.5:
                    tp_price = candidate
                break
            elif direction == -1 and tp_price < z.level < entry_price:
                candidate = round(z.top + atr * 0.1, 5)
                new_rr    = abs(candidate - entry_price) / (abs(sl_price - entry_price) + 1e-9)
                if new_rr >= 1.5:
                    tp_price = candidate
                break

        # ── Position sizing ─────────────────────────────────────── #
        risk_amount        = account_nav * 0.005   # hard 0.5% risk
        pip_value_per_unit = self._pip_value_per_unit(instrument, entry_price)
        sl_pips_actual     = actual_sl_dist / pip

        if sl_pips_actual <= 0 or pip_value_per_unit <= 0:
            logger.error(f"Invalid pip calculation for {instrument}.")
            return None

        # Lot cap scaled by account size
        if account_nav < 50:
            max_lots = 0.05
        elif account_nav < 500:
            max_lots = 0.15
        elif account_nav < 5_000:
            max_lots = 0.50
        else:
            max_lots = 1.00

        pip_value_per_lot = pip_value_per_unit * 100_000
        if pip_value_per_lot > 0 and sl_pips_actual > 0:
            calculated_lots = risk_amount / (sl_pips_actual * pip_value_per_lot)
        else:
            calculated_lots = 0.01

        lots  = min(calculated_lots, max_lots)
        lots  = max(0.05, round(lots, 2))
        units = int(round(lots * 100_000, 0))

        logger.info(
            f"[SIZING] {instrument} | NAV=${account_nav:.2f} | "
            f"Risk=${risk_amount:.2f} | SL={sl_pips_actual:.1f}p | "
            f"Calc={calculated_lots:.3f}L | Cap={max_lots}L | "
            f"Final={lots:.2f}L ({units:,} units)"
        )

        # ── Margin check ─────────────────────────────────────────── #
        margin_limit    = 0.50 if account_nav < 500 else 0.90
        margin_required = (units * entry_price) / leverage
        if margin_required > account_nav * margin_limit:
            logger.warning(f"{instrument}: reducing units for margin limit.")
            units           = max(1, int((account_nav * margin_limit * leverage) / entry_price))
            margin_required = (units * entry_price) / leverage
        if margin_required > account_nav * 0.90:
            logger.error(f"{instrument}: margin_required exceeds 90% of NAV — rejected.")
            return None

        # ── Pips and RR ──────────────────────────────────────────── #
        sl_pips = self.price_to_pips(abs(entry_price - sl_price), instrument)
        tp_pips = self.price_to_pips(abs(entry_price - tp_price), instrument)
        rr      = round(tp_pips / (sl_pips + 1e-9), 2)

        signed_units  = units if direction == 1 else -units
        entry_rounded = _round_price(instrument, entry_price)
        sl_rounded    = _round_price(instrument, sl_price)
        tp_rounded    = _round_price(instrument, tp_price)

        setup = TradeSetup(
            instrument      = instrument,
            direction       = direction,
            entry_price     = entry_rounded,
            sl_price        = sl_rounded,
            tp_price        = tp_rounded,
            units           = signed_units,
            risk_amount     = round(risk_amount, 2),
            sl_pips         = round(sl_pips, 1),
            tp_pips         = round(tp_pips, 1),
            rr_ratio        = rr,
            atr             = round(atr, 5),
            spread_cost     = round(spread, 6),
            slippage_cost   = round(slippage, 6),
            margin_required = round(margin_required, 2),
            confidence      = confidence,
        )

        if not setup.verify():
            logger.error(f"Setup verification FAILED for {instrument}.")
            return None

        logger.info(str(setup))
        return setup

    # ── Pip value per unit ─────────────────────────────────────────── #

    @staticmethod
    def _pip_value_per_unit(instrument: str, price: float) -> float:
        """
        Returns the dollar value of 1 pip movement per 1 unit.

        Synthetics: price is an index level (e.g. R_75 ≈ 750,000).
        1 pip = pip_size = 0.001.  1 unit = 1 contract.
        pip_value = pip_size / price  (same logic as USD-base forex).

        Forex USD-quote (GBP_USD, EUR_USD):
          pip_value = pip_size  (0.0001 per unit)
        Forex USD-base (USD_JPY):
          pip_value = pip_size / price
        XAU_USD:
          pip_value = pip_size  (0.01 per oz)
        """
        pip = PIP_SIZE.get(instrument, 0.0001)

        # Deriv synthetics — price is a large index number
        SYNTHETICS = (
            "R_10", "R_25", "R_50", "R_75", "R_100",
            "CRASH300N", "CRASH500", "CRASH1000",
            "BOOM300N", "BOOM500", "BOOM1000",
            "STPIDX", "JD10", "JD25", "JD50", "JD75", "JD100",
        )
        if instrument in SYNTHETICS:
            return pip / (price + 1e-9)

        if instrument == "USD_JPY":
            return pip / price
        elif instrument in ("GBP_JPY", "NZD_JPY", "EUR_JPY", "AUD_JPY"):
            return pip / price
        elif instrument == "XAU_USD":
            return pip
        else:
            return pip   # USD-quoted standard

    # ── Daily loss circuit breaker ─────────────────────────────────── #

    def check_daily_loss(self, current_nav: float) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date         = today
            self._daily_open_nav     = current_nav
            self._daily_loss_tripped = False
            logger.info(f"New trading day: opening NAV = {current_nav:.2f}")
        if self._daily_loss_tripped:
            return False
        if self._daily_open_nav and self._daily_open_nav > 0:
            loss_pct = (self._daily_open_nav - current_nav) / self._daily_open_nav * 100
            if loss_pct >= self.max_daily_loss:
                logger.warning(f"Daily loss limit breached: {loss_pct:.2f}%. Halted.")
                self._daily_loss_tripped = True
                return False
        return True

    def can_open_new(self, open_position_count: int) -> bool:
        if open_position_count >= self.max_positions:
            logger.info(f"Max positions ({self.max_positions}) reached.")
            return False
        return True

    # ── Trailing stop ──────────────────────────────────────────────── #

    @staticmethod
    def trail_sl(
        direction:     int,
        current_price: float,
        current_sl:    float,
        entry_price:   float,
        tp_price:      float,
        atr:           float,
        trail_mult:    float = 1.5,
    ) -> Tuple[float, str]:
        tp_dist    = abs(tp_price - entry_price)
        halfway    = (entry_price + tp_dist * 0.5 if direction == 1
                      else entry_price - tp_dist * 0.5)
        trail_dist = trail_mult * atr
        if direction == 1:
            if current_price >= halfway:
                new_sl = max(current_price - trail_dist, current_sl)
                return new_sl, "trail"
            elif current_price >= entry_price + atr:
                return max(entry_price, current_sl), "breakeven"
            return current_sl, "hold"
        else:
            if current_price <= halfway:
                new_sl = min(current_price + trail_dist, current_sl)
                return new_sl, "trail"
            elif current_price <= entry_price - atr:
                return min(entry_price, current_sl), "breakeven"
            return current_sl, "hold"
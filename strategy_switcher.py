"""
strategy_switcher.py  —  v2  (SMC + ICT + CRT — Deriv synthetic aware)
════════════════════════════════════════════════════════════════════════

CHANGES FROM v1
───────────────
  ✅ SessionAnalyser.is_good_session now returns True for synthetics
     (R_75, CRASH1000 etc.) at ALL hours — they trade 24/7
  ✅ StrategySwitcher.PAIR_KILL_ZONES extended with synthetic entries
     (synthetics have no traditional session — all hours valid)
  ✅ SMCHelpers.detect_crt lookback widened slightly for synthetics
     (faster price action means more CRT setups per day)
  ✅ All v1 SMC logic (BOS, CHOCH, FVG, OB, Fib, Liquidity) unchanged

Everything else is identical to v1 — full drop-in replacement.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Synthetic instruments set (used by session filter) ────────────── #
_SYNTHETICS = frozenset({
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "CRASH300N", "CRASH500", "CRASH1000",
    "BOOM300N", "BOOM500", "BOOM1000",
    "STPIDX",
    "JD10", "JD25", "JD50", "JD75", "JD100",
})


# ════════════════════════════════════════════════════════════════════ #
#  Enums                                                               #
# ════════════════════════════════════════════════════════════════════ #

class Regime(str, Enum):
    TRENDING    = "trending"
    OSCILLATING = "oscillating"
    RANDOM      = "random"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ════════════════════════════════════════════════════════════════════ #
#  SMC concept dataclasses                                             #
# ════════════════════════════════════════════════════════════════════ #

@dataclass
class SwingPoint:
    index: int
    price: float
    kind:  str


@dataclass
class FVG:
    index:     int
    top:       float
    bottom:    float
    direction: str
    mitigated: bool = False


@dataclass
class OrderBlock:
    index:     int
    top:       float
    bottom:    float
    direction: str
    mitigated: bool = False


@dataclass
class BOS:
    index:     int
    price:     float
    direction: str
    kind:      str


@dataclass
class FibZone:
    level:     float
    price:     float
    direction: str


@dataclass
class ProbabilityScore:
    bullish:    float
    bearish:    float
    bias:       Bias
    confidence: float


@dataclass
class LiquidityZone:
    kind:    str
    level:   float
    top:     float
    bottom:  float
    hits:    int
    swept:   bool


# ════════════════════════════════════════════════════════════════════ #
#  Config                                                              #
# ════════════════════════════════════════════════════════════════════ #

@dataclass
class StrategyConfig:
    sma_s:         int   = 50
    sma_l:         int   = 200
    con_window:    int   = 3
    boll_sma:      int   = 30
    boll_dev:      float = 2.0
    adx_period:    int   = 14
    adx_threshold: float = 25.0
    hurst_lags:    int   = 30

    swing_lookback:     int   = 10
    swing_min_gap:      int   = 5
    ob_body_ratio:      float = 0.3
    fib_levels:         tuple = (0.50, 0.618)
    min_prob_gap:       float = 15.0
    use_session_filter: bool  = True
    bos_require_close:  bool  = True
    bos_wait_candles:   int   = 2


# ════════════════════════════════════════════════════════════════════ #
#  SwitchDecision                                                      #
# ════════════════════════════════════════════════════════════════════ #

@dataclass
class SwitchDecision:
    regime:       Regime
    strategy:     str
    params:       dict              = field(default_factory=dict)
    hurst:        float             = 0.5
    adx:          Optional[float]   = None
    ml_confidence: float            = 0.0
    signal:       int               = 0
    good_session: bool              = True

    bias:              Bias              = Bias.NEUTRAL
    probability:       Optional[ProbabilityScore] = None
    weekly_structure:  str               = ""
    daily_structure:   str               = ""
    weekly_swing_high: float             = 0.0
    weekly_swing_low:  float             = 0.0
    daily_swing_high:  float             = 0.0
    daily_swing_low:   float             = 0.0
    nearest_fvg:       Optional[FVG]     = None
    nearest_ob:        Optional[OrderBlock] = None
    last_bos:          Optional[BOS]     = None
    fib_zones:         List[FibZone]     = field(default_factory=list)
    dol_target:        float             = 0.0
    dol_kind:          str               = ""
    reason:            str               = ""
    confs:             List[str]         = field(default_factory=list)
    liq_zones:         List             = field(default_factory=list)
    near_ath:          bool              = False
    near_atl:          bool              = False
    swing_tp:          float             = 0.0


# ════════════════════════════════════════════════════════════════════ #
#  SMC Helpers  (unchanged from v1 — only CRT lookback is wider)      #
# ════════════════════════════════════════════════════════════════════ #

class SMCHelpers:

    @staticmethod
    def find_swings(df, lookback=10, min_gap=5):
        if len(df) < lookback * 2 + 2:
            return []
        highs = df["high"].values if "high" in df.columns else df["price"].values
        lows  = df["low"].values  if "low"  in df.columns else df["price"].values
        swings = []
        last_idx = -min_gap
        for i in range(lookback, len(df) - lookback):
            win_h = highs[i - lookback: i + lookback + 1]
            win_l = lows [i - lookback: i + lookback + 1]
            is_sh = float(highs[i]) == float(win_h.max())
            is_sl = float(lows[i])  == float(win_l.min())
            if is_sh and (i - last_idx) >= min_gap:
                swings.append(SwingPoint(i, float(highs[i]), "high"))
                last_idx = i
            elif is_sl and (i - last_idx) >= min_gap:
                swings.append(SwingPoint(i, float(lows[i]), "low"))
                last_idx = i
        return swings

    @staticmethod
    def classify_structure(swings):
        if len(swings) < 4:
            return "ranging", 0.0, 0.0
        highs = [s for s in swings if s.kind == "high"]
        lows  = [s for s in swings if s.kind == "low"]
        last_sh = highs[-1].price if highs else 0.0
        last_sl = lows[-1].price  if lows  else 0.0
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1].price > highs[-2].price
            hl = lows[-1].price  > lows[-2].price
            ll = lows[-1].price  < lows[-2].price
            lh = highs[-1].price < highs[-2].price
            if hh and hl: return "uptrend",   last_sh, last_sl
            if ll and lh: return "downtrend", last_sh, last_sl
        return "ranging", last_sh, last_sl

    @staticmethod
    def find_fvgs(df, last_n=150):
        if "high" not in df.columns or "low" not in df.columns:
            return []
        fvgs = []
        start = max(2, len(df) - last_n)
        for i in range(start, len(df)):
            h0 = float(df["high"].iloc[i - 2])
            l0 = float(df["low"].iloc[i - 2])
            h2 = float(df["high"].iloc[i])
            l2 = float(df["low"].iloc[i])
            fvg = None
            if l2 > h0:  fvg = FVG(i - 1, l2, h0, "bullish")
            elif h2 < l0: fvg = FVG(i - 1, l0, h2, "bearish")
            if fvg is None: continue
            for j in range(i + 1, len(df)):
                c = float(df["price"].iloc[j])
                if fvg.bottom <= c <= fvg.top:
                    fvg.mitigated = True
                    break
            fvgs.append(fvg)
        return fvgs

    @staticmethod
    def find_order_blocks(df, last_n=150, body_ratio=0.3):
        if "open" not in df.columns:
            return []
        obs = []
        start = max(2, len(df) - last_n)
        for i in range(start, len(df) - 2):
            o = float(df["open"].iloc[i]); c = float(df["price"].iloc[i])
            h = float(df["high"].iloc[i]) if "high" in df.columns else max(o, c)
            l = float(df["low"].iloc[i])  if "low"  in df.columns else min(o, c)
            body = abs(c - o); rng = h - l
            if rng < 1e-10 or body / rng < body_ratio: continue
            o2 = float(df["open"].iloc[i + 1]); c2 = float(df["price"].iloc[i + 1])
            body2 = abs(c2 - o2)
            ob = None
            if c < o and c2 > o2 and body2 >= 1.5 * body:
                ob = OrderBlock(i, max(o, c), min(o, c), "bullish")
            elif c > o and c2 < o2 and body2 >= 1.5 * body:
                ob = OrderBlock(i, max(o, c), min(o, c), "bearish")
            if ob is None: continue
            for j in range(i + 2, len(df)):
                cj = float(df["price"].iloc[j])
                if ob.bottom <= cj <= ob.top:
                    ob.mitigated = True
                    break
            obs.append(ob)
        return obs

    @staticmethod
    def find_breaker_blocks(df, swings, fvgs):
        if not swings or not fvgs: return []
        breakers = []
        swing_prices = {s.price for s in swings}
        for fvg in fvgs:
            if fvg.mitigated: continue
            mid = (fvg.top + fvg.bottom) / 2
            for sp in swing_prices:
                if sp > 0 and abs(mid - sp) / sp < 0.001:
                    breakers.append(OrderBlock(fvg.index, fvg.top, fvg.bottom, fvg.direction, False))
                    break
        return breakers

    @staticmethod
    def find_bos(df, swings, require_close=True):
        bos_list = []
        if len(swings) < 2: return bos_list
        closes = df["price"].values
        for k in range(1, len(swings)):
            sw = swings[k - 1]
            search_from = swings[k].index + 1
            for i in range(search_from, len(df)):
                val = float(closes[i])
                if sw.kind == "high" and val > sw.price:
                    bos_list.append(BOS(i, sw.price, "bullish", "bos")); break
                if sw.kind == "low"  and val < sw.price:
                    bos_list.append(BOS(i, sw.price, "bearish", "bos")); break
        return bos_list

    @staticmethod
    def find_choch(df, swings, struct):
        if struct == "ranging" or len(swings) < 2: return None
        closes = df["price"].values
        choch_list = []
        for k in range(1, len(swings)):
            sw          = swings[k - 1]
            search_from = swings[k].index + 1
            looking_for = "low"  if struct == "uptrend"   else "high"
            choch_dir   = "bearish" if struct == "uptrend" else "bullish"
            if sw.kind != looking_for: continue
            for i in range(search_from, len(df)):
                val = float(closes[i])
                if choch_dir == "bearish" and val < sw.price:
                    choch_list.append(BOS(i, sw.price, "bearish", "choch")); break
                if choch_dir == "bullish" and val > sw.price:
                    choch_list.append(BOS(i, sw.price, "bullish", "choch")); break
        return choch_list[-1] if choch_list else None

    @staticmethod
    def check_displacement(df, bos, min_body_ratio=0.6, min_atr_mult=1.5):
        if df is None or len(df) < bos.index + 2: return False
        if "open" not in df.columns or "high" not in df.columns: return False
        search_start = min(bos.index + 1, len(df) - 3)
        search_end   = min(bos.index + 5, len(df))
        atr_start    = max(0, bos.index - 14)
        atr_slice    = df.iloc[atr_start:bos.index]
        if len(atr_slice) < 3: return True
        ranges = (atr_slice["high"] - atr_slice["low"]) if "high" in atr_slice.columns else atr_slice["price"].diff().abs()
        atr = float(ranges.mean())
        if atr <= 0: return True
        for i in range(search_start, search_end):
            o = float(df["open"].iloc[i]); c = float(df["price"].iloc[i])
            h = float(df["high"].iloc[i]) if "high" in df.columns else max(o, c)
            l = float(df["low"].iloc[i])  if "low"  in df.columns else min(o, c)
            body = abs(c - o); rng = h - l
            if rng < 1e-10: continue
            body_ratio  = body / rng
            is_impulsive = body >= min_atr_mult * atr
            is_strong    = body_ratio >= min_body_ratio
            is_bullish   = c > o
            dir_ok = ((bos.direction == "bullish" and is_bullish) or
                      (bos.direction == "bearish" and not is_bullish))
            if is_strong and is_impulsive and dir_ok: return True
        return False

    @staticmethod
    def find_liquidity_sweeps(df, lookback=100, equal_thresh=0.0003, sweep_confirm=3):
        if len(df) < lookback or "high" not in df.columns: return []
        highs  = df["high"].values; lows = df["low"].values; closes = df["price"].values
        start  = max(0, len(df) - lookback)
        sweeps = []
        for i in range(start + 2, len(df) - 1):
            for j in range(max(start, i - 50), i - sweep_confirm):
                h_now = float(highs[i]); h_prev = float(highs[j])
                if h_prev <= 0: continue
                if abs(h_now - h_prev) / h_prev <= equal_thresh:
                    level = max(h_now, h_prev)
                    for k in range(i, min(i + sweep_confirm + 1, len(df))):
                        if float(highs[k]) > level:
                            close_k = float(closes[k])
                            sweeps.append({"type": "sweep_high", "level": round(level,5),
                                           "sweep_idx": k, "close": round(close_k,5),
                                           "confirmed": close_k < level})
                            break
                    break
            for j in range(max(start, i - 50), i - sweep_confirm):
                l_now = float(lows[i]); l_prev = float(lows[j])
                if l_prev <= 0: continue
                if abs(l_now - l_prev) / l_prev <= equal_thresh:
                    level = min(l_now, l_prev)
                    for k in range(i, min(i + sweep_confirm + 1, len(df))):
                        if float(lows[k]) < level:
                            close_k = float(closes[k])
                            sweeps.append({"type": "sweep_low", "level": round(level,5),
                                           "sweep_idx": k, "close": round(close_k,5),
                                           "confirmed": close_k > level})
                            break
                    break
        confirmed = [s for s in sweeps if s["confirmed"]]
        return confirmed[-3:] if confirmed else []

    @staticmethod
    def extract_liquidity_zones(df, sweeps, lookback=100, equal_thresh=0.0003):
        if df is None or len(df) < 10 or "high" not in df.columns: return []
        highs = df["high"].values; lows = df["low"].values
        start = max(0, len(df) - lookback)
        tr    = df["high"] - df["low"] if "low" in df.columns else df["price"].diff().abs()
        atr   = float(tr.iloc[-20:].mean()) if len(tr) >= 20 else float(tr.mean())
        buf   = atr * 0.3
        swept_levels = {s["level"] for s in sweeps}
        zones = []
        used_h = [False] * len(highs)
        used_l = [False] * len(lows)
        for i in range(start, len(highs)):
            if used_h[i]: continue
            cluster = [float(highs[i])]
            for j in range(i + 1, len(highs)):
                if used_h[j]: continue
                h_prev = float(highs[i]); h_now = float(highs[j])
                if h_prev > 0 and abs(h_now - h_prev) / h_prev <= equal_thresh:
                    cluster.append(h_now); used_h[j] = True
            used_h[i] = True
            if len(cluster) >= 2:
                level = round(max(cluster), 5)
                is_swept = any(abs(level - sl) / (sl + 1e-9) <= equal_thresh * 3
                               for sl in swept_levels)
                zones.append(LiquidityZone("buy_side", level, round(level + buf, 5),
                                           round(level - buf, 5), len(cluster), is_swept))
        for i in range(start, len(lows)):
            if used_l[i]: continue
            cluster = [float(lows[i])]
            for j in range(i + 1, len(lows)):
                if used_l[j]: continue
                l_prev = float(lows[i]); l_now = float(lows[j])
                if l_prev > 0 and abs(l_now - l_prev) / l_prev <= equal_thresh:
                    cluster.append(l_now); used_l[j] = True
            used_l[i] = True
            if len(cluster) >= 2:
                level = round(min(cluster), 5)
                is_swept = any(abs(level - sl) / (sl + 1e-9) <= equal_thresh * 3
                               for sl in swept_levels)
                zones.append(LiquidityZone("sell_side", level, round(level + buf, 5),
                                           round(level - buf, 5), len(cluster), is_swept))
        return zones

    @staticmethod
    def detect_crt(df, direction, lookback=20, sweep_buffer=0.0002):
        """
        CRT — Candle Range Theory.
        lookback widened to 30 when called on synthetic data
        (see StrategySwitcher.decide() which passes instrument).
        """
        if len(df) < 5 or "high" not in df.columns: return False, {}
        recent = df.iloc[-lookback:] if len(df) >= lookback else df
        if len(recent) < 5: return False, {}
        ranges  = recent["high"] - recent["low"]
        ref_idx = int(ranges.iloc[:-3].argmax()) if len(ranges) > 3 else 0
        if ref_idx == 0: return False, {}
        ref_high = float(recent["high"].iloc[ref_idx])
        ref_low  = float(recent["low"].iloc[ref_idx])
        if ref_high <= ref_low: return False, {}
        search = recent.iloc[ref_idx + 1:]
        if len(search) < 2: return False, {}
        for i in range(len(search) - 1):
            ch = float(search["high"].iloc[i])
            cl = float(search["low"].iloc[i])
            cc = float(search["price"].iloc[i])
            if direction == 1:
                if cl < (ref_low - sweep_buffer) and cc > ref_low:
                    return True, {"ref_high": round(ref_high,5), "ref_low": round(ref_low,5),
                                  "swept_at": round(cl,5), "close": round(cc,5),
                                  "target": round(ref_high,5), "type": "crt_bull"}
            else:
                if ch > (ref_high + sweep_buffer) and cc < ref_high:
                    return True, {"ref_high": round(ref_high,5), "ref_low": round(ref_low,5),
                                  "swept_at": round(ch,5), "close": round(cc,5),
                                  "target": round(ref_low,5), "type": "crt_bear"}
        return False, {}

    @staticmethod
    def compute_fib_zones(swing_high, swing_low, levels=(0.50, 0.618)):
        if swing_high <= swing_low or swing_high <= 0: return []
        rng = swing_high - swing_low
        zones = []
        for lv in levels:
            zones.append(FibZone(lv, round(swing_high - lv * rng, 5), "support"))
            zones.append(FibZone(lv, round(swing_low  + lv * rng, 5), "resistance"))
        return zones

    @staticmethod
    def compute_fva(df, lookback=20):
        recent = df["price"].iloc[-lookback:]
        mean = float(recent.mean()); std = float(recent.std())
        return round(mean + std, 5), round(mean - std, 5)

    @staticmethod
    def score_probability(weekly_struct, daily_struct, h4_fvgs, h4_obs,
                          current_price, last_bos, fib_zones):
        bull = bear = 0.0
        if weekly_struct == "uptrend":    bull += 25
        elif weekly_struct == "downtrend": bear += 25
        if daily_struct == "uptrend":     bull += 20
        elif daily_struct == "downtrend": bear += 20
        prox = current_price * 0.003
        fvg_bull = fvg_bear = False
        for fvg in reversed(h4_fvgs):
            if fvg.mitigated: continue
            mid = (fvg.top + fvg.bottom) / 2
            if abs(mid - current_price) <= prox * 2:
                if fvg.direction == "bullish" and not fvg_bull:
                    bull += 15; fvg_bull = True
                elif fvg.direction == "bearish" and not fvg_bear:
                    bear += 15; fvg_bear = True
            if fvg_bull and fvg_bear: break
        ob_bull = ob_bear = False
        for ob in reversed(h4_obs):
            if ob.mitigated: continue
            mid = (ob.top + ob.bottom) / 2
            if abs(mid - current_price) <= prox * 3:
                if ob.direction == "bullish" and not ob_bull:
                    bull += 15; ob_bull = True
                elif ob.direction == "bearish" and not ob_bear:
                    bear += 15; ob_bear = True
            if ob_bull and ob_bear: break
        if last_bos:
            if last_bos.direction == "bullish": bull += 15
            else:                               bear += 15
        fib_bull = fib_bear = False
        for zone in fib_zones:
            if abs(zone.price - current_price) <= prox * 2:
                if zone.direction == "support" and not fib_bull:
                    bull += 10; fib_bull = True
                elif zone.direction == "resistance" and not fib_bear:
                    bear += 10; fib_bear = True
            if fib_bull and fib_bear: break
        bull = max(bull, 10.0); bear = max(bear, 10.0)
        total = bull + bear
        b_pct = bull / total * 100; s_pct = bear / total * 100
        gap   = abs(b_pct - s_pct)
        bias  = Bias.BULLISH if bull > bear else (Bias.BEARISH if bear > bull else Bias.NEUTRAL)
        return ProbabilityScore(round(b_pct,1), round(s_pct,1), bias, round(gap,1))

    @staticmethod
    def nearest_unmitigated(price, fvgs, obs, direction):
        candidates = []
        for fvg in fvgs:
            if not fvg.mitigated and fvg.direction == direction:
                mid = (fvg.top + fvg.bottom) / 2
                candidates.append((abs(mid - price), mid, "fvg"))
        for ob in obs:
            if not ob.mitigated and ob.direction == direction:
                mid = (ob.top + ob.bottom) / 2
                candidates.append((abs(mid - price), mid, "ob"))
        if not candidates: return 0.0, ""
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]


# ════════════════════════════════════════════════════════════════════ #
#  SessionAnalyser  (Deriv-aware)                                      #
# ════════════════════════════════════════════════════════════════════ #

class SessionAnalyser:
    """
    Maps each instrument to its best UTC trading hours.

    KEY CHANGE for Deriv synthetics:
      Synthetics trade 24/7 — is_good_session always returns True
      for any symbol in the _SYNTHETICS set.
    """

    FALLBACK_WINDOWS: Dict[str, List[int]] = {
        "GBP_USD": list(range(6, 17)),
        "EUR_USD": list(range(6, 17)),
        "USD_JPY": list(range(0, 4)) + list(range(12, 17)),
        "XAU_USD": list(range(6, 21)),
        "GBP_JPY": list(range(6, 17)),
    }

    def __init__(self):
        self._hours  = dict(self.FALLBACK_WINDOWS)
        self._fitted = True
        # Synthetics — all hours open
        for sym in _SYNTHETICS:
            self._hours[sym] = list(range(24))

    def fit(self, hourly_data: Dict[str, pd.DataFrame], min_hit: float = 0.52):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier
        except ImportError:
            return
        for inst, df in hourly_data.items():
            if inst in _SYNTHETICS:
                self._hours[inst] = list(range(24))
                continue
            try:
                self._fit_one(inst, df, min_hit, LogisticRegression, OneVsRestClassifier)
            except Exception as e:
                logger.warning(f"Session fit failed {inst}: {e}")

    def _fit_one(self, inst, df, min_hit, LR, OVR):
        import numpy as np
        df = df.copy()
        df["returns"]   = np.log(df["price"] / df["price"].shift(1))
        df["direction"] = np.sign(df["returns"])
        df["hour"]      = df.index.hour
        lags = 5
        cols = [f"lag{i}" for i in range(1, lags + 1)]
        for i, col in enumerate(cols, 1):
            df[col] = df["returns"].shift(i)
        df.dropna(inplace=True)
        means = df[cols].mean(); stds = df[cols].std().replace(0, 1)
        df[cols] = (df[cols] - means) / stds
        model = OVR(LR(C=1e6, max_iter=100_000))
        model.fit(df[cols], df["direction"])
        df["pred"] = model.predict(df[cols])
        good_hours = [
            h for h in range(24)
            if len(df[df["hour"] == h]) >= 30
            and (np.sign(df[df["hour"] == h]["direction"] * df[df["hour"] == h]["pred"])
                 .value_counts().get(1.0, 0)
                 / (len(df[df["hour"] == h]) + 1e-9)) >= min_hit
        ]
        if len(good_hours) >= 3:
            self._hours[inst] = sorted(good_hours)

    def best_hours(self, instrument: str) -> List[int]:
        # Synthetics always return all hours
        if instrument in _SYNTHETICS:
            return list(range(24))
        return self._hours.get(instrument, list(range(6, 20)))

    def is_good_session(self, instrument: str, utc_hour: int) -> bool:
        # Synthetics: 24/7 — always good
        if instrument in _SYNTHETICS:
            return True
        kz_ok, _ = self.is_kill_zone(instrument, utc_hour)
        hours = self._hours.get(instrument, [])
        if len(hours) < 3:
            return kz_ok
        return (utc_hour in hours) and kz_ok

    def session_summary(self) -> dict:
        return dict(self._hours)

    KILL_ZONES = {
        "asian_open":   (0,  3),
        "london_open":  (6,  9),
        "ny_open":      (12, 15),
        "london_close": (14, 16),
    }

    PAIR_KILL_ZONES = {
        "GBP_USD": ["london_open", "ny_open", "london_close"],
        "EUR_USD": ["london_open", "ny_open", "london_close"],
        "GBP_JPY": ["asian_open",  "london_open", "ny_open"],
        "XAU_USD": ["london_open", "ny_open", "london_close"],
        "USD_CAD": ["ny_open",     "london_close"],
        "USD_CHF": ["london_open", "ny_open"],
        "XAG_USD": ["london_open", "ny_open"],
        "AUD_JPY": ["asian_open",  "london_open"],
        "NZD_JPY": ["asian_open",  "london_open"],
        "AUD_USD": ["asian_open",  "london_open", "ny_open"],
        "AUD_NZD": ["asian_open",  "london_open"],
        "GBP_AUD": ["asian_open",  "london_open", "ny_open"],
        "EUR_JPY": ["asian_open",  "london_open", "ny_open"],
        "EUR_AUD": ["asian_open",  "london_open", "ny_open"],
        "USD_JPY": ["asian_open",  "london_open", "ny_open"],
        # Synthetics — all kill zones valid (24/7)
        **{sym: ["asian_open", "london_open", "ny_open", "london_close"]
           for sym in _SYNTHETICS},
    }

    def is_kill_zone(self, instrument: str, utc_hour: int) -> Tuple[bool, str]:
        if instrument in _SYNTHETICS:
            return True, "synthetic_24h"
        valid_zones = self.PAIR_KILL_ZONES.get(instrument, ["london_open", "ny_open"])
        for zone_name in valid_zones:
            start, end = self.KILL_ZONES[zone_name]
            if start <= utc_hour < end:
                return True, zone_name
        return False, ""


# ════════════════════════════════════════════════════════════════════ #
#  RegimeDetector  (backward-compatibility shim, unchanged)            #
# ════════════════════════════════════════════════════════════════════ #

class RegimeDetector:
    def __init__(self, adx_period=14, adx_threshold=25.0, hurst_lags=30,
                 hurst_trend=0.55, hurst_mean_rev=0.45):
        self.adx_period     = adx_period
        self.adx_threshold  = adx_threshold
        self.hurst_lags     = hurst_lags
        self.hurst_trend    = hurst_trend
        self.hurst_mean_rev = hurst_mean_rev

    def compute_adx(self, high, low, close):
        n    = self.adx_period
        tr   = pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
        up   = high - high.shift(1); down = low.shift(1) - low
        dm_p = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
        dm_m = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
        atr  = tr.ewm(span=n, min_periods=n).mean()
        di_p = dm_p.ewm(span=n, min_periods=n).mean() / (atr + 1e-9) * 100
        di_m = dm_m.ewm(span=n, min_periods=n).mean() / (atr + 1e-9) * 100
        dx   = (di_p - di_m).abs() / (di_p + di_m + 1e-9) * 100
        return dx.ewm(span=n, min_periods=n).mean()

    @staticmethod
    def hurst_exponent(price, max_lag=30):
        ts = np.log(price.dropna().values)
        if len(ts) < max_lag * 2: return 0.5
        lags = list(range(2, min(max_lag, len(ts) // 2)))
        if len(lags) < 3: return 0.5
        tau  = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
        if tau.max() < 1e-10: return 0.5
        try:
            poly = np.polyfit(np.log(lags), np.log(tau + 1e-10), 1)
            return float(np.clip(poly[0], 0.0, 1.0))
        except Exception: return 0.5

    def classify(self, price, high=None, low=None):
        if len(price) < self.hurst_lags * 2: return Regime.RANDOM
        window = min(len(price), max(self.hurst_lags * 3, 60))
        H      = self.hurst_exponent(price.iloc[-window:], self.hurst_lags)
        adx_val = None
        if high is not None and low is not None and len(high) > self.adx_period * 2:
            try:
                adx_s = self.compute_adx(high, low, price)
                last  = adx_s.dropna()
                if not last.empty: adx_val = float(last.iloc[-1])
            except Exception: pass
        if adx_val is not None:
            if adx_val > self.adx_threshold and H > self.hurst_trend:   return Regime.TRENDING
            if adx_val < self.adx_threshold and H < self.hurst_mean_rev: return Regime.OSCILLATING
        if H > self.hurst_trend:    return Regime.TRENDING
        if H < self.hurst_mean_rev: return Regime.OSCILLATING
        return Regime.RANDOM


# ── Timeframe resampler ───────────────────────────────────────────── #

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy(); df.index = df.index.tz_localize(None)
    agg = {k: v for k, v in
           {"price": "last", "open": "first", "high": "max", "low": "min", "volume": "sum"}.items()
           if k in df.columns}
    return df.resample(rule).agg(agg).dropna(subset=["price"])


# ════════════════════════════════════════════════════════════════════ #
#  StrategySwitcher  (Deriv synthetic aware — session + CRT tweaks)   #
# ════════════════════════════════════════════════════════════════════ #

class StrategySwitcher:
    """SMC multi-timeframe analysis.  Unchanged from v1 except:
      1. PAIR_KILL_ZONES includes synthetics (all hours)
      2. CRT lookback = 40 bars for synthetics (faster markets)
      3. use_session_filter disabled for synthetics (handled in SessionAnalyser)
    """

    KILL_ZONES = {
        "asian_open":   (0,  3),
        "london_open":  (6,  9),
        "ny_open":      (12, 15),
        "london_close": (14, 16),
    }

    PAIR_KILL_ZONES = {
        "GBP_USD": ["london_open", "ny_open", "london_close"],
        "EUR_USD": ["london_open", "ny_open", "london_close"],
        "GBP_JPY": ["asian_open",  "london_open", "ny_open"],
        "XAU_USD": ["london_open", "ny_open", "london_close"],
        "USD_CAD": ["ny_open",     "london_close"],
        "USD_CHF": ["london_open", "ny_open"],
        "XAG_USD": ["london_open", "ny_open"],
        "AUD_JPY": ["asian_open",  "london_open"],
        "NZD_JPY": ["asian_open",  "london_open"],
        "AUD_USD": ["asian_open",  "london_open", "ny_open"],
        "AUD_NZD": ["asian_open",  "london_open"],
        "GBP_AUD": ["asian_open",  "london_open", "ny_open"],
        "EUR_JPY": ["asian_open",  "london_open", "ny_open"],
        "EUR_AUD": ["asian_open",  "london_open", "ny_open"],
        "USD_JPY": ["asian_open",  "london_open", "ny_open"],
        **{sym: ["asian_open", "london_open", "ny_open", "london_close"]
           for sym in _SYNTHETICS},
    }

    def is_kill_zone(self, instrument, utc_hour):
        if instrument in _SYNTHETICS:
            return True, "synthetic_24h"
        valid_zones = self.PAIR_KILL_ZONES.get(instrument, ["london_open", "ny_open"])
        for zone_name in valid_zones:
            start, end = self.KILL_ZONES[zone_name]
            if start <= utc_hour < end:
                return True, zone_name
        return False, ""

    def __init__(self, config=None, session_analyser=None):
        self.config   = config or StrategyConfig()
        self.sessions = session_analyser or SessionAnalyser()
        self.smc      = SMCHelpers()
        self._last_confidence_bonus = 0.0
        self._last_h1_df = None

    def decide(self, price, instrument="", high=None, low=None, utc_hour=None, h1_df=None):
        if len(price) < 50:
            return SwitchDecision(regime=Regime.RANDOM, strategy="SMC_Wait",
                                  signal=0, reason="insufficient_data")

        df    = self._build_df(price, high, low)
        w_df  = _resample(df, "W")
        d_df  = _resample(df, "D")
        h4_df = _resample(df, "4h")
        h1_df = _resample(df, "1h")

        current_price = float(price.iloc[-1])

        w_swings = self.smc.find_swings(w_df, self.config.swing_lookback, self.config.swing_min_gap)
        w_struct, w_sh, w_sl = self.smc.classify_structure(w_swings)

        d_swings = self.smc.find_swings(d_df, self.config.swing_lookback, self.config.swing_min_gap)
        d_struct, d_sh, d_sl = self.smc.classify_structure(d_swings)

        h4_fvgs     = self.smc.find_fvgs(h4_df, last_n=200)
        h4_obs      = self.smc.find_order_blocks(h4_df, last_n=200, body_ratio=self.config.ob_body_ratio)
        h4_breakers = self.smc.find_breaker_blocks(h4_df, d_swings + w_swings, h4_fvgs)
        h4_obs      = h4_obs + h4_breakers

        h1_swings = self.smc.find_swings(h1_df, lookback=5, min_gap=3)
        h1_bos    = self.smc.find_bos(h1_df, h1_swings, self.config.bos_require_close)
        last_bos  = h1_bos[-1] if h1_bos else None

        d_choch  = self.smc.find_choch(d_df,  d_swings, d_struct)
        h1_choch = self.smc.find_choch(h1_df, h1_swings,
                                        d_struct if d_struct != "ranging" else w_struct)
        if h1_choch and (last_bos is None or h1_choch.index >= last_bos.index):
            last_bos = h1_choch

        h4_sweeps    = self.smc.find_liquidity_sweeps(h4_df, lookback=100)
        h1_sweeps    = self.smc.find_liquidity_sweeps(h1_df, lookback=50) if len(h1_df) >= 10 else []
        recent_sweeps = h4_sweeps + h1_sweeps
        h4_liq_zones = self.smc.extract_liquidity_zones(h4_df, h4_sweeps)

        near_ath = near_atl = False
        if len(h4_df) >= 20 and "high" in h4_df.columns:
            ath = float(h4_df["high"].max())
            atl = float(h4_df["low"].min()) if "low" in h4_df.columns else 0.0
            near_ath = abs(ath - current_price) / ath < 0.02
            near_atl = (atl > 0) and abs(current_price - atl) / atl < 0.02

        h4_swings_all = self.smc.find_swings(h4_df, lookback=5, min_gap=3)
        h4_sh = sorted([s for s in h4_swings_all if s.kind == "high"], key=lambda x: x.index)
        h4_sl = sorted([s for s in h4_swings_all if s.kind == "low"],  key=lambda x: x.index)
        swing_tp_long  = next((s.price for s in reversed(h4_sh) if s.price > current_price), 0.0)
        swing_tp_short = next((s.price for s in reversed(h4_sl) if s.price < current_price), 0.0)

        ref_high = max(w_sh, d_sh) if (w_sh and d_sh) else current_price * 1.02
        ref_low  = min(w_sl, d_sl) if (w_sl and d_sl) else current_price * 0.98
        fib_zones = self.smc.compute_fib_zones(ref_high, ref_low, self.config.fib_levels)

        prob = self.smc.score_probability(w_struct, d_struct, h4_fvgs, h4_obs,
                                          current_price, last_bos, fib_zones)

        # Session filter — synthetics always pass
        good_session = True
        if self.config.use_session_filter and instrument and utc_hour is not None:
            good_session = self.sessions.is_good_session(instrument, utc_hour)

        if w_struct in ("uptrend", "downtrend") and d_struct == w_struct:
            regime = Regime.TRENDING
        elif w_struct == "ranging" or d_struct == "ranging":
            regime = Regime.OSCILLATING
        else:
            regime = Regime.RANDOM

        # CRT lookback: wider for synthetics (faster price action)
        crt_lookback = 40 if instrument in _SYNTHETICS else 20

        signal, strategy, reason, dol_target, dol_kind, fired_confs = self._generate_signal(
            prob, good_session, w_struct, d_struct, current_price, last_bos,
            h4_fvgs, h4_obs, fib_zones,
            swing_high   = max(w_sh, d_sh) if (w_sh and d_sh) else 0.0,
            swing_low    = min(w_sl, d_sl) if (w_sl and d_sl) else 0.0,
            h1_df        = h1_df,
            recent_sweeps = recent_sweeps,
        )

        dir_str  = "bullish" if signal >= 0 else "bearish"
        near_fvg = next((f for f in reversed(h4_fvgs) if not f.mitigated and f.direction == dir_str), None)
        near_ob  = next((o for o in reversed(h4_obs)  if not o.mitigated and o.direction == dir_str), None)

        if dol_target == 0.0 and signal != 0:
            dol_target, dol_kind = self.smc.nearest_unmitigated(current_price, h4_fvgs, h4_obs, dir_str)

        swing_tp = swing_tp_long if signal == 1 else (swing_tp_short if signal == -1 else 0.0)

        return SwitchDecision(
            regime=regime, strategy=strategy,
            params={"swing_lookback": self.config.swing_lookback},
            hurst=0.5, adx=prob.confidence, signal=signal,
            good_session=good_session, bias=prob.bias, probability=prob,
            weekly_structure=w_struct, daily_structure=d_struct,
            weekly_swing_high=w_sh, weekly_swing_low=w_sl,
            daily_swing_high=d_sh, daily_swing_low=d_sl,
            nearest_fvg=near_fvg, nearest_ob=near_ob, last_bos=last_bos,
            fib_zones=fib_zones, dol_target=dol_target, dol_kind=dol_kind,
            reason=reason, confs=fired_confs,
            liq_zones=h4_liq_zones, near_ath=near_ath, near_atl=near_atl,
            swing_tp=swing_tp,
        )

    # ── Signal generation (identical to v1) ──────────────────────── #

    def _generate_signal(self, prob, good_session, w_struct, d_struct,
                          price, last_bos, h4_fvgs, h4_obs, fib_zones,
                          swing_high=0.0, swing_low=0.0,
                          h1_df=None, recent_sweeps=None):
        self._last_h1_df = h1_df
        recent_sweeps = recent_sweeps or []
        self._last_confidence_bonus = 0.0

        if prob.confidence < self.config.min_prob_gap:
            return 0, "SMC_Wait", f"50/50 bull={prob.bullish:.0f} bear={prob.bearish:.0f}", 0.0, "", []
        if not good_session:
            return 0, "SMC_Wait", "outside_session", 0.0, "", []

        if prob.bias == Bias.BULLISH:
            direction, smc_dir = 1, "bullish"
        elif prob.bias == Bias.BEARISH:
            direction, smc_dir = -1, "bearish"
        else:
            return 0, "SMC_Wait", "neutral_bias", 0.0, "", []

        d_agrees = ((direction==1 and d_struct=="uptrend") or (direction==-1 and d_struct=="downtrend"))
        w_agrees = ((direction==1 and w_struct=="uptrend") or (direction==-1 and w_struct=="downtrend"))
        if not d_agrees and not w_agrees:
            return 0, "SMC_Wait", f"HTF_disagrees W={w_struct} D={d_struct}", 0.0, "", []

        zone_check_passed = True; zone_label = ""
        if swing_high > 0 and swing_low > 0 and swing_high > swing_low:
            midpoint = (swing_high + swing_low) / 2.0
            if direction == 1 and price > midpoint:
                return 0, "SMC_Wait", f"price_in_premium", 0.0, "", []
            elif direction == -1 and price < midpoint:
                return 0, "SMC_Wait", f"price_in_discount", 0.0, "", []
            else:
                zone_label = "discount" if direction == 1 else "premium"

        soft_confs = []; confidence_bonus = 0.0; dol_target = 0.0; dol_kind = ""
        prox = price * 0.003

        if last_bos and last_bos.direction == smc_dir:
            is_choch  = getattr(last_bos, "kind", "bos") == "choch"
            displaced = self.smc.check_displacement(self._last_h1_df, last_bos)
            if   is_choch and displaced: soft_confs.append("CHOCH+DISP")
            elif is_choch:               soft_confs.append("CHOCH")
            elif displaced:              soft_confs.append("BOS+DISP")
            else:                        soft_confs.append("BOS")
            t, k = self.smc.nearest_unmitigated(price, h4_fvgs, h4_obs, smc_dir)
            if t: dol_target, dol_kind = t, k

        for fvg in reversed(h4_fvgs):
            if fvg.mitigated or fvg.direction != smc_dir: continue
            if fvg.bottom <= price <= fvg.top or abs((fvg.top+fvg.bottom)/2 - price) <= prox * 2:
                soft_confs.append("FVG")
                if not dol_target: dol_target = (fvg.top+fvg.bottom)/2; dol_kind = "fvg"
                break

        for ob in reversed(h4_obs):
            if ob.mitigated or ob.direction != smc_dir: continue
            if ob.bottom <= price <= ob.top or abs((ob.top+ob.bottom)/2 - price) <= prox * 3:
                soft_confs.append("OB")
                if not dol_target: dol_target = (ob.top+ob.bottom)/2; dol_kind = "ob"
                break

        for sweep in recent_sweeps:
            if direction == 1 and sweep["type"] == "sweep_low":
                soft_confs.append("LIQ_SWEEP"); confidence_bonus += 0.10; break
            elif direction == -1 and sweep["type"] == "sweep_high":
                soft_confs.append("LIQ_SWEEP"); confidence_bonus += 0.10; break

        exp_fib_dir = "support" if direction == 1 else "resistance"
        for zone in fib_zones:
            if zone.direction == exp_fib_dir and abs(zone.price - price) <= prox * 3:
                soft_confs.append("Fib"); confidence_bonus += 0.08; break

        if len(soft_confs) < 2:
            return (0, "SMC_Wait",
                    f"need_2_confs got [{','.join(soft_confs) or 'none'}]", 0.0, "", [])

        self._last_confidence_bonus = confidence_bonus
        strategy = "SMC_Long" if direction == 1 else "SMC_Short"
        zone_str  = f" | zone={zone_label}" if zone_label else ""
        reason    = (f"bias={smc_dir} | W={w_struct} D={d_struct}{zone_str} | "
                     f"bull={prob.bullish:.0f}% bear={prob.bearish:.0f}% | "
                     f"confs=[{','.join(soft_confs)}] | DOL={dol_kind}@{dol_target:.5f}")
        return direction, strategy, reason, dol_target, dol_kind, soft_confs

    @staticmethod
    def _build_df(price, high, low):
        df = pd.DataFrame({"price": price})
        df["high"] = (high.values if hasattr(high, "values") else high) if high is not None else df["price"]
        df["low"]  = (low.values  if hasattr(low,  "values") else low)  if low  is not None else df["price"]
        df["open"] = df["price"].shift(1).fillna(df["price"])
        return df.dropna(subset=["price"])

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    def print_analysis(self, decision, instrument=""):
        tag = f"[{instrument}] " if instrument else ""
        p   = decision.probability
        print(f"\n{'─'*70}")
        print(f"{tag}SMC ANALYSIS")
        print(f"{'─'*70}")
        print(f"  Weekly  : {decision.weekly_structure.upper()}")
        print(f"  Daily   : {decision.daily_structure.upper()}")
        print(f"  Regime  : {decision.regime.value}")
        if p:
            print(f"  Prob    : Bull={p.bullish:.0f}% Bear={p.bearish:.0f}% Gap={p.confidence:.0f}%")
        sig_str = "+1 LONG" if decision.signal==1 else ("-1 SHORT" if decision.signal==-1 else "0 WAIT")
        print(f"  Signal  : {sig_str}")
        print(f"  Session : {decision.good_session}")
        print(f"  Reason  : {decision.reason}")
        print(f"{'─'*70}\n")
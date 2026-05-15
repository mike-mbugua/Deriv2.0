"""
paper_trader.py  —  v5  (ML-enhanced, M5 confluence, M2 entry)
═══════════════════════════════════════════════════════════════

TIMEFRAME STACK
───────────────
  Weekly  → market structure bias                        [SMC layer 1]
  Daily   → structure + DOL targets  (PRIMARY SMC call)  [SMC layer 2]
  H4      → trend filter + PD arrays                     [filter 1]
  H1      → entry zone BOS/FVG/OB                        [filter 2]
  M5      → micro-structure confluence gate              [filter 3]
  M2      → precision entry price (fill price only)      [entry tick]

MACHINE LEARNING  (built-in, no external training step needed)
──────────────────────────────────────────────────────────────
  MLSignalFilter — Random Forest trained on M5 features:
    • Returns over 3/5/10/20 bars
    • ATR ratio (volatility context)
    • RSI (momentum)
    • Distance from 20-bar SMA (mean-reversion)
    • Time-of-day (UTC hour) encoded as sin/cos
    • Day-of-week encoded as sin/cos
    • SMC probability score from higher TF

  Training:   Happens automatically when ≥200 M5 bars available.
              Re-trains every 500 new M5 bars (continuous learning).
  Prediction: Outputs probability(long) and probability(short).
              Acts as a GATE — blocks entry if ML disagrees with SMC.
              Does NOT override SMC; it filters low-probability setups.

  Why Random Forest?
    • Handles non-linear interactions between features
    • Resistant to overfitting on small datasets (100-tree ensemble)
    • Built-in feature importance for explainability
    • Fast inference (<1ms per prediction)

SIGNAL PIPELINE (trade fires only when ALL pass)
─────────────────────────────────────────────────
  1. Daily SMC    → weekly/daily structure + probability > threshold
  2. H4 filter    → trend agrees (or flat)
  3. H1 filter    → entry zone agrees (or flat)
  4. M5 filter    → micro confluence agrees (or flat)
  5. ML gate      → Random Forest P(direction) > ml_threshold (default 0.55)
  6. M2 price     → fill price (most precise available)
  7. Risk engine  → SL = 1.5×ATR, TP = 3×ATR (1:2 RR), fixed or % units

REFRESH SCHEDULE
─────────────────
  M2 + M5 → every tick (loop_interval, default 120s for M2 precision)
  H1      → every new UTC hour
  H4      → every new 4-hour block
  D + W   → once per UTC day
"""


import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_feed import OandaClient, DataFeed
from strategy_switcher import StrategySwitcher, StrategyConfig, SwitchDecision
from risk_manager import RiskManager, TradeSetup
from config import INSTRUMENTS
from execution_broker import ExecutionBroker, SimBroker, OandaBroker, TradeResult



logger = logging.getLogger(__name__)

try:
    from smc_logger import (
        print_smc_dashboard, print_trade_opened,
        print_trade_closed, print_no_signal_summary,
    )
    _HAS_SMC_LOGGER = True
except ImportError:
    _HAS_SMC_LOGGER = False
    def print_smc_dashboard(*a, **kw): pass
    def print_trade_opened(*a, **kw):  pass
    def print_trade_closed(*a, **kw):  pass
    def print_no_signal_summary(*a, **kw): pass

try:
    from telegram_notifier import TelegramNotifier
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False
# ══════════════════════════════════════════════════════════════════ #
#  Timeframe configuration                                          #
# ══════════════════════════════════════════════════════════════════ #

TF_BARS = {
    "W":  200,    # ~4 years weekly   → swing structures
    "D":  500,    # ~2 years daily    → primary SMC
    "H4": 500,    # ~83 days H4       → trend filter
    "H1": 300,    # ~12 days H1       → entry zone
    "M5": 576,    # ~48 hrs M5        → confluence + ML training
    "M2": 450,    # ~15 hrs M2        → entry price
}

TF_MIN_BARS = {
    "W": 20, "D": 60, "H4": 50, "H1": 50, "M5": 30, "M2": 10
}

# Confidence weights (must sum to 1.0)
CONF_WEIGHTS = {
    "prob_gap":   0.30,   # SMC probability gap
    "ml_prob":    0.25,   # ML model probability  ← new primary weight
    "h4_agrees":  0.15,   # H4 agrees
    "h1_agrees":  0.12,   # H1 agrees
    "m5_agrees":  0.10,   # M5 agrees
    "session":    0.04,   # session quality
    "tf_aligned": 0.04,   # weekly+daily same structure
}


# ══════════════════════════════════════════════════════════════════ #
#  ML Signal Filter                                                 #
# ══════════════════════════════════════════════════════════════════ #

class MLSignalFilter:
    """
    Random Forest trained on M5 bar features to predict short-term
    price direction. Acts as a probability gate before trade execution.

    Features (per bar):
      ret_3, ret_5, ret_10, ret_20  — log returns over N bars
      atr_ratio                     — ATR / price (volatility context)
      rsi_14                        — momentum
      dist_sma20                    — (price - SMA20) / price
      hour_sin, hour_cos            — UTC hour encoded cyclically
      dow_sin,  dow_cos             — day-of-week encoded cyclically
      smc_prob                      — SMC probability score (passed in)

    Label:
      +1 if price rose over next 5 M2 bars (10 min)
      -1 if price fell

    Training:
      Triggered automatically when ≥200 M5 bars available.
      Re-trains every retrain_interval new bars (continuous learning).
    """
    

    def __init__(
        self,
        min_train_bars:   int   = 200,
        retrain_interval: int   = 500,
        ml_threshold:     float = 0.55,
        n_estimators:     int   = 100,
        model_dir:        str   = "logs",
    ):
        self.min_train_bars   = min_train_bars
        self.retrain_interval = retrain_interval
        self.ml_threshold     = ml_threshold
        self.n_estimators     = n_estimators
        self.model_dir        = model_dir

        self._models:          Dict[str, object] = {}   # pair → fitted RF
        self._bars_since_fit:  Dict[str, int]    = {}   # pair → bars since last fit
        self._feature_names:   List[str]         = []
        self._fitted_pairs:    set               = set()
        self._importances:     Dict[str, dict]   = {}   # pair → feature importances

    # ── Feature engineering ──────────────────────────────────────── #

    @staticmethod
    def _build_features(df: pd.DataFrame, smc_prob: float = 50.0) -> Optional[pd.DataFrame]:
        """
        Build feature matrix from M5 OHLC DataFrame.
        Returns None if not enough data.
        """
        if len(df) < 25 or "price" not in df.columns:
            return None

        d = df.copy()
        d["ret"]    = np.log(d["price"] / d["price"].shift(1))
        d["ret_3"]  = d["ret"].rolling(3).sum()
        d["ret_5"]  = d["ret"].rolling(5).sum()
        d["ret_10"] = d["ret"].rolling(10).sum()
        d["ret_20"] = d["ret"].rolling(20).sum()

        # ATR-based volatility
        if "high" in d.columns and "low" in d.columns:
            tr = pd.concat([
                d["high"] - d["low"],
                (d["high"] - d["price"].shift(1)).abs(),
                (d["low"]  - d["price"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            d["atr"] = tr.ewm(span=14, min_periods=14).mean()
        else:
            d["atr"] = d["ret"].abs().rolling(14).mean() * d["price"]
        d["atr_ratio"] = d["atr"] / (d["price"] + 1e-9)

        # RSI
        delta    = d["price"].diff()
        gain     = delta.clip(lower=0).ewm(span=14, min_periods=14).mean()
        loss     = (-delta.clip(upper=0)).ewm(span=14, min_periods=14).mean()
        rs       = gain / (loss + 1e-9)
        d["rsi"] = 100 - 100 / (1 + rs)

        # Distance from SMA
        d["sma20"]     = d["price"].rolling(20).mean()
        d["dist_sma20"] = (d["price"] - d["sma20"]) / (d["price"] + 1e-9)

        # Time encoding (cyclical)
        if hasattr(d.index, "hour"):
            hours       = d.index.hour
            dows        = d.index.dayofweek
        else:
            hours       = np.zeros(len(d), dtype=int)
            dows        = np.zeros(len(d), dtype=int)
        d["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        d["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        d["dow_sin"]  = np.sin(2 * np.pi * dows / 7)
        d["dow_cos"]  = np.cos(2 * np.pi * dows / 7)

        # SMC probability (constant for this bar — passed from caller)
        d["smc_prob"] = smc_prob / 100.0   # normalise to 0–1

        feature_cols = [
            "ret_3", "ret_5", "ret_10", "ret_20",
            "atr_ratio", "rsi", "dist_sma20",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "smc_prob",
        ]
        return d[feature_cols].dropna()

    # ── Training ─────────────────────────────────────────────────── #

    def fit(self, pair: str, m5_df: pd.DataFrame, smc_prob: float = 50.0):
        """
        Train a Random Forest for one pair on its M5 history.
        Called automatically from predict() when enough bars available.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.warning("scikit-learn not installed — ML filter disabled.")
            return

        if len(m5_df) < self.min_train_bars:
            return

        feat_df = self._build_features(m5_df, smc_prob)
        if feat_df is None or len(feat_df) < self.min_train_bars - 25:
            return

        # Label: +1 if price rises over next 5 bars, -1 otherwise
        price_series = m5_df["price"].reindex(feat_df.index)
        future       = price_series.shift(-5)
        labels       = np.where(future > price_series, 1, -1)

        # Align — drop last 5 rows (no future label)
        X = feat_df.values[:-5]
        y = labels[:-5]

        if len(np.unique(y)) < 2:
            return   # degenerate — all same class

        clf = RandomForestClassifier(
            n_estimators = self.n_estimators,
            max_depth    = 6,
            min_samples_leaf = 10,
            n_jobs       = -1,
            random_state = 42,
        )
        clf.fit(X, y)

        self._models[pair]         = clf
        self._feature_names        = list(feat_df.columns)
        self._bars_since_fit[pair] = 0
        self._fitted_pairs.add(pair)

        # Store feature importances
        self._importances[pair] = dict(zip(
            feat_df.columns,
            [round(float(v), 4) for v in clf.feature_importances_]
        ))

        # Save model to disk
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            path = os.path.join(self.model_dir, f"ml_{pair}.pkl")
            with open(path, "wb") as f:
                pickle.dump(clf, f)
        except Exception:
            pass

        acc = self._in_sample_accuracy(clf, X, y)
        logger.info(
            f"ML [{pair}] trained on {len(X)} M5 bars | "
            f"in-sample acc={acc:.1%} | "
            f"top feature: {max(self._importances[pair], key=self._importances[pair].get)}"
        )

    def load_saved(self, pair: str):
        """Load a previously saved model from disk (warm-start on restart)."""
        try:
            path = os.path.join(self.model_dir, f"ml_{pair}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._models[pair] = pickle.load(f)
                self._fitted_pairs.add(pair)
                logger.info(f"ML [{pair}] loaded saved model from {path}")
        except Exception as e:
            logger.debug(f"ML [{pair}] could not load saved model: {e}")

    @staticmethod
    def _in_sample_accuracy(clf, X, y) -> float:
        preds = clf.predict(X)
        return float((preds == y).mean())

    # ── Prediction ───────────────────────────────────────────────── #

    def predict(
        self,
        pair:       str,
        m5_df:      pd.DataFrame,
        direction:  int,            # +1 or -1 (SMC signal direction)
        smc_prob:   float = 50.0,
    ) -> Tuple[bool, float]:
        """
        Returns (allow_trade, ml_probability).

        allow_trade = True  → ML agrees with SMC direction
        allow_trade = False → ML disagrees or not confident enough
        ml_probability      → P(direction) from Random Forest

        Auto-trains if not fitted yet and enough bars available.
        Auto-retrains every retrain_interval bars.
        """
        # Try to load saved model on first call
        if pair not in self._bars_since_fit:
            self._bars_since_fit[pair] = 0
            self.load_saved(pair)

        self._bars_since_fit[pair] += 1

        # Trigger training / retraining
        should_train = (
            pair not in self._fitted_pairs
            and len(m5_df) >= self.min_train_bars
        ) or (
            self._bars_since_fit.get(pair, 0) >= self.retrain_interval
            and len(m5_df) >= self.min_train_bars
        )

        if should_train:
            logger.info(f"ML [{pair}] training/retraining on {len(m5_df)} M5 bars...")
            self.fit(pair, m5_df, smc_prob)

        # No model yet — allow trade (don't block on missing ML)
        if pair not in self._fitted_pairs or pair not in self._models:
            return True, 0.5

        # Build features for latest bar
        feat_df = self._build_features(m5_df.iloc[-50:], smc_prob)
        if feat_df is None or feat_df.empty:
            return True, 0.5

        clf = self._models[pair]
        X   = feat_df.values[[-1]]   # latest bar only

        try:
            proba = clf.predict_proba(X)[0]
            classes = list(clf.classes_)

            if direction == 1:   # long — want P(class=+1)
                idx      = classes.index(1) if 1 in classes else 0
                ml_prob  = float(proba[idx])
            else:                # short — want P(class=-1)
                idx      = classes.index(-1) if -1 in classes else 0
                ml_prob  = float(proba[idx])

            allow = ml_prob >= self.ml_threshold
            return allow, round(ml_prob, 3)

        except Exception as e:
            logger.debug(f"ML [{pair}] predict error: {e}")
            return True, 0.5

    def feature_importance(self, pair: str) -> dict:
        return self._importances.get(pair, {})

    def is_fitted(self, pair: str) -> bool:
        return pair in self._fitted_pairs


# ══════════════════════════════════════════════════════════════════ #
#  Trade record                                                     #
# ══════════════════════════════════════════════════════════════════ #
# ══════════════════════════════════════════════════════════════════ #
#  Confluence Performance Tracker                                    #
# ══════════════════════════════════════════════════════════════════ #

class ConfluenceTracker:
    """
    Tracks win/loss outcomes per confluence combination.
    Dynamically adjusts soft confirmation weights based on
    real trading performance.

    After each closed trade, the confluence combination that
    triggered the entry is scored. Over time, combinations
    with higher win rates get higher weights, making the bot
    prefer setups that have historically worked.

    Storage: logs/confluence_stats.json (persists across restarts)
    """

    # Base weights — starting point before learning kicks in
    BASE_WEIGHTS = {
        "BOS":        0.15,
        "BOS+DISP":   0.25,
        "CHOCH":      0.20,
        "CHOCH+DISP": 0.35,
        "FVG":        0.20,
        "OB":         0.18,
        "LIQ_SWEEP":  0.22,
        "CRT":        0.28,
        "Fib":        0.08,
    }

    # Minimum trades before weight adjustment kicks in
    MIN_TRADES_TO_ADJUST = 5

    # How much to shift weight per win/loss (learning rate)
    LEARNING_RATE = 0.02

    # Weight boundaries
    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.50

    def __init__(self, log_dir: str = "logs"):
        self.log_dir   = log_dir
        self._stats:   Dict[str, dict] = {}   # key=confluence, value=stats
        self._weights: Dict[str, float] = dict(self.BASE_WEIGHTS)
        self._path     = os.path.join(log_dir, "confluence_stats.json")
        self._load()

    # ── Persistence ──────────────────────────────────────────────── #

    def _load(self):
        """Load saved stats from disk on startup."""
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    data = json.load(f)
                self._stats   = data.get("stats",   {})
                self._weights = data.get("weights", dict(self.BASE_WEIGHTS))
                logger.info(
                    f"ConfluenceTracker loaded | "
                    f"{len(self._stats)} confluence types tracked"
                )
        except Exception as e:
            logger.warning(f"ConfluenceTracker load failed: {e}")

    def _save(self):
        """Save stats to disk after every update."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({
                    "stats":   self._stats,
                    "weights": self._weights,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"ConfluenceTracker save failed: {e}")

    # ── Recording ────────────────────────────────────────────────── #

    def record_trade(
        self,
        confs:    List[str],   # e.g. ["BOS+DISP", "FVG", "Fib"]
        pnl:      float,       # actual PnL of the closed trade
        pips:     float,       # pips won/lost
    ):
        """
        Called after every trade closes.
        Records outcome per confluence and adjusts weights.
        """
        won = pnl > 0

        for conf in confs:
            if conf not in self._stats:
                self._stats[conf] = {
                    "trades": 0,
                    "wins":   0,
                    "losses": 0,
                    "total_pips": 0.0,
                    "win_rate":   0.0,
                    "avg_pips":   0.0,
                }

            s = self._stats[conf]
            s["trades"]     += 1
            s["total_pips"] += pips
            if won:
                s["wins"] += 1
            else:
                s["losses"] += 1

            s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1)
            s["avg_pips"] = round(s["total_pips"] / s["trades"], 1)

            # Adjust weight if enough data
            if s["trades"] >= self.MIN_TRADES_TO_ADJUST:
                self._adjust_weight(conf, won, s["win_rate"])

        self._save()
        logger.info(
            f"[CONFLUENCE] recorded | "
            f"confs={confs} | "
            f"{'WIN' if won else 'LOSS'} | "
            f"pips={pips:+.1f}"
        )

    def _adjust_weight(self, conf: str, won: bool, win_rate: float):
        """
        Shift weight up on win, down on loss.
        Extra boost if win_rate is very high/low.
        """
        current = self._weights.get(conf, self.BASE_WEIGHTS.get(conf, 0.15))

        if won:
            # Win — increase weight, more so if win rate is high
            boost = self.LEARNING_RATE * (1 + (win_rate - 50) / 100)
            new_weight = current + boost
        else:
            # Loss — decrease weight, more so if win rate is low
            penalty = self.LEARNING_RATE * (1 + (50 - win_rate) / 100)
            new_weight = current - penalty

        self._weights[conf] = round(
            max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight)), 4
        )

    # ── Query ────────────────────────────────────────────────────── #

    def get_weight(self, conf: str) -> float:
        """Get current learned weight for a confluence type."""
        return self._weights.get(conf, self.BASE_WEIGHTS.get(conf, 0.15))

    def get_combined_score(self, confs: List[str]) -> float:
        """
        Score a confluence combination using learned weights.
        Returns a confidence bonus (0.0 - 1.0).
        """
        if not confs:
            return 0.0
        total = sum(self.get_weight(c) for c in confs)
        return round(min(total, 1.0), 3)

    def summary(self) -> dict:
        """Return performance summary sorted by win rate."""
        if not self._stats:
            return {}
        return dict(sorted(
            self._stats.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True,
        ))

    def print_report(self):
        """Print confluence performance table to console."""
        print(f"\n{'─'*65}")
        print(f"  CONFLUENCE PERFORMANCE TRACKER")
        print(f"{'─'*65}")
        print(f"  {'Confluence':<15} {'Trades':>6} {'WR%':>6} {'AvgPips':>8} {'Weight':>8}")
        print(f"  {'─'*55}")
        for conf, s in self.summary().items():
            w = self._weights.get(conf, 0.0)
            print(
                f"  {conf:<15} {s['trades']:>6} "
                f"{s['win_rate']:>5.1f}% {s['avg_pips']:>+8.1f} "
                f"{w:>8.3f}"
            )
        print(f"{'─'*65}\n")


@dataclass
class Trade:
    timestamp:   str
    pair:        str
    direction:   str
    units:       int
    price:       float
    sl_price:    float
    tp_price:    float
    strategy:    str
    regime:      str
    sl_pips:     float = 0.0
    tp_pips:     float = 0.0
    rr_ratio:    float = 0.0
    pnl:         float = 0.0
    pnl_pips:    float = 0.0
    balance:     float = 0.0
    confidence:  float = 0.0
    ml_prob:     float = 0.0   # ML gate probability at entry
    confs:       List[str] = field(default_factory=list)  # ← add this


# ══════════════════════════════════════════════════════════════════ #
#  Per-pair position state                                          #
# ══════════════════════════════════════════════════════════════════ #

@dataclass
class PairState:
    pair:             str
    position:         int   = 0
    units:            int   = 0
    entry_price:      float = 0.0
    sl_price:         float = 0.0
    tp_price:         float = 0.0
    unrealized_pnl:   float = 0.0
    oanda_trade_id:   str   = ""
    ml_prob_at_entry: float = 0.0


# ══════════════════════════════════════════════════════════════════ #
#  Per-pair multi-TF history                                        #
# ══════════════════════════════════════════════════════════════════ #

@dataclass
class PairHistory:
    pair: str
    W:    pd.DataFrame = field(default_factory=pd.DataFrame)
    D:    pd.DataFrame = field(default_factory=pd.DataFrame)
    H4:   pd.DataFrame = field(default_factory=pd.DataFrame)
    H1:   pd.DataFrame = field(default_factory=pd.DataFrame)
    M5:   pd.DataFrame = field(default_factory=pd.DataFrame)
    M2:   pd.DataFrame = field(default_factory=pd.DataFrame)

    def get(self, gran: str) -> pd.DataFrame:
        return getattr(self, gran, pd.DataFrame())

    def set(self, gran: str, df: pd.DataFrame):
        setattr(self, gran, df.copy())

    def append(self, gran: str, df: pd.DataFrame, max_bars: int = 2000):
        existing = self.get(gran)
        if existing.empty:
            self.set(gran, df)
        else:
            combined = pd.concat([existing, df]).drop_duplicates().sort_index()
            self.set(gran, combined.iloc[-max_bars:])

    def latest_price(self) -> Optional[float]:
        for gran in ("M2", "M5", "H1", "H4", "D"):
            df = self.get(gran)
            if not df.empty and "price" in df.columns:
                v = df["price"].dropna()
                if not v.empty:
                    return float(v.iloc[-1])
        return None

    def has_enough(self, gran: str) -> bool:
        return len(self.get(gran)) >= TF_MIN_BARS.get(gran, 20)


# ══════════════════════════════════════════════════════════════════ #
#  Main PaperTrader                                                 #
# ══════════════════════════════════════════════════════════════════ #

class PaperTrader:
    """
    Complete multi-pair SMC + ML trading engine.

    Signal pipeline:
      Daily SMC → H4 filter → H1 filter → M5 confluence
      → ML gate → M2 entry price → Risk engine → OANDA order

    ML is built-in: no separate training script needed.
    The MLSignalFilter trains itself from M5 bars as they accumulate,
    and retrains every 500 new bars (continuous learning).
    """

    def __init__(
        self,
        instruments:      List[str],
        config:           StrategyConfig = None,
        oanda_client:     OandaClient    = None,
        initial_balance:  float  = 10_000.0,
        risk_pct:         float  = 10.0,
        rr_ratio:         float  = 2.0,
        sl_atr_mult:      float  = 1.5,
        max_positions:    int    = 2,
        max_daily_loss:   float  = 5.0,
        loop_interval:    int    = 120,   # 2 min default (matches M2)
        log_dir:          str    = "logs",
        sim_mode:         bool   = True,
        fixed_units:      int    = 0,
        spread_pips:      float  = 1.5,
        slippage_pips:    float  = 0.5,
        # ML parameters
        ml_enabled:       bool   = True,
        ml_threshold:     float  = 0.55,
        ml_min_bars:      int    = 200,
        ml_retrain_every: int    = 500,
        broker = None,
    ):
        from execution_broker import SimBroker, OandaBroker
        if broker is not None:
            self.broker = broker
        elif oanda_client and not sim_mode:
            self.broker = OandaBroker(oanda_client)
        else:
            self.broker = SimBroker()
        
        # DEBUG — remove after fix confirmed


        self.instruments     = instruments
        self.config          = config or StrategyConfig()
        self.oanda           = oanda_client
        self.initial_balance = initial_balance
        self.cash_balance    = initial_balance
        self.loop_interval   = loop_interval
        self.log_dir         = log_dir
        self.sim_mode        = sim_mode
        self._loop_count     = 0
        self.ml_enabled      = ml_enabled

        os.makedirs(log_dir, exist_ok=True)

        self.switcher = StrategySwitcher(self.config)
        self.risk_mgr = RiskManager(
            risk_pct       = risk_pct,
            rr_ratio       = rr_ratio,
            sl_atr_mult    = sl_atr_mult,
            max_positions  = max_positions,
            max_daily_loss = max_daily_loss,
        )

        # ML filter — one shared instance, trains per pair
        self.ml = MLSignalFilter(
            min_train_bars   = ml_min_bars,
            retrain_interval = ml_retrain_every,
            ml_threshold     = ml_threshold,
            model_dir        = log_dir,
        )
        self.confluence_tracker = ConfluenceTracker(log_dir=log_dir)

        self.history: Dict[str, PairHistory] = {p: PairHistory(pair=p) for p in instruments}
        self.states:  Dict[str, PairState]   = {p: PairState(pair=p)   for p in instruments}
        self.trade_log: List[Trade] = []

        self._log_path   = os.path.join(log_dir, "trades.jsonl")
        self._state_path = os.path.join(log_dir, "bot_state.json")
        utc_now = datetime.now(timezone.utc)
        self._last_refresh: Dict[str, Dict[str, str]] = {
            p: {
                "H1": str(utc_now.hour),
                "H4": str(utc_now.hour // 4),
                "D":  utc_now.strftime("%Y-%m-%d"),
                "W":  utc_now.strftime("%Y-%m-%d"),
            }
            for p in instruments
        }

        if oanda_client:
            self.feed = DataFeed(oanda_client, cache_dir="data")
        else:
            self.feed = None

        logger.info(
            f"PaperTrader v5 ready | pairs={instruments} | sim={sim_mode} | "
            f"risk={risk_pct}% | RR=1:{rr_ratio} | max_pos={max_positions} | "
            f"loop={loop_interval}s | TFs=W/D/H4/H1/M5/M2 | "
            f"ML={'ON (thresh=' + str(ml_threshold) + ')' if ml_enabled else 'OFF'}"
        )
        # ── Telegram notifications ────────────────────────────────── #
        self.telegram: "TelegramNotifier | None" = None
        if _HAS_TELEGRAM:
            try:
                self.telegram = TelegramNotifier()
                logger.info("Telegram notifier ready.")
            except ValueError as e:
                logger.warning(f"Telegram disabled: {e}")
        # How often (in loops) to send WAIT/no-signal updates.
        # 30 loops × 120s = every ~60 min. Set to 0 to disable WAIT messages.
        self._tg_wait_every   = 30
        self._tg_loop_counter = 0   # tracks loops for WAIT throttling

    # ══════════════════════════════════════════════════════════════ #
    #  NAV                                                           #
    # ══════════════════════════════════════════════════════════════ #

    def _nav(self, prices: Dict[str, float] = None) -> float:
        upnl = sum(s.unrealized_pnl for s in self.states.values() if s.position != 0)
        return round(self.cash_balance + upnl, 2)

    def _update_unrealised_pnl(self, pair: str, price: float):
        s = self.states[pair]
        if s.position == 0:
            s.unrealized_pnl = 0.0
            return

        s.unrealized_pnl = round(s.units * (price - s.entry_price), 2)

        # ── Two-stage trailing stop ──────────────────────────────── #
        d_df = self.history[pair].get("D")
        atr  = self.risk_mgr.compute_atr(d_df) if d_df is not None else 0.0
        if atr > 0 and s.sl_price and s.tp_price and s.entry_price:
            # Check if price is inside an active liquidity zone mid-trade
            # If so, tighten the trail — price may reverse hard from here
            liq_zones  = getattr(s, "liq_zones", [])
            trail_mult = 1.5   # default
            in_zone    = None
            for z in liq_zones:
                if z.bottom <= price <= z.top and not z.swept:
                    in_zone    = z
                    trail_mult = 0.75   # tighter inside zone
                    break
            if in_zone:
                logger.info(
                    f"[ZONE] {pair} price={price:.5f} inside active "
                    f"{in_zone.kind} zone — trail tightened to {trail_mult}×ATR"
                )
            new_sl, stage = self.risk_mgr.trail_sl(
                direction     = s.position,
                current_price = price,
                current_sl    = s.sl_price,
                entry_price   = s.entry_price,
                tp_price      = s.tp_price,
                atr           = atr,
            )
            if new_sl != s.sl_price:
                logger.info(
                    f"[TRAIL] {pair} SL moved {s.sl_price:.5f} → {new_sl:.5f} "
                    f"({stage}) | price={price:.5f}"
                )
                s.sl_price = new_sl
                # Push updated SL to broker if live
                if not self.sim_mode and s.oanda_trade_id:
                    try:
                        self.broker.modify_sl(s.oanda_trade_id, pair, new_sl)
                    except Exception as e:
                        logger.warning(f"[TRAIL] Could not update SL on broker: {e}")

    # ══════════════════════════════════════════════════════════════ #
    #  Portfolio                                                     #
    # ══════════════════════════════════════════════════════════════ #

    def portfolio_summary(self, prices: Dict[str, float] = None) -> dict:
        prices = prices or {}
        for pair, state in self.states.items():
            if state.position != 0 and pair in prices:
                self._update_unrealised_pnl(pair, prices[pair])
        nav    = self._nav(prices)
        closed = [t for t in self.trade_log if t.direction == "close"]
        wins   = [t for t in closed if t.pnl > 0]
        return {
            "cash_balance": round(self.cash_balance, 2),
            "nav":          nav,
            "total_pnl":    round(sum(t.pnl for t in closed), 2),
            "total_trades": len(closed),
            "wins":         len(wins),
            "losses":       len(closed) - len(wins),
            "win_rate":     round(len(wins) / max(len(closed), 1) * 100, 1),
            "open_positions": {
                p: {
                    "direction": "long" if s.position == 1 else "short",
                    "units":     abs(s.units),
                    "entry":     s.entry_price,
                    "sl":        s.sl_price,
                    "tp":        s.tp_price,
                    "upnl":      round(s.unrealized_pnl, 2),
                    "ml_prob":   s.ml_prob_at_entry,
                }
                for p, s in self.states.items() if s.position != 0
            },
            "return_pct": round((nav - self.initial_balance) / self.initial_balance * 100, 2),
            "ml_fitted":  [p for p in self.instruments if self.ml.is_fitted(p)],
            "confluence_stats": self.confluence_tracker.summary(),
        }

    # ══════════════════════════════════════════════════════════════ #
    #  Data seeding + refresh                                        #
    # ══════════════════════════════════════════════════════════════ #

    def _seed_all(self):
        # ── Sync real broker balance before sizing anything ───────── #
        # If running against MT5/OANDA with a real account, the bot's
        # internal cash_balance must match the broker's actual balance.
        # Without this, position sizing uses the config initial_balance
        # ($10,000) while the broker only has e.g. $50 — causing every
        # MT5 order to be rejected for insufficient margin.
        if hasattr(self.broker, 'account_info'):
            try:
                info = self.broker.account_info()
                real_balance = info.get('balance', 0.0)
                if real_balance > 0:
                    logger.info(
                        f"Syncing NAV from broker: ${real_balance:.2f} "
                        f"(was ${self.cash_balance:.2f})"
                    )
                    self.cash_balance    = real_balance
                    self.initial_balance = real_balance
            except Exception as e:
                logger.warning(f"Could not sync broker balance: {e}")

        logger.info("Seeding W / D / H4 / H1 / M5 / M2 from OANDA...")
        for pair in self.instruments:
            for gran, bars in TF_BARS.items():
                try:
                    df = self.feed.get_history(pair, gran, bars)
                    self.history[pair].set(gran, df)
                    logger.info(
                        f"  {pair} {gran:3s}: {len(df):4d} bars "
                        f"({df.index[0].date()} -> {df.index[-1].date()})"
                    )
                except Exception as e:
                    logger.warning(f"  {pair} {gran}: seed failed — {e}")

        # Trigger initial ML training from seeded M5 data
        if self.ml_enabled:
            for pair in self.instruments:
                m5_df = self.history[pair].get("M5")
                if len(m5_df) >= self.ml.min_train_bars:
                    logger.info(f"ML [{pair}] initial training from seeded M5 data...")
                    self.ml.fit(pair, m5_df)

    def _refresh_pair(self, pair: str, utc_now: datetime):
        utc_hour = utc_now.hour
        utc_h4   = str(utc_hour // 4)
        utc_day  = utc_now.strftime("%Y-%m-%d")
        last     = self._last_refresh[pair]

        # M2 + M5 every tick
        schedule = ["M2", "M5"]

        if str(utc_hour) != last["H1"]:
            schedule.append("H1")
            last["H1"] = str(utc_hour)

        if utc_h4 != last["H4"]:
            schedule.append("H4")
            last["H4"] = utc_h4

        if utc_day != last.get("D", ""):
            schedule.extend(["D", "W"])
            last["D"] = utc_day
            last["W"] = utc_day

        for gran in schedule:
            try:
                df = self.feed.get_history(pair, gran, 5, force_refresh=True)
                self.history[pair].append(gran, df, max_bars=TF_BARS[gran] * 2)
            except Exception as e:
                logger.debug(f"Refresh {pair} {gran}: {e}")

    # ══════════════════════════════════════════════════════════════ #
    #  SMC + ML analysis                                             #
    # ══════════════════════════════════════════════════════════════ #

    def _analyse_pair(
        self,
        pair:     str,
        utc_hour: int,
    ) -> Tuple[
        Optional[SwitchDecision],  # daily_dec
        Optional[SwitchDecision],  # h4_dec
        Optional[SwitchDecision],  # h1_dec
        Optional[SwitchDecision],  # m5_dec
        int,                       # final_signal
        float,                     # confidence
        float,                     # entry_price (M2)
        float,                     # ml_prob
    ]:
        h = self.history[pair]

        if not h.has_enough("D") or not h.has_enough("H4"):
            return None, None, None, None, 0, 0.0, 0.0, 0.5, []

        d_df  = h.get("D")
        h4_df = h.get("H4")
        h1_df = h.get("H1")
        m5_df = h.get("M5")
        m2_df = h.get("M2")

        # ── Layer 1: Daily — primary SMC ──────────────────────── #
        daily_dec = self.switcher.decide(
            d_df["price"].dropna(), pair,
            d_df["high"] if "high" in d_df.columns else None,
            d_df["low"]  if "low"  in d_df.columns else None,
            utc_hour,
        )

        # ── Layer 2: H4 — trend filter ────────────────────────── #
        h4_dec = self.switcher.decide(
            h4_df["price"].dropna(), pair,
            h4_df["high"] if "high" in h4_df.columns else None,
            h4_df["low"]  if "low"  in h4_df.columns else None,
            utc_hour,
        )

        # ── Layer 3: H1 — entry zone ──────────────────────────── #
        h1_dec = None
        if h.has_enough("H1"):
            h1_dec = self.switcher.decide(
                h1_df["price"].dropna(), pair,
                h1_df["high"] if "high" in h1_df.columns else None,
                h1_df["low"]  if "low"  in h1_df.columns else None,
                utc_hour,
            )

        # ── Layer 4: M5 — micro confluence ────────────────────── #
        # ── Layer 4: M5 — micro confluence ────────────────────── #
        m5_dec = None
        if h.has_enough("M5"):
            m5_dec = self.switcher.decide(
                m5_df["price"].dropna(), pair,
                m5_df["high"] if "high" in m5_df.columns else None,
                m5_df["low"]  if "low"  in m5_df.columns else None,
                utc_hour,
            )

        # ── Layer 4b: CRT — manipulation confirmation ─────────── #
        # Uses H1 as reference candle timeframe
        # M5 for sweep + rejection detection
        crt_confirmed = False
        crt_details   = {}
        d_signal = daily_dec.signal if daily_dec else 0
        if d_signal != 0 and h.has_enough("H1") and h.has_enough("M5"):
            crt_confirmed, crt_details = self.switcher.smc.detect_crt(
                m5_df, d_signal, lookback=30
            )
            if crt_confirmed:
                logger.info(
                    f"  {pair}: CRT confirmed | "
                    f"type={crt_details.get('type')} | "
                    f"swept={crt_details.get('swept_at')} | "
                    f"target={crt_details.get('target')}"
                )

        # ── Layer 5: M2 — entry price only ────────────────────── #

        # ── Layer 5: M2 — entry price only ────────────────────── #
        if not m2_df.empty and "price" in m2_df.columns:
            v = m2_df["price"].dropna()
            entry_price = float(v.iloc[-1]) if not v.empty else (h.latest_price() or 0.0)
        else:
            entry_price = h.latest_price() or 0.0

        # ── Signal alignment ───────────────────────────────────── #
        d_signal  = daily_dec.signal
        h4_signal = h4_dec.signal if h4_dec else 0
        h1_signal = h1_dec.signal if h1_dec else 0
        m5_signal = m5_dec.signal if m5_dec else 0

        final_signal = d_signal

        if h4_signal != 0 and h4_signal != d_signal:
            final_signal = 0
        if final_signal != 0 and h1_signal != 0 and h1_signal != d_signal:
            final_signal = 0
        if final_signal != 0 and m5_signal != 0 and m5_signal != d_signal:
            final_signal = 0

        # ── CRT gate — if CRT data available, prefer confirmed setups ── #
        # CRT doesn't block trades but boosts confidence when confirmed
        # Store for confidence scoring
        _crt_confirmed = crt_confirmed
        # ── Layer 6: ML gate ───────────────────────────────────── #
        ml_prob  = 0.5
        ml_allow = True

        if self.ml_enabled and final_signal != 0:
            prob     = getattr(daily_dec, "probability", None)
            smc_prob = prob.bullish if final_signal == 1 else prob.bearish if prob else 50.0
            ml_allow, ml_prob = self.ml.predict(pair, m5_df, final_signal, smc_prob)
            if not ml_allow:
                logger.debug(
                    f"  {pair}: ML blocked (P={ml_prob:.2f} < {self.ml.ml_threshold})"
                )
                final_signal = 0

        confidence = self._compute_confidence(
            daily_dec, h4_dec, h1_dec, m5_dec, final_signal, ml_prob,
            crt_confirmed=crt_confirmed,
        )
        entry_confs = getattr(daily_dec, 'confs', [])

        #return daily_dec, h4_dec, h1_dec, m5_dec, final_signal, confidence, entry_price, ml_prob

        return daily_dec, h4_dec, h1_dec, m5_dec, final_signal, confidence, entry_price, ml_prob, entry_confs


    # ══════════════════════════════════════════════════════════════ #
    #  Confidence scoring                                            #
    # ══════════════════════════════════════════════════════════════ #

    def _compute_confidence(
        self,
        daily_dec:     Optional[SwitchDecision],
        h4_dec:        Optional[SwitchDecision],
        h1_dec:        Optional[SwitchDecision],
        m5_dec:        Optional[SwitchDecision],
        signal:        int,
        ml_prob:       float = 0.5,
        crt_confirmed: bool  = False,
    ) -> float:
        if signal == 0 or daily_dec is None:
            return 0.0

        w     = CONF_WEIGHTS
        score = 0.0

        # SMC probability gap
        prob = getattr(daily_dec, "probability", None)
        if prob:
            score += min(prob.confidence / 100.0, 1.0) * w["prob_gap"]

        # ML probability (normalise: 0.5 = no info, 1.0 = full confidence)
        ml_norm = max(0.0, (ml_prob - 0.5) * 2)   # 0.5→0, 1.0→1
        score  += ml_norm * w["ml_prob"]

        # TF agreement
        if h4_dec and h4_dec.signal == signal: score += w["h4_agrees"]
        if h1_dec and h1_dec.signal == signal: score += w["h1_agrees"]
        if m5_dec and m5_dec.signal == signal: score += w["m5_agrees"]

        # Session
        if getattr(daily_dec, "good_session", False): score += w["session"]

        # Both TFs same structure
        ws = getattr(daily_dec, "weekly_structure", "")
        ds = getattr(daily_dec, "daily_structure",  "")
        if ws and ds and ws == ds and ws != "ranging":
            score += w["tf_aligned"]
       # CRT bonus — manipulation confirmed = higher quality setup
        if crt_confirmed:
            score += 0.10

        # Soft confluence bonus (Fib + LiqSweep bonuses from _generate_signal)
        bonus = getattr(self.switcher, '_last_confidence_bonus', 0.0)
        score += bonus

        return round(min(score, 1.0), 3)

    # ══════════════════════════════════════════════════════════════ #
    #  Core step                                                     #
    # ══════════════════════════════════════════════════════════════ #

    def step(
        self,
        pair:       str,
        utc_hour:   int   = None,
        live_price: float = None,
    ) -> dict:
        utc_hour = utc_hour or datetime.now(timezone.utc).hour

        result = self._analyse_pair(pair, utc_hour)
        daily_dec, h4_dec, h1_dec, m5_dec, final_signal, confidence, entry_price, ml_prob, entry_confs = result

        if daily_dec is None:
            return {"pair": pair, "action": "insufficient_data", "signal": 0}

        current_price = live_price or entry_price or self.history[pair].latest_price()
        if not current_price:
            return {"pair": pair, "action": "no_price", "signal": 0}

        state = self.states[pair]
        self._update_unrealised_pnl(pair, current_price)
        nav = self._nav({pair: current_price})

        trading_ok   = self.risk_mgr.check_daily_loss(nav)
        open_count   = sum(1 for s in self.states.values() if s.position != 0)
        # ── Dynamic max positions based on current NAV ────────────── #
        if nav < 50:
            dynamic_max = 4
        elif nav < 200:
            dynamic_max = 7
        elif nav < 1_000:
            dynamic_max = 8
        else:
            dynamic_max = 9
        self.risk_mgr.max_positions = dynamic_max
        can_open_new = self.risk_mgr.can_open_new(open_count)
        prev_signal  = state.position
        action       = "hold"
        trade_record = None

        # SL/TP check (sim)
        if state.position != 0 and self.sim_mode:
            hit = self._check_sl_tp(state, current_price)
            if hit:
                trade_record = self._close_position(
                    pair, current_price,
                    daily_dec.strategy, daily_dec.regime.value, reason=hit
                )
                action, final_signal = f"closed_{hit}", 0

        # Act on signal
        if trade_record is None and final_signal != prev_signal:
            if not trading_ok:
                action = "halted_daily_loss"
            elif final_signal != 0 and not can_open_new and prev_signal == 0:
                action = "halted_max_positions"
            else:
                if prev_signal != 0 and final_signal != prev_signal:
                    self._close_position(
                        pair, current_price,
                        daily_dec.strategy, daily_dec.regime.value,
                        reason="signal_flip"
                    )
                if final_signal != 0 and trading_ok and can_open_new:
                    d_df  = self.history[pair].get("D")
                    h4_df = self.history[pair].get("H4")
                    setup = self.risk_mgr.build_trade_setup(
                        instrument  = pair,
                        direction   = final_signal,
                        entry_price = current_price,
                        account_nav = nav,
                        price_df    = d_df,
                        h4_df       = h4_df,
                        confidence  = confidence,
                        swing_tp    = getattr(daily_dec, "swing_tp",  0.0),
                        liq_zones   = getattr(daily_dec, "liq_zones", []),
                        near_ath    = getattr(daily_dec, "near_ath",  False),
                        near_atl    = getattr(daily_dec, "near_atl",  False),
                    )
                    if setup:
                        trade_record = self._open_position(
                            pair, setup,
                            daily_dec.strategy, daily_dec.regime.value,
                            ml_prob=ml_prob,
                            confs=entry_confs,
                        )
                        action = ("open_long" if final_signal == 1 else "open_short")                                  if trade_record else "order_failed"
                    else:
                        action = "setup_invalid"

        # Pip P&L for closed trade
        pip_pnl = 0.0
        if trade_record and trade_record.direction == "close":
            from config import PIP_SIZE
            pip  = PIP_SIZE.get(pair, 0.0001)
            pnl  = trade_record.pnl
            pip_pnl = round(pnl / (abs(state.units or 1) * pip + 1e-9), 1)

        status = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "pair":          pair,
            "price":         current_price,
            "entry_price":   entry_price,
            "final_signal":  final_signal,
            "daily_signal":  daily_dec.signal,
            "h4_signal":     h4_dec.signal    if h4_dec else 0,
            "h1_signal":     h1_dec.signal    if h1_dec else 0,
            "m5_signal":     m5_dec.signal    if m5_dec else 0,
            "ml_prob":       ml_prob,
            "ml_fitted":     self.ml.is_fitted(pair),
            "weekly_struct": getattr(daily_dec, "weekly_structure", ""),
            "daily_struct":  getattr(daily_dec, "daily_structure",  ""),
            "regime":        daily_dec.regime.value,
            "strategy":      daily_dec.strategy,
            "confidence":    confidence,
            "action":        action,
            "good_session":  daily_dec.good_session,
            "reason":        daily_dec.reason,
            "trade":         asdict(trade_record) if trade_record else None,
            "position":      state.position,
            "unrealized_pnl": round(state.unrealized_pnl, 2),
            "cash_balance":  round(self.cash_balance, 2),
            "nav":           self._nav({pair: current_price}),
        }
        self._write_log(status)
        return status

    # ══════════════════════════════════════════════════════════════ #
    #  Position management                                           #
    # ══════════════════════════════════════════════════════════════ #

    def _open_position(
        self, pair, setup, strategy, regime, ml_prob=0.5, confs=None
    ) -> Optional[Trade]:
        print(f"[DEBUG] _open_position called | broker={self.broker.name} | pair={pair}")  # remove after confirmed

        state = self.states[pair]

        if not self.sim_mode:
            result = self.broker.open_trade(
                pair, setup.units, setup.sl_price, setup.tp_price
            )
            if not result.success:
                logger.error(f"Broker open failed {pair}: {result.error}")
                return None
            state.oanda_trade_id = result.trade_id

        state.position          = setup.direction
        state.units             = setup.units
        state.entry_price       = setup.entry_price
        state.sl_price          = setup.sl_price
        state.tp_price          = setup.tp_price
        state.unrealized_pnl    = 0.0
        state.ml_prob_at_entry  = ml_prob
        state.liq_zones         = getattr(setup, "liq_zones", [])

        t = Trade(
            timestamp  = datetime.now(timezone.utc).isoformat(),
            pair       = pair,
            direction  = "long" if setup.direction == 1 else "short",
            units      = abs(setup.units),
            price      = setup.entry_price,
            sl_price   = setup.sl_price,
            tp_price   = setup.tp_price,
            strategy   = strategy,
            regime     = regime,
            sl_pips    = setup.sl_pips,
            tp_pips    = setup.tp_pips,
            rr_ratio   = setup.rr_ratio,
            pnl        = 0.0,
            balance    = round(self.cash_balance, 2),
            confidence = getattr(setup, "score", 0.0),
            ml_prob    = ml_prob,
            confs      = confs or [],

        )
        self.trade_log.append(t)
        ml_tag = f"ML={ml_prob:.2f}" if self.ml_enabled else "ML=OFF"
        logger.info(
            f"OPEN  {pair} {'LONG' if setup.direction==1 else 'SHORT'} "
            f"@ {setup.entry_price:.5f} [M2] | "
            f"SL={setup.sl_price:.5f} ({setup.sl_pips:.0f}p) | "
            f"TP={setup.tp_price:.5f} ({setup.tp_pips:.0f}p) | "
            f"units={abs(setup.units):,} | Risk=${setup.risk_amount:.2f} | "
            f"RR=1:{setup.rr_ratio:.1f} | conf={getattr(setup,'score',0):.2f} | {ml_tag}"
        )
        print_trade_opened(pair, "long" if setup.direction == 1 else "short", setup)
 
        # ── Telegram signal ──────────────────────────────────────── #
        if self.telegram:
            try:
                # Pull SMC context from the daily decision stored on state
                # (daily_dec is not a parameter here, so we reconstruct from
                #  the history we already have — cheap because it's cached)
                h         = self.history[pair]
                d_df      = h.get("D")
                h4_df     = h.get("H4")
                utc_hour  = datetime.now(timezone.utc).hour
                daily_dec = self.switcher.decide(
                    d_df["price"].dropna(), pair,
                    d_df["high"] if "high" in d_df.columns else None,
                    d_df["low"]  if "low"  in d_df.columns else None,
                    utc_hour,
                )
                h4_dec = self.switcher.decide(
                    h4_df["price"].dropna(), pair,
                    h4_df["high"] if "high" in h4_df.columns else None,
                    h4_df["low"]  if "low"  in h4_df.columns else None,
                    utc_hour,
                ) if not h4_df.empty else None
 
                prob     = getattr(daily_dec, "probability", None)
                bull_pct = getattr(prob, "bullish",   50.0) if prob else 50.0
                bear_pct = getattr(prob, "bearish",   50.0) if prob else 50.0
 
                # Build PD array strings from daily_dec
                pd_strings  = []
                fib_strings = []
                pd_arrays   = getattr(daily_dec, "pd_arrays",  [])
                fib_zones   = getattr(daily_dec, "fib_zones",  [])
                for a in (pd_arrays or []):
                    kind  = getattr(a, "kind",   "?")
                    bias  = getattr(a, "bias",   "")
                    bot   = getattr(a, "bottom", 0.0)
                    top   = getattr(a, "top",    0.0)
                    pd_strings.append(f"{kind:<4} [{bias.upper()}] {bot:.5f} — {top:.5f}   ACTIVE")
                for z in (fib_zones or []):
                    level = getattr(z, "level",  "")
                    ztype = getattr(z, "ztype",  "")
                    price = getattr(z, "price",  0.0)
                    fib_strings.append(f"Fib {level} [{ztype.upper()}] @ {price:.5f}")
 
                self.telegram.send_signal(
                    pair             = pair,
                    direction        = "LONG" if setup.direction == 1 else "SHORT",
                    entry            = setup.entry_price,
                    sl               = setup.sl_price,
                    tp               = setup.tp_price,
                    units            = abs(setup.units),
                    confidence       = getattr(setup, "score", 0.0),
                    ml_prob          = ml_prob,
                    regime           = getattr(daily_dec, "regime", type("R", (), {"value": regime})()).value,
                    strategy         = strategy,
                    weekly_structure = getattr(daily_dec, "weekly_structure", ""),
                    daily_structure  = getattr(daily_dec, "daily_structure",  ""),
                    h4_context       = getattr(h4_dec,    "daily_structure",  "") if h4_dec else "",
                    bull_pct         = bull_pct,
                    bear_pct         = bear_pct,
                    pd_arrays        = pd_strings,
                    fib_zones        = fib_strings,
                    reason           = getattr(daily_dec, "reason", ""),
                    session_ok       = getattr(daily_dec, "good_session", True),
                    price            = setup.entry_price,
                )
            except Exception as tg_err:
                logger.debug(f"[TELEGRAM] send_signal failed: {tg_err}")
 
        return t

    def _close_position(
        self, pair, price, strategy, regime, reason="signal"
    ) -> Optional[Trade]:
        state = self.states[pair]
        if state.position == 0:
            return None

        pnl = state.units * (price - state.entry_price)
        self.cash_balance += pnl

        if not self.sim_mode:
            if state.oanda_trade_id:
                self.broker.close_trade(state.oanda_trade_id, pair)
            else:
                self.broker.close_all(pair)

        from config import PIP_SIZE
        pip     = PIP_SIZE.get(pair, 0.0001)
        pnl_pip = round(pnl / (abs(state.units) * pip + 1e-9), 1)

        t = Trade(
            timestamp = datetime.now(timezone.utc).isoformat(),
            pair=pair, direction="close", units=abs(state.units),
            price=price, sl_price=state.sl_price, tp_price=state.tp_price,
            strategy=strategy, regime=regime,
            pnl=round(pnl, 2), pnl_pips=pnl_pip,
            balance=round(self.cash_balance, 2),
        )
        self.trade_log.append(t)

        # ── Record confluence outcome for learning ─────────────── #
        # Find the matching open trade to get its confluence list
        open_trade = next(
            (tr for tr in reversed(self.trade_log)
             if tr.pair == pair and tr.direction in ("long", "short")
             and tr.confs),
            None
        )
        if open_trade and open_trade.confs:
            self.confluence_tracker.record_trade(
                confs = open_trade.confs,
                pnl   = round(pnl, 2),
                pips  = pnl_pip,
            )
        prev = "LONG" if state.position == 1 else "SHORT"
        logger.info(
            f"CLOSE {pair} {prev} @ {price:.5f} | "
            f"PnL=${pnl:.2f} ({pnl_pip:+.1f} pips) | "
            f"Cash=${self.cash_balance:.2f} | reason={reason}"
        )
        print_trade_closed(pair, reason, round(pnl, 2), round(self.cash_balance, 2))
 
        # ── Telegram close notification ──────────────────────────── #
        if self.telegram:
            try:
                self.telegram.send_close(
                    pair      = pair,
                    direction = prev,
                    pnl       = round(pnl, 2),
                    pnl_pips  = pnl_pip,
                    reason    = reason,
                    cash      = round(self.cash_balance, 2),
                    nav       = self._nav(),
                )
            except Exception as tg_err:
                logger.debug(f"[TELEGRAM] send_close failed: {tg_err}")
 
        state.position = state.units = 0

        state.entry_price = state.sl_price = state.tp_price = 0.0
        state.unrealized_pnl = state.ml_prob_at_entry = 0.0
        state.oanda_trade_id = ""
        return t

    def _check_sl_tp(self, state: PairState, price: float) -> Optional[str]:
        if state.position == 0: return None
        if state.position == 1:
            if price <= state.sl_price: return "sl"
            if price >= state.tp_price: return "tp"
        else:
            if price >= state.sl_price: return "sl"
            if price <= state.tp_price: return "tp"
        return None

    # ══════════════════════════════════════════════════════════════ #
    #  Legacy compatibility                                          #
    # ══════════════════════════════════════════════════════════════ #

    def load_history(self, pair: str, df: pd.DataFrame, gran: str = "H1"):
        self.history[pair].set(gran, df)

    # ══════════════════════════════════════════════════════════════ #
    #  Logging                                                       #
    # ══════════════════════════════════════════════════════════════ #

    def _write_log(self, status: dict):
        with open(self._log_path, "a") as f:
            f.write(json.dumps(status, default=str) + "\n")
        with open(self._state_path, "w") as f:
            json.dump({**status, "portfolio": self.portfolio_summary()},
                      f, indent=2, default=str)

    def load_trade_log(self) -> pd.DataFrame:
        rows = []
        if os.path.exists(self._log_path):
            with open(self._log_path) as f:
                for line in f:
                    try: rows.append(json.loads(line))
                    except: pass
        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════ #
    #  Live loop                                                     #
    # ══════════════════════════════════════════════════════════════ #

    def run_live(self, bars: int = None):
        if not self.oanda:
            raise RuntimeError("OandaClient required for live mode.")

        self._seed_all()   # W+D+H4+H1+M5+M2 + initial ML training

        logger.info(
            f"Live loop started | {len(self.instruments)} pairs | "
            f"interval={self.loop_interval}s | "
            f"stack=W/D/H4/H1/M5(conf)/M2(entry) | "
            f"ML={'ENABLED' if self.ml_enabled else 'DISABLED'}"
        )

        while True:
            try:
                self._loop_count += 1
                utc_now  = datetime.now(timezone.utc)
                utc_hour = utc_now.hour

                # Refresh all TFs on their schedules
                prices: Dict[str, float] = {}
                for pair in self.instruments:
                    self._refresh_pair(pair, utc_now)
                    try:
                        tick         = self.oanda.get_current_price(pair)
                        prices[pair] = tick["mid"]
                    except Exception:
                        prices[pair] = self.history[pair].latest_price() or 0.0

                # Update open positions + SL/TP
                open_count = 0
                for pair, state in self.states.items():
                    if state.position == 0:
                        continue
                    open_count += 1
                    price = prices.get(pair, state.entry_price)
                    self._update_unrealised_pnl(pair, price)
                    if self.sim_mode:
                        hit = self._check_sl_tp(state, price)
                        if hit:
                            d_df = self.history[pair].get("D")
                            dec  = self.switcher.decide(
                                d_df["price"].dropna(), pair,
                                d_df.get("high"), d_df.get("low"), utc_hour
                            ) if not d_df.empty else None
                            self._close_position(
                                pair, price,
                                dec.strategy    if dec else "unknown",
                                dec.regime.value if dec else "unknown",
                                reason=hit,
                            )
                            open_count -= 1

                # Analyse all pairs
                all_daily: Dict[str, Optional[SwitchDecision]] = {}
                all_h4:    Dict[str, Optional[SwitchDecision]] = {}
                candidates = []

                for pair in self.instruments:
                    (daily_dec, h4_dec, h1_dec, m5_dec,
                     final_signal, confidence, entry_price, ml_prob, entry_confs) = \
                        self._analyse_pair(pair, utc_hour)

                    all_daily[pair] = daily_dec
                    all_h4[pair]    = h4_dec

                    if daily_dec is None:
                        d_n  = len(self.history[pair].get("D"))
                        h4_n = len(self.history[pair].get("H4"))
                        logger.info(
                            f"  {pair}: data seeding "
                            f"(D={d_n}/{TF_MIN_BARS['D']} H4={h4_n}/{TF_MIN_BARS['H4']})"
                        )
                        continue

                    if final_signal != 0 and self.states[pair].position == 0:
                        dir_str  = "LONG" if final_signal == 1 else "SHORT"
                        prob     = getattr(daily_dec, "probability", None)
                        h4_ok    = "✓" if (h4_dec and h4_dec.signal == final_signal) else "~"
                        h1_ok    = "✓" if (h1_dec and h1_dec.signal == final_signal) else "~"
                        m5_ok    = "✓" if (m5_dec and m5_dec.signal == final_signal) else "~"
                        ml_ok    = f"ML={ml_prob:.2f}" if self.ml_enabled else "ML=OFF"
                        logger.info(
                            f"  ▶ {pair} {dir_str} @ {entry_price:.5f} [M2] | "
                            f"W={daily_dec.weekly_structure} D={daily_dec.daily_structure} | "
                            f"H4{h4_ok} H1{h1_ok} M5{m5_ok} | {ml_ok} | "
                            f"Bull={prob.bullish:.0f}% Bear={prob.bearish:.0f}% | "
                            f"conf={confidence:.2f}"
                        )
                        candidates.append({
                            "pair":        pair,
                            "signal":      final_signal,
                            "confidence":  confidence,
                            "price":       entry_price,
                            "daily_dec":   daily_dec,
                            "ml_prob":     ml_prob,
                            "entry_confs": entry_confs,
                        })
                    elif daily_dec and daily_dec.signal != 0:
                        # Signal existed but filtered — log reason
                        d_s  = daily_dec.signal
                        h4_s = h4_dec.signal if h4_dec else 0
                        h1_s = h1_dec.signal if h1_dec else 0
                        m5_s = m5_dec.signal if m5_dec else 0
                        reasons = []
                        if h4_s != 0 and h4_s != d_s: reasons.append(f"H4≠D")
                        if h1_s != 0 and h1_s != d_s: reasons.append(f"H1≠D")
                        if m5_s != 0 and m5_s != d_s: reasons.append(f"M5≠D")
                        if ml_prob < self.ml.ml_threshold and self.ml_enabled:
                            reasons.append(f"ML={ml_prob:.2f}<{self.ml.ml_threshold}")
                        if reasons:
                            logger.info(f"  {pair}: filtered [{', '.join(reasons)}]")

                # Execute top-ranked signals (max 2)
                # ── Dynamic max positions (live loop) ─────────────── #
                # ── Dynamic max positions (live loop) ─────────────── #
                # Check REAL open positions from MT5 directly,
                # not just what the bot opened this session.
                nav = self._nav(prices)
                if nav < 50:
                    dynamic_max = 4
                elif nav < 200:
                    dynamic_max = 7
                elif nav < 1_000:
                    dynamic_max = 8
                else:
                    dynamic_max = 9
                self.risk_mgr.max_positions = dynamic_max

                # Get real count from MT5 if available
                if hasattr(self.broker, "get_open_position_count"):
                    real_open = self.broker.get_open_position_count()
                else:
                    real_open = open_count

                available_slots = dynamic_max - real_open
                logger.info(
                    f"[POSITIONS] NAV=${nav:.2f} → "
                    f"max_trades={dynamic_max} | "
                    f"open_MT5={real_open} | "
                    f"open_bot={open_count} | "
                    f"slots={available_slots}"
                )


                if available_slots > 0 and candidates:
                    candidates.sort(key=lambda x: x["confidence"], reverse=True)
                    nav = self._nav(prices)
                    for cand in candidates[:available_slots]:
                        if not self.risk_mgr.check_daily_loss(nav):
                            logger.warning("Daily loss limit — no new trades.")
                            break
                        d_df  = self.history[cand["pair"]].get("D")
                        h4_df    = self.history[cand["pair"]].get("H4")
                        d_dec    = cand["daily_dec"]
                        setup = self.risk_mgr.build_trade_setup(
                            instrument  = cand["pair"],
                            direction   = cand["signal"],
                            entry_price = cand["price"],
                            account_nav = nav,
                            price_df    = d_df,
                            h4_df       = h4_df,
                            confidence  = cand["confidence"],
                            swing_tp    = getattr(d_dec, "swing_tp",  0.0),
                            liq_zones   = getattr(d_dec, "liq_zones", []),
                            near_ath    = getattr(d_dec, "near_ath",  False),
                            near_atl    = getattr(d_dec, "near_atl",  False),
                        )
                        if setup:
                            trade = self._open_position(
                                cand["pair"], setup,
                                cand["daily_dec"].strategy,
                                cand["daily_dec"].regime.value,
                                ml_prob=cand["ml_prob"],
                                confs=cand.get("entry_confs", []),
                            )
                            if trade:
                                open_count      += 1
                                available_slots -= 1

                # Portfolio + ML status
                port = self.portfolio_summary(prices)
                print_smc_dashboard(
                    decisions=all_daily, h4_decisions=all_h4,
                    prices=prices, portfolio=port, open_states=self.states,
                    loop_count=self._loop_count,
                    next_check_secs=self.loop_interval,
                    instruments=self.instruments,
                )
                logger.info(
                    f"[{utc_now.strftime('%H:%M')} UTC #{self._loop_count}] "
                    f"Cash=${port['cash_balance']:,.2f} | NAV=${port['nav']:,.2f} | "
                    f"PnL=${port['total_pnl']:+,.2f} | Return={port['return_pct']:+.2f}% | "
                    f"Trades={port['total_trades']} (W:{port.get('wins',0)}/L:{port.get('losses',0)}) | "
                    f"Open={open_count}/{self.risk_mgr.max_positions} | "
                    f"ML fitted={port['ml_fitted']}"
                )
                self._tg_loop_counter += 1
                if (
                    self.telegram
                    and self._tg_wait_every > 0
                    and self._tg_loop_counter % self._tg_wait_every == 0
                ):
                    for pair in self.instruments:
                        state = self.states[pair]
                        if state.position != 0:
                            continue   # already open — no WAIT message needed
                        d_dec = all_daily.get(pair)
                        if d_dec is None:
                            continue
                        h4_dec  = all_h4.get(pair)
                        prob    = getattr(d_dec, "probability", None)
                        bull_pct = getattr(prob, "bullish",  50.0) if prob else 50.0
                        bear_pct = getattr(prob, "bearish",  50.0) if prob else 50.0
 
                        pd_strings  = []
                        fib_strings = []
                        for a in (getattr(d_dec, "pd_arrays",  []) or []):
                            kind  = getattr(a, "kind",   "?")
                            bias  = getattr(a, "bias",   "")
                            bot   = getattr(a, "bottom", 0.0)
                            top   = getattr(a, "top",    0.0)
                            pd_strings.append(f"{kind:<4} [{bias.upper()}] {bot:.5f} — {top:.5f}   ACTIVE")
                        for z in (getattr(d_dec, "fib_zones",  []) or []):
                            level = getattr(z, "level",  "")
                            ztype = getattr(z, "ztype",  "")
                            zprice = getattr(z, "price", 0.0)
                            fib_strings.append(f"Fib {level} [{ztype.upper()}] @ {zprice:.5f}")
 
                        try:
                            self.telegram.send_no_signal(
                                pair             = pair,
                                price            = prices.get(pair, 0.0),
                                session_ok       = getattr(d_dec, "good_session",     True),
                                weekly_structure = getattr(d_dec, "weekly_structure", ""),
                                daily_structure  = getattr(d_dec, "daily_structure",  ""),
                                h4_context       = getattr(h4_dec, "daily_structure", "") if h4_dec else "",
                                regime           = d_dec.regime.value,
                                bull_pct         = bull_pct,
                                bear_pct         = bear_pct,
                                pd_arrays        = pd_strings,
                                fib_zones        = fib_strings,
                                reason           = getattr(d_dec, "reason", ""),
                            )
                        except Exception as tg_err:
                            logger.debug(f"[TELEGRAM] send_no_signal {pair}: {tg_err}")

                for p, pos in port["open_positions"].items():
                    logger.info(
                        f"  OPEN {p} {pos['direction'].upper()} @ {pos['entry']:.5f} | "
                        f"uPnL=${pos['upnl']:+.2f} | "
                        f"SL={pos['sl']:.5f} TP={pos['tp']:.5f} | "
                        f"ML@entry={pos['ml_prob']:.2f}"
                    )

            except KeyboardInterrupt:
                logger.info("Stopped — closing all positions.")
                self._close_all_on_exit(prices)
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)

            time.sleep(self.loop_interval)

    def _close_all_on_exit(self, prices: Dict[str, float] = None):
        prices = prices or {}
        for pair, state in self.states.items():
            if state.position != 0:
                price = (prices.get(pair)
                         or self.history[pair].latest_price()
                         or state.entry_price)
                self._close_position(pair, price, "shutdown", "shutdown", reason="shutdown")

    # ══════════════════════════════════════════════════════════════ #
    #  Backtest                                                      #
    # ══════════════════════════════════════════════════════════════ #

    def run_backtest_mode(self) -> pd.DataFrame:
        min_warmup = max(
            self.config.sma_l, self.config.boll_sma * 3,
            self.config.hurst_lags * 3, TF_MIN_BARS["D"],
        )
        gran    = "D"
        lengths = [len(self.history[p].get(gran)) for p in self.instruments]
        if max(lengths, default=0) < min_warmup:
            gran    = "H1"
            lengths = [len(self.history[p].get(gran)) for p in self.instruments]
        lengths = [l for l in lengths if l > 0]
        if not lengths:
            raise RuntimeError("No price history loaded.")

        sim_len = min(lengths)
        logger.info(f"Backtest: {sim_len} {gran} bars x {len(self.instruments)} pairs")
        full    = {p: self.history[p].get(gran).copy() for p in self.instruments}
        results = []

        for i in range(min_warmup, sim_len):
            for pair in self.instruments:
                if len(full[pair]) <= i:
                    continue
                self.history[pair].set(gran, full[pair].iloc[:i + 1])
                h4_full = self.history[pair].get("H4")
                if not h4_full.empty:
                    self.history[pair].set("H4", h4_full.iloc[:min(i, len(h4_full)-1) + 1])
                # Also slice M5 for ML training during backtest
                m5_full = self.history[pair].get("M5")
                if not m5_full.empty:
                    m5_i = min(i * 3, len(m5_full) - 1)
                    self.history[pair].set("M5", m5_full.iloc[:m5_i + 1])
                status = self.step(pair, utc_hour=9)
                results.append(status)

        for pair in self.instruments:
            self.history[pair].set(gran, full[pair])

        logger.info(f"Backtest complete. {len(results)} steps.")
        return pd.DataFrame(results)
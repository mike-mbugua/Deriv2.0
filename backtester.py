"""
backtester.py
Three vectorized strategy backtesting classes that all operate on
DataFrames fetched from OANDA via DataFeed.

Classes:
  BaseBacktester      — shared data loading & performance metrics
  SMABacktester       — SMA crossover (trending regime)
  ConBacktester       — contrarian rolling-sign (random regime)
  MeanRevBacktester   — Bollinger Bands mean reversion (oscillating regime)
"""

import logging
from itertools import product
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ================================================================== #
#  Base                                                               #
# ================================================================== #

class BaseBacktester:
    """
    Shared data loading, ATR calculation, and performance summary.
    Sub-classes call load_from_df() then implement test_strategy().
    """

    def __init__(self, symbol: str, tc: float = 0.00007):
        self.symbol  = symbol
        self.tc      = tc
        self.results: Optional[pd.DataFrame] = None
        self.data:    Optional[pd.DataFrame] = None

    # ── Data loading ─────────────────────────────────────────────── #

    def load_from_df(self, df: pd.DataFrame):
        """
        Accept a DataFrame from DataFeed (columns: price, open, high, low, volume).
        Normalises the index, adds 'returns', keeps all OHLC columns.
        """
        df = df.copy()
        # Normalise price column name
        if "price" not in df.columns:
            for col in ("close", "Close", "Adj Close"):
                if col in df.columns:
                    df = df.rename(columns={col: "price"})
                    break
        required = {"price"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df["returns"] = np.log(df["price"] / df["price"].shift(1))
        self.data = df.dropna(subset=["price"])

    # ── ATR helper (used for SL/TP sizing) ───────────────────────── #

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range — requires high / low / price (close)."""
        if not {"high", "low", "price"}.issubset(df.columns):
            # Proxy ATR from close only (less accurate)
            return df["price"].pct_change().abs().rolling(period).mean() * df["price"]
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["price"].shift(1)).abs(),
            (df["low"]  - df["price"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, min_periods=period).mean()

    # ── Performance summary ──────────────────────────────────────── #

    def performance_summary(self) -> dict:
        if self.results is None:
            return {}
        r          = self.results
        ann        = 252
        s          = r["strategy"]
        s_std      = float(s.std())
        s_mean     = float(s.mean())
        cr         = float(r["cstrategy"].iloc[-1])
        bh         = float(r["creturns"].iloc[-1])
        drawdown   = float((r["cstrategy"].cummax() - r["cstrategy"]).max())
        sharpe     = (s_mean * ann) / (s_std * np.sqrt(ann) + 1e-9)
        num_trades = int(r["trades"].sum()) if "trades" in r.columns else 0
        return {
            "total_return":   round(cr,   4),
            "buy_hold":       round(bh,   4),
            "outperformance": round(cr - bh, 4),
            "ann_return":     round(s_mean * ann, 6),
            "ann_risk":       round(s_std * np.sqrt(ann), 6),
            "sharpe":         round(sharpe, 4),
            "max_drawdown":   round(drawdown, 4),
            "num_trades":     num_trades,
        }

    def _assert_data(self):
        if self.data is None:
            raise RuntimeError("Call load_from_df() first.")


# ================================================================== #
#  SMA Crossover                                                      #
# ================================================================== #

class SMABacktester(BaseBacktester):
    """SMA crossover: go long when SMA_S > SMA_L, short otherwise."""

    def __init__(self, symbol: str, SMA_S: int, SMA_L: int, tc: float = 0.00007):
        super().__init__(symbol, tc)
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L

    def __repr__(self):
        return f"SMABacktester({self.symbol}, S={self.SMA_S}, L={self.SMA_L})"

    def prepare_data(self):
        self._assert_data()
        self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()

    def set_parameters(self, SMA_S: int = None, SMA_L: int = None):
        if SMA_S is not None:
            self.SMA_S = SMA_S
        if SMA_L is not None:
            self.SMA_L = SMA_L
        if self.data is not None:
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()

    def test_strategy(self) -> Tuple[float, float]:
        self._assert_data()
        if "SMA_S" not in self.data.columns:
            self.prepare_data()
        d = self.data.copy().dropna()
        d["position"] = np.where(d["SMA_S"] > d["SMA_L"], 1, -1)
        d["strategy"] = d["position"].shift(1) * d["returns"]
        d["trades"]   = d["position"].diff().fillna(0).abs()
        d["strategy"] -= d["trades"] * self.tc
        d.dropna(inplace=True)
        d["creturns"]  = d["returns"].cumsum().apply(np.exp)
        d["cstrategy"] = d["strategy"].cumsum().apply(np.exp)
        self.results   = d
        perf    = round(float(d["cstrategy"].iloc[-1]), 6)
        outperf = round(float(perf - d["creturns"].iloc[-1]), 6)
        return perf, outperf

    def optimize(
        self,
        SMA_S_range: Tuple[int, int, int],
        SMA_L_range: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int], float]:
        combos = [
            (s, l)
            for s, l in product(range(*SMA_S_range), range(*SMA_L_range))
            if s < l
        ]
        if not combos:
            raise ValueError("No valid SMA_S < SMA_L combinations.")
        results = []
        for s, l in combos:
            self.set_parameters(s, l)
            try:
                results.append(self.test_strategy()[0])
            except Exception:
                results.append(0.0)
        best_idx = int(np.argmax(results))
        opt = combos[best_idx]
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
        self.results_overview = pd.DataFrame(combos, columns=["SMA_S", "SMA_L"])
        self.results_overview["performance"] = results
        return opt, round(float(np.max(results)), 6)


# ================================================================== #
#  Contrarian                                                         #
# ================================================================== #

class ConBacktester(BaseBacktester):
    """Contrarian: fade recent momentum using rolling return sign."""

    def __init__(self, symbol: str, tc: float = 0.00007):
        super().__init__(symbol, tc)
        self.window = 1

    def __repr__(self):
        return f"ConBacktester({self.symbol}, window={self.window})"

    def test_strategy(self, window: int = 1) -> Tuple[float, float]:
        self._assert_data()
        self.window = window
        d = self.data.copy().dropna()
        d["position"] = -np.sign(d["returns"].rolling(window).mean())
        d["strategy"] = d["position"].shift(1) * d["returns"]
        d["trades"]   = d["position"].diff().fillna(0).abs()
        d["strategy"] -= d["trades"] * self.tc
        d.dropna(inplace=True)
        d["creturns"]  = d["returns"].cumsum().apply(np.exp)
        d["cstrategy"] = d["strategy"].cumsum().apply(np.exp)
        self.results   = d
        perf    = round(float(d["cstrategy"].iloc[-1]), 6)
        outperf = round(float(perf - d["creturns"].iloc[-1]), 6)
        return perf, outperf

    def optimize(
        self, window_range: Tuple[int, int, int]
    ) -> Tuple[int, float]:
        windows = list(range(*window_range))
        if not windows:
            raise ValueError("Empty window_range.")
        results = []
        for w in windows:
            try:
                results.append(self.test_strategy(w)[0])
            except Exception:
                results.append(0.0)
        best_w = windows[int(np.argmax(results))]
        self.test_strategy(best_w)
        self.results_overview = pd.DataFrame({"window": windows, "performance": results})
        return best_w, round(float(np.max(results)), 6)


# ================================================================== #
#  Bollinger Bands Mean Reversion                                     #
# ================================================================== #

class MeanRevBacktester(BaseBacktester):
    """
    Bollinger Band mean reversion:
      price < lower band → long
      price > upper band → short
      price crosses SMA  → close
    """

    def __init__(self, symbol: str, SMA: int, dev: float, tc: float = 0.00007):
        super().__init__(symbol, tc)
        self.SMA = SMA
        self.dev = dev

    def __repr__(self):
        return f"MeanRevBacktester({self.symbol}, SMA={self.SMA}, dev={self.dev})"

    def test_strategy(
        self, SMA: int = None, dev: float = None
    ) -> Tuple[float, float]:
        self._assert_data()
        if SMA is not None:
            self.SMA = SMA
        if dev is not None:
            self.dev = dev
        d = self.data.copy().dropna()
        d["SMA"]      = d["price"].rolling(self.SMA).mean()
        d["std"]      = d["price"].rolling(self.SMA).std()
        d["Lower"]    = d["SMA"] - self.dev * d["std"]
        d["Upper"]    = d["SMA"] + self.dev * d["std"]
        d["distance"] = d["price"] - d["SMA"]
        d.dropna(inplace=True)

        d["position"] = np.where(d["price"] < d["Lower"],  1, np.nan)
        d["position"] = np.where(d["price"] > d["Upper"], -1, d["position"])
        # Zero out when price crosses SMA
        d["position"] = np.where(
            d["distance"] * d["distance"].shift(1) < 0, 0, d["position"]
        )
        d["position"] = d["position"].ffill().fillna(0)

        d["strategy"] = d["position"].shift(1) * d["returns"]
        d["trades"]   = d["position"].diff().fillna(0).abs()
        d["strategy"] -= d["trades"] * self.tc
        d.dropna(inplace=True)
        d["creturns"]  = d["returns"].cumsum().apply(np.exp)
        d["cstrategy"] = d["strategy"].cumsum().apply(np.exp)
        self.results   = d
        perf    = round(float(d["cstrategy"].iloc[-1]), 6)
        outperf = round(float(perf - d["creturns"].iloc[-1]), 6)
        return perf, outperf

    def optimize(
        self,
        SMA_range: Tuple[int, int, int],
        dev_range: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, float], float]:
        combos  = list(product(
            range(*SMA_range),
            [round(x * 0.5, 1) for x in range(*dev_range)]
        ))
        results = []
        for s, d in combos:
            try:
                results.append(self.test_strategy(s, d)[0])
            except Exception:
                results.append(0.0)
        best_idx = int(np.argmax(results))
        opt = combos[best_idx]
        self.test_strategy(opt[0], opt[1])
        self.results_overview = pd.DataFrame(combos, columns=["SMA", "dev"])
        self.results_overview["performance"] = results
        return opt, round(float(np.max(results)), 6)

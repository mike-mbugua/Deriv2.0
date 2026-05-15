"""
backtest_runner.py
──────────────────
Orchestrates backtesting of all three strategies + the regime switcher
on the same OANDA dataset, for any of the 5 supported pairs.
Also trains the ML session analyser on hourly data.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtester import SMABacktester, ConBacktester, MeanRevBacktester
from config import BotConfig, INSTRUMENTS
from data_feed import DataFeed, OandaClient
from strategy_switcher import (
    RegimeDetector, StrategySwitcher, StrategyConfig,
    SessionAnalyser, Regime,
)

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Usage
    -----
        runner = BacktestRunner(oanda_client, config)
        runner.load_data("GBP_USD")
        results = runner.run_all()
        runner.print_summary(results)
        runner.compare_with_regime_switcher()

    Or run all pairs:
        runner.run_all_pairs()
    """

    def __init__(
        self,
        oanda_client: OandaClient,
        config:       BotConfig = None,
    ):
        self.oanda   = oanda_client
        self.cfg     = config or BotConfig()
        self.feed    = DataFeed(oanda_client, cache_dir="data")
        self._df:    Optional[pd.DataFrame] = None
        self._symbol: str = ""

        self.sma: Optional[SMABacktester]      = None
        self.con: Optional[ConBacktester]      = None
        self.mrv: Optional[MeanRevBacktester]  = None

    # ── Data loading ─────────────────────────────────────────────── #

    def load_data(
        self,
        instrument: str,
        granularity: str = "D",
        bars:        int = None,
    ):
        """Fetch history from OANDA (or local cache) and prepare backtester instances."""
        bars = bars or self.cfg.history_bars
        logger.info(f"Loading {instrument} | {granularity} | {bars} bars")

        self._df     = self.feed.get_history(instrument, granularity, bars)
        self._symbol = instrument

        self.sma = SMABacktester(instrument, self.cfg.sma_s, self.cfg.sma_l, self.cfg.tc)
        self.con = ConBacktester(instrument, self.cfg.tc)
        self.mrv = MeanRevBacktester(instrument, self.cfg.boll_sma, self.cfg.boll_dev, self.cfg.tc)

        for bt in (self.sma, self.con, self.mrv):
            bt.load_from_df(self._df)

        self.sma.prepare_data()
        logger.info(
            f"Data ready: {len(self._df)} bars | "
            f"{self._df.index[0].date()} → {self._df.index[-1].date()}"
        )

    def _assert_data(self):
        if self._df is None:
            raise RuntimeError("Call load_data() first.")

    # ── Individual strategy runs ─────────────────────────────────── #

    def run_sma(self) -> dict:
        self._assert_data()
        self.sma.test_strategy()
        s = self.sma.performance_summary()
        s["strategy_name"] = f"SMA({self.cfg.sma_s}/{self.cfg.sma_l})"
        return s

    def run_contrarian(self) -> dict:
        self._assert_data()
        self.con.test_strategy(window=self.cfg.con_window)
        s = self.con.performance_summary()
        s["strategy_name"] = f"Contrarian(w={self.cfg.con_window})"
        return s

    def run_mean_rev(self) -> dict:
        self._assert_data()
        self.mrv.test_strategy(SMA=self.cfg.boll_sma, dev=self.cfg.boll_dev)
        s = self.mrv.performance_summary()
        s["strategy_name"] = f"BollingerBands({self.cfg.boll_sma},{self.cfg.boll_dev})"
        return s

    def run_all(self) -> Dict[str, dict]:
        return {
            "SMA":        self.run_sma(),
            "Contrarian": self.run_contrarian(),
            "MeanRev":    self.run_mean_rev(),
        }

    # ── Optimization ─────────────────────────────────────────────── #

    def run_optimization(self):
        self._assert_data()
        print("\n[SMA Optimization]")
        try:
            opt, perf = self.sma.optimize(self.cfg.sma_s_range, self.cfg.sma_l_range)
            print(f"  Best params : SMA_S={opt[0]}, SMA_L={opt[1]}")
            print(f"  Best perf   : {perf:.4f}")
        except Exception as e:
            print(f"  SMA failed  : {e}")

        print("\n[Contrarian Optimization]")
        try:
            opt, perf = self.con.optimize(self.cfg.con_range)
            print(f"  Best window : {opt}")
            print(f"  Best perf   : {perf:.4f}")
        except Exception as e:
            print(f"  CON failed  : {e}")

        print("\n[Bollinger Bands Optimization]")
        try:
            opt, perf = self.mrv.optimize(self.cfg.boll_sma_range, self.cfg.boll_dev_range)
            print(f"  Best params : SMA={opt[0]}, dev={opt[1]}")
            print(f"  Best perf   : {perf:.4f}")
        except Exception as e:
            print(f"  MRV failed  : {e}")

    # ── Regime-switching backtest ─────────────────────────────────── #

    def _run_regime_backtest(self) -> pd.DataFrame:
        """Bar-by-bar simulation mirroring live StrategySwitcher logic."""
        self._assert_data()
        s_cfg = StrategyConfig(
            sma_s         = self.cfg.sma_s,
            sma_l         = self.cfg.sma_l,
            con_window    = self.cfg.con_window,
            boll_sma      = self.cfg.boll_sma,
            boll_dev      = self.cfg.boll_dev,
            adx_period    = self.cfg.adx_period,
            adx_threshold = self.cfg.adx_threshold,
            hurst_lags    = self.cfg.hurst_lags,
        )
        switcher   = StrategySwitcher(s_cfg)
        df         = self._df.copy()
        df["returns"] = np.log(df["price"] / df["price"].shift(1))

        min_warmup  = max(self.cfg.sma_l, self.cfg.boll_sma * 3, self.cfg.hurst_lags * 3)
        records     = []
        prev_signal = 0

        for i in range(min_warmup, len(df)):
            window   = df.iloc[:i + 1]
            price_s  = window["price"]
            high_s   = window["high"] if "high" in window.columns else None
            low_s    = window["low"]  if "low"  in window.columns else None
            decision = switcher.decide(price_s, self._symbol, high_s, low_s)
            signal   = decision.signal
            trade    = 1 if signal != prev_signal else 0
            ret      = float(df["returns"].iloc[i])
            strat_ret = signal * ret - trade * self.cfg.tc

            records.append({
                "date":             df.index[i],
                "price":            float(df["price"].iloc[i]),
                "returns":          ret,
                "regime":           decision.regime.value,
                "strategy":         decision.strategy,
                "position":         signal,
                "strategy_return":  strat_ret,
                "trade":            trade,
                "hurst":            decision.hurst,
                "adx":              decision.adx,
            })
            prev_signal = signal

        result = pd.DataFrame(records).set_index("date")
        result["creturns"]  = result["returns"].cumsum().apply(np.exp)
        result["cstrategy"] = result["strategy_return"].cumsum().apply(np.exp)
        return result

    def compare_with_regime_switcher(self):
        all_results = self.run_all()
        print(f"\n[Regime-Switching Backtest] — {self._symbol}")
        try:
            regime_df = self._run_regime_backtest()
            all_results["RegimeSwitcher"] = self._summarise_regime(regime_df)
        except Exception as e:
            logger.error(f"Regime backtest failed: {e}", exc_info=True)
            print(f"  Failed: {e}")
        self.print_summary(all_results)
        return all_results

    @staticmethod
    def _summarise_regime(df: pd.DataFrame) -> dict:
        ann  = 252
        s    = df["strategy_return"]
        risk = float(s.std() * np.sqrt(ann))
        ret  = float(s.mean() * ann)
        return {
            "strategy_name":    "RegimeSwitcher",
            "total_return":     round(float(df["cstrategy"].iloc[-1]), 4),
            "buy_hold":         round(float(df["creturns"].iloc[-1]), 4),
            "outperformance":   round(float(df["cstrategy"].iloc[-1] - df["creturns"].iloc[-1]), 4),
            "ann_return":       round(ret, 6),
            "ann_risk":         round(risk, 6),
            "sharpe":           round(ret / (risk + 1e-9), 4),
            "max_drawdown":     round(float((df["cstrategy"].cummax() - df["cstrategy"]).max()), 4),
            "num_trades":       int(df["trade"].sum()),
            "regime_breakdown": df["regime"].value_counts().to_dict(),
        }

    # ── Session analysis ─────────────────────────────────────────── #

    def run_session_analysis(
        self,
        instruments: list = None,
        hourly_bars: int = 5000,
    ) -> SessionAnalyser:
        """
        Download hourly data for each instrument and train the ML
        session analyser. Returns a fitted SessionAnalyser.
        """
        instruments = instruments or INSTRUMENTS
        hourly_data = {}
        for inst in instruments:
            try:
                logger.info(f"Fetching H1 data for session analysis: {inst}")
                df = self.feed.get_history(inst, "H1", hourly_bars)
                hourly_data[inst] = df
            except Exception as e:
                logger.warning(f"Could not fetch H1 data for {inst}: {e}")

        sa = SessionAnalyser()
        sa.fit(hourly_data, min_hit=self.cfg.ml_min_hit_ratio)
        logger.info(f"Session analysis complete: {sa.session_summary()}")

        # Print best hours
        print("\n[ML Session Analysis — Best UTC Hours to Trade]")
        for inst, hours in sa.session_summary().items():
            print(f"  {inst:10s}: {hours}")

        return sa

    # ── Run all pairs ─────────────────────────────────────────────── #

    def run_all_pairs(
        self,
        instruments: list = None,
        granularity: str  = "D",
        bars:        int  = None,
        optimize:    bool = False,
    ) -> Dict[str, Dict]:
        """Run full backtest comparison for every supported instrument."""
        instruments = instruments or INSTRUMENTS
        all_pair_results = {}

        for inst in instruments:
            print(f"\n{'='*70}")
            print(f"  {inst}")
            print(f"{'='*70}")
            try:
                self.load_data(inst, granularity, bars)
                results = self.compare_with_regime_switcher()
                all_pair_results[inst] = results
                if optimize:
                    self.run_optimization()
            except Exception as e:
                logger.error(f"Backtest failed for {inst}: {e}", exc_info=True)
                print(f"  ERROR: {e}")

        return all_pair_results

    # ── Pretty-print ─────────────────────────────────────────────── #

    @staticmethod
    def print_summary(results: Dict[str, dict]):
        cols = [
            "strategy_name", "total_return", "buy_hold", "outperformance",
            "ann_return", "ann_risk", "sharpe", "max_drawdown", "num_trades",
        ]
        rows = [{c: r.get(c, "N/A") for c in cols} for r in results.values()]
        df   = pd.DataFrame(rows).set_index("strategy_name")

        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 130)

        print("\n" + "=" * 90)
        print(df.to_string())
        print("=" * 90)

        try:
            winner = df["sharpe"].astype(float).idxmax()
            print(f"\n  ★  Best Sharpe: {winner}  (Sharpe = {df.loc[winner, 'sharpe']})")
        except Exception:
            pass

        for key, r in results.items():
            if "regime_breakdown" in r:
                bd = "  |  ".join(f"{k}: {v}" for k, v in r["regime_breakdown"].items())
                print(f"\n  RegimeSwitcher breakdown → {bd}")

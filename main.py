"""
main.py — Unified Forex + Deriv Synthetics Bot  (v6 — SMC + ML + Deriv)
═════════════════════════════════════════════════════════════════════════

BROKER FLAGS
────────────
  --broker oanda   (default) OANDA REST for data + execution
  --broker deriv   Deriv WebSocket for data; Deriv MT5 for execution
  --broker mt5     OANDA data + MT5 execution (original v5 behaviour)

DATA SOURCES
────────────
  OANDA broker  → OandaClient  (OANDA REST API, forex only)
  Deriv broker  → DerivDataFeed (WebSocket or MT5) for synthetics + forex

The analysis (SMC/ICT/CRT) and ML pipeline are IDENTICAL for both.
Only the data client and execution broker change.

TIMEFRAME STACK (unchanged)
────────────────────────────
  Weekly → Daily → H4 → H1 → M5 (confluence) → M2 (entry)

ML TRAINING ON DERIV DATA
──────────────────────────
  When --broker deriv is set:
  • MLSignalFilter trains on Deriv's own tick/candle history
  • Synthetics (R_75, CRASH1000 etc.) train their own separate models
  • Forex pairs use Deriv frx data (same price, slightly different spread)
  • Models auto-retrain every 500 new M5 bars (continuous learning)
  • Saved to logs/ml_{pair}.pkl per pair

Usage
──────
  # Simulate on Deriv synthetics (no real orders)
  python main.py --broker deriv --mode live

  # Live execution on Deriv via MT5
  python main.py --broker deriv --mode live --execute

  # Paper sim on single synthetic
  python main.py --broker deriv --mode paper --pair R_75

  # Paper sim — all Deriv instruments
  python main.py --broker deriv --mode paper --all-pairs

  # Backtest on Deriv data
  python main.py --broker deriv --mode backtest --pair R_75

  # OANDA forex (original behaviour)
  python main.py --broker oanda --mode live --execute

  # MT5 execution with OANDA data
  python main.py --broker mt5 --mode live --execute
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────── #
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════ #
#  Print helpers (unchanged)                                         #
# ══════════════════════════════════════════════════════════════════ #

def _banner(title: str, width: int = 70):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def _section(title: str, width: int = 70):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")

def _row(label: str, value, width: int = 30):
    print(f"  {label:<{width}}: {value}")

def _print_config(cfg, pairs, sim_mode, broker_name):
    _banner("BOT CONFIGURATION")
    _row("Broker",          broker_name)
    _row("Mode",            "SIMULATE (no orders)" if sim_mode else "⚠  LIVE EXECUTE")
    _row("Pairs",           " | ".join(pairs[:5]) + (f" +{len(pairs)-5} more" if len(pairs) > 5 else ""))
    _row("Risk per trade",  f"{cfg.risk_pct}% of NAV")
    _row("R:R",             f"1:{cfg.rr_ratio}")
    _row("SL",              f"{cfg.sl_atr_mult}× ATR(14)")
    _row("Max open trades", cfg.max_positions)
    _row("Daily loss limit",f"{cfg.max_daily_loss}%")
    _row("Loop interval",   f"{cfg.loop_interval}s")
    _row("ML gate",         "ON — RandomForest P>0.55")
    _row("Initial balance", f"${cfg.initial_balance:,.2f}")
    print()


# ══════════════════════════════════════════════════════════════════ #
#  Broker + config factory                                           #
# ══════════════════════════════════════════════════════════════════ #

def _build_oanda_client():
    from data_feed import OandaClient
    api_key    = os.getenv("OANDA_API_KEY", "")
    account_id = os.getenv("OANDA_ACCOUNT_ID", "")
    practice   = os.getenv("OANDA_PRACTICE", "true").lower() == "true"
    if not api_key or not account_id:
        sys.exit(
            "\n[ERROR] OANDA credentials missing.\n"
            "Set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env\n"
        )
    client = OandaClient(api_key, account_id, practice=practice)
    logger.info(f"OANDA client ready ({'practice' if practice else 'LIVE ⚠'})")
    return client


def _build_deriv_feed(backend: str = None):
    """
    Returns a DerivDataFeed wrapping either the WebSocket or MT5 client.
    backend: "websocket" | "mt5" (default from DERIV_BACKEND env var)
    """
    from deriv_feed import DerivWebSocketClient, DerivMT5Client, DerivDataFeed
    backend = backend or os.getenv("DERIV_BACKEND", "websocket")

    if backend == "mt5":
        client = DerivMT5Client(
            login    = int(os.getenv("MT5_LOGIN",    0)),
            password = os.getenv("MT5_PASSWORD",    ""),
            server   = os.getenv("MT5_SERVER",      "Deriv-Demo"),
            path     = os.getenv("MT5_PATH",        ""),
        )
        if not client.connect():
            logger.warning("DerivMT5Client connect failed — falling back to WebSocket.")
            client = DerivWebSocketClient(api_token=os.getenv("DERIV_API_TOKEN", ""))
    else:
        client = DerivWebSocketClient(api_token=os.getenv("DERIV_API_TOKEN", ""))

    feed = DerivDataFeed(client, cache_dir="data")
    logger.info(f"Deriv feed ready (backend={backend})")
    return feed


def _build_execution_broker(args, oanda_client=None):
    """
    Build the correct ExecutionBroker from CLI args.
    """
    from execution_broker import SimBroker, OandaBroker, MT5Broker

    if not args.execute:
        return SimBroker()

    broker_arg = getattr(args, "broker", "oanda")

    if broker_arg == "deriv":
        from deriv_broker import DerivMT5Broker, DerivWebSocketBroker
        backend = os.getenv("DERIV_BACKEND", "mt5")
        if backend == "websocket":
            broker = DerivWebSocketBroker(api_token=os.getenv("DERIV_API_TOKEN", ""))
            logger.info("Execution broker: Deriv WebSocket (multiplier contracts)")
        else:
            broker = DerivMT5Broker(
                login    = int(os.getenv("MT5_LOGIN",    0)),
                password = os.getenv("MT5_PASSWORD",    ""),
                server   = os.getenv("MT5_SERVER",      "Deriv-Demo"),
                path     = os.getenv("MT5_PATH",        ""),
            )
            if not broker.connect():
                sys.exit("[ERROR] Deriv MT5 connection failed. Check MT5_LOGIN/PASSWORD/SERVER in .env")
            logger.info(f"Execution broker: Deriv MT5 | {broker.account_info()}")
        return broker

    elif broker_arg == "mt5":
        broker = MT5Broker(
            login    = int(os.getenv("MT5_LOGIN",    0)),
            password = os.getenv("MT5_PASSWORD",    ""),
            server   = os.getenv("MT5_SERVER",      ""),
            path     = os.getenv("MT5_PATH",        ""),
        )
        if not broker.connect():
            sys.exit("[ERROR] MT5 connection failed. Check MT5_LOGIN/PASSWORD/SERVER in .env")
        logger.info(f"Execution broker: MT5 | {broker.account_info()}")
        return broker

    else:  # oanda
        if oanda_client is None:
            oanda_client = _build_oanda_client()
        return OandaBroker(oanda_client)


def _load_config(args):
    """
    Return (cfg, instruments, pip_size, spread, slippage, leverage, session_windows).
    For Deriv mode merges DerivBotConfig + Deriv market data tables.
    """
    broker_arg = getattr(args, "broker", "oanda")

    if broker_arg == "deriv":
        from deriv_config import (
            DerivBotConfig,
            DERIV_INSTRUMENTS,
            DERIV_PIP_SIZE,
            DERIV_SPREAD,
            DERIV_SLIPPAGE,
            DERIV_LEVERAGE,
            DERIV_SESSION_WINDOWS,
        )
        cfg = DerivBotConfig()

        # CLI overrides
        if getattr(args, "risk",  None): cfg.risk_pct   = args.risk
        if getattr(args, "rr",    None): cfg.rr_ratio   = args.rr
        if getattr(args, "units", None): cfg.fixed_units = args.units

        instruments = DERIV_INSTRUMENTS
        return cfg, instruments, DERIV_PIP_SIZE, DERIV_SPREAD, DERIV_SLIPPAGE, \
               DERIV_LEVERAGE, DERIV_SESSION_WINDOWS
    else:
        from config import BotConfig, INSTRUMENTS, PIP_SIZE, SPREAD, SLIPPAGE, LEVERAGE
        cfg = BotConfig()
        if getattr(args, "risk",  None): cfg.risk_pct   = args.risk
        if getattr(args, "rr",    None): cfg.rr_ratio   = args.rr
        if getattr(args, "tc",    None): cfg.tc         = args.tc
        if getattr(args, "units", None): cfg.fixed_units = args.units

        # SESSION_WINDOWS not in original config — fall back to None
        session_windows = getattr(__import__("config"), "SESSION_WINDOWS", None)
        return cfg, INSTRUMENTS, PIP_SIZE, SPREAD, SLIPPAGE, LEVERAGE, session_windows


# ══════════════════════════════════════════════════════════════════ #
#  DataFeed adapter                                                  #
#  Wraps both OandaClient-based DataFeed and DerivDataFeed behind   #
#  a common get_history(pair, gran, bars, force_refresh) API.        #
# ══════════════════════════════════════════════════════════════════ #

def _make_feed(args, oanda_client=None):
    broker_arg = getattr(args, "broker", "oanda")
    if broker_arg == "deriv":
        return _build_deriv_feed()
    else:
        from data_feed import DataFeed
        if oanda_client is None:
            oanda_client = _build_oanda_client()
        return DataFeed(oanda_client, cache_dir="data")


# ══════════════════════════════════════════════════════════════════ #
#  Strategy config builder                                           #
# ══════════════════════════════════════════════════════════════════ #

def _build_strategy_config(cfg):
    from strategy_switcher import StrategyConfig
    return StrategyConfig(
        sma_s         = cfg.sma_s,
        sma_l         = cfg.sma_l,
        con_window    = cfg.con_window,
        boll_sma      = cfg.boll_sma,
        boll_dev      = cfg.boll_dev,
        adx_period    = cfg.adx_period,
        adx_threshold = cfg.adx_threshold,
        hurst_lags    = cfg.hurst_lags,
    )


# ══════════════════════════════════════════════════════════════════ #
#  PaperTrader builder                                               #
# ══════════════════════════════════════════════════════════════════ #

def _build_trader(pairs, s_cfg, feed, cfg, sim_mode, broker=None):
    from paper_trader import PaperTrader
    from execution_broker import SimBroker

    # PaperTrader still accepts oanda_client for legacy compatibility.
    # For Deriv mode we pass feed directly via a thin shim.
    oanda_client = _feed_as_oanda_shim(feed)

    return PaperTrader(
        instruments      = pairs,
        config           = s_cfg,
        oanda_client     = oanda_client,
        initial_balance  = cfg.initial_balance,
        risk_pct         = cfg.risk_pct,
        rr_ratio         = cfg.rr_ratio,
        sl_atr_mult      = cfg.sl_atr_mult,
        max_positions    = cfg.max_positions,
        max_daily_loss   = cfg.max_daily_loss,
        fixed_units      = cfg.fixed_units,
        loop_interval    = cfg.loop_interval,
        log_dir          = "logs",
        sim_mode         = sim_mode,
        ml_enabled       = True,
        ml_threshold     = 0.55,
        ml_min_bars      = 200,
        ml_retrain_every = 500,
        broker           = broker or (SimBroker() if sim_mode else None),
    )


def _feed_as_oanda_shim(feed):
    """
    Wrap DerivDataFeed (or DataFeed) so PaperTrader's internal
    DataFeed constructor still works.  PaperTrader calls:
      self.feed = DataFeed(oanda_client, cache_dir="data")
      self.feed.get_history(pair, gran, bars, force_refresh=True)
    We return a shim object whose DataFeed wraps the real feed.
    """
    class _FeedShim:
        """Minimal OandaClient-like object that delegates to the real feed."""
        def __init__(self, real_feed):
            self._feed = real_feed
        def get_candles(self, instrument, granularity="H1", count=500, **kw):
            return self._feed.get_history(instrument, granularity, count)
        def get_current_price(self, instrument):
            try:
                return self._feed.client.get_current_price(instrument)
            except Exception:
                df = self._feed.get_history(instrument, "M1", 2)
                if not df.empty:
                    p = float(df["price"].iloc[-1])
                    return {"bid": p, "ask": p, "mid": p, "spread": 0.0}
                return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "spread": 0.0}
        def get_account_summary(self):
            try:
                return self._feed.client.get_account_summary()
            except Exception:
                return {}

    # If it's already an OandaClient-based feed just return a wrapper
    from data_feed import DataFeed
    if isinstance(feed, DataFeed):
        return feed.client   # already OandaClient

    return _FeedShim(feed)


# ══════════════════════════════════════════════════════════════════ #
#  Session analyser helper                                           #
# ══════════════════════════════════════════════════════════════════ #

def _build_session_analyser(feed, pairs, cfg, broker_arg="oanda"):
    from strategy_switcher import SessionAnalyser
    sa = SessionAnalyser()

    print("\n[ML Session Analyser] Training from H1 history...")
    hourly_data = {}
    for pair in pairs:
        try:
            df = feed.get_history(pair, "H1", getattr(cfg, "ml_h1_bars", 3000))
            hourly_data[pair] = df
            logger.info(f"  Fetched H1 history for {pair}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  H1 fetch failed {pair}: {e}")

    if hourly_data:
        sa.fit(hourly_data, min_hit=getattr(cfg, "ml_min_hit_ratio", 0.52))
        logger.info(f"  Session analysis complete: {sa.session_summary()}")
    else:
        logger.warning("  No H1 data — using research-based session windows.")

    # For synthetics (24/7), override session filter: all hours allowed
    if broker_arg == "deriv":
        from deriv_config import DERIV_SYNTHETIC_INSTRUMENTS
        for sym in DERIV_SYNTHETIC_INSTRUMENTS:
            sa._hours[sym] = list(range(24))

    _print_ml_session(sa, pairs)
    return sa


def _print_ml_session(sa, pairs):
    _section("ML SESSION ANALYSIS — Best UTC Hours")
    for pair in pairs[:10]:   # cap display at 10
        hours = sa.best_hours(pair)
        bar   = "".join("█" if h in hours else "·" for h in range(24))
        print(f"  {pair:14s} [{bar}]")
    if len(pairs) > 10:
        print(f"  ... and {len(pairs)-10} more pairs")
    print()


# ══════════════════════════════════════════════════════════════════ #
#  Mode: live                                                        #
# ══════════════════════════════════════════════════════════════════ #

def run_live(args, cfg, instruments, *_):
    broker_arg = getattr(args, "broker", "oanda")
    sim_mode   = not args.execute

    # Data feed
    oanda_client = None
    if broker_arg != "deriv":
        oanda_client = _build_oanda_client()
    feed = _make_feed(args, oanda_client)

    # Execution broker
    broker = _build_execution_broker(args, oanda_client)

    pairs  = instruments
    s_cfg  = _build_strategy_config(cfg)

    _print_config(cfg, pairs, sim_mode, broker.name)

    if not sim_mode:
        print(f"⚠️  LIVE EXECUTION via {broker.name}")
        print(f"   Pairs     : {pairs[:5]}{'...' if len(pairs)>5 else ''}")
        print(f"   Risk/trade: {cfg.risk_pct}% of NAV")
        confirm = input("\n  Type YES to confirm and start: ").strip()
        if confirm != "YES":
            print("  Aborted.")
            return
    else:
        print("  Running in SIMULATE mode — no real orders.")
        input("  Press Enter to start...")

    # Session analyser
    sa     = _build_session_analyser(feed, pairs, cfg, broker_arg)
    trader = _build_trader(pairs, s_cfg, feed, cfg, sim_mode, broker)
    trader.switcher.sessions = sa

    # Inject Deriv-aware pip/spread tables into risk_manager
    if broker_arg == "deriv":
        _patch_risk_manager_tables(cfg)

    _banner(f"LIVE BOT STARTING  [{broker.name}]")
    _row("UTC start time",  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    _row("Pairs",           f"{len(pairs)} instruments")
    _row("ML gate",         "ON — RandomForest P>0.55")
    _row("Loop interval",   f"{cfg.loop_interval}s")
    print()

    trader.run_live()


# ══════════════════════════════════════════════════════════════════ #
#  Mode: paper simulation                                            #
# ══════════════════════════════════════════════════════════════════ #

def run_paper(args, cfg, instruments, *_):
    broker_arg   = getattr(args, "broker", "oanda")
    oanda_client = None
    if broker_arg != "deriv":
        oanda_client = _build_oanda_client()
    feed  = _make_feed(args, oanda_client)
    pairs = instruments if args.all_pairs else [args.pair]
    s_cfg = _build_strategy_config(cfg)

    from execution_broker import SimBroker
    trader = _build_trader(pairs, s_cfg, feed, cfg, sim_mode=True, broker=SimBroker())

    if broker_arg == "deriv":
        _patch_risk_manager_tables(cfg)

    _banner(f"PAPER SIMULATION — {pairs}")
    print("  Loading history...")
    for pair in pairs:
        for gran, bars in [("D", 500), ("H4", 500), ("H1", 300), ("M5", 576)]:
            try:
                df = feed.get_history(pair, gran, bars)
                trader.load_history(pair, df, gran=gran)
                logger.info(f"  {pair} {gran:3s}: {len(df)} bars")
            except Exception as e:
                logger.warning(f"  {pair} {gran}: {e}")

    print("  Running bar-by-bar simulation...")
    sim_results = trader.run_backtest_mode()
    port        = trader.portfolio_summary()

    _banner("SIMULATION RESULTS")
    _row("Initial Balance", f"${cfg.initial_balance:,.2f}")
    _row("Final NAV",       f"${port['nav']:,.2f}")
    _row("Total P&L",       f"${port['total_pnl']:+,.2f}")
    _row("Win Rate",        f"{port.get('win_rate',0):.1f}%")
    _row("Total Trades",    port['total_trades'])
    print()

    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", "paper_sim_results.csv")
    sim_results.to_csv(out, index=False)
    logger.info(f"Results saved → {out}")


# ══════════════════════════════════════════════════════════════════ #
#  Mode: backtest                                                    #
# ══════════════════════════════════════════════════════════════════ #

def run_backtest(args, cfg, instruments, *_):
    from backtest_runner import BacktestRunner
    broker_arg   = getattr(args, "broker", "oanda")
    oanda_client = _build_oanda_client() if broker_arg != "deriv" else None
    feed         = _make_feed(args, oanda_client)
    pairs        = instruments if args.all_pairs else [args.pair]
    gran         = args.granularity or "H1"

    _banner(f"BACKTEST — {pairs} [{broker_arg.upper()}]")

    # BacktestRunner still expects an oanda_client-like object
    # We pass a shim for the feed
    client_shim = _feed_as_oanda_shim(feed)

    runner = BacktestRunner(client_shim, cfg)
    for pair in pairs:
        _section(f"{pair} | {gran}")
        runner.load_data(pair, gran, cfg.history_bars)
        results = runner.run_all()
        runner.print_summary(results)
        if getattr(args, "optimize", False):
            runner.run_optimization()


# ══════════════════════════════════════════════════════════════════ #
#  Mode: compare                                                     #
# ══════════════════════════════════════════════════════════════════ #

def run_compare(args, cfg, instruments, *_):
    from backtest_runner import BacktestRunner
    broker_arg   = getattr(args, "broker", "oanda")
    oanda_client = _build_oanda_client() if broker_arg != "deriv" else None
    feed         = _make_feed(args, oanda_client)
    pairs        = instruments if args.all_pairs else [args.pair]
    gran         = args.granularity or "H1"
    client_shim  = _feed_as_oanda_shim(feed)
    runner       = BacktestRunner(client_shim, cfg)

    _banner(f"STRATEGY COMPARISON — {pairs}")
    for pair in pairs:
        _section(f"{pair} | {gran}")
        runner.load_data(pair, gran, cfg.history_bars)
        runner.compare_with_regime_switcher()


# ══════════════════════════════════════════════════════════════════ #
#  Mode: session analysis only                                       #
# ══════════════════════════════════════════════════════════════════ #

def run_session(args, cfg, instruments, *_):
    broker_arg   = getattr(args, "broker", "oanda")
    oanda_client = _build_oanda_client() if broker_arg != "deriv" else None
    feed         = _make_feed(args, oanda_client)
    pairs        = instruments if args.all_pairs else [args.pair]
    _banner("ML SESSION ANALYSIS")
    _build_session_analyser(feed, pairs, cfg, broker_arg)


# ══════════════════════════════════════════════════════════════════ #
#  Deriv risk-manager patch                                          #
#  Injects Deriv pip/spread/leverage tables into risk_manager module #
# ══════════════════════════════════════════════════════════════════ #

def _patch_risk_manager_tables(cfg):
    """
    risk_manager.py imports PIP_SIZE, SPREAD, SLIPPAGE, LEVERAGE from
    config.py at module load time.  For Deriv we monkey-patch the
    module-level dicts after import so all subsequent calls use the
    Deriv tables — no changes to risk_manager.py needed.
    """
    try:
        import risk_manager as rm
        from deriv_config import (
            DERIV_PIP_SIZE  as pip,
            DERIV_SPREAD    as spread,
            DERIV_SLIPPAGE  as slip,
            DERIV_LEVERAGE  as lev,
        )
        # Merge Deriv tables on top of existing (keeps forex defaults)
        rm.PIP_SIZE.update(pip)
        # SPREAD and SLIPPAGE are used via config imports at call time —
        # patch the config module directly
        import config
        config.PIP_SIZE.update(pip)
        config.SPREAD.update(spread)
        config.SLIPPAGE.update(slip)
        config.LEVERAGE.update(lev)
        logger.info("Risk manager tables patched with Deriv data.")
    except Exception as e:
        logger.warning(f"Could not patch risk manager tables: {e}")


# ══════════════════════════════════════════════════════════════════ #
#  CLI                                                               #
# ══════════════════════════════════════════════════════════════════ #

def parse_args():
    p = argparse.ArgumentParser(
        description = "Unified SMC+ML Bot — OANDA forex + Deriv synthetics",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )
    p.add_argument("--mode",
        choices=["backtest", "compare", "paper", "live", "session"],
        default="live")
    p.add_argument("--pair",    default="GBP_USD")
    p.add_argument("--all-pairs", action="store_true")
    p.add_argument("--granularity", default=None)
    p.add_argument("--optimize",    action="store_true")
    p.add_argument("--execute",     action="store_true",
        help="Place real orders (requires broker credentials in .env)")
    p.add_argument("--risk",  type=float, default=None)
    p.add_argument("--rr",    type=float, default=None)
    p.add_argument("--tc",    type=float, default=None)
    p.add_argument("--units", type=int,   default=None)
    p.add_argument("--no-ml", action="store_true")
    p.add_argument("--broker",
        choices=["oanda", "mt5", "deriv"],
        default="oanda",
        help=(
            "oanda = OANDA REST (default) | "
            "mt5   = OANDA data + MT5 execution | "
            "deriv = Deriv data + Deriv execution (MT5 or WebSocket)"
        ))
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════ #
#  Entry point                                                       #
# ══════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    args = parse_args()

    try:
        cfg, instruments, pip_size, spread, slippage, leverage, session_windows = \
            _load_config(args)
    except ImportError as e:
        sys.exit(f"Import error: {e}\nRun: pip install -r requirements.txt")

    # Single-pair override
    if not args.all_pairs and args.pair not in instruments:
        logger.warning(
            f"--pair {args.pair} not in configured instruments. "
            f"Using it anyway."
        )
        instruments = [args.pair]
    elif not args.all_pairs:
        instruments = [args.pair]

    dispatch = {
        "backtest": run_backtest,
        "compare":  run_compare,
        "paper":    run_paper,
        "live":     run_live,
        "session":  run_session,
    }
    dispatch[args.mode](args, cfg, instruments)
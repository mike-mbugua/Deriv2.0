# Bot Fix Changelog — v2
> Every fix mapped to a root cause. Most beginner bots fail from risk engine
> mistakes, not strategy mistakes. Here's what was wrong and why.

---

## 1. NAV Math — CRITICAL FIX (`paper_trader.py`)

**Bug:** `Balance=10,000 | NAV=99,989 | Return=899%` with 0 trades.

**Root cause:** `_nav()` was computing `balance + Σ(units × price)` which
multiplied full position notional (e.g. 50,000 units × 1.36 = $68,000) into NAV.
Cash was also being debited on open (`self.balance -= cost`), mixing cost-basis
and mark-to-market accounting.

**Fix:**
```
NAV = cash_balance + Σ(unrealised_pnl per open position)
unrealised_pnl = signed_units × (current_price - entry_price)
```
Cash only changes on CLOSE (realised PnL added to cash). No deduction on open.

---

## 2. Position Sizing — FIXED (`risk_manager.py`)

**Bug:** `units = risk_amount / sl_dist` was wrong for two reasons:
- Ignored pip value (a 0.0001 SL on JPY pair ≠ same dollar value as on EUR/USD)
- Ignored leverage (could request more margin than available)

**Fix:**
```
sl_pips       = sl_dist / pip_size
pip_value     = _pip_value_per_unit(instrument, price)
units         = risk_amount / (sl_pips × pip_value)
margin_check  = (units × price) / leverage  ← must be < 90% of NAV
```
Leverage per instrument defined in `config.py`. Margin guard added.

---

## 3. Spread + Slippage Modelled — NEW (`config.py`, `risk_manager.py`)

**Bug:** Not modelled at all. Bot would win on paper, lose live.

**Fix:** `SPREAD` and `SLIPPAGE` dicts added to `config.py` per instrument.
Applied in `build_trade_setup()`:
```
tx_cost  = spread/2 + slippage
Long :  SL = entry - sl_dist - tx_cost   (widens risk)
         TP = entry + tp_dist - tx_cost   (shrinks reward)
Short:  SL = entry + sl_dist + tx_cost
         TP = entry - tp_dist + tx_cost
```

---

## 4. Stop Loss Verified — NEW (`risk_manager.py`)

**Bug:** No verification that SL/TP were on the correct side of entry.

**Fix:** `TradeSetup.verify()` method added — asserts:
- Long: `sl < entry < tp`
- Short: `tp < entry < sl`
- units > 0
- RR ≥ 1.9 (warns if below 2.0 target)
Returns `False` and rejects trade if any check fails.

---

## 5. Leverage Understood — NEW (`config.py`, `risk_manager.py`)

**Bug:** Leverage not considered in position sizing or margin check.

**Fix:** `LEVERAGE` dict in `config.py`:
```python
LEVERAGE = {
    "GBP_USD": 30, "XAU_USD": 20, "GBP_JPY": 20,
    "USD_JPY": 30, "AUD_CAD": 30, "NZD_JPY": 20,
}
```
Margin check in `build_trade_setup()` prevents opening positions requiring
more than 90% of NAV as margin.

---

## 6. Granularity Fixed — CRITICAL (`config.py`, `paper_trader.py`, `main.py`)

**Bug:** `Granularity=D | Interval=300s` — fetching Daily candles every 5 min.
Daily candles don't change for 24 hours, so the strategy signal was always
stale. The regime/signal never updated between midnight and midnight.

**Fix:** Multi-timeframe cascade:
```
H4  → trend context (bias filter)   — checked on each loop
H1  → signal confirmation            — strategy fires here
M5  → entry candle precision         — price used for order
Loop interval = 300s (every 5 min to catch M5 candles)
```
H4 and H1 signals must agree. If H4 is bullish and H1 gives short signal
→ signal is suppressed (conflicting timeframes = no trade).

---

## 7. Pair Waterfall — NEW (`config.py`, `paper_trader.py`, `main.py`)

**Bug:** Only GBP_USD monitored. If no signal, bot sat idle.

**Fix:** Priority-ordered waterfall in `INSTRUMENTS`:
```
GBP_USD → XAU_USD → GBP_JPY → USD_JPY → AUD_CAD → NZD_JPY
```
Bot scans all pairs every loop. If GBP_USD has no signal, it checks the
next pair. All 6 pairs get pip sizes, spread, slippage, leverage defined.
New pairs added to `PAIR_MAP` in `data_feed.py`.

---

## 8. Max 2 Trades, Highest Probability — NEW (`paper_trader.py`)

**Bug:** `max_positions=3`, no ranking — any signal was taken.

**Fix:**
- `max_positions=2` in `config.py`
- All signalling pairs are scored by `_compute_confidence()`:
  ```
  ADX component   (0–0.4): stronger trend = higher score
  Hurst component (0–0.3): further from 0.5 = clearer regime
  H4/H1 agreement (0–0.2): both timeframes agree = bonus
  Session quality  (0–0.1): ML-identified good hours = bonus
  ```
- Candidates ranked by confidence score, top 2 taken.
- Already-open positions never evicted by a higher-confidence new signal.

---

## 9. Dynamic Session Hours — PRESERVED & CLARIFIED (`backtest_runner.py`)

**Original:** Hours hardcoded as `[0, 1, 2, 4, 5, 10, 11, 12, 14, 16, 21, 22]`
(ML analysis was running but on wrong data — Daily bars, not H1).

**Fix:** ML SessionAnalyser now fits on H1 bars (`ml_h1_bars=3000`, ~4 months).
H1 volume patterns are much better than Daily for session identification.
Fallback windows remain in `config.py` if ML fails.
Output now labelled "data-driven" vs "fallback" in the log.

---

## 10. Risk Parameters — FIXED (`config.py`)

| Parameter     | Old   | New   | Reason |
|---------------|-------|-------|--------|
| `risk_pct`    | 1.0%  | 10.0% | Your requirement |
| `sl_atr_mult` | 1.5×  | 1.0×  | Paired with RR below |
| `tp_atr_mult` | 2.5×  | 2.0×  | 1:2 RR: TP = 2 × SL always |
| `max_positions`| 3    | 2     | Your requirement |
| `rr_ratio`    | ~1.67 | 2.0   | Exact 1:2 enforced |

The old `tp_atr_mult/sl_atr_mult = 2.5/1.5 = 1.67` gave a ~1:1.67 RR,
not 1:2 as intended. Now TP distance is always exactly `2 × SL distance`.

---

## How to Run After Update

```bash
# Install any new deps (dotenv should already be there)
pip install -r requirements.txt

# Live mode — all 6 pairs, waterfall, max 2 trades, 10% risk
python main.py --mode live --execute

# Paper simulation (historical H1 data)
python main.py --mode paper --all-pairs

# Session analysis (studies best hours)
python main.py --mode session

# Expected live log (healthy):
# PORTFOLIO | Cash=$10000.00 | NAV=$10000.00 | PnL=$0.00 | Return=0.00% | OpenTrades=0/2
# (NAV and Cash match when flat — no more 899% phantom return)
```

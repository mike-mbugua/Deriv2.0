# Deriv Bot — Complete Setup & Run Guide
> Your codebase is complete. All files present. Focus: Deriv WebSocket mode.

---

## What You Have (All Files Present ✅)

```
main.py              ✅  entry point
paper_trader.py      ✅  core SMC + ML engine
strategy_switcher.py ✅  signal generation
risk_manager.py      ✅  position sizing, SL/TP
deriv_feed.py        ✅  Deriv WebSocket data client
deriv_broker.py      ✅  Deriv MT5 + WebSocket execution
deriv_config.py      ✅  pip sizes, spreads, synthetics list
execution_broker.py  ✅  SimBroker / OandaBroker / MT5Broker
telegram_notifier.py ✅  Telegram alerts (pip bug fixed — use patched version)
config.py            ✅  OANDA config (still imported by backtest_runner)
data_feed.py         ✅  OANDA feed (used by backtest mode only)
backtest_runner.py   ✅  backtesting
backtester.py        ✅  strategy classes
requirements.txt     ✅  (websocket-client was missing — use patched version)
```

---

## Step 1 — Install dependencies

Use the patched `requirements.txt` (adds `websocket-client` which the original was missing):

```bash
pip install -r requirements.txt
```

If you're on a VPS or fresh Python install, do this first:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the key Deriv package installed:

```bash
python -c "import websocket; print('websocket-client OK')"
python -c "import sklearn; print('scikit-learn OK')"
```

---

## Step 2 — Get your Deriv API Token

1. Log in at **app.deriv.com**
2. Click your profile icon → **Account Settings** → **API Token**
3. Click **Create new token**
4. Name: `BotToken` (anything)
5. Tick **Read** and **Trade** scopes
6. Click **Create** → copy the token immediately (it's shown once)

> **Read** = fetch market data  
> **Trade** = place real orders (needed only for `--execute` mode)

---

## Step 3 — Set up .env

Copy the provided `.env.deriv` file to `.env` in your bot folder:

```bash
cp .env.deriv .env
```

Open `.env` and fill in exactly these fields:

```
DERIV_API_TOKEN=your_token_from_step_2
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Everything else has sensible defaults. **Do not change DERIV_BACKEND** — leave it as `websocket`.

### Get Telegram credentials

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Follow prompts → copy the token it gives you → paste as `TELEGRAM_BOT_TOKEN`
3. Send any message to your new bot, then open this URL in a browser:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
4. Find `result[0]["message"]["chat"]["id"]` → paste as `TELEGRAM_CHAT_ID`

---

## Step 4 — Create required directories

```bash
mkdir logs data
```

These auto-create on first run too, but it's cleaner to do it manually.

---

## Step 5 — Replace two files

Replace these in your bot folder with the patched versions provided:

| File | What changed |
|---|---|
| `requirements.txt` | Added `websocket-client>=1.7` (Deriv WebSocket won't work without it) |
| `telegram_notifier.py` | Pip calculation now uses instrument-aware pip size. R_75 was showing millions of pips because it hardcoded `0.0001` — now it reads from `deriv_config.DERIV_PIP_SIZE`. Also added `send_error()` method for crash alerts. |

---

## Step 6 — Run commands

### Start here: paper simulation on R_75

```bash
python main.py --broker deriv --mode paper --pair R_75
```

This loads ~500 bars of historical R_75 data from Deriv, runs bar-by-bar simulation, and prints results. No real money. No live connection needed beyond the API token.

### Watch live signals (no real orders)

```bash
python main.py --broker deriv --mode live --pair R_75
```

Connects to Deriv WebSocket, fetches live data every `LOOP_INTERVAL` seconds, runs the full SMC + ML pipeline, and sends results to Telegram. Still no real orders.

### Watch all Deriv instruments

```bash
python main.py --broker deriv --mode live --all-pairs
```

Runs all 15 instruments simultaneously: R_10, R_25, R_50, R_75, R_100, CRASH500, CRASH1000, BOOM500, BOOM1000, STPIDX + 5 forex pairs.

### Live execution — real orders via Deriv WebSocket

```bash
python main.py --broker deriv --mode live --execute
```

Places real orders when signals fire. Requires `DERIV_API_TOKEN` with **Trade** scope. The bot will ask you to type `YES` before starting.

### Backtest R_75

```bash
python main.py --broker deriv --mode backtest --pair R_75
```

### Override risk and R:R at runtime

```bash
python main.py --broker deriv --mode live --pair R_75 --risk 0.5 --rr 2.5
```

### Disable ML filter (pure SMC only)

```bash
python main.py --broker deriv --mode live --pair R_75 --no-ml
```

---

## What the bot does on startup (Deriv mode)

1. Loads `DerivBotConfig` from `deriv_config.py` (pip sizes, spreads, leverages for all synthetics)
2. Creates `DerivWebSocketClient` with your token
3. Patches `risk_manager.py` with Deriv pip/spread tables (so SL/TP math is correct for synthetics)
4. Fetches historical data: W → D → H4 → H1 → M5 → M2 for each pair
5. Trains `MLSignalFilter` once ≥200 M5 bars are available (~first 16 hours)
6. Runs the signal loop every `LOOP_INTERVAL` seconds
7. Sends Telegram alerts: WAIT messages every 30 loops, SIGNAL/CLOSE on events

---

## Recommended run order

```
Stage 1  python main.py --broker deriv --mode paper --pair R_75
         → Check the output makes sense. Look for SIGNAL or WAIT lines.

Stage 2  python main.py --broker deriv --mode live --pair R_75
         → Watch live signals in console + Telegram. No money involved.

Stage 3  python main.py --broker deriv --mode live --all-pairs
         → Expand to all instruments. Still no real orders.

Stage 4  python main.py --broker deriv --mode live --execute
         → Live orders on Deriv DEMO account first.
           Change MT5_SERVER=Deriv-Demo to Deriv-Server only when profitable.
```

---

## What each Deriv instrument looks like (price ranges)

Understanding price ranges matters for reading Telegram messages:

| Symbol | Price range | Pip size | Notes |
|---|---|---|---|
| R_10 | ~5,000 | 0.001 | Least volatile |
| R_25 | ~4,000 | 0.001 | |
| R_50 | ~7,000 | 0.001 | |
| R_75 | ~800,000 | 0.001 | Most popular — start here |
| R_100 | ~1,500 | 0.001 | Most volatile |
| CRASH1000 | ~9,000 | 0.001 | Sudden crashes, slow recoveries |
| BOOM1000 | ~9,000 | 0.001 | Sudden booms, slow declines |
| CRASH500 | ~7,000 | 0.001 | Crashes more frequently |
| BOOM500 | ~7,000 | 0.001 | |
| STPIDX | ~100 | 0.10 | Step Index — moves in fixed steps |
| GBP_USD | ~1.27 | 0.0001 | Standard forex |
| EUR_USD | ~1.08 | 0.0001 | Standard forex |

---

## Common errors and fixes

### `ModuleNotFoundError: No module named 'websocket'`
```bash
pip install websocket-client
```

### `Deriv API error: ... InvalidToken`
Your `DERIV_API_TOKEN` is wrong or expired. Go back to app.deriv.com → API Token and create a new one.

### `Deriv WebSocket timed out after 30s`
Deriv's API is occasionally slow. The bot will retry on the next loop. If it persists, check your internet connection.

### `Bot shows WAIT for all pairs constantly`
This is normal for the first 1–2 loops while historical data loads. Wait for the second loop. If it continues, check that your API token has **Read** scope.

### `ML fitted=[]` in the log
Normal at startup. The ML model needs 200 M5 bars before it trains. For M5 bars at 5-min intervals that's about 16 hours of live data. Until then, the ML gate is bypassed (signals still fire based on pure SMC).

### `Telegram disabled: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID`
Add both values to your `.env` file. Double-check there are no spaces around the `=`.

### `no analysis data` for all pairs
Normal on the very first loop. Wait one more loop (2 minutes) for H4 data to load.

### `Loop error: Deriv WebSocket error: ...`
The WebSocket connection dropped. The bot catches this and retries next loop. If it happens every loop, your token may have expired or Deriv's API is down.

### Pip values look wrong in Telegram (millions of pips for R_75)
You're using the original `telegram_notifier.py`. Replace it with the patched version provided — it uses the correct pip size for each instrument.

---

## Synthetics vs Forex — key differences for this bot

| | Synthetics (R_75 etc.) | Forex (GBP_USD etc.) |
|---|---|---|
| Trading hours | 24/7 always open | Weekdays only, session windows |
| Session filter | Bypassed (all hours OK) | Active (best hours only) |
| Pip size | 0.001 (not 0.0001) | 0.0001 |
| Price range | Very high (R_75 ~800k) | ~1–200 |
| Spread | Very tight (algorithmic) | Slightly wider |
| Volatility | Controlled (Deriv sets it) | Market-driven |
| Best to start | **R_75** — most liquid | GBP_USD if you prefer forex |

---

## How to stop the bot safely

```
Ctrl + C
```

The bot catches `SIGINT` and closes all open paper positions before shutting down. Never close the terminal window without Ctrl+C — it won't log the shutdown cleanly.

---

## File structure after first run

```
your_bot_folder/
├── .env                     ← your credentials
├── main.py
├── paper_trader.py
├── [... all other .py files ...]
├── data/
│   ├── R_75_H4.pkl          ← cached Deriv candle data
│   ├── R_75_H1.pkl
│   └── [... per pair per timeframe ...]
└── logs/
    ├── trades.jsonl          ← every trade opened/closed
    ├── bot_state.json        ← latest bot snapshot
    ├── ml_R_75.pkl           ← trained ML model for R_75 (after 16h)
    ├── ml_R_100.pkl
    └── confluence_stats.json ← win/loss per confluence type
```

---

*Patched for Deriv WebSocket mode | May 2026*
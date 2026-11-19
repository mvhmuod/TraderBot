# TraderBot

TraderBot is a desktop app for Windows that **paper-trades** one crypto trading pair with a
simple moving-average (SMA) crossover strategy. It reads live, public market data (candles and
prices) from a crypto exchange through the [ccxt](https://github.com/ccxt/ccxt) library,
simulates every order against a pretend account, and shows the result in a window: a candlestick
chart with the two moving averages and trade markers, the account balance and P&L, open
positions, trade history and a log. It is a learning and testing project.

> **Paper trading only.**
> Every order is **simulated**. No real money is used and there is **no code that places real
> orders**. The app does **not use API keys** and never asks for them; it only reads public price
> data.
>
> **Not financial advice.** This project is for learning. Nothing in it is a recommendation to
> buy or sell anything. The default numbers (moving-average lengths, position size, fees, and so
> on) are **placeholders** so the app starts with valid settings. They are **not
> recommendations**.

## Features

- Live public market data from Binance, Kraken, Coinbase, Bybit or OKX (no account needed).
- Any spot trading pair the exchange lists (searchable list, or type the pair yourself).
- Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d (Coinbase does not offer 4h candles).
- SMA crossover strategy with adjustable fast and slow lengths.
- Simulated fills with a configurable fee and slippage.
- Optional stop-loss, take-profit and daily loss limit (each can be switched off).
- Candlestick chart with fast/slow SMA lines, BUY/SELL markers and a last-price line.
- Summary bar: cash, equity, total P&L, realized P&L, current position, last price and status.
- Positions table, trade history table and a log.
- Everything is saved in a local SQLite database, so the paper account, trades and settings are
  kept between runs.
- Reset the paper account at any time while the bot is stopped.

## Requirements

- Windows
- Python 3.15
- An internet connection (for public market data)

## Setup

Open a terminal (Command Prompt or PowerShell) in the project folder and run:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --only-binary=:all: -r requirements.txt
```

`requirements.txt` pins the exact library versions the project was tested with.
`--only-binary=:all:` installs pre-built packages only, so no compiler is needed.

## Running

Double-click `run.bat`, or run:

```bat
.venv\Scripts\python app.py
```

Pick the exchange, trading pair, timeframe and other settings on the left, then press **Start**.
Press **Stop** to stop the bot. Open positions are kept when you stop, and the bot continues
with them the next time you start it. If you switch to another trading pair while a position is
open, the bot keeps valuing that position at its latest price and still applies the stop-loss,
take-profit and daily loss limit to it, but it only buys and sells on crossover signals for the
selected pair.

The paper account's cash is in one currency: the quote currency of the first pair you start
(USDT for BTC/USDT), or the currency of the pair selected when you press **Reset paper account**.
An account created by an earlier version of TraderBot takes the currency of its past trades and
open positions; if those used more than one currency, the account has to be reset.
Any pair priced in that currency can be traded (for example ETH/USDT next to BTC/USDT). If you
press Start on a pair priced in a different currency (for example ETH/BTC on a USDT account), the
bot does not start and offers to reset the paper account in the new currency. Answering No keeps
everything as it was.

Only one TraderBot window can use the paper account at a time. If TraderBot is already running
(or an old window is still shutting down), a second copy shows "TraderBot is already running"
and does not open the account.

## How it works

1. **Closed candles only.** Every poll (every *Poll interval* seconds) the bot downloads recent
   candles and drops any candle that is still forming. The strategy only ever looks at candles
   that have closed, so signals do not flicker. Because the PC clock can run a few seconds ahead
   of the exchange, the newest candle is only treated as closed once the exchange has started the
   next candle, or 10 seconds after its close time. If the exchange returns fewer closed candles
   than the slow SMA needs, the status shows an error ("Not enough candle history") and the bot
   cannot trade until you choose a smaller *Slow SMA*.
2. **The first candle after Start is only observed.** When you press Start, the newest closed
   candle already existed before the bot was watching, so its signal is never traded. The bot
   waits for the next candle to close.
3. **SMA crossover rule.** For each newly closed candle the bot compares the fast and slow simple
   moving averages of the close price on the last two candles:
   - **BUY** when the fast SMA was at or below the slow SMA and is now above it.
   - **SELL** when the fast SMA was at or above the slow SMA and is now below it.
   - Otherwise nothing happens. The bot is long-only: it buys only when it holds nothing, and a
     SELL signal sells the whole position.
4. **Order size.** A buy spends *Position size* % of the account equity, but never more than the
   available cash. Orders smaller than 1 unit of the quote currency are skipped.
5. **Simulated fills with fee and slippage.** Orders fill at the latest price, made slightly
   worse by the slippage (higher when buying, lower when selling). The fee is a percentage of
   the order value and is taken from the money spent (buy) or received (sell).
6. **Average cost includes fees.** The average cost of a position is everything paid for it,
   fees included, divided by the amount held. It is the break-even price.
7. **Stop-loss and take-profit.** On every poll, for every open position (also one in a pair
   other than the selected one), the latest price is compared with the average cost. The position
   is sold if the price is *Stop-loss* % or more below it, or *Take-profit* % or more above it.
   If no current price can be fetched for a position, it is not sold, and the log says so.
8. **Daily loss limit.** The bot records the account equity (cash plus positions at the latest
   price) at its first poll of each UTC day. If equity falls by *Daily loss limit* % or more from
   that value, the bot sells every open position and halts. Pressing Start again on the same UTC
   day is refused while equity is still at or below that limit.
9. **Equity when a price is missing.** When a poll fails (for example a network problem), equity
   and Total P&L keep using the last known price of each position. A position is valued at its
   average cost only while no price for it has been seen yet (for example right after the app
   starts).

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| Exchange | binance | Where public market data comes from. |
| Trading pair | BTC/USDT | The pair to paper-trade, written as BASE/QUOTE. Its QUOTE must be the paper account's currency (see Running). |
| Timeframe | 5m | Candle length. The bot acts at most once per closed candle. Coinbase does not offer 4h. |
| Fast SMA | 9 | Number of candles in the fast moving average (1 to 500). |
| Slow SMA | 21 | Number of candles in the slow moving average (2 to 500, larger than the fast one). OKX and Coinbase return at most 300 candles, so there it can be at most 298. |
| Position size | 10 % | Share of equity used for each buy, limited by available cash. |
| Stop-loss | Off (0) | Sell if the price falls this % below the average cost. |
| Take-profit | Off (0) | Sell if the price rises this % above the average cost. |
| Daily loss limit | Off (0) | Halt the bot if equity falls this % below its value at the first poll of the UTC day. |
| Fee | 0.100 % | Simulated exchange fee on every fill (below 10 %). |
| Slippage | 0.050 % | Simulated price slippage on every fill (below 10 %). |
| Poll interval | 15 s | How often the bot checks prices (at least 5 s). |
| Starting balance | 10,000.00 | Cash for a fresh paper account, in the selected pair's quote currency. Only used when the account is created or when you press **Reset paper account**. |

All settings are locked while the bot is running. They are saved when the bot starts and when
the window closes. Data is stored in `data/traderbot.db`, which stays on your computer and is
not part of the repository. While the app is open it also keeps a lock file,
`data/traderbot.db.lock`, so that a second copy of the app cannot use the same account.

## Running tests

```bat
.venv\Scripts\python -m pytest -q
```

The tests use fake market data and temporary databases; they do not access the network. The
`run.bat` tests start a copy of the launcher with a stand-in for the app.

## Project structure

```text
TraderBot/
├── app.py                    Starts the desktop app
├── run.bat                   Double-click launcher for Windows
├── requirements.txt          Pinned library versions
├── traderbot/
│   ├── config.py             Settings (BotConfig), validation, supported exchanges/timeframes
│   ├── models.py             Shared data types (Trade, Position, AccountSnapshot, StepResult, ...)
│   ├── storage.py            SQLite storage for the account, positions, trades and settings
│   ├── engine.py             The trading loop: one step per poll
│   ├── risk.py               Order size, stop-loss / take-profit, daily loss limit
│   ├── broker/paper.py       Simulated broker (fills, fees, slippage, P&L)
│   ├── data/feed.py          Public market data via ccxt (closed candles, last price, pairs)
│   ├── strategy/ma_cross.py  SMA crossover strategy
│   └── gui/
│       ├── app.py            Qt application setup
│       ├── main_window.py    Main window
│       ├── chart.py          Candlestick chart (pyqtgraph)
│       └── worker.py         Background thread that runs the engine
└── tests/                    Automated tests (pytest)
```

## Roadmap

### v1 (done)

- Desktop app (PySide6) with settings panel, summary bar, chart, positions, trade history and log.
- Live public market data from five exchanges via ccxt, closed candles only.
- SMA crossover strategy.
- Paper broker with fees, slippage, average cost and realized P&L.
- Stop-loss, take-profit and daily loss limit.
- Local SQLite storage of the paper account, trades and settings; paper account reset.
- The paper account remembers its currency and refuses pairs priced in another one (with an
  offer to reset).
- Automated tests for every module.

### Ideas under discussion (not yet approved)

- A backtest tab to try settings on past data.
- More strategies (for example RSI).
- Trading several pairs at once.
- Live trading, only after a long period of paper testing and with explicit safeguards.

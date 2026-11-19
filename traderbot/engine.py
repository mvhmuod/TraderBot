"""The trading loop: one call to ``BotEngine.step()`` = one poll of the market.

This module is plain Python (no Qt), so it can be tested with fake data feeds.
All orders go to the PaperBroker, which only SIMULATES fills: nothing here can
ever place a real order.

What one step does, in this order:
1. Stop early if the bot halted itself (daily loss limit).
2. Load the latest CLOSED candles and the last price.
3. Remember the account equity at the start of the UTC day (once per day).
4. Exit an open position if the stop-loss or take-profit price was reached.
5. If the daily loss limit is reached: close the position and halt.
6. If a NEW candle has closed since the last step, act on the crossover signal.
7. Report candles, price, account and trades back to the caller (the GUI).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pandas as pd

from traderbot.broker.paper import BrokerError, PaperBroker
from traderbot.config import BotConfig
from traderbot.data.feed import DataFeedError, MarketDataFeed
from traderbot.models import Side, Signal, StepResult, Trade, TradeReason
from traderbot.risk import RiskManager, utc_day_key
from traderbot.storage import Storage
from traderbot.strategy.ma_cross import MACrossoverStrategy

logger = logging.getLogger(__name__)

WATCHING_MESSAGE = "Watching for the next crossover (waiting for the next candle to close)"

# Short labels used in log messages, e.g. "SELL 0.01 BTC/USDT @ 60,000.00 (stop-loss)".
_REASON_LABELS = {
    TradeReason.SIGNAL: "signal",
    TradeReason.STOP_LOSS: "stop-loss",
    TradeReason.TAKE_PROFIT: "take-profit",
    TradeReason.DAILY_LIMIT: "daily loss limit",
}


class EngineError(Exception):
    """The bot cannot start (message is safe to show to the user)."""


def _fmt_money(value: float) -> str:
    """Quote amounts with thousands separators and 2 decimals, e.g. 10,000.00."""
    return f"{value:,.2f}"


def _fmt_price(price: float) -> str:
    """Prices: 2 decimals for normal prices, up to 8 decimals for very small ones."""
    if abs(price) >= 1:
        return f"{price:,.2f}"
    text = f"{price:,.8f}".rstrip("0").rstrip(".")
    return text or "0"


def _fmt_amount(amount: float) -> str:
    """Base amounts with up to 8 decimals and no trailing zeros, e.g. 0.0153."""
    text = f"{amount:,.8f}".rstrip("0").rstrip(".")
    return text or "0"


def describe_trade(trade: Trade) -> str:
    """One short human-readable line for a fill, e.g. "BUY 0.0153 BTC/USDT @ 62,690.12 (signal)"."""
    label = _REASON_LABELS.get(trade.reason, str(trade.reason))
    text = (f"{trade.side.value} {_fmt_amount(trade.amount)} {trade.symbol} "
            f"@ {_fmt_price(trade.price)} ({label})")
    if trade.side == Side.SELL:
        text += f", P&L {trade.realized_pnl:+,.2f}"
    return text


class BotEngine:
    """Runs the SMA-crossover strategy against a paper account, one poll at a time.

    ``clock`` returns "now" in UTC milliseconds; tests inject a fake one.
    The engine is not thread-safe: use it from one thread only (the GUI worker thread).
    """

    def __init__(self, config: BotConfig, feed: MarketDataFeed, broker: PaperBroker, storage: Storage,
                 clock: Callable[[], int] | None = None) -> None:
        self.config: BotConfig = config
        self._feed = feed
        self._broker = broker
        self._storage = storage
        self._clock: Callable[[], int] = clock if clock is not None else (lambda: int(time.time() * 1000))
        self._risk = RiskManager(config)
        # With invalid MA settings the strategy cannot be built yet; start() then
        # reports the problem through config.validate() instead of crashing here.
        self._strategy: MACrossoverStrategy | None = self._build_strategy()
        # Per-run state (reset by start()).
        self._last_processed_ts: int | None = None
        self._halted = False

    def _build_strategy(self) -> MACrossoverStrategy | None:
        try:
            return MACrossoverStrategy(self.config.fast_period, self.config.slow_period)
        except ValueError:
            return None

    @property
    def halted(self) -> bool:
        """True after the daily loss limit stopped the bot (until the next start())."""
        return self._halted

    # ------------------------------------------------------------------ start
    def start(self) -> list[str]:
        """Prepare a new run and return messages for the log.

        Raises EngineError if the settings are invalid, or if today's daily loss
        limit was already reached (the bot must stay off until the next UTC day).
        """
        errors = self.config.validate()
        if errors:
            raise EngineError("Invalid settings: " + "; ".join(errors))

        cfg = self.config
        # Rebuild from the (possibly edited) config so this run uses current settings.
        risk = RiskManager(cfg)

        # Refuse to start if the daily loss limit was already hit today.
        if cfg.max_daily_loss_pct > 0:
            now = int(self._clock())
            day_start = self._storage.get_meta(utc_day_key(now))
            if day_start is not None:
                day_start = float(day_start)
                mark_prices: dict[str, float] = {}
                try:
                    for symbol in self._broker.positions():
                        mark_prices[symbol] = self._feed.fetch_last_price(symbol)
                except DataFeedError as e:
                    raise EngineError(f"Could not check the daily loss limit: {e}") from e
                equity = self._broker.equity(mark_prices)
                if risk.daily_limit_hit(day_start, equity):
                    raise EngineError(
                        f"Daily loss limit already reached today: equity {_fmt_money(equity)} vs "
                        f"{_fmt_money(day_start)} at the start of the UTC day. "
                        "The bot can be started again tomorrow (UTC)."
                    )

        # All checks passed: reset the per-run state (nothing changes if start() raised).
        self._strategy = MACrossoverStrategy(cfg.fast_period, cfg.slow_period)
        self._risk = risk
        self._broker.set_costs(cfg.fee_rate, cfg.slippage_rate)
        self._last_processed_ts = None
        self._halted = False

        message = (f"Started paper trading {cfg.symbol} on {cfg.exchange_id} ({cfg.timeframe} candles, "
                   f"SMA {cfg.fast_period}/{cfg.slow_period}, simulated orders only)")
        logger.info(message)
        return [message]

    # ------------------------------------------------------------------- step
    def step(self) -> StepResult:
        """Run one poll. Never raises: problems are reported in ``result.error``."""
        now = int(self._clock())
        cfg = self.config
        result = StepResult(timestamp=now, symbol=cfg.symbol, timeframe=cfg.timeframe)

        # 1. A halted bot does nothing until it is started again.
        if self._halted:
            result.halted = True
            return result

        mark_prices: dict[str, float] = {}
        try:
            self._run_step(result, now, mark_prices)
        except Exception as e:  # noqa: BLE001 - the step must never crash the bot
            logger.exception("Unexpected error during step")
            result.error = f"Unexpected error: {e}"
            try:
                result.account = self._broker.snapshot(mark_prices)
            except Exception:  # noqa: BLE001
                logger.exception("Could not read the account after an error")
        return result

    def _run_step(self, result: StepResult, now: int, mark_prices: dict[str, float]) -> None:
        """Steps 2-7 of the algorithm. Fills ``result`` and ``mark_prices`` in place."""
        cfg = self.config
        symbol = cfg.symbol
        strategy = self._strategy
        if strategy is None:
            # Only possible when step() is called without a successful start().
            result.error = "Invalid moving-average settings: fast SMA must be at least 1 and below slow SMA"
            result.account = self._broker.snapshot({})
            return

        # 2. Market data (closed candles only) and the latest price.
        limit = max(cfg.history_limit, strategy.min_candles + 1)
        try:
            candles = self._feed.fetch_closed_candles(symbol, cfg.timeframe, limit=limit, now_ms=now)
            last_price = self._feed.fetch_last_price(symbol)
        except DataFeedError as e:
            result.error = str(e)
            result.account = self._broker.snapshot({})
            return   # the bot keeps running; the next step tries again
        candles = strategy.compute(candles)

        # 3. Equity at the start of the UTC day (stored once per day).
        mark_prices[symbol] = last_price
        day_key = utc_day_key(now)
        day_start = self._storage.get_meta(day_key)
        if day_start is None:
            day_start = self._broker.equity(mark_prices)
            self._storage.set_meta(day_key, day_start)
            result.messages.append(f"Equity at the start of the UTC day: {_fmt_money(day_start)}")
        day_start = float(day_start)

        # 4. Stop-loss / take-profit.
        position = self._broker.get_position(symbol)
        if position is not None:
            reason = self._risk.exit_reason(position, last_price)
            if reason is not None:
                self._close(result, symbol, last_price, now, reason)

        # 5. Daily loss limit: close everything for this symbol and halt.
        equity = self._broker.equity(mark_prices)
        if self._risk.daily_limit_hit(day_start, equity):
            if self._broker.get_position(symbol) is not None:
                self._close(result, symbol, last_price, now, TradeReason.DAILY_LIMIT)
            self._halted = True
            result.halted = True
            result.messages.append(
                f"Daily loss limit reached: equity {_fmt_money(equity)} is down "
                f"{cfg.max_daily_loss_pct:g}% or more from {_fmt_money(day_start)} at the start of "
                "the UTC day. Bot halted."
            )
        else:
            # 6. Act on the signal only once per newly closed candle.
            self._handle_signal(result, candles, symbol, last_price, now, equity, strategy)

        # 7. Report back (account recomputed after any trades made in this step).
        result.candles = candles
        result.last_price = last_price
        result.account = self._broker.snapshot(mark_prices)

    def _handle_signal(self, result: StepResult, candles: pd.DataFrame, symbol: str, last_price: float,
                       now: int, equity: float, strategy: MACrossoverStrategy) -> None:
        """Step 6: evaluate the crossover on a NEW closed candle and trade on it."""
        if candles.empty:
            return
        newest_ts = int(candles["timestamp"].iloc[-1])

        if self._last_processed_ts is None:
            # First step after start: the newest candle closed before we were
            # watching, so its signal is never traded.
            self._last_processed_ts = newest_ts
            result.messages.append(WATCHING_MESSAGE)
            return
        if newest_ts <= self._last_processed_ts:
            return   # no new candle since the last step

        self._last_processed_ts = newest_ts
        sig = strategy.signal(candles)
        result.signal = sig
        position = self._broker.get_position(symbol)

        if sig == Signal.BUY:
            if position is not None:
                result.messages.append("Buy signal ignored: already holding")
                return
            quote = self._risk.order_quote_amount(self._broker.cash, equity)
            if quote <= 0:
                result.messages.append("Buy signal skipped: order size below minimum")
                return
            try:
                trade = self._broker.market_buy(symbol, quote, last_price, now, TradeReason.SIGNAL)
            except (BrokerError, ValueError) as e:
                self._order_failed(result, "Buy", e)
                return
            self._add_trade(result, trade)
        elif sig == Signal.SELL:
            if position is None:
                result.messages.append("Sell signal ignored: nothing to sell")
                return
            self._close(result, symbol, last_price, now, TradeReason.SIGNAL)
        else:
            result.messages.append("New candle closed: no crossover")

    # ---------------------------------------------------------------- helpers
    def _close(self, result: StepResult, symbol: str, price: float, now: int, reason: TradeReason) -> None:
        """Sell the whole position; an order problem becomes a message + result.error."""
        try:
            trade = self._broker.close_position(symbol, price, now, reason)
        except (BrokerError, ValueError) as e:
            self._order_failed(result, "Sell", e)
            return
        self._add_trade(result, trade)

    @staticmethod
    def _add_trade(result: StepResult, trade: Trade) -> None:
        result.new_trades.append(trade)
        message = describe_trade(trade)
        result.messages.append(message)
        logger.info(message)

    @staticmethod
    def _order_failed(result: StepResult, side: str, error: Exception) -> None:
        message = f"{side} order failed: {error}"
        logger.warning(message)
        result.messages.append(message)
        result.error = message

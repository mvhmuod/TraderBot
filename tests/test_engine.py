"""Tests for traderbot.engine.BotEngine (no network).

A FakeFeed serves scripted candles: candle i starts at T0 + i*TF and closes at
T0 + (i+1)*TF, and only candles that have closed at the engine's clock time are
returned (like the real feed). The clock is injected so tests control "now".

Strategy settings in these tests: fast SMA 2, slow SMA 3 (min_candles = 4).
Close series used for crossovers:
  [10, 10, 10, 10, 20] -> at the 4th candle fast=10, slow=10 (touching);
                          at the 5th fast=15 > slow=13.33 -> BUY
  [.., 10, 20, 20, 5]  -> fast 20 > slow 16.67, then fast 12.5 < slow 15 -> SELL
Default costs: fee_rate = 0.001, slippage_rate = 0.0005.
"""
from __future__ import annotations

import pandas as pd
import pytest

from traderbot.broker.paper import BrokerError, InsufficientFunds, PaperBroker
from traderbot.config import BotConfig
from traderbot.data.feed import DataFeedError
from traderbot.engine import WATCHING_MESSAGE, BotEngine, EngineError, describe_trade
from traderbot.models import CANDLE_COLUMNS, AccountSnapshot, Side, Signal, TradeReason
from traderbot.risk import utc_day_key
from traderbot.storage import Storage

SYM = "BTC/USDT"
TF = 5 * 60_000                      # 5m in ms
DAY0 = 1_704_067_200_000             # 2024-01-01 00:00:00 UTC
DAY_MS = 86_400_000
T0 = DAY0 + 3_600_000                # first candle starts at 01:00 UTC

FLAT = [10.0] * 5
BUY_CROSS = [10.0, 10.0, 10.0, 10.0, 10.0, 20.0]          # candle 5 is a BUY crossover
BUY_THEN_SELL = BUY_CROSS + [20.0, 5.0]                    # candle 7 is a SELL crossover


def at(n: int) -> int:
    """A clock time at which candles 0 .. n-1 are closed (1 s after candle n-1 closed)."""
    return T0 + n * TF + 1000


class Clock:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class FakeFeed:
    """Mimics MarketDataFeed.fetch_closed_candles / fetch_last_price with scripted data."""

    def __init__(self, closes: list[float], price: float = 20.0) -> None:
        self.closes = list(closes)
        self.price = price
        self.error: Exception | None = None          # raised by both calls
        self.price_error: Exception | None = None    # raised by fetch_last_price only
        self.candle_calls: list[tuple] = []
        self.price_calls: list[str] = []

    def fetch_closed_candles(self, symbol, timeframe, limit, now_ms=None):
        self.candle_calls.append((symbol, timeframe, limit, now_ms))
        if self.error is not None:
            raise self.error
        rows = [[T0 + i * TF, c, c, c, c, 1.0] for i, c in enumerate(self.closes)
                if T0 + (i + 1) * TF <= now_ms]
        df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
        df["timestamp"] = df["timestamp"].astype("int64")
        for col in CANDLE_COLUMNS[1:]:
            df[col] = df[col].astype("float64")
        return df.tail(limit).reset_index(drop=True)

    def fetch_last_price(self, symbol):
        self.price_calls.append(symbol)
        if self.error is not None:
            raise self.error
        if self.price_error is not None:
            raise self.price_error
        return self.price


def make_config(**overrides) -> BotConfig:
    values = dict(exchange_id="binance", symbol=SYM, timeframe="5m", fast_period=2, slow_period=3,
                  history_limit=5, starting_balance=10_000.0, fee_rate=0.001, slippage_rate=0.0005,
                  position_size_pct=10.0, stop_loss_pct=0.0, take_profit_pct=0.0,
                  max_daily_loss_pct=0.0, poll_interval_sec=15)
    values.update(overrides)
    return BotConfig(**values)


class Setup:
    """Real Storage(':memory:') + PaperBroker, FakeFeed and an injectable clock."""

    def __init__(self, closes: list[float], price: float = 20.0, now: int | None = None,
                 broker_fee: float = 0.001, **config_overrides) -> None:
        self.config = make_config(**config_overrides)
        self.storage = Storage(":memory:")
        self.broker = PaperBroker(self.storage, fee_rate=broker_fee, slippage_rate=0.0005,
                                  starting_balance=10_000.0)
        self.feed = FakeFeed(closes, price)
        self.clock = Clock(now if now is not None else at(len(closes)))
        self.engine = BotEngine(self.config, self.feed, self.broker, self.storage, clock=self.clock)

    def step_at(self, n: int):
        self.clock.now = at(n)
        return self.engine.step()


@pytest.fixture
def make():
    setups: list[Setup] = []

    def _make(*args, **kwargs) -> Setup:
        s = Setup(*args, **kwargs)
        setups.append(s)
        return s

    yield _make
    for s in setups:
        s.storage.close()


# --------------------------------------------------------------------------- helpers
def test_describe_trade_messages(make):
    s = make(BUY_CROSS)
    buy = s.broker.market_buy(SYM, 1000.0, 20.0, at(6), TradeReason.SIGNAL)
    # amount = 999 / 20.01 = 49.925037481...
    assert describe_trade(buy) == "BUY 49.92503748 BTC/USDT @ 20.01 (signal)"
    sell = s.broker.close_position(SYM, 0.5, at(6), TradeReason.STOP_LOSS)
    text = describe_trade(sell)
    assert text.startswith("SELL 49.92503748 BTC/USDT @ 0.49975 (stop-loss), P&L -")


# --------------------------------------------------------------------------- start()
def test_start_returns_messages_and_resets_state(make):
    s = make(BUY_CROSS)
    messages = s.engine.start()
    assert isinstance(messages, list) and messages
    assert all(isinstance(m, str) for m in messages)
    assert s.engine.halted is False


@pytest.mark.parametrize("overrides", [
    dict(poll_interval_sec=1),
    dict(fast_period=5, slow_period=3, history_limit=10),
    dict(symbol="BTCUSDT"),
    dict(position_size_pct=0.0),
])
def test_start_refuses_invalid_config(make, overrides):
    s = make(BUY_CROSS, **overrides)    # constructing the engine must not crash
    with pytest.raises(EngineError):
        s.engine.start()


def test_step_with_unbuildable_strategy_reports_error(make):
    s = make(BUY_CROSS, fast_period=5, slow_period=3, history_limit=10)
    result = s.engine.step()
    assert result.error
    assert result.new_trades == []
    assert isinstance(result.account, AccountSnapshot)


def test_start_uses_config_costs(make):
    # Broker created with fee 0, config says 0.001 -> start() applies the config's costs.
    s = make(BUY_CROSS, broker_fee=0.0)
    s.engine.start()
    trade = s.broker.market_buy(SYM, 1000.0, 20.0, at(6), TradeReason.SIGNAL)
    assert trade.fee == 1.0


def test_start_refuses_when_daily_limit_already_hit_and_needs_price(make):
    s = make(BUY_CROSS, price=50.0, max_daily_loss_pct=5.0)
    # Hold 999/20.01 = 49.925... units; cash 9000.
    s.broker.market_buy(SYM, 1000.0, 20.0, T0, TradeReason.SIGNAL)
    s.storage.set_meta(utc_day_key(s.clock.now), 12_000.0)
    # Threshold = 12000 * 0.95 = 11400.
    # At price 50 equity = 9000 + 49.925*50 = 11496.25 > 11400 -> may start.
    s.engine.start()
    assert s.feed.price_calls == [SYM]
    # At price 40 equity = 9000 + 1997.0 = 10997.0 <= 11400 -> refused.
    s.feed.price = 40.0
    with pytest.raises(EngineError, match="Daily loss limit"):
        s.engine.start()


def test_start_daily_check_price_error_becomes_engine_error(make):
    s = make(BUY_CROSS, max_daily_loss_pct=5.0)
    s.broker.market_buy(SYM, 1000.0, 20.0, T0, TradeReason.SIGNAL)
    s.storage.set_meta(utc_day_key(s.clock.now), 12_000.0)
    s.feed.price_error = DataFeedError("Network problem talking to binance: timeout")
    with pytest.raises(EngineError, match="Network problem"):
        s.engine.start()


def test_start_daily_check_skipped_without_day_start_or_limit(make):
    s = make(BUY_CROSS, max_daily_loss_pct=5.0)
    s.broker.market_buy(SYM, 1000.0, 20.0, T0, TradeReason.SIGNAL)
    s.engine.start()                        # no day_start stored yet -> nothing to check
    assert s.feed.price_calls == []
    t = make(BUY_CROSS, max_daily_loss_pct=0.0)
    t.storage.set_meta(utc_day_key(t.clock.now), 1e9)
    t.engine.start()                        # limit disabled
    assert t.feed.price_calls == []


# --------------------------------------------------------------------------- data requests
def test_step_requests_closed_candles_with_clock_time(make):
    s = make(BUY_CROSS, history_limit=5)
    s.engine.start()
    s.step_at(6)
    # limit = max(history_limit=5, min_candles + 1 = 4 + 1) = 5 ; now_ms = clock time
    assert s.feed.candle_calls == [(SYM, "5m", 5, at(6))]
    t = make(BUY_CROSS, history_limit=50)
    t.engine.start()
    t.step_at(6)
    assert t.feed.candle_calls[-1] == (SYM, "5m", 50, at(6))


# --------------------------------------------------------------------------- signals
def test_first_step_never_trades_even_on_crossover(make):
    s = make(BUY_CROSS)
    s.engine.start()
    result = s.step_at(6)                   # newest closed candle IS a BUY crossover
    assert result.new_trades == []
    assert result.signal == Signal.HOLD
    assert WATCHING_MESSAGE in result.messages
    assert s.broker.get_position(SYM) is None
    assert s.broker.cash == 10_000.0
    assert result.error is None
    # Same candle again: still nothing.
    again = s.step_at(6)
    assert again.new_trades == [] and again.signal == Signal.HOLD
    assert WATCHING_MESSAGE not in again.messages


def test_buy_on_new_closed_candle_with_right_size(make):
    s = make(BUY_CROSS, price=20.0)
    s.engine.start()
    first = s.step_at(5)                    # candles 0-4 (flat): only starts watching
    assert first.new_trades == []
    result = s.step_at(6)                   # candle 5 closed: BUY crossover
    assert result.error is None
    assert result.signal == Signal.BUY
    assert len(result.new_trades) == 1
    trade = result.new_trades[0]
    # q = min(cash 10000, equity 10000 * 10 / 100) = 1000
    # fill = 20 * 1.0005 = 20.01 ; fee = 1000 * 0.001 = 1 ; amount = 999 / 20.01
    assert trade.side == Side.BUY
    assert trade.reason == TradeReason.SIGNAL
    assert trade.quote_value == 1000.0
    assert trade.fee == 1.0
    assert trade.price == pytest.approx(20.01, rel=1e-15)
    assert trade.amount == pytest.approx(999 / 20.01, rel=1e-15)
    assert trade.timestamp == at(6)
    assert trade.id is not None
    assert s.broker.cash == 9000.0
    assert s.storage.load_trades() == [trade]
    assert describe_trade(trade) in result.messages


def test_buy_size_capped_by_cash(make):
    s = make(BUY_CROSS, price=20.0, position_size_pct=50.0)
    s.broker.market_buy("ETH/USDT", 9_800.0, 100.0, T0, TradeReason.SIGNAL)
    # cash 200, equity (ETH valued at avg_cost) = 10000 -> 50% = 5000 -> capped to cash 200
    s.engine.start()
    s.step_at(5)
    result = s.step_at(6)
    assert result.new_trades[0].quote_value == 200.0
    assert s.broker.cash == 0.0


def test_buy_skipped_when_order_below_minimum(make):
    s = make(BUY_CROSS, price=20.0, position_size_pct=100.0)
    s.broker.market_buy("ETH/USDT", 9_999.5, 100.0, T0, TradeReason.SIGNAL)   # cash 0.5
    s.engine.start()
    s.step_at(5)
    result = s.step_at(6)
    assert result.signal == Signal.BUY
    assert result.new_trades == []
    assert "Buy signal skipped: order size below minimum" in result.messages
    assert result.error is None


def test_same_candle_never_acted_on_twice(make):
    s = make(BUY_CROSS, price=20.0)
    s.engine.start()
    s.step_at(5)
    bought = s.step_at(6)
    assert len(bought.new_trades) == 1
    # Sell outside the engine: if the engine re-read the same BUY candle it would buy again.
    s.broker.close_position(SYM, 20.0, at(6), TradeReason.SIGNAL)
    for extra_ms in (0, 5_000, 60_000, TF - 2_000):
        s.clock.now = at(6) + extra_ms     # still before candle 6 closes
        result = s.engine.step()
        assert result.new_trades == []
        assert result.signal == Signal.HOLD
        assert result.messages == []
    assert s.broker.get_position(SYM) is None
    assert len(s.storage.load_trades()) == 2


def test_hold_on_new_candle_reports_no_crossover(make):
    s = make(FLAT + [10.0])
    s.engine.start()
    s.step_at(5)
    result = s.step_at(6)
    assert result.signal == Signal.HOLD
    assert result.new_trades == []
    assert result.messages == ["New candle closed: no crossover"]


def test_sell_signal_closes_position(make):
    s = make(BUY_THEN_SELL, price=20.0)
    s.engine.start()
    s.step_at(5)
    buy = s.step_at(6).new_trades[0]
    hold = s.step_at(7)
    assert hold.signal == Signal.HOLD and hold.new_trades == []
    s.feed.price = 15.0
    result = s.step_at(8)
    assert result.signal == Signal.SELL
    assert len(result.new_trades) == 1
    sell = result.new_trades[0]
    assert sell.side == Side.SELL and sell.reason == TradeReason.SIGNAL
    # Sells the full amount: fill = 15 * 0.9995 = 14.9925 ; gross = amount * fill
    amount = buy.amount
    gross = amount * (15.0 * (1 - 0.0005))
    net = gross - gross * 0.001
    assert sell.amount == amount
    assert sell.quote_value == pytest.approx(net, rel=1e-12)
    assert sell.realized_pnl == pytest.approx(net - 1000.0, rel=1e-9)
    assert s.broker.get_position(SYM) is None
    assert s.broker.cash == pytest.approx(9000.0 + net, rel=1e-12)
    assert result.account.positions == {}


def test_buy_signal_ignored_while_holding(make):
    s = make(BUY_CROSS, price=20.0)
    s.broker.market_buy(SYM, 500.0, 20.0, T0, TradeReason.SIGNAL)
    s.engine.start()
    s.step_at(5)
    result = s.step_at(6)
    assert result.signal == Signal.BUY
    assert result.new_trades == []
    assert "Buy signal ignored: already holding" in result.messages
    assert len(s.storage.load_trades()) == 1


def test_sell_signal_ignored_while_flat(make):
    s = make(BUY_THEN_SELL, price=20.0)
    s.engine.start()
    s.step_at(7)                            # start watching after the BUY candle
    result = s.step_at(8)
    assert result.signal == Signal.SELL
    assert result.new_trades == []
    assert "Sell signal ignored: nothing to sell" in result.messages
    assert result.error is None


def test_empty_candles_does_not_start_watching(make):
    s = make(BUY_CROSS)
    s.engine.start()
    result = s.step_at(0)                   # no candle closed yet
    assert result.error is None
    assert result.candles is not None and result.candles.empty
    assert WATCHING_MESSAGE not in result.messages
    nxt = s.step_at(6)
    assert WATCHING_MESSAGE in nxt.messages
    assert nxt.new_trades == []


# --------------------------------------------------------------------------- stop-loss / take-profit
def _open_position(s: Setup) -> float:
    """Buy 1000 quote at 100: amount = 999/100.05, avg_cost = 1000/amount = 100.1501501..."""
    s.broker.market_buy(SYM, 1000.0, 100.0, T0, TradeReason.SIGNAL)
    return s.broker.get_position(SYM).avg_cost


def test_stop_loss_exit(make):
    s = make(FLAT, price=100.0, stop_loss_pct=5.0)
    avg = _open_position(s)
    assert avg == pytest.approx(100.0 * 1.0005 * 1000 / 999, rel=1e-12)
    s.engine.start()
    # threshold = avg * 0.95 = 95.1426...
    s.feed.price = 95.2
    assert s.step_at(5).new_trades == []
    s.feed.price = 95.0
    result = s.step_at(5)
    assert len(result.new_trades) == 1
    trade = result.new_trades[0]
    assert trade.side == Side.SELL and trade.reason == TradeReason.STOP_LOSS
    assert trade.realized_pnl < 0
    assert s.broker.get_position(SYM) is None
    assert describe_trade(trade) in result.messages
    assert "(stop-loss)" in describe_trade(trade)


def test_take_profit_exit(make):
    s = make(FLAT, price=100.0, take_profit_pct=10.0)
    _open_position(s)
    s.engine.start()
    # threshold = avg * 1.10 = 110.165...
    s.feed.price = 110.1
    assert s.step_at(5).new_trades == []
    s.feed.price = 110.2
    result = s.step_at(5)
    assert len(result.new_trades) == 1
    trade = result.new_trades[0]
    assert trade.reason == TradeReason.TAKE_PROFIT
    assert trade.realized_pnl > 0
    assert s.broker.get_position(SYM) is None


def test_exit_then_sell_signal_same_step_is_ignored(make):
    # Stop-loss closes first; the SELL crossover on the new candle then finds nothing to sell.
    s = make(BUY_THEN_SELL, price=20.0, stop_loss_pct=5.0)
    s.engine.start()
    s.step_at(5)
    s.step_at(6)                            # buys at 20
    s.step_at(7)
    s.feed.price = 5.0
    result = s.step_at(8)
    assert [t.reason for t in result.new_trades] == [TradeReason.STOP_LOSS]
    assert result.signal == Signal.SELL
    assert "Sell signal ignored: nothing to sell" in result.messages


# --------------------------------------------------------------------------- daily loss limit
def test_daily_limit_closes_halts_and_stops_working(make):
    s = make(FLAT, price=100.0, max_daily_loss_pct=5.0)
    s.broker.market_buy(SYM, 5_000.0, 100.0, T0, TradeReason.SIGNAL)
    amount = s.broker.get_position(SYM).amount         # 4995 / 100.05
    s.engine.start()
    first = s.step_at(5)
    day_start = 5_000.0 + amount * 100.0                # ~ 9992.50
    assert s.storage.get_meta(utc_day_key(at(5))) == pytest.approx(day_start, rel=1e-15)
    assert first.halted is False
    # Small drop: 95 -> equity ~ 9745.6 > 9492.9 threshold -> keep going.
    s.feed.price = 95.0
    assert s.step_at(5).halted is False
    # Big drop: 80 -> equity ~ 8994 <= threshold -> close + halt.
    s.feed.price = 80.0
    result = s.step_at(5)
    assert result.halted is True
    assert s.engine.halted is True
    assert [t.reason for t in result.new_trades] == [TradeReason.DAILY_LIMIT]
    assert s.broker.get_position(SYM) is None
    assert any("Daily loss limit reached" in m for m in result.messages)
    assert result.account is not None and result.account.positions == {}
    assert result.signal == Signal.HOLD

    # Later steps do nothing at all (no data requests, no trades).
    calls_before = (len(s.feed.candle_calls), len(s.feed.price_calls))
    s.feed.price = 200.0
    later = s.step_at(6)
    assert later.halted is True
    assert later.new_trades == [] and later.account is None and later.error is None
    assert (len(s.feed.candle_calls), len(s.feed.price_calls)) == calls_before
    assert len(s.storage.load_trades()) == 2

    # Restarting the same UTC day is refused (equity is still below the limit).
    with pytest.raises(EngineError, match="Daily loss limit"):
        s.engine.start()
    assert s.engine.halted is True

    # The next UTC day it can start again; start() clears the halt.
    s.clock.now = DAY0 + DAY_MS + 3_600_000
    s.engine.start()
    assert s.engine.halted is False


def test_daily_limit_without_position_halts_and_skips_signal(make):
    s = make(BUY_CROSS, price=20.0, max_daily_loss_pct=5.0)
    s.engine.start()
    s.step_at(5)                                        # day start = 10000
    s.storage.set_meta(utc_day_key(at(6)), 20_000.0)    # pretend the day started much higher
    result = s.step_at(6)                               # BUY crossover candle
    assert result.halted is True
    assert result.new_trades == []
    assert result.signal == Signal.HOLD                 # step 6 skipped
    assert s.broker.cash == 10_000.0


# --------------------------------------------------------------------------- day start
def test_day_start_set_once_per_utc_day(make):
    s = make(BUY_CROSS, price=20.0)
    s.engine.start()
    key1 = utc_day_key(at(5))
    assert key1 == "day_start:2024-01-01"
    assert s.storage.get_meta(key1) is None
    s.step_at(5)
    assert s.storage.get_meta(key1) == 10_000.0
    s.step_at(6)                                        # buys
    s.feed.price = 30.0                                 # equity changes
    s.step_at(6)
    assert s.storage.get_meta(key1) == 10_000.0         # unchanged
    # Next UTC day -> new key holding the equity at that moment.
    s.clock.now = DAY0 + DAY_MS + 1
    s.engine.step()
    key2 = utc_day_key(s.clock.now)
    assert key2 == "day_start:2024-01-02"
    expected = s.broker.equity({SYM: 30.0})
    assert s.storage.get_meta(key2) == pytest.approx(expected, rel=1e-15)
    assert s.storage.get_meta(key1) == 10_000.0


# --------------------------------------------------------------------------- errors
def test_data_feed_error_reported_and_bot_continues(make):
    s = make(BUY_CROSS, price=20.0)
    s.engine.start()
    s.feed.error = DataFeedError("Network problem talking to binance: timeout")
    result = s.step_at(5)
    assert result.error == "Network problem talking to binance: timeout"
    assert isinstance(result.account, AccountSnapshot)
    assert result.account.cash == 10_000.0
    assert result.new_trades == [] and result.halted is False
    assert s.engine.halted is False
    # Recovers on the next poll: the failed step did not count as "first step".
    s.feed.error = None
    ok = s.step_at(5)
    assert ok.error is None
    assert WATCHING_MESSAGE in ok.messages
    assert len(s.step_at(6).new_trades) == 1


def test_price_error_alone_is_reported(make):
    s = make(BUY_CROSS)
    s.engine.start()
    s.feed.price_error = DataFeedError("No valid price for BTC/USDT on binance")
    result = s.step_at(5)
    assert result.error == "No valid price for BTC/USDT on binance"
    assert result.account is not None


def test_unexpected_feed_exception_never_raises(make):
    s = make(BUY_CROSS)
    s.engine.start()
    s.feed.error = RuntimeError("boom")
    result = s.step_at(5)
    assert result.error == "Unexpected error: boom"
    assert result.new_trades == []
    assert isinstance(result.account, AccountSnapshot)
    s.feed.error = None
    assert s.step_at(5).error is None


@pytest.mark.parametrize("exc", [
    InsufficientFunds("Not enough cash"),
    BrokerError("Paper broker unavailable"),
    ValueError("Order size must be at least 1.0"),
])
def test_broker_error_on_buy_is_handled(make, monkeypatch, exc):
    s = make(BUY_CROSS, price=20.0)
    s.engine.start()
    s.step_at(5)

    def failing_buy(*args, **kwargs):
        raise exc

    monkeypatch.setattr(s.broker, "market_buy", failing_buy)
    result = s.step_at(6)
    assert result.signal == Signal.BUY
    assert result.new_trades == []
    assert result.error == f"Buy order failed: {exc}"
    assert result.error in result.messages
    assert result.account is not None and result.account.cash == 10_000.0
    assert result.candles is not None and result.last_price == 20.0
    # The candle is not retried.
    monkeypatch.undo()
    assert s.step_at(6).new_trades == []


def test_broker_error_on_exit_is_handled(make, monkeypatch):
    s = make(FLAT, price=100.0, stop_loss_pct=5.0)
    _open_position(s)
    s.engine.start()

    def failing_close(*args, **kwargs):
        raise BrokerError("Cannot sell")

    monkeypatch.setattr(s.broker, "close_position", failing_close)
    s.feed.price = 90.0
    result = s.step_at(5)
    assert result.error == "Sell order failed: Cannot sell"
    assert result.new_trades == []
    assert s.broker.get_position(SYM) is not None
    assert result.account is not None and result.candles is not None


def test_real_broker_rejection_below_minimum(make):
    # The real broker rejects a buy below MIN_ORDER_QUOTE (ValueError) ...
    s = make(BUY_CROSS, price=20.0)
    with pytest.raises(ValueError):
        s.broker.market_buy(SYM, 0.5, 20.0, T0, TradeReason.SIGNAL)
    # ... and the engine never sends such an order (risk sizing returns 0 -> skipped).
    s.engine.start()
    s.step_at(5)
    assert s.step_at(6).new_trades[0].quote_value >= 1.0


# --------------------------------------------------------------------------- reporting
def test_account_snapshot_reflects_trades_in_same_step(make):
    s = make(BUY_CROSS, price=20.0)
    s.engine.start()
    s.step_at(5)
    s.feed.price = 21.0
    result = s.step_at(6)
    trade = result.new_trades[0]
    acct = result.account
    assert acct.cash == 9000.0
    assert acct.positions[SYM].amount == trade.amount
    # equity marked at the step's last price (21), not the fill price
    assert acct.equity == pytest.approx(9000.0 + trade.amount * 21.0, rel=1e-15)
    assert acct.realized_pnl_total == 0.0
    assert result.last_price == 21.0
    assert list(result.candles.columns) == CANDLE_COLUMNS + ["fast_ma", "slow_ma"]
    assert result.candles["timestamp"].iloc[-1] == T0 + 5 * TF
    assert result.timestamp == at(6)
    assert result.symbol == SYM and result.timeframe == "5m"


def test_restart_resets_first_step_watching(make):
    s = make(BUY_THEN_SELL, price=20.0)
    s.engine.start()
    s.step_at(5)
    s.engine.start()                        # restart: the next step only watches again
    result = s.step_at(6)
    assert result.new_trades == []
    assert WATCHING_MESSAGE in result.messages

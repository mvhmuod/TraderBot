"""Tests for traderbot.data.feed (no network: a fake exchange is injected)."""
from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from traderbot.data.feed import DataFeedError, MarketDataFeed, timeframe_to_ms
from traderbot.models import CANDLE_COLUMNS

MIN = 60_000  # one minute in ms
T0 = 1_700_000_000_000 - (1_700_000_000_000 % (5 * MIN))  # aligned to a 5m boundary


class FakeExchange:
    """Mimics the few public ccxt methods MarketDataFeed uses."""

    def __init__(self, ohlcv=None, ticker=None, markets=None, now=None, error=None):
        self.ohlcv = ohlcv if ohlcv is not None else []
        self.ticker = ticker if ticker is not None else {"last": 100.0}
        self.markets = markets if markets is not None else {}
        self.now = now if now is not None else T0
        self.error = error              # exception raised by every data call
        self.ohlcv_calls: list[tuple] = []
        self.load_markets_calls = 0

    def fetch_ohlcv(self, symbol, timeframe, limit=None):
        self.ohlcv_calls.append((symbol, timeframe, limit))
        if self.error:
            raise self.error
        return [list(r) for r in self.ohlcv]

    def fetch_ticker(self, symbol):
        if self.error:
            raise self.error
        return dict(self.ticker)

    def load_markets(self):
        self.load_markets_calls += 1
        if self.error:
            raise self.error
        return self.markets

    def milliseconds(self):
        return self.now


def candle(ts, close=100.0):
    return [ts, close - 1, close + 2, close - 2, close, 10.0]


# --------------------------------------------------------------------------- timeframe_to_ms
@pytest.mark.parametrize(
    "tf, ms",
    [("1m", 60_000), ("5m", 300_000), ("15m", 900_000), ("30m", 1_800_000),
     ("1h", 3_600_000), ("4h", 14_400_000), ("1d", 86_400_000)],
)
def test_timeframe_to_ms(tf, ms):
    assert timeframe_to_ms(tf) == ms
    assert isinstance(timeframe_to_ms(tf), int)


@pytest.mark.parametrize("tf", ["abc", "5x", "", "0m"])
def test_timeframe_to_ms_invalid(tf):
    with pytest.raises(DataFeedError):
        timeframe_to_ms(tf)


# --------------------------------------------------------------------------- construction
def test_unknown_exchange_id():
    with pytest.raises(DataFeedError, match="Unknown exchange"):
        MarketDataFeed("definitely_not_an_exchange")


def test_base_class_name_is_not_an_exchange():
    with pytest.raises(DataFeedError):
        MarketDataFeed("Exchange")


def test_real_exchange_construction_is_offline_and_public():
    # Building a ccxt exchange object does not touch the network.
    feed = MarketDataFeed("binance")
    assert feed.exchange_id == "binance"
    ex = feed._exchange
    assert isinstance(ex, ccxt.binance)
    assert ex.enableRateLimit is True
    assert ex.timeout == 15000
    assert not ex.apiKey and not ex.secret


def test_injected_exchange_used():
    fake = FakeExchange()
    feed = MarketDataFeed("whatever", exchange=fake)
    assert feed.exchange_id == "whatever"
    assert feed._exchange is fake


# --------------------------------------------------------------------------- candles
def test_forming_candle_boundary_exact():
    tf = 5 * MIN
    rows = [candle(T0 - 3 * tf), candle(T0 - 2 * tf), candle(T0 - tf), candle(T0)]
    fake = FakeExchange(ohlcv=rows)
    feed = MarketDataFeed("fake", exchange=fake)

    # now == T0: candle T0-tf closes exactly at now -> CLOSED; candle T0 is forming.
    df = feed.fetch_closed_candles("BTC/USDT", "5m", limit=10, now_ms=T0)
    assert df["timestamp"].tolist() == [T0 - 3 * tf, T0 - 2 * tf, T0 - tf]

    # One ms earlier: candle T0-tf is still forming (ts + tf > now).
    df = feed.fetch_closed_candles("BTC/USDT", "5m", limit=10, now_ms=T0 - 1)
    assert df["timestamp"].tolist() == [T0 - 3 * tf, T0 - 2 * tf]

    # Exactly when candle T0 closes it becomes closed too.
    df = feed.fetch_closed_candles("BTC/USDT", "5m", limit=10, now_ms=T0 + tf)
    assert df["timestamp"].tolist() == [T0 - 3 * tf, T0 - 2 * tf, T0 - tf, T0]


def test_now_defaults_to_exchange_milliseconds():
    tf = 5 * MIN
    rows = [candle(T0 - tf), candle(T0)]
    fake = FakeExchange(ohlcv=rows, now=T0 + 10)
    feed = MarketDataFeed("fake", exchange=fake)
    df = feed.fetch_closed_candles("BTC/USDT", "5m", limit=5)
    assert df["timestamp"].tolist() == [T0 - tf]


def test_requests_limit_plus_one():
    fake = FakeExchange(ohlcv=[])
    feed = MarketDataFeed("fake", exchange=fake)
    feed.fetch_closed_candles("ETH/USDT", "1h", limit=50, now_ms=T0)
    assert fake.ohlcv_calls == [("ETH/USDT", "1h", 51)]


def test_duplicates_and_unsorted_rows():
    tf = MIN
    rows = [
        candle(T0 - 1 * tf, 103.0),
        candle(T0 - 3 * tf, 101.0),
        candle(T0 - 2 * tf, 102.0),
        candle(T0 - 3 * tf, 101.5),   # duplicate timestamp, newer data -> kept
        candle(T0 - 2 * tf, 102.0),
    ]
    feed = MarketDataFeed("fake", exchange=FakeExchange(ohlcv=rows))
    df = feed.fetch_closed_candles("BTC/USDT", "1m", limit=10, now_ms=T0)
    assert df["timestamp"].tolist() == [T0 - 3 * tf, T0 - 2 * tf, T0 - tf]
    assert df["timestamp"].is_unique
    assert df["close"].tolist() == [101.5, 102.0, 103.0]
    assert df.index.tolist() == [0, 1, 2]


def test_limit_trims_to_newest_rows_after_dropping_forming():
    tf = MIN
    rows = [candle(T0 - k * tf, 100.0 + k) for k in range(6, -1, -1)]  # T0-6tf .. T0 (T0 forming)
    feed = MarketDataFeed("fake", exchange=FakeExchange(ohlcv=rows))
    df = feed.fetch_closed_candles("BTC/USDT", "1m", limit=3, now_ms=T0)
    assert len(df) == 3
    assert df["timestamp"].tolist() == [T0 - 3 * tf, T0 - 2 * tf, T0 - tf]
    assert df.index.tolist() == [0, 1, 2]


def test_dtypes_and_columns():
    rows = [[T0 - MIN, 1, 2, 0, 1, 5], [T0 - 2 * MIN, "1.5", 2.5, 0.5, 2, None]]
    feed = MarketDataFeed("fake", exchange=FakeExchange(ohlcv=rows))
    df = feed.fetch_closed_candles("BTC/USDT", "1m", limit=5, now_ms=T0)
    assert list(df.columns) == CANDLE_COLUMNS
    assert df["timestamp"].dtype == "int64"
    for col in CANDLE_COLUMNS[1:]:
        assert df[col].dtype == "float64", col
    assert df["open"].tolist() == [1.5, 1.0]
    assert pd.isna(df["volume"].iloc[0])


def test_empty_response_gives_empty_typed_frame():
    feed = MarketDataFeed("fake", exchange=FakeExchange(ohlcv=[]))
    df = feed.fetch_closed_candles("BTC/USDT", "1m", limit=5, now_ms=T0)
    assert df.empty
    assert list(df.columns) == CANDLE_COLUMNS
    assert df["timestamp"].dtype == "int64"
    assert df["close"].dtype == "float64"


def test_all_forming_gives_empty():
    feed = MarketDataFeed("fake", exchange=FakeExchange(ohlcv=[candle(T0)]))
    df = feed.fetch_closed_candles("BTC/USDT", "5m", limit=5, now_ms=T0 + 1)
    assert df.empty


def test_invalid_limit():
    feed = MarketDataFeed("fake", exchange=FakeExchange())
    with pytest.raises(ValueError):
        feed.fetch_closed_candles("BTC/USDT", "1m", limit=0, now_ms=T0)


# --------------------------------------------------------------------------- last price
def test_last_price_uses_last():
    feed = MarketDataFeed("fake", exchange=FakeExchange(ticker={"last": 123.45, "close": 1.0}))
    assert feed.fetch_last_price("BTC/USDT") == 123.45


def test_last_price_falls_back_to_close():
    feed = MarketDataFeed("fake", exchange=FakeExchange(ticker={"last": None, "close": 99.5}))
    assert feed.fetch_last_price("BTC/USDT") == 99.5
    feed = MarketDataFeed("fake", exchange=FakeExchange(ticker={"close": 42.0}))
    assert feed.fetch_last_price("BTC/USDT") == 42.0


@pytest.mark.parametrize(
    "ticker",
    [{"last": 0.0}, {"last": -1.0}, {"last": None, "close": 0}, {"last": None, "close": None},
     {}, {"last": float("nan")}, {"last": "abc"}],
)
def test_last_price_invalid_raises(ticker):
    feed = MarketDataFeed("fake", exchange=FakeExchange(ticker=ticker))
    with pytest.raises(DataFeedError):
        feed.fetch_last_price("BTC/USDT")


def test_last_price_returns_float():
    feed = MarketDataFeed("fake", exchange=FakeExchange(ticker={"last": 7}))
    price = feed.fetch_last_price("BTC/USDT")
    assert price == 7.0 and isinstance(price, float)


# --------------------------------------------------------------------------- error mapping
NETWORK_ERRORS = [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
                  ccxt.DDoSProtection, ccxt.RateLimitExceeded]
OTHER_ERRORS = [ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.NotSupported, ccxt.BaseError]


def _call_all(feed: MarketDataFeed):
    return [
        lambda: feed.fetch_closed_candles("XXX/YYY", "1m", limit=5, now_ms=T0),
        lambda: feed.fetch_last_price("XXX/YYY"),
        lambda: feed.list_symbols(),
    ]


def test_bad_symbol_mapped():
    feed = MarketDataFeed("kraken", exchange=FakeExchange(error=ccxt.BadSymbol("binance does not have market symbol XXX/YYY")))
    for call in _call_all(feed)[:2]:
        with pytest.raises(DataFeedError) as info:
            call()
        assert "Unknown trading pair XXX/YYY on kraken" in str(info.value)
        assert isinstance(info.value.__cause__, ccxt.BadSymbol)
    with pytest.raises(DataFeedError, match="Unknown trading pair"):
        _call_all(feed)[2]()


@pytest.mark.parametrize("exc_class", NETWORK_ERRORS)
def test_network_errors_mapped(exc_class):
    feed = MarketDataFeed("kraken", exchange=FakeExchange(error=exc_class("timed out")))
    for call in _call_all(feed):
        with pytest.raises(DataFeedError) as info:
            call()
        assert str(info.value).startswith("Network problem")
        assert "kraken" in str(info.value)
        assert not isinstance(info.value, ccxt.BaseError)
        assert isinstance(info.value.__cause__, exc_class)


@pytest.mark.parametrize("exc_class", OTHER_ERRORS)
def test_other_ccxt_errors_mapped(exc_class):
    feed = MarketDataFeed("kraken", exchange=FakeExchange(error=exc_class("something odd")))
    for call in _call_all(feed):
        with pytest.raises(DataFeedError) as info:
            call()
        assert str(info.value) == "something odd"
        assert not isinstance(info.value, ccxt.BaseError)


def test_ccxt_error_with_empty_message_still_has_text():
    feed = MarketDataFeed("kraken", exchange=FakeExchange(error=ccxt.ExchangeError()))
    with pytest.raises(DataFeedError) as info:
        feed.fetch_last_price("BTC/USDT")
    assert str(info.value) == "ExchangeError"


# --------------------------------------------------------------------------- symbols
MARKETS = {
    "ETH/USDT": {"symbol": "ETH/USDT", "spot": True, "type": "spot", "active": True},
    "BTC/USDT": {"symbol": "BTC/USDT", "spot": True, "type": "spot", "active": True},
    "OLD/USDT": {"symbol": "OLD/USDT", "spot": True, "type": "spot", "active": False},
    "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "spot": False, "type": "swap", "active": True},
    "BTC-FUT": {"symbol": "BTC/USD:BTC-250101", "spot": False, "type": "future", "active": True},
    "ADA/BTC": {"symbol": "ADA/BTC", "spot": True, "type": "spot", "active": None},  # unknown -> kept
    "SOL/USDT": {"symbol": "SOL/USDT", "type": "spot", "active": True},  # no "spot" key, type says spot
}


def test_list_symbols_filters_and_sorts():
    feed = MarketDataFeed("fake", exchange=FakeExchange(markets=MARKETS))
    assert feed.list_symbols() == ["ADA/BTC", "BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_list_symbols_cached_after_success():
    fake = FakeExchange(markets=MARKETS)
    feed = MarketDataFeed("fake", exchange=fake)
    first = feed.list_symbols()
    first.append("MUTATED")  # caller mutating the result must not affect the cache
    second = feed.list_symbols()
    assert fake.load_markets_calls == 1
    assert "MUTATED" not in second


def test_list_symbols_not_cached_after_failure():
    fake = FakeExchange(markets=MARKETS, error=ccxt.RequestTimeout("slow"))
    feed = MarketDataFeed("fake", exchange=fake)
    with pytest.raises(DataFeedError):
        feed.list_symbols()
    fake.error = None
    assert feed.list_symbols() == ["ADA/BTC", "BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert fake.load_markets_calls == 2

"""Public market data (candles and prices) from a crypto exchange via ccxt.

Only PUBLIC endpoints are used here (markets, candles, ticker). This module
never uses API keys and never places orders: it only reads prices.
"""
from __future__ import annotations

import logging
import math
from typing import Any, NoReturn

import ccxt
import pandas as pd

from traderbot.models import CANDLE_COLUMNS

logger = logging.getLogger(__name__)

# The local PC clock can run a few seconds ahead of the exchange. The newest candle in a
# response is only trusted as closed once the exchange has started a newer candle, or once
# its close time is at least this long ago by the local clock.
CLOSE_GRACE_MS = 10_000


class DataFeedError(Exception):
    """A market-data problem with a friendly message that is safe to show in the GUI."""


def timeframe_to_ms(timeframe: str) -> int:
    """Convert a ccxt timeframe string such as "5m" or "1h" to milliseconds.

    Raises DataFeedError if the timeframe cannot be understood.
    """
    try:
        seconds = ccxt.Exchange.parse_timeframe(timeframe)
    except (ccxt.BaseError, ValueError, TypeError, AttributeError) as e:
        raise DataFeedError(f"Unsupported timeframe: {timeframe!r}") from e
    ms = int(seconds * 1000)
    if ms <= 0:
        raise DataFeedError(f"Unsupported timeframe: {timeframe!r}")
    return ms


class MarketDataFeed:
    """Reads public market data for one exchange.

    `exchange` lets tests inject a fake object that has the same methods as a
    ccxt exchange (load_markets, fetch_ohlcv, fetch_ticker, milliseconds).
    """

    def __init__(self, exchange_id: str, exchange: object | None = None) -> None:
        self.exchange_id: str = exchange_id
        self._symbols_cache: list[str] | None = None
        if exchange is not None:
            self._exchange: Any = exchange
            return
        # Only accept real exchange ids from ccxt's own list, so a typo (or a
        # name like "Exchange") can never resolve to something unexpected.
        if not isinstance(exchange_id, str) or exchange_id not in ccxt.exchanges:
            raise DataFeedError(f"Unknown exchange: {exchange_id!r}")
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise DataFeedError(f"Unknown exchange: {exchange_id!r}")
        try:
            # No API keys are ever passed: public data only.
            self._exchange = exchange_class({"enableRateLimit": True, "timeout": 15000})
        except ccxt.BaseError as e:
            raise DataFeedError(f"Could not set up exchange {exchange_id}: {e}") from e

    # ------------------------------------------------------------------ helpers
    def _raise_friendly(self, error: ccxt.BaseError, symbol: str | None = None) -> NoReturn:
        """Turn a ccxt exception into a DataFeedError with a readable message."""
        if isinstance(error, ccxt.BadSymbol):
            pair = symbol if symbol else "(unknown)"
            raise DataFeedError(f"Unknown trading pair {pair} on {self.exchange_id}") from error
        if isinstance(error, ccxt.NetworkError):
            raise DataFeedError(
                f"Network problem talking to {self.exchange_id}: {str(error) or type(error).__name__}"
            ) from error
        raise DataFeedError(str(error) or type(error).__name__) from error

    # ------------------------------------------------------------------ public API
    def list_symbols(self) -> list[str]:
        """Return the sorted symbols of active spot markets (cached after first success)."""
        if self._symbols_cache is not None:
            return list(self._symbols_cache)
        try:
            markets = self._exchange.load_markets()
        except ccxt.BaseError as e:
            self._raise_friendly(e)
        symbols: set[str] = set()
        for key, market in (markets or {}).items():
            if not isinstance(market, dict):
                continue
            # ccxt uses active=None when the exchange does not say; only an
            # explicit False means the market is closed.
            if market.get("active") is False:
                continue
            is_spot = market.get("spot") is True or market.get("type") == "spot"
            if not is_spot:
                continue
            symbol = market.get("symbol") or key
            if isinstance(symbol, str) and symbol:
                symbols.add(symbol)
        self._symbols_cache = sorted(symbols)
        logger.info("Loaded %d spot symbols from %s", len(self._symbols_cache), self.exchange_id)
        return list(self._symbols_cache)

    def fetch_closed_candles(
        self, symbol: str, timeframe: str, limit: int, now_ms: int | None = None
    ) -> pd.DataFrame:
        """Fetch up to `limit` CLOSED candles, oldest first.

        A candle that started at `timestamp` closes at `timestamp + timeframe_ms`.
        If that moment is later than `now_ms` the candle is still forming and is
        dropped, so strategies only ever see final prices. Because the local clock
        may run ahead of the exchange, the newest row is also dropped unless it closed
        at least CLOSE_GRACE_MS ago (rows followed by a newer row are certainly closed).
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        tf_ms = timeframe_to_ms(timeframe)
        timeframes = getattr(self._exchange, "timeframes", None)
        if isinstance(timeframes, dict) and timeframes and timeframe not in timeframes:
            raise DataFeedError(
                f"{self.exchange_id} does not offer {timeframe} candles. Choose another timeframe."
            )
        try:
            # Ask for one extra row because the newest one is usually still forming.
            rows = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit + 1)
            if now_ms is None:
                now_ms = int(self._exchange.milliseconds())
        except ccxt.BaseError as e:
            self._raise_friendly(e, symbol)

        # Keep exactly the six standard fields of each row.
        clean_rows = [list(row[:6]) for row in (rows or []) if row is not None and len(row) >= 6]
        df = pd.DataFrame(clean_rows, columns=CANDLE_COLUMNS)

        # Rows without a usable timestamp cannot be placed on the timeline.
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")
        for col in CANDLE_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        df = df.drop_duplicates(subset="timestamp", keep="last")
        df = df.sort_values("timestamp", kind="stable")
        newest_ts = df["timestamp"].iloc[-1] if not df.empty else None
        # Exactly at the close time the candle is final; any later close time = still forming.
        df = df[df["timestamp"] + tf_ms <= now_ms]
        if not df.empty and int(df["timestamp"].iloc[-1]) == int(newest_ts) \
                and int(newest_ts) + tf_ms + CLOSE_GRACE_MS > now_ms:
            # The newest row may still be forming if the local clock runs ahead.
            df = df.iloc[:-1]
        df = df.tail(limit).reset_index(drop=True)
        return df

    def fetch_last_price(self, symbol: str) -> float:
        """Return the latest traded price for `symbol` (falls back to the ticker's close)."""
        try:
            ticker = self._exchange.fetch_ticker(symbol)
        except ccxt.BaseError as e:
            self._raise_friendly(e, symbol)
        ticker = ticker or {}
        price = ticker.get("last")
        if price is None:
            price = ticker.get("close")
        try:
            value = float(price)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            raise DataFeedError(f"No valid price for {symbol} on {self.exchange_id}")
        return value

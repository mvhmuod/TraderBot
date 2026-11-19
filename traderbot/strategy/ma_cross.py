"""Simple moving-average (SMA) crossover strategy.

Idea: a "fast" average follows the price closely, a "slow" average lags behind.
- When the fast average crosses ABOVE the slow one -> BUY signal.
- When the fast average crosses BELOW the slow one -> SELL signal.
Signals are only evaluated on CLOSED candles, so they never flicker.
"""
from __future__ import annotations

import math
import numbers

import pandas as pd

from traderbot.models import Signal


class MACrossoverStrategy:
    """Generates BUY/SELL/HOLD from a fast and a slow simple moving average of the close."""

    def __init__(self, fast_period: int, slow_period: int) -> None:
        for value in (fast_period, slow_period):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError("MA periods must be whole numbers")
        if not (1 <= fast_period < slow_period):
            raise ValueError("Need 1 <= fast_period < slow_period")
        self.fast_period: int = int(fast_period)
        self.slow_period: int = int(slow_period)

    @property
    def min_candles(self) -> int:
        """Candles needed to compare two consecutive slow-MA values."""
        return self.slow_period + 1

    def compute(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return a COPY of `candles` with "fast_ma" and "slow_ma" columns added.

        Values are NaN until enough candles exist for the average.
        """
        out = candles.copy()
        close = out["close"].astype("float64")
        out["fast_ma"] = close.rolling(self.fast_period, min_periods=self.fast_period).mean()
        out["slow_ma"] = close.rolling(self.slow_period, min_periods=self.slow_period).mean()
        return out

    def signal(self, candles_with_ma: pd.DataFrame) -> Signal:
        """Look at the last two (closed) candles and report a crossover, if any."""
        if "fast_ma" not in candles_with_ma.columns or "slow_ma" not in candles_with_ma.columns:
            raise ValueError("candles must come from compute() (fast_ma/slow_ma missing)")
        if len(candles_with_ma) < self.min_candles:
            return Signal.HOLD

        prev_fast = float(candles_with_ma["fast_ma"].iloc[-2])
        prev_slow = float(candles_with_ma["slow_ma"].iloc[-2])
        curr_fast = float(candles_with_ma["fast_ma"].iloc[-1])
        curr_slow = float(candles_with_ma["slow_ma"].iloc[-1])
        if any(math.isnan(v) for v in (prev_fast, prev_slow, curr_fast, curr_slow)):
            return Signal.HOLD

        # "<=" / ">=" on the previous candle: if the averages were touching and
        # now separate, that still counts as a cross.
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return Signal.BUY
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return Signal.SELL
        return Signal.HOLD

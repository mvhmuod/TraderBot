"""Tests for traderbot.strategy.ma_cross."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from traderbot.models import CANDLE_COLUMNS, Signal
from traderbot.strategy.ma_cross import MACrossoverStrategy


def make_candles(closes: list[float]) -> pd.DataFrame:
    """Build a candle frame with the given closes (other columns are filler)."""
    n = len(closes)
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype="int64") * 60_000,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1.0] * n,
        },
        columns=CANDLE_COLUMNS,
    )
    df["close"] = df["close"].astype("float64")
    return df


def signal_for(fast: int, slow: int, closes: list[float]) -> Signal:
    strat = MACrossoverStrategy(fast, slow)
    return strat.signal(strat.compute(make_candles(closes)))


# --------------------------------------------------------------------------- construction
def test_valid_construction():
    s = MACrossoverStrategy(9, 21)
    assert s.fast_period == 9 and s.slow_period == 21
    assert s.min_candles == 22
    assert MACrossoverStrategy(1, 2).min_candles == 3


@pytest.mark.parametrize("fast, slow", [(0, 5), (5, 5), (6, 5), (-1, 3), (2.5, 5), (2, 5.0), (True, 5)])
def test_invalid_construction(fast, slow):
    with pytest.raises(ValueError):
        MACrossoverStrategy(fast, slow)


# --------------------------------------------------------------------------- compute
def test_compute_values_and_nans():
    strat = MACrossoverStrategy(2, 3)
    out = strat.compute(make_candles([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert math.isnan(out["fast_ma"].iloc[0])
    assert out["fast_ma"].iloc[1:].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert math.isnan(out["slow_ma"].iloc[0]) and math.isnan(out["slow_ma"].iloc[1])
    assert out["slow_ma"].iloc[2:].tolist() == [2.0, 3.0, 4.0]
    assert list(out.columns) == CANDLE_COLUMNS + ["fast_ma", "slow_ma"]


def test_compute_does_not_mutate_input():
    candles = make_candles([1.0, 2.0, 3.0, 4.0])
    before = candles.copy(deep=True)
    out = MACrossoverStrategy(2, 3).compute(candles)
    assert out is not candles
    assert list(candles.columns) == CANDLE_COLUMNS
    pd.testing.assert_frame_equal(candles, before)
    out.loc[0, "close"] = 999.0  # changing the result must not touch the input
    assert candles.loc[0, "close"] == 1.0


def test_compute_nan_in_window_gives_nan():
    out = MACrossoverStrategy(2, 3).compute(make_candles([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]))
    # Any window containing the NaN close is NaN (min_periods = full window).
    assert math.isnan(out["fast_ma"].iloc[2]) and math.isnan(out["fast_ma"].iloc[3])
    assert out["fast_ma"].iloc[4] == 4.5
    assert math.isnan(out["slow_ma"].iloc[4])
    assert out["slow_ma"].iloc[5] == 5.0


def test_compute_empty_frame():
    out = MACrossoverStrategy(2, 3).compute(make_candles([]))
    assert out.empty
    assert "fast_ma" in out.columns and "slow_ma" in out.columns
    assert MACrossoverStrategy(2, 3).signal(out) == Signal.HOLD


# --------------------------------------------------------------------------- signals
def test_buy_crossover():
    # fast(2): 8.5 -> 10.0 ; slow(3): 9.0 -> 9.667  => fast crosses above
    assert signal_for(2, 3, [10.0, 9.0, 8.0, 12.0]) == Signal.BUY


def test_sell_crossover():
    # fast(2): 9.5 -> 8.0 ; slow(3): 9.0 -> 8.333  => fast crosses below
    assert signal_for(2, 3, [8.0, 9.0, 10.0, 6.0]) == Signal.SELL


def test_only_last_two_rows_matter():
    # An old crossover earlier in the series must not trigger a signal now.
    assert signal_for(2, 3, [10.0, 9.0, 8.0, 12.0, 13.0]) == Signal.HOLD


def test_prev_equal_then_cross_up_is_buy():
    # fast(1)=close, slow(2)=avg of last two. prev: 5 == 5 ; curr: 7 > 6
    assert signal_for(1, 2, [5.0, 5.0, 5.0, 7.0]) == Signal.BUY


def test_prev_equal_then_cross_down_is_sell():
    # prev: 5 == 5 ; curr: 3 < 4
    assert signal_for(1, 2, [5.0, 5.0, 5.0, 3.0]) == Signal.SELL


def test_touch_from_below_without_cross_is_hold():
    # prev: fast 4 < slow 5 ; curr: fast 4 == slow 4  (touch, no cross)
    assert signal_for(1, 2, [6.0, 4.0, 4.0]) == Signal.HOLD


def test_touch_from_above_without_cross_is_hold():
    # prev: fast 6 > slow 5 ; curr: fast 6 == slow 6
    assert signal_for(1, 2, [4.0, 6.0, 6.0]) == Signal.HOLD


def test_equal_both_rows_is_hold():
    assert signal_for(1, 2, [5.0, 5.0, 5.0]) == Signal.HOLD


def test_staying_above_or_below_is_hold():
    assert signal_for(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0]) == Signal.HOLD
    assert signal_for(2, 3, [5.0, 4.0, 3.0, 2.0, 1.0]) == Signal.HOLD


def test_insufficient_data_is_hold():
    strat = MACrossoverStrategy(2, 3)
    # 3 rows < min_candles (4)
    assert strat.signal(strat.compute(make_candles([10.0, 9.0, 12.0]))) == Signal.HOLD
    # Even with hand-made non-NaN MAs, too few rows -> HOLD
    df = make_candles([1.0, 2.0, 3.0])
    df["fast_ma"] = [0.0, 0.0, 10.0]
    df["slow_ma"] = [1.0, 1.0, 1.0]
    assert strat.signal(df) == Signal.HOLD


def test_nan_in_last_rows_is_hold():
    # A NaN close in the newest candle makes the current MAs NaN.
    assert signal_for(2, 3, [10.0, 9.0, 8.0, float("nan")]) == Signal.HOLD
    strat = MACrossoverStrategy(1, 2)
    base = make_candles([5.0, 5.0, 5.0, 7.0])
    for col, row in [("fast_ma", -1), ("slow_ma", -1), ("fast_ma", -2), ("slow_ma", -2)]:
        df = strat.compute(base)
        assert strat.signal(df) == Signal.BUY
        df.loc[df.index[row], col] = float("nan")
        assert strat.signal(df) == Signal.HOLD, (col, row)


def test_signal_requires_ma_columns():
    with pytest.raises(ValueError):
        MACrossoverStrategy(1, 2).signal(make_candles([1.0, 2.0, 3.0]))


def test_signal_does_not_mutate_input():
    strat = MACrossoverStrategy(2, 3)
    df = strat.compute(make_candles([10.0, 9.0, 8.0, 12.0]))
    before = df.copy(deep=True)
    strat.signal(df)
    pd.testing.assert_frame_equal(df, before)

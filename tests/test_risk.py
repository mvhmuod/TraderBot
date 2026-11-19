"""Tests for traderbot.risk."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from traderbot.config import MIN_ORDER_QUOTE, BotConfig
from traderbot.models import Position, TradeReason
from traderbot.risk import RiskManager, utc_day_key


def risk(**overrides) -> RiskManager:
    return RiskManager(BotConfig(**overrides))


def pos(avg_cost: float = 100.0) -> Position:
    return Position(symbol="BTC/USDT", amount=0.5, avg_cost=avg_cost, opened_at=0)


def ms(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


# --------------------------------------------------------------------------- sizing
def test_size_is_percent_of_equity():
    assert risk(position_size_pct=10.0).order_quote_amount(cash=10_000.0, equity=10_000.0) == 1_000.0


def test_size_capped_by_cash():
    assert risk(position_size_pct=50.0).order_quote_amount(cash=300.0, equity=10_000.0) == 300.0
    assert risk(position_size_pct=100.0).order_quote_amount(cash=9_999.5, equity=10_000.0) == 9_999.5


def test_size_below_minimum_returns_zero():
    r = risk(position_size_pct=1.0)
    assert r.order_quote_amount(cash=10_000.0, equity=50.0) == 0.0           # 0.5 < 1.0
    assert r.order_quote_amount(cash=0.5, equity=10_000.0) == 0.0            # capped by cash -> 0.5
    assert r.order_quote_amount(cash=0.0, equity=10_000.0) == 0.0
    assert r.order_quote_amount(cash=-5.0, equity=10_000.0) == 0.0


def test_size_exactly_minimum_is_allowed():
    r = risk(position_size_pct=100.0)
    assert r.order_quote_amount(cash=MIN_ORDER_QUOTE, equity=500.0) == MIN_ORDER_QUOTE
    assert r.order_quote_amount(cash=MIN_ORDER_QUOTE - 1e-6, equity=500.0) == 0.0


def test_reads_config_changes_live():
    cfg = BotConfig(position_size_pct=10.0)
    r = RiskManager(cfg)
    cfg.position_size_pct = 20.0
    assert r.order_quote_amount(cash=10_000.0, equity=10_000.0) == 2_000.0


# --------------------------------------------------------------------------- exits
def test_exits_disabled_by_default():
    r = risk(stop_loss_pct=0.0, take_profit_pct=0.0)
    for price in (0.01, 50.0, 100.0, 1_000_000.0):
        assert r.exit_reason(pos(100.0), price) is None


def test_stop_loss_boundary():
    r = risk(stop_loss_pct=5.0)
    threshold = 100.0 * (1 - 5.0 / 100.0)
    assert r.exit_reason(pos(100.0), threshold) == TradeReason.STOP_LOSS        # exactly at -> hit
    assert r.exit_reason(pos(100.0), threshold - 0.01) == TradeReason.STOP_LOSS
    assert r.exit_reason(pos(100.0), threshold + 0.01) is None
    assert r.exit_reason(pos(100.0), 500.0) is None                            # TP disabled


def test_take_profit_boundary():
    r = risk(take_profit_pct=10.0)
    threshold = 100.0 * (1 + 10.0 / 100.0)
    assert r.exit_reason(pos(100.0), threshold) == TradeReason.TAKE_PROFIT     # exactly at -> hit
    assert r.exit_reason(pos(100.0), threshold + 1.0) == TradeReason.TAKE_PROFIT
    assert r.exit_reason(pos(100.0), threshold - 0.01) is None
    assert r.exit_reason(pos(100.0), 1.0) is None                              # SL disabled


def test_both_enabled_in_between_is_none():
    r = risk(stop_loss_pct=5.0, take_profit_pct=10.0)
    assert r.exit_reason(pos(100.0), 100.0) is None
    assert r.exit_reason(pos(100.0), 94.0) == TradeReason.STOP_LOSS
    assert r.exit_reason(pos(100.0), 111.0) == TradeReason.TAKE_PROFIT


def test_stop_loss_has_priority_over_take_profit():
    # Not reachable with valid configs and a positive avg_cost, so force both
    # conditions true with a zero cost basis: price 0 <= 0 and price 0 >= 0.
    r = risk(stop_loss_pct=5.0, take_profit_pct=10.0)
    assert r.exit_reason(pos(0.0), 0.0) == TradeReason.STOP_LOSS
    # Take-profit alone on the same input does fire, proving the order matters.
    assert risk(take_profit_pct=10.0).exit_reason(pos(0.0), 0.0) == TradeReason.TAKE_PROFIT


# --------------------------------------------------------------------------- daily limit
def test_daily_limit_disabled_when_zero():
    r = risk(max_daily_loss_pct=0.0)
    assert r.daily_limit_hit(10_000.0, 0.0) is False
    assert r.daily_limit_hit(10_000.0, 1.0) is False


def test_daily_limit_boundary_exactly_equal_counts():
    r = risk(max_daily_loss_pct=5.0)
    limit = 10_000.0 * (1 - 5.0 / 100.0)
    assert limit == 9_500.0
    assert r.daily_limit_hit(10_000.0, 9_500.0) is True          # exactly at the limit -> hit
    assert r.daily_limit_hit(10_000.0, 9_499.99) is True
    assert r.daily_limit_hit(10_000.0, 9_500.01) is False
    assert r.daily_limit_hit(10_000.0, 12_000.0) is False


def test_daily_limit_odd_numbers_boundary():
    r = risk(max_daily_loss_pct=3.3)
    start = 1234.56
    limit = start * (1 - 3.3 / 100.0)
    assert r.daily_limit_hit(start, limit) is True
    assert r.daily_limit_hit(start, limit + 1e-6) is False


def test_daily_limit_needs_positive_day_start():
    r = risk(max_daily_loss_pct=5.0)
    assert r.daily_limit_hit(0.0, -100.0) is False
    assert r.daily_limit_hit(-10.0, -100.0) is False


def test_daily_limit_returns_bool():
    assert isinstance(risk(max_daily_loss_pct=5.0).daily_limit_hit(100.0, 100.0), bool)
    assert isinstance(risk(max_daily_loss_pct=0.0).daily_limit_hit(100.0, 100.0), bool)


# --------------------------------------------------------------------------- utc_day_key
def test_utc_day_key_format():
    assert utc_day_key(ms(2024, 5, 1, 12, 30)) == "day_start:2024-05-01"
    assert utc_day_key(0) == "day_start:1970-01-01"


@pytest.mark.parametrize(
    "ts, expected",
    [
        (ms(2024, 5, 1), "day_start:2024-05-01"),                 # exactly midnight UTC
        (ms(2024, 5, 1) - 1, "day_start:2024-04-30"),             # 1 ms before midnight
        (ms(2024, 5, 1) + 1, "day_start:2024-05-01"),
        (ms(2024, 12, 31, 23, 59, 59) + 999, "day_start:2024-12-31"),
        (ms(2025, 1, 1), "day_start:2025-01-01"),                 # year rollover
        (ms(2024, 2, 29, 23, 59, 59), "day_start:2024-02-29"),    # leap day
        (ms(2024, 3, 1), "day_start:2024-03-01"),
        (-1, "day_start:1969-12-31"),
    ],
)
def test_utc_day_key_midnight_boundaries(ts, expected):
    assert utc_day_key(ts) == expected

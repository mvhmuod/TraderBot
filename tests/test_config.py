"""Tests for BotConfig: validate() (one test per rule) and to_dict()/from_dict()."""
from __future__ import annotations

import math
from dataclasses import fields

import pytest

from traderbot.config import (
    DEFAULT_DB_PATH, MAX_MA_PERIOD, MIN_ORDER_QUOTE, MIN_POLL_SECONDS, PROJECT_ROOT,
    SUPPORTED_EXCHANGES, SUPPORTED_TIMEFRAMES, BotConfig,
)


def errors_for(**changes: object) -> list[str]:
    """Validation errors of the default config with some fields changed."""
    return BotConfig(**changes).validate()


def has_error(errors: list[str], text: str) -> bool:
    return any(text in error for error in errors)


# ---------------------------------------------------------------------- defaults
def test_defaults_are_valid() -> None:
    assert BotConfig().validate() == []


def test_default_db_path_is_inside_project_data_folder() -> None:
    assert DEFAULT_DB_PATH == PROJECT_ROOT / "data" / "traderbot.db"


def test_every_supported_exchange_and_timeframe_is_valid() -> None:
    for exchange in SUPPORTED_EXCHANGES:
        assert errors_for(exchange_id=exchange) == []
    for timeframe in SUPPORTED_TIMEFRAMES:
        assert errors_for(timeframe=timeframe) == []


# ---------------------------------------------------------------------- validate(): each rule
def test_unsupported_exchange() -> None:
    errors = errors_for(exchange_id="notanexchange")
    assert errors == ["Unsupported exchange: 'notanexchange'"]


@pytest.mark.parametrize("symbol", ["", "BTCUSDT", "btc-usdt"])
def test_symbol_must_contain_slash(symbol: str) -> None:
    errors = errors_for(symbol=symbol)
    assert errors == ["Trading pair must look like BASE/QUOTE, e.g. BTC/USDT"]


def test_symbol_with_slash_is_accepted() -> None:
    assert errors_for(symbol="ETH/BTC") == []


@pytest.mark.parametrize("timeframe", ["2m", "1w", "", "5M"])
def test_unsupported_timeframe(timeframe: str) -> None:
    assert errors_for(timeframe=timeframe) == [f"Unsupported timeframe: {timeframe!r}"]


def test_fast_period_range() -> None:
    assert has_error(errors_for(fast_period=0), "Fast MA length must be between 1 and")
    assert errors_for(fast_period=1) == []
    # Upper bound (slow must also be raised so only the range rule is tested).
    too_big = errors_for(fast_period=MAX_MA_PERIOD + 1, slow_period=MAX_MA_PERIOD,
                         history_limit=MAX_MA_PERIOD + 2)
    assert has_error(too_big, f"Fast MA length must be between 1 and {MAX_MA_PERIOD}")


def test_slow_period_range() -> None:
    assert has_error(errors_for(fast_period=1, slow_period=1),
                     f"Slow MA length must be between 2 and {MAX_MA_PERIOD}")
    assert errors_for(fast_period=1, slow_period=2) == []
    assert errors_for(fast_period=1, slow_period=MAX_MA_PERIOD,
                      history_limit=MAX_MA_PERIOD + 2) == []
    assert has_error(errors_for(slow_period=MAX_MA_PERIOD + 1, history_limit=1000),
                     f"Slow MA length must be between 2 and {MAX_MA_PERIOD}")


@pytest.mark.parametrize("fast, slow", [(21, 21), (30, 21)])
def test_fast_must_be_smaller_than_slow(fast: int, slow: int) -> None:
    errors = errors_for(fast_period=fast, slow_period=slow)
    assert errors == ["Fast MA length must be smaller than slow MA length"]


def test_starting_balance_minimum() -> None:
    assert errors_for(starting_balance=MIN_ORDER_QUOTE) == []
    assert errors_for(starting_balance=MIN_ORDER_QUOTE - 0.01) == [
        f"Starting balance must be at least {MIN_ORDER_QUOTE}"]
    assert has_error(errors_for(starting_balance=math.nan), "Starting balance")


@pytest.mark.parametrize("field_name, label", [("fee_rate", "Fee rate"),
                                               ("slippage_rate", "Slippage rate")])
def test_cost_rates_between_0_and_10_percent(field_name: str, label: str) -> None:
    assert errors_for(**{field_name: 0.0}) == []
    assert errors_for(**{field_name: 0.0999}) == []
    expected = [f"{label} must be between 0 and 10%"]
    assert errors_for(**{field_name: 0.1}) == expected
    assert errors_for(**{field_name: -0.0001}) == expected
    assert errors_for(**{field_name: math.nan}) == expected


def test_position_size_pct() -> None:
    assert errors_for(position_size_pct=100.0) == []
    assert errors_for(position_size_pct=0.01) == []
    expected = ["Position size must be above 0% and at most 100%"]
    assert errors_for(position_size_pct=0.0) == expected
    assert errors_for(position_size_pct=100.01) == expected
    assert errors_for(position_size_pct=-5.0) == expected


def test_stop_loss_pct() -> None:
    assert errors_for(stop_loss_pct=0.0) == []
    assert errors_for(stop_loss_pct=99.99) == []
    expected = ["Stop-loss must be between 0% (off) and 100%"]
    assert errors_for(stop_loss_pct=100.0) == expected
    assert errors_for(stop_loss_pct=-1.0) == expected


def test_take_profit_pct() -> None:
    assert errors_for(take_profit_pct=0.0) == []
    assert errors_for(take_profit_pct=1000.0) == []
    expected = ["Take-profit must be between 0% (off) and 1000%"]
    assert errors_for(take_profit_pct=1000.01) == expected
    assert errors_for(take_profit_pct=-1.0) == expected


def test_daily_loss_pct() -> None:
    assert errors_for(max_daily_loss_pct=0.0) == []
    assert errors_for(max_daily_loss_pct=99.99) == []
    expected = ["Daily loss limit must be between 0% (off) and 100%"]
    assert errors_for(max_daily_loss_pct=100.0) == expected
    assert errors_for(max_daily_loss_pct=-0.5) == expected


def test_poll_interval_minimum() -> None:
    assert errors_for(poll_interval_sec=MIN_POLL_SECONDS) == []
    assert errors_for(poll_interval_sec=MIN_POLL_SECONDS - 1) == [
        f"Poll interval must be at least {MIN_POLL_SECONDS} seconds"]


def test_history_limit_covers_slow_ma() -> None:
    # Default slow_period is 21 -> history_limit must be at least 23.
    assert errors_for(history_limit=23) == []
    assert errors_for(history_limit=22) == ["History limit must be at least slow MA length + 2"]


def test_several_problems_are_all_reported() -> None:
    errors = errors_for(exchange_id="nope", symbol="BTCUSDT", poll_interval_sec=1)
    assert len(errors) == 3


# ---------------------------------------------------------------------- to_dict / from_dict
def test_to_dict_has_every_field() -> None:
    data = BotConfig().to_dict()
    assert set(data) == {f.name for f in fields(BotConfig)}
    assert data["fee_rate"] == 0.001


def test_round_trip() -> None:
    original = BotConfig(exchange_id="kraken", symbol="ETH/EUR", timeframe="1h", fast_period=5,
                         slow_period=50, starting_balance=2_500.0, fee_rate=0.0026,
                         slippage_rate=0.001, position_size_pct=25.0, stop_loss_pct=3.5,
                         take_profit_pct=7.0, max_daily_loss_pct=4.0, poll_interval_sec=30,
                         history_limit=400)
    assert BotConfig.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("data", [None, {}])
def test_from_empty_gives_defaults(data: dict | None) -> None:
    assert BotConfig.from_dict(data) == BotConfig()  # type: ignore[arg-type]


def test_unknown_keys_are_ignored() -> None:
    config = BotConfig.from_dict({"symbol": "ETH/USDT", "api_key": "x", "something_new": 1})
    assert config == BotConfig(symbol="ETH/USDT")
    assert not hasattr(config, "api_key")


def test_missing_keys_keep_defaults() -> None:
    config = BotConfig.from_dict({"fast_period": 4})
    assert config.fast_period == 4
    assert config.slow_period == BotConfig().slow_period
    assert config.exchange_id == BotConfig().exchange_id


def test_bad_values_fall_back_to_defaults() -> None:
    defaults = BotConfig()
    config = BotConfig.from_dict({
        "fee_rate": "not a number",       # ValueError
        "starting_balance": None,          # TypeError
        "fast_period": "5.5",              # int("5.5") -> ValueError
        "slow_period": [1, 2],             # TypeError
        "poll_interval_sec": {"a": 1},     # TypeError
    })
    assert config == defaults


def test_values_are_converted_to_the_field_type() -> None:
    config = BotConfig.from_dict({"fast_period": "7", "slow_period": 30.0, "fee_rate": "0.002",
                                  "poll_interval_sec": "20", "position_size_pct": 15})
    assert config.fast_period == 7 and isinstance(config.fast_period, int)
    assert config.slow_period == 30 and isinstance(config.slow_period, int)
    assert config.fee_rate == 0.002 and isinstance(config.fee_rate, float)
    assert config.poll_interval_sec == 20
    assert config.position_size_pct == 15.0 and isinstance(config.position_size_pct, float)


def test_out_of_range_saved_values_are_loaded_but_fail_validation() -> None:
    # from_dict only converts types; range checks are validate()'s job.
    config = BotConfig.from_dict({"poll_interval_sec": 1})
    assert config.poll_interval_sec == 1
    assert config.validate() == [f"Poll interval must be at least {MIN_POLL_SECONDS} seconds"]

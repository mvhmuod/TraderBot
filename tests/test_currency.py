"""Tests for the paper account currency (no network).

The account's cash is in one currency (e.g. USDT). Starting the bot on a pair priced in
another currency (e.g. ETH/BTC) is refused until the account is reset in that currency.
"""
from __future__ import annotations

import pytest

from tests.test_engine import SYM, Clock, FakeFeed, at, make_config
from traderbot.broker.paper import PaperBroker
from traderbot.config import quote_currency
from traderbot.engine import BotEngine, CurrencyMismatchError, EngineError
from traderbot.models import TradeReason
from traderbot.storage import ACCOUNT_CURRENCY_KEY, Storage

CLOSES = [10.0] * 5


@pytest.mark.parametrize("symbol, expected", [
    ("BTC/USDT", "USDT"),
    ("ETH/BTC", "BTC"),
    ("btc/usd", "USD"),
    ("BTC/USDT:USDT", "USDT"),
    ("  SOL/EUR ", "EUR"),
    ("BTCUSDT", ""),
    ("", ""),
])
def test_quote_currency(symbol: str, expected: str) -> None:
    assert quote_currency(symbol) == expected


# ---------------------------------------------------------------------- storage / broker
def test_reset_account_sets_and_clears_currency() -> None:
    storage = Storage(":memory:")
    storage.reset_account(500.0, "usdt")
    assert storage.get_meta(ACCOUNT_CURRENCY_KEY) == "USDT"
    storage.reset_account(500.0, "BTC")
    assert storage.get_meta(ACCOUNT_CURRENCY_KEY) == "BTC"
    storage.reset_account(500.0)                    # no currency -> decided on next start
    assert storage.get_meta(ACCOUNT_CURRENCY_KEY) is None
    storage.close()


def test_reset_account_currency_is_atomic() -> None:
    # If writing the new currency fails, the whole reset is rolled back.
    storage = Storage(":memory:")
    storage.reset_account(500.0, "USDT")
    storage._conn.execute(
        "CREATE TRIGGER fail_meta BEFORE UPDATE ON meta WHEN NEW.key = 'account_currency' "
        "BEGIN SELECT RAISE(ABORT, 'boom'); END")
    with pytest.raises(Exception):
        storage.reset_account(999.0, "BTC")
    assert storage.get_meta(ACCOUNT_CURRENCY_KEY) == "USDT"
    assert storage.load_account() == (500.0, 500.0)
    storage.close()


def test_broker_currency_persists_across_reopen(tmp_path) -> None:
    db = tmp_path / "acct.db"
    storage = Storage(db)
    broker = PaperBroker(storage, 0.001, 0.0005, 10_000.0)
    assert broker.currency is None                  # new account: not decided yet
    broker.set_currency("usdt")
    assert broker.currency == "USDT"
    storage.close()

    storage = Storage(db)
    broker = PaperBroker(storage, 0.001, 0.0005, 10_000.0)
    assert broker.currency == "USDT"
    broker.reset(2_000.0, "BTC")
    assert broker.currency == "BTC" and broker.cash == 2_000.0
    with pytest.raises(ValueError):
        broker.set_currency("  ")
    storage.close()


# ---------------------------------------------------------------------- engine.start()
def make_engine(symbol: str = SYM, **overrides):
    storage = Storage(":memory:")
    broker = PaperBroker(storage, fee_rate=0.001, slippage_rate=0.0005, starting_balance=10_000.0)
    engine = BotEngine(make_config(symbol=symbol, **overrides), FakeFeed(CLOSES),
                       broker, storage, clock=Clock(at(len(CLOSES))))
    return engine, broker, storage


def test_first_start_adopts_pair_currency() -> None:
    engine, broker, storage = make_engine("BTC/USDT")
    messages = engine.start()
    assert broker.currency == "USDT"
    assert "Paper account currency set to USDT." in messages
    # Later starts with the same currency do not announce it again.
    assert not any("currency set" in m for m in engine.start())
    storage.close()


def test_start_refused_for_pair_in_other_currency() -> None:
    engine, broker, storage = make_engine("ETH/BTC")
    broker.set_currency("USDT")
    with pytest.raises(CurrencyMismatchError) as info:
        engine.start()
    err = info.value
    assert isinstance(err, EngineError)
    assert (err.account_currency, err.pair_currency, err.symbol) == ("USDT", "BTC", "ETH/BTC")
    assert "This paper account is in USDT, but ETH/BTC is priced in BTC" in str(err)
    assert broker.currency == "USDT"                # nothing changed
    storage.close()


def test_start_allowed_for_other_pair_in_same_currency() -> None:
    engine, broker, storage = make_engine("ETH/USDT")
    broker.set_currency("USDT")
    engine.start()
    assert broker.currency == "USDT"
    storage.close()


def test_derivative_style_symbol_uses_settle_free_quote() -> None:
    engine, broker, storage = make_engine("BTC/USDT:USDT")
    broker.set_currency("USDT")
    engine.start()                                  # quote is USDT, not "USDT:USDT"
    storage.close()


def test_old_account_without_currency_follows_open_positions() -> None:
    # An account from before this feature: no currency recorded, an open ETH/BTC position.
    engine, broker, storage = make_engine("BTC/USDT")
    broker.market_buy("ETH/BTC", 100.0, 0.05, 1_700_000_000_000, TradeReason.SIGNAL)
    with pytest.raises(CurrencyMismatchError) as info:
        engine.start()
    assert (info.value.account_currency, info.value.pair_currency) == ("BTC", "USDT")
    assert broker.currency is None                  # not adopted on a refused start
    storage.close()


def test_old_account_with_positions_in_two_currencies_must_reset() -> None:
    engine, broker, storage = make_engine("BTC/USDT")
    broker.market_buy("ETH/BTC", 100.0, 0.05, 1_700_000_000_000, TradeReason.SIGNAL)
    broker.market_buy("ETH/USDT", 100.0, 2000.0, 1_700_000_000_000, TradeReason.SIGNAL)
    with pytest.raises(CurrencyMismatchError) as info:
        engine.start()
    assert info.value.account_currency == "BTC, USDT" and info.value.mixed
    text = str(info.value)
    assert "more than one currency (BTC, USDT)" in text
    assert "Reset the paper account to start fresh in USDT" in text
    assert "Choose a pair" not in text      # no pair could work; only a reset helps
    storage.close()


def test_old_account_with_closed_trades_in_other_currency_is_not_relabelled() -> None:
    # Older account: bought and sold ETH/BTC (cash is in BTC), nothing open now.
    engine, broker, storage = make_engine("BTC/USDT")
    broker.market_buy("ETH/BTC", 100.0, 0.05, 1_700_000_000_000, TradeReason.SIGNAL)
    broker.close_position("ETH/BTC", 0.05, 1_700_000_100_000, TradeReason.SIGNAL)
    assert broker.positions() == {}
    with pytest.raises(CurrencyMismatchError) as info:
        engine.start()
    assert (info.value.account_currency, info.value.pair_currency, info.value.mixed) == ("BTC", "USDT", False)
    assert broker.currency is None
    storage.close()


def test_old_account_with_trades_in_two_currencies_is_mixed() -> None:
    engine, broker, storage = make_engine("BTC/USDT")
    broker.market_buy("ETH/BTC", 100.0, 0.05, 1_700_000_000_000, TradeReason.SIGNAL)
    broker.close_position("ETH/BTC", 0.05, 1_700_000_100_000, TradeReason.SIGNAL)
    broker.market_buy("ETH/USDT", 100.0, 2000.0, 1_700_000_200_000, TradeReason.SIGNAL)
    with pytest.raises(CurrencyMismatchError) as info:
        engine.start()
    assert info.value.mixed
    storage.close()


def test_old_account_with_same_currency_history_adopts_it() -> None:
    engine, broker, storage = make_engine("ETH/USDT")
    broker.market_buy("BTC/USDT", 100.0, 60_000.0, 1_700_000_000_000, TradeReason.SIGNAL)
    broker.close_position("BTC/USDT", 60_000.0, 1_700_000_100_000, TradeReason.SIGNAL)
    engine.start()
    assert broker.currency == "USDT"
    storage.close()


def test_trade_symbols() -> None:
    storage = Storage(":memory:")
    broker = PaperBroker(storage, 0.0, 0.0, 10_000.0)
    assert storage.trade_symbols() == []
    broker.market_buy("ETH/USDT", 100.0, 2000.0, 1, TradeReason.SIGNAL)
    broker.market_buy("BTC/USDT", 100.0, 60_000.0, 2, TradeReason.SIGNAL)
    broker.market_buy("ETH/USDT", 100.0, 2000.0, 3, TradeReason.SIGNAL)
    assert storage.trade_symbols() == ["BTC/USDT", "ETH/USDT"]
    storage.close()


@pytest.mark.parametrize("symbol", ["ETH/", "/USDT", " / ", "ETH/:USDT"])
def test_pair_without_base_or_quote_is_invalid(symbol: str) -> None:
    errors = make_config(symbol=symbol).validate()
    assert any("BASE/QUOTE" in e for e in errors)
    engine, broker, storage = make_engine(symbol)
    with pytest.raises(EngineError) as info:
        engine.start()
    assert not isinstance(info.value, CurrencyMismatchError)
    assert broker.currency is None
    storage.close()


def test_old_account_with_matching_position_adopts_currency() -> None:
    engine, broker, storage = make_engine("BTC/USDT")
    broker.market_buy("ETH/USDT", 100.0, 2000.0, 1_700_000_000_000, TradeReason.SIGNAL)
    engine.start()
    assert broker.currency == "USDT"
    storage.close()


def test_currency_not_adopted_when_start_fails_for_other_reason() -> None:
    engine, broker, storage = make_engine("BTC/USDT", fast_period=5, slow_period=3)  # invalid
    with pytest.raises(EngineError):
        engine.start()
    assert broker.currency is None
    storage.close()

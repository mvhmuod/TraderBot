"""Tests for traderbot.storage.Storage (no network, in-memory or tmp_path databases)."""
from __future__ import annotations

import dataclasses
import sqlite3
import threading

import pytest

from traderbot.config import BotConfig
from traderbot.models import Position, Side, Trade, TradeReason
from traderbot.storage import Storage


def make_trade(ts: int = 1_000, symbol: str = "BTC/USDT", side: Side = Side.BUY,
               reason: TradeReason = TradeReason.SIGNAL, realized_pnl: float = 0.0) -> Trade:
    return Trade(timestamp=ts, symbol=symbol, side=side, price=100.0, amount=2.0, fee=0.2,
                 quote_value=200.0, realized_pnl=realized_pnl, reason=reason)


@pytest.fixture
def storage() -> Storage:
    s = Storage(":memory:")
    s.reset_account(1_000.0)
    yield s
    s.close()


# ----------------------------------------------------------------- account
def test_new_storage_has_no_account() -> None:
    s = Storage(":memory:")
    try:
        assert s.load_account() is None
        assert s.load_positions() == {}
        assert s.load_trades() == []
        assert s.realized_pnl_total() == 0.0
    finally:
        s.close()


def test_reset_account_initialises_cash_and_starting_balance() -> None:
    s = Storage(":memory:")
    try:
        s.reset_account(5_000.0)
        assert s.load_account() == (5_000.0, 5_000.0)
    finally:
        s.close()


def test_reset_clears_trades_positions_and_day_start_meta_only(storage: Storage) -> None:
    pos = Position("BTC/USDT", 2.0, 100.1, 1_000)
    storage.record_fill(make_trade(), 800.0, pos, "BTC/USDT")
    storage.set_meta("day_start:2026-11-19", 1_000.0)
    storage.set_meta("day_start:2026-11-18", 990.0)
    storage.set_meta("dayXstart:keep", 1)          # "_" must NOT act as a wildcard
    storage.save_settings({"symbol": "ETH/USDT"})
    storage.set_meta("other", "keep me")

    storage.reset_account(2_500.0)

    assert storage.load_account() == (2_500.0, 2_500.0)
    assert storage.load_trades() == []
    assert storage.load_positions() == {}
    assert storage.realized_pnl_total() == 0.0
    assert storage.get_meta("day_start:2026-11-19") is None
    assert storage.get_meta("day_start:2026-11-18") is None
    assert storage.get_meta("dayXstart:keep") == 1
    assert storage.load_settings() == {"symbol": "ETH/USDT"}
    assert storage.get_meta("other") == "keep me"


def test_reset_account_is_one_transaction(storage: Storage) -> None:
    pos = Position("BTC/USDT", 2.0, 100.1, 1_000)
    storage.record_fill(make_trade(), 800.0, pos, "BTC/USDT")
    storage.set_meta("day_start:2026-11-19", 1_000.0)
    # Make the final account write fail AFTER the deletes have run.
    storage._conn.execute(
        "CREATE TRIGGER fail_account BEFORE UPDATE ON account BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        storage.reset_account(50.0)
    assert storage.load_account() == (800.0, 1_000.0)
    assert len(storage.load_trades()) == 1
    assert "BTC/USDT" in storage.load_positions()
    assert storage.get_meta("day_start:2026-11-19") == 1_000.0


def test_file_database_creates_parent_dir_and_persists(tmp_path) -> None:
    db = tmp_path / "nested" / "dir" / "bot.db"
    s = Storage(db)
    s.reset_account(1_000.0)
    s.record_fill(make_trade(), 800.0, Position("BTC/USDT", 2.0, 100.1, 1_000), "BTC/USDT")
    s.set_meta("k", [1, 2])
    s.close()
    assert db.exists()

    s2 = Storage(str(db))
    try:
        assert s2.load_account() == (800.0, 1_000.0)
        assert s2.load_positions() == {"BTC/USDT": Position("BTC/USDT", 2.0, 100.1, 1_000)}
        assert len(s2.load_trades()) == 1
        assert s2.get_meta("k") == [1, 2]
    finally:
        s2.close()


# ------------------------------------------------------------- record_fill
def test_record_fill_inserts_trade_updates_cash_and_upserts_position(storage: Storage) -> None:
    trade = make_trade(ts=1_000)
    pos = Position("BTC/USDT", 2.0, 100.1, 1_000)
    stored = storage.record_fill(trade, 800.0, pos, "BTC/USDT")

    assert stored.id is not None and stored.id >= 1
    assert stored == dataclasses.replace(trade, id=stored.id)
    assert trade.id is None                       # original is not mutated (frozen dataclass)
    assert storage.load_account() == (800.0, 1_000.0)
    assert storage.load_positions() == {"BTC/USDT": pos}
    assert storage.load_trades() == [stored]

    # Upsert: same symbol again replaces the row.
    pos2 = Position("BTC/USDT", 3.0, 101.0, 1_000)
    stored2 = storage.record_fill(make_trade(ts=2_000), 700.0, pos2, "BTC/USDT")
    assert stored2.id > stored.id
    assert storage.load_positions() == {"BTC/USDT": pos2}
    assert storage.load_account() == (700.0, 1_000.0)

    # position=None deletes it.
    sell = make_trade(ts=3_000, side=Side.SELL, realized_pnl=12.5)
    storage.record_fill(sell, 950.0, None, "BTC/USDT")
    assert storage.load_positions() == {}
    assert storage.load_account() == (950.0, 1_000.0)
    assert storage.realized_pnl_total() == pytest.approx(12.5)


def _state(storage: Storage):
    return (storage.load_account(), storage.load_positions(), storage.load_trades(),
            storage.realized_pnl_total())


def test_record_fill_is_atomic_when_position_write_fails(storage: Storage) -> None:
    """A failure on the LAST statement (position upsert) must undo the trade insert and cash update."""
    storage.record_fill(make_trade(ts=1_000), 800.0, Position("BTC/USDT", 2.0, 100.1, 1_000), "BTC/USDT")
    before = _state(storage)

    storage._conn.execute(
        "CREATE TRIGGER fail_pos BEFORE UPDATE ON positions BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        storage.record_fill(make_trade(ts=2_000, realized_pnl=5.0), 1.0,
                            Position("BTC/USDT", 9.0, 1.0, 1_000), "BTC/USDT")

    assert _state(storage) == before
    # Storage is still usable afterwards (no transaction left open).
    storage._conn.execute("DROP TRIGGER fail_pos")
    stored = storage.record_fill(make_trade(ts=3_000), 600.0, None, "BTC/USDT")
    assert storage.load_trades()[-1] == stored
    assert storage.load_account() == (600.0, 1_000.0)


def test_record_fill_is_atomic_when_delete_fails(storage: Storage) -> None:
    storage.record_fill(make_trade(ts=1_000), 800.0, Position("BTC/USDT", 2.0, 100.1, 1_000), "BTC/USDT")
    before = _state(storage)
    storage._conn.execute(
        "CREATE TRIGGER fail_del BEFORE DELETE ON positions BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        storage.record_fill(make_trade(ts=2_000, side=Side.SELL, realized_pnl=3.0), 999.0, None, "BTC/USDT")
    assert _state(storage) == before


def test_record_fill_is_atomic_on_python_error_mid_transaction(storage: Storage) -> None:
    """A bad value in the position (fails while binding, after insert + cash update) rolls back."""
    before = _state(storage)

    class BadInt:
        def __int__(self) -> int:
            raise RuntimeError("cannot convert")

    bad_pos = Position("BTC/USDT", 1.0, 1.0, BadInt())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="cannot convert"):
        storage.record_fill(make_trade(), 1.0, bad_pos, "BTC/USDT")
    assert _state(storage) == before


def test_record_fill_without_account_raises_and_rolls_back() -> None:
    s = Storage(":memory:")
    try:
        with pytest.raises(RuntimeError):
            s.record_fill(make_trade(), 1.0, None, "BTC/USDT")
        assert s.load_trades() == []
        assert s.load_account() is None
    finally:
        s.close()


# ------------------------------------------------------------- load_trades
def test_load_trades_ordering_symbol_filter_and_limit(storage: Storage) -> None:
    ids = []
    # Timestamps deliberately NOT in id order: ordering must follow the id.
    for ts, sym in [(5_000, "BTC/USDT"), (1_000, "ETH/USDT"), (3_000, "BTC/USDT"),
                    (2_000, "ETH/USDT"), (4_000, "BTC/USDT")]:
        ids.append(storage.record_fill(make_trade(ts=ts, symbol=sym), 900.0, None, sym).id)

    all_trades = storage.load_trades()
    assert [t.id for t in all_trades] == ids
    assert ids == sorted(ids)

    # limit -> the NEWEST n, still ascending
    assert [t.id for t in storage.load_trades(limit=2)] == ids[-2:]
    assert [t.id for t in storage.load_trades(limit=1)] == ids[-1:]
    assert [t.id for t in storage.load_trades(limit=99)] == ids
    assert storage.load_trades(limit=0) == []

    btc = storage.load_trades(symbol="BTC/USDT")
    assert [t.id for t in btc] == [ids[0], ids[2], ids[4]]
    assert [t.timestamp for t in btc] == [5_000, 3_000, 4_000]
    assert [t.id for t in storage.load_trades(symbol="BTC/USDT", limit=2)] == [ids[2], ids[4]]
    assert [t.id for t in storage.load_trades(symbol="ETH/USDT", limit=5)] == [ids[1], ids[3]]
    assert storage.load_trades(symbol="XRP/USDT") == []


def test_enum_round_trip(storage: Storage) -> None:
    for i, reason in enumerate(TradeReason):
        for side in Side:
            storage.record_fill(make_trade(ts=i, side=side, reason=reason), 900.0, None, "BTC/USDT")
    trades = storage.load_trades()
    assert len(trades) == len(TradeReason) * len(Side)
    for t in trades:
        assert type(t.side) is Side
        assert type(t.reason) is TradeReason
    assert {(t.side, t.reason) for t in trades} == {(s, r) for r in TradeReason for s in Side}
    # Stored as the enum's .value
    raw = storage._conn.execute("SELECT DISTINCT side, reason FROM trades").fetchall()
    assert {r[0] for r in raw} == {"BUY", "SELL"}
    assert {r[1] for r in raw} == {r.value for r in TradeReason}


def test_trade_field_types_round_trip(storage: Storage) -> None:
    trade = Trade(timestamp=1_731_974_400_123, symbol="ETH/BTC", side=Side.SELL, price=0.0371234,
                  amount=1.23456789, fee=0.0000458, quote_value=0.04578, realized_pnl=-0.00012,
                  reason=TradeReason.STOP_LOSS)
    stored = storage.record_fill(trade, 1.0, None, "ETH/BTC")
    loaded = storage.load_trades()[0]
    assert loaded == stored
    assert isinstance(loaded.timestamp, int)
    assert isinstance(loaded.price, float)


def test_realized_pnl_total(storage: Storage) -> None:
    assert storage.realized_pnl_total() == 0.0
    storage.record_fill(make_trade(realized_pnl=0.0), 900.0, None, "BTC/USDT")
    storage.record_fill(make_trade(side=Side.SELL, realized_pnl=10.25), 900.0, None, "BTC/USDT")
    storage.record_fill(make_trade(side=Side.SELL, realized_pnl=-4.0), 900.0, None, "BTC/USDT")
    assert storage.realized_pnl_total() == pytest.approx(6.25)


# ------------------------------------------------------- meta and settings
def test_meta_round_trip_and_default(storage: Storage) -> None:
    assert storage.get_meta("missing") is None
    assert storage.get_meta("missing", 42) == 42
    values = {"f": 1234.5, "i": 7, "s": "text", "l": [1, "a", None], "d": {"a": 1, "b": [2.5]},
              "b": True, "n": None}
    for k, v in values.items():
        storage.set_meta(k, v)
    for k, v in values.items():
        assert storage.get_meta(k, "DEFAULT") == v
    storage.set_meta("f", 99.0)  # overwrite
    assert storage.get_meta("f") == 99.0


def test_settings_round_trip(storage: Storage) -> None:
    assert storage.load_settings() is None
    cfg = BotConfig(symbol="ETH/USDT", fast_period=5, slow_period=30, fee_rate=0.002)
    storage.save_settings(cfg.to_dict())
    loaded = storage.load_settings()
    assert loaded == cfg.to_dict()
    assert BotConfig.from_dict(loaded) == cfg
    assert storage.get_meta("settings") == cfg.to_dict()


def test_concurrent_access_from_threads(storage: Storage) -> None:
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                storage.record_fill(make_trade(ts=n * 100 + i), 900.0, None, "BTC/USDT")
                storage.set_meta(f"t{n}", i)
                storage.load_trades(limit=5)
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(storage.load_trades()) == 80

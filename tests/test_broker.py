"""Tests for traderbot.broker.paper.PaperBroker (no network).

Expected numbers are worked out by hand from the spec formulas; the derivation
is written next to each assertion.
Default costs in these tests: fee_rate = 0.001 (0.1%), slippage_rate = 0.0005 (0.05%).
"""
from __future__ import annotations

import pytest

from traderbot.broker.paper import BrokerError, InsufficientFunds, NoPosition, PaperBroker
from traderbot.config import MIN_ORDER_QUOTE
from traderbot.models import AccountSnapshot, Position, Side, TradeReason
from traderbot.storage import Storage

SYM = "BTC/USDT"
FEE = 0.001
SLIP = 0.0005

# --- Hand-computed numbers --------------------------------------------------
# BUY 1000 quote @ 20000:
#   fill = 20000 * 1.0005 = 20010 ; fee = 1000 * 0.001 = 1
#   amount = (1000 - 1) / 20010 = 999 / 20010 = 0.04992503748125937...
#   avg_cost = 1000 / amount = 1000 * 20010 / 999 = 20030.03003003003...
BUY1_FILL = 20010.0
BUY1_FEE = 1.0
BUY1_AMOUNT = 0.04992503748125937
BUY1_AVG = 20030.03003003003


@pytest.fixture
def storage() -> Storage:
    s = Storage(":memory:")
    yield s
    s.close()


@pytest.fixture
def broker(storage: Storage) -> PaperBroker:
    return PaperBroker(storage, fee_rate=FEE, slippage_rate=SLIP, starting_balance=10_000.0)


# ---------------------------------------------------------------- init/reset
def test_new_broker_initialises_account(storage: Storage, broker: PaperBroker) -> None:
    assert broker.cash == 10_000.0
    assert broker.starting_balance == 10_000.0
    assert broker.positions() == {}
    assert broker.get_position(SYM) is None
    assert storage.load_account() == (10_000.0, 10_000.0)


def test_existing_account_is_kept_and_starting_balance_arg_ignored(storage: Storage) -> None:
    storage.reset_account(1_234.0)
    b = PaperBroker(storage, FEE, SLIP, starting_balance=99_999.0)
    assert b.cash == 1_234.0
    assert b.starting_balance == 1_234.0


def test_reset_clears_everything(storage: Storage, broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    storage.set_meta("day_start:2026-11-19", 10_000.0)
    broker.reset(500.0)
    assert broker.cash == 500.0
    assert broker.starting_balance == 500.0
    assert broker.positions() == {}
    assert storage.load_trades() == []
    assert storage.load_positions() == {}
    assert storage.get_meta("day_start:2026-11-19") is None
    assert broker.snapshot({}).realized_pnl_total == 0.0


def test_invalid_cost_rates_rejected(storage: Storage, broker: PaperBroker) -> None:
    with pytest.raises(ValueError):
        PaperBroker(storage, fee_rate=-0.1, slippage_rate=0.0, starting_balance=100.0)
    with pytest.raises(ValueError):
        broker.set_costs(0.001, 1.0)


# ------------------------------------------------------------------------ BUY
def test_buy_math(storage: Storage, broker: PaperBroker) -> None:
    trade = broker.market_buy(SYM, 1_000.0, 20_000.0, 1_700_000_000_000, TradeReason.SIGNAL)

    assert trade.id is not None
    assert trade.side is Side.BUY
    assert trade.symbol == SYM
    assert trade.timestamp == 1_700_000_000_000
    assert trade.reason is TradeReason.SIGNAL
    assert trade.price == pytest.approx(BUY1_FILL)
    assert trade.fee == pytest.approx(BUY1_FEE)
    assert trade.amount == pytest.approx(BUY1_AMOUNT, rel=1e-12)
    assert trade.quote_value == pytest.approx(1_000.0)
    assert trade.realized_pnl == 0.0

    assert broker.cash == pytest.approx(9_000.0)        # 10000 - 1000
    pos = broker.get_position(SYM)
    assert pos is not None
    assert pos.amount == pytest.approx(BUY1_AMOUNT, rel=1e-12)
    assert pos.avg_cost == pytest.approx(BUY1_AVG, rel=1e-12)
    assert pos.opened_at == 1_700_000_000_000

    # Persisted identically.
    assert storage.load_account() == (pytest.approx(9_000.0), 10_000.0)
    assert storage.load_positions() == {SYM: pos}
    assert storage.load_trades() == [trade]


def test_buy_without_costs_is_exact(storage: Storage) -> None:
    b = PaperBroker(storage, 0.0, 0.0, 1_000.0)
    t = b.market_buy(SYM, 250.0, 50.0, 1, TradeReason.SIGNAL)
    # amount = 250 / 50 = 5 ; avg_cost = 250 / 5 = 50 ; cash = 750
    assert t.amount == pytest.approx(5.0)
    assert t.fee == 0.0
    assert b.get_position(SYM) == Position(SYM, pytest.approx(5.0), pytest.approx(50.0), 1)
    assert b.cash == pytest.approx(750.0)


def test_buy_averages_into_existing_position(broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    t2 = broker.market_buy(SYM, 500.0, 25_000.0, 2_000, TradeReason.SIGNAL)
    # 2nd buy: fill = 25000 * 1.0005 = 25012.5 ; fee = 0.5
    #   amount = 499.5 / 25012.5 = 0.019970014992503747...
    assert t2.price == pytest.approx(25_012.5)
    assert t2.fee == pytest.approx(0.5)
    assert t2.amount == pytest.approx(0.019970014992503747, rel=1e-12)
    pos = broker.get_position(SYM)
    # total amount = 0.04992503748125937 + 0.019970014992503747 = 0.06989505247376313
    # avg_cost = (old.amount*old.avg_cost + 500) / total = 1500 / 0.06989505247376313 = 21460.74646074646
    assert pos.amount == pytest.approx(0.06989505247376313, rel=1e-12)
    assert pos.avg_cost == pytest.approx(21_460.74646074646, rel=1e-12)
    assert pos.opened_at == 1_000                       # unchanged by averaging in
    assert broker.cash == pytest.approx(8_500.0)        # 10000 - 1000 - 500


def test_buy_clamps_to_cash_within_tolerance(storage: Storage) -> None:
    b = PaperBroker(storage, 0.0, 0.0, 100.0)
    t = b.market_buy(SYM, 100.0 + 5e-10, 10.0, 1, TradeReason.SIGNAL)
    assert t.quote_value == 100.0                       # clamped to exactly the cash
    assert t.amount == pytest.approx(10.0)
    assert b.cash == 0.0


def test_buy_insufficient_funds(storage: Storage, broker: PaperBroker) -> None:
    with pytest.raises(InsufficientFunds):
        broker.market_buy(SYM, 10_000.01, 20_000.0, 1, TradeReason.SIGNAL)
    assert issubclass(InsufficientFunds, BrokerError)
    assert broker.cash == 10_000.0
    assert broker.positions() == {}
    assert storage.load_trades() == []


def test_buy_minimum_order_size(broker: PaperBroker) -> None:
    with pytest.raises(ValueError):
        broker.market_buy(SYM, MIN_ORDER_QUOTE - 0.01, 20_000.0, 1, TradeReason.SIGNAL)
    with pytest.raises(ValueError):
        broker.market_buy(SYM, 0.0, 20_000.0, 1, TradeReason.SIGNAL)
    # exactly the minimum is allowed
    t = broker.market_buy(SYM, MIN_ORDER_QUOTE, 20_000.0, 1, TradeReason.SIGNAL)
    assert t.quote_value == pytest.approx(MIN_ORDER_QUOTE)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, float("nan"), float("inf")])
def test_buy_rejects_bad_price(broker: PaperBroker, bad_price: float) -> None:
    with pytest.raises(ValueError):
        broker.market_buy(SYM, 100.0, bad_price, 1, TradeReason.SIGNAL)
    assert broker.cash == 10_000.0


# ----------------------------------------------------------------------- SELL
def test_partial_sell_math(storage: Storage, broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    half = BUY1_AMOUNT / 2                              # 0.024962518740629686
    t = broker.market_sell(SYM, half, 22_000.0, 2_000, TradeReason.TAKE_PROFIT)
    # fill  = 22000 * (1 - 0.0005) = 21989
    # gross = 0.024962518740629686 * 21989 = 548.9008245877061
    # fee   = gross * 0.001 = 0.5489008245877062
    # net   = gross - fee = 548.3519237631184
    # pnl   = net - half * avg_cost = 548.3519237631184 - 500 = 48.35192376311844
    assert t.side is Side.SELL
    assert t.reason is TradeReason.TAKE_PROFIT
    assert t.price == pytest.approx(21_989.0)
    assert t.amount == pytest.approx(half, rel=1e-12)
    assert t.fee == pytest.approx(0.5489008245877062, rel=1e-12)
    assert t.quote_value == pytest.approx(548.3519237631184, rel=1e-12)
    assert t.realized_pnl == pytest.approx(48.35192376311844, rel=1e-10)
    assert broker.cash == pytest.approx(9_548.351923763119, rel=1e-12)

    pos = broker.get_position(SYM)
    assert pos.amount == pytest.approx(half, rel=1e-12)
    assert pos.avg_cost == pytest.approx(BUY1_AVG, rel=1e-12)   # unchanged on sells
    assert pos.opened_at == 1_000
    assert storage.load_positions() == {SYM: pos}
    assert storage.realized_pnl_total() == pytest.approx(48.35192376311844, rel=1e-10)


def test_sell_without_costs_is_exact(storage: Storage) -> None:
    b = PaperBroker(storage, 0.0, 0.0, 1_000.0)
    b.market_buy(SYM, 1_000.0, 100.0, 1, TradeReason.SIGNAL)     # 10 units @ 100
    t = b.market_sell(SYM, 4.0, 110.0, 2, TradeReason.SIGNAL)
    # net = 4 * 110 = 440 ; pnl = 440 - 4 * 100 = 40 ; remaining 6
    assert t.quote_value == pytest.approx(440.0)
    assert t.realized_pnl == pytest.approx(40.0)
    assert b.cash == pytest.approx(440.0)
    assert b.get_position(SYM).amount == pytest.approx(6.0)


def test_full_close_removes_position(storage: Storage, broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    t = broker.close_position(SYM, 19_000.0, 2_000, TradeReason.STOP_LOSS)
    # fill  = 19000 * 0.9995 = 18990.5
    # gross = 0.04992503748125937 * 18990.5 = 948.101424287856
    # fee   = 0.948101424287856 ; net = 947.1533228635682
    # pnl   = net - 1000 = -52.846677136431786
    assert t.amount == pytest.approx(BUY1_AMOUNT, rel=1e-12)
    assert t.price == pytest.approx(18_990.5)
    assert t.fee == pytest.approx(0.948101424287856, rel=1e-12)
    assert t.quote_value == pytest.approx(947.1533228635682, rel=1e-12)
    assert t.realized_pnl == pytest.approx(-52.846677136431786, rel=1e-10)
    assert t.reason is TradeReason.STOP_LOSS
    assert broker.cash == pytest.approx(9_947.153322863569, rel=1e-12)
    assert broker.get_position(SYM) is None
    assert broker.positions() == {}
    assert storage.load_positions() == {}
    assert [x.side for x in storage.load_trades()] == [Side.BUY, Side.SELL]


def test_sell_clamps_within_tolerance_and_closes(broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    held = broker.get_position(SYM).amount
    t = broker.market_sell(SYM, held + 5e-10, 20_000.0, 2_000, TradeReason.SIGNAL)
    assert t.amount == held                              # clamped to exactly what is held
    assert broker.get_position(SYM) is None


def test_sell_leaving_only_dust_removes_position(broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    held = broker.get_position(SYM).amount
    broker.market_sell(SYM, held - 1e-13, 20_000.0, 2_000, TradeReason.SIGNAL)
    assert broker.get_position(SYM) is None


def test_oversell_beyond_tolerance_raises(storage: Storage, broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    held = broker.get_position(SYM).amount
    cash = broker.cash
    with pytest.raises(BrokerError):
        broker.market_sell(SYM, held + 1e-6, 20_000.0, 2_000, TradeReason.SIGNAL)
    assert broker.get_position(SYM).amount == held
    assert broker.cash == cash
    assert len(storage.load_trades()) == 1


def test_sell_without_position_raises_no_position(broker: PaperBroker) -> None:
    with pytest.raises(NoPosition):
        broker.market_sell(SYM, 1.0, 20_000.0, 1, TradeReason.SIGNAL)
    with pytest.raises(NoPosition):
        broker.close_position(SYM, 20_000.0, 1, TradeReason.SIGNAL)
    assert issubclass(NoPosition, BrokerError)


@pytest.mark.parametrize("bad_price", [0.0, -5.0, float("nan")])
def test_sell_rejects_bad_price(broker: PaperBroker, bad_price: float) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    with pytest.raises(ValueError):
        broker.market_sell(SYM, 0.01, bad_price, 2, TradeReason.SIGNAL)


@pytest.mark.parametrize("bad_amount", [0.0, -0.01, float("nan")])
def test_sell_rejects_bad_amount(broker: PaperBroker, bad_amount: float) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    with pytest.raises(ValueError):
        broker.market_sell(SYM, bad_amount, 20_000.0, 2, TradeReason.SIGNAL)


# ------------------------------------------------------------ atomicity/costs
def test_memory_unchanged_when_storage_write_fails(storage: Storage, broker: PaperBroker,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    cash, pos = broker.cash, broker.get_position(SYM)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(storage, "record_fill", boom)
    with pytest.raises(RuntimeError):
        broker.market_buy(SYM, 100.0, 20_000.0, 2_000, TradeReason.SIGNAL)
    with pytest.raises(RuntimeError):
        broker.close_position(SYM, 20_000.0, 2_000, TradeReason.SIGNAL)
    assert broker.cash == cash
    assert broker.get_position(SYM) == pos


def test_set_costs_applies_to_next_fill(storage: Storage) -> None:
    b = PaperBroker(storage, 0.0, 0.0, 1_000.0)
    b.set_costs(0.01, 0.1)
    t = b.market_buy(SYM, 100.0, 10.0, 1, TradeReason.SIGNAL)
    # fill = 10 * 1.1 = 11 ; fee = 1 ; amount = 99 / 11 = 9
    assert t.price == pytest.approx(11.0)
    assert t.fee == pytest.approx(1.0)
    assert t.amount == pytest.approx(9.0)


# ------------------------------------------------------------- persistence
def test_state_persists_across_new_broker(tmp_path) -> None:
    db = tmp_path / "paper.db"
    s1 = Storage(db)
    b1 = PaperBroker(s1, FEE, SLIP, 10_000.0)
    b1.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    b1.market_buy("ETH/USDT", 300.0, 3_000.0, 1_500, TradeReason.SIGNAL)
    b1.market_sell(SYM, BUY1_AMOUNT / 2, 22_000.0, 2_000, TradeReason.SIGNAL)
    cash1, positions1 = b1.cash, b1.positions()
    s1.close()

    s2 = Storage(db)
    try:
        b2 = PaperBroker(s2, FEE, SLIP, starting_balance=1.0)   # arg ignored: account exists
        assert b2.cash == cash1
        assert b2.starting_balance == 10_000.0
        assert b2.positions() == positions1
        assert len(s2.load_trades()) == 3
        assert b2.snapshot({}).realized_pnl_total == pytest.approx(48.35192376311844, rel=1e-10)
        # and trading continues from the stored state
        b2.close_position(SYM, 22_000.0, 3_000, TradeReason.SIGNAL)
        assert b2.get_position(SYM) is None
        assert set(b2.positions()) == {"ETH/USDT"}
    finally:
        s2.close()


# ----------------------------------------------------------- equity/snapshot
def test_equity_and_snapshot(storage: Storage) -> None:
    b = PaperBroker(storage, 0.0, 0.0, 1_000.0)
    b.market_buy(SYM, 500.0, 100.0, 1, TradeReason.SIGNAL)          # 5 BTC @ 100, cash 500
    b.market_buy("ETH/USDT", 200.0, 10.0, 1, TradeReason.SIGNAL)    # 20 ETH @ 10, cash 300
    # With no marks, positions are valued at avg_cost -> 300 + 500 + 200
    assert b.equity({}) == pytest.approx(1_000.0)
    # marks: BTC 120 -> 600 ; ETH missing -> avg_cost 10 -> 200 ; total 300 + 600 + 200
    assert b.equity({SYM: 120.0}) == pytest.approx(1_100.0)

    b.market_sell(SYM, 1.0, 120.0, 2, TradeReason.SIGNAL)           # +120 cash, pnl +20
    snap = b.snapshot({SYM: 130.0, "ETH/USDT": 9.0})
    assert isinstance(snap, AccountSnapshot)
    assert snap.cash == pytest.approx(420.0)
    assert snap.starting_balance == 1_000.0
    assert set(snap.positions) == {SYM, "ETH/USDT"}
    assert snap.positions[SYM].amount == pytest.approx(4.0)
    # equity = 420 + 4*130 + 20*9 = 420 + 520 + 180 = 1120
    assert snap.equity == pytest.approx(1_120.0)
    assert snap.realized_pnl_total == pytest.approx(20.0)
    assert snap.total_pnl == pytest.approx(120.0)
    assert snap.total_pnl_pct == pytest.approx(12.0)

    # Snapshot is a copy: later trades don't change it.
    b.close_position(SYM, 130.0, 3, TradeReason.SIGNAL)
    assert SYM in snap.positions
    assert SYM not in b.positions()


def test_equity_with_costs_uses_mark_as_is(broker: PaperBroker) -> None:
    broker.market_buy(SYM, 1_000.0, 20_000.0, 1_000, TradeReason.SIGNAL)
    # 9000 + 0.04992503748125937 * 21000 = 9000 + 1048.4257871064468 = 10048.425787106447
    assert broker.equity({SYM: 21_000.0}) == pytest.approx(10_048.425787106447, rel=1e-12)
    # no mark -> avg_cost -> 9000 + 1000
    assert broker.equity({}) == pytest.approx(10_000.0, rel=1e-12)

"""GUI smoke tests (offscreen, no network).

A fake market data feed supplies deterministic candles. The tests drive the real MainWindow,
EngineWorker thread, engine, paper broker and a temporary SQLite database.
"""
from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # must be set before Qt is imported

import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QEventLoop, QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QMessageBox

from traderbot.gui.chart import PriceChart, format_price
from traderbot.gui.main_window import MainWindow, format_amount, format_money
from traderbot.broker.paper import PaperBroker
from traderbot.models import CANDLE_COLUMNS, Side, Trade, TradeReason
from traderbot.storage import Storage

MINUTE_MS = 60_000


# ---------------------------------------------------------------------- fakes & helpers
class FakeFeed:
    """Deterministic stand-in for MarketDataFeed (same public methods, no network).

    40 one-minute candles with steadily falling closes (fast SMA below slow SMA).
    ``advance(close)`` appends one more closed candle; a big jump creates a BUY crossover.
    """

    def __init__(self, exchange_id: str) -> None:
        self.exchange_id = exchange_id
        self._lock = threading.Lock()   # the worker thread reads while the test thread advances
        # Start 2 hours in the past so every candle is safely closed.
        self._start_ms = (int(time.time() * 1000) // MINUTE_MS - 120) * MINUTE_MS
        self._closes: list[float] = [100.0 - 0.5 * i for i in range(40)]

    def advance(self, close: float) -> None:
        with self._lock:
            self._closes.append(float(close))

    def list_symbols(self) -> list[str]:
        return ["BTC/USDT", "ETH/USDT"]

    def fetch_closed_candles(self, symbol: str, timeframe: str, limit: int,
                             now_ms: int | None = None) -> pd.DataFrame:
        with self._lock:
            closes = list(self._closes)
        rows = []
        for i, close in enumerate(closes):
            open_ = closes[i - 1] if i > 0 else close + 0.2
            rows.append([self._start_ms + i * MINUTE_MS, open_, max(open_, close) + 0.3,
                         min(open_, close) - 0.3, close, 10.0])
        frame = pd.DataFrame(rows[-limit:], columns=CANDLE_COLUMNS)
        frame["timestamp"] = frame["timestamp"].astype("int64")
        return frame.reset_index(drop=True)

    def fetch_last_price(self, symbol: str) -> float:
        with self._lock:
            return self._closes[-1]


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Process Qt events until `condition()` is true or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while True:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if condition():
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def problems(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Collects exceptions raised inside Qt slots and Qt threading/timer warnings."""
    found: list[str] = []

    def excepthook(exc_type, exc, tb):  # noqa: ANN001
        found.append(f"{exc_type.__name__}: {exc}")

    def qt_handler(mode, context, message):  # noqa: ANN001
        lowered = message.lower()
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg) and (
                "thread" in lowered or "timer" in lowered):
            found.append(f"Qt: {message}")

    monkeypatch.setattr(sys, "excepthook", excepthook)
    previous = qInstallMessageHandler(qt_handler)
    yield found
    qInstallMessageHandler(previous)


def make_window(tmp_path: Path, feed: FakeFeed) -> MainWindow:
    window = MainWindow(db_path=str(tmp_path / "traderbot_test.db"), feed_factory=lambda _id: feed)
    window.show()
    assert wait_until(lambda: window.is_ready, 10), "worker did not initialise"
    return window


def configure_fast_bot(window: MainWindow) -> None:
    window.timeframe_combo.setCurrentText("1m")
    window.fast_spin.setValue(3)
    window.slow_spin.setValue(8)
    window.poll_spin.setValue(5)        # minimum allowed poll interval
    window.size_spin.setValue(10.0)


def close_window(window: MainWindow) -> None:
    window.close()
    assert window.worker_thread.wait(5000)
    assert window.worker_thread.isFinished()
    window.deleteLater()
    QApplication.processEvents()


# ---------------------------------------------------------------------- MainWindow smoke tests
def test_main_window_runs_bot_trades_and_stops(qapp: QApplication, tmp_path: Path,
                                               problems: list[str]) -> None:
    feed = FakeFeed("binance")
    window = make_window(tmp_path, feed)

    assert window.windowTitle() == "TraderBot — Paper Trading"
    assert "PAPER TRADING" in window.paper_badge.text()
    assert window.status_label.text() == "Stopped"
    assert window.start_button.isEnabled() and not window.stop_button.isEnabled()
    assert window.cash_card.value.text() == "10,000.00"

    # Startup preview draws the chart (fake feed, no trading).
    assert wait_until(lambda: window.chart.candles.count == 40, 10), "no chart preview"
    assert window.trades_table.rowCount() == 0
    assert window.pair_combo.count() == 2   # symbols loaded from the fake feed

    configure_fast_bot(window)
    window.start_button.click()
    assert wait_until(lambda: window.step_count >= 1, 10), "no step_done within 10 s"
    assert window.is_running
    assert window.status_label.text() == "Running"
    assert not window.start_button.isEnabled() and window.stop_button.isEnabled()
    assert not window.fast_spin.isEnabled() and not window.reset_button.isEnabled()
    assert window.chart.candles.count == 40
    assert len(window.chart.fast_curve.getData()[0]) == 40 - 3 + 1
    assert len(window.chart.slow_curve.getData()[0]) == 40 - 8 + 1
    assert window.chart.price_line.isVisible()
    assert window.chart.price_line.value() == pytest.approx(80.5)
    assert window.last_price_card.value.text() == "80.5000"
    assert window.trades_table.rowCount() == 0     # first step only watches

    # A new closed candle that jumps up -> fast SMA crosses above slow SMA -> paper BUY.
    feed.advance(130.0)
    assert wait_until(lambda: window.trades_table.rowCount() >= 1, 10), "no trade after crossover"
    assert window.trades_table.rowCount() == 1
    assert window.trades_table.item(0, 1).text() == "BTC/USDT"
    assert window.trades_table.item(0, 2).text() == "BUY"
    assert window.trades_table.item(0, 8).text() == "Signal"
    assert window.positions_table.rowCount() == 1
    assert window.positions_table.item(0, 0).text() == "BTC/USDT"
    assert window.chart.marker_counts() == (1, 0)
    assert window.chart.candles.count == 41
    # The view followed the new, much higher candle (the user did not zoom).
    view_box = window.chart.plot_widget.getPlotItem().getViewBox()
    assert view_box.viewRange()[1][1] >= 130.0
    assert "BTC @" in window.position_card.value.text()
    assert "BUY" in window.log_view.toPlainText()

    window.stop_button.click()
    assert wait_until(lambda: not window.is_running and window.status_label.text() == "Stopped", 10)
    assert window.start_button.isEnabled() and window.fast_spin.isEnabled()

    # Reset the paper account (confirmation dialog answered automatically).
    window._confirm = lambda title, text: True  # type: ignore[method-assign]
    window.balance_spin.setValue(5000.0)
    window.reset_button.click()
    assert wait_until(lambda: window.trades_table.rowCount() == 0
                      and window.positions_table.rowCount() == 0, 10)
    assert wait_until(lambda: window.cash_card.value.text() == "5,000.00", 10)
    # The old trade's marker disappears from the chart after the reset.
    assert wait_until(lambda: window.chart.marker_counts() == (0, 0), 10)
    assert window.chart.candles.count > 0

    close_window(window)
    assert problems == []


def test_close_while_running_shuts_down_cleanly(qapp: QApplication, tmp_path: Path,
                                                problems: list[str]) -> None:
    feed = FakeFeed("binance")
    window = make_window(tmp_path, feed)
    configure_fast_bot(window)
    window.start_button.click()
    assert wait_until(lambda: window.step_count >= 1, 10), "no step_done within 10 s"

    asked: list[str] = []

    def confirm(title: str, text: str) -> bool:
        asked.append(text)
        return True

    window._confirm = confirm  # type: ignore[method-assign]
    close_window(window)
    assert asked and "Stop the bot and quit?" in asked[0]
    # The database was closed by the worker, so the file is no longer locked (Windows).
    db_file = tmp_path / "traderbot_test.db"
    assert db_file.exists()
    db_file.rename(tmp_path / "renamed.db")
    assert problems == []


def test_settings_persist_and_invalid_settings_do_not_start(qapp: QApplication, tmp_path: Path,
                                                            problems: list[str]) -> None:
    feed = FakeFeed("binance")
    window = make_window(tmp_path, feed)
    window.fast_spin.setValue(10)
    window.slow_spin.setValue(10)      # fast must be smaller than slow -> invalid
    window.start_button.click()
    QApplication.processEvents()
    assert not window.is_running
    assert window.status_label.text() == "Stopped"
    boxes = [w for w in window.findChildren(QMessageBox) if w.isVisible()]
    assert boxes and "smaller than slow" in boxes[0].text()
    for box in boxes:
        box.close()

    window.slow_spin.setValue(30)
    window.timeframe_combo.setCurrentText("15m")
    close_window(window)

    # Settings typed in the window are remembered for the next start.
    window2 = make_window(tmp_path, feed)
    assert window2.slow_spin.value() == 30
    assert window2.fast_spin.value() == 10
    assert window2.timeframe_combo.currentText() == "15m"
    close_window(window2)
    assert problems == []


def test_second_window_on_same_database_is_refused(qapp: QApplication, tmp_path: Path,
                                                  problems: list[str]) -> None:
    # Two app instances on one database would overwrite each other's cash balance.
    feed = FakeFeed("binance")
    window = make_window(tmp_path, feed)
    second = MainWindow(db_path=str(tmp_path / "traderbot_test.db"), feed_factory=lambda _id: feed)
    try:
        second.show()
        assert wait_until(lambda: second.status_label.text() == "Error", 10), "second window opened the db"
        assert not second.is_ready and not second.start_button.isEnabled()
        boxes = [w for w in second.findChildren(QMessageBox) if w.isVisible()]
        assert boxes and "already running" in boxes[0].text()
        for box in boxes:
            box.close()
        assert window.is_ready
    finally:
        close_window(second)
        close_window(window)
    # Once the first window is closed the database can be opened again.
    third = make_window(tmp_path, feed)
    close_window(third)
    assert problems == []


def test_position_in_other_pair_is_valued_at_market_on_startup(qapp: QApplication, tmp_path: Path,
                                                               problems: list[str]) -> None:
    storage = Storage(str(tmp_path / "traderbot_test.db"))
    broker = PaperBroker(storage, 0.0, 0.0, 10_000.0)
    broker.market_buy("ETH/USDT", 1000.0, 50.0, 1_700_000_000_000, TradeReason.SIGNAL)   # 20 ETH
    storage.close()

    feed = FakeFeed("binance")          # last price 80.5 for every symbol; the window shows BTC/USDT
    window = make_window(tmp_path, feed)
    try:
        expected_equity = 9000.0 + 20.0 * 80.5
        assert wait_until(lambda: window.equity_card.value.text() == format_money(expected_equity), 10)
        assert window.positions_table.item(0, 0).text() == "ETH/USDT"
        assert window.positions_table.item(0, 3).text() == format_price(80.5)
        assert window.positions_table.item(0, 4).text() == format_money(20.0 * 80.5)
    finally:
        close_window(window)
    assert problems == []


def test_pair_in_other_currency_offers_reset(qapp: QApplication, tmp_path: Path,
                                             problems: list[str]) -> None:
    # A USDT account with one trade; the user then picks ETH/BTC (priced in BTC).
    storage = Storage(str(tmp_path / "traderbot_test.db"))
    broker = PaperBroker(storage, 0.0, 0.0, 10_000.0)
    broker.set_currency("USDT")
    broker.market_buy("ETH/USDT", 1000.0, 50.0, 1_700_000_000_000, TradeReason.SIGNAL)
    storage.close()

    feed = FakeFeed("binance")
    window = make_window(tmp_path, feed)
    try:
        configure_fast_bot(window)
        window.pair_combo.setEditText("ETH/BTC")
        asked: list[str] = []
        answer = {"value": False}

        def confirm(title: str, text: str) -> bool:
            asked.append(text)
            return answer["value"]

        window._confirm = confirm  # type: ignore[method-assign]

        # 1) Start is refused and a reset is offered; answering No changes nothing.
        window.start_button.click()
        assert wait_until(lambda: len(asked) == 1, 10), "no reset offer"
        assert "This paper account is in USDT, but ETH/BTC is priced in BTC" in asked[0]
        assert "10,000.00 BTC" in asked[0]
        assert not window.is_running and window.status_label.text() == "Error"
        assert window.start_button.isEnabled()
        assert wait_until(lambda: window.positions_table.rowCount() == 1, 5)
        # No plain error box on top of the question.
        assert not [w for w in window.findChildren(QMessageBox) if w.isVisible()]

        # 2) Answering Yes resets the account in BTC; Start then works.
        answer["value"] = True
        window.start_button.click()
        assert wait_until(lambda: len(asked) == 2, 10)
        assert wait_until(lambda: window.positions_table.rowCount() == 0
                          and window.trades_table.rowCount() == 0, 10), "account not reset"
        assert wait_until(lambda: "10,000.00 BTC" in window.log_view.toPlainText(), 10)
        window.start_button.click()
        assert wait_until(lambda: window.is_running, 10), "bot did not start after reset"
        assert len(asked) == 2
        window.stop_button.click()
        assert wait_until(lambda: not window.is_running, 10)
    finally:
        close_window(window)

    check = Storage(str(tmp_path / "traderbot_test.db"))
    assert PaperBroker(check, 0.0, 0.0, 1.0).currency == "BTC"
    check.close()
    assert problems == []


# ---------------------------------------------------------------------- PriceChart unit tests
def _candles_with_ma(n: int = 30) -> pd.DataFrame:
    start = 1_700_000_000_000
    closes = np.linspace(100.0, 130.0, n)
    frame = pd.DataFrame({
        "timestamp": np.array([start + i * MINUTE_MS for i in range(n)], dtype="int64"),
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 5.0),
    })
    frame["fast_ma"] = frame["close"].rolling(3, min_periods=3).mean()
    frame["slow_ma"] = frame["close"].rolling(8, min_periods=8).mean()
    return frame


def _trade(ts: int, side: Side, price: float) -> Trade:
    return Trade(timestamp=ts, symbol="BTC/USDT", side=side, price=price, amount=0.1, fee=0.01,
                 quote_value=price * 0.1, realized_pnl=0.0, reason=TradeReason.SIGNAL)


def test_chart_set_data_empty(qapp: QApplication) -> None:
    chart = PriceChart()
    chart.set_data(None, [], None, 9, 21)
    assert chart.candles.count == 0
    assert chart.marker_counts() == (0, 0)
    assert not chart.price_line.isVisible()

    empty = pd.DataFrame(columns=CANDLE_COLUMNS + ["fast_ma", "slow_ma"])
    chart.set_data(empty, [_trade(1_700_000_000_000, Side.BUY, 100.0)], 101.5, 9, 21)
    assert chart.candles.count == 0
    assert chart.marker_counts() == (0, 0)
    assert chart.price_line.isVisible()
    assert chart.price_line.value() == pytest.approx(101.5)
    chart.clear()
    assert not chart.price_line.isVisible()
    chart.deleteLater()


def test_chart_set_data_with_trades(qapp: QApplication) -> None:
    chart = PriceChart()
    chart.resize(800, 400)
    chart.show()
    candles = _candles_with_ma(30)
    first_ts = int(candles["timestamp"].iloc[0])
    trades = [
        _trade(first_ts + 5 * MINUTE_MS + 1000, Side.BUY, 105.0),
        _trade(first_ts + 12 * MINUTE_MS + 1000, Side.SELL, 112.0),
        _trade(first_ts - 10 * MINUTE_MS, Side.SELL, 90.0),   # before the visible history: skipped
    ]
    chart.set_data(candles, trades, 130.25, 3, 8)
    QApplication.processEvents()

    assert chart.candles.count == 30
    assert len(chart.fast_curve.getData()[0]) == 28        # NaN head of the MA is not drawn
    assert len(chart.slow_curve.getData()[0]) == 23
    assert chart.marker_counts() == (1, 1)
    buy = chart.buy_markers.data[0]
    sell = chart.sell_markers.data[0]
    assert buy["x"] == pytest.approx((first_ts + 5 * MINUTE_MS + 1000) / 1000)
    assert buy["y"] < candles["low"].iloc[5]                # below the candle
    assert sell["y"] > candles["high"].iloc[12]             # above the candle
    assert chart.price_line.isVisible()
    assert chart.price_line.value() == pytest.approx(130.25)
    assert "130.25" in chart.price_line.label.toPlainText()
    labels = [label.text for _sample, label in chart._legend.items]
    assert labels == ["Fast SMA (3)", "Slow SMA (8)"]

    # Bounds are in seconds and cover all candles.
    x_min, x_max = chart.candles.dataBounds(0)
    assert x_min < first_ts / 1000 < x_max
    y_min, y_max = chart.candles.dataBounds(1)
    assert y_min == pytest.approx(candles["low"].min())
    assert y_max == pytest.approx(candles["high"].max())

    # The view was fitted to the data on the first call.
    (vx_min, vx_max), _ = chart.plot_widget.getPlotItem().getViewBox().viewRange()
    assert vx_min <= first_ts / 1000 and vx_max >= candles["timestamp"].iloc[-1] / 1000

    view_box = chart.plot_widget.getPlotItem().getViewBox()

    # Without a user zoom, the view follows new data (e.g. a new candle with a higher high).
    taller = candles.copy()
    taller.loc[taller.index[-1], "high"] = 200.0
    chart.set_data(taller, trades, 130.25, 3, 8)
    QApplication.processEvents()
    assert view_box.viewRange()[1][1] >= 200.0

    # A user zoom is kept on the next update; reset_view() fits again.
    zoom_x = (first_ts / 1000 + 600, first_ts / 1000 + 1200)
    view_box.setXRange(*zoom_x, padding=0)
    chart.set_data(candles, trades, 130.5, 3, 8)
    QApplication.processEvents()
    assert view_box.viewRange()[0] == pytest.approx(list(zoom_x))
    chart.reset_view()
    QApplication.processEvents()
    assert view_box.viewRange()[0][0] <= first_ts / 1000

    # Paint once to make sure drawing code runs without errors.
    image = chart.grab()
    assert not image.isNull()

    chart.set_data(candles.iloc[0:0], trades, None, 3, 8)
    assert chart.candles.count == 0
    assert chart.marker_counts() == (0, 0)
    assert not chart.price_line.isVisible()
    chart.deleteLater()


def test_number_formatting() -> None:
    assert format_price(62690.123) == "62,690.12"
    assert format_price(1.23456) == "1.2346"
    assert format_price(0.5) == "0.50"
    assert format_price(0.000008123) == "0.000008123"
    assert format_price(None) == "—"
    assert format_money(-0.001) == "0.00"
    assert format_money(1234.5, signed=True) == "+1,234.50"
    assert format_amount(0.01530000) == "0.0153"
    assert format_amount(1500.0) == "1,500"

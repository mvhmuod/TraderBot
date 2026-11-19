"""Main application window.

The window never touches the database, the exchange or the engine directly: it sends requests
to :class:`EngineWorker` (running in its own thread) through queued signals and updates the
widgets when the worker answers.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QCompleter, QDoubleSpinBox,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from traderbot.config import (
    DEFAULT_DB_PATH, MAX_MA_PERIOD, MIN_ORDER_QUOTE, MIN_POLL_SECONDS, SUPPORTED_EXCHANGES,
    SUPPORTED_TIMEFRAMES, BotConfig,
)
from traderbot.data.feed import MarketDataFeed
from traderbot.gui.chart import PriceChart, format_price
from traderbot.gui.worker import (
    PREVIEW_SKIPPED_RUNNING, STOP_REASON_HALTED, STOP_REASON_START_FAILED, TRADE_HISTORY_LIMIT,
    EngineWorker,
)
from traderbot.models import AccountSnapshot, Side, StepResult, Trade, TradeReason

logger = logging.getLogger(__name__)

WINDOW_TITLE = "TraderBot — Paper Trading"
PAPER_BADGE_TEXT = "PAPER TRADING — simulated orders, no real money"

PREVIEW_DEBOUNCE_MS = 800
SHUTDOWN_WAIT_MS = 5000
LOG_MAX_LINES = 2000
EPS = 1e-9

# Colors that read well on light and dark backgrounds.
POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
STATUS_COLORS = {
    "Stopped": "#6b7280",
    "Starting…": "#2563eb",
    "Stopping…": "#2563eb",
    "Running": "#16a34a",
    "Halted": "#d97706",
    "Error": "#dc2626",
}
BADGE_COLOR = "#b45309"

REASON_TEXT = {
    TradeReason.SIGNAL: "Signal",
    TradeReason.STOP_LOSS: "Stop-loss",
    TradeReason.TAKE_PROFIT: "Take-profit",
    TradeReason.DAILY_LIMIT: "Daily loss limit",
}

POSITION_HEADERS = ["Symbol", "Amount", "Avg cost", "Last price", "Value", "Unrealized P&L"]
TRADE_HEADERS = ["Time (local)", "Symbol", "Side", "Price", "Amount", "Fee", "Value",
                 "Realized P&L", "Reason"]
_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
_LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


# ---------------------------------------------------------------------- formatting helpers
def format_money(value: float | None, signed: bool = False) -> str:
    """Money with thousands separators and 2 decimals ("—" when unknown)."""
    if value is None or not math.isfinite(value):
        return "—"
    if abs(value) < 0.005:
        value = 0.0  # avoid "-0.00"
    return f"{value:+,.2f}" if signed else f"{value:,.2f}"


def format_amount(value: float | None) -> str:
    """Coin amount with up to 8 decimals, trailing zeros removed."""
    if value is None or not math.isfinite(value):
        return "—"
    text = f"{value:,.8f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def format_fee(value: float | None) -> str:
    """Fees are often tiny, so show 4 decimals."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:,.4f}"


def pnl_color(value: float | None) -> str | None:
    """Green for profit, red for loss, None (default text color) for zero/unknown."""
    if value is None or not math.isfinite(value):
        return None
    if value > EPS:
        return POSITIVE_COLOR
    if value < -EPS:
        return NEGATIVE_COLOR
    return None


class _WheelGuard(QObject):
    """Stops the mouse wheel from changing spin/combo boxes that do not have keyboard focus.

    Without this, scrolling the settings panel would accidentally change values.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt naming)
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


class _StatCard(QFrame):
    """Small caption + value box used in the summary bar."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        self.caption = QLabel(caption.upper())
        caption_font = self.caption.font()
        caption_font.setPointSizeF(max(7.0, caption_font.pointSizeF() * 0.8))
        caption_font.setBold(True)
        self.caption.setFont(caption_font)
        palette = self.caption.palette()
        muted = QColor(palette.color(QPalette.ColorRole.WindowText))
        muted.setAlpha(150)
        palette.setColor(QPalette.ColorRole.WindowText, muted)
        self.caption.setPalette(palette)
        self.value = QLabel("—")
        value_font = self.value.font()
        value_font.setPointSizeF(value_font.pointSizeF() * 1.25)
        value_font.setBold(True)
        self.value.setFont(value_font)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_value(self, text: str, color: str | None = None, tooltip: str = "") -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color};" if color else "")
        self.value.setToolTip(tooltip)


class MainWindow(QMainWindow):
    """TraderBot main window (paper trading only)."""

    # Requests to the worker (always delivered with Qt.QueuedConnection).
    _request_initialize = Signal(str)
    _request_start = Signal(object)
    _request_stop = Signal()
    _request_symbols = Signal(str)
    _request_preview = Signal(object)
    _request_reset = Signal(float)
    _request_save_settings = Signal(object)
    _request_shutdown = Signal()

    def __init__(self, db_path: str | None = None,
                 feed_factory: Callable[[str], MarketDataFeed] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1440, 900)
        self.setMinimumSize(1000, 640)

        self._db_path = str(db_path) if db_path is not None else str(DEFAULT_DB_PATH)
        self._ready = False             # worker initialised the account
        self._running = False
        self._starting = False
        self._stopping = False
        self._shutdown_complete = False
        self._close_pending = False
        self._applying_settings = False  # True while widgets are filled programmatically
        self._history_limit = BotConfig().history_limit
        self._run_config: BotConfig | None = None
        self._account: AccountSnapshot | None = None
        self._trades: list[Trade] = []
        self._last_price: float | None = None
        self._last_price_symbol: str | None = None
        self._chart_key: tuple[str, str, str] | None = None
        self._chart_data: tuple[StepResult, BotConfig] | None = None   # what the chart shows now
        self._pending_previews: deque[BotConfig] = deque()
        self.step_count = 0             # number of step results received (handy for tests)

        self._wheel_guard = _WheelGuard(self)
        self._build_ui()

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._send_preview)

        self._setup_worker(feed_factory)
        self._set_status("Stopped")
        self._update_controls()
        self._append_log("TraderBot started in PAPER TRADING mode: orders are simulated, "
                         "market data is public, no API keys are used.")
        self._request_initialize.emit(self._db_path)

    # ------------------------------------------------------------------ public helpers
    @property
    def is_ready(self) -> bool:
        """True once the worker has opened the paper account."""
        return self._ready

    @property
    def is_running(self) -> bool:
        """True while the bot is running (as last reported by the worker)."""
        return self._running

    # ------------------------------------------------------------------ worker / thread
    def _setup_worker(self, feed_factory: Callable[[str], MarketDataFeed] | None) -> None:
        self.worker_thread = QThread()
        self.worker_thread.setObjectName("TraderBotEngineThread")
        self.worker = EngineWorker(feed_factory)   # no parent, so it can be moved
        self.worker.moveToThread(self.worker_thread)

        queued = Qt.ConnectionType.QueuedConnection
        # GUI -> worker
        self._request_initialize.connect(self.worker.initialize, queued)
        self._request_start.connect(self.worker.start_bot, queued)
        self._request_stop.connect(self.worker.stop_bot, queued)
        self._request_symbols.connect(self.worker.load_symbols, queued)
        self._request_preview.connect(self.worker.preview, queued)
        self._request_reset.connect(self.worker.reset_account, queued)
        self._request_save_settings.connect(self.worker.save_settings, queued)
        self._request_shutdown.connect(self.worker.shutdown, queued)
        # worker -> GUI
        self.worker.step_done.connect(self._on_step_done, queued)
        self.worker.account_changed.connect(self._on_account_changed, queued)
        self.worker.trades_loaded.connect(self._on_trades_loaded, queued)
        self.worker.symbols_loaded.connect(self._on_symbols_loaded, queued)
        self.worker.preview_ready.connect(self._on_preview_ready, queued)
        self.worker.settings_loaded.connect(self._on_settings_loaded, queued)
        self.worker.initialized.connect(self._on_initialized, queued)
        self.worker.started.connect(self._on_started, queued)
        self.worker.stopped.connect(self._on_stopped, queued)
        self.worker.error.connect(self._on_error, queued)
        self.worker.log.connect(self._append_log, queued)

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_about_to_quit)
        self.worker_thread.start()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(10)

        root.addLayout(self._build_header())
        root.addWidget(self._build_summary_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_settings_panel())

        right = QSplitter(Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        right.addWidget(self._build_chart_panel())
        right.addWidget(self._build_tabs())
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setSizes([520, 300])
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1100])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Paper trading only: public market data, simulated orders, no API keys.")
        self.setStyleSheet(
            "QFrame#statCard { border: 1px solid palette(mid); border-radius: 6px;"
            " background: palette(base); }"
            "QGroupBox { font-weight: 600; margin-top: 14px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }"
        )

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(12)
        title = QLabel("TraderBot")
        title_font = title.font()
        title_font.setPointSizeF(title_font.pointSizeF() * 1.6)
        title_font.setBold(True)
        title.setFont(title_font)
        self.paper_badge = QLabel(PAPER_BADGE_TEXT)
        self.paper_badge.setStyleSheet(
            f"background: {BADGE_COLOR}; color: white; border-radius: 5px;"
            " padding: 4px 12px; font-weight: 700;")
        self.paper_badge.setToolTip("Every order is simulated. No real money, no API keys.")
        header.addWidget(title)
        header.addWidget(self.paper_badge)
        header.addStretch(1)
        return header

    def _build_summary_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.cash_card = _StatCard("Cash")
        self.equity_card = _StatCard("Equity")
        self.total_pnl_card = _StatCard("Total P&L")
        self.realized_pnl_card = _StatCard("Realized P&L")
        self.position_card = _StatCard("Position")
        self.last_price_card = _StatCard("Last price")
        for card in (self.cash_card, self.equity_card, self.total_pnl_card,
                     self.realized_pnl_card, self.position_card, self.last_price_card):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            layout.addWidget(card)

        # Status card: same caption style, but the value is a colored "pill".
        status_card = _StatCard("Status")
        status_card.value.hide()
        self.status_label = QLabel("Stopped")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_card.layout().addWidget(self.status_label)
        layout.addWidget(status_card)
        return bar

    def _make_double_spin(self, minimum: float, maximum: float, decimals: int, step: float,
                          suffix: str = "", tooltip: str = "", off_at_zero: bool = False
                          ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setAlignment(_RIGHT)
        spin.setGroupSeparatorShown(True)
        spin.setToolTip(tooltip)
        if off_at_zero:
            spin.setSpecialValueText("Off")   # shown when the value is 0
        self._guard_wheel(spin)
        return spin

    def _make_int_spin(self, minimum: int, maximum: int, suffix: str = "", tooltip: str = ""
                       ) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setAlignment(_RIGHT)
        spin.setToolTip(tooltip)
        self._guard_wheel(spin)
        return spin

    def _guard_wheel(self, widget: QWidget) -> None:
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(self._wheel_guard)

    @staticmethod
    def _new_form(group: QGroupBox) -> QFormLayout:
        form = QFormLayout(group)
        form.setContentsMargins(10, 8, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(_LEFT)
        return form

    def _build_settings_panel(self) -> QWidget:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(8)

        # --- Market
        market = QGroupBox("Market")
        form = self._new_form(market)
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(SUPPORTED_EXCHANGES)
        self.exchange_combo.setToolTip("Exchange used for PUBLIC price data only.")
        self._guard_wheel(self.exchange_combo)
        self.pair_combo = QComboBox()
        self.pair_combo.setEditable(True)
        self.pair_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.pair_combo.setMaxVisibleItems(20)
        self.pair_combo.setToolTip("Trading pair, e.g. BTC/USDT. Type to search.")
        completer = self.pair_combo.completer()
        if completer is not None:
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._guard_wheel(self.pair_combo)
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(SUPPORTED_TIMEFRAMES)
        self.timeframe_combo.setToolTip("Candle length. The bot acts once per closed candle.")
        self._guard_wheel(self.timeframe_combo)
        form.addRow("Exchange", self.exchange_combo)
        form.addRow("Trading pair", self.pair_combo)
        form.addRow("Timeframe", self.timeframe_combo)
        column.addWidget(market)

        # --- Strategy
        strategy = QGroupBox("Strategy (SMA crossover)")
        form = self._new_form(strategy)
        self.fast_spin = self._make_int_spin(
            1, MAX_MA_PERIOD, tooltip="Number of candles in the fast simple moving average.")
        self.slow_spin = self._make_int_spin(
            2, MAX_MA_PERIOD, tooltip="Number of candles in the slow simple moving average. "
                                      "Must be larger than the fast one.")
        form.addRow("Fast SMA", self.fast_spin)
        form.addRow("Slow SMA", self.slow_spin)
        column.addWidget(strategy)

        # --- Risk
        risk = QGroupBox("Risk")
        form = self._new_form(risk)
        self.size_spin = self._make_double_spin(
            0.01, 100.0, 2, 1.0, " %",
            "Percent of account equity used for each buy (limited by available cash).")
        self.stop_loss_spin = self._make_double_spin(
            0.0, 99.99, 2, 0.5, " %",
            "Sell if the price falls this % below the average cost. 0 = off.", off_at_zero=True)
        self.take_profit_spin = self._make_double_spin(
            0.0, 1000.0, 2, 0.5, " %",
            "Sell if the price rises this % above the average cost. 0 = off.", off_at_zero=True)
        self.daily_loss_spin = self._make_double_spin(
            0.0, 99.99, 2, 0.5, " %",
            "Stop the bot if equity falls this % below its value at the start of the UTC day. "
            "0 = off.", off_at_zero=True)
        form.addRow("Position size", self.size_spin)
        form.addRow("Stop-loss", self.stop_loss_spin)
        form.addRow("Take-profit", self.take_profit_spin)
        form.addRow("Daily loss limit", self.daily_loss_spin)
        column.addWidget(risk)

        # --- Costs & polling
        costs = QGroupBox("Simulation")
        form = self._new_form(costs)
        self.fee_spin = self._make_double_spin(
            0.0, 9.999, 3, 0.01, " %", "Simulated exchange fee charged on every fill.")
        self.slippage_spin = self._make_double_spin(
            0.0, 9.999, 3, 0.01, " %", "Simulated slippage: every fill gets a slightly worse price.")
        self.poll_spin = self._make_int_spin(
            MIN_POLL_SECONDS, 3600, " s",
            f"How often the bot checks prices (minimum {MIN_POLL_SECONDS} s).")
        form.addRow("Fee", self.fee_spin)
        form.addRow("Slippage", self.slippage_spin)
        form.addRow("Poll interval", self.poll_spin)
        column.addWidget(costs)

        # --- Paper account
        account = QGroupBox("Paper account")
        form = self._new_form(account)
        self.balance_spin = self._make_double_spin(
            MIN_ORDER_QUOTE, 1_000_000_000.0, 2, 1000.0, "",
            "Balance used when you press 'Reset paper account'. "
            "Changing it does not affect the current account until you reset.")
        self.reset_button = QPushButton("Reset paper account")
        self.reset_button.setToolTip("Delete all paper trades and positions and start again "
                                     "with the starting balance. Only possible while stopped.")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        form.addRow("Starting balance", self.balance_spin)
        form.addRow(self.reset_button)
        column.addWidget(account)
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)

        # Start / Stop buttons stay visible below the scroll area.
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        for button, color, hover in ((self.start_button, "#16a34a", "#15803d"),
                                     (self.stop_button, "#dc2626", "#b91c1c")):
            button.setMinimumHeight(36)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(
                f"QPushButton {{ background: {color}; color: white; border: none;"
                " border-radius: 6px; font-weight: 700; padding: 6px 14px; }"
                f"QPushButton:hover {{ background: {hover}; }}"
                "QPushButton:disabled { background: palette(mid); color: palette(window); }")
        self.start_button.setToolTip("Start paper trading with these settings.")
        self.stop_button.setToolTip("Stop the bot. Open positions are kept.")
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)

        panel = QWidget()
        # Wide enough for the form (depends on the system font) plus the vertical scroll bar.
        needed = content.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4
        panel.setMinimumWidth(max(290, needed))
        panel.setMaximumWidth(max(460, needed))
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)

        self._settings_inputs: list[QWidget] = [
            self.exchange_combo, self.pair_combo, self.timeframe_combo, self.fast_spin,
            self.slow_spin, self.size_spin, self.stop_loss_spin, self.take_profit_spin,
            self.daily_loss_spin, self.fee_spin, self.slippage_spin, self.poll_spin,
            self.balance_spin,
        ]
        # Fill with defaults until the saved settings arrive from the worker.
        self._apply_settings(BotConfig().to_dict())

        self.exchange_combo.currentTextChanged.connect(self._on_exchange_changed)
        self.pair_combo.currentTextChanged.connect(self._on_market_input_changed)
        self.timeframe_combo.currentTextChanged.connect(self._on_market_input_changed)
        self.fast_spin.valueChanged.connect(self._on_market_input_changed)
        self.slow_spin.valueChanged.connect(self._on_market_input_changed)
        return panel

    def _build_chart_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.chart_title = QLabel("")
        title_font = self.chart_title.font()
        title_font.setBold(True)
        self.chart_title.setFont(title_font)
        self.chart = PriceChart()
        self.chart.setMinimumHeight(260)
        layout.addWidget(self.chart_title)
        layout.addWidget(self.chart, 1)
        return panel

    @staticmethod
    def _make_table(headers: list[str], numeric_columns: set[int]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setHighlightSections(False)
        for column in range(len(headers)):
            item = table.horizontalHeaderItem(column)
            if item is not None:
                item.setTextAlignment(_RIGHT if column in numeric_columns else _LEFT)
        return table

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.positions_table = self._make_table(POSITION_HEADERS, {1, 2, 3, 4, 5})
        self.trades_table = self._make_table(TRADE_HEADERS, {3, 4, 5, 6, 7})
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.tabs.addTab(self.positions_table, "Positions")
        self.tabs.addTab(self.trades_table, "Trade history")
        self.tabs.addTab(self.log_view, "Log")
        return self.tabs

    # ------------------------------------------------------------------ settings <-> widgets
    def _apply_settings(self, data: dict) -> None:
        """Put saved settings into the widgets without triggering previews."""
        config = BotConfig.from_dict(data)
        self._applying_settings = True
        try:
            index = self.exchange_combo.findText(config.exchange_id)
            if index >= 0:
                self.exchange_combo.setCurrentIndex(index)
            self.pair_combo.setEditText(config.symbol)
            index = self.timeframe_combo.findText(config.timeframe)
            if index >= 0:
                self.timeframe_combo.setCurrentIndex(index)
            self.fast_spin.setValue(config.fast_period)
            self.slow_spin.setValue(config.slow_period)
            self.size_spin.setValue(config.position_size_pct)
            self.stop_loss_spin.setValue(config.stop_loss_pct)
            self.take_profit_spin.setValue(config.take_profit_pct)
            self.daily_loss_spin.setValue(config.max_daily_loss_pct)
            self.fee_spin.setValue(config.fee_rate * 100.0)
            self.slippage_spin.setValue(config.slippage_rate * 100.0)
            self.poll_spin.setValue(config.poll_interval_sec)
            self.balance_spin.setValue(config.starting_balance)
            self._history_limit = config.history_limit
        finally:
            self._applying_settings = False

    def _current_symbol_text(self) -> str:
        """The pair typed/selected by the user, matched to the exchange's spelling if possible."""
        text = self.pair_combo.currentText().strip()
        if not text:
            return ""
        index = self.pair_combo.findText(text, Qt.MatchFlag.MatchFixedString)  # case-insensitive
        if index >= 0:
            return self.pair_combo.itemText(index)
        return text.upper()

    def current_config(self) -> BotConfig:
        """Build a BotConfig from the widgets (not validated)."""
        slow = self.slow_spin.value()
        return BotConfig(
            exchange_id=self.exchange_combo.currentText(),
            symbol=self._current_symbol_text(),
            timeframe=self.timeframe_combo.currentText(),
            fast_period=self.fast_spin.value(),
            slow_period=slow,
            starting_balance=self.balance_spin.value(),
            fee_rate=self.fee_spin.value() / 100.0,
            slippage_rate=self.slippage_spin.value() / 100.0,
            position_size_pct=self.size_spin.value(),
            stop_loss_pct=self.stop_loss_spin.value(),
            take_profit_pct=self.take_profit_spin.value(),
            max_daily_loss_pct=self.daily_loss_spin.value(),
            poll_interval_sec=self.poll_spin.value(),
            # Not editable in the GUI; make sure it always covers the slow MA.
            history_limit=max(self._history_limit, slow + 2),
        )

    # ------------------------------------------------------------------ state / controls
    def _set_status(self, status: str, tooltip: str = "") -> None:
        color = STATUS_COLORS.get(status, STATUS_COLORS["Stopped"])
        self.status_label.setText(status)
        self.status_label.setToolTip(tooltip)
        self.status_label.setStyleSheet(
            f"background: {color}; color: white; border-radius: 10px;"
            " padding: 3px 14px; font-weight: 700;")

    def _update_controls(self) -> None:
        idle = self._ready and not (self._running or self._starting or self._stopping)
        for widget in self._settings_inputs:
            widget.setEnabled(idle)
        self.reset_button.setEnabled(idle)
        self.start_button.setEnabled(idle)
        self.stop_button.setEnabled(self._running and not self._stopping)

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for line in str(message).splitlines() or [""]:
            self.log_view.appendPlainText(f"{stamp}  {line}")

    def _show_message(self, icon: QMessageBox.Icon, title: str, text: str) -> None:
        """Non-blocking message box (never freezes the event loop)."""
        box = QMessageBox(icon, title, text, QMessageBox.StandardButton.Ok, self)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.open()

    def _confirm(self, title: str, text: str) -> bool:
        """Ask a yes/no question (blocking). Tests replace this method."""
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ user actions
    @Slot()
    def _on_start_clicked(self) -> None:
        if not self._ready or self._running or self._starting or self._stopping:
            return
        config = self.current_config()
        errors = config.validate()
        if errors:
            self._show_message(QMessageBox.Icon.Warning, "Please check the settings",
                               "\n".join(f"• {error}" for error in errors))
            return
        self._preview_timer.stop()
        self._run_config = config
        self._starting = True
        self._set_status("Starting…")
        self._update_controls()
        self._request_start.emit(config)

    @Slot()
    def _on_stop_clicked(self) -> None:
        if not self._running or self._stopping:
            return
        self._stopping = True
        self._set_status("Stopping…")
        self._update_controls()
        self._request_stop.emit()

    @Slot()
    def _on_reset_clicked(self) -> None:
        if not self._ready or self._running or self._starting or self._stopping:
            return
        balance = self.balance_spin.value()
        if self._confirm(
                "Reset paper account",
                f"Delete all paper trades and positions and start again with "
                f"{format_money(balance)}?\n\nThis cannot be undone."):
            self._request_reset.emit(float(balance))

    @Slot(str)
    def _on_exchange_changed(self, exchange_id: str) -> None:
        if self._applying_settings or not self._ready:
            return
        self._request_symbols.emit(exchange_id)
        self._schedule_preview()

    @Slot()
    def _on_market_input_changed(self, *_args: object) -> None:
        if self._applying_settings:
            return
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if not self._ready or self._running or self._starting or self._stopping:
            return
        self._preview_timer.start()   # (re)start the debounce timer

    @Slot()
    def _send_preview(self) -> None:
        if not self._ready or self._running or self._starting or self._stopping:
            return
        config = self.current_config()
        if config.validate():
            return  # incomplete input (e.g. half-typed pair); wait for more typing
        self._pending_previews.append(config)
        self._request_preview.emit(config)

    # ------------------------------------------------------------------ worker callbacks
    @Slot(object)
    def _on_settings_loaded(self, data: dict) -> None:
        self._apply_settings(data)

    @Slot(bool)
    def _on_initialized(self, ok: bool) -> None:
        self._ready = ok
        self._update_controls()
        if not ok:
            self._set_status("Error", "The paper account database could not be opened.")
            return
        self._request_symbols.emit(self.exchange_combo.currentText())
        self._send_preview()

    @Slot(object)
    def _on_account_changed(self, snapshot: AccountSnapshot) -> None:
        self._account = snapshot
        self._refresh_summary()
        self._refresh_positions()

    @Slot(object)
    def _on_trades_loaded(self, trades: list[Trade]) -> None:
        self._trades = list(trades)[-TRADE_HISTORY_LIMIT:]
        self._refresh_trades_table()
        if self._chart_data is not None:
            # Redraw so the trade markers match the new history (e.g. after a reset).
            self._show_result_on_chart(*self._chart_data)

    @Slot(str, object)
    def _on_symbols_loaded(self, exchange_id: str, symbols: list[str]) -> None:
        if exchange_id != self.exchange_combo.currentText():
            return  # answer for an exchange the user already switched away from
        text = self.pair_combo.currentText()
        self._applying_settings = True
        try:
            self.pair_combo.clear()
            self.pair_combo.addItems(list(symbols))
            self.pair_combo.setEditText(text)
        finally:
            self._applying_settings = False
        self._append_log(f"Loaded {len(symbols):,} trading pairs from {exchange_id}.")

    @Slot(object)
    def _on_preview_ready(self, result: StepResult) -> None:
        config = self._pending_previews.popleft() if self._pending_previews else None
        if self._pending_previews or config is None:
            return  # a newer preview request is on its way; ignore this stale answer
        if self._running or self._starting:
            return
        if result.error:
            if result.error != PREVIEW_SKIPPED_RUNNING:
                self._append_log(f"Chart preview: {result.error}")
            self.chart.clear()
            self._chart_key = None
            self._chart_data = None
            self.chart_title.setText(f"{config.symbol} · {config.timeframe} · {config.exchange_id}"
                                     " — no data")
            return
        self._show_result_on_chart(result, config)

    @Slot()
    def _on_started(self) -> None:
        self._starting = False
        self._running = True
        self._set_status("Running")
        self._update_controls()

    @Slot(str)
    def _on_stopped(self, reason: str) -> None:
        was_starting = self._starting
        self._running = False
        self._starting = False
        self._stopping = False
        if reason == STOP_REASON_HALTED:
            self._set_status("Halted", reason)
        elif reason.startswith(STOP_REASON_START_FAILED) or was_starting:
            self._set_status("Error", reason)
        else:
            self._set_status("Stopped", reason)
        self._update_controls()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        if not (self._close_pending or self._shutdown_complete):
            self._show_message(QMessageBox.Icon.Warning, "TraderBot", message)

    @Slot(object)
    def _on_step_done(self, result: StepResult) -> None:
        self.step_count += 1
        config = self._run_config
        if result.new_trades:
            self._trades.extend(result.new_trades)
            self._trades = self._trades[-TRADE_HISTORY_LIMIT:]
            self._refresh_trades_table()
        if result.last_price is not None:
            self._last_price = result.last_price
            self._last_price_symbol = result.symbol
        if config is not None and result.candles is not None:
            self._show_result_on_chart(result, config)
        if self._running and not self._stopping:
            if result.error:
                self._set_status("Error", f"Last check failed (the bot keeps trying): {result.error}")
            else:
                self._set_status("Running")
        self._refresh_summary()
        self._refresh_positions()

    # ------------------------------------------------------------------ rendering
    def _show_result_on_chart(self, result: StepResult, config: BotConfig) -> None:
        if result.last_price is not None:
            self._last_price = result.last_price
            self._last_price_symbol = config.symbol
        self._chart_data = (result, config)
        trades = [trade for trade in self._trades if trade.symbol == config.symbol]
        self.chart.set_data(result.candles, trades, result.last_price,
                            config.fast_period, config.slow_period)
        key = (config.exchange_id, config.symbol, config.timeframe)
        if key != self._chart_key:
            self._chart_key = key
            self.chart.reset_view()
        self.chart_title.setText(f"{config.symbol} · {config.timeframe} · {config.exchange_id}")
        self._refresh_summary()
        self._refresh_positions()

    def _display_symbol(self) -> str:
        if (self._running or self._starting) and self._run_config is not None:
            return self._run_config.symbol
        return self._current_symbol_text()

    def _refresh_summary(self) -> None:
        account = self._account
        symbol = self._display_symbol()
        last_price = self._last_price if self._last_price_symbol == symbol else None
        self.last_price_card.set_value(format_price(last_price))
        if account is None:
            for card in (self.cash_card, self.equity_card, self.total_pnl_card,
                         self.realized_pnl_card, self.position_card):
                card.set_value("—")
            return
        self.cash_card.set_value(format_money(account.cash))
        self.equity_card.set_value(format_money(account.equity),
                                   tooltip="Cash plus open positions at the last price.")
        total = account.total_pnl
        self.total_pnl_card.set_value(
            f"{format_money(total, signed=True)} ({account.total_pnl_pct:+.2f}%)", pnl_color(total),
            tooltip=f"Compared with the starting balance of {format_money(account.starting_balance)}")
        realized = account.realized_pnl_total
        self.realized_pnl_card.set_value(format_money(realized, signed=True), pnl_color(realized))

        position = account.positions.get(symbol)
        others = len(account.positions) - (1 if position is not None else 0)
        if position is not None:
            base = symbol.split("/")[0]
            text = f"{format_amount(position.amount)} {base} @ {format_price(position.avg_cost)}"
        else:
            text = "None"
        if others > 0:
            text += f"  (+{others} other)"
        self.position_card.set_value(text, tooltip="Amount held @ average cost (incl. buy fees)")

    @staticmethod
    def _cell(text: str, align: Qt.AlignmentFlag = _RIGHT, color: str | None = None
              ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        if color:
            item.setForeground(QColor(color))
        return item

    def _refresh_positions(self) -> None:
        positions = sorted(self._account.positions.values(), key=lambda p: p.symbol) \
            if self._account is not None else []
        table = self.positions_table
        table.setRowCount(len(positions))
        for row, pos in enumerate(positions):
            last = self._last_price if pos.symbol == self._last_price_symbol else None
            value = pos.amount * last if last is not None else None
            unrealized = (value - pos.amount * pos.avg_cost) if value is not None else None
            table.setItem(row, 0, self._cell(pos.symbol, _LEFT))
            table.setItem(row, 1, self._cell(format_amount(pos.amount)))
            table.setItem(row, 2, self._cell(format_price(pos.avg_cost)))
            table.setItem(row, 3, self._cell(format_price(last)))
            table.setItem(row, 4, self._cell(format_money(value)))
            table.setItem(row, 5, self._cell(format_money(unrealized, signed=True), _RIGHT,
                                             pnl_color(unrealized)))

    def _refresh_trades_table(self) -> None:
        table = self.trades_table
        table.setUpdatesEnabled(False)
        try:
            trades = list(reversed(self._trades))   # newest first
            table.setRowCount(len(trades))
            for row, trade in enumerate(trades):
                when = datetime.fromtimestamp(trade.timestamp / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                is_buy = trade.side == Side.BUY
                side_color = POSITIVE_COLOR if is_buy else NEGATIVE_COLOR
                table.setItem(row, 0, self._cell(when, _LEFT))
                table.setItem(row, 1, self._cell(trade.symbol, _LEFT))
                side_item = self._cell(trade.side.value, _LEFT, side_color)
                bold = QFont(side_item.font())
                bold.setBold(True)
                side_item.setFont(bold)
                table.setItem(row, 2, side_item)
                table.setItem(row, 3, self._cell(format_price(trade.price)))
                table.setItem(row, 4, self._cell(format_amount(trade.amount)))
                table.setItem(row, 5, self._cell(format_fee(trade.fee)))
                table.setItem(row, 6, self._cell(format_money(trade.quote_value)))
                if is_buy:
                    table.setItem(row, 7, self._cell("—"))
                else:
                    table.setItem(row, 7, self._cell(format_money(trade.realized_pnl, signed=True),
                                                     _RIGHT, pnl_color(trade.realized_pnl)))
                table.setItem(row, 8, self._cell(REASON_TEXT.get(trade.reason, str(trade.reason)),
                                                 _LEFT))
        finally:
            table.setUpdatesEnabled(True)

    # ------------------------------------------------------------------ shutdown
    def _request_worker_shutdown(self) -> bool:
        """Ask the worker to stop and close the database; wait up to 5 s. True if it finished."""
        self._preview_timer.stop()
        if self.worker_thread.isFinished() or not self.worker_thread.isRunning():
            return True
        if self._ready:
            self._request_save_settings.emit(self.current_config().to_dict())
        # The worker quits its own thread at the end of shutdown(), so every request queued
        # before this one is still handled first.
        self._request_shutdown.emit()
        return self.worker_thread.wait(SHUTDOWN_WAIT_MS)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        if self._shutdown_complete:
            event.accept()
            return
        if self._close_pending:
            event.ignore()   # already waiting for the worker to finish
            return
        if (self._running or self._starting) and not self._confirm(
                "Quit TraderBot", "The bot is running. Stop the bot and quit?"):
            event.ignore()
            return
        if self._request_worker_shutdown():
            self._shutdown_complete = True
            event.accept()
            return
        # The worker is still busy (e.g. a slow network request). Hide the window and close
        # for real once the thread has finished, so the thread is never destroyed while running.
        logger.warning("Worker thread still busy after %d ms; closing when it finishes",
                       SHUTDOWN_WAIT_MS)
        self._close_pending = True
        self.worker_thread.finished.connect(self._on_worker_finished_after_close,
                                            Qt.ConnectionType.QueuedConnection)
        if self.worker_thread.isFinished():   # finished just before we connected
            QTimer.singleShot(0, self._on_worker_finished_after_close)
        self.hide()
        event.ignore()

    @Slot()
    def _on_worker_finished_after_close(self) -> None:
        self._shutdown_complete = True
        self._close_pending = False
        self.close()
        if not any(widget.isVisible() for widget in QApplication.topLevelWidgets()):
            QApplication.quit()

    @Slot()
    def _on_about_to_quit(self) -> None:
        """Last-chance clean shutdown if the application quits without closing the window."""
        if self._shutdown_complete:
            return
        if self._request_worker_shutdown():
            self._shutdown_complete = True
        else:
            logger.warning("Worker thread did not finish before the application quit")

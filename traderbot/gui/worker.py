"""Background worker that owns the trading engine.

The :class:`EngineWorker` object is moved to its own ``QThread``. Every Storage / PaperBroker /
MarketDataFeed / BotEngine call happens in that thread, so slow network requests never freeze
the window. The GUI talks to the worker ONLY through queued signals, and the worker reports
back through its own signals (Qt delivers them to the GUI thread).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal, Slot

from traderbot.broker.paper import PaperBroker
from traderbot.config import MIN_ORDER_QUOTE, BotConfig
from traderbot.data.feed import DataFeedError, MarketDataFeed
from traderbot.engine import BotEngine, EngineError
from traderbot.models import StepResult
from traderbot.storage import Storage
from traderbot.strategy.ma_cross import MACrossoverStrategy

logger = logging.getLogger(__name__)

# Reasons sent with the `stopped` signal (the GUI uses them to pick the status text).
STOP_REASON_USER = "Stopped by user"
STOP_REASON_HALTED = "Daily loss limit reached"
STOP_REASON_START_FAILED = "Could not start"
STOP_REASON_SHUTDOWN = "Shutting down"

# Error text of a preview that was not run because the bot is running (not a real problem).
PREVIEW_SKIPPED_RUNNING = "Preview skipped: the bot is running"

# How many trades to load into the GUI's history table at most.
TRADE_HISTORY_LIMIT = 2000


def _now_ms() -> int:
    return int(time.time() * 1000)


class EngineWorker(QObject):
    """Runs the bot in a worker thread. Create without a parent, then ``moveToThread``."""

    step_done = Signal(object)          # StepResult
    account_changed = Signal(object)    # AccountSnapshot
    trades_loaded = Signal(object)      # list[Trade], oldest -> newest
    symbols_loaded = Signal(str, object)  # exchange_id, list[str]
    preview_ready = Signal(object)      # StepResult (candles + last price, no trading)
    settings_loaded = Signal(object)    # dict (BotConfig.to_dict())
    initialized = Signal(bool)          # emitted last by initialize(): True on success
    started = Signal()
    stopped = Signal(str)               # reason
    error = Signal(str)
    log = Signal(str)

    def __init__(self, feed_factory: Callable[[str], MarketDataFeed] | None = None) -> None:
        super().__init__()  # no parent: required for moveToThread
        self._feed_factory: Callable[[str], MarketDataFeed] = feed_factory or MarketDataFeed
        self._feeds: dict[str, MarketDataFeed] = {}
        self._storage: Storage | None = None
        self._broker: PaperBroker | None = None
        self._engine: BotEngine | None = None
        self._timer: QTimer | None = None
        self._running = False
        self._stepping = False

    # ------------------------------------------------------------------ helpers
    def _get_feed(self, exchange_id: str) -> MarketDataFeed:
        """One MarketDataFeed per exchange id, created on first use (may raise DataFeedError)."""
        feed = self._feeds.get(exchange_id)
        if feed is None:
            feed = self._feed_factory(exchange_id)
            self._feeds[exchange_id] = feed
        return feed

    def _ensure_timer(self) -> QTimer:
        """Create the poll timer lazily, so it is created in (and belongs to) the worker thread."""
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setSingleShot(False)
            self._timer.timeout.connect(self._on_timer)
        return self._timer

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _fail_start(self, message: str) -> None:
        self._running = False
        self._engine = None
        self._stop_timer()
        self.error.emit(message)
        self.stopped.emit(f"{STOP_REASON_START_FAILED}: {message}")

    # ------------------------------------------------------------------ slots
    @Slot(str)
    def initialize(self, db_path: str) -> None:
        """Open the database, create the paper broker and send the saved state to the GUI."""
        try:
            if self._storage is not None:
                self._storage.close()
            self._storage = Storage(db_path)
            saved = self._storage.load_settings()
            config = BotConfig.from_dict(saved) if saved else BotConfig()
            self._broker = PaperBroker(self._storage, config.fee_rate, config.slippage_rate,
                                       config.starting_balance)
            self.settings_loaded.emit(config.to_dict())
            self.account_changed.emit(self._broker.snapshot({}))
            self.trades_loaded.emit(self._storage.load_trades(limit=TRADE_HISTORY_LIMIT))
            self.log.emit(f"Paper account loaded ({Path(db_path).name}).")
            self.initialized.emit(True)
        except Exception as exc:  # noqa: BLE001 - report anything to the GUI instead of crashing
            logger.exception("Initialisation failed")
            if self._storage is not None:
                try:
                    self._storage.close()
                except Exception:  # noqa: BLE001
                    logger.exception("Error while closing the database")
            self._storage = None
            self._broker = None
            self.error.emit(f"Could not open the paper account database: {exc}")
            self.initialized.emit(False)

    @Slot(object)
    def start_bot(self, config: BotConfig) -> None:
        """Validate settings, build the engine, start polling and run the first step now."""
        if self._running:
            self.log.emit("The bot is already running.")
            return
        if self._storage is None or self._broker is None:
            self._fail_start("The paper account is not ready.")
            return
        errors = config.validate()
        if errors:
            self._fail_start("Invalid settings:\n" + "\n".join(errors))
            return
        try:
            self._storage.save_settings(config.to_dict())
            self._broker.set_costs(config.fee_rate, config.slippage_rate)
            feed = self._get_feed(config.exchange_id)
            engine = BotEngine(config, feed, self._broker, self._storage)
            messages = engine.start()
        except (EngineError, DataFeedError, ValueError) as exc:
            self._fail_start(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while starting the bot")
            self._fail_start(f"Unexpected error: {exc}")
            return

        self._engine = engine
        self._running = True
        for message in messages:
            self.log.emit(message)
        timer = self._ensure_timer()
        timer.setInterval(int(config.poll_interval_sec) * 1000)
        timer.start()
        self.log.emit(f"Checking prices every {config.poll_interval_sec} s.")
        self.started.emit()
        self._run_step()  # first step immediately

    @Slot()
    def stop_bot(self) -> None:
        """Stop polling. Open positions are kept."""
        if not self._running:
            return
        self._stop_timer()
        self._running = False
        self._engine = None
        self.log.emit("Bot stopped. Open positions are kept.")
        self.stopped.emit(STOP_REASON_USER)

    @Slot(str)
    def load_symbols(self, exchange_id: str) -> None:
        """Fetch the list of tradable spot pairs for an exchange (public data)."""
        try:
            symbols = self._get_feed(exchange_id).list_symbols()
        except DataFeedError as exc:
            self.log.emit(f"Could not load trading pairs for {exchange_id}: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while loading symbols")
            self.log.emit(f"Could not load trading pairs for {exchange_id}: {exc}")
            return
        self.symbols_loaded.emit(exchange_id, list(symbols))

    @Slot(object)
    def preview(self, config: BotConfig) -> None:
        """Fetch candles + last price once and compute the MAs for the chart. Never trades.

        Always emits ``preview_ready`` exactly once (with ``error`` set if it failed), so the
        GUI can match answers to requests.
        """
        result = StepResult(timestamp=_now_ms(), symbol=config.symbol, timeframe=config.timeframe)
        if self._running:
            result.error = PREVIEW_SKIPPED_RUNNING
            self.preview_ready.emit(result)
            return
        try:
            errors = config.validate()
            if errors:
                result.error = errors[0]
            else:
                strategy = MACrossoverStrategy(config.fast_period, config.slow_period)
                feed = self._get_feed(config.exchange_id)
                limit = max(config.history_limit, strategy.min_candles + 1)
                candles = feed.fetch_closed_candles(config.symbol, config.timeframe, limit=limit,
                                                    now_ms=result.timestamp)
                last_price = feed.fetch_last_price(config.symbol)
                result.candles = strategy.compute(candles)
                result.last_price = last_price
                if self._broker is not None:
                    result.account = self._broker.snapshot({config.symbol: last_price})
        except (DataFeedError, ValueError) as exc:
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during preview")
            result.error = f"Unexpected error: {exc}"
        self.preview_ready.emit(result)
        if result.account is not None:
            self.account_changed.emit(result.account)

    @Slot(float)
    def reset_account(self, starting_balance: float) -> None:
        """Wipe trades/positions and start again with `starting_balance` (only while stopped)."""
        if self._running:
            self.error.emit("Stop the bot before resetting the paper account.")
            return
        if self._broker is None or self._storage is None:
            self.error.emit("The paper account is not ready.")
            return
        if not starting_balance >= MIN_ORDER_QUOTE:
            self.error.emit(f"Starting balance must be at least {MIN_ORDER_QUOTE}.")
            return
        try:
            self._broker.reset(starting_balance)
            settings = self._storage.load_settings() or BotConfig().to_dict()
            settings["starting_balance"] = float(starting_balance)
            self._storage.save_settings(settings)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reset failed")
            self.error.emit(f"Could not reset the paper account: {exc}")
            return
        self.account_changed.emit(self._broker.snapshot({}))
        self.trades_loaded.emit(self._storage.load_trades(limit=TRADE_HISTORY_LIMIT))
        self.log.emit(f"Paper account reset to {starting_balance:,.2f}.")

    @Slot(object)
    def save_settings(self, settings: dict) -> None:
        """Remember the GUI settings (used when the window closes without starting the bot)."""
        if self._storage is None:
            return
        try:
            self._storage.save_settings(dict(settings))
        except Exception:  # noqa: BLE001
            logger.exception("Could not save settings")

    @Slot()
    def shutdown(self) -> None:
        """Stop the timer, close the database and end the worker thread's event loop."""
        was_running = self._running
        self._running = False
        self._engine = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None  # deleted here, in the thread that owns it
        if was_running:
            self.stopped.emit(STOP_REASON_SHUTDOWN)
        if self._storage is not None:
            try:
                self._storage.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error while closing the database")
        self._storage = None
        self._broker = None
        self._feeds.clear()
        # End the worker thread's event loop from inside the thread, so this always happens
        # AFTER every request that was queued before shutdown. Never quit the GUI thread.
        thread = self.thread()
        app = QCoreApplication.instance()
        if thread is not None and (app is None or thread != app.thread()):
            thread.quit()

    # ------------------------------------------------------------------ polling
    @Slot()
    def _on_timer(self) -> None:
        self._run_step()

    def _run_step(self) -> None:
        """Run one engine step and report it. Never raises."""
        if not self._running or self._engine is None:
            return
        if self._stepping:
            logger.info("Previous step still running; skipping this tick")
            return
        self._stepping = True
        try:
            result = self._engine.step()
        except Exception as exc:  # noqa: BLE001 - step() should never raise, but be safe
            logger.exception("Engine step raised")
            engine = self._engine
            result = StepResult(timestamp=_now_ms(), symbol=engine.config.symbol,
                                timeframe=engine.config.timeframe, error=f"Unexpected error: {exc}")
        finally:
            self._stepping = False

        for message in result.messages:
            self.log.emit(message)
        if result.error:
            self.log.emit(f"Problem: {result.error}")
        self.step_done.emit(result)
        if result.account is not None:
            self.account_changed.emit(result.account)

        if result.halted:
            self._stop_timer()
            self._running = False
            self._engine = None
            self.log.emit("Bot halted: daily loss limit reached.")
            self.stopped.emit(STOP_REASON_HALTED)

    @property
    def is_running(self) -> bool:
        """True while the bot is polling (read only from the worker thread or after it ended)."""
        return self._running


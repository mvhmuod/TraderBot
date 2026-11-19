"""SQLite persistence for the paper account, positions, trade history and settings.

Everything lives in one small SQLite file (``data/traderbot.db`` by default).
Each public method takes a re-entrant lock, so a single ``Storage`` object can be
shared safely between threads. Writes that must happen together (for example a
trade plus the matching cash and position change) run inside ONE transaction:
either all of them are saved or none of them are.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from traderbot.models import Position, Side, Trade, TradeReason

logger = logging.getLogger(__name__)

# Meta keys starting with this prefix hold the equity at the start of a UTC day.
DAY_START_PREFIX = "day_start:"
SETTINGS_KEY = "settings"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL,
    starting_balance REAL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    amount REAL,
    avg_cost REAL,
    opened_at INTEGER
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    symbol TEXT,
    side TEXT,
    price REAL,
    amount REAL,
    fee REAL,
    quote_value REAL,
    realized_pnl REAL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_TRADE_COLUMNS = "id, timestamp, symbol, side, price, amount, fee, quote_value, realized_pnl, reason"


class Storage:
    """Thread-safe SQLite store for the paper account.

    Pass ``":memory:"`` as ``db_path`` for a throw-away in-memory database (tests).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # autocommit=True: SQLite commits each statement on its own unless we
        # explicitly open a transaction with BEGIN (see _transaction below).
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, autocommit=True)
        with self._lock:
            self._conn.executescript(_SCHEMA)
        logger.debug("Storage opened at %s", self._db_path)

    # ------------------------------------------------------------------ helpers
    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the enclosed statements as ONE transaction (all-or-nothing)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @staticmethod
    def _row_to_trade(row: tuple) -> Trade:
        (trade_id, timestamp, symbol, side, price, amount, fee, quote_value, realized_pnl, reason) = row
        return Trade(
            timestamp=int(timestamp),
            symbol=symbol,
            side=Side(side),
            price=float(price),
            amount=float(amount),
            fee=float(fee),
            quote_value=float(quote_value),
            realized_pnl=float(realized_pnl),
            reason=TradeReason(reason),
            id=int(trade_id),
        )

    def close(self) -> None:
        """Close the database connection. The object must not be used afterwards."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ account
    def load_account(self) -> tuple[float, float] | None:
        """Return ``(cash, starting_balance)``, or ``None`` if the account was never initialised."""
        with self._lock:
            row = self._conn.execute(
                "SELECT cash, starting_balance FROM account WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return float(row[0]), float(row[1])

    def reset_account(self, starting_balance: float) -> None:
        """Start a fresh paper account in ONE transaction.

        Deletes all trades, all positions and every ``day_start:*`` meta row, then
        sets ``cash = starting_balance = starting_balance``. Other meta rows
        (such as saved settings) are kept.
        """
        balance = float(starting_balance)
        with self._transaction() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM positions")
            # GLOB (not LIKE): "_" is a wildcard in LIKE but a plain character in GLOB.
            conn.execute("DELETE FROM meta WHERE key GLOB ?", (DAY_START_PREFIX + "*",))
            conn.execute(
                "INSERT INTO account (id, cash, starting_balance) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET cash = excluded.cash, "
                "starting_balance = excluded.starting_balance",
                (balance, balance),
            )
        logger.info("Paper account reset to %s", balance)

    def load_positions(self) -> dict[str, Position]:
        """Return all open positions keyed by symbol."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, amount, avg_cost, opened_at FROM positions ORDER BY symbol"
            ).fetchall()
        return {
            symbol: Position(symbol=symbol, amount=float(amount), avg_cost=float(avg_cost),
                             opened_at=int(opened_at))
            for symbol, amount, avg_cost, opened_at in rows
        }

    # ------------------------------------------------------------------- trades
    def load_trades(self, symbol: str | None = None, limit: int | None = None) -> list[Trade]:
        """Return trades ordered oldest -> newest (ascending id).

        ``symbol`` filters to one pair. ``limit`` keeps only the NEWEST ``limit``
        trades (still returned oldest -> newest); ``limit <= 0`` returns ``[]``.
        """
        if limit is not None and limit <= 0:
            return []
        sql = f"SELECT {_TRADE_COLUMNS} FROM trades"
        params: list[Any] = []
        if symbol is not None:
            sql += " WHERE symbol = ?"
            params.append(symbol)
        if limit is not None:
            # Take the newest N (descending), then flip them back to ascending order.
            sql = f"SELECT * FROM ({sql} ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
            params.append(int(limit))
        else:
            sql += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_trade(row) for row in rows]

    def realized_pnl_total(self) -> float:
        """Sum of ``realized_pnl`` over all trades (0.0 when there are none)."""
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(realized_pnl), 0.0) FROM trades").fetchone()
        return float(row[0])

    def record_fill(self, trade: Trade, new_cash: float, position: Position | None, symbol: str) -> Trade:
        """Save one fill atomically and return the trade with its database ``id``.

        In ONE transaction: insert ``trade``, set the account cash to ``new_cash``,
        and upsert the position for ``symbol`` (or delete it when ``position`` is
        ``None``). Any error rolls everything back and is re-raised.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO trades (timestamp, symbol, side, price, amount, fee, quote_value, "
                "realized_pnl, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(trade.timestamp),
                    trade.symbol,
                    Side(trade.side).value,
                    float(trade.price),
                    float(trade.amount),
                    float(trade.fee),
                    float(trade.quote_value),
                    float(trade.realized_pnl),
                    TradeReason(trade.reason).value,
                ),
            )
            trade_id = cursor.lastrowid
            updated = conn.execute("UPDATE account SET cash = ? WHERE id = 1", (float(new_cash),))
            if updated.rowcount != 1:
                raise RuntimeError("Paper account is not initialised; call reset_account() first")
            if position is None:
                conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                conn.execute(
                    "INSERT INTO positions (symbol, amount, avg_cost, opened_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(symbol) DO UPDATE SET amount = excluded.amount, "
                    "avg_cost = excluded.avg_cost, opened_at = excluded.opened_at",
                    (symbol, float(position.amount), float(position.avg_cost), int(position.opened_at)),
                )
        return dataclasses.replace(trade, id=int(trade_id))

    # --------------------------------------------------------------------- meta
    def get_meta(self, key: str, default: Any = None) -> Any:
        """Return the JSON-decoded value stored under ``key``, or ``default`` if missing."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def set_meta(self, key: str, value: Any) -> None:
        """Store ``value`` (must be JSON-serialisable) under ``key``, replacing any old value."""
        encoded = json.dumps(value)
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, encoded),
            )

    def save_settings(self, data: dict) -> None:
        """Save bot settings (``BotConfig.to_dict()``) under the meta key ``"settings"``."""
        self.set_meta(SETTINGS_KEY, dict(data))

    def load_settings(self) -> dict | None:
        """Return the saved settings dict, or ``None`` if nothing was saved yet."""
        value = self.get_meta(SETTINGS_KEY)
        return value if isinstance(value, dict) else None

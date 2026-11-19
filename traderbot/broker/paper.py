"""Simulated (paper) broker. It NEVER talks to an exchange and NEVER places real orders.

Orders are "filled" instantly at the given price adjusted for slippage, a fee is
charged, and the result is saved through ``Storage.record_fill`` in one atomic
transaction. In-memory state is only updated after the database write succeeds,
so memory and disk can never disagree.
"""
from __future__ import annotations

import logging
import math
import threading

from traderbot.config import MIN_ORDER_QUOTE
from traderbot.models import AccountSnapshot, Position, Side, Trade, TradeReason
from traderbot.storage import Storage

logger = logging.getLogger(__name__)

# Tolerance for floating-point comparisons where rounding matters.
EPS = 1e-9
# A position whose remaining amount is at or below this is treated as fully closed.
DUST_AMOUNT = 1e-12


class BrokerError(Exception):
    """An order could not be executed (message is safe to show to the user)."""


class InsufficientFunds(BrokerError):
    """A buy needs more cash than the paper account has."""


class NoPosition(BrokerError):
    """A sell was requested for a symbol that is not held."""


def _check_rate(name: str, value: float) -> float:
    """Validate a fee/slippage rate: a finite number with 0 <= rate < 1."""
    rate = float(value)
    if not math.isfinite(rate) or not (0.0 <= rate < 1.0):
        raise ValueError(f"{name} must be between 0 and 1 (got {value!r})")
    return rate


def _check_price(price: float) -> float:
    """Validate a market price: a finite number above zero."""
    value = float(price)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Price must be above 0 (got {price!r})")
    return value


class PaperBroker:
    """Long-only spot paper broker backed by :class:`Storage`.

    If the storage has no account yet it is created with ``starting_balance``;
    otherwise the stored cash/positions are kept and ``starting_balance`` is ignored.
    """

    def __init__(self, storage: Storage, fee_rate: float, slippage_rate: float,
                 starting_balance: float) -> None:
        self._lock = threading.RLock()
        self._storage = storage
        self._fee_rate = _check_rate("Fee rate", fee_rate)
        self._slippage_rate = _check_rate("Slippage rate", slippage_rate)
        with self._lock:
            if storage.load_account() is None:
                storage.reset_account(starting_balance)
            self._load_state()

    def _load_state(self) -> None:
        """(Re)load cash, starting balance and positions from storage."""
        account = self._storage.load_account()
        if account is None:  # should be impossible right after reset_account
            raise BrokerError("Paper account could not be loaded")
        self._cash, self._starting_balance = account
        self._positions: dict[str, Position] = self._storage.load_positions()

    # --------------------------------------------------------------- settings
    def set_costs(self, fee_rate: float, slippage_rate: float) -> None:
        """Change the fee and slippage rates used for future fills (fractions, e.g. 0.001 = 0.1%)."""
        fee = _check_rate("Fee rate", fee_rate)
        slippage = _check_rate("Slippage rate", slippage_rate)
        with self._lock:
            self._fee_rate = fee
            self._slippage_rate = slippage

    # ------------------------------------------------------------------ state
    @property
    def cash(self) -> float:
        """Quote units available to spend."""
        with self._lock:
            return self._cash

    @property
    def starting_balance(self) -> float:
        """Quote units the account was (re)set with."""
        with self._lock:
            return self._starting_balance

    def get_position(self, symbol: str) -> Position | None:
        """Return the open position for ``symbol``, or ``None``."""
        with self._lock:
            return self._positions.get(symbol)

    def positions(self) -> dict[str, Position]:
        """Return a copy of all open positions keyed by symbol."""
        with self._lock:
            return dict(self._positions)

    def equity(self, mark_prices: dict[str, float]) -> float:
        """Cash plus every position valued at its mark price (avg_cost if no mark is given).

        Mark prices are used as-is: no fees or slippage are deducted.
        """
        with self._lock:
            total = self._cash
            for symbol, pos in self._positions.items():
                total += pos.amount * mark_prices.get(symbol, pos.avg_cost)
            return total

    def snapshot(self, mark_prices: dict[str, float]) -> AccountSnapshot:
        """Return a read-only copy of the account (safe to hand to the GUI thread)."""
        with self._lock:
            return AccountSnapshot(
                cash=self._cash,
                starting_balance=self._starting_balance,
                positions=dict(self._positions),
                equity=self.equity(mark_prices),
                realized_pnl_total=self._storage.realized_pnl_total(),
            )

    # ----------------------------------------------------------------- orders
    def market_buy(self, symbol: str, quote_amount: float, price: float, timestamp: int,
                   reason: TradeReason) -> Trade:
        """Spend ``quote_amount`` of cash (fee included) to buy ``symbol`` at ``price`` + slippage.

        Raises ValueError for a bad price or an amount below MIN_ORDER_QUOTE, and
        InsufficientFunds if ``quote_amount`` exceeds the available cash.
        """
        price = _check_price(price)
        quote = float(quote_amount)
        if not math.isfinite(quote) or quote < MIN_ORDER_QUOTE:
            raise ValueError(f"Order size must be at least {MIN_ORDER_QUOTE} (got {quote_amount!r})")
        with self._lock:
            if quote > self._cash + EPS:
                raise InsufficientFunds(
                    f"Not enough cash: order needs {quote:,.2f} but only {self._cash:,.2f} is available"
                )
            if quote > self._cash:
                quote = self._cash  # within rounding tolerance -> spend exactly what is left

            fill_price = price * (1 + self._slippage_rate)   # buying: pay slightly more
            fee = quote * self._fee_rate                     # fee is taken out of the quote spent
            amount = (quote - fee) / fill_price
            new_cash = self._cash - quote

            old = self._positions.get(symbol)
            if old is None:
                # avg_cost includes the fee, so it is the break-even price.
                new_position = Position(symbol=symbol, amount=amount, avg_cost=quote / amount,
                                        opened_at=int(timestamp))
            else:
                total_amount = old.amount + amount
                new_position = Position(
                    symbol=symbol,
                    amount=total_amount,
                    avg_cost=(old.amount * old.avg_cost + quote) / total_amount,
                    opened_at=old.opened_at,
                )

            trade = Trade(timestamp=int(timestamp), symbol=symbol, side=Side.BUY, price=fill_price,
                          amount=amount, fee=fee, quote_value=quote, realized_pnl=0.0, reason=reason)
            stored = self._storage.record_fill(trade, new_cash, new_position, symbol)
            # Only touch memory after the database write succeeded.
            self._cash = new_cash
            self._positions[symbol] = new_position
        logger.info("Paper BUY %.8g %s @ %.8g (fee %.8g, %s)", amount, symbol, fill_price, fee,
                    TradeReason(reason).value)
        return stored

    def market_sell(self, symbol: str, base_amount: float, price: float, timestamp: int,
                    reason: TradeReason) -> Trade:
        """Sell ``base_amount`` of ``symbol`` at ``price`` - slippage; the fee is taken from the proceeds.

        Raises NoPosition if nothing is held, ValueError for a bad price/amount and
        BrokerError when selling more than is held (beyond rounding tolerance).
        """
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                raise NoPosition(f"No open position in {symbol}")
            price = _check_price(price)
            amount = float(base_amount)
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError(f"Sell amount must be above 0 (got {base_amount!r})")
            if amount > pos.amount + EPS * max(1.0, pos.amount):
                raise BrokerError(f"Cannot sell {amount:.8g} {symbol}: only {pos.amount:.8g} held")
            if amount > pos.amount:
                amount = pos.amount  # within rounding tolerance -> sell exactly what is held

            fill_price = price * (1 - self._slippage_rate)   # selling: receive slightly less
            gross = amount * fill_price
            fee = gross * self._fee_rate
            net = gross - fee
            new_cash = self._cash + net
            realized_pnl = net - amount * pos.avg_cost

            remaining = pos.amount - amount
            if remaining <= DUST_AMOUNT or remaining <= EPS * pos.amount:
                new_position: Position | None = None   # fully closed (ignore leftover dust)
            else:
                new_position = Position(symbol=symbol, amount=remaining, avg_cost=pos.avg_cost,
                                        opened_at=pos.opened_at)

            trade = Trade(timestamp=int(timestamp), symbol=symbol, side=Side.SELL, price=fill_price,
                          amount=amount, fee=fee, quote_value=net, realized_pnl=realized_pnl,
                          reason=reason)
            stored = self._storage.record_fill(trade, new_cash, new_position, symbol)
            # Only touch memory after the database write succeeded.
            self._cash = new_cash
            if new_position is None:
                del self._positions[symbol]
            else:
                self._positions[symbol] = new_position
        logger.info("Paper SELL %.8g %s @ %.8g (fee %.8g, P&L %.8g, %s)", amount, symbol, fill_price,
                    fee, realized_pnl, TradeReason(reason).value)
        return stored

    def close_position(self, symbol: str, price: float, timestamp: int, reason: TradeReason) -> Trade:
        """Sell the full amount held in ``symbol``. Raises NoPosition if nothing is held."""
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                raise NoPosition(f"No open position in {symbol}")
            return self.market_sell(symbol, pos.amount, price, timestamp, reason)

    def reset(self, starting_balance: float) -> None:
        """Wipe trades/positions and start again with ``starting_balance`` cash."""
        with self._lock:
            self._storage.reset_account(starting_balance)
            self._load_state()

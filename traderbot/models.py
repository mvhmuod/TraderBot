"""Shared data types used across TraderBot.

Conventions (apply everywhere in the code base):
- Timestamps are integer milliseconds since the Unix epoch, UTC.
- "base" = the coin being traded (BTC in BTC/USDT); "quote" = the money it is
  priced in (USDT in BTC/USDT). Paper cash is held in a generic quote unit.
- The bot is long-only spot: it can only BUY with cash and SELL what it holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

# Column layout of every candle DataFrame passed between modules.
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeReason(str, Enum):
    SIGNAL = "signal"            # strategy crossover
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    DAILY_LIMIT = "daily_limit"  # daily loss limit reached, position closed


@dataclass(frozen=True)
class Trade:
    """One executed (simulated) fill."""
    timestamp: int          # ms, when the fill happened
    symbol: str             # e.g. "BTC/USDT"
    side: Side
    price: float            # fill price per 1 base unit, after slippage
    amount: float           # base units bought or sold
    fee: float              # quote units paid as fee
    quote_value: float      # BUY: total quote spent incl. fee; SELL: quote received after fee
    realized_pnl: float     # SELL: profit/loss vs cost basis (quote). BUY: always 0.0
    reason: TradeReason
    id: int | None = None   # database id, set once stored


@dataclass(frozen=True)
class Position:
    """An open holding of one symbol."""
    symbol: str
    amount: float           # base units held (> 0)
    avg_cost: float         # quote paid per base unit INCLUDING buy fees (break-even price)
    opened_at: int          # ms


@dataclass(frozen=True)
class AccountSnapshot:
    """Read-only copy of the paper account, safe to hand to the GUI thread."""
    cash: float                         # quote units available
    starting_balance: float             # quote units the account was (re)set with
    positions: dict[str, Position]      # symbol -> position
    equity: float                       # cash + positions valued at mark prices
    realized_pnl_total: float           # sum of realized_pnl over all trades

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting_balance

    @property
    def total_pnl_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return self.total_pnl / self.starting_balance * 100.0


@dataclass
class StepResult:
    """Everything one engine poll produced; emitted to the GUI after every step."""
    timestamp: int                          # ms, when the step ran
    symbol: str
    timeframe: str
    candles: pd.DataFrame | None = None     # CANDLE_COLUMNS + "fast_ma", "slow_ma"; closed candles only
    last_price: float | None = None
    signal: Signal = Signal.HOLD            # signal evaluated on this step (HOLD if none/not evaluated)
    account: AccountSnapshot | None = None
    new_trades: list[Trade] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)   # human-readable log lines
    error: str | None = None                # set when the step failed (e.g. network); bot keeps running
    halted: bool = False                    # True when the bot stopped itself (daily loss limit)

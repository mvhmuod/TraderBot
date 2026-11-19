"""Risk rules: how big each buy is, when to exit early, and the daily loss limit.

All percentages come from BotConfig. A value of 0 switches that rule off.
"""
from __future__ import annotations

from datetime import datetime, timezone

from traderbot.config import MIN_ORDER_QUOTE, BotConfig
from traderbot.models import Position, TradeReason


class RiskManager:
    """Applies the risk settings of a BotConfig. Reads the config on every call."""

    def __init__(self, config: BotConfig) -> None:
        self.config: BotConfig = config

    def order_quote_amount(self, cash: float, equity: float) -> float:
        """Quote amount to spend on the next buy, or 0.0 if it would be below the minimum.

        Size = position_size_pct of equity, but never more than the available cash.
        """
        amount = min(cash, equity * self.config.position_size_pct / 100.0)
        if amount < MIN_ORDER_QUOTE:
            return 0.0
        return amount

    def exit_reason(self, position: Position, price: float) -> TradeReason | None:
        """Return STOP_LOSS / TAKE_PROFIT if the price hit a limit, else None.

        Stop-loss is checked first, so it wins if both somehow apply.
        """
        stop_loss_pct = self.config.stop_loss_pct
        take_profit_pct = self.config.take_profit_pct
        if stop_loss_pct > 0 and price <= position.avg_cost * (1 - stop_loss_pct / 100.0):
            return TradeReason.STOP_LOSS
        if take_profit_pct > 0 and price >= position.avg_cost * (1 + take_profit_pct / 100.0):
            return TradeReason.TAKE_PROFIT
        return None

    def daily_limit_hit(self, day_start_equity: float, equity: float) -> bool:
        """True when equity fell by max_daily_loss_pct (or more) since the start of the UTC day."""
        pct = self.config.max_daily_loss_pct
        return pct > 0 and day_start_equity > 0 and equity <= day_start_equity * (1 - pct / 100.0)


def utc_day_key(timestamp_ms: int) -> str:
    """Meta key for the start-of-day equity, e.g. "day_start:2024-05-01" (UTC date)."""
    # Floor division keeps the correct day for every millisecond (even before 1970).
    day = datetime.fromtimestamp(timestamp_ms // 1000, tz=timezone.utc)
    return f"day_start:{day:%Y-%m-%d}"

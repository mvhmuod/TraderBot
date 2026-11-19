"""Bot settings. Every value here can be changed in the GUI.

The default numbers are placeholders so the app starts with something valid;
they are NOT recommendations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "traderbot.db"

SUPPORTED_EXCHANGES = ["binance", "kraken", "coinbase", "bybit", "okx"]
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

MIN_ORDER_QUOTE = 1.0       # smallest simulated order, in quote units
MAX_MA_PERIOD = 500
MIN_POLL_SECONDS = 5        # keep well inside public API rate limits


@dataclass
class BotConfig:
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    fast_period: int = 9
    slow_period: int = 21
    starting_balance: float = 10_000.0
    fee_rate: float = 0.001             # 0.1% per fill
    slippage_rate: float = 0.0005       # 0.05% worse price per fill
    position_size_pct: float = 10.0     # % of equity per buy (capped by available cash)
    stop_loss_pct: float = 0.0          # 0 = disabled; % below avg_cost
    take_profit_pct: float = 0.0        # 0 = disabled; % above avg_cost
    max_daily_loss_pct: float = 0.0     # 0 = disabled; % drop from equity at start of UTC day
    poll_interval_sec: int = 15
    history_limit: int = 300            # closed candles to load each poll

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty list means valid."""
        errors: list[str] = []
        if self.exchange_id not in SUPPORTED_EXCHANGES:
            errors.append(f"Unsupported exchange: {self.exchange_id!r}")
        if not self.symbol or "/" not in self.symbol:
            errors.append("Trading pair must look like BASE/QUOTE, e.g. BTC/USDT")
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            errors.append(f"Unsupported timeframe: {self.timeframe!r}")
        if not (1 <= self.fast_period <= MAX_MA_PERIOD):
            errors.append(f"Fast MA length must be between 1 and {MAX_MA_PERIOD}")
        if not (2 <= self.slow_period <= MAX_MA_PERIOD):
            errors.append(f"Slow MA length must be between 2 and {MAX_MA_PERIOD}")
        if self.fast_period >= self.slow_period:
            errors.append("Fast MA length must be smaller than slow MA length")
        if not (self.starting_balance >= MIN_ORDER_QUOTE):
            errors.append(f"Starting balance must be at least {MIN_ORDER_QUOTE}")
        if not (0 <= self.fee_rate < 0.1):
            errors.append("Fee rate must be between 0 and 10%")
        if not (0 <= self.slippage_rate < 0.1):
            errors.append("Slippage rate must be between 0 and 10%")
        if not (0 < self.position_size_pct <= 100):
            errors.append("Position size must be above 0% and at most 100%")
        if not (0 <= self.stop_loss_pct < 100):
            errors.append("Stop-loss must be between 0% (off) and 100%")
        if not (0 <= self.take_profit_pct <= 1000):
            errors.append("Take-profit must be between 0% (off) and 1000%")
        if not (0 <= self.max_daily_loss_pct < 100):
            errors.append("Daily loss limit must be between 0% (off) and 100%")
        if self.poll_interval_sec < MIN_POLL_SECONDS:
            errors.append(f"Poll interval must be at least {MIN_POLL_SECONDS} seconds")
        if self.history_limit < self.slow_period + 2:
            errors.append("History limit must be at least slow MA length + 2")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BotConfig":
        """Build from saved settings, ignoring unknown keys and keeping defaults for missing ones."""
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        defaults = cls()
        for name, value in (data or {}).items():
            if name not in known:
                continue
            default_value = getattr(defaults, name)
            try:
                kwargs[name] = type(default_value)(value)
            except (TypeError, ValueError):
                continue  # bad saved value -> keep default
        return cls(**kwargs)

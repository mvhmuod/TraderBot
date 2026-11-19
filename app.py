"""Start TraderBot (paper trading only: simulated orders, public market data, no API keys)."""
import logging

from traderbot.gui.app import run

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run())

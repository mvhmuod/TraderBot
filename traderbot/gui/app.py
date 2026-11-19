"""Application entry point for the desktop GUI."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from traderbot import __version__
from traderbot.gui.main_window import MainWindow


def run() -> int:
    """Create the Qt application and the main window, and run the event loop.

    Returns the application's exit code.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("TraderBot")
    app.setApplicationVersion(__version__)
    # Fusion looks the same on every OS and follows the system light/dark color scheme.
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()

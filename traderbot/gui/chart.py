"""Price chart widget: candlesticks, two moving averages, trade markers and a last-price line.

The chart only draws the data it is given through :meth:`PriceChart.set_data`; it never
fetches anything itself. All x positions are candle/trade timestamps in SECONDS (pyqtgraph's
``DateAxisItem`` expects seconds since the Unix epoch).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette
from PySide6.QtWidgets import QVBoxLayout, QWidget

from traderbot.models import Side, Trade

logger = logging.getLogger(__name__)

# Colors chosen to read well on both light and dark backgrounds.
UP_COLOR = "#26a69a"        # rising candle (close >= open)
DOWN_COLOR = "#ef5350"      # falling candle (close < open)
FAST_MA_COLOR = "#f5a623"   # orange
SLOW_MA_COLOR = "#4f8ff7"   # blue
BUY_MARKER_COLOR = "#16a34a"
SELL_MARKER_COLOR = "#dc2626"
LAST_PRICE_COLOR = "#9e9e9e"

# Fraction of a candle interval used for the candle body width.
_BODY_WIDTH_FRACTION = 0.7
# Fallback candle interval (seconds) when it cannot be derived from the data (e.g. one candle).
_DEFAULT_INTERVAL_SEC = 60.0
# Trade markers are drawn this fraction of the visible price span away from the candle.
_MARKER_OFFSET_FRACTION = 0.015


def format_price(value: float | None) -> str:
    """Format a price with thousands separators; small prices get up to 8 significant digits."""
    if value is None or not math.isfinite(value):
        return "—"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:,.4f}"
    if magnitude == 0:
        return "0.00"
    # Small prices: show 8 significant digits, trim trailing zeros, keep at least 2 decimals.
    decimals = min(12, max(2, 8 - 1 - math.floor(math.log10(magnitude))))
    text = f"{value:.{decimals}f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(2, '0')}"


class CandlestickItem(pg.GraphicsObject):
    """Draws OHLC candles: green when close >= open, red otherwise.

    Candles are painted directly in :meth:`paint` (instead of a cached QPicture) so that very
    small prices and very large epoch-second x values keep full floating point precision.
    """

    def __init__(self) -> None:
        super().__init__()
        self._x = np.empty(0, dtype=float)
        self._low = np.empty(0, dtype=float)
        self._high = np.empty(0, dtype=float)
        self._width = 0.0
        self._bounds = QRectF()
        # Pre-built drawing primitives, grouped by color.
        self._up_wicks: list[QLineF] = []
        self._down_wicks: list[QLineF] = []
        self._up_bodies: list[QRectF] = []
        self._down_bodies: list[QRectF] = []
        self._up_flat: list[QLineF] = []     # bodies with open == close drawn as a line
        self._down_flat: list[QLineF] = []

        self._up_pen = pg.mkPen(UP_COLOR, width=1)       # pyqtgraph pens are cosmetic (1 px)
        self._down_pen = pg.mkPen(DOWN_COLOR, width=1)
        self._up_brush = QBrush(QColor(UP_COLOR))
        self._down_brush = QBrush(QColor(DOWN_COLOR))

    @property
    def count(self) -> int:
        """Number of candles currently drawn."""
        return int(self._x.size)

    def set_data(self, x: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray, width: float) -> None:
        """Replace all candles. ``x`` is in seconds; ``width`` is the body width in seconds."""
        self.prepareGeometryChange()
        self._x = np.asarray(x, dtype=float)
        self._low = np.asarray(low, dtype=float)
        self._high = np.asarray(high, dtype=float)
        self._width = float(width)
        self._up_wicks, self._down_wicks = [], []
        self._up_bodies, self._down_bodies = [], []
        self._up_flat, self._down_flat = [], []

        half = self._width / 2.0
        for xi, o, h, lo, c in zip(self._x, open_, high, low, close):
            if not all(math.isfinite(v) for v in (xi, o, h, lo, c)):
                continue  # skip broken rows instead of drawing garbage
            up = c >= o
            wick = QLineF(QPointF(xi, lo), QPointF(xi, h))
            (self._up_wicks if up else self._down_wicks).append(wick)
            bottom, top = min(o, c), max(o, c)
            if top > bottom:
                body = QRectF(xi - half, bottom, self._width, top - bottom)
                (self._up_bodies if up else self._down_bodies).append(body)
            else:
                flat = QLineF(QPointF(xi - half, o), QPointF(xi + half, o))
                (self._up_flat if up else self._down_flat).append(flat)

        finite = np.isfinite(self._low) & np.isfinite(self._high) & np.isfinite(self._x)
        if finite.any():
            x_min = float(self._x[finite].min()) - half
            x_max = float(self._x[finite].max()) + half
            y_min = float(self._low[finite].min())
            y_max = float(self._high[finite].max())
            self._bounds = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            self._bounds = QRectF()
        self.update()
        self.informViewBoundsChanged()   # lets the ViewBox auto-range react to the new data

    def clear(self) -> None:
        """Remove all candles."""
        empty = np.empty(0, dtype=float)
        self.set_data(empty, empty, empty, empty, empty, 0.0)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt naming)
        return QRectF(self._bounds)

    def dataBounds(self, ax: int, frac: float = 1.0,  # noqa: N802 (pyqtgraph naming)
                   orthoRange: Sequence[float] | None = None) -> tuple[float | None, float | None]:
        """Data range on one axis; lets the view fit the y-axis to the visible candles."""
        finite = np.isfinite(self._low) & np.isfinite(self._high) & np.isfinite(self._x)
        if not finite.any():
            return (None, None)
        half = self._width / 2.0
        if ax == 0:
            return (float(self._x[finite].min()) - half, float(self._x[finite].max()) + half)
        mask = finite
        if orthoRange is not None and orthoRange[0] is not None and orthoRange[1] is not None:
            mask = mask & (self._x >= orthoRange[0] - half) & (self._x <= orthoRange[1] + half)
            if not mask.any():
                return (None, None)
        return (float(self._low[mask].min()), float(self._high[mask].max()))

    def paint(self, painter: QPainter, option: object, widget: QWidget | None = None) -> None:
        if self._x.size == 0:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for pen, brush, wicks, bodies, flats in (
            (self._up_pen, self._up_brush, self._up_wicks, self._up_bodies, self._up_flat),
            (self._down_pen, self._down_brush, self._down_wicks, self._down_bodies, self._down_flat),
        ):
            painter.setPen(pen)
            if wicks:
                painter.drawLines(wicks)
            if flats:
                painter.drawLines(flats)
            if bodies:
                painter.setBrush(brush)
                painter.drawRects(bodies)


class PriceChart(QWidget):
    """Candlestick chart with fast/slow SMA lines, BUY/SELL markers and a last-price line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)

        self._date_axis = pg.DateAxisItem(orientation="bottom")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": self._date_axis})
        self._plot = self.plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        # Price axis on the right, like most trading charts.
        self._plot.showAxis("right")
        self._plot.hideAxis("left")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        view_box = self._plot.getViewBox()
        view_box.setAutoVisible(y=True)   # when the user pans in time, fit prices to what is visible

        self.candles = CandlestickItem()
        self._plot.addItem(self.candles)

        self.fast_curve = pg.PlotDataItem(pen=pg.mkPen(FAST_MA_COLOR, width=1.6), connect="finite")
        self.slow_curve = pg.PlotDataItem(pen=pg.mkPen(SLOW_MA_COLOR, width=1.6), connect="finite")
        self._plot.addItem(self.fast_curve)
        self._plot.addItem(self.slow_curve)

        self.buy_markers = pg.ScatterPlotItem(
            symbol="t1", size=13, pen=pg.mkPen(None), brush=pg.mkBrush(BUY_MARKER_COLOR))   # up triangle
        self.sell_markers = pg.ScatterPlotItem(
            symbol="t", size=13, pen=pg.mkPen(None), brush=pg.mkBrush(SELL_MARKER_COLOR))   # down triangle
        self._plot.addItem(self.buy_markers)
        self._plot.addItem(self.sell_markers)

        self.price_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(LAST_PRICE_COLOR, width=1, style=Qt.PenStyle.DashLine),
            # Label hugs the right edge, just above the line (colors set in _apply_theme).
            label="", labelOpts={"position": 0.995, "anchors": [(1, 1), (1, 1)]},
        )
        # ignoreBounds: the price line should not stretch the auto-range.
        self._plot.addItem(self.price_line, ignoreBounds=True)
        self.price_line.setVisible(False)

        self._legend = pg.LegendItem(offset=(10, 10))
        self._legend.setParentItem(self._plot.getViewBox())
        self._legend_periods: tuple[int, int] | None = None

        self._needs_reset = True     # auto-range on the first data after construction / reset_view()
        self._has_data = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self._apply_theme()

    # ------------------------------------------------------------------ public API
    def set_data(self, candles: pd.DataFrame | None, trades: list[Trade] | None,
                 last_price: float | None, fast_period: int, slow_period: int) -> None:
        """Redraw everything from the given data.

        ``candles`` has CANDLE_COLUMNS and optionally "fast_ma"/"slow_ma". Empty or None data
        clears the candles, lines and markers (the last-price line is still shown if given).
        """
        self._update_legend(fast_period, slow_period)
        self._set_last_price(last_price)

        if candles is None or len(candles) == 0 or not {"timestamp", "open", "high", "low",
                                                          "close"}.issubset(candles.columns):
            self._clear_series()
            return

        x = candles["timestamp"].to_numpy(dtype=float) / 1000.0
        open_ = candles["open"].to_numpy(dtype=float)
        high = candles["high"].to_numpy(dtype=float)
        low = candles["low"].to_numpy(dtype=float)
        close = candles["close"].to_numpy(dtype=float)

        interval = _DEFAULT_INTERVAL_SEC
        if x.size >= 2:
            diffs = np.diff(x)
            diffs = diffs[diffs > 0]
            if diffs.size:
                interval = float(np.median(diffs))
        self.candles.set_data(x, open_, high, low, close, interval * _BODY_WIDTH_FRACTION)

        self._set_ma_curve(self.fast_curve, candles, "fast_ma", x)
        self._set_ma_curve(self.slow_curve, candles, "slow_ma", x)
        self._set_markers(trades or [], x, high, low)

        self._has_data = True
        if self._needs_reset:
            self._needs_reset = False
            self._auto_range()

    def clear(self) -> None:
        """Remove all drawn data (candles, lines, markers, last-price line)."""
        self._set_last_price(None)
        self._clear_series()

    def reset_view(self) -> None:
        """Fit the view to the data again (call on symbol/timeframe change).

        If there is no data yet, the fit happens when the next data arrives.
        """
        if self._has_data:
            self._needs_reset = False
            self._auto_range()
        else:
            self._needs_reset = True

    def marker_counts(self) -> tuple[int, int]:
        """(number of BUY markers, number of SELL markers) currently drawn."""
        return len(self.buy_markers.data), len(self.sell_markers.data)

    # ------------------------------------------------------------------ internals
    def _auto_range(self) -> None:
        """Fit the view to the data and keep following new data until the user zooms/pans."""
        view_box = self._plot.getViewBox()
        view_box.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        view_box.updateAutoRange()

    def _clear_series(self) -> None:
        self.candles.clear()
        self.fast_curve.setData([], [])
        self.slow_curve.setData([], [])
        self.buy_markers.setData([], [])
        self.sell_markers.setData([], [])
        self._has_data = False

    def _set_ma_curve(self, curve: pg.PlotDataItem, candles: pd.DataFrame, column: str,
                      x: np.ndarray) -> None:
        if column not in candles.columns:
            curve.setData([], [])
            return
        y = pd.to_numeric(candles[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y)
        if not finite.any():
            curve.setData([], [])
            return
        # Only pass the defined part (MAs are NaN until enough candles exist).
        curve.setData(x[finite], y[finite])

    def _set_markers(self, trades: list[Trade], x: np.ndarray, high: np.ndarray,
                     low: np.ndarray) -> None:
        finite = np.isfinite(low) & np.isfinite(high)
        span = float(high[finite].max() - low[finite].min()) if finite.any() else 0.0
        buys_x: list[float] = []
        buys_y: list[float] = []
        sells_x: list[float] = []
        sells_y: list[float] = []
        for trade in trades:
            tx = trade.timestamp / 1000.0
            # The candle that contains the trade time = the last candle starting at or before it.
            idx = int(np.searchsorted(x, tx, side="right")) - 1
            if idx < 0:
                continue  # trade happened before the visible history
            offset = span * _MARKER_OFFSET_FRACTION if span > 0 else abs(trade.price) * 0.002
            if trade.side == Side.BUY:
                buys_x.append(tx)
                buys_y.append(min(float(low[idx]), trade.price) - offset)
            else:
                sells_x.append(tx)
                sells_y.append(max(float(high[idx]), trade.price) + offset)
        self.buy_markers.setData(buys_x, buys_y)
        self.sell_markers.setData(sells_x, sells_y)

    def _set_last_price(self, last_price: float | None) -> None:
        if last_price is None or not math.isfinite(last_price) or last_price <= 0:
            self.price_line.setVisible(False)
            return
        self.price_line.setValue(last_price)
        self.price_line.setVisible(True)
        self.price_line.label.setVisible(True)
        self.price_line.label.setFormat(f"Last {format_price(last_price)}")

    def _update_legend(self, fast_period: int, slow_period: int) -> None:
        periods = (int(fast_period), int(slow_period))
        if periods == self._legend_periods:
            return
        self._legend_periods = periods
        self._legend.clear()
        self._legend.addItem(self.fast_curve, f"Fast SMA ({periods[0]})")
        self._legend.addItem(self.slow_curve, f"Slow SMA ({periods[1]})")

    def _apply_theme(self) -> None:
        """Match the chart background and axis colors to the current (light or dark) palette."""
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        text = palette.color(QPalette.ColorRole.Text)
        self.plot_widget.setBackground(background)
        axis_pen = pg.mkPen(text)
        for name in ("bottom", "right"):
            axis = self._plot.getAxis(name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
        self._legend.setLabelTextColor(text)
        legend_background = QColor(background)
        legend_background.setAlpha(200)
        self._legend.setBrush(pg.mkBrush(legend_background))
        # The last-price label sits on top of the newest candles: give it a solid-ish
        # background and the normal text color so it stays readable in light and dark themes.
        label = self.price_line.label
        label.fill = pg.mkBrush(legend_background)
        label.border = pg.mkPen(LAST_PRICE_COLOR)
        label.setColor(text)
        label.update()
        if self._legend_periods is not None:
            # Re-create labels so they pick up the new text color.
            fast, slow = self._legend_periods
            self._legend_periods = None
            self._update_legend(fast, slow)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 (Qt naming)
        if (event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange)
                and hasattr(self, "_legend")):
            self._apply_theme()
        super().changeEvent(event)

"""
Per-bar order-flow indicators: DeltaBars, CVD and DeltaVolumeInfo.

All three are tick-mounted and aggregate the trade stream into fixed-interval
bars (``period``), classifying the aggressor side of every trade with the shared
``classify_aggressor``. They expose the per-bar figures to ``on_data()`` as
developing bands and draw their geometry in an own panel below price through the
declarative ``draw()`` primitive contract (no plotting-package code):

    - DeltaBars       - a diverging histogram of the per-bar delta (ask - bid).
    - CVD             - cumulative-volume-delta candles (running cumulative
                        delta sampled into per-bar OHLC).
    - VolumeInfo      - a configurable numeric volume breakdown: buy / sell /
                        delta / total / delta_max / delta_min cells per bar
                        (the volume "recuadros" panel). Default rows:
                        delta_max / delta_min / delta.

Detection is causal (a bar depends only on the trades up to it), so the result
is identical in backtest, live and replay; the tick-mounted ``live_refresh``
hook re-runs the same ``calculate`` over the live tick window for parity, exactly
like ``LargeTrades``.

Usage::

    self.ticks = self.subscribe_ticks("BTCUSDT")
    self.cvd   = self.add_indicator(CVD.refs(self.ticks), CVD(period="1m"))
    self.delta = self.add_indicator(DeltaBars.refs(self.ticks), DeltaBars("1m"))

    def on_data(self):
        if self.cvd.close[-1] > 0 and self.delta.delta[-1] > 0:
            ...
"""

from __future__ import annotations

import numpy as np

from tradetropy.core.constants import parse_timeframe
from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.draw import Labels, Rects, Segments
from tradetropy.ta.order_flow._core import (
    bar_delta,
    cumulative_delta_ohlc,
    format_magnitude,
)

# Semantic colors (object defaults, theme-independent).
DEFAULT_UP_COLOR = "#0ECB81"     # green - positive delta / bullish CVD
DEFAULT_DOWN_COLOR = "#F6465D"   # red - negative delta / bearish CVD
DEFAULT_TOTAL_COLOR = "#808080"  # gray - side-neutral total volume row


def _tick_refs(tick_proxy):
    """Build the [ts, price, volume, flags, bid, ask] ColumnRef list."""
    return [
        tick_proxy.col_ref("ts"),
        tick_proxy.price_ref,
        tick_proxy.col_ref("volume"),
        tick_proxy.col_ref("flags"),
        tick_proxy.bid_ref,
        tick_proxy.ask_ref,
    ]


class _TickBarFlow(Indicator):
    """
    Shared base for tick-mounted, per-bar order-flow panels.

    Subclasses implement ``calculate`` (storing their per-bar arrays on the
    instance for ``draw()``) and ``draw``. The base provides the tick column
    refs, the source-array split, the bar interval parsing and the live refresh
    hook so all three panels stay consistent.
    """

    category = "volume"

    def __init__(self, period="1m", anchor: int = 0):
        if isinstance(period, str):
            self.interval_ms = int(parse_timeframe(period))
        else:
            self.interval_ms = int(period)
        if self.interval_ms <= 0:
            raise ValueError(f"period must be positive, got {period!r}")
        self.period = period
        self.anchor = int(anchor)

    @property
    def min_periods(self) -> int:
        return 1

    @staticmethod
    def refs(tick_proxy):
        """Build the ColumnRef list: [ts, price, volume, flags, bid, ask]."""
        return _tick_refs(tick_proxy)

    def default_refs(self, proxy):
        """ColumnRefs resolved from a proxy so add_indicator(proxy, ind) works."""
        return type(self).refs(proxy)

    def display_name(self) -> str:
        return f"{type(self).__name__}({self.period})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"{type(self).__name__.lower()}_{self.period}_{symbol}"

    def _split_source(self, source: np.ndarray):
        """Return (ts, price, volume, flags, bid, ask) from the [N x 6] source."""
        ts = source[:, 0].astype(np.int64)
        price = source[:, 1].astype(np.float64)
        volume = source[:, 2].astype(np.float64)
        flags = source[:, 3] if source.shape[1] > 3 else None
        bid = source[:, 4] if source.shape[1] > 4 else None
        ask = source[:, 5] if source.shape[1] > 5 else None
        return ts, price, volume, flags, bid, ask

    def live_refresh(self, proxy) -> None:
        """Re-run calculate over the live tick window (parity with backtest)."""
        if proxy is None or len(proxy) == 0:
            return
        try:
            ts = np.asarray(proxy.ts[:], dtype=np.float64)
            price = np.asarray(proxy.price[:], dtype=np.float64)
            volume = np.asarray(proxy.volume[:], dtype=np.float64)
            flags = np.asarray(proxy.flags[:], dtype=np.float64)
            bid = np.asarray(proxy.bid[:], dtype=np.float64)
            ask = np.asarray(proxy.ask[:], dtype=np.float64)
        except Exception:
            return
        if ts.size == 0:
            return
        self.calculate(np.column_stack([ts, price, volume, flags, bid, ask]))

    def _half_width(self) -> int:
        """
        Half bar width (ms) for a glyph centered on the bar's start ts.

        Candles are drawn with ``vbar(x=ts, width=0.9 * interval)`` - centered
        on the bar's start timestamp and spanning 0.9 of the interval. Centering
        the order-flow geometry on ``bar_ts`` with the same width aligns each bar
        directly under its candle (instead of anchoring to the [bar_ts,
        bar_ts + interval] span, which shifts it half a bar to the right).

        Returns:
            int: Half width in milliseconds (0.45 * interval).
        """
        return int(self.interval_ms * 0.45)


class DeltaBars(_TickBarFlow):
    """
    Per-bar delta histogram (ask volume - bid volume), own panel.

    Source columns (use ``DeltaBars.refs(tick_proxy)``):
    ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [4 x N] anchored at each bar's last tick (NaN elsewhere):
        row 0 : delta   - ask_vol - bid_vol of the bar.
        row 1 : ask_vol - buy-aggressor volume.
        row 2 : bid_vol - sell-aggressor volume.
        row 3 : total   - ask_vol + bid_vol.

    Args:
        period (str | int): Bar size ('1m', '5m', ... or ms). Match it to your
            chart's candle interval to align the bars under the candles.
        up_color / down_color: Bar colors for positive / negative delta.
        anchor (int): Epoch-ms origin of the bar grid.
    """

    name = "delta_bars"
    output_names = ["delta"]
    ts_band_indices = [1, 2, 3]
    ts_output_names = ["ask_vol", "bid_vol", "total"]

    def __init__(self, period="1m", up_color: str = DEFAULT_UP_COLOR,
                 down_color: str = DEFAULT_DOWN_COLOR, anchor: int = 0):
        super().__init__(period=period, anchor=anchor)
        self.up_color = up_color
        self.down_color = down_color
        self._bars: dict = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=False, renderer="none", panel_title="Delta",
            name="Delta", autoscale=True,
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 3 or len(source) == 0:
            self._bars = {}
            return np.full((4, 0), np.nan, dtype=np.float64)
        n = len(source)
        ts, price, volume, flags, bid, ask = self._split_source(source)
        bars = bar_delta(ts, price, volume, interval_ms=self.interval_ms,
                         flags=flags, bid=bid, ask=ask, anchor=self.anchor)
        self._bars = bars
        out = np.full((4, n), np.nan, dtype=np.float64)
        idx = bars["rep_idx"]
        if idx.size:
            out[0, idx] = bars["delta"]
            out[1, idx] = bars["ask_vol"]
            out[2, idx] = bars["bid_vol"]
            out[3, idx] = bars["total"]
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        bars = getattr(self, "_bars", {}) or {}
        bar_ts = np.asarray(bars.get("bar_ts", []), dtype=np.int64)
        if bar_ts.size == 0:
            return []
        delta = np.asarray(bars["delta"], dtype=np.float64)
        half = self._half_width()
        x0 = list(bar_ts - half)
        x1 = list(bar_ts + half)
        y0 = [0.0] * len(bar_ts)
        y1 = list(delta)
        colors = [self.up_color if d >= 0 else self.down_color for d in delta]
        return [Rects(
            x0=x0, x1=x1, y0=y0, y1=y1,
            fill_color=colors, fill_alpha=0.7,
            line_color=colors, line_alpha=0.9, line_width=1.0,
        )]


class CVD(_TickBarFlow):
    """
    Cumulative Volume Delta panel.

    The running cumulative delta (+volume buy / -volume sell) is sampled per bar
    into open / high / low / close. Two visual styles are available:

        - ``style='candle'`` (default): classic candles (wick + body) of the
          cumulative delta OHLC, so the panel reads like a price chart of net
          aggression.
        - ``style='bar'``: one diverging bar per period whose height is the
          cumulative delta (``close``), colored by whether the cumulative delta
          rose (green) or fell (red) versus the bar's open. The simple
          single-bar CVD histogram.

    Source columns (use ``CVD.refs(tick_proxy)``):
    ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [4 x N] anchored at each bar's last tick (NaN elsewhere):
        row 0 : open, row 1 : high, row 2 : low, row 3 : close (cumulative delta).

    Args:
        period (str | int): Bar size ('1m', ... or ms).
        up_color / down_color: Rising / falling cumulative-delta colors.
        style (str): 'candle' (OHLC candles, default) or 'bar' (single histogram).
        anchor (int): Epoch-ms origin of the bar grid.
    """

    name = "cvd"
    output_names = ["open", "high", "low", "close"]

    def __init__(self, period="1m", up_color: str = DEFAULT_UP_COLOR,
                 down_color: str = DEFAULT_DOWN_COLOR, style: str = "candle",
                 anchor: int = 0):
        super().__init__(period=period, anchor=anchor)
        self.up_color = up_color
        self.down_color = down_color
        style = str(style).lower()
        if style not in ("bar", "candle"):
            raise ValueError(
                f"style must be 'bar' or 'candle', got {style!r}"
            )
        self.style = style
        self._bars: dict = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=False, renderer="none", panel_title="CVD", name="CVD",
            autoscale=True,
            reference_lines=[{"value": 0.0, "color": "#888888",
                              "dash": "dashed", "label": "0"}],
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 3 or len(source) == 0:
            self._bars = {}
            return np.full((4, 0), np.nan, dtype=np.float64)
        n = len(source)
        ts, price, volume, flags, bid, ask = self._split_source(source)
        bars = cumulative_delta_ohlc(ts, price, volume, interval_ms=self.interval_ms,
                                     flags=flags, bid=bid, ask=ask, anchor=self.anchor)
        self._bars = bars
        out = np.full((4, n), np.nan, dtype=np.float64)
        # rep tick of each bar = last trade; recover its index from the source.
        # bar_delta-style rep_idx is not returned here, so re-derive via the grid.
        if bars["bar_ts"].size:
            grid = (ts - self.anchor) // self.interval_ms
            _, first_idx = np.unique(grid, return_index=True)
            rep_idx = np.empty(first_idx.shape[0], dtype=np.int64)
            rep_idx[:-1] = first_idx[1:] - 1
            rep_idx[-1] = n - 1
            out[0, rep_idx] = bars["open"]
            out[1, rep_idx] = bars["high"]
            out[2, rep_idx] = bars["low"]
            out[3, rep_idx] = bars["close"]
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        bars = getattr(self, "_bars", {}) or {}
        bar_ts = np.asarray(bars.get("bar_ts", []), dtype=np.int64)
        if bar_ts.size == 0:
            return []
        o = np.asarray(bars["open"], dtype=np.float64)
        h = np.asarray(bars["high"], dtype=np.float64)
        low = np.asarray(bars["low"], dtype=np.float64)
        c = np.asarray(bars["close"], dtype=np.float64)
        half = self._half_width()
        if self.style == "bar":
            return self._draw_bar(bar_ts, o, c, half)
        return self._draw_candle(bar_ts, o, h, low, c, half)

    def _draw_bar(self, bar_ts, o, c, half) -> list:
        """Single diverging bar per period: height = cumulative delta close."""
        colors = [self.up_color if cc >= oo else self.down_color
                  for oo, cc in zip(o, c)]
        return [Rects(
            x0=list(bar_ts - half), x1=list(bar_ts + half),
            y0=[0.0] * len(bar_ts), y1=list(c),
            fill_color=colors, fill_alpha=0.7,
            line_color=colors, line_alpha=0.9, line_width=1.0,
        )]

    def _draw_candle(self, bar_ts, o, h, low, c, half) -> list:
        """Classic candle per period: high-low wick plus open-close body."""
        center = bar_ts
        bull = c >= o
        colors = [self.up_color if b else self.down_color for b in bull]
        body_lo = np.minimum(o, c)
        body_hi = np.maximum(o, c)
        # Zero-height bodies (doji) would be invisible; give them a hairline.
        flat = body_hi <= body_lo
        eps = np.where(flat, np.maximum(np.abs(c) * 1e-3, 1e-9), 0.0)
        return [
            Segments(
                x0=list(center), y0=list(low), x1=list(center), y1=list(h),
                color=colors, alpha=0.9, width=1.0,
            ),
            Rects(
                x0=list(bar_ts - half), x1=list(bar_ts + half),
                y0=list(body_lo - eps), y1=list(body_hi + eps),
                fill_color=colors, fill_alpha=0.7,
                line_color=colors, line_alpha=0.95, line_width=1.0,
            ),
        ]


class VolumeInfo(_TickBarFlow):
    """
    Per-bar volume breakdown panel: configurable rows of numeric cells.

    Each row is drawn as a stack of colored cells with its number, one per bar.
    The ``rows`` parameter controls which rows appear and in what order (top to
    bottom). Available rows:

        - 'delta_max' - max intra-bar cumulative delta (green by default).
        - 'delta_min' - min intra-bar cumulative delta (red by default).
        - 'delta'     - net delta of the bar (buy - sell), gray by default.
        - 'buy'       - buy-aggressor (ask) volume, green.
        - 'sell'      - sell-aggressor (bid) volume, red.
        - 'total'     - total volume (buy + sell), gray.

    ``draw()`` emits one toggleable legend group per row so a single legend
    click hides that row's cells and numbers.

    Source columns (use ``VolumeInfo.refs(tick_proxy)``):
    ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [6 x N] anchored at each bar's last tick (NaN elsewhere):
        row 0 : delta, row 1 : ask_vol (buy), row 2 : bid_vol (sell),
        row 3 : total, row 4 : delta_max, row 5 : delta_min.

    Args:
        period (str | int): Bar size ('1m', ... or ms).
        rows (list[str]): Rows to display, top to bottom.
            Default: ['delta_max', 'delta_min', 'delta'].
        up_color: Buy / positive-delta cell color.
        down_color: Sell / negative-delta cell color.
        delta_max_color: Delta Max cell color (green by default).
        delta_min_color: Delta Min cell color (red by default).
        delta_color: Delta cell color (gray by default).
        total_color: Total cell color (gray by default).
        text_color: Number text color.
        cell_alpha (float): Cell fill opacity.
        font_size (str): Number font size.
        anchor (int): Epoch-ms origin of the bar grid.
    """

    name = "volume_info"
    output_names = ["delta"]
    ts_band_indices = [1, 2, 3, 4, 5]
    ts_output_names = ["ask_vol", "bid_vol", "total", "delta_max", "delta_min"]

    _ALL_ROWS = ("delta_max", "delta_min", "delta", "buy", "sell", "total")

    def __init__(self, period="1m",
                 rows: list | None = None,
                 up_color: str = DEFAULT_UP_COLOR,
                 down_color: str = DEFAULT_DOWN_COLOR,
                 delta_max_color: str = DEFAULT_UP_COLOR,
                 delta_min_color: str = DEFAULT_DOWN_COLOR,
                 delta_color: str = DEFAULT_TOTAL_COLOR,
                 total_color: str = DEFAULT_TOTAL_COLOR,
                 text_color: str = "#FFFFFF", cell_alpha: float = 0.85,
                 font_size: str = "8pt", anchor: int = 0):
        super().__init__(period=period, anchor=anchor)
        rows = list(rows) if rows is not None else ["delta_max", "delta_min", "delta"]
        bad = [r for r in rows if r not in self._ALL_ROWS]
        if bad:
            raise ValueError(
                f"Unknown rows {bad}; valid rows: {list(self._ALL_ROWS)}"
            )
        self.rows = rows
        self.up_color = up_color
        self.down_color = down_color
        self.delta_max_color = delta_max_color
        self.delta_min_color = delta_min_color
        self.delta_color = delta_color
        self.total_color = total_color
        self.text_color = text_color
        self.cell_alpha = float(cell_alpha)
        self.font_size = font_size
        self._bars: dict = {}
        n_rows = len(rows)
        panel_h = max(60, n_rows * 20 + 20)
        title = "/".join(r.replace("_", " ").title() for r in rows)
        self.plot_config = IndicatorPlotConfig(
            overlay=False, renderer="none", panel_title=title,
            name="VolumeInfo", panel_height=panel_h, show_legend=True,
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 3 or len(source) == 0:
            self._bars = {}
            return np.full((6, 0), np.nan, dtype=np.float64)
        n = len(source)
        ts, price, volume, flags, bid, ask = self._split_source(source)
        bars = bar_delta(ts, price, volume, interval_ms=self.interval_ms,
                         flags=flags, bid=bid, ask=ask, anchor=self.anchor)
        self._bars = bars
        out = np.full((6, n), np.nan, dtype=np.float64)
        idx = bars["rep_idx"]
        if idx.size:
            out[0, idx] = bars["delta"]
            out[1, idx] = bars["ask_vol"]
            out[2, idx] = bars["bid_vol"]
            out[3, idx] = bars["total"]
            out[4, idx] = bars["delta_max"]
            out[5, idx] = bars["delta_min"]
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> dict:
        bars = getattr(self, "_bars", {}) or {}
        bar_ts = np.asarray(bars.get("bar_ts", []), dtype=np.int64)
        if bar_ts.size == 0:
            return {}
        half = self._half_width()
        center = bar_ts
        nb = len(bar_ts)
        x0 = list(bar_ts - half)
        x1 = list(bar_ts + half)

        data = {
            "buy":       np.asarray(bars["ask_vol"],   dtype=np.float64),
            "sell":      np.asarray(bars["bid_vol"],   dtype=np.float64),
            "delta":     np.asarray(bars["delta"],     dtype=np.float64),
            "total":     np.asarray(bars["total"],     dtype=np.float64),
            "delta_max": np.asarray(bars["delta_max"], dtype=np.float64),
            "delta_min": np.asarray(bars["delta_min"], dtype=np.float64),
        }
        labels_map = {
            "buy": "Buy", "sell": "Sell", "delta": "Delta",
            "total": "Total", "delta_max": "Delta Max", "delta_min": "Delta Min",
        }

        def _cell_color(row_key, values):
            if row_key == "buy":
                return self.up_color
            if row_key == "sell":
                return self.down_color
            if row_key == "total":
                return self.total_color
            if row_key == "delta":
                return self.delta_color
            if row_key == "delta_max":
                return self.delta_max_color
            if row_key == "delta_min":
                return self.delta_min_color
            return self.total_color

        def _signed(row_key):
            return row_key in ("delta", "delta_max", "delta_min")

        result = {}
        n_rows = len(self.rows)
        for i, row_key in enumerate(self.rows):
            row_y = float(n_rows - i)   # top row = highest y value
            values = data[row_key]
            fill = _cell_color(row_key, values)
            signed = _signed(row_key)
            cell = Rects(
                x0=x0, x1=x1,
                y0=[row_y - 0.5] * nb, y1=[row_y + 0.5] * nb,
                fill_color=fill, fill_alpha=self.cell_alpha,
                line_color=fill, line_alpha=0.5, line_width=1.0,
            )
            txt = [
                (("+" if v >= 0 else "") + format_magnitude(float(v))) if signed
                else format_magnitude(float(v))
                for v in values
            ]
            lbl = Labels(
                x=list(center), y=[row_y] * nb, text=txt,
                color=self.text_color, font_size=self.font_size,
                x_offset=0, y_offset=0,
                text_align="center", text_baseline="middle",
            )
            result[labels_map[row_key]] = [cell, lbl]
        return result



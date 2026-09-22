"""
L2 order-book order-flow indicators: DeepWall, DeepReload and StopRun.

All three read the live L2 order book (an ``OrderbookProxy`` passed to the
constructor, exactly like ``DeepTrades``) and overlay their findings on the
price panel via the declarative ``draw()`` contract. Detection is pure and
causal (``order_flow/_core.py``), shared identically by backtest, live and
replay; because they need depth they are meaningful in live mode and in replay
of a recorded book - a plain backtest with no book yields nothing.

    - DeepWall   - resting liquidity walls (anomalously large levels that
                   persist), drawn as horizontal level spans.
    - DeepReload - liquidity replenishment: a level repeatedly consumed and
                   refilled (the L2 analogue of an iceberg), drawn as markers.
    - StopRun    - stop sweeps: a fast directional run through several ticks
                   followed by a reversal, drawn as a sweep segment + marker.

Usage::

    self.ticks = self.subscribe_ticks("BTCUSDT", window_size=5000)
    self.book  = self.subscribe_orderbook("BTCUSDT", depth=20)
    self.walls = self.add_indicator(DeepWall.refs(self.ticks), DeepWall(self.book))
    self.reload = self.add_indicator(DeepReload.refs(self.ticks), DeepReload(self.book))
    self.stops = self.add_indicator(StopRun.refs(self.ticks), StopRun(self.book))
"""

from __future__ import annotations

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.draw import HLines, Labels, Points, Segments
from tradetropy.ta.order_flow._core import (
    detect_reload_l2,
    detect_stop_run_l2,
    detect_walls,
    format_magnitude,
)

# Semantic colors (object defaults, theme-independent).
DEFAULT_BID_WALL_COLOR = "#0ECB81"   # green - bid (support) wall
DEFAULT_ASK_WALL_COLOR = "#F6465D"   # red - ask (resistance) wall
DEFAULT_RELOAD_COLOR = "#10B981"     # green - iceberg-like replenishment
DEFAULT_STOPRUN_UP_COLOR = "#F6465D"   # red - up-sweep then down reversal
DEFAULT_STOPRUN_DOWN_COLOR = "#0ECB81"  # green - down-sweep then up reversal


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


def _anchor_idx(tick_ts: np.ndarray, event_ts: np.ndarray) -> np.ndarray:
    """Index of the tick at or just before each event ts (causal anchoring)."""
    if len(tick_ts) == 0 or len(event_ts) == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.searchsorted(tick_ts, event_ts, side="right") - 1
    return np.clip(idx, 0, len(tick_ts) - 1).astype(np.int64)


class _L2BookIndicator(Indicator):
    """
    Shared base for tick-mounted, book-reading L2 order-flow overlays.

    The order book is supplied to the constructor (not through the column refs,
    since the proxy is not a numeric source). Subclasses implement ``calculate``
    (reading ``self._book_window()`` and storing events for ``draw()``) and
    ``draw``. The base provides tick refs, the book accessor, event-to-tick
    anchoring and the live refresh hook.
    """

    category = "annotation"

    def __init__(self, orderbook=None):
        self._book = orderbook

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

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"{type(self).__name__.lower()}_{symbol}"

    def _book_window(self):
        """Causal book history dict, or None when no book is attached."""
        if self._book is None:
            return None
        try:
            return self._book.book_window()
        except Exception:
            return None

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


class DeepWall(_L2BookIndicator):
    """
    Resting liquidity walls (L2), drawn as horizontal level spans.

    Source columns (use ``DeepWall.refs(tick_proxy)``):
    ``[ts, price, volume, flags, bid, ask]``; the book comes from the
    constructor.

    Outputs - [3 x N] anchored at each wall's first tick (NaN elsewhere):
        row 0 : price - wall price level.
        row 1 : side  - +1 bid (support) / -1 ask (resistance).
        row 2 : size  - peak resting size of the wall.

    Args:
        orderbook: OrderbookProxy supplying the L2 book (book_window).
        min_volume (float): Absolute minimum resting size to qualify as a wall.
        rel_multiple (float): Multiple of the snapshot's median level size
            (self-adjusting; 0 disables the relative term).
        persistence_ms (int): Minimum lifetime of a wall to report.
        top_n (int | None): Levels per side to scan (None -> all retained).
        bid_color / ask_color: Wall colors by side.
        show_size (bool): Draw the wall's size as a label.
    """

    name = "deep_wall"
    output_names = ["price"]
    ts_band_indices = [1, 2]
    ts_output_names = ["side", "size"]

    def __init__(self, orderbook=None, min_volume: float = 0.0,
                 rel_multiple: float = 5.0, persistence_ms: int = 0,
                 top_n: "int | None" = None,
                 bid_color: str = DEFAULT_BID_WALL_COLOR,
                 ask_color: str = DEFAULT_ASK_WALL_COLOR,
                 show_size: bool = True):
        super().__init__(orderbook)
        self.min_volume = float(min_volume)
        self.rel_multiple = float(rel_multiple)
        self.persistence_ms = int(persistence_ms)
        self.top_n = top_n
        self.bid_color = bid_color
        self.ask_color = ask_color
        self.show_size = bool(show_size)
        self._events: dict = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=True, renderer="none", exclude_from_autoscale=True,
            name="Deep Wall",
        )

    def display_name(self) -> str:
        return "DeepWall"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source) if source.ndim == 2 else 0
        out = np.full((3, max(n, 0)), np.nan, dtype=np.float64)
        if n == 0:
            self._events = {}
            return out
        tick_ts = source[:, 0].astype(np.int64)
        book = self._book_window()
        if book is None or len(book.get("ts", [])) == 0:
            self._events = {}
            return out
        ev = detect_walls(
            book, min_volume=self.min_volume, rel_multiple=self.rel_multiple,
            persistence_ms=self.persistence_ms, top_n=self.top_n,
        )
        self._events = ev
        idx = _anchor_idx(tick_ts, ev["first_ts"])
        if idx.size:
            out[0, idx] = ev["price"]
            out[1, idx] = ev["side"].astype(np.float64)
            out[2, idx] = ev["max_size"]
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        ev = getattr(self, "_events", {}) or {}
        price = np.asarray(ev.get("price", []), dtype=np.float64)
        if price.size == 0:
            return []
        first_ts = np.asarray(ev["first_ts"], dtype=np.int64)
        last_ts = np.asarray(ev["last_ts"], dtype=np.int64)
        side = np.asarray(ev["side"], dtype=np.int64)
        size = np.asarray(ev["max_size"], dtype=np.float64)
        pad = int(interval_ms or 60_000)
        x1 = np.where(last_ts > first_ts, last_ts, first_ts + pad)
        colors = [self.bid_color if s > 0 else self.ask_color for s in side]
        prims = [HLines(
            x0=list(first_ts), x1=list(x1), y=list(price),
            color=colors, alpha=0.7, width=3.0,
        )]
        if self.show_size:
            prims.append(Labels(
                x=list(x1), y=list(price),
                text=[format_magnitude(float(v)) for v in size],
                color=colors, font_size="8pt",
                x_offset=4, y_offset=0,
                text_align="left", text_baseline="middle",
            ))
        return prims


class DeepReload(_L2BookIndicator):
    """
    Liquidity replenishment (L2 iceberg-like), drawn as markers.

    A level repeatedly consumed and refilled is flagged as a reload. Source
    columns: ``[ts, price, volume, flags, bid, ask]`` (book from constructor).

    Outputs - [3 x N] anchored at each reload's tick (NaN elsewhere):
        row 0 : price, row 1 : side (+1 bid / -1 ask), row 2 : reloads.

    Args:
        orderbook: OrderbookProxy supplying the L2 book.
        drop_frac / recover_frac: Consume / refill fractions (see
            detect_reload_l2).
        reload_threshold (int): Reloads needed to flag a level.
        window_ms (int): Max gap between a consumption and the refill.
        top_n (int | None): Levels per side to scan.
        min_volume (float): Ignore levels below this reference size.
        color: Marker color.
    """

    name = "deep_reload"
    output_names = ["price"]
    ts_band_indices = [1, 2]
    ts_output_names = ["side", "reloads"]

    def __init__(self, orderbook=None, drop_frac: float = 0.5,
                 recover_frac: float = 0.8, reload_threshold: int = 2,
                 window_ms: int = 0, top_n: "int | None" = None,
                 min_volume: float = 0.0,
                 color: str = DEFAULT_RELOAD_COLOR):
        super().__init__(orderbook)
        self.drop_frac = float(drop_frac)
        self.recover_frac = float(recover_frac)
        self.reload_threshold = int(reload_threshold)
        self.window_ms = int(window_ms)
        self.top_n = top_n
        self.min_volume = float(min_volume)
        self.color = color
        self._events: dict = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=True, renderer="none", exclude_from_autoscale=True,
            name="Deep Reload",
        )

    def display_name(self) -> str:
        return "DeepReload"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source) if source.ndim == 2 else 0
        out = np.full((3, max(n, 0)), np.nan, dtype=np.float64)
        if n == 0:
            self._events = {}
            return out
        tick_ts = source[:, 0].astype(np.int64)
        book = self._book_window()
        if book is None or len(book.get("ts", [])) == 0:
            self._events = {}
            return out
        ev = detect_reload_l2(
            book, drop_frac=self.drop_frac, recover_frac=self.recover_frac,
            reload_threshold=self.reload_threshold, window_ms=self.window_ms,
            top_n=self.top_n, min_volume=self.min_volume,
        )
        self._events = ev
        idx = _anchor_idx(tick_ts, ev["ts"])
        if idx.size:
            out[0, idx] = ev["price"]
            out[1, idx] = ev["side"].astype(np.float64)
            out[2, idx] = ev["reloads"].astype(np.float64)
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        ev = getattr(self, "_events", {}) or {}
        ts = np.asarray(ev.get("ts", []), dtype=np.int64)
        if ts.size == 0:
            return []
        price = np.asarray(ev["price"], dtype=np.float64)
        reloads = np.asarray(ev["reloads"], dtype=np.int64)
        return [
            Points(
                x=list(ts), y=list(price), color=self.color, alpha=0.9,
                size=11, marker="diamond", fill=False, line_width=1.8,
            ),
            Labels(
                x=list(ts), y=list(price),
                text=[f"x{int(c)}" for c in reloads],
                color=self.color, font_size="8pt",
                x_offset=5, y_offset=0,
                text_align="left", text_baseline="middle",
            ),
        ]


class StopRun(_L2BookIndicator):
    """
    Stop sweeps (stop runs), drawn as a sweep segment plus reversal marker.

    A fast directional run of at least ``sweep_levels`` ticks within
    ``sweep_window_ms`` followed by a reversal back through the start within
    ``reversal_window_ms``. Operates on the trade (tick) price path; the book is
    accepted for API symmetry with the other L2 overlays. Source columns:
    ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [3 x N] anchored at each run's reversal tick (NaN elsewhere):
        row 0 : extreme_px - swept extreme price.
        row 1 : side       - +1 down-sweep then up / -1 up-sweep then down.
        row 2 : start_px   - sweep start (the level reclaimed on reversal).

    Args:
        orderbook: OrderbookProxy (optional; reserved for L2 confirmation).
        tick_size (float): Price step to count swept levels.
        sweep_levels (int): Minimum ticks moved to qualify as a sweep.
        sweep_window_ms (int): Max duration of the sweep run.
        reversal_window_ms (int): Window after the extreme to observe a reversal.
        up_color / down_color: Colors by reversal direction.
    """

    name = "stop_run"
    output_names = ["extreme_px"]
    ts_band_indices = [1, 2]
    ts_output_names = ["side", "start_px"]

    def __init__(self, orderbook=None, tick_size: float = 1.0,
                 sweep_levels: int = 3, sweep_window_ms: int = 1000,
                 reversal_window_ms: int = 3000,
                 up_color: str = DEFAULT_STOPRUN_UP_COLOR,
                 down_color: str = DEFAULT_STOPRUN_DOWN_COLOR):
        super().__init__(orderbook)
        self.tick_size = float(tick_size)
        self.sweep_levels = int(sweep_levels)
        self.sweep_window_ms = int(sweep_window_ms)
        self.reversal_window_ms = int(reversal_window_ms)
        self.up_color = up_color
        self.down_color = down_color
        self._events: dict = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=True, renderer="none", exclude_from_autoscale=True,
            name="Stop Run",
        )

    def display_name(self) -> str:
        return "StopRun"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source) if source.ndim == 2 else 0
        out = np.full((3, max(n, 0)), np.nan, dtype=np.float64)
        if n == 0:
            self._events = {}
            return out
        tick_ts = source[:, 0].astype(np.int64)
        price = source[:, 1].astype(np.float64)
        ev = detect_stop_run_l2(
            tick_ts, price, tick_size=self.tick_size,
            sweep_levels=self.sweep_levels, sweep_window_ms=self.sweep_window_ms,
            reversal_window_ms=self.reversal_window_ms,
        )
        self._events = ev
        idx = _anchor_idx(tick_ts, ev["ts"])
        if idx.size:
            out[0, idx] = ev["extreme_px"]
            out[1, idx] = ev["side"].astype(np.float64)
            out[2, idx] = ev["start_px"]
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        ev = getattr(self, "_events", {}) or {}
        rev_ts = np.asarray(ev.get("ts", []), dtype=np.int64)
        if rev_ts.size == 0:
            return []
        start_ts = np.asarray(ev["start_ts"], dtype=np.int64)
        start_px = np.asarray(ev["start_px"], dtype=np.float64)
        ext_ts = np.asarray(ev["extreme_ts"], dtype=np.int64)
        ext_px = np.asarray(ev["extreme_px"], dtype=np.float64)
        side = np.asarray(ev["side"], dtype=np.int64)
        colors = [self.down_color if s > 0 else self.up_color for s in side]
        return [
            Segments(
                x0=list(start_ts), y0=list(start_px),
                x1=list(ext_ts), y1=list(ext_px),
                color=colors, alpha=0.9, width=1.8, dash="dashed",
            ),
            Points(
                x=list(ext_ts), y=list(ext_px), color=colors, alpha=0.95,
                size=12, marker="inverted_triangle", fill=True, line_width=1.0,
            ),
            Labels(
                x=list(ext_ts), y=list(ext_px), text=["stop"] * len(ext_ts),
                color=colors, font_size="8pt",
                x_offset=0, y_offset=8,
                text_align="center", text_baseline="bottom",
            ),
        ]

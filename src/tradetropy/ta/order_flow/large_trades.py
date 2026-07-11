"""
LargeTrades - large trade ("whale") bubbles over price.

Mounted on a tick stream, ``LargeTrades`` flags trades (or aggregated execution
bursts) whose magnitude (raw ``volume``, ``notional`` or net ``delta``) stands
out against a causal threshold and exposes them as an order-flow overlay: each
detected event is drawn as a hollow circle on the price panel, sized by
magnitude and colored by aggressor side (buy = blue, sell = pink), with a small
label showing the real magnitude number. Detection is causal (relative
thresholds use only trailing events), so it is safe for backtesting and live
use.

This is a statistical detector of large *aggressive executions* read from the
time-and-sales (tick) feed; it does not use order-book depth. A future
``DeepTrades`` is reserved for a depth-of-market (DOM/MBO) detector
(absorption, icebergs, sweeps); see ``docs/design/deep_trades_dom.md``.

Usage::

    self.ticks  = self.subscribe_ticks("BTCUSDT")
    self.whales = self.add_indicator(
        LargeTrades.refs(self.ticks),
        LargeTrades(threshold="p99", by="notional"),
    )

    def on_data(self):
        if not np.isnan(self.whales.price[-1]):   # current tick is a large trade
            vol  = self.whales.volume[-1]
            notl = self.whales.notional[-1]
            side = self.whales.side[-1]            # +1 buy / -1 sell
"""

from __future__ import annotations

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.order_flow._core import (
    DEFAULT_BUY_COLOR,
    DEFAULT_SELL_COLOR,
    detect_large_trades,
    format_magnitude,
    merge_live_events,
)


class LargeTrades(Indicator):
    """
    Large-trade detector and bubble overlay (order flow).

    Source columns (use ``LargeTrades.refs(tick_proxy)`` or pass the tick proxy
    directly): ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [4 x N] (1 price band + 3 auxiliary bands):

    Price band (accessible by the user, drawn as bubbles):
        row 0 : price  - price of a detected large trade at this tick (NaN
                         otherwise).

    Auxiliary bands (excluded from rendering, exposed for ``on_data()`` access
    via ts_output_names):
        row 1 : volume   - traded volume of the large trade (NaN otherwise).
        row 2 : notional - price * volume of the large trade (NaN otherwise).
        row 3 : side     - aggressor side (+1 buy / -1 sell), NaN otherwise.

    Args:
        threshold (float | str): What makes a trade "large".
            - float: absolute magnitude threshold.
            - 'pXX': trailing quantile (e.g. 'p99' = top 1% within ``window``).
            - 'Nx': N times the trailing median (e.g. '5x').
        by (str): Magnitude metric: 'volume', 'notional' (price * volume) or
            'delta' (net aggression). 'delta' is only meaningful with
            ``aggregate_ms`` > 0, where buys and sells inside a burst net out.
        window (int): Trailing window (in events) for relative thresholds.
        aggregate_ms (int): Execution-burst window in ms. When > 0, consecutive
            trades are merged before thresholding (same-side for volume/notional,
            net signed for delta); 0 disables aggregation (per-trade detection).
        min_gap_ms (int): Minimum spacing in ms between detections (anti-cluster;
            0 disables it).
        scale (str): Bubble sizing: 'sqrt' (area proportional, default) or
            'linear'.
        min_size (float): Smallest bubble diameter in px.
        max_size (float): Largest bubble diameter in px.
        buy_color (str): Color for buy-aggressor bubbles (default blue).
        sell_color (str): Color for sell-aggressor bubbles (default pink).
        fill (bool): True fills the circle, False border only (hollow).
        fill_alpha (float): Fill transparency (0.0-1.0). Only applies
            when fill=True. None = uses default alpha (0.35).
        label_color (str | None): Color of the internal labels (magnitude
            labels). None = uses the same color as the circle border.
        line_width (float): Border width of the hollow circles.
        label (str | None): Magnitude number drawn next to each bubble:
            'volume', 'notional', 'delta', or None to hide labels. The label
            always shows the trade/burst's real magnitude (not a sequential
            index).
        label_font_size (str): Font size for the magnitude labels.
    """

    name = "large_trades"
    category = "annotation"

    output_names = ["price"]
    ts_band_indices = [1, 2, 3]
    ts_output_names = ["volume", "notional", "side"]

    def __init__(
        self,
        threshold="p99",
        by: str = "volume",
        window: int = 2000,
        aggregate_ms: int = 0,
        min_gap_ms: int = 0,
        scale: str = "sqrt",
        min_size: float = 10.0,
        max_size: float = 40.0,
        buy_color: str = DEFAULT_BUY_COLOR,
        sell_color: str = DEFAULT_SELL_COLOR,
        fill: bool = False,
        fill_alpha: float | None = None,
        label_color: str | None = None,
        line_width: float = 1.6,
        label: "str | None" = "volume",
        label_font_size: str = "7pt",
    ):
        if by not in ("volume", "notional", "delta"):
            raise ValueError(
                f"by must be 'volume', 'notional' or 'delta', not {by!r}"
            )
        if label not in (None, "volume", "notional", "delta"):
            raise ValueError(
                f"label must be None, 'volume', 'notional' or 'delta', "
                f"not {label!r}"
            )
        if scale not in ("sqrt", "linear"):
            raise ValueError(f"scale must be 'sqrt' or 'linear', not {scale!r}")

        self.threshold = threshold
        self.by = by
        self.window = int(window)
        self.aggregate_ms = int(aggregate_ms)
        self.min_gap_ms = int(min_gap_ms)
        self.scale = scale
        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.buy_color = buy_color
        self.sell_color = sell_color
        self.fill = bool(fill)
        self.fill_alpha = float(fill_alpha) if fill_alpha is not None else None
        self.label_color = label_color
        self.line_width = float(line_width)
        self.label = label
        self.label_font_size = label_font_size

        # length drives the engine warmup; relative thresholds need the window.
        self.length = self.window if isinstance(threshold, str) else 1

        # Filled by calculate for the plotting layer.
        self._deep_events: dict = {}
        self._deep_style: dict = self._build_style()

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="none",   # draw-only: bubbles come from draw(), no series line
            name="Large Trades",
        )

    @property
    def min_periods(self) -> int:
        return 1

    @staticmethod
    def refs(tick_proxy):
        """
        Build the ColumnRef list for this indicator in the expected order.

        Args:
            tick_proxy (TickProxy): Proxy returned by subscribe_ticks().

        Returns:
            list[ColumnRef]: [ts, price, volume, flags, bid, ask] refs.
        """
        return [
            tick_proxy.col_ref("ts"),
            tick_proxy.price_ref,
            tick_proxy.col_ref("volume"),
            tick_proxy.col_ref("flags"),
            tick_proxy.bid_ref,
            tick_proxy.ask_ref,
        ]

    def default_refs(self, proxy):
        """ColumnRefs resolved from a proxy so add_indicator(proxy, ind) works."""
        return type(self).refs(proxy)

    def display_name(self) -> str:
        return f"LargeTrades({self.threshold})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"largetrades_{symbol}"

    def _build_style(self) -> dict:
        """Snapshot of the visual configuration shared by both render paths."""
        return {
            "scale": self.scale,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "buy_color": self.buy_color,
            "sell_color": self.sell_color,
            "fill": self.fill,
            "fill_alpha": self.fill_alpha,
            "label_color": self.label_color,
            "line_width": self.line_width,
            "label": self.label,
            "label_font_size": self.label_font_size,
            "by": self.by,
        }

    def detect(self, source: np.ndarray) -> dict:
        """
        Run causal detection over a [N x 6] source matrix.

        Shared by calculate (backtest) and the live streaming path so both
        flag the exact same trades.

        Args:
            source (np.ndarray): [N x 6] columns [ts, price, volume, flags,
                bid, ask].

        Returns:
            dict: ``detect_large_trades`` result with keys 'mask', 'metric',
            'side' and 'events'.
        """
        ts = source[:, 0].astype(np.float64)
        price = source[:, 1].astype(np.float64)
        volume = source[:, 2].astype(np.float64)
        flags = source[:, 3] if source.shape[1] > 3 else None
        bid = source[:, 4] if source.shape[1] > 4 else None
        ask = source[:, 5] if source.shape[1] > 5 else None
        return detect_large_trades(
            ts, price, volume, flags=flags, bid=bid, ask=ask,
            threshold=self.threshold, by=self.by, window=self.window,
            min_gap_ms=self.min_gap_ms, aggregate_ms=self.aggregate_ms,
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Detect large trades and emit the [4 x N] band matrix.

        Bands are anchored at each detected event's representative tick (the
        last trade of an aggregated burst) and carry the burst's aggregated
        magnitudes, so ``self.volume`` / ``self.notional`` reflect the merged
        execution, not a single child print.

        Args:
            source (np.ndarray): [N x 6] columns [ts, price, volume, flags,
                bid, ask].

        Returns:
            np.ndarray: [4 x N] - row 0 price (NaN unless large), rows 1-3
            volume / notional / side (NaN unless large).
        """
        if source.ndim != 2 or source.shape[1] < 3 or len(source) == 0:
            self._deep_events = {}
            return np.full((4, 0), np.nan, dtype=np.float64)

        n = len(source)
        res = self.detect(source)
        mask = res["mask"]
        events = res["events"]

        out = np.full((4, n), np.nan, dtype=np.float64)
        idx = np.where(mask)[0]
        if idx.size:
            out[0, idx] = events["price"]
            out[1, idx] = events["volume"]
            out[2, idx] = events["notional"]
            out[3, idx] = events["side"].astype(np.float64)

        self._deep_events = {
            "ts": np.asarray(events["ts"], dtype=np.int64),
            "price": np.asarray(events["price"], dtype=np.float64),
            "volume": np.asarray(events["volume"], dtype=np.float64),
            "notional": np.asarray(events["notional"], dtype=np.float64),
            "delta": np.asarray(events["delta"], dtype=np.float64),
            "side": np.asarray(events["side"], dtype=np.int8),
            "metric": np.asarray(events["metric"], dtype=np.float64),
        }
        self._deep_style = self._build_style()
        return out

    def event_labels(self) -> list:
        """
        Build the magnitude label strings for the detected events.

        Returns:
            list[str]: One compact magnitude string per detected trade, or an
            empty list when ``label`` is None.
        """
        if not self.label or not self._deep_events:
            return []
        src = self._deep_events.get(self.label)
        if src is None:
            return []
        return [format_magnitude(float(v)) for v in src]

    def draw(self, cfg=None, *, interval_ms=None) -> dict:
        """
        Emit the large-trade bubbles as a Points primitive plus magnitude Labels.

        Bubbles are hollow circles (border only) sized per trade and colored by
        aggressor side; one label per bubble shows the real magnitude. Colors
        come from the indicator object (buy_color / sell_color), theme-independent.
        """
        from tradetropy.ta.draw import Points, Labels
        from tradetropy.ta.order_flow._core import build_bubble_columns

        events = getattr(self, "_deep_events", {}) or {}
        style = getattr(self, "_deep_style", None) or self._build_style()
        cols = build_bubble_columns(events, style, self.event_labels())
        if len(cols["ts"]) == 0:
            return {}

        prims: dict[str, list] = {}
        prims["Large Trades"] = [Points(
            x=list(cols["ts"]), y=list(cols["price"]),
            color=list(cols["color"]), alpha=0.95,
            size=list(cols["size"]), marker="circle",
            fill=style.get("fill", False),
            fill_alpha=style.get("fill_alpha"),
            line_width=style.get("line_width", 1.6),
        )]
        if style.get("label"):
            lbl_color = style.get("label_color") or list(cols["color"])
            prims["Large Trades Labels"] = [Labels(
                x=list(cols["ts"]), y=list(cols["price"]),
                text=list(cols["text"]), color=lbl_color,
                font_size=style.get("label_font_size", "8pt"),
                x_offset=0, y_offset=0,
                text_align="center", text_baseline="middle",
            )]
        return prims

    def live_refresh(self, proxy) -> None:
        """
        Re-run causal detection over the source tick window (live mode).

        Tick-mounted indicators are not recomputed by the standard live indicator
        path (which reads OHLC ring bands), so the generic live primitive updater
        calls this hook before draw(): it rebuilds the source matrix from the
        tick proxy's current window and reuses calculate for exact parity with
        the backtest. No-op when the proxy is empty.
        """
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
        self._accumulate_live_bubbles(ts)

    def _accumulate_live_bubbles(self, window_ts) -> None:
        """
        Persist bubbles across refreshes so replay matches the backtest window.

        ``calculate`` recomputes over the current tick-ring window, whose oldest
        ``window`` events sit in the relative-threshold warmup and are never
        flagged. Without this, a bubble detected while its tick was fresh (the
        same causal decision the full-series backtest makes) would disappear once
        the tick aged into that warmup band, so replay shows fewer bubbles than
        the backtest even within the same window. Merging the freshly-detected
        events into the accumulated set (pruned to the visible span) restores
        parity.

        Only applied for per-trade detection (``aggregate_ms == 0``); with burst
        aggregation an in-progress burst's anchor ts shifts as it grows, which
        would accumulate partial duplicates, so that mode keeps the plain
        windowed behavior.

        Args:
            window_ts (np.ndarray): Tick timestamps of the current window (ms).
        """
        if self.aggregate_ms != 0:
            return
        merged, self._live_accum_ts = merge_live_events(
            getattr(self, "_live_accum", None), self._deep_events, window_ts,
            getattr(self, "_live_accum_ts", None),
        )
        self._deep_events = merged
        self._live_accum = merged

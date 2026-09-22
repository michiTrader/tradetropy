"""
DeepTrades - depth-aware (L2) order-flow detector.

DeepTrades extends :class:`LargeTrades`: it detects the same outsized aggressive
executions from the time-and-sales feed, then classifies each one against the
resting L2 order-book liquidity it hit - read causally as-of the trade via an
OrderbookProxy - into one of:

    - Large Aggressor : a big print that neither sweeps the stack nor is
      absorbed by a wall.
    - Absorption      : a big print into a resting wall that holds (price stalls).
    - Sweep           : a single order that fully clears several book levels.

Because classification needs depth, DeepTrades requires an order book and is
meaningful in live mode (with a streaming order book) and in replay of a
recorded book. Si no se provee orderbook, se lanza ``ConfigError``; usa
``LargeTrades`` si no tienes datos de book.

The order book is supplied to the constructor (the proxy is not a numeric
column source, so it cannot be passed through the usual column refs):

    self.ticks = self.subscribe_ticks('BTCUSDT')
    self.book  = self.subscribe_orderbook('BTCUSDT', depth=20)
    self.deep  = self.add_indicator(
        DeepTrades.refs(self.ticks),
        DeepTrades(self.book, threshold='p99', by='notional'),
    )

    def on_data(self):
        if self.deep.event_type[-1] == 2:   # a sweep just printed
            ...
"""

from __future__ import annotations

import numpy as np

from tradetropy.exceptions import ConfigError
from tradetropy.ta.base import IndicatorPlotConfig
from tradetropy.ta.order_flow.large_trades import LargeTrades
from tradetropy.ta.order_flow._core import (
    DEEP_TRADE_LABELS,
    EVENT_ABSORPTION,
    EVENT_ICEBERG,
    EVENT_LIQUIDITY_GRAB,
    EVENT_SWEEP,
    DEFAULT_BUY_COLOR,
    DEFAULT_SELL_COLOR,
    apply_deep_autofilter,
    deep_trade_class_name,
    detect_deep_trades,
    detect_iceberg,
    detect_liquidity_grab,
    map_bubble_style,
    resolve_autofilter,
)

DEFAULT_ABSORPTION_COLOR = "#F59E0B"   # amber
DEFAULT_SWEEP_COLOR = "#8B5CF6"        # violet
DEFAULT_ICEBERG_COLOR = "#10B981"      # green
DEFAULT_GRAB_COLOR = "#EF4444"         # red


class DeepTrades(LargeTrades):
    """
    Depth-aware large-trade detector (L2 absorption / sweep classification).

    Source columns (use ``DeepTrades.refs(tick_proxy)``):
    ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [7 x N] (1 price band + 6 auxiliary bands):
        row 0 : price       - price of a detected event (NaN otherwise).
        row 1 : volume      - traded volume of the event.
        row 2 : notional    - price * volume.
        row 3 : side        - aggressor side (+1 buy / -1 sell).
        row 4 : event_type  - 0 aggressor / 1 absorption / 2 sweep.
        row 5 : resting     - resting size at the touched best level.
        row 6 : consumed    - number of book levels fully cleared.

    Args:
        orderbook: OrderbookProxy supplying the L2 book (book_as_of).
        threshold, by, window, aggregate_ms, min_gap_ms: large-trade detection
            (same semantics as LargeTrades).
        min_resting_volume (float): minimum resting size for a level to count as
            a wall (absorption candidate).
        absorption_ratio (float): traded / resting threshold for absorption.
        stack_depth (int): levels fully cleared to call a sweep.
        buy_color / sell_color: aggressor bubble colors (by side).
        absorption_color / sweep_color: colors for absorption / sweep events.
        label (str | None): 'class' (default) draws the classification tag;
            'volume' / 'notional' / 'delta' draw the magnitude; None hides labels.
        autofilter: Optional book-aware class filter (default None = off, every
            detected event is emitted). Accepts:
            - 'significant': keep only classified signals (absorption, sweep,
              iceberg, liquidity_grab), dropping the plain Large Aggressor noise.
            - a class name or an iterable of class names ('aggressor',
              'absorption', 'sweep', 'iceberg', 'liquidity_grab').
            The filter is book-aware: an event that could not be classified for
            lack of book depth (resting is NaN) is kept by default.
        keep_unclassified (bool): Controls the book-aware rule above. True
            (default) keeps unclassifiable events; False is strict mode -
            only positively classified events survive.
    """

    name = "deep_trades"
    category = "annotation"

    output_names = ["price"]
    ts_band_indices = [1, 2, 3, 4, 5, 6]
    ts_output_names = ["volume", "notional", "side", "event_type", "resting", "consumed"]

    def __init__(
        self,
        orderbook,
        threshold="p99",
        by: str = "volume",
        window: int = 2000,
        aggregate_ms: int = 0,
        min_gap_ms: int = 0,
        min_resting_volume: float = 0.0,
        absorption_ratio: float = 0.8,
        stack_depth: int = 3,
        scale: str = "sqrt",
        min_size: float = 8.0,
        max_size: float = 40.0,
        buy_color: str = DEFAULT_BUY_COLOR,
        sell_color: str = DEFAULT_SELL_COLOR,
        absorption_color: str = DEFAULT_ABSORPTION_COLOR,
        sweep_color: str = DEFAULT_SWEEP_COLOR,
        fill: bool = False,
        fill_alpha: float | None = None,
        label_color: str | None = None,
        line_width: float = 1.6,
        label: "str | None" = "class",
        label_font_size: str = "8pt",
        mbo=None,
        reload_threshold: int = 3,
        reload_window_ms: int = 2000,
        sweep_levels: int = 3,
        sweep_window_ms: int = 1000,
        reversal_window_ms: int = 3000,
        tick_size: float = 1.0,
        iceberg_color: str = DEFAULT_ICEBERG_COLOR,
        grab_color: str = DEFAULT_GRAB_COLOR,
        autofilter=None,
        keep_unclassified: bool = True,
    ):
        if orderbook is None:
            raise ConfigError(
                "DeepTrades requiere un OrderbookProxy. "
                "Usa LargeTrades si no tienes datos de orderbook."
            )
        if label not in (None, "class", "volume", "notional", "delta"):
            raise ValueError(
                f"label must be None, 'class', 'volume', 'notional' or "
                f"'delta', not {label!r}"
            )
        super().__init__(
            threshold=threshold, by=by, window=window,
            aggregate_ms=aggregate_ms, min_gap_ms=min_gap_ms,
            scale=scale, min_size=min_size, max_size=max_size,
            buy_color=buy_color, sell_color=sell_color,
            fill=fill, fill_alpha=fill_alpha, label_color=label_color,
            line_width=line_width,
            label=(None if label == "class" else label),
            label_font_size=label_font_size,
        )
        self._orderbook = orderbook
        self.min_resting_volume = float(min_resting_volume)
        self.absorption_ratio = float(absorption_ratio)
        self.stack_depth = int(stack_depth)
        self.absorption_color = absorption_color
        self.sweep_color = sweep_color
        self._class_label = label == "class"
        # L3 / MBO configuration.
        self._mbo = mbo
        self.reload_threshold = int(reload_threshold)
        self.reload_window_ms = int(reload_window_ms)
        self.sweep_levels = int(sweep_levels)
        self.sweep_window_ms = int(sweep_window_ms)
        self.reversal_window_ms = int(reversal_window_ms)
        self.tick_size = float(tick_size)
        self.iceberg_color = iceberg_color
        self.grab_color = grab_color
        # Book-aware class autofilter (None = off). Resolved once to a frozenset
        # of event-type codes so calculate() stays cheap.
        self.autofilter = autofilter
        self.keep_unclassified = bool(keep_unclassified)
        self._autofilter_classes = resolve_autofilter(autofilter)

    @staticmethod
    def refs(tick_proxy, orderbook_proxy=None):
        """
        Build the tick ColumnRef list (same columns as LargeTrades).

        ``orderbook_proxy`` is accepted for API symmetry but the book is read
        from the proxy passed to the constructor (it is not a column source).
        """
        return LargeTrades.refs(tick_proxy)

    def display_name(self) -> str:
        return f"DeepTrades({self.threshold})"

    @staticmethod
    def class_name(event_type) -> str:
        """
        Map an event_type band value to its class name (NaN-safe).

        Convenience for on_data() so a strategy can read the class by name
        instead of comparing magic numbers::

            if DeepTrades.class_name(self.deep.event_type[-1]) == 'sweep':
                ...

        Args:
            event_type: Numeric class code from the event_type band (NaN when
                the current tick is not a detected event).

        Returns:
            str: 'aggressor', 'absorption', 'sweep', 'iceberg' or
            'liquidity_grab'; '' when there is no event (NaN) at this tick.
        """
        return deep_trade_class_name(event_type)

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"deeptrades_{symbol}"

    def _book_fn(self):
        ob = self._orderbook
        if ob is None:
            return lambda ts: None
        return ob.book_as_of

    def detect(self, source: np.ndarray) -> dict:
        """Run causal large-trade detection + L2 classification over [N x 6]."""
        ts = source[:, 0].astype(np.float64)
        price = source[:, 1].astype(np.float64)
        volume = source[:, 2].astype(np.float64)
        flags = source[:, 3] if source.shape[1] > 3 else None
        bid = source[:, 4] if source.shape[1] > 4 else None
        ask = source[:, 5] if source.shape[1] > 5 else None
        return detect_deep_trades(
            ts, price, volume, self._book_fn(),
            flags=flags, bid=bid, ask=ask,
            threshold=self.threshold, by=self.by, window=self.window,
            min_gap_ms=self.min_gap_ms, aggregate_ms=self.aggregate_ms,
            min_resting_volume=self.min_resting_volume,
            absorption_ratio=self.absorption_ratio,
            stack_depth=self.stack_depth,
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """Detect + classify events and emit the [7 x N] band matrix."""
        if source.ndim != 2 or source.shape[1] < 3 or len(source) == 0:
            self._deep_events = {}
            return np.full((7, 0), np.nan, dtype=np.float64)

        n = len(source)
        res = self.detect(source)
        mask = res["mask"]
        events = res["events"]

        out = np.full((7, n), np.nan, dtype=np.float64)
        idx = np.where(mask)[0]
        if idx.size:
            out[0, idx] = events["price"]
            out[1, idx] = events["volume"]
            out[2, idx] = events["notional"]
            out[3, idx] = events["side"].astype(np.float64)
            out[4, idx] = events["event_type"].astype(np.float64)
            out[5, idx] = events["resting"]
            out[6, idx] = events["consumed"].astype(np.float64)

        self._deep_events = {
            "ts": np.asarray(events["ts"], dtype=np.int64),
            "price": np.asarray(events["price"], dtype=np.float64),
            "volume": np.asarray(events["volume"], dtype=np.float64),
            "notional": np.asarray(events["notional"], dtype=np.float64),
            "delta": np.asarray(events["delta"], dtype=np.float64),
            "side": np.asarray(events["side"], dtype=np.int8),
            "metric": np.asarray(events["metric"], dtype=np.float64),
            "event_type": np.asarray(events["event_type"], dtype=np.int8),
            "resting": np.asarray(events["resting"], dtype=np.float64),
            "consumed": np.asarray(events["consumed"], dtype=np.int64),
        }
        # L3 upgrade: when an MBO stream is available, reclassify coincident
        # events as Iceberg / Liquidity Grab (these dominate the L2 classes).
        if self._mbo is not None:
            self._apply_mbo_classification(out, idx)
        # Book-aware autofilter: drop low-conviction classes from out, idx and
        # _deep_events. Applied last so MBO upgrades (iceberg/grab) are kept.
        if self._autofilter_classes is not None and idx.size:
            self._apply_autofilter(out, idx)
        self._deep_style = self._build_style()
        return out

    def _apply_autofilter(self, out: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        Drop events not in the autofilter keep-set (book-aware) in place.

        Sets the band columns of dropped events back to NaN in ``out`` and
        prunes every per-event array in ``self._deep_events`` so on_data() and
        draw() see the same filtered set.

        Args:
            out (np.ndarray): The [7 x N] band matrix (mutated in place).
            idx (np.ndarray): Column indices of the detected events.

        Returns:
            np.ndarray: The filtered column indices (idx[keep]).
        """
        keep = apply_deep_autofilter(
            self._deep_events["event_type"],
            self._deep_events["resting"],
            self._autofilter_classes,
            keep_unclassified=self.keep_unclassified,
        )
        if keep.all():
            return idx
        out[:, idx[~keep]] = np.nan
        for key, arr in self._deep_events.items():
            self._deep_events[key] = arr[keep]
        return idx[keep]

    def _apply_mbo_classification(self, out: np.ndarray, idx: np.ndarray) -> None:
        """
        Upgrade coincident detected events to Iceberg / Liquidity Grab using the
        MBO event window. A large-trade event at (or just after) an iceberg /
        grab timestamp is reclassified; grabs take precedence over icebergs.

        Args:
            out (np.ndarray): The [7 x N] band matrix (event_type is row 4).
            idx (np.ndarray): Column indices of the detected events.
        """
        ev = self._mbo.events()
        if ev is None or len(ev) == 0 or len(self._deep_events.get("ts", [])) == 0:
            return
        ts = ev[:, 0]
        order_id = ev[:, 1]
        side = ev[:, 2]
        price = ev[:, 3]
        size = ev[:, 4]
        action = ev[:, 5]

        ice = detect_iceberg(
            ts, order_id, side, price, size, action,
            reload_threshold=self.reload_threshold,
            window_ms=self.reload_window_ms,
        )
        grab = detect_liquidity_grab(
            ts, price, action, tick_size=self.tick_size,
            sweep_levels=self.sweep_levels,
            sweep_window_ms=self.sweep_window_ms,
            reversal_window_ms=self.reversal_window_ms,
        )
        ice_ts = set(int(x) for x in ice["ts"])
        grab_ts = set(int(x) for x in grab["ts"])
        if not ice_ts and not grab_ts:
            return

        etypes = self._deep_events["event_type"]
        ev_ts = self._deep_events["ts"]
        for k, et in enumerate(etypes):
            t = int(ev_ts[k])
            if t in grab_ts:
                etypes[k] = EVENT_LIQUIDITY_GRAB
            elif t in ice_ts:
                etypes[k] = EVENT_ICEBERG
        # Reflect the upgraded classes in the output band (row 4).
        if idx.size:
            out[4, idx] = etypes.astype(np.float64)

    def event_labels(self) -> list:
        """Class tags ('aggressor'/'absorption'/'sweep') or magnitude labels."""
        if not self._deep_events:
            return []
        if self._class_label:
            etypes = self._deep_events.get("event_type")
            if etypes is None:
                return []
            return [DEEP_TRADE_LABELS.get(int(e), "") for e in etypes]
        return super().event_labels()

    def _event_colors(self) -> list:
        """Per-event colors: aggressor by side, absorption/sweep by class."""
        ev = self._deep_events or {}
        etypes = ev.get("event_type", np.zeros(0, dtype=np.int8))
        sides = ev.get("side", np.zeros(0, dtype=np.int8))
        colors = []
        for et, sd in zip(etypes, sides):
            if int(et) == EVENT_LIQUIDITY_GRAB:
                colors.append(self.grab_color)
            elif int(et) == EVENT_ICEBERG:
                colors.append(self.iceberg_color)
            elif int(et) == EVENT_SWEEP:
                colors.append(self.sweep_color)
            elif int(et) == EVENT_ABSORPTION:
                colors.append(self.absorption_color)
            else:
                colors.append(self.buy_color if sd > 0 else self.sell_color)
        return colors

    def draw(self, cfg=None, *, interval_ms=None) -> dict:
        """Emit classified bubbles (colored by class) plus class/magnitude labels."""
        from tradetropy.ta.draw import Points, Labels

        events = getattr(self, "_deep_events", {}) or {}
        ts = np.asarray(events.get("ts", []), dtype=np.int64)
        if len(ts) == 0:
            return {}

        style = getattr(self, "_deep_style", None) or self._build_style()
        price = np.asarray(events.get("price", []), dtype=np.float64)
        metric = np.asarray(events.get("metric", []), dtype=np.float64)
        side = np.asarray(events.get("side", []))
        sizes, _ = map_bubble_style(
            metric, side,
            scale=style.get("scale", "sqrt"),
            min_size=style.get("min_size", 8.0),
            max_size=style.get("max_size", 40.0),
        )
        colors = self._event_colors()
        labels = self.event_labels()
        if len(labels) != len(ts):
            labels = [""] * len(ts)

        prims: dict[str, list] = {}
        prims["Large Trades"] = [Points(
            x=list(ts), y=list(price), color=colors, alpha=0.95,
            size=list(sizes), marker="circle",
            fill=style.get("fill", False),
            fill_alpha=style.get("fill_alpha"),
            line_width=style.get("line_width", 1.6),
        )]
        if self.label is not None or self._class_label:
            lbl_color = style.get("label_color") or colors
            prims["Large Trades Labels"] = [Labels(
                x=list(ts), y=list(price), text=list(labels), color=lbl_color,
                font_size=style.get("label_font_size", "8pt"),
                x_offset=0, y_offset=0,
                text_align="center", text_baseline="middle",
            )]
        return prims

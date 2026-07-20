"""
ManualMarks - strategy-driven manual marks (segments / horizontal levels /
points / labels) with a start and an end.

Unlike every other indicator, ManualMarks does not detect anything from its
source: the source is only a causal clock (its ts column) so the indicator can
resolve a default "now" for an open-ended mark. The actual marks are pushed by
the strategy from on_data() through the on-demand public query API
(exposes_query_api = True, the same mechanism as Heatmap.liquidity_at or the
Volume Profile hvn/lvn nodes), so this is the supported way to let a strategy
draw ad-hoc lines with a start and an end from its own decision logic.
"""

from __future__ import annotations

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.exceptions import ConfigError

_DEFAULT_COLOR = '#3B82F6'


class ManualMarks(Indicator):
    """
    Strategy-driven manual marks: segments with a start and an end that the
    strategy adds/updates/removes from on_data().

    A mark is a line in price x time space defined by two endpoints
    (ts0, price0) -> (ts1, price1). Use the same price for both endpoints for
    a horizontal level; use different prices for a sloped line. ts1 (and
    price1) can be left open (None) while the mark is "live" - draw() extends
    it to the current bar/tick until the strategy closes it.

    This indicator draws no series (its ``clock`` band is a NaN placeholder,
    required only so ``add_indicator()`` returns a query-capable proxy - see
    ``n_outputs`` note above the class attributes). Attach it to any
    already-subscribed source purely as a causal clock
    (``ManualMarks.refs(proxy)``), then reach it through the handle
    ``add_indicator()`` returns.

    Usage:
        def init(self):
            self.btc = self.subscribe_ohlc('BTCUSDT', '5m')
            self.marks = self.add_indicator(
                ManualMarks.refs(self.btc), ManualMarks(),
            )

        def on_data(self):
            if some_signal:
                self.mark_id = self.marks.add_mark(
                    price0=self.btc.close[-1], ts0=self.ts,
                    color='#F6465D', label='Signal',
                )
            if close_condition and self.mark_id is not None:
                self.marks.close_mark(self.mark_id, ts1=self.ts,
                                       price1=self.btc.close[-1])

    Query API (through the add_indicator() handle):
        add_mark(price0, ts0=None, price1=None, ts1=None, color=None,
                  dash='solid', width=1.6, label=None, marker=None) -> int
            Adds a mark and returns its id. ts0 defaults to the current clock
            (self.ts equivalent); price1/ts1 left None keep the mark open
            (draw() extends it to the latest known ts). marker (optional)
            also drops a point at (ts0, price0) - e.g. to flag the pivot that
            triggered the mark.
        update_mark(mark_id, **fields) -> None
            Updates any field of an existing mark (e.g. drag the open end).
        close_mark(mark_id, ts1=None, price1=None) -> None
            Fixes the end of an open mark. ts1 defaults to the current clock;
            price1 defaults to the mark's own price0 (horizontal level).
        remove_mark(mark_id) -> None
            Deletes a mark (no-op if the id does not exist).
        clear_marks() -> None
            Deletes every mark.
        marks -> list[dict]
            Read-only snapshot of the current marks (copies).

    Args:
        max_marks (int): Ring cap on stored marks (oldest dropped first, 0 =
            unlimited). Keeps a long-running live session bounded.
    """

    name = 'manualmarks'
    category = 'annotation'
    source_cols = ('ts',)
    # Two named dummy bands (always NaN) - not because ManualMarks has price
    # series, but because add_indicator() only returns a MultiBandProxy (the
    # proxy that supports on-demand query delegation via __getattr__) when
    # n_outputs > 1; the slotted, single-band IndicatorProxy cannot delegate
    # (see creating-indicators.md, "Exposing a public query API"). This keeps
    # ManualMarks on the same supported mechanism as Heatmap / Volume Profile
    # instead of adding a new one-off path in the engine.
    output_names = ['clock', 'count']
    ts_band_indices: list = []
    exposes_query_api = True

    def __init__(self, max_marks: int = 500):
        self.max_marks = int(max_marks)
        self._marks: dict[int, dict] = {}
        self._next_id = 1
        self._last_ts: int | None = None

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer='none',
            plot=True,
            name='Marks',
        )

    @property
    def min_periods(self) -> int:
        return 1

    def display_name(self) -> str:
        return 'ManualMarks'

    def col_name(self, symbol: str, col_source: str = '') -> str:
        return f'manualmarks_{symbol}'

    @staticmethod
    def refs(proxy):
        """
        Build the ColumnRef list for this indicator: only a causal clock.

        Args:
            proxy (OhlcProxy | TickProxy): Any already-subscribed source.

        Returns:
            list[ColumnRef]: [ts] ref (close is not needed, kept minimal).
        """
        return [proxy.ts_ref]

    def calculate(self, source: np.ndarray) -> np.ndarray:
        ts_col = source[:, 0] if source.ndim == 2 else source
        n = len(ts_col)
        if n:
            self._last_ts = int(ts_col[-1])
        out = np.full((2, n), np.nan, dtype=np.float64)
        if n:
            out[1, -1] = float(len(self._marks))  # 'count' - current mark count
        return out

    # -----
    # Public query API (reached via the add_indicator() handle)
    # -----

    def add_mark(
        self,
        price0: float,
        ts0: "int | None" = None,
        price1: "float | None" = None,
        ts1: "int | None" = None,
        *,
        color: "str | None" = None,
        dash: str = 'solid',
        width: float = 1.6,
        alpha: float = 0.9,
        label: "str | None" = None,
        marker: "str | None" = None,
    ) -> int:
        """
        Add a manual mark and return its id.

        Args:
            price0 (float): Start price.
            ts0 (int | None): Start timestamp (epoch ms). None -> current
                causal clock (the ts of the last processed bar/tick).
            price1 (float | None): End price. None -> mark stays open
                (horizontal at price0 until closed or drawn to "now").
            ts1 (int | None): End timestamp. None -> mark stays open.
            color (str | None): Line/point color. None -> module default.
            dash (str): Line style ('solid'|'dashed'|'dotted'|'dashdot'|'dotdash').
            width (float): Line width.
            alpha (float): Line opacity.
            label (str | None): Optional floating text at the start point.
            marker (str | None): Optional marker shape dropped at (ts0, price0).

        Returns:
            int: The new mark's id (pass it to update_mark/close_mark/remove_mark).
        """
        ts0 = int(ts0) if ts0 is not None else self._last_ts
        if ts0 is None:
            raise ConfigError(
                'ManualMarks.add_mark(): ts0 is required until the first bar/'
                'tick has been processed (no causal clock yet).'
            )
        mark_id = self._next_id
        self._next_id += 1
        self._marks[mark_id] = {
            'id': mark_id,
            'ts0': ts0,
            'price0': float(price0),
            'ts1': int(ts1) if ts1 is not None else None,
            'price1': float(price1) if price1 is not None else None,
            'color': color or _DEFAULT_COLOR,
            'dash': dash,
            'width': float(width),
            'alpha': float(alpha),
            'label': label,
            'marker': marker,
        }
        if self.max_marks and len(self._marks) > self.max_marks:
            oldest_id = next(iter(self._marks))
            del self._marks[oldest_id]
        return mark_id

    def update_mark(self, mark_id: int, **fields) -> None:
        """
        Update fields of an existing mark (e.g. drag the open end as price
        moves). Unknown mark_id is a no-op.

        Args:
            mark_id (int): Id returned by add_mark().
            **fields: Any of price0/ts0/price1/ts1/color/dash/width/alpha/
                label/marker.
        """
        mark = self._marks.get(mark_id)
        if mark is None:
            return
        valid = {'price0', 'ts0', 'price1', 'ts1', 'color', 'dash', 'width',
                 'alpha', 'label', 'marker'}
        bad = set(fields) - valid
        if bad:
            raise ConfigError(f'ManualMarks.update_mark(): unknown fields {bad}')
        for key, value in fields.items():
            if key in ('ts0', 'ts1') and value is not None:
                value = int(value)
            elif key in ('price0', 'price1') and value is not None:
                value = float(value)
            mark[key] = value

    def close_mark(
        self,
        mark_id: int,
        ts1: "int | None" = None,
        price1: "float | None" = None,
    ) -> None:
        """
        Fix the end of an open mark.

        Args:
            mark_id (int): Id returned by add_mark().
            ts1 (int | None): End timestamp. None -> current causal clock.
            price1 (float | None): End price. None -> the mark's own price0
                (horizontal level).
        """
        mark = self._marks.get(mark_id)
        if mark is None:
            return
        mark['ts1'] = int(ts1) if ts1 is not None else self._last_ts
        mark['price1'] = float(price1) if price1 is not None else mark['price0']

    def remove_mark(self, mark_id: int) -> None:
        """Delete a mark. No-op if mark_id does not exist."""
        self._marks.pop(mark_id, None)

    def clear_marks(self) -> None:
        """Delete every mark."""
        self._marks.clear()

    @property
    def marks(self) -> list:
        """Read-only snapshot of the current marks (list of dict copies)."""
        return [dict(m) for m in self._marks.values()]

    # -----
    # Drawing
    # -----

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """
        Emit the current marks as Segments (sloped/horizontal lines), plus
        optional Points (markers) and Labels for marks that requested them.

        Open marks (price1/ts1 is None) are extended to the latest causal
        clock so a live/backtest mark is visible as soon as it is added,
        without waiting for it to be closed.
        """
        from tradetropy.ta.draw import Segments, Points, Labels

        if not self._marks:
            return []

        now_ts = self._last_ts
        # Segments.width/dash are scalar per primitive (color/alpha are per-
        # element), so marks are grouped by (dash, width) into one Segments
        # primitive per group instead of losing per-mark line style.
        groups: dict[tuple, dict] = {}
        pt_x, pt_y, pt_color, pt_marker_groups = [], [], [], {}
        lbl_x, lbl_y, lbl_text, lbl_color = [], [], [], []

        for mark in self._marks.values():
            ts0 = mark['ts0']
            price0 = mark['price0']
            ts1 = mark['ts1'] if mark['ts1'] is not None else (now_ts if now_ts is not None else ts0)
            price1 = mark['price1'] if mark['price1'] is not None else price0
            if ts1 < ts0:
                ts0, ts1 = ts1, ts0
                price0, price1 = price1, price0

            key = (mark['dash'], mark['width'])
            grp = groups.setdefault(key, {'x0': [], 'y0': [], 'x1': [], 'y1': [],
                                           'color': [], 'alpha': []})
            grp['x0'].append(int(ts0)); grp['y0'].append(float(price0))
            grp['x1'].append(int(ts1)); grp['y1'].append(float(price1))
            grp['color'].append(mark['color'])
            grp['alpha'].append(mark['alpha'])

            if mark['marker']:
                pt_x.append(int(mark['ts0'])); pt_y.append(float(mark['price0']))
                pt_color.append(mark['color'])
                pt_marker_groups.setdefault(mark['marker'], []).append(len(pt_x) - 1)

            if mark['label']:
                lbl_x.append(int(mark['ts0'])); lbl_y.append(float(mark['price0']))
                lbl_text.append(mark['label'])
                lbl_color.append(mark['color'])

        prims: list = []
        for (dash, width), grp in groups.items():
            prims.append(Segments(
                x0=grp['x0'], y0=grp['y0'], x1=grp['x1'], y1=grp['y1'],
                color=grp['color'], alpha=grp['alpha'], width=width, dash=dash,
            ))
        # Points.marker is also scalar per primitive: one Points group per
        # distinct marker shape.
        for marker_shape, idxs in pt_marker_groups.items():
            prims.append(Points(
                x=[pt_x[i] for i in idxs], y=[pt_y[i] for i in idxs],
                color=[pt_color[i] for i in idxs], marker=marker_shape,
            ))
        if lbl_x:
            prims.append(Labels(
                x=lbl_x, y=lbl_y, text=lbl_text, color=lbl_color,
                font_size='8pt', x_offset=4, y_offset=2,
                text_align='left', text_baseline='middle',
            ))
        return prims

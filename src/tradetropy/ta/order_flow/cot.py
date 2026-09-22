"""
Commitment of Traders (COT) order-flow indicator (GoCharting style).

``COT`` is a tick-mounted, per-bar order-flow overlay that prints the bar's
*commitment* figures as numeric labels over each candle, the way the COT study
behaves on GoCharting / Order Flow Talks (NOT the weekly CFTC report):

    - COT High (COTH): the intra-bar cumulative delta (running sum of signed
      aggressor volume, +volume buy / -volume sell, relative to the bar start)
      *at the last tick where price touched the bar's high*. It is the net
      aggression committed when price reached the top of the bar.
    - COT Low (COTL): the same cumulative delta *at the last tick where price
      touched the bar's low*.
    - Delta: the bar's net delta (buy volume - sell volume).

COTH / COTL are delta NUMBERS (signed volume), not price levels. A new high made
with a small or negative COTH (or a new low with a positive COTL) signals weak
commitment at the extreme - a classic order-flow divergence / exhaustion read.

The figures are anchored to each candle (they reset every bar), so there is no
session ``period`` to configure beyond the bar interval - match ``period`` to
your candle interval. Detection is causal (a bar depends only on the trades up
to it), so the result is identical in backtest, live and replay; the
tick-mounted ``live_refresh`` hook re-runs ``calculate`` over the live tick
window for parity, exactly like the other order-flow panels.

Usage::

    self.ticks = self.subscribe_ticks('BTCUSDT', window_size=5000)
    self.cot   = self.add_indicator(COT.refs(self.ticks), COT(period='1m'))

    def on_data(self):
        # Developing per-bar commitment figures (NaN except on the bar's last
        # tick), readable on the latest closed/forming bar:
        if self.cot.cot_high[-1] < 0:
            ...                       # new push up with no committed buying
"""

from __future__ import annotations

import numpy as np

from tradetropy.ta.base import IndicatorPlotConfig
from tradetropy.ta.draw import Labels
from tradetropy.ta.order_flow._core import bar_delta, format_magnitude
from tradetropy.ta.order_flow.delta import (
    DEFAULT_DOWN_COLOR,
    DEFAULT_TOTAL_COLOR,
    DEFAULT_UP_COLOR,
    _TickBarFlow,
)


def _signed_mag(value: float) -> str:
    """Compact signed magnitude label, e.g. '+1.2k' / '-226'."""
    if value is None or not np.isfinite(value):
        return ""
    sign = "+" if value >= 0 else ""
    return sign + format_magnitude(float(value))


class COT(_TickBarFlow):
    """
    Commitment of Traders per-bar figures (GoCharting style, price overlay).

    Source columns (use ``COT.refs(tick_proxy)``):
    ``[ts, price, volume, flags, bid, ask]``.

    Outputs - [3 x N], anchored at each bar's last tick (NaN elsewhere):
        row 0 : cot_high - cumulative delta at the bar's high (COTH).
        row 1 : cot_low  - cumulative delta at the bar's low (COTL).
        row 2 : delta    - net bar delta (buy - sell).

    Args:
        period (str | int): Bar size ('1m', '5m', ... or ms). Match it to your
            chart's candle interval to align the labels with the candles.
        up_color: COTH (buy-committed) label color.
        down_color: COTL (sell-committed) label color.
        delta_color: Delta label color (overridden per-bar by sign when
            ``delta_sign_color`` is True).
        delta_sign_color (bool): Color the Delta label green/red by its sign.
        text_offset (int): Vertical pixel gap from the candle high/low to the
            COTH / COTL labels.
        font_size (str): Label font size.
        anchor (int): Epoch-ms origin of the bar grid.
    """

    name = "cot"
    output_names = ["cot_high", "cot_low", "delta"]

    def __init__(self, period="1m",
                 up_color: str = DEFAULT_UP_COLOR,
                 down_color: str = DEFAULT_DOWN_COLOR,
                 delta_color: str = DEFAULT_TOTAL_COLOR,
                 delta_sign_color: bool = True,
                 text_offset: int = 6,
                 font_size: str = "6pt",
                 anchor: int = 0):
        super().__init__(period=period, anchor=anchor)
        self.up_color = up_color
        self.down_color = down_color
        self.delta_color = delta_color
        self.delta_sign_color = bool(delta_sign_color)
        self.text_offset = int(text_offset)
        self.font_size = font_size
        self._bars: dict = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=True, renderer="none", name="COT",
            exclude_from_autoscale=True, show_legend=True,
        )

    def display_name(self) -> str:
        return f"COT({self.period})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"cot_{self.period}_{symbol}"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 3 or len(source) == 0:
            self._bars = {}
            return np.full((3, 0), np.nan, dtype=np.float64)
        n = len(source)
        ts, price, volume, flags, bid, ask = self._split_source(source)
        bars = bar_delta(ts, price, volume, interval_ms=self.interval_ms,
                         flags=flags, bid=bid, ask=ask, anchor=self.anchor)
        self._bars = bars
        out = np.full((3, n), np.nan, dtype=np.float64)
        idx = bars["rep_idx"]
        if idx.size:
            out[0, idx] = bars["cot_high"]
            out[1, idx] = bars["cot_low"]
            out[2, idx] = bars["delta"]
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> dict:
        bars = getattr(self, "_bars", {}) or {}
        bar_ts = np.asarray(bars.get("bar_ts", []), dtype=np.int64)
        if bar_ts.size == 0:
            return {}
        x = list(bar_ts)
        nb = len(bar_ts)
        hi = np.asarray(bars["high_price"], dtype=np.float64)
        lo = np.asarray(bars["low_price"], dtype=np.float64)
        cot_high = np.asarray(bars["cot_high"], dtype=np.float64)
        cot_low = np.asarray(bars["cot_low"], dtype=np.float64)
        delta = np.asarray(bars["delta"], dtype=np.float64)

        off = self.text_offset

        # COT High: above the candle high.
        coth = Labels(
            x=x, y=list(hi), text=[f"COTH:{_signed_mag(v)}" for v in cot_high],
            color=self.up_color, font_size=self.font_size,
            x_offset=0, y_offset=off,
            text_align="center", text_baseline="bottom",
        )
        # Delta: stacked above COT High.
        if self.delta_sign_color:
            delta_color = [self.up_color if d >= 0 else self.down_color
                           for d in delta]
        else:
            delta_color = self.delta_color
        dlt = Labels(
            x=x, y=list(hi), text=[f"Delta:{_signed_mag(v)}" for v in delta],
            color=delta_color, font_size=self.font_size,
            x_offset=0, y_offset=off + 14,
            text_align="center", text_baseline="bottom",
        )
        # COT Low: below the candle low.
        cotl = Labels(
            x=x, y=list(lo), text=[f"COTL:{_signed_mag(v)}" for v in cot_low],
            color=self.down_color, font_size=self.font_size,
            x_offset=0, y_offset=-off,
            text_align="center", text_baseline="top",
        )
        return {"COT": [coth, cotl, dlt]}

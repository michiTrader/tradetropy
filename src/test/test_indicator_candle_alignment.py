"""
Tests for align_indicators_to_candle (phase-aware candle-grid alignment of the
draw() primitives every indicator emits).

Indicators draw geometry (LargeTrades bubbles, Volume Profile bars, COT labels,
zones, ...) at the exact event time - a tick's millisecond timestamp - which
puts a bubble at 12:10:17.4 on a chart whose 15s candles sit at 12:10:15. The
plotting layer floors every time-valued field of every draw primitive to the
candle it belongs to (via _align_ts_to_candle), so all indicator geometry shares
the OHLC time unit. Per-bar value series are already on the OHLC grid (resampled
to the candle index), so only draw() geometry needs this.

The same _snap_primitive_groups helper is applied on the static plot path
(plotting.plot) and the live/replay path (_vp_mixin), gated by the global
PlotConfig.align_indicators_to_candle flag (default True).
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.ta import LargeTrades
from tradetropy.ta.draw import HBars, HLines, Labels, Points, Rects, Segments


_INTERVAL = 60_000
# A candle origin whose remainder mod the interval is 20_000, i.e. candles open
# at ...000000, ...060000, ... rather than on the raw epoch grid. Exercises the
# phase handling (the classic fractional-candle offset).
_ORIGIN = 1_700_000_000_000
assert _ORIGIN % _INTERVAL == 20_000


@pytest.mark.unit
class TestSnapPrimitiveGroups:
    def _snap(self, groups, origin=_ORIGIN):
        from tradetropy.plotting._util import _snap_primitive_groups
        return _snap_primitive_groups(groups, _INTERVAL, origin)

    def test_points_snapped_to_phase_grid(self):
        # x at +50s and +65s past two candle opens -> floored to those opens.
        p = Points(x=[_ORIGIN + 50_000, _ORIGIN + 65_000], y=[1.0, 2.0],
                   color="#fff")
        self._snap({"g": [p]})
        assert list(np.asarray(p.x)) == [_ORIGIN, _ORIGIN + _INTERVAL]

    def test_points_snapped_spans_untouched(self):
        off = _ORIGIN + 17_400  # 17.4s into the first candle
        off2 = _ORIGIN + _INTERVAL + 3_000
        prims = {
            "hl":  [HLines(x0=[off], x1=[off2], y=[1.0], color="#fff")],
            "seg": [Segments(x0=[off], y0=[1.0], x1=[off2], y1=[2.0], color="#fff")],
            "pts": [Points(x=[off], y=[1.0], color="#fff")],
            "hb":  [HBars(y=[1.0], height=[0.5], left=[off], right=[off2],
                          color="#fff")],
            "rc":  [Rects(x0=[off], x1=[off2], y0=[1.0], y1=[2.0],
                          fill_color="#fff")],
            "lb":  [Labels(x=[off], y=[1.0], text=["x"])],
        }
        self._snap(prims)
        assert list(np.asarray(prims["hl"][0].x0)) == [off]
        assert list(np.asarray(prims["hl"][0].x1)) == [off2]
        assert list(np.asarray(prims["seg"][0].x0)) == [off]
        assert list(np.asarray(prims["seg"][0].x1)) == [off2]
        assert list(np.asarray(prims["pts"][0].x)) == [_ORIGIN]
        assert list(np.asarray(prims["hb"][0].left)) == [off]
        assert list(np.asarray(prims["hb"][0].right)) == [off2]
        assert list(np.asarray(prims["rc"][0].x0)) == [off]
        assert list(np.asarray(prims["rc"][0].x1)) == [off2]
        assert list(np.asarray(prims["lb"][0].x)) == [_ORIGIN]

    def test_non_time_fields_untouched(self):
        p = Points(x=[_ORIGIN + 50_000], y=[123.45], color="#abc",
                   size=[9], alpha=0.7)
        self._snap({"g": [p]})
        assert list(p.y) == [123.45]
        assert p.color == "#abc"
        assert list(p.size) == [9]

    def test_epoch_grid_when_origin_zero(self):
        p = Points(x=[125_000, 60_001], y=[1.0, 2.0], color="#fff")
        self._snap({"g": [p]}, origin=0)
        assert list(np.asarray(p.x)) == [120_000, 60_000]

    def test_empty_arrays_noop(self):
        p = Points(x=[], y=[], color="#fff")
        self._snap({"g": [p]})
        assert list(p.x) == []

    def test_zero_interval_noop(self):
        from tradetropy.plotting._util import _snap_primitive_groups
        p = Points(x=[123, 456], y=[1.0, 2.0], color="#fff")
        _snap_primitive_groups({"g": [p]}, 0, 0)
        assert list(p.x) == [123, 456]

    def test_none_groups_noop(self):
        from tradetropy.plotting._util import _snap_primitive_groups
        assert _snap_primitive_groups(None, _INTERVAL, _ORIGIN) is None


@pytest.mark.unit
class TestPlotConfigFlag:
    def test_default_is_on(self):
        from tradetropy.plotting.config import PlotConfig
        assert PlotConfig().align_indicators_to_candle is True

    def test_can_disable(self):
        from tradetropy.plotting.config import PlotConfig
        assert PlotConfig(align_indicators_to_candle=False).align_indicators_to_candle is False


def _ticks_with_whales():
    """
    Build a tick matrix with strictly-increasing, off-grid timestamps and a
    handful of large trades so LargeTrades emits bubbles at sub-candle times.

    A fixed 997 ms stride (coprime with the 60s interval) keeps every timestamp
    unique and almost never on a candle open, while spanning ~10 candles.
    """
    rng = np.random.default_rng(0)
    n = 600
    ts = (_ORIGIN + 17_400 + np.arange(n) * 997).astype(np.float64)

    price = 50_000 + np.cumsum(rng.standard_normal(n) * 2)
    volume = rng.uniform(0.5, 5.0, n)
    # Inject clearly-large trades (volume >> threshold) at a few rows.
    for r in (50, 150, 320, 480, 559):
        volume[r] = 5_000.0
    flags = np.zeros(n)  # unknown aggressor -> tick-rule fallback

    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = ts
    out[:, _TICK_COL["bid"]] = price
    out[:, _TICK_COL["ask"]] = price
    out[:, _TICK_COL["volume"]] = volume
    out[:, _TICK_COL["flags"]] = flags
    out[:, _TICK_COL["price"]] = price
    return out


class _Whale(Strategy):
    def init(self):
        self.t = self.subscribe_ticks("BTCUSDT", window_size=5_000)
        self.b = self.subscribe_ohlc("BTCUSDT", _INTERVAL, window_size=500)
        self.w = self.add_indicator(
            LargeTrades.refs(self.t),
            LargeTrades(threshold=100.0, by="volume", window=50),
        )

    def on_data(self):
        pass


@pytest.mark.unit
class TestStaticIntegration:
    def _gather(self):
        from tradetropy import BacktestEngine
        from tradetropy.plotting._util import _gather_plot_data
        eng = BacktestEngine.by_ticks(
            _Whale(), data=(TickData("BTCUSDT", _ticks_with_whales(), tick_size=0.5),))
        eng.run()
        return _gather_plot_data(eng)

    def _bubble_meta(self, data):
        for meta in data["indicators"]:
            prims = getattr(meta, "_draw_primitives", None)
            if prims and any("Large Trades" in k for k in prims):
                return meta
        return None

    def test_bubbles_are_off_grid_before_snapping(self):
        # Sanity: the real draw() output carries raw sub-candle tick timestamps,
        # so the snap is actually doing something.
        data = self._gather()
        meta = self._bubble_meta(data)
        assert meta is not None, "LargeTrades emitted no bubbles"
        opens = data["ohlc_array"][:, 0].astype(np.int64)
        pts = meta._draw_primitives["Large Trades"][0]
        x_raw = np.asarray(list(pts.x), dtype=np.int64)
        assert x_raw.size > 0
        assert not np.all(np.isin(x_raw, opens))

    def test_snapped_bubbles_land_on_candle_opens(self):
        from tradetropy.plotting._util import _snap_primitive_groups, _align_ts_to_candle
        data = self._gather()
        meta = self._bubble_meta(data)
        assert meta is not None
        opens = data["ohlc_array"][:, 0].astype(np.int64)
        origin = int(opens[0])
        pts = meta._draw_primitives["Large Trades"][0]
        x_before = np.asarray(list(pts.x), dtype=np.int64)

        _snap_primitive_groups(meta._draw_primitives, _INTERVAL, origin)

        x_after = np.asarray(list(pts.x), dtype=np.int64)
        # Every bubble now sits on an actual candle open ...
        assert np.all(np.isin(x_after, opens))
        # ... on the SAME candle the raw tick fell in (floor, not shift).
        expected = _align_ts_to_candle(x_before, _INTERVAL, origin)
        assert list(x_after) == list(np.asarray(expected, dtype=np.int64))

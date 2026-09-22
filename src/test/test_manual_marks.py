"""Tests for ManualMarks (strategy-driven manual marks with start/end)."""

import numpy as np

from tradetropy.ta.manual_marks import ManualMarks
from tradetropy.ta.draw import Segments, Points, Labels


def _rows(n=5, interval_ms=60_000):
    ts = np.arange(n, dtype=np.float64) * interval_ms
    return ts


class TestManualMarksCore:
    def test_calculate_is_nan_and_advances_clock(self):
        mm = ManualMarks()
        ts = _rows(5)
        out = mm.calculate(ts.reshape(-1, 1))
        assert out.shape == (2, 5)
        assert np.all(np.isnan(out[0]))
        assert mm._last_ts == int(ts[-1])

    def test_add_mark_requires_clock_or_explicit_ts0(self):
        mm = ManualMarks()
        try:
            mm.add_mark(price0=100.0)
            assert False, "expected ConfigError without a clock"
        except Exception as exc:
            assert "ts0" in str(exc)
        # Explicit ts0 works even with no clock yet.
        mark_id = mm.add_mark(price0=100.0, ts0=0)
        assert mark_id == 1

    def test_open_mark_extends_to_current_clock_on_draw(self):
        mm = ManualMarks()
        ts = _rows(5)
        mm.calculate(ts.reshape(-1, 1))
        mark_id = mm.add_mark(price0=100.0, ts0=int(ts[1]), color="#FF0000")
        prims = mm.draw()
        segs = [p for p in prims if isinstance(p, Segments)]
        assert len(segs) == 1
        seg = segs[0]
        assert seg.x0[0] == int(ts[1])
        assert seg.x1[0] == int(ts[-1])   # extended to the last causal clock
        assert seg.y0[0] == 100.0
        assert seg.y1[0] == 100.0         # open end mirrors price0 (horizontal)

    def test_close_mark_fixes_the_end(self):
        mm = ManualMarks()
        ts = _rows(5)
        mm.calculate(ts.reshape(-1, 1))
        mark_id = mm.add_mark(price0=100.0, ts0=int(ts[0]))
        mm.close_mark(mark_id, ts1=int(ts[3]), price1=110.0)
        prims = mm.draw()
        seg = [p for p in prims if isinstance(p, Segments)][0]
        assert seg.x1[0] == int(ts[3])
        assert seg.y1[0] == 110.0

    def test_sloped_mark_start_end_different_prices(self):
        mm = ManualMarks()
        mark_id = mm.add_mark(price0=100.0, ts0=0, price1=120.0, ts1=60_000)
        seg = [p for p in mm.draw() if isinstance(p, Segments)][0]
        assert seg.y0[0] == 100.0 and seg.y1[0] == 120.0

    def test_update_mark_changes_fields(self):
        mm = ManualMarks()
        mark_id = mm.add_mark(price0=100.0, ts0=0)
        mm.update_mark(mark_id, color="#00FF00", price0=105.0)
        m = mm.marks[0]
        assert m["color"] == "#00FF00"
        assert m["price0"] == 105.0

    def test_remove_and_clear(self):
        mm = ManualMarks()
        id1 = mm.add_mark(price0=100.0, ts0=0)
        id2 = mm.add_mark(price0=101.0, ts0=0)
        mm.remove_mark(id1)
        assert len(mm.marks) == 1
        mm.clear_marks()
        assert mm.marks == []
        assert mm.draw() == []

    def test_max_marks_caps_oldest_dropped(self):
        mm = ManualMarks(max_marks=2)
        mm.add_mark(price0=1.0, ts0=0)
        mm.add_mark(price0=2.0, ts0=0)
        mm.add_mark(price0=3.0, ts0=0)
        prices = sorted(m["price0"] for m in mm.marks)
        assert prices == [2.0, 3.0]

    def test_marker_and_label_emit_points_and_labels(self):
        mm = ManualMarks()
        mm.add_mark(price0=100.0, ts0=0, ts1=1000, price1=100.0,
                    marker="circle", label="Signal")
        prims = mm.draw()
        assert any(isinstance(p, Points) for p in prims)
        assert any(isinstance(p, Labels) for p in prims)

    def test_dash_width_grouping_produces_separate_segments(self):
        mm = ManualMarks()
        mm.add_mark(price0=1.0, ts0=0, ts1=10, dash="solid", width=1.0)
        mm.add_mark(price0=2.0, ts0=0, ts1=10, dash="dashed", width=2.0)
        segs = [p for p in mm.draw() if isinstance(p, Segments)]
        assert len(segs) == 2
        dashes = sorted(s.dash for s in segs)
        assert dashes == ["dashed", "solid"]


class TestManualMarksQueryApiWiring:
    """exposes_query_api wires add_mark/close_mark through add_indicator()'s
    returned handle, the same mechanism as Heatmap's query API."""

    def test_indicator_declares_query_api_and_multi_band(self):
        mm = ManualMarks()
        assert mm.exposes_query_api is True
        # n_outputs must be > 1 so add_indicator() wires a MultiBandProxy
        # (the only proxy type that supports query delegation).
        assert mm.n_outputs > 1
        assert mm.output_names == ['clock', 'count']

    def test_refs_returns_ts_ref_only(self):
        class FakeProxy:
            ts_ref = object()

        refs = ManualMarks.refs(FakeProxy())
        assert refs == [FakeProxy.ts_ref]

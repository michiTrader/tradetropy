"""Tests for the neutral draw-primitive contract (tradetropy.ta.draw)."""

import numpy as np


class TestSegmentsPrimitive:
    """The Segments primitive (arbitrary endpoints) renders via the generic path."""

    def test_reexported_from_tool_package(self):
        from tradetropy.ta.draw import Segments as S1, Primitive
        from tradetropy.ta.tool import Segments as S2
        assert S1 is S2
        # Segments is part of the Primitive union.
        seg = S1(x0=[0], y0=[1.0], x1=[10], y1=[2.0], color="#123456")
        assert isinstance(seg, Primitive.__args__)

    def test_build_data_columns(self):
        from tradetropy.ta.draw import Segments
        from tradetropy.plotting.render._tools import _build_data

        seg = Segments(
            x0=[0, 100], y0=[1.0, 3.0], x1=[50, 150], y1=[2.0, 4.0],
            color=["#FF0000", "#00FF00"], alpha=[0.5, 0.9], width=2.0,
        )
        data = _build_data(Segments, [seg], theme=None)
        assert data is not None
        assert list(data["color"]) == ["#FF0000", "#00FF00"]
        np.testing.assert_allclose(data["y0"], [1.0, 3.0])
        np.testing.assert_allclose(data["y1"], [2.0, 4.0])
        np.testing.assert_allclose(data["alpha"], [0.5, 0.9])
        assert data["x0"].dtype == np.dtype("datetime64[ms]")
        assert data["x1"].dtype == np.dtype("datetime64[ms]")

    def test_renderer_creates_glyph(self):
        from bokeh.plotting import figure
        from tradetropy.ta.draw import Segments
        from tradetropy.plotting.render._tools import render_tool_groups

        seg = Segments(
            x0=[0], y0=[1.0], x1=[60_000], y1=[2.0], color="#3B82F6",
        )
        fig = figure(x_axis_type="datetime")
        before = len(fig.renderers)
        registry = render_tool_groups(
            fig, {"Swing": [seg]}, theme=None, interval_ms=60_000, registry={},
        )
        assert len(fig.renderers) > before
        assert ("Swing", "Segments") in registry

    def test_live_update_in_place(self):
        """A second render with the same registry updates data without new glyphs."""
        from bokeh.plotting import figure
        from tradetropy.ta.draw import Segments
        from tradetropy.plotting.render._tools import render_tool_groups

        fig = figure(x_axis_type="datetime")
        registry = render_tool_groups(
            fig, {"Swing": [Segments(x0=[0], y0=[1.0], x1=[10], y1=[2.0], color="#000")]},
            theme=None, interval_ms=60_000, registry={},
        )
        n_after_first = len(fig.renderers)
        render_tool_groups(
            fig, {"Swing": [Segments(x0=[0, 5], y0=[1.0, 2.0], x1=[10, 15],
                                     y1=[2.0, 3.0], color="#000")]},
            theme=None, interval_ms=60_000, registry=registry,
        )
        # No new glyph created; the existing source was updated in place.
        assert len(fig.renderers) == n_after_first
        assert len(registry[("Swing", "Segments")].data["y0"]) == 2


class TestAnnotationDrawMigration:
    """Annotations emit declarative primitives via Indicator.draw()."""

    def test_fvg_emits_rects_with_extension_and_mitigation(self):
        import numpy as np
        from tradetropy.ta.annotations import FairValueGap
        from tradetropy.ta.draw import Rects

        # Bars: ts, open, high, low, close, vol.
        # Bullish FVG at bar 1: low[2]=105 > high[0]=100. Open zone (never
        # mitigated here) must extend 3 bars to the right.
        rows = np.array([
            [0,       99, 100, 98, 99,  1],
            [60_000, 100, 101, 99, 101, 1],
            [120_000, 106, 107, 105, 106, 1],
            [180_000, 106, 107, 104, 105, 1],
        ], dtype=np.float64)
        fvg = FairValueGap(mitigate=True, min_gap=0.0)
        fvg.calculate(rows)
        prims = fvg.draw(fvg.plot_config, interval_ms=60_000)
        assert len(prims) == 1 and isinstance(prims[0], Rects)
        r = prims[0]
        assert len(r.x0) == 1
        # Open zone extends 3 bars from ts_open.
        ts_open = int(r.x0[0])
        assert int(r.x1[0]) == ts_open + 3 * 60_000

    def test_geometric_indicator_skips_series_render(self):
        from tradetropy.plotting.plotting import _is_geometric_renderer
        assert _is_geometric_renderer("rect")
        assert _is_geometric_renderer("span")
        assert _is_geometric_renderer(["rect", "rect"])
        assert not _is_geometric_renderer("step")
        assert not _is_geometric_renderer("line")


class TestPointCloudDecimation:
    """Zoom-aware level-of-detail decimation for large scatter clouds."""

    def test_decimate_indices_caps_and_orders(self):
        from tradetropy.plotting.render._tools import _decimate_indices

        # Small clouds are untouched (all indices, in order).
        small = _decimate_indices(10, 4000)
        assert list(small) == list(range(10))

        # Large clouds are subsampled to the cap, strictly increasing, in range.
        big = _decimate_indices(100_000, 4000)
        assert len(big) == 4000
        assert big[0] == 0
        assert big.max() < 100_000
        assert np.all(np.diff(big) >= 0)

    def test_decimate_point_data_preserves_columns(self):
        from tradetropy.plotting.render._tools import _decimate_point_data

        n = 10_000
        data = {
            "x": np.arange(n).astype("datetime64[ms]"),
            "y": np.arange(n, dtype=np.float64),
            "size": np.full(n, 6.0),
            "color": [f"#{i:06x}" for i in range(n)],       # list column
            "fill_color": [f"#{i:06x}" for i in range(n)],
        }
        out = _decimate_point_data(data, 4000)
        # Every column is present and shares the reduced length.
        assert set(out) == set(data)
        lengths = {len(v) for v in out.values()}
        assert lengths == {4000}
        # List columns stay lists; array columns stay arrays.
        assert isinstance(out["color"], list)
        assert isinstance(out["y"], np.ndarray)

    def test_large_points_cloud_draws_decimated_view_with_backing_source(self):
        from bokeh.plotting import figure
        from bokeh.models import Range1d
        from tradetropy.ta.draw import Points
        from tradetropy.plotting.render._tools import (
            render_tool_groups, _POINTS_LOD_THRESHOLD, _POINTS_LOD_MAX_VISIBLE,
        )

        n = _POINTS_LOD_THRESHOLD + 5_000
        pts = Points(
            x=list(range(n)), y=[float(i) for i in range(n)],
            color="#FF0000", size=6.0,
        )
        fig = figure(x_axis_type="datetime", x_range=Range1d(start=0, end=n))
        registry = render_tool_groups(
            fig, {"Cloud": [pts]}, theme=None, interval_ms=60_000, registry={},
            lazy_x_range=fig.x_range, enable_point_lod=True,
        )
        key = ("Cloud", "Points")
        assert key in registry
        # Backing (full) source keeps all points.
        assert len(registry[key].data["x"]) == n
        # A zoom callback was attached to refine the drawn view on pan/zoom.
        assert len(fig.x_range.js_property_callbacks) > 0
        # The drawn glyph reads a decimated source capped at max_visible.
        scatter_renderers = [r for r in fig.renderers
                             if r.data_source is not registry[key]]
        assert scatter_renderers, "expected a separate drawn (view) source"
        drawn = scatter_renderers[0].data_source
        assert len(drawn.data["x"]) == _POINTS_LOD_MAX_VISIBLE

    def test_small_points_cloud_not_decimated(self):
        from bokeh.plotting import figure
        from bokeh.models import Range1d
        from tradetropy.ta.draw import Points
        from tradetropy.plotting.render._tools import render_tool_groups

        n = 100
        pts = Points(x=list(range(n)), y=[float(i) for i in range(n)],
                     color="#00FF00", size=6.0)
        fig = figure(x_axis_type="datetime", x_range=Range1d(start=0, end=n))
        registry = render_tool_groups(
            fig, {"Cloud": [pts]}, theme=None, interval_ms=60_000, registry={},
            lazy_x_range=fig.x_range, enable_point_lod=True,
        )
        # Below threshold: drawn directly, full data, no LOD callback needed.
        assert len(registry[("Cloud", "Points")].data["x"]) == n
        assert len(fig.x_range.js_property_callbacks) == 0

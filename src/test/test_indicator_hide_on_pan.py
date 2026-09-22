"""
Per-indicator hide_on_pan flag + dense bar-panel decimation instead of blanking.

- IndicatorPlotConfig.hide_on_pan (ta/base.py) resolves per renderer type via
  resolve_hide_on_pan() when left at its default (None): scatter/label/rect/
  arrow hide during pan by default, line/step/bar/span stay visible.
- render_indicator() registers scatter/label-style renderers into
  hide_on_pan_sink (feeding the Task 6 global derender registry), but a
  renderer="bar" dimension NEVER enters that registry - dense bar panels
  (MACD histogram, etc.) rely on decimate_bar.js's level-of-detail bucketing
  instead of blanking.
"""
import numpy as np
import pytest

pytest.importorskip("bokeh")

from bokeh.plotting import figure

from tradetropy.ta.base import resolve_hide_on_pan, IndicatorPlotConfig
from tradetropy.plotting.config import IndicatorPlotMeta
from tradetropy.plotting.render._indicators import render_indicator, _BAR_LOD_THRESHOLD


def _meta(renderer, hide_on_pan=None, n=10):
    ts = 1_700_000_000_000 + np.arange(n) * 60_000.0
    values = np.linspace(-1, 1, n)
    return IndicatorPlotMeta(
        name="Ind",
        values=values,
        timestamps=ts,
        overlay=False,
        renderer=renderer,
        color="#ff0000",
        hide_on_pan=hide_on_pan,
    )


class TestResolveHideOnPanDefaults:
    def test_scatter_defaults_to_hidden_on_pan(self):
        assert resolve_hide_on_pan("scatter", None) is True

    def test_label_defaults_to_hidden_on_pan(self):
        assert resolve_hide_on_pan("label", None) is True

    def test_line_defaults_to_visible_during_pan(self):
        assert resolve_hide_on_pan("line", None) is False

    def test_bar_defaults_to_visible_during_pan(self):
        assert resolve_hide_on_pan("bar", None) is False

    def test_explicit_override_wins_over_renderer_default(self):
        assert resolve_hide_on_pan("scatter", False) is False
        assert resolve_hide_on_pan("line", True) is True


class TestIndicatorPlotConfigPropagation:
    def test_default_is_none(self):
        assert IndicatorPlotConfig().hide_on_pan is None

    def test_merged_propagates_explicit_value(self):
        pc = IndicatorPlotConfig().merged(hide_on_pan=False)
        assert pc.hide_on_pan is False


class TestRenderIndicatorHideOnPanSink:
    def test_scatter_indicator_registers_in_sink(self):
        fig = figure(width=300, height=150)
        meta = _meta("scatter")
        sink = []
        render_indicator(fig, [self._src(meta)], meta, hide_on_pan_sink=sink)
        assert len(sink) == 1

    def test_line_indicator_does_not_register(self):
        fig = figure(width=300, height=150)
        meta = _meta("line")
        sink = []
        render_indicator(fig, [self._src(meta)], meta, hide_on_pan_sink=sink)
        assert sink == []

    def test_scatter_with_explicit_hide_on_pan_false_is_excluded(self):
        fig = figure(width=300, height=150)
        meta = _meta("scatter", hide_on_pan=False)
        sink = []
        render_indicator(fig, [self._src(meta)], meta, hide_on_pan_sink=sink)
        assert sink == []

    def test_bar_indicator_never_registers_even_with_explicit_override(self):
        fig = figure(width=300, height=150)
        meta = _meta("bar", hide_on_pan=True)
        sink = []
        render_indicator(fig, [self._src(meta)], meta, hide_on_pan_sink=sink)
        assert sink == []

    @staticmethod
    def _src(meta):
        from tradetropy.plotting.sources import build_indicator_source
        return build_indicator_source(meta)[0]


class TestDenseBarPanelDecimation:
    def test_dense_bar_source_gets_lod_wired(self):
        fig = figure(width=300, height=150)
        n = _BAR_LOD_THRESHOLD + 500
        meta = _meta("bar", n=n)
        from tradetropy.plotting.sources import build_indicator_source
        src = build_indicator_source(meta)[0]

        from bokeh.models import Range1d
        x_range = Range1d(start=0, end=1)

        render_indicator(fig, [src], meta, x_range=x_range)

        # The vbar glyph's source must be a DIFFERENT ColumnDataSource (the
        # decimated view), not the original full source, and the full source
        # must have a js_on_change callback wired for pan/zoom refinement.
        vbar_renderer = fig.renderers[-1]
        assert vbar_renderer.data_source is not src
        assert len(vbar_renderer.data_source.data["ts"]) <= n

    def test_sparse_bar_source_is_not_decimated(self):
        fig = figure(width=300, height=150)
        meta = _meta("bar", n=50)
        from tradetropy.plotting.sources import build_indicator_source
        src = build_indicator_source(meta)[0]

        from bokeh.models import Range1d
        x_range = Range1d(start=0, end=1)

        render_indicator(fig, [src], meta, x_range=x_range)

        vbar_renderer = fig.renderers[-1]
        assert vbar_renderer.data_source is src

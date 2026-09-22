"""
Tests for the P&L panel's backtesting.py-style trade markers.

render_pl_bars() now draws each closed trade as an entry->exit segment plus a
circle marker at the exit (size scaled by |pnl|, colored win/loss), replacing
the previous quad-bar rendering. build_trades_source() also derives a signed
`size_signed` column (positive for long, negative for short) consumed by the
panel's hover tooltip.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Stats. insufficient sample:UserWarning"
)


def _make_strategy():
    from tradetropy.models.strategy import Strategy
    from tradetropy.ta import SMA

    class SmaCross(Strategy):
        def init(self):
            self.px = self.subscribe_ohlc("GOOG", "1d", window_size=200)
            self.fast = self.add_indicator(self.px.close, SMA(10))
            self.slow = self.add_indicator(self.px.close, SMA(30))

        def on_data(self):
            if self.fast[-1] > self.slow[-1]:
                if not self.sesh.positions("GOOG"):
                    self.sesh.buy("GOOG", volume=1)
            else:
                for pos in self.sesh.positions("GOOG"):
                    self.sesh.position_close(pos.ticket)

    return SmaCross


def _run_backtest():
    from tradetropy import BacktestEngine
    from tradetropy.datasets import load_goog_1d

    SmaCross = _make_strategy()
    bt = BacktestEngine.by_klines(SmaCross(), data=(load_goog_1d(),))
    bt.run()
    return bt


@pytest.mark.unit
class TestSignedSize:
    def test_size_signed_matches_direction(self):
        from tradetropy.plotting.sources import build_trades_source

        bt = _run_backtest()
        src = build_trades_source(bt.stats.trades, interval_ms=0)
        assert src is not None
        size = np.asarray(src.data["size"], dtype=np.float64)
        size_signed = np.asarray(src.data["size_signed"], dtype=np.float64)
        direction = src.data["direction"]

        # size_signed always has the same magnitude as size (only sign flips).
        assert np.allclose(np.abs(size_signed), np.abs(size))

        for d, s in zip(direction, size_signed):
            if str(d).lower() == "short":
                assert s <= 0
            else:
                assert s >= 0


@pytest.mark.unit
class TestPlPanelMarkers:
    def _built_source(self):
        from tradetropy.plotting.sources import build_trades_source

        bt = _run_backtest()
        src = build_trades_source(bt.stats.trades, interval_ms=0)
        assert src is not None
        return src

    def test_render_draws_segment_and_scatter(self):
        from tradetropy.plotting.render import render_pl_bars
        from tradetropy.plotting._fig_factory import _new_fig
        from tradetropy.plotting._util import _theme

        src = self._built_source()
        theme = _theme.__wrapped__() if hasattr(_theme, "__wrapped__") else None
        if theme is None:
            from tradetropy.plotting.theme.registry import get_theme
            theme = get_theme("light")

        fig = _new_fig(100, 800, theme, name="pl")
        render_pl_bars(fig, src, theme)

        renderer_types = [type(r.glyph).__name__ for r in fig.renderers]
        assert "Segment" in renderer_types
        assert "Scatter" in renderer_types

    def test_marker_size_scales_with_abs_pnl(self):
        from tradetropy.plotting.render import render_pl_bars
        from tradetropy.plotting._fig_factory import _new_fig
        from tradetropy.plotting.theme.registry import get_theme

        src = self._built_source()
        theme = get_theme("light")
        fig = _new_fig(100, 800, theme, name="pl")
        render_pl_bars(fig, src, theme)

        sizes = np.asarray(src.data["_pl_marker_size"], dtype=np.float64)
        pnl = np.asarray(src.data["pnl"], dtype=np.float64)
        assert len(sizes) == len(pnl)
        # Larger |pnl| trades must not have a smaller marker than a smaller
        # |pnl| trade (monotonic scaling, allowing ties).
        order = np.argsort(np.abs(pnl))
        assert np.all(np.diff(sizes[order]) >= -1e-9)

    def test_hover_tooltip_includes_signed_size(self):
        from tradetropy.plotting.render import render_pl_bars
        from tradetropy.plotting._fig_factory import _new_fig
        from tradetropy.plotting.theme.registry import get_theme
        from bokeh.models import HoverTool

        src = self._built_source()
        theme = get_theme("light")
        fig = _new_fig(100, 800, theme, name="pl")
        render_pl_bars(fig, src, theme)

        hover_tools = [t for t in fig.tools if isinstance(t, HoverTool)]
        assert hover_tools
        tooltip_fields = [field for _, field in hover_tools[0].tooltips]
        assert any("size_signed" in f for f in tooltip_fields)
        assert any("pnl" in f for f in tooltip_fields)
        assert any("entry_ts" in f for f in tooltip_fields)
        assert any("exit_ts" in f for f in tooltip_fields)

    def test_no_crash_on_empty_trades(self):
        from tradetropy.plotting.render import render_pl_bars
        from tradetropy.plotting._fig_factory import _new_fig
        from tradetropy.plotting.theme.registry import get_theme
        from bokeh.models import ColumnDataSource

        theme = get_theme("light")
        fig = _new_fig(100, 800, theme, name="pl")
        empty_src = ColumnDataSource(dict(
            entry_ts=[], exit_ts=[], pnl=[], pnl_pct=[], is_win=[],
            direction=[], size_signed=[], entry_price=[], exit_price=[],
            trade_label=[],
        ))
        render_pl_bars(fig, empty_src, theme)  # must not raise
        assert fig.renderers == []


@pytest.mark.unit
class TestFullPlotWithPlPanel:
    def test_plot_pl_end_to_end(self, tmp_path):
        bt = _run_backtest()
        out = tmp_path / "chart.html"
        bt.plot(output="file", filename=str(out), plot_pl=True)
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "size_signed" in html
        assert "_pl_marker_size" in html

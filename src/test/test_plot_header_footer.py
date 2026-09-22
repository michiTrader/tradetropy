"""
Manual header/footer notes (PlotConfig.header_text / footer_text).

Verifies the Divs built by ``_build_note_div`` render the given HTML, and
that a real ``Backtest.plot()`` run with both texts sets produces an HTML
file containing them - while the default (None) leaves the layout unchanged
(no extra Div injected).
"""
import numpy as np
import pytest

pytest.importorskip("bokeh")

from tradetropy.plotting._layout import _build_note_div

_LIGHT_THEME = {"bg": "#FFFFFF", "text": "#000000", "grid_line": "#DDDDDD"}


class TestBuildNoteDiv:
    def test_header_renders_text(self):
        div = _build_note_div("<b>My Strategy</b>", _LIGHT_THEME, position="header")
        assert "<b>My Strategy</b>" in div.text
        assert "border-bottom" in div.text

    def test_footer_renders_text(self):
        div = _build_note_div("Note: fees included", _LIGHT_THEME, position="footer")
        assert "Note: fees included" in div.text
        assert "border-top" in div.text


@pytest.mark.unit
class TestPlotHeaderFooterIntegration:
    def _run_backtest(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core.data_types import KlineData

        rng = np.random.default_rng(7)
        n = 200
        m = np.zeros((n, 7), dtype=np.float64)
        m[:, 0] = 1_700_000_000_000 + np.arange(n) * 60_000.0
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        open_ = np.r_[close[0], close[:-1]]
        m[:, 1] = open_
        m[:, 2] = np.maximum(open_, close) + 0.2
        m[:, 3] = np.minimum(open_, close) - 0.2
        m[:, 4] = close
        m[:, 5] = rng.uniform(1, 10, n)
        kl = KlineData(symbol="SYM", data=m, timeframe="1m", tick_size=0.01)

        class S(Strategy):
            def init(self):
                self.o = self.subscribe_ohlc("SYM", "1m", window_size=n)

            def on_data(self):
                pass

        bt = BacktestEngine.by_klines(S(), data=(kl,))
        bt.run()
        return bt

    def test_header_and_footer_text_appear_in_output(self, tmp_path):
        bt = self._run_backtest()
        out = tmp_path / "notes.html"
        bt.plot(
            output="file", filename=str(out), plot_stats=False,
            header_text="<b>Strategy X - walk-forward Q3</b>",
            footer_text="Note: commissions included",
        )
        html = out.read_text(encoding="utf-8")
        assert "Strategy X - walk-forward Q3" in html
        assert "Note: commissions included" in html

    def test_no_notes_by_default(self, tmp_path):
        bt = self._run_backtest()
        out = tmp_path / "no_notes.html"
        bt.plot(output="file", filename=str(out), plot_stats=False)
        html = out.read_text(encoding="utf-8")
        assert "plot-note" not in html

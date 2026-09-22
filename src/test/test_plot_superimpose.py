"""
PlotConfig.superimpose - translucent higher-timeframe candle bodies behind
the main OHLC panel.

Covers: explicit-string resolution producing the expected bar count, warn +
no-op when the target TF is invalid (<= own interval), and that the
superimposed layer only draws quads (no segment/wick).
"""
import numpy as np
import pytest

pytest.importorskip("bokeh")

from tradetropy.plotting._util import _resolve_superimpose_interval


class TestResolveSuperimposeInterval:
    def test_auto_true_minute_chart_targets_1h(self):
        assert _resolve_superimpose_interval(60_000, 5000, True) == 3_600_000

    def test_auto_true_hourly_chart_targets_1d(self):
        assert _resolve_superimpose_interval(3_600_000, 500, True) == 86_400_000

    def test_explicit_string_parsed_interval_used_as_is(self):
        # PlotConfig.__post_init__ already parsed "4h" -> ms before this call.
        assert _resolve_superimpose_interval(900_000, 2000, 14_400_000) == 14_400_000

    def test_target_not_greater_than_own_interval_warns_and_disables(self):
        with pytest.warns(UserWarning, match="strictly greater"):
            result = _resolve_superimpose_interval(3_600_000, 500, 1_800_000)
        assert result is None

    def test_too_few_resulting_bars_warns_and_disables(self):
        # 10 bars of 1m span only 10 minutes; a 1D target would yield 0 bars.
        with pytest.warns(UserWarning, match="fewer than"):
            result = _resolve_superimpose_interval(60_000, 10, 86_400_000)
        assert result is None

    def test_false_disables_without_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_superimpose_interval(60_000, 5000, False) is None


@pytest.mark.unit
class TestSuperimposeIntegration:
    def _klines(self, n=600, interval_ms=60_000, seed=3):
        from tradetropy.core.data_types import KlineData
        rng = np.random.default_rng(seed)
        m = np.zeros((n, 7), dtype=np.float64)
        m[:, 0] = 1_700_000_000_000 + np.arange(n) * interval_ms
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        open_ = np.r_[close[0], close[:-1]]
        m[:, 1] = open_
        m[:, 2] = np.maximum(open_, close) + 0.2
        m[:, 3] = np.minimum(open_, close) - 0.2
        m[:, 4] = close
        m[:, 5] = rng.uniform(1, 10, n)
        return KlineData(symbol="SYM", data=m, timeframe="1m", tick_size=0.01)

    def _run(self, n=600):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine

        kl = self._klines(n=n)

        class S(Strategy):
            def init(self):
                self.o = self.subscribe_ohlc("SYM", "1m", window_size=n)

            def on_data(self):
                pass

        bt = BacktestEngine.by_klines(S(), data=(kl,))
        bt.run()
        return bt

    def test_superimpose_explicit_tf_renders_expected_quad_count(self, tmp_path):
        bt = self._run(n=600)  # 600 minutes = 10h -> "1h" yields ~10 bars
        out = tmp_path / "superimpose.html"
        bt.plot(output="file", filename=str(out), plot_stats=False, superimpose="1h")
        html = out.read_text(encoding="utf-8")
        assert "Superimpose 1h" in html

    def test_superimpose_disabled_by_default(self, tmp_path):
        bt = self._run(n=600)
        out = tmp_path / "no_superimpose.html"
        bt.plot(output="file", filename=str(out), plot_stats=False)
        html = out.read_text(encoding="utf-8")
        assert "Superimpose" not in html

    def test_superimpose_invalid_target_warns_but_still_plots(self, tmp_path):
        bt = self._run(n=600)
        out = tmp_path / "superimpose_invalid.html"
        with pytest.warns(UserWarning, match="strictly greater"):
            bt.plot(output="file", filename=str(out), plot_stats=False, superimpose="30s")
        assert out.exists() and out.stat().st_size > 0
        html = out.read_text(encoding="utf-8")
        assert "Superimpose" not in html

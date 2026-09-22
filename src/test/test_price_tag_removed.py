"""
Price tag removal + show_price_tag deprecation.

The floating price box that used to follow the crosshair (TradingView-style)
has been removed entirely. ``show_price_tag`` is still accepted by
``PlotConfig``/``Backtest.plot()`` for backwards compatibility but is a no-op
that emits a ``DeprecationWarning``.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("bokeh")

from tradetropy.plotting.config import PlotConfig
from tradetropy.plotting._layout import _configure_crosshair


class TestShowPriceTagDeprecation:
    def test_plotconfig_warns_on_show_price_tag_true(self):
        with pytest.warns(DeprecationWarning, match="show_price_tag"):
            PlotConfig(show_price_tag=True)

    def test_plotconfig_default_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            PlotConfig()  # must not raise

    def test_configure_crosshair_no_longer_accepts_show_price_tag_kwarg(self):
        # The floating tag's render path (Label/CustomAction/mousemove) was
        # removed along with the show_price_tag parameter of the renderer
        # itself (PlotConfig keeps the deprecated kwarg; the low-level
        # crosshair configurator does not).
        import inspect
        sig = inspect.signature(_configure_crosshair)
        assert "show_price_tag" not in sig.parameters


@pytest.mark.unit
def test_backtest_plot_with_deprecated_kwarg_still_renders(tmp_path):
    from tradetropy.models.strategy import Strategy
    from tradetropy.backtest.engine import BacktestEngine
    from tradetropy.core.data_types import KlineData

    rng = np.random.default_rng(11)
    n = 150
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
    out = tmp_path / "deprecated_price_tag.html"

    with pytest.warns(DeprecationWarning, match="show_price_tag"):
        bt.plot(output="file", filename=str(out), plot_stats=False, show_price_tag=True)

    assert out.exists() and out.stat().st_size > 0
    html = out.read_text(encoding="utf-8")
    assert "Toggle price tag" not in html

"""
Backtest equity curve with no closed trades.

A strategy whose position never closes still has a full equity curve: the
broker records balance + floating PnL on every bar. These tests guard the
backtest<->replay parity fix - previously the backtest discarded the whole
Stats object when no trade had closed (``if not trades: return None``), so the
equity of an open position was invisible in backtest while replay/live showed
it. The fix gates on the equity curve, not on closed trades, and
``compute_stats`` tolerates an empty trade list.
"""

import numpy as np
import pytest

from tradetropy.core.data_types import KlineData
from tradetropy.backtest import BacktestEngine
from tradetropy.models.strategy import Strategy


def _kline_ramp(n=200, start=100.0, step=0.05):
    """Build a [N x 7] kline matrix: ts, open, high, low, close, vol, turnover."""
    arr = np.zeros((n, 7), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64) * 60_000.0
    close = start + np.arange(n, dtype=np.float64) * step
    open_ = close - step
    arr[:, 1] = open_
    arr[:, 2] = np.maximum(open_, close) + 0.02
    arr[:, 3] = np.minimum(open_, close) - 0.02
    arr[:, 4] = close
    arr[:, 5] = 10.0
    arr[:, 6] = close * 10.0
    return arr


class BuyAndHoldNoClose(Strategy):
    """Open a single long position early and never close it."""

    def init(self):
        self.k = self.subscribe_ohlc('SYM', 60_000, window_size=50)
        self._n = 0

    def on_data(self):
        self._n += 1
        if self._n == 10 and not self.sesh.positions('SYM'):
            self.sesh.buy('SYM', volume=1)


@pytest.mark.integration
class TestEquityWithoutClosedTrades:
    def _run(self):
        kd = KlineData('SYM', _kline_ramp(n=200), timeframe=60_000)
        return BacktestEngine.by_klines(BuyAndHoldNoClose(), data=(kd,)).run()

    def test_stats_built_without_closed_trades(self):
        bt = self._run()
        # Stats must exist even with zero closed trades (parity with replay).
        assert bt.stats is not None
        assert int(bt.stats['# Trades']) == 0
        assert bt.stats.trades.empty

    def test_equity_curve_present_and_reflects_floating_pnl(self):
        bt = self._run()
        eq = bt.stats.equity_curve
        assert not eq.empty
        # The open long on a rising ramp produces floating PnL, so the curve
        # must move away from the initial balance (it is not flat).
        assert float(eq.iloc[-1]) != pytest.approx(float(eq.iloc[0]))
        assert float(eq.iloc[-1]) > float(eq.iloc[0])

    def test_equity_panel_renders_with_na_trade_metrics(self):
        # Task 4: the equity panel derives peak/final/maxDD from the equity
        # curve, not from trades, so it must build even when trade metrics are
        # N/A. Exercise the real panel builder (avoids show(), which opens a
        # browser).
        pytest.importorskip('bokeh')
        from tradetropy.plotting.plotting import _build_equity_panel
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme

        bt = self._run()
        config = PlotConfig()
        theme = _theme(config)
        all_figs = []
        fig_eq, eq_source, baseline, _ = _build_equity_panel(
            config, theme, bt.stats, all_figs,
        )
        assert fig_eq is not None
        assert len(eq_source.data['equity']) == len(bt.stats.equity_curve)
        assert len(all_figs) == 1


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:Stats. insufficient sample:UserWarning")
class TestEquityPanelDuplicateTimestamps:
    """A tick-driven backtest records equity per tick, so its curve can carry
    duplicate timestamps (several ticks in one millisecond). The equity panel
    must resolve peak / max-drawdown positionally - label-based .loc[ts] on a
    duplicated timestamp returns a Series and used to crash float()."""

    def _dup_ts_stats(self):
        import pandas as pd
        from tradetropy.stats import Stats

        # Two ticks per millisecond; the worst drawdown lands on a duplicated ts.
        ts = pd.to_datetime([0, 0, 1, 1, 2, 2], unit='ms', utc=True)
        vals = [1000.0, 1000.0, 990.0, 985.0, 995.0, 998.0]
        eq = pd.Series(vals, index=ts, name='equity')
        return Stats(eq, [], 1000.0)

    def test_panel_builds_on_duplicate_timestamps(self):
        pytest.importorskip('bokeh')
        from tradetropy.plotting.plotting import _build_equity_panel
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme

        stats = self._dup_ts_stats()
        config = PlotConfig()
        theme = _theme(config)
        fig_eq, eq_source, baseline, _ = _build_equity_panel(config, theme, stats, [])
        assert fig_eq is not None
        assert len(eq_source.data['equity']) == 6

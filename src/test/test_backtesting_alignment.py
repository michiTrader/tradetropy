"""
Tests for the backtesting.py alignment work:

- KlineData / TickData ``tick_value`` defaults to ``tick_size`` so PnL per unit
  equals the pure price difference (backtesting.py's unit model).
- Out-of-money guard (equity <= 0) closes positions and stops the backtest
  cleanly, on by default.
- ``finalize_trades`` closes open positions at the end so they enter stats.

These document and lock the parity choices made to compare tradetropy directly
against backtesting.py.
"""

import numpy as np
import pytest

from tradetropy import Strategy
from tradetropy.core.data_types import KlineData, TickData


# ══════════════════════════════════════════════════════════════════════════════
# tick_value defaults to tick_size (contract_size=1 -> PnL == price difference)
# ══════════════════════════════════════════════════════════════════════════════


class TestTickValueDefault:
    def test_kline_tick_value_defaults_to_tick_size(self):
        kd = KlineData("X", np.zeros((3, 7)), timeframe="1m", tick_size=0.1)
        assert kd.tick_value == 0.1
        assert kd.contract_size == 1.0
        assert kd.config.tick_value == 0.1

    def test_tick_tick_value_defaults_to_tick_size(self):
        td = TickData("X", np.zeros((3, 7)), tick_size=0.25)
        assert td.tick_value == 0.25
        assert td.config.tick_value == 0.25

    def test_explicit_tick_value_is_preserved(self):
        # Futures-like instrument: a tick is worth more than its size.
        kd = KlineData(
            "MES", np.zeros((3, 7)), timeframe="1m",
            tick_size=0.25, tick_value=1.25, contract_size=5.0,
        )
        assert kd.tick_value == 1.25
        assert kd.contract_size == 5.0

    def test_default_tick_value_makes_pnl_equal_price_diff(self):
        # With tick_value == tick_size and contract_size == 1, closing a 1-unit
        # long that moved by +10.0 in price yields exactly +10.0 profit.
        cfg = KlineData(
            "X", np.zeros((3, 7)), timeframe="1m", tick_size=0.1
        ).config
        profit = cfg.calculate_profit(
            volume=1.0, price_open=100.0, price_close=110.0, is_buy=True
        )
        assert profit == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════════════════════
# Single aggregate progress bar in optimize (PoolEvaluator)
# ══════════════════════════════════════════════════════════════════════════════


class TestOptimizeProgressBar:
    def test_wrap_progress_disabled_returns_raw_iterable(self):
        from tradetropy.backtest.pool_adapter import PoolEvaluator

        ev = PoolEvaluator(lambda c, d: None, data=None, progress=False)
        it = iter(range(5))
        assert ev._wrap_progress(it, 5) is it

    def test_wrap_progress_builds_single_bar_with_total(self):
        pytest.importorskip("tqdm")
        from tradetropy.backtest.pool_adapter import PoolEvaluator

        ev = PoolEvaluator(lambda c, d: None, data=None, progress=True,
                           desc="Optimize")
        bar = ev._wrap_progress(iter(range(7)), 7)
        # tqdm exposes the aggregate total; one bar for all candidates.
        assert getattr(bar, "total", None) == 7
        assert list(bar) == list(range(7))

    def test_optimize_runs_with_progress_and_matches_no_progress(self):
        # End-to-end: the aggregate bar must not change the ranked results.
        import numpy as np
        from tradetropy import BacktestEngine, Strategy
        from tradetropy.core.data_types import KlineData
        from tradetropy.ta import SMA
        from tradetropy.datasets import load_btcusd_1m

        base = load_btcusd_1m().data
        mat = np.tile(base, (3, 1))[:1200].copy()
        mat[:, 0] = base[0, 0] + np.arange(1200) * 60_000
        k = KlineData("B", mat, timeframe="1m", tick_size=0.1)

        res_on = BacktestEngine.by_klines(_ProgStrat(), data=(k,)).optimize(
            maximize="Return [%]", method="grid",
            fast=[5, 10], slow=[30, 40],
            constraints=lambda p: p["fast"] < p["slow"], progress=True,
        )
        res_off = BacktestEngine.by_klines(_ProgStrat(), data=(k,)).optimize(
            maximize="Return [%]", method="grid",
            fast=[5, 10], slow=[30, 40],
            constraints=lambda p: p["fast"] < p["slow"], progress=False,
        )
        assert res_on.best_params == res_off.best_params
        assert res_on.best_fitness == pytest.approx(res_off.best_fitness)


class _ProgStrat(Strategy):
    """Module-level SMA-cross strategy (picklable for the process pool)."""

    fast = 10
    slow = 30

    def init(self):
        from tradetropy.ta import SMA
        self.px = self.subscribe_ohlc("B", "1m", window_size=200)
        self.f = self.add_indicator(self.px.close, SMA(self.fast))
        self.s = self.add_indicator(self.px.close, SMA(self.slow))

    def on_data(self):
        if self.f[-1] > self.s[-1]:
            if not self.sesh.positions("B"):
                self.sesh.buy("B", volume=1)
        else:
            for p in self.sesh.positions("B"):
                self.sesh.position_close(p.ticket)


# ══════════════════════════════════════════════════════════════════════════════
# Out-of-money guard (equity <= 0) + graceful stop, on by default
# ══════════════════════════════════════════════════════════════════════════════


def _crash_klines(n_pre=5, crash_to=0.0):
    """
    OHLC series that holds flat at 1000 for a few bars then crashes, so a long
    position taken at the start is wiped (equity <= 0).
    """
    prices = [1000.0] * n_pre + [900.0, 700.0, 400.0, crash_to, crash_to]
    rows = []
    ts0 = 1_600_000_000_000
    for i, p in enumerate(prices):
        rows.append([ts0 + i * 60_000, p, p, p, p, 1.0, 0.0])
    return KlineData("C", np.array(rows, dtype=np.float64), timeframe="1m",
                     tick_size=1.0)


class _BuyAndHold(Strategy):
    """Open one big long on the first on_data bar and hold it forever."""

    def init(self):
        self.px = self.subscribe_ohlc("C", "1m", window_size=50)

    def on_data(self):
        if not self.sesh.positions("C"):
            self.sesh.buy("C", volume=20)


class TestOutOfMoneyGuard:
    def _run(self, **sesh_kwargs):
        from tradetropy import BacktestEngine
        from tradetropy.session.base import SeshSimulatorBase

        sesh = SeshSimulatorBase(initial_balance=10_000.0, **sesh_kwargs)
        eng = BacktestEngine.by_klines(_BuyAndHold(), data=(_crash_klines(),),
                                       sesh=sesh)
        eng.run()
        return eng

    def test_default_stops_out_of_money_cleanly(self):
        # volume 20 * a >100 price drop from 1000 wipes the 10k account.
        eng = self._run()  # stop_out_of_money defaults True
        assert eng.out_of_money is True
        assert eng.stopped_early is True
        # Equity was frozen at zero on the wipe bar (no exception raised).
        assert eng.broker._eq_vals[-1] == 0.0
        # Stopped before consuming every bar.
        assert len(eng.broker._eq_vals) < 10

    def test_disabled_does_not_stop(self):
        eng = self._run(stop_out_of_money=False)
        assert eng.out_of_money is False
        assert eng.stopped_early is False
        # Ran to the end: equity recorded for (nearly) all bars, can go < 0.
        assert eng.broker._eq_vals[-1] < 0


# ══════════════════════════════════════════════════════════════════════════════
# finalize_trades (close open positions at end so they enter stats)
# ══════════════════════════════════════════════════════════════════════════════


def _flat_klines(n=40, price=1000.0):
    rows = []
    ts0 = 1_600_000_000_000
    for i in range(n):
        # tiny drift so trades have non-zero pnl but never wipe the account.
        p = price + (i % 5)
        rows.append([ts0 + i * 60_000, p, p + 1, p - 1, p, 1.0, 0.0])
    return KlineData("H", np.array(rows, dtype=np.float64), timeframe="1m",
                     tick_size=1.0)


class _OpenNeverClose(Strategy):
    def init(self):
        self.px = self.subscribe_ohlc("H", "1m", window_size=50)

    def on_data(self):
        if not self.sesh.positions("H"):
            self.sesh.buy("H", volume=1)


class TestFinalizeTrades:
    def _run(self, finalize):
        from tradetropy import BacktestEngine
        from tradetropy.session.base import SeshSimulatorBase

        sesh = SeshSimulatorBase(initial_balance=100_000.0,
                                 finalize_trades=finalize)
        eng = BacktestEngine.by_klines(_OpenNeverClose(), data=(_flat_klines(),),
                                       sesh=sesh)
        eng.run()
        return eng

    def test_finalize_true_closes_open_position_into_trades(self):
        eng = self._run(finalize=True)
        assert len(eng.broker.get_trades()) == 1

    def test_finalize_false_leaves_open_and_warns(self):
        with pytest.warns(UserWarning, match="position.*still.*open"):
            eng = self._run(finalize=False)
        # The still-open position is not a closed Trade.
        assert len(eng.broker.get_trades()) == 0
        assert len(eng.broker.positions) == 1

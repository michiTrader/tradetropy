# Tests for the consolidated optimization system (module tradetropy.optimize).
import pickle

import numpy as np
import pytest

from tradetropy.core.data_types import TickData
from tradetropy.backtest import BacktestEngine
from tradetropy.models import Strategy
from tradetropy.optimize import FitnessMetric, OptimizationResult, create_evaluation_function  # noqa: F401 (public alias)
from tradetropy.backtest.engine import _run_backtest_candidate


def _ramp_ticks(n=400, start=100.0, step=0.1):
    arr = np.zeros((n, 7), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64) * 1000.0
    price = start + np.arange(n, dtype=np.float64) * step
    arr[:, 1] = price
    arr[:, 2] = price
    arr[:, 3] = 1.0
    arr[:, 6] = price
    return arr


class _HoldStrat(Strategy):
    hold = 10

    def init(self):
        self._tp = self.subscribe_ticks('SYM', window_size=10)
        self._n = 0

    def on_data(self):
        self._n += 1
        if self._n == 50:
            self.sesh.buy('SYM', volume=1)
        elif self._n == 50 + self.hold:
            for p in self.sesh.positions('SYM'):
                self.sesh.position_close(p.ticket)


def test_evaluation_function_picklable():
    fm = FitnessMetric(metric='Return [%]')
    fn = create_evaluation_function(_run_backtest_candidate, fm)
    fn2 = pickle.loads(pickle.dumps(fn))
    assert fn2.fitness_metric.metric == 'Return [%]'


@pytest.mark.integration
def test_optimize_grid_maximize():
    ticks = _ramp_ticks()
    bt = BacktestEngine.by_ticks(_HoldStrat(), data=(TickData('SYM', ticks, tick_size=0.1),))
    result = bt.optimize(maximize='Return [%]', method='grid', workers=1, hold=[5, 20, 60])
    assert isinstance(result, OptimizationResult)
    df = result.to_dataframe()
    assert len(df) == 3
    assert result.best_params['hold'] == 60
    assert np.isfinite(result.best_fitness)


@pytest.mark.integration
def test_optimize_random_method():
    ticks = _ramp_ticks()
    bt = BacktestEngine.by_ticks(_HoldStrat(), data=(TickData('SYM', ticks, tick_size=0.1),))
    result = bt.optimize(maximize='Return [%]', method='random', iterations=4, workers=1, hold=[5, 20, 60])
    assert len(result.to_dataframe()) == 4


@pytest.mark.integration
def test_optimize_multiworker_shm_matches_single():
    """
    The shared-memory input transport (multi-worker) must yield the SAME
    ranking as a single worker: the matrices are read-only and identical, so
    only HOW they reach the workers changes, never the metrics.
    """
    ticks = _ramp_ticks()
    data = (TickData('SYM', ticks, tick_size=0.1),)

    r1 = BacktestEngine.by_ticks(_HoldStrat(), data=data).optimize(
        maximize='Return [%]', method='grid', workers=1, hold=[5, 20, 60])
    r2 = BacktestEngine.by_ticks(_HoldStrat(), data=data).optimize(
        maximize='Return [%]', method='grid', workers=2, hold=[5, 20, 60])

    assert r1.best_params == r2.best_params
    assert r1.best_fitness == pytest.approx(r2.best_fitness)

    # Same fitness for every candidate regardless of worker count.
    f1 = {tuple(sorted(row.metrics['params'].items())): row.fitness for row in r1.top(99)}
    f2 = {tuple(sorted(row.metrics['params'].items())): row.fitness for row in r2.top(99)}
    assert f1.keys() == f2.keys()
    for k in f1:
        assert f1[k] == pytest.approx(f2[k]), f"candidate {k}: {f1[k]} != {f2[k]}"

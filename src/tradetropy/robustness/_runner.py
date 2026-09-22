"""
Re-run path for data and parameter level Monte Carlo.

Data and parameter perturbations cannot be replayed on a finished trade list:
they change what the strategy sees, so every simulation needs a fresh backtest.
This module reuses the same parallel infrastructure as the optimizer
(``PoolEvaluator`` + a picklable, module-level evaluation callable) to run the
simulations across processes.

The evaluator applies the configured methods to a per-simulation copy of the
inputs/parameters, runs a fresh ``BacktestEngine`` and returns only the scalar
metrics needed by ``MonteCarloResult`` (keeping inter-process payloads small).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from tradetropy.backtest.engine import BacktestEngine
from tradetropy.backtest.pool_adapter import PoolEvaluator
from tradetropy.optimize.task import Candidate, Result
from tradetropy.robustness.result import MonteCarloResult

_FINAL_EQUITY_KEY = 'Equity Final [$]'
_MAX_DD_KEY = 'Max. Drawdown [%]'


def _as_float(value) -> float:
    """Coerce a value to float, mapping failures to NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _seed_for(base_seed, i: int):
    """Return a per-simulation seed (None stays None for full randomness)."""
    return None if base_seed is None else base_seed + i


class _RerunEvaluator:
    """
    Picklable evaluation callable for one re-run simulation.

    Built at module import time semantics (no closures), so it survives the
    'spawn' start method used on Windows/macOS.
    """

    def __init__(self, metrics: List[str]):
        self.metrics = list(metrics)

    def __call__(self, candidate: Candidate, data: Dict) -> Result:
        """
        Run one perturbed backtest and return its scalar metrics.

        Args:
            candidate (Candidate): Carries the per-simulation seed in params.
            data (dict): Bundle with strategy_cls, sesh, inputs and methods.

        Returns:
            Result: metrics dict on success, or error string on failure.
        """
        try:
            rng = np.random.default_rng(candidate.params['seed'])
            tick_inputs = data['tick_inputs']
            kline_inputs = data['kline_inputs']
            params: Dict = dict(data['base_params'])

            for method in data['methods']:
                if method.level == 'data':
                    if tick_inputs:
                        tick_inputs = method.apply(tick_inputs, rng)
                    if kline_inputs:
                        kline_inputs = method.apply(kline_inputs, rng)
                elif method.level == 'params':
                    params = method.apply(params, rng)

            strategy = data['strategy_cls']()
            if params:
                strategy.update_params(params)

            if tick_inputs:
                eng = BacktestEngine.by_ticks(
                    strategy, data=tick_inputs,
                    sesh=data['sesh'].clone(), align_by_ts=data['align_by_ts'],
                )
            elif kline_inputs:
                eng = BacktestEngine.by_klines(
                    strategy, data=kline_inputs, sesh=data['sesh'].clone(),
                )
            else:
                return Result(candidate.id, float('nan'), {}, error='no inputs')

            eng.run()
            stats = eng.stats
            if stats is None:
                return Result(candidate.id, float('nan'), {})

            out = {k: _as_float(stats.get(k, np.nan)) for k in self.metrics}
            out[_FINAL_EQUITY_KEY] = _as_float(stats.get(_FINAL_EQUITY_KEY, np.nan))
            out[_MAX_DD_KEY] = _as_float(stats.get(_MAX_DD_KEY, np.nan))
            return Result(candidate.id, 0.0, out)
        except Exception as exc:  # noqa: BLE001 - report failure as NaN sample
            return Result(candidate.id, float('nan'), {}, error=str(exc))


def run_rerun_simulations(engine, config) -> MonteCarloResult:
    """
    Run data/parameter-level simulations in parallel and aggregate them.

    Args:
        engine (BacktestEngine): Finished baseline engine to clone per sim.
        config (MonteCarloConfig): Run configuration (mode == 'rerun').

    Returns:
        MonteCarloResult: Distribution analytics over the simulations.
    """
    base = engine.stats
    metric_keys = list(config.metrics)

    bundle = {
        'strategy_cls': type(engine.strategy),
        'sesh': engine._sesh,
        'tick_inputs': engine._tick_inputs,
        'kline_inputs': engine._kline_inputs,
        'align_by_ts': engine._align_by_ts,
        'methods': config.resolved,
        'base_params': dict(getattr(engine.strategy, '_custom_params', {})),
    }

    evaluator = PoolEvaluator(
        _RerunEvaluator(metric_keys), bundle, workers=config.workers,
        desc="Montecarlo",
    )
    candidates = [
        Candidate(i, {'seed': _seed_for(config.seed, i)})
        for i in range(config.n_sims)
    ]
    results = evaluator.evaluate(candidates)

    rows: List[dict] = []
    final_equity = np.empty(config.n_sims, dtype=float)
    max_dd = np.empty(config.n_sims, dtype=float)

    for i, res in enumerate(sorted(results, key=lambda r: r.candidate_id)):
        m = res.metrics or {}
        rows.append({k: _as_float(m.get(k, np.nan)) for k in metric_keys})
        final_equity[i] = _as_float(m.get(_FINAL_EQUITY_KEY, np.nan))
        max_dd[i] = _as_float(m.get(_MAX_DD_KEY, np.nan))

    sims = pd.DataFrame(rows, columns=metric_keys)
    return MonteCarloResult(
        sims,
        base,
        final_equity=final_equity,
        max_drawdown=max_dd,
        initial_balance=base.initial_balance,
        confidence=config.confidence,
        equity_paths=None,
        method_names=[m.name for m in config.resolved],
    )

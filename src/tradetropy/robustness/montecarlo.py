"""
Monte Carlo orchestrator.

``MonteCarlo`` drives a robustness run over a finished backtest. It accepts a
``BacktestEngine`` (or a ``Stats`` object for the trade-level path) plus a
``MonteCarloConfig``, runs ``n_sims`` perturbed simulations and returns a
``MonteCarloResult``.

Two execution paths, selected by ``config.mode``:

    - 'trades'  Perturb the closed-trade list and recompute metrics in-process
                (fast, no engine re-run).
    - 'rerun'   Perturb data/parameters and re-run the engine for every
                simulation through a process pool (implemented in _runner).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from tradetropy.exceptions import ConfigError, DataError
from tradetropy.robustness.config import MonteCarloConfig
from tradetropy.robustness.core import recompute_stats
from tradetropy.robustness.result import MonteCarloResult

# Metrics always extracted internally for analytics, regardless of config.
_FINAL_EQUITY_KEY = 'Equity Final [$]'
_MAX_DD_KEY = 'Max. Drawdown [%]'


class MonteCarlo:
    """
    Monte Carlo robustness runner.

    Args:
        source: A finished BacktestEngine, or a Stats object (trade-level only).
        config (MonteCarloConfig): Run configuration.

    Raises:
        ConfigError: If a re-run method is requested without an engine source.
        DataError: If the baseline backtest has no trades.

    Example:
        from tradetropy.robustness import MonteCarlo, MonteCarloConfig
        cfg = MonteCarloConfig(n_sims=1000, methods=['shuffle_order'], seed=42)
        result = MonteCarlo(bt, cfg).run()
    """

    def __init__(self, source, config: MonteCarloConfig):
        self.config = config
        self._engine = None
        self._stats = self._resolve_source(source)

        if self.config.mode == 'rerun' and self._engine is None:
            raise ConfigError(
                'Data/parameter-level methods require a BacktestEngine source, '
                'not a bare Stats object.'
            )

        if self._stats is None or self._stats.trades is None \
                or len(self._stats.trades) == 0:
            raise DataError(
                'Monte Carlo requires a finished backtest with at least one '
                'closed trade.'
            )

    def _resolve_source(self, source):
        """
        Extract the baseline Stats from the source and keep the engine if any.

        Args:
            source: A BacktestEngine or a Stats object.

        Returns:
            Stats: The baseline statistics.

        Raises:
            ConfigError: If the source type is unsupported.
        """
        if hasattr(source, 'stats') and hasattr(source, 'run'):
            self._engine = source
            return source.stats
        # Duck-type a Stats object (pd.Series subclass with .trades).
        if hasattr(source, 'trades') and hasattr(source, 'equity_curve'):
            return source
        raise ConfigError(
            'MonteCarlo source must be a BacktestEngine or a Stats object.'
        )

    # ==========================================================================
    # Execution
    # ==========================================================================

    def run(self) -> MonteCarloResult:
        """
        Run all simulations and return the aggregated result.

        Returns:
            MonteCarloResult: Distribution analytics over the simulations.
        """
        if self.config.mode == 'trades':
            return self._run_trades()
        return self._run_rerun()

    def _run_trades(self) -> MonteCarloResult:
        """
        Trade-level path: perturb the trade list and recompute in-process.

        Returns:
            MonteCarloResult: Aggregated simulation analytics.
        """
        base = self._stats
        trades = base.trades
        initial_balance = base.initial_balance
        freq = base.freq
        metric_keys = list(self.config.metrics)
        methods = self.config.resolved

        rows: List[dict] = []
        final_equity = np.empty(self.config.n_sims, dtype=float)
        max_dd = np.empty(self.config.n_sims, dtype=float)
        equity_paths: List[pd.Series] = []

        for i in range(self.config.n_sims):
            rng = self._rng(i)
            perturbed = trades
            for method in methods:
                perturbed = method.apply(perturbed, rng)

            stats = recompute_stats(perturbed, trades, initial_balance, freq)

            rows.append({k: _as_float(stats.get(k, np.nan)) for k in metric_keys})
            final_equity[i] = _as_float(stats.get(_FINAL_EQUITY_KEY, np.nan))
            max_dd[i] = _as_float(stats.get(_MAX_DD_KEY, np.nan))
            equity_paths.append(stats.equity_curve)

        sims = pd.DataFrame(rows, columns=metric_keys)
        return MonteCarloResult(
            sims,
            base,
            final_equity=final_equity,
            max_drawdown=max_dd,
            initial_balance=initial_balance,
            confidence=self.config.confidence,
            equity_paths=equity_paths,
            method_names=[m.name for m in methods],
        )

    def _run_rerun(self) -> MonteCarloResult:
        """
        Data/parameter-level path: re-run the engine per simulation.

        Returns:
            MonteCarloResult: Aggregated simulation analytics.
        """
        from tradetropy.robustness._runner import run_rerun_simulations
        return run_rerun_simulations(self._engine, self.config)

    def _rng(self, i: int) -> np.random.Generator:
        """
        Build a per-simulation seeded generator.

        Args:
            i (int): Simulation index.

        Returns:
            np.random.Generator: Independent generator for this simulation.
        """
        if self.config.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.config.seed + i)


def _as_float(value) -> float:
    """
    Coerce a metric value to float, mapping failures to NaN.

    Args:
        value: Any metric value.

    Returns:
        float: The value as float, or NaN if not convertible.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')

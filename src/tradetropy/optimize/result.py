"""
Optimization results container.

OptimizationResult is the public class returned by BacktestEngine.optimize()
and by any Optimizer in this module.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from tradetropy.optimize.task import Result


class OptimizationResult:
    """
    Immutable container for optimization results.

    Provides access to best parameters, fitness values, and complete results
    as DataFrame for analysis.

    Attributes:
        best_params: Dictionary of optimal parameter values
        best_fitness: Fitness value of best candidate
        best_stats: Stats object of best candidate (if available)

    Example:
        result = bt.optimize(
            maximize='Sharpe Ratio',
            fast_ma=(5, 50, int, 5)
        )
        print(result.best_params)
        print(result.best_fitness)
        result.to_dataframe().sort_values('fitness', ascending=False)

        # Re-run backtest with optimal parameters
        bt_optimal = bt.rerun(params=result.best_params)
        bt_optimal.plot()
    """

    def __init__(self, results: List[Result], maximize: bool = True):
        """
        Initialize optimization result container.

        Args:
            results: List of Result objects from optimization
            maximize: Whether fitness was maximized (vs minimized)
        """
        # Sort once (fitness already has correct sign for maximize)
        self._results: List[Result] = sorted(
            results, key=lambda r: r.fitness, reverse=True
        )
        self._maximize = maximize

    # ---- Primary accessors ---------------------------------------------------

    def top(self, n: int = 1) -> List[Result]:
        """
        Get top n results by fitness.

        Args:
            n: Number of results to return

        Returns:
            List of Result objects ranked by fitness descending
        """
        return self._results[:n]

    @property
    def best_params(self) -> dict:
        """
        Parameters of best candidate.

        Returns:
            Dictionary of parameter values; empty dict if no results
        """
        if not self._results:
            return {}
        return (self._results[0].metrics or {}).get('params', {})

    @property
    def best_fitness(self) -> Optional[float]:
        """
        Fitness value of best candidate.

        Returns:
            Float fitness value; None if no results available
        """
        if not self._results:
            return None
        return self._results[0].fitness

    @property
    def best_stats(self):
        """
        Stats object of best candidate.

        Constructed from backtest metrics if available.

        Returns:
            Stats object or None if no results or empty metrics
        """
        if not self._results:
            return None

        metrics = self._results[0].metrics or {}
        # Exclude internal 'params' key
        stats_dict = {k: v for k, v in metrics.items() if k != 'params'}
        if not stats_dict:
            return None

        try:
            from collections import OrderedDict
            from tradetropy.stats import Stats
            return Stats(OrderedDict(stats_dict))
        except (TypeError, ValueError) as e:
            import warnings
            warnings.warn(
                f'OptimizationResult: unable to construct Stats from metrics: '
                f'{type(e).__name__}: {e}.'
            )
            return None

    # ---- DataFrame export ---------------------------------------------------

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert all results to DataFrame.

        Columns:
            - fitness: fitness value
            - param_<name>: one per parameter
            - <metric>: one per backtest statistic
            - error: error message or None

        Returns:
            pandas DataFrame with all results
        """
        import pandas as pd

        rows = []
        for r in self._results:
            row: dict = {'fitness': r.fitness, 'error': r.error}

            metrics = r.metrics or {}
            params = metrics.get('params', {})

            # Backtest metrics (excluding internal 'params' key)
            for k, v in metrics.items():
                if k != 'params':
                    row[k] = v

            # Parameters with prefix to avoid collisions
            for k, v in params.items():
                row[f'param_{k}'] = v

            rows.append(row)

        return pd.DataFrame(rows)

    # ---- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._results)
        bp = self.best_params
        bf = self.best_fitness
        return (
            f'OptimizationResult('
            f'n={n}, '
            f'best_fitness={bf:.4f}, '
            f'best_params={bp})'
        )
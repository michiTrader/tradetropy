"""
Base abstract class for parameter optimizers.
"""

from abc import ABC, abstractmethod
from typing import List, Any
from tradetropy.optimize.task import Result


class Optimizer(ABC):
    """
    Abstract base class defining the common interface for all optimizers.

    Subclasses implement different optimization strategies (grid search,
    random search, etc.) while maintaining a consistent results interface.

    Args:
        space: Parameter search space configuration
        fitness_metric: Metric to optimize (maximize or minimize)
    """

    def __init__(self, space: Any, fitness_metric: Any):
        self.space = space
        self.fitness_metric = fitness_metric
        self.results: List[Result] = []

    @abstractmethod
    def run(self, evaluator) -> None:
        """
        Execute the complete optimization process.

        Args:
            evaluator: Callable that evaluates candidates and returns results
        """
        ...

    def best(self, n: int = 1) -> List[Result]:
        """
        Return the top n results ranked by fitness.

        Args:
            n: Number of results to return

        Returns:
            List of Result objects sorted by fitness descending
        """
        sorted_results = sorted(self.results, key=lambda r: r.fitness, reverse=True)
        return sorted_results[:n]

    def to_dataframe(self):
        """
        Convert all results to a pandas DataFrame.

        Returns:
            DataFrame with columns:
                - fitness: fitness value
                - param_<name>: one column per parameter
                - <metric>: one column per backtest metric (excluding 'params')
                - error: error message or None
        """
        import pandas as pd

        rows = []
        for r in self.results:
            row: dict = {'fitness': r.fitness, 'error': r.error}

            # Expand backtest metrics, excluding internal 'params' key
            # which is expanded separately below
            if r.metrics:
                for k, v in r.metrics.items():
                    if k != 'params':
                        row[k] = v

            # Candidate parameters with 'param_' prefix to avoid
            # collisions with metric column names
            params = (r.metrics or {}).get('params', {})
            for k, v in params.items():
                row[f'param_{k}'] = v

            rows.append(row)

        return pd.DataFrame(rows)
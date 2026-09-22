"""
Candidate, Result, FitnessMetric definitions and evaluation functions.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from tradetropy.exceptions import ConfigError


@dataclass
class Candidate:
    """
    Parameter set to evaluate.

    Attributes:
        id: Unique candidate identifier
        params: Dictionary of parameter values
    """
    id: int
    params: Dict[str, Any]


@dataclass
class Result:
    """
    Result of evaluating a candidate.

    Attributes:
        candidate_id: ID of the evaluated candidate
        fitness: Computed fitness score
        metrics: Backtest statistics (always contains 'params' key)
        error: Error message if evaluation failed, None otherwise
    """
    candidate_id: int
    fitness: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


class FitnessMetric:
    """
    Extracts fitness value from a statistics dictionary.

    Attributes:
        metric: Statistic key name (e.g., 'Sharpe Ratio')
        maximize: True to maximize, False to minimize
        custom_fn: Optional custom extraction function
        missing_value: Value returned when metric is not found in stats
    """

    def __init__(
        self,
        metric: Optional[str] = None,
        maximize: bool = True,
        custom_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        missing_value: float = float('-inf'),
    ):
        """
        Initialize fitness metric extractor.

        Args:
            metric: Key in stats dict (required if custom_fn is None)
            maximize: True to maximize, False to minimize
            custom_fn: Optional function (stats: dict) -> float
                       Takes precedence over metric
            missing_value: Value if metric not found in stats

        Raises:
            ConfigError: When called without metric or custom_fn
        """
        self.metric = metric
        self.maximize = maximize
        self.custom_fn = custom_fn
        self.missing_value = missing_value

    def __call__(self, stats: Dict[str, Any]) -> float:
        """
        Extract fitness value from statistics.

        Args:
            stats: Backtest statistics dictionary

        Returns:
            Float fitness value (sign adjusted for maximization)

        Raises:
            ConfigError: If neither metric nor custom_fn is defined
        """
        if self.custom_fn:
            value = self.custom_fn(stats)
            return value if self.maximize else -value

        if self.metric is None:
            raise ConfigError("Must specify 'metric' or 'custom_fn'")

        if self.metric not in stats:
            return self.missing_value

        value = stats[self.metric]
        if value is None:
            return self.missing_value

        return float(value) if self.maximize else -float(value)


class _EvaluationFunction:
    """
    Picklable callable that executes a backtest and returns a Result.

    Used for multiprocessing compatibility (spawn method on Windows/macOS).
    Instance attributes must be picklable.
    """

    def __init__(self, backtest_fn, fitness_metric):
        self.backtest_fn = backtest_fn
        self.fitness_metric = fitness_metric

    def __call__(self, candidate, data):
        """
        Evaluate a candidate parameter set.

        Args:
            candidate: Candidate object with params
            data: Backtest data

        Returns:
            Result object with fitness and metrics
        """
        try:
            stats = self.backtest_fn(data, candidate.params)
            fitness = self.fitness_metric(stats)
            stats_with_params = dict(stats)
            stats_with_params['params'] = candidate.params
            return Result(candidate.id, fitness, stats_with_params)
        except Exception as exc:
            return Result(
                candidate.id,
                float('-inf'),
                {'params': candidate.params},
                error=str(exc)
            )


def _create_evaluation_function(backtest_fn, fitness_metric):
    """
    Factory function for creating picklable evaluation function.

    Args:
        backtest_fn: Module-level backtest function
        fitness_metric: FitnessMetric instance

    Returns:
        _EvaluationFunction instance compatible with multiprocessing spawn
    """
    return _EvaluationFunction(backtest_fn, fitness_metric)
"""
Grid search optimization strategy.
"""

from itertools import product
from tradetropy.optimize.optimizer import Optimizer
from tradetropy.optimize.task import Candidate


class GridSearchOptimizer(Optimizer):
    """
    Exhaustive search over all parameter combinations.

    Evaluates every possible combination of parameter values. Each parameter
    is an explicit list of values - no step or type specification needed.

    Args:
        space: Parameter search space
        fitness_metric: Metric to optimize
    """

    def __init__(self, space, fitness_metric=None):
        super().__init__(space, fitness_metric)

    def _generate_grid(self) -> list:
        """
        Generate all valid parameter combinations.

        Returns:
            List of valid parameter dictionaries
        """
        keys = list(self.space.ranges.keys())
        value_lists = [list(self.space.ranges[k]) for k in keys]
        all_combinations = [
            dict(zip(keys, combo)) for combo in product(*value_lists)
        ]
        return [c for c in all_combinations if self.space.is_valid(c)]

    def run(self, evaluator) -> None:
        """
        Run grid search over all parameter combinations.

        Args:
            evaluator: Evaluator to score candidates
        """
        param_list = self._generate_grid()
        candidates = [Candidate(i, params) for i, params in enumerate(param_list)]
        self.results = evaluator.evaluate(candidates)
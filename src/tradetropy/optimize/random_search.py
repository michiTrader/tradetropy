"""
Random search optimization strategy.
"""

from tradetropy.optimize.optimizer import Optimizer
from tradetropy.optimize.task import Candidate


class RandomSearchOptimizer(Optimizer):
    """
    Evaluate random parameter combinations with constraint validation.

    Samples random combinations from the parameter space, respecting any
    defined constraints. Useful for large search spaces where exhaustive
    grid search is impractical.

    Args:
        space: Parameter search space
        fitness_metric: Metric to optimize
        iterations: Number of random combinations to evaluate (default 100)
    """

    def __init__(self, space, fitness_metric=None, iterations=100):
        super().__init__(space, fitness_metric)
        self.iterations = iterations

    def run(self, evaluator):
        """
        Execute random search for specified number of iterations.

        Args:
            evaluator: Evaluator to score candidates
        """
        param_list = [self.space.sample() for _ in range(self.iterations)]
        candidates = [Candidate(i, p) for i, p in enumerate(param_list)]
        self.results = evaluator.evaluate(candidates)

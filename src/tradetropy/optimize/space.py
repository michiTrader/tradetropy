"""
Define parameter search space and constraints.

Each parameter is specified as an explicit list of values to explore:

    bt.optimize(
        maximize  = 'Sharpe Ratio',
        fast_ma   = [10, 20, 30, 50],
        threshold = [0.1, 0.5, 1.0],
        mode      = ['trend', 'mean_revert'],
    )

Values can be int, float, or str. The range() built-in is also supported.
"""

import random
from typing import Any, Callable, Dict, List

from tradetropy.exceptions import ConfigError


class ParamValues:
    """
    Discrete list of possible values for a parameter.

    Args:
        name: Parameter name
        values: List of possible values
    """

    def __init__(self, name: str, values: List[Any]):
        if not values:
            raise ConfigError(f"Value list for '{name}' cannot be empty.")
        self.name = name
        self.values = list(values)

    def sample(self) -> Any:
        """
        Return a random value from the list.

        Returns:
            A randomly selected value
        """
        return random.choice(self.values)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class ParameterSpace:
    """
    Search space with parameters and constraints.

    Manages parameter ranges and validation constraints for optimization.
    Supports both explicit value lists and range objects.

    Args:
        **param_lists: Keyword arguments where each value is a list of
                       parameter values (e.g., fast_ma=[10, 20, 30])

    Example:
        space = ParameterSpace(
            fast_ma=[10, 20, 30],
            threshold=[0.1, 0.5]
        )
    """

    def __init__(self, **param_lists: List):
        self.ranges: Dict[str, ParamValues] = {}
        self.constraints: List[Callable[[Dict[str, Any]], bool]] = []
        for name, values in param_lists.items():
            self.add_param(name, values)

    def add_param(self, name: str, values) -> None:
        """
        Add a parameter to the search space.

        Args:
            name: Parameter name
            values: List of values or range object (e.g., range(10, 51, 5))

        Raises:
            ConfigError: If values is not a list or range
        """
        if isinstance(values, range):
            values = list(values)
        if not isinstance(values, list):
            raise ConfigError(
                f"Parameter '{name}' must be a list or range. "
                f"Example: {name}=[10, 20, 30] or {name}=range(10, 51, 5)"
            )
        self.ranges[name] = ParamValues(name, values)

    def add_constraint(self, fn: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Add a validation constraint.

        The constraint function receives the parameter dict and returns True
        if valid, False otherwise.

        Args:
            fn: Callable that validates a parameter dict
        """
        self.constraints.append(fn)

    def is_valid(self, params: Dict[str, Any]) -> bool:
        """
        Check if a parameter combination satisfies all constraints.

        Args:
            params: Dictionary of parameter values

        Returns:
            True if all constraints are satisfied
        """
        return all(c(params) for c in self.constraints)

    def sample(self, max_attempts: int = 1000) -> Dict[str, Any]:
        """
        Generate a random valid parameter dictionary.

        Args:
            max_attempts: Maximum attempts to generate valid combination

        Returns:
            Random valid parameter dictionary

        Raises:
            ConfigError: If unable to generate valid params after max_attempts
        """
        for _ in range(max_attempts):
            params = {name: pv.sample() for name, pv in self.ranges.items()}
            if self.is_valid(params):
                return params
        raise ConfigError(
            'Unable to generate valid parameter set after maximum attempts. '
            'Check your constraints.'
        )
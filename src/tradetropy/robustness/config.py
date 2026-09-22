"""
Monte Carlo configuration.

``MonteCarloConfig`` holds every option of a robustness run and is responsible
for validating them and resolving method specifications (strings or instances)
into concrete ``MCMethod`` objects. From the resolved methods it derives the
execution mode:

    - 'trades'  All methods operate on the closed-trade list (no re-run).
    - 'rerun'   At least one method operates on data or parameters and the
                engine must be re-run for every simulation.

Mixing trade-level methods with data/parameter-level methods is rejected: the
former post-process a finished backtest while the latter require fresh runs, so
they cannot be composed in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from tradetropy.exceptions import ConfigError
from tradetropy.robustness.methods import (
    MCMethod,
    MethodSpec,
    RERUN_LEVELS,
    resolve_method,
)

# Metrics tracked by default across simulations. Each must be a valid Stats key.
DEFAULT_METRICS: Tuple[str, ...] = (
    'Return [%]',
    'Max. Drawdown [%]',
    'Sharpe Ratio',
    'Profit Factor',
    'Win Rate [%]',
)

DEFAULT_CONFIDENCE: Tuple[float, ...] = (0.95, 0.99)


@dataclass
class MonteCarloConfig:
    """
    Configuration for a Monte Carlo robustness run.

    Args:
        n_sims (int): Number of simulations to run. Must be > 0.
        methods (Sequence): Method identifiers (str) or MCMethod instances.
        metrics (Sequence[str]): Stats keys to track across simulations.
        confidence (Sequence[float]): Confidence levels in (0, 1) for intervals.
        seed (int | None): Base seed for reproducibility (None for random).
        workers (int | None): Parallel processes for the re-run path
            (None -> cpu_count). Ignored on the trade-level path.

    Raises:
        ConfigError: If any option is invalid or method levels are incompatible.

    Example:
        cfg = MonteCarloConfig(
            n_sims=1000,
            methods=['shuffle_order', 'resample_trades'],
            confidence=(0.95, 0.99),
            seed=42,
        )
        cfg.mode          # -> 'trades'
        cfg.resolved      # -> [ShuffleOrder(), ResampleTrades(replace=True)]
    """

    n_sims: int = 1000
    methods: Sequence[MethodSpec] = field(default_factory=lambda: ['resample_trades'])
    metrics: Sequence[str] = DEFAULT_METRICS
    confidence: Sequence[float] = DEFAULT_CONFIDENCE
    seed: Optional[int] = None
    workers: Optional[int] = None

    # Derived fields (populated in __post_init__).
    resolved: List[MCMethod] = field(default_factory=list, init=False, repr=False)
    mode: str = field(default='trades', init=False)

    def __post_init__(self) -> None:
        if self.n_sims <= 0:
            raise ConfigError('MonteCarloConfig n_sims must be > 0.')

        if not self.methods:
            raise ConfigError('MonteCarloConfig requires at least one method.')

        if not self.metrics:
            raise ConfigError('MonteCarloConfig requires at least one metric.')

        for c in self.confidence:
            if not 0.0 < c < 1.0:
                raise ConfigError(
                    f'Confidence level {c} must be in the open interval (0, 1).'
                )

        if self.workers is not None and self.workers <= 0:
            raise ConfigError('MonteCarloConfig workers must be > 0 or None.')

        self.resolved = [resolve_method(m) for m in self.methods]
        self.mode = self._derive_mode(self.resolved)

    @staticmethod
    def _derive_mode(methods: List[MCMethod]) -> str:
        """
        Derive the execution mode from the resolved methods.

        Args:
            methods (list[MCMethod]): Resolved perturbation methods.

        Returns:
            str: 'trades' if all methods are trade-level, else 'rerun'.

        Raises:
            ConfigError: If trade-level methods are mixed with re-run methods.
        """
        levels = {m.level for m in methods}
        has_trades = 'trades' in levels
        has_rerun = bool(levels & set(RERUN_LEVELS))
        if has_trades and has_rerun:
            raise ConfigError(
                'Cannot mix trade-level methods with data/parameter-level '
                'methods in the same run: the former post-process a finished '
                'backtest while the latter require fresh re-runs. Run them '
                'separately.'
            )
        return 'trades' if has_trades else 'rerun'

"""
Monte Carlo robustness testing.

Generate many randomized variants of a finished backtest to estimate the
distribution of performance metrics, confidence intervals, the probability of
loss, the risk of ruin and a composite robustness score.

Public API:

    from tradetropy.robustness import MonteCarlo, MonteCarloConfig
    from tradetropy.robustness.methods import (
        ShuffleOrder, ResampleTrades, SkipTrades, RandomizeSlippage,
        RandomStartIndex,
    )

    result = MonteCarlo(bt, MonteCarloConfig(n_sims=1000, seed=42)).run()
    print(result.summary())

Or through the convenience method on the engine:

    result = bt.montecarlo(n_sims=1000, methods=['shuffle_order'], seed=42)
"""

from tradetropy.robustness.config import MonteCarloConfig
from tradetropy.robustness.montecarlo import MonteCarlo
from tradetropy.robustness.result import MonteCarloResult
from tradetropy.robustness.methods import (
    MCMethod,
    TradeMethod,
    DataMethod,
    ParamMethod,
    ShuffleOrder,
    ResampleTrades,
    SkipTrades,
    RandomizeSlippage,
    RandomStartIndex,
    RandomizePrices,
    RandomStartBar,
    RandomizeParameters,
    available_methods,
    resolve_method,
)

__all__ = [
    'MonteCarloConfig',
    'MonteCarlo',
    'MonteCarloResult',
    'MCMethod',
    'TradeMethod',
    'DataMethod',
    'ParamMethod',
    'ShuffleOrder',
    'ResampleTrades',
    'SkipTrades',
    'RandomizeSlippage',
    'RandomStartIndex',
    'RandomizePrices',
    'RandomStartBar',
    'RandomizeParameters',
    'available_methods',
    'resolve_method',
]

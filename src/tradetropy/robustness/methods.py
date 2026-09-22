"""
Monte Carlo perturbation methods.

A perturbation method takes the outcome of a baseline backtest and produces a
randomized variant of it for a single Monte Carlo simulation. Methods are
grouped by the level at which they operate:

    - 'trades'  Operate on the closed-trade list of a finished backtest and do
                NOT re-run the engine. Fast: the equity path is rebuilt from the
                perturbed per-trade PnL series and metrics are recomputed.
    - 'data'    Perturb the input price series (klines/ticks) and require a full
                re-run of the engine for every simulation.
    - 'params'  Perturb the strategy parameters around their baseline value and
                require a full re-run of the engine for every simulation.

All methods are seeded through an explicit ``numpy.random.Generator`` so that a
configured ``seed`` yields fully reproducible simulations.

New methods register themselves with the ``@register`` decorator, which exposes
them to the public API by a short string identifier (e.g. ``'shuffle_order'``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd

from tradetropy.exceptions import ConfigError

# Levels that require re-running the engine (vs. post-processing the trades).
RERUN_LEVELS = ('data', 'params')

# Method identifier registry, populated by the @register decorator.
_REGISTRY: Dict[str, Type['MCMethod']] = {}


def register(name: str):
    """
    Class decorator that registers a perturbation method by string id.

    Args:
        name (str): Public identifier, e.g. 'shuffle_order'.

    Returns:
        Callable: The decorator that records the class in the registry.

    Raises:
        ConfigError: If the identifier is already registered.
    """
    def _decorator(cls: Type['MCMethod']) -> Type['MCMethod']:
        key = name.lower()
        if key in _REGISTRY:
            raise ConfigError(f"Monte Carlo method '{name}' is already registered.")
        cls.name = key
        _REGISTRY[key] = cls
        return cls
    return _decorator


def available_methods() -> List[str]:
    """
    List the registered method identifiers.

    Returns:
        list[str]: Sorted list of available method names.
    """
    return sorted(_REGISTRY)


# Accepts a registered string id or a ready-made MCMethod instance.
MethodSpec = Union[str, 'MCMethod']


def resolve_method(spec: MethodSpec) -> 'MCMethod':
    """
    Resolve a method specification into an MCMethod instance.

    Args:
        spec: Registered string identifier or an MCMethod instance.

    Returns:
        MCMethod: The resolved instance (instances are returned unchanged).

    Raises:
        ConfigError: If the string identifier is unknown or the type is invalid.
    """
    if isinstance(spec, MCMethod):
        return spec
    if isinstance(spec, str):
        key = spec.lower()
        if key not in _REGISTRY:
            raise ConfigError(
                f"Unknown Monte Carlo method '{spec}'. "
                f"Available methods: {available_methods()}."
            )
        return _REGISTRY[key]()
    raise ConfigError(
        f"A Monte Carlo method must be a string id or an MCMethod instance, "
        f"got {type(spec).__name__}."
    )


# ==========================================================================
# Base classes
# ==========================================================================


class MCMethod(ABC):
    """
    Abstract base class for a Monte Carlo perturbation method.

    Attributes:
        level (str): Operating level, one of 'trades', 'data' or 'params'.
        name (str): Registered identifier, set by the @register decorator.
    """

    level: str = 'trades'
    name: str = 'method'

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


class TradeMethod(MCMethod):
    """
    Perturbation that rewrites the closed-trade list without re-running.

    Subclasses transform the normalized trades DataFrame (columns described in
    ``tradetropy.stats``: pnl_net, size, entry_price, exit_price, direction,
    entry_time, exit_time, commission, symbol) and return a new DataFrame.
    """

    level = 'trades'

    @abstractmethod
    def apply(self, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """
        Produce a perturbed copy of the trades for one simulation.

        Args:
            trades (pd.DataFrame): Baseline closed-trade list.
            rng (np.random.Generator): Seeded random generator.

        Returns:
            pd.DataFrame: Perturbed trades (index reset).
        """
        ...


class DataMethod(MCMethod):
    """
    Perturbation that rewrites the input price data, requiring a re-run.

    Subclasses transform the tuple of typed inputs (KlineData/TickData) and
    return a new tuple to feed a fresh BacktestEngine.
    """

    level = 'data'

    @abstractmethod
    def apply(self, inputs: Tuple, rng: np.random.Generator) -> Tuple:
        """
        Produce a perturbed copy of the input data for one simulation.

        Args:
            inputs (tuple): Baseline typed inputs (KlineData/TickData).
            rng (np.random.Generator): Seeded random generator.

        Returns:
            tuple: Perturbed inputs of the same type and shape.
        """
        ...


class ParamMethod(MCMethod):
    """
    Perturbation that rewrites the strategy parameters, requiring a re-run.
    """

    level = 'params'

    @abstractmethod
    def apply(self, params: Dict, rng: np.random.Generator) -> Dict:
        """
        Produce a perturbed copy of the strategy parameters.

        Args:
            params (dict): Baseline parameter values.
            rng (np.random.Generator): Seeded random generator.

        Returns:
            dict: Perturbed parameter values.
        """
        ...


# ==========================================================================
# Trade-level methods
# ==========================================================================


@register('shuffle_order')
class ShuffleOrder(TradeMethod):
    """
    Randomly reorder the sequence of trades.

    Net profit is preserved but the equity path changes, exposing how much the
    drawdown profile depends on the specific ordering of wins and losses.
    """

    def apply(self, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        order = rng.permutation(len(trades))
        return trades.iloc[order].reset_index(drop=True)


@register('resample_trades')
class ResampleTrades(TradeMethod):
    """
    Bootstrap resample of the trade list.

    Draws ``len(trades)`` trades from the baseline. With replacement (default)
    this is a standard bootstrap; without replacement it is a pure reshuffle.

    Args:
        replace (bool): Sample with replacement when True.
    """

    def __init__(self, replace: bool = True):
        self.replace = bool(replace)

    def apply(self, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        n = len(trades)
        idx = rng.choice(n, size=n, replace=self.replace)
        return trades.iloc[idx].reset_index(drop=True)

    def __repr__(self) -> str:
        return f'ResampleTrades(replace={self.replace})'


@register('skip_trades')
class SkipTrades(TradeMethod):
    """
    Randomly drop a fraction of trades.

    Each trade is independently removed with probability ``prob``. Tests how
    dependent the result is on a small number of individual trades.

    Args:
        prob (float): Per-trade removal probability in [0, 1).

    Raises:
        ConfigError: If prob is outside [0, 1).
    """

    def __init__(self, prob: float = 0.1):
        if not 0.0 <= prob < 1.0:
            raise ConfigError('SkipTrades prob must be in [0, 1).')
        self.prob = float(prob)

    def apply(self, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        keep = rng.random(len(trades)) >= self.prob
        kept = trades.iloc[keep]
        # Guarantee at least one trade so metrics remain defined.
        if kept.empty:
            kept = trades.iloc[[rng.integers(len(trades))]]
        return kept.reset_index(drop=True)

    def __repr__(self) -> str:
        return f'SkipTrades(prob={self.prob})'


@register('randomize_slippage')
class RandomizeSlippage(TradeMethod):
    """
    Add random execution cost (slippage) to every trade.

    A random slippage drawn uniformly in [0, ``bps``] basis points of the
    traded notional is subtracted from each trade's net PnL, plus an optional
    flat ``commission`` per trade. Models execution uncertainty.

    Args:
        bps (float): Maximum one-way slippage in basis points (1 bp = 0.01%).
        commission (float): Additional flat cost subtracted per trade.

    Raises:
        ConfigError: If bps or commission is negative.
    """

    def __init__(self, bps: float = 1.0, commission: float = 0.0):
        if bps < 0.0 or commission < 0.0:
            raise ConfigError('RandomizeSlippage bps and commission must be >= 0.')
        self.bps = float(bps)
        self.commission = float(commission)

    def apply(self, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = trades.copy()
        n = len(out)
        notional = (out['entry_price'].to_numpy(dtype=float)
                    + out['exit_price'].to_numpy(dtype=float)) \
            * out['size'].to_numpy(dtype=float)
        notional = np.nan_to_num(np.abs(notional), nan=0.0)
        slip_bps = rng.uniform(0.0, self.bps, size=n) / 10_000.0
        cost = notional * slip_bps + self.commission
        out['pnl_net'] = out['pnl_net'].to_numpy(dtype=float) - cost
        return out.reset_index(drop=True)

    def __repr__(self) -> str:
        return f'RandomizeSlippage(bps={self.bps}, commission={self.commission})'


@register('random_start_index')
class RandomStartIndex(TradeMethod):
    """
    Start the trade sequence at a random offset.

    Drops a random number of leading trades (up to ``max_frac`` of the list),
    keeping the chronological tail. Tests dependence on the earliest trades.

    Args:
        max_frac (float): Maximum fraction of leading trades to drop, in [0, 1).

    Raises:
        ConfigError: If max_frac is outside [0, 1).
    """

    def __init__(self, max_frac: float = 0.25):
        if not 0.0 <= max_frac < 1.0:
            raise ConfigError('RandomStartIndex max_frac must be in [0, 1).')
        self.max_frac = float(max_frac)

    def apply(self, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        n = len(trades)
        max_skip = int(n * self.max_frac)
        start = int(rng.integers(0, max_skip + 1)) if max_skip > 0 else 0
        kept = trades.iloc[start:]
        if kept.empty:
            kept = trades.iloc[-1:]
        return kept.reset_index(drop=True)

    def __repr__(self) -> str:
        return f'RandomStartIndex(max_frac={self.max_frac})'



# ==========================================================================
# Data-level methods
# ==========================================================================
#
# Column layouts (see tradetropy.core.data_types):
#   KlineData.data : [N x 7] -> ts, open, high, low, close, volume, turnover
#   TickData.data  : [N x 7] -> ts, bid, ask, volume, flags, volume_real, price

# Price column indices perturbed by RandomizePrices, per input type.
_KLINE_PRICE_COLS = (1, 2, 3, 4)   # open, high, low, close
_TICK_PRICE_COLS = (1, 2, 6)       # bid, ask, price


def _is_kline(inp) -> bool:
    """Return True when the input carries an OHLC interval (KlineData)."""
    return getattr(inp, 'interval_ms', None) is not None


@register('randomize_prices')
class RandomizePrices(DataMethod):
    """
    Add bounded multiplicative noise to the input price series.

    Each price is multiplied by ``1 + u`` with ``u`` drawn uniformly in
    [-noise_pct, +noise_pct]. For klines the OHLC relationship is restored
    afterwards (high = max, low = min) when ``respect_ohlc`` is True. Models
    the impact of small data/quote differences on the result.

    Args:
        noise_pct (float): Maximum relative perturbation (e.g. 0.001 = 0.1%).
        respect_ohlc (bool): Re-derive high/low so candles stay valid.

    Raises:
        ConfigError: If noise_pct is negative.
    """

    def __init__(self, noise_pct: float = 0.001, respect_ohlc: bool = True):
        if noise_pct < 0.0:
            raise ConfigError('RandomizePrices noise_pct must be >= 0.')
        self.noise_pct = float(noise_pct)
        self.respect_ohlc = bool(respect_ohlc)

    def apply(self, inputs: Tuple, rng: np.random.Generator) -> Tuple:
        import dataclasses
        out = []
        for inp in inputs:
            data = np.array(inp.data, dtype=np.float64, copy=True)
            n = len(data)
            kline = _is_kline(inp)
            cols = _KLINE_PRICE_COLS if kline else _TICK_PRICE_COLS
            for col in cols:
                factor = 1.0 + rng.uniform(-self.noise_pct, self.noise_pct, size=n)
                data[:, col] = data[:, col] * factor
            if kline and self.respect_ohlc:
                ohlc = data[:, 1:5]
                data[:, 2] = ohlc.max(axis=1)
                data[:, 3] = ohlc.min(axis=1)
            out.append(dataclasses.replace(inp, data=data))
        return tuple(out)

    def __repr__(self) -> str:
        return (f'RandomizePrices(noise_pct={self.noise_pct}, '
                f'respect_ohlc={self.respect_ohlc})')


@register('random_start_bar')
class RandomStartBar(DataMethod):
    """
    Start the backtest at a random leading offset.

    Drops a random number of leading rows (up to ``max_frac`` of the series),
    keeping the tail. Tests dependence on the exact start of the data window.

    Args:
        max_frac (float): Maximum fraction of leading rows to drop, in [0, 1).

    Raises:
        ConfigError: If max_frac is outside [0, 1).
    """

    def __init__(self, max_frac: float = 0.1):
        if not 0.0 <= max_frac < 1.0:
            raise ConfigError('RandomStartBar max_frac must be in [0, 1).')
        self.max_frac = float(max_frac)

    def apply(self, inputs: Tuple, rng: np.random.Generator) -> Tuple:
        import dataclasses
        # A single shared offset fraction keeps multi-symbol inputs aligned.
        frac = rng.uniform(0.0, self.max_frac)
        out = []
        for inp in inputs:
            n = len(inp.data)
            start = min(int(n * frac), max(n - 1, 0))
            out.append(dataclasses.replace(inp, data=inp.data[start:]))
        return tuple(out)

    def __repr__(self) -> str:
        return f'RandomStartBar(max_frac={self.max_frac})'



# ==========================================================================
# Parameter-level methods
# ==========================================================================


@register('randomize_parameters')
class RandomizeParameters(ParamMethod):
    """
    Re-run the strategy with randomly permuted parameters.

    Samples a valid parameter combination from a search space for every
    simulation, measuring how sensitive the result is to the exact parameter
    values (parameter permutation / sensitivity). Reuses the optimizer's
    ``ParameterSpace`` so constraints and value lists behave identically.

    Args:
        space (ParameterSpace | None): Explicit search space. If omitted, one
            is built from the keyword value lists.
        **param_lists: Parameter name -> list/range of values, used only when
            ``space`` is not given.

    Raises:
        ConfigError: If both a space and keyword lists are provided.

    Example:
        RandomizeParameters(fast=[10, 20, 30], slow=[50, 100, 200])
    """

    def __init__(self, space=None, **param_lists):
        from tradetropy.optimize import ParameterSpace
        if space is not None and param_lists:
            raise ConfigError(
                'RandomizeParameters: pass either a ParameterSpace or keyword '
                'value lists, not both.'
            )
        self.space = space if space is not None else ParameterSpace(**param_lists)

    def apply(self, params: Dict, rng: np.random.Generator) -> Dict:
        ranges = self.space.ranges
        if not ranges:
            return dict(params)
        for _ in range(1000):
            candidate = {
                name: pv.values[int(rng.integers(len(pv.values)))]
                for name, pv in ranges.items()
            }
            if self.space.is_valid(candidate):
                return candidate
        raise ConfigError(
            'RandomizeParameters: could not sample a valid parameter set; '
            'check your constraints.'
        )

    def __repr__(self) -> str:
        return f'RandomizeParameters(params={list(self.space.ranges)})'

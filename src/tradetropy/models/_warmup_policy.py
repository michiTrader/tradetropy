"""
Unified warmup policy shared across all engines.

Strategy.warmup has a unified semantic across backtest, live, and replay:

- warmup as int (including 0) -> interpreted literally and identically
  in all engines. Backtest silences first N bars/ticks; live requests
  N historical bars from broker; replay reserves N from dataset as history.

- warmup as None -> all engines auto-calculate using the same criterion:
  maximum min_periods of OHLC indicators in the strategy (pure, in bars).

This module centralizes that criterion to prevent engine divergence.
"""

from __future__ import annotations

import os

import numpy as np

# Warmup diagnostics are opt-in: a published library must not print to stdout on
# every run. Set the TRADETROPY_DEBUG_WARMUP environment variable to any non-empty
# value to re-enable the warmup summary / debug block during development.
_WARMUP_DEBUG_ENV = 'TRADETROPY_DEBUG_WARMUP'


def warmup_debug_enabled() -> bool:
    """Return True when warmup diagnostics are explicitly enabled via env var."""
    return bool(os.environ.get(_WARMUP_DEBUG_ENV))


def _TickProxy():
    """
    Lazy import to avoid circular import cycle.

    Returns:
        TickProxy class
    """
    from tradetropy.data.data import TickProxy
    return TickProxy


def estimate_ticks_per_candle(strategy, tick_matrices: dict) -> int:
    """
    Estimate ticks per bar from data.

    Computes N_ticks / n_unique_bars from first OhlcProxy. Used to convert
    auto warmup (bars) to ticks in tick mode. Returns 1 if no OHLC proxies
    or tick data available.

    Args:
        strategy: Strategy instance
        tick_matrices: Dictionary mapping symbols to tick arrays

    Returns:
        Estimated ticks per bar (minimum 1)
    """
    if not strategy._ohlc_proxies or not tick_matrices:
        return 1
    intervalo = strategy._ohlc_proxies[0].interval_ms
    if not intervalo or intervalo <= 0:
        return 10
    sym = strategy._ohlc_proxies[0].symbol
    arr = tick_matrices.get(sym)
    if arr is None or len(arr) == 0:
        arr = next(iter(tick_matrices.values()), None)
    if arr is None or len(arr) == 0:
        return 1
    ts = np.asarray(arr)[:, 0]
    ts_bars = (ts // intervalo) * intervalo
    n_bars = max(1, len(np.unique(ts_bars)))
    return max(1, len(arr) // n_bars)


#: Default convergence factor applied to recursive indicators that opt in via
#: ``Indicator.warmup_factor`` (RSI = Wilder, MACD = EMA). Their seed influence
#: decays geometrically with the smoothing period, so a warmup of a few times
#: the indicator ``length`` makes the first evaluated bar effectively converged.
INDICATOR_CONVERGENCE_FACTOR = 5


def auto_warmup_candles(strategy) -> int:
    """
    Calculate auto warmup criterion for OHLC indicators.

    Returns the maximum warmup-in-bars over the strategy's OHLC indicators.
    Each indicator needs at least its ``min_periods`` bars to emit a value;
    recursive indicators (RSI, MACD) additionally opt into a larger warmup via
    their ``warmup_factor`` (so their value converges before on_data() starts).
    The per-indicator requirement is therefore
    ``max(min_periods, warmup_factor * length)``, which leaves windowed
    indicators (SMA, Volume Profile, ...) at ``min_periods`` and only lengthens
    the warmup for indicators that declare they need it. Excludes indicators on
    TickProxy (warmup is measured in bars). Returns 0 if no OHLC indicators are
    defined.

    Args:
        strategy: Strategy instance

    Returns:
        Maximum warmup in bars
    """

    def _bars(indicator) -> int:
        min_periods = int(getattr(indicator, 'min_periods', 0) or 0)
        length = int(getattr(indicator, 'length', 0) or 0)
        factor = int(getattr(indicator, 'warmup_factor', 1) or 1)
        return max(min_periods, factor * length)

    return max(
        (
            _bars(defn['indicator'])
            for defn in strategy._indicator_defs
            if not isinstance(defn['source'].proxy, _TickProxy())
        ),
        default=0,
    )


def describe_warmup(strategy, feed_type: str, value: int, context: str = '') -> str:
    """
    Human-readable description of selected warmup and its unit.

    Args:
        strategy: Strategy instance
        feed_type: 'tick' or 'kline'
        value: Warmup value in feed units
        context: Optional context label

    Returns:
        Formatted description string with units and source (explicit/auto)
    """
    w = getattr(strategy, 'warmup', None)
    source = 'explicit' if w is not None else 'auto (min_periods)'
    unit = 'ticks' if feed_type == 'tick' else 'bars'
    prefix = f'[{context}] ' if context else ''
    return (
        f'{prefix}warmup selected = {value} {unit} '
        f'(mode={feed_type}, source={source})'
    )


def log_warmup(strategy, feed_type: str, value: int, context: str = '') -> None:
    """
    Print the one-line warmup summary, but only when diagnostics are enabled.

    A published library must stay silent on stdout by default; set the
    TRADETROPY_DEBUG_WARMUP environment variable to see the summary.
    """
    if warmup_debug_enabled():
        print(describe_warmup(strategy, feed_type, value, context))


def debug_warmup_block(
    context: str,
    feed_type: str,
    *,
    total_data_points: int,
    warmup: int,
    n_indicators: int = 0,
    auto_bars: int = 0,
    ticks_per_candle: int = 1,
    n_candles_in_dataset: int = 0,
    extra: dict | None = None,
) -> None:
    """
    Print uniform debug block for warmup and data processing.

    Used to diagnose differences between engines (BacktestEngine vs
    ReplayEngine) on the same dataset.

    Args:
        context: Context label for the debug block
        feed_type: 'tick' or 'kline'
        total_data_points: Total data points
        warmup: Warmup value
        n_indicators: Number of OHLC indicators
        auto_bars: Max min_periods from indicators
        ticks_per_candle: Estimated ticks per bar
        n_candles_in_dataset: Number of bars in dataset
        extra: Optional dictionary of additional debug info
    """
    if not warmup_debug_enabled():
        return
    unit = 'ticks' if feed_type == 'tick' else 'bars'
    lines = [
        f'[{context}] DEBUG warmup/data ========',
        f'Feed mode           : {feed_type} (warmup unit = {unit})',
        f'Total data points   : {total_data_points} {unit if feed_type == "kline" else "ticks"}',
        f'Bars in dataset     : {n_candles_in_dataset}',
        f'Est. ticks per bar  : {ticks_per_candle}',
        f'OHLC indicators     : {n_indicators}  (max min_periods = {auto_bars} bars)',
        f'Warmup resolved     : {warmup} {unit}',
        f'Data post-warmup    : {total_data_points - warmup}',
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f'{k:<19}: {v}')
    lines.append('=' * 40)
    print('\n'.join(lines))


def resolve_warmup(strategy, feed_type: str = 'kline', ticks_per_candle: int = 1) -> int:
    """
    Calculate effective warmup in loop units.

    In tick mode: unit is ticks. In kline mode: unit is bars.

    Args:
        strategy: Strategy instance
        feed_type: 'tick' or 'kline'
        ticks_per_candle: Estimated ticks per bar (only used for auto in tick mode)

    Returns:
        Warmup value in feed units
    """
    w = getattr(strategy, 'warmup', None)
    if w is not None:
        return int(w)
    auto = auto_warmup_candles(strategy)
    if feed_type == 'tick':
        return auto * max(1, int(ticks_per_candle))
    return auto

# Legacy aliases
resolver_warmup = resolve_warmup  # noqa: F841
estimate_ticks_per_candle = estimate_ticks_per_candle  # noqa: F841

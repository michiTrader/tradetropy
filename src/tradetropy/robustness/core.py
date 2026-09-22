"""
Trade-level recomputation core.

For trade-level perturbations the engine is never re-run. Instead the perturbed
closed-trade list is replayed onto a synthetic timeline (reusing the original
trade time slots so the overall span and spacing are preserved) and the full
metric set is recomputed through ``tradetropy.stats.compute_stats``. Reusing
``compute_stats`` guarantees the perturbed metrics are defined identically to
the baseline ones.

Statistical reliability gating is disabled here (``min_trades=1``,
``min_duration_ann=0``): a Monte Carlo run must always yield a numeric value per
simulation to build a distribution, so the gating that ``compute_stats`` applies
to tiny one-off backtests would be counterproductive.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tradetropy.stats.stats import compute_stats


def _direction_to_type(direction: str) -> int:
    """
    Map a direction label to the broker trade type code.

    Args:
        direction (str): 'long' or 'short'.

    Returns:
        int: 0 for long, 1 for short.
    """
    return 0 if str(direction).lower().startswith('l') else 1


class _SyntheticTrade:
    """
    Minimal trade object exposing the attributes compute_stats reads.

    Mirrors the duck-typed contract of broker Trade objects so the perturbed
    trade list can flow through the standard stats pipeline unchanged.
    """

    __slots__ = (
        'time', 'time_close', 'volume', 'price', 'price_close',
        'type', 'commission', 'pnl_net', 'symbol',
    )

    def __init__(self, *, entry_time, exit_time, volume, entry_price,
                 exit_price, direction, commission, pnl_net, symbol):
        self.time = entry_time
        self.time_close = exit_time
        self.volume = float(volume)
        self.price = float(entry_price)
        self.price_close = float(exit_price)
        self.type = _direction_to_type(direction)
        self.commission = float(commission)
        self.pnl_net = float(pnl_net)
        self.symbol = str(symbol)


def recompute_stats(
    perturbed: pd.DataFrame,
    timeline: pd.DataFrame,
    initial_balance: float,
    freq: str,
):
    """
    Recompute a full Stats object from a perturbed trade list.

    The perturbed trades supply the PnL, prices, size, direction, commission
    and symbol; the chronological timeline (entry/exit slots of the baseline
    trades) supplies the timestamps so the equity path keeps the original span.

    Args:
        perturbed (pd.DataFrame): Perturbed closed-trade list.
        timeline (pd.DataFrame): Baseline trades, sorted, used for time slots.
        initial_balance (float): Starting account capital.
        freq (str): Frequency for annualized ratios (baseline stats.freq).

    Returns:
        Stats: Recomputed metrics for this simulation.
    """
    n = len(perturbed)
    entry_slots = timeline['entry_time'].to_numpy()[:n]
    exit_slots = timeline['exit_time'].to_numpy()[:n]

    sizes = perturbed['size'].to_numpy(dtype=float)
    entry_p = perturbed['entry_price'].to_numpy(dtype=float)
    exit_p = perturbed['exit_price'].to_numpy(dtype=float)
    directions = perturbed['direction'].to_numpy()
    commissions = perturbed['commission'].to_numpy(dtype=float)
    pnl = perturbed['pnl_net'].to_numpy(dtype=float)
    symbols = perturbed['symbol'].to_numpy()

    trades = [
        _SyntheticTrade(
            entry_time=entry_slots[i],
            exit_time=exit_slots[i],
            volume=sizes[i],
            entry_price=entry_p[i],
            exit_price=exit_p[i],
            direction=directions[i],
            commission=commissions[i],
            pnl_net=pnl[i],
            symbol=symbols[i],
        )
        for i in range(n)
    ]

    equity = _build_equity_curve(entry_slots, exit_slots, pnl, initial_balance)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_stats(
            equity_curve=equity,
            trades=trades,
            initial_balance=initial_balance,
            freq=freq,
            min_trades=1,
            min_duration_ann=pd.Timedelta(0),
        )


def _build_equity_curve(entry_slots, exit_slots, pnl, initial_balance):
    """
    Build an equity curve from a per-trade PnL series.

    Starts at ``initial_balance`` at the first entry slot and steps by each
    trade's PnL at its exit slot.

    Args:
        entry_slots (np.ndarray): Entry timestamps (datetime64).
        exit_slots (np.ndarray): Exit timestamps (datetime64).
        pnl (np.ndarray): Per-trade net PnL in the perturbed order.
        initial_balance (float): Starting capital.

    Returns:
        pd.Series: Equity curve with a UTC DatetimeIndex, sorted ascending.
    """
    cum = initial_balance + np.cumsum(pnl)
    start = pd.to_datetime(entry_slots[:1], utc=True)
    exits = pd.to_datetime(exit_slots, utc=True)

    index = start.append(pd.DatetimeIndex(exits))
    values = np.concatenate(([float(initial_balance)], cum))

    equity = pd.Series(values, index=index)
    equity = equity[~equity.index.duplicated(keep='last')].sort_index()
    return equity

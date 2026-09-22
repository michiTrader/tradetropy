"""
Parity guard: the pandas-free compute_stats_fast (optimize/pool worker path)
must produce numeric metrics IDENTICAL to the authoritative, pandas-backed
compute_stats, so the fitness the optimizer ranks on matches a real run().
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tradetropy.stats.stats import compute_stats
from tradetropy.stats._fast import compute_stats_fast

# Numeric metric keys shared by both engines (time-typed fields checked apart).
_NUMERIC_KEYS = [
    "Exposure Time [%]",
    "Equity Final [$]",
    "Equity Peak [$]",
    "Return [%]",
    "Return (Ann.) [%]",
    "Volatility (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "# Trades",
    "# Positions",
    "# Positions Long",
    "# Positions Short",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg. Trade [%]",
    "Profit Factor",
    "Expectancy [%]",
    "SQN",
    "Total Commissions [$]",
    "_low_sample",
]

_TD_KEYS = [
    "Duration",
    "Max. Drawdown Duration",
    "Avg. Drawdown Duration",
    "Max. Trade Duration",
    "Avg. Trade Duration",
]


class _Trade:
    """Minimal Trade stand-in matching the attributes stats.py reads."""

    def __init__(self, type_, time, time_close, volume, price, price_close,
                 commission, pnl_net, symbol="SYM", ticket=0, position_id=0):
        self.type = type_
        self.time = time
        self.time_close = time_close
        self.volume = volume
        self.price = price
        self.price_close = price_close
        self.commission = commission
        self.pnl_net = pnl_net
        self.symbol = symbol
        self.ticket = ticket
        self.position_id = position_id


def _make_scenario(rng, n_points, step_ms, n_trades, start_ms=1_600_000_000_000):
    ts = start_ms + np.arange(n_points, dtype=np.float64) * step_ms
    # Random-walk equity, strictly positive.
    vals = 100_000.0 + np.cumsum(rng.normal(0, 50, size=n_points))
    vals = np.maximum(vals, 1.0)

    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    trades = []
    open_position_id = None
    for k in range(n_trades):
        i = int(rng.integers(0, n_points - 1))
        j = int(rng.integers(i, n_points))
        closed = rng.random() > 0.15  # ~15% left open
        type_ = int(rng.integers(0, 2))  # 0 long, 1 short
        volume = float(rng.uniform(0.5, 5.0))
        price = float(rng.uniform(90, 110))
        pnl = float(rng.normal(0, 200))
        ticket = k + 1
        # ~30% chance of sharing the previous trade's position_id, to exercise
        # partial-close grouping (# Positions must collapse those together).
        if open_position_id is not None and rng.random() < 0.3:
            position_id = open_position_id
        else:
            position_id = ticket
        open_position_id = position_id
        trades.append(_Trade(
            type_=type_,
            time=_dt(float(ts[i])),                       # broker Trade.time is datetime
            time_close=_dt(float(ts[j])) if closed else None,
            volume=volume,
            price=price,
            price_close=price + rng.normal(0, 1) if closed else float("nan"),
            commission=float(rng.uniform(0, 5)),
            pnl_net=pnl if closed else float("nan"),
            ticket=ticket,
            position_id=position_id,
        ))
    return ts, vals, trades


def _equity_series(ts, vals):
    idx = pd.to_datetime(ts, unit="ms", utc=True)
    return pd.Series(vals, index=idx, name="equity", dtype=float)


def _eq(a, b, key):
    """Assert numeric parity handling NaN/inf/int."""
    if isinstance(a, bool) or isinstance(b, bool):
        assert bool(a) == bool(b), f"{key}: {a} != {b}"
        return
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        assert math.isnan(fa) and math.isnan(fb), f"{key}: {a} != {b}"
        return
    if math.isinf(fa) or math.isinf(fb):
        assert fa == fb, f"{key}: {a} != {b}"
        return
    assert abs(fa - fb) <= 1e-6 + 1e-6 * abs(fb), f"{key}: {a} != {b}"


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    "n_points,step_ms,n_trades",
    [
        (500, 60_000, 0),      # 1min bars, no trades (open-position equity only)
        (500, 60_000, 8),      # 1min bars, short duration -> low sample gating
        (3000, 60_000, 40),    # ~2 days -> 5min freq
        (20000, 60_000, 60),   # ~14 days -> 1h freq, annualized reliable
        (50000, 60_000, 120),  # ~34 days -> 1D freq
    ],
)
def test_fast_matches_compute_stats(seed, n_points, step_ms, n_trades):
    rng = np.random.default_rng(seed * 100 + n_points)
    ts, vals, trades = _make_scenario(rng, n_points, step_ms, n_trades)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compute_stats(_equity_series(ts, vals), trades, initial_balance=100_000.0)
    fast = compute_stats_fast(ts, vals, trades, initial_balance=100_000.0)

    for key in _NUMERIC_KEYS:
        _eq(ref[key], fast[key], key)

    for key in _TD_KEYS:
        ref_s = pd.Timedelta(ref[key]).total_seconds()
        fast_s = fast[key].total_seconds() if isinstance(fast[key], timedelta) else float(fast[key])
        assert abs(ref_s - fast_s) <= 1e-3, f"{key}: {ref[key]} != {fast[key]}"


def test_fast_empty_equity_returns_none():
    assert compute_stats_fast(np.array([]), np.array([]), [], initial_balance=100_000.0) is None


@pytest.mark.parametrize("freq", ["1min", "5min", "1h", "1D"])
def test_fast_explicit_freq_matches(freq):
    rng = np.random.default_rng(7)
    ts, vals, trades = _make_scenario(rng, 20000, 60_000, 50)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compute_stats(_equity_series(ts, vals), trades, initial_balance=100_000.0, freq=freq)
    fast = compute_stats_fast(ts, vals, trades, initial_balance=100_000.0, freq=freq)
    for key in ("Sharpe Ratio", "Sortino Ratio", "Volatility (Ann.) [%]"):
        _eq(ref[key], fast[key], f"{freq}:{key}")

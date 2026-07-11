"""
_fast.py - pandas-free statistics for the optimize/pool worker hot path.

``compute_stats`` (stats.py) is the authoritative, pandas-backed metric engine
used by ``run()`` / ``BacktestEngine.stats``; it returns a ``Stats`` (a
``pd.Series`` subclass) whose ``Duration`` / ``Start`` / ``End`` values are
``pd.Timedelta`` / ``pd.Timestamp`` - part of the public API.

That path imports pandas, which on a spawned optimize/pool worker (Windows)
costs ~0.5-0.6 s of import time that is otherwise unnecessary: the worker only
needs the NUMERIC metric values to rank candidates and to fill
``OptimizationResult.to_dataframe()``.

``compute_stats_fast`` reproduces the SAME numeric metric values as
``compute_stats`` using only NumPy + the stdlib ``datetime``, so a worker never
imports pandas. Parity of every numeric metric is asserted against
``compute_stats`` in ``test/test_stats_fast_parity.py`` - the fitness the
optimizer ranks on is therefore identical to what a ``run()`` would report.

Representation of the time-typed fields differs by TYPE only (never by value):
``Start`` / ``End`` are timezone-aware ``datetime`` and the duration fields are
``datetime.timedelta`` instead of the pandas equivalents. Both flow through
``pandas.DataFrame`` (in ``to_dataframe``) exactly like the pandas types, so the
optimize result table is equivalent.

The resample used for the annualized metrics (Volatility/Sharpe/Sortino) is the
pandas ``equity.resample(freq).last().dropna()`` rule. For every frequency this
engine emits (1min/5min/1h/1D, all aligned to midnight UTC == the epoch for
these divisors) that is exactly integer floor-division binning on the
millisecond timestamps, taking the last value in each bin - which is what
``_resample_last_by_bin`` does.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

# Mirror the authoritative tables in stats.py (kept in sync; the parity test
# guards any divergence).
_FREQ_ANN_FACTORS: dict[str, float] = {
    "1W": 52.0,
    "1D": 252.0,
    "4h": 252.0 * 1.625,
    "2h": 252.0 * 3.25,
    "1h": 252.0 * 6.5,
    "30min": 252.0 * 13.0,
    "15min": 252.0 * 26.0,
    "5min": 252.0 * 78.0,
    "1min": 252.0 * 390.0,
    "30s": 252.0 * 780.0,
    "15s": 252.0 * 1560.0,
    "10s": 252.0 * 2340.0,
    "5s": 252.0 * 4680.0,
    "1s": 252.0 * 23400.0,
}

# Millisecond span of the fixed frequencies this fast path resamples on. Only
# the frequencies auto-selected by _resolve_freq (+ the kline-interval presets)
# are needed; all are epoch/midnight aligned so floor-division binning matches
# pandas' resample exactly.
_FREQ_MS: dict[str, int] = {
    "1s": 1_000,
    "5s": 5_000,
    "10s": 10_000,
    "15s": 15_000,
    "30s": 30_000,
    "1min": 60_000,
    "5min": 300_000,
    "15min": 900_000,
    "30min": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1D": 86_400_000,
    "1W": 604_800_000,
}

_ANN_METRICS = (
    "Return (Ann.) [%]",
    "Volatility (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
)
_TRADE_DIST_METRICS = ("Profit Factor", "SQN")

# Defaults mirror stats.MIN_TRADES_FOR_STATS / MIN_DURATION_ANN.
MIN_TRADES_FOR_STATS = 2
MIN_DURATION_ANN_MS = 7 * 86_400_000  # 7 days


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ts_to_ms(value: Any) -> float:
    """
    Normalize a trade timestamp to epoch milliseconds (float).

    Mirrors stats._to_datetime_utc's unit auto-detection: a datetime is taken
    directly; a numeric value > 1e11 is milliseconds, otherwise seconds.
    Returns NaN for None / unparseable values (an open trade has no exit).
    """
    if value is None:
        return math.nan
    if isinstance(value, datetime):
        return value.timestamp() * 1000.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(v):
        return math.nan
    return v if abs(v) > 1e11 else v * 1000.0


def _resolve_freq_ms(eq_ts_ms: np.ndarray, freq: str | None) -> str:
    """Mirror stats._resolve_freq: explicit freq wins, else auto by duration."""
    if freq is not None:
        return freq
    duration_ms = float(eq_ts_ms[-1] - eq_ts_ms[0])
    hours = duration_ms / 3_600_000.0
    if hours < 2:
        return "1min"
    if hours < 48:
        return "5min"
    if hours < 30 * 24:
        return "1h"
    return "1D"


def _resample_last_by_bin(eq_ts_ms: np.ndarray, eq_vals: np.ndarray, freq_ms: int) -> np.ndarray:
    """
    Reproduce ``equity.resample(freq).last().dropna()`` as the value at the
    largest timestamp within each floor-division bin, ordered by bin.

    Input is assumed time-ordered (the broker appends monotonically). Returns
    the resampled equity VALUES; if fewer than 2 bins result, returns the
    original values unchanged (mirrors calc_equity_resampled's guard).
    """
    bins = np.floor_divide(eq_ts_ms.astype(np.int64), freq_ms)
    # Last occurrence per bin: a change in bin marks the end of the previous
    # bin. Since input is time-ordered, the last row of each bin is its .last().
    last_mask = np.empty(len(bins), dtype=bool)
    last_mask[-1] = True
    last_mask[:-1] = bins[1:] != bins[:-1]
    resampled = eq_vals[last_mask]
    if len(resampled) < 2:
        return eq_vals
    return resampled


def _pct_change(values: np.ndarray) -> np.ndarray:
    """Reproduce Series.pct_change().dropna(): (v[i]-v[i-1])/v[i-1]."""
    if len(values) < 2:
        return np.array([], dtype=float)
    prev = values[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (values[1:] - prev) / prev


def _round4(x: Any) -> Any:
    """Mirror stats._round4 for the numeric fields (time types pass through)."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (datetime, timedelta)):
        return x
    if isinstance(x, (int, np.integer)) and np.isfinite(x):
        return int(x)
    if isinstance(x, (float, np.floating)) and np.isfinite(x):
        return float(np.round(float(x), 4))
    return x


def _drawdown_episodes(eq_ts_ms: np.ndarray, eq_vals: np.ndarray):
    """
    Reproduce stats.drawdown_episodes on raw arrays.

    Returns (depths, durations_ms) as float arrays (empty if no drawdown).
    """
    peak = np.maximum.accumulate(eq_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = eq_vals / peak - 1.0
    flags = dd < 0
    if not flags.any():
        return np.array([], dtype=float), np.array([], dtype=float)
    depths, durations = [], []
    start_i, current_min = None, 0.0
    n = len(flags)
    for i in range(n):
        if flags[i] and start_i is None:
            start_i, current_min = i, float(dd[i])
        elif flags[i]:
            current_min = min(current_min, float(dd[i]))
        elif start_i is not None:
            depths.append(current_min)
            durations.append(float(eq_ts_ms[i] - eq_ts_ms[start_i]))
            start_i, current_min = None, 0.0
    if start_i is not None:
        depths.append(current_min)
        durations.append(float(eq_ts_ms[-1] - eq_ts_ms[start_i]))
    return np.asarray(depths, dtype=float), np.asarray(durations, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# compute_stats_fast
# ══════════════════════════════════════════════════════════════════════════════


def compute_stats_fast(
    eq_ts_ms,
    eq_vals,
    trades: list,
    initial_balance: float,
    freq: str | None = None,
    *,
    min_trades: int = MIN_TRADES_FOR_STATS,
    min_duration_ann_ms: int = MIN_DURATION_ANN_MS,
) -> "OrderedDict | None":
    """
    Pandas-free equivalent of stats.compute_stats for worker use.

    Args:
        eq_ts_ms: equity-curve timestamps in epoch ms (broker._eq_ts).
        eq_vals:  equity-curve values (broker._eq_vals).
        trades:   list of Trade objects (broker.get_trades()).
        initial_balance: starting capital.
        freq: resampling frequency; None auto-detects (mirrors _resolve_freq).

    Returns:
        OrderedDict of metrics with numeric values IDENTICAL to compute_stats
        (time fields as datetime/timedelta), or None if the equity curve is
        empty (mirrors _build_stats gating on an empty curve).
    """
    ib = float(initial_balance)
    if not math.isfinite(ib) or ib < 0:
        raise ValueError(f"initial_balance must be a finite number >= 0, got {ib}")

    eq_ts = np.asarray(eq_ts_ms, dtype=np.float64)
    eq = np.asarray(eq_vals, dtype=np.float64)
    if eq.size == 0:
        return None

    resolved_freq = _resolve_freq_ms(eq_ts, freq)
    ann = _FREQ_ANN_FACTORS.get(resolved_freq, 252.0)
    freq_ms = _FREQ_MS.get(resolved_freq)
    if freq_ms is None:
        # Unknown freq: fall back to treating each recorded point as a period
        # (no resample). This matches compute_stats only for known freqs; the
        # engine only ever emits known freqs, and the parity test enforces it.
        resampled = eq
    else:
        resampled = _resample_last_by_bin(eq_ts, eq, freq_ms)
    returns = _pct_change(resampled)

    # ── Trades to arrays ──────────────────────────────────────────────────────
    entry_ms, exit_ms, size, entry_px, pnl_net, direction, commission = _trade_arrays(trades)
    closed = ~np.isnan(exit_ms)
    n_closed = int(closed.sum())

    start_ms = float(eq_ts[0])
    end_ms = float(eq_ts[-1])
    duration_ms = end_ms - start_ms

    equity_final = float(eq[-1])
    equity_peak = float(np.max(eq))
    return_pct = _return_pct(equity_final, ib)
    return_ann = _return_ann_pct(equity_final, ib, duration_ms)
    max_dd = _max_drawdown_pct(eq)

    depths, dd_durs = _drawdown_episodes(eq_ts, eq)

    # Closed-trade slices (order matches compute_stats' sort by entry then exit).
    order = _closed_order(entry_ms, exit_ms, closed)
    c_pnl = pnl_net[order]
    c_ret = _trade_return_pct(pnl_net[order], size[order], entry_px[order])
    c_dir = direction[order]
    c_dur_ms = exit_ms[order] - entry_ms[order]

    longs = int(np.sum(c_dir == 1)) if n_closed else 0
    shorts = int(np.sum(c_dir == -1)) if n_closed else 0

    metrics: "OrderedDict[str, Any]" = OrderedDict([
        ("Start", _to_dt(start_ms)),
        ("End", _to_dt(end_ms)),
        ("Duration", _to_td(duration_ms)),
        ("Exposure Time [%]", _exposure_time_pct(
            eq_ts, entry_ms, exit_ms, size, direction, closed, start_ms, end_ms)),
        ("Equity Final [$]", equity_final),
        ("Equity Peak [$]", equity_peak),
        ("Return [%]", return_pct),
        ("Return (Ann.) [%]", return_ann),
        ("Volatility (Ann.) [%]", _volatility_ann_pct(returns, ann)),
        ("Sharpe Ratio", _sharpe_ratio(returns, ann)),
        ("Sortino Ratio", _sortino_ratio(returns, ann)),
        ("Calmar Ratio", _calmar_ratio(return_ann, max_dd)),
        ("Max. Drawdown [%]", max_dd),
        ("Avg. Drawdown [%]", float(depths.mean() * 100.0) if depths.size else 0.0),
        ("Max. Drawdown Duration", _to_td(float(dd_durs.max())) if dd_durs.size else timedelta(0)),
        ("Avg. Drawdown Duration", _to_td(float(dd_durs.mean())) if dd_durs.size else timedelta(0)),
        ("# Trades", n_closed),
        ("# Trades Long", longs),
        ("# Trades Short", shorts),
        ("Win Rate [%]", float((c_pnl > 0).mean() * 100.0) if n_closed else 0.0),
        ("Best Trade [%]", float(np.nanmax(c_ret)) if c_ret.size else 0.0),
        ("Worst Trade [%]", float(np.nanmin(c_ret)) if c_ret.size else 0.0),
        ("Avg. Trade [%]", float(np.nanmean(c_ret)) if c_ret.size else 0.0),
        ("Max. Trade Duration", _to_td(float(c_dur_ms.max())) if n_closed else timedelta(0)),
        ("Avg. Trade Duration", _to_td(float(c_dur_ms.mean())) if n_closed else timedelta(0)),
        ("Profit Factor", _profit_factor(c_pnl)),
        ("Expectancy [%]", _expectancy_pct(c_ret)),
        ("SQN", _sqn(c_ret)),
        ("Total Commissions [$]", float(np.nansum(commission))),
        ("_low_sample", False),
    ])

    ann_unreliable = duration_ms < min_duration_ann_ms
    trades_unreliable = n_closed < min_trades
    if ann_unreliable:
        for k in _ANN_METRICS:
            metrics[k] = np.nan
    if trades_unreliable:
        for k in _TRADE_DIST_METRICS:
            metrics[k] = np.nan
    if ann_unreliable or trades_unreliable:
        metrics["_low_sample"] = True

    for k in metrics:
        metrics[k] = _round4(metrics[k])
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Numeric metric helpers (mirror stats.py exactly)
# ══════════════════════════════════════════════════════════════════════════════


def _to_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _to_td(ms: float) -> timedelta:
    return timedelta(milliseconds=ms)


def _return_pct(equity_final: float, ib: float) -> float:
    if ib == 0:
        return math.inf if equity_final > 0 else math.nan
    return (equity_final / ib - 1.0) * 100.0


def _return_ann_pct(equity_final: float, ib: float, duration_ms: float) -> float:
    total_seconds = duration_ms / 1000.0
    if total_seconds <= 0:
        return math.nan
    seconds_per_year = 365.25 * 24 * 3600
    years = max(total_seconds / seconds_per_year, 1.0 / 365.25)
    if ib == 0:
        return math.inf if equity_final > 0 else math.nan
    ratio = equity_final / ib
    if ratio <= 0:
        return math.nan
    return (ratio ** (1.0 / years) - 1.0) * 100.0


def _max_drawdown_pct(eq: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = eq / peak - 1.0
    return float(np.min(dd) * 100.0)


def _volatility_ann_pct(returns: np.ndarray, ann: float) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.std(returns, ddof=0) * np.sqrt(ann) * 100.0)


def _sharpe_ratio(returns: np.ndarray, ann: float) -> float:
    if returns.size == 0:
        return math.nan
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=0))
    if sd == 0:
        return math.inf if mu > 0 else (-math.inf if mu < 0 else math.nan)
    return float(mu / sd * math.sqrt(ann))


def _sortino_ratio(returns: np.ndarray, ann: float) -> float:
    if returns.size == 0:
        return math.nan
    mu = float(np.mean(returns))
    downside = np.where(returns < 0, returns, 0.0)
    dd = float(np.std(downside, ddof=0))
    if dd == 0:
        return math.inf if mu > 0 else (-math.inf if mu < 0 else math.nan)
    return float(mu / dd * math.sqrt(ann))


def _calmar_ratio(return_ann_pct: float, max_drawdown_pct: float) -> float:
    if not np.isfinite(return_ann_pct):
        return math.nan
    if max_drawdown_pct == 0:
        return math.inf if return_ann_pct > 0 else math.nan
    return float(return_ann_pct / abs(max_drawdown_pct))


def _trade_return_pct(pnl: np.ndarray, size: np.ndarray, entry_px: np.ndarray) -> np.ndarray:
    if pnl.size == 0:
        return np.array([], dtype=float)
    notional = size * entry_px
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(notional > 0, pnl / notional * 100.0, np.nan)


def _profit_factor(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return math.nan
    gains = float(pnl[pnl > 0].sum())
    losses = float(pnl[pnl < 0].sum())
    if losses == 0:
        return math.inf if gains > 0 else math.nan
    return gains / abs(losses)


def _expectancy_pct(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    wins = r[r > 0]
    losses = r[r <= 0]
    p = float((r > 0).mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    return p * avg_win + (1.0 - p) * avg_loss


def _sqn(r: np.ndarray) -> float:
    n = int(r.size)
    if n == 0:
        return math.nan
    exp = _expectancy_pct(r)
    sd = float(np.std(r, ddof=0))
    if sd == 0:
        return math.inf if exp > 0 else (-math.inf if exp < 0 else math.nan)
    return float(math.sqrt(n) * (exp / sd))


def _exposure_time_pct(eq_ts, entry_ms, exit_ms, size, direction, closed, start_ms, end_ms):
    """Mirror stats.calc_exposure_time_pct on raw ms arrays."""
    n_idx = len(eq_ts)
    if n_idx < 2:
        return 0.0
    total = (end_ms - start_ms) / 1000.0
    if total <= 0:
        return 0.0
    if entry_ms.size == 0:
        return 0.0
    w = direction.astype(float) * size
    entry_idx = np.searchsorted(eq_ts, entry_ms, side="left")
    exit_idx = np.full(entry_ms.size, n_idx, dtype=int)
    exit_idx[closed] = np.searchsorted(eq_ts, exit_ms[closed], side="left")
    a_diff = np.zeros(n_idx + 1, dtype=float)
    np.add.at(a_diff, entry_idx, w)
    np.add.at(a_diff, exit_idx, -w)
    a = np.cumsum(a_diff[:-1])
    in_pos = a != 0.0
    deltas = (eq_ts[1:] - eq_ts[:-1]) / 1000.0
    exposed = float(np.sum(deltas * in_pos[:-1]))
    return exposed / total * 100.0


def _trade_arrays(trades: list):
    """Extract trade fields into parallel numpy arrays (pandas-free)."""
    n = len(trades)
    entry_ms = np.full(n, np.nan)
    exit_ms = np.full(n, np.nan)
    size = np.zeros(n)
    entry_px = np.zeros(n)
    pnl_net = np.full(n, np.nan)
    commission = np.zeros(n)
    direction = np.zeros(n, dtype=int)  # 1 long, -1 short
    for i, t in enumerate(trades):
        raw_type = getattr(t, "type", None)
        if isinstance(raw_type, int) or hasattr(raw_type, "value"):
            direction[i] = 1 if int(raw_type) == 0 else -1
        else:
            direction[i] = 1 if "buy" in str(raw_type).lower() else -1
        entry_ms[i] = _ts_to_ms(getattr(t, "time", None))
        exit_ms[i] = _ts_to_ms(getattr(t, "time_close", None))
        size[i] = float(getattr(t, "volume", 0.0))
        entry_px[i] = float(getattr(t, "price", 0.0))
        pnl_net[i] = float(getattr(t, "pnl_net", np.nan))
        commission[i] = float(getattr(t, "commission", 0.0))
    return entry_ms, exit_ms, size, entry_px, pnl_net, direction, commission


def _closed_order(entry_ms, exit_ms, closed) -> np.ndarray:
    """
    Indices of CLOSED trades sorted by (entry_time, exit_time), matching
    compute_stats' ``sort_values(["entry_time","exit_time"])`` then
    ``closed_trades`` filter. Stable to mirror pandas' mergesort default use.
    """
    idx = np.nonzero(closed)[0]
    if idx.size == 0:
        return idx
    order = np.lexsort((exit_ms[idx], entry_ms[idx]))
    return idx[order]

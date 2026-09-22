"""
tradetropy.stats.stats - backwards-compatible facade.

Historically every stats symbol lived in this single ~1550-line module. It is
now split by responsibility:

    _freq.py     - frequency resolution & annualization factors
    _prepare.py  - input validation & trades-list normalization
    metrics.py   - pure calculation functions (ratios, drawdown, per-trade)
    _compute.py  - compute_stats() + statistical reliability gating
    _result.py   - Stats (presentation-only pd.Series subclass)

This module re-exports every public and previously-public name so existing
code importing ``tradetropy.stats.stats`` (directly, bypassing the package's
lazy ``__init__``) keeps working unchanged.
"""

from __future__ import annotations

# ── Frequency resolution ─────────────────────────────────────────────────────
from tradetropy.stats._freq import (
    _FREQ_ANN_FACTORS,
    _ann_factor,
    _resolve_freq,
    _ts_precision,
)

# ── Input validation / normalization ─────────────────────────────────────────
from tradetropy.stats._prepare import (
    _empty_trades_df,
    _round4,
    _to_datetime_utc,
    _trades_list_to_df,
    _validate_equity_curve,
    _validate_initial_balance,
)

# ── Pure calculation functions ───────────────────────────────────────────────
from tradetropy.stats.metrics import (
    _position_group_key,
    build_positions_df,
    calc_avg_drawdown_duration,
    calc_avg_drawdown_pct,
    calc_avg_trade_duration,
    calc_avg_trade_pct,
    calc_best_trade_pct,
    calc_calmar_ratio,
    calc_daily_equity,
    calc_daily_returns,
    calc_equity_resampled,
    calc_exposure_time_pct,
    calc_expectancy_pct,
    calc_max_drawdown_duration,
    calc_max_drawdown_pct,
    calc_max_trade_duration,
    calc_num_positions,
    calc_num_trades,
    calc_profit_factor,
    calc_return_ann_pct,
    calc_return_pct,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_sqn,
    calc_volatility_ann_pct,
    calc_win_rate_pct,
    calc_worst_trade_pct,
    closed_trades,
    count_positions_by_direction,
    count_trades_by_direction,
    downside_std_dev,
    drawdown_episodes,
    drawdown_series,
    mean,
    std_dev,
    trade_durations,
    trade_durations_days,
    trade_pnl_net,
    trade_return_pct,
)

# ── compute_stats() + reliability thresholds ─────────────────────────────────
from tradetropy.stats._compute import (
    MIN_DURATION_ANN,
    MIN_TRADES_FOR_STATS,
    _ANN_METRICS,
    _TRADE_DIST_METRICS,
    compute_stats,
)

# ── Stats (presentation-only result container) ──────────────────────────────
from tradetropy.stats._result import Stats

__all__ = [
    "Stats",
    "compute_stats",
    "drawdown_series",
    "calc_daily_equity",
    "MIN_TRADES_FOR_STATS",
    "MIN_DURATION_ANN",
]

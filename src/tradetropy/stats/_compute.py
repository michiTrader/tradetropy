"""
tradetropy.stats._compute - compute_stats() and statistical reliability gating.

With very few trades or too short a duration, annualized metrics
(Return Ann., Volatility, Sharpe, Sortino, Calmar) and those based on the
trade distribution (SQN, Profit Factor) become uninterpretable and
can produce astronomical numbers or inf. Below these thresholds those
metrics are reported as NaN and Stats["_low_sample"] = True is set.

The user can override the thresholds per call in compute_stats().
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from tradetropy.stats._freq import _ann_factor, _resolve_freq
from tradetropy.stats._prepare import (
    _round4,
    _trades_list_to_df,
    _validate_equity_curve,
    _validate_initial_balance,
)
from tradetropy.stats.metrics import (
    build_positions_df,
    calc_avg_drawdown_duration,
    calc_avg_drawdown_pct,
    calc_avg_trade_duration,
    calc_avg_trade_pct,
    calc_best_trade_pct,
    calc_calmar_ratio,
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
)

MIN_TRADES_FOR_STATS: int = 2
MIN_DURATION_ANN: pd.Timedelta = pd.Timedelta(days=7)

# Metrics that are zeroed when duration is too short to annualize.
_ANN_METRICS = (
    "Return (Ann.) [%]",
    "Volatility (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
)

# Metrics that are zeroed when there are too few closed trades.
_TRADE_DIST_METRICS = (
    "Profit Factor",
    "SQN",
)


def compute_stats(
    equity_curve: pd.Series,
    trades: list,
    initial_balance: float,
    strategy=None,
    freq: str | None = None,
    *,
    min_trades: int = MIN_TRADES_FOR_STATS,
    min_duration_ann: pd.Timedelta = MIN_DURATION_ANN,
    warn: bool = True,
) -> 'Stats':
    """
    Calculate all performance metrics and return Stats object.

    This is the main entry point for computing backtest statistics. It handles
    frequency auto-detection, metric calculation, and reliability gating.

    Args:
        equity_curve (pd.Series): Broker equity curve with DatetimeIndex (UTC).
                                 Use broker.equity_curve directly.
        trades (list): List of Trade objects from broker.get_trades().
        initial_balance (float): Starting account capital.
        strategy (optional): Strategy instance (used in __repr__).
        freq (str | None): Resampling frequency for returns/volatility/ratios.
                          If None, auto-detects based on backtest duration:
                            - < 2h    -> '1min'
                            - < 2d    -> '5min'
                            - < 30d   -> '1h'
                            - >= 30d  -> '1D'
                          Use explicit values for consistency: '1D', '1h',
                          '5min', '1min'.
        warn (bool): Emit a UserWarning when the sample is too small to trust
                    the annualized/trade-distribution metrics (see "Statistical
                    reliability gating" below). The gating itself (zeroing
                    those metrics to NaN and setting Stats["_low_sample"]) always
                    happens regardless of this flag - only the warning message
                    is optional. Quick exploratory runs on a small dataset can
                    set this to False to silence the notice, mirroring
                    backtesting.py's quieter behavior (which reports 0/NaN with
                    no explicit warning in that case).

    Returns:
        Stats: Object containing all metrics accessible via dict interface.

    Raises:
        DataError: If equity_curve or trades are invalid.
        ConfigError: If initial_balance is invalid.

    Important - Drawdown calculation:
    All drawdown metrics (Max DD, Avg DD, durations) are calculated on the
    original unsampled equity curve. This captures the true worst moment
    regardless of backtest frequency.

    Important - Statistical reliability (low_sample):
    Annualizing a few-minute session or calculating ratios with one trade
    produces mathematically correct but uninterpretable results (e.g. infinity).
    To avoid misleading output:

      - Annualized metrics (Return Ann., Volatility Ann., Sharpe, Sortino,
        Calmar) report as NaN if duration < min_duration_ann (default 7 days).
      - Trade distribution metrics (Profit Factor, SQN) report as NaN if
        closed trades < min_trades (default 2).
      - Non-annualized metrics (Return %, Win Rate, Expectancy) always report
        valid values for any sample size.
      - When metrics are gated, Stats['_low_sample'] = True and a warning
        is emitted.

    To recover old behavior without gating, pass min_duration_ann=pd.Timedelta(0).

    Example:
        stats = compute_stats(
            equity_curve=broker.equity_curve,
            trades=broker.get_trades(),
            initial_balance=broker.initial_balance,
            freq='1h'
        )
        print(stats['Sharpe Ratio'])
        print(stats['# Trades'])
    """
    # Lazy import: Stats() can itself call compute_stats() (legacy direct
    # construction API), so importing it at module scope would create a
    # circular import between this module and _result.py.
    from tradetropy.stats._result import Stats

    ib = _validate_initial_balance(initial_balance)
    eq = _validate_equity_curve(equity_curve)
    df = _trades_list_to_df(trades)

    # ── Frequency resolution ─────────────────────────────────────────────────
    resolved_freq = _resolve_freq(eq, freq)
    ann = _ann_factor(resolved_freq)

    # Resampled equity for returns and ratios
    # Drawdown and duration use the ORIGINAL equity (no information loss)
    eq_resampled = calc_equity_resampled(eq, resolved_freq)
    returns = calc_daily_returns(eq_resampled)

    tc = closed_trades(df)

    start = eq.index[0]
    end   = eq.index[-1]
    duration = end - start  # pd.Timedelta — "0 days 00:12:46"

    equity_final = float(eq.iloc[-1])
    equity_peak  = float(eq.max())
    return_ann   = calc_return_ann_pct(equity_final, initial_balance=ib, duration=duration)
    max_dd       = calc_max_drawdown_pct(eq)

    metrics = OrderedDict([
        # Period
        ("Start",                 start),
        ("End",                   end),
        ("Duration",              duration),
        ("Exposure Time [%]",     calc_exposure_time_pct(eq.index, df, start=start, end=end)),

        # Equity
        ("Equity Final [$]",      equity_final),
        ("Equity Peak [$]",       equity_peak),
        ("Return [%]",            calc_return_pct(equity_final, initial_balance=ib)),
        ("Return (Ann.) [%]",     return_ann),
        ("Volatility (Ann.) [%]", calc_volatility_ann_pct(returns, ann)),

        # Risk ratios
        ("Sharpe Ratio",          calc_sharpe_ratio(returns, ann)),
        ("Sortino Ratio",         calc_sortino_ratio(returns, ann)),
        ("Calmar Ratio",          calc_calmar_ratio(return_ann, max_drawdown_pct=max_dd)),

        # Drawdown - on original equity, not resampled
        ("Max. Drawdown [%]",      max_dd),
        ("Avg. Drawdown [%]",      calc_avg_drawdown_pct(eq)),
        ("Max. Drawdown Duration", calc_max_drawdown_duration(eq)),
        ("Avg. Drawdown Duration", calc_avg_drawdown_duration(eq)),

        # Trades
        ("# Trades",              calc_num_trades(df)),
        ("# Positions",           calc_num_positions(tc)),
        ("# Positions Long",      count_positions_by_direction(tc)[0]),
        ("# Positions Short",     count_positions_by_direction(tc)[1]),
        ("Win Rate [%]",          calc_win_rate_pct(tc)),
        ("Best Trade [%]",        calc_best_trade_pct(tc)),
        ("Worst Trade [%]",       calc_worst_trade_pct(tc)),
        ("Avg. Trade [%]",        calc_avg_trade_pct(tc)),
        ("Max. Trade Duration",   calc_max_trade_duration(tc)),
        ("Avg. Trade Duration",   calc_avg_trade_duration(tc)),
        ("Profit Factor",         calc_profit_factor(tc)),
        ("Expectancy [%]",        calc_expectancy_pct(tc)),
        ("SQN",                   calc_sqn(tc)),
        ("Total Commissions [$]", df["commission"].sum()),

        # Internal context (prefix _ = not shown in public repr)
        ("_low_sample",       False),
        ("_trades",           df),
        ("_positions",        build_positions_df(tc)),
        ("_equity_curve",     eq),
        ("_initial_balance",  ib),
        ("_strategy",         strategy),
        ("_freq",             resolved_freq),
        ("_ann_factor",       ann),
    ])

    # ── Statistical reliability gating ───────────────────────────────────────
    # Zeros uninterpretable metrics with tiny samples (see docstring).
    n_closed = int(metrics["# Trades"])
    ann_unreliable    = duration < min_duration_ann
    trades_unreliable = n_closed < min_trades

    if ann_unreliable:
        for k in _ANN_METRICS:
            metrics[k] = np.nan
    if trades_unreliable:
        for k in _TRADE_DIST_METRICS:
            metrics[k] = np.nan

    if ann_unreliable or trades_unreliable:
        metrics["_low_sample"] = True
        reasons = []
        if ann_unreliable:
            reasons.append(
                f"duration {duration} < {min_duration_ann} "
                f"(zeros {', '.join(_ANN_METRICS)})"
            )
        if trades_unreliable:
            reasons.append(
                f"{n_closed} closed trade(s) < {min_trades} "
                f"(zeros {', '.join(_TRADE_DIST_METRICS)})"
            )
        if warn:
            warnings.warn(
                "Stats: insufficient sample, metrics zeroed to NaN -- "
                + "; ".join(reasons),
                stacklevel=2,
            )

    for k in metrics:
        metrics[k] = _round4(metrics[k])

    return Stats(metrics)

"""
tradetropy.stats.metrics - pure calculation functions.

Every performance metric (ratios, drawdown, per-trade statistics) lives here
as a standalone, side-effect-free function operating on plain arrays/Series/
DataFrames. ``compute_stats()`` (in ``_compute.py``) is the only caller that
assembles these into the final ``Stats`` object; that separation keeps every
metric independently testable and reusable (e.g. by ``stats._fast`` parity
tests or by robustness/optimize code that only needs one or two numbers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradetropy.stats._freq import _ann_factor  # noqa: F401  (re-exported)


# ══════════════════════════════════════════════════════════════════════════════
# Basic statistics
# ══════════════════════════════════════════════════════════════════════════════


def mean(x: np.ndarray) -> float:
    """
    Calculate arithmetic mean.

    Args:
        x (np.ndarray): Array of values

    Returns:
        float: Mean value or NaN if empty
    """
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else np.nan


def std_dev(x: np.ndarray, *, ddof: int = 0) -> float:
    """
    Calculate standard deviation.

    Args:
        x (np.ndarray): Array of values
        ddof (int): Delta degrees of freedom for unbiased estimator

    Returns:
        float: Standard deviation or NaN if empty
    """
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=ddof)) if x.size else np.nan


def downside_std_dev(x: np.ndarray, *, ddof: int = 0) -> float:
    """
    Calculate standard deviation of negative returns only.

    Used for Sortino ratio calculation.

    Args:
        x (np.ndarray): Array of returns
        ddof (int): Delta degrees of freedom for unbiased estimator

    Returns:
        float: Downside deviation or NaN if empty
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    x = np.where(x < 0, x, 0.0)
    return float(np.std(x, ddof=ddof))


# ══════════════════════════════════════════════════════════════════════════════
# Equity & drawdown
# ══════════════════════════════════════════════════════════════════════════════


def calc_equity_resampled(equity_curve: pd.Series, freq: str) -> pd.Series:
    """
    Resample equity curve to specified frequency.

    If resampling results in fewer than 2 points, returns the original equity
    curve unmodified. This ensures downstream calculations always have
    sufficient data.

    Args:
        equity_curve (pd.Series): Original equity curve with DatetimeIndex
        freq (str): Target frequency (e.g. '1D', '1h', '5min')

    Returns:
        pd.Series: Resampled equity curve, or original if too few points
    """
    resampled = equity_curve.resample(freq).last().dropna()
    if len(resampled) < 2:
        return equity_curve
    return resampled


def calc_daily_equity(equity_curve: pd.Series) -> pd.Series:
    """
    Resample equity curve to daily frequency.

    Compatibility alias for calc_equity_resampled(). Use the latter in
    new code.

    Args:
        equity_curve (pd.Series): Equity curve with DatetimeIndex

    Returns:
        pd.Series: Daily-resampled equity curve
    """
    return calc_equity_resampled(equity_curve, "1D")


def calc_daily_returns(resampled_equity: pd.Series) -> pd.Series:
    """
    Calculate period-to-period returns from resampled equity.

    Args:
        resampled_equity (pd.Series): Resampled equity curve

    Returns:
        pd.Series: Percentage change between consecutive periods
    """
    return resampled_equity.pct_change().dropna()


def calc_volatility_ann_pct(returns: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Calculate annualized volatility in percentage.

    Args:
        returns (pd.Series): Period returns
        ann_factor (float): Annualization factor matching return frequency.
                           Use _ann_factor(freq) for correct value

    Returns:
        float: Annualized volatility percentage (0.0 if empty)
    """
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(ann_factor) * 100.0)


def calc_sharpe_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Calculate annualized Sharpe ratio (risk-free rate = 0).

    Args:
        returns (pd.Series): Period returns
        ann_factor (float): Annualization factor matching return frequency.
                           Use _ann_factor(freq) for correct value

    Returns:
        float: Sharpe ratio or NaN if returns are empty
    """
    if returns.empty:
        return np.nan
    r = returns.to_numpy(dtype=float)
    mu = mean(r)
    sd = std_dev(r, ddof=0)
    if sd == 0:
        return np.inf if mu > 0 else (-np.inf if mu < 0 else np.nan)
    return float(mu / sd * np.sqrt(ann_factor))


def calc_sortino_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Calculate annualized Sortino ratio.

    Uses downside deviation (negative returns only) as denominator.

    Args:
        returns (pd.Series): Period returns
        ann_factor (float): Annualization factor matching return frequency.
                           Use _ann_factor(freq) for correct value

    Returns:
        float: Sortino ratio or NaN if returns are empty
    """
    if returns.empty:
        return np.nan
    r = returns.to_numpy(dtype=float)
    mu = mean(r)
    dd = downside_std_dev(r, ddof=0)
    if dd == 0:
        return np.inf if mu > 0 else (-np.inf if mu < 0 else np.nan)
    return float(mu / dd * np.sqrt(ann_factor))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Calculate drawdown for each point in equity curve.

    Operates directly on the given curve without resampling. Pass original
    (unsampled) equity to capture actual maximum drawdown.

    Args:
        equity (pd.Series): Equity curve with DatetimeIndex

    Returns:
        pd.Series: Drawdown ratio (negative or 0.0)
    """
    peak = equity.cummax()
    return equity / peak - 1.0


def drawdown_episodes(equity: pd.Series) -> pd.DataFrame:
    """
    Identify and measure individual drawdown episodes.

    Args:
        equity (pd.Series): Equity curve with DatetimeIndex (unsampled)

    Returns:
        pd.DataFrame: Episodes with columns (depth, duration). Empty if no
                     drawdowns occurred

    Example:
        dd = drawdown_episodes(equity)
        max_duration = dd['duration'].max()
    """
    dd = drawdown_series(equity)
    in_dd = dd < 0
    if not in_dd.any():
        return pd.DataFrame(columns=["depth", "duration"])

    # Iterate positionally (not by label) so a non-unique index - e.g. the
    # equity curve of a multi-symbol backtest, where several symbols advance on
    # the same bar timestamp - does not turn dd.loc[t] into a Series.
    dd_vals = dd.to_numpy(dtype=float)
    idx = dd.index
    flags = in_dd.to_numpy()
    eps, start_i, current_min = [], None, 0.0
    for i in range(len(flags)):
        if flags[i] and start_i is None:
            start_i, current_min = i, float(dd_vals[i])
        elif flags[i]:
            current_min = min(current_min, float(dd_vals[i]))
        elif start_i is not None:
            eps.append({"depth": current_min, "duration": idx[i] - idx[start_i]})
            start_i, current_min = None, 0.0
    if start_i is not None:
        eps.append({
            "depth":    current_min,
            "duration": idx[-1] - idx[start_i],
        })
    return pd.DataFrame(eps)


def calc_max_drawdown_pct(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown in percentage.

    Uses unsampled equity curve to capture worst peak-to-trough regardless
    of backtest frequency.

    Args:
        equity (pd.Series): Equity curve with DatetimeIndex

    Returns:
        float: Maximum drawdown percentage (0.0 if no losses)
    """
    return (
        float(drawdown_series(equity).min() * 100.0)
        if not equity.empty
        else 0.0
    )


def calc_avg_drawdown_pct(equity: pd.Series) -> float:
    """
    Calculate average drawdown in percentage.

    Args:
        equity (pd.Series): Equity curve with DatetimeIndex (unsampled)

    Returns:
        float: Average drawdown percentage (0.0 if no drawdowns)
    """
    eps = drawdown_episodes(equity)
    return float(eps["depth"].mean() * 100.0) if not eps.empty else 0.0


def calc_max_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """
    Calculate maximum drawdown duration.

    Args:
        equity (pd.Series): Equity curve with DatetimeIndex

    Returns:
        pd.Timedelta: Maximum episode duration (0 if no drawdowns)

    Example:
        max_dd_duration = calc_max_drawdown_duration(equity)
    """
    eps = drawdown_episodes(equity)
    if eps.empty:
        return pd.Timedelta(0)
    return pd.Timedelta(eps["duration"].max())


def calc_avg_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """
    Calculate average drawdown duration.

    Args:
        equity (pd.Series): Equity curve with DatetimeIndex

    Returns:
        pd.Timedelta: Mean episode duration (0 if no drawdowns)
    """
    eps = drawdown_episodes(equity)
    if eps.empty:
        return pd.Timedelta(0)
    return pd.Timedelta(eps["duration"].mean())


def calc_calmar_ratio(return_ann_pct: float, *, max_drawdown_pct: float) -> float:
    """
    Calculate Calmar ratio (annual return / maximum drawdown).

    Args:
        return_ann_pct (float): Annualized return percentage
        max_drawdown_pct (float): Maximum drawdown percentage (absolute)

    Returns:
        float: Calmar ratio or NaN if return is invalid
    """
    if not np.isfinite(return_ann_pct):
        return np.nan
    if max_drawdown_pct == 0:
        return np.inf if return_ann_pct > 0 else np.nan
    return float(return_ann_pct / abs(max_drawdown_pct))


# ══════════════════════════════════════════════════════════════════════════════
# Account return
# ══════════════════════════════════════════════════════════════════════════════

def calc_return_pct(equity_final: float, *, initial_balance: float) -> float:
    """
    Calculate total return in percentage.

    Args:
        equity_final (float): Final equity value
        initial_balance (float): Starting capital

    Returns:
        float: Total return percentage
    """
    if initial_balance == 0:
        return np.inf if equity_final > 0 else np.nan
    return (equity_final / float(initial_balance) - 1.0) * 100.0


def calc_return_ann_pct(
    equity_final: float,
    *,
    initial_balance: float,
    duration: pd.Timedelta,
) -> float:
    """
    Calculate annualized return in percentage.

    Uses pd.Timedelta for higher precision with intraday backtests. Never
    returns NaN for short durations - uses 1 day minimum to prevent extreme
    exponents. Mathematically, annualizing a 12-minute session produces very
    large numbers - that is correct. Interpret results in context.

    Args:
        equity_final (float): Final equity value
        initial_balance (float): Starting capital
        duration (pd.Timedelta): Total backtest duration

    Returns:
        float: Annualized return percentage
    """
    total_seconds = duration.total_seconds()
    if total_seconds <= 0:
        return np.nan
    seconds_per_year = 365.25 * 24 * 3600
    # Minimum 1 day to avoid extreme exponent powers
    years = max(total_seconds / seconds_per_year, 1.0 / 365.25)
    if initial_balance == 0:
        return np.inf if equity_final > 0 else np.nan
    ratio = equity_final / float(initial_balance)
    if ratio <= 0:
        return np.nan
    return (ratio ** (1.0 / years) - 1.0) * 100.0


def calc_exposure_time_pct(
    equity_index: pd.DatetimeIndex,
    trades: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """
    Calculate percentage of time with open positions.

    Args:
        equity_index (pd.DatetimeIndex): Timestamps of equity curve
        trades (pd.DataFrame): Trades DataFrame with direction, size, entry/exit times
        start (pd.Timestamp): Backtest start time
        end (pd.Timestamp): Backtest end time

    Returns:
        float: Exposure percentage (0.0 if no trades)
    """
    if len(equity_index) < 2:
        return 0.0
    total = (end - start).total_seconds()
    if total <= 0:
        return 0.0

    sign = np.where(trades["direction"].to_numpy() == "long", 1.0, -1.0)
    size = trades["size"].to_numpy(dtype=float)
    w = sign * size
    is_closed = trades["exit_time"].notna().to_numpy()
    entry_idx = equity_index.searchsorted(trades["entry_time"].to_numpy(), side="left")
    exit_idx = np.full(len(trades), len(equity_index), dtype=int)
    exit_idx[is_closed] = equity_index.searchsorted(
        trades["exit_time"][is_closed].to_numpy(), side="left"
    )

    a_diff = np.zeros(len(equity_index) + 1, dtype=float)
    a_diff[entry_idx] += w
    a_diff[exit_idx] -= w
    a = np.cumsum(a_diff[:-1])
    in_pos = a != 0.0
    deltas = (
        (equity_index[1:] - equity_index[:-1])
        .to_numpy(dtype="timedelta64[ns]")
        .astype("timedelta64[s]")
        .astype(float)
    )
    exposed = float(np.sum(deltas * in_pos[:-1]))
    return exposed / total * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Per-trade metrics
# ══════════════════════════════════════════════════════════════════════════════

def closed_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to closed trades only.

    Args:
        trades (pd.DataFrame): All trades DataFrame

    Returns:
        pd.DataFrame: Subset of trades with exit_time (closed trades)
    """
    return trades.loc[trades['exit_time'].notna()].copy()


def calc_num_trades(trades: pd.DataFrame) -> int:
    """
    Count closed trades.

    Args:
        trades (pd.DataFrame): All trades DataFrame

    Returns:
        int: Number of trades with exit_time
    """
    return int(trades['exit_time'].notna().sum())


def count_trades_by_direction(trades_closed: pd.DataFrame) -> tuple[int, int]:
    """
    Count trades by direction.

    Args:
        trades_closed (pd.DataFrame): Closed trades only

    Returns:
        tuple: (long_count, short_count)
    """
    if trades_closed.empty:
        return 0, 0
    longs = int((trades_closed['direction'] == 'long').sum())
    shorts = int((trades_closed['direction'] == 'short').sum())
    return longs, shorts


def _position_group_key(trades_closed: pd.DataFrame) -> pd.Series:
    """
    Resolve the grouping key used to aggregate closed trades into positions.

    A single position can be closed across several ``Trade`` rows (partial
    closes), all sharing the same ``position_id``. Some connectors (CCXT,
    Bybit) do not track a real position id and report ``0`` for every trade;
    in that case grouping by the raw value would collapse ALL trades into one
    fake "position". The fallback keeps the count honest: any row whose
    ``position_id`` is missing/zero falls back to its own ``ticket``, so it is
    counted as its own position (# Positions degrades to # Trades for that
    connector instead of silently under-counting).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        pd.Series: Grouping key, one value per row of trades_closed
    """
    pos_id = trades_closed["position_id"]
    ticket = trades_closed["ticket"]
    return pos_id.where(pos_id != 0, ticket)


def calc_num_positions(trades_closed: pd.DataFrame) -> int:
    """
    Count distinct closed positions (grouping partial closes together).

    Several ``Trade`` rows can belong to the same position when it was closed
    in more than one partial close; this counts the position once. See
    ``_position_group_key`` for the fallback used when ``position_id`` is not
    tracked by the connector.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        int: Number of distinct closed positions (0 if empty)
    """
    if trades_closed.empty:
        return 0
    return int(_position_group_key(trades_closed).nunique())


def count_positions_by_direction(trades_closed: pd.DataFrame) -> tuple[int, int]:
    """
    Count distinct closed positions by direction.

    A position's direction is taken from its first trade (all partial closes
    of the same position share the same direction).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        tuple: (long_count, short_count) of distinct positions
    """
    if trades_closed.empty:
        return 0, 0
    key = _position_group_key(trades_closed)
    first_dir = trades_closed.groupby(key)["direction"].first()
    longs = int((first_dir == "long").sum())
    shorts = int((first_dir == "short").sum())
    return longs, shorts


def build_positions_df(trades_closed: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate closed trades into one row per position.

    Groups partial closes of the same position (see ``_position_group_key``)
    into a single row: entry time/price come from the first partial close,
    exit time/price from the last, size/pnl_net/commission are summed, and
    ``n_partials`` counts how many trade rows made up the position (1 when the
    position was closed in a single fill).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        pd.DataFrame: One row per position, columns (position_id, symbol,
                     direction, entry_time, exit_time, size, entry_price,
                     exit_price, commission, pnl_net, n_partials). Empty
                     DataFrame with this schema if trades_closed is empty.
    """
    cols = ["position_id", "symbol", "direction", "entry_time", "exit_time",
            "size", "entry_price", "exit_price", "commission", "pnl_net",
            "n_partials"]
    if trades_closed.empty:
        return pd.DataFrame({c: pd.Series([], dtype="object") for c in cols})

    tc = trades_closed.copy()
    tc["_group_key"] = _position_group_key(tc)
    tc = tc.sort_values(["_group_key", "entry_time", "exit_time"])

    grouped = tc.groupby("_group_key", sort=False)
    out = pd.DataFrame({
        "position_id": grouped["_group_key"].first(),
        "symbol":      grouped["symbol"].first(),
        "direction":   grouped["direction"].first(),
        "entry_time":  grouped["entry_time"].first(),
        "exit_time":   grouped["exit_time"].last(),
        "size":        grouped["size"].sum(),
        "entry_price": grouped["entry_price"].first(),
        "exit_price":  grouped["exit_price"].last(),
        "commission":  grouped["commission"].sum(),
        "pnl_net":     grouped["pnl_net"].sum(),
        "n_partials":  grouped.size(),
    }).reset_index(drop=True)
    return out.sort_values(["entry_time", "exit_time"], na_position="last").reset_index(drop=True)


def trade_return_pct(trades_closed: pd.DataFrame) -> np.ndarray:
    """
    Calculate return percentage for each closed trade.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        np.ndarray: Array of return percentages
    """
    if trades_closed.empty:
        return np.array([], dtype=float)
    pnl = trades_closed['pnl_net'].to_numpy(dtype=float)
    notional = (trades_closed['size'].to_numpy(dtype=float)
                * trades_closed['entry_price'].to_numpy(dtype=float))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(notional > 0, pnl / notional * 100.0, np.nan)
    return r


def trade_pnl_net(trades_closed: pd.DataFrame) -> np.ndarray:
    """
    Extract net PnL for each closed trade.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        np.ndarray: Array of PnL values
    """
    if trades_closed.empty:
        return np.array([], dtype=float)
    return trades_closed['pnl_net'].to_numpy(dtype=float)


def trade_durations(trades_closed: pd.DataFrame) -> pd.Series:
    """
    Calculate hold time for each trade.

    Returns duration as pd.Timedelta (replacing trade_durations_days).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        pd.Series: Timedelta objects representing hold duration
    """
    if trades_closed.empty:
        return pd.Series(dtype='timedelta64[ns]')
    return trades_closed['exit_time'] - trades_closed['entry_time']


def trade_durations_days(trades_closed: pd.DataFrame) -> np.ndarray:
    """
    Calculate hold time in decimal days (compatibility alias).

    Use trade_durations() in new code to get pd.Timedelta objects.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        np.ndarray: Duration in days as floats
    """
    d = trade_durations(trades_closed)
    if d.empty:
        return np.array([], dtype=float)
    return (d / pd.Timedelta(days=1)).to_numpy(dtype=float)


def calc_win_rate_pct(trades_closed: pd.DataFrame) -> float:
    """
    Calculate win rate (profitable trades / total trades).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: Win rate percentage (0.0 if empty)
    """
    pnl = trade_pnl_net(trades_closed)
    return float((pnl > 0).mean() * 100.0) if pnl.size else 0.0


def calc_best_trade_pct(trades_closed: pd.DataFrame) -> float:
    """
    Find best-performing trade return.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: Best trade return percentage (0.0 if empty)
    """
    r = trade_return_pct(trades_closed)
    return float(np.nanmax(r)) if r.size else 0.0


def calc_worst_trade_pct(trades_closed: pd.DataFrame) -> float:
    """
    Find worst-performing trade return.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: Worst trade return percentage (0.0 if empty)
    """
    r = trade_return_pct(trades_closed)
    return float(np.nanmin(r)) if r.size else 0.0


def calc_avg_trade_pct(trades_closed: pd.DataFrame) -> float:
    """
    Calculate average trade return.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: Average return percentage (0.0 if empty)
    """
    r = trade_return_pct(trades_closed)
    return float(np.nanmean(r)) if r.size else 0.0


def calc_max_trade_duration(trades_closed: pd.DataFrame) -> pd.Timedelta:
    """
    Find longest-held trade.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        pd.Timedelta: Maximum hold duration (0 if empty)
    """
    d = trade_durations(trades_closed)
    if d.empty:
        return pd.Timedelta(0)
    return pd.Timedelta(d.max())


def calc_avg_trade_duration(trades_closed: pd.DataFrame) -> pd.Timedelta:
    """
    Calculate average trade duration.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        pd.Timedelta: Average hold duration (0 if empty)
    """
    d = trade_durations(trades_closed)
    if d.empty:
        return pd.Timedelta(0)
    return pd.Timedelta(d.mean())


def calc_profit_factor(trades_closed: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: Profit factor or NaN if no losing trades
    """
    pnl = trade_pnl_net(trades_closed)
    if pnl.size == 0:
        return np.nan
    gains = float(pnl[pnl > 0].sum())
    losses = float(pnl[pnl < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def calc_expectancy_pct(trades_closed: pd.DataFrame) -> float:
    """
    Calculate mathematical expectancy (expected return per trade).

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: Expected value percentage (0.0 if empty)
    """
    r = trade_return_pct(trades_closed)
    if r.size == 0:
        return 0.0
    wins = r[r > 0]
    losses = r[r <= 0]
    p = float((r > 0).mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    return p * avg_win + (1.0 - p) * avg_loss


def calc_sqn(trades_closed: pd.DataFrame) -> float:
    """
    Calculate System Quality Number (quality metric for trading systems).

    Higher SQN indicates more reliable system. Threshold: SQN > 2.5 is good.

    Args:
        trades_closed (pd.DataFrame): Closed trades DataFrame

    Returns:
        float: SQN value or NaN if no trades
    """
    r = trade_return_pct(trades_closed)
    n = int(r.size)
    if n == 0:
        return np.nan
    exp = calc_expectancy_pct(trades_closed)
    sd = float(np.std(r, ddof=0))
    if sd == 0:
        return np.inf if exp > 0 else (-np.inf if exp < 0 else np.nan)
    return float(np.sqrt(n) * (exp / sd))

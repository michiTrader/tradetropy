from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from tradetropy.exceptions import ConfigError, DataError


# ══════════════════════════════════════════════════════════════════════════════
# Statistical reliability thresholds
# ══════════════════════════════════════════════════════════════════════════════
#
# With very few trades or too short a duration, annualized metrics
# (Return Ann., Volatility, Sharpe, Sortino, Calmar) and those based on the
# trade distribution (SQN, Profit Factor) become uninterpretable and
# can produce astronomical numbers or inf. Below these thresholds those
# metrics are reported as NaN and Stats["_low_sample"] = True is set.
#
# The user can override the thresholds per call in compute_stats().

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 - FREQUENCY RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════
#
# _resolve_freq()  - chooses the right resample based on backtest duration.
# _ann_factor()    - returns the correct annualization factor for each freq.
#
# Auto-detect table (freq=None):
#   duration < 2h    -> "1min"
#   duration < 2d    -> "5min"
#   duration < 30d   -> "1h"
#   duration >= 30d  -> "1D"   (previous behavior)
#
# The resolved freq is stored in Stats as "_freq" so the user
# can inspect it and, if desired, override it on the next call.
#
# Annualization factors:
#   Base: 252 trading days, 6.5 market hours per day, 390 minutes per day.
#   For crypto/24h the correct factor would be higher, but the standard
#   equities convention is used. The user can pass a custom freq
#   if a different factor is needed -- the fallback will be 252.
#
#   "1W"    -> 52           (weeks per year)
#   "1D"    -> 252          (trading days)
#   "4h"    -> 252 * 1.625  (4h periods in 6.5h of market)
#   "2h"    -> 252 * 3.25
#   "1h"    -> 252 * 6.5
#   "30min" -> 252 * 13
#   "15min" -> 252 * 26
#   "5min"  -> 252 * 78
#   "1min"  -> 252 * 390
#   "30s"   -> 252 * 780
#   "15s"   -> 252 * 1560
#   "10s"   -> 252 * 2340
#   "5s"    -> 252 * 4680
#   "1s"    -> 252 * 23400
#   other   -> 252          (conservative fallback)

_FREQ_ANN_FACTORS: dict[str, float] = {
    "1W":   52.0,
    "1D":   252.0,
    "4h":   252.0 * 1.625,
    "2h":   252.0 * 3.25,
    "1h":   252.0 * 6.5,
    "30min": 252.0 * 13.0,
    "15min": 252.0 * 26.0,
    "5min": 252.0 * 78.0,
    "1min": 252.0 * 390.0,
    "30s":  252.0 * 780.0,
    "15s":  252.0 * 1560.0,
    "10s":  252.0 * 2340.0,
    "5s":   252.0 * 4680.0,
    "1s":   252.0 * 23400.0,
}


def _resolve_freq(equity_curve: pd.Series, freq: str | None) -> str:
    """
    Determine the resampling frequency for the equity curve.

    If freq is provided, returns it as-is. Otherwise, auto-detects based on
    the total backtest duration.

    Args:
        equity_curve (pd.Series): Equity curve with DatetimeIndex
        freq (str | None): User-provided frequency or None for auto-detect

    Returns:
        str: Resampling frequency (e.g. '1min', '5min', '1h', '1D')
    """
    if freq is not None:
        return freq

    duration = equity_curve.index[-1] - equity_curve.index[0]
    hours = duration.total_seconds() / 3600.0

    if hours < 2:
        return "1min"
    if hours < 48:
        return "5min"
    if hours < 30 * 24:
        return "1h"
    return "1D"


def _ann_factor(freq: str) -> float:
    """
    Get the annualization factor for a given frequency.

    Args:
        freq (str): Resampling frequency (e.g. '1D', '1h', '5min')

    Returns:
        float: Annualization factor for converting period returns to annual
    """
    return _FREQ_ANN_FACTORS.get(freq, 252.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 - DATA TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════════════


def _ts_precision(freq: str) -> str | None:
    """
    Get the timestamp rounding unit for a given frequency.

    For frequencies >= 1 minute, milliseconds and nanoseconds are rounded
    away as they provide no meaningful information. For sub-minute frequencies,
    full precision is preserved.

    Args:
        freq (str): Resampling frequency

    Returns:
        str | None: Timestamp unit ('s' for rounding, None for full precision)
    """
    _SUBMINUTE = {"30s", "15s", "10s", "5s", "1s"}
    if freq in _SUBMINUTE:
        return None
    # Any other known freq (1min, 5min, 1h, 1D, 1W...) -> round to seconds
    return "s"


def _round4(x: Any) -> Any:
    """
    Round numeric values to 4 decimal places while preserving other types.

    Args:
        x (Any): Value to round (numeric, bool, timestamp, or None)

    Returns:
        Any: Rounded value preserving the original type
    """
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x
    if isinstance(x, (int, np.integer)) and np.isfinite(x):
        return int(x)
    if isinstance(x, (float, np.floating)) and np.isfinite(x):
        return float(np.round(float(x), 4))
    return x


def _to_datetime_utc(s: pd.Series, *, name: str) -> pd.Series:
    """
    Convert Series to UTC datetime, auto-detecting timestamp format.

    Handles numeric (unix timestamp) and string datetime representations,
    automatically selecting milliseconds or seconds based on magnitude.

    Args:
        s (pd.Series): Series with timestamps (numeric, string, or datetime)
        name (str): Column name for error messages

    Returns:
        pd.Series: UTC datetime Series

    Raises:
        DataError: If timestamps are invalid or cannot be parsed
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, utc=True)
    else:
        a = pd.to_numeric(s, errors="coerce")
        if a.isna().all():
            out = pd.to_datetime(s, errors="coerce", utc=True)
        else:
            mx = float(np.nanmax(a.to_numpy(dtype=float)))
            unit = "ms" if mx > 1e11 else "s"
            out = pd.to_datetime(a, unit=unit, errors="coerce", utc=True)
    if out.isna().any():
        raise DataError(f"{name} contains invalid or unparseable timestamps.")
    return out


def _validate_initial_balance(initial_balance: float) -> float:
    """
    Validate and normalize initial balance.

    Args:
        initial_balance (float): Initial account balance

    Returns:
        float: Validated balance

    Raises:
        ConfigError: If balance is not a finite number >= 0
    """
    ib = float(initial_balance)
    if not math.isfinite(ib) or ib < 0:
        raise ConfigError(f'initial_balance must be a finite number >= 0, got {ib}')
    return ib


def _validate_equity_curve(equity_curve: pd.Series) -> pd.Series:
    """
    Validate equity curve structure and content.

    Args:
        equity_curve (pd.Series): Equity curve with DatetimeIndex (UTC)

    Returns:
        pd.Series: Validated equity curve

    Raises:
        DataError: If equity curve is invalid or empty
    """
    return equity_curve


def _empty_trades_df() -> pd.DataFrame:
    """
    Build an empty trades DataFrame with the standard schema and dtypes.

    Used when a backtest produced no closed trades (e.g. a position stays open
    to the end). Downstream metric functions guard on ``.empty`` and report
    trade metrics as N/A, while the equity curve is still computed - keeping
    backtest parity with replay/live, where the equity of an open position is
    always visible.

    Returns:
        pd.DataFrame: Empty frame with columns (entry_time, exit_time, size,
                     entry_price, exit_price, direction, commission, pnl_net,
                     symbol) and their expected dtypes.
    """
    return pd.DataFrame({
        "entry_time":  pd.Series([], dtype="datetime64[ns, UTC]"),
        "exit_time":   pd.Series([], dtype="datetime64[ns, UTC]"),
        "size":        pd.Series([], dtype="float64"),
        "entry_price": pd.Series([], dtype="float64"),
        "exit_price":  pd.Series([], dtype="float64"),
        "direction":   pd.Series([], dtype="object"),
        "commission":  pd.Series([], dtype="float64"),
        "pnl_net":     pd.Series([], dtype="float64"),
        "symbol":      pd.Series([], dtype="object"),
    })


def _trades_list_to_df(trades: list) -> pd.DataFrame:
    """
    Convert trade list to standardized DataFrame.

    Extracts OHLC, direction, and PnL from trade objects and normalizes
    timestamps to UTC datetime format.

    An empty trade list is valid: it yields an empty DataFrame with the
    standard schema (see ``_empty_trades_df``) instead of raising, so a
    backtest whose only position never closed still produces stats and an
    equity curve.

    Args:
        trades (list): List of Trade objects from broker

    Returns:
        pd.DataFrame: Normalized trades with columns (entry_time, exit_time,
                     size, entry_price, exit_price, direction, commission,
                     pnl_net, symbol)
    """
    rows = []
    for t in trades:
        raw_type = getattr(t, "type", None)
        if isinstance(raw_type, int) or hasattr(raw_type, "value"):
            direction = "long" if int(raw_type) == 0 else "short"
        else:
            s = str(raw_type).lower()
            direction = "long" if "buy" in s else "short"

        rows.append({
            "entry_time":  getattr(t, "time",        None),
            "exit_time":   getattr(t, "time_close",  None),
            "size":        float(getattr(t, "volume",      0.0)),
            "entry_price": float(getattr(t, "price",       0.0)),
            "exit_price":  float(getattr(t, "price_close", np.nan)),
            "direction":   direction,
            "commission":  float(getattr(t, "commission",  0.0)),
            "pnl_net":    float(getattr(t, "pnl_net",     np.nan)),
            "symbol":     getattr(t, "symbol", ""),
        })

    if not rows:
        return _empty_trades_df()

    df = pd.DataFrame(rows)
    df["entry_time"] = _to_datetime_utc(df["entry_time"], name="trades.entry_time")
    df["exit_time"] = pd.Series(
        pd.to_datetime(df["exit_time"], utc=True, errors="coerce"),
        dtype="datetime64[ns, UTC]",
    )
    for col in ["size", "entry_price", "exit_price", "commission", "pnl_net"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["direction"] = df["direction"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    return df.sort_values(["entry_time", "exit_time"], na_position="last").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - PURE CALCULATION FUNCTIONS
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 - COMPUTE_STATS()
# ══════════════════════════════════════════════════════════════════════════════


def compute_stats(
    equity_curve: pd.Series,
    trades: list,
    initial_balance: float,
    strategy=None,
    freq: str | None = None,
    *,
    min_trades: int = MIN_TRADES_FOR_STATS,
    min_duration_ann: pd.Timedelta = MIN_DURATION_ANN,
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
        ("# Trades Long",         count_trades_by_direction(tc)[0]),
        ("# Trades Short",        count_trades_by_direction(tc)[1]),
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
        warnings.warn(
            "Stats: insufficient sample, metrics zeroed to NaN -- "
            + "; ".join(reasons),
            stacklevel=2,
        )

    for k in metrics:
        metrics[k] = _round4(metrics[k])

    return Stats(metrics)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 - STATS (PRESENTATION ONLY)
# ══════════════════════════════════════════════════════════════════════════════


class Stats(pd.Series):
    """
    Backtest results container. Presentation and data access only.

    Encapsulates all performance metrics calculated by compute_stats().
    Inherits from pd.Series for dict-like access and pandas integration.

    Preferred construction:
        stats = compute_stats(equity_curve, trades, initial_balance)

    Direct construction (wrapping pre-calculated metrics):
        stats = Stats(ordered_dict)

    Legacy API (deprecated but functional):
        stats = Stats(equity_curve, trades, initial_balance)

    Metric access:
        stats['Sharpe Ratio']        -> float
        stats['# Trades']            -> int
        stats['Duration']            -> pd.Timedelta
        stats['Max. Trade Duration'] -> pd.Timedelta
        stats.trades                 -> pd.DataFrame
        stats.equity_curve           -> pd.Series
        stats.initial_balance        -> float
        stats.freq                   -> str (frequency used for ratios)
        stats.ann_factor             -> float (annualization factor used)

    Inspect the frequency and settings:
        stats.freq        # -> '1min', '5min', '1h' or '1D'
        stats.ann_factor  # -> 97.5, 252, etc.
    """

    _KEYS = (
        "Start",
        "End",
        "Duration",
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
        "Max. Drawdown Duration",
        "Avg. Drawdown Duration",
        "# Trades",
        "# Trades Long",
        "# Trades Short",
        "Win Rate [%]",
        "Best Trade [%]",
        "Worst Trade [%]",
        "Avg. Trade [%]",
        "Max. Trade Duration",
        "Avg. Trade Duration",
        "Profit Factor",
        "Expectancy [%]",
        "SQN",
        "Total Commissions [$]",
        "_low_sample",
        "_trades",
        "_equity_curve",
        "_initial_balance",
        "_strategy",
        "_freq",
        "_ann_factor",
    )

    def __new__(cls, *args, **kwargs):
        return pd.Series.__new__(cls)

    def __init__(
        self,
        equity_curve_or_metrics: "pd.Series | OrderedDict",
        trades: Any = None,
        initial_balance: float = 0.0,
        strategy: Any = None,
        freq: str | None = None,
        *,
        min_trades: int = MIN_TRADES_FOR_STATS,
        min_duration_ann: pd.Timedelta = MIN_DURATION_ANN,
    ):
        if isinstance(equity_curve_or_metrics, OrderedDict):
            metrics = equity_curve_or_metrics
        else:
            metrics = compute_stats(
                equity_curve=equity_curve_or_metrics,
                trades=trades,
                initial_balance=initial_balance,
                strategy=strategy,
                freq=freq,
                min_trades=min_trades,
                min_duration_ann=min_duration_ann,
            )
        pd.Series.__init__(self, data=metrics, name="Stats")

    @property
    def _constructor(self):
        return pd.Series

    @property
    def trades(self) -> pd.DataFrame:
        return self["_trades"]

    @property
    def equity_curve(self) -> pd.Series:
        return self["_equity_curve"]

    @property
    def initial_balance(self) -> float:
        return self["_initial_balance"]

    @property
    def freq(self) -> str:
        """
        Frequency used for return, volatility, and ratio calculations.

        Returns:
            str: Frequency identifier (e.g. '1D', '1h', '5min', '1min')
        """
        return self.get('_freq', '1D')

    @property
    def ann_factor(self) -> float:
        """
        Annualization factor used in Sharpe, Sortino, and volatility metrics.

        Returns:
            float: Factor (e.g. 252 for daily, 252*6.5 for hourly)
        """
        return float(self.get('_ann_factor', 252.0))

    @property
    def low_sample(self) -> bool:
        """
        Indicates whether sample size was insufficient for reliable metrics.

        When True, annualized metrics (Return Ann., Volatility, Sharpe,
        Sortino, Calmar) and/or trade distribution metrics (Profit Factor,
        SQN) are reported as NaN.

        Returns:
            bool: True if metrics were gated due to low sample
        """
        return bool(self.get("_low_sample", False))

    def to_dict(self) -> OrderedDict:
        """
        Export stats to ordered dictionary.

        Returns:
            OrderedDict: All metrics with internal fields (prefixed with '_')
        """
        return OrderedDict(self)

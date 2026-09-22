"""
tradetropy.stats._prepare - input validation and data normalization.

Validates/normalizes the raw inputs to compute_stats() before any metric is
calculated: initial balance, equity curve, and the trades list (converted to
a standardized DataFrame with a fixed schema).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tradetropy.exceptions import ConfigError, DataError


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
        pd.DataFrame: Empty frame with columns (ticket, position_id, entry_time,
                     exit_time, size, entry_price, exit_price, direction,
                     commission, pnl_net, symbol) and their expected dtypes.
    """
    return pd.DataFrame({
        "ticket":      pd.Series([], dtype="int64"),
        "position_id": pd.Series([], dtype="int64"),
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
        pd.DataFrame: Normalized trades with columns (ticket, position_id,
                     entry_time, exit_time, size, entry_price, exit_price,
                     direction, commission, pnl_net, symbol)
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
            "ticket":      int(getattr(t, "ticket", 0) or 0),
            "position_id": int(getattr(t, "position_id", 0) or 0),
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

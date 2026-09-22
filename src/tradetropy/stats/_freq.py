"""
tradetropy.stats._freq - frequency resolution and annualization factors.

_resolve_freq()  - chooses the right resample based on backtest duration.
_ann_factor()    - returns the correct annualization factor for each freq.
_ts_precision()  - timestamp rounding unit for a given frequency.

Auto-detect table (freq=None):
  duration < 2h    -> "1min"
  duration < 2d    -> "5min"
  duration < 30d   -> "1h"
  duration >= 30d  -> "1D"   (previous behavior)

The resolved freq is stored in Stats as "_freq" so the user
can inspect it and, if desired, override it on the next call.

Annualization factors:
  Base: 252 trading days, 6.5 market hours per day, 390 minutes per day.
  For crypto/24h the correct factor would be higher, but the standard
  equities convention is used. The user can pass a custom freq
  if a different factor is needed -- the fallback will be 252.

  "1W"    -> 52           (weeks per year)
  "1D"    -> 252          (trading days)
  "4h"    -> 252 * 1.625  (4h periods in 6.5h of market)
  "2h"    -> 252 * 3.25
  "1h"    -> 252 * 6.5
  "30min" -> 252 * 13
  "15min" -> 252 * 26
  "5min"  -> 252 * 78
  "1min"  -> 252 * 390
  "30s"   -> 252 * 780
  "15s"   -> 252 * 1560
  "10s"   -> 252 * 2340
  "5s"    -> 252 * 4680
  "1s"    -> 252 * 23400
  other   -> 252          (conservative fallback)
"""

from __future__ import annotations

import pandas as pd

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

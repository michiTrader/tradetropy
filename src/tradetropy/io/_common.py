"""
Helpers shared by the ticks/klines readers and writers: source normalization
(TickProxy/OhlcProxy/ndarray/DataFrame -> DataFrame) and timestamp conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Literal

from tradetropy.exceptions import TradingError, ConnectionError, DataError

# Internal alias kept for implementation-detail signatures in this package
# (public functions expose the expanded Literal[...] directly).
_TsFormat = Literal["iso", "ms", "s"]


def _source_to_df_ticks(source, n_cols: int, cols: tuple) -> pd.DataFrame:
    """
    Normalize TickProxy, ndarray or DataFrame to DataFrame with tick columns.

    Args:
        source: TickProxy, ndarray [N×7] or DataFrame to normalize.
        n_cols: Number of columns to extract.
        cols: Tuple of column names for the output DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standard tick columns.

    Raises:
        ConnectionError: If TickProxy has no connected backend.
        TradingError: If source type is not TickProxy, ndarray or DataFrame.
    """
    from tradetropy.data import TickProxy

    if isinstance(source, TickProxy):
        view = next(iter(source._views.values()), None)
        if view is None:
            raise ConnectionError(f"TickProxy '{source.symbol}' is not connected.")

        if view._tick_store is not None:
            cursor = view._cursor.pos
            if cursor < 0:
                return pd.DataFrame(columns=cols)
            start = max(0, cursor + 1 - source._window_size)
            data   = view._tick_store.matrix[start : cursor + 1, :n_cols].copy()

        elif view._tick_ring is not None:
            ring = view._tick_ring
            n    = ring.n_available
            if n == 0:
                return pd.DataFrame(columns=cols)
            start = ring._head + (ring._W - n)
            data   = ring._buf[start : start + n, :n_cols].copy()

        else:
            raise ConnectionError(f"TickProxy '{source.symbol}' has no recognized backend.")

        return pd.DataFrame(data, columns=cols)

    elif isinstance(source, np.ndarray):
        return pd.DataFrame(source[:, :n_cols], columns=cols)

    elif isinstance(source, pd.DataFrame):
        return source.copy()

    raise TradingError(
        f"save_ticks() expected TickProxy, ndarray or DataFrame, "
        f"got {type(source).__name__}."
    )


def _source_to_df_klines(source, n_cols: int, cols: tuple, include_partial: bool) -> pd.DataFrame:
    """
    Normalize OhlcProxy, ndarray or DataFrame to DataFrame with OHLC columns.

    Args:
        source: OhlcProxy, ndarray [N×6] or DataFrame to normalize.
        n_cols: Number of columns to extract.
        cols: Tuple of column names for the output DataFrame.
        include_partial: Include partial candle if source is OhlcProxy.

    Returns:
        pd.DataFrame: DataFrame with standard OHLC columns and 'partial' flag.

    Raises:
        ConnectionError: If OhlcProxy has no connected backend.
        TradingError: If source type is not OhlcProxy, ndarray or DataFrame.
    """
    from tradetropy.data import OhlcProxy

    if isinstance(source, OhlcProxy):
        rows: list[pd.DataFrame] = []

        if source._ohlc_store is not None:
            store  = source._ohlc_store
            view   = next(iter(source._views.values()), None)
            cursor = view._cursor.pos if view is not None else -1

            if cursor >= 0:
                n_candle = int(store.tick_to_candle_mapping[cursor])
                if n_candle > 0:
                    df_c = pd.DataFrame(store.matrix[:n_candle, :n_cols], columns=cols)
                    df_c["partial"] = 0
                    rows.append(df_c)
                if include_partial:
                    df_p = pd.DataFrame([store.partial_candle_at_tick(cursor)], columns=cols)
                    df_p["partial"] = 1
                    rows.append(df_p)

        elif source._ohlc_ring is not None:
            ring       = source._ohlc_ring
            n_closed   = min(ring._n_closed, ring._W)

            if n_closed > 0:
                start = ring._head + (ring._W - n_closed)
                df_c   = pd.DataFrame(
                    ring._buf[start : start + n_closed, :n_cols], columns=cols
                )
                df_c["partial"] = 0
                rows.append(df_c)

            if include_partial and ring._current_candle_ts >= 0:
                df_p = pd.DataFrame([ring._partial_candle[:n_cols]], columns=cols)
                df_p["partial"] = 1
                rows.append(df_p)

        else:
            raise ConnectionError(f"OhlcProxy '{source.symbol}' is not connected.")

        if not rows:
            return pd.DataFrame(columns=list(cols) + ["partial"])
        return pd.concat(rows, ignore_index=True)

    elif isinstance(source, np.ndarray):
        # KlineData.data is [N x 7] (col 6 = turnover); a bare OHLCV array is
        # [N x 6]. The OhlcProxy path above uses n_cols=6 on purpose because in
        # the backtest store column 6+ are pre-calculated indicators, not
        # turnover. For a raw ndarray there is no such ambiguity: keep the
        # turnover column when it is present so it round-trips through save.
        from tradetropy.core.constants import (
            OHLCV_TURNOVER_COLS, N_OHLCV_TURNOVER_COLS,
        )
        if source.ndim == 2 and source.shape[1] >= N_OHLCV_TURNOVER_COLS:
            return pd.DataFrame(
                source[:, :N_OHLCV_TURNOVER_COLS], columns=OHLCV_TURNOVER_COLS
            )
        return pd.DataFrame(source[:, :n_cols], columns=cols)

    elif isinstance(source, pd.DataFrame):
        return source.copy()

    raise TradingError(
        f"save_klines() expected OhlcProxy, ndarray or DataFrame, "
        f"got {type(source).__name__}."
    )


def _ts_to_datetime(df: pd.DataFrame, ts_format: _TsFormat) -> pd.DataFrame:
    """
    Convert Unix timestamp column to datetime in specified format.

    Args:
        df: DataFrame with 'ts' column containing milliseconds since epoch.
        ts_format: Output format ('iso', 'ms', 's').

    Returns:
        pd.DataFrame: DataFrame with 'ts' replaced by 'datetime' column.

    Raises:
        DataError: If ts_format is invalid.
    """
    ts_ms = df["ts"].astype(np.int64)
    df    = df.drop(columns=["ts"])

    if ts_format == "iso":
        dt = pd.to_datetime(ts_ms, unit="ms", utc=True).dt.tz_localize(None).astype(str)
    elif ts_format == "ms":
        dt = pd.to_datetime(ts_ms, unit="ms", utc=True)
    elif ts_format == "s":
        dt = ts_ms // 1000
    else:
        raise DataError(f"Unknown ts_format: {ts_format!r}. Use 'iso', 'ms' or 's'.")

    df.insert(0, "datetime", dt)
    return df


def _datetime_to_ts_ms(column: pd.Series) -> np.ndarray:
    """
    Convert datetime or numeric column to Unix timestamp in milliseconds.

    Handles datetime objects, ISO strings and numeric timestamps.

    Args:
        column: pandas Series with datetime or numeric values.

    Returns:
        np.ndarray: Array of timestamps in milliseconds since epoch.
    """
    if pd.api.types.is_datetime64_any_dtype(column):
        # Do NOT assume nanosecond resolution: pandas >= 2.0 preserves the
        # source unit (a column written with unit='ms' round-trips as
        # datetime64[ms], not [ns]), so a hardcoded // 1_000_000 corrupts the
        # timestamp. Truncate to millisecond resolution explicitly, then
        # reinterpret as integer milliseconds. Normalize tz-aware to UTC first.
        s = column
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.to_numpy(dtype="datetime64[ms]").astype(np.int64)

    ts_numeric = pd.to_numeric(column, errors="coerce")
    if ts_numeric.isna().any():
        # Datetime strings: parse and truncate to ms (unit-agnostic).
        dt = pd.to_datetime(column, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        return dt.to_numpy(dtype="datetime64[ms]").astype(np.int64)

    ts = ts_numeric.astype(np.int64)
    if ts.max() < 1e11:
        ts = ts * 1000
    return ts

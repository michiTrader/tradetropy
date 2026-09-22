"""
Kline (OHLC candle) data IO: save_klines / read_klines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal

from tradetropy.io._backends import (
    _detect_format, _resolve_write_format, _read, _write,
    _read_attrs, _write_hdf5_attrs,
)
from tradetropy.io._common import (
    _source_to_df_klines, _ts_to_datetime, _datetime_to_ts_ms,
)
from tradetropy.exceptions import DataError


# MT5 bar export header, as written by MetaTrader 5's "Export Bars" tool.
# The export is tab-separated and stores date/time in two separate columns.
_MT5_KLINE_REQUIRED_COLS = (
    "<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
)
_MT5_KLINE_VOLUME_COLS = ("<TICKVOL>", "<VOL>")


def _sniff_mt5_kline_header(path: Path) -> bool:
    """
    Check whether a CSV has the MT5 bar-export header.

    Args:
        path: CSV file path.

    Returns:
        bool: True when the first tab-separated columns match the MT5 bar
            schema, otherwise False.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
    except OSError:
        return False
    cols = [c.strip() for c in first_line.split("\t")]
    return cols[: len(_MT5_KLINE_REQUIRED_COLS)] == list(_MT5_KLINE_REQUIRED_COLS)


def _read_mt5_klines_csv(
    path: Path,
    symbol: str,
    timeframe: int,
    *,
    tick_size: float,
    tick_value: float,
    contract_size: float,
    digits: int,
    avg_spread: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> "KlineData":
    """
    Read an MT5 bar export and return KlineData.

    MT5 exports bars as tab-separated rows with ``<DATE>`` and ``<TIME>``
    columns. ``<TICKVOL>`` is used as the canonical volume because it is
    populated consistently across MT5 symbols; ``<VOL>`` is accepted as a
    fallback for exports that omit tick volume. MT5 does not export turnover,
    so that output column is filled with NaN.

    Args:
        path: MT5 bar-export CSV path.
        symbol: Trading symbol.
        timeframe: Candle interval in milliseconds.
        tick_size: Minimum price move.
        tick_value: Value of one tick.
        contract_size: Contract size.
        digits: Number of decimal places.
        avg_spread: Average spread in ticks.
        volume_min: Minimum order volume.
        volume_max: Maximum order volume.
        volume_step: Minimum volume increment.

    Returns:
        KlineData: Parsed OHLCV data with a NaN turnover column.

    Raises:
        DataError: If the header or required values are invalid.
    """
    from tradetropy.core.data_types import KlineData

    df = pd.read_csv(
        path, sep="\t", encoding="utf-8-sig",
        dtype={"<DATE>": str, "<TIME>": str},
    )
    missing = [c for c in _MT5_KLINE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataError(
            f"{path.name!r} does not look like an MT5 bar export: missing "
            f"columns {missing}. Found columns: {list(df.columns)}."
        )

    volume_col = next((c for c in _MT5_KLINE_VOLUME_COLS if c in df.columns), None)
    if volume_col is None:
        raise DataError(
            f"{path.name!r} is missing an MT5 volume column. Expected one of "
            f"{list(_MT5_KLINE_VOLUME_COLS)}; found {list(df.columns)}."
        )

    date_text = df["<DATE>"].astype(str).str.strip()
    time_text = df["<TIME>"].astype(str).str.strip()
    try:
        dt = pd.to_datetime(date_text + " " + time_text, format="mixed")
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"{path.name!r} contains invalid MT5 date/time values in "
            f"'<DATE>'/'<TIME>'."
        ) from exc
    ts_ms = dt.to_numpy(dtype="datetime64[ms]").astype(np.int64)

    def _numeric(column: str, *, default: float | None = None) -> np.ndarray:
        values = pd.to_numeric(df[column], errors="coerce")
        if default is not None:
            values = values.fillna(default)
        if values.isna().any():
            raise DataError(
                f"{path.name!r} contains non-numeric values in {column!r}."
            )
        return values.to_numpy(dtype=np.float64)

    open_col = _numeric("<OPEN>")
    high_col = _numeric("<HIGH>")
    low_col = _numeric("<LOW>")
    close_col = _numeric("<CLOSE>")
    volume_col_data = _numeric(volume_col, default=0.0)
    data = np.column_stack([
        ts_ms.astype(np.float64), open_col, high_col, low_col, close_col,
        volume_col_data, np.full(len(df), np.nan, dtype=np.float64),
    ])

    return KlineData(
        symbol=symbol, data=data, timeframe=timeframe,
        tick_size=tick_size, tick_value=tick_value,
        contract_size=contract_size, digits=digits,
        avg_spread=avg_spread, volume_min=volume_min,
        volume_max=volume_max, volume_step=volume_step,
    )


def save_klines(
    data: "OhlcProxy | np.ndarray | pd.DataFrame",
    path: str | Path,
    format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    include_partial: bool = True,
    ts_format: Literal["iso", "ms", "s"] = "iso",
    hdf5_key: str = "data",
    compression: str = "snappy",
    metadata: "dict | None" = None,
) -> pd.DataFrame:
    """
    Save klines to file.

    Args:
        data: OhlcProxy, ndarray [N×6] or DataFrame with standard OHLC columns.
        path: Destination file path.
        format: Output format ('csv', 'parquet', 'hdf5', 'npz'). None (default)
            infers from the extension, falling back to 'npz'.
        include_partial: Include partial candle (partial=1) in OhlcProxy.
        ts_format: Timestamp format for CSV ('iso', 'ms', 's').
        hdf5_key: Key inside the HDF5 file.
        compression: Parquet compression method ('snappy', 'gzip', 'zstd', None).
        metadata: Optional dict of tradetropy metadata to store as HDF5 attrs.

    Returns:
        pd.DataFrame: The saved kline data as DataFrame.

    Example:
        save_klines(self.btc_1m, 'ohlc.npz')
        save_klines(self.btc_1m, 'ohlc.parquet', format='parquet')
        save_klines(self.btc_1m, 'session.h5', format='hdf5', hdf5_key='btc/1m')
    """
    from tradetropy.core.constants import OHLC_COLS, N_OHLC_COLS

    format = _resolve_write_format(path, format)
    df = _source_to_df_klines(data, N_OHLC_COLS, OHLC_COLS, include_partial)
    if format == "csv":
        df = _ts_to_datetime(df, ts_format)
    elif format != "npz":
        df = _ts_to_datetime(df, "ms")
    # npz keeps the raw millisecond 'ts' column (read_klines handles both).
    _write(df, path, format, hdf5_key, compression, metadata)
    if format == "hdf5" and metadata:
        _write_hdf5_attrs(Path(path), hdf5_key, metadata)
    return df


def read_klines(
    path: str | Path,
    symbol: "str | None" = None,
    timeframe: "str | int | None" = None,
    format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    source: str = "auto",
    hdf5_key: str = "data",
    col_datetime: str = "datetime",
    tick_size: float = 0.01,
    tick_value: float = 0.01,
    contract_size: float = 1.0,
    digits: int = 2,
    avg_spread: float = 0.0,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> "KlineData":
    """
    Read klines from file and return KlineData.

    Accepts external files (broker CSV) and files generated by save_klines().

    Required columns: datetime, open, high, low, close, volume
    Optional columns: turnover (filled with NaN if missing)

    Args:
        path: File path to read from.
        symbol: Trading symbol. If None, read from HDF5 attrs (saved by
            KlineData.save()). Required for CSV/Parquet or non-tradetropy files.
        timeframe: Candle interval. Accepts a timeframe string ('1m', '5m',
            '1h', '1d', etc.) or an integer number of milliseconds, parsed via
            parse_timeframe(). If None, read from HDF5 attrs.
        format: File format ('csv', 'parquet', 'hdf5'). None -> auto-detect.
        source: Kline file schema. 'auto' (default) reads the generic
            datetime/open/high/low/close/volume schema; 'mt5' reads an MT5
            tab-separated bar export with <DATE>/<TIME>/<OPEN>/<HIGH>/<LOW>/
            <CLOSE>/<TICKVOL> columns.
        hdf5_key: Key inside HDF5 file.
        col_datetime: Datetime column name.
        tick_size: Minimum price move.
        tick_value: Value of one tick.
        contract_size: Size of contract.
        digits: Number of decimal places.
        volume_min: Minimum order volume.
        volume_max: Maximum order volume.
        volume_step: Minimum volume increment (lot step).

    Returns:
        KlineData: Kline data object.

    Example:
        klines = read_klines('ohlc.parquet', 'BTCUSDT', timeframe='1m')
        klines = read_klines('session.h5')  # all read from attrs
        klines = read_klines('bybit_export.csv', 'BTCUSDT', timeframe=60_000)
    """
    from tradetropy.core.constants import parse_timeframe
    from tradetropy.core.data_types import KlineData

    path    = Path(path)
    format  = format or _detect_format(path)

    if source == "mt5":
        if format != "csv":
            raise DataError("source='mt5' is only supported for CSV files.")
        if symbol is None:
            raise ValueError(
                "symbol is required for source='mt5' (an MT5 export carries "
                "no tradetropy attrs)."
            )
        if timeframe is None:
            raise ValueError(
                "read_klines() requires timeframe= for source='mt5' because "
                "an MT5 export carries no tradetropy attrs."
            )
        return _read_mt5_klines_csv(
            path, symbol, parse_timeframe(timeframe),
            tick_size=tick_size, tick_value=tick_value,
            contract_size=contract_size, digits=digits,
            avg_spread=avg_spread, volume_min=volume_min,
            volume_max=volume_max, volume_step=volume_step,
        )

    if source != "auto":
        raise DataError(
            f"Unknown source {source!r}. Available sources: 'auto', 'mt5'."
        )

    if format in ("hdf5", "npz") and (symbol is None or timeframe is None):
        attrs = _read_attrs(path, format, hdf5_key)
        if symbol is None:
            symbol = attrs.get("tradetropy_symbol")
        if attrs.get("tradetropy_symbol") is not None:
            tick_size    = attrs.get("tradetropy_tick_size", tick_size)
            tick_value   = attrs.get("tradetropy_tick_value", tick_value)
            contract_size = attrs.get("tradetropy_contract_size", contract_size)
            digits       = attrs.get("tradetropy_digits", digits)
            avg_spread   = attrs.get("tradetropy_avg_spread", avg_spread)
            volume_min   = attrs.get("tradetropy_volume_min", volume_min)
            volume_max   = attrs.get("tradetropy_volume_max", volume_max)
            volume_step  = attrs.get("tradetropy_volume_step", volume_step)
            if timeframe is None:
                stored_interval = attrs.get("tradetropy_interval_ms")
                if stored_interval is not None:
                    timeframe = int(stored_interval)

    if symbol is None:
        raise ValueError(
            "symbol is required. Pass it explicitly or use a file saved by "
            "KlineData.save() with format='hdf5'."
        )

    if timeframe is None:
        raise ValueError(
            "read_klines() requires a timeframe (e.g. timeframe='5m' or "
            "timeframe=60_000). The file has no tradetropy_interval_ms attr "
            "(not saved by KlineData.save(), or saved before the timeframe "
            "was set) - pass timeframe explicitly to fix it going forward."
        )
    interval_ms = parse_timeframe(timeframe)

    df = _read(path, format, hdf5_key)

    if "ts" not in df.columns and col_datetime not in df.columns:
        if format == "csv" and _sniff_mt5_kline_header(path):
            raise DataError(
                f"{path.name!r} looks like an MT5 bar export, not the generic "
                f"tradetropy kline schema (datetime, open, high, low, close, "
                f"volume). Re-run with source='mt5' to read it directly."
            )
        raise DataError(
            f"{path.name!r} is missing a {col_datetime!r} or 'ts' column. "
            f"Found columns: {list(df.columns)}. If this is a broker-specific "
            f"export, pass source=... (available: 'mt5') or convert it to the "
            f"generic schema first (datetime, open, high, low, close, volume)."
        )

    # Discard partial candles if coming from save_klines()
    if "partial" in df.columns:
        df = (
            df[df["partial"] == 0]
            .drop(columns=["partial"])
            .reset_index(drop=True)
        )

    # Accept a raw 'ts' column (record path / _append_klines_hdf5) or a
    # human-readable 'datetime' column (save_klines csv), mirroring read_ticks.
    if "ts" in df.columns:
        ts_ms = df["ts"].to_numpy(dtype=np.int64)
        df    = df.drop(columns=["ts"])
    else:
        ts_ms = _datetime_to_ts_ms(df[col_datetime])
        df    = df.drop(columns=[col_datetime])

    cols_map = {c.lower(): c for c in df.columns}

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        real = cols_map.get(name)
        return (
            df[real].to_numpy(dtype=np.float64)
            if real
            else np.full(len(df), default, dtype=np.float64)
        )

    data = np.column_stack([
        ts_ms,
        _col("open"), _col("high"), _col("low"), _col("close"),
        _col("volume"), _col("turnover", np.nan),
    ])

    return KlineData(
        symbol=symbol, data=data, timeframe=interval_ms,
        tick_size=tick_size, tick_value=tick_value,
        contract_size=contract_size, digits=digits,
        volume_min=volume_min, volume_max=volume_max,
        volume_step=volume_step,
    )

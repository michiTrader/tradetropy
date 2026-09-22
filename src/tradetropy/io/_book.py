"""
L2 order-book data IO: save_book / read_book / convert_book, and the
wide/long on-disk layout helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal

from tradetropy.core.data_types import BookData
from tradetropy.exceptions import TradingError, DataError

from tradetropy.io._backends import (
    _detect_format, _resolve_write_format, _read, _write,
    _read_attrs, _write_hdf5_attrs,
)
from tradetropy.io._common import _ts_to_datetime, _datetime_to_ts_ms

# Internal alias kept for implementation-detail signatures in this package
# (public functions expose the expanded Literal[...] directly).
_Layout = Literal["auto", "wide", "long"]


def _source_to_df_book(source, levels: "int | None"):
    """
    Normalize BookData, ndarray or DataFrame to (DataFrame, levels, metadata).

    Args:
        source: BookData, ndarray [N x (2+4*levels)] or DataFrame with book cols.
        levels: Number of book levels K (required for ndarray, ignored for
            BookData which carries its own; inferred for DataFrame if None).

    Returns:
        tuple[pd.DataFrame, int, dict]: The book DataFrame, resolved levels, and
        default tradetropy metadata (empty for ndarray/DataFrame).

    Raises:
        DataError: If levels cannot be resolved.
        TradingError: If source type is unsupported.
    """
    from tradetropy.core.data_types import (
        BookData, book_flat_columns, book_row_width,
    )

    if isinstance(source, BookData):
        k = int(source.levels)
        df = pd.DataFrame(
            np.asarray(source.data, dtype=np.float64)[:, : book_row_width(k)],
            columns=book_flat_columns(k),
        )
        meta = {
            "tradetropy_symbol": source.symbol,
            "tradetropy_levels": k,
            "tradetropy_tick_size": source.tick_size,
        }
        return df, k, meta

    if isinstance(source, pd.DataFrame):
        if levels is None:
            n_bid = sum(1 for c in source.columns if str(c).startswith("bid_px_"))
            if n_bid <= 0:
                raise DataError(
                    "save_book(): cannot infer 'levels' from the DataFrame. "
                    "Pass levels=K explicitly."
                )
            levels = n_bid
        return source.copy(), int(levels), {}

    if isinstance(source, np.ndarray):
        if levels is None:
            raise DataError(
                "save_book(): 'levels' is required when saving a raw ndarray."
            )
        k = int(levels)
        return (
            pd.DataFrame(source[:, : book_row_width(k)], columns=book_flat_columns(k)),
            k,
            {},
        )

    raise TradingError(
        f"save_book() expected BookData, ndarray or DataFrame, "
        f"got {type(source).__name__}."
    )


def save_book(
    data,
    path: str | Path,
    format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    *,
    levels: "int | None" = None,
    layout: Literal["auto", "wide", "long"] = "wide",
    ts_format: Literal["iso", "ms", "s"] = "iso",
    hdf5_key: str = "book",
    compression: str = "snappy",
    metadata: "dict | None" = None,
) -> pd.DataFrame:
    """
    Save L2 order-book rows to file (symmetric to save_ticks / save_klines).

    Two on-disk layouts are supported (both round-trip through read_book):

    - layout='wide' (default, tradetropy native): grouped bid_px_*/bid_sz_*/
      ask_px_*/ask_sz_* columns, one book event per row.
    - layout='long' (Binance bookDepth-style): 'ts'/'percentage'/'depth'/
      'notional' columns, one row per price level per side with CUMULATIVE
      depth/notional (percentage = signed 1-based level index; negative = bid,
      positive = ask). This is a lossless re-encoding of the wide data that
      de-cumulates back exactly on read.

    Args:
        data: BookData, ndarray [N x (2+4*levels)] or DataFrame with book cols.
        path: Destination file path.
        format: Output format ('csv', 'parquet', 'hdf5').
        levels: Book levels K (required for a raw ndarray; taken from BookData).
        layout: On-disk layout ('wide' or 'long').
        ts_format: Timestamp format for CSV ('iso', 'ms', 's').
        hdf5_key: Key inside the HDF5 file (default 'book').
        compression: Parquet compression method ('snappy', 'gzip', 'zstd', None).
        metadata: Optional dict of tradetropy metadata to store as HDF5 attrs
            (merged over the defaults derived from a BookData).

    Returns:
        pd.DataFrame: The saved book data as DataFrame (in the chosen layout).

    Example:
        book.save('book.h5')                        # via BookData.save()
        save_book(book, 'book.parquet', format='parquet')
        save_book(arr, 'book.csv', format='csv', levels=20)
        save_book(book, 'bookdepth.csv', layout='long')
    """
    from tradetropy.core.data_types import wide_rows_to_long, book_flat_columns

    format = _resolve_write_format(path, format)
    df, resolved_levels, default_meta = _source_to_df_book(data, levels)

    if layout == "long":
        wide_arr = df[book_flat_columns(resolved_levels)].to_numpy(dtype=np.float64)
        df = wide_rows_to_long(wide_arr, resolved_levels)

    # For CSV keep a human-readable 'datetime' column; for hdf5/parquet/npz keep
    # the raw 'ts' column (matching the record-path schema of _append_book, and
    # avoiding a datetime64 resolution round-trip).
    if format == "csv":
        df = _ts_to_datetime(df, ts_format)

    attrs = {**default_meta, "tradetropy_levels": int(resolved_levels)}
    if metadata:
        attrs.update(metadata)
    _write(df, path, format, hdf5_key, compression, attrs)
    if format == "hdf5":
        _write_hdf5_attrs(Path(path), hdf5_key, attrs)
    return df


def _detect_book_layout(df: pd.DataFrame) -> str:
    """
    Detect the on-disk order-book layout of a DataFrame by its columns.

    Args:
        df: DataFrame read from an order-book file.

    Returns:
        str: 'long' if the DataFrame carries the long (bookDepth-style)
        'percentage'/'depth'/'notional' columns, otherwise 'wide' (the grouped
        bid_px_*/ask_px_* layout, the default and back-compatible case).
    """
    cols = {str(c).lower() for c in df.columns}
    if {"percentage", "depth", "notional"} <= cols:
        return "long"
    return "wide"


def _normalize_long_book_df(df: pd.DataFrame, col_datetime: str) -> pd.DataFrame:
    """
    Normalize a long (bookDepth-style) DataFrame to canonical columns with 'ts'.

    Accepts case-insensitive column names and a time column named 'ts',
    'timestamp', 'datetime', 'time' or ``col_datetime`` (a real Binance bookDepth
    file uses 'timestamp'). The time column is converted to epoch milliseconds
    and named 'ts'; 'percentage'/'depth'/'notional' are lower-cased to canonical.

    Args:
        df: Raw long DataFrame as read from file.
        col_datetime: Preferred datetime column name to look for.

    Returns:
        pd.DataFrame: DataFrame with columns ['ts', 'percentage', 'depth',
        'notional'].

    Raises:
        DataError: If a usable time column cannot be found.
    """
    lower = {str(c).lower(): c for c in df.columns}
    d = df.copy()

    if "ts" in lower:
        d = d.rename(columns={lower["ts"]: "ts"})
    else:
        tcol = None
        for cand in (col_datetime, "datetime", "timestamp", "time"):
            if cand.lower() in lower:
                tcol = lower[cand.lower()]
                break
        if tcol is None:
            raise DataError(
                "read_book(): long layout requires a time column named "
                "'ts', 'timestamp', 'datetime' or 'time'."
            )
        ts_ms = _datetime_to_ts_ms(d[tcol])
        d = d.drop(columns=[tcol])
        d.insert(0, "ts", ts_ms)

    for name in ("percentage", "depth", "notional"):
        if name not in d.columns and name in lower:
            d = d.rename(columns={lower[name]: name})

    return d


def read_book(
    path: "str | Path",
    symbol: "str | None" = None,
    levels: "int | None" = None,
    format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    layout: Literal["auto", "wide", "long"] = "auto",
    hdf5_key: str = "book",
    col_datetime: str = "datetime",
    tick_size: float = 0.01,
) -> BookData:
    """
    Read L2 order-book rows and return a BookData (multi-format, multi-layout).

    Round-trips files written by save_book() / BookData.save() in any format
    (csv, parquet, hdf5) and the append-only HDF5 written by the live record=
    path (_append_book_hdf5). Two on-disk layouts are supported and normalized
    to the same wide BookData in memory:

    - 'wide' (tradetropy native): grouped bid_px_*/bid_sz_*/ask_px_*/ask_sz_*
      columns, one book event per row.
    - 'long' (Binance bookDepth-style): 'timestamp'/'percentage'/'depth'/
      'notional' columns, one row per price level per side, with cumulative
      depth/notional. De-cumulated back to per-level price/size on read (exact
      for tradetropy-written files, approximate per-band VWAP for a real Binance
      bookDepth file, which carries no explicit per-level price).

    The 'ts' column is accepted directly, or rebuilt from a 'datetime' /
    'timestamp' column. For the wide layout the level count K is taken from the
    argument, from HDF5 metadata, or inferred from the bid_px_* columns; for the
    long layout it is inferred from the largest |percentage| present.

    Args:
        path: File path (as written by save_book / BookData.save / record=).
        symbol: Trading symbol. If None, read from HDF5 attrs when available.
        levels: Number of book levels K (wide only). If None, from metadata or
            inferred.
        format: File format ('csv', 'parquet', 'hdf5'). None -> auto-detect.
        layout: On-disk layout ('auto', 'wide', 'long'). 'auto' detects from the
            columns (long if percentage/depth/notional present, else wide).
        hdf5_key: Key inside the HDF5 file (default 'book').
        col_datetime: Datetime column name (when no 'ts' column is present).
        tick_size: Minimum price step propagated to BookData.

    Returns:
        BookData: Reconstructed order-book data (always the wide in-memory form).

    Raises:
        DataError: If the level count cannot be determined.
        ValueError: If the symbol cannot be resolved.
    """
    from tradetropy.core.data_types import BookData, book_flat_columns, long_rows_to_wide

    path   = Path(path)
    format = format or _detect_format(path)

    if format in ("hdf5", "npz"):
        attrs = _read_attrs(path, format, hdf5_key)
        if symbol is None:
            symbol = attrs.get("tradetropy_symbol")
        if levels is None and attrs.get("tradetropy_levels") is not None:
            levels = int(attrs["tradetropy_levels"])
        tick_size = attrs.get("tradetropy_tick_size", tick_size)

    df = _read(path, format, hdf5_key)

    resolved_layout = layout if layout != "auto" else _detect_book_layout(df)

    if symbol is None:
        raise ValueError(
            "symbol is required. Pass it explicitly or use a file saved by "
            "BookData.save() with format='hdf5'."
        )

    if resolved_layout == "long":
        long_df = _normalize_long_book_df(df, col_datetime)
        data, levels = long_rows_to_wide(long_df)
        return BookData(symbol=symbol, data=data, levels=levels, tick_size=tick_size)

    if levels is None:
        n_bid = sum(1 for c in df.columns if str(c).startswith("bid_px_"))
        if n_bid <= 0:
            raise DataError(
                f"read_book(): could not determine 'levels' from {path}. "
                f"Pass levels=K explicitly."
            )
        levels = n_bid
    levels = int(levels)

    if "ts" not in df.columns and col_datetime in df.columns:
        ts_ms = _datetime_to_ts_ms(df[col_datetime])
        df = df.drop(columns=[col_datetime])
        df.insert(0, "ts", ts_ms)

    cols = book_flat_columns(levels)
    data = df[cols].to_numpy(dtype=np.float64)
    return BookData(symbol=symbol, data=data, levels=levels, tick_size=tick_size)


def convert_book(
    src: "str | Path",
    dst: "str | Path",
    *,
    symbol: "str | None" = None,
    to_layout: Literal["wide", "long"] = "wide",
    src_layout: Literal["auto", "wide", "long"] = "auto",
    src_format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    dst_format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    levels: "int | None" = None,
    tick_size: float = 0.01,
    **save_kwargs,
):
    """
    Convert an order-book file between layouts and/or container formats.

    Reads ``src`` (any supported layout/format) into a BookData - which is always
    the wide in-memory form - and writes it to ``dst`` in ``to_layout``. Layout
    and container format are auto-detected from the file contents/extension
    unless overridden. This is thin sugar over read_book() + save_book().

    Args:
        src: Source order-book file path.
        dst: Destination file path.
        symbol: Trading symbol (required if src carries no HDF5 metadata).
        to_layout: Output layout ('wide' or 'long').
        src_layout: Input layout ('auto', 'wide', 'long').
        src_format: Source format ('csv'/'parquet'/'hdf5'); None -> auto-detect.
        dst_format: Destination format; None -> auto-detect from dst extension.
        levels: Book levels K for the source (wide only; inferred otherwise).
        tick_size: Minimum price step propagated to the intermediate BookData.
        **save_kwargs: Extra arguments forwarded to save_book (ts_format,
            hdf5_key, compression, metadata).

    Returns:
        BookData: The intermediate (wide) BookData, for convenience.

    Example:
        convert_book('btc_bookdepth.csv', 'btc_book.h5', symbol='BTCUSDT')
        convert_book('btc_book.h5', 'btc_bookdepth.csv', to_layout='long')
    """
    book = read_book(
        src, symbol=symbol, levels=levels, format=src_format,
        layout=src_layout, tick_size=tick_size,
    )
    dst_format = dst_format or _detect_format(Path(dst))
    save_book(book, dst, format=dst_format, layout=to_layout, **save_kwargs)
    return book

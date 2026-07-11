from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal, Union

from tradetropy.data.data import TickProxy, OhlcProxy
from tradetropy.core.data_types import TickData, KlineData, BookData, MboData
from tradetropy.exceptions import TradingError, ConnectionError
from tradetropy.exceptions import DataError


_Format   = Literal["csv", "parquet", "hdf5", "npz"]
_TsFormat = Literal["iso", "ms", "s"]
_Layout   = Literal["auto", "wide", "long"]


def _require_tables() -> None:
    """
    Ensure PyTables is importable before using the HDF5 path.

    HDF5 is an optional format (``pip install tradetropy[hdf5]``): PyTables cannot
    build on some platforms (notably Termux/Android), so the base install ships
    without it and uses the NumPy ``.npz`` binary format instead. This raises a
    clear, actionable error when the HDF5 path is reached without PyTables,
    rather than surfacing pandas' generic backend error.

    Raises:
        DataError: If PyTables (the ``tables`` package) is not installed.
    """
    try:
        import tables  # noqa: F401
    except ImportError as exc:
        raise DataError(
            "HDF5 support requires PyTables, an optional extra. Install it with "
            "'pip install tradetropy[hdf5]', or use format='npz' (the base binary "
            "format, no extra needed). On Termux prefer 'npz': PyTables cannot "
            "build there."
        ) from exc


def _ensure_parent_dir(path: "str | Path") -> Path:
    """
    Ensure parent directory exists, creating it if necessary.

    Centralizes the pattern Path(path).parent.mkdir(parents=True,
    exist_ok=True) to allow users to pass paths with non-existent
    directories that are created automatically.

    Args:
        path: File or directory path to normalize.

    Returns:
        Path: Normalized Path object for chaining.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_hdf5_attrs(path, key, attrs: dict) -> None:
    """Write tradetropy metadata as HDF5 table attributes."""
    try:
        with pd.HDFStore(path, mode="a") as store:
            storer = store.get_storer(key)
            if storer is not None:
                for k, v in attrs.items():
                    setattr(storer.attrs, k, v)
    except Exception:
        pass


def _read_hdf5_attrs(path, key) -> dict:
    """Read tradetropy metadata from HDF5 table attributes."""
    try:
        with pd.HDFStore(path, mode="r") as store:
            storer = store.get_storer(key)
            if storer is None:
                return {}
            names = storer.attrs._f_list()
            return {
                k: getattr(storer.attrs, k)
                for k in names
                if k.startswith("tradetropy_")
            }
    except Exception:
        return {}


# -----
# NumPy .npz backend (base binary format; no PyTables/HDF5 required)
# -----

def _write_npz(df: pd.DataFrame, path: "str | Path", metadata: "dict | None") -> None:
    """
    Write a numeric DataFrame to a NumPy ``.npz`` file with embedded metadata.

    The frame is stored as a single 2D float64 matrix plus a unicode array of
    column names and a JSON metadata string. No pickle is used, so the file is
    portable and safe to load. This is the base binary format (a dependency-free
    alternative to HDF5 that works everywhere NumPy does, including Termux).

    Args:
        df: DataFrame with all-numeric columns (the canonical ``ts``-first
            layout, raw millisecond timestamps).
        path: Destination ``.npz`` file path.
        metadata: Optional tradetropy metadata dict (tradetropy_-prefixed keys),
            embedded so the reader can recover symbol/levels/interval, etc.
    """
    path = _ensure_parent_dir(path)
    cols = [str(c) for c in df.columns]
    arr = df.to_numpy(dtype=np.float64)
    payload = {
        "_columns": np.array(cols),
        "_data": arr,
        "_meta": np.array(json.dumps(metadata or {})),
    }
    # Open the handle ourselves so np.savez does not append a second ".npz".
    with open(path, "wb") as fh:
        np.savez(fh, **payload)


def _read_npz(path: "str | Path") -> pd.DataFrame:
    """
    Read a ``.npz`` file written by :func:`_write_npz` back to a DataFrame.

    Args:
        path: Source ``.npz`` file path.

    Returns:
        pd.DataFrame: The stored matrix with its original column names.
    """
    with np.load(path) as z:
        cols = [str(c) for c in z["_columns"]]
        data = np.asarray(z["_data"], dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, len(cols))
    return pd.DataFrame(data, columns=cols)


def _read_npz_attrs(path: "str | Path") -> dict:
    """
    Read the embedded tradetropy metadata dict from a ``.npz`` file.

    Args:
        path: Source ``.npz`` file path.

    Returns:
        dict: The tradetropy_-prefixed metadata, or an empty dict if absent.
    """
    try:
        with np.load(path) as z:
            if "_meta" in z.files:
                return json.loads(str(z["_meta"]))
    except Exception:
        return {}
    return {}


def _read_attrs(path: "str | Path", format: _Format, key: str) -> dict:
    """
    Read tradetropy metadata for any format that carries it (hdf5 / npz).

    Args:
        path: File path.
        format: Resolved file format.
        key: HDF5 key (ignored for npz).

    Returns:
        dict: The tradetropy metadata, or an empty dict for formats without attrs.
    """
    if format == "hdf5":
        return _read_hdf5_attrs(Path(path), key)
    if format == "npz":
        return _read_npz_attrs(path)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Write
# ══════════════════════════════════════════════════════════════════════════════
def save_ticks(
    data: "TickProxy | np.ndarray | pd.DataFrame",
    path: str | Path,
    format: "_Format | None" = None,
    ts_format: _TsFormat = "iso",
    hdf5_key: str = "data",
    compression: str = "snappy",
    metadata: "dict | None" = None,
) -> pd.DataFrame:
    """
    Save ticks to file.

    Args:
        data: TickProxy, ndarray [N×7] or DataFrame with standard tick columns.
        path: Destination file path.
        format: Output format ('csv', 'parquet', 'hdf5', 'npz'). None (default)
            infers from the extension, falling back to 'npz'.
        ts_format: Timestamp format for CSV ('iso', 'ms', 's').
        hdf5_key: Key inside the HDF5 file.
        compression: Parquet compression method ('snappy', 'gzip', 'zstd', None).
        metadata: Optional dict of tradetropy metadata to store as HDF5 attrs.

    Returns:
        pd.DataFrame: The saved tick data as DataFrame.

    Example:
        save_ticks(self.btc_tick, 'ticks.npz')
        save_ticks(self.btc_tick, 'ticks.parquet', format='parquet')
        save_ticks(self.btc_tick, 'session.h5', format='hdf5', hdf5_key='btc/ticks')
        save_ticks(my_array, 'ticks.csv', format='csv')
    """
    from tradetropy.core.constants import TICK_COLS, N_TICK_COLS

    format = _resolve_write_format(path, format)
    df = _source_to_df_ticks(data, N_TICK_COLS, TICK_COLS)
    if format == "csv":
        df = _ts_to_datetime(df, ts_format)
    elif format != "npz":
        df = _ts_to_datetime(df, "ms")
    # npz keeps the raw millisecond 'ts' column (read_ticks handles both).
    _write(df, path, format, hdf5_key, compression, metadata)
    if format == "hdf5" and metadata:
        _write_hdf5_attrs(Path(path), hdf5_key, metadata)
    return df

def save_klines(
    data: "OhlcProxy | np.ndarray | pd.DataFrame",
    path: str | Path,
    format: "_Format | None" = None,
    include_partial: bool = True,
    ts_format: _TsFormat = "iso",
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


# ══════════════════════════════════════════════════════════════════════════════
# Read
# ══════════════════════════════════════════════════════════════════════════════
def read_ticks(
    path: str | Path,
    symbol: "str | None" = None,
    format: _Format | None = None,
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
) -> "TickData":
    """
    Read ticks from file and return TickData.

    Accepts external files (broker CSV) and files generated by save_ticks().

    Required columns: datetime, bid, ask
    Optional columns: volume, flags, volume_real, price
                      (filled by normalize_ticks() if missing)

    Args:
        path: File path to read from.
        symbol: Trading symbol. If None, read from HDF5 attrs (saved by
            TickData.save()). Required for CSV/Parquet or non-tradetropy files.
        format: File format ('csv', 'parquet', 'hdf5'). None -> auto-detect.
        hdf5_key: Key inside HDF5 file.
        col_datetime: Datetime column name.
        tick_size: Minimum price move.
        tick_value: Value of one tick.
        contract_size: Size of contract.
        digits: Number of decimal places.
        avg_spread: Average bid-ask spread.
        volume_min: Minimum order volume.
        volume_max: Maximum order volume.
        volume_step: Minimum volume increment (lot step).

    Returns:
        TickData: Normalized tick data object.

    Example:
        ticks = read_ticks('ticks.parquet', 'BTCUSDT')
        ticks = read_ticks('session.h5')  # symbol read from attrs
        ticks = read_ticks('broker_export.csv', 'BTCUSDT')
    """
    from tradetropy.core.data_types import TickData
    from tradetropy.data import normalize_ticks

    path    = Path(path)
    format  = format or _detect_format(path)

    if symbol is None and format in ("hdf5", "npz"):
        attrs = _read_attrs(path, format, hdf5_key)
        symbol = attrs.get("tradetropy_symbol")
        if symbol is not None:
            tick_size   = attrs.get("tradetropy_tick_size", tick_size)
            tick_value  = attrs.get("tradetropy_tick_value", tick_value)
            contract_size = attrs.get("tradetropy_contract_size", contract_size)
            digits      = attrs.get("tradetropy_digits", digits)
            avg_spread  = attrs.get("tradetropy_avg_spread", avg_spread)
            volume_min  = attrs.get("tradetropy_volume_min", volume_min)
            volume_max  = attrs.get("tradetropy_volume_max", volume_max)
            volume_step = attrs.get("tradetropy_volume_step", volume_step)

    if symbol is None:
        raise ValueError(
            "symbol is required. Pass it explicitly or use a file saved by "
            "TickData.save() with format='hdf5'."
        )

    df = _read(path, format, hdf5_key)

    if "ts" in df.columns:
        ts_ms = df["ts"].to_numpy(dtype=np.int64)
        df = df.drop(columns=["ts"])
    else:
        ts_ms = _datetime_to_ts_ms(df[col_datetime])
        df = df.drop(columns=[col_datetime])

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
        _col("bid"), _col("ask"), _col("volume"), _col("flags"),
        _col("volume_real", np.nan), _col("price"),
    ])

    tick_data = TickData(
        symbol=symbol, data=data,
        tick_size=tick_size, tick_value=tick_value,
        contract_size=contract_size, digits=digits,
        avg_spread=avg_spread,
        volume_min=volume_min, volume_max=volume_max,
        volume_step=volume_step,
    )
    if "ts" not in df.columns:
        tick_data.data = normalize_ticks(tick_data.data)
    return tick_data


# ------------------------------------------------------------------------------
# Trades (times and sales) -- per-source schemas
# ------------------------------------------------------------------------------
def _binance_is_buyer_maker_to_flags(column: pd.Series) -> np.ndarray:
    """
    Convert a Binance 'is_buyer_maker' column to tradetropy aggressor flags.

    The aggressor is the taker, which is the OPPOSITE side of the maker:
        - is_buyer_maker=True  -> the buyer is the maker (passive), so the
          aggressor is the seller -> flags = -1 (sell).
        - is_buyer_maker=False -> the aggressor is the buyer -> flags = +1 (buy).

    Accepts booleans, integers or the string forms 'true'/'false' (any case),
    since CSV readers may deliver any of these.

    Args:
        column (pd.Series): The 'is_buyer_maker' column from the trades file.

    Returns:
        np.ndarray: Float flags array [N] with +1.0 (buy) or -1.0 (sell), using
            the tradetropy sign convention (+1 buy / -1 sell / 0 unknown).
    """
    text = column.astype(str).str.strip().str.lower()
    is_maker = text.isin(("true", "1", "t", "yes")).to_numpy()
    return np.where(is_maker, -1.0, 1.0)


# Registry of external trades schemas keyed by source. Each entry maps the
# tradetropy tick fields to the source column names and provides a per-source
# aggressor-side conversion. Add a new source by adding one entry here; the
# generic read_trades() below needs no change.
_TRADE_SCHEMAS: "dict[str, dict]" = {
    "binance": {
        "time":          "time",
        "price":         "price",
        "volume":        "qty",
        "side":          "is_buyer_maker",
        "side_to_flags": _binance_is_buyer_maker_to_flags,
    },
}


def read_trades(
    path: str | Path,
    symbol: str,
    source: str = "binance",
    format: _Format | None = None,
    hdf5_key: str = "data",
    tick_size: float = 0.01,
    tick_value: float = 0.01,
    contract_size: float = 1.0,
    digits: int = 2,
    avg_spread: float = 0.0,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> "TickData":
    """
    Read raw exchange trades (times and sales) and return TickData.

    Normalizes a venue-specific trades export into the tradetropy tick array,
    filling bid/ask from the trade price (a trades file carries no quotes) and
    mapping the aggressor side onto the 'flags' column (+1 buy / -1 sell). The
    supported schemas live in the _TRADE_SCHEMAS registry, keyed by 'source';
    'binance' handles the columns id, price, qty, quote_qty, time,
    is_buyer_maker (the id and quote_qty columns are ignored).

    The returned TickData is a standard object, so it is saved to any format
    with its usual .save() (parquet / hdf5 / csv) and re-read with read_ticks().

    Args:
        path: File path to read from.
        symbol: Trading symbol (required; a trades file has no tradetropy attrs).
        source: Trades schema key. One of the keys in _TRADE_SCHEMAS.
        format: File format ('csv', 'parquet', 'hdf5'). None -> auto-detect.
        hdf5_key: Key inside the HDF5 file.
        tick_size: Minimum price move.
        tick_value: Value of one tick.
        contract_size: Size of contract.
        digits: Number of decimal places.
        avg_spread: Average bid-ask spread.
        volume_min: Minimum order volume.
        volume_max: Maximum order volume.
        volume_step: Minimum volume increment (lot step).

    Returns:
        TickData: Normalized tick data object.

    Raises:
        DataError: If the source is unknown or a required column is missing.

    Example:
        ticks = read_trades('ADAUSDT-trades.csv', 'ADAUSD', source='binance')
        ticks.save('ada_ticks.parquet')
    """
    from tradetropy.core.data_types import TickData
    from tradetropy.data import normalize_ticks

    schema = _TRADE_SCHEMAS.get(source.lower())
    if schema is None:
        raise DataError(
            f"Unknown trades source {source!r}. "
            f"Available sources: {sorted(_TRADE_SCHEMAS)}."
        )

    path   = Path(path)
    format = format or _detect_format(path)
    df     = _read(path, format, hdf5_key)

    cols_map = {c.lower(): c for c in df.columns}

    def _need(field: str) -> pd.Series:
        real = cols_map.get(schema[field].lower())
        if real is None:
            raise DataError(
                f"{source!r} trades file is missing required column "
                f"{schema[field]!r} (for '{field}'). "
                f"Found columns: {list(df.columns)}."
            )
        return df[real]

    ts_ms  = _datetime_to_ts_ms(_need("time"))
    price  = _need("price").to_numpy(dtype=np.float64)
    volume = _need("volume").to_numpy(dtype=np.float64)
    flags  = schema["side_to_flags"](_need("side"))

    n = len(df)
    # Column order must match TICK_COLS: ts, bid, ask, volume, flags,
    # volume_real, price. bid/ask are left at 0.0 so normalize_ticks() fills
    # them from price (a trades file has no quotes); volume_real is NaN.
    data = np.column_stack([
        ts_ms.astype(np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        volume,
        flags,
        np.full(n, np.nan, dtype=np.float64),
        price,
    ])

    tick_data = TickData(
        symbol=symbol, data=data,
        tick_size=tick_size, tick_value=tick_value,
        contract_size=contract_size, digits=digits,
        avg_spread=avg_spread,
        volume_min=volume_min, volume_max=volume_max,
        volume_step=volume_step,
    )
    tick_data.data = normalize_ticks(tick_data.data)
    return tick_data

def read_klines(
    path: str | Path,
    symbol: "str | None" = None,
    timeframe: "str | int | None" = None,
    format: _Format | None = None,
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


# -----
# Compatibility Aliases
# -----

def ticks_to_file(data, path, **kwargs):
    """Compatibility alias for save_ticks(). Use save_ticks() for new code."""
    return save_ticks(data, path, **kwargs)

def klines_to_file(data, path, **kwargs):
    """Compatibility alias for save_klines(). Use save_klines() for new code."""
    return save_klines(data, path, **kwargs)

def ticks_from_file(path, symbol, **kwargs):
    """Compatibility alias for read_ticks(). Use read_ticks() for new code."""
    return read_ticks(path, symbol, **kwargs)

def klines_from_file(path, symbol, timeframe=None, **kwargs):
    """Compatibility alias for read_klines(). Use read_klines() for new code."""
    return read_klines(path, symbol, timeframe, **kwargs)

def read_klines_csv(path, symbol, timeframe=None, **kwargs):
    """Compatibility alias for read_klines(format='csv'). Use read_klines() for new code."""
    return read_klines(path, symbol, timeframe, format="csv", **kwargs)

def read_ticks_csv(path, symbol, **kwargs):
    """Compatibility alias for read_ticks(format='csv'). Use read_ticks() for new code."""
    return read_ticks(path, symbol, format="csv", **kwargs)

def book_to_file(data, path, **kwargs):
    """Compatibility alias for save_book(). Use save_book() for new code."""
    return save_book(data, path, **kwargs)

def book_from_file(path, symbol=None, **kwargs):
    """Compatibility alias for read_book(). Use read_book() for new code."""
    return read_book(path, symbol, **kwargs)

def save_proxy(proxy, path, format="parquet", **kwargs):
    """
    Save TickProxy or OhlcProxy to file.

    Automatically dispatches to save_ticks() or save_klines() based on proxy type.

    Args:
        proxy: TickProxy or OhlcProxy instance.
        path: Destination file path.
        format: Output format ('csv', 'parquet', 'hdf5').
        **kwargs: Additional arguments passed to save_ticks() or save_klines().

    Raises:
        TradingError: If proxy type is not TickProxy or OhlcProxy.
    """

# -----
# Internal Helpers
# -----

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

def _detect_format(path: Path) -> _Format:
    """
    Auto-detect file format from path extension.

    Args:
        path: File path to analyze.

    Returns:
        _Format: Detected format ('csv', 'parquet', 'hdf5').

    Raises:
        DataError: If file extension is not recognized.
    """
    ext = path.suffix.lower()
    if ext == ".csv":                   return "csv"
    if ext == ".parquet":               return "parquet"
    if ext == ".npz":                   return "npz"
    if ext in (".h5", ".hdf", ".hdf5"): return "hdf5"
    raise DataError(
        f"Cannot infer format from '{path.name}'. "
        f"Specify format='csv'|'parquet'|'hdf5'|'npz'."
    )


def _resolve_write_format(path: "str | Path", format: "_Format | None") -> _Format:
    """
    Resolve the on-disk write format for a save call.

    An explicit ``format`` wins. Otherwise it is inferred from the file
    extension (``.npz`` / ``.csv`` / ``.parquet`` / ``.h5``); an unknown or
    missing extension falls back to ``npz``, the base binary format (so a save
    works out of the box without the optional HDF5/Parquet extras).

    Args:
        path: Destination file path.
        format: Explicit format, or None to infer.

    Returns:
        _Format: The resolved format.
    """
    if format is not None:
        return format
    try:
        return _detect_format(Path(path))
    except DataError:
        return "npz"


def _read(path: Path, format: _Format, hdf5_key: str) -> pd.DataFrame:
    """
    Read data file in specified format.

    Args:
        path: File path to read.
        format: File format ('csv', 'parquet', 'hdf5').
        hdf5_key: Key inside HDF5 file (ignored for other formats).

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        DataError: If format is invalid.
    """
    if format == "csv":     return pd.read_csv(path)
    if format == "parquet": return pd.read_parquet(path)
    if format == "npz":     return _read_npz(path)
    if format == "hdf5":
        _require_tables()
        return pd.read_hdf(path, key=hdf5_key)
    raise DataError(f"Unknown format: {format!r}.")

def _write(
    df: pd.DataFrame,
    path: str | Path,
    format: _Format,
    hdf5_key: str,
    compression: str,
    metadata: "dict | None" = None,
) -> None:
    """
    Write DataFrame to file in specified format.

    Creates parent directories if they don't exist.

    Args:
        df: DataFrame to write.
        path: Output file path.
        format: File format ('csv', 'parquet', 'hdf5', 'npz').
        hdf5_key: Key inside HDF5 file (ignored for other formats).
        compression: Parquet compression method.
        metadata: Optional tradetropy metadata; embedded in the file for 'npz'
            (HDF5 writes it separately via _write_hdf5_attrs).

    Raises:
        DataError: If format is invalid.
    """
    path = _ensure_parent_dir(path)
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, index=False, compression=compression)
    elif format == "npz":
        _write_npz(df, path, metadata)
    elif format == "hdf5":
        _require_tables()
        # Drop the key first if it already exists (re-saving with a changed
        # schema/dtype, e.g. after "improving" a symbol's parameters, must
        # REPLACE the table rather than let PyTables reconcile two schemas -
        # a mismatch there is what silently drops the tradetropy_* attrs written
        # by _write_hdf5_attrs on a previous save).
        if Path(path).exists():
            try:
                with pd.HDFStore(path, mode="a") as store:
                    if hdf5_key in store:
                        store.remove(hdf5_key)
            except Exception:
                pass
        df.to_hdf(
            path, key=hdf5_key, mode="a", format="table",
            complevel=5, complib="blosc", data_columns=True,
        )
    else:
        raise DataError(f"Unknown format: {format!r}.")

# -----
# Internal Append Helpers (Live Recording)
# -----
#
# The record= path appends buffered events to disk repeatedly during a live
# session, then a final flush runs on stop(). Two on-disk record backends:
#
# - hdf5 (.h5/.hdf5): PyTables native append (format='table', append=True).
#   Requires the optional [hdf5] extra.
# - npz (.npz): the base binary format. .npz itself is NOT appendable (adding
#   rows would rewrite the whole file, O(N^2) over a session), so during the
#   session rows are appended to a raw float64 sidecar '<path>.part' with a
#   '<path>.meta' JSON header (columns + metadata). engine.stop() calls
#   _consolidate_npz_record() once after the final flush to turn the sidecar
#   into the final .npz. This keeps per-flush cost O(rows-in-flush), constant.

def _npz_record_sidecars(path: "str | Path") -> "tuple[Path, Path]":
    """Return the (raw-part, meta) sidecar paths for an .npz record file."""
    p = str(path)
    return Path(p + ".part"), Path(p + ".meta")


def _append_record_npz(
    arr: np.ndarray, path: "str | Path", columns: "tuple | list",
    metadata: "dict | None" = None,
) -> None:
    """
    Append raw rows to an .npz record's binary sidecar (appendable, O(rows)).

    Writes the column layout and metadata once to a '<path>.meta' JSON header
    on the first call, then appends the float64 rows to '<path>.part'. The
    sidecars are consolidated into the final .npz by _consolidate_npz_record().

    Args:
        arr: Event rows [N x >=len(columns)] (extra columns are dropped).
        path: Destination .npz file path.
        columns: Canonical column names for the stored matrix.
        metadata: Optional tradetropy metadata, persisted once in the header.
    """
    path = _ensure_parent_dir(path)
    part, meta = _npz_record_sidecars(path)
    width = len(columns)
    if not meta.exists():
        meta.write_text(json.dumps({
            "columns": [str(c) for c in columns],
            "width": int(width),
            "metadata": metadata or {},
        }))
    elif metadata:
        # A later flush carrying metadata (e.g. levels resolved lazily) updates
        # the header if the first one had none.
        try:
            info = json.loads(meta.read_text())
            if not info.get("metadata"):
                info["metadata"] = metadata
                meta.write_text(json.dumps(info))
        except Exception:
            pass
    rows = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)[:, :width])
    with open(part, "ab") as fh:
        rows.tofile(fh)


def _consolidate_npz_record(path: "str | Path") -> None:
    """
    Consolidate an .npz record's binary sidecars into the final .npz file.

    Reads '<path>.part' (raw float64 rows) and '<path>.meta' (columns +
    metadata), writes a proper .npz via _write_npz and removes the sidecars.
    A no-op when there are no sidecars (e.g. an HDF5 recording, or nothing was
    ever flushed).

    Args:
        path: The .npz record file path passed to record=.
    """
    path = Path(path)
    part, meta = _npz_record_sidecars(path)
    if not meta.exists():
        return
    try:
        info = json.loads(meta.read_text())
    except Exception:
        return
    cols = info["columns"]
    width = int(info["width"])
    metadata = info.get("metadata") or {}
    if part.exists():
        raw = np.fromfile(part, dtype=np.float64)
    else:
        raw = np.empty(0, dtype=np.float64)
    data = raw.reshape(-1, width) if raw.size else np.empty((0, width), dtype=np.float64)
    _write_npz(pd.DataFrame(data, columns=cols), path, metadata)
    part.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)


# -- Format-dispatching append entry points (used by live/_loop.py) -----------

def _append_ticks(arr: np.ndarray, path: "str | Path", metadata: "dict | None" = None) -> None:
    """Append tick rows to a record file, routing by extension (.npz / .h5)."""
    from tradetropy.core.constants import TICK_COLS, N_TICK_COLS
    fmt = _detect_format(Path(path))
    if fmt == "npz":
        _append_record_npz(arr[:, :N_TICK_COLS], path, TICK_COLS, metadata)
    elif fmt == "hdf5":
        _append_ticks_hdf5(arr, Path(path), metadata)
    else:
        raise DataError(f"record= path must be .npz or .h5/.hdf5, got '{path}'.")


def _append_klines(arr: np.ndarray, path: "str | Path", metadata: "dict | None" = None) -> None:
    """Append kline rows to a record file, routing by extension (.npz / .h5)."""
    from tradetropy.core.constants import OHLC_COLS, N_OHLC_COLS
    fmt = _detect_format(Path(path))
    if fmt == "npz":
        _append_record_npz(arr[:, :N_OHLC_COLS], path, OHLC_COLS, metadata)
    elif fmt == "hdf5":
        _append_klines_hdf5(arr, Path(path), metadata)
    else:
        raise DataError(f"record= path must be .npz or .h5/.hdf5, got '{path}'.")


def _append_book(
    arr: np.ndarray, path: "str | Path", levels: int, metadata: "dict | None" = None
) -> None:
    """Append order-book rows to a record file, routing by extension."""
    from tradetropy.core.data_types import book_flat_columns, book_row_width
    fmt = _detect_format(Path(path))
    if fmt == "npz":
        cols = book_flat_columns(levels)
        meta = {"tradetropy_levels": int(levels)}
        if metadata:
            meta.update(metadata)
        _append_record_npz(arr[:, :book_row_width(levels)], path, cols, meta)
    elif fmt == "hdf5":
        _append_book_hdf5(arr, Path(path), levels, metadata)
    else:
        raise DataError(f"record= path must be .npz or .h5/.hdf5, got '{path}'.")


def _append_mbo(arr: np.ndarray, path: "str | Path", metadata: "dict | None" = None) -> None:
    """Append MBO rows to a record file, routing by extension (.npz / .h5)."""
    from tradetropy.core.data_types import MBO_COLS, N_MBO_COLS
    fmt = _detect_format(Path(path))
    if fmt == "npz":
        _append_record_npz(arr[:, :N_MBO_COLS], path, MBO_COLS, metadata)
    elif fmt == "hdf5":
        _append_mbo_hdf5(arr, Path(path), metadata)
    else:
        raise DataError(f"record= path must be .npz or .h5/.hdf5, got '{path}'.")


# -----
# Internal HDF5 Append Helpers (Live Recording)
# -----

def _append_ticks_hdf5(
    arr: np.ndarray, path: Path, metadata: "dict | None" = None
) -> None:
    """
    Append tick array to an HDF5 file.

    Creates the file and directory structure if they don't exist.
    Compatible with read_ticks() - uses same key and column layout.

    Args:
        arr: Tick array with shape [N×7].
        path: HDF5 file path.
        metadata: Optional tradetropy metadata (e.g. symbol) persisted once as
            table attributes so read_ticks() can recover it without an explicit
            symbol argument.
    """
    from tradetropy.core.constants import TICK_COLS, N_TICK_COLS
    _require_tables()
    path = _ensure_parent_dir(path)
    df = pd.DataFrame(arr[:, :N_TICK_COLS], columns=TICK_COLS)
    df.to_hdf(
        path, key="data", mode="a", format="table",
        append=True, complevel=5, complib="blosc",
        data_columns=True, index=False,
    )
    if metadata:
        _write_hdf5_attrs(path, "data", metadata)


def _append_klines_hdf5(
    arr: np.ndarray, path: Path, metadata: "dict | None" = None
) -> None:
    """
    Append kline array to an HDF5 file.

    Creates the file and directory structure if they don't exist.
    Compatible with read_klines() - uses same key and column layout.

    Args:
        arr: Kline array with shape [N×6].
        path: HDF5 file path.
        metadata: Optional tradetropy metadata (e.g. symbol, interval_ms) persisted
            once as table attributes so read_klines() can recover the symbol and
            timeframe without explicit arguments.
    """
    from tradetropy.core.constants import OHLC_COLS, N_OHLC_COLS
    _require_tables()
    path = _ensure_parent_dir(path)
    df = pd.DataFrame(arr[:, :N_OHLC_COLS], columns=OHLC_COLS)
    df.to_hdf(
        path, key="data", mode="a", format="table",
        append=True, complevel=5, complib="blosc",
        data_columns=True, index=False,
    )
    if metadata:
        _write_hdf5_attrs(path, "data", metadata)


def _append_book_hdf5(
    arr: np.ndarray, path: Path, levels: int, metadata: "dict | None" = None
) -> None:
    """
    Append L2 order-book rows to an HDF5 file.

    Creates the file and directory structure if they don't exist. Rows use the
    flat layout from core.data_types.book_flat_columns and round-trip via
    read_book(). The level count is stored once as table metadata so read_book()
    can reconstruct the BookData without being told K.

    Args:
        arr: Book rows [N x (2 + 4*levels)].
        path: HDF5 file path.
        levels: Number of book levels K per side.
        metadata: Optional extra tradetropy metadata (e.g. symbol) persisted as
            table attributes so read_book() can recover it.
    """
    from tradetropy.core.data_types import book_flat_columns, book_row_width
    _require_tables()
    path = _ensure_parent_dir(path)
    width = book_row_width(levels)
    cols = book_flat_columns(levels)
    df = pd.DataFrame(np.asarray(arr, dtype=np.float64)[:, :width], columns=cols)
    df.to_hdf(
        path, key="book", mode="a", format="table",
        append=True, complevel=5, complib="blosc",
        data_columns=True, index=False,
    )
    # Persist the level count (and any extra metadata) as table attributes.
    attrs = {"tradetropy_levels": int(levels)}
    if metadata:
        attrs.update(metadata)
    _write_hdf5_attrs(path, "book", attrs)


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
    format: "_Format | None" = None,
    *,
    levels: "int | None" = None,
    layout: _Layout = "wide",
    ts_format: _TsFormat = "iso",
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
    format: _Format | None = None,
    layout: _Layout = "auto",
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
    to_layout: _Layout = "wide",
    src_layout: _Layout = "auto",
    src_format: _Format | None = None,
    dst_format: _Format | None = None,
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


def _append_mbo_hdf5(
    arr: np.ndarray, path: Path, metadata: "dict | None" = None
) -> None:
    """
    Append L3 / MBO event rows to an HDF5 file (key 'mbo').

    Rows use the flat layout from core.data_types.MBO_COLS and round-trip via
    read_mbo().

    Args:
        arr: MBO rows [M x 6].
        path: HDF5 file path.
        metadata: Optional tradetropy metadata (e.g. symbol) persisted once as
            table attributes so read_mbo() can recover it.
    """
    from tradetropy.core.data_types import MBO_COLS, N_MBO_COLS
    _require_tables()
    path = _ensure_parent_dir(path)
    df = pd.DataFrame(np.asarray(arr, dtype=np.float64)[:, :N_MBO_COLS], columns=MBO_COLS)
    df.to_hdf(
        path, key="mbo", mode="a", format="table",
        append=True, complevel=5, complib="blosc",
        data_columns=True, index=False,
    )
    if metadata:
        _write_hdf5_attrs(path, "mbo", metadata)


def read_mbo(
    path: "str | Path",
    symbol: "str | None" = None,
    hdf5_key: str = "mbo",
    tick_size: float = 0.01,
):
    """
    Read recorded L3 / MBO event rows and return an MboData.

    Args:
        path: HDF5 file path (as written by _append_mbo_hdf5).
        symbol: Trading symbol. If None, read from HDF5 attrs (persisted by the
            live record= path). Required when the file carries no attrs.
        hdf5_key: Key inside the HDF5 file (default 'mbo').
        tick_size: Minimum price step propagated to MboData.

    Returns:
        MboData: Reconstructed market-by-order data.

    Raises:
        ValueError: If the symbol cannot be resolved from arg or attrs.
    """
    from tradetropy.core.data_types import MboData, MBO_COLS

    path = Path(path)
    format = _detect_format(path)
    if symbol is None:
        attrs = _read_attrs(path, format, hdf5_key)
        symbol = attrs.get("tradetropy_symbol")
        tick_size = attrs.get("tradetropy_tick_size", tick_size)
    if symbol is None:
        raise ValueError(
            "symbol is required. Pass it explicitly or use a file recorded by "
            "the live record= path (which persists it as HDF5 attrs)."
        )

    df = _read(path, format, hdf5_key)
    data = df[list(MBO_COLS)].to_numpy(dtype=np.float64)
    return MboData(symbol=symbol, data=data, tick_size=tick_size)
    
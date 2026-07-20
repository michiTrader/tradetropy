"""
Tick data IO: save_ticks / read_ticks, the MT5 tick-export reader and
read_trades (raw exchange trades / times-and-sales, per-source schemas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal

from tradetropy.exceptions import DataError

from tradetropy.io._backends import (
    _detect_format, _resolve_write_format, _read, _write,
    _read_attrs, _write_hdf5_attrs,
)
from tradetropy.io._common import (
    _source_to_df_ticks, _ts_to_datetime, _datetime_to_ts_ms,
)


def save_ticks(
    data: "TickProxy | np.ndarray | pd.DataFrame",
    path: str | Path,
    format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
    ts_format: Literal["iso", "ms", "s"] = "iso",
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


# MT5 tick export header, as written by MetaTrader 5's "Export Ticks" tool.
# Tab-separated, with <DATE> and <TIME> as two separate columns (no single
# 'datetime' column) - this is what read_ticks() sniffs to recognize the file
# and suggest source='mt5' instead of failing with a raw KeyError.
_MT5_TICK_HEADER_COLS = (
    "<DATE>", "<TIME>", "<BID>", "<ASK>", "<LAST>", "<VOLUME>", "<FLAGS>",
)

# MT5 <FLAGS> bitmask bits used for the aggressor side (bid/ask/last/volume
# bits vary by broker/build and are not needed here: this module reads
# whichever of bid/ask/last is non-empty in each row instead of decoding
# those bits).
_MT5_FLAG_BUY = 32
_MT5_FLAG_SELL = 64


def _sniff_mt5_tick_header(path: Path) -> bool:
    """
    Check whether a CSV's first line is the MT5 tick export header.

    Args:
        path: CSV file path.

    Returns:
        bool: True if the first line matches the MT5 <DATE>/<TIME>/<BID>/
            <ASK>/<LAST>/<VOLUME>/<FLAGS> tab-separated header (a UTF-8 BOM,
            if present, is ignored).
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
    except OSError:
        return False
    cols = [c.strip() for c in first_line.split("\t")]
    return cols[: len(_MT5_TICK_HEADER_COLS)] == list(_MT5_TICK_HEADER_COLS)


def _mt5_aggressor_flags(mt5_flags: "np.ndarray") -> "np.ndarray":
    """
    Map the MT5 <FLAGS> bitmask to the tradetropy aggressor sign.

    Args:
        mt5_flags: Integer bitmask column from the MT5 export.

    Returns:
        np.ndarray: Float flags [N] with +1.0 (buy, bit 32), -1.0 (sell,
            bit 64) or 0.0 (neither bit set / unknown).
    """
    flags = mt5_flags.astype(np.int64)
    out = np.zeros(len(flags), dtype=np.float64)
    out[(flags & _MT5_FLAG_BUY) != 0] = 1.0
    out[(flags & _MT5_FLAG_SELL) != 0] = -1.0
    return out


def _read_mt5_ticks_csv(path: Path, symbol: str, **kwargs) -> "TickData":
    """
    Read an MT5 tick export ("Export Ticks" tool) and return TickData.

    MT5 tick exports use a fixed tab-separated schema with EMPTY fields left
    blank (not a variable column count): <DATE> <TIME> <BID> <ASK> <LAST>
    <VOLUME> <FLAGS>. Each row updates only a SUBSET of fields (a bid-only
    quote tick, a full bid+ask quote tick, or a last/volume trade tick), so
    bid/ask are forward-filled from the last known value - a row with a blank
    ask does not mean the ask disappeared, it means this tick did not touch
    it. Leading rows with no prior value to carry forward are back-filled from
    the first known bid/ask (their "opening price"), so the series never
    starts with a NaN that would crash downstream price handling. This
    stateful, MT5-specific semantics is why this is a dedicated reader rather
    than something normalize_ticks() (which only fills columns from OTHER
    columns within the SAME row) can express.

    The <FLAGS> bitmask's bid/ask/last/volume bits vary by broker/build, so
    this reader does not decode them for that purpose; it uses whichever of
    <BID>/<ASK>/<LAST> is non-empty in each row, which is unambiguous. Bits 32
    (buy) / 64 (sell) ARE used, to fill tradetropy's 'flags' aggressor column
    (+1 buy / -1 sell / 0 unknown), since a quote-only row has no other source
    for it.

    Args:
        path: MT5 tick export CSV path.
        symbol: Trading symbol (an MT5 export carries no tradetropy attrs).
        **kwargs: Forwarded to the TickData constructor (tick_size,
            tick_value, contract_size, digits, avg_spread, volume_min,
            volume_max, volume_step).

    Returns:
        TickData: Normalized tick data object.

    Example:
        ticks = read_ticks('MESU26_ticks.csv', 'MESU26', source='mt5')
    """
    from tradetropy.core.data_types import TickData
    from tradetropy.data import normalize_ticks

    df = pd.read_csv(
        path, sep="\t", encoding="utf-8-sig",
        dtype={"<DATE>": str, "<TIME>": str},
    )
    missing = [c for c in _MT5_TICK_HEADER_COLS if c not in df.columns]
    if missing:
        raise DataError(
            f"{path.name!r} does not look like an MT5 tick export: missing "
            f"columns {missing}. Found columns: {list(df.columns)}."
        )

    dt = pd.to_datetime(
        df["<DATE>"].str.strip() + " " + df["<TIME>"].str.strip(),
        format="%Y.%m.%d %H:%M:%S.%f",
    )
    ts_ms = dt.to_numpy(dtype="datetime64[ms]").astype(np.int64)

    # Forward-fill carries the last known bid/ask across rows that did not
    # touch them. The trailing .bfill() covers the leading edge: if the very
    # first rows are bid-only (or ask-only) ticks, there is no PRIOR value to
    # carry forward, so those cells would stay NaN and later crash downstream
    # (e.g. broker price normalization). Backfilling the first known value is
    # the correct "opening price" for those leading ticks.
    bid = df["<BID>"].ffill().bfill().to_numpy(dtype=np.float64)
    ask = df["<ASK>"].ffill().bfill().to_numpy(dtype=np.float64)
    last = df["<LAST>"].to_numpy(dtype=np.float64)
    volume = df["<VOLUME>"].to_numpy(dtype=np.float64)
    mt5_flags = df["<FLAGS>"].fillna(0).to_numpy(dtype=np.int64)

    n = len(df)
    data = np.column_stack([
        ts_ms.astype(np.float64),
        bid, ask, np.nan_to_num(volume, nan=0.0),
        _mt5_aggressor_flags(mt5_flags),
        np.full(n, np.nan, dtype=np.float64),
        last,
    ])

    # Fills 'price' from (bid+ask)/2 on quote-only rows (no <LAST>), matching
    # the generic CSV path's normalize_ticks() call - forward-filled bid/ask
    # is MT5-specific (handled above), but the same-row price/bid/ask fallback
    # is the generic rule and should behave identically here.
    tick_data = TickData(symbol=symbol, data=data, **kwargs)
    tick_data.data = normalize_ticks(tick_data.data)
    return tick_data


def read_ticks(
    path: str | Path,
    symbol: "str | None" = None,
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
        source: Tick file schema. 'auto' (default) reads the generic
            datetime/bid/ask/volume/flags/price schema; on failure it sniffs
            for a known broker-specific schema and raises an actionable error
            suggesting the right `source=` instead of a raw KeyError. 'mt5'
            reads an MT5 "Export Ticks" CSV directly (tab-separated <DATE>/
            <TIME>/<BID>/<ASK>/<LAST>/<VOLUME>/<FLAGS>, forward-filling bid/ask
            across rows) - see _read_mt5_ticks_csv() for the schema details.
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

    Raises:
        DataError: If source='mt5' is used on a file that is not an MT5 tick
            export, or if the generic schema is missing required columns and
            no known broker-specific schema is detected.

    Example:
        ticks = read_ticks('ticks.parquet', 'BTCUSDT')
        ticks = read_ticks('session.h5')  # symbol read from attrs
        ticks = read_ticks('broker_export.csv', 'BTCUSDT')
        ticks = read_ticks('MESU26_ticks.csv', 'MESU26', source='mt5')
    """
    from tradetropy.core.data_types import TickData
    from tradetropy.data import normalize_ticks

    path    = Path(path)
    format  = format or _detect_format(path)

    tick_kwargs = dict(
        tick_size=tick_size, tick_value=tick_value,
        contract_size=contract_size, digits=digits,
        avg_spread=avg_spread,
        volume_min=volume_min, volume_max=volume_max,
        volume_step=volume_step,
    )

    if source == "mt5":
        if symbol is None:
            raise ValueError(
                "symbol is required for source='mt5' (an MT5 export carries "
                "no tradetropy attrs)."
            )
        return _read_mt5_ticks_csv(path, symbol, **tick_kwargs)

    if source != "auto":
        raise DataError(
            f"Unknown source {source!r}. Available sources: 'auto', 'mt5'."
        )

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

    if "ts" not in df.columns and col_datetime not in df.columns:
        if format == "csv" and _sniff_mt5_tick_header(path):
            raise DataError(
                f"{path.name!r} looks like an MT5 tick export (columns "
                f"{list(_MT5_TICK_HEADER_COLS)}), not the generic tradetropy "
                f"tick schema (datetime, bid, ask, ...). "
                f"Re-run with source='mt5' to read it directly."
            )
        raise DataError(
            f"{path.name!r} is missing a {col_datetime!r} or 'ts' column. "
            f"Found columns: {list(df.columns)}. If this is a broker-specific "
            f"export, pass source=... (available: 'mt5') or convert it to the "
            f"generic schema first (datetime, bid, ask, volume, flags, price)."
        )

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

    tick_data = TickData(symbol=symbol, data=data, **tick_kwargs)
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
    format: 'Literal["csv", "parquet", "hdf5", "npz"] | None' = None,
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

"""
Live-recording append helpers (used by the ``record=`` path).

The record= path appends buffered events to disk repeatedly during a live
session, then a final flush runs on stop(). Two on-disk record backends:

- hdf5 (.h5/.hdf5): PyTables native append (format='table', append=True).
  Requires the optional [hdf5] extra.
- npz (.npz): the base binary format. .npz itself is NOT appendable (adding
  rows would rewrite the whole file, O(N^2) over a session), so during the
  session rows are appended to a raw float64 sidecar '<path>.part' with a
  '<path>.meta' JSON header (columns + metadata). engine.stop() calls
  _consolidate_npz_record() once after the final flush to turn the sidecar
  into the final .npz. This keeps per-flush cost O(rows-in-flush), constant.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pathlib import Path

from tradetropy.exceptions import DataError

from tradetropy.io._backends import (
    _ensure_parent_dir, _detect_format, _write_npz, _require_tables,
    _write_hdf5_attrs,
)


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

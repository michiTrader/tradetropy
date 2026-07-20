"""
Generic file-format backends shared by every IO submodule.

Covers auto-detecting a format from a path, reading/writing a DataFrame in
csv/parquet/hdf5/npz, and the metadata ("tradetropy attrs") sidecar for the
formats that carry it (hdf5 table attributes, npz embedded JSON).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal

from tradetropy.exceptions import DataError

# Internal alias kept for implementation-detail signatures in this package
# (public functions expose the expanded Literal[...] directly).
_Format = Literal["csv", "parquet", "hdf5", "npz"]


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
    path: "str | Path",
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

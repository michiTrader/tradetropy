"""
L3 / market-by-order (MBO) data IO: read_mbo.
"""

from __future__ import annotations

import numpy as np

from pathlib import Path

from tradetropy.io._backends import _detect_format, _read, _read_attrs


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

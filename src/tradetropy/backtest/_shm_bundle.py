"""
_shm_bundle.py - shared-memory transport for optimize() input matrices.

BacktestEngine.optimize() runs each candidate in a spawned worker process
(Windows/macOS). Previously the whole data bundle - including the full tick /
kline matrices - was pickled and shipped to every worker through the Pool
``initargs``, so an N-worker run made N pickled copies of the data travel over
IPC and sit in N separate process heaps. For a large series (millions of rows)
that is hundreds of MB of serialization and resident memory that buys nothing:
the data is read-only and identical for every worker.

This module moves the matrices into ``multiprocessing.shared_memory`` ONCE in
the parent and ships only tiny descriptors (name/shape/dtype) plus the small
metadata. Each worker maps the same block (zero copy) and rebuilds its
``TickData`` / ``KlineData`` around the shared view. It mirrors the design
already used by ``PoolBacktestEngine`` (pool.py) and relies on the same
invariant: workers only READ the input arrays.

The full pandas-free stats path and every other optimize behaviour are
unchanged; this only changes HOW the input data reaches the workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from tradetropy.core.data_types import TickData, KlineData

# Metadata fields carried alongside the shared matrix (everything a
# TickData/KlineData needs except the ``data`` array itself).
_META_FIELDS = (
    "symbol", "tick_size", "tick_value", "contract_size", "digits",
    "avg_spread", "volume_min", "volume_max", "volume_step",
)


@dataclass
class ShmMatrix:
    """Serializable descriptor of a matrix living in shared_memory."""

    name: str
    shape: tuple
    dtype: str

    def open(self):
        """Map the block. Returns (shm_handle, ndarray_view). Zero copies."""
        shm = shared_memory.SharedMemory(name=self.name)
        array = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return shm, array


def _allocate(array: np.ndarray, shm_refs: list) -> ShmMatrix:
    """Copy an array into a fresh shared_memory block; return its descriptor."""
    arr = np.ascontiguousarray(array, dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    dest = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    dest[:] = arr
    shm_refs.append(shm)
    return ShmMatrix(name=shm.name, shape=arr.shape, dtype=str(arr.dtype))


def _input_meta(inp) -> dict:
    meta = {f: getattr(inp, f) for f in _META_FIELDS}
    if isinstance(inp, KlineData):
        # Reconstruct from the normalized ms interval (not the original
        # timeframe string) so the child rebuilds an identical KlineData.
        meta["timeframe"] = inp.interval_ms
    return meta


def build_shm_bundle(strategy_cls, sesh, tick_inputs, kline_inputs, align_by_ts):
    """
    Build the picklable optimize data bundle with matrices in shared_memory.

    Returns (bundle, shm_refs). The caller MUST keep ``shm_refs`` alive for the
    whole optimization and call ``release_shm(shm_refs)`` when done.
    """
    shm_refs: list = []
    bundle = {
        "_shm_bundle": True,
        "strategy_cls": strategy_cls,
        "sesh": sesh,
        "align_by_ts": align_by_ts,
        "tick_meta": [_input_meta(t) for t in tick_inputs],
        "tick_blocks": [_allocate(t.data, shm_refs) for t in tick_inputs],
        "kline_meta": [_input_meta(k) for k in kline_inputs],
        "kline_blocks": [_allocate(k.data, shm_refs) for k in kline_inputs],
    }
    return bundle, shm_refs


def hydrate_bundle(bundle: dict, opened_shm: list) -> dict:
    """
    Rebuild the normal optimize data bundle in a worker from shared memory.

    Opened shm handles are appended to ``opened_shm`` (which the caller must
    keep alive for the worker's lifetime so the mapped views stay valid).
    """
    tick_inputs = []
    for meta, block in zip(bundle["tick_meta"], bundle["tick_blocks"]):
        shm, arr = block.open()
        opened_shm.append(shm)
        tick_inputs.append(TickData(data=arr, **meta))

    kline_inputs = []
    for meta, block in zip(bundle["kline_meta"], bundle["kline_blocks"]):
        shm, arr = block.open()
        opened_shm.append(shm)
        kline_inputs.append(KlineData(data=arr, **meta))

    return {
        "strategy_cls": bundle["strategy_cls"],
        "sesh": bundle["sesh"],
        "tick_inputs": tuple(tick_inputs),
        "kline_inputs": tuple(kline_inputs),
        "align_by_ts": bundle["align_by_ts"],
    }


def release_shm(shm_refs: list) -> None:
    """Close and unlink every shared_memory block created by the parent."""
    for shm in shm_refs:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    shm_refs.clear()

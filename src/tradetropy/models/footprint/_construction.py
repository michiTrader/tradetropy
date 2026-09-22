import numpy as np
from dataclasses import dataclass
from typing import Any

from ._store import FootprintStore
from ._ring import LiveFpRing
from ._config import FootprintConfig, _resolve_tick_size
from ._types import N_FP_SCALAR_COLS, N_FP_LEVEL_COLS, _FP_SCALAR_COL
from ._compute import _compute_scalars


def build_footprint_from_ticks(
    tick_matrix: np.ndarray,
    tick_to_candle_map: np.ndarray,
    n_closed_candles: int,
    config: FootprintConfig,
    _TICK_COL_: dict,
    interval_ms: int,
) -> FootprintStore:
    C = _TICK_COL_
    price_col = tick_matrix[:, C["price"]]

    config = _resolve_tick_size(config, price_col, tick_to_candle_map, n_closed_candles)
    vol_col = tick_matrix[:, C[config.vol_col]]

    if config.aggressor_col is not None and config.aggressor_col in C:
        flags = tick_matrix[:, C[config.aggressor_col]].astype(np.int64)
        is_buy  = (flags & 32) != 0
        is_sell = (flags & 64) != 0
        is_zero = flags == 0
        has_other = ~(is_buy | is_sell | is_zero)
        bid_col = tick_matrix[:, C["bid"]]
        ask_col = tick_matrix[:, C["ask"]]
        mid     = (bid_col + ask_col) / 2.0
        is_ask  = is_buy | (has_other & (price_col >= mid))
    else:
        bid_col = tick_matrix[:, C["bid"]]
        ask_col = tick_matrix[:, C["ask"]]
        is_ask = price_col >= (bid_col + ask_col) / 2.0

    vol_ask = np.where(is_ask, vol_col, 0.0)
    vol_bid = np.where(~is_ask, vol_col, 0.0)
    price_level = np.round(price_col / config.tick_size) * config.tick_size

    all_levels: list[np.ndarray] = []
    all_scalars = np.zeros((n_closed_candles, N_FP_SCALAR_COLS), dtype=np.float64)
    fp_idx = np.zeros(n_closed_candles + 1, dtype=np.int64)
    cvd_acc = 0.0

    for v_idx in range(n_closed_candles):
        mask = tick_to_candle_map == v_idx
        if not np.any(mask):
            fp_idx[v_idx + 1] = fp_idx[v_idx]
            continue

        unique_levels, inv = np.unique(price_level[mask], return_inverse=True)
        n = len(unique_levels)
        nb = np.bincount(inv, weights=vol_bid[mask], minlength=n)
        na = np.bincount(inv, weights=vol_ask[mask], minlength=n)
        nt = np.bincount(inv, weights=vol_col[mask], minlength=n)
        nc = np.bincount(inv, minlength=n).astype(np.float64)

        levels_arr = np.column_stack([unique_levels, nb, na, nt, na - nb, nc])
        scalars = _compute_scalars(levels_arr, config.value_area_pct, cvd_acc)
        cvd_acc = float(scalars[_FP_SCALAR_COL["cvd"]])

        all_levels.append(levels_arr)
        all_scalars[v_idx] = scalars
        fp_idx[v_idx + 1] = fp_idx[v_idx] + len(levels_arr)

    fp_data = (
        np.ascontiguousarray(np.vstack(all_levels), dtype=np.float64)
        if all_levels
        else np.empty((0, N_FP_LEVEL_COLS), dtype=np.float64)
    )

    return FootprintStore(
        fp_data=fp_data,
        fp_idx=fp_idx,
        fp_scalars=np.ascontiguousarray(all_scalars),
        config=config,
        cvd_total=cvd_acc,
        interval_ms=interval_ms,
    )


def build_fp_stores_for_strategy(
    strategy,
    data: dict,
    ohlc_stores: dict,
    _TICK_COL_: dict,
) -> dict:
    from tradetropy.core.constants import _TICK_COL as _TC
    from tradetropy.data import build_candles_from_ticks

    fp_stores = {}

    for fp_proxy in getattr(strategy, "_fp_proxies", []):
        symbol = fp_proxy.symbol
        interval_ms = fp_proxy.interval_ms

        ohlc_store_compatible = next(
            (
                s
                for s in ohlc_stores.values()
                if s.symbol == symbol and s.interval_ms == interval_ms
            ),
            None,
        )
        mapping = (
            ohlc_store_compatible.tick_to_candle_mapping if ohlc_store_compatible else None
        )
        n_closed_candles = (
            ohlc_store_compatible.n_closed_candles if ohlc_store_compatible else None
        )

        tick_matrix = data[symbol].astype(float)

        if mapping is None:
            result = build_candles_from_ticks(
                tick_matrix[:, _TC["ts"]],
                tick_matrix[:, _TC["price"]],
                tick_matrix[:, _TC["volume"]],
                interval_ms,
            )
            mapping = result["tick_to_candle_map"]
            n_closed_candles = int(result["closed_candles"].shape[0])

        store = build_footprint_from_ticks(
            tick_matrix=tick_matrix,
            tick_to_candle_map=mapping,
            n_closed_candles=n_closed_candles,
            config=fp_proxy.config,
            _TICK_COL_=_TICK_COL_,
            interval_ms=interval_ms,
        )
        fp_stores[id(fp_proxy)] = store
        fp_proxy._connect(store)
        fp_proxy._mapping = mapping

    return fp_stores


def connect_fp_proxies_live(strategy) -> dict:
    rings = {}
    for fp_proxy in getattr(strategy, "_fp_proxies", []):
        ring = LiveFpRing(
            window_size=fp_proxy._window_size,
            config=fp_proxy.config,
            interval_ms=fp_proxy.interval_ms,
        )
        fp_proxy._connect_live(ring)
        rings[id(fp_proxy)] = ring
    return rings


@dataclass
class FpShmBlocks:
    fp_data: Any
    fp_idx: Any
    fp_scalars: Any
    config: FootprintConfig
    cvd_total: float
    interval_ms: int


def footprint_store_from_shm(
    blocks: FpShmBlocks,
    shm_opened: list,
) -> FootprintStore:
    def open_shm(block):
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(name=block.name)
        arr = np.ndarray(block.shape, dtype=block.dtype, buffer=shm.buf)
        shm_opened.append(shm)
        return arr

    return FootprintStore(
        fp_data=open_shm(blocks.fp_data),
        fp_idx=open_shm(blocks.fp_idx),
        fp_scalars=open_shm(blocks.fp_scalars),
        config=blocks.config,
        cvd_total=blocks.cvd_total,
        interval_ms=blocks.interval_ms,
    )

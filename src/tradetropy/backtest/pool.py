# region pool.py v2.1
"""
pool.py v2.1
============
Parallel execution of multiple strategies with shared memory.

Changes v2.1:
  - OHLC data support [M x 6] ("bar" mode), in addition to ticks [N x 7] ("tick" mode).
    PoolBacktestEngine.run() detects the format automatically by shape[1].
    In bar mode:
      - No tick_store or tick_proxies exist.
      - OhlcDataStores are built directly from klines using
        _build_candles_from_klines (same helper as BacktestEngine).
      - fp_proxies are not supported in bar mode (raises ValueError).
      - Warmup uses min_periods directly on bars, same as
        BacktestEngine._calculate_warmup_bars().
    SharedStore receives a `mode` field ("tick" | "bar") and `kline_blocks`
    for raw kline matrices, needed in the worker to determine N and to
    build OHLC indicators without reprocessing OHLC blocks.

Changes from v2.0:
  - Fixed imports: tradetropy.* instead of playg_tick_ohlc_2
  - on_tick() -> on_data() (aligned with current Strategy)
  - fp_proxies support (subscribe_footprint).
  - _scan_strategies collects fp_proxies and fp intervals.
  - _worker feeds fp_proxies tick by tick and calls on_data().
  - Explicit warmup.
  - Multi-symbol.
  - _TickStoreHybrid unchanged.

USAGE (ticks):
    results = PoolBacktestEngine.by_ticks(
        strategies = [Strategy1(), Strategy2()],
        data       = {"BTCUSDT": tick_matrix},   # [N x >=7]
        workers    = 4,
    )

USAGE (OHLC / bars):
    results = PoolBacktestEngine.by_klines(
        strategies = [Strategy1(), Strategy2()],
        data       = {"BTCUSDT": kline_matrix},  # [M x >=6]
        workers    = 4,
    )

STRATEGY CONTRACT:
    - Define at module level (not inside functions) for spawn.
    - Implement result() to return whatever you need.
    - If result() is not implemented, the worker returns None.

SHARED MEMORY AND DEDUPLICATION:
    Tick mode:
      - tick_raw  [N x 7]                -> shm
      - ohlc_raw  [M x 6] + mappings     -> shm per (symbol, interval)
      - fp_data / fp_idx / fp_scalars    -> shm per (symbol, interval, config)
      - deduplicated indicators          -> shm

    Bar mode:
      - kline_raw [M x 6]                -> shm
      - ohlc_raw  [M_ag x 6] + mappings  -> shm per (symbol, interval) [if interval > kline]
      - deduplicated OHLC indicators     -> shm
"""

from __future__ import annotations

import multiprocessing as mp
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tradetropy.core.constants import (
    TICK_COLS,
    OHLC_COLS,
    N_TICK_COLS,
    N_OHLC_COLS,
    _TICK_COL,
    _OHLC_COL,
)
from tradetropy.models.strategy import FeedType

from tradetropy.data.data import (
    OhlcDataStore,
    TickProxy,
    OhlcProxy,
    WindowView,
    OhlcIndicatorView,
    build_candles_from_ticks,
)
from tradetropy.models.footprint import (
    FootprintConfig,
    FootprintStore,
    FpProxy,
    FpShmBlocks,
    footprint_store_from_shm,
    build_footprint_from_ticks,
    _FP_LEVEL_COL,
    _FP_SCALAR_COL,
    N_FP_LEVEL_COLS,
    N_FP_SCALAR_COLS,
)
from tradetropy.models.strategy import Strategy
from tradetropy.core.data_types import TickData, KlineData, SymbolInput, _normalize_data, _normalize_book
from tradetropy.exceptions import ConfigError, StopEngine


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- SHARED MEMORY BLOCK DESCRIPTOR
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ShmBlock:
    """
    Serializable descriptor of a numpy array in shared_memory.
    Only contains metadata -- child processes open by name.
    """

    name: str
    shape: tuple
    dtype: str

    def open(self) -> tuple["shared_memory.SharedMemory", np.ndarray]:
        """Open the block. Returns (shm, numpy_view). Zero copies."""
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=self.name)
        array = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return shm, array


def _allocate(array: np.ndarray, shm_refs: list) -> ShmBlock:
    """Copy an array to shared_memory. Returns the ShmBlock descriptor."""
    from multiprocessing import shared_memory
    arr = np.ascontiguousarray(array, dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    dest = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    dest[:] = arr
    shm_refs.append(shm)
    return ShmBlock(name=shm.name, shape=arr.shape, dtype=str(arr.dtype))


def _open_shm(block: ShmBlock, opened_shm: list) -> np.ndarray:
    shm, arr = block.open()
    opened_shm.append(shm)
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- SHM BLOCK CONTAINERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RawTickBlocks:
    tick_raw: ShmBlock  # [N × 7]


@dataclass
class RawKlineBlocks:
    kline_raw: ShmBlock  # [M x 6]  -- raw klines in bar mode


@dataclass
class RawOhlcBlocks:
    ohlc_raw: ShmBlock  # [M x 6]
    tick_to_candle: ShmBlock  # [N]        -- tick_to_candle_map
    accumulated: ShmBlock  # [N x 4]   -- [open, high, low, vol] accumulated
    candle_ts: ShmBlock  # [N]        -- candle_ts_per_tick
    prices: ShmBlock  # [N]        -- price per tick


@dataclass
class RawFpBlocks:
    """SHM blocks for a FootprintStore (per symbol + interval + config)."""

    fp_data: ShmBlock  # [total_levels x N_FP_LEVEL_COLS]
    fp_idx: ShmBlock  # [M_closed + 1]   -- CSR indices (float64, cast to int64)
    fp_scalars: ShmBlock  # [M_closed x N_FP_SCALAR_COLS]
    config: FootprintConfig
    cvd_total: float
    interval_ms: int


@dataclass
class SharedStore:
    """
    All SHM blocks for a backtest session.

    mode             : "tick" | "bar"
    tick_blocks      : {symbol: RawTickBlocks}          -- tick mode only
    kline_blocks     : {symbol: RawKlineBlocks}         -- bar mode only
    ohlc_blocks      : {(symbol, interval): RawOhlcBlocks}
    fp_blocks        : {(symbol, interval, fp_key): RawFpBlocks}
                         fp_key = str(config) -- to support different configs on
                         the same symbol/interval without collision.
    ind_tick_blocks  : {(symbol, col_name): ShmBlock [N]}
    ind_ohlc_blocks  : {(symbol, interval, col_name): ShmBlock [M]}
    pattern_stores   : {pattern_id: PatternStore} -- read-only, shared between workers.
                         pattern_id = id(pattern) of the original PatternMatcherDef.
                         On Linux (fork) it is shared without copying.
                         On Windows/macOS (spawn) it is serialized once per worker.
    """

    feed_type: FeedType  # "tick" | "kline"
    tick_blocks: dict
    kline_blocks: dict
    ohlc_blocks: dict
    fp_blocks: dict
    ind_tick_blocks: dict
    ind_ohlc_blocks: dict
    pattern_stores: dict = field(default_factory=dict)  # {pattern_id: PatternStore}
    _shm_refs: list = field(default_factory=list, repr=False)

    def release(self):
        for shm in self._shm_refs:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_refs.clear()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- BUILDING THE SHARED STORE
# ══════════════════════════════════════════════════════════════════════════════


def _fp_config_key(config: FootprintConfig) -> str:
    """Unique string key for a FootprintConfig. Used for deduplication."""
    return f"{config.tick_size}_{config.value_area_pct}_{config.aggressor_col}_{config.vol_col}"


def _build_tick_shared_store(data, intervals_per_symbol, fp_infos, indicator_defs_union, shm_refs):
    tick_blocks = {}
    ohlc_blocks = {}
    fp_blocks = {}
    ind_tick_blocks = {}
    ind_ohlc_blocks = {}
    candles_per_key = {}

    raw_per_symbol = {}
    for symbol, tick_matrix in data.items():
        ticks = np.ascontiguousarray(tick_matrix[:, :N_TICK_COLS], dtype=np.float64)
        raw_per_symbol[symbol] = ticks
        tick_blocks[symbol] = RawTickBlocks(tick_raw=_allocate(ticks, shm_refs))

    for symbol, ticks in raw_per_symbol.items():
        for interval in intervals_per_symbol.get(symbol, []):
            r = build_candles_from_ticks(
                ticks[:, _TICK_COL["ts"]], ticks[:, _TICK_COL["price"]],
                ticks[:, _TICK_COL["volume"]], interval,
            )
            candles_per_key[(symbol, interval)] = r
            ohlc_blocks[(symbol, interval)] = RawOhlcBlocks(
                ohlc_raw=_allocate(r["closed_candles"], shm_refs),
                tick_to_candle=_allocate(r["tick_to_candle_map"], shm_refs),
                accumulated=_allocate(r["accumulated_per_tick"], shm_refs),
                candle_ts=_allocate(r["candle_ts_per_tick"], shm_refs),
                prices=_allocate(r["prices"], shm_refs),
            )

    for info in fp_infos:
        symbol = info["symbol"]
        interval = info["interval_ms"]
        config = info["config"]
        cfg_key = _fp_config_key(config)
        fp_key = (symbol, interval, cfg_key)
        if fp_key in fp_blocks:
            continue

        ticks = raw_per_symbol[symbol]

        if (symbol, interval) in candles_per_key:
            r = candles_per_key[(symbol, interval)]
            mapping = r["tick_to_candle_map"]
            n_closed_candles = int(r["closed_candles"].shape[0])
        else:
            r = build_candles_from_ticks(
                ticks[:, _TICK_COL["ts"]], ticks[:, _TICK_COL["price"]],
                ticks[:, _TICK_COL["volume"]], interval,
            )
            candles_per_key[(symbol, interval)] = r
            mapping = r["tick_to_candle_map"]
            n_closed_candles = int(r["closed_candles"].shape[0])

        store = build_footprint_from_ticks(
            tick_matrix=ticks, tick_to_candle_map=mapping,
            n_closed_candles=n_closed_candles, config=config,
            _TICK_COL_=_TICK_COL, interval_ms=interval,
        )
        fp_blocks[fp_key] = RawFpBlocks(
            fp_data=_allocate(store.fp_data, shm_refs),
            fp_idx=_allocate(store.fp_idx.astype(np.float64), shm_refs),
            fp_scalars=_allocate(store.fp_scalars, shm_refs),
            config=config, cvd_total=store._cvd_total, interval_ms=interval,
        )

    for defn in indicator_defs_union:
        col_name = defn["col_name"]
        indicator = defn["indicator"]
        src_key = defn["src_key"]

        if src_key[0] == "tick":
            symbol = src_key[1]
            shm_key = (symbol, col_name)
            if shm_key in ind_tick_blocks:
                continue
            src_col = _TICK_COL[defn["source"]._col_name]
            src_arr = raw_per_symbol[symbol][:, src_col]
            ind_tick_blocks[shm_key] = _allocate(indicator.calculate(src_arr), shm_refs)
        else:
            symbol, interval = src_key[1], src_key[2]
            shm_key = (symbol, interval, col_name)
            if shm_key in ind_ohlc_blocks:
                continue
            src_col = _OHLC_COL[defn["source"]._col_name]
            candles_raw = candles_per_key[(symbol, interval)]["closed_candles"]
            src_arr = (
                candles_raw[:, src_col] if len(candles_raw) > 0
                else np.array([], dtype=np.float64)
            )
            ind_ohlc_blocks[shm_key] = _allocate(indicator.calculate(src_arr), shm_refs)

    return tick_blocks, ohlc_blocks, fp_blocks, ind_tick_blocks, ind_ohlc_blocks, candles_per_key


def _build_kline_shared_store(data, intervals_per_symbol, indicator_defs_union, shm_refs):
    from tradetropy.backtest._build import _build_candles_from_klines

    kline_blocks = {}
    ohlc_blocks = {}
    ind_ohlc_blocks = {}
    candles_per_key = {}

    raw_per_symbol = {}
    for symbol, kline_matrix in data.items():
        klines = np.ascontiguousarray(kline_matrix[:, :N_OHLC_COLS], dtype=np.float64)
        raw_per_symbol[symbol] = klines
        kline_blocks[symbol] = RawKlineBlocks(kline_raw=_allocate(klines, shm_refs))

    for symbol, klines in raw_per_symbol.items():
        for interval in intervals_per_symbol.get(symbol, []):
            r = _build_candles_from_klines(klines, interval)
            candles_per_key[(symbol, interval)] = r
            ohlc_blocks[(symbol, interval)] = RawOhlcBlocks(
                ohlc_raw=_allocate(r["closed_candles"], shm_refs),
                tick_to_candle=_allocate(r["tick_to_candle_map"], shm_refs),
                accumulated=_allocate(r["accumulated_per_tick"], shm_refs),
                candle_ts=_allocate(r["candle_ts_per_tick"], shm_refs),
                prices=_allocate(r["prices"], shm_refs),
            )

    for defn in indicator_defs_union:
        col_name = defn["col_name"]
        indicator = defn["indicator"]
        src_key = defn["src_key"]

        if src_key[0] == "tick":
            continue

        symbol, interval = src_key[1], src_key[2]
        shm_key = (symbol, interval, col_name)
        if shm_key in ind_ohlc_blocks:
            continue
        src_col = _OHLC_COL[defn["source"]._col_name]
        candles_raw = candles_per_key[(symbol, interval)]["closed_candles"]
        src_arr = (
            candles_raw[:, src_col] if len(candles_raw) > 0
            else np.array([], dtype=np.float64)
        )
        ind_ohlc_blocks[shm_key] = _allocate(indicator.calculate(src_arr), shm_refs)

    return kline_blocks, ohlc_blocks, ind_ohlc_blocks, candles_per_key


def build_shared_store(
    data: dict,
    intervals_per_symbol: dict,
    fp_infos: list,
    indicator_defs_union: list,
    feed_type: FeedType = "tick",
    pattern_defs_union: list | None = None,
) -> SharedStore:
    shm_refs = []

    if feed_type == "tick":
        tick_blocks, ohlc_blocks, fp_blocks, ind_tick_blocks, ind_ohlc_blocks, candles_by_key = \
            _build_tick_shared_store(data, intervals_per_symbol, fp_infos, indicator_defs_union, shm_refs)
        kline_blocks = {}
    else:
        kline_blocks, ohlc_blocks, ind_ohlc_blocks, candles_by_key = \
            _build_kline_shared_store(data, intervals_per_symbol, indicator_defs_union, shm_refs)
        tick_blocks = {}
        fp_blocks = {}
        ind_tick_blocks = {}

    pattern_stores = _build_pattern_stores_from_shm(
        pattern_defs_union=pattern_defs_union,
        indicator_defs_union=indicator_defs_union,
        candles_by_key=candles_by_key,
        ind_ohlc_blocks=ind_ohlc_blocks,
    )

    return SharedStore(
        feed_type=feed_type,
        tick_blocks=tick_blocks, kline_blocks=kline_blocks,
        ohlc_blocks=ohlc_blocks, fp_blocks=fp_blocks,
        ind_tick_blocks=ind_tick_blocks, ind_ohlc_blocks=ind_ohlc_blocks,
        pattern_stores=pattern_stores, _shm_refs=shm_refs,
    )


def _find_ind_def_in_union(proxy, indicator_defs_union: list) -> "dict | None":
    if proxy is None:
        return None
    for defn in indicator_defs_union:
        if defn.get("proxy") is proxy:
            return defn
    return None


def _build_pattern_stores_from_shm(
    pattern_defs_union: list,
    indicator_defs_union: list,
    candles_by_key: dict,
    ind_ohlc_blocks: dict,
) -> dict:
    if not pattern_defs_union:
        return {}
    from tradetropy.ta.pattern.sequence    import FrozenPivotSequence
    from tradetropy.ta.pattern.store       import PatternStore
    from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin

    pattern_stores: dict = {}

    for pm_def in pattern_defs_union:
        base_pivot_proxy = pm_def.pivots[0]
        base_def = _find_ind_def_in_union(base_pivot_proxy, indicator_defs_union)
        if base_def is None:
            continue

        base_indicator = base_def["indicator"]
        if not isinstance(base_indicator, PivotIndicatorMixin):
            continue

        src_key = base_def["src_key"]
        if src_key[0] != "ohlc":
            continue

        symbol  = src_key[1]
        interval = src_key[2]

        if (symbol, interval) not in candles_by_key:
            continue

        candles_raw = candles_by_key[(symbol, interval)]["closed_candles"]
        n_bars  = len(candles_raw)
        if n_bars == 0:
            continue

        base_cols = base_indicator.pivot_col_names(symbol)

        def _read_col(col_name):
            key = (symbol, interval, col_name)
            if key in ind_ohlc_blocks:
                _, arr = ind_ohlc_blocks[key].open()
                return arr.copy()
            return np.full(n_bars, np.nan, dtype=np.float64)

        ph    = _read_col(base_cols[0])
        pl    = _read_col(base_cols[1])
        ph_ts = _read_col(base_cols[2])
        pl_ts = _read_col(base_cols[3])

        decorator_arrays: dict[str, np.ndarray]       = {}
        tag_decoders:     dict[str, dict[float, str]] = {}
        for dec_proxy in pm_def.pivots[1:]:
            dec_def = _find_ind_def_in_union(dec_proxy, indicator_defs_union)
            if dec_def is None:
                continue
            dec_ind = dec_def["indicator"]
            if not isinstance(dec_ind, PivotIndicatorMixin):
                continue
            dec_cols = dec_ind.pivot_col_names(symbol)
            if dec_cols:
                tag_name = dec_ind.tag_name
                decorator_arrays[tag_name] = _read_col(dec_cols[0])
                decoder = getattr(dec_ind, "TAG_DECODE", None)
                if decoder is not None:
                    tag_decoders[tag_name] = decoder

        sequence = FrozenPivotSequence.from_arrays(
            ph_array=ph, pl_array=pl, ph_ts_array=ph_ts, pl_ts_array=pl_ts,
            decorator_arrays=decorator_arrays, tag_decoders=tag_decoders,
        )
        store = PatternStore(sequence, pm_def.pattern)
        pattern_stores[id(pm_def.pattern)] = store

    return pattern_stores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- SCAN STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════


def _scan_strategies(
    classes: list,
) -> tuple[dict, list, list, list]:
    """
    Instantiate each class temporarily to collect:
      - OHLC intervals needed for each symbol
      - deduplicated list of all indicators
      - deduplicated list of footprints to pre-calculate
      - deduplicated list of pattern matcher defs

    Receives classes (not instances) - so scanning doesn't consume the instance
    that the worker will later use, and works correctly with spawn because
    the class was already pickled by name to be sent to the worker.

    Returns (intervals_per_symbol, fp_infos, indicator_defs_union, pattern_defs_union).
    """
    intervals_per_symbol: dict[str, set] = {}
    seen_ind:     dict[tuple, dict] = {}
    seen_fp:      dict[tuple, dict] = {}
    seen_pattern: dict[int, object]  = {}  # {id(pattern): PatternMatcherDef}

    for item in classes:
        is_direct_instance = isinstance(item, Strategy)

        if isinstance(item, type):
            strat = item()
        elif callable(item) and not isinstance(item, Strategy):
            strat = item()
        else:
            strat = item
        strat.declare()

        for op in strat._ohlc_proxies:
            intervals_per_symbol.setdefault(op.symbol, set()).add(op.interval_ms)

        for defn in strat._indicator_defs:
            source = defn["source"]
            indicator = defn["indicator"]

            if isinstance(source.proxy, TickProxy):
                symbol = source.symbol
                src_key = ("tick", symbol)
            else:
                symbol = source.proxy.symbol
                interval = source.proxy.interval_ms
                src_key = ("ohlc", symbol, interval)

            col_name = indicator.col_name(source._col_name, symbol)
            key = (src_key, col_name)
            if key not in seen_ind:
                seen_ind[key] = {
                    "source": source,
                    "indicator": indicator,
                    "col_name": col_name,
                    "src_key": src_key,
                }

        for fp_proxy in strat._fp_proxies:
            symbol = fp_proxy.symbol
            interval = fp_proxy.interval_ms
            config = fp_proxy.config
            cfg_key = _fp_config_key(config)
            key = (symbol, interval, cfg_key)

            if key not in seen_fp:
                seen_fp[key] = {
                    "symbol": symbol,
                    "interval_ms": interval,
                    "config": config,
                }

        # collect pattern matcher defs -- deduplicate by id(pattern)
        for pm_def in strat._pattern_matcher_defs:
            pm_key = id(pm_def.pattern)
            if pm_key not in seen_pattern:
                seen_pattern[pm_key] = pm_def

        if is_direct_instance:
            strat._tick_proxies          = []
            strat._ohlc_proxies          = []
            strat._indicator_defs        = []
            strat._fp_proxies            = []
            strat._pattern_matcher_defs  = []

    return (
        {s: list(v) for s, v in intervals_per_symbol.items()},
        list(seen_fp.values()),
        list(seen_ind.values()),
        list(seen_pattern.values()),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- HYBRID TICK DATA STORE
# ══════════════════════════════════════════════════════════════════════════════


class _TickStoreHybrid:
    """
    TickDataStore where .matrix = raw (shm) + indicator columns.
    Built ONCE in __init__. Subsequent accesses are O(1).
    If no indicator columns, .matrix is the direct shm view (zero copies).
    """

    __slots__ = ("matrix", "col_index", "n_ticks")

    def __init__(self, raw: np.ndarray, ind_cols: list, col_index: dict):
        if ind_cols:
            self.matrix = np.ascontiguousarray(np.hstack([raw] + ind_cols))
        else:
            self.matrix = raw
        self.col_index = col_index
        self.n_ticks = len(raw)

    def col(self, name: str) -> np.ndarray:
        return self.matrix[:, self.col_index[name]]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- WORKER HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _open_shm(block: ShmBlock, opened_shm: list) -> np.ndarray:
    shm, arr = block.open()
    opened_shm.append(shm)
    return arr

_open_shm = _open_shm


def _open_ohlc_stores_worker(strategy, shared_store, opened_shm) -> dict:
    ohlc_stores = {}
    inds_per_proxy: dict[int, list] = defaultdict(list)
    for defn in strategy._indicator_defs:
        source = defn["source"]
        if isinstance(source.proxy, OhlcProxy):
            inds_per_proxy[id(source.proxy)].append(defn)

    for ohlc_proxy in strategy._ohlc_proxies:
        sym = ohlc_proxy.symbol
        interval = ohlc_proxy.interval_ms
        ob = shared_store.ohlc_blocks[(sym, interval)]

        ohlc_raw = _open_shm(ob.ohlc_raw, opened_shm)
        mapping = _open_shm(ob.tick_to_candle, opened_shm).astype(np.intp)
        accumulated = _open_shm(ob.accumulated, opened_shm)
        candle_ts = _open_shm(ob.candle_ts, opened_shm)
        prices = _open_shm(ob.prices, opened_shm)

        ohlc_col_index = dict(zip(OHLC_COLS, range(N_OHLC_COLS)))
        ohlc_ind_cols = []

        for defn in inds_per_proxy[id(ohlc_proxy)]:
            source = defn["source"]
            indicator = defn["indicator"]
            col_name = indicator.col_name(source._col_name, sym)
            shm_key = (sym, interval, col_name)

            if shm_key in shared_store.ind_ohlc_blocks:
                col_arr = _open_shm(shared_store.ind_ohlc_blocks[shm_key], opened_shm)
            else:
                col_source = _OHLC_COL[source._col_name]
                source_arr = (
                    ohlc_raw[:, col_source]
                    if len(ohlc_raw) > 0
                    else np.array([], dtype=np.float64)
                )
                col_arr = indicator.calculate(source_arr)

            ohlc_col_index[col_name] = N_OHLC_COLS + len(ohlc_ind_cols)
            defn["col_name"] = col_name
            defn["en_tick_store"] = False
            defn["ohlc_proxy"] = ohlc_proxy
            defn["src_col_idx"] = _OHLC_COL[source._col_name]
            ohlc_ind_cols.append(
                col_arr.reshape(-1, 1) if col_arr.ndim == 1 else col_arr
            )

        if len(ohlc_raw) > 0 and ohlc_ind_cols:
            ohlc_matrix = np.ascontiguousarray(np.hstack([ohlc_raw] + ohlc_ind_cols))
        elif len(ohlc_raw) > 0:
            ohlc_matrix = ohlc_raw
        else:
            ohlc_matrix = np.empty(
                (0, N_OHLC_COLS + len(ohlc_ind_cols)), dtype=np.float64
            )

        ohlc_stores[id(ohlc_proxy)] = OhlcDataStore(
            matrix=ohlc_matrix, col_index=ohlc_col_index,
            tick_to_candle_mapping=mapping, accumulated_by_tick=accumulated,
            ts_candle_by_tick=candle_ts, prices_per_tick=prices,
            interval_ms=interval, symbol=sym,
        )

    return ohlc_stores


def _run_bar_mode_worker(strategy, shared_store, ohlc_stores, opened_shm):
    for defn in strategy._indicator_defs:
        proxy = defn["proxy"]
        indicator = defn["indicator"]
        op = defn["ohlc_proxy"]
        ohlc_store = ohlc_stores[id(op)]
        view = OhlcIndicatorView(
            ohlc_store=ohlc_store, indicator=indicator,
            ind_col_idx=ohlc_store.col_index[defn["col_name"]],
            src_col_idx=defn["src_col_idx"], size=op._window_size,
        )
        proxy._connect(view)

    for pm_def in strategy._pattern_matcher_defs:
        pm_key = id(pm_def.pattern)
        store = shared_store.pattern_stores.get(pm_key)
        if store is not None:
            pm_def.proxy._connect_backtest(store)

    primary_symbol = (
        strategy._ohlc_proxies[0].symbol
        if strategy._ohlc_proxies
        else next(iter(shared_store.kline_blocks))
    )
    kline_raw = _open_shm(shared_store.kline_blocks[primary_symbol].kline_raw, opened_shm)
    N = len(kline_raw)

    ohlc_proxies    = strategy._ohlc_proxies
    ind_proxies     = [d["proxy"] for d in strategy._indicator_defs]
    pattern_proxies = [d.proxy for d in strategy._pattern_matcher_defs]

    warmup = max(
        (d["indicator"].min_periods for d in strategy._indicator_defs
         if not isinstance(d["source"].proxy, TickProxy)),
        default=0,
    )

    for idx in range(N):
        for op in ohlc_proxies:
            op._advance(idx)
        for ip in ind_proxies:
            ip._advance(idx)
        for pm in pattern_proxies:
            pm._advance(idx)

        if idx >= warmup:
            try:
                strategy._in_on_data = True
                strategy.on_data()
            except StopEngine:
                pass
            finally:
                strategy._in_on_data = False

    for shm in opened_shm:
        shm.close()

    return strategy.result() if hasattr(strategy, "result") else None


def _run_tick_mode_worker(strategy, shared_store, ohlc_stores, opened_shm,
                          book_payload=None):
    if strategy._tick_proxies:
        primary_symbol = strategy._tick_proxies[0].symbol
    elif strategy._ohlc_proxies:
        primary_symbol = strategy._ohlc_proxies[0].symbol
    elif strategy._fp_proxies:
        primary_symbol = strategy._fp_proxies[0].symbol
    else:
        primary_symbol = next(iter(shared_store.tick_blocks))

    tick_raw = _open_shm(shared_store.tick_blocks[primary_symbol].tick_raw, opened_shm)
    col_index = dict(zip(TICK_COLS, range(N_TICK_COLS)))
    ind_cols = []

    # Build + fully populate the L2 book rings BEFORE the indicator precompute
    # so DeepTrades / L2 indicators (computed once over the whole series) read a
    # causal (as-of by ts) book. The rings are reset before the tick loop, which
    # re-drains them incrementally for causal direct book access in on_data().
    book_rings: dict = {}
    book_replayer = None
    if book_payload is not None:
        from tradetropy.data._ring import LiveBookRing
        from tradetropy.data._book_replay import BookReplayer
        books, offsets = book_payload
        for bp in getattr(strategy, "_book_proxies", []):
            ring = LiveBookRing(window_size=bp.window_size, levels=bp.depth)
            bp._book_ring = ring
            book_rings.setdefault(bp.symbol, []).append(ring)
        book_replayer = BookReplayer(books, offsets)
        for _sym, _rings in book_rings.items():
            book_replayer.drain_to(_sym, float("inf"), _rings)

    for defn in strategy._indicator_defs:
        source = defn["source"]
        if not isinstance(source.proxy, TickProxy):
            continue
        indicator = defn["indicator"]
        col_name = indicator.col_name(source._col_name, source.symbol)
        shm_key = (source.symbol, col_name)

        if shm_key in shared_store.ind_tick_blocks:
            col_arr = _open_shm(shared_store.ind_tick_blocks[shm_key], opened_shm)
        else:
            col_source = _TICK_COL[source._col_name]
            col_arr = indicator.calculate(tick_raw[:, col_source])

        col_index[col_name] = N_TICK_COLS + len(ind_cols)
        defn["col_name"] = col_name
        defn["en_tick_store"] = True
        ind_cols.append(col_arr.reshape(-1, 1) if col_arr.ndim == 1 else col_arr)

    tick_store = _TickStoreHybrid(tick_raw, ind_cols, col_index)

    for fp_proxy in strategy._fp_proxies:
        symbol = fp_proxy.symbol
        interval = fp_proxy.interval_ms
        config = fp_proxy.config
        cfg_key = _fp_config_key(config)
        fp_key = (symbol, interval, cfg_key)

        rb = shared_store.fp_blocks[fp_key]
        store = FootprintStore(
            fp_data=_open_shm(rb.fp_data, opened_shm),
            fp_idx=_open_shm(rb.fp_idx, opened_shm).astype(np.int64),
            fp_scalars=_open_shm(rb.fp_scalars, opened_shm),
            config=rb.config, cvd_total=rb.cvd_total, interval_ms=rb.interval_ms,
        )
        fp_proxy._connect(store)

    for tp in strategy._tick_proxies:
        tp._connect_backtest(tick_store)

    for defn in strategy._indicator_defs:
        proxy = defn["proxy"]
        indicator = defn["indicator"]
        source = defn["source"]

        if defn.get("en_tick_store", True) and isinstance(source.proxy, TickProxy):
            col_idx = tick_store.col_index[defn["col_name"]]
            view = WindowView(col_idx=col_idx, size=source.proxy._window_size, tick_store=tick_store)
            proxy._connect(view)
        else:
            op = defn["ohlc_proxy"]
            ohlc_store = ohlc_stores[id(op)]
            view = OhlcIndicatorView(
                ohlc_store=ohlc_store, indicator=indicator,
                ind_col_idx=ohlc_store.col_index[defn["col_name"]],
                src_col_idx=defn["src_col_idx"], size=op._window_size,
            )
            proxy._connect(view)

    for pm_def in strategy._pattern_matcher_defs:
        pm_key = id(pm_def.pattern)
        store  = shared_store.pattern_stores.get(pm_key)
        if store is not None:
            pm_def.proxy._connect_backtest(store)

    tick_proxies    = strategy._tick_proxies
    ohlc_proxies    = strategy._ohlc_proxies
    ind_proxies     = [d["proxy"] for d in strategy._indicator_defs]
    fp_proxies      = strategy._fp_proxies
    pattern_proxies = [d.proxy for d in strategy._pattern_matcher_defs]
    N = len(tick_raw)

    warmup = max(
        (d["indicator"].min_periods - 1 for d in strategy._indicator_defs),
        default=0,
    )

    # Reset the rings so the loop re-drains them incrementally (causal direct
    # book access in on_data); the precompute above already used the full book.
    if book_replayer is not None:
        for _rings in book_rings.values():
            for _r in _rings:
                _r.reset()
        book_replayer.reset()

    for idx in range(N):
        for tp in tick_proxies:
            tp._advance(idx)
        for op in ohlc_proxies:
            op._advance(idx)
        for ip in ind_proxies:
            ip._advance(idx)
        for pm in pattern_proxies:
            pm._advance(idx)

        if book_replayer is not None:
            _ts_now = int(tick_raw[idx, _TICK_COL["ts"]])
            for _sym, _rings in book_rings.items():
                book_replayer.drain_to(_sym, _ts_now, _rings)

        if fp_proxies:
            row = tick_raw[idx]
            ts_ms = int(row[_TICK_COL["ts"]])
            price = float(row[_TICK_COL["price"]])
            vol = float(row[_TICK_COL["volume"]])
            bid = float(row[_TICK_COL["bid"]])
            ask = float(row[_TICK_COL["ask"]])

            for fp in fp_proxies:
                aggressor_col = fp.config.aggressor_col
                flag = (
                    float(row[_TICK_COL[aggressor_col]])
                    if aggressor_col and aggressor_col in _TICK_COL
                    else 0.0
                )
                fp.process_tick(ts_ms, price, vol, bid, ask, flag)

        if idx >= warmup:
            try:
                strategy._in_on_data = True
                strategy.on_data()
            except StopEngine:
                pass
            finally:
                strategy._in_on_data = False

    for shm in opened_shm:
        shm.close()

    return strategy.result() if hasattr(strategy, "result") else None


def _worker(args: tuple) -> Any:
    strategy_cls, shared_store = args[0], args[1]
    book_payload = args[2] if len(args) > 2 else None
    strategy = strategy_cls()
    strategy._feed_type = shared_store.feed_type
    strategy._set_run_mode("pool")
    strategy.init()

    opened_shm = []
    ohlc_stores = _open_ohlc_stores_worker(strategy, shared_store, opened_shm)

    for op in strategy._ohlc_proxies:
        op._connect_backtest(ohlc_stores[id(op)])

    if shared_store.feed_type == "kline":
        return _run_bar_mode_worker(strategy, shared_store, ohlc_stores, opened_shm)

    return _run_tick_mode_worker(
        strategy, shared_store, ohlc_stores, opened_shm, book_payload,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- POOL BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class PoolBacktestEngine:
    """
    Runs multiple strategies in parallel with shared memory.

    USAGE:
        # With ticks [N x >=7]:
        results = PoolBacktestEngine.by_ticks(
            strategies = [S1, S2, S3],
            data       = (TickData("BTCUSDT", tick_matrix, tick_size=0.01),),
            workers    = 8,
        )

        # With klines:
        results = PoolBacktestEngine.by_klines(
            strategies = [S1, S2, S3],
            data       = (KlineData("BTCUSDT", kline_matrix, timeframe=60_000),),
            workers    = 8,
        )

    The data type (TickData/KlineData) determines the mode -- it is never
    inferred from shape. Symbol config travels with the data.

    Advantages:
      - SMA(20) on BTCUSDT shared across 400 strategies -> computed only once.
      - Footprints pre-computed in bulk -> zero recalculation in workers.
      - Works on Linux (fork), macOS and Windows (spawn).
    """

    @classmethod
    def by_ticks(
        cls,
        strategies: list,
        data,
        workers: int | None = None,
        book: "BookData | tuple | list | None" = None,
        sync_book: bool = False,
    ) -> list:
        """
        Run strategies with tick data.

        data : TickData or tuple/list of TickData (one per symbol).
        subscribe_ticks() and subscribe_footprint() are valid in this mode.

        book : BookData or tuple/list of BookData with the recorded L2 book
            to replay alongside ticks (enables DeepTrades / L2 metrics in the
            pool, same as in backtest). The symbol is taken from each
            ``BookData.symbol`` (no ``{symbol: BookData}`` map). A sync
            preflight warns if the book is out of sync with the trades.
        sync_book : if True and the preflight finds a recoverable clock offset,
            shifts the book to the trades' clock. Default False (warn only).
        """
        inputs, feed_type = _normalize_data(data)
        if feed_type != "tick":
            raise ConfigError(
                "PoolBacktestEngine.by_ticks() received KlineData. "
                "Use by_klines() for bars."
            )
        data_dict = {inp.symbol: inp.data for inp in inputs}
        return cls._run(
            strategies, data_dict, workers, feed_type="tick",
            book=_normalize_book(book), sync_book=sync_book,
        )

    @classmethod
    def by_klines(
        cls,
        strategies: list,
        data,
        workers: int | None = None,
        book: "BookData | tuple | list | None" = None,
    ) -> list:
        """
        Run strategies with OHLC bar data.

        data : KlineData or tuple/list of KlineData (each with interval_ms).
        subscribe_ticks() and subscribe_footprint() are not supported.
        """
        inputs, feed_type = _normalize_data(data)
        if feed_type != "kline":
            raise ConfigError(
                "PoolBacktestEngine.by_klines() received TickData. "
                "Use by_ticks() for ticks."
            )
        if book is not None:
            raise ConfigError(
                "PoolBacktestEngine.by_klines() does not support book=. The L2 "
                "book needs per-trade timestamps for causal book_as_of; use "
                "by_ticks(..., book=...)."
            )
        data_dict = {inp.symbol: inp.data for inp in inputs}
        return cls._run(strategies, data_dict, workers, feed_type="kline")

    @classmethod
    def _run(
        cls,
        strategies: list,
        data: dict,
        workers: int | None,
        feed_type: FeedType,  # "tick" | "kline"
        book: "dict | None" = None,
        sync_book: bool = False,
    ) -> list:
        """Common implementation. feed_type is always passed explicitly."""
        if not strategies:
            return []

        def _to_factory(s):
            if isinstance(s, type):
                return s
            if isinstance(s, Strategy):
                return type(s)
            return s

        factories = [_to_factory(s) for s in strategies]

        intervals_per_symbol, fp_infos, ind_union, pattern_defs = (
            _scan_strategies(factories)
        )

        shared_store = build_shared_store(
            data                 = data,
            intervals_per_symbol = intervals_per_symbol,
            fp_infos             = fp_infos,
            indicator_defs_union = ind_union,
            feed_type            = feed_type,
            pattern_defs_union   = pattern_defs,
        )

        # Sync preflight once in the parent (emit warnings, resolve offsets) so
        # workers only rebuild the rings and replay - no per-worker re-warn.
        book_payload = None
        if book and feed_type == "tick":
            from tradetropy.data._book_replay import resolve_book_sync
            from tradetropy.core.constants import _TICK_COL as _TC
            trades = {
                sym: (arr[:, _TC["ts"]], arr[:, _TC["price"]])
                for sym, arr in data.items() if sym in book
            }
            reports = resolve_book_sync(
                book, trades, sync_book=sync_book,
                engine_label="PoolBacktestEngine",
            )
            offsets = {s: off for s, (_r, off) in reports.items()}
            book_payload = (book, offsets)

        n_workers = workers or min(len(factories), mp.cpu_count())
        args = [(f, shared_store, book_payload) for f in factories]

        ctx_name = "fork" if platform.system() == "Linux" else "spawn"
        ctx = mp.get_context(ctx_name)

        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                results = list(ex.map(_worker, args))
        finally:
            shared_store.release()

        return results

    @classmethod
    def run(
        cls,
        strategies: list,
        data: "SymbolInput | tuple[SymbolInput, ...] | list[SymbolInput]",
        workers: int | None = None,
        save_log: "bool | None" = None,
        book: "BookData | tuple | list | None" = None,
        sync_book: bool = False,
    ) -> list:
        """
        Unified entry point for PoolBacktestEngine.

        .. note::
            ``save_log`` is accepted for API consistency with other
            engines, but is a **no-op** in pool/optimize: each worker uses
            ``NullLogger`` (zero I/O), so nothing is ever written to a file.

        Accepts two forms:

        **Typed API (v2.2+, preferred):**
            data is ``TickData``, ``KlineData``, or a list of them.
            The mode is inferred from the type -- there is no ``mode`` parameter.
            Symbol config travels with the data; no need to add it manually.

            Example::

                PoolBacktestEngine.run(
                    strategies = [S1, S2],
                    data       = TickData("BTCUSDT", tick_matrix,
                                          tick_step=0.25, tick_value=1.25),
                    workers    = 4,
                )

                PoolBacktestEngine.run(
                    strategies = [S1, S2],
                    data       = [
                        TickData("BTCUSDT", btc_ticks),
                        TickData("ETHUSDT", eth_ticks),
                    ],
                )

        **Legacy API (dict, compatible with v2.0/v2.1):**
            data is a dict ``{symbol: ndarray}``. The mode is inferred from
            ``shape[1]`` as in earlier versions. Kept for backwards
            compatibility.

            Example::

                PoolBacktestEngine.run(
                    strategies = [S1, S2],
                    data       = {"BTCUSDT": tick_matrix},   # [N x >=7]
                    workers    = 4,
                )
        """
        if not strategies:
            return []
        inputs, feed_type = _normalize_data(data)
        data_dict = {inp.symbol: inp.data for inp in inputs}
        return cls._run(
            strategies, data_dict, workers, feed_type=feed_type,
            book=_normalize_book(book), sync_book=sync_book,
        )


# endregion

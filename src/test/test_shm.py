# region test_shm_v1.0
"""
Tests for the Shared Memory (SHM) layer of the parallel backtesting system.

Coverage:
  · ShmBlock              — serializable descriptor, open/close, zero-copy views
  · _allocate             — copies array → shm, returns ShmBlock
  · RawTickBlocks / RawOhlcBlocks / SharedStore — metadata containers
  · build_shared_store    — tick raw, ohlc raw, deduplicated indicators
  · SharedStore.release   — clears all shm blocks without errors
  · _scan_strategies      — collects intervals and deduplicated indicators
  · Deduplication         — same SMA on the same source → single block

NOTE: pool.py has a legacy import ("playg_tick_ohlc_2") that doesn't exist in
the current package. All these tests import directly from tradetropy.* modules and
replicate SharedStore logic using the same primitives.
The tests are self-contained: they don't import pool.py.

Usage:
    pytest test_shm.py -v
    pytest test_shm.py -v -k "TestShmBlock"
"""

import pytest
import numpy as np
from multiprocessing import shared_memory
from dataclasses import dataclass, field
from typing import Any

from tradetropy.core.constants import (
    TICK_COLS, OHLC_COLS,
    N_TICK_COLS, N_OHLC_COLS,
    _TICK_COL, _OHLC_COL,
)
from tradetropy.data.data import (
    OhlcDataStore,
    build_candles_from_ticks,
    TickProxy, OhlcProxy,
    WindowView, OhlcIndicatorView,
)
from tradetropy.ta import SMA, EMA
from tradetropy.models.strategy import Strategy


# ══════════════════════════════════════════════════════════════════════════════
# SHM PRIMITIVES REPLICA (without depending on pool.py)
# ══════════════════════════════════════════════════════════════════════════════
# These functions are exact copies from pool.py, with imports corrected to
# the tradetropy package. They are tested directly here.

@dataclass
class ShmBlock:
    name  : str
    shape : tuple
    dtype : str

    def open(self):
        shm   = shared_memory.SharedMemory(name=self.name)
        array = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return shm, array


def _allocate(array: np.ndarray, shm_refs: list) -> ShmBlock:
    arr  = np.ascontiguousarray(array, dtype=np.float64)
    shm  = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    dest = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    dest[:] = arr
    shm_refs.append(shm)
    return ShmBlock(name=shm.name, shape=arr.shape, dtype=str(arr.dtype))


@dataclass
class RawTickBlocks:
    tick_raw : ShmBlock

@dataclass
class RawOhlcBlocks:
    ohlc_raw    : ShmBlock
    mapping     : ShmBlock
    accumulated : ShmBlock
    candle_ts   : ShmBlock
    prices      : ShmBlock

@dataclass
class SharedStore:
    tick_blocks      : dict
    ohlc_blocks      : dict
    ind_tick_blocks  : dict
    ind_ohlc_blocks  : dict
    _shm_refs        : list = field(default_factory=list, repr=False)

    def release(self):
        for shm in self._shm_refs:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_refs.clear()


def build_shared_store(
    data                   : dict,
    intervals_per_symbol   : dict,
    indicator_defs_union   : list,
) -> SharedStore:
    shm_refs        = []
    tick_blocks     = {}
    ohlc_blocks     = {}
    ind_tick_blocks = {}
    ind_ohlc_blocks = {}

    raw_per_symbol = {}
    for symbol, tick_matrix in data.items():
        ticks = np.ascontiguousarray(tick_matrix[:, :N_TICK_COLS], dtype=np.float64)
        raw_per_symbol[symbol] = ticks
        tick_blocks[symbol] = RawTickBlocks(tick_raw=_allocate(ticks, shm_refs))

    candles_by_key = {}
    for symbol, ticks in raw_per_symbol.items():
        for interval in intervals_per_symbol.get(symbol, []):
            r = build_candles_from_ticks(
                ticks[:, _TICK_COL["ts"]],
                ticks[:, _TICK_COL["price"]],
                ticks[:, _TICK_COL["volume"]],
                interval,
            )
            candles_by_key[(symbol, interval)] = r
            ohlc_blocks[(symbol, interval)] = RawOhlcBlocks(
                ohlc_raw    = _allocate(r["closed_candles"],     shm_refs),
                mapping     = _allocate(r["tick_to_candle_map"],  shm_refs),
                accumulated = _allocate(r["accumulated_per_tick"], shm_refs),
                candle_ts   = _allocate(r["candle_ts_per_tick"],   shm_refs),
                prices      = _allocate(r["prices"],            shm_refs),
            )

    for defn in indicator_defs_union:
        col_name  = defn["col_name"]
        indicator = defn["indicator"]
        src_key   = defn["src_key"]

        if src_key[0] == "tick":
            symbol     = src_key[1]
            shm_key    = (symbol, col_name)
            if shm_key in ind_tick_blocks:
                continue
            source_col = _TICK_COL[defn["source"]._col_name]
            source_arr = raw_per_symbol[symbol][:, source_col]
            result     = indicator.calculate(source_arr)
            ind_tick_blocks[shm_key] = _allocate(result, shm_refs)
        else:
            symbol, interval = src_key[1], src_key[2]
            shm_key = (symbol, interval, col_name)
            if shm_key in ind_ohlc_blocks:
                continue
            source_col = _OHLC_COL[defn["source"]._col_name]
            candles_raw = candles_by_key[(symbol, interval)]["closed_candles"]
            source_arr  = candles_raw[:, source_col] if len(candles_raw) > 0 \
                           else np.array([], dtype=np.float64)
            result      = indicator.calculate(source_arr)
            ind_ohlc_blocks[shm_key] = _allocate(result, shm_refs)

    return SharedStore(
        tick_blocks      = tick_blocks,
        ohlc_blocks      = ohlc_blocks,
        ind_tick_blocks  = ind_tick_blocks,
        ind_ohlc_blocks  = ind_ohlc_blocks,
        _shm_refs        = shm_refs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS: klines → ticks (same as in test_tradetropy.py)
# ══════════════════════════════════════════════════════════════════════════════

def klines_a_ticks(klines: np.ndarray) -> np.ndarray:
    klines = np.asarray(klines, dtype=np.float64)
    close  = klines[:, 4]
    out    = np.empty((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]]          = klines[:, 0]
    out[:, _TICK_COL["bid"]]         = close
    out[:, _TICK_COL["ask"]]         = close
    out[:, _TICK_COL["volume"]]      = klines[:, 5]
    out[:, _TICK_COL["flags"]]       = 0.0
    out[:, _TICK_COL["volume_real"]] = np.nan
    out[:, _TICK_COL["price"]]       = close
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ticks_btc(klines_1m):
    return klines_a_ticks(klines_1m)


@pytest.fixture
def ticks_20k(klines_20k):
    return klines_a_ticks(klines_20k)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: ShmBlock
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestShmBlock:

    def test_open_returns_correct_array(self):
        arr  = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        refs = []
        block = _allocate(arr, refs)
        try:
            shm, view = block.open()
            np.testing.assert_array_equal(view, arr)
            shm.close()
        finally:
            for s in refs:
                s.close(); s.unlink()

    def test_shape_dtype_preserved(self):
        arr  = np.zeros((100, 7), dtype=np.float64)
        refs = []
        block = _allocate(arr, refs)
        try:
            assert block.shape == (100, 7)
            assert "float64" in block.dtype
        finally:
            for s in refs: s.close(); s.unlink()

    def test_view_is_zero_copy(self):
        """The numpy view shares memory with shm — it is not a copy."""
        arr  = np.arange(10, dtype=np.float64)
        refs = []
        block = _allocate(arr, refs)
        try:
            shm, view = block.open()
            # modify shm directly and verify that the view reflects the change
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[0] = 999.0
            assert view[0] == 999.0
            shm.close()
        finally:
            for s in refs: s.close(); s.unlink()

    def test_empty_array_doesnt_fail(self):
        """_allocate must handle empty arrays (size=0 → nbytes=0 → size=1)."""
        arr  = np.empty((0, 6), dtype=np.float64)
        refs = []
        block = _allocate(arr, refs)
        try:
            assert block.shape == (0, 6)
        finally:
            for s in refs: s.close(); s.unlink()

    def test_multiple_blocks_unique_names(self):
        """Each block must have a distinct shm name."""
        refs    = []
        arr     = np.ones(50, dtype=np.float64)
        blocks  = [_allocate(arr, refs) for _ in range(5)]
        names   = [b.name for b in blocks]
        try:
            assert len(set(names)) == 5
        finally:
            for s in refs: s.close(); s.unlink()

    def test_release_clears_all(self):
        """After release, trying to open the shm by name must fail."""
        arr   = np.ones(20, dtype=np.float64)
        refs  = []
        block = _allocate(arr, refs)
        name  = block.name
        for s in refs: s.close(); s.unlink()
        refs.clear()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: _allocate
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAllocate:

    def test_correct_content_2d(self):
        arr  = np.arange(12, dtype=np.float64).reshape(3, 4)
        refs = []
        b = _allocate(arr, refs)
        try:
            shm, view = b.open()
            np.testing.assert_array_equal(view, arr)
            shm.close()
        finally:
            for s in refs: s.close(); s.unlink()

    def test_converts_to_float64(self):
        arr  = np.array([1, 2, 3], dtype=np.int32)
        refs = []
        b = _allocate(arr, refs)
        try:
            assert "float64" in b.dtype
        finally:
            for s in refs: s.close(); s.unlink()

    def test_adds_to_shm_refs(self):
        arr  = np.zeros(10, dtype=np.float64)
        refs = []
        _allocate(arr, refs)
        _allocate(arr, refs)
        try:
            assert len(refs) == 2
        finally:
            for s in refs: s.close(); s.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — tick blocks
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSharedStoreTicks:

    def test_tick_block_shape(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, []
        )
        try:
            shm, view = store.tick_blocks["BTCUSDT"].tick_raw.open()
            assert view.shape == (len(ticks_btc), N_TICK_COLS)
            shm.close()
        finally:
            store.release()

    def test_tick_block_correct_values(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, []
        )
        try:
            shm, view = store.tick_blocks["BTCUSDT"].tick_raw.open()
            np.testing.assert_array_equal(view[:, _TICK_COL["price"]],
                                          ticks_btc[:, _TICK_COL["price"]])
            shm.close()
        finally:
            store.release()

    def test_multiples_simbolos(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc, "ETHUSDT": ticks_btc},
            {}, []
        )
        try:
            assert "BTCUSDT" in store.tick_blocks
            assert "ETHUSDT" in store.tick_blocks
        finally:
            store.release()

    def test_release_empties_refs(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, []
        )
        name = store.tick_blocks["BTCUSDT"].tick_raw.name
        store.release()
        assert store._shm_refs == []
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — ohlc blocks
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSharedStoreOhlc:

    def test_ohlc_block_exists(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            assert ("BTCUSDT", 300_000) in store.ohlc_blocks
        finally:
            store.release()

    def test_ohlc_raw_contains_closed_candles(self, ticks_btc):
        """Closed candles have M-1 rows relative to the total."""
        r = build_candles_from_ticks(
            ticks_btc[:, _TICK_COL["ts"]],
            ticks_btc[:, _TICK_COL["price"]],
            ticks_btc[:, _TICK_COL["volume"]],
            300_000,
        )
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm, view = ok.ohlc_raw.open()
            assert view.shape[0] == len(r["closed_candles"])
            shm.close()
        finally:
            store.release()

    def test_mapping_shape(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm, view = ok.mapping.open()
            assert view.shape == (len(ticks_btc),)
            shm.close()
        finally:
            store.release()

    def test_accumulated_shape(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm, view = ok.accumulated.open()
            assert view.shape == (len(ticks_btc), 4)
            shm.close()
        finally:
            store.release()

    def test_multiples_intervalos(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000, 900_000, 3_600_000]},
            []
        )
        try:
            for iv in [300_000, 900_000, 3_600_000]:
                assert ("BTCUSDT", iv) in store.ohlc_blocks
        finally:
            store.release()

    def test_candle_ts_monotonic(self, ticks_btc):
        """Candle start timestamps must be non-decreasing."""
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm, view = ok.candle_ts.open()
            assert np.all(np.diff(view) >= 0)
            shm.close()
        finally:
            store.release()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — indicators
# ══════════════════════════════════════════════════════════════════════════════

def _make_tick_proxy(symbol="BTCUSDT", window=20):
    return TickProxy(symbol, window)

def _make_ohlc_proxy(symbol="BTCUSDT", interval=300_000, window=50):
    return OhlcProxy(symbol, interval, window)

def _make_ind_def_tick(symbol, col_name_proxy, sma_length, src_tick_proxy):
    from tradetropy.data.data import ColumnRef
    sma = SMA(length=sma_length)
    col_ref = ColumnRef(src_tick_proxy, col_name_proxy)
    return {
        "source"   : col_ref,
        "indicator": sma,
        "col_name" : sma.col_name(col_name_proxy, symbol),
        "src_key"  : ("tick", symbol),
    }

def _make_ind_def_ohlc(symbol, col_name_proxy, sma_length, src_ohlc_proxy):
    from tradetropy.data.data import ColumnRef
    sma = SMA(length=sma_length)
    col_ref = ColumnRef(src_ohlc_proxy, col_name_proxy)
    return {
        "source"   : col_ref,
        "indicator": sma,
        "col_name" : sma.col_name(col_name_proxy, symbol),
        "src_key"  : ("ohlc", symbol, src_ohlc_proxy.interval_ms),
    }


@pytest.mark.unit
class TestSharedStoreIndicators:

    def test_ind_tick_calculated_in_shm(self, ticks_btc):
        tp = _make_tick_proxy()
        defn = _make_ind_def_tick("BTCUSDT", "price", 3, tp)
        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, [defn]
        )
        try:
            assert ("BTCUSDT", defn["col_name"]) in store.ind_tick_blocks
        finally:
            store.release()

    def test_ind_tick_correct_values(self, ticks_btc):
        tp    = _make_tick_proxy()
        defn  = _make_ind_def_tick("BTCUSDT", "price", 3, tp)
        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, [defn]
        )
        try:
            shm, view = store.ind_tick_blocks[("BTCUSDT", defn["col_name"])].open()
            expected_sma = SMA(3).calculate(ticks_btc[:, _TICK_COL["price"]])
            np.testing.assert_array_almost_equal(view, expected_sma)
            shm.close()
        finally:
            store.release()

    def test_ind_ohlc_calculated_in_shm(self, ticks_btc):
        op   = _make_ohlc_proxy(interval=300_000)
        defn = _make_ind_def_ohlc("BTCUSDT", "close", 3, op)
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            [defn]
        )
        try:
            assert ("BTCUSDT", 300_000, defn["col_name"]) in store.ind_ohlc_blocks
        finally:
            store.release()

    def test_ind_ohlc_shape_equals_closed_candles(self, ticks_btc):
        op   = _make_ohlc_proxy(interval=300_000)
        defn = _make_ind_def_ohlc("BTCUSDT", "close", 3, op)
        r    = build_candles_from_ticks(
            ticks_btc[:, _TICK_COL["ts"]],
            ticks_btc[:, _TICK_COL["price"]],
            ticks_btc[:, _TICK_COL["volume"]],
            300_000,
        )
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            [defn]
        )
        try:
            shm, view = store.ind_ohlc_blocks[("BTCUSDT", 300_000, defn["col_name"])].open()
            assert len(view) == len(r["closed_candles"])
            shm.close()
        finally:
            store.release()

    def test_dedup_same_key(self, ticks_btc):
        """Two defns with the same col_name → single shm block."""
        tp    = _make_tick_proxy()
        defn1 = _make_ind_def_tick("BTCUSDT", "price", 5, tp)
        defn2 = _make_ind_def_tick("BTCUSDT", "price", 5, tp)  # identical
        assert defn1["col_name"] == defn2["col_name"]

        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, [defn1, defn2]
        )
        try:
            # Only one block should exist (deduplicated)
            assert len(store.ind_tick_blocks) == 1
        finally:
            store.release()

    def test_no_dedup_different_key(self, ticks_btc):
        """SMA(5) and SMA(10) are different → two blocks."""
        tp    = _make_tick_proxy()
        defn1 = _make_ind_def_tick("BTCUSDT", "price", 5,  tp)
        defn2 = _make_ind_def_tick("BTCUSDT", "price", 10, tp)
        assert defn1["col_name"] != defn2["col_name"]

        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, [defn1, defn2]
        )
        try:
            assert len(store.ind_tick_blocks) == 2
        finally:
            store.release()

    def test_shm_refs_count_correct(self, ticks_btc):
        """
        Without indicators:
          1 tick raw
          + per interval: 5 blocks (ohlc_raw, mapping, accumulated, candle_ts, prices)
        """
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            # 1 tick_raw + 5 ohlc = 6
            assert len(store._shm_refs) == 6
        finally:
            store.release()

    def test_ind_tick_on_bid_col(self, ticks_btc):
        """Indicator on the 'bid' column of tick — not just 'price'."""
        tp   = _make_tick_proxy()
        from tradetropy.data.data import ColumnRef
        sma  = SMA(length=3)
        col_ref = ColumnRef(tp, "bid")
        defn = {
            "source"   : col_ref,
            "indicator": sma,
            "col_name" : sma.col_name("bid", "BTCUSDT"),
            "src_key"  : ("tick", "BTCUSDT"),
        }
        store = build_shared_store(
            {"BTCUSDT": ticks_btc}, {}, [defn]
        )
        try:
            shm, view = store.ind_tick_blocks[("BTCUSDT", defn["col_name"])].open()
            expected = SMA(3).calculate(ticks_btc[:, _TICK_COL["bid"]])
            np.testing.assert_array_almost_equal(view, expected)
            shm.close()
        finally:
            store.release()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SharedStore.release
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSharedStoreRelease:

    def test_release_leaves_refs_empty(self, ticks_btc):
        store = build_shared_store({"BTCUSDT": ticks_btc}, {}, [])
        store.release()
        assert store._shm_refs == []

    def test_release_twice_no_error(self, ticks_btc):
        store = build_shared_store({"BTCUSDT": ticks_btc}, {}, [])
        store.release()
        store.release()   # second time must be silent

    def test_shm_inaccessible_after_release(self, ticks_btc):
        store = build_shared_store({"BTCUSDT": ticks_btc}, {}, [])
        name  = store.tick_blocks["BTCUSDT"].tick_raw.name
        store.release()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    def test_release_with_multiple_intervals(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000, 900_000]},
            []
        )
        store.release()
        assert store._shm_refs == []


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Raw data integrity in SHM (exact values)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSharedStoreIntegrity:

    def test_tick_prices_preserved(self, ticks_btc):
        store = build_shared_store({"BTCUSDT": ticks_btc}, {}, [])
        try:
            shm, view = store.tick_blocks["BTCUSDT"].tick_raw.open()
            np.testing.assert_array_equal(
                view[:, _TICK_COL["price"]],
                ticks_btc[:, _TICK_COL["price"]]
            )
            shm.close()
        finally:
            store.release()

    def test_tick_timestamps_preserved(self, ticks_btc):
        store = build_shared_store({"BTCUSDT": ticks_btc}, {}, [])
        try:
            shm, view = store.tick_blocks["BTCUSDT"].tick_raw.open()
            np.testing.assert_array_equal(
                view[:, _TICK_COL["ts"]],
                ticks_btc[:, _TICK_COL["ts"]]
            )
            shm.close()
        finally:
            store.release()

    def test_ohlc_raw_valid_columns(self, ticks_btc):
        """Closed candles must have high >= low in all cases."""
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok   = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm, ohlc = ok.ohlc_raw.open()
            if len(ohlc) > 0:
                assert np.all(ohlc[:, _OHLC_COL["high"]] >= ohlc[:, _OHLC_COL["low"]])
            shm.close()
        finally:
            store.release()

    def test_mapping_monotonic_non_decreasing(self, ticks_btc):
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok   = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm, mapping = ok.mapping.open()
            assert np.all(np.diff(mapping.astype(np.int64)) >= 0)
            shm.close()
        finally:
            store.release()

    def test_accumulated_open_is_first_candle_price(self, ticks_btc):
        """
        accumulated[:, 0] = open of the current candle for each tick.
        The first tick of each candle must have open == price of that tick.
        """
        store = build_shared_store(
            {"BTCUSDT": ticks_btc},
            {"BTCUSDT": [300_000]},
            []
        )
        try:
            ok = store.ohlc_blocks[("BTCUSDT", 300_000)]
            shm_mapping,   mapping   = ok.mapping.open()
            shm_accum,     accum     = ok.accumulated.open()
            shm_prices,    prices    = ok.prices.open()

            # For each candle change (first tick), open must equal price
            changes = np.r_[True, np.diff(mapping.astype(np.int64)) > 0]
            np.testing.assert_array_almost_equal(
                accum[changes, 0],   # open at first tick of candle
                prices[changes],     # price at that tick
            )
            shm_mapping.close(); shm_accum.close(); shm_prices.close()
        finally:
            store.release()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SHM with large dataset (20k ticks)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestSharedStoreGrande:

    def ok(self, ticks_20k):
        store = build_shared_store({"BTCUSDT": ticks_20k}, {}, [])
        try:
            shm, view = store.tick_blocks["BTCUSDT"].tick_raw.open()
            assert view.shape == (20_000, N_TICK_COLS)
            shm.close()
        finally:
            store.release()

    def test_20k_multiple_intervals_and_sma(self, ticks_20k):
        tp   = _make_tick_proxy()
        defn = _make_ind_def_tick("BTCUSDT", "price", 20, tp)
        store = build_shared_store(
            {"BTCUSDT": ticks_20k},
            {"BTCUSDT": [300_000, 900_000]},
            [defn]
        )
        try:
            assert ("BTCUSDT", 300_000)  in store.ohlc_blocks
            assert ("BTCUSDT", 900_000) in store.ohlc_blocks
            assert ("BTCUSDT", defn["col_name"]) in store.ind_tick_blocks
        finally:
            store.release()

    def test_release_20k_clears_all(self, ticks_20k):
        tp   = _make_tick_proxy()
        defn = _make_ind_def_tick("BTCUSDT", "price", 10, tp)
        store = build_shared_store(
            {"BTCUSDT": ticks_20k},
            {"BTCUSDT": [300_000]},
            [defn]
        )
        n_refs = len(store._shm_refs)
        assert n_refs > 0
        store.release()
        assert store._shm_refs == []

# endregion

# region test_pool.py v1.0
"""
Tests for pool.py — Suite v1.0

Coverage:
  · ShmBlock / _allocate            — descriptor, open, zero-copy, shape/dtype
  · RawFpBlocks / SharedStore     — containers and release()
  · _fp_config_key                — FootprintConfig deduplication key
  · build_shared_store        — tick raw, ohlc raw, fp blocks, deduplicated indicators
  · _scan_strategies         — collects intervals, fp_infos, indicators
  · _TickStoreHybrid              — construction, column access
  · _worker                       — complete loop in child process
  · PoolBacktestEngine.by_ticks — end-to-end integration (1 worker, N workers)
  · Equivalence with backtest     — identical results to BacktestEngine.for_ticks
  · Indicator deduplication       — same SMA → single shm block
  · Footprint in pool             — subscribe_footprint works in workers
  · Warmup                        — on_data not called before min_periods

IMPORTANT: all strategies must be defined at module level
(not inside functions) so that pickle works in 'spawn' mode.

Usage:
    pytest test_pool.py -v
    pytest test_pool.py -v -m unit
    pytest test_pool.py -v -m integration
    pytest test_pool.py -v -m slow
"""

import pytest
import numpy as np
from multiprocessing import shared_memory

from tradetropy.core.constants import (
    TICK_COLS,
    OHLC_COLS,
    N_TICK_COLS,
    N_OHLC_COLS,
    _TICK_COL,
    _OHLC_COL,
)
from tradetropy.data.data import build_candles_from_ticks
from tradetropy.data.data import TickProxy, OhlcProxy
from tradetropy.ta import SMA, EMA
from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.core.data_types import TickData
from tradetropy.models.footprint import FootprintConfig
from tradetropy.session.base import SeshSimulatorBase

from tradetropy.backtest.pool import (
    ShmBlock,
    _allocate,
    RawTickBlocks,
    RawOhlcBlocks,
    RawFpBlocks,
    SharedStore,
    _fp_config_key,
    build_shared_store,
    _scan_strategies,
    _TickStoreHybrid,
    _worker,
    PoolBacktestEngine,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    """Converts klines [N×6+] to tick_matrix [N×7] using close as price."""
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.empty((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["flags"]] = 0.0
    out[:, _TICK_COL["volume_real"]] = np.nan
    out[:, _TICK_COL["price"]] = close
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES AT MODULE LEVEL (required for pickle with spawn)
# ══════════════════════════════════════════════════════════════════════════════


class _CounterStrategy(Strategy):
    """Counts how many times on_data is called and returns the result."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.n_calls = 0

    def on_data(self):
        self.n_calls += 1

    def result(self):
        return {"n_calls": self.n_calls}


class _LastPriceStrategy(Strategy):
    """Captures the last price seen."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
        self.last_price = np.nan

    def on_data(self):
        self.last_price = float(self._tp.price[-1])

    def result(self):
        return {"price": self.last_price}


class _SmaStrategy(Strategy):
    """SMA(3) on price. Returns last non-nan value."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
        self._sma = self.add_indicator(self._tp.price_ref, SMA(length=3))
        self.last_sma = np.nan
        self.n_calls = 0

    def on_data(self):
        v = float(self._sma[-1])
        if not np.isnan(v):
            self.last_sma = v
        self.n_calls += 1

    def result(self):
        return {"sma": self.last_sma, "n_calls": self.n_calls}


class _SmaOhlcStrategy(Strategy):
    """SMA(3) on close of 5m candles."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=50)
        self._sma = self.add_indicator(self._ohlc.close_ref, SMA(length=3))
        self.last_sma = np.nan
        self.last_close = np.nan
        self.n_candles = 0

    def on_data(self):
        self.last_close = float(self._ohlc.close[-1])
        self.n_candles = self._ohlc.n_klines
        v = float(self._sma[-1])
        if not np.isnan(v):
            self.last_sma = v

    def result(self):
        return {
            "sma": self.last_sma,
            "close": self.last_close,
            "n_candles": self.n_candles,
        }


class _EmaStrategy(Strategy):
    """EMA(3) on price."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
        self._ema = self.add_indicator(self._tp.price_ref, EMA(length=3))
        self.last_ema = np.nan

    def on_data(self):
        v = float(self._ema[-1])
        if not np.isnan(v):
            self.last_ema = v

    def result(self):
        return {"ema": self.last_ema}


class _FpStrategy(Strategy):
    """subscribe_footprint — returns n_fp_candles from footprint at end."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._fp = self.subscribe_footprint(
            "BTCUSDT",
            300_000,
            window_size=20,
            tick_size=1.0,
        )
        self.n_fp_candles = 0

    def on_data(self):
        self.n_fp_candles = self._fp.n_candles

    def result(self):
        return {"n_fp_candles": self.n_fp_candles}


class _MultiSymbolStrategy(Strategy):
    """Two simultaneous symbols."""

    def init(self):
        self._btc = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._eth = self.subscribe_ticks("ETHUSDT", window_size=50)
        self.last_btc = np.nan
        self.last_eth = np.nan

    def on_data(self):
        self.last_btc = float(self._btc.price[-1])
        self.last_eth = float(self._eth.price[-1])

    def result(self):
        return {"btc": self.last_btc, "eth": self.last_eth}


class _NoResultStrategy(Strategy):
    """Strategy without result() method -- should return None."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=10)

    def on_data(self):
        pass


class _WarmupStrategy(Strategy):
    """Verifies on_data is not called until after warmup (SMA length=5)."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._sma5 = self.add_indicator(self._tp.price_ref, SMA(length=5))
        self.first_smas = []

    def on_data(self):
        self.first_smas.append(float(self._sma5[-1]))

    def result(self):
        return {"first_smas": self.first_smas}


class _OhlcOnlyStrategy(Strategy):
    """Only subscribe_ohlc (no subscribe_ticks) — primary symbol from ohlc_proxy."""

    def init(self):
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=20)
        self.last_close = np.nan

    def on_data(self):
        self.last_close = float(self._ohlc.close[-1])

    def result(self):
        return {"close": self.last_close}


class _Sma3Strategy(Strategy):
    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._sma = self.add_indicator(self._tp.price_ref, SMA(length=3))
        self.val = np.nan

    def on_data(self):
        v = float(self._sma[-1])
        if not np.isnan(v):
            self.val = v

    def result(self):
        return self.val


class _Sma5Strategy(Strategy):
    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._sma = self.add_indicator(self._tp.price_ref, SMA(length=5))
        self.val = np.nan

    def on_data(self):
        v = float(self._sma[-1])
        if not np.isnan(v):
            self.val = v

    def result(self):
        return self.val


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ticks_btc(klines_1m):
    return klines_to_ticks(klines_1m)


@pytest.fixture
def ticks_20k(klines_20k):
    return klines_to_ticks(klines_20k)


@pytest.fixture
def data_btc(ticks_btc):
    return (TickData("BTCUSDT", ticks_btc, tick_size=0.01),)


@pytest.fixture
def data_20k(ticks_20k):
    return (TickData("BTCUSDT", ticks_20k, tick_size=0.01),)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: _fp_config_key
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFpConfigKey:
    def test_same_config_same_key(self):
        c1 = FootprintConfig(tick_size=0.5, value_area_pct=0.70, aggressor_col="flags")
        c2 = FootprintConfig(tick_size=0.5, value_area_pct=0.70, aggressor_col="flags")
        assert _fp_config_key(c1) == _fp_config_key(c2)

    def test_different_tick_size_different_key(self):
        c1 = FootprintConfig(tick_size=0.5)
        c2 = FootprintConfig(tick_size=1.0)
        assert _fp_config_key(c1) != _fp_config_key(c2)

    def test_different_value_area_different_key(self):
        c1 = FootprintConfig(value_area_pct=0.70)
        c2 = FootprintConfig(value_area_pct=0.80)
        assert _fp_config_key(c1) != _fp_config_key(c2)

    def test_different_aggressor_col_different_key(self):
        c1 = FootprintConfig(aggressor_col="flags")
        c2 = FootprintConfig(aggressor_col=None)
        assert _fp_config_key(c1) != _fp_config_key(c2)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — tick blocks
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSharedStoreTicks:
    def test_tick_block_shape(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            shm, vista = store.tick_blocks["BTCUSDT"].tick_raw.open()
            assert vista.shape == (len(ticks_btc), N_TICK_COLS)
            shm.close()
        finally:
            store.release()

    def test_tick_block_correct_prices(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            shm, vista = store.tick_blocks["BTCUSDT"].tick_raw.open()
            np.testing.assert_array_equal(
                vista[:, _TICK_COL["price"]],
                ticks_btc[:, _TICK_COL["price"]],
            )
            shm.close()
        finally:
            store.release()

    def test_multiple_symbols(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc, "ETHUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            assert "BTCUSDT" in store.tick_blocks
            assert "ETHUSDT" in store.tick_blocks
        finally:
            store.release()

    def test_release_empties_refs_and_blocks_access(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        shm_name = store.tick_blocks["BTCUSDT"].tick_raw.name
        store.release()
        assert store._shm_refs == []
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — OHLC blocks
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSharedStoreOhlc:
    def test_ohlc_block_exists_for_interval(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            assert ("BTCUSDT", 300_000) in store.ohlc_blocks
        finally:
            store.release()

    def test_ohlc_raw_rows_equals_closed_candles(self, ticks_btc):
        r = build_candles_from_ticks(
            ticks_btc[:, _TICK_COL["ts"]],
            ticks_btc[:, _TICK_COL["price"]],
            ticks_btc[:, _TICK_COL["volume"]],
            300_000,
        )
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            shm, vista = store.ohlc_blocks[("BTCUSDT", 300_000)].ohlc_raw.open()
            assert vista.shape[0] == len(r["closed_candles"])
            shm.close()
        finally:
            store.release()

    def test_mapping_shape_equals_n_ticks(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            shm, vista = store.ohlc_blocks[("BTCUSDT", 300_000)].tick_to_candle.open()
            assert vista.shape == (len(ticks_btc),)
            shm.close()
        finally:
            store.release()

    def test_mapping_non_decreasing(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            shm, mapping = store.ohlc_blocks[("BTCUSDT", 300_000)].tick_to_candle.open()
            assert np.all(np.diff(mapping.astype(np.int64)) >= 0)
            shm.close()
        finally:
            store.release()

    def test_multiple_intervals(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000, 900_000]},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            assert ("BTCUSDT", 300_000) in store.ohlc_blocks
            assert ("BTCUSDT", 900_000) in store.ohlc_blocks
        finally:
            store.release()

    def test_high_gte_low_in_closed_candles(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            shm, ohlc = store.ohlc_blocks[("BTCUSDT", 300_000)].ohlc_raw.open()
            if len(ohlc) > 0:
                assert np.all(ohlc[:, _OHLC_COL["high"]] >= ohlc[:, _OHLC_COL["low"]])
            shm.close()
        finally:
            store.release()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — FP blocks
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSharedStoreFp:
    def _fp_infos(self, config=None):
        config = config or FootprintConfig(tick_size=1.0)
        return [
            {
                "symbol": "BTCUSDT",
                "interval_ms": 300_000,
                "config": config,
            }
        ]

    def test_fp_block_exists(self, ticks_btc):
        cfg = FootprintConfig(tick_size=1.0)
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=self._fp_infos(cfg),
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            key = ("BTCUSDT", 300_000, _fp_config_key(cfg))
            assert key in store.fp_blocks
        finally:
            store.release()

    def test_fp_block_fp_idx_shape(self, ticks_btc):
        cfg = FootprintConfig(tick_size=1.0)
        r = build_candles_from_ticks(
            ticks_btc[:, _TICK_COL["ts"]],
            ticks_btc[:, _TICK_COL["price"]],
            ticks_btc[:, _TICK_COL["volume"]],
            300_000,
        )
        n_closed = len(r["closed_candles"])
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=self._fp_infos(cfg),
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            key = ("BTCUSDT", 300_000, _fp_config_key(cfg))
            shm, fp_idx = store.fp_blocks[key].fp_idx.open()
            # fp_idx has n_closed + 1 entries (CSR)
            assert fp_idx.shape[0] == n_closed + 1
            shm.close()
        finally:
            store.release()

    def test_fp_dedup_same_config(self, ticks_btc):
        """Two fp_infos with same config → single shm block."""
        cfg = FootprintConfig(tick_size=1.0)
        infos = self._fp_infos(cfg) + self._fp_infos(cfg)
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=infos,
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            assert len(store.fp_blocks) == 1
        finally:
            store.release()

    def test_fp_two_different_configs(self, ticks_btc):
        """Two different configs → two shm blocks."""
        cfg1 = FootprintConfig(tick_size=0.5)
        cfg2 = FootprintConfig(tick_size=1.0)
        infos = [
            {"symbol": "BTCUSDT", "interval_ms": 300_000, "config": cfg1},
            {"symbol": "BTCUSDT", "interval_ms": 300_000, "config": cfg2},
        ]
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=infos,
            indicator_defs_union=[],
            feed_type="tick",
        )
        try:
            assert len(store.fp_blocks) == 2
        finally:
            store.release()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: build_shared_store — deduplicated indicators
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSharedStoreIndicators:
    def _defn_tick(self, symbol="BTCUSDT", length=3, col="price"):
        tp = TickProxy(symbol, 50)
        sma = SMA(length=length)
        from tradetropy.data.data import ColumnRef

        return {
            "source": ColumnRef(tp, col),
            "indicator": sma,
            "col_name": sma.col_name(col, symbol),
            "src_key": ("tick", symbol),
        }

    def _defn_ohlc(self, symbol="BTCUSDT", length=3, col="close", interval=300_000):
        op = OhlcProxy(symbol, interval, 50)
        sma = SMA(length=length)
        from tradetropy.data.data import ColumnRef

        return {
            "source": ColumnRef(op, col),
            "indicator": sma,
            "col_name": sma.col_name(col, symbol),
            "src_key": ("ohlc", symbol, interval),
        }

    def test_ind_tick_exists_in_shm(self, ticks_btc):
        defn = self._defn_tick()
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[defn],
            feed_type="tick",
        )
        try:
            assert ("BTCUSDT", defn["col_name"]) in store.ind_tick_blocks
        finally:
            store.release()

    def test_ind_tick_correct_values(self, ticks_btc):
        defn = self._defn_tick(length=3)
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[defn],
            feed_type="tick",
        )
        try:
            shm, vista = store.ind_tick_blocks[("BTCUSDT", defn["col_name"])].open()
            expected = SMA(3).calculate(ticks_btc[:, _TICK_COL["price"]])
            np.testing.assert_array_almost_equal(vista, expected)
            shm.close()
        finally:
            store.release()

    def test_dedup_same_tick_key(self, ticks_btc):
        defn1 = self._defn_tick(length=5)
        defn2 = self._defn_tick(length=5)  # identical
        assert defn1["col_name"] == defn2["col_name"]
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[defn1, defn2],
            feed_type="tick",
        )
        try:
            assert len(store.ind_tick_blocks) == 1
        finally:
            store.release()

    def test_no_dedup_different_length(self, ticks_btc):
        defn5 = self._defn_tick(length=5)
        defn10 = self._defn_tick(length=10)
        assert defn5["col_name"] != defn10["col_name"]
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[defn5, defn10],
            feed_type="tick",
        )
        try:
            assert len(store.ind_tick_blocks) == 2
        finally:
            store.release()

    def test_ind_ohlc_exists_in_shm(self, ticks_btc):
        defn = self._defn_ohlc(length=3)
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[defn],
            feed_type="tick",
        )
        try:
            assert ("BTCUSDT", 300_000, defn["col_name"]) in store.ind_ohlc_blocks
        finally:
            store.release()

    def test_ind_ohlc_shape_equals_closed_candles(self, ticks_btc):
        defn = self._defn_ohlc(length=3)
        r = build_candles_from_ticks(
            ticks_btc[:, _TICK_COL["ts"]],
            ticks_btc[:, _TICK_COL["price"]],
            ticks_btc[:, _TICK_COL["volume"]],
            300_000,
        )
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={"BTCUSDT": [300_000]},
            fp_infos=[],
            indicator_defs_union=[defn],
            feed_type="tick",
        )
        try:
            shm, vista = store.ind_ohlc_blocks[
                ("BTCUSDT", 300_000, defn["col_name"])
            ].open()
            assert len(vista) == len(r["closed_candles"])
            shm.close()
        finally:
            store.release()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SharedStore.liberar
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSharedStoreRelease:
    def test_release_empties_refs(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        store.release()
        assert store._shm_refs == []

    def test_release_twice_no_error(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        store.release()
        store.release()  # second call is silent

    def test_shm_inaccessible_after_release(self, ticks_btc):
        store = build_shared_store(
            data={"BTCUSDT": ticks_btc},
            intervals_per_symbol={},
            fp_infos=[],
            indicator_defs_union=[],
            feed_type="tick",
        )
        shm_name = store.tick_blocks["BTCUSDT"].tick_raw.name
        store.release()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: _scan_strategies
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestScanStrategies:
    def test_collects_ohlc_intervals(self):
        intervals, fp_infos, ind_union, _ = _scan_strategies([_SmaOhlcStrategy()])
        assert "BTCUSDT" in intervals
        assert 300_000 in intervals["BTCUSDT"]

    def test_collects_tick_indicators(self):
        _, _, ind_union, _ = _scan_strategies([_SmaStrategy()])
        assert len(ind_union) == 1
        assert ind_union[0]["src_key"][0] == "tick"

    def test_collects_ohlc_indicators(self):
        _, _, ind_union, _ = _scan_strategies([_SmaOhlcStrategy()])
        assert any(d["src_key"][0] == "ohlc" for d in ind_union)

    def test_dedup_same_indicators(self):
        """Two strategies with same SMA → single defn in ind_union."""
        _, _, ind_union, _ = _scan_strategies([_SmaStrategy(), _SmaStrategy()])
        col_names = [d["col_name"] for d in ind_union]
        assert len(col_names) == len(set(col_names))

    def test_collects_fp_infos(self):
        _, fp_infos, _, _ = _scan_strategies([_FpStrategy()])
        assert len(fp_infos) == 1
        assert fp_infos[0]["symbol"] == "BTCUSDT"
        assert fp_infos[0]["interval_ms"] == 300_000

    def test_resets_strategy_proxies(self):
        """After scanning, _tick_proxies etc. must be empty."""
        s = _SmaStrategy()
        _scan_strategies([s])
        assert s._tick_proxies == []
        assert s._ohlc_proxies == []
        assert s._indicator_defs == []
        assert s._fp_proxies == []

    def test_strategy_without_subscriptions(self):
        """Empty strategy must not raise."""

        class S(Strategy):
            def init(self):
                pass

            def on_data(self):
                pass

        intervals, fp_infos, ind, _ = _scan_strategies([S()])
        assert intervals == {}
        assert fp_infos == []
        assert ind == []


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: _TickStoreHybrid
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTickStoreHybrid:
    def test_no_ind_cols_matrix_is_raw(self, ticks_btc):
        col_index = dict(zip(TICK_COLS, range(N_TICK_COLS)))
        store = _TickStoreHybrid(ticks_btc, [], col_index)
        assert store.matrix is ticks_btc or np.shares_memory(store.matrix, ticks_btc)

    def test_with_ind_cols_extends_matrix(self, ticks_btc):
        col_index = dict(zip(TICK_COLS, range(N_TICK_COLS)))
        ind_col = SMA(3).calculate(ticks_btc[:, _TICK_COL["price"]]).reshape(-1, 1)
        col_index["sma3_BTCUSDT_price"] = N_TICK_COLS
        store = _TickStoreHybrid(ticks_btc, [ind_col], col_index)
        assert store.matrix.shape == (len(ticks_btc), N_TICK_COLS + 1)

    def test_col_method_returns_correct_column(self, ticks_btc):
        col_index = dict(zip(TICK_COLS, range(N_TICK_COLS)))
        store = _TickStoreHybrid(ticks_btc, [], col_index)
        np.testing.assert_array_equal(
            store.col("price"),
            ticks_btc[:, _TICK_COL["price"]],
        )

    def test_n_ticks_correct(self, ticks_btc):
        col_index = dict(zip(TICK_COLS, range(N_TICK_COLS)))
        store = _TickStoreHybrid(ticks_btc, [], col_index)
        assert store.n_ticks == len(ticks_btc)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PoolBacktestEngine.by_ticks — integration
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestPoolBacktestEngineRun:
    def test_empty_list_returns_empty(self, data_btc):
        results = PoolBacktestEngine.by_ticks([], data=data_btc, workers=1)
        assert results == []

    def test_one_strategy_one_worker(self, data_btc):
        results = PoolBacktestEngine.by_ticks(
            [_CounterStrategy()], data=data_btc, workers=1
        )
        assert len(results) == 1
        assert results[0]["n_calls"] == len(data_btc[0].data)

    def test_multiple_strategies_same_type(self, data_btc, ticks_btc):
        n = len(ticks_btc)
        results = PoolBacktestEngine.by_ticks(
            [_CounterStrategy(), _CounterStrategy(), _CounterStrategy()],
            data=data_btc,
            workers=1,
        )
        assert len(results) == 3
        assert all(r["n_calls"] == n for r in results)

    def test_strategy_without_result_returns_none(self, data_btc):
        results = PoolBacktestEngine.by_ticks(
            [_NoResultStrategy()], data=data_btc, workers=1
        )
        assert results[0] is None

    def test_multiple_strategy_types(self, data_btc):
        results = PoolBacktestEngine.by_ticks(
            [_LastPriceStrategy(), _SmaStrategy()],
            data=data_btc,
            workers=1,
        )
        assert len(results) == 2
        assert not np.isnan(results[0]["price"])
        assert not np.isnan(results[1]["sma"])

    def test_workers_2(self, data_btc, ticks_btc):
        """With 2 workers and 4 identical strategies results must match."""
        strats = [_LastPriceStrategy() for _ in range(4)]
        results = PoolBacktestEngine.by_ticks(strats, data=data_btc, workers=2)
        prices = [r["price"] for r in results]
        assert all(p == pytest.approx(prices[0]) for p in prices)

    def test_last_price_result_correct(self, data_btc, ticks_btc):
        """Last price in pool == last price in BacktestEngine.for_ticks."""

        # Reference backtest
        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
                self.last = np.nan

            def on_data(self):
                self.last = float(self._tp.price[-1])

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_LastPriceStrategy()], data=data_btc, workers=1
        )
        assert results[0]["price"] == pytest.approx(ref.last, rel=1e-6)

    def test_n_calls_equals_n_ticks(self, data_btc, ticks_btc):
        results = PoolBacktestEngine.by_ticks(
            [_CounterStrategy()], data=data_btc, workers=1
        )
        assert results[0]["n_calls"] == len(ticks_btc)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Equivalence Pool vs BacktestEngine.for_ticks
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestEquivalencePoolVsBacktest:
    def test_sma_tick_matches(self, data_btc, ticks_btc):
        # Reference with BacktestEngine.for_ticks
        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
                self._sma = self.add_indicator(self._tp.price_ref, SMA(length=3))
                self.last_sma = np.nan

            def on_data(self):
                v = float(self._sma[-1])
                if not np.isnan(v):
                    self.last_sma = v

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_SmaStrategy()], data=data_btc, workers=1
        )
        assert results[0]["sma"] == pytest.approx(ref.last_sma, rel=1e-5)

    def test_ema_tick_matches(self, data_btc, ticks_btc):
        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
                self._ema = self.add_indicator(self._tp.price_ref, EMA(length=3))
                self.last_ema = np.nan

            def on_data(self):
                v = float(self._ema[-1])
                if not np.isnan(v):
                    self.last_ema = v

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_EmaStrategy()], data=data_btc, workers=1
        )
        assert results[0]["ema"] == pytest.approx(ref.last_ema, rel=1e-5)

    def test_sma_ohlc_close_matches(self, data_btc, ticks_btc):
        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=50)
                self._sma = self.add_indicator(self._ohlc.close_ref, SMA(length=3))
                self.last_sma = np.nan
                self.last_close = np.nan

            def on_data(self):
                self.last_close = float(self._ohlc.close[-1])
                v = float(self._sma[-1])
                if not np.isnan(v):
                    self.last_sma = v

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_SmaOhlcStrategy()], data=data_btc, workers=1
        )
        r = results[0]
        assert r["close"] == pytest.approx(ref.last_close, rel=1e-6)
        assert r["sma"] == pytest.approx(ref.last_sma, rel=1e-5)

    def test_n_candles_ohlc_matches(self, data_btc, ticks_btc):
        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=50)
                self.n_candles = 0

            def on_data(self):
                self.n_candles = self._ohlc.n_klines

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_SmaOhlcStrategy()], data=data_btc, workers=1
        )
        assert results[0]["n_candles"] == ref.n_candles

    def test_multiple_strategies_same_results(self, data_btc):
        """N identical strategies must produce exactly the same result."""
        n = 5
        strats = [_SmaStrategy() for _ in range(n)]
        results = PoolBacktestEngine.by_ticks(strats, data=data_btc, workers=1)
        smas = [r["sma"] for r in results]
        assert all(s == pytest.approx(smas[0]) for s in smas)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Warmup
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestWarmup:
    def test_on_data_not_called_before_min_periods(self, data_btc, ticks_btc):
        """
        SMA(5) has min_periods=5.
        on_data must be called N - (5-1) = N - 4 times.
        """
        results = PoolBacktestEngine.by_ticks(
            [_WarmupStrategy()], data=data_btc, workers=1
        )
        first_smas = results[0]["first_smas"]
        n_ticks = len(ticks_btc)
        # Called from tick 5 onwards → n_ticks - 4 times
        assert len(first_smas) == n_ticks - 4

    def test_first_sma_value_is_valid(self, data_btc):
        """The first value received in on_data must be a valid SMA (not nan)."""
        results = PoolBacktestEngine.by_ticks(
            [_WarmupStrategy()], data=data_btc, workers=1
        )
        first_smas = results[0]["first_smas"]
        assert len(first_smas) > 0
        assert not np.isnan(first_smas[0])

    def test_warmup_matches_backtest(self, data_btc, ticks_btc):
        """The number of calls in pool with warmup must match backtest."""

        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
                self._sma = self.add_indicator(self._tp.price_ref, SMA(length=5))
                self.n = 0

            def on_data(self):
                # Only count when indicator has valid value
                if not np.isnan(self._sma[-1]):
                    self.n += 1

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_WarmupStrategy()], data=data_btc, workers=1
        )
        assert len(results[0]["first_smas"]) == ref.n


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Footprint in Pool
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestPoolFootprint:
    def test_fp_n_candles_positive(self, data_btc):
        results = PoolBacktestEngine.by_ticks(
            [_FpStrategy()], data=data_btc, workers=1
        )
        assert results[0]["n_fp_candles"] >= 1

    def test_fp_matches_backtest(self, data_btc, ticks_btc):
        class _Ref(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
                self._fp = self.subscribe_footprint(
                    "BTCUSDT",
                    300_000,
                    window_size=20,
                    tick_size=1.0,
                )
                self.n_fp_candles = 0

            def on_data(self):
                self.n_fp_candles = self._fp.n_candles

        ref = _Ref()
        BacktestEngine.by_ticks(
            ref,
            data=(TickData("BTCUSDT", ticks_btc, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        results = PoolBacktestEngine.by_ticks(
            [_FpStrategy()], data=data_btc, workers=1
        )
        assert results[0]["n_fp_candles"] == ref.n_fp_candles

    def test_fp_multiple_strategies_same_results(self, data_btc):
        strats = [_FpStrategy(), _FpStrategy(), _FpStrategy()]
        results = PoolBacktestEngine.by_ticks(strats, data=data_btc, workers=1)
        n_candles = [r["n_fp_candles"] for r in results]
        assert all(v == n_candles[0] for v in n_candles)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Primary symbol from different proxies
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestPrimarySymbol:
    def test_from_tick_proxy(self, data_btc, ticks_btc):
        results = PoolBacktestEngine.by_ticks(
            [_LastPriceStrategy()], data=data_btc, workers=1
        )
        assert not np.isnan(results[0]["price"])

    def test_from_ohlc_proxy(self, data_btc):
        results = PoolBacktestEngine.by_ticks(
            [_OhlcOnlyStrategy()], data=data_btc, workers=1
        )
        assert not np.isnan(results[0]["close"])

    def test_from_fp_proxy(self, data_btc):
        results = PoolBacktestEngine.by_ticks(
            [_FpStrategy()], data=data_btc, workers=1
        )
        assert results[0]["n_fp_candles"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Multiple indicators and deduplication in worker
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestDeduplicationInWorker:
    def test_sma_shared_between_strategies(self, data_btc):
        """
        Two strategies with SMA(3) on BTCUSDT price:
        the key in ind_tick_blocks is the same → single block.
        Results must be identical.
        """
        s1, s2 = _SmaStrategy(), _SmaStrategy()
        results = PoolBacktestEngine.by_ticks([s1, s2], data=data_btc, workers=1)
        assert results[0]["sma"] == pytest.approx(results[1]["sma"])

    def test_two_different_sma(self, data_btc, ticks_btc):
        r = PoolBacktestEngine.by_ticks(
            [_Sma3Strategy(), _Sma5Strategy()], data=data_btc, workers=1
        )
        assert r[0] != pytest.approx(r[1], rel=1e-6)

        prices = ticks_btc[:, _TICK_COL["price"]]
        sma3 = SMA(3).calculate(prices)
        sma5 = SMA(5).calculate(prices)
        assert r[0] == pytest.approx(sma3[~np.isnan(sma3)][-1], rel=1e-5)
        assert r[1] == pytest.approx(sma5[~np.isnan(sma5)][-1], rel=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Scale — large dataset
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.slow
class TestPoolScale:
    def test_20k_ticks_one_worker(self, data_20k, ticks_20k):
        results = PoolBacktestEngine.by_ticks(
            [_SmaStrategy()], data=data_20k, workers=1
        )
        sma_bulk = SMA(3).calculate(ticks_20k[:, _TICK_COL["price"]])
        expected = sma_bulk[~np.isnan(sma_bulk)][-1]
        assert results[0]["sma"] == pytest.approx(expected, rel=1e-5)

    def test_20k_ticks_multiple_strategies(self, data_20k):
        strats = [_SmaStrategy() for _ in range(6)]
        results = PoolBacktestEngine.by_ticks(strats, data=data_20k, workers=2)
        assert len(results) == 6
        smas = [r["sma"] for r in results]
        assert all(s == pytest.approx(smas[0], rel=1e-5) for s in smas)

    def test_20k_fp_n_candles(self, data_20k):
        results = PoolBacktestEngine.by_ticks(
            [_FpStrategy()], data=data_20k, workers=1
        )
        assert results[0]["n_fp_candles"] > 1


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "--tb=short"])

# endregion

# region test_footprint.py
"""
Tests for the footprint module — Suite v1.0

Covers:
  · _price_level          — rounding to tick_size
  · _compute_scalars      — POC, value area, delta, CVD
  · _dict_to_levels       — dict → sorted array conversion
  · _accumulate_to_dict   — bid/ask classification by flag and by mid
  · FootprintStore bulk   — construction, closed candle, partial candle
  · FpProxy (backtest)    — access [-1]/[-2], n_candles, iteration
  · LiveFpRing            — process_tick, confirm candle, double buffer
  · FpProxy (live)        — access [-1]/[-2] over LiveFpRing
  · Engine integration    — BacktestEngine + subscribe_footprint

Usage:
    pytest test_footprint.py -v
    pytest test_footprint.py -v -m unit
    pytest test_footprint.py -v -m integration
"""

import pytest
import numpy as np

from tradetropy.models.footprint import (
    FootprintConfig,
    FpCandle,
    FootprintStore,
    LiveFpRing,
    FpProxy,
    _price_level,
    _compute_scalars,
    _dict_to_levels,
    _accumulate_to_dict,
    build_footprint_from_ticks,
    _FP_LEVEL_COL,
    _FP_SCALAR_COL,
    N_FP_LEVEL_COLS,
    N_FP_SCALAR_COLS,
)
from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.exceptions import DataError


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _make_config(tick_size=0.5, value_area_pct=0.70, aggressor_col="flags"):
    return FootprintConfig(
        tick_size=tick_size,
        value_area_pct=value_area_pct,
        aggressor_col=aggressor_col,
    )


def _make_tick_matrix(rows):
    """
    rows: list of (ts_ms, price, volume, bid, ask, flag)
    Returns ndarray [N × N_TICK_COLS] with the correct columns.
    """
    out = np.zeros((len(rows), N_TICK_COLS), dtype=np.float64)
    for i, (ts, price, vol, bid, ask, flag) in enumerate(rows):
        out[i, _TICK_COL["ts"]] = ts
        out[i, _TICK_COL["price"]] = price
        out[i, _TICK_COL["volume"]] = vol
        out[i, _TICK_COL["bid"]] = bid
        out[i, _TICK_COL["ask"]] = ask
        out[i, _TICK_COL["flags"]] = flag
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

BASE_TS = 1_700_000_000_000  # ms — 2023-11-14T22:13:20 UTC
MIN_MS = 60_000


@pytest.fixture
def tick_matrix_2velas():
    """
    2 candles of 1 minute (60 ticks per candle, here 3 per candle for simplicity).
    Candle 0 (minute 0): 3 ticks at BASE_TS + 0..2s
    Candle 1 (minute 1): 3 ticks at BASE_TS + 60s + 0..2s
    """
    rows = [
        # ts,                   price,  vol,  bid,    ask,    flag
        (BASE_TS + 0 * 1000, 100.0, 2.0, 99.75, 100.25, 1.0),  # vela0, ask
        (BASE_TS + 1 * 1000, 100.5, 1.0, 100.25, 100.75, 0.0),  # vela0, bid
        (BASE_TS + 2 * 1000, 100.0, 3.0, 99.75, 100.25, 1.0),  # vela0, ask
        (BASE_TS + MIN_MS + 0, 101.0, 1.0, 100.75, 101.25, 1.0),  # vela1, ask
        (BASE_TS + MIN_MS + 1, 100.5, 2.0, 100.25, 100.75, 0.0),  # vela1, bid
        (BASE_TS + MIN_MS + 2, 101.0, 1.0, 100.75, 101.25, 1.0),  # vela1, ask
    ]
    return _make_tick_matrix(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PRIMITIVE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPriceLevel:
    def test_exact_on_multiple(self):
        assert _price_level(100.0, 0.5) == pytest.approx(100.0)

    def test_rounds_up(self):
        assert _price_level(100.3, 0.5) == pytest.approx(100.5)

    def test_rounds_down(self):
        assert _price_level(100.2, 0.5) == pytest.approx(100.0)

    def test_tick_size_1(self):
        assert _price_level(42.7, 1.0) == pytest.approx(43.0)

    def test_small_tick_size(self):
        assert _price_level(42500.3, 0.25) == pytest.approx(42500.25)

    def test_negative_price(self):
        # Uncommon, but the code must be stable
        assert _price_level(-1.3, 0.5) == pytest.approx(-1.5)


@pytest.mark.unit
class TestDictToLevels:
    def test_empty_returns_empty_array(self):
        arr = _dict_to_levels({})
        assert arr.shape == (0, N_FP_LEVEL_COLS)

    def test_one_level(self):
        d = {100.0: [1.0, 2.0, 3]}
        arr = _dict_to_levels(d)
        assert arr.shape == (1, N_FP_LEVEL_COLS)
        assert arr[0, _FP_LEVEL_COL["price"]] == pytest.approx(100.0)
        assert arr[0, _FP_LEVEL_COL["vol_bid"]] == pytest.approx(1.0)
        assert arr[0, _FP_LEVEL_COL["vol_ask"]] == pytest.approx(2.0)
        assert arr[0, _FP_LEVEL_COL["vol_total"]] == pytest.approx(3.0)
        assert arr[0, _FP_LEVEL_COL["delta"]] == pytest.approx(1.0)  # ask-bid
        assert arr[0, _FP_LEVEL_COL["n_trades"]] == pytest.approx(3.0)

    def test_sorted_by_price(self):
        d = {101.0: [1.0, 0.0, 1], 99.5: [0.5, 0.5, 2], 100.0: [0.0, 2.0, 1]}
        arr = _dict_to_levels(d)
        prices = arr[:, _FP_LEVEL_COL["price"]]
        assert list(prices) == pytest.approx([99.5, 100.0, 101.0])

    def test_negative_delta(self):
        d = {100.0: [3.0, 1.0, 4]}  # more bid than ask → negative delta
        arr = _dict_to_levels(d)
        assert arr[0, _FP_LEVEL_COL["delta"]] == pytest.approx(-2.0)


@pytest.mark.unit
class TestAccumulateToDict:
    def test_positive_flag_goes_to_ask(self):
        d = {}
        _accumulate_to_dict(d, 100.0, 1.0, 99.75, 100.25, flag=1.0, config=_make_config())
        assert d[100.0][1] == pytest.approx(1.0)  # vol_ask
        assert d[100.0][0] == pytest.approx(0.0)  # vol_bid

    def test_zero_flag_goes_to_bid(self):
        d = {}
        _accumulate_to_dict(d, 100.0, 1.0, 99.75, 100.25, flag=0.0, config=_make_config())
        assert d[100.0][0] == pytest.approx(1.0)  # vol_bid
        assert d[100.0][1] == pytest.approx(0.0)  # vol_ask

    def test_accumulation_same_level(self):
        d = {}
        _accumulate_to_dict(d, 100.0, 2.0, 99.75, 100.25, flag=1.0, config=_make_config())
        _accumulate_to_dict(d, 100.0, 3.0, 99.75, 100.25, flag=1.0, config=_make_config())
        assert d[100.0][1] == pytest.approx(5.0)
        assert d[100.0][2] == 2  # n_trades

    def test_two_different_levels(self):
        d = {}
        _accumulate_to_dict(d, 100.0, 1.0, 99.75, 100.25, flag=1.0, config=_make_config())
        _accumulate_to_dict(d, 100.5, 2.0, 99.75, 100.25, flag=0.0, config=_make_config())
        assert len(d) == 2

    def test_no_aggressor_uses_mid(self):
        """aggressor_col=None → compares price with mid(bid, ask)."""
        cfg = _make_config(aggressor_col=None)
        d = {}
        # mid = (99.75 + 100.25) / 2 = 100.0 → price >= mid → ask
        _accumulate_to_dict(d, 100.0, 1.0, 99.75, 100.25, flag=0.0, config=cfg)
        assert d[100.0][1] == pytest.approx(1.0)  # ask

        d2 = {}
        # price=99.7 < mid=100.0 → bid
        # round(99.7 / 0.5) * 0.5 = round(199.4) * 0.5 = 199 * 0.5 = 99.5
        _accumulate_to_dict(d2, 99.7, 1.0, 99.75, 100.25, flag=0.0, config=cfg)
        assert d2[99.5][0] == pytest.approx(1.0)  # level rounded to 99.5, bid


@pytest.mark.unit
class TestComputeScalars:
    def _simple_levels(self):
        """3 levels: [99.5, 100.0, 100.5] with vols [1, 5, 2]."""
        arr = np.array(
            [
                [99.5, 1.0, 0.0, 1.0, -1.0, 1],
                [100.0, 0.0, 5.0, 5.0, 5.0, 3],
                [100.5, 2.0, 0.0, 2.0, -2.0, 2],
            ],
            dtype=np.float64,
        )
        return arr

    def test_poc_is_highest_volume_level(self):
        sc = _compute_scalars(self._simple_levels(), 0.70, 0.0)
        assert sc[_FP_SCALAR_COL["poc_price"]] == pytest.approx(100.0)
        assert sc[_FP_SCALAR_COL["poc_vol"]] == pytest.approx(5.0)

    def test_vol_total(self):
        sc = _compute_scalars(self._simple_levels(), 0.70, 0.0)
        assert sc[_FP_SCALAR_COL["vol_total"]] == pytest.approx(8.0)

    def test_delta_total(self):
        sc = _compute_scalars(self._simple_levels(), 0.70, 0.0)
        # deltas: -1 + 5 - 2 = 2
        assert sc[_FP_SCALAR_COL["delta_total"]] == pytest.approx(2.0)

    def test_cvd_acumula(self):
        sc = _compute_scalars(self._simple_levels(), 0.70, 10.0)
        assert sc[_FP_SCALAR_COL["cvd"]] == pytest.approx(12.0)  # 10 + 2

    def test_n_levels(self):
        sc = _compute_scalars(self._simple_levels(), 0.70, 0.0)
        assert sc[_FP_SCALAR_COL["levels"]] == 3

    def test_value_area_cubre_pct(self):
        """VAH and VAL must contain at least 70% of the volume."""
        sc = _compute_scalars(self._simple_levels(), 0.70, 0.0)
        # POC=100.0 (vol=5). 70% of 8 = 5.6. From POC, go up to 100.5 (vol=2)
        # → 5+2=7 >= 5.6. So VAH=100.5, VAL=100.0
        assert sc[_FP_SCALAR_COL["vah"]] == pytest.approx(100.5)
        assert sc[_FP_SCALAR_COL["val"]] == pytest.approx(100.0)

    def test_empty_levels_returns_zeros(self):
        sc = _compute_scalars(np.empty((0, N_FP_LEVEL_COLS)), 0.70, 0.0)
        assert sc[_FP_SCALAR_COL["vol_total"]] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: FOOTPRINT STORE (backtest, bulk construction)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFootprintStoreBulk:
    @pytest.fixture
    def store_2velas(self, tick_matrix_2velas):
        """FootprintStore with 2 candles of 1 minute, 1 closed candle."""
        from tradetropy.data.data import build_candles_from_ticks

        ts = tick_matrix_2velas[:, _TICK_COL["ts"]]
        price = tick_matrix_2velas[:, _TICK_COL["price"]]
        vol = tick_matrix_2velas[:, _TICK_COL["volume"]]
        r = build_candles_from_ticks(ts, price, vol, 60_000)
        store = build_footprint_from_ticks(
            tick_matrix=tick_matrix_2velas,
            tick_to_candle_map=r["tick_to_candle_map"],
            n_closed_candles=int(r["closed_candles"].shape[0]),
            config=_make_config(),
            _TICK_COL_=_TICK_COL,
            interval_ms=60_000,
        )
        return store

    def test_n_closed(self, store_2velas):
        """With 2 tick candles, there is 1 closed."""
        assert store_2velas._n_closed == 1

    def test_fp_idx_consistent(self, store_2velas):
        """fp_idx[0]=0, fp_idx[1]=levels of the first candle."""
        assert store_2velas.fp_idx[0] == 0
        n = store_2velas.fp_idx[1] - store_2velas.fp_idx[0]
        assert n > 0

    def test_closed_candle_has_poc(self, store_2velas):
        candle = store_2velas.closed_candle(0)
        assert isinstance(candle, FpCandle)
        assert candle.poc_price > 0
        assert not candle.is_partial

    def test_closed_candle_total_volume(self, store_2velas, tick_matrix_2velas):
        """Volume of the closed candle = sum of vols from the ticks of that candle."""
        candle = store_2velas.closed_candle(0)
        expected_vol = float(tick_matrix_2velas[:3, _TICK_COL["volume"]].sum())
        assert candle.vol_total == pytest.approx(expected_vol)

    def test_closed_candle_delta(self, store_2velas):
        """
        Candle 0: ask ticks vol=2+3=5, bid tick vol=1 → delta = 5-1=4.
        Price tick 2 (100.5, flag=0) goes to bid.
        """
        candle = store_2velas.closed_candle(0)
        # ask: ticks 0 (vol=2) and 2 (vol=3) → 5. bid: tick 1 (vol=1) → 1.
        assert candle.delta_total == pytest.approx(4.0)

    def test_partial_candle_none_before_processing(self):
        """A freshly built store without calling process_tick → partial candle None."""
        store = FootprintStore(
            fp_data=np.empty((0, N_FP_LEVEL_COLS)),
            fp_idx=np.array([0], dtype=np.int64),
            fp_scalars=np.empty((0, N_FP_SCALAR_COLS)),
            config=_make_config(),
            cvd_total=0.0,
            interval_ms=60_000,
        )
        assert store.partial_candle() is None

    def test_partial_candle_after_processing(self, store_2velas):
        """After process_tick, there is a partial candle."""
        store_2velas.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        candle = store_2velas.partial_candle()
        assert candle is not None
        assert candle.is_partial
        assert candle.vol_total == pytest.approx(1.0)

    def test_process_tick_new_interval_resets(self, store_2velas):
        """Tick in new interval → fresh partial candle."""
        store_2velas.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        store_2velas.process_tick(
            BASE_TS + 2 * MIN_MS, 102.0, 5.0, 101.75, 102.25, 1.0
        )
        candle = store_2velas.partial_candle()
        assert candle.vol_total == pytest.approx(5.0)  # only the last tick

    def test_cvd_accumulates_between_closed_candles(self, store_2velas):
        """CVD of the closed candle is delta_total (previous CVD=0)."""
        candle = store_2velas.closed_candle(0)
        assert candle.cvd == pytest.approx(candle.delta_total)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: FP PROXY (backtest)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFpProxyBacktest:
    @pytest.fixture
    def connected_proxy(self, tick_matrix_2velas):
        """FpProxy connected to a FootprintStore with 2 candles."""
        from tradetropy.data.data import build_candles_from_ticks

        ts = tick_matrix_2velas[:, _TICK_COL["ts"]]
        price = tick_matrix_2velas[:, _TICK_COL["price"]]
        vol = tick_matrix_2velas[:, _TICK_COL["volume"]]
        r = build_candles_from_ticks(ts, price, vol, 60_000)
        store = build_footprint_from_ticks(
            tick_matrix=tick_matrix_2velas,
            tick_to_candle_map=r["tick_to_candle_map"],
            n_closed_candles=int(r["closed_candles"].shape[0]),
            config=_make_config(),
            _TICK_COL_=_TICK_COL,
            interval_ms=60_000,
        )
        proxy = FpProxy("SYM", 60_000, window_size=50, tick_size=0.5, value_area_pct=0.70, aggressor_col="flags")
        proxy._connect(store)
        return proxy

    def test_n_candles_without_partial(self, connected_proxy):
        """Without process_tick, only the closed candle → n_candles=1."""
        assert connected_proxy.n_candles == 1

    def test_n_candles_with_partial(self, connected_proxy):
        connected_proxy.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        assert connected_proxy.n_candles == 2

    def test_negative_index_1_is_partial(self, connected_proxy):
        connected_proxy.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        candle = connected_proxy[-1]
        assert candle is not None
        assert candle.is_partial

    def test_negative_index_2_is_closed(self, connected_proxy):
        connected_proxy.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        candle = connected_proxy[-2]
        assert candle is not None
        assert not candle.is_partial

    def test_out_of_range_returns_none(self, connected_proxy):
        connected_proxy.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        assert connected_proxy[-100] is None

    def test_positive_index_raises_error(self, connected_proxy):
        with pytest.raises(DataError):
            _ = connected_proxy[0]

    def test_iter(self, connected_proxy):
        connected_proxy.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        candles = list(connected_proxy)
        assert len(candles) == 2
        assert all(isinstance(v, FpCandle) for v in candles)

    def test_window_size_limits_closed(self, tick_matrix_2velas):
        """window_size=0 → only partial candle accessible."""
        from tradetropy.data.data import build_candles_from_ticks

        ts = tick_matrix_2velas[:, _TICK_COL["ts"]]
        price = tick_matrix_2velas[:, _TICK_COL["price"]]
        vol = tick_matrix_2velas[:, _TICK_COL["volume"]]
        r = build_candles_from_ticks(ts, price, vol, 60_000)
        store = build_footprint_from_ticks(
            tick_matrix=tick_matrix_2velas,
            tick_to_candle_map=r["tick_to_candle_map"],
            n_closed_candles=int(r["closed_candles"].shape[0]),
            config=_make_config(),
            _TICK_COL_=_TICK_COL,
            interval_ms=60_000,
        )
        proxy = FpProxy("SYM", 60_000, window_size=0, tick_size=0.5, value_area_pct=0.70, aggressor_col="flags")
        proxy._connect(store)
        proxy.process_tick(BASE_TS + MIN_MS, 101.0, 1.0, 100.75, 101.25, 1.0)
        assert proxy[-2] is None  # outside window


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: LIVE FP RING
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveFpRing:
    @pytest.fixture
    def ring(self):
        return LiveFpRing(window_size=10, config=_make_config(), interval_ms=60_000)

    def test_initially_no_candles(self, ring):
        assert ring.n_available_candles == 0
        assert ring.partial_candle() is None

    def test_first_tick_creates_partial_candle(self, ring):
        ring.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        assert ring.n_available_candles == 1
        candle = ring.partial_candle()
        assert candle is not None
        assert candle.is_partial

    def test_tick_same_interval_updates_partial(self, ring):
        ring.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        ring.process_tick(BASE_TS + 1000, 100.5, 2.0, 99.75, 100.25, 0.0)
        candle = ring.partial_candle()
        assert candle.vol_total == pytest.approx(3.0)

    def test_new_interval_closes_previous_candle(self, ring):
        ring.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        ring.process_tick(BASE_TS + MIN_MS, 101.0, 2.0, 100.75, 101.25, 1.0)
        # 1 closed + 1 partial
        assert ring._n_closed == 1
        assert ring.n_available_candles == 2

    def test_closed_candle_access(self, ring):
        ring.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        ring.process_tick(BASE_TS + MIN_MS, 101.0, 2.0, 100.75, 101.25, 1.0)
        candle = ring.closed_candle(0)  # last closed
        assert candle is not None
        assert not candle.is_partial
        assert candle.vol_total == pytest.approx(1.0)

    def test_double_buffer_survives_wrap(self):
        """Ring of size 3: make sure wrap does not corrupt access."""
        ring = LiveFpRing(window_size=3, config=_make_config(), interval_ms=60_000)
        for i in range(5):
            ring.process_tick(BASE_TS + i * MIN_MS, 100.0 + i, 1.0, 99.75, 100.25, 1.0)
        # Should have closed candles (4) but window=3 → 3 available
        assert ring.n_available_candles == 4  # 3 closed in window + 1 partial

    def test_cvd_accumulates_between_candles(self, ring):
        """CVD of candle N = delta_candle_N + CVD of candle N-1."""
        ring.process_tick(BASE_TS, 100.0, 5.0, 99.75, 100.25, 1.0)  # ask
        ring.process_tick(BASE_TS + MIN_MS, 100.0, 2.0, 99.75, 100.25, 0.0)  # bid
        ring.process_tick(BASE_TS + 2 * MIN_MS, 100.0, 1.0, 99.75, 100.25, 1.0)
        v0 = ring.closed_candle(1)  # first candle (penultimate closed)
        v1 = ring.closed_candle(0)  # second candle (last closed)
        assert v1.cvd == pytest.approx(v0.cvd + v1.delta_total)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: FP PROXY (live)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFpProxyLive:
    @pytest.fixture
    def proxy_live(self):
        ring = LiveFpRing(window_size=10, config=_make_config(), interval_ms=60_000)
        proxy = FpProxy("SYM", 60_000, window_size=10, tick_size=0.5, value_area_pct=0.70, aggressor_col="flags")
        proxy._connect_live(ring)
        return proxy

    def test_partial_access_live(self, proxy_live):
        proxy_live.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        candle = proxy_live[-1]
        assert candle is not None and candle.is_partial

    def test_closed_access_live(self, proxy_live):
        proxy_live.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        proxy_live.process_tick(BASE_TS + MIN_MS, 101.0, 2.0, 100.75, 101.25, 1.0)
        closed = proxy_live[-2]
        assert closed is not None and not closed.is_partial

    def test_iter_live(self, proxy_live):
        proxy_live.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        proxy_live.process_tick(BASE_TS + MIN_MS, 101.0, 2.0, 100.75, 101.25, 1.0)
        candles = list(proxy_live)
        assert len(candles) == 2

    def test_len_live(self, proxy_live):
        proxy_live.process_tick(BASE_TS, 100.0, 1.0, 99.75, 100.25, 1.0)
        proxy_live.process_tick(BASE_TS + MIN_MS, 101.0, 2.0, 100.75, 101.25, 1.0)
        assert len(proxy_live) == 2


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: INTEGRATION WITH TICK BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestFootprintIntegrationEngine:
    @pytest.fixture
    def ticks_3velas(self):
        """3 candles of 1 minute, 3 ticks each."""
        rows = []
        for candle_idx in range(3):
            ts_base = BASE_TS + candle_idx * MIN_MS
            rows.extend(
                [
                    (ts_base + 0, 100.0 + candle_idx, 1.0, 99.75 + candle_idx, 100.25 + candle_idx, 1.0),
                    (ts_base + 1, 100.5 + candle_idx, 2.0, 99.75 + candle_idx, 100.25 + candle_idx, 0.0),
                    (ts_base + 2, 100.0 + candle_idx, 1.0, 99.75 + candle_idx, 100.25 + candle_idx, 1.0),
                ]
            )
        return _make_tick_matrix(rows)

    def test_on_data_receives_partial_candle(self, ticks_3velas):
        """subscribe_footprint delivers fp[-1] non-None in on_data."""
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        capturas = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("SYM", window_size=20)
                self._fp = self.subscribe_footprint(
                    "SYM",
                    timeframe=60_000,
                    window_size=50,
                    tick_size=0.5,
                    value_area_pct=0.70,
                    aggressor_col="flags",
                )

            def on_data(self):
                v = self._fp[-1]
                capturas.append(v)

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks_3velas, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        assert len(capturas) == 9  # one on_data per tick
        assert all(v is not None for v in capturas)

    def test_closed_candles_correct(self, ticks_3velas):
        """After 2 complete candles, fp[-2] is the first closed candle."""
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        state = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("SYM", window_size=20)
                self._fp = self.subscribe_footprint(
                    "SYM",
                    timeframe=60_000,
                    window_size=50,
                    tick_size=0.5,
                    value_area_pct=0.70,
                    aggressor_col="flags",
                )

            def on_data(self):
                if self._tp.n_ticks == 9:  # last tick
                    state["n_candles"] = self._fp.n_candles
                    state["partial"] = self._fp[-1]
                    state["cerrada_1"] = self._fp[-2]
                    state["cerrada_0"] = self._fp[-3]

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks_3velas, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        assert state["n_candles"] == 3
        assert state["partial"].is_partial
        assert not state["cerrada_1"].is_partial
        assert not state["cerrada_0"].is_partial

    def test_without_subscribe_ohlc_works(self, ticks_3velas):
        """subscribe_footprint without prior subscribe_ohlc must not fail."""
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ok = []

        class S(Strategy):
            def init(self):
                # NO subscribe_ohlc — engine builds the mapping internally
                self._fp = self.subscribe_footprint(
                    "SYM", 60_000, window_size=50, tick_size=0.5, value_area_pct=0.70, aggressor_col="flags"
                )

            def on_data(self):
                ok.append(True)

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks_3velas, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        assert len(ok) == 9

    def test_total_volume_matches_ticks(self, ticks_3velas):
        """Total vol of closed candle 0 = sum of vols from the first 3 ticks."""
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        state = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("SYM", window_size=20)
                self._fp = self.subscribe_footprint(
                    "SYM", 60_000, window_size=50, tick_size=0.5, value_area_pct=0.70, aggressor_col="flags"
                )

            def on_data(self):
                if self._tp.n_ticks == 9:
                    state["v0"] = self._fp[-3]  # first closed candle

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks_3velas, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        expected_vol = float(ticks_3velas[:3, _TICK_COL["volume"]].sum())
        assert state["v0"].vol_total == pytest.approx(expected_vol)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestEdgeCases:
    def test_single_tick_single_level(self):
        """One tick → one candle with one level."""
        rows = [(BASE_TS, 100.0, 5.0, 99.75, 100.25, 1.0)]
        matrix = _make_tick_matrix(rows)
        from tradetropy.data.data import build_candles_from_ticks

        r = build_candles_from_ticks(
            matrix[:, _TICK_COL["ts"]],
            matrix[:, _TICK_COL["price"]],
            matrix[:, _TICK_COL["volume"]],
            1.0,
        )
        # Only 1 tick → 0 closed candles
        store = build_footprint_from_ticks(
            tick_matrix=matrix,
            tick_to_candle_map=r["tick_to_candle_map"],
            n_closed_candles=int(r["closed_candles"].shape[0]),
            config=_make_config(),
            _TICK_COL_=_TICK_COL,
            interval_ms=60_000,
        )
        assert store._n_closed == 0

    def test_multiple_levels_same_price(self):
        """Multiple ticks at the same price consolidate into 1 level."""
        d = {}
        for _ in range(5):
            _accumulate_to_dict(
                d, 100.0, 1.0, 99.75, 100.25, flag=1.0, config=_make_config()
            )
        arr = _dict_to_levels(d)
        assert len(arr) == 1
        assert arr[0, _FP_LEVEL_COL["vol_ask"]] == pytest.approx(5.0)

    def test_value_area_100pct_covers_all(self):
        """With value_area_pct=1.0, VAH and VAL cover all levels."""
        levels = np.array(
            [
                [99.5, 2.0, 0.0, 2.0, -2.0, 1],
                [100.0, 0.0, 3.0, 3.0, 3.0, 2],
                [100.5, 1.0, 0.0, 1.0, -1.0, 1],
            ],
            dtype=np.float64,
        )
        sc = _compute_scalars(levels, 1.0, 0.0)
        assert sc[_FP_SCALAR_COL["vah"]] == pytest.approx(100.5)
        assert sc[_FP_SCALAR_COL["val"]] == pytest.approx(99.5)

    def test_fp_candle_repr(self):
        """FpCandle.__repr__ does not raise an exception."""
        levels = np.array([[100.0, 1.0, 2.0, 3.0, 1.0, 5]], dtype=np.float64)
        sc = _compute_scalars(levels, 0.70, 0.0)
        v = FpCandle(levels, sc, is_partial=True)
        s = repr(v)
        assert "FpCandle" in s
        assert "partial" in s


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "--tb=short"])

# endregion

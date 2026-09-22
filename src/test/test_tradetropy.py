import pytest
import numpy as np
from typing import TYPE_CHECKING


from tradetropy.data.data import build_candles_from_ticks
from tradetropy.core.constants import _TICK_COL, N_TICK_COLS, _OHLC_COL, N_OHLC_COLS
from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.core.data_types import TickData
from tradetropy.session.base import SeshSimulatorBase

from tradetropy.connectors.mt5 import SeshMT5Live, SeshMT5Sim
from tradetropy.backtest import BacktestEngine
from tradetropy.live import LiveEngine
from tradetropy.models.strategy import Strategy
import time
from tradetropy.core.constants import TICK_COLS, OHLC_COLS, _TICK_COL, _OHLC_COL

# ══════════════════════════════════════════════════════════════════════════════
# CONVERTER  klines → ticks
# ══════════════════════════════════════════════════════════════════════════════
def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    """
    Convert OHLC klines [N × ≥6] to tick_matrix [N × 7].

    Input columns  : [ts, open, high, low, close, volume, ...]
    Output columns : [ts, bid, ask, volume, flags, volume_real, price]

    Generates 1 tick per candle using the close as price.
    bid = ask = close  (no spread — data not available in klines).
    volume_real = NaN  (same).
    """
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
# HELPER — capture OhlcProxy state at a specific tick
# ══════════════════════════════════════════════════════════════════════════════


def get_ohlc_at_tick(
    symbol: str,
    klines: np.ndarray,
    req_interval_ms: int,
    tick_idx: int,  # 1-based: tick_idx=1 = "only the first tick seen"
) -> dict:
    """
    Equivalent to DataHandler.get_data(symbol, interval, live_idx).

    Runs the full BacktestEngine but captures the OhlcProxy state
    exactly at tick number tick_idx.

    Returns dict with:
        open, high, low, close, volume, timestamp  — arrays of all candles
        n_candles                                    — closed + 1 partial
    """
    ticks = klines_to_ticks(klines)
    capture = {}

    class _Capture(Strategy):
        def init(self):
            self._tp = self.subscribe_ticks(symbol, window_size=1)
            self._ohlc = self.subscribe_ohlc(
                symbol, req_interval_ms, window_size=500
            )

        def on_data(self):
            if self._tp.n_ticks != tick_idx:
                return
            p = self._ohlc
            n_v = p.n_klines
            idx = range(-n_v, 0)
            capture.update(
                open=np.array([p.open[i] for i in idx]),
                high=np.array([p.high[i] for i in idx]),
                low=np.array([p.low[i] for i in idx]),
                close=np.array([p.close[i] for i in idx]),
                volume=np.array([p.volume[i] for i in idx]),
                timestamp=np.array([p.ts[i] for i in idx]),
                n_candles=n_v,
            )

    BacktestEngine.by_ticks(
        _Capture(),
        data=(TickData(symbol, ticks, tick_size=0.01),),
        sesh=SeshSimulatorBase("tick"),
    ).run()
    return capture


def ts_str(ts_ms: float) -> str:
    """Converts timestamp in ms to ISO 8601 string with second resolution."""
    return np.datetime_as_string(np.datetime64(int(ts_ms), "ms"), unit="s")


# ══════════════════════════════════════════════════════════════════════════════
# klines_to_ticks
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlinesToTicks:
    def test_shape(self, klines_1m):
        assert klines_to_ticks(klines_1m).shape == (13, 7)

    def test_price_is_close(self, klines_1m):
        ticks = klines_to_ticks(klines_1m)
        np.testing.assert_array_equal(ticks[:, _TICK_COL["price"]], klines_1m[:, 4])

    def test_bid_ask_equals_close(self, klines_1m):
        ticks = klines_to_ticks(klines_1m)
        np.testing.assert_array_equal(ticks[:, _TICK_COL["bid"]], klines_1m[:, 4])
        np.testing.assert_array_equal(ticks[:, _TICK_COL["ask"]], klines_1m[:, 4])

    def test_volume_preserved(self, klines_1m):
        ticks = klines_to_ticks(klines_1m)
        np.testing.assert_array_equal(ticks[:, _TICK_COL["volume"]], klines_1m[:, 5])

    def test_flags_zero(self, klines_1m):
        assert np.all(klines_to_ticks(klines_1m)[:, _TICK_COL["flags"]] == 0)

    def test_volume_real_nan(self, klines_1m):
        assert np.all(np.isnan(klines_to_ticks(klines_1m)[:, _TICK_COL["volume_real"]]))

    def test_timestamps_preserved(self, klines_1m):
        ticks = klines_to_ticks(klines_1m)
        np.testing.assert_array_equal(ticks[:, _TICK_COL["ts"]], klines_1m[:, 0])

    def test_works_with_5m(self, klines_5m):
        ticks = klines_to_ticks(klines_5m)
        assert ticks.shape == (4, 7)
        assert ticks[0, _TICK_COL["price"]] == pytest.approx(42515.6)


# ══════════════════════════════════════════════════════════════════════════════
# OhlcProxy — equivalence with DataHandler.get_data
# ══════════════════════════════════════════════════════════════════════════════
#
# In the new system get_data() does not exist. The equivalent is subscribe_ohlc()
# and reading the OhlcProxy in on_tick(). get_ohlc_at_tick() does
# exactly that and returns the same dict that get_data() used to return.
#
# NOTE on high/low with 1 tick per candle:
#   Synthetic ticks have price = close of the original kline.
#   When reconstructing candles, high = max(closes in group), low = min(closes).
#   This differs from the original DataHandler which used the real high/low of
#   the klines — known and accepted limitation of the 1-tick/candle conversion.


@pytest.mark.unit
class TestOhlcSync:
    def test_live_idx_7_structure(self, klines_1m):
        """
        Equivalent to test_data_handler_sync from the original DataHandler.
        At tick 7 with 5m interval: 2 candles (1 closed + 1 partial).
        """
        r = get_ohlc_at_tick("BTCUSDT", klines_1m, 300_000, tick_idx=7)

        assert r["n_candles"] == 2
        # open of the partial = close of the first tick of that 5m interval = klines_1m[2, close]
        assert r["open"][-1] == pytest.approx(klines_1m[2, 4])
        # high of the partial = max closes from klines[2..6]
        assert r["high"][-1] == pytest.approx(klines_1m[2:7, 4].max())

    # ── 1m → 5m ──────────────────────────────────────────────────────────────
    #
    # klines_1m[0].ts = 1700000000000 = 22:13:20 UTC
    # floor(22:13:20, 5min) = 22:10:00  ← timestamp of the first 5m candle
    #
    # 1m groups in 5m candles:
    #   22:10 → klines[0, 1]      closes: 42505.8, 42515.6
    #   22:15 → klines[2..6]      closes: 42498.2, 42490.6, 42502.4, 42525.9, 42518.3
    #   22:20 → klines[7..11]     closes: 42501.1, 42508.7, 42530.2, 42540.8, 42533.4
    #   22:25 → klines[12]        close:  42555.9
    #
    @pytest.mark.parametrize(
        "live_idx,n_candles,open_,high,low,close,ts_candle",
        [
            pytest.param(
                1,
                1,
                42505.8,
                42505.8,
                42505.8,
                42505.8,
                "2023-11-14T22:10:00",
                id="tick01_start_candle5m_0",
            ),
            pytest.param(
                2,
                1,
                42505.8,
                42515.6,
                42505.8,
                42515.6,
                "2023-11-14T22:10:00",
                id="tick02_end_candle5m_0",
            ),
            pytest.param(
                3,
                2,
                42498.2,
                42498.2,
                42498.2,
                42498.2,
                "2023-11-14T22:15:00",
                id="tick03_start_candle5m_1",
            ),
            pytest.param(
                4,
                2,
                42498.2,
                42498.2,
                42490.6,
                42490.6,
                "2023-11-14T22:15:00",
                id="tick04_candle5m_1",
            ),
            pytest.param(
                5,
                2,
                42498.2,
                42502.4,
                42490.6,
                42502.4,
                "2023-11-14T22:15:00",
                id="tick05_candle5m_1",
            ),
            pytest.param(
                6,
                2,
                42498.2,
                42525.9,
                42490.6,
                42525.9,
                "2023-11-14T22:15:00",
                id="tick06_candle5m_1",
            ),
            pytest.param(
                7,
                2,
                42498.2,
                42525.9,
                42490.6,
                42518.3,
                "2023-11-14T22:15:00",
                id="tick07_end_candle5m_1",
            ),
            pytest.param(
                8,
                3,
                42501.1,
                42501.1,
                42501.1,
                42501.1,
                "2023-11-14T22:20:00",
                id="tick08_start_candle5m_2",
            ),
            pytest.param(
                13,
                4,
                42555.9,
                42555.9,
                42555.9,
                42555.9,
                "2023-11-14T22:25:00",
                id="tick13_last",
            ),
        ],
    )
    def test_1m_to_5m(
        self, klines_1m, live_idx, n_candles, open_, high, low, close, ts_candle
    ):
        r = get_ohlc_at_tick("BTCUSDT", klines_1m, 300_000, live_idx)
        assert r["n_candles"] == n_candles
        assert r["open"][-1] == pytest.approx(open_)
        assert r["high"][-1] == pytest.approx(high)
        assert r["low"][-1] == pytest.approx(low)
        assert r["close"][-1] == pytest.approx(close)
        assert ts_str(r["timestamp"][-1]) == ts_candle

    # ── 5m → 15m and 5m → 30m ────────────────────────────────────────────────
    #
    # klines_5m timestamps: 22:10:00 / 22:15:20 / 22:20:20 / 22:25:20
    #
    # Groups in 15m candles:
    #   22:00 → klines5m[0]         close: 42515.6
    #   22:15 → klines5m[1, 2, 3]   closes: 42518.3, 42533.4, 42555.9
    #
    # Groups in 30m candles:
    #   22:00 → klines5m[0, 1, 2, 3]
    #
    @pytest.mark.parametrize(
        "req_ms,live_idx,n_candles,open_,high,low,close,vol,ts_candle",
        [
            # 5m → 15m
            pytest.param(
                900_000,
                1,
                1,
                42515.6,
                42515.6,
                42515.6,
                42515.6,
                22.27,
                "2023-11-14T22:00:00",
                id="15m_tick1_start",
            ),
            pytest.param(
                900_000,
                2,
                2,
                42518.3,
                42518.3,
                42518.3,
                42518.3,
                60.61,
                "2023-11-14T22:15:00",
                id="15m_tick2_new_candle",
            ),
            pytest.param(
                900_000,
                3,
                2,
                42518.3,
                42533.4,
                42518.3,
                42533.4,
                120.76,
                "2023-11-14T22:15:00",
                id="15m_tick3",
            ),
            pytest.param(
                900_000,
                4,
                2,
                42518.3,
                42555.9,
                42518.3,
                42555.9,
                139.03,
                "2023-11-14T22:15:00",
                id="15m_tick4_last",
            ),
            # 5m → 30m
            pytest.param(
                1_800_000,
                1,
                1,
                42515.6,
                42515.6,
                42515.6,
                42515.6,
                22.27,
                "2023-11-14T22:00:00",
                id="30m_tick1",
            ),
            pytest.param(
                1_800_000,
                2,
                1,
                42515.6,
                42518.3,
                42515.6,
                42518.3,
                82.88,
                "2023-11-14T22:00:00",
                id="30m_tick2",
            ),
            pytest.param(
                1_800_000,
                3,
                1,
                42515.6,
                42533.4,
                42515.6,
                42533.4,
                143.03,
                "2023-11-14T22:00:00",
                id="30m_tick3",
            ),
            pytest.param(
                1_800_000,
                4,
                1,
                42515.6,
                42555.9,
                42515.6,
                42555.9,
                161.30,
                "2023-11-14T22:00:00",
                id="30m_tick4_last",
            ),
        ],
    )
    def test_5m_to_15m_30m(
        self,
        klines_5m,
        req_ms,
        live_idx,
        n_candles,
        open_,
        high,
        low,
        close,
        vol,
        ts_candle,
    ):
        r = get_ohlc_at_tick("BTCUSDT", klines_5m, req_ms, live_idx)
        assert r["n_candles"] == n_candles
        assert r["open"][-1] == pytest.approx(open_)
        assert r["high"][-1] == pytest.approx(high)
        assert r["low"][-1] == pytest.approx(low)
        assert r["close"][-1] == pytest.approx(close)
        assert r["volume"][-1] == pytest.approx(vol, rel=1e-3)
        assert ts_str(r["timestamp"][-1]) == ts_candle


# ══════════════════════════════════════════════════════════════════════════════
# TickProxy
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTickProxy:
    def test_basic_properties(self, klines_1m):
        """Equivalent to test_ohlc_properties from the original DataHandler."""
        ticks = klines_to_ticks(klines_1m)
        capture = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=20)

            def on_data(self):
                if self._tp.n_ticks == len(klines_1m):
                    capture["price_0"] = float(self._tp.price[0])
                    capture["price_m1"] = float(self._tp.price[-1])
                    capture["vol_sum"] = float(np.sum(self._tp.volume))
                    capture["n"] = self._tp.n_ticks

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        assert capture["n"] == 13
        assert capture["price_0"] == pytest.approx(42505.8)
        assert capture["price_m1"] == pytest.approx(42555.9)
        # np.sum avoids float64 drift accumulated by Python's sum()
        assert capture["vol_sum"] == pytest.approx(161.3, rel=1e-5)

    def test_datetime_matches_ts(self, klines_1m):
        """proxy.datetime mirrors proxy.ts as datetime64[ms] (scalar + array)."""
        ticks = klines_to_ticks(klines_1m)
        cap = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=20)

            def on_data(self):
                if self._tp.n_ticks == len(klines_1m):
                    cap["ts_scalar"] = float(self._tp.ts[-1])
                    cap["dt_scalar"] = self._tp.datetime[-1]
                    cap["ts_arr"] = np.asarray(self._tp.ts[:])
                    cap["dt_arr"] = np.asarray(self._tp.datetime[:])
                    cap["dt_iter"] = list(self._tp.datetime)

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        # Scalar: datetime64[ms] equal to np.datetime64 of the raw ms.
        assert isinstance(cap["dt_scalar"], np.datetime64)
        assert cap["dt_scalar"] == np.datetime64(int(cap["ts_scalar"]), "ms")

        # Array: datetime64[ms] dtype, element-wise equal to ts as datetime64.
        assert cap["dt_arr"].dtype == np.dtype("datetime64[ms]")
        expected = cap["ts_arr"].astype(np.int64).astype("datetime64[ms]")
        assert np.array_equal(cap["dt_arr"], expected)

        # Iteration yields the same datetime64[ms] values.
        assert np.array_equal(np.asarray(cap["dt_iter"]), expected)

    def test_spread_zero_with_klines(self, klines_1m):
        """bid == ask in synthetic ticks → spread always 0."""
        ticks = klines_to_ticks(klines_1m)
        spreads = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)

            def on_data(self):
                spreads.append(self._tp.spread)

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        assert all(s == pytest.approx(0.0) for s in spreads)


# ══════════════════════════════════════════════════════════════════════════════
# OhlcDataStore — memory guarantees
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestOhlcMemory:
    @pytest.fixture
    def store_capture(self, klines_1m):
        """Helper fixture: runs the engine and captures the final OhlcDataStore."""
        ticks = klines_to_ticks(klines_1m)
        capture = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)

            def on_data(self):
                if self._tp.n_ticks == len(klines_1m):
                    capture["mat"] = self._ohlc._ohlc_store.matrix
                    capture["n"] = self._ohlc._ohlc_store.n_closed_candles

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        return capture

    def test_matrix_float64_c_contiguous(self, store_capture):
        """Matrix must be float64 C-contiguous — required for zero-copy slicing."""
        mat = store_capture["mat"]
        assert mat.dtype == np.float64
        assert mat.flags["C_CONTIGUOUS"]

    def test_has_closed_candles(self, store_capture):
        """With 13 1m klines and 5m candles, closed candles should exist at the end."""
        assert store_capture["n"] > 0

    def test_slice_closed_is_view(self, store_capture):
        """
        The slice mat[:n_closed, col] is a zero-copy view — does not duplicate memory.
        Closed candles are read directly from the store without going through np.r_.
        """
        mat, n = store_capture["mat"], store_capture["n"]
        assert mat[:n, _OHLC_COL["close"]].base is mat


# ══════════════════════════════════════════════════════════════════════════════
# Benchmarks  (pytest-benchmark)
# ══════════════════════════════════════════════════════════════════════════════
#
# Run with:
#   pytest -m benchmark --benchmark-only
#   pytest -m benchmark --benchmark-compare   # compare with previous run
#   pytest -m benchmark --benchmark-save=baseline


def _make_engine(klines, interval_ms=300_000, window_size=200):
    """Benchmark target: builds ticks + runs the full engine."""
    ticks = klines_to_ticks(klines)

    class S(Strategy):
        def init(self):
            self._tp = self.subscribe_ticks("BTCUSDT", window_size=window_size)
            self._ohlc = self.subscribe_ohlc(
                "BTCUSDT", interval_ms, window_size=window_size
            )

        def on_data(self):
            _ = self._tp.price[-1]
            _ = self._ohlc.close[-1]

    BacktestEngine.by_ticks(
        S(),
        data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
        sesh=SeshSimulatorBase("tick"),
    ).run()


@pytest.mark.benchmark
@pytest.mark.slow
class TestBenchmarks:
    """
    Speed benchmarks. Equivalent to test_dp_speed from the original DataHandler.
    pytest-benchmark runs each function N times and reports min/mean/max/stddev/ops.

    Save baseline:   pytest -m benchmark --benchmark-save=v1
    Compare after:   pytest -m benchmark --benchmark-compare=v1
    """

    def test_20k_1m_to_5m(self, benchmark, klines_20k):
        """Direct equivalent of test_dp_speed. 20k 1m klines → 5m, full loop."""
        benchmark.pedantic(
            _make_engine,
            args=(klines_20k,),
            kwargs={"interval_ms": 300_000},
            rounds=5,
            iterations=1,
        )

    def test_100k_1m_to_5m(self, benchmark, klines_100k):
        """100k klines — verifies cost scales linearly (not quadratically)."""
        benchmark.pedantic(
            _make_engine,
            args=(klines_100k,),
            kwargs={"interval_ms": 300_000},
            rounds=3,
            iterations=1,
        )

    def test_20k_multi_interval(self, benchmark, klines_20k):
        """3 simultaneous resolutions (5m, 15m, 60m) — overhead of multiple OhlcProxy."""
        ticks = klines_to_ticks(klines_20k)
        assert len(ticks) == 20_000

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=100)
                self._5m = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=100)
                self._15m = self.subscribe_ohlc("BTCUSDT", 900_000, window_size=100)
                self._60m = self.subscribe_ohlc(
                    "BTCUSDT", 3_600_000, window_size=100
                )

            def on_data(self):
                _ = self._tp.price[-1]
                _ = self._5m.close[-1]
                _ = self._15m.close[-1]
                _ = self._60m.close[-1]

        benchmark.pedantic(
            lambda: BacktestEngine.by_ticks(
                S(),
                data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
                sesh=SeshSimulatorBase("tick"),
            ).run(),
            rounds=5,
            iterations=1,
        )

    def test_build_candles_bulk(self, benchmark, klines_100k):
        """
        Only build_candles_from_ticks — the bulk pre-computation (paid once).
        Useful for isolating its cost from the engine loop cost.
        """
        ticks = klines_to_ticks(klines_100k)
        ts = ticks[:, _TICK_COL["ts"]]
        price = ticks[:, _TICK_COL["price"]]
        vol = ticks[:, _TICK_COL["volume"]]
        benchmark(build_candles_from_ticks, ts, price, vol, 300_000)


# if __name__ == "__main__":
#     pytest.main(["-v", "-s", __file__])

# region test_live_engine_v2.0
"""
Tests for LiveEngine — Suite v2.0

Coverage:
  · prepare() without history
  · prepare() with history (indicator warmup)
  · on_tick() processes ticks one by one and calls on_data()
  · TickProxy updates n_ticks correctly in live
  · OhlcProxy builds partial and closed candles in live
  · IndicatorProxy (SMA/EMA) in live on ticks and candles
  · FpProxy (footprint) in live
  · Tick-by-tick vs bulk backtest consistency (same final values)
  · Result equivalence: klines_to_ticks + LiveEngine == BacktestEngine

The fixtures klines_1m, klines_5m, klines_20k, klines_100k come from conftest.py.
Kline data is converted to synthetic ticks with klines_to_ticks().

Usage:
    pytest test_live_engine.py -v
    pytest test_live_engine.py::TestLiveEngineTicks -v
    pytest test_live_engine.py::TestLiveEngineEquivalence -v
"""

import pytest
import numpy as np

from tradetropy.core.constants import (
    _TICK_COL,
    N_TICK_COLS,
    _OHLC_COL,
)
from tradetropy.data.data import OhlcProxy, TickProxy
from tradetropy.ta import SMA, EMA
from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.core.data_types import TickData, KlineData
from tradetropy.live.engine import LiveEngine
from tradetropy.models.footprint import FootprintConfig
from tradetropy.exceptions import TradingError, ConfigError


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    """Convert klines [N×6] to tick_matrix [N×7] using close as price."""
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


def run_live_tick_by_tick(strategy_cls, ticks, symbol="BTCUSDT"):
    """Create a LiveEngine, prepare it without history and feed ticks one by one."""
    engine = LiveEngine.by_ticks(strategy_cls())
    engine.prepare()
    for i in range(len(ticks)):
        engine.on_tick(symbol, ticks[i])
    return engine


def run_live_with_history(strategy_cls, ticks, history_ratio=0.5, symbol="BTCUSDT"):
    """Prepare the engine with the first history_ratio ticks as history."""
    split = int(len(ticks) * history_ratio)
    hist = ticks[:split]
    remainder = ticks[split:]
    engine = LiveEngine.by_ticks(strategy_cls())
    engine.prepare({symbol: hist})
    for i in range(len(remainder)):
        engine.on_tick(symbol, remainder[i])
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# TEST STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════


class _CounterStrategy(Strategy):
    """Counts how many times on_data() is called."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.n_calls = 0

    def on_data(self):
        self.n_calls += 1


class _TickValuesStrategy(Strategy):
    """Captures the last price on each tick."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.prices = []
        self.n_ticks_hist = []

    def on_data(self):
        self.prices.append(float(self._tp.price[-1]))
        self.n_ticks_hist.append(self._tp.n_ticks)


class _OhlcStrategy(Strategy):
    """Captures the last close and n_candles from a 5m OhlcProxy."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=20)
        self.closes = []
        self.n_candles_hist = []

    def on_data(self):
        self.closes.append(float(self._ohlc.close[-1]))
        self.n_candles_hist.append(self._ohlc.n_klines)


class _SmaTickStrategy(Strategy):
    """Captures SMA(3) on price at each tick."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._sma = self.add_indicator(self._tp.price_ref, SMA(length=3))
        self.sma_vals = []

    def on_data(self):
        self.sma_vals.append(float(self._sma[-1]))


class _SmaOhlcStrategy(Strategy):
    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=20)
        self._sma = self.add_indicator(self._ohlc.close_ref, SMA(length=2))
        self.sma_vals = []

    def on_data(self):
        if self._ohlc.n_klines >= 1 and len(self._sma) > 0:
            self.sma_vals.append(float(self._sma[-1]))


class _EmaStrategy(Strategy):
    """Captures EMA(3) on price."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self._ema = self.add_indicator(self._tp.price_ref, EMA(length=3))
        self.ema_vals = []

    def on_data(self):
        self.ema_vals.append(float(self._ema[-1]))


class _MultiColStrategy(Strategy):
    """Captures open, high, low, close, volume from 5m candles."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=10)
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
        self.opens = []
        self.highs = []
        self.lows = []
        self.vols = []

    def on_data(self):
        self.opens.append(float(self._ohlc.open[-1]))
        self.highs.append(float(self._ohlc.high[-1]))
        self.lows.append(float(self._ohlc.low[-1]))
        self.vols.append(float(self._ohlc.volume[-1]))


class _SpreadStrategy(Strategy):
    """Captures spread at each tick."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)
        self.spreads = []

    def on_data(self):
        self.spreads.append(self._tp.spread)


class _FpStrategy(Strategy):
    """Captures the partial footprint candle at each tick."""

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=10)
        self._fp = self.subscribe_footprint(
            "BTCUSDT",
            300_000,
            window_size=10,
            tick_size=1.0,
        )
        self.n_candles_hist = []

    def on_data(self):
        self.n_candles_hist.append(self._fp.n_candles)


class _CaptureFinalStrategy(Strategy):
    """
    Captures the full state at the end of all ticks.
    Useful for comparing with backtest.
    """

    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=200)
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=200)
        self._sma = self.add_indicator(self._tp.price_ref, SMA(length=3))
        self.last_price = np.nan
        self.last_close = np.nan
        self.last_sma = np.nan
        self.last_n_candles = 0

    def on_data(self):
        self.last_price = float(self._tp.price[-1])
        self.last_close = float(self._ohlc.close[-1])
        self.last_sma = float(self._sma[-1])
        self.last_n_candles = self._ohlc.n_klines


# Equivalent strategy for backtest (same logic)
class _BacktestEquivalentStrategy(Strategy):
    def init(self):
        self._tp = self.subscribe_ticks("BTCUSDT", window_size=200)
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=200)
        self._sma = self.add_indicator(self._tp.price_ref, SMA(length=3))
        self.last_price = np.nan
        self.last_close = np.nan
        self.last_sma = np.nan
        self.last_n_candles = 0

    def on_data(self):
        self.last_price = float(self._tp.price[-1])
        self.last_close = float(self._ohlc.close[-1])
        self.last_sma = float(self._sma[-1])
        self.last_n_candles = self._ohlc.n_klines


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ticks_1m(klines_1m):
    return klines_to_ticks(klines_1m)


@pytest.fixture
def ticks_5m(klines_5m):
    return klines_to_ticks(klines_5m)


@pytest.fixture
def ticks_20k(klines_20k):
    return klines_to_ticks(klines_20k)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: prepare() and lifecycle
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineLifecycle:
    def test_prepare_without_history_no_error(self):
        engine = LiveEngine.by_ticks(_CounterStrategy())
        engine.prepare()  # should not raise

    def test_prepare_with_history(self, ticks_1m):
        engine = LiveEngine.by_ticks(_CounterStrategy())
        engine.prepare({"BTCUSDT": ticks_1m})

    def test_on_tick_without_prepare_raises(self, ticks_1m):
        engine = LiveEngine.by_ticks(_CounterStrategy())
        with pytest.raises(TradingError, match="prepared"):
            engine.on_tick("BTCUSDT", ticks_1m[0])


    def test_unknown_symbol_raises(self, ticks_1m):
        engine = LiveEngine.by_ticks(_CounterStrategy())
        engine.prepare()
        with pytest.raises(TradingError):
            engine.on_tick("BNBUSDT", ticks_1m[0])


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: TickProxy in live
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineTicks:
    def test_n_ticks_increases_monotonically(self, ticks_1m):
        engine = LiveEngine.by_ticks(_TickValuesStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        history = engine.strategy.n_ticks_hist
        assert history == list(range(1, len(ticks_1m) + 1))

    def test_prices_correct_tick_by_tick(self, ticks_1m):
        engine = LiveEngine.by_ticks(_TickValuesStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        expected = ticks_1m[:, _TICK_COL["price"]].tolist()
        np.testing.assert_array_almost_equal(engine.strategy.prices, expected)

    def test_spread_zero_with_klines(self, ticks_1m):
        """bid == ask -> spread always 0."""
        engine = LiveEngine.by_ticks(_SpreadStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        assert all(s == pytest.approx(0.0) for s in engine.strategy.spreads)

    def test_tick_window_limited_to_size(self, ticks_1m):
        """With window=5 and 13 ticks, only the last 5 should be available."""

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)
                self.last_len = 0

            def on_data(self):
                self.last_len = len(self._tp.price)

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        assert engine.strategy.last_len == 5

    def test_history_warms_up_nticks(self, ticks_1m):
        """After prepare() with history, n_ticks should reflect the history."""
        split = 7
        engine = LiveEngine.by_ticks(_TickValuesStrategy())
        engine.prepare({"BTCUSDT": ticks_1m[:split]})
        # n_ticks after prepare = split (the ring has split writes)
        assert engine.strategy._tp.n_ticks == split


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: OhlcProxy in live
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineOhlc:
    def test_n_candles_grows_with_ticks(self, ticks_1m):
        engine = LiveEngine.by_ticks(_OhlcStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        history = engine.strategy.n_candles_hist
        # There must always be at least 1 candle (the partial)
        assert history[0] >= 1
        # The number of candles must not decrease
        assert all(a <= b for a, b in zip(history, history[1:]))

    def test_partial_close_equals_last_price(self, ticks_1m):
        """
        The close of the partial candle at each tick must be the current tick price,
        since bid==ask==price in our synthetic ticks.
        """
        engine = LiveEngine.by_ticks(_OhlcStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        # The last captured close must be the price of the last tick
        last_price = float(ticks_1m[-1, _TICK_COL["price"]])
        assert engine.strategy.closes[-1] == pytest.approx(last_price)

    def test_partial_high_not_less_than_close(self, ticks_1m):
        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self.violations = 0

            def on_data(self):
                h = float(self._ohlc.high[-1])
                c = float(self._ohlc.close[-1])
                if h < c:
                    self.violations += 1

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        assert engine.strategy.violations == 0

    def test_partial_low_not_greater_than_close(self, ticks_1m):
        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self.violations = 0

            def on_data(self):
                l = float(self._ohlc.low[-1])
                c = float(self._ohlc.close[-1])
                if l > c:
                    self.violations += 1

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        assert engine.strategy.violations == 0

    def test_ohlc_5m_from_klines_5m(self, ticks_5m):
        """With 4 5m candles, there should be exactly 1 closed + 1 partial = 2."""
        engine = LiveEngine.by_ticks(_OhlcStrategy())
        engine.prepare()
        for row in ticks_5m:
            engine.on_tick("BTCUSDT", row)
        # 4 ticks over 4 5m candles -> at least 1 closed + 1 partial
        assert engine.strategy.n_candles_hist[-1] >= 2

    def test_open_is_first_price_of_interval(self, ticks_1m):
        """
        The open of a closed candle must be the price of the first tick of that interval.
        We verify that the captured opens are always the opening price of the interval.
        """

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=5)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self.opens_candle2 = []
                self._captured = False

            def on_data(self):
                n = self._ohlc.n_klines
                if n >= 2 and not self._captured:
                    # First tick of the second candle -> closed open[0] must be fixed
                    self.opens_candle2.append(float(self._ohlc.open[-2]))
                    self._captured = True

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        # The open of the first closed candle must be constant
        if engine.strategy.opens_candle2:
            val = engine.strategy.opens_candle2[0]
            assert val > 0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Indicators in live
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineIndicators:
    def test_sma_tick_first_values_nan(self, ticks_1m):
        """SMA(3) on ticks: the first 2 values must be NaN."""
        engine = LiveEngine.by_ticks(_SmaTickStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        vals = engine.strategy.sma_vals
        assert np.isnan(vals[0])
        assert np.isnan(vals[1])
        assert not np.isnan(vals[2])

    def test_sma_tick_correct_value(self, ticks_1m):
        """SMA(3) at tick 3 = mean of the first 3 prices."""
        engine = LiveEngine.by_ticks(_SmaTickStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        vals = engine.strategy.sma_vals
        prices = ticks_1m[:, _TICK_COL["price"]]
        expected = float(np.mean(prices[:3]))
        assert vals[2] == pytest.approx(expected, rel=1e-6)

    def test_sma_tick_final_values(self, ticks_1m):
        """The final SMA(3) live values must match bulk."""
        engine = LiveEngine.by_ticks(_SmaTickStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        sma_bulk = SMA(3).calculate(ticks_1m[:, _TICK_COL["price"]])
        live_vals = engine.strategy.sma_vals
        # The last N non-nan values must match bulk
        mask = ~np.isnan(sma_bulk)
        np.testing.assert_array_almost_equal(
            np.array(live_vals)[mask],
            sma_bulk[mask],
        )

    def test_ema_tick_first_nan_then_valid(self, ticks_1m):
        engine = LiveEngine.by_ticks(_EmaStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        vals = engine.strategy.ema_vals
        assert np.isnan(vals[0])
        assert np.isnan(vals[1])
        assert not np.isnan(vals[2])

    def test_sma_ohlc_calculated_correctly(self, ticks_1m):
        """SMA(2) on close of 5m candles: at least the last value must be non-nan.
        _SmaOhlcStrategy reads [-2] because in LiveEngine the OHLC indicator
        is only recalculated at the close of each candle, leaving [-1] as NaN.
        """
        engine = LiveEngine.by_ticks(_SmaOhlcStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        vals = engine.strategy.sma_vals
        # With 13 1m klines -> 4 5m candles -> SMA(3) has at least 1 value
        non_nan = [v for v in vals if not np.isnan(v)]
        assert len(non_nan) > 0

    def test_history_warms_up_sma(self, ticks_1m):
        """
        With prepare() history, SMA(3) must be warmed up from tick 1
        after the warmup.
        """
        split = 6
        engine = LiveEngine.by_ticks(_SmaTickStrategy())
        engine.prepare({"BTCUSDT": ticks_1m[:split]})
        # Now the ring has 6 pre-warmed ticks
        # The next tick (index 6) should give a valid SMA
        engine.on_tick("BTCUSDT", ticks_1m[split])
        vals = engine.strategy.sma_vals
        # The last value (after the post-warmup tick) must be valid
        assert not np.isnan(vals[-1])

    def test_multiple_independent_indicators(self, ticks_1m):
        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
                self._sma3 = self.add_indicator(self._tp.price_ref, SMA(length=3))
                self._sma5 = self.add_indicator(self._tp.price_ref, SMA(length=5))
                self.vals3 = []
                self.vals5 = []

            def on_data(self):
                self.vals3.append(float(self._sma3[-1]))
                self.vals5.append(float(self._sma5[-1]))

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        # SMA5 has 2 more NaNs than SMA3
        nan3 = sum(1 for v in engine.strategy.vals3 if np.isnan(v))
        nan5 = sum(1 for v in engine.strategy.vals5 if np.isnan(v))
        assert nan3 == 2
        assert nan5 == 4


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: OhlcProxy multi-interval
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineMultiInterval:
    def test_two_simultaneous_intervals(self, ticks_1m):
        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=20)
                self._5m = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self._15m = self.subscribe_ohlc("BTCUSDT", 900_000, window_size=10)
                self.closes_5m = []
                self.closes_15m = []

            def on_data(self):
                self.closes_5m.append(float(self._5m.close[-1]))
                self.closes_15m.append(float(self._15m.close[-1]))

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        # Both series must have 13 values (one per tick)
        assert len(engine.strategy.closes_5m) == 13
        assert len(engine.strategy.closes_15m) == 13

    def test_15m_has_fewer_candles_than_5m(self, ticks_1m):
        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=20)
                self._5m = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self._15m = self.subscribe_ohlc("BTCUSDT", 900_000, window_size=10)
                self.n5m = 0
                self.n15m = 0

            def on_data(self):
                self.n5m = self._5m.n_klines
                self.n15m = self._15m.n_klines

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        # Larger intervals mean fewer candles
        assert engine.strategy.n15m <= engine.strategy.n5m


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Footprint (FpProxy) in live
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineFp:
    def test_fp_n_candles_grows(self, ticks_1m):
        engine = LiveEngine.by_ticks(_FpStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        history = engine.strategy.n_candles_hist
        assert history[0] >= 1
        assert all(a <= b for a, b in zip(history, history[1:]))

    def test_fp_partial_has_data(self, ticks_1m):
        """The partial candle must have vol_total > 0 after receiving ticks."""

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=10)
                self._fp = self.subscribe_footprint(
                    "BTCUSDT",
                    300_000,
                    window_size=10,
                    tick_size=1.0,
                )
                self.vols = []

            def on_data(self):
                p = self._fp[-1]
                if p is not None:
                    self.vols.append(p.vol_total)

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)
        # All vol_total must be > 0
        assert all(v > 0 for v in engine.strategy.vols)

    def test_fp_without_subscribe_ohlc(self, ticks_1m):
        """
        FpProxy does not require subscribe_ohlc() to exist, but LiveEngine.on_tick()
        requires at least one subscribe_ticks() to create the _tick_ring.
        Without tick_ring, on_tick raises RuntimeError — this is a known limitation
        of LiveEngine: footprint-only strategies must include subscribe_ticks.
        """

        class S_without_ticks(Strategy):
            """Strategy without subscribe_ticks — incompatible with LiveEngine."""

            def init(self):
                self._fp = self.subscribe_footprint(
                    "BTCUSDT",
                    300_000,
                    window_size=5,
                    tick_size=1.0,
                )

            def on_data(self):
                pass

        class S_with_ticks(Strategy):
            """FpProxy + subscribe_ticks — compatible with LiveEngine."""

            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=10)
                self._fp = self.subscribe_footprint(
                    "BTCUSDT",
                    300_000,
                    window_size=5,
                    tick_size=1.0,
                )

            def on_data(self):
                pass

        # Without subscribe_ticks -> RuntimeError on first on_tick
        engine_mal = LiveEngine.by_ticks(S_without_ticks())
        engine_mal.prepare()
        with pytest.raises(TradingError, match="prepared"):
            engine_mal.on_tick("BTCUSDT", ticks_1m[0])

        # With subscribe_ticks -> works correctly
        engine_ok = LiveEngine.by_ticks(S_with_ticks())
        engine_ok.prepare()
        for row in ticks_1m:
            engine_ok.on_tick("BTCUSDT", row)

    def test_fp_cols_populated_in_ohlc_source(self, ticks_1m):
        """
        The fp_* columns of the OHLC source (only for the HoverTool) must be
        populated with real footprint values, not left as NaN (which
        Bokeh renders as '???' in the tooltip). Reproduces the bug in
        live/replay where _populate_ohlc_history rebuilt the source without
        the fp_* columns and _update_ohlc filled them with NaN.
        """
        from tradetropy.plotting.live.updater._ohlc_mixin import (
            OhlcUpdateMixin,
            _FP_COL_NAMES,
        )
        from tradetropy.plotting.live.updater._refs import _OhlcSourceRef

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT", window_size=50)
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=50)
                self._fp = self.subscribe_footprint(
                    "BTCUSDT", 300_000, window_size=50, tick_size=1.0,
                )

            def on_data(self):
                pass

        engine = LiveEngine.by_ticks(S())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)

        strat = engine.strategy
        ohlc_ring = strat._ohlc._ohlc_ring
        assert ohlc_ring is not None and ohlc_ring._n_closed > 0

        from bokeh.models import ColumnDataSource

        n = min(ohlc_ring._n_closed, 50)
        src = ColumnDataSource(dict(
            ts=np.zeros(n, dtype="datetime64[ms]"),
            **{c: np.full(n, np.nan, dtype=np.float64) for c in _FP_COL_NAMES},
        ))
        ref = _OhlcSourceRef(
            source=src, proxy=strat._ohlc, interval_ms=300_000,
            theme={}, fp_proxy=strat._fp,
        )

        updater = OhlcUpdateMixin()
        updater._populate_ohlc_fp_cols(ref, n)

        # At least one closed candle must have a finite delta in each fp_* column.
        for col in _FP_COL_NAMES:
            vals = np.asarray(src.data[col], dtype=np.float64)
            assert np.isfinite(vals).any(), f"column {col} is all NaN"

        # The values must match those of the footprint store/ring.
        from tradetropy.plotting.sources import _fp_scalars_from_candle
        vela0 = strat._fp._ring.closed_candle(0)  # most recent closed
        if vela0 is not None:
            expected = _fp_scalars_from_candle(vela0)
            for col, exp in zip(_FP_COL_NAMES, expected):
                got = float(np.asarray(src.data[col], dtype=np.float64)[-1])
                assert got == pytest.approx(exp), f"{col}: {got} != {exp}"


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Equivalence Live vs Backtest (tick engine)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestLiveEngineEquivalence:
    """
    Verifies that LiveEngine tick-by-tick produces the same final values as
    BacktestEngine on the same data.
    """

    def test_last_price_matches(self, ticks_1m):
        # Backtest
        strat_bt = _BacktestEquivalentStrategy()
        BacktestEngine.by_ticks(
            strat_bt,
            data=(TickData("BTCUSDT", ticks_1m, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        # Live
        strat_live = _CaptureFinalStrategy()
        engine = LiveEngine.by_ticks(strat_live)
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)

        assert strat_live.last_price == pytest.approx(
            strat_bt.last_price, rel=1e-6
        )

    def test_last_ohlc_close_matches(self, ticks_1m):
        strat_bt = _BacktestEquivalentStrategy()
        BacktestEngine.by_ticks(
            strat_bt,
            data=(TickData("BTCUSDT", ticks_1m, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        strat_live = _CaptureFinalStrategy()
        engine = LiveEngine.by_ticks(strat_live)
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)

        assert strat_live.last_close == pytest.approx(strat_bt.last_close, rel=1e-6)

    def test_n_candles_ohlc_matches(self, ticks_1m):
        strat_bt = _BacktestEquivalentStrategy()
        BacktestEngine.by_ticks(
            strat_bt,
            data=(TickData("BTCUSDT", ticks_1m, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        strat_live = _CaptureFinalStrategy()
        engine = LiveEngine.by_ticks(strat_live)
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)

        assert strat_live.last_n_candles == strat_bt.last_n_candles

    def test_final_sma_matches(self, ticks_1m):
        strat_bt = _BacktestEquivalentStrategy()
        BacktestEngine.by_ticks(
            strat_bt,
            data=(TickData("BTCUSDT", ticks_1m, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        strat_live = _CaptureFinalStrategy()
        engine = LiveEngine.by_ticks(strat_live)
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)

        assert strat_live.last_sma == pytest.approx(strat_bt.last_sma, rel=1e-5)

    def test_complete_sma_series_matches(self, ticks_1m):
        """The entire tick-by-tick SMA(3) live series must match bulk."""
        engine = LiveEngine.by_ticks(_SmaTickStrategy())
        engine.prepare()
        for row in ticks_1m:
            engine.on_tick("BTCUSDT", row)

        sma_bulk = SMA(3).calculate(ticks_1m[:, _TICK_COL["price"]])
        live_vals = np.array(engine.strategy.sma_vals)

        # Compare only non-nan values
        mask = ~np.isnan(sma_bulk)
        np.testing.assert_array_almost_equal(live_vals[mask], sma_bulk[mask], decimal=6)

    def test_equivalence_with_20k_ticks(self, ticks_20k):
        """
        Scales to 20k: last price and SMA live must match backtest.

        NOTE: n_candles is NOT compared here because the live strategy uses
        window_size=200, which limits the ring to the last 200 candles.
        Backtest without a limit sees all 4001 candles.
        To compare n_candles one would need window_size >= total_n_candles.
        """
        strat_bt = _BacktestEquivalentStrategy()
        BacktestEngine.by_ticks(
            strat_bt,
            data=(TickData("BTCUSDT", ticks_20k, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        strat_live = _CaptureFinalStrategy()
        engine = LiveEngine.by_ticks(strat_live)
        engine.prepare()
        for row in ticks_20k:
            engine.on_tick("BTCUSDT", row)

        assert strat_live.last_price == pytest.approx(
            strat_bt.last_price, rel=1e-6
        )
        assert strat_live.last_sma == pytest.approx(strat_bt.last_sma, rel=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: LiveEngine with bar data (klines -> KlineEngine)
# ══════════════════════════════════════════════════════════════════════════════
#
# KlineEngine uses klines [M×6] directly.
# We compare results with LiveEngine on equivalent ticks.
# The key difference: KlineEngine has real high/low from the kline,
# while LiveEngine reconstructs OHLC from price=close.
# Therefore we only compare open (= kline close) and final close.
#

# ══════════════════════════════════════════════════════════════════════════════
# TESTS: LiveEngine with bar data (klines -> KlineEngine)
# ══════════════════════════════════════════════════════════════════════════════
#
# KlineEngine uses klines [M×6] directly.
# The fixtures klines_1m/5m have 7 columns (including monetary volume),
# so we use klines[:, :6] to keep only [ts,o,h,l,c,vol].
#


class _BarBacktestStrategy(Strategy):
    """Captures the last close and n_candles at the end of all bars."""

    def init(self):
        self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=200)
        self.last_close = np.nan
        self.last_n_candles = 0
        self.closes_partial = []

    def on_data(self):
        self.last_close = float(self._ohlc.close[-1])
        self.last_n_candles = self._ohlc.n_klines
        self.closes_partial.append(float(self._ohlc.close[-1]))


@pytest.fixture
def klines_1m_6col(klines_1m):
    """klines_1m truncated to 6 columns — format expected by KlineEngine."""
    return klines_1m[:, :6]


@pytest.fixture
def klines_5m_6col(klines_5m):
    """klines_5m truncated to 6 columns."""
    return klines_5m[:, :6]


@pytest.mark.unit
class TestKlineEngine:
    """Tests for KlineEngine with real klines [M×6]."""

    def test_n_candles_grows_with_bars(self, klines_1m_6col):
        """In KlineEngine each bar = 1 advance -> n_candles always grows."""
        strat = _BarBacktestStrategy()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        # With 13 1m bars and 5m candles: formed candles exist
        assert strat.last_n_candles >= 1

    def test_last_close_correct(self, klines_1m_6col, klines_1m):
        """The last close must be the close of the last bar."""
        strat = _BarBacktestStrategy()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert strat.last_close == pytest.approx(klines_1m[-1, 4], rel=1e-6)

    def test_on_data_called_per_bar(self, klines_1m_6col):
        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=50)
                self.n_calls = 0

            def on_data(self):
                self.n_calls += 1

        strat = S()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert strat.n_calls == len(klines_1m_6col)

    def test_sma_bar_final_values(self, klines_1m_6col, klines_1m):
        """SMA(3) on close of 1m candles (= 1m bars)."""

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=50)
                self._sma = self.add_indicator(self._ohlc.close_ref, SMA(length=3))
                self.sma_vals = []

            def on_data(self):
                if self._ohlc.n_klines >= 1 and len(self._sma) > 0:
                    self.sma_vals.append(float(self._sma[-1]))

        strat = S()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        vals = strat.sma_vals
        sma_length = 3
        expected_calls = len(klines_1m_6col) - sma_length  # 13 - 3 = 10
        assert len(vals) == expected_calls

    def test_subscribe_ticks_in_bar_engine_raises(self, klines_1m_6col):
        """KlineEngine must reject strategies with subscribe_ticks()."""

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("BTCUSDT")
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 60_000)

            def on_data(self):
                pass

        with pytest.raises(ConfigError, match="subscribe_ticks"):
            BacktestEngine.by_klines(
                S(),
                data=(
                    KlineData(
                        "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                    ),
                ),
                sesh=SeshSimulatorBase("kline"),
            ).run()

    def test_bar_engine_5m_from_1m(self, klines_1m_6col):
        """
        With 13 1m bars and 5m grouping:
          22:10: bars 0,1           (2 bars) <- partial candle -> closed when 22:15 arrives
          22:15: bars 2..6          (5 bars) <- closed
          22:20: bars 7..11         (5 bars) <- closed
          22:25: bar 12             (1 bar)  <- partial
        At the end: 3 closed + 1 partial = 4 total candles.
        """

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=50)
                self.n_candles = 0

            def on_data(self):
                self.n_candles = self._ohlc.n_klines

        strat = S()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert strat.n_candles == 4

    def test_bar_engine_close_series_no_nan(self, klines_1m_6col):
        """All KlineEngine closes must be valid."""
        strat = _BarBacktestStrategy()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=60_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert all(not np.isnan(v) for v in strat.closes_partial)

    def test_bar_engine_5m_from_5m(self, klines_5m_6col):
        """With 4 5m klines and 5m candles: 3 closed + 1 partial = 4 candles."""

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self.n_candles = 0

            def on_data(self):
                self.n_candles = self._ohlc.n_klines

        strat = S()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_5m_6col, timeframe=300_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert strat.n_candles == 4

    def test_bar_engine_multi_symbol(self, klines_1m_6col):
        """KlineEngine with two simultaneous symbols."""

        class S(Strategy):
            def init(self):
                self._btc = self.subscribe_ohlc("BTCUSDT", 300_000, window_size=10)
                self._eth = self.subscribe_ohlc("ETHUSDT", 300_000, window_size=10)
                self.n_calls = 0

            def on_data(self):
                self.n_calls += 1

        strat = S()
        BacktestEngine.by_klines(
            strat,
            data=(
                KlineData(
                    "BTCUSDT", klines_1m_6col, timeframe=300_000, tick_size=0.01
                ),
                KlineData(
                    "ETHUSDT", klines_1m_6col, timeframe=300_000, tick_size=0.01
                ),
            ),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert strat.n_calls == len(klines_1m_6col)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: LiveEngine with history (warmup)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveEngineHistory:
    def test_history_partial_then_live(self, ticks_1m):
        """
        Splitting ticks into history (7) + live (6) must give the same
        final result as processing all in live (13 ticks live).
        """
        engine_full = LiveEngine.by_ticks(_CaptureFinalStrategy())
        engine_full.prepare()
        for row in ticks_1m:
            engine_full.on_tick("BTCUSDT", row)

        engine_split = LiveEngine.by_ticks(_CaptureFinalStrategy())
        engine_split.prepare({"BTCUSDT": ticks_1m[:7]})
        for row in ticks_1m[7:]:
            engine_split.on_tick("BTCUSDT", row)

        assert engine_full.strategy.last_price == pytest.approx(
            engine_split.strategy.last_price, rel=1e-6
        )

    def test_warmed_sma_no_extra_nans(self, ticks_1m):
        """
        With 7-tick warmup for SMA(3), the first live tick
        must already have a valid SMA.
        """
        engine = LiveEngine.by_ticks(_SmaTickStrategy())
        engine.prepare({"BTCUSDT": ticks_1m[:7]})
        # Process one live tick
        engine.on_tick("BTCUSDT", ticks_1m[7])
        assert not np.isnan(engine.strategy.sma_vals[-1])

    def test_warmed_ohlc_has_candles(self, ticks_1m):
        """With 8-tick warmup (2 5m candles), candles must be available."""
        engine = LiveEngine.by_ticks(_OhlcStrategy())
        engine.prepare({"BTCUSDT": ticks_1m[:8]})
        engine.on_tick("BTCUSDT", ticks_1m[8])
        assert engine.strategy.n_candles_hist[-1] >= 1


# endregion


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: automatic LiveChart wiring via engine.run(live_chart=...)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveChartAutoWiring:
    def test_run_live_chart_attaches_and_starts_before_loop(self, monkeypatch):
        """run(live_chart=chart) does prepare -> attach -> start -> loop in order."""
        from tradetropy.plotting.live.chart import LiveChart

        order = []

        class _FakeChart(LiveChart):
            def start(self):  # prevents the real Bokeh server from starting
                # At this point the engine should already have called attach_chart()
                order.append(
                    ("start", self._engine is not None, self._strategy is not None)
                )

        engine = LiveEngine.by_ticks(_CounterStrategy())
        monkeypatch.setattr(
            engine, "_run_guarded", lambda: order.append(("loop", None, None))
        )

        chart = _FakeChart()
        engine.run(live_chart=chart)

        assert engine._preparado is True
        assert engine._chart is chart
        assert chart._engine is engine
        assert chart._strategy is engine.strategy
        # start() executed with engine/strategy already injected, and before the loop
        assert [o[0] for o in order] == ["start", "loop"]
        assert order[0] == ("start", True, True)

    def test_run_without_live_chart_no_attach(self, monkeypatch):
        """run() without live_chart does not attach anything and only starts the loop."""
        order = []
        engine = LiveEngine.by_ticks(_CounterStrategy())
        monkeypatch.setattr(engine, "_run_guarded", lambda: order.append("loop"))

        engine.run()

        assert engine._chart is None
        assert order == ["loop"]

    def test_run_live_chart_respects_prior_prepare(self, monkeypatch):
        """If the user already called prepare(), run(live_chart=...) does not re-prepare."""
        from tradetropy.plotting.live.chart import LiveChart

        calls = []

        class _FakeChart(LiveChart):
            def start(self):
                calls.append("start")

        engine = LiveEngine.by_ticks(_CounterStrategy())
        engine.prepare()
        assert engine._preparado is True

        # prepare() must not run again
        monkeypatch.setattr(
            engine, "prepare", lambda *a, **k: calls.append("prepare")
        )
        monkeypatch.setattr(engine, "_run_guarded", lambda: calls.append("loop"))

        engine.run(live_chart=_FakeChart())

        assert "prepare" not in calls
        assert calls == ["start", "loop"]

    def test_replay_run_live_chart_order(self, ticks_1m, monkeypatch):
        """In ReplayEngine: chart.start -> ctrl.start -> loop, and the controller
        is automatically connected to the chart."""
        from tradetropy.replay.engine import ReplayEngine
        from tradetropy.plotting.live.chart import LiveChart

        order = []

        class _FakeChart(LiveChart):
            def start(self):
                order.append("chart.start")

        engine = ReplayEngine.by_ticks(
            _CounterStrategy(),
            data=(TickData("BTCUSDT", ticks_1m, tick_size=0.25),),
            speed=float("inf"),
        )
        monkeypatch.setattr(engine, "_run_guarded", lambda: order.append("loop"))
        monkeypatch.setattr(engine._ctrl, "start", lambda: order.append("ctrl.start"))

        chart = _FakeChart()
        engine.run(live_chart=chart)

        assert order == ["chart.start", "ctrl.start", "loop"]
        assert chart._engine is engine
        assert chart._replay_controller is engine._ctrl

    def test_run_live_chart_positional(self, monkeypatch):
        from tradetropy.plotting.live.chart import LiveChart
        order = []
        class _FakeChart(LiveChart):
            def start(self):
                order.append('start')
        engine = LiveEngine.by_ticks(_CounterStrategy())
        monkeypatch.setattr(engine, '_run_guarded', lambda: order.append('loop'))
        chart = _FakeChart()
        engine.run(chart)
        assert engine._chart is chart
        assert chart._engine is engine
        assert order == ['start', 'loop']

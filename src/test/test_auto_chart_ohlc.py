"""
Tests for the chart-only OHLC proxy auto-injected for tick strategies.

A live/replay/training chart needs at least one OHLC proxy to draw candles.
A pure order-flow strategy (only subscribe_ticks) should not be forced to
declare subscribe_ohlc() just for plotting: the engine injects a default 1m
OHLC proxy (configurable via chart_ohlc_interval_ms), fed from the tick stream,
and excluded from the REST top-up so it never causes a network fetch.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.replay import ReplayEngine


def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.zeros((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["price"]] = close
    return out


class TickOnly(Strategy):
    """Pure order-flow style strategy: ticks only, no subscribe_ohlc()."""

    def init(self):
        self.ticks = self.subscribe_ticks("BTCUSDT")

    def on_data(self):
        pass


class WithOhlc(Strategy):
    """Strategy that declares its own OHLC; no injection must happen."""

    def init(self):
        self.ticks = self.subscribe_ticks("BTCUSDT")
        self.bars = self.subscribe_ohlc("BTCUSDT", 60_000)

    def on_data(self):
        pass


@pytest.fixture
def ticks(klines_20k):
    return klines_to_ticks(klines_20k[:300])


@pytest.mark.unit
class TestAutoChartOhlc:
    def test_injects_default_1m_for_tick_only(self, ticks):
        strat = TickOnly()
        eng = ReplayEngine.by_ticks(
            strat, data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        )
        eng.prepare()

        # Exactly the proxy LiveChart._setup validates exists now.
        assert len(strat._ohlc_proxies) == 1
        proxy = strat._ohlc_proxies[0]
        assert proxy.symbol == "BTCUSDT"
        assert proxy.interval_ms == 60_000
        # It is tracked as auto-injected (so top-up skips it).
        assert proxy in eng._auto_ohlc_proxies
        # Its ring was built and aggregates the tick stream into candles.
        assert proxy._ohlc_ring is not None
        ds = eng._sesh._datasets["BTCUSDT"]
        for i in range(len(ds)):
            eng._sesh._cursor["BTCUSDT"] = i
            eng._apply_tick("BTCUSDT", ds[i], with_broker=False)
        # 300 ticks at 1-min spacing -> many closed 1m candles.
        assert proxy._ohlc_ring._n_closed > 0

    def test_interval_is_configurable(self, ticks):
        strat = TickOnly()
        eng = ReplayEngine.by_ticks(
            strat, data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
            chart_ohlc_interval_ms=15_000,
        )
        eng.prepare()

        assert len(strat._ohlc_proxies) == 1
        assert strat._ohlc_proxies[0].interval_ms == 15_000

    def test_no_injection_when_strategy_declares_ohlc(self, ticks):
        strat = WithOhlc()
        eng = ReplayEngine.by_ticks(
            strat, data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        )
        eng.prepare()

        # Only the user's own proxy; nothing auto-injected.
        assert len(strat._ohlc_proxies) == 1
        assert eng._auto_ohlc_proxies == []

    def test_primary_symbol_resolves_via_injected_proxy(self, ticks):
        strat = TickOnly()
        eng = ReplayEngine.by_ticks(
            strat, data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        )
        eng.prepare()
        assert eng.primary_symbol() == "BTCUSDT"

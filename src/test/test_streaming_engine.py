"""
Tests for the unified event-driven engine path (LiveEngine._loop_streaming).

A streaming-capable fake session feeds scripted events through a FakeFeed; the
engine drains them from the EventBus and dispatches via _process_event. The
streaming path must NOT de-duplicate by timestamp (same-ms trades are kept).
"""

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.streaming import FakeFeed, KlineEvent, TradeEvent


class _FakeStreamSesh(SeshSimulatorBase):
    """Simulated session that advertises streaming and returns a scripted feed."""

    def __init__(self, events, *, raise_after=None, **kw):
        super().__init__(**kw)
        self._events = events
        self._raise_after = raise_after

    @property
    def supports_streaming(self) -> bool:
        return True

    def create_feed(self, **kwargs):
        return FakeFeed(self._events, raise_after=self._raise_after)


class _CounterStrat(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=100)
        self.n_calls = 0
        self.prices = []
        self.ts_seen = []

    def on_data(self):
        self.n_calls += 1
        self.prices.append(float(self.tp.price[-1]))
        self.ts_seen.append(int(self.tp.ts[-1]))


class _KlineStrat(Strategy):
    warmup = 0

    def init(self):
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=50)
        self.n_calls = 0
        self.closes = []

    def on_data(self):
        self.n_calls += 1
        self.closes.append(float(self.ohlc.close[-1]))


def _hist_ticks(n=5, base_ts=1_000_000):
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i in range(n):
        out[i, _TICK_COL["ts"]] = base_ts + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = 100.0
        out[i, _TICK_COL["volume"]] = 1.0
    return out


# =====
# Trades drive on_data end-to-end
# =====
class TestStreamingTicks:
    def test_trades_reach_on_data(self):
        events = [
            TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0, side=1)
            for i in range(8)
        ]
        sesh = _FakeStreamSesh(events, feed_type="tick")
        engine = LiveEngine.by_ticks(_CounterStrat(), sesh=sesh)
        engine.prepare({"BTCUSDT": _hist_ticks()})
        engine.run(blocking=True)

        strat = engine.strategy
        assert strat.n_calls == 8
        assert strat.prices[0] == 100.0
        assert strat.prices[-1] == 107.0

    def test_same_millisecond_trades_not_dropped(self):
        # Three trades share the same ts: the polling loop's `ts > last` guard
        # would drop two of them; the streaming path must keep all three.
        events = [
            TradeEvent("BTCUSDT", 2_000_000, 100.0, 1.0, side=1),
            TradeEvent("BTCUSDT", 2_000_000, 100.5, 2.0, side=1),
            TradeEvent("BTCUSDT", 2_000_000, 101.0, 1.0, side=-1),
        ]
        sesh = _FakeStreamSesh(events, feed_type="tick")
        engine = LiveEngine.by_ticks(_CounterStrat(), sesh=sesh)
        engine.prepare({"BTCUSDT": _hist_ticks()})
        engine.run(blocking=True)

        assert engine.strategy.n_calls == 3
        assert engine.strategy.prices == [100.0, 100.5, 101.0]


# =====
# Closed klines drive on_data (kline streaming mode)
# =====
class TestStreamingKlines:
    def test_closed_klines_reach_on_data(self):
        events = [
            KlineEvent("BTCUSDT", 3_000_000, 60_000, 1, 2, 0, 1.0, 5.0,
                       is_closed=True),
            KlineEvent("BTCUSDT", 3_060_000, 60_000, 1, 2, 0, 2.0, 6.0,
                       is_closed=True),
            KlineEvent("BTCUSDT", 3_120_000, 60_000, 1, 2, 0, 3.0, 7.0,
                       is_closed=True),
        ]
        sesh = _FakeStreamSesh(events, feed_type="kline")
        engine = LiveEngine.by_klines(_KlineStrat(), sesh=sesh)
        hist = np.array(
            [
                [2_400_000, 1.0, 2.0, 0.0, 1.0, 5.0],
                [2_460_000, 1.0, 2.0, 0.0, 1.0, 5.0],
                [2_520_000, 1.0, 2.0, 0.0, 1.0, 5.0],
            ],
            dtype=np.float64,
        )
        engine.prepare({("BTCUSDT", 60_000): hist})
        engine.run(blocking=True)

        # on_data fires at least on each new closed candle; closes are tracked.
        assert engine.strategy.n_calls >= 2
        assert 2.0 in engine.strategy.closes


# =====
# Feed error surfaces through the loop
# =====
class TestStreamingErrors:
    def test_feed_error_propagates_to_on_crash(self):
        captured = {}

        class _Strat(_CounterStrat):
            def on_crash(self, exc):
                captured["exc"] = exc
                return True  # suppress re-raise

        events = [TradeEvent("BTCUSDT", 2_000_000 + i, 100.0, 1.0) for i in range(5)]
        sesh = _FakeStreamSesh(events, raise_after=2, feed_type="tick")
        engine = LiveEngine.by_ticks(_Strat(), sesh=sesh)
        engine.prepare({"BTCUSDT": _hist_ticks()})
        engine.run(blocking=True)

        assert isinstance(captured.get("exc"), RuntimeError)

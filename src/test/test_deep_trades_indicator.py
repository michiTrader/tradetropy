"""Tests for the DeepTrades (L2) indicator: classification through the engine."""

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.ta import DeepTrades
from tradetropy.ta.order_flow._core import EVENT_ABSORPTION, EVENT_SWEEP
from tradetropy.streaming import (
    FakeFeed,
    OrderbookSnapshot,
    TradeEvent,
)


class _BookStreamSesh(SeshSimulatorBase):
    def __init__(self, events, **kw):
        super().__init__(**kw)
        self._events = events

    @property
    def supports_streaming(self):
        return True

    def create_feed(self, **kwargs):
        return FakeFeed(self._events)


class _DeepStrat(Strategy):
    warmup = 0

    def init(self):
        self.ticks = self.subscribe_ticks("BTCUSDT", window_size=200)
        self.book = self.subscribe_orderbook("BTCUSDT", depth=5)
        self.deep = self.add_indicator(
            DeepTrades.refs(self.ticks),
            DeepTrades(
                self.book, threshold=5.0, by="volume", window=10,
                min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
            ),
        )
        self.events = []

    def on_data(self):
        et = self.deep.event_type[-1]
        if not np.isnan(et):
            self.events.append((int(self.ticks.ts[-1]), int(et)))


def _hist_ticks(n=3, base_ts=1, price=100.0):
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i in range(n):
        out[i, _TICK_COL["ts"]] = base_ts + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = price
        out[i, _TICK_COL["volume"]] = 1.0
    return out


def test_deeptrades_classifies_sweep_and_absorption():
    events = [
        # Sweep book: thin stacked asks; big buy clears 3 levels.
        OrderbookSnapshot("BTCUSDT", 1_001, ((99.0, 5.0),),
                          ((100.0, 5.0), (101.0, 5.0), (102.0, 5.0), (103.0, 5.0))),
        TradeEvent("BTCUSDT", 1_002, 100.0, 16.0, side=1),
        # Absorption book: big ask wall; big buy is absorbed (not cleared).
        OrderbookSnapshot("BTCUSDT", 1_003, ((99.0, 5.0),),
                          ((100.0, 100.0), (101.0, 5.0))),
        TradeEvent("BTCUSDT", 1_004, 100.0, 90.0, side=1),
        # Small trade: below threshold, not detected.
        TradeEvent("BTCUSDT", 1_006, 100.0, 1.0, side=1),
    ]
    sesh = _BookStreamSesh(events, feed_type="tick")
    engine = LiveEngine.by_ticks(_DeepStrat(), sesh=sesh)
    engine.prepare({"BTCUSDT": _hist_ticks()})
    engine.run(blocking=True)

    by_ts = dict(engine.strategy.events)
    assert by_ts.get(1_002) == EVENT_SWEEP
    assert by_ts.get(1_004) == EVENT_ABSORPTION
    assert 1_006 not in by_ts  # below threshold


def test_deeptrades_requires_orderbook():
    from tradetropy.exceptions import ConfigError
    with pytest.raises(ConfigError):
        DeepTrades(orderbook=None, threshold=5.0, by="volume", window=5)

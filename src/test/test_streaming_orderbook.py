"""
OrderbookProxy + subscribe_orderbook + engine wiring.

A streaming session emits interleaved trades and order-book events; the strategy
reads live book metrics in on_data(). The book is stale until the first snapshot
and updates as snapshots/deltas arrive.
"""

import math

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.streaming import (
    FakeFeed,
    OrderbookDelta,
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


class _BookStrat(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.book = self.subscribe_orderbook("BTCUSDT", depth=5)
        self.records = []

    def on_data(self):
        self.records.append(
            (
                int(self.tp.ts[-1]),
                self.book.imbalance(),
                self.book.mid,
                self.book.stale,
            )
        )


def _hist_ticks(n=3, base_ts=1, price=100.0):
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i in range(n):
        out[i, _TICK_COL["ts"]] = base_ts + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = price
        out[i, _TICK_COL["volume"]] = 1.0
    return out


def test_orderbook_updates_through_engine():
    events = [
        TradeEvent("BTCUSDT", 1_000, 100.0, 1.0, side=1),           # before snapshot
        OrderbookSnapshot("BTCUSDT", 1_001, ((100.0, 3.0),), ((101.0, 1.0),)),
        TradeEvent("BTCUSDT", 1_002, 100.5, 1.0, side=1),           # imbalance 0.75
        OrderbookDelta("BTCUSDT", 1_003, 0, 0, ((100.0, 1.0),), ()),  # bid 3->1
        TradeEvent("BTCUSDT", 1_004, 100.5, 1.0, side=-1),         # imbalance 0.5
    ]
    sesh = _BookStreamSesh(events, feed_type="tick")
    engine = LiveEngine.by_ticks(_BookStrat(), sesh=sesh)
    engine.prepare({"BTCUSDT": _hist_ticks()})
    engine.run(blocking=True)

    recs = engine.strategy.records
    assert len(recs) == 3
    ts, imb, mid, stale = zip(*recs)
    assert list(ts) == [1_000, 1_002, 1_004]

    # 1st trade: no snapshot yet -> stale, NaN metrics.
    assert stale[0] is True
    assert math.isnan(imb[0]) and math.isnan(mid[0])

    # 2nd trade: snapshot applied -> bid_vol 3 / total 4 = 0.75, mid 100.5.
    assert stale[1] is False
    assert imb[1] == pytest.approx(0.75)
    assert mid[1] == pytest.approx(100.5)

    # 3rd trade: delta dropped best bid to 1 -> imbalance 0.5.
    assert imb[2] == pytest.approx(0.5)
    assert mid[2] == pytest.approx(100.5)


def test_book_empty_in_plain_backtest_is_stale():
    # Without a streaming feed, the proxy stays stale and metrics are NaN.
    strat = _BookStrat()
    engine = LiveEngine.by_ticks(strat, sesh=_BookStreamSesh([], feed_type="tick"))
    engine.prepare({"BTCUSDT": _hist_ticks()})
    # Book ring built but never fed -> stale.
    assert strat.book.stale is True
    assert math.isnan(strat.book.mid)

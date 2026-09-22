"""Tests for the streaming event protocol and the channel-aware EventBus."""

import threading
import time

import pytest

from tradetropy.streaming import (
    EventBus,
    FillEvent,
    KlineEvent,
    OrderEvent,
    OrderbookDelta,
    OrderbookSnapshot,
    TickEvent,
    TradeEvent,
)
from tradetropy.streaming._protocol import (
    BOOK_DELTA,
    BOOK_SNAPSHOT,
    FILL,
    KLINE,
    ORDER,
    TICK,
    TRADE,
)


# =====
# Event protocol
# =====
class TestProtocol:
    def test_channels_assigned(self):
        assert TradeEvent("BTC", 1, 100.0, 1.0).channel == TRADE
        assert TickEvent("BTC", 1, 99.0, 101.0, 100.0).channel == TICK
        assert KlineEvent("BTC", 1, 60_000, 1, 2, 0, 1, 5.0).channel == KLINE
        assert OrderbookSnapshot("BTC", 1, (), ()).channel == BOOK_SNAPSHOT
        assert OrderbookDelta("BTC", 1, 0, 0).channel == BOOK_DELTA

    def test_events_are_immutable(self):
        tr = TradeEvent("BTC", 1, 100.0, 1.0)
        with pytest.raises(Exception):
            tr.price = 200.0  # frozen dataclass

    def test_trade_defaults(self):
        tr = TradeEvent("BTC", 1, 100.0, 2.0)
        assert tr.side == 0
        assert tr.trade_id == -1
        assert tr.is_maker is False

    def test_kline_partial_default(self):
        k = KlineEvent("BTC", 1, 60_000, 1, 2, 0, 1, 5.0)
        assert k.is_closed is False

    def test_user_data_channels(self):
        oe = OrderEvent("BTC", 1, order_id=10, status="open", side=1,
                        price=100.0, volume=2.0)
        fe = FillEvent("BTC", 2, order_id=10, trade_id=5, side=1,
                       price=100.0, volume=1.0, fee=0.1)
        assert oe.channel == ORDER
        assert fe.channel == FILL
        assert oe.status == "open" and oe.filled == 0.0
        assert fe.trade_id == 5 and fe.position_id == -1


# =====
# EventBus ordering (trades / deltas never dropped or reordered)
# =====
class TestOrdering:
    def test_trades_preserved_in_order(self):
        bus = EventBus()
        for i in range(5):
            bus.put(TradeEvent("BTC", i, 100.0 + i, 1.0))
        out = bus.get_batch()
        assert [e.ts for e in out] == [0, 1, 2, 3, 4]
        assert all(isinstance(e, TradeEvent) for e in out)

    def test_book_events_preserved(self):
        bus = EventBus()
        bus.put(OrderbookSnapshot("BTC", 1, ((100.0, 1.0),), ((101.0, 1.0),)))
        bus.put(OrderbookDelta("BTC", 2, 1, 2, ((100.0, 0.5),), ()))
        bus.put(OrderbookDelta("BTC", 3, 2, 3, (), ((101.0, 0.0),)))
        out = bus.get_batch()
        assert [e.channel for e in out] == [
            BOOK_SNAPSHOT,
            BOOK_DELTA,
            BOOK_DELTA,
        ]
        assert [e.ts for e in out] == [1, 2, 3]

    def test_closed_klines_never_coalesced(self):
        bus = EventBus()
        bus.put(KlineEvent("BTC", 60_000, 60_000, 1, 2, 0, 1, 5.0, is_closed=True))
        bus.put(KlineEvent("BTC", 120_000, 60_000, 1, 2, 0, 1, 6.0, is_closed=True))
        out = bus.get_batch()
        assert len(out) == 2
        assert [e.ts for e in out] == [60_000, 120_000]

    def test_interleaved_order_preserved(self):
        bus = EventBus()
        bus.put(TradeEvent("BTC", 1, 100.0, 1.0))
        bus.put(OrderbookDelta("BTC", 2, 1, 2, ((100.0, 0.5),), ()))
        bus.put(TradeEvent("BTC", 3, 101.0, 1.0))
        out = bus.get_batch()
        assert [e.ts for e in out] == [1, 2, 3]

    def test_user_data_events_ordered_never_coalesced(self):
        bus = EventBus()
        bus.put(OrderEvent("BTC", 1, order_id=10, status="open", side=1))
        bus.put(OrderEvent("BTC", 2, order_id=10, status="filled", side=1,
                           filled=2.0))
        bus.put(FillEvent("BTC", 3, order_id=10, trade_id=1, side=1,
                          price=100.0, volume=2.0))
        out = bus.get_batch()
        # All three kept in order (no coalescing on ORDER/FILL).
        assert [e.channel for e in out] == [ORDER, ORDER, FILL]
        assert [e.ts for e in out] == [1, 2, 3]


# =====
# EventBus coalescing (quotes / partial klines keep only the latest)
# =====
class TestCoalescing:
    def test_quotes_coalesce_to_latest(self):
        bus = EventBus()
        for i in range(10):
            bus.put(TickEvent("BTC", i, 99.0, 101.0, 100.0 + i))
        out = bus.get_batch()
        assert len(out) == 1
        # The surviving quote is the most recent one.
        assert out[0].ts == 9
        assert out[0].price == 109.0

    def test_quotes_coalesce_per_symbol(self):
        bus = EventBus()
        bus.put(TickEvent("BTC", 1, 99.0, 101.0, 100.0))
        bus.put(TickEvent("ETH", 1, 9.0, 11.0, 10.0))
        bus.put(TickEvent("BTC", 2, 99.0, 101.0, 100.5))
        out = bus.get_batch()
        assert len(out) == 2
        by_sym = {e.symbol: e for e in out}
        assert by_sym["BTC"].ts == 2
        assert by_sym["ETH"].ts == 1

    def test_partial_klines_coalesce(self):
        bus = EventBus()
        for i in range(5):
            bus.put(
                KlineEvent("BTC", 60_000, 60_000, 1, 2, 0, 1.0 + i, 5.0)
            )
        out = bus.get_batch()
        assert len(out) == 1
        assert out[0].close == 5.0

    def test_partial_then_closed_kline_both_kept(self):
        bus = EventBus()
        bus.put(KlineEvent("BTC", 60_000, 60_000, 1, 2, 0, 1.0, 5.0))
        bus.put(
            KlineEvent("BTC", 60_000, 60_000, 1, 2, 0, 1.5, 5.0, is_closed=True)
        )
        out = bus.get_batch()
        # The partial coalesced into nothing more, the closed one is separate.
        assert len(out) == 2
        assert out[0].is_closed is False
        assert out[1].is_closed is True

    def test_coalesce_position_after_drain_resets(self):
        bus = EventBus()
        bus.put(TickEvent("BTC", 1, 99.0, 101.0, 100.0))
        first = bus.get_batch()
        assert len(first) == 1
        # After draining, a new quote must create a fresh slot (not coalesce
        # into the already-consumed one).
        bus.put(TickEvent("BTC", 2, 99.0, 101.0, 100.5))
        second = bus.get_batch()
        assert len(second) == 1
        assert second[0].ts == 2


# =====
# get_batch semantics
# =====
class TestGetBatch:
    def test_max_items_respected(self):
        bus = EventBus()
        for i in range(10):
            bus.put(TradeEvent("BTC", i, 100.0, 1.0))
        out = bus.get_batch(max_items=4)
        assert len(out) == 4
        assert [e.ts for e in out] == [0, 1, 2, 3]
        rest = bus.get_batch()
        assert len(rest) == 6

    def test_timeout_returns_empty(self):
        bus = EventBus()
        t0 = time.perf_counter()
        out = bus.get_batch(timeout=0.05)
        elapsed = time.perf_counter() - t0
        assert out == []
        assert elapsed >= 0.04

    def test_get_single(self):
        bus = EventBus()
        bus.put(TradeEvent("BTC", 1, 100.0, 1.0))
        ev = bus.get(timeout=0.0)
        assert ev is not None and ev.ts == 1
        assert bus.get(timeout=0.0) is None

    def test_clear(self):
        bus = EventBus()
        bus.put(TradeEvent("BTC", 1, 100.0, 1.0))
        bus.clear()
        assert bus.depth == 0
        assert bus.get_batch() == []


# =====
# Metrics
# =====
class TestStats:
    def test_depth_and_max_depth(self):
        bus = EventBus()
        for i in range(3):
            bus.put(TradeEvent("BTC", i, 100.0, 1.0))
        assert bus.depth == 3
        assert bus.max_depth == 3
        bus.get_batch()
        assert bus.depth == 0
        # High-water mark persists after draining.
        assert bus.max_depth == 3

    def test_coalesced_counter(self):
        bus = EventBus()
        for i in range(5):
            bus.put(TickEvent("BTC", i, 99.0, 101.0, 100.0))
        s = bus.stats
        assert s["total_put"] == 5
        assert s["total_coalesced"] == 4
        assert s["depth"] == 1


# =====
# Thread-safety smoke
# =====
class TestThreadSafety:
    def test_producer_consumer(self):
        bus = EventBus()
        n = 2000
        produced = {"count": 0}

        def producer():
            for i in range(n):
                bus.put(TradeEvent("BTC", i, 100.0 + i, 1.0))
                produced["count"] += 1

        t = threading.Thread(target=producer)
        t.start()

        collected = []
        deadline = time.perf_counter() + 5.0
        while len(collected) < n and time.perf_counter() < deadline:
            collected.extend(bus.get_batch(timeout=0.05))
        t.join(timeout=5.0)
        collected.extend(bus.get_batch())

        # No trades lost; strictly increasing ts (order preserved).
        assert len(collected) == n
        ts = [e.ts for e in collected]
        assert ts == sorted(ts)
        assert ts[0] == 0 and ts[-1] == n - 1

"""Tests for the WebSocketFeed ABC, FeedRunner thread, and FakeFeed."""

import threading
import time

import pytest

from tradetropy.streaming import (
    EventBus,
    FakeFeed,
    FeedRunner,
    TradeEvent,
)
from tradetropy.streaming._protocol import TRADE


def _trades(n):
    return [TradeEvent("BTC", i, 100.0 + i, 1.0) for i in range(n)]


# =====
# FakeFeed basic contract
# =====
class TestFakeFeed:
    def test_delivers_all_events_in_order(self):
        bus = EventBus()
        runner = FeedRunner(FakeFeed(_trades(50)), bus, ["BTC"], [TRADE])
        runner.start()
        runner.wait(timeout=5.0)  # one pass then listen() returns

        collected = bus.get_batch()
        assert len(collected) == 50
        assert [e.ts for e in collected] == list(range(50))
        assert not runner.is_alive()
        assert runner.error is None

    def test_connect_subscribe_disconnect_called(self):
        feed = FakeFeed(_trades(3))
        runner = FeedRunner(feed, EventBus(), ["BTC", "ETH"], [TRADE])
        runner.start()
        runner.wait(timeout=5.0)

        assert feed.connected is True
        assert feed.disconnected is True
        assert feed.subscribed == (["BTC", "ETH"], [TRADE])


# =====
# Clean shutdown of a never-ending feed
# =====
class TestShutdown:
    def test_stop_joins_cleanly(self):
        bus = EventBus()
        feed = FakeFeed(_trades(5), delay=0.001, loop_forever=True)
        runner = FeedRunner(feed, bus, ["BTC"], [TRADE])
        runner.start()

        time.sleep(0.05)  # let it pump a while
        assert runner.is_alive()

        runner.stop(timeout=5.0)
        assert not runner.is_alive()
        assert feed.disconnected is True
        # Some events made it through before stopping.
        assert bus.depth > 0 or bus.max_depth > 0


# =====
# Error propagation
# =====
class TestErrorPropagation:
    def test_error_surfaces_via_callback_and_property(self):
        seen = {}

        def on_error(exc):
            seen["exc"] = exc

        bus = EventBus()
        feed = FakeFeed(_trades(10), raise_after=3)
        runner = FeedRunner(feed, bus, ["BTC"], [TRADE], on_error=on_error)
        runner.start()
        runner.wait(timeout=5.0)

        assert isinstance(runner.error, RuntimeError)
        assert isinstance(seen.get("exc"), RuntimeError)
        # The events emitted before the failure still reached the bus.
        assert len(bus.get_batch()) == 3
        # disconnect still ran during teardown.
        assert feed.disconnected is True

    def test_error_does_not_kill_silently_without_callback(self):
        runner = FeedRunner(
            FakeFeed(_trades(10), raise_after=1), EventBus(), ["BTC"], [TRADE]
        )
        runner.start()
        runner.wait(timeout=5.0)
        assert isinstance(runner.error, RuntimeError)


# =====
# Guards
# =====
class TestGuards:
    def test_double_start_raises(self):
        runner = FeedRunner(FakeFeed(_trades(1)), EventBus(), ["BTC"], [TRADE])
        runner.start()
        runner.wait(timeout=5.0)
        with pytest.raises(RuntimeError):
            runner.start()

    def test_stop_before_start_is_noop(self):
        runner = FeedRunner(FakeFeed(_trades(1)), EventBus(), ["BTC"], [TRADE])
        runner.stop()  # must not raise
        assert not runner.is_alive()

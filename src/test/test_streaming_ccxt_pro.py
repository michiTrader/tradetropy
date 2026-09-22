"""Tests for CCXTProFeed normalization against a mock ccxt.pro exchange."""

import asyncio
import time

import pytest

from tradetropy.streaming import (
    CCXTProFeed,
    EventBus,
    FeedRunner,
    KlineEvent,
    OrderbookSnapshot,
    TradeEvent,
)
from tradetropy.streaming._protocol import BOOK_SNAPSHOT, KLINE, TRADE


class _MockProExchange:
    """
    Minimal stand-in for a ccxt.pro exchange.

    Each watch_* returns its scripted payload on the first call, then blocks
    (awaits a long sleep) so the per-pair loop stays alive until cancelled,
    mirroring a real never-ending WebSocket subscription.
    """

    id = "mock"

    def __init__(self, *, trades=None, ohlcv=None, order_book=None,
                 orders=None, mytrades=None):
        self._trades = trades
        self._ohlcv = ohlcv
        self._order_book = order_book
        self._orders = orders
        self._mytrades = mytrades
        self.closed = False
        self._calls = {"trades": 0, "ohlcv": 0, "book": 0,
                       "orders": 0, "mytrades": 0}

    async def _once(self, key, payload):
        self._calls[key] += 1
        if self._calls[key] == 1:
            return payload
        await asyncio.sleep(3600)

    async def watch_trades(self, symbol):
        return await self._once("trades", self._trades)

    async def watch_ohlcv(self, symbol, timeframe):
        return await self._once("ohlcv", self._ohlcv)

    async def watch_order_book(self, symbol, limit=None):
        return await self._once("book", self._order_book)

    async def watch_orders(self, symbol):
        return await self._once("orders", getattr(self, "_orders", None))

    async def watch_my_trades(self, symbol):
        return await self._once("mytrades", getattr(self, "_mytrades", None))

    async def close(self):
        self.closed = True


def _drain_until(bus, count, timeout=2.0):
    collected = []
    deadline = time.perf_counter() + timeout
    while len(collected) < count and time.perf_counter() < deadline:
        collected.extend(bus.get_batch(timeout=0.02))
    return collected


# =====
# Trades
# =====
class TestTrades:
    def test_trade_normalization(self):
        trades = [
            {"timestamp": 1000, "price": 100.0, "amount": 2.0,
             "side": "buy", "id": "5", "takerOrMaker": "taker"},
            {"timestamp": 1001, "price": 99.0, "amount": 1.0,
             "side": "sell", "id": "6"},
        ]
        bus = EventBus()
        feed = CCXTProFeed(_MockProExchange(trades=trades))
        runner = FeedRunner(feed, bus, ["BTC/USDT"], [TRADE])
        runner.start()
        out = _drain_until(bus, 2)
        runner.stop()

        assert len(out) == 2
        assert all(isinstance(e, TradeEvent) for e in out)
        assert [e.side for e in out] == [1, -1]
        assert out[0].price == 100.0 and out[0].volume == 2.0
        assert out[0].trade_id == 5
        assert out[0].is_maker is False
        assert out[1].ts == 1001


# =====
# Order book (snapshot)
# =====
class TestOrderBook:
    def test_book_snapshot_normalization(self):
        ob = {
            "timestamp": 2000,
            "nonce": 42,
            "bids": [[100.0, 1.0], [99.5, 2.0], [99.0, 3.0]],
            "asks": [[100.5, 1.5], [101.0, 2.5]],
        }
        bus = EventBus()
        feed = CCXTProFeed(_MockProExchange(order_book=ob), book_limit=2)
        runner = FeedRunner(feed, bus, ["BTC/USDT"], [BOOK_SNAPSHOT])
        runner.start()
        out = _drain_until(bus, 1)
        runner.stop()

        assert len(out) == 1
        snap = out[0]
        assert isinstance(snap, OrderbookSnapshot)
        assert snap.ts == 2000 and snap.last_id == 42
        # book_limit=2 truncates levels.
        assert snap.bids == ((100.0, 1.0), (99.5, 2.0))
        assert snap.asks == ((100.5, 1.5), (101.0, 2.5))


# =====
# Klines (closed vs partial)
# =====
class TestKlines:
    def test_closed_and_partial(self):
        ohlcv = [
            [60_000, 1.0, 2.0, 0.5, 1.5, 5.0],
            [120_000, 1.5, 2.5, 1.0, 2.0, 6.0],   # last -> partial
        ]
        bus = EventBus()
        feed = CCXTProFeed(_MockProExchange(ohlcv=ohlcv), timeframe_ms=60_000)
        runner = FeedRunner(feed, bus, ["BTC/USDT"], [KLINE])
        runner.start()
        out = _drain_until(bus, 2)
        runner.stop()

        assert len(out) == 2
        assert all(isinstance(e, KlineEvent) for e in out)
        closed, partial = out[0], out[1]
        assert closed.is_closed is True and closed.ts == 60_000
        assert closed.close == 1.5 and closed.interval_ms == 60_000
        assert partial.is_closed is False and partial.ts == 120_000


# =====
# Private user-data channels (orders / fills)
# =====
class TestUserData:
    def test_order_and_fill_normalization(self):
        from tradetropy.streaming import FillEvent, OrderEvent
        from tradetropy.streaming._protocol import FILL, ORDER

        orders = [{"id": "100", "symbol": "BTC/USDT", "side": "buy",
                   "price": 50000.0, "amount": 2.0, "filled": 1.0,
                   "status": "open", "type": "limit", "timestamp": 1000,
                   "clientOrderId": "abc"}]
        mytrades = [{"id": "9", "order": "100", "symbol": "BTC/USDT",
                     "side": "buy", "price": 50000.0, "amount": 1.0,
                     "fee": {"cost": 0.5}, "takerOrMaker": "taker",
                     "timestamp": 1001}]
        bus = EventBus()
        feed = CCXTProFeed(_MockProExchange(orders=orders, mytrades=mytrades))
        runner = FeedRunner(feed, bus, ["BTC/USDT"], [ORDER, FILL])
        runner.start()
        out = _drain_until(bus, 2)
        runner.stop()

        by_ch = {e.channel: e for e in out}
        oe = by_ch.get(ORDER)
        fe = by_ch.get(FILL)
        assert isinstance(oe, OrderEvent)
        assert oe.order_id == 100 and oe.status == "open" and oe.side == 1
        assert oe.filled == 1.0 and oe.client_id == "abc"
        assert isinstance(fe, FillEvent)
        assert fe.order_id == 100 and fe.trade_id == 9 and fe.side == 1
        assert fe.fee == 0.5 and fe.is_maker is False


# =====
# Capability / config guards (surfaced via runner.error)
# =====
class TestGuards:
    def test_missing_capability_errors(self):
        class _NoTrades:
            id = "notrades"
            async def close(self):
                pass

        runner = FeedRunner(
            CCXTProFeed(_NoTrades()), EventBus(), ["BTC/USDT"], [TRADE]
        )
        runner.start()
        runner.wait(timeout=2.0)
        from tradetropy.exceptions import TradingError
        assert isinstance(runner.error, TradingError)

    def test_unknown_channel_errors(self):
        runner = FeedRunner(
            CCXTProFeed(_MockProExchange(trades=[])),
            EventBus(),
            ["BTC/USDT"],
            ["bogus"],
        )
        runner.start()
        runner.wait(timeout=2.0)
        from tradetropy.exceptions import ConfigError
        assert isinstance(runner.error, ConfigError)

    def test_unknown_exchange_string_errors(self):
        # connect() resolves a string id; an absurd id must raise clearly.
        runner = FeedRunner(
            CCXTProFeed("definitely_not_an_exchange"),
            EventBus(),
            ["BTC/USDT"],
            [TRADE],
        )
        runner.start()
        runner.wait(timeout=3.0)
        assert runner.error is not None

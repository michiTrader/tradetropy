"""
Real-time WebSocket streaming for Tradetropy.

This package bridges asynchronous exchange WebSocket feeds to the synchronous
engine loop:

- ``_protocol``: normalized, exchange-agnostic event types.
- ``base``: the channel-aware EventBus (and, later, the WebSocketFeed ABC and
  feed-thread runner).

Strategies are unaffected: streaming is a live-mode transport detail behind the
same Sesh / proxy interfaces used by backtest and replay.
"""

from tradetropy.streaming._protocol import (
    BOOK_DELTA,
    BOOK_SNAPSHOT,
    FILL,
    FeedEvent,
    FillEvent,
    KLINE,
    KlineEvent,
    Level,
    MBO,
    MboEvent,
    ORDER,
    OrderEvent,
    OrderbookDelta,
    OrderbookSnapshot,
    TICK,
    TRADE,
    TickEvent,
    TradeEvent,
)
from tradetropy.streaming.base import EventBus, FeedRunner, WebSocketFeed
from tradetropy.streaming.ccxt_pro import CCXTProFeed
from tradetropy.streaming.fake import FakeFeed
from tradetropy.streaming.replay_feed import ReplayFeed

__all__ = [
    "EventBus",
    "WebSocketFeed",
    "FeedRunner",
    "FakeFeed",
    "CCXTProFeed",
    "ReplayFeed",
    "FeedEvent",
    "TradeEvent",
    "TickEvent",
    "KlineEvent",
    "OrderbookSnapshot",
    "OrderbookDelta",
    "OrderEvent",
    "FillEvent",
    "MboEvent",
    "Level",
    "TRADE",
    "TICK",
    "KLINE",
    "BOOK_SNAPSHOT",
    "BOOK_DELTA",
    "ORDER",
    "FILL",
    "MBO",
]

"""
Normalized streaming event protocol.

This module is dependency-light (standard library only): it defines the
immutable, exchange-agnostic events that every WebSocketFeed produces and that
the engine consumes through the EventBus. Connectors convert their raw venue
messages into these types so the engine never sees exchange-specific shapes.

Event channels
--------------
- TRADE          : an executed trade (time and sales), one per print.
- TICK           : an L1 quote update (best bid / best ask).
- KLINE          : an OHLCV candle update (partial or closed).
- BOOK_SNAPSHOT  : a full L2 order-book image (best N levels).
- BOOK_DELTA     : an incremental L2 order-book update.

Ordering semantics (see EventBus): TRADE, KLINE(closed) and the order-book
channels are an ordered log that must never be dropped; TICK and KLINE(partial)
are state that may be coalesced to the latest value per symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Union

# Channel identifiers (used by the EventBus for ordering / coalescing policy).
TRADE = "trade"
TICK = "tick"
KLINE = "kline"
BOOK_SNAPSHOT = "book_snapshot"
BOOK_DELTA = "book_delta"
# Private (authenticated) user-data channels.
ORDER = "order"
FILL = "fill"
# L3 / market-by-order channel.
MBO = "mbo"

# A single price level: (price, quantity).
Level = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """
    A single executed trade from the time-and-sales feed.

    Args:
        symbol (str): Trading symbol (venue native, e.g. 'BTC/USDT').
        ts (int): Trade timestamp in milliseconds (UTC).
        price (float): Execution price.
        volume (float): Executed size (base units).
        side (int): Aggressor side, +1 buy / -1 sell / 0 unknown.
        trade_id (int): Venue trade id (-1 when unavailable).
        is_maker (bool): True if the resting side was the maker (when known).
    """

    symbol: str
    ts: int
    price: float
    volume: float
    side: int = 0
    trade_id: int = -1
    is_maker: bool = False
    channel: str = field(default=TRADE, init=False)


@dataclass(frozen=True, slots=True)
class TickEvent:
    """
    An L1 quote update (top of book).

    Args:
        symbol (str): Trading symbol.
        ts (int): Quote timestamp in milliseconds (UTC).
        bid (float): Best bid price.
        ask (float): Best ask price.
        price (float): Reference/last price (mid or last trade).
        volume (float): Size associated with the update (0.0 if not provided).
        flags (float): Aggressor encoding compatible with tick `flags`
            (+1 buy / -1 sell / 0 unknown).
    """

    symbol: str
    ts: int
    bid: float
    ask: float
    price: float
    volume: float = 0.0
    flags: float = 0.0
    channel: str = field(default=TICK, init=False)


@dataclass(frozen=True, slots=True)
class KlineEvent:
    """
    An OHLCV candle update.

    Args:
        symbol (str): Trading symbol.
        ts (int): Candle open timestamp in milliseconds (UTC).
        interval_ms (int): Candle interval in milliseconds.
        open (float): Open price.
        high (float): High price.
        low (float): Low price.
        close (float): Close price.
        volume (float): Candle volume (base units).
        turnover (float): Quote-volume / turnover (NaN when unavailable).
        is_closed (bool): True when the candle is final (closed). Partial
            candles are coalesced by the EventBus; closed candles are never
            dropped.
    """

    symbol: str
    ts: int
    interval_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float = float("nan")
    is_closed: bool = False
    channel: str = field(default=KLINE, init=False)


@dataclass(frozen=True, slots=True)
class OrderbookSnapshot:
    """
    A full L2 order-book image (best N levels).

    A snapshot resets the reconstructed book. CCXT Pro's watch_order_book
    returns a fully reconstructed book on each update, so most order-book events
    on that path are snapshots.

    Args:
        symbol (str): Trading symbol.
        ts (int): Snapshot timestamp in milliseconds (UTC).
        bids (tuple[Level, ...]): Bid levels, best first, each (price, qty).
        asks (tuple[Level, ...]): Ask levels, best first, each (price, qty).
        last_id (int): Venue sequence/update id of this image (-1 if unknown).
    """

    symbol: str
    ts: int
    bids: Tuple[Level, ...]
    asks: Tuple[Level, ...]
    last_id: int = -1
    channel: str = field(default=BOOK_SNAPSHOT, init=False)


@dataclass(frozen=True, slots=True)
class OrderbookDelta:
    """
    An incremental L2 order-book update.

    Each change is a (price, qty) pair; a qty of 0.0 removes the level. Deltas
    are an ordered log and must never be dropped or reordered, since each one
    mutates the book relative to the previous state.

    Args:
        symbol (str): Trading symbol.
        ts (int): Delta timestamp in milliseconds (UTC).
        first_id (int): First venue sequence id covered (-1 if unknown).
        last_id (int): Last venue sequence id covered (-1 if unknown).
        bids (tuple[Level, ...]): Changed bid levels, each (price, qty).
        asks (tuple[Level, ...]): Changed ask levels, each (price, qty).
    """

    symbol: str
    ts: int
    first_id: int
    last_id: int
    bids: Tuple[Level, ...] = ()
    asks: Tuple[Level, ...] = ()
    channel: str = field(default=BOOK_DELTA, init=False)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """
    A private order-lifecycle update from the authenticated user-data stream.

    Reports the current state of one of the account's orders. Ordered and never
    dropped (each update mutates the known order state). Used to keep the
    session's order cache current without polling.

    Args:
        symbol (str): Trading symbol.
        ts (int): Update timestamp in milliseconds (UTC).
        order_id (int): Venue order id.
        status (str): Lifecycle status: 'open', 'closed', 'filled',
            'canceled', 'rejected', 'expired'.
        side (int): +1 buy / -1 sell.
        price (float): Order price (0 for market orders).
        volume (float): Order size (base units).
        filled (float): Cumulative filled size so far.
        order_type (str): Venue order type (e.g. 'limit', 'market').
        client_id (str): Client order id / comment (when set).
    """

    symbol: str
    ts: int
    order_id: int
    status: str
    side: int = 0
    price: float = 0.0
    volume: float = 0.0
    filled: float = 0.0
    order_type: str = ""
    client_id: str = ""
    channel: str = field(default=ORDER, init=False)


@dataclass(frozen=True, slots=True)
class FillEvent:
    """
    A private execution (deal/fill) from the authenticated user-data stream.

    Reports one execution against the account. Ordered and never dropped (each
    fill changes realized position / balance). Used to keep the session's deal
    history and positions current without polling.

    Args:
        symbol (str): Trading symbol.
        ts (int): Fill timestamp in milliseconds (UTC).
        order_id (int): Venue order id this fill belongs to.
        trade_id (int): Venue trade/execution id (-1 if unavailable).
        side (int): +1 buy / -1 sell.
        price (float): Fill price.
        volume (float): Filled size (base units).
        fee (float): Fee/commission paid for this fill.
        is_maker (bool): True if the account was the maker.
        position_id (int): Position id this fill applies to (-1 if unknown).
    """

    symbol: str
    ts: int
    order_id: int
    trade_id: int = -1
    side: int = 0
    price: float = 0.0
    volume: float = 0.0
    fee: float = 0.0
    is_maker: bool = False
    position_id: int = -1
    channel: str = field(default=FILL, init=False)


# Union of all concrete event types. Used for type hints across the engine.
FeedEvent = Union[
    TradeEvent,
    TickEvent,
    KlineEvent,
    OrderbookSnapshot,
    OrderbookDelta,
    OrderEvent,
    FillEvent,
    "MboEvent",
]


# MBO action codes (mirror core.data_types.MBO_*).
MBO_ADD = 0
MBO_MODIFY = 1
MBO_CANCEL = 2
MBO_TRADE = 3


@dataclass(frozen=True, slots=True)
class MboEvent:
    """
    A single L3 / market-by-order event.

    Args:
        symbol (str): Trading symbol.
        ts (int): Event timestamp in milliseconds (UTC).
        order_id (int): Venue order id.
        side (int): +1 bid / -1 ask.
        price (float): Order price.
        size (float): New resting size (ADD/MODIFY) or remaining (TRADE).
        action (int): MBO_ADD / MBO_MODIFY / MBO_CANCEL / MBO_TRADE.
    """

    symbol: str
    ts: int
    order_id: int
    side: int
    price: float
    size: float
    action: int
    channel: str = field(default=MBO, init=False)

"""
ReplayFeed - deterministically replay a recorded event stream.

Reads recorded tick rows (as written by the live recording path and loaded with
io.read_ticks) and re-emits them as normalized TradeEvents through the SAME
WebSocketFeed interface the live feed uses. Driven by the engine's streaming
loop, this reproduces the exact on_data() decision stream of the original live
session - the backtest == live parity guarantee for the streaming path.

Replay is as-fast-as-possible (no pacing) and ends after the recorded events
are exhausted, so a blocking engine run returns when the replay completes.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict, Iterable, List, Optional

import numpy as np

from tradetropy.core.constants import _TICK_COL
from tradetropy.streaming._protocol import (
    FeedEvent,
    OrderbookSnapshot,
    TradeEvent,
)
from tradetropy.streaming.base import WebSocketFeed


class ReplayFeed(WebSocketFeed):
    """
    A WebSocketFeed that replays a fixed, time-ordered list of events.

    Args:
        events (Iterable[FeedEvent]): Events to replay, merged in ts order.
    """

    def __init__(self, events: Iterable[FeedEvent]):
        self._events: List[FeedEvent] = sorted(events, key=lambda e: e.ts)
        self.connected: bool = False
        self.disconnected: bool = False
        self.subscribed = None

    @classmethod
    def from_ticks(cls, ticks_by_symbol: Dict[str, np.ndarray]) -> "ReplayFeed":
        """
        Build a ReplayFeed from recorded tick arrays.

        Each recorded row was originally produced from a TradeEvent (bid = ask =
        price, volume_real = volume, flags = aggressor side), so reconstructing
        a TradeEvent and re-running it through the engine yields the identical
        tick row - hence identical indicators and on_data() decisions.

        Args:
            ticks_by_symbol (dict[str, np.ndarray]): symbol -> [N x 7] tick rows
                in the standard column layout.

        Returns:
            ReplayFeed: A feed that emits the reconstructed trades in ts order.
        """
        events: List[FeedEvent] = []
        for symbol, arr in ticks_by_symbol.items():
            a = np.asarray(arr, dtype=np.float64)
            for row in a:
                events.append(
                    TradeEvent(
                        symbol=symbol,
                        ts=int(row[_TICK_COL["ts"]]),
                        price=float(row[_TICK_COL["price"]]),
                        volume=float(row[_TICK_COL["volume"]]),
                        side=int(row[_TICK_COL["flags"]]),
                    )
                )
        return cls(events)

    @staticmethod
    def _book_events(books_by_symbol: Dict[str, object]) -> List[FeedEvent]:
        """
        Build OrderbookSnapshot events from recorded BookData.

        Every recorded book row stores the full reconstructed top-K image (not
        the incremental change), so replaying each row as a snapshot reproduces
        the exact book state - regardless of whether it was originally a
        snapshot or a delta.

        Args:
            books_by_symbol (dict[str, BookData]): symbol -> recorded BookData.

        Returns:
            list[FeedEvent]: OrderbookSnapshot events.
        """
        events: List[FeedEvent] = []
        for symbol, book in books_by_symbol.items():
            ts = book.ts
            bid_px, bid_sz = book.bid_px, book.bid_sz
            ask_px, ask_sz = book.ask_px, book.ask_sz
            for i in range(len(book.data)):
                bids = tuple(
                    (float(p), float(s))
                    for p, s in zip(bid_px[i], bid_sz[i])
                    if not np.isnan(p) and not np.isnan(s)
                )
                asks = tuple(
                    (float(p), float(s))
                    for p, s in zip(ask_px[i], ask_sz[i])
                    if not np.isnan(p) and not np.isnan(s)
                )
                events.append(
                    OrderbookSnapshot(symbol, int(ts[i]), bids, asks)
                )
        return events

    @classmethod
    def from_records(
        cls,
        ticks_by_symbol: "Optional[Dict[str, np.ndarray]]" = None,
        books_by_symbol: "Optional[Dict[str, object]]" = None,
    ) -> "ReplayFeed":
        """
        Build a ReplayFeed merging recorded trades and order-book snapshots.

        Events from both streams are merged and replayed in timestamp order, so
        a strategy that reads the book in on_data() sees the same book-as-of
        each trade as it did live - the order-book replay parity guarantee.

        Note: exact interleaving of trade and book events that share the same
        millisecond is resolved by a stable ts sort (trades inserted before
        book), not by a unified sequence; record with distinct timestamps for
        strict same-ms ordering.

        Args:
            ticks_by_symbol (dict | None): symbol -> recorded [N x 7] tick rows.
            books_by_symbol (dict | None): symbol -> recorded BookData.

        Returns:
            ReplayFeed: A feed replaying the merged, ts-ordered event stream.
        """
        events: List[FeedEvent] = []
        if ticks_by_symbol:
            events.extend(cls.from_ticks(ticks_by_symbol)._events)
        if books_by_symbol:
            events.extend(cls._book_events(books_by_symbol))
        return cls(events)

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(
        self, symbols: Iterable[str], channels: Iterable[str]
    ) -> None:
        self.subscribed = (list(symbols), list(channels))

    async def listen(self) -> AsyncIterator[FeedEvent]:
        for ev in self._events:
            yield ev

    async def disconnect(self) -> None:
        self.disconnected = True

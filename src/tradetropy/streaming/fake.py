"""
FakeFeed - a scripted WebSocketFeed for tests and demos.

Replays a fixed list of normalized events (mirroring the role of FakeLiveSesh
for sessions), so the EventBus, FeedRunner and engine event path can be tested
deterministically without a network connection.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable, List, Optional

from tradetropy.streaming._protocol import FeedEvent
from tradetropy.streaming.base import WebSocketFeed


class FakeFeed(WebSocketFeed):
    """
    A WebSocketFeed that yields a predetermined sequence of events.

    Args:
        events (Iterable[FeedEvent]): Events to emit, in order.
        delay (float): Optional per-event sleep in seconds (simulates pacing).
        loop_forever (bool): If True, replay the sequence endlessly (useful for
            testing clean shutdown via FeedRunner.stop()). If False (default),
            ``listen`` returns after one pass.
        raise_after (int | None): If set, raise RuntimeError after emitting this
            many events (to test error propagation).
    """

    def __init__(
        self,
        events: Iterable[FeedEvent],
        *,
        delay: float = 0.0,
        loop_forever: bool = False,
        raise_after: Optional[int] = None,
    ):
        self._events: List[FeedEvent] = list(events)
        self._delay = float(delay)
        self._loop_forever = bool(loop_forever)
        self._raise_after = raise_after

        self.connected: bool = False
        self.disconnected: bool = False
        self.subscribed: Optional[tuple] = None

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(
        self, symbols: Iterable[str], channels: Iterable[str]
    ) -> None:
        self.subscribed = (list(symbols), list(channels))

    async def listen(self) -> AsyncIterator[FeedEvent]:
        emitted = 0
        while True:
            for ev in self._events:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield ev
                emitted += 1
                if self._raise_after is not None and emitted >= self._raise_after:
                    raise RuntimeError(
                        f"FakeFeed scripted failure after {emitted} events"
                    )
            if not self._loop_forever:
                return

    async def disconnect(self) -> None:
        self.disconnected = True

"""
Streaming core: the channel-aware EventBus bridging the async feed to the
synchronous engine loop.

The WebSocketFeed listener runs in its own thread (with its own asyncio loop)
and calls ``EventBus.put`` from that thread; the engine thread drains events
with ``EventBus.get_batch``. The bus is therefore a cross-thread hand-off and
uses a plain threading primitive (NOT asyncio.Queue, which is not thread-safe).

Backpressure policy is per channel, not global:

- Ordered, never-dropped channels (TRADE, closed KLINE, BOOK_SNAPSHOT,
  BOOK_DELTA): preserved in arrival order. Dropping a trade corrupts order
  flow; dropping a delta corrupts the reconstructed book.
- Coalescible state channels (TICK, partial KLINE): only the latest value per
  symbol matters, so a newer update replaces the pending one in place rather
  than growing the queue.

The producer is never blocked (blocking the socket read causes TCP backpressure
and the venue disconnects). Queue depth is exposed as a metric so a slow engine
(the per-tick indicator recompute cost) surfaces as growing depth instead of
silent lag.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import AsyncIterator, Callable, Iterable, List, Optional

from tradetropy.streaming._protocol import (
    FeedEvent,
    KLINE,
    TICK,
)

# Channels whose events may be coalesced to the latest value per symbol.
_DEFAULT_COALESCE = frozenset({TICK, KLINE})


class _Slot:
    """
    A mutable holder for one queued event.

    Coalescing replaces ``event`` in place while the slot still sits in the
    deque, so a burst of quote updates collapses to one slot without changing
    its queue position. ``consumed`` guards against coalescing into a slot that
    has already been drained.
    """

    __slots__ = ("event", "consumed")

    def __init__(self, event: FeedEvent):
        self.event = event
        self.consumed = False


class EventBus:
    """
    Thread-safe, channel-aware buffer between a feed thread and the engine.

    Args:
        coalesce_channels (Iterable[str]): Channels whose events are coalesced
            to the latest value per symbol. Defaults to TICK and partial KLINE.
            Closed klines are never coalesced regardless of this setting.

    Example:
        bus = EventBus()
        bus.put(TradeEvent('BTC/USDT', 1, 100.0, 1.0))   # feed thread
        events = bus.get_batch(timeout=0.1)               # engine thread
    """

    def __init__(self, coalesce_channels=_DEFAULT_COALESCE):
        self._deque: "deque[_Slot]" = deque()
        # Live coalesce slots: coalesce-key -> slot still pending in the deque.
        self._slots: dict = {}
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._coalesce = frozenset(coalesce_channels)
        self._max_depth: int = 0
        self._total_put: int = 0
        self._total_coalesced: int = 0

    def _coalesce_key(self, event: FeedEvent):
        """
        Compute the coalescing key for an event, or None if it must be ordered.

        Args:
            event (FeedEvent): The event to classify.

        Returns:
            tuple | None: A hashable key when the event is coalescible, else
            None (the event is kept in strict arrival order).
        """
        ch = event.channel
        if ch not in self._coalesce:
            return None
        if ch == KLINE:
            # Closed candles are final - never coalesce them away.
            if getattr(event, "is_closed", False):
                return None
            return (KLINE, event.symbol, event.interval_ms)
        return (ch, event.symbol)

    def put(self, event: FeedEvent) -> None:
        """
        Enqueue an event from the feed thread. Never blocks.

        Coalescible events (TICK, partial KLINE) replace any pending event of
        the same key in place; all other events are appended in arrival order.

        Args:
            event (FeedEvent): The normalized event to enqueue.
        """
        with self._not_empty:
            self._total_put += 1
            key = self._coalesce_key(event)
            if key is not None:
                slot = self._slots.get(key)
                if slot is not None and not slot.consumed:
                    slot.event = event
                    self._total_coalesced += 1
                    return
                slot = _Slot(event)
                self._slots[key] = slot
                self._deque.append(slot)
            else:
                self._deque.append(_Slot(event))

            depth = len(self._deque)
            if depth > self._max_depth:
                self._max_depth = depth
            self._not_empty.notify()

    def get_batch(
        self, timeout: float = 0.0, max_items: int = 100_000
    ) -> List[FeedEvent]:
        """
        Drain up to ``max_items`` events for the engine thread.

        Draining many events per call amortizes the engine's heavy per-event
        work (indicator recompute). If the bus is empty, waits up to ``timeout``
        seconds for the first event.

        Args:
            timeout (float): Max seconds to wait when the bus is empty
                (0.0 returns immediately).
            max_items (int): Max events to return in one batch.

        Returns:
            list[FeedEvent]: Events in arrival order (oldest first). Empty list
            when nothing arrived within ``timeout``.
        """
        with self._not_empty:
            if not self._deque and timeout > 0.0:
                self._not_empty.wait(timeout)

            out: List[FeedEvent] = []
            while self._deque and len(out) < max_items:
                slot = self._deque.popleft()
                slot.consumed = True
                key = self._coalesce_key(slot.event)
                if key is not None and self._slots.get(key) is slot:
                    del self._slots[key]
                out.append(slot.event)
            return out

    def get(self, timeout: float = 0.0) -> Optional[FeedEvent]:
        """
        Drain a single event, or None if none arrived within ``timeout``.

        Args:
            timeout (float): Max seconds to wait when the bus is empty.

        Returns:
            FeedEvent | None: The oldest event, or None.
        """
        batch = self.get_batch(timeout=timeout, max_items=1)
        return batch[0] if batch else None

    def clear(self) -> None:
        """Drop all pending events and reset coalesce slots."""
        with self._lock:
            self._deque.clear()
            self._slots.clear()

    @property
    def depth(self) -> int:
        """Current number of pending events (queue depth)."""
        with self._lock:
            return len(self._deque)

    @property
    def max_depth(self) -> int:
        """High-water mark of queue depth since creation (backpressure metric)."""
        return self._max_depth

    @property
    def stats(self) -> dict:
        """
        Snapshot of throughput / backpressure counters.

        Returns:
            dict: keys 'depth', 'max_depth', 'total_put', 'total_coalesced'.
        """
        with self._lock:
            return {
                "depth": len(self._deque),
                "max_depth": self._max_depth,
                "total_put": self._total_put,
                "total_coalesced": self._total_coalesced,
            }

    def __len__(self) -> int:
        return self.depth


class WebSocketFeed(ABC):
    """
    Abstract base for an asynchronous market-data feed.

    A feed connects to a venue, subscribes to symbols/channels, and yields
    normalized :class:`FeedEvent` objects from :meth:`listen`. It owns no
    threads and never touches engine state; the :class:`FeedRunner` drives it on
    a dedicated thread and forwards every event to an :class:`EventBus`.

    Implementations normalize raw venue messages into the protocol events
    (TradeEvent, TickEvent, KlineEvent, OrderbookSnapshot, OrderbookDelta).
    """

    @abstractmethod
    async def connect(self) -> None:
        """Open the underlying connection. Idempotent where possible."""
        raise NotImplementedError

    @abstractmethod
    async def subscribe(
        self, symbols: Iterable[str], channels: Iterable[str]
    ) -> None:
        """
        Subscribe to the given symbols and channels.

        Args:
            symbols (Iterable[str]): Venue-native symbols (e.g. 'BTC/USDT').
            channels (Iterable[str]): Channel identifiers (TRADE, TICK, KLINE,
                BOOK_SNAPSHOT, BOOK_DELTA).
        """
        raise NotImplementedError

    @abstractmethod
    def listen(self) -> AsyncIterator[FeedEvent]:
        """
        Yield normalized events as they arrive.

        Returns:
            AsyncIterator[FeedEvent]: An async iterator of normalized events.
            Implemented as an async generator; runs until the connection closes
            or the driving task is cancelled.
        """
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the underlying connection and release resources."""
        raise NotImplementedError


class FeedRunner:
    """
    Drive a :class:`WebSocketFeed` on a dedicated daemon thread.

    The runner creates its own asyncio event loop on the thread, calls
    ``connect`` then ``subscribe``, and forwards every event from ``listen`` to
    the :class:`EventBus`. This is the async-to-sync bridge: the engine thread
    only ever touches the bus, never the feed or its loop.

    Args:
        feed (WebSocketFeed): The feed to drive.
        bus (EventBus): Destination for events.
        symbols (Iterable[str]): Symbols to subscribe to.
        channels (Iterable[str]): Channels to subscribe to.
        on_error (Callable[[Exception], None] | None): Called (from the feed
            thread) if the feed raises. The error is also stored on ``error``.
        name (str): Thread name (for debugging).

    Example:
        runner = FeedRunner(feed, bus, ['BTC/USDT'], [TRADE])
        runner.start()
        ...                       # engine drains bus.get_batch()
        runner.stop()
    """

    def __init__(
        self,
        feed: WebSocketFeed,
        bus: EventBus,
        symbols: Iterable[str],
        channels: Iterable[str],
        *,
        on_error: Optional[Callable[[Exception], None]] = None,
        name: str = "FeedRunner",
    ):
        self._feed = feed
        self._bus = bus
        self._symbols = list(symbols)
        self._channels = list(channels)
        self._on_error = on_error
        self._name = name

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._started = threading.Event()
        self._error: Optional[Exception] = None

    def start(self, timeout: float = 5.0) -> None:
        """
        Launch the feed thread and wait until its loop is running.

        Args:
            timeout (float): Max seconds to wait for the thread to come up.

        Raises:
            RuntimeError: If already started.
        """
        if self._thread is not None:
            raise RuntimeError("FeedRunner already started")
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=self._name
        )
        self._thread.start()
        self._started.wait(timeout=timeout)

    def _run(self) -> None:
        """Thread entry point: own an event loop and run the driver task."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._task = self._loop.create_task(self._driver())
            self._started.set()
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - surfaced via on_error/error
            self._error = exc
            self._started.set()
            self._dispatch_error(exc)
        finally:
            if self._loop is not None:
                try:
                    self._loop.run_until_complete(
                        self._loop.shutdown_asyncgens()
                    )
                except Exception:  # noqa: BLE001
                    pass
                self._loop.close()

    async def _driver(self) -> None:
        """Connect, subscribe, then pump events into the bus until stopped."""
        try:
            await self._feed.connect()
            await self._feed.subscribe(self._symbols, self._channels)
            async for event in self._feed.listen():
                self._bus.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._dispatch_error(exc)
        finally:
            try:
                await self._feed.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def _dispatch_error(self, exc: Exception) -> None:
        """Invoke the error callback, swallowing any callback failure."""
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:  # noqa: BLE001
                pass

    def stop(self, timeout: float = 5.0) -> None:
        """
        Cancel the feed task and join the thread cleanly.

        Args:
            timeout (float): Max seconds to wait for the thread to finish.
        """
        if self._thread is None:
            return
        if (
            self._loop is not None
            and self._task is not None
            and not self._task.done()
        ):
            self._loop.call_soon_threadsafe(self._task.cancel)
        self._thread.join(timeout=timeout)

    def wait(self, timeout: Optional[float] = None) -> None:
        """
        Block until the feed finishes on its own (e.g. a scripted feed drains).

        Args:
            timeout (float | None): Max seconds to wait, or None for no limit.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """True while the feed thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Optional[Exception]:
        """The exception that stopped the feed, or None."""
        return self._error

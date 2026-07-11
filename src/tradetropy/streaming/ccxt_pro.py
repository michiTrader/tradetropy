"""
CCXTProFeed - the single WebSocketFeed implementation, built on CCXT Pro.

CCXT Pro's WebSocket layer is merged into the MIT-licensed ``ccxt`` package
(install with ``pip install "tradetropy[ccxt]"``), exposing ``watch_trades``,
``watch_ohlcv`` and ``watch_order_book`` across 100+ venues with unified message
shapes. One adapter therefore covers Binance, Bybit, OKX and the rest, and CCXT
handles order-book reconstruction and sequence-gap resync internally - so this
feed mostly emits full OrderbookSnapshot images rather than hand-managed deltas.

Each subscribed (symbol, channel) pair runs its own ``watch_*`` loop; the loops
are multiplexed into a single async event stream consumed by FeedRunner.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable, List, Optional

from tradetropy.exceptions import ConfigError, ConnectionError, TradingError
from tradetropy.streaming._protocol import (
    BOOK_DELTA,
    BOOK_SNAPSHOT,
    FILL,
    FillEvent,
    KLINE,
    ORDER,
    OrderEvent,
    TICK,
    TRADE,
    FeedEvent,
    KlineEvent,
    OrderbookSnapshot,
    TickEvent,
    TradeEvent,
)
from tradetropy.streaming.base import WebSocketFeed


def _import_ccxt_pro():
    """
    Lazily import the CCXT Pro (async/WebSocket) module.

    Returns:
        The ``ccxt.pro`` module.

    Raises:
        ImportError: If ccxt (with pro support) is not installed.
    """
    try:
        import ccxt.pro as ccxtpro  # type: ignore
        return ccxtpro
    except ImportError:
        pass
    try:
        import ccxtpro  # type: ignore  # legacy standalone package
        return ccxtpro
    except ImportError as exc:
        raise ImportError(
            "CCXT Pro (WebSocket) is not available. Install with "
            "pip install \"tradetropy[ccxt]\" (ccxt>=4 includes ccxt.pro)."
        ) from exc


def _to_int_id(raw) -> int:
    """Coerce a venue trade id to int (digit string -> int, else stable hash)."""
    if raw is None:
        return -1
    s = str(raw)
    if s.isdigit():
        return int(s)
    return abs(hash(s)) % 1_000_000_000


def _side_to_int(side) -> int:
    """Map ccxt trade side ('buy'/'sell') to +1 / -1 / 0."""
    s = str(side).lower()
    if s == "buy":
        return 1
    if s == "sell":
        return -1
    return 0


class CCXTProFeed(WebSocketFeed):
    """
    WebSocketFeed backed by a CCXT Pro exchange.

    Args:
        exchange: A CCXT Pro exchange id (str, e.g. 'binance'), an already
            constructed ccxt.pro instance, or a compatible mock.
        config (dict | None): Exchange constructor config (apiKey, options, ...)
            used only when ``exchange`` is a string.
        timeframe_ms (int): Candle interval for the KLINE channel (default 1m).
        book_limit (int): Number of order-book levels to request for the book
            channels.
        sandbox (bool): Enable the venue's testnet/sandbox mode (applied only
            when ``exchange`` is a string and the venue supports it).
        demo (bool): Enable the venue's demo-trading mode (Bybit demo account,
            via ccxt.enable_demo_trading). Applied only when ``exchange`` is a
            string and the venue supports it. Mutually exclusive with sandbox.

    Notes:
        - Use a public (unauthenticated) instance for market-data streaming.
        - watch_order_book returns a fully reconstructed book; this feed emits
          it as an OrderbookSnapshot. A `stale`/resync flag is handled later by
          the OrderbookProxy for the gaps CCXT cannot paper over.
    """

    def __init__(
        self,
        exchange,
        config: Optional[dict] = None,
        *,
        timeframe_ms: int = 60_000,
        book_limit: int = 20,
        sandbox: bool = False,
        demo: bool = False,
    ):
        self._exchange_arg = exchange
        self._config = config
        self._timeframe_ms = int(timeframe_ms)
        self._book_limit = int(book_limit)
        self._sandbox = bool(sandbox)
        self._demo = bool(demo)

        self._ex = None
        self._symbols: List[str] = []
        self._channels: List[str] = []
        # Track the last emitted closed-candle ts per symbol to avoid
        # re-emitting closed klines (watch_ohlcv returns the cached list).
        self._last_closed_kline: dict = {}

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Resolve / construct the CCXT Pro exchange instance."""
        self._ex = self._resolve_exchange()

    def _resolve_exchange(self):
        ex = self._exchange_arg
        if isinstance(ex, str):
            pro = _import_ccxt_pro()
            if not hasattr(pro, ex):
                raise ConnectionError(
                    f"Unknown CCXT Pro exchange: {ex!r}. "
                    f"Check the id in ccxt.pro."
                )
            instance = getattr(pro, ex)(self._config or {})
            if self._sandbox and hasattr(instance, "set_sandbox_mode"):
                instance.set_sandbox_mode(True)
            elif self._demo and hasattr(instance, "enable_demo_trading"):
                instance.enable_demo_trading(True)
            return instance
        return ex

    async def subscribe(
        self, symbols: Iterable[str], channels: Iterable[str]
    ) -> None:
        """Record the symbols/channels to watch (loops start in listen())."""
        self._symbols = list(symbols)
        self._channels = list(channels)
        unknown = [
            c
            for c in self._channels
            if c not in (TRADE, TICK, KLINE, BOOK_SNAPSHOT, BOOK_DELTA, ORDER, FILL)
        ]
        if unknown:
            raise ConfigError(f"Unsupported channels for CCXTProFeed: {unknown}")

    async def disconnect(self) -> None:
        """Close the exchange connection if it exposes an async close()."""
        ex = self._ex
        if ex is not None and hasattr(ex, "close"):
            res = ex.close()
            if asyncio.iscoroutine(res):
                await res

    # ── event stream ────────────────────────────────────────────────────────

    async def listen(self) -> AsyncIterator[FeedEvent]:
        """
        Run one watch loop per (symbol, channel) and yield normalized events.

        The per-pair loops push event batches into an internal asyncio.Queue;
        this generator forwards them in arrival order. A watcher exception is
        forwarded and re-raised here so FeedRunner can surface it.
        """
        queue: "asyncio.Queue" = asyncio.Queue()
        tasks: List[asyncio.Task] = [
            asyncio.ensure_future(self._watch(sym, ch, queue))
            for sym in self._symbols
            for ch in self._channels
        ]
        try:
            while True:
                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                for ev in item:
                    yield ev
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch(self, symbol: str, channel: str, queue: "asyncio.Queue") -> None:
        """One venue watch loop for a single (symbol, channel) pair."""
        try:
            method = self._watch_method(channel)
            while True:
                events = await method(symbol, channel)
                if events:
                    await queue.put(events)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - forwarded to listen()
            await queue.put(exc)

    def _watch_method(self, channel: str):
        """Bind a channel to its async normalizer, checking venue capability."""
        if channel == TRADE:
            self._require("watch_trades")
            return self._watch_trades
        if channel == TICK:
            self._require("watch_ticker")
            return self._watch_ticker
        if channel == KLINE:
            self._require("watch_ohlcv")
            return self._watch_ohlcv
        if channel in (BOOK_SNAPSHOT, BOOK_DELTA):
            self._require("watch_order_book")
            return self._watch_order_book
        if channel == ORDER:
            self._require("watch_orders")
            return self._watch_orders
        if channel == FILL:
            self._require("watch_my_trades")
            return self._watch_my_trades
        raise ConfigError(f"Unsupported channel: {channel!r}")

    def _require(self, attr: str) -> None:
        if not hasattr(self._ex, attr):
            raise TradingError(
                f"Exchange {getattr(self._ex, 'id', '?')!r} does not provide "
                f"{attr}() in CCXT Pro."
            )

    # ── normalizers ───────────────────────────────────────────────────────────

    async def _watch_trades(self, symbol: str, _channel: str) -> List[FeedEvent]:
        trades = await self._ex.watch_trades(symbol)
        return [self._trade_event(symbol, t) for t in (trades or [])]

    async def _watch_ticker(self, symbol: str, _channel: str) -> List[FeedEvent]:
        t = await self._ex.watch_ticker(symbol)
        return [self._tick_event(symbol, t)] if t else []

    async def _watch_ohlcv(self, symbol: str, _channel: str) -> List[FeedEvent]:
        tf = self._timeframe_str()
        ohlcv = await self._ex.watch_ohlcv(symbol, tf)
        return self._kline_events(symbol, ohlcv or [])

    async def _watch_order_book(self, symbol: str, _channel: str) -> List[FeedEvent]:
        ob = await self._ex.watch_order_book(symbol, self._book_limit)
        return [self._book_snapshot(symbol, ob)] if ob else []

    async def _watch_orders(self, symbol: str, _channel: str) -> List[FeedEvent]:
        orders = await self._ex.watch_orders(symbol)
        return [self._order_event(o) for o in (orders or [])]

    async def _watch_my_trades(self, symbol: str, _channel: str) -> List[FeedEvent]:
        trades = await self._ex.watch_my_trades(symbol)
        return [self._fill_event(t) for t in (trades or [])]

    def _trade_event(self, symbol: str, t: dict) -> TradeEvent:
        return TradeEvent(
            symbol=symbol,
            ts=int(t.get("timestamp") or 0),
            price=float(t.get("price") or 0.0),
            volume=float(t.get("amount") or 0.0),
            side=_side_to_int(t.get("side")),
            trade_id=_to_int_id(t.get("id")),
            is_maker=(t.get("takerOrMaker") == "maker"),
        )

    def _tick_event(self, symbol: str, t: dict) -> TickEvent:
        bid = float(t.get("bid") or 0.0)
        ask = float(t.get("ask") or 0.0)
        last = t.get("last")
        if last is not None:
            price = float(last)
        elif bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        else:
            price = 0.0
        return TickEvent(
            symbol=symbol,
            ts=int(t.get("timestamp") or 0),
            bid=bid,
            ask=ask,
            price=price,
            volume=float(t.get("baseVolume") or 0.0),
        )

    def _kline_events(self, symbol: str, ohlcv: list) -> List[FeedEvent]:
        """
        Normalize a watch_ohlcv batch.

        CCXT returns the cached candle list each update; the last candle is the
        forming (partial) one, earlier candles are closed. Closed candles are
        emitted once (tracked per symbol); the partial is always emitted (the
        EventBus coalesces it to the latest).
        """
        out: List[FeedEvent] = []
        n = len(ohlcv)
        if n == 0:
            return out
        last_closed_seen = self._last_closed_kline.get(symbol, -1)
        for i, c in enumerate(ohlcv):
            ts = int(c[0])
            is_closed = i < n - 1
            if is_closed:
                if ts <= last_closed_seen:
                    continue
                self._last_closed_kline[symbol] = ts
            out.append(
                KlineEvent(
                    symbol=symbol,
                    ts=ts,
                    interval_ms=self._timeframe_ms,
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]) if len(c) > 5 else 0.0,
                    is_closed=is_closed,
                )
            )
        return out

    def _order_event(self, o: dict) -> OrderEvent:
        return OrderEvent(
            symbol=str(o.get("symbol", "")),
            ts=int(o.get("timestamp") or 0),
            order_id=_to_int_id(o.get("id")),
            status=str(o.get("status") or "open"),
            side=_side_to_int(o.get("side")),
            price=float(o.get("price") or 0.0),
            volume=float(o.get("amount") or 0.0),
            filled=float(o.get("filled") or 0.0),
            order_type=str(o.get("type") or ""),
            client_id=str(o.get("clientOrderId") or ""),
        )

    def _fill_event(self, t: dict) -> FillEvent:
        fee = t.get("fee") or {}
        return FillEvent(
            symbol=str(t.get("symbol", "")),
            ts=int(t.get("timestamp") or 0),
            order_id=_to_int_id(t.get("order")),
            trade_id=_to_int_id(t.get("id")),
            side=_side_to_int(t.get("side")),
            price=float(t.get("price") or 0.0),
            volume=float(t.get("amount") or 0.0),
            fee=float(fee.get("cost") or 0.0),
            is_maker=(t.get("takerOrMaker") == "maker"),
        )

    def _book_snapshot(self, symbol: str, ob: dict) -> OrderbookSnapshot:
        k = self._book_limit
        bids = tuple(
            (float(p), float(q)) for p, q in (ob.get("bids") or [])[:k]
        )
        asks = tuple(
            (float(p), float(q)) for p, q in (ob.get("asks") or [])[:k]
        )
        nonce = ob.get("nonce")
        return OrderbookSnapshot(
            symbol=symbol,
            ts=int(ob.get("timestamp") or 0),
            bids=bids,
            asks=asks,
            last_id=int(nonce) if nonce is not None else -1,
        )

    def _timeframe_str(self) -> str:
        """Map the configured interval (ms) to a CCXT timeframe string."""
        from tradetropy.connectors.ccxt import _MS_TO_CCXT_TF

        tf = _MS_TO_CCXT_TF.get(self._timeframe_ms)
        if tf is None:
            raise ConfigError(
                f"interval_ms={self._timeframe_ms} has no CCXT timeframe "
                f"equivalent. Supported: {sorted(_MS_TO_CCXT_TF.keys())}"
            )
        return tf

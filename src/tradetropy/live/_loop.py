"""
Automatic feed loop for LiveEngine.
Contains _loop_tick, _loop_kline and flushing/dispatcher helpers.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from tradetropy.core.constants import _TICK_COL
from tradetropy.io.io import _append_ticks, _append_klines, _append_book, _append_mbo
from tradetropy.models.strategy import StopEngine


def _run_guarded(self):
    """
    Wrapper that catches KeyboardInterrupt and loop exceptions.

    Ensures strategy.on_crash() is called for non-keyboard exceptions
    and strategy.on_stop() is called on graceful shutdown.
    """
    try:
        self._loop()
    except KeyboardInterrupt:
        self.stop()
        self.strategy.on_stop()
    except StopEngine as exc:
        warnings.warn(
            f"StopEngine in Live: '{exc.reason}'. ",
            stacklevel=2,
        )
    except Exception as exc:
        suppress = self.strategy.on_crash(exc)
        if not suppress:
            raise


def _flush_tick_proxy(self, proxy) -> None:
    """
    Flush tick buffer to disk.

    Writes pending ticks to HDF5 file and clears buffer. On the first flush it
    also persists the symbol as table metadata so read_ticks() can recover it
    without an explicit symbol argument.

    Args:
        proxy: TickProxy with recording enabled.
    """
    cfg = proxy._record_config
    if not cfg._buffer:
        return
    arr = np.array(cfg._buffer, dtype=np.float64)
    metadata = None
    if not cfg._meta_written:
        metadata = {"tradetropy_symbol": proxy.symbol}
    _append_ticks(arr, cfg.path, metadata)
    cfg._meta_written = True
    cfg._buffer.clear()


def _flush_ohlc_proxy(self, proxy) -> None:
    """
    Flush kline buffer to disk.

    Writes pending klines to HDF5 file and clears buffer. On the first flush it
    also persists the symbol and interval as table metadata so read_klines() can
    recover the symbol and timeframe without explicit arguments.

    Args:
        proxy: OhlcProxy with recording enabled.
    """
    cfg = proxy._record_config
    if not cfg._buffer:
        return
    arr = np.array(cfg._buffer, dtype=np.float64)
    metadata = None
    if not cfg._meta_written:
        metadata = {
            "tradetropy_symbol": proxy.symbol,
            "tradetropy_interval_ms": int(proxy.interval_ms),
        }
    _append_klines(arr, cfg.path, metadata)
    cfg._meta_written = True
    cfg._buffer.clear()


def _flush_book_proxy(self, proxy) -> None:
    """
    Flush recorded order-book rows to disk.

    Writes pending book rows to HDF5 (with the proxy's level count) and clears
    the buffer. On the first flush it also persists the symbol as table metadata
    so read_book() can recover it. Compatible with io.read_book() /
    ReplayFeed.from_records().

    Args:
        proxy: OrderbookProxy with recording enabled.
    """
    cfg = proxy._record_config
    if not cfg._buffer:
        return
    arr = np.array(cfg._buffer, dtype=np.float64)
    metadata = None
    if not cfg._meta_written:
        metadata = {"tradetropy_symbol": proxy.symbol}
    _append_book(arr, cfg.path, proxy.depth, metadata)
    cfg._meta_written = True
    cfg._buffer.clear()


def _flush_mbo_proxy(self, proxy) -> None:
    """
    Flush recorded MBO event rows to disk (read back with io.read_mbo()).

    On the first flush it also persists the symbol as table metadata so
    read_mbo() can recover it without an explicit symbol argument.

    Args:
        proxy: MboProxy with recording enabled.
    """
    cfg = proxy._record_config
    if not cfg._buffer:
        return
    arr = np.array(cfg._buffer, dtype=np.float64)
    metadata = None
    if not cfg._meta_written:
        metadata = {"tradetropy_symbol": proxy.symbol}
    _append_mbo(arr, cfg.path, metadata)
    cfg._meta_written = True
    cfg._buffer.clear()


def _loop(self):
    """
    Dispatch main loop based on streaming capability and feed_type.

    A streaming-capable session (sesh.supports_streaming) drives the unified
    event-driven loop (_loop_streaming). Otherwise the existing polling loops
    run unchanged (also used by ReplayEngine, which overrides them).
    """
    simbolos = self._simbolos_del_loop()
    if self._sesh is not None and getattr(self._sesh, "supports_streaming", False):
        self._loop_streaming(simbolos)
    elif self._feed_type == "tick":
        self._loop_tick(simbolos)
    else:
        self._loop_kline(simbolos)


def _on_feed_error(self, exc: Exception) -> None:
    """
    Feed-thread error callback: record it and ask the loop to stop.

    Args:
        exc (Exception): The exception raised inside the feed.
    """
    self._feed_error = exc
    self._stop_event.set()


def _loop_streaming(self, symbols: list):
    """
    Unified event-driven loop: drain the EventBus and dispatch each event.

    Builds the session's WebSocketFeed, runs it on a FeedRunner thread, and
    drains batches from the EventBus on this (engine) thread, routing every
    event through _process_event. Draining in batches amortizes the heavy
    per-event indicator work. Exits when stopped, or when a finite feed has
    drained completely (e.g. a scripted/replay feed); a feed-thread error is
    re-raised so strategy.on_crash() runs.

    Args:
        symbols (list): Symbols to subscribe to.
    """
    from tradetropy.streaming import EventBus, FeedRunner, BOOK_SNAPSHOT, FILL, KLINE, MBO, ORDER, TRADE

    self._feed_error = None

    book_proxies = getattr(self.strategy, "_book_proxies", [])
    mbo_proxies = getattr(self.strategy, "_mbo_proxies", [])
    book_limit = max((bp.depth for bp in book_proxies), default=20)

    if self._feed_type == "tick":
        channels = [TRADE]
        feed_kwargs = {"book_limit": book_limit}
        # Parity with the polling tick loop's start-up.
        self._topup_ohlc_rings()
        self._resync_chart_sources()
    else:
        interval = next(
            (op.interval_ms for op in self.strategy._ohlc_proxies), 60_000
        )
        channels = [KLINE]
        feed_kwargs = {"timeframe_ms": interval, "book_limit": book_limit}

    if book_proxies:
        channels = channels + [BOOK_SNAPSHOT]

    if mbo_proxies:
        channels = channels + [MBO]

    # Private user-data channels (orders/fills) when the session is
    # authenticated and the feed delivers them - replaces sync() polling.
    if getattr(self._sesh, "supports_user_stream", False):
        channels = channels + [ORDER, FILL]

    feed = self._sesh.create_feed(**feed_kwargs)

    bus = EventBus()
    runner = FeedRunner(
        feed, bus, symbols, channels, on_error=self._on_feed_error,
        name="LiveEngine-feed",
    )
    self._event_bus = bus
    self._feed_runner = runner
    runner.start()

    try:
        while not self._stop_event.is_set():
            batch = bus.get_batch(timeout=0.1, max_items=10_000)
            for event in batch:
                if self._stop_event.is_set():
                    break
                self._process_event(event)
            # A finite feed (scripted/replay) drains then ends; exit once the
            # feed thread has stopped and the bus is empty. A live feed keeps
            # the runner alive, so the loop keeps draining.
            if not batch and not runner.is_alive() and bus.depth == 0:
                break
    finally:
        runner.stop()

    if self._feed_error is not None:
        raise self._feed_error


def _simbolos_del_loop(self) -> list:
    """
    Collect all subscribed symbols.

    Gathers symbols from tick proxies, OHLC proxies and footprint proxies.

    Returns:
        list: List of unique trading symbols.
    """
    syms = set()
    for tp in self.strategy._tick_proxies:
        syms.add(tp.symbol)
    for op in self.strategy._ohlc_proxies:
        syms.add(op.symbol)
    for fp in self.strategy._fp_proxies:
        syms.add(fp.symbol)
    for bp in getattr(self.strategy, "_book_proxies", []):
        syms.add(bp.symbol)
    for mp in getattr(self.strategy, "_mbo_proxies", []):
        syms.add(mp.symbol)
    if not syms:
        syms = set(self._tick_rings.keys())
    return list(syms)


def _loop_tick(self, simbolos: list):
    """
    Main loop in tick mode.

    Repeatedly fetches last tick from session for each symbol and
    processes it via on_tick(). Sleeps between cycles.

    Args:
        simbolos: List of trading symbols to monitor.
    """
    ultimo_ts = {sym: -1 for sym in simbolos}

    self._topup_ohlc_rings()
    self._resync_chart_sources()

    while not self._stop_event.is_set():
        for sym in simbolos:
            if self._stop_event.is_set():
                break

            try:
                tick = self.sesh._fetch_last_tick(sym)
                ts = int(tick[_TICK_COL["ts"]])
            except Exception as e:
                import traceback
                warnings.warn(
                    f"LiveEngine fetch error ({sym}): {e}\n"
                    f"{traceback.format_exc()}"
                )
                continue

            if ts > ultimo_ts[sym]:
                ultimo_ts[sym] = ts
                self.on_tick(sym, tick)

        time.sleep(self._poll_interval)


def _loop_kline(self, symbols: list):
    """
    Main loop in kline mode.

    Repeatedly fetches last kline from session for each symbol and
    processes it via on_kline(). Sleeps between cycles.

    Args:
        symbols: List of trading symbols to monitor.
    """
    intervalos = {}
    for op in self.strategy._ohlc_proxies:
        if op.symbol not in intervalos:
            intervalos[op.symbol] = op.interval_ms

    while not self._stop_event.is_set():
        for sym in symbols:
            if self._stop_event.is_set():
                break
            intervalo = intervalos.get(sym)
            if intervalo is None:
                continue
            try:
                kline = self.sesh._fetch_last_kline(sym, intervalo)
                self.on_kline(sym, kline)
            except Exception as e:
                import traceback
                warnings.warn(
                    f"LiveEngine kline loop error ({sym}): {e}\n"
                    f"{traceback.format_exc()}"
                )
                time.sleep(0.1)

        if not self._stop_event.is_set():
            time.sleep(1.0)

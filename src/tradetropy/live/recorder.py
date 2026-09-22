"""
High-level data recorder.

`Recorder` captures one or more symbols' live market data (ticks, klines or
the L2 order book) to disk without writing a Strategy. The base binary format
is NumPy `.npz` (no extra needed, works on Termux); pass a `.h5`/`.hdf5` path to
record HDF5 instead (needs the optional `tradetropy[hdf5]` extra). It drives an
internal minimal strategy under a single `LiveEngine`, so it reuses the exact
same transport and persistence the engine already implements:

- WebSocket streaming is used automatically when the session supports it
  (`sesh.supports_streaming`, e.g. SeshCCXTLive over CCXT Pro); otherwise the
  engine falls back to REST polling (`poll_interval`).
- The recording itself (buffered append, periodic flush, final flush on stop)
  is the same `record=` mechanism used by `subscribe_ticks` / `subscribe_ohlc`
  / `subscribe_orderbook`. For `.npz` (not an appendable container) each flush
  appends to a raw sidecar and the file is consolidated into the final `.npz`
  when the recorder stops; the HDF5 path appends natively.

Recording is a live-only operation: a simulated (backtest) session records
nothing, so `Recorder` rejects one up front.

There is no playback-speed control here: recording happens in real time, as
events arrive. To replay a recorded file at an arbitrary speed use
`ReplayEngine` (its controller exposes the speed control).

Note: in a 'kline' stream the file may begin with one boundary warmup candle
(the last historical candle, recorded when the first live candle closes it),
followed by the live-streamed candles. Tick and order-book recordings contain
only live-streamed events.

Streams are declared one at a time with `add_stream(stream, symbol, path, ...)`,
which returns the recorder for chaining. The first argument is the stream:

- `add_stream('tick', symbol, path)`
- `add_stream('kline', symbol, path, timeframe=...)`       (timeframe required)
- `add_stream('orderbook', symbol, path, depth=...)`       (depth optional)

Example (single symbol):
    from tradetropy import Recorder
    from tradetropy.connectors.ccxt import SeshCCXTLive

    sesh = SeshCCXTLive('binance')                     # public data, no keys

    # Record trades until Ctrl+C (final flush guaranteed by the context manager):
    with Recorder(sesh) as rec:
        rec.add_stream('tick', 'BTC/USDT', 'rec/btc_ticks.npz')
        rec.run()

    # Record the L2 book for one hour, then auto-stop (chained):
    Recorder(sesh).add_stream(
        'orderbook', 'BTC/USDT', 'rec/btc_book.npz', depth=20
    ).run(duration=3600)

    # Record 1m candles in the background (notebook):
    rec = Recorder(sesh)
    rec.add_stream('kline', 'BTC/USDT', 'rec/btc_1m.npz', timeframe='1m')
    rec.run(blocking=False)
    # ... later ...
    rec.stop()

Example (multiple symbols and/or streams, one shared session/connection):
    # A single WebSocket connection records several streams at once. Mixing
    # streams is allowed: each add_stream() call is validated on its own.
    rec = Recorder(sesh)
    rec.add_stream('tick', 'BTC/USDT', 'rec/btc_ticks.npz')
    rec.add_stream('kline', 'ETH/USDT', 'rec/eth_1m.npz', timeframe='1m')
    rec.add_stream('orderbook', 'BTC/USDT', 'rec/btc_book.npz', depth=20)
    rec.run(duration=3600)

Recording several symbols this way reuses one `Sesh` / one `LiveEngine`, i.e.
one WebSocket connection, instead of opening one connection per symbol - the
same pattern `LivePool` groups use to share a feed. There is no manual async
needed either: the streaming layer already serializes events from every
symbol through a single background feed thread into one engine thread (see
`.kiro/steering/streaming.md`), so multi-symbol recording is naturally
ordered and thread-safe without extra code here.
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

from tradetropy.exceptions import ConfigError
from tradetropy.models.strategy import Strategy

if TYPE_CHECKING:
    from tradetropy.session.base import Sesh

#: Valid stream identifiers (first argument of add_stream).
_STREAMS = ("tick", "kline", "orderbook")

#: Default in-memory window sizes for the internal recorder proxies.
_TICK_WINDOW = 1000
_OHLC_WINDOW = 300
_BOOK_WINDOW = 5000


class _RecorderStrategy(Strategy):
    """
    Internal minimal strategy that subscribes every recorded stream.

    Not part of the public API. Its ``on_data()`` is a no-op: the strategy
    exists solely so the engine's ``record=`` path writes the normalized
    stream(s) to disk. ``warmup = 0`` whenever any stream is tick/orderbook, so
    recording starts on the first live event.

    Recording configuration is injected as an instance attribute (a list of
    spec dicts) before the engine runs ``init()`` (during ``prepare()``).
    """

    # Injected by Recorder before the engine prepares.
    _rec_specs: "list[dict]" = []

    def init(self):
        """
        Subscribe every recorded stream, isolating runtime failures per stream.

        A symbol that already has a tick subscription (recorded, from an
        explicit ``'tick'`` stream, or a dummy one from another ``'orderbook'``
        stream of the same symbol) is not subscribed again, so two streams
        sharing a symbol never duplicate the tick feed.

        Stream arguments are validated up front in ``add_stream()``, so by the
        time this runs every spec is structurally valid. A spec that still
        fails to subscribe at the session level (e.g. the venue rejects the
        symbol) is skipped with a warning; the remaining streams still
        subscribe normally.
        """
        symbols_with_tick: set[str] = set()

        for i, spec in enumerate(self._rec_specs):
            try:
                self._subscribe_one(spec, symbols_with_tick)
            except Exception as exc:
                warnings.warn(
                    f"Recorder: stream #{i} (symbol={spec.get('symbol')!r}, "
                    f"stream={spec.get('stream')!r}) failed to subscribe and "
                    f"will not be recorded: {exc!r}",
                    stacklevel=2,
                )

    def _subscribe_one(self, spec: dict, symbols_with_tick: "set[str]") -> None:
        """Subscribe a single (already-validated) stream, recording it via ``record=``."""
        sym = spec["symbol"]
        path = spec["path"]
        flush = spec["flush_every"]
        stream = spec["stream"]

        if stream == "tick":
            self.subscribe_ticks(
                sym, window_size=_TICK_WINDOW,
                record=path, record_flush_every=flush,
            )
            symbols_with_tick.add(sym)
        elif stream == "kline":
            self.subscribe_ohlc(
                sym, timeframe=spec["timeframe"], window_size=_OHLC_WINDOW,
                record=path, record_flush_every=flush,
            )
        elif stream == "orderbook":
            # The tick feed always subscribes the TRADE channel, and the engine
            # raises if a streamed trade has no tick ring. A non-recorded tick
            # proxy gives those trades a home; only the book proxy is recorded.
            # Skip it if this symbol already has one (explicit or dummy) so two
            # streams sharing a symbol never subscribe ticks twice.
            if sym not in symbols_with_tick:
                self.subscribe_ticks(sym, window_size=_TICK_WINDOW)
                symbols_with_tick.add(sym)
            self.subscribe_orderbook(
                sym, depth=spec["depth"], window_size=_BOOK_WINDOW,
                record=path, record_flush_every=flush,
            )

    def on_data(self):
        """No-op: recording happens in the engine's record= path."""
        pass


class Recorder:
    """
    Record one or more symbols' live tick / kline / order-book stream(s) to
    disk (NumPy ``.npz`` by default; ``.h5``/``.hdf5`` with the optional
    ``tradetropy[hdf5]`` extra).

    Drives an internal minimal strategy under a single LiveEngine, using
    WebSocket streaming when the session supports it and REST polling
    otherwise. Reuses the engine's existing record= persistence, so recorded
    files round-trip through tradetropy.io (read_ticks / read_klines /
    read_book).

    Streams are declared with `add_stream(stream, symbol, path, ...)`, which
    returns the recorder for chaining. Recording several symbols (optionally
    mixing streams) with one Recorder reuses a single Sesh / LiveEngine, i.e.
    one WebSocket connection, instead of opening one connection per symbol.

    Example:
        rec = Recorder(sesh)
        rec.add_stream('tick', 'BTC/USDT', 'btc.npz')
        rec.add_stream('kline', 'ETH/USDT', 'eth.npz', timeframe='1m')
        rec.run(duration=60)

        # Chained single-symbol one-liner:
        Recorder(sesh).add_stream('tick', 'BTC/USDT', 'btc.npz').run(duration=60)
    """

    def __init__(
        self,
        sesh: "Sesh",
        *,
        depth: int = 20,
        poll_interval: float = 1.0,
        flush_every: int = 500,
    ):
        """
        Build an empty recorder; declare what to record with `add_stream()`.

        Args:
            sesh: A live session (e.g. ``SeshCCXTLive``). Must support live
                data (``supports_streaming`` and/or REST polling). A simulated
                (backtest) session is rejected: ``ConfigError`` is raised up
                front, since recording is a live-only operation. All streams
                share this single session, i.e. a single WebSocket connection
                when the session supports streaming.
            depth (int): Default number of order-book levels (K) per side for
                ``'orderbook'`` streams added without their own ``depth``.
                Default ``20``.
            poll_interval (float): REST polling interval in seconds, used only
                as a fallback when ``sesh.supports_streaming`` is False (no
                WebSocket support). Ignored when streaming is available.
                Default ``1.0``.
            flush_every (int): Default number of recorded events between disk
                flushes, for streams added without their own ``flush_every``.
                Lower values reduce data loss on an abrupt stop at the cost of
                more frequent disk writes. Default ``500``.

        Raises:
            ConfigError: If ``sesh`` is a simulated (backtest) session.

        Example:
            sesh = SeshCCXTLive('binance')
            rec = Recorder(sesh)
            rec.add_stream('kline', 'BTC/USDT', 'btc_1m.npz', timeframe='1m')
            rec.run(duration=3600)
        """
        from tradetropy.session.base import SeshSimulatorBase
        if isinstance(sesh, SeshSimulatorBase):
            raise ConfigError(
                "Recorder needs a live session; a simulated (backtest) session "
                "records nothing. Use a live session such as SeshCCXTLive."
            )

        self._sesh = sesh
        self._depth = depth
        self._poll_interval = poll_interval
        self._flush_every = flush_every

        self._specs: "list[dict]" = []
        self._timer: Optional[threading.Timer] = None
        self._engine = None      # built lazily on run(), once streams are known
        self._started = False

    # -- Building ---------------------------------------------------------------

    def add_stream(
        self,
        stream: Literal["tick", "kline", "orderbook"],
        symbol: str,
        path: "str | Path",
        *,
        timeframe: "str | int | None" = None,
        depth: "int | None" = None,
        flush_every: "int | None" = None,
    ) -> "Recorder":
        """
        Declare one stream to record, validating its arguments up front.

        Args:
            stream (str): One of ``'tick'`` (trade prints), ``'kline'`` (OHLC
                candles) or ``'orderbook'`` (the L2 depth book).
            symbol (str): Trading symbol in the venue's native format (e.g.
                ``'BTC/USDT'`` for a CCXT session). Must be non-empty.
            path (str | Path): Destination file (``.npz`` by default, or
                ``.h5``/``.hdf5`` with the ``hdf5`` extra), written via the
                engine's append-on-event ``record=`` mechanism (buffered,
                flushed periodically and on ``stop()``). Must be non-empty.
            timeframe (str | int | None): Candle interval, required for
                ``stream='kline'`` (e.g. ``'1m'``, ``'5m'``, or milliseconds as
                an int like ``60_000``). Ignored for ``'tick'`` and
                ``'orderbook'``.
            depth (int | None): Number of order-book levels (K) per side for
                ``stream='orderbook'``. When ``None`` the recorder's
                instance-level ``depth`` default is used. Ignored for other
                streams.
            flush_every (int | None): Events between disk flushes for this
                stream. When ``None`` the recorder's instance-level
                ``flush_every`` default is used.

        Returns:
            Recorder: self, so calls can be chained.

        Raises:
            ConfigError: If the recorder has already started; if ``stream`` is
                not one of ``'tick'``, ``'kline'`` or ``'orderbook'``; if
                ``symbol`` or ``path`` is empty; if ``stream='kline'`` and no
                ``timeframe`` is given; or if an ``'orderbook'`` ``depth`` is
                not an int.

        Example:
            rec.add_stream('tick', 'BTC/USDT', 'btc.npz')
            rec.add_stream('kline', 'ETH/USDT', 'eth.npz', timeframe='1m')
            rec.add_stream('orderbook', 'BTC/USDT', 'book.npz', depth=20)
        """
        if self._started:
            raise ConfigError(
                "Recorder.add_stream() cannot be called after run(); declare "
                "every stream before starting the recording."
            )
        if stream not in _STREAMS:
            raise ConfigError(
                f"Recorder.add_stream(): stream must be one of {_STREAMS}, "
                f"got {stream!r}."
            )
        if not symbol:
            raise ConfigError("Recorder.add_stream(): symbol must be non-empty.")
        if not path:
            raise ConfigError("Recorder.add_stream(): path must be non-empty.")

        spec_depth = self._depth
        if stream == "kline":
            if timeframe is None or timeframe == "":
                raise ConfigError(
                    f"Recorder.add_stream({symbol!r}): stream='kline' requires "
                    f"a non-empty timeframe (e.g. timeframe='1m' or 60_000)."
                )
        elif stream == "orderbook":
            if depth is not None:
                if not isinstance(depth, int) or isinstance(depth, bool):
                    raise ConfigError(
                        f"Recorder.add_stream({symbol!r}): orderbook depth must "
                        f"be an int, got {depth!r}."
                    )
                spec_depth = depth

        self._specs.append({
            "stream": stream,
            "symbol": symbol,
            "path": path,
            "timeframe": timeframe,
            "depth": spec_depth,
            "flush_every": self._flush_every if flush_every is None else flush_every,
        })
        return self

    def _build_engine(self) -> None:
        """
        Build the internal strategy and LiveEngine from the declared streams.

        Deferred until run() because the engine type (by_ticks vs by_klines)
        and the strategy warmup depend on which streams were added.
        """
        strat = _RecorderStrategy()
        strat._rec_specs = self._specs
        # Tick recording is gated by warmup readiness, so warmup=0 ensures every
        # live tick is recorded from the start (tick + orderbook streams). Kline
        # recording is not readiness-gated, and warmup=0 would make the engine
        # request zero warmup candles (an error); leave it on auto when every
        # stream is a kline.
        if any(s["stream"] in ("tick", "orderbook") for s in self._specs):
            strat.warmup = 0
        self._strategy = strat

        from tradetropy.live.engine import LiveEngine
        # by_klines() rejects any subscribe_ticks() (TradingError), and an
        # 'orderbook' stream always subscribes a companion tick proxy. So
        # by_klines() is only usable when EVERY stream is 'kline'; any tick or
        # orderbook stream in the mix requires by_ticks(), which already
        # coexists with OHLC proxies (a strategy may subscribe ticks and
        # klines together).
        #
        # require_warmup=False: a broker/venue with no historical data yet
        # (new listing, thin market, or simply none available) is expected and
        # must not prevent recording - it just means recording starts with
        # empty rings and fills in from the live feed itself.
        if all(s["stream"] == "kline" for s in self._specs):
            self._engine = LiveEngine.by_klines(
                strat, sesh=self._sesh, poll_interval=self._poll_interval,
                require_warmup=False,
            )
        else:
            self._engine = LiveEngine.by_ticks(
                strat, sesh=self._sesh, poll_interval=self._poll_interval,
                require_warmup=False,
            )

    # -- Read-only info ---------------------------------------------------------

    @property
    def stream(self) -> Literal["tick", "kline", "orderbook"]:
        """
        What is being recorded.

        For a single-stream recorder this is that stream ('tick', 'kline' or
        'orderbook'). For a multi-stream recorder, use `specs` to inspect each
        one individually.
        """
        return self._specs[0]["stream"]

    @property
    def symbol(self) -> str:
        """
        The recorded symbol.

        For a single-stream recorder this is that stream's symbol. For a
        multi-stream recorder, use `specs` to inspect each one individually.
        """
        return self._specs[0]["symbol"]

    @property
    def path(self) -> Path:
        """
        Destination file path.

        For a single-stream recorder this is that stream's path. For a
        multi-stream recorder, use `specs` to inspect each one individually.
        """
        return Path(self._specs[0]["path"])

    @property
    def specs(self) -> "list[dict]":
        """Normalized list of recording specs (stream, symbol, path, ...)."""
        return list(self._specs)

    @property
    def uses_streaming(self) -> bool:
        """True if the session will stream over WebSocket (else REST polling)."""
        return bool(getattr(self._sesh, "supports_streaming", False))

    # -- Control ----------------------------------------------------------------

    def run(self, duration: "float | None" = None, blocking: bool = True) -> "Recorder":
        """
        Start recording every declared stream.

        Args:
            duration (float | None): If set, auto-stop after this many seconds.
                If None, record until stop() or Ctrl+C (blocking mode).
            blocking (bool): True (default) blocks the current thread until the
                recording stops. False runs it on a background daemon thread.

        Returns:
            Recorder: self, for chaining.

        Raises:
            ConfigError: If no stream has been declared with add_stream().
        """
        if not self._specs:
            raise ConfigError(
                "Recorder.run(): no stream to record. Declare at least one with "
                "add_stream('tick'|'kline'|'orderbook', symbol, path, ...)."
            )
        if self._engine is None:
            self._build_engine()
        self._started = True

        self._cancel_timer()
        if duration is not None:
            self._timer = threading.Timer(float(duration), self._engine.stop)
            self._timer.daemon = True
            self._timer.start()

        if blocking:
            try:
                self._engine.run(blocking=True)
            finally:
                self._cancel_timer()
                self._engine.stop()   # guarantees the final buffer flush
        else:
            self._engine.run(blocking=False)
        return self

    def stop(self) -> "Recorder":
        """
        Stop recording and flush any pending buffered events to disk.

        Returns:
            Recorder: self, for chaining.
        """
        self._cancel_timer()
        if self._engine is not None:
            self._engine.stop()
        return self

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:
        if not self._specs:
            return f"Recorder(empty, streaming={self.uses_streaming})"
        if len(self._specs) == 1:
            s = self._specs[0]
            return (
                f"Recorder(stream={s['stream']!r}, symbol={s['symbol']!r}, "
                f"path={str(s['path'])!r}, streaming={self.uses_streaming})"
            )
        summary = ", ".join(
            f"{s['symbol']}:{s['stream']}" for s in self._specs
        )
        return (
            f"Recorder(specs=[{summary}], streaming={self.uses_streaming})"
        )

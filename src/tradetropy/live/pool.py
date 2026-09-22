"""
pool.py
=======
Run several live strategies at once under a single supervisor (``LivePool``).

Design
------
The live engine is deliberately single-writer: one thread mutates the rings,
with no locks. ``LivePool`` preserves that property by organizing strategies
into groups:

- A **group** is a set of strategies that share one ``Sesh`` (and therefore one
  WebSocket feed and one broker). Its strategies are co-located on a single
  engine thread: the group owns one ``FeedRunner`` + ``EventBus`` and dispatches
  each event to the strategies subscribed to that symbol, sequentially. Sharing
  a ``Sesh`` means sharing the broker / net positions, which is safe because a
  single thread applies every event.
- The **pool** supervises several groups. Each group runs with a selectable
  isolation level:

  - ``"thread"`` - the group runs on a daemon thread in this process (light
    isolation, shared memory across groups possible).
  - ``"process"`` - the group runs in its own child process (hard isolation: a
    crash cannot propagate; suitable for separate accounts / venues).

If a strategy needs its own ``Sesh``, put it in its own group.

Strategies and sessions are supplied as **factories** (a ``Strategy`` subclass,
a zero-arg callable, or - for threads only - an instance). Factories must be
picklable for ``"process"`` groups (spawn), mirroring ``PoolBacktestEngine``.

Fault handling
--------------
When a strategy raises inside ``on_data()``:

- the group isolates (quarantines) that strategy and keeps the others running,
- it calls the strategy's ``on_crash(exc)`` hook,
- it emits a :class:`StrategyErrorEvent` to the pool-level ``on_error`` callback
  (the "manual protocol").

There is no implicit auto-restart; the callback may call
``pool.restart_group(name)`` if desired.

Example
-------
    from tradetropy.live import LivePool
    from tradetropy.connectors.ccxt import SeshCCXTLive

    def on_fail(ev):
        print(f"[{ev.group}] {ev.strategy_name} crashed: {ev.exc!r}")

    pool = LivePool(on_error=on_fail)
    pool.add_group(
        strategies=[MakerStrat, TakerStrat],
        sesh=lambda: SeshCCXTLive("binance", cfg),   # one shared sesh
        isolation="thread",
        name="binance-main",
    )
    pool.add_group(
        strategies=[ArbStrat],
        sesh=lambda: SeshCCXTLive("okx", cfg2),
        isolation="process",
        name="okx-arb",
    )
    pool.run()                # blocking; supervises all groups
    # pool.status(); pool.restart_group("okx-arb"); pool.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import platform
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from tradetropy.exceptions import ConfigError
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import Sesh


# =============================================================================
# PUBLIC DATA TYPES
# =============================================================================


@dataclass(frozen=True)
class StrategyErrorEvent:
    """
    Immutable record of a strategy (or group) failure, handed to the pool
    ``on_error`` callback.

    Attributes:
        group (str): Name of the group the failure occurred in.
        strategy_name (str): Name of the faulted strategy (``ClassName#index``),
            or ``''`` for a group-level failure not tied to one strategy.
        strategy_index (int): Index of the strategy inside the group, or -1.
        exc (BaseException): The exception raised.
        traceback_str (str): Formatted traceback for logging / diagnosis.
        ts (float): Wall-clock time of the failure (``time.time()``).
        isolation (str): Isolation of the group (``"thread"`` / ``"process"``).
        phase (str): Where it failed (``"on_data"``, ``"order_event"``,
            ``"prepare"``, ``"feed"``).
    """

    group: str
    strategy_name: str
    strategy_index: int
    exc: BaseException
    traceback_str: str
    ts: float
    isolation: str
    phase: str


@dataclass
class GroupStatus:
    """
    Snapshot of a group's health, returned by :meth:`LivePool.status`.

    Attributes:
        name (str): Group name.
        isolation (str): ``"thread"`` or ``"process"``.
        alive (bool): Whether the group is still running.
        n_strategies (int): Total strategies configured in the group.
        n_active (int): Strategies still running (not quarantined).
        n_faulted (int): Strategies quarantined after a crash.
        faulted (list[str]): Names of the quarantined strategies.
        bus_depth (int): Current EventBus queue depth (backpressure metric).
        bus_max_depth (int): High-water mark of the EventBus depth.
        restarts (int): Number of times the group has been restarted.
        last_error (str | None): ``repr`` of the last error, if any.
    """

    name: str
    isolation: str
    alive: bool
    n_strategies: int
    n_active: int
    n_faulted: int
    faulted: list
    bus_depth: int
    bus_max_depth: int
    restarts: int
    last_error: Optional[str]


@dataclass
class GroupSpec:
    """
    Normalized, (mostly) picklable configuration for one group.

    ``on_error`` is a parent-process callback and is stripped before a spec is
    sent to a child process (see :meth:`child_copy`).
    """

    name: str
    strategy_factories: tuple
    sesh_factory: Callable[[], Sesh]
    isolation: str = "thread"
    feed_type: str = "tick"
    save_log: bool = False
    on_error: Optional[Callable[[StrategyErrorEvent], None]] = None
    _restarts: int = field(default=0, repr=False)

    def build_sesh(self) -> Sesh:
        """Build the group's shared session from its factory."""
        return self.sesh_factory()

    def child_copy(self) -> "GroupSpec":
        """Return a copy safe to pickle to a child process (no callback)."""
        return replace(self, on_error=None)


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================


def _as_strategy_factory(s):
    """
    Normalize a strategy provider to a zero-arg callable.

    Accepts a ``Strategy`` subclass (returned as-is), a ``Strategy`` instance
    (converted to its class, matching ``PoolBacktestEngine``; constructor args
    are lost, so prefer a factory for parameterized strategies), or any zero-arg
    callable returning a ``Strategy``.

    Args:
        s: The strategy provider.

    Returns:
        Callable[[], Strategy]: A factory building a fresh strategy.

    Raises:
        ConfigError: If ``s`` is not a valid strategy provider.
    """
    if isinstance(s, type) and issubclass(s, Strategy):
        return s
    if isinstance(s, Strategy):
        return type(s)
    if callable(s):
        return s
    raise ConfigError(
        "Each strategy must be a Strategy subclass, a Strategy instance, or a "
        f"zero-arg callable returning a Strategy. Got: {type(s).__name__}."
    )


class _ConstSesh:
    """
    Picklable-if-the-instance-is wrapper turning a Sesh instance into a factory.

    Used only for thread groups (a shared instance). Process groups require a
    real callable factory so each child builds its own session.
    """

    __slots__ = ("_inst",)

    def __init__(self, inst: Sesh):
        self._inst = inst

    def __call__(self) -> Sesh:
        return self._inst


def _as_sesh_factory(sesh, isolation: str) -> Callable[[], Sesh]:
    """
    Normalize the ``sesh`` argument to a zero-arg factory.

    Args:
        sesh: A ``Sesh`` instance or a zero-arg callable returning one.
        isolation: The group's isolation level.

    Returns:
        Callable[[], Sesh]: The session factory.

    Raises:
        ConfigError: If a raw instance is passed to a process group, or the
            value is neither a Sesh nor callable.
    """
    if isinstance(sesh, Sesh):
        if isolation == "process":
            raise ConfigError(
                "A process group requires a sesh FACTORY (callable), not a Sesh "
                "instance: the instance cannot be shared across processes. Pass "
                "e.g. sesh=lambda: SeshCCXTLive(...) defined at module level."
            )
        return _ConstSesh(sesh)
    if callable(sesh):
        return sesh
    raise ConfigError(
        "sesh must be a Sesh instance or a zero-arg callable returning a Sesh. "
        f"Got: {type(sesh).__name__}."
    )


def _make_group_spec(
    strategies,
    sesh,
    *,
    isolation: str,
    name: Optional[str],
    feed_type: str,
    save_log: bool,
    on_error,
    auto_index: int,
) -> GroupSpec:
    """Validate inputs and build a normalized :class:`GroupSpec`."""
    if isolation not in ("thread", "process"):
        raise ConfigError(
            f"isolation must be 'thread' or 'process', got {isolation!r}."
        )
    if feed_type not in ("tick", "kline"):
        raise ConfigError(
            f"feed_type must be 'tick' or 'kline', got {feed_type!r}."
        )
    if isinstance(strategies, (Strategy, type)) or callable(strategies):
        if not isinstance(strategies, (list, tuple)):
            strategies = [strategies]
    if not strategies:
        raise ConfigError("A group needs at least one strategy.")

    factories = tuple(_as_strategy_factory(s) for s in strategies)
    sesh_factory = _as_sesh_factory(sesh, isolation)
    group_name = name or f"group-{auto_index}"

    if isolation == "process":
        _assert_picklable(factories, sesh_factory, group_name)

    return GroupSpec(
        name=group_name,
        strategy_factories=factories,
        sesh_factory=sesh_factory,
        isolation=isolation,
        feed_type=feed_type,
        save_log=save_log,
        on_error=on_error,
    )


def _assert_picklable(factories, sesh_factory, group_name: str) -> None:
    """Raise a clear error early if a process group's factories cannot pickle."""
    import pickle

    try:
        pickle.dumps(factories)
        pickle.dumps(sesh_factory)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"Process group {group_name!r} needs picklable factories (spawn). "
            f"Define strategy classes and the sesh factory at module level "
            f"(no lambdas / local closures capturing unpicklable state). "
            f"Underlying error: {exc}"
        ) from exc


# =============================================================================
# THREAD GROUP - multiplexed single-feed engine driver
# =============================================================================


class LiveGroup:
    """
    A set of strategies sharing one ``Sesh`` + feed, driven on one thread.

    The group owns a single ``FeedRunner`` + ``EventBus`` built from the union
    of its strategies' symbols/channels, drains events and dispatches each to
    the strategies subscribed to that symbol (``LiveEngine._process_event``).
    Private ORDER/FILL events are applied once to the shared session (never
    per-strategy) so the shared broker state is not double-updated.

    A failing strategy is quarantined; the rest keep running.
    """

    def __init__(self, spec: GroupSpec):
        self.spec = spec
        self.name = spec.name

        self._sesh: Optional[Sesh] = None
        self._engines: list = []
        self._names: list = []
        self._active: list = []
        self._faulted: list = []
        self._route: dict = {}

        self._bus = None
        self._runner = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._prepared = False
        self._feed_error: Optional[BaseException] = None
        self._last_error: Optional[str] = None

        # Error sink: defaults to the pool callback; a process child swaps this
        # for a queue push so failures reach the parent.
        self._error_sink = spec.on_error

    # ---- lifecycle ----------------------------------------------------------

    def set_error_sink(self, fn) -> None:
        """Override where StrategyErrorEvents are delivered (used by the child)."""
        self._error_sink = fn

    @property
    def strategies(self) -> list:
        """The live strategy instances (thread mode only; empty until prepared)."""
        return [eng.strategy for eng in self._engines]

    def prepare(self) -> None:
        """Build the shared session and one prepared LiveEngine per strategy."""
        from tradetropy.live.engine import LiveEngine

        self._sesh = self.spec.build_sesh()
        if not getattr(self._sesh, "supports_streaming", False):
            raise ConfigError(
                f"Group {self.name!r}: the session "
                f"{type(self._sesh).__name__} does not support streaming "
                f"(supports_streaming is False). LivePool drives strategies over "
                f"the event-driven streaming path."
            )

        self._engines = []
        self._names = []
        for i, factory in enumerate(self.spec.strategy_factories):
            strat = factory()
            if not isinstance(strat, Strategy):
                raise ConfigError(
                    f"Group {self.name!r}: factory #{i} did not return a "
                    f"Strategy (got {type(strat).__name__})."
                )
            if self.spec.feed_type == "tick":
                eng = LiveEngine.by_ticks(strat, sesh=self._sesh)
            else:
                eng = LiveEngine.by_klines(strat, sesh=self._sesh)
            eng._save_log = self.spec.save_log
            eng.prepare()
            self._engines.append(eng)
            self._names.append(f"{type(strat).__name__}#{i}")

        self._active = [True] * len(self._engines)
        self._faulted = []
        self._build_route()
        self._prepared = True

    def _build_route(self) -> None:
        """Rebuild the symbol -> active engine-index routing table."""
        route: dict = {}
        for i, eng in enumerate(self._engines):
            if not self._active[i]:
                continue
            for sym in eng._simbolos_del_loop():
                route.setdefault(sym, []).append(i)
        self._route = route

    def _feed_plan(self):
        """Compute the union of symbols/channels and feed kwargs for the group."""
        from tradetropy.streaming import (
            BOOK_SNAPSHOT,
            FILL,
            KLINE,
            MBO,
            ORDER,
            TRADE,
        )

        symbols: set = set()
        has_book = False
        has_mbo = False
        book_limit = 20
        interval = 60_000

        for eng in self._engines:
            symbols.update(eng._simbolos_del_loop())
            bps = getattr(eng.strategy, "_book_proxies", [])
            if bps:
                has_book = True
                book_limit = max([book_limit] + [bp.depth for bp in bps])
            if getattr(eng.strategy, "_mbo_proxies", []):
                has_mbo = True
            for op in eng.strategy._ohlc_proxies:
                interval = op.interval_ms
                break

        if self.spec.feed_type == "tick":
            channels = [TRADE]
            feed_kwargs = {"book_limit": book_limit}
        else:
            channels = [KLINE]
            feed_kwargs = {"timeframe_ms": interval, "book_limit": book_limit}

        if has_book:
            channels.append(BOOK_SNAPSHOT)
        if has_mbo:
            channels.append(MBO)
        if getattr(self._sesh, "supports_user_stream", False):
            channels += [ORDER, FILL]

        return list(symbols), channels, feed_kwargs

    def start(self) -> None:
        """Prepare (if needed) and launch the group's driver thread."""
        if not self._prepared:
            self.prepare()
        self._stop.clear()
        self._feed_error = None
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=f"LiveGroup-{self.name}"
        )
        self._thread.start()

    def _on_feed_error(self, exc: Exception) -> None:
        """Feed-thread error callback: record and ask the loop to stop."""
        self._feed_error = exc
        self._stop.set()

    def _run_loop(self) -> None:
        """Own the shared feed and dispatch every event until stopped/drained."""
        from tradetropy.streaming import EventBus, FeedRunner

        symbols, channels, feed_kwargs = self._feed_plan()
        feed = self._sesh.create_feed(**feed_kwargs)
        bus = EventBus()
        runner = FeedRunner(
            feed,
            bus,
            symbols,
            channels,
            on_error=self._on_feed_error,
            name=f"LiveGroup-{self.name}-feed",
        )
        self._bus = bus
        self._runner = runner
        runner.start()

        try:
            while not self._stop.is_set():
                batch = bus.get_batch(timeout=0.1, max_items=10_000)
                for event in batch:
                    if self._stop.is_set():
                        break
                    self._dispatch(event)
                if not batch and not runner.is_alive() and bus.depth == 0:
                    break
        finally:
            runner.stop()

        if self._feed_error is not None:
            self._last_error = repr(self._feed_error)
            self._emit_error(self._feed_error, index=-1, phase="feed")

    # ---- dispatch + fault isolation -----------------------------------------

    def _dispatch(self, event) -> None:
        """
        Route one event: private ORDER/FILL once to the shared session; all
        other events to every active engine subscribed to the symbol.
        """
        ch = event.channel
        if ch in ("order", "fill"):
            self._apply_user_event(event)
            return

        symbol = getattr(event, "symbol", None)
        for i in self._route.get(symbol, ()):  # only active engines are routed
            if not self._active[i]:
                continue
            try:
                self._engines[i]._process_event(event)
            except Exception as exc:  # noqa: BLE001 - quarantine, keep others
                self._quarantine(i, exc)

    def _apply_user_event(self, event) -> None:
        """Apply a private order/fill event once to the shared session."""
        try:
            if event.channel == "order":
                self._sesh.apply_order_event(event)
            else:
                self._sesh.apply_fill_event(event)
        except Exception as exc:  # noqa: BLE001
            self._last_error = repr(exc)
            self._emit_error(exc, index=-1, phase="order_event")

    def _quarantine(self, index: int, exc: Exception) -> None:
        """Isolate a crashed strategy, call on_crash, emit the error event."""
        with self._lock:
            if self._active[index]:
                self._active[index] = False
                self._faulted.append(self._names[index])
                self._build_route()
        self._last_error = repr(exc)

        strat = self._engines[index].strategy
        try:
            strat.on_crash(exc)
        except Exception:  # noqa: BLE001 - never let the hook break the group
            pass

        self._emit_error(exc, index=index, phase="on_data")

    def _emit_error(self, exc: BaseException, *, index: int, phase: str) -> None:
        """Build and deliver a StrategyErrorEvent to the error sink."""
        name = self._names[index] if 0 <= index < len(self._names) else ""
        ev = StrategyErrorEvent(
            group=self.name,
            strategy_name=name,
            strategy_index=index,
            exc=exc,
            traceback_str="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            ts=time.time(),
            isolation=self.spec.isolation,
            phase=phase,
        )
        if self._error_sink is not None:
            try:
                self._error_sink(ev)
            except Exception:  # noqa: BLE001
                pass

    # ---- control + status ---------------------------------------------------

    def is_alive(self) -> bool:
        """True while the group's driver thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the driver, join the thread, flush records and call on_stop."""
        self._stop.set()
        if self._runner is not None:
            self._runner.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        for eng in self._engines:
            try:
                eng.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                eng.strategy.on_stop()
            except Exception:  # noqa: BLE001
                pass

    def status(self) -> GroupStatus:
        """Return a health snapshot of the group."""
        return GroupStatus(
            name=self.name,
            isolation=self.spec.isolation,
            alive=self.is_alive(),
            n_strategies=len(self._engines),
            n_active=sum(self._active) if self._active else 0,
            n_faulted=len(self._faulted),
            faulted=list(self._faulted),
            bus_depth=self._bus.depth if self._bus is not None else 0,
            bus_max_depth=self._bus.max_depth if self._bus is not None else 0,
            restarts=self.spec._restarts,
            last_error=self._last_error,
        )


# =============================================================================
# PROCESS GROUP - a LiveGroup running in its own child process
# =============================================================================


def _process_group_main(child_spec: GroupSpec, stop_event, status_q) -> None:
    """
    Child-process entry point: run a LiveGroup and report status/errors.

    Runs in a spawned process. Group failures are pushed to ``status_q`` as
    ``("error", StrategyErrorEvent)`` (the parent invokes the real callback);
    periodic ``("status", GroupStatus)`` messages keep the parent's cache warm.

    Args:
        child_spec (GroupSpec): Group config (callback already stripped).
        stop_event: multiprocessing.Event signalling shutdown.
        status_q: multiprocessing.Queue for status/error/lifecycle messages.
    """
    group = LiveGroup(child_spec)
    group.set_error_sink(lambda ev: status_q.put(("error", ev)))

    try:
        group.prepare()
    except Exception as exc:  # noqa: BLE001
        status_q.put((
            "error",
            StrategyErrorEvent(
                group=child_spec.name,
                strategy_name="",
                strategy_index=-1,
                exc=exc,
                traceback_str="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
                ts=time.time(),
                isolation="process",
                phase="prepare",
            ),
        ))
        status_q.put(("stopped", None))
        return

    group.start()
    try:
        while not stop_event.is_set() and group.is_alive():
            status_q.put(("status", group.status()))
            time.sleep(0.5)
    finally:
        group.stop()
        status_q.put(("status", group.status()))
        status_q.put(("stopped", None))


class ProcessGroup:
    """
    Supervisor-side handle for a group running in its own child process.

    Spawns the child, then drains its status queue on a parent monitor thread:
    it caches the latest :class:`GroupStatus` and forwards each
    :class:`StrategyErrorEvent` to the pool ``on_error`` callback.
    """

    def __init__(self, spec: GroupSpec):
        self.spec = spec
        self.name = spec.name
        self._on_error = spec.on_error

        ctx_name = "fork" if platform.system() == "Linux" else "spawn"
        self._ctx = mp.get_context(ctx_name)
        self._stop_event = self._ctx.Event()
        self._status_q = self._ctx.Queue()
        self._proc: Optional[mp.process.BaseProcess] = None
        self._monitor: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()

        self._last_status: Optional[GroupStatus] = None
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """Spawn the child process and start the parent monitor thread."""
        self._proc = self._ctx.Process(
            target=_process_group_main,
            args=(self.spec.child_copy(), self._stop_event, self._status_q),
            name=f"ProcessGroup-{self.name}",
            daemon=True,
        )
        self._proc.start()
        self._monitor = threading.Thread(
            target=self._drain, daemon=True, name=f"ProcessGroup-{self.name}-mon"
        )
        self._monitor.start()

    def _drain(self) -> None:
        """Parent monitor: forward errors to the callback, cache status."""
        import queue as _queue

        while not self._monitor_stop.is_set():
            try:
                kind, payload = self._status_q.get(timeout=0.2)
            except _queue.Empty:
                if self._proc is not None and not self._proc.is_alive():
                    break
                continue
            except (EOFError, OSError):
                break

            if kind == "status":
                self._last_status = payload
            elif kind == "error":
                self._last_error = repr(payload.exc)
                if self._on_error is not None:
                    try:
                        self._on_error(payload)
                    except Exception:  # noqa: BLE001
                        pass
            elif kind == "stopped":
                break

    def is_alive(self) -> bool:
        """True while the child process is running."""
        return self._proc is not None and self._proc.is_alive()

    def stop(self, timeout: float = 12.0) -> None:
        """Signal the child to stop, join it (terminate on timeout)."""
        self._stop_event.set()
        if self._proc is not None:
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=3.0)
        self._monitor_stop.set()
        if self._monitor is not None and self._monitor.is_alive():
            self._monitor.join(timeout=2.0)

    def status(self) -> GroupStatus:
        """Return the latest cached status from the child (or a minimal one)."""
        if self._last_status is not None:
            return replace(self._last_status, alive=self.is_alive())
        return GroupStatus(
            name=self.name,
            isolation="process",
            alive=self.is_alive(),
            n_strategies=len(self.spec.strategy_factories),
            n_active=len(self.spec.strategy_factories),
            n_faulted=0,
            faulted=[],
            bus_depth=0,
            bus_max_depth=0,
            restarts=self.spec._restarts,
            last_error=self._last_error,
        )


# =============================================================================
# LIVE POOL - supervisor over groups
# =============================================================================


class LivePool:
    """
    Supervise several live strategy groups (thread- or process-isolated).

    Args:
        on_error (Callable[[StrategyErrorEvent], None] | None): Called when any
            strategy (or group) fails. This is the "manual protocol" hook - it
            runs alongside the strategy's own ``on_crash`` and never stops the
            other strategies. It may call :meth:`restart_group`.

    Example:
        pool = LivePool(on_error=lambda ev: print(ev.strategy_name, ev.exc))
        pool.add_group(strategies=[S1, S2], sesh=factory, isolation="thread")
        pool.run()
    """

    def __init__(self, on_error: Optional[Callable[[StrategyErrorEvent], None]] = None):
        self._on_error = on_error
        self._specs: "OrderedDict[str, GroupSpec]" = OrderedDict()
        self._groups: "OrderedDict[str, object]" = OrderedDict()
        self._running = False
        self._auto_index = 0
        self._lock = threading.Lock()

    # ---- configuration ------------------------------------------------------

    def add_group(
        self,
        strategies,
        sesh,
        *,
        isolation: str = "thread",
        name: Optional[str] = None,
        feed_type: str = "tick",
        save_log: bool = False,
    ) -> str:
        """
        Register (and, if the pool is already running, start) a group.

        Args:
            strategies: A strategy provider or a list of them (see module docs).
            sesh: A ``Sesh`` instance (thread groups) or a zero-arg factory
                returning one (required for process groups).
            isolation: ``"thread"`` (default) or ``"process"``.
            name: Unique group name (auto-generated if omitted).
            feed_type: ``"tick"`` (default) or ``"kline"``.
            save_log: Persist each strategy's log to file (default False, since
                several strategies run headless under one supervisor).

        Returns:
            str: The group's name.

        Raises:
            ConfigError: On invalid inputs or a duplicate name.
        """
        with self._lock:
            spec = _make_group_spec(
                strategies,
                sesh,
                isolation=isolation,
                name=name,
                feed_type=feed_type,
                save_log=save_log,
                on_error=self._on_error,
                auto_index=self._auto_index,
            )
            self._auto_index += 1
            if spec.name in self._specs:
                raise ConfigError(f"Group name {spec.name!r} already exists.")
            self._specs[spec.name] = spec
            if self._running:
                self._start_group(spec)
        return spec.name

    @classmethod
    def from_strategies(
        cls,
        pairs,
        *,
        isolation: str = "thread",
        feed_type: str = "tick",
        on_error: Optional[Callable[[StrategyErrorEvent], None]] = None,
    ) -> "LivePool":
        """
        Build a pool where each ``(strategy, sesh)`` pair is its own group.

        Convenience for the "each strategy fully independent (own sesh)" case.

        Args:
            pairs: Iterable of ``(strategy_provider, sesh_provider)``.
            isolation: Isolation for every group.
            feed_type: Feed type for every group.
            on_error: Pool-level failure callback.

        Returns:
            LivePool: The configured (not yet running) pool.
        """
        pool = cls(on_error=on_error)
        for strat, sesh in pairs:
            pool.add_group(
                strategies=strat,
                sesh=sesh,
                isolation=isolation,
                feed_type=feed_type,
            )
        return pool

    # ---- lifecycle ----------------------------------------------------------

    def _start_group(self, spec: GroupSpec) -> None:
        """Instantiate and start the concrete group for a spec."""
        group = ProcessGroup(spec) if spec.isolation == "process" else LiveGroup(spec)
        group.start()
        self._groups[spec.name] = group

    def run(self, blocking: bool = True) -> "LivePool":
        """
        Start every configured group and supervise them.

        Args:
            blocking: If True (default), block until all groups finish (finite
                feeds) or a KeyboardInterrupt, then stop cleanly. If False,
                start the groups and return immediately.

        Returns:
            LivePool: self (for chaining).
        """
        self._running = True
        for spec in list(self._specs.values()):
            if spec.name not in self._groups:
                self._start_group(spec)

        if blocking:
            try:
                while self._running and any(
                    g.is_alive() for g in self._groups.values()
                ):
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
            self.stop()
        return self

    def stop(self) -> None:
        """Stop all groups cleanly. Group handles are retained for inspection."""
        self._running = False
        for group in list(self._groups.values()):
            try:
                group.stop()
            except Exception:  # noqa: BLE001
                pass

    # ---- hot control --------------------------------------------------------

    def stop_group(self, name: str) -> None:
        """Stop a single group without affecting the others."""
        group = self._groups.get(name)
        if group is None:
            raise ConfigError(f"No running group named {name!r}.")
        group.stop()

    def restart_group(self, name: str) -> None:
        """
        Restart a group from its factories (fresh session + strategies).

        Works for a running or a stopped/quarantined group. The restart count is
        preserved on the spec and reflected in :meth:`status`.
        """
        spec = self._specs.get(name)
        if spec is None:
            raise ConfigError(f"No group named {name!r}.")
        old = self._groups.get(name)
        if old is not None:
            try:
                old.stop()
            except Exception:  # noqa: BLE001
                pass
        spec._restarts += 1
        if self._running:
            self._start_group(spec)

    def remove_group(self, name: str) -> None:
        """Stop and forget a group entirely (its name becomes reusable)."""
        group = self._groups.pop(name, None)
        if group is not None:
            try:
                group.stop()
            except Exception:  # noqa: BLE001
                pass
        self._specs.pop(name, None)

    # ---- introspection ------------------------------------------------------

    def group(self, name: str):
        """Return the concrete group handle (``LiveGroup`` / ``ProcessGroup``)."""
        return self._groups.get(name)

    @property
    def group_names(self) -> list:
        """Names of the configured groups, in insertion order."""
        return list(self._specs.keys())

    def status(self) -> dict:
        """Return ``{group_name: GroupStatus}`` for every started group."""
        return {name: g.status() for name, g in self._groups.items()}

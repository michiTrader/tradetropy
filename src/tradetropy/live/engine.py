"""
live_engine.py v4.5
====================
Engine for live trading with automatic warmup and feed loop.

FULL FLOW
---------
    engine = LiveEngine.by_ticks(MyStrategy(), sesh=sesh)
    engine.run()   # prepare() is called automatically

    # With chart (automatic wiring: attach + start in the correct order):
    engine.run(live_chart=chart)

    # Equivalent manual flow (for fine-grained control):
    engine.prepare()
    engine.attach_chart(chart)
    chart.start()
    engine.run()
"""

from __future__ import annotations

import threading
from typing import Optional, Literal

import numpy as np

from tradetropy.models.strategy import Strategy, FeedType, StopEngine
from tradetropy.models.footprint import LiveFpRing
from tradetropy.data.data import (
    TickProxy,
    OhlcProxy,
    IndicatorProxy,
    LiveRingBuffer,
    LiveOhlcRing,
    MultiBandProxy,
)
from tradetropy.exceptions import TradingError, ConnectionError


class LiveEngine:
    """
    Engine for live trading.

    Handles automatic warmup (historical data loading), indicator initialization,
    and real-time feed loop for tick or candle-based strategies.

    Preferred constructors:
    - LiveEngine.by_ticks(strategy, sesh=sesh) - warmup in ticks
    - LiveEngine.by_klines(strategy, sesh=sesh) - warmup in candles

    Automatic Warmup:
    - by_ticks: Loads strategy.warmup ticks (with hybrid historical candle support)
    - by_klines: Loads strategy.warmup candles

    Typical Usage:
        engine = LiveEngine.by_ticks(MyStrategy(), sesh=sesh)
        engine.run()

        # With chart (automatic wiring):
        engine.run(live_chart=chart)

        # With chart (manual, fine control):
        engine.prepare()
        engine.attach_chart(chart)
        chart.start()
        engine.run()
    """

    #: Default policy for writing the log to file (opt-in).
    #: Live DOES persist the log by default (production). Override via save_log
    #: in run(). ReplayEngine overrides this to False.
    _DEFAULT_SAVE_LOG: bool = True

    def __init__(
        self,
        strategy: Strategy,
        sesh=None,
        _feed_type: Literal["tick", "kline"] = "tick",
        _poll_interval: float = 1.0,
        chart_ohlc_interval_ms: int = 60_000,
        require_warmup: bool = True,
    ):
        self.strategy = strategy
        self._sesh = sesh
        self._feed_type: FeedType = _feed_type
        # When False, a broker returning zero historical ticks/klines is
        # tolerated (warning instead of DataError): the engine starts with
        # empty rings and warms up from the live feed itself. Used by Recorder,
        # where "no history yet" (new listing, quiet market) is expected and
        # recording should still start. Strategies keep the strict default
        # (True) so they never run silently blind on missing data.
        self._require_warmup: bool = bool(require_warmup)

        # Interval (ms) of the chart-only OHLC proxy auto-injected for tick
        # strategies that declare no subscribe_ohlc(). Default 1 minute.
        self._chart_ohlc_interval_ms: int = int(chart_ohlc_interval_ms)
        # OHLC proxies created by the engine purely so the chart has a candle
        # backbone (NOT declared by the strategy). They are excluded from the
        # REST top-up so a headless tick strategy triggers no network fetch.
        self._auto_ohlc_proxies: list[OhlcProxy] = []
        # If the session is simulated and was created in auto mode (feed_type=None),
        # builds its internal broker according to the engine mode. For real live
        # sessions (without _bind_feed_type) this is a no-op.
        if sesh is not None and hasattr(sesh, "_bind_feed_type"):
            sesh._bind_feed_type(_feed_type)
        self._poll_interval: float = _poll_interval

        self._tick_rings: dict[str, LiveRingBuffer] = {}
        self._ohlc_rings: dict[str, list[LiveOhlcRing]] = {}
        self._fp_rings: dict[str, list[LiveFpRing]] = {}
        self._book_rings: dict[str, list] = {}
        self._mbo_rings: dict[str, list] = {}
        self._ind_state: list[dict] = []

        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._preparado: bool = False

        # Rebuild gating (used by ReplayEngine to rewind a replay in place).
        # When _suppress_on_data is True, _process_tick/_process_bar update the
        # rings/indicators/broker but skip strategy.on_data() (warmup re-feed).
        # When _suppress_chart is True, the chart is not notified per tick during
        # a bulk re-feed; a single resync is issued at the end instead.
        self._suppress_on_data: bool = False
        self._suppress_chart: bool = False

        self._historical_candles: int = 0

        self._chart: "LiveChart | None" = None

        from tradetropy.session.base import SeshSimulatorBase
        self._is_simulated: bool = isinstance(sesh, SeshSimulatorBase)

    # ==========================================================================
    # CONSTRUCTORS
    # ==========================================================================

    @classmethod
    def by_ticks(
        cls,
        strategy: Strategy,
        sesh=None,
        poll_interval: float = 1.0,
        chart_ohlc_interval_ms: int = 60_000,
        require_warmup: bool = True,
    ) -> "LiveEngine":
        """
        Create LiveEngine with tick-based warmup.

        Args:
            strategy: Strategy instance (must have warmup attribute).
            sesh: Optional session (broker) instance.
            poll_interval: Polling interval in seconds.
            chart_ohlc_interval_ms: Interval (ms) of the chart-only OHLC proxy
                auto-injected when the strategy subscribes ticks but no OHLC.
                Default 60_000 (1 minute). Used only for the chart; never fed
                to on_data().
            require_warmup: If False, a broker returning zero historical ticks
                or klines is tolerated (warning instead of DataError) and the
                engine starts with empty rings. Default True (strict).

        Returns:
            LiveEngine: Engine configured for tick-based operation.
        """
        return cls(
            strategy,
            sesh=sesh,
            _feed_type="tick",
            _poll_interval=poll_interval,
            chart_ohlc_interval_ms=chart_ohlc_interval_ms,
            require_warmup=require_warmup,
        )

    @classmethod
    def by_klines(
        cls,
        strategy: Strategy,
        sesh=None,
        poll_interval: float = 1.0,
        require_warmup: bool = True,
    ) -> "LiveEngine":
        """
        Create LiveEngine with candle-based warmup.

        Args:
            strategy: Strategy instance (must have warmup attribute).
            sesh: Optional session (broker) instance.
            poll_interval: Polling interval in seconds.
            require_warmup: If False, a broker returning zero historical klines
                is tolerated (warning instead of DataError) and the engine
                starts with empty rings. Default True (strict).

        Returns:
            LiveEngine: Engine configured for candle-based operation.
        """
        return cls(
            strategy, sesh=sesh, _feed_type="kline",
            _poll_interval=poll_interval, require_warmup=require_warmup,
        )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    @property
    def sesh(self):
        """
        Get the session (broker) connected to the engine.

        Returns:
            Session: The broker instance.
        """
        return self._sesh

    @property
    def feed_type(self) -> FeedType:
        """
        Get the engine feed type.

        Returns:
            FeedType: Either 'tick' or 'kline'.
        """
        return self._feed_type

    def attach_chart(self, chart) -> None:
        """
        Connect a LiveChart to the engine for real-time visualization.

        Args:
            chart: LiveChart instance to attach.

        Raises:
            TradingError: If chart is not a LiveChart instance.
        """
        from tradetropy.plotting.live.chart import LiveChart as _LiveChart

        if not isinstance(chart, _LiveChart):
            raise TradingError(
                "attach_chart() requires a LiveChart instance. "
                f"Received: {type(chart).__name__}"
            )

        self._chart          = chart
        chart._engine        = self
        chart._strategy      = self.strategy
        _inner_broker = getattr(self.sesh, "_broker", None)
        chart._broker = _inner_broker if _inner_broker is not None else self.sesh

    def _maybe_inject_chart_ohlc(self) -> None:
        """
        Auto-inject a chart-only OHLC proxy for tick strategies lacking one.

        A live/replay/training chart needs at least one OHLC proxy to draw its
        candle backbone, but a pure order-flow strategy (only subscribe_ticks
        + tick-mounted indicators like DeepTrades/LargeTrades) has no reason to
        declare subscribe_ohlc(). To remove that friction, when the feed is
        tick-based and the strategy declared ticks but no OHLC, the engine
        injects a default OHLC proxy (interval chart_ohlc_interval_ms, 1 minute
        by default) on the first tick symbol.

        The proxy is built and fed by the existing machinery: _build_rings
        creates its LiveOhlcRing from the warmup ticks and _process_tick
        aggregates each live tick into candles. It mirrors what the static
        backtest plot already does (synthesize 1m candles from ticks).

        The proxy is recorded in self._auto_ohlc_proxies and excluded from the
        REST top-up, so it never causes a network fetch in a headless live
        session. It is not exposed to the strategy (no reference is handed back)
        and no indicator attaches to it, so on_data() decisions are unchanged.
        """
        if self._feed_type != "tick":
            return
        if self.strategy._ohlc_proxies:
            return
        if not self.strategy._tick_proxies:
            return

        symbol = self.strategy._tick_proxies[0].symbol
        proxy = OhlcProxy(symbol, self._chart_ohlc_interval_ms)
        self.strategy._ohlc_proxies.append(proxy)
        self._auto_ohlc_proxies.append(proxy)

    def prepare(self, historico: "dict | None" = None) -> None:
        """
        Prepares the engine: fetches historical data, builds rings, warms
        indicators and calls strategy.init().

        Args:
            historico: optional dict with preloaded data (for tests).
                       Format: {symbol: tick_array} and/or
                               {(symbol, interval_ms): kline_array}.
        """
        if self._feed_type == "kline" and self.strategy._tick_proxies:
            syms = [tp.symbol for tp in self.strategy._tick_proxies]
            raise TradingError(
                f"LiveEngine.by_klines() does not support subscribe_ticks(). "
                f"Problematic symbols: {syms}."
            )

        self.strategy._sesh = self.sesh
        self.strategy._feed_type = self._feed_type
        self.strategy._set_run_mode("live")
        self.strategy._save_log = getattr(self, "_save_log", self._DEFAULT_SAVE_LOG)

        self.strategy.init()

        self._maybe_inject_chart_ohlc()

        if historico is not None:
            hist = historico
        elif self.sesh is not None:
            hist = _fetch_auto_history(self)
        else:
            hist = {}

        _build_rings(self, hist)
        _warm(self, hist)

        if self._feed_type == "tick" and historico is None and self.sesh is not None:
            self._topup_ohlc_rings()

        _build_fp_rings(self, hist)
        _build_pattern_stores_live(self, hist)
        _build_book_rings(self, hist)
        _build_mbo_rings(self, hist)
        self._historical_candles = _count_closed_candles_rings(self)

        # Print warmup only for pure LiveEngine. ReplayEngine already prints its
        # own real warmup in _calc_warmup; the one here would be a reconstructed
        # value (misleading) because it does not know ticks_per_candle.
        if type(self).__name__ == "LiveEngine":
            from tradetropy.models._warmup_policy import resolve_warmup, log_warmup
            _wu = resolve_warmup(self.strategy, feed_type=self._feed_type)
            log_warmup(self.strategy, self._feed_type, _wu, "LiveEngine")

        self._preparado = True

    def run(
        self,
        params: dict = None,
        verbose: bool = False,
        blocking: bool = True,
        live_chart=None,
        save_log: "bool | None" = None,
    ):
        """
        Start the feed loop. Returns self for chaining.

        Args:
            params: Optional dict to override strategy parameters before starting
                    (equivalent to BacktestEngine.run).
            verbose: Enable diagnostic prints.
            blocking: True -> blocks current thread. False -> launches daemon thread.
            live_chart: Optional LiveChart. If passed, engine connects and starts
                        it automatically (equivalent to: prepare() -> attach_chart()
                        -> start() -> run()). Done in correct order, so it's
                        sufficient to call: engine.run(live_chart=chart)
            save_log: Write strategy log to file (strategy.log_file). None -> uses
                      engine default (_DEFAULT_SAVE_LOG = True in live, False in
                      replay). Requires log_file to be defined.

        Returns:
            self: Engine instance for chaining.

        Raises:
            TradingError: If both positional and live_chart= LiveChart passed.
        """
        from tradetropy.plotting.live.chart import LiveChart as _LiveChart
        if isinstance(params, _LiveChart):
            if live_chart is not None:
                raise TradingError('run(): passed a positional LiveChart and also live_chart=. Use only one.')
            live_chart, params = params, None
        if params is not None:
            self.strategy.update_params(params)
        self._verbose = verbose

        if save_log is None:
            save_log = self._DEFAULT_SAVE_LOG
        # Store so that prepare() can pick it up; _set_save_log invalidates the
        # lazy logger if prepare() already ran (init() may have created it).
        self._save_log = save_log
        self.strategy._set_save_log(save_log)

        if not self._preparado:
            self.prepare()

        # Automatic chart wiring: attach + start in the correct order,
        # right after prepare() (which creates the proxies the chart validates)
        # and before the loop (the Bokeh server must be ready before data).
        if live_chart is not None:
            self.attach_chart(live_chart)
            live_chart.start()

        # Hook for subclasses that need to start something between chart.start()
        # and the loop (e.g. ReplayEngine starts its ReplayController here).
        # No-op by default in LiveEngine.
        self._before_loop()

        self._stop_event.clear()

        if blocking:
            self._run_guarded()
        else:
            self._loop_thread = threading.Thread(
                target=self._run_guarded,
                daemon=True,
                name="LiveEngine-loop",
            )
            self._loop_thread.start()

        return self

    def _before_loop(self) -> None:
        """
        Hook called by run() after chart attach/start and before loop start.

        No-op in LiveEngine. Subclasses can override (e.g., ReplayEngine
        starts the ReplayController here).
        """
        pass

    def stop(self):
        """
        Stop the run() loop cleanly.

        In live mode, flushes any pending tick/OHLC records to disk before
        stopping the loop thread.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if not self._is_simulated:
            from tradetropy.io.io import _consolidate_npz_record
            for tp in self.strategy._tick_proxies:
                if tp._record_config is not None:
                    self._flush_tick_proxy(tp)
                    _consolidate_npz_record(tp._record_config.path)
            for op in self.strategy._ohlc_proxies:
                if op._record_config is not None:
                    self._flush_ohlc_proxy(op)
                    _consolidate_npz_record(op._record_config.path)
            for bp in self.strategy._book_proxies:
                if bp._record_config is not None:
                    self._flush_book_proxy(bp)
                    _consolidate_npz_record(bp._record_config.path)
            for mp in self.strategy._mbo_proxies:
                if mp._record_config is not None:
                    self._flush_mbo_proxy(mp)
                    _consolidate_npz_record(mp._record_config.path)
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=10)

    def on_tick(self, symbol: str, tick: np.ndarray):
        """
        Process a tick in tick mode (manual feed).

        Args:
            symbol: Trading symbol.
            tick: Tick array [ts_ms, bid, ask, volume, flags, volume_real, price].
        """
        _process_tick(self, symbol, tick)

    def on_kline(self, symbol: str, kline: np.ndarray):
        """
        Process a candle (kline) in kline mode (manual feed).

        Args:
            symbol: Trading symbol.
            kline: Kline array [ts_ms, open, high, low, close, volume, turnover].
        """
        _process_bar(self, symbol, kline)

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _get_ind_def_for_proxy(self, proxy) -> "dict | None":
        """
        Find the indicator_def corresponding to a proxy.

        Args:
            proxy: Proxy instance to search for.

        Returns:
            dict | None: Indicator definition or None if not found.
        """
        for defn in self.strategy._indicator_defs:
            if defn["proxy"] is proxy:
                return defn
        return None


# =============================================================================
# INTERNAL METHOD ASSIGNMENT
# =============================================================================

from tradetropy.live._warmup import (                    # noqa: E402
    _compute_required_ticks,
    _compute_required_klines,
    _fetch_auto_history,
    _fetch_ticks_for_warmup,
    _fetch_klines_for_warmup,
)
from tradetropy.live._builder import (                   # noqa: E402
    _build_rings,
    _warm,
    _resync_chart_sources,
    _topup_ohlc_rings,
    _rewarm_ohlc_indicators,
    _build_fp_rings,
    _build_pattern_stores_live,
    _build_book_rings,
    _build_mbo_rings,
)
from tradetropy.live._loop import (                      # noqa: E402
    _run_guarded,
    _flush_tick_proxy,
    _flush_ohlc_proxy,
    _flush_book_proxy,
    _flush_mbo_proxy,
    _loop,
    _simbolos_del_loop,
    _loop_tick,
    _loop_kline,
    _loop_streaming,
    _on_feed_error,
)
from tradetropy.live._feed import (                      # noqa: E402
    _count_closed_candles_rings,
    _strategy_ready,
    _update_ohlc_indicators_on_close,
    _update_ohlc_indicators_partial,
    _update_pattern_stores_live,
    _process_tick,
    _process_bar,
    _process_event,
    _event_to_tick_row,
    _event_to_kline_row,
)

LiveEngine._compute_required_ticks = _compute_required_ticks
LiveEngine._compute_required_klines = _compute_required_klines
LiveEngine._fetch_auto_history = _fetch_auto_history
LiveEngine._fetch_ticks_for_warmup = _fetch_ticks_for_warmup
LiveEngine._fetch_klines_for_warmup = _fetch_klines_for_warmup

LiveEngine._build_rings = _build_rings
LiveEngine._warm = _warm
LiveEngine._resync_chart_sources = _resync_chart_sources
LiveEngine._topup_ohlc_rings = _topup_ohlc_rings
LiveEngine._rewarm_ohlc_indicators = _rewarm_ohlc_indicators
LiveEngine._build_fp_rings = _build_fp_rings
LiveEngine._build_pattern_stores_live = _build_pattern_stores_live
LiveEngine._build_book_rings = _build_book_rings
LiveEngine._build_mbo_rings = _build_mbo_rings

LiveEngine._run_guarded = _run_guarded
LiveEngine._flush_tick_proxy = _flush_tick_proxy
LiveEngine._flush_ohlc_proxy = _flush_ohlc_proxy
LiveEngine._flush_book_proxy = _flush_book_proxy
LiveEngine._flush_mbo_proxy = _flush_mbo_proxy
LiveEngine._loop = _loop
LiveEngine._simbolos_del_loop = _simbolos_del_loop
LiveEngine._loop_tick = _loop_tick
LiveEngine._loop_kline = _loop_kline
LiveEngine._loop_streaming = _loop_streaming
LiveEngine._on_feed_error = _on_feed_error

LiveEngine._count_closed_candles_rings = _count_closed_candles_rings
LiveEngine._strategy_ready = _strategy_ready
LiveEngine._update_ohlc_indicators_on_close = _update_ohlc_indicators_on_close
LiveEngine._update_ohlc_indicators_partial = _update_ohlc_indicators_partial
LiveEngine._update_pattern_stores_live = _update_pattern_stores_live
LiveEngine._process_tick = _process_tick
LiveEngine._process_bar = _process_bar
LiveEngine._process_event = _process_event
LiveEngine._event_to_tick_row = _event_to_tick_row
LiveEngine._event_to_kline_row = _event_to_kline_row

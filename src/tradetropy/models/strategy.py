from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Union

import numpy as np

FeedType = Literal["tick", "kline"]
RunMode  = Literal["backtest", "live", "optimize", "pool"]

from tradetropy.data.data import (
    TickProxy,
    OhlcProxy,
    IndicatorProxy,
    ColumnRef,
    WindowView,
    OrderbookProxy,
    MboProxy,
)
from tradetropy.ta.base import Indicator
from tradetropy.ta.trend import SMA, EMA
from tradetropy.models.footprint import FpProxy

# Pattern-matching support is a gated (tier-3) feature. The DSL modules are
# absent from lower-tier builds, so the type is imported only for annotations;
# the runtime construction lives in the pattern methods, which import it lazily.
if TYPE_CHECKING:
    from tradetropy.ta.pattern import PatternMatcherProxy

from tradetropy.exceptions import ConfigError, StopEngine, TradingError
from tradetropy.models.decorators import live_only

# if TYPE_CHECKING: moved outside block — Sesh is needed at runtime for type hints
from tradetropy.session.base import Sesh

if TYPE_CHECKING:
    from tradetropy.ta.tool import Tool, VolumeProfileResult


# ══════════════════════════════════════════════════════════════════════════════
# NULL LOGGER
# ══════════════════════════════════════════════════════════════════════════════
#
# No-op logger for modes where output makes no sense
# (optimize, pool with logs disabled).
#
# Implements the exact same public interface as logging.Logger
# plus the custom methods from _logger.py (perf / signal / trading),
# but all operations are empty — zero I/O, zero allocations.
#
# Does not inherit from logging.Logger: prevents the logging module from
# registering anything or creating phantom handlers in child processes.
#
class _NullLogger:
    """Empty logger — same interface as logging.Logger + custom methods."""

    # Custom levels (mirror of logger.py, in case anyone queries them)
    PERF    = 5
    SIGNAL  = 15
    TRADING = 25

    # Standard methods
    def debug(self, *a, **kw):     pass
    def info(self, *a, **kw):      pass
    def warning(self, *a, **kw):   pass
    def error(self, *a, **kw):     pass
    def critical(self, *a, **kw):  pass
    def exception(self, *a, **kw): pass
    def log(self, *a, **kw):       pass

    # Custom methods from logger.py
    def perf(self, *a, **kw):    pass
    def signal(self, *a, **kw):  pass
    def trading(self, *a, **kw): pass

    # Common utilities
    def isEnabledFor(self, level: int) -> bool: return False
    def setLevel(self, level):                  pass
    def addHandler(self, hdlr):                 pass
    def removeHandler(self, hdlr):              pass

    @property
    def handlers(self) -> list:                 return []

    def __repr__(self) -> str:
        return "NullLogger()"


# Singleton — one instance for the whole session; no mutable state.
_NULL_LOGGER = _NullLogger()

# Type annotation for self.log
_AnyLogger = Union[logging.Logger, _NullLogger]


# ══════════════════════════════════════════════════════════════════════════════
# RECORD CONFIG (internal)
# ══════════════════════════════════════════════════════════════════════════════
#
# Internal configuration for live data recording.
# Not part of the public API — users only see the `record` and
# `record_flush_every` parameters in subscribe_ticks / subscribe_ohlc.
#
@dataclass
class _RecordConfig:
    """Internal recording configuration. Do not expose in public API."""
    path: Path
    flush_every: int = 1000
    _buffer: list = field(default_factory=list)
    _meta_written: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY BASE
# ══════════════════════════════════════════════════════════════════════════════
#
# Single callback: on_data().
# The strategy doesn't know whether it receives a tick or a bar — it doesn't care.
# The broker already has updated prices when on_data() is called.
#
class Strategy:
    """
    Base class for strategies. Compatible with BacktestEngine, LiveEngine
    and PoolBacktestEngine.

    BUILT-IN LOGGER
    ────────────────
    Access the logger through ``self.log``. It is created lazily the first
    time it is accessed — no cost if never called.

    Customization via class attributes in the subclass:

        class MyStrategy(Strategy):
            log_color           = "#FF6B35"        # color of the {name} field
            log_file            = "logs/bt.log"    # None → console only
            log_level           = SIGNAL           # minimum level
            log_enabled_in_pool = True             # enable logs in pool

    Or at runtime, before self.log is first accessed
    (typically in the subclass __init__):

        def __init__(self):
            super().__init__()
            self.log_color = "#FF6B35"

    Behavior by execution mode
    ──────────────────────────────
    ┌──────────────────┬──────────────────────────────────────────────────────┐
    │ run_mode         │ self.log returns …                                  │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ "backtest"       │ Real logger (console; file only if save_log=True)   │
    │ "live"           │ Real logger (console; file if save_log, def True)   │
    │ "optimize"       │ NullLogger — always silent, no exception            │
    │ "pool"           │ NullLogger by default                               │
    │                  │ Real logger if log_enabled_in_pool = True            │
    └──────────────────┴──────────────────────────────────────────────────────┘

    Log file writing (opt-in)
    ──────────────────────────────
    Defining ``log_file`` is NOT enough to write to disk: the file is
    opt-in and controlled by the ``save_log`` parameter of ``engine.run(...)``:

        · BacktestEngine.run / ReplayEngine.run → save_log=False by default.
        · LiveEngine.run                        → save_log=True  by default.
        · In all cases, save_log=True/False forces the behavior.

    If save_log is False, ``self.log`` only emits to console (when a real
    logger exists). This prevents accumulating log files in repeated
    backtests/replays.

    LOGGER METHODS
    ──────────────────
    Standard  : self.log.debug / .info / .warning / .error / .critical
    Custom    : self.log.perf(msg)     — tick-to-tick metrics  (level  5)
                self.log.signal(msg)   — detected signals      (level 15)
                self.log.trading(msg)  — orders / fills        (level 25)

    Minimal example:
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='1m')
                self.sma = self.add_indicator(self.btc.close_ref, SMA(10))

            def on_data(self):
                if self.sma[-1] > self.btc.close[-2]:
                    self.log.signal("bullish cross → %.2f", self.btc.close[-1])
    """

    # ── Class attributes for the logger (override in subclass or __init__) ──
    #: Color of the {name} field in console. Hex str or any _ColorInput.
    log_color: str = "#BFD8FF"
    #: Path to the log file. Only written if the engine runs with
    #: save_log=True (def: True in live, False in backtest/replay). None → never.
    log_file: "str | None" = None
    #: Minimum logging level. Import PERF / SIGNAL / TRADING from logger.
    log_level: int = 5   # PERF — captures everything by default
    #: Enable logger in "pool" mode (disabled by default).
    log_enabled_in_pool: bool = False
    #: Bars/ticks to process before calling on_data() for the first time.
    #: Unified semantics across BacktestEngine, LiveEngine and ReplayEngine:
    #:
    #:   warmup = int (including 0)  → LITERAL value, identical across all modes:
    #:     BacktestEngine
    #:       · Silences on_data() during the first N bars/ticks of the dataset.
    #:       · Indicators are calculated normally — only on_data() is not called.
    #:     LiveEngine
    #:       · Requests N historical bars/ticks from the broker in prepare().
    #:       · If it gets K >= N → on_data() active from the first live tick.
    #:       · If it gets K < N  → waits for N-K additional live bars/ticks.
    #:       · Historical + live sum always reaches N before the first on_data().
    #:     ReplayEngine
    #:       · Reserves N ticks/bars from the dataset as simulated history;
    #:         on_data() starts at tick/bar N (same logical point as backtest).
    #:
    #:   warmup = None (default) → AUTO across all modes with the same criterion:
    #:     max(min_periods) of OHLC indicators (pure, in bars). Guarantees
    #:     that on_data() won't start with cold indicators and that backtest,
    #:     live and replay coincide at the starting point with the same data.
    #:
    #:   In LiveEngine.by_ticks(), if the broker cannot provide the requested
    #:   amount of historical ticks, warmup is completed automatically with
    #:   available OHLC bars. See LiveEngine documentation for details.
    #:
    #: Note: PoolBacktestEngine (massive optimization) always auto-calculates
    #: from min_periods and ignores an explicit warmup — the exact starting
    #: point is not relevant in that path.
    warmup: "int | None" = None

    def __init__(self):
        self._tick_proxies:         list[TickProxy]   = []
        self._ohlc_proxies:         list[OhlcProxy]   = []
        self._book_proxies:         list[OrderbookProxy] = []
        self._mbo_proxies:          list[MboProxy]    = []
        self._indicator_defs:       list[dict]        = []
        self._fp_proxies:           list[FpProxy]     = []
        self._pattern_matcher_defs: list              = []  # list[PatternMatcherDef]
        # Snapshots accumulated by use_tool() (plot=True) so the engine can
        # draw them after run(). Each item: {"tool", "result", "overrides"}.
        self._tool_snapshots:       list[dict]        = []
        self._broker:               "BrokerSimulator | None" = None
        self._sesh:                 "Sesh | None"     = None
        self._feed_type:            FeedType          = "tick"
        self._run_mode:             RunMode           = "backtest"
        self._in_on_data:           bool              = False
        self._custom_params:        dict              = {}
        self._verbose:              bool              = True
        # Log file writing (opt-in). The engine sets this according to its
        # policy (backtest/replay → False, live → True). When False,
        # log_file is ignored and self.log only emits to console (if verbose).
        self._save_log:             bool              = False

        # Lazy logger — None means "not yet created".
        # Reset to None each time run_mode changes.
        self._logger: "_AnyLogger | None" = None


    # ══════════════════════════════════════════════════════════════════════════
    # LOGGER
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def log(self) -> "_AnyLogger":
        """
        Ready-to-use logger. Lazy: created the first time it is accessed.

        Returns NullLogger (zero I/O) in "optimize" mode and in "pool" mode
        unless ``log_enabled_in_pool = True`` is defined on the class.
        In any other mode returns a real Logger with the PERF / SIGNAL /
        TRADING levels registered.

        The logger is automatically invalidated when the engine changes the
        run_mode (through the setter), so it is always consistent with
        the current execution mode.
        """
        if self._logger is None:
            self._logger = self._crear_logger()
        return self._logger


    def _crear_logger(self) -> "_AnyLogger":
        if self._run_mode == "optimize":
            return _NULL_LOGGER
        if self._run_mode == "pool" and not self.log_enabled_in_pool:
            return _NULL_LOGGER
        if not self._verbose:
            return _NULL_LOGGER
        try:
            from tradetropy.logger import get_strategy_logger, wrap_backtest_logger
        except ImportError:
            return _NULL_LOGGER

        # Log file is opt-in: only written if _save_log is active.
        # Having log_file defined is NOT enough — prevents silent writes in
        # backtest/replay from inheriting log_file from the class.
        log_file_efectivo = self.log_file if (self._save_log and self.log_file) else None

        # Presentation zone for the log time column. Read from the session's
        # display_tz so the log matches sesh.str_time in every run mode
        # (backtest / live / replay). Falls back to UTC when unavailable.
        from datetime import timezone as _timezone
        display_tz = getattr(self._sesh, "display_tz", _timezone.utc) or _timezone.utc

        logger = get_strategy_logger(
            name       = self.__class__.__name__,
            name_color = self.log_color,
            log_file   = log_file_efectivo,
            level      = self.log_level,
            display_tz = display_tz,
        )

        # In backtest, replace the clock timestamp with the data timestamp
        if self._run_mode == "backtest" and self._sesh is not None:
            sesh_ref = self._sesh  # capture for the closure
            logger = wrap_backtest_logger(
                logger,
                ts_provider=lambda: getattr(sesh_ref, "_ultimo_ts", 0),
            )

        return logger

    def set_log_file(self, path: str | None) -> None:
        self.log_file = path
        if self._logger is not None:
            self._reset_logger()

    def _set_save_log(self, value: bool) -> None:
        """Internal use by engines: enables/disables log file writing and
        invalidates the lazy logger if changed, so self.log rebuilds
        with (or without) FileHandler as appropriate."""
        if self._save_log != value:
            self._save_log = value
            if self._logger is not None:
                self._reset_logger()

    def _reset_logger(self) -> None:
        import logging as _logging
        existing = _logging.getLogger(self.__class__.__name__)
        existing.handlers.clear()
        self._logger = None

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL PROPERTIES
    # ══════════════════════════════════════════════════════════════════════════

    def __repr__(self):
        params = f"({', '.join([f'{k}:{v}' for k, v in self._custom_params.items()])})"
        return f"{self.__class__.__name__}{params}"

    @property
    def feed_type(self) -> FeedType:
        """Feed type injected by the engine: 'tick' or 'kline'."""
        return self._feed_type

    @property
    def run_mode(self) -> RunMode:
        """Execution environment injected by the engine. Read-only."""
        return self._run_mode

    def _set_run_mode(self, value: RunMode) -> None:
        """Internal use by engines: changes the mode and invalidates the lazy logger."""
        if self._run_mode != value:
            self._run_mode = value
            self._reset_logger()

    @property
    def is_live(self) -> bool:
        """True if running in production (LiveEngine)."""
        return self._run_mode == "live"

    @property
    def is_backtest(self) -> bool:
        """True if running in backtest (BacktestEngine / PoolBacktestEngine)."""
        return self._run_mode in ("backtest", "pool", "optimize")

    def run_if(self, *, live: Callable | None = None, backtest: Callable | None = None):
        """
        Executes the callable for the current environment and returns its result.
        The callable for the other environment is NEVER executed.
        """
        fn = live if self._run_mode == "live" else backtest
        return fn() if fn is not None else None

    # ── On-demand data fetch (live only, anti-lookahead) ──────────────────────
    # Thin ergonomic wrappers over the session's public fetch API. The
    # authoritative guard is run_mode: fetch is allowed ONLY in live, so a
    # backtest / optimize / pool / replay run can never pull current ("future")
    # data on demand. The session also self-guards via supports_data_fetch, so
    # the two layers reinforce each other.

    def _require_live_fetch(self, op: str) -> None:
        """
        Guard the fetch API to live mode only, to prevent lookahead bias.

        Args:
            op (str): The public method name, for the error message.

        Raises:
            TradingError: If no session is configured.
            ConfigError: If run_mode is not 'live'.
        """
        if self._sesh is None:
            raise TradingError("No session configured")
        if self._run_mode != "live":
            raise ConfigError(
                f"self.{op}() is only available in live mode "
                f"(run_mode='live'), not in {self._run_mode!r}. On-demand data "
                f"fetch is disabled in backtest/optimize/pool/replay to prevent "
                f"lookahead bias - the backtest would otherwise pull current "
                f"'future' data. Preload historical data instead (a KlineData/"
                f"TickData passed to the engine, or tradetropy.connectors.ccxt."
                f"fetch_klines/fetch_ticks for offline preparation)."
            )

    def fetch_klines(self, symbol: str, timeframe, *, limit: int = 200):
        """
        Fetch recent OHLC candles on demand from the venue (live only).

        Delegates to the live session's fetch_klines. Raises ConfigError in any
        non-live run_mode (anti-lookahead).

        Args:
            symbol (str): Trading symbol.
            timeframe (int | str): Candle interval ('1m', '1h', ... or ms).
                Standard recommended set: '1m', '15m', '1h', '4h', '1d',
                '1w', '1mo' ('min'/'wk' accepted as aliases for 'm'/'w').
            limit (int): Maximum number of candles. Default 200.

        Returns:
            KlineData: The fetched candles.

        Example:
            klines = self.fetch_klines('BTC/USDT', '1h', limit=500)
        """
        self._require_live_fetch("fetch_klines")
        return self._sesh.fetch_klines(symbol, timeframe, limit=limit)

    def fetch_ticks(self, symbol: str, *, limit: int = 500):
        """
        Fetch recent public trades (ticks) on demand from the venue (live only).

        Delegates to the live session's fetch_ticks. Raises ConfigError in any
        non-live run_mode (anti-lookahead).

        Args:
            symbol (str): Trading symbol.
            limit (int): Maximum number of trades. Default 500.

        Returns:
            TickData: The fetched ticks.

        Example:
            ticks = self.fetch_ticks('BTC/USDT', limit=1000)
        """
        self._require_live_fetch("fetch_ticks")
        return self._sesh.fetch_ticks(symbol, limit=limit)

    def fetch_orderbook(
        self, symbol: str, *, depth: int = 20, tick_size: float = 0.01
    ):
        """
        Fetch a single L2 order-book image on demand from the venue (live only).

        Delegates to the live session's fetch_orderbook, returning a BookData
        that round-trips through tradetropy.io (save/read_book). Raises ConfigError
        in any non-live run_mode (anti-lookahead).

        Args:
            symbol (str): Trading symbol.
            depth (int): Number of book levels K per side. Default 20.
            tick_size (float): Minimum price step propagated to BookData.

        Returns:
            BookData: One book event (a REST snapshot is a single row).

        Example:
            book = self.fetch_orderbook('BTC/USDT', depth=20)
        """
        self._require_live_fetch("fetch_orderbook")
        return self._sesh.fetch_orderbook(symbol, depth=depth, tick_size=tick_size)

    @property
    def ts(self) -> int:
        """Timestamp in ms of the last processed data point."""
        if self._sesh is None:
            raise TradingError("No session configured")
        return self._sesh.ts

    @property
    def time(self) -> datetime:
        """Current datetime (timezone-aware)."""
        if self._sesh is None:
            raise TradingError("No session configured")
        return self._sesh.time

    @property
    def str_time(self) -> str:
        """Formatted string '%Y-%m-%d %H:%M:%S'."""
        if self._sesh is None:
            raise TradingError("No session configured")
        return self._sesh.str_time

    def update_params(self, params: dict | None = None):
        if params is None:
            return {}
        for k, v in params.items():
            if not hasattr(self, k):
                raise ConfigError(
                    f"Strategy '{self.__class__.__name__}' has no parameter '{k}'. "
                    "Strategy class parameters must be defined as class variables "
                    "before they can be optimized or executed."
                )
            setattr(self, k, v)
            self._custom_params[k] = v
        return self._custom_params

    @classmethod
    def set_warmup(cls, n: "int | None"):
        cls.warmup = n

    # ══════════════════════════════════════════════════════════════════════════
    # SUBSCRIPTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def subscribe_ticks(
        self,
        symbol: str,
        window_size: int = 1000,
        record: "str | Path | None" = None,
        record_flush_every: int = 500,
    ) -> TickProxy:
        proxy = TickProxy(symbol, window_size)
        if record is not None:
            proxy._record_config = _RecordConfig(
                path=Path(record),
                flush_every=record_flush_every,
            )
        self._tick_proxies.append(proxy)
        return proxy

    def subscribe_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        window_size: int = 5000,
        record: "str | Path | None" = None,
        record_flush_every: int = 500,
    ) -> OrderbookProxy:
        """
        Subscribe to the real-time L2 order book (depth) of a symbol.

        Returns an OrderbookProxy exposing top-of-book metrics (imbalance,
        spread, mid, best_bid/ask) and a causal book_as_of() for order-flow
        detectors. The engine feeds it from the streaming feed's order-book
        events. Available on streaming-capable live sessions and in replay of a
        recorded book; in plain backtest the book stays empty (stale).

        Args:
            symbol (str): Trading symbol.
            depth (int): Number of book levels K retained per side.
            window_size (int): Max number of book events kept for book_as_of.
            record (str | Path | None): If set, record every book event to this
                HDF5 file for later replay (read_book / ReplayFeed.from_records).
            record_flush_every (int): Flush the record buffer every N events.

        Returns:
            OrderbookProxy: Proxy to read the live book in on_data().
        """
        proxy = OrderbookProxy(symbol, depth=depth, window_size=window_size)
        if record is not None:
            proxy._record_config = _RecordConfig(
                path=Path(record),
                flush_every=record_flush_every,
            )
        self._book_proxies.append(proxy)
        return proxy

    def subscribe_mbo(
        self,
        symbol: str,
        window_size: int = 50_000,
        record: "str | Path | None" = None,
        record_flush_every: int = 1000,
    ) -> MboProxy:
        """
        Subscribe to the L3 / market-by-order (per-order) stream of a symbol.

        Returns an MboProxy exposing the recent per-order event window and
        reconstructed resting size per level - the data the L3 order-flow
        detectors (iceberg reloads, liquidity grabs) need. Available where the
        venue/feed delivers MBO and in replay of a recorded MBO log; empty
        otherwise.

        Args:
            symbol (str): Trading symbol.
            window_size (int): Max number of MBO events retained.
            record (str | Path | None): If set, record every MBO event to this
                HDF5 file for later replay (read_mbo).
            record_flush_every (int): Flush the record buffer every N events.

        Returns:
            MboProxy: Proxy to read the L3 stream in on_data() / DeepTrades.
        """
        proxy = MboProxy(symbol, window_size=window_size)
        if record is not None:
            proxy._record_config = _RecordConfig(
                path=Path(record),
                flush_every=record_flush_every,
            )
        self._mbo_proxies.append(proxy)
        return proxy

    def subscribe_ohlc(
        self,
        symbol: str,
        timeframe: "str | int | None" = None,
        window_size: int = 300,
        record: "str | Path | None" = None,
        record_flush_every: int = 100,
    ) -> OhlcProxy:
        """
        Subscribe to OHLC candles for a symbol.

        Args:
            symbol (str): Trading symbol (e.g. 'BTCUSDT').
            timeframe (str | int): Candle duration. Accepts a timeframe string
                (e.g. '1m', '5m', '1h', '1d') or an integer number of
                milliseconds, parsed via parse_timeframe(). Standard
                recommended set: '1m', '15m', '1h', '4h', '1d', '1w', '1mo'
                ('min'/'wk' accepted as aliases for 'm'/'w'; 'mo' is a fixed
                30-day month).
            window_size (int): Maximum number of closed bars kept in memory
                (default 300).
            record (str | Path | None): Path to record the stream to HDF5.
            record_flush_every (int): How often to flush the recording buffer.

        Returns:
            OhlcProxy: Proxy giving access to the OHLC window.

        Raises:
            ConfigError: If timeframe is not provided or is invalid.

        Example:
            self.btc = self.subscribe_ohlc('BTCUSDT', timeframe='1m')
            self.btc = self.subscribe_ohlc('BTCUSDT', 60_000)
        """
        from tradetropy.core.constants import parse_timeframe as _parse_tf

        if timeframe is None:
            raise ConfigError(
                "subscribe_ohlc() requires a timeframe "
                "(e.g. timeframe='1m' or 60_000)."
            )
        interval_ms = _parse_tf(timeframe)

        proxy = OhlcProxy(symbol, interval_ms, window_size)
        if record is not None:
            proxy._record_config = _RecordConfig(
                path=Path(record),
                flush_every=record_flush_every,
            )
        self._ohlc_proxies.append(proxy)
        return proxy

    def subscribe_footprint(
        self,
        symbol: str,
        timeframe: "str | int | None" = None,
        window_size: int = 50,
        *,
        tick_size: float | None = None,
        levels: int = 4,
        value_area_pct: float = 0.70,
        aggressor_col: str | None = "flags",
        vol_col: str = "volume",
    ) -> FpProxy:
        """
        Subscribe to the footprint (volume profile) of a symbol at the given interval.

        Does not require calling subscribe_ohlc() first — the engine builds
        the tick→bar mapping internally if no compatible OhlcProxy exists.
        If one does exist, it reuses the already-computed mapping.

        Args:
            symbol (str): Trading symbol (e.g. "BTCUSDT").
            timeframe (str | int): Candle duration. Accepts a timeframe string
                (e.g. '1m', '5m', '1h', '1d') or an integer number of
                milliseconds, parsed via parse_timeframe(). Standard
                recommended set: '1m', '15m', '1h', '4h', '1d', '1w', '1mo'
                ('min'/'wk' accepted as aliases for 'm'/'w'; 'mo' is a fixed
                30-day month).
            window_size (int): Maximum number of closed bars kept in memory.

        Configuration parameters (keyword-only):
            tick_size (float | None): Price level grouping size (e.g. 10 for BTC).
                If None (default), automatically inferred as
                average_bar_range / levels, rounded to a nice value.
            levels (int): Desired levels per bar for auto-inference (default 20).
                Only used when tick_size=None.
            value_area_pct (float): Percentage of volume defining the value area
                (default 0.70).
            aggressor_col (str | None): Tick column for aggressor classification
                (default "flags").
            vol_col (str): Volume column to use (default "volume").

        Raises:
            ConfigError: If timeframe is not provided or is invalid.

        Example:
            self.fp = self.subscribe_footprint('BTCUSDT', timeframe='5m')
            self.fp = self.subscribe_footprint('BTCUSDT', 300_000)
        """
        from tradetropy.core.constants import parse_timeframe as _parse_tf

        if timeframe is None:
            raise ConfigError(
                "subscribe_footprint() requires a timeframe "
                "(e.g. timeframe='5m' or 300_000)."
            )
        interval_ms = _parse_tf(timeframe)

        proxy = FpProxy(
            symbol,
            interval_ms,
            window_size,
            tick_size=tick_size,
            levels=levels,
            value_area_pct=value_area_pct,
            aggressor_col=aggressor_col,
            vol_col=vol_col,
        )
        self._fp_proxies.append(proxy)
        return proxy

    def add_indicator(
        self,
        source: "ColumnRef | list[ColumnRef] | OhlcProxy | TickProxy",
        indicator: Indicator,
        **plot_overrides,
    ) -> "IndicatorProxy | MultiBandProxy":
        """
        Declares an indicator and returns the proxy to access its values.

        ``source`` can be:
            - ColumnRef           — a single column (e.g. ``proxy.close_ref``).
            - list[ColumnRef]     — multiple columns from the same proxy.
            - OhlcProxy/TickProxy — the proxy directly; the indicator resolves
                                     its columns via ``default_refs(proxy)`` (only
                                     indicators that implement it, e.g.
                                     VolumeProfile / RollingVolumeProfile).

        Visualization parameters (optional, override the defaults defined
        in the indicator's plot_config):

            plot          : bool — False to skip rendering
            overlay       : bool | None — True=on OHLC, False=own panel
            name          : str | list[str] — legend label.
                            str → single entry controlling all dimensions.
                            list[str] → one entry per dimension.
            panel_height  : int — own panel height in px
            panel_title   : str | None — Y-axis title in own panel.
                            None → uses name (if str) or display_name().
            zorder        : int — rendering order (higher = on top)
            color         : str | list[str] — CSS color(s) per dimension
            line_width    : float | list[float] — line width per dimension
            line_dash     : str | list[str] — line style per dimension
            line_alpha    : float | list[float] — transparency per dimension
            scatter       : bool — True to render as points
            marker        : str | list[str] — marker shape per dimension
            marker_size   : int | list[int] — marker size per dimension
            marker_alpha  : float | list[float] — transparency per dimension
            marker_fill   : bool | list[bool] — fill per dimension
            marker_line_width : float | list[float] — border width per dimension
            reference_lines   : list[dict] — horizontal reference lines
        """
        # ── Normalize source to list ──────────────────────────────────────────
        if isinstance(source, (OhlcProxy, TickProxy)):
            if not hasattr(indicator, "default_refs"):
                raise ConfigError(
                    f"{type(indicator).__name__} does not accept a direct proxy. "
                    "Pass a ColumnRef (e.g. proxy.close_ref) or list[ColumnRef]."
                )
            sources = indicator.default_refs(source)
            if not sources or not all(isinstance(f, ColumnRef) for f in sources):
                raise ConfigError(
                    f"{type(indicator).__name__}.default_refs() must return "
                    "a non-empty list[ColumnRef]."
                )
        elif isinstance(source, ColumnRef):
            sources = [source]
        elif isinstance(source, list):
            if not source or not all(isinstance(f, ColumnRef) for f in source):
                raise ConfigError("source must be ColumnRef or non-empty list[ColumnRef]")
            sources = source
        else:
            raise ConfigError(
                f"source must be a proxy, ColumnRef or list[ColumnRef], got {type(source)}"
            )

        if not isinstance(indicator, Indicator):
            raise ConfigError(f"indicator must be an Indicator instance, got {type(indicator)}")

        # Validate that all ColumnRef belong to the same proxy
        proxies_ids = {id(f._proxy) for f in sources}
        if len(proxies_ids) > 1:
            raise ConfigError(
                "All ColumnRef must come from the same proxy. "
                "Cannot mix sources from different symbols or intervals."
            )

        # ── Resolve final plot_config: indicator defaults + overrides ────────
        plot_config = (
            indicator.plot_config.merged(**plot_overrides)
            if plot_overrides
            else indicator.plot_config
        )

        n_outputs = indicator.n_outputs

        if n_outputs > 1:
            from tradetropy.data.data import MultiBandProxy
            price_output_names = indicator.output_names or [
                f"output_{i}" for i in range(n_outputs - len(indicator.ts_band_indices))
            ]
            proxy = MultiBandProxy(price_output_names)
            # Inject ts metadata for attribute access (opt-in)
            ts_names = getattr(indicator, "ts_output_names", [])
            if ts_names:
                proxy._ts_output_names = list(ts_names)
                proxy._ts_band_indices = list(indicator.ts_band_indices)
            # On-demand access to HVN/LVN nodes (volume profiles with nodes!=None).
            if hasattr(indicator, "compute_nodes") and getattr(indicator, "nodes", None):
                proxy._set_node_provider(indicator, sources[0]._proxy)
            # On-demand public query API (e.g. Heatmap.liquidity_at / hottest).
            if getattr(indicator, "exposes_query_api", False):
                proxy._set_query_provider(indicator)
                # Opt-in: give the indicator the source proxy so its query
                # methods can recompute causally from the current window
                # (e.g. CandlePatterns.last_pattern / efficacy).
                if hasattr(indicator, "set_query_source"):
                    indicator.set_query_source(sources[0]._proxy)
        else:
            proxy = IndicatorProxy()

        self._indicator_defs.append({
            "sources":      sources,
            "source":       sources[0],
            "indicator":    indicator,
            "proxy":        proxy,
            "multi_source": len(sources) > 1,
            "multi_band":   n_outputs > 1,
            "plot_config":  plot_config,
        })
        return proxy

    def use_tool(
        self,
        source: "TickProxy | OhlcProxy",
        tool: "Tool",
        *,
        start=None,
        end=None,
        plot: bool = True,
        **plot_overrides,
    ):
        """
        Run an on-demand analysis tool over ``source`` and return its snapshot.

        A pull operation fully controlled by the strategy: call it inside
        ``on_data()`` with the source, the tool instance and the range. Nothing
        is precomputed and plotting is skipped in optimize/pool.

            def on_data(self):
                vp = self.use_tool(
                    self.ticks, FixedRangeVP(nodes="both"),
                    start=self.ts - 3_600_000, end=self.ts,
                )
                if vp and self.ticks.price[-1] > vp.vah:
                    self.sesh.buy("BTCUSDT", volume=1)

        When ``plot`` is True the snapshot is stored for rendering; tool
        snapshots are grouped into one legend entry per tool type (e.g. "VP",
        "Fib"), so one legend click toggles all of that tool's drawings in both
        backtest and livechart. Per-call ``**plot_overrides`` (e.g. name, color,
        line_width) tweak the tool's plot config for that snapshot.

        Args:
            source (TickProxy | OhlcProxy): Data the tool reads its slice from.
            tool (Tool): A configured tool instance (e.g. FixedRangeVP(nodes=...)).
            start: Range start (epoch ms, datetime, ISO str, or None=oldest row).
            end: Range end (epoch ms, datetime, ISO str, or None=newest row).
            plot (bool): If True, store the snapshot for rendering. Default True.
            **plot_overrides: Per-snapshot visualization overrides (e.g. color).

        Returns:
            The computed snapshot (e.g. VolumeProfileResult). Falsy when the
            requested range has no data in the source buffer.
        """
        if not self._in_on_data:
            raise ConfigError(
                "use_tool() can only be called inside on_data(). "
                "Stores and proxy connections do not exist in init()/declare()."
            )
        result = tool.run(source, start=start, end=end)
        # No rendering in optimize/pool: no plotting and avoids accumulating memory
        # in child processes of the parameter sweep.
        if plot and result and self._run_mode not in ("optimize", "pool"):
            self._tool_snapshots.append({
                "tool": tool,
                "result": result,
                "overrides": dict(plot_overrides),
            })
        return result

    def add_pattern_matcher(
        self,
        base_pivot,
        pattern: "Pattern | str",
        *,
        decorators: list | None = None,
        tag: str = "pattern",
    ) -> "PatternMatcherProxy":
        """
        Declares a pattern matcher and returns the proxy to use it in on_data().

        The engine builds the PatternStore (once) before the backtest/live loop
        and connects the proxy. In on_data() just access ``.last``
        (O(log n) in backtest, O(1) in live).

        ──────────────────────────────────────────────────────────────────
        ═══════════  DSL (STRING-BASED)  ═════════════════════════════════
        ──────────────────────────────────────────────────────────────────

        You can pass a DSL string instead of instantiating Pattern manually.
        The DSL is parsed automatically with ``parse_pattern()``:

            self.setup = self.add_pattern_matcher(
                base_pivot=self.cpivot,
                decorators=[self.nbs],
                pattern = \"\"\"
                    L[nbs=boo]
                    H[nbs=neu]  > $0
                \"\"\",
                tag="impulse",
            )

        Format of each line::

            TYPE[?] [TAGS] [CONDITIONS]

        TYPE
            H | L | any

            Expected pivot type:
            · 'H'   → high pivots only
            · 'L'   → low pivots only
            · 'any' → both (use when type doesn't matter but tags or
                       conditions do)

        ? — optional suffix (immediately after TYPE, no space)
            Marks the node as optional. The pattern can match whether
            or not the pivot exists.

            Examples:  H?   L?[nbs=boo]   any? > $0

        TAGS — [key=val, key=val, ...]
            Filters the pivot must satisfy.
            Tags come from decorator indicators:
            · 'type' → from ConfirmedPivot ('H' / 'L')
            · 'nbs'  → from NBS indicator ('neu', 'boo', 'shk', 'emp')
            · 'hhll' → from HHLL indicator ('HH', 'HL', 'LH', 'LL')

            Alternative values can be specified with ``|``:
                [nbs=neu|shk]   → tag 'nbs' == 'neu' OR 'shk'

            Examples:
                [nbs=neu]               → 'nbs' exactly 'neu'
                [nbs=neu|shk]           → 'nbs' is 'neu' or 'shk'
                [nbs=neu, hhll=HH]      → AND of two decorators
                [hhll=LL]               → only tag 'hhll'

        CONDITIONS — optional
            Sequence of expressions separated by & (AND) or | (OR).
            Can be grouped with parentheses.

            Operators::

                >   <   >=   <=   ==   !=

            Operands::

                50000           → absolute number (price)
                $0              → NodeRef(0, 'value')   price of node 0
                $0.value        → same as $0
                $0.index        → bar index of node 0
                $0.timestamp    → timestamp of node 0
                $0*1.02         → node 0 price × 1.02
                $-1             → previous node price
                $-1*0.98        → previous node price × 0.98
                type(0)         → type ('H'/'L') of node 0
                type(-1)        → type of previous node

            Tag conditions (tag function)::

                tag(nbs)==neu       → current.tags['nbs'] == 'neu'
                tag(nbs)!=shk       → current.tags['nbs'] != 'shk'
                tag(hhll)==HH       → current.tags['hhll'] == 'HH'
                tag(nbs)==neu|tag(nbs)==shk   → OR between values (more
                  readable with [nbs=neu|shk] in TAGS)

            Time conditions (@ prefix)::

                @between(09:30, 16:00)
                @between(09:30, 16:00, tz=America/New_York)
                @after(14:00)
                @after(09:30, tz=America/New_York)
                @before(22:00)
                @before(16:00, tz=Europe/London)
                @weekday(monday, tuesday, wednesday)
                @weekday(friday, tz=America/New_York)
                @time_since($0.timestamp, min=2, max=168)          # unit=h (default)
                @time_since($0.timestamp, min=5, max=15, unit=m)   # minutes
                @time_since($0.timestamp, min=30, unit=s)          # seconds
                @hours_since($0.timestamp, min=2, max=168)         # alias, always hours
                @hours_since($-1.timestamp, min=0, max=24)

            The lhs of the comparison uses the SAME attribute as the rhs:
                > $0.value     → current_price    > node0_price
                > $0.index     → current_index    > node0_index
                > $0.timestamp → current_timestamp > node0_timestamp

        Comments
            Lines starting with '#' are ignored.

        Anchor end
            If the last line of the DSL is a standalone ``$``, anchor_end
            is activated: the match is only valid if it ends at the LAST
            confirmed pivot in the sequence (useful for detecting the exact
            end of a pattern).

        Complete DSL examples::

            # Minimal — 2 nodes
            pattern = \"\"\"
                L
                H > $0
            \"\"\"

            # With NBS tags and price/time conditions
            pattern = \"\"\"
                L[nbs=boo]  @between(09:30, 16:00, tz=America/New_York)
                H[nbs=neu]  > $0 & @between(09:30, 16:00, tz=America/New_York)
                L[nbs=boo]  > $0 & @hours_since($1.timestamp, min=0, max=48)
                H[nbs=neu]  > $1
            \"\"\"

            # With optional node and anchor end
            pattern = \"\"\"
                H[nbs=neu]
                L?[nbs=boo]           # optional pullback
                H[nbs=neu]  > $0
                $
            \"\"\"

            # OR between grouped conditions
            pattern = \"\"\"
                L
                H  > $0 & (@between(09:30, 16:00, tz=America/New_York) | @after(20:00))
            \"\"\"

            # NodeRef by index and multiplier
            pattern = \"\"\"
                H[nbs=neu, hhll=HH]
                L[nbs=boo, hhll=HL]  > $0*0.95
                H[nbs=neu, hhll=HH]  > $1
            \"\"\"

        ──────────────────────────────────────────────────────────────────
        ═══════════  OBJECT-BASED API  ═══════════════════════════════════
        ──────────────────────────────────────────────────────────────────

        If you prefer the Python API, import the classes from
        ``tradetropy.ta.pattern``:

            from tradetropy.ta.pattern import (
                Pattern, PatternNode,
                Condition, ConditionAnd, ConditionOr,
                NodeRef, NodeTypeRef, TagCondition, TimeCondition,
            )

        ── Pattern ────────────────────────────────────────────────────
        Ordered sequence of PatternNodes with an identifying tag.

            pattern = Pattern(
                nodes=[...],
                tag="pattern_name",
                anchor_end=False,   # optional
            )

        Properties:
            · length     — total nodes (mandatory + optional)
            · min_length — mandatory nodes only
            · tag        — pattern name
            · anchor_end — True only if the match must end at the
                           last confirmed pivot

        ── PatternNode ────────────────────────────────────────────────
        Describes a pivot within a pattern.

            PatternNode(
                type='H',                             # 'H' | 'L' | 'any'
                tag_filters={'nbs': 'neu'},           # dict of required tags
                conditions=[Condition(...)],           # implicit AND list
                optional=False,                        # True if optional
            )

        · type : expected type.  'any' accepts both H and L.
        · tag_filters : dict {decorator_name: expected_value}.
          The pivot must have ALL specified tags.
          E.g. {'nbs': 'neu', 'hhll': 'HH'}.
          {} = no filter.
        · conditions : list of ConditionExpr with implicit AND.
          For OR use ConditionOr.
        · optional : if True, this node may be present or absent
          in the match. The pattern is considered valid in both cases.
          NodeRefs pointing to an absent optional node return False
          safely (the candidate is discarded).

        ── Condition ──────────────────────────────────────────────────
        Atomic condition: compares the current node's attribute against
        an operand.

            Condition('>', 3500)                          # absolute price
            Condition('<', NodeRef(0, 'value'))            # relative to node 0
            Condition('>', NodeRef(0, 'value', 1.02))     # > 2% above node 0
            Condition('>', NodeRef(-1, 'value'))           # > previous node
            Condition('>', NodeRef(0, 'index'))            # current index > node 0 index
            Condition('<', NodeRef(-1, 'timestamp'))       # current ts < previous node ts
            Condition('==', NodeTypeRef(0))                # same type as node 0
            Condition('!=', NodeTypeRef(-1))               # opposite type from previous

        The lhs is derived from the same attribute as the rhs (value, index,
        timestamp, or type for NodeTypeRef), ensuring coherent units.

        ── NodeRef ────────────────────────────────────────────────────
        Reference to another node's attribute in the pattern.

            NodeRef(position, attribute='value', multiplier=1.0)

        · position : >= 0 → absolute ($0, $1, $2...)
                     < 0  → relative ($-1 = previous node)
        · attribute : 'value' | 'index' | 'timestamp'
        · multiplier : factor applied to the referenced value.
          E.g. NodeRef(0, 'value', 0.98) → 98% of node 0's price

        ── NodeTypeRef ────────────────────────────────────────────────
        Reference to the type ('H' or 'L') of another node.

            NodeTypeRef(position)
            Condition('==', NodeTypeRef(0))   # same type as node 0
            Condition('!=', NodeTypeRef(-1))  # opposite type from previous

        Only supports '==' and '!=' operators.

        ── TimeCondition ──────────────────────────────────────────────
        Time condition on the current node's timestamp.

            'between'
                TimeCondition('between', start='09:30', end='16:00',
                              tz='America/New_York')
                Overnight range: start='22:00', end='04:00'

            'after'
                TimeCondition('after', start='14:00')
                TimeCondition('after', start='09:30', tz='America/New_York')

            'before'
                TimeCondition('before', end='22:00')
                TimeCondition('before', end='16:00', tz='Europe/London')

            'weekday'
                TimeCondition('weekday',
                              days=['monday','tuesday','wednesday'])
                TimeCondition('weekday', days=['friday'],
                              tz='America/New_York')
                Valid days: monday..sunday or mon..sun

            'hours_since'
                TimeCondition('hours_since',
                              ref=NodeRef(0, 'timestamp'),
                              min_hours=2, max_hours=48)
                TimeCondition('hours_since',
                              ref=NodeRef(-1, 'timestamp'),
                              min_hours=0, max_hours=24)
                max_hours=None = no upper limit

                In the DSL this is written @time_since($N.timestamp,
                min=..., max=..., unit=h|m|s) (minutes/seconds converted to
                hours by the parser); @hours_since is the hours-only alias.

        ── TagCondition ────────────────────────────────────────────────
        Condition on a tag value of the current node.
        Useful for OR between tags of different keys (nbs vs hhll).

            TagCondition('nbs',  '==', 'neu')
            TagCondition('hhll', '==', 'HH')
            TagCondition('nbs',  '!=', 'shk')

            # OR between tags of different keys
            ConditionOr([
                TagCondition('nbs',  '==', 'neu'),
                TagCondition('hhll', '==', 'HH'),
            ])

        ── ConditionAnd / ConditionOr ─────────────────────────────────
        Logical combinators for ConditionExpr.

            # (A AND B) OR C
            ConditionOr([
                ConditionAnd([
                    Condition('>', 3500),
                    TimeCondition('between', '09:00', '10:00'),
                ]),
                Condition('==', NodeTypeRef(0)),
            ])

        ──────────────────────────────────────────────────────────────────
        ═══════════  RUNTIME OBJECTS  ═══════════════════════════════════
        ──────────────────────────────────────────────────────────────────

        ── PatternMatcherProxy ────────────────────────────────────────
        The proxy returned by add_pattern_matcher().

            proxy = self.add_pattern_matcher(...)
            match = proxy.last   # MatchResult | None

        Main property:
            .last : MatchResult | None

        ── MatchResult ────────────────────────────────────────────────
        Result of a successful pattern match.
        It is frozen (immutable) and supports indexed access like a tuple.

            match = self.setup.last
            if match:
                n0 = match[0]             # PivotPoint of the first node
                match[-1]                 # last node
                len(match)                # number of matched nodes

                n0.value                  # pivot price
                n0.index                  # bar index
                n0.timestamp              # ts_ms UTC epoch
                n0.type                   # 'H' or 'L'
                n0.tags                   # {"type": "H", "nbs": "neu"}

                match.tag                 # pattern name
                match.first               # first node (PivotPoint)
                match.last_node           # last node (PivotPoint)
                match.indices             # [45, 48, 52]
                match.values              # [3500.0, 3200.0, 3600.0]
                match.timestamps          # [t1, t2, t3]

                # Optional nodes:
                match.matched_optional    # {node_idx: bool, ...} or None
                match.node_map            # {node_idx: PivotPoint|None}

        ── PivotPoint ─────────────────────────────────────────────────
        Represents a confirmed pivot at runtime.
        It is frozen (immutable, hashable).

            PivotPoint(
                index=45,                       # bar index
                timestamp=1234567890000.0,       # ts_ms UTC
                value=3500.0,                    # price
                type='H',                        # 'H' | 'L'
                tags={'type':'H', 'nbs':'neu'},  # combined tags
            )

        Shortcuts from MatchResult:
            match.indices     → [n.index for n in match]
            match.values      → [n.value for n in match]
            match.timestamps  → [n.timestamp for n in match]

        ──────────────────────────────────────────────────────────────────
        ═══════════  PARAMETERS  ════════════════════════════════════════
        ──────────────────────────────────────────────────────────────────

        base_pivot
            Proxy returned by add_indicator() for the base ConfirmedPivot.
            It provides the H/L pivot sequence and is mandatory.

        decorators
            Optional list of pivot indicator proxies such as NBS or HHLL.
            They add tags to the base pivot sequence. All decorators must use
            the same OhlcProxy as ``base_pivot``.

        pattern
            Can be:
            · A DSL string (see DSL section above).
            · A Pattern instance built with the object API.

            If it's a string, the pattern tag is taken from the ``tag``
            parameter. If it's a Pattern, ``pattern.tag`` is used.

        tag
            Pattern name. Used as the tag in MatchResult.
            Only relevant when ``pattern`` is a DSL string;
            if pattern is a Pattern object, this parameter is ignored.

        Returns
            PatternMatcherProxy — use ``.last`` in on_data() to get
            the most recent MatchResult.

        ──────────────────────────────────────────────────────────────────
        ═══════════  COMPLETE EXAMPLES  ═════════════════════════════════
        ──────────────────────────────────────────────────────────────────

        Example 1 — DSL string (recommended for readability)::

            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.cpivot = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    ConfirmedPivot(swing=3),
                )
                self.nbs = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    NBS(swing=3),
                )

                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                decorators=[self.nbs],
                    pattern = \"\"\"
                        # Base support — Booster in NY session
                        L[nbs=boo]  @between(09:30, 16:00, tz=America/New_York)
                        # Bullish impulse — Neutralizer in same timeframe
                        H[nbs=neu]  > $0 & @between(09:30, 16:00, tz=America/New_York)
                        # Second support within 48h
                        L[nbs=boo]  > $0 & @hours_since($1.timestamp, min=0, max=48)
                        # Second impulse — Higher High
                        H[nbs=neu]  > $1
                    \"\"\",
                    tag="bullish_impulse_ny",
                )

            def on_data(self):
                match = self.setup.last
                if match:
                    self.log.signal(
                        "L=%.2f H=%.2f HL=%.2f HH=%.2f",
                        match[0].value, match[1].value,
                        match[2].value, match[3].value,
                    )

        Example 2 — Object API with optionals and OR::

            from tradetropy.ta.pattern import (
                Pattern, PatternNode,
                Condition, ConditionOr,
                NodeRef,
            )

            def init(self):
                # ... ohlc and pivots setup ...

                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                decorators=[self.nbs],
                    pattern = Pattern([
                        PatternNode('H', {'nbs': 'neu'}, []),
                        PatternNode('L', {'nbs': 'boo'}, [],
                                    optional=True),          # optional pullback
                        PatternNode('H', {'nbs': 'neu'}, [
                            Condition('>', NodeRef(0, 'value')),
                            ConditionOr([
                                Condition('>', 3700),
                                Condition('>', NodeRef(0, 'value', 1.02)),
                            ]),
                        ]),
                    ], tag="hh_optional_pullback"),
                )

            def on_data(self):
                match = self.setup.last
                if match:
                    n0 = match[0]
                    n1 = match[1]   # if optional matched, it's a PivotPoint
                    n2 = match[2]

                    # Was the optional node (position 1) present?
                    if match.matched_optional and match.matched_optional.get(1):
                        pullback = match.node_map[1]
                        self.log.info("Pullback at %.2f", pullback.value)

        Example 3 — DSL with anchor_end and optional node::

            def init(self):
                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                decorators=[self.nbs],
                    pattern = \"\"\"
                        H[nbs=neu]
                        L?[nbs=boo]
                        H[nbs=neu]  > $0
                        $
                    \"\"\",
                    tag="anchored_setup",
                )

            def on_data(self):
                match = self.setup.last
                if match and match.last_node.index == self.cpivot.last_pivot_index:
                    # The match ends exactly at the last confirmed pivot
                    self.log.info("Pattern just completed: %s", match.values)

        Example 4 — OR inline in tag_filters and TagCondition::

            def init(self):
                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                decorators=[self.nbs],
                    pattern = \"\"\"
                        # OR inline: nbs='neu' OR 'shk' (same key)
                        H[nbs=neu|shk]  > $0
                        # TagCondition: OR between different keys
                        L  (tag(nbs)==boo | tag(hhll)==LL)
                        # Combined: TagCondition + price
                        H  > $0 & (tag(nbs)==neu | tag(hhll)==HH)
                    \"\"\",
                    tag="mixed_tag_or",
                )

            def on_data(self):
                match = self.setup.last
                if match:
                    self.log.info(
                        "Match: tag=%s values=%s",
                        match.tag, match.values,
                    )
        """
        from tradetropy.ta.pattern.matcher import PatternMatcherProxy, PatternMatcherDef
        from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin

        if decorators is None:
            decorators = []
        elif not isinstance(decorators, (list, tuple)):
            raise ConfigError(
                "'decorators' must be a list or tuple of pivot indicators."
            )

        pivot_list = [base_pivot, *decorators]

        def _indicator_def(proxy):
            for indicator_def in self._indicator_defs:
                if indicator_def["proxy"] is proxy:
                    return indicator_def
            return None

        base_def = _indicator_def(pivot_list[0])
        if base_def is None:
            raise ConfigError(
                "'base_pivot' must be a proxy returned by add_indicator()."
            )

        base_indicator = base_def["indicator"]
        if not isinstance(base_indicator, PivotIndicatorMixin):
            raise ConfigError(
                "'base_pivot' must be a pivot indicator implementing "
                "PivotIndicatorMixin, normally ConfirmedPivot."
            )
        if not getattr(base_indicator, "is_base_pivot", False):
            raise ConfigError(
                "The first pattern matcher pivot must be a base pivot "
                "(ConfirmedPivot). Indicators such as NBS and HHLL belong "
                "in 'decorators'."
            )

        base_source = base_def["sources"][0]._proxy
        for position, decorator_proxy in enumerate(pivot_list[1:], start=1):
            decorator_def = _indicator_def(decorator_proxy)
            if decorator_def is None:
                raise ConfigError(
                    f"Pattern matcher decorator at position {position} must "
                    "be a proxy returned by add_indicator()."
                )

            decorator_indicator = decorator_def["indicator"]
            if not isinstance(decorator_indicator, PivotIndicatorMixin):
                raise ConfigError(
                    f"Pattern matcher decorator at position {position} must "
                    "implement PivotIndicatorMixin."
                )
            if getattr(decorator_indicator, "is_base_pivot", False):
                raise ConfigError(
                    "Only one base pivot is allowed. Additional pivot "
                    "indicators must be decorators."
                )
            if decorator_def["sources"][0]._proxy is not base_source:
                raise ConfigError(
                    "The base pivot and all pattern decorators must use the "
                    "same OHLC subscription."
                )

        if pattern is None:
            raise ConfigError("add_pattern_matcher() requires a pattern.")

        # Accept DSL string directly — no need to import parse_pattern
        if isinstance(pattern, str):
            from tradetropy.ta.pattern.pattern_dsl import parse_pattern
            pattern = parse_pattern(pattern, tag=tag)

        proxy = PatternMatcherProxy()
        defn = PatternMatcherDef(
            base_pivot=pivot_list[0],
            decorators=list(pivot_list[1:]),
            pattern=pattern,
            proxy=proxy,
        )
        self._pattern_matcher_defs.append(defn)
        return proxy

    # ══════════════════════════════════════════════════════════════════════════
    # SESH
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def sesh(self) -> Sesh | None:
        """Session injected by the engine (SeshSimulator in backtest, SeshMT5/SeshLive
        in live trading). None if no broker is configured."""
        return self._sesh

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def declare(self):
        """
        Called by the pool to scan subscriptions without executing init()
        logic. By default delegates to init(), which is correct for the
        vast majority of strategies where init() only declares proxies.

        Override only if init() has heavy side effects (DB connections,
        file reads, expensive calculations, etc.) that should not run
        during the pool scan:

            class MyStrategy(Strategy):
                def declare(self):
                    # Declarations only — called during pool scan
                    self._tp  = self.subscribe_ticks("BTCUSDT", window_size=50)
                    self._sma = self.add_indicator(self._tp.price_ref, SMA(3))

                def init(self):
                    self.declare()
                    # Heavy logic — only runs in real execution
                    self.model = load_ml_model("weights.pkl")
                    self.db    = connect_database()
        """
        self.init()

    def init(self):
        """Called once before the loop. Declare proxies and indicators here."""
        pass

    def on_data(self):
        """
        Called per tick (BacktestEngine) or per bar (BacktestEngine).
        The broker already has updated prices when this executes.
        """
        pass
    
    def on_stop(self):
        """
        Called when the engine stops cleanly.
        Useful for closing positions, saving state, final logging, etc.

        Invoked in these cases:
        · Ctrl+C (KeyboardInterrupt)
        · Explicit engine.stop()
        · End of loop (kline mode with no more data)

        Example:
            def on_stop(self):
                self.log.trading("Engine stopped — closing open positions")
                for p in self.sesh.positions():
                    self.sesh.position_close(p.ticket)
        """
        pass

    def on_crash(self, exc: Exception):
        """
        Called when an unhandled exception occurs in the feed loop.
        The engine has already stopped the loop when this method is called.

        The exception is available as `exc` for logging or diagnosis.
        If not overridden, the engine re-raises it after calling this.

        Example:
            def on_crash(self, exc: Exception):
                self.log.error("Loop error: %s — %s", type(exc).__name__, exc)
                # notify via telegram, save state, etc.
        """
        pass
    
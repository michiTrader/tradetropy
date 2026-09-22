"""
tradetropy.playback.base
======================
BaseEngine -- shared foundation for the cursor-driven playback engines:

    - ReplayEngine    : automated, bi-directional auditing of a programmatic
                        strategy over historical data.
    - PaperEngine     : manual, forward-only simulation driven by a UI Order
                        Ticket (no programmatic strategy).

BaseEngine extends LiveEngine and reuses its ring/warmup/indicator/charting
logic. On top of that it adds everything that makes a historical dataset behave
like a controllable live feed:

    - A cursor-driven ReplaySesh bound via _bind_data().
    - A PlaybackController (pause/play/step/speed + restart/step-back) wired in
      automatically on attach_chart().
    - Deterministic in-place rewind (restart / step-back) that rebuilds the full
      engine state by re-feeding the dataset up to a target cursor, preserving
      object identity so an attached chart keeps its references valid.
    - A thread-safe command queue (_cmd_queue) so UI callbacks running on the
      Bokeh IOLoop can route orders/actions to the single engine thread (the
      only writer of broker/ring state), consistent with the streaming layer's
      single-writer model.

Subclasses customize two things:

    - How the strategy is obtained (ReplayEngine: user strategy; PaperEngine:
      a strategy synthesized from a PaperConfig).
    - Whether backward navigation is allowed (ReplayEngine: yes; PaperEngine:
      overrides _rebuild_step_back to forbid it and builds its controller with
      allow_backward=False).

ESCAPE HATCH
============
    engine._sesh   -> ReplaySesh
    engine._ctrl   -> PlaybackController
"""

from __future__ import annotations

import numpy as np
import threading
import warnings
from collections import deque
from typing import TYPE_CHECKING

from tradetropy.core.data_types import _normalize_data, _normalize_book
from tradetropy.exceptions import ConfigError
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy

if TYPE_CHECKING:
    from tradetropy.plotting.live.chart import LiveChart
    from tradetropy.replay.replay_sesh import ReplaySesh as _ReplaySesh


# =============================================================================
# HELPER -- normalize input array
# =============================================================================

def _to_array(obj) -> np.ndarray:
    """
    Convert data dict entry to numpy ndarray.

    Accepts ndarray directly or any object with a .data attribute.

    Args:
        obj: Ndarray or object with .data attribute.

    Returns:
        np.ndarray: Converted array with float64 dtype.
    """
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "data"):
        return np.asarray(obj.data, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


# =============================================================================
# BASE ENGINE
# =============================================================================

class BaseEngine(LiveEngine):
    """
    Shared base for the cursor-driven playback engines (replay / paper).

    Inherits LiveEngine and reuses ring, warmup, indicator and charting logic.
    Overrides the minimum necessary to support pause/play/step/speed controls
    and deterministic in-place rewind over a historical dataset.

    - Builds or accepts a ReplaySesh and binds data via _bind_data().
    - Builds a PlaybackController internally for the playback controls.
    - Calculates n_warmup automatically from indicator min_periods.
    - attach_chart() connects the controller to the chart automatically.
    - _loop_tick/_loop_kline omit _topup_ohlc_rings (not applicable to fixed
      historical timestamps) and drain the UI command queue every iteration.
    - _before_loop() starts the controller between chart.start() and the loop.

    Subclasses: ReplayEngine (bi-directional) and PaperEngine (forward-only).
    """

    #: Playback does NOT persist log to file by default (same as backtest).
    #: Override with save_log=True in run().
    _DEFAULT_SAVE_LOG: bool = False

    #: Whether this engine allows backward navigation (step-back / backward
    #: direction). ReplayEngine keeps True; PaperEngine overrides to False.
    _ALLOW_BACKWARD: bool = True

    #: Whether the chart should render the manual Order Ticket + Position
    #: Modifier panels. PaperEngine overrides to True.
    _WANTS_ORDER_TICKET: bool = False

    def __init__(
        self,
        strategy: Strategy,
        data,
        sesh: "_ReplaySesh | None" = None,
        history: "dict | None" = None,
        warmup_ticks: "int | None" = None,
        warmup_pct: float = 0.20,
        _feed_type: str = "tick",
        _poll_interval: float = 0.0,
        speed: float = 1.0,
        base_rate: float = 1.0,
        book: "BookData | tuple | list | None" = None,
        sync_book: bool = False,
        chart_ohlc_interval_ms: int = 60_000,
    ):
        if data is None:
            raise ConfigError(f"{type(self).__name__} requires `data`.")

        # Lazy imports break the import cycle (replay package imports BaseEngine).
        from tradetropy.replay.replay_sesh import ReplaySesh as _ReplaySesh
        from tradetropy.replay.controller import PlaybackController as _PlaybackController

        # Normalize typed data (TickData/KlineData) -> tuple + feed_type.
        inputs, data_feed_type = _normalize_data(data)
        if data_feed_type != _feed_type:
            method = "by_ticks" if _feed_type == "tick" else "by_klines"
            expected = "TickData" if _feed_type == "tick" else "KlineData"
            raise ConfigError(
                f"{type(self).__name__}.{method}() expected {expected}, "
                f"received data of type {data_feed_type!r}."
            )

        # Internal dict {symbol: ndarray} consumed by _calc_warmup/_bind_data.
        data_dict = {inp.symbol: inp.data for inp in inputs}

        # Normalize the recorded book: single BookData or tuple/list of them,
        # keyed by each BookData.symbol (symmetric with `data`).
        book = _normalize_book(book)

        # Resolve sesh
        if sesh is None:
            sesh = _ReplaySesh()
        elif not isinstance(sesh, _ReplaySesh):
            raise ConfigError(
                f"`sesh` must be a ReplaySesh, received {type(sesh).__name__}. "
                "Use ReplaySesh(commission=...) to create one."
            )

        # Register SymbolConfig for each input (same as BacktestEngine).
        # add_symbol queues if the broker is not yet built; it is flushed when
        # the feed_type is bound in super().__init__.
        for inp in inputs:
            sesh.add_symbol(inp.config)

        # Calculate warmup_ticks if not passed explicitly
        if warmup_ticks is None:
            warmup_ticks = self._calc_warmup(
                strategy, data_dict, _feed_type, warmup_pct, sesh,
            )

        # Bind data to the sesh
        sesh._bind_data(data_dict, history=history, warmup_ticks=warmup_ticks)

        super().__init__(
            strategy       = strategy,
            sesh           = sesh,
            _feed_type     = _feed_type,
            _poll_interval = _poll_interval,
            chart_ohlc_interval_ms = chart_ohlc_interval_ms,
        )

        self._ctrl = _PlaybackController(
            sesh, speed=speed, base_rate=base_rate,
            allow_backward=self._ALLOW_BACKWARD,
        )
        # Let the controller drive in-place rewinds (restart / step back).
        self._ctrl._engine = self

        # Serializes the feed loop against an in-place rewind so rings are never
        # mutated by both at the same time.
        self._rebuild_lock = threading.RLock()
        # Per-symbol cursor tracking for the tick loop's drain. Stored on the
        # instance (not as loop locals) so a rewind can reset them coherently.
        self._replay_ultimo_idx: dict[str, int] = {}
        self._replay_ultimo_ts:  dict[str, int] = {}

        # Thread-safe command queue: UI callbacks (Bokeh IOLoop thread) enqueue
        # callables that mutate broker/ring state; they are drained and executed
        # ON THE ENGINE THREAD at the top of each feed iteration, preserving the
        # single-writer model (only the engine thread mutates engine state).
        self._cmd_queue: "deque" = deque()
        self._cmd_lock = threading.Lock()

        # Recorded L2 order book to replay alongside the ticks (optional). Each
        # recorded row is a full top-K image; the shared BookReplayer applies it
        # as a snapshot to the book ring when the tick cursor reaches its
        # timestamp, so book_as_of / DeepTrades behave identically to the
        # live/streaming replay path and to the tick backtest.
        from tradetropy.data._book_replay import BookReplayer, resolve_book_sync
        self._book_replayer = BookReplayer(book or {})

        # Sync preflight: measure how the recorded book aligns with the ticks,
        # warn on desync, and (only when recoverable and sync_book=True) shift
        # the book clock onto the trade clock. Tick mode only - klines have no
        # per-trade timestamp for book_as_of.
        if book and _feed_type == "tick":
            from tradetropy.core.constants import _TICK_COL as _TC
            trades = {
                sym: (arr[:, _TC["ts"]], arr[:, _TC["price"]])
                for sym, arr in data_dict.items()
            }
            reports = resolve_book_sync(
                book, trades, sync_book=sync_book,
                engine_label=type(self).__name__,
            )
            for sym, (_rep, off) in reports.items():
                if off:
                    self._book_replayer.set_offset(sym, off)
            self._book_sync_reports = {s: r for s, (r, _o) in reports.items()}
        else:
            self._book_sync_reports = {}

    # =========================================================================
    # CONSTRUCTORS
    # =========================================================================

    @classmethod
    def by_ticks(
        cls,
        strategy: Strategy,
        data,
        sesh: "_ReplaySesh | None" = None,
        history: "dict | None" = None,
        warmup_ticks: "int | None" = None,
        warmup_pct: float = 0.20,
        speed: float = 1.0,
        base_rate: float = 1.0,
        poll_interval: float = 0.0,
        book: "BookData | tuple | list | None" = None,
        sync_book: bool = False,
        chart_ohlc_interval_ms: int = 60_000,
    ) -> "BaseEngine":
        """
        Playback over historical ticks.

        Args:
            strategy (Strategy): Strategy to execute.
            data: Tuple of TickData (one per symbol).
            sesh (ReplaySesh | None): Optional. If omitted, default ReplaySesh()
                is created (balance 10000, no commission). To configure
                commission, balance, spread, pass a pre-built ReplaySesh.
            history: Optional warmup fine-tuning.
            warmup_ticks: Optional warmup fine-tuning.
            speed (float): Playback speed multiplier.
            base_rate (float): Base ticks/sec at speed=1.0.
            warmup_pct (float): Fraction of dataset used as warmup.
            poll_interval (float): Polling interval in seconds.
            book (BookData | tuple | list | None): Optional recorded L2 book to
                replay alongside the ticks (enables DeepTrades / book metrics in
                the playback UI). Pass a single BookData or a tuple/list of them;
                the symbol is read from each ``BookData.symbol``. Read back with
                io.read_book().
            sync_book (bool): When True, and the sync preflight finds a
                recoverable clock offset between the book and the ticks, the
                book timestamps are shifted onto the trade clock. Default False
                (only warn on desync). An unrecoverable desync is never shifted.
            chart_ohlc_interval_ms (int): Interval (ms) of the chart-only OHLC
                proxy auto-injected when the strategy subscribes ticks but no
                OHLC. Default 60_000 (1 minute). Used only to draw candles;
                never fed to on_data().

        Returns:
            BaseEngine: Configured engine instance (subclass type).
        """
        if sesh is None:
            from tradetropy.replay.replay_sesh import ReplaySesh as _ReplaySesh
            sesh = _ReplaySesh()
        return cls(
            strategy       = strategy,
            data           = data,
            sesh           = sesh,
            history        = history,
            warmup_ticks   = warmup_ticks,
            warmup_pct     = warmup_pct,
            _feed_type     = "tick",
            _poll_interval = poll_interval,
            speed          = speed,
            base_rate      = base_rate,
            book           = book,
            sync_book      = sync_book,
            chart_ohlc_interval_ms = chart_ohlc_interval_ms,
        )

    @classmethod
    def by_klines(
        cls,
        strategy: Strategy,
        data,
        sesh: "_ReplaySesh | None" = None,
        history: "dict | None" = None,
        warmup_ticks: "int | None" = None,
        warmup_pct: float = 0.20,
        speed: float = 1.0,
        base_rate: float = 1.0,
        poll_interval: float = 0.0,
    ) -> "BaseEngine":
        """
        Playback over historical klines.

        Same parameters as by_ticks, but data is a tuple of KlineData
        (each with its interval_ms).

        Args:
            strategy (Strategy): Strategy to execute.
            data: Tuple of KlineData (one per symbol-interval pair).
            sesh (ReplaySesh | None): Optional pre-configured session.
            history: Optional warmup fine-tuning.
            warmup_ticks: Optional warmup fine-tuning.
            speed (float): Playback speed multiplier.
            base_rate (float): Base klines/sec at speed=1.0.
            warmup_pct (float): Fraction of dataset used as warmup.
            poll_interval (float): Polling interval in seconds.

        Returns:
            BaseEngine: Configured engine instance (subclass type).
        """
        if sesh is None:
            from tradetropy.replay.replay_sesh import ReplaySesh as _ReplaySesh
            sesh = _ReplaySesh()
        return cls(
            strategy       = strategy,
            data           = data,
            sesh           = sesh,
            history        = history,
            warmup_ticks   = warmup_ticks,
            warmup_pct     = warmup_pct,
            _feed_type     = "kline",
            _poll_interval = poll_interval,
            speed          = speed,
            base_rate      = base_rate,
        )

    # =========================================================================
    # WARMUP CALCULATION
    # =========================================================================

    @classmethod
    def _calc_warmup(
        cls,
        strategy: Strategy,
        data: dict,
        feed_type: str,
        warmup_pct: float,
        sesh=None,
    ) -> int:
        """
        Calculates n_warmup (number of ticks/candles to reserve as history).

        Unified semantics with backtest/live:
        - strategy.warmup int -> LITERAL value. In tick mode interpreted as
          ticks; in kline mode, as candles (same as by_ticks/by_klines).
        - strategy.warmup None -> AUTO: max(min_periods) of OHLC indicators
          (pure). In tick mode converted to ticks via ticks_per_candle.

        A safety clamp is kept (n_warmup <= 0.50*N) to avoid running out of
        data to replay. The old heuristic (*1.5, minimum 50, warmup_pct)
        was removed so replay matches backtest.
        """
        from tradetropy.models._warmup_policy import auto_warmup_candles

        data_norm: dict = {}
        for key, val in data.items():
            data_norm[key] = _to_array(val)

        warmup_usuario = getattr(strategy, "warmup", None)

        _interval_ms = None
        _auto_bars = 0

        try:
            _strat_tmp = strategy.__class__()
            _strat_tmp._set_run_mode("live")
            _strat_tmp._feed_type = feed_type
            # Wire sesh/broker so init() can touch the broker without crashing
            # (e.g. self.sesh.positions()/trades). Without this, a strategy
            # that accesses the broker in init() leaves _auto_bars=0 and
            # the replay starts with empty history.
            if sesh is not None:
                _strat_tmp._sesh = sesh
                _strat_tmp._broker = getattr(sesh, "_broker", None)
            _strat_tmp.init()
        except Exception as exc:
            warnings.warn(
                f"{cls.__name__}: could not instantiate strategy to auto-calculate "
                f"n_warmup ({exc}). Using warmup_pct={warmup_pct}.",
                stacklevel=3,
            )
            _strat_tmp = None

        if _strat_tmp is not None:
            _auto_bars = auto_warmup_candles(_strat_tmp)
            if _strat_tmp._ohlc_proxies:
                _interval_ms = _strat_tmp._ohlc_proxies[0].interval_ms
            del _strat_tmp

        warmup_by_symbol: dict[str, int] = {}

        for key, arr in data_norm.items():
            if isinstance(key, tuple):
                continue

            symbol = key
            N = len(arr)
            if N == 0:
                warmup_by_symbol[symbol] = 0
                continue

            if warmup_usuario is not None:
                # Literal: ticks (tick mode) or candles (kline mode).
                n_warmup = int(warmup_usuario)
            elif feed_type == "tick":
                # Auto in candles -> convert to ticks. Same criterion as
                # BacktestEngine.by_ticks (estimate_ticks_per_candle): N // n_candles.
                if _interval_ms is not None and _interval_ms > 0:
                    ts_col   = arr[:, 0]
                    ts_velas = (ts_col // _interval_ms) * _interval_ms
                    n_candles  = max(1, len(np.unique(ts_velas)))
                    ticks_per_candle = max(1, N // n_candles)
                else:
                    ticks_per_candle = 10
                n_warmup = _auto_bars * ticks_per_candle
            else:
                n_warmup = _auto_bars

            # Parity with backtest: warmup is respected as-is (no 50% clamp).
            # In backtest warmup silences the first N bars/ticks; in replay
            # they are reserved as history -- same on_data() start point with
            # the same data. Only prevent negative or exceeding the dataset
            # (degenerates same as backtest: nothing would remain).
            n_warmup = max(0, min(n_warmup, N))

            warmup_by_symbol[symbol] = n_warmup

            from tradetropy.models._warmup_policy import debug_warmup_block
            _ivl = _interval_ms or 0
            _nvelas = (
                len(np.unique((arr[:, 0] // _ivl))) if _ivl else 0
            )
            _tpv = max(1, N // _nvelas) if _nvelas else 1
            debug_warmup_block(
                f"{cls.__name__} '{symbol}'", feed_type,
                total_data_points=N, warmup=n_warmup,
                n_indicators=0, auto_bars=_auto_bars,
                ticks_per_candle=_tpv, n_candles_in_dataset=_nvelas,
                extra={
                    "reserved data (hist)": n_warmup,
                    "data to replay (live)": N - n_warmup,
                    "user warmup": warmup_usuario,
                },
            )

        _final = min(warmup_by_symbol.values()) if warmup_by_symbol else 0
        return _final

    # =========================================================================
    # ATTACH CHART -- connects the controller automatically
    # =========================================================================

    def attach_chart(self, chart: "LiveChart") -> None:
        """
        Attach chart and connect the playback controller to it.

        In addition to LiveEngine.attach_chart(), connects the controller so the
        pause/play/step/speed controls appear in the browser.

        Args:
            chart (LiveChart): Chart instance to attach.
        """
        super().attach_chart(chart)
        chart.attach_replay_controller(self._ctrl)

    # =========================================================================
    # PREPARE -- connects broker to the sesh after LiveEngine setup
    # =========================================================================

    def prepare(self, historico=None) -> None:
        super().prepare(historico=historico)

    # =========================================================================
    # BEFORE LOOP -- starts the controller between chart.start() and the loop
    # =========================================================================
    def _before_loop(self) -> None:
        """
        Start the playback controller before entering the feed loop.

        Called after prepare() and chart.start(). Guaranteed order:
            prepare() -> attach_chart() -> chart.start() -> ctrl.start() -> loop

        This ensures the ring is ready and the Bokeh server has started.
        """
        self._ctrl.start()

    # =========================================================================
    # STOP -- also stops the controller
    # =========================================================================

    def stop(self) -> None:
        """
        Stop the engine and the playback controller.

        Stops the controller then calls parent LiveEngine.stop().
        """
        try:
            self._ctrl.stop()
        except Exception:
            pass
        super().stop()

    # =========================================================================
    # COMMAND QUEUE -- thread-safe UI -> engine-thread routing
    # =========================================================================

    def submit_command(self, fn) -> None:
        """
        Enqueue a callable to run on the engine thread.

        UI widgets (Order Ticket, Position Modifier) run their callbacks on the
        Bokeh IOLoop thread, which must NOT mutate broker/ring state directly.
        Instead they submit a command here; it is drained and executed on the
        engine thread (the single writer) at the top of the next feed iteration.

        Args:
            fn (callable): Receives the engine as its single argument.
        """
        with self._cmd_lock:
            self._cmd_queue.append(fn)

    def _drain_commands(self) -> None:
        """
        Execute all queued UI commands on the engine thread.

        Called from the feed loops under _rebuild_lock so order routing is
        serialized against ring/broker mutation and in-place rewinds.
        """
        while True:
            with self._cmd_lock:
                if not self._cmd_queue:
                    return
                fn = self._cmd_queue.popleft()
            try:
                fn(self)
            except Exception as exc:
                warnings.warn(f"{type(self).__name__} command error: {exc}")

    # =========================================================================
    # MANUAL ORDER ROUTING -- used by the UI Order Ticket / Position Modifier
    # =========================================================================

    def primary_symbol(self) -> "str | None":
        """The primary chart symbol (first subscribed OHLC proxy), or None."""
        if self.strategy._ohlc_proxies:
            return self.strategy._ohlc_proxies[0].symbol
        syms = list(getattr(self._sesh, "_datasets", {}).keys())
        return syms[0] if syms else None

    def place_order(
        self,
        side: str,
        order_type: str = "market",
        volume: float = 1.0,
        price: "float | None" = None,
        limit_price: "float | None" = None,
        sl: float = 0.0,
        tp: float = 0.0,
        symbol: "str | None" = None,
        comment: str = "manual",
    ):
        """
        Route a manual order to the session. Runs on the engine thread.

        Submitted from UI callbacks via submit_command(); never call directly
        from the Bokeh IOLoop. Supports Market, Limit, Stop and Stop-Limit.

        Args:
            side (str): 'buy' or 'sell'.
            order_type (str): 'market' | 'limit' | 'stop' | 'stop_limit'.
            volume (float): Order size (> 0).
            price (float | None): For limit/stop: the order price. For
                stop_limit: the stop (trigger) price. Ignored for market.
            limit_price (float | None): For stop_limit: the limit price placed
                once the stop triggers.
            sl (float): Stop-loss price (0 = none).
            tp (float): Take-profit price (0 = none).
            symbol (str | None): Defaults to the primary chart symbol.
            comment (str): Order comment/tag.

        Returns:
            TradeResult | None: The broker result, or None on invalid input.
        """
        from tradetropy.core.broker import (
            OrderType, TradeRequest, TradeRequestActions,
        )

        sym = symbol or self.primary_symbol()
        if sym is None:
            warnings.warn(f"{type(self).__name__}.place_order: no symbol.")
            return None
        side = str(side).lower()
        order_type = str(order_type).lower()
        if side not in ("buy", "sell"):
            warnings.warn(f"place_order: invalid side {side!r}.")
            return None
        try:
            volume = float(volume)
        except (TypeError, ValueError):
            volume = 0.0
        if volume <= 0:
            warnings.warn("place_order: volume must be > 0.")
            return None

        is_buy = side == "buy"

        # Market order: delegate to the session's market helper.
        if order_type == "market":
            fn = self._sesh.buy if is_buy else self._sesh.sell
            return fn(sym, volume, sl=sl, tp=tp, comment=comment)

        if price is None:
            warnings.warn(f"place_order: {order_type} requires a price.")
            return None

        # Pending order with an explicit type so the user's Limit/Stop choice is
        # honored regardless of where the price sits relative to the market.
        _PENDING = {
            ("buy", "limit"):       OrderType.ORDER_TYPE_BUY_LIMIT,
            ("buy", "stop"):        OrderType.ORDER_TYPE_BUY_STOP,
            ("buy", "stop_limit"):  OrderType.ORDER_TYPE_BUY_STOP_LIMIT,
            ("sell", "limit"):      OrderType.ORDER_TYPE_SELL_LIMIT,
            ("sell", "stop"):       OrderType.ORDER_TYPE_SELL_STOP,
            ("sell", "stop_limit"): OrderType.ORDER_TYPE_SELL_STOP_LIMIT,
        }
        ot = _PENDING.get((side, order_type))
        if ot is None:
            warnings.warn(f"place_order: unknown order type {order_type!r}.")
            return None

        broker = getattr(self._sesh, "_broker", None)
        if broker is None or not hasattr(broker, "order_send"):
            # Fall back to the session helper (auto limit/stop by price).
            fn = self._sesh.buy if is_buy else self._sesh.sell
            return fn(sym, volume, price=price, sl=sl, tp=tp, comment=comment)

        req = TradeRequest(
            symbol=sym,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=ot,
            volume=volume,
            price=float(price),
            price_limit=float(limit_price) if limit_price is not None else 0.0,
            sl=sl, tp=tp,
            comment=comment,
        )
        return broker.order_send(req)

    def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        """
        Attach/adjust SL and TP on an open position. Runs on the engine thread.

        Args:
            ticket (int): Position ticket.
            sl (float): New stop-loss price (0 = clear).
            tp (float): New take-profit price (0 = clear).

        Returns:
            bool: True on success.
        """
        try:
            return bool(self._sesh.position_modify(int(ticket), sl=sl, tp=tp))
        except Exception as exc:
            warnings.warn(f"{type(self).__name__} modify_position error: {exc}")
            return False

    def close_position(self, ticket: int, volume: "float | None" = None) -> bool:
        """
        Close an open position fully or partially. Runs on the engine thread.

        Args:
            ticket (int): Position ticket.
            volume (float | None): Volume to close; None = full close.

        Returns:
            bool: True on success.
        """
        try:
            return bool(self._sesh.position_close(int(ticket), volume))
        except Exception as exc:
            warnings.warn(f"{type(self).__name__} close_position error: {exc}")
            return False

    def cancel_order(self, ticket: int) -> bool:
        """
        Cancel a pending limit/stop order. Runs on the engine thread.

        Args:
            ticket (int): Order ticket.

        Returns:
            bool: True on success.
        """
        try:
            return bool(self._sesh.order_delete(int(ticket)))
        except Exception as exc:
            warnings.warn(f"{type(self).__name__} cancel_order error: {exc}")
            return False

    def net_position(self, symbol: "str | None" = None) -> "dict | None":
        """
        Current net position for a symbol as a light dict, or None when flat.

        A cheap, non-mutating read safe to call from the Bokeh IOLoop (Position
        Modifier widget, position-tracker line). Aggregates open positions into a
        single volume-weighted average (netting); for a hedging book it nets the
        signed volume and reports the first ticket.

        Returns:
            dict | None: {'ticket', 'avg_price', 'size', 'side'} or None.
        """
        from tradetropy.core.broker import OrderType

        sym = symbol or self.primary_symbol()
        if sym is None:
            return None
        try:
            positions = self._sesh.positions(sym)
        except Exception:
            return None
        if not positions:
            return None

        _BUY = (OrderType.ORDER_TYPE_BUY, OrderType.ORDER_TYPE_BUY_LIMIT,
                OrderType.ORDER_TYPE_BUY_STOP, OrderType.ORDER_TYPE_BUY_STOP_LIMIT)
        net = 0.0
        notional = 0.0
        first_ticket = None
        for pos in positions:
            vol = float(getattr(pos, "volume", 0.0) or 0.0)
            px = float(getattr(pos, "price_open", 0.0) or 0.0)
            signed = vol if getattr(pos, "type", None) in _BUY else -vol
            net += signed
            notional += abs(vol) * px
            if first_ticket is None:
                first_ticket = int(getattr(pos, "ticket", 0))
        total_vol = sum(abs(float(getattr(p, "volume", 0.0) or 0.0)) for p in positions)
        if total_vol <= 0:
            return None
        avg_price = notional / total_vol
        return {
            "ticket": first_ticket,
            "avg_price": avg_price,
            "size": abs(net) if net != 0 else total_vol,
            "side": "buy" if net >= 0 else "sell",
        }

    # =========================================================================
    # REWIND / SEEK -- restart and step-back support
    # =========================================================================

    def _intervals_for(self, symbol: str) -> list:
        """Distinct OHLC interval_ms declared for a symbol (kline re-feed)."""
        seen: list[int] = []
        for op in self.strategy._ohlc_proxies:
            if op.symbol == symbol and op.interval_ms not in seen:
                seen.append(op.interval_ms)
        return seen

    def _apply_tick(self, symbol: str, tick, with_broker: bool) -> None:
        """
        Apply one tick to the engine, mirroring the live loop exactly.

        Runs on_tick() (rings/indicators/on_data) and, when with_broker is
        True, feeds the tick to the internal broker (current prices, SL/TP,
        market fills) in the same order the feed loop uses. The broker step is
        skipped during the warmup re-feed (no trading happens in warmup).

        Args:
            symbol (str): Trading symbol.
            tick (np.ndarray): Tick row.
            with_broker (bool): Whether to forward the tick to the broker.
        """
        from tradetropy.core.constants import _TICK_COL
        from datetime import datetime, timezone

        self._drain_book_to(symbol, int(tick[_TICK_COL["ts"]]))
        self.on_tick(symbol, tick)
        if not with_broker:
            return
        inner = getattr(self._sesh, "_broker", None)
        if inner is None or not hasattr(inner, "update_tick"):
            return
        ts = int(tick[_TICK_COL["ts"]])
        try:
            inner.update_tick(
                symbol      = symbol,
                timestamp   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                bid         = float(tick[_TICK_COL["bid"]]),
                ask         = float(tick[_TICK_COL["ask"]]),
                volume      = float(tick[_TICK_COL["volume"]]),
                flags       = int(tick[_TICK_COL["flags"]]),
                volume_real = float(tick[_TICK_COL["volume_real"]]),
                price       = float(tick[_TICK_COL["price"]]),
            )
        except Exception as exc:
            warnings.warn(f"{type(self).__name__} broker update_tick error: {exc}")

    def _drain_book_to(self, symbol: str, ts: int) -> None:
        """
        Apply recorded book rows with timestamp <= ts to the symbol's book ring.

        Delegates to the shared BookReplayer so the causal book replay is
        identical across replay, paper trading and the tick backtest.

        Args:
            symbol (str): Trading symbol.
            ts (int): Current tick timestamp (ms).
        """
        self._book_replayer.drain_to(symbol, ts, self._book_rings.get(symbol))

    def _feed_index(self, symbol: str, i: int, with_broker: bool) -> None:
        """Re-feed dataset row i for a symbol (tick or kline mode)."""
        if self._feed_type == "tick":
            ds = self._sesh._datasets.get(symbol)
            if ds is None or i >= len(ds):
                return
            self._apply_tick(symbol, ds[i], with_broker)
        else:
            self._sesh._cursor[symbol] = i
            for intervalo in self._intervals_for(symbol):
                kline = self._sesh._fetch_last_kline(symbol, intervalo)
                if kline is not None:
                    self.on_kline(symbol, kline)

    def _cursor_ts(self, symbol: str, idx: int) -> int:
        """Timestamp (ms) of dataset row idx, or -1 if out of range."""
        from tradetropy.core.constants import _TICK_COL
        ds = self._sesh._datasets.get(symbol)
        if ds is None or idx < 0 or idx >= len(ds):
            return -1
        try:
            return int(ds[idx][_TICK_COL["ts"]])
        except Exception:
            return -1

    def _reset_engine_state(self) -> None:
        """
        Reset broker, rings, footprint, pattern stores and counters in place.

        Object identity is preserved (no ring/proxy/broker is recreated) so the
        attached chart keeps its references valid; only their contents are
        cleared. Called at the start of every rewind.
        """
        try:
            self._sesh.reset()
        except Exception as exc:
            warnings.warn(f"{type(self).__name__} reset broker error: {exc}")

        for ring in self._tick_rings.values():
            ring.reset()
        for rings in self._ohlc_rings.values():
            for r in rings:
                r.reset()
        for rings in self._fp_rings.values():
            for r in rings:
                r.reset()

        # Reset replayed book rings and their cursor so a rewind re-feeds the
        # recorded book from the start in lockstep with the ticks.
        for rings in self._book_rings.values():
            for r in rings:
                r.reset()
        self._book_replayer.reset()

        # Pattern stores are rebuilt empty (their source rings are now cleared);
        # they refill as the re-feed re-closes candles.
        if self.strategy._pattern_matcher_defs:
            try:
                self._build_pattern_stores_live({})
            except Exception as exc:
                warnings.warn(f"{type(self).__name__} reset pattern stores error: {exc}")

        for tp in self.strategy._tick_proxies:
            tp._n_total = 0

        self._historical_candles = 0
        self._warmup_hibrido_avisado = False

    def _rebuild_to_cursor(self, targets: "dict[str, int]") -> None:
        """
        Rewind the playback in place so each symbol's cursor lands on its target.

        Rebuilds the full engine state deterministically: reset everything, then
        re-feed dataset rows [0 .. target]. Rows in the warmup region are fed
        without on_data()/broker (history), the rest with on_data()/broker (so
        trades reproduce exactly). The chart is notified once at the end.

        Args:
            targets (dict[str, int]): Desired final cursor per symbol.
        """
        with self._rebuild_lock:
            self._reset_engine_state()
            prev_chart = self._suppress_chart
            self._suppress_chart = True
            symbols = self._simbolos_del_loop()
            try:
                # Phase 1 -- warmup re-feed (no on_data, no broker).
                self._suppress_on_data = True
                for sym in symbols:
                    n_warmup = self._sesh._warmup_ticks.get(sym, 0)
                    target   = targets.get(sym, n_warmup - 1)
                    warm_end = min(n_warmup, target + 1)
                    for i in range(0, warm_end):
                        self._feed_index(sym, i, with_broker=False)

                self._historical_candles = self._count_closed_candles_rings()

                # Phase 2 -- live re-feed (on_data + broker).
                self._suppress_on_data = False
                for sym in symbols:
                    n_warmup = self._sesh._warmup_ticks.get(sym, 0)
                    target   = targets.get(sym, n_warmup - 1)
                    for i in range(n_warmup, target + 1):
                        self._feed_index(sym, i, with_broker=True)
                    self._sesh._cursor[sym]      = target
                    self._replay_ultimo_idx[sym] = target
                    self._replay_ultimo_ts[sym]  = self._cursor_ts(sym, target)
            finally:
                self._suppress_on_data = False
                self._suppress_chart   = prev_chart

        # Resync the chart once (schedules a repopulate on the Bokeh IOLoop).
        self._resync_chart_sources()

    def _rebuild_restart(self) -> None:
        """Rewind to the very start of the playback (before the first live tick)."""
        targets = {
            sym: self._sesh._warmup_ticks.get(sym, 0) - 1
            for sym in self._simbolos_del_loop()
        }
        self._rebuild_to_cursor(targets)

    def _rebuild_step_back(self, n: int = 1) -> None:
        """
        Rewind n steps back (clamped to the start of the live region).

        Args:
            n (int): Number of cursor units to move back (default 1).
        """
        targets = {}
        for sym in self._simbolos_del_loop():
            n_warmup = self._sesh._warmup_ticks.get(sym, 0)
            cur      = self._sesh._cursor.get(sym, n_warmup - 1)
            targets[sym] = max(cur - int(n), n_warmup - 1)
        self._rebuild_to_cursor(targets)

    # =========================================================================
    # LOOP TICK -- without _topup_ohlc_rings
    # =========================================================================

    def _drain_pending_ticks(self, sym: str) -> None:
        """
        Apply every dataset tick accumulated since the last drain for a symbol.

        The cursor is snapshotted ONCE per drain and used both to bound the
        fetched range and to advance ``_replay_ultimo_idx``. This is what keeps
        the drain race-safe: two threads advance the cursor (the controller loop
        while playing and the Bokeh Step button on the IOLoop), so reading the
        cursor a SECOND time to set ``_replay_ultimo_idx`` would skip the ticks
        that landed in the window between the two reads - permanently shrinking
        candle volume and every order-flow metric relative to the backtest.
        With a single snapshot, any tick that arrives after it is simply picked
        up by the next drain.

        ``_fetch_pending_ticks`` returns each row in ``(last_idx, up_to]`` once,
        by index and in order, so all of them are applied - including several
        trades sharing the same millisecond. There is deliberately NO ``ts >
        last`` gate: dropping same-millisecond trades would shrink candle volume
        and every order-flow metric relative to the backtest and the rewind path
        (which sum all ticks), breaking replay parity. This mirrors the
        streaming contract: same-millisecond trades are all delivered.

        Args:
            sym (str): Symbol to drain.
        """
        from tradetropy.core.constants import _TICK_COL

        last_idx = self._replay_ultimo_idx.get(sym, -1)
        # Snapshot the cursor once; drain to it and mark it processed with the
        # SAME value so a concurrent step() cannot make us skip ticks.
        cur = self._sesh._cursor.get(sym, -1)
        pendientes = self._sesh._fetch_pending_ticks(sym, last_idx, up_to=cur)
        for tick in pendientes:
            ts = int(tick[_TICK_COL["ts"]])
            if ts > self._replay_ultimo_ts.get(sym, -1):
                self._replay_ultimo_ts[sym] = ts
            self._apply_tick(sym, tick, with_broker=True)
        if pendientes:
            self._replay_ultimo_idx[sym] = cur

    def _loop_tick(self, simbolos: list) -> None:
        """
        Event-driven loop with full drain of pending ticks.

        When the controller advances multiple ticks between engine wakeups
        (typical at speed=Max/turbo), _fetch_pending_ticks returns all ticks in
        range (last_idx, cursor] so none are lost.

        On finish the loop does NOT tear the engine down: it idles (the
        controller pauses and notifies), so a Restart/Back rewind can rewind the
        cursor and playback resumes. Per-symbol drain tracking lives on the
        instance so a rewind can reset it coherently.

        No side effects: does not call _topup_ohlc_rings() since historical
        timestamps are fixed and topup would corrupt chronological order.
        """
        from tradetropy.core.constants import _TICK_COL

        self._resync_chart_sources()

        self._replay_ultimo_ts  = {sym: -1 for sym in simbolos}
        self._replay_ultimo_idx = {
            sym: self._sesh._warmup_ticks.get(sym, 0) - 1
            for sym in simbolos
        }
        _tick_event = threading.Event()

        _sesh = self._sesh
        _original_step = _sesh.step

        def _step_con_signal(symbol=None):
            _original_step(symbol)
            _tick_event.set()

        _sesh.step = _step_con_signal

        try:
            while not self._stop_event.is_set():
                # Idle (do not tear down) when the replay reached the end. The
                # controller handles the finish notification + pause; a rewind
                # resets the cursor so playback can resume from here.
                _tick_event.wait(timeout=0.1)

                # Drain UI commands on the engine thread even while idle/paused
                # so manual orders are routed promptly.
                with self._rebuild_lock:
                    self._drain_commands()

                if not _tick_event.is_set():
                    continue
                _tick_event.clear()

                # Serialize against an in-place rewind so rings are never
                # mutated concurrently.
                with self._rebuild_lock:
                    for sym in simbolos:
                        if self._stop_event.is_set():
                            break
                        try:
                            self._drain_pending_ticks(sym)
                        except Exception as exc:
                            import traceback
                            warnings.warn(
                                f"{type(self).__name__} tick error ({sym}): {exc}\n"
                                f"{traceback.format_exc()}"
                            )
        finally:
            _sesh.step = _original_step

    # =========================================================================
    # LOOP KLINE -- without _topup_ohlc_rings, with finished check
    # =========================================================================

    def _loop_kline(self, symbols: list) -> None:
        """
        Kline feed loop without _topup_ohlc_rings.

        Same as LiveEngine._loop_kline() but without topup and with automatic
        stop when replay finishes. No side effects on ring: historical data
        is replayed progressively, topup not applicable.
        """
        import time

        intervalos = {}
        for op in self.strategy._ohlc_proxies:
            if op.symbol not in intervalos:
                intervalos[op.symbol] = op.interval_ms

        while not self._stop_event.is_set():
            # Drain UI commands on the engine thread first.
            with self._rebuild_lock:
                self._drain_commands()

            # Idle (do not tear down) when the replay reached the end so a
            # Restart/Back rewind can resume playback.
            if getattr(self._sesh, "finished", False):
                time.sleep(0.05)
                continue

            with self._rebuild_lock:
                for sym in symbols:
                    if self._stop_event.is_set():
                        break
                    intervalo = intervalos.get(sym)
                    if intervalo is None:
                        continue
                    try:
                        kline = self.sesh._fetch_last_kline(sym, intervalo)
                        if kline is None:
                            continue
                        self.on_kline(sym, kline)
                    except Exception as exc:
                        import traceback
                        warnings.warn(
                            f"{type(self).__name__} kline loop error ({sym}): {exc}\n"
                            f"{traceback.format_exc()}"
                        )
                        time.sleep(0.05)

            time.sleep(self._poll_interval)

    # =========================================================================
    # TOPUP -- no-op in playback
    # =========================================================================

    def _topup_ohlc_rings(self) -> None:
        """
        No-op in playback engines.

        LiveEngine calls this at end of prepare() to cover gap between
        last historical kline and 'now'. In playback, data is historical and
        fixed - gap does not exist. Applying topup would corrupt the ring with
        klines from dataset end, breaking chronological order.
        """
        pass

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        syms  = list(self._sesh._datasets.keys())
        mode  = self._feed_type
        speed = self._ctrl.speed
        speed_str = "Max" if speed == float("inf") else f"{speed:.1f}x"
        state = "paused" if self._ctrl.paused else "playing"
        return (
            f"{type(self).__name__}(symbols={syms}, mode={mode!r}, "
            f"speed={speed_str}, base_rate={self._ctrl.base_rate:.2f}, "
            f"state={state!r})"
        )

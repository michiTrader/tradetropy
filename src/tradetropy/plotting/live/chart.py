"""
chart.py
========
LiveChart - main class for the real-time charting system.

Public API
----------
    chart = LiveChart(
        config               = PlotConfig(theme="dark", width=1400),
        max_candles          = 300,
        port                 = 5006,
        trades_poll_interval = 5.0,   # seconds between new trade detection checks
    )

    engine.attach_chart(chart)
    chart.start()    # opens localhost:5006 and starts the server in background
    engine.run()     # blocking loop; the chart updates on its own

    # To stop cleanly from on_stop():
    chart.stop()

Thread model
------------
- Engine thread  : calls _on_new_data(bar_closed) after each on_data().
- Server thread  : Tornado/Bokeh IOLoop (daemon). Serves WebSockets.
- Communication : doc.add_next_tick_callback() - the only thread-safe way
                   to mutate a Bokeh Document from outside the IOLoop.

Rate limiting
-------------
If the engine produces ticks faster than the browser can consume (high-
frequency feeds), _pending_update prevents stacking callbacks. The browser
always receives the MOST RECENT state when the IOLoop processes the callback,
never a stale intermediate state.

Live trades
-----------
The chart automatically monitors for new closed trades in the broker.
To avoid excessive broker calls, the check runs every trades_poll_interval
seconds (default 5.0), not on every tick. When new trades are detected,
it updates the trades source via a full .data= assignment and updates the
legend_label with the count.

Behavior is identical to the static backtest plot: each closed trade
appears as a diagonal line (entry -> exit) colored green for winners
and red for losers.

Trade data sources:
  - SeshMT5Sim  -> broker.stats.trades (DataFrame, same source as bt.plot())
  - SeshMT5Live -> sesh.deals(symbol) matching IN/OUT by position_id
"""

from __future__ import annotations

import threading
import time
import webbrowser
from typing import TYPE_CHECKING, Literal

from tradetropy.plotting.config import PlotConfig

if TYPE_CHECKING:
    from tradetropy.models.strategy import Strategy
    from tradetropy.plotting.live.updater import LiveSourceUpdater
    from tradetropy.replay.replay_controller import ReplayController


_State = Literal["init", "ready", "running", "stopped"]


class LiveChart:
    """
    Interactive real-time chart powered by a LiveEngine.

    Typical usage
    -------------
        chart = LiveChart(PlotConfig(theme="dark"), max_candles=300)
        engine.attach_chart(chart)
        chart.start()
        engine.run()

    The chart updates automatically on every call to on_data().
    No changes to the strategy or indicators are required.

    Live trades
    -----------
    Closed trades appear automatically as diagonal lines on the OHLC
    panel, just like in bt.plot(). The chart queries the broker every
    `trades_poll_interval` seconds to detect new trades. This can be
    disabled by setting config.plot_trades=False.
    """
    config = PlotConfig

    def __init__(
        self,
        config: "PlotConfig | None" = None,
        max_candles: int = 500,
        port: int = 5006,
        open_browser: bool = True,
        trades_poll_interval: float = 5.0,
    ):
        """
        Parameters
        ----------
        config               : PlotConfig with visual options.
        max_candles          : maximum number of candles to keep in the chart.
        port                 : Bokeh server port (default 5006).
        open_browser         : open the browser automatically on startup.
        trades_poll_interval : seconds between broker trade detection checks
                               (default 5.0 s). The check is cheap (counter
                               comparison) when there are no changes - it does
                               not call the broker. Set to 0.0 to disable
                               monitoring.
        """
        self._config:       PlotConfig            = config or PlotConfig()
        self._max_candles:  int                   = max_candles
        self._port:         int                   = port
        self._open_browser: bool                  = open_browser
        self._trades_poll_interval: float         = trades_poll_interval

        # Injected by LiveEngine.attach_chart()
        self._strategy:  "Strategy | None"        = None
        self._engine:    object                   = None   # LiveEngine
        self._broker:    object                   = None

        # Internal state
        self._doc:           object               = None   # bokeh Document
        self._updater:       "LiveSourceUpdater | None" = None
        self._server:        object               = None   # bokeh Server
        self._server_thread: "threading.Thread | None"  = None
        self._io_loop:       object               = None   # tornado IOLoop

        self._state:          _State              = "init"
        self._pending_update: bool                = False
        self._pending_bar_closed: bool            = False

        # Frame-rate decoupling: the engine only marks the chart as 'dirty'
        # (_dirty) from its thread; a periodic callback on the Bokeh IOLoop
        # (_frame_tick) consumes that flag and redraws at most once every
        # PlotConfig.chart_refresh_ms. This prevents the IOLoop from being
        # saturated with back-to-back redraws at high replay speeds and keeps
        # controls (pause/step/speed) responsive. _periodic_cb stores the
        # handle returned by doc.add_periodic_callback for removal in stop().
        self._dirty:          bool                = False
        self._periodic_cb:    object              = None
        # Timestamp (perf_counter) of the last refresh for heavy drawing
        # layers (VP/Heatmap), spaced out to heavy_refresh_ms.
        self._last_heavy_s:   float               = 0.0

        # Replay (optional) - injected via attach_replay_controller()
        self._replay_controller: "ReplayController | None" = None

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def attach_replay_controller(self, controller: "ReplayController") -> None:
        """
        Attaches a ReplayController to this chart.

        Must be called BEFORE chart.start(). The controller is passed to
        build_live_document() which adds the pause/play/step/speed buttons
        to the Bokeh Document layout.

        The controller is NOT started here - the user must call
        ctrl.start() after chart.start() so the advance loop begins
        once the server is ready.

        Parameters
        ----------
        controller : ReplayController linked to a ReplaySesh.
        """
        self._replay_controller = controller

    def start(self) -> None:
        """
        Prepares the document, loads the available history from the rings,
        starts the Bokeh server in a daemon thread and (optionally) opens
        the browser.

        Must be called AFTER engine.prepare() and engine.attach_chart().
        """
        if self._state == "running":
            return
        if self._strategy is None:
            raise RuntimeError(
                "LiveChart has no strategy. "
                "Call engine.attach_chart(chart) before chart.start()."
            )

        self._setup()

        # Keep the websocket push alive if a non-finite float ever slips into a
        # message payload (Bokeh's serializer rejects NaN/Inf, crashing the send
        # coroutine outside our reach). Idempotent, scoped to live charting.
        from tradetropy.plotting.live._json_safety import install_nan_safe_serialization
        install_nan_safe_serialization()

        self._start_server_thread()

        if self._open_browser:
            # Opening the browser from the server's IOLoop ensures the
            # server already accepts sessions when the browser connects.
            # call_later(0.3) gives the IOLoop time to process start().
            port = self._port
            def _open():
                webbrowser.open(f"http://localhost:{port}")
            if self._io_loop is not None:
                self._io_loop.call_later(0.3, _open)
            else:
                time.sleep(0.5)
                webbrowser.open(f"http://localhost:{self._port}")

        self._state = "running"

    def stop(self) -> None:
        """
        Stops the Bokeh server cleanly. Call from engine.on_stop() or at
        the end of the script to release the port.
        """
        if self._state == "stopped":
            return
        self._state = "stopped"

        # Remove the periodic refresh callback cleanly. Document mutation must
        # happen on the IOLoop thread, so schedule the removal there (it runs
        # before io_loop.stop below, since add_callback is FIFO). Best-effort:
        # if the loop is already tearing down, the daemon thread dies with the
        # process anyway.
        _doc = self._doc
        _cb  = self._periodic_cb
        if self._io_loop is not None and _doc is not None and _cb is not None:
            def _remove_periodic():
                try:
                    _doc.remove_periodic_callback(_cb)
                except Exception:
                    pass
            try:
                self._io_loop.add_callback(_remove_periodic)
            except Exception:
                pass
        self._periodic_cb = None

        if self._io_loop is not None:
            try:
                self._io_loop.add_callback(self._io_loop.stop)
            except Exception:
                pass

        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)

    # -- Hook called by LiveEngine ------------------------------------------------

    def _on_new_data(self, bar_closed: bool = False) -> None:
        """
        Called by LiveEngine (engine thread) after each on_data().

        Does NOT enqueue work on the IOLoop: it only marks the chart as 'dirty'.
        The redraw is triggered by _frame_tick, a periodic callback running on
        the Bokeh IOLoop at most once every PlotConfig.chart_refresh_ms. This
        decouples the refresh from the engine data rate: at high replay speed
        the IOLoop is not saturated with consecutive redraws and control clicks
        (pause/step/speed) are processed immediately.

        Setting boolean flags is atomic in CPython (protected by the GIL), so
        it is safe to do from the engine thread without a lock.
        """
        if self._state != "running":
            return

        # Propagate bar_closed to the next frame; it is not lost even if
        # multiple ticks arrive between frames (the frame reads the latest state).
        if bar_closed:
            self._pending_bar_closed = True

        self._dirty = True

    def _frame_tick(self) -> None:
        """
        Periodic callback on the Bokeh IOLoop: redraws if there is new data.

        Runs at fixed intervals (PlotConfig.chart_refresh_ms). If the chart is
        not 'dirty' it does nothing, leaving the IOLoop free to handle UI events
        (replay control clicks). If it is dirty, it clears the flag and executes
        a single refresh, reflecting the latest engine state.

        Heavy drawing layers (VP/Heatmap, tool snapshots) refresh at a coarser
        cadence (heavy_refresh_ms): they recompute their full geometry each time,
        so spacing them out avoids occupying the IOLoop on every frame and keeps
        controls responsive even at high speed.
        """
        if not self._dirty:
            return
        self._dirty = False

        now_s = time.perf_counter()
        heavy_interval = max(0.0, int(getattr(self._config, "heavy_refresh_ms", 200)) / 1000.0)
        heavy = (now_s - self._last_heavy_s) >= heavy_interval
        if heavy:
            self._last_heavy_s = now_s
        self._safe_update(heavy=heavy)

    # ======================================================================
    # INTERNAL SETUP
    # ======================================================================

    def _setup(self) -> None:
        """
        Validates that the strategy has at least one subscribe_ohlc().
        The actual Document construction happens in make_doc() per session,
        inside _start_server_thread().
        """
        if not self._strategy._ohlc_proxies:
            raise ValueError(
                "LiveChart requires at least one subscribe_ohlc() in the strategy."
            )
        self._state = "ready"

    # ======================================================================
    # BOKEH SERVER
    # ======================================================================

    def _start_server_thread(self) -> None:
        """
        Starts the Bokeh server in a daemon thread.

        Architecture:
          - A dedicated Tornado IOLoop for the server.
          - Application with FunctionHandler that copies the pre-built Document
            to each new session (one session = one browser tab).
          - The IOLoop runs in the daemon thread - it dies with the main process.
        """
        import asyncio
        from bokeh.server.server import Server
        from bokeh.application import Application
        from bokeh.application.handlers.function import FunctionHandler
        from tornado.ioloop import IOLoop

        # We need an asyncio event loop in the server thread.
        # Tornado/Bokeh requires it on Python 3.10+.
        ready_event = threading.Event()

        # Capture poll_interval and replay controller here for the closure
        _trades_poll_interval   = self._trades_poll_interval
        _replay_controller      = self._replay_controller

        def run_server():
            # Create a dedicated event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            io_loop = IOLoop.current()
            self._io_loop = io_loop

            def make_doc(doc):
                # print(f"[MAKE_DOC] strategy._sesh = {getattr(self._strategy, '_sesh', None)!r}")
                # print(f"[MAKE_DOC] strategy._run_mode = {getattr(self._strategy, '_run_mode', None)!r}")

                # Build all content directly in the session Document that
                # Bokeh provides - roots cannot be copied between documents
                # in Bokeh 3.x.
                from tradetropy.plotting.live.document import build_live_document

                # Top up the ring just before building the Document.
                # The browser may take seconds or minutes to connect after
                # chart.start() - during that time the engine already processed
                # ticks and closed candles that are not in the historical ring.
                # Without this topup the chart starts with a gap between the
                # last historical candle and the current live candle.
                feed_type = getattr(self._engine, "_feed_type", "kline")
                _engine_sesh = getattr(self._engine, "sesh", None) if self._engine is not None else None
                if feed_type == "tick" and self._engine is not None and _engine_sesh is not None:
                    try:
                        self._engine._topup_ohlc_rings()
                    except Exception:
                        pass  # topup is best-effort - should not break the chart

                _, updater = build_live_document(
                    strategy             = self._strategy,
                    config               = self._config,
                    max_candles          = self._max_candles,
                    broker               = self._broker,
                    doc                  = doc,
                    trades_poll_interval = _trades_poll_interval,
                    replay_controller    = _replay_controller,
                    io_loop              = io_loop,
                    engine               = self._engine,
                )
                # Only save updater and doc from the FIRST session.
                # Subsequent sessions (reconnection, second tab) receive
                # their own updater but the engine only notifies the first one.
                if self._updater is None:
                    self._doc     = doc
                    self._updater = updater
                    # Register the periodic callback that decouples the refresh
                    # from the engine data rate. Runs on the IOLoop (make_doc
                    # already executes there, so it is thread-safe) at fixed
                    # intervals of chart_refresh_ms, redrawing only when there
                    # is new data (_dirty). This leaves regular gaps in the
                    # IOLoop for handling replay control clicks instantly.
                    period_ms = max(16, int(getattr(self._config, "chart_refresh_ms", 66)))
                    try:
                        self._periodic_cb = doc.add_periodic_callback(
                            self._frame_tick, period_ms,
                        )
                    except Exception:
                        self._periodic_cb = None
                # Load history now that the session doc is ready.
                # max_age=0 -> always load in tick mode (ring data is fresh
                # thanks to the topup above, not old klines).
                max_age = 0 if feed_type == "tick" else 30
                updater.populate_history(max_age_minutes=max_age)

                # DEBUG - verify source has data after populate
                # src = updater._ohlc_refs[0].source
                # print(f"[MAKE_DOC] after populate: ts len={len(src.data['ts'])}")

            app = Application(FunctionHandler(make_doc))

            server = Server(
                {"/": app},
                port          = self._port,
                io_loop       = io_loop,
                allow_websocket_origin = [
                    f"localhost:{self._port}",
                    f"127.0.0.1:{self._port}",
                ],
                num_procs = 1,
                # The default session token expires in ~30s. If prepare() is
                # slow or the browser loads slowly, the WebSocket fails with
                # "Token is expired". 600_000 ms (10 min) gives enough margin.
                session_token_expiration = 600_000,
            )
            self._server = server
            server.start()

            ready_event.set()   # signal that the server is ready
            io_loop.start()     # blocking - runs until io_loop.stop()

        self._server_thread = threading.Thread(
            target = run_server,
            daemon = True,
            name   = "LiveChart-server",
        )
        self._server_thread.start()

        # Wait for the server to be ready (max 5s)
        if not ready_event.wait(timeout=5.0):
            raise RuntimeError(
                f"LiveChart: the Bokeh server did not start on port {self._port}. "
                "Check that the port is not already in use."
            )

    # ======================================================================
    # UPDATE CALLBACK
    # ======================================================================

    def _safe_update(self, heavy: bool = True) -> None:
        """
        Chart refresh. Always runs on the Bokeh IOLoop (invoked by
        _frame_tick), so ColumnDataSource access is safe.

        Args:
            heavy (bool): Forwarded to LiveSourceUpdater.update(); if False,
                heavy drawing layers (VP/Heatmap, tool snapshots) are skipped
                in this frame (they refresh at the coarse heavy_refresh_ms
                cadence or on candle close).
        """
        # Read and reset the bar_closed flag atomically
        bar_closed = self._pending_bar_closed
        self._pending_bar_closed = False

        if self._updater is None or self._state != "running":
            return

        try:
            self._updater.update(bar_closed=bar_closed, heavy=heavy)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"LiveChart._safe_update error: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )

    # ======================================================================
    # REPR
    # ======================================================================

    def __repr__(self) -> str:
        sym = (
            self._strategy._ohlc_proxies[0].symbol
            if self._strategy and self._strategy._ohlc_proxies
            else "?"
        )
        return (
            f"LiveChart(symbol={sym!r}, port={self._port}, "
            f"state={self._state!r}, max_candles={self._max_candles}, "
            f"trades_poll={self._trades_poll_interval}s)"
        )
        
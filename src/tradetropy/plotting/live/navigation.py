"""
navigation.py
=============
LiveNavigationController - real-time candlestick tracking system.

Manages automatic movement of the OHLC chart X range to follow
the candle under construction, just like MT5 or TradingView.

Behavior
--------
- By default follow=True: the X range moves with each new candle.
- The user can pan/zoom freely -> follow is disabled.
- Double-click -> follow re-enables (same as MT5).
- "Last" button -> equivalent to double-click, accessible without mouse.
- The window width (follow_window_ms) is preserved when resuming follow
  so the user's zoom is respected.

Integration
-----------
    # In document.py
    nav = LiveNavigationController(fig_ohlc, interval_ms)
    nav.attach(source_ohlc, btn_last)   # registers callbacks

    # In LiveSourceUpdater._update_ohlc()
    if nav is not None:
        nav.update_view(ts_ms)

Thread-safety
-------------
All methods are called from the Bokeh IOLoop (callbacks registered
with on_change, on_event or add_next_tick_callback). They are never called
directly from the engine thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bokeh.models import ColumnDataSource, Button
    from bokeh.plotting import figure


# Right padding in multiples of interval_ms
_RIGHT_PADDING_CANDLES = 5

# Default window in multiples of interval_ms when there is no history
_DEFAULT_WINDOW_CANDLES = 100


class LiveNavigationController:
    """
    Navigation controller for the live chart.

    Parameters
    ----------
    fig_ohlc     : Bokeh figure of the main OHLC panel.
    interval_ms  : duration of a candle in milliseconds.
    """

    def __init__(self, fig_ohlc: "figure", interval_ms: int, follow_state=None,
                 right_padding_candles: int = _RIGHT_PADDING_CANDLES):
        self.fig           = fig_ohlc
        self.interval_ms   = interval_ms

        # Candle space reserved to the right of the live edge. By default
        # _RIGHT_PADDING_CANDLES, but expanded when a Volume Profile in
        # "visible" view (VPVR) is present whose histogram is anchored to the
        # right and must fit in the view during tracking.
        self.right_padding_candles = max(int(right_padding_candles), _RIGHT_PADDING_CANDLES)

        # Public state
        self.follow           = True
        self.autoscale_y      = True

        # ColumnDataSource shared with the autoscale JS ("follow" column).
        # The JS reads it to decide whether to rescale on each data change.
        # Only this class writes to it, from Bokeh IOLoop callbacks.
        self._follow_state = follow_state

        # Window width in ms; initialized in attach() or on the first
        # call to update_view() if the range has not been set yet.
        self.follow_window_ms: int = interval_ms * _DEFAULT_WINDOW_CANDLES

        # Internal flag to prevent on_xrange_change from reacting to
        # movements that we ourselves make in update_view/goto_last.
        self._updating_range: bool = False

        # Bokeh fires on_change("start") synchronously the first time
        # a model with an already-assigned value receives a new subscriber
        # (happens when calling attach() if x_range.start was already set in
        # document.py before registration). That spurious fire would set
        # follow=False before the user has touched anything.
        # _ignore_next_change absorbs exactly that first false event.
        self._ignore_next_change: bool = False

        # Last known ts_ms - needed for goto_last() when a double-click
        # arrives between updates.
        self._last_ts_ms: int = 0

        # OHLC source - assigned in attach()
        self._source: "ColumnDataSource | None" = None

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP
    # ══════════════════════════════════════════════════════════════════════════

    def attach(
        self,
        source_ohlc: "ColumnDataSource",
        btn_last: "Button | None" = None,
    ) -> None:
        """
        Registers event callbacks on fig_ohlc and the button.

        Must be called after fig_ohlc has been added to the Document.

        Parameters
        ----------
        source_ohlc : ColumnDataSource with columns ts, High, Low.
        btn_last    : optional "Last" button; if provided, its on_click
                      callback is connected.
        """
        from bokeh.events import DoubleTap, Reset, PanStart, MouseWheel

        self._source = source_ohlc

        # Detect pan/zoom from the user via x_range changes.
        # _ignore_next_change absorbs ONLY a possible synchronous fire that
        # some versions of Bokeh emit when registering on_change on a
        # range that already has a value. It is activated right before
        # registration and cleared immediately after: if such a fire
        # occurs, it is ignored; and if it does not occur (the common case),
        # the flag does NOT survive to swallow the user's first real pan
        # - a bug that left follow active on the first interaction because
        # the _updating_range check (in programmatic moves) returns before
        # consuming the flag.
        self._ignore_next_change = True
        self.fig.x_range.on_change("start", self._on_xrange_change)
        self._ignore_next_change = False

        # Double-click -> resume tracking.
        # IMPORTANT: attach() must be called BEFORE doc.add_root() so that
        # Bokeh registers the handler before the figure enters the Document.
        self.fig.on_event(DoubleTap, self._on_double_tap)

        # "Reset" button (ResetTool from toolbar) -> resume tracking.
        # Without this, Bokeh's native reset reverts ranges to the initial
        # construction window (warmup), anchoring the view to the session
        # start and desynchronizing the Y axis.
        self.fig.on_event(Reset, self._on_reset)

        # User interaction gestures -> disable follow immediately.
        # PanStart (drag) and MouseWheel (zoom) are ONLY emitted by the user,
        # never by programmatic _move_range movements, so they are an
        # unambiguous signal to drop tracking and allow free movement,
        # without relying on flag heuristics or "fighting" with follow
        # during the drag.
        self.fig.on_event(PanStart, self._on_user_interact)
        self.fig.on_event(MouseWheel, self._on_user_interact)

        # "Last" button
        if btn_last is not None:
            btn_last.on_click(self._on_btn_last)

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def enable_follow(self) -> None:
        """Enable automatic candle tracking."""
        self.follow = True
        self._push_follow_state()

    def disable_follow(self) -> None:
        """Disable tracking; the range stays fixed where it is."""
        self.follow = False
        self._push_follow_state()

    def goto_last(self) -> None:
        """
        Move the X range to the last known candle and re-enable tracking.

        Equivalent to double-click or the "Last" button.

        The Y axis autoscale is handled by the autoscale CustomJS when
        x_range changes (when coming from pan/zoom the range changes and
        triggers rescaling). This class no longer writes y_range directly.
        """
        self.follow = True
        self._push_follow_state()
        if self._last_ts_ms > 0:
            self._move_range(self._last_ts_ms)

    def update_view(self, ts_ms: int) -> None:
        """
        Called by LiveSourceUpdater._update_ohlc() on each tick.

        If follow=True, shifts the X range so the current candle is
        always visible with right padding. The Y axis rescaling is done
        by the autoscale CustomJS (sole authority over y_range), triggered
        by x_range changes and source data changes.

        Parameter
        ----------
        ts_ms : timestamp of the current partial candle (epoch ms).
        """
        self._last_ts_ms = ts_ms

        # Anchor the X axis reset bounds to the live edge (even if
        # follow is disabled) so the ResetTool goes directly to the
        # last candle, without flashing to the initial construction window.
        self._sync_x_reset_bounds(ts_ms)

        if not self.follow:
            return

        self._move_range(ts_ms)

    # ══════════════════════════════════════════════════════════════════════════
    # X RANGE MOVEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _move_range(self, last_ts_ms: int) -> None:
        """
        Shifts x_range.start and x_range.end to center the view on
        last_ts_ms with:
          - follow_window_ms of history to the left
          - RIGHT_PADDING_CANDLES candles of space to the right

        Uses _updating_range=True so that on_xrange_change does not disable
        tracking while we apply the programmatic move.
        """
        right_padding_ms = self.interval_ms * self.right_padding_candles
        new_start = last_ts_ms - self.follow_window_ms
        new_end   = last_ts_ms + right_padding_ms

        self._updating_range = True
        try:
            self.fig.x_range.start = new_start
            self.fig.x_range.end   = new_end
            # Also anchor the Y axis reset bounds to the current range
            # (which maintains the JS autoscale) so that if the ResetTool is
            # pressed while tracking (X unchanged), the Y axis does not revert
            # to the initial construction prices.
            y_start = self.fig.y_range.start
            y_end   = self.fig.y_range.end
            if y_start is not None and y_end is not None:
                self.fig.y_range.reset_start = y_start
                self.fig.y_range.reset_end   = y_end
        finally:
            self._updating_range = False

    def _sync_x_reset_bounds(self, last_ts_ms: int) -> None:
        """
        Keeps x_range.reset_start/reset_end anchored to the live edge
        window (same geometry as _move_range), so the toolbar ResetTool
        takes the view directly to the last candle instead of reverting
        to the initial warmup window.

        Only touches reset_start/reset_end (not start/end), so it does not
        move the current view or trigger _on_xrange_change.
        """
        right_padding_ms = self.interval_ms * self.right_padding_candles
        try:
            self.fig.x_range.reset_start = last_ts_ms - self.follow_window_ms
            self.fig.x_range.reset_end   = last_ts_ms + right_padding_ms
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Y RANGE AUTOSCALE
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # FOLLOW STATE (synchronization with JS autoscale)
    # ══════════════════════════════════════════════════════════════════════════

    def _push_follow_state(self) -> None:
        """
        Propagates the follow state to the ColumnDataSource shared with the
        JS autoscale. The JS reads it to decide whether to rescale the Y axis
        on each data change (only when follow is active).

        Reassigning `.data` (instead of mutating in-place) ensures Bokeh
        serializes the change and sends it to the client.
        """
        if self._follow_state is not None:
            self._follow_state.data = dict(follow=[1 if self.follow else 0])

    # ══════════════════════════════════════════════════════════════════════════
    # EVENT CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_xrange_change(self, attr: str, old, new) -> None:
        # print(f"[DBG NAV] on_xrange_change: _updating={self._updating_range}  _ignore={self._ignore_next_change}  follow={self.follow}  new={new}")
        """
        Called by Bokeh when x_range.start changes (user pan or zoom).

        If the change was initiated by the controller itself
        (_updating_range=True) we ignore it. The first fire after
        registering on_change is also ignored (_ignore_next_change=True)
        because Bokeh may fire it synchronously when registering the
        callback on a range that is already configured.
        Otherwise we disable tracking and store the current window width
        to preserve the zoom.
        """
        if self._updating_range:
            return

        if self._ignore_next_change:
            self._ignore_next_change = False
            return

        # Calculate and persist the current window width before disabling
        try:
            width = int(self.fig.x_range.end) - int(self.fig.x_range.start)
            if width > 0:
                self.follow_window_ms = width - self.interval_ms * self.right_padding_candles
                self.follow_window_ms = max(
                    self.follow_window_ms,
                    self.interval_ms * 10,   # minimum 10 candles
                )
        except Exception:
            pass

        self.follow = False
        self._push_follow_state()

    def _on_double_tap(self, event) -> None:
        """
        Double-click on the chart -> resume tracking.
        Equivalent to `goto_last()`.
        """
        self.goto_last()

    def _on_user_interact(self, event) -> None:
        """
        Pan gesture (PanStart) or zoom (MouseWheel) from the user -> disable
        tracking immediately to allow free chart movement.

        The window width (follow_window_ms) is stored by _on_xrange_change
        when the range changes, preserving the zoom when resuming tracking.
        """
        if self.follow:
            self.follow = False
            self._push_follow_state()

    def _on_reset(self, event) -> None:
        """
        "Reset" button (ResetTool from toolbar) -> resume tracking.

        Bokeh's ResetTool reverts ranges to their initial construction
        values (the warmup window), which anchors the view to the session
        start and desynchronizes the Y axis. In a live chart the expected
        behavior is to return to the live edge, so we redirect the reset
        to `goto_last()`.

        Deferred with add_next_tick_callback to execute AFTER Bokeh
        processes the native range reset and the x_range on_change that
        reset fires (which would set follow=False). This way the
        repositioning to the live edge and follow reactivation have the
        final word and the race condition is avoided.
        """
        doc = getattr(self.fig, "document", None)
        if doc is not None:
            doc.add_next_tick_callback(self._goto_last_deferred)
        else:
            self.goto_last()

    def _goto_last_deferred(self) -> None:
        self.goto_last()

    def _on_btn_last(self, _clicks: int = 0) -> None:
        """
        Callback for the "Last" button.
        Bokeh passes the cumulative click count; we ignore it.
        """
        self.goto_last()

    # ══════════════════════════════════════════════════════════════════════════
    # REPR
    # ══════════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        window_candles = self.follow_window_ms // self.interval_ms
        return (
            f"LiveNavigationController("
            f"follow={self.follow}, "
            f"autoscale_y={self.autoscale_y}, "
            f"window={window_candles} candles, "
            f"last_ts={self._last_ts_ms})"
        )
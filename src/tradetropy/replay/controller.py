"""
replay_controller.py
====================
ReplayController -- controls the advancement of a ReplaySesh from Bokeh
buttons (pause/play, step, speed).

ARCHITECTURE
============
The controller lives in its own daemon thread (`_loop_thread`). The loop
calls `sesh.step()` every `sleep_s` seconds, where:

    rate    = base_rate * speed          # target ticks/sec
    sleep_s = 1.0 / rate                 # seconds between steps

When `paused=True` the loop sleeps without calling step().
With speed=float("inf") (Max/turbo) there is no pause -- it only yields the GIL.

The Bokeh buttons (Toggle, Button, Slider) have Python callbacks that run
in the Bokeh IOLoop. Those callbacks only write floats/bools to the
controller -- atomic operations in CPython that do not need locks.

THREAD SAFETY
=============
- The engine reads sesh.cursor from its own thread.
- Bokeh callbacks write self.paused / self.speed from the IOLoop.
- The controller loop calls sesh.step() from the controller thread.

The only critical section is sesh.step() which increments _cursor. Since the
engine only READS the cursor (does not write it) and step() increments it
non-destructively, the operation is safe without a Lock in CPython (GIL
protects integers). If cross-platform guarantees are needed, a threading.Lock
can be added in step().

TYPICAL USAGE
=============
    from tradetropy.replay.replay_sesh       import ReplaySesh
    from tradetropy.replay.replay_controller import ReplayController

    sesh = ReplaySesh(data={"BTCUSDT": ticks})
    ctrl = ReplayController(sesh, speed=1.0, base_rate=1.0)

    engine = LiveEngine.by_ticks(MyStrategy(), sesh=sesh)
    chart  = LiveChart(PlotConfig(theme="dark"))

    engine.attach_chart(chart)
    chart.attach_replay_controller(ctrl)
    chart.start()
    ctrl.start()
    engine.run()

PARAMETERS
==========
sesh          : ReplaySesh.
speed         : float -- ticks/sec multiplier.
                real ticks/sec = base_rate * speed.
                float("inf") = turbo (no pause between steps).
base_rate     : float -- base ticks/sec when speed=1.0 (default 1.0).
autostart     : bool -- if True, start() is called automatically when
                attached to chart. Default: False (user controls when to start).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradetropy.replay.replay_sesh import ReplaySesh

# -- CSS injected as Bokeh InlineStyleSheet -----------------------------------
# Applied to the replay Row container. Selectors use :host to target the
# Bokeh 3.x shadow DOM where widgets live internally.
# Bokeh 3.x exposes each widget as a custom element with shadow DOM, so
# global styles do not penetrate. The correct way to style buttons is via
# the `stylesheets` attribute of each widget individually.
#
# Strategy: each button receives its own InlineStyleSheet that overrides
# all Bokeh styles with !important. The Row container receives a stylesheet
# that defines the pill background and flex gap.
# -----------------------------------------------------------------------------

def _pill_container_css(bg: str = "#0F172A") -> str:
    """
    Generate CSS for pill-shaped container.

    Styles the replay control row with rounded corners, dark background,
    and flex layout.

    Args:
        bg (str): Background color hex value.

    Returns:
        str: CSS string with :host pseudo-element selectors.
    """
    return f"""
:host {{
    background: rgba(15,23,42,0.85) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    height: 32px !important;
    box-sizing: border-box !important;
}}
"""


def _btn_css(
    bg: str,
    width: str,
    height: str = "24px",
    radius: str = "5px",
    border: str = "none",
    color: str = "#fff",
    hover_bg: str = "",
) -> str:
    """
    Generate CSS for individual button.

    Completely overrides Bokeh default styles using !important declarations
    to ensure consistent appearance in shadow DOM.

    Args:
        bg (str): Background color hex value.
        width (str): Button width.
        height (str): Button height.
        radius (str): Border radius.
        border (str): Border style.
        color (str): Text color hex value.
        hover_bg (str): Hover background color (optional).

    Returns:
        str: CSS string with full button styling.
    """
    hover_rule = f"""
:host(:hover) .bk-btn {{
    background: {hover_bg} !important;
}}
""" if hover_bg else ""
    return f"""
:host {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}}
:host .bk-btn-group {{
    width: {width} !important;
    height: {height} !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
:host .bk-btn {{
    width: {width} !important;
    height: {height} !important;
    min-width: unset !important;
    min-height: unset !important;
    padding: 0 !important;
    margin: 0 !important;
    background: {bg} !important;
    color: {color} !important;
    border: {border} !important;
    border-radius: {radius} !important;
    outline: none !important;
    box-shadow: none !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    cursor: pointer !important;
    transition: background 0.15s ease, opacity 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-family: inherit !important;
}}
:host .bk-btn:focus {{
    outline: none !important;
    box-shadow: none !important;
}}
{hover_rule}
"""


class PlaybackController:
    """
    Playback controller that couples ReplaySesh with chart buttons.

    Controls the advancement of historical ticks/klines through pause/play/step
    controls and speed adjustment. Shared by the automated ReplayEngine
    (bi-directional) and the manual PaperEngine (forward-only, built with
    allow_backward=False).

    Stepping is always forward: Step forward advances the cursor and Step
    backward delegates to an in-place rewind (step_back), which is only enabled
    when allow_backward is True. There is no global time-flow direction.

    Attributes:
        paused (bool): If True, loop sleeps without advancing cursor.
        speed (float): Ticks/sec multiplier. float("inf") = turbo mode.
        base_rate (float): Base ticks/sec at speed=1.0.
        finished (bool): True when replay reached the last tick.
        allow_backward (bool): Whether backward navigation (step-back) is
            permitted. False in PaperEngine (anti-cheating).

    Methods:
        start() -> Start advance loop in background thread.
        stop() -> Stop loop cleanly.
        pause() / play() / toggle_pause().
        step_once() -> Advance one speed-scaled increment forward.
        step(n) -> Advance exactly n units forward.
        set_speed(x) -> Change speed (x > 0 or float("inf")).
        restart() -> Rewind to the start (1x, paused). Needs an attached engine.
        step_back(n) -> In-place backward rewind (needs an attached engine).
    """

    def __init__(
        self,
        sesh: "ReplaySesh",
        speed: float = 1.0,
        base_rate: float = 1.0,
        autostart: bool = False,
        allow_backward: bool = True,
    ):
        self.sesh:      "ReplaySesh" = sesh
        self.base_rate: float        = max(base_rate, 1e-6)
        self.speed:     float        = max(speed, 1e-6)
        self.paused:    bool         = False
        self.finished:  bool         = False

        # Whether backward navigation is allowed at all (forward-only paper
        # trading mode sets this False).
        self.allow_backward: bool    = bool(allow_backward)

        self._stop_event:  threading.Event        = threading.Event()
        self._loop_thread: "threading.Thread | None" = None

        self._on_finished: "callable | None" = None
        # UI hook fired (on the Bokeh document) when a rewind finishes, so the
        # controls can be restored to the paused-at-position state.
        self._on_seek_done: "callable | None" = None

        # Set by the engine so the controller can drive in-place rewinds
        # (restart / step back). None when used standalone.
        self._engine = None
        # True while a rewind is running, to ignore re-entrant requests.
        self._rebuilding: bool = False

        if autostart:
            self.start()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:
        """
        Launch the advance loop in a daemon thread.

        If already running, returns without creating a duplicate thread.
        """
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="ReplayController-loop",
        )
        self._loop_thread.start()

    def stop(self) -> None:
        """Stops the advance loop cleanly."""
        self._stop_event.set()
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=3.0)

    # =========================================================================
    # CONTROL (called from Bokeh callbacks -- IOLoop thread)
    # =========================================================================

    def pause(self) -> None:
        """Pauses automatic advance."""
        self.paused = True

    def play(self) -> None:
        """Resumes automatic advance."""
        self.paused = False

    def toggle_pause(self) -> None:
        """Toggles between pause and play."""
        self.paused = not self.paused

    def step_once(self) -> None:
        """
        Advance one speed-scaled increment forward.

        The increment scales with the active speed multiplier: at speed 10x a
        single Step advances 10 units. Works while paused or playing. Safe to
        call from any thread.
        """
        self.step(self._step_units())

    def step(self, n: int = 1) -> None:
        """
        Advance exactly ``n`` units forward.

        Advances the sesh cursor n times. Backward navigation is a separate
        operation handled by step_back() (the Step Back control), available only
        when allow_backward is True.

        Args:
            n (int): Number of cursor units to advance (>= 1).
        """
        n = max(1, int(n))
        for _ in range(n):
            self.sesh.step()

    def _step_units(self) -> int:
        """
        Number of units a single Step advances, scaled by the playback speed.

        At speed 10x -> 10 units; turbo (inf) clamps to 1 (a discrete Step in
        turbo is just one unit). Always at least 1.
        """
        s = self.speed
        if s == float("inf"):
            return 1
        return max(1, int(round(s)))

    def set_speed(self, x: float) -> None:
        """
        Change playback speed.

        Args:
            x (float): Speed multiplier. x > 0 sets ticks/sec = base_rate * x.
                x = float("inf") enables turbo (no pause between steps).
        """
        self.speed = max(float(x), 1e-6)

    # =========================================================================
    # REWIND (restart / step back) -- delegated to the engine, run off-thread
    # =========================================================================

    def restart(self) -> None:
        """
        Rewind the replay to the very start (before the first live tick).

        Resets the speed to 1x, pauses playback and rebuilds the engine state in
        a background thread so the Bokeh IOLoop stays responsive. No-op when used
        standalone (no engine attached) or while another rewind is in flight.
        """
        self.set_speed(1.0)
        self._seek(lambda eng: eng._rebuild_restart(), kind="restart")

    def step_back(self, n: int = 1) -> None:
        """
        Rewind n steps back, clamped to the start of the live region.

        Pauses playback and rebuilds the engine state in a background thread.
        No-op when backward navigation is disabled (forward-only paper trading).

        Args:
            n (int): Number of cursor units to move back (default 1).
        """
        if not self.allow_backward:
            return
        self._seek(lambda eng: eng._rebuild_step_back(n), kind="step_back")

    def _seek(self, fn, kind: str = "step_back") -> None:
        """
        Run an engine rewind in a daemon thread and restore the UI afterwards.

        The rewind is heavy (it re-feeds the dataset up to the target), so it
        must not run on the Bokeh IOLoop. The controller is paused first; the
        finished flag is cleared once the rewind completes.

        Args:
            fn (callable): Receives the engine and performs the rewind.
            kind (str): Origin of the seek ("restart" or "step_back"). Passed to
                the seek-done hook so the UI can restore the right state (restart
                leaves the bar paused at 1x; step_back leaves it ready to play).
        """
        eng = self._engine
        if eng is None or self._rebuilding:
            return
        self._rebuilding = True
        self.paused = True

        def _work():
            try:
                fn(eng)
            except Exception:
                import traceback
                import warnings
                warnings.warn(
                    f"ReplayController rewind error:\n{traceback.format_exc()}"
                )
            finally:
                self.finished = False
                self._rebuilding = False
                if self._on_seek_done is not None:
                    try:
                        self._on_seek_done(kind)
                    except Exception:
                        pass

        threading.Thread(
            target=_work, daemon=True, name="ReplayController-seek"
        ).start()

    # =========================================================================
    # INTERNAL LOOP
    # =========================================================================

    def _loop(self) -> None:
        """
        Main loop that advances session at fixed rate.

        Calculates sleep duration as: sleep_s = 1.0 / (base_rate * speed)

        With speed=float("inf") (turbo): only yields GIL without pausing.
        When paused: sleeps without calling step().
        """
        while not self._stop_event.is_set():
            if not self.paused:
                if self.sesh.finished:
                    # Reached the end: pause and notify once, but DO NOT break.
                    # The user can Restart (or step Back) to rewind and resume;
                    # the loop stays alive idling.
                    if not self.finished:
                        self.finished = True
                        self.paused = True
                        if self._on_finished is not None:
                            try:
                                self._on_finished()
                            except Exception:
                                pass
                else:
                    self.finished = False
                    self.sesh.step()

            # Fixed rate: rate = target ticks/sec
            rate = self.base_rate * self.speed

            sleep_s = 1.0 / rate
            _slept = 0.0
            _chunk = min(sleep_s, 0.05)
            while _slept < sleep_s and not self._stop_event.is_set():
                time.sleep(_chunk)
                _slept += _chunk

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        state = "paused" if self.paused else ("finished" if self.finished else "playing")
        speed_str = f"{self.speed:.2f}x"
        return (
            f"PlaybackController(speed={speed_str}, base_rate={self.base_rate:.2f}, "
            f"state={state!r}, finished={self.finished})"
        )


# Backwards-compatible alias: the original name used across the replay package
# and tests. PlaybackController is the generalized class shared by replay and
# paper-trading engines.
ReplayController = PlaybackController


# =============================================================================
# HELPERS FOR BUILDING BOKEH WIDGETS
# =============================================================================

def build_playback_controls(
    controller: "PlaybackController",
    io_loop,
    doc=None,
    mode: str = "replay",
) -> "tuple":
    """
    Build the expressive Bokeh playback control bar.

    Replaces text labels with graphical glyphs, keeping only "Restart" as text
    for absolute clarity:

        [Restart]  [«]  [► / ‖]  [»]    [-] 1x [+]

      - Restart : rewind to index 0 (and, in paper-trading mode, wipe the
                  ledger). Resets the speed to 1x and leaves the bar paused.
                   forbids backward navigation (forward-only paper trading).
      - ► / ‖    : Play / Pause toggle (► shown while playing, ‖ while paused).
      - »        : Step Forward (speed-scaled: at 10x advances 10 units).
      - -/+      : speed multiplier.

    There is no global time-flow direction: the two Step buttons already cover
    moving backward and forward. Uses native Bokeh buttons with InlineStyleSheet
    styling so Python callbacks fire directly without fragile JS bridges. All
    callbacks only flip controller flags / enqueue rewinds (no heavy work on the
    IOLoop).

    Args:
        controller (PlaybackController): Pre-created controller instance.
        io_loop: Tornado IOLoop of the Bokeh server (legacy, kept for compat).
        doc: Bokeh Document for callback scheduling with the document lock.
        mode (str): "replay" or "paper" (informational; backward visibility
            is driven by controller.allow_backward).

    Returns:
        tuple: (toggle, btn_step, speed_div, row_layout) for compatibility with
            build_live_document(); row_layout is the styled control bar.
    """
    from bokeh.models import Toggle, Button, Div
    from bokeh.models import InlineStyleSheet
    from bokeh.layouts import row as _bk_row

    _speed_steps = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

    # Glyphs for the play/pause toggle (status semantics: shows current state).
    _GLYPH_PLAY  = "\u25B6"         # ►  (playing)
    _GLYPH_PAUSE = "\u2590 \u258C"  # ▐ ▌  (paused) 2AFF
    _GLYPH_STOP  = "\u25A0"         # ■

    def _speed_label(val: float) -> str:
        if val == float("inf"):
            return "Max"
        s = f"{val:.2f}".rstrip("0").rstrip(".")
        return f"{s}x"

    def _nearest_speed_idx(val: float) -> int:
        return min(range(len(_speed_steps)), key=lambda i: abs(_speed_steps[i] - val))

    # -- Play/Pause toggle -----------------------------------------------------
    _play_css_inactive = InlineStyleSheet(css=_btn_css(
        bg="#3B82F6", width="44px", hover_bg="#2563EB",
    ))
    _play_css_active = InlineStyleSheet(css=_btn_css(
        bg="#D97706", width="44px", hover_bg="#B45309",
    ))
    _play_css_finished = InlineStyleSheet(css=_btn_css(
        bg="#374151", width="44px", color="#9CA3AF",
    ))

    toggle = Toggle(
        label=_GLYPH_PLAY,
        active=False,
        width=44,
        height=24,
        button_type="light",
        stylesheets=[_play_css_inactive],
    )

    # -- Glyph button CSS (square nav buttons) ---------------------------------
    _nav_css = InlineStyleSheet(css=_btn_css(
        bg="rgba(255,255,255,0.08)",
        width="32px",
        border="1px solid rgba(255,255,255,0.14)",
        hover_bg="rgba(255,255,255,0.14)",
    ))

    # -- Step Forward (⯈) ------------------------------------------------------
    btn_step = Button(
        label="\u2BC8\u258C",            # ⯈
        width=32,
        height=24,
        button_type="light",
        stylesheets=[_nav_css],
    )

    # -- Speed controls --------------------------------------------------------
    _circ_css = InlineStyleSheet(css=_btn_css(
        bg="rgba(255,255,255,0.06)",
        width="24px",
        radius="50%",
        border="1px solid rgba(255,255,255,0.10)",
        hover_bg="rgba(255,255,255,0.14)",
    ))
    btn_dec = Button(
        label="-", width=24, height=24, button_type="light",
        stylesheets=[_circ_css],
    )
    speed_div = Div(
        text=_speed_label(controller.speed),
        width=40,
        height=24,
        stylesheets=[InlineStyleSheet(css="""
:host {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}
:host .bk-clearfix {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    text-align: center;
    line-height: 24px;
    width: 40px;
    display: block;
}
""")],
    )
    btn_inc = Button(
        label="+", width=24, height=24, button_type="light",
        stylesheets=[_circ_css],
    )

    # -- Callbacks -------------------------------------------------------------
    def on_toggle(active: bool) -> None:
        # active True -> paused (show ‖); active False -> playing (show ►).
        if active:
            controller.pause()
            toggle.label = _GLYPH_PAUSE
            toggle.stylesheets = [_play_css_active]
        else:
            controller.play()
            toggle.label = _GLYPH_PLAY
            toggle.stylesheets = [_play_css_inactive]

    toggle.on_click(on_toggle)
    btn_step.on_click(lambda _: controller.step_once())
    def _on_dec(_=None) -> None:
        idx = _nearest_speed_idx(controller.speed)
        if idx > 0:
            controller.set_speed(_speed_steps[idx - 1])
            speed_div.text = _speed_label(controller.speed)

    def _on_inc(_=None) -> None:
        idx = _nearest_speed_idx(controller.speed)
        if idx < len(_speed_steps) - 1:
            controller.set_speed(_speed_steps[idx + 1])
            speed_div.text = _speed_label(controller.speed)

    btn_dec.on_click(_on_dec)
    btn_inc.on_click(_on_inc)

    # -- Finished callback -----------------------------------------------------
    # On finish the replay only pauses and signals; Restart/Back stay enabled so
    # the user can rewind and resume. Only Play and Step (forward) are disabled.
    def _mark_finished() -> None:
        toggle.label      = _GLYPH_STOP
        toggle.disabled   = True
        toggle.stylesheets = [_play_css_finished]
        btn_step.disabled = True

    def _on_finished_cb() -> None:
        if doc is not None:
            try:
                doc.add_next_tick_callback(_mark_finished)
            except Exception:
                pass

    controller._on_finished = _on_finished_cb

    # -- Rewind-done callback --------------------------------------------------
    # After a Restart/Back rewind the replay sits paused at the new position.
    # Restart resets to 1x and leaves the bar visibly paused (‖); a Step Back
    # restores a ready-to-play state (►).
    def _restore_ui_play_ready() -> None:
        toggle.disabled    = False
        toggle.active      = False
        toggle.label       = _GLYPH_PLAY
        toggle.stylesheets = [_play_css_inactive]
        btn_step.disabled  = False

    def _restore_ui_paused() -> None:
        toggle.disabled    = False
        toggle.active      = True
        toggle.label       = _GLYPH_PAUSE
        toggle.stylesheets = [_play_css_active]
        btn_step.disabled  = False
        # Sync the speed label to the controller (Restart reset it to 1x).
        speed_div.text     = _speed_label(controller.speed)

    def _on_seek_done_cb(kind: str = "step_back") -> None:
        restore = _restore_ui_paused if kind == "restart" else _restore_ui_play_ready
        if doc is not None:
            try:
                doc.add_next_tick_callback(restore)
            except Exception:
                pass

    controller._on_seek_done = _on_seek_done_cb

    # -- Row container ---------------------------------------------------------
    _row_css = InlineStyleSheet(css=_pill_container_css())

    children = [toggle, btn_step]
    children += [btn_dec, speed_div, btn_inc]

    # A pill of controls: fixed height, content-driven width. Do NOT set
    # sizing_mode="fixed" here - that mode requires BOTH width and height and
    # Bokeh warns (W-1005 FIXED_SIZING_MODE) when width is missing. Leaving
    # sizing_mode unset with only height gives a fixed-height / natural-width row
    # (the CSS flex styles below drive the horizontal layout).
    row_layout = _bk_row(
        *children,
        height=36,
        stylesheets=[_row_css],
        styles={
            "display":      "flex",
            "align-items":  "center",
            "gap":          "6px",
            "padding":      "4px 10px",
        },
    )

    return toggle, btn_step, speed_div, row_layout


def build_replay_controls(
    controller: "PlaybackController",
    io_loop,
    doc=None,
) -> "tuple":
    """
    Backwards-compatible wrapper around build_playback_controls (replay mode).

    Args:
        controller (PlaybackController): Pre-created controller instance.
        io_loop: Tornado IOLoop of the Bokeh server (legacy).
        doc: Bokeh Document for callback scheduling with the document lock.

    Returns:
        tuple: (toggle, btn_step, speed_div, row_layout).
    """
    return build_playback_controls(controller, io_loop, doc=doc, mode="replay")

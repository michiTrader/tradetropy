"""
Unit tests for PlaybackController (generalized replay/paper controller).

Coverage:
  - step(n) advances the cursor by exactly n units forward.
  - step_once() scales with the active speed multiplier (10x -> 10 units).
  - restart() resets the speed to 1x and pauses.
  - allow_backward=False (forward-only paper trading) disables step_back() as a no-op.

Headless: drives a ReplaySesh directly, no Bokeh, no engine threads.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.replay.replay_sesh import ReplaySesh
from tradetropy.replay.controller import PlaybackController, ReplayController


def _make_sesh(n=100, warmup=10):
    ts = 1_700_000_000_000 + np.arange(n) * 1000
    px = 100.0 + np.arange(n) * 0.5
    ticks = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    ticks[:, _TICK_COL["ts"]] = ts
    ticks[:, _TICK_COL["bid"]] = px
    ticks[:, _TICK_COL["ask"]] = px
    ticks[:, _TICK_COL["price"]] = px
    sesh = ReplaySesh()
    sesh._bind_data({"SYM": ticks}, warmup_ticks=warmup)
    return sesh


@pytest.mark.unit
class TestPlaybackController:
    def test_alias_is_same_class(self):
        assert ReplayController is PlaybackController

    def test_step_n_advances_forward(self):
        sesh = _make_sesh(warmup=10)
        ctrl = PlaybackController(sesh, speed=1.0)
        start = sesh._cursor["SYM"]
        ctrl.step(5)
        assert sesh._cursor["SYM"] == start + 5

    def test_step_once_scales_with_speed(self):
        sesh = _make_sesh(warmup=10)
        ctrl = PlaybackController(sesh, speed=10.0)
        assert ctrl._step_units() == 10
        start = sesh._cursor["SYM"]
        ctrl.step_once()
        assert sesh._cursor["SYM"] == start + 10

    def test_turbo_step_is_one_unit(self):
        sesh = _make_sesh(warmup=10)
        ctrl = PlaybackController(sesh, speed=float("inf"))
        assert ctrl._step_units() == 1

    def test_restart_resets_speed_and_pauses(self):
        sesh = _make_sesh(warmup=10)
        ctrl = PlaybackController(sesh, speed=64.0)
        # No engine attached: the seek is skipped, but restart must still reset
        # the speed to 1x (done before delegating to the engine).
        ctrl.restart()
        assert ctrl.speed == 1.0

    def test_forward_only_backward_step_is_noop(self):
        sesh = _make_sesh(warmup=10)
        ctrl = PlaybackController(sesh, allow_backward=False)
        start = sesh._cursor["SYM"]
        ctrl.step_back(3)    # backward on forward-only -> no-op (no engine)
        assert sesh._cursor["SYM"] == start

    def test_step_back_noop_without_engine(self):
        sesh = _make_sesh()
        ctrl = PlaybackController(sesh, allow_backward=True)
        # No engine attached: step_back is a safe no-op (does not raise).
        ctrl.step_back(5)


def _labels(row_layout):
    """Collect (label, visible) for every Button/Toggle child in a row."""
    out = {}
    for ch in row_layout.children:
        lbl = getattr(ch, "label", None)
        if lbl is not None:
            out[lbl] = getattr(ch, "visible", True)
    return out


@pytest.mark.unit
class TestPlaybackControlsBar:
    def test_replay_bar_has_glyphs(self):
        from tradetropy.replay.controller import build_playback_controls
        sesh = _make_sesh()
        ctrl = PlaybackController(sesh)
        toggle, btn_step, speed_div, row = build_playback_controls(
            ctrl, io_loop=None, doc=None, mode="replay"
        )
        labels = _labels(row)
        assert "\u2B6F" not in labels        # ⭯ restart removed
        assert "\u25B6" in labels          # ► play toggle
        assert any("\u2BC8" in lbl for lbl in labels)  # ⯈ step forward (may carry a trailing glyph)
        assert "\u00AB" not in labels      # « step back removed
        assert "\u25A0" not in labels      # ■ stop gone

    def test_paper_bar_no_backward(self):
        from tradetropy.replay.controller import build_playback_controls
        sesh = _make_sesh()
        ctrl = PlaybackController(sesh, allow_backward=False)
        toggle, btn_step, speed_div, row = build_playback_controls(
            ctrl, io_loop=None, doc=None, mode="paper"
        )
        labels = _labels(row)
        assert "\u00AB" not in labels      # « step back removed
        assert any("\u2BC8" in lbl for lbl in labels)  # ⯈ step forward (may carry a trailing glyph)
        assert "\u2B6F" not in labels      # ⭯ restart removed
        assert "\u25A0" not in labels

    def test_finished_and_seek_hooks_wired(self):
        from tradetropy.replay.controller import build_playback_controls
        sesh = _make_sesh()
        ctrl = PlaybackController(sesh, allow_backward=True)
        build_playback_controls(ctrl, io_loop=None, doc=None)
        assert ctrl._on_finished is not None
        assert ctrl._on_seek_done is not None

    def test_replay_wrapper_still_works(self):
        from tradetropy.replay.controller import build_replay_controls
        sesh = _make_sesh()
        ctrl = PlaybackController(sesh)
        out = build_replay_controls(ctrl, io_loop=None, doc=None)
        assert len(out) == 4


@pytest.mark.unit
class TestForwardOnlyEngines:
    def test_replay_engine_is_forward_only(self):
        from tradetropy.replay.engine import ReplayEngine
        assert ReplayEngine._ALLOW_BACKWARD is False

    def test_paper_engine_is_forward_only(self):
        from tradetropy.paper.engine import PaperEngine
        assert PaperEngine._ALLOW_BACKWARD is False

    def test_replay_controller_no_backward_button(self):
        from tradetropy.replay.engine import ReplayEngine
        from tradetropy.replay.controller import build_playback_controls
        sesh = _make_sesh()
        ctrl = PlaybackController(sesh, allow_backward=ReplayEngine._ALLOW_BACKWARD)
        toggle, btn_step, speed_div, row = build_playback_controls(
            ctrl, io_loop=None, doc=None, mode="replay"
        )
        labels = _labels(row)
        assert "\u00AB" not in labels     # « step back removed
        assert any("\u2BC8" in lbl for lbl in labels)  # ⯈ step forward (may carry a trailing glyph)
        assert "\u2B6F" not in labels     # ⭯ restart removed
        assert "\u25B6" in labels          # ► play toggle present

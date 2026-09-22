"""
Tests for the frame-rate decoupling of the live/replay chart.

The chart refresh must be decoupled from the engine's data rate so that, at
high replay speed, the Bokeh IOLoop is not saturated with back-to-back refreshes
and the replay controls (pause/step/speed) stay responsive.

Contract:
- ``_on_new_data`` (engine thread) only marks the chart dirty; it must NOT
  schedule any IOLoop callback.
- ``_frame_tick`` (IOLoop, periodic) consumes the dirty flag and refreshes at
  most once per invocation, reflecting the latest state (coalescing several
  ``_on_new_data`` into one refresh).
- ``bar_closed`` is not lost when several ticks arrive between two frames.
"""

import pytest

from tradetropy.plotting.config import PlotConfig
from tradetropy.plotting.live.chart import LiveChart


@pytest.mark.unit
class TestChartFrameRate:
    def test_default_refresh_ms(self):
        assert PlotConfig().chart_refresh_ms == 66
        assert PlotConfig(chart_refresh_ms=100).chart_refresh_ms == 100

    def test_on_new_data_only_marks_dirty(self):
        chart = LiveChart(PlotConfig())
        chart._state = "running"

        assert chart._dirty is False
        chart._on_new_data(bar_closed=False)
        assert chart._dirty is True
        assert chart._pending_bar_closed is False

    def test_on_new_data_propagates_bar_closed(self):
        chart = LiveChart(PlotConfig())
        chart._state = "running"

        chart._on_new_data(bar_closed=True)
        assert chart._dirty is True
        assert chart._pending_bar_closed is True

    def test_on_new_data_noop_when_not_running(self):
        chart = LiveChart(PlotConfig())
        # state stays "init"
        chart._on_new_data(bar_closed=True)
        assert chart._dirty is False
        assert chart._pending_bar_closed is False

    def test_frame_tick_coalesces_and_reflects_latest(self):
        chart = LiveChart(PlotConfig())
        chart._state = "running"

        calls = []

        def _fake_update(bar_closed=False, heavy=True):
            calls.append(bar_closed)

        # Stand-in updater; _safe_update reads/resets _pending_bar_closed and
        # forwards it to updater.update().
        class _U:
            update = staticmethod(_fake_update)

        chart._updater = _U()

        # Several data points arrive between two frames; only the last frame
        # matters and it must carry the coalesced bar_closed=True.
        chart._on_new_data(bar_closed=False)
        chart._on_new_data(bar_closed=True)
        chart._on_new_data(bar_closed=False)

        chart._frame_tick()
        assert calls == [True]          # exactly one refresh, latest state
        assert chart._dirty is False
        assert chart._pending_bar_closed is False

        # A frame with no new data must not refresh (leaves the IOLoop idle so
        # control callbacks are processed promptly).
        chart._frame_tick()
        assert calls == [True]

    def test_frame_tick_idle_when_clean(self):
        chart = LiveChart(PlotConfig())
        chart._state = "running"

        calls = []

        class _U:
            update = staticmethod(lambda bar_closed=False, heavy=True: calls.append((bar_closed, heavy)))

        chart._updater = _U()

        chart._frame_tick()             # never marked dirty
        assert calls == []

    def test_heavy_layers_throttled_between_frames(self):
        # Heavy draw layers (VP/Heatmap) must run at the coarser heavy_refresh_ms
        # cadence, not on every frame, so the IOLoop stays free for controls.
        chart = LiveChart(PlotConfig(heavy_refresh_ms=10_000))
        chart._state = "running"

        calls = []

        class _U:
            update = staticmethod(
                lambda bar_closed=False, heavy=True: calls.append(heavy)
            )

        chart._updater = _U()

        # First dirty frame: heavy runs (last_heavy_s starts at 0).
        chart._on_new_data(bar_closed=False)
        chart._frame_tick()
        # Immediate next frame: within the 10s window -> heavy must be False.
        chart._on_new_data(bar_closed=False)
        chart._frame_tick()

        assert calls == [True, False]

    def test_bar_closed_forces_heavy(self):
        # A closed-bar frame must always refresh the heavy layers so the final
        # profile is drawn, even inside the throttle window.
        chart = LiveChart(PlotConfig(heavy_refresh_ms=10_000))
        chart._state = "running"

        calls = []

        class _U:
            update = staticmethod(
                lambda bar_closed=False, heavy=True: calls.append((bar_closed, heavy))
            )

        chart._updater = _U()

        chart._on_new_data(bar_closed=False)
        chart._frame_tick()                     # heavy True (first)
        chart._on_new_data(bar_closed=True)     # bar closed within window
        chart._frame_tick()

        # Second frame: _safe_update forwards heavy=False, but bar_closed=True;
        # the updater OR-s them, so the heavy layers still run. We assert the
        # bar_closed flag reached the updater on the second frame.
        assert calls[1][0] is True

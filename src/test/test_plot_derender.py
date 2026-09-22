"""
Global pan/zoom derender mechanism (PlotConfig.derender_on_pan + _configure_derender).

Heavy overlay glyphs (scatter markers, text labels) collected via the same
(label, group) contract used by ``_configure_lazy_labels`` are hidden while
the user pans/wheel-zooms any panel and restored once the interaction
settles. Verified at the wiring level (CustomJS attached to PanStart/PanEnd/
MouseWheel on every figure, with the registered renderer list non-empty) and
at the end-to-end PlotConfig level (enabled by default, disable via kwarg).
"""
import pytest

pytest.importorskip("bokeh")

from bokeh.plotting import figure

from tradetropy.plotting._layout import _configure_derender
from tradetropy.plotting.config import PlotConfig


def _fig():
    return figure(width=400, height=200)


class TestConfigureDerender:
    def test_wires_pan_and_wheel_events_when_enabled(self):
        fig = _fig()
        renderer = fig.scatter(x=[1, 2], y=[1, 2])
        entries = [(renderer, None)]

        _configure_derender([fig], entries, enabled=True)

        js_callbacks = fig.js_event_callbacks
        # Bokeh keys js_event_callbacks by the event class' fully qualified
        # event_name; PanStart/PanEnd/MouseWheel must all have a registered
        # CustomJS after wiring.
        registered_events = {name for name in js_callbacks.keys()}
        assert any("pan" in name.lower() for name in registered_events)
        assert any("wheel" in name.lower() for name in registered_events)

    def test_does_nothing_when_disabled(self):
        fig = _fig()
        renderer = fig.scatter(x=[1, 2], y=[1, 2])
        entries = [(renderer, None)]

        _configure_derender([fig], entries, enabled=False)

        assert not fig.js_event_callbacks

    def test_does_nothing_when_entries_empty(self):
        fig = _fig()
        _configure_derender([fig], [], enabled=True)
        assert not fig.js_event_callbacks

    def test_callback_args_carry_the_registered_renderers(self):
        fig = _fig()
        r1 = fig.scatter(x=[1], y=[1])
        r2 = fig.scatter(x=[2], y=[2])
        entries = [(r1, None), (r2, None)]

        _configure_derender([fig], entries, enabled=True)

        any_cb = next(iter(fig.js_event_callbacks.values()))[0]
        assert list(any_cb.args["renderers"]) == [r1, r2]


class TestPlotConfigDerenderFlag:
    def test_default_is_enabled(self):
        assert PlotConfig().derender_on_pan is True

    def test_can_be_disabled(self):
        assert PlotConfig(derender_on_pan=False).derender_on_pan is False

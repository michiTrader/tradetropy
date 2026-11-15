"""
Global pan/zoom derender mechanism (PlotConfig.derender_on_pan + _configure_derender).

Scatter markers and text labels are hidden during a gesture. Markers restore
their previous visibility and labels follow their legend group and zoom level.
"""
import pytest

pytest.importorskip("bokeh")

from bokeh.plotting import figure

from tradetropy.plotting._layout import _configure_derender
from tradetropy.plotting.config import PlotConfig


def _fig():
    return figure(width=400, height=200)


def _wire(fig, renderers, *, enabled=True):
    _configure_derender([fig], renderers, label_entries=[], x_range=fig.x_range,
                         interval_ms=1, zoom_range=3, enabled=enabled)


def test_pan_and_wheel_preserve_series_visibility():
    import shutil
    import subprocess
    from tradetropy.plotting._layout import _JS_DIR

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the chart callback")
    script = """
const assert = require('node:assert/strict');
const vm = require('node:vm');
const code = require('node:fs').readFileSync(process.argv[1], 'utf8');
const hidden = {visible: false};
const visible = {visible: true};
const label = {visible: true};
const window = {};
const range = {start: 0, end: 2};
let timer = null;
function dispatch(kind, cb_obj = {}) {
    vm.runInNewContext(code, {
        kind, cb_obj, window, x_range: range, interval_ms: 1, zoom_range: 3,
        renderers: [hidden, visible], labels: [label], groups: [visible],
        clearTimeout: () => { timer = null; }, setTimeout: fn => { timer = fn; },
        queueMicrotask: fn => fn(),
    });
}
dispatch('start');
assert.equal(visible.visible, false);
range.end = 5;
dispatch('end');
assert.equal(hidden.visible, false);
assert.equal(visible.visible, true);
assert.equal(label.visible, false);
dispatch('wheel');
dispatch('wheel');
timer();
assert.equal(hidden.visible, false);
assert.equal(visible.visible, true);
dispatch('wheel');
// Native legend handling flips the temporarily hidden glyph before the callback runs.
visible.visible = true;
dispatch('legend', {item: {renderers: [visible]}});
timer();
assert.equal(visible.visible, false);
"""
    subprocess.run([node, "-e", script, str(_JS_DIR / "derender.js")], check=True)


class TestConfigureDerender:
    def test_wires_pan_and_wheel_events_when_enabled(self):
        fig = _fig()
        renderer = fig.scatter(x=[1, 2], y=[1, 2], legend_label="Swings")
        fig.legend.click_policy = "hide"

        _wire(fig, [renderer])

        js_callbacks = fig.js_event_callbacks
        # Bokeh keys js_event_callbacks by the event class' fully qualified
        # event_name; PanStart/PanEnd/MouseWheel must all have a registered
        # CustomJS after wiring.
        registered_events = {name for name in js_callbacks.keys()}
        assert any("pan" in name.lower() for name in registered_events)
        assert any("wheel" in name.lower() for name in registered_events)
        assert "legend_item_click" in fig.legend[0].js_event_callbacks

    def test_does_nothing_when_disabled(self):
        fig = _fig()
        renderer = fig.scatter(x=[1, 2], y=[1, 2])
        _wire(fig, [renderer], enabled=False)

        assert not fig.js_event_callbacks

    def test_does_nothing_when_entries_empty(self):
        fig = _fig()
        _wire(fig, [])
        assert not fig.js_event_callbacks

    def test_callback_args_carry_the_registered_renderers(self):
        fig = _fig()
        r1 = fig.scatter(x=[1], y=[1])
        r2 = fig.scatter(x=[2], y=[2])
        _wire(fig, [r1, r2])

        any_cb = next(iter(fig.js_event_callbacks.values()))[0]
        assert list(any_cb.args["renderers"]) == [r1, r2]


class TestPlotConfigDerenderFlag:
    def test_default_is_enabled(self):
        assert PlotConfig().derender_on_pan is True

    def test_can_be_disabled(self):
        assert PlotConfig(derender_on_pan=False).derender_on_pan is False

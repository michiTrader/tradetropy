// Preserve the visibility set by the legend across temporary pan/zoom hiding.
// All panels in one chart share the x_range and the same gesture state.
const states = window._tradetropyPanState ??= new WeakMap();
const state = states.get(x_range) ?? {visible: null, dragging: false, timer: null};
states.set(x_range, state);

function hide() {
    state.visible ??= new Map(renderers.map(renderer => [renderer, renderer.visible]));
    for (const renderer of renderers) renderer.visible = false;
    for (const label of labels) label.visible = false;
}

function restore() {
    if (state.visible === null || state.dragging) return;
    for (const [renderer, visible] of state.visible) renderer.visible = visible;
    state.visible = null;
    const zoomed_in = (x_range.end - x_range.start) / interval_ms <= zoom_range;
    for (let i = 0; i < labels.length; i++) {
        labels[i].visible = zoomed_in && (groups[i]?.visible ?? true);
    }
}

if (kind === "legend") {
    if (state.visible !== null) {
        for (const renderer of cb_obj.item.renderers) {
            if (state.visible.has(renderer)) {
                state.visible.set(renderer, !state.visible.get(renderer));
            }
        }
        // Bokeh applies its own toggle on the same pointer event.
        queueMicrotask(() => { if (state.visible !== null) hide(); });
    }
} else {
    clearTimeout(state.timer);
    if (kind === "start") {
        state.dragging = true;
        hide();
    } else if (kind === "end") {
        state.dragging = false;
        restore();
    } else if (kind === "wheel") {
        hide();
        state.timer = setTimeout(restore, 150);
    }
}

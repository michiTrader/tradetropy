// Global derender: temporarily hide heavy overlay glyphs (scatter markers,
// text labels) while the user is actively panning/zooming the chart, and
// restore them once the interaction settles.
//
// Rationale: dragging or wheel-zooming a chart with many marker-based
// indicators (e.g. NBS triangles, ConfirmedPivot labels) repaints every one
// of those glyphs on every frame, which is what makes interaction feel
// sluggish. Bar/line/step renderers are NOT covered here - they stay visible
// and rely on level-of-detail decimation instead (decimate_candles.js /
// decimate_points.js), because blanking a bar mid-drag reads as a visual
// glitch in a way a handful of missing markers does not.
//
// Two distinct triggers drive the same show/hide logic:
//   - PanStart/PanEnd (drag): an exact begin/end pair, no heuristic needed.
//   - MouseWheel (zoom): has no "end" event, so a debounce timer stands in
//     for PanEnd - the same pattern already used by lazy_labels.js.
//
// ``renderers`` and ``groups`` are parallel lists: ``groups[i]`` is the
// renderer's legend group glyph (or null), so a renderer restored after the
// interaction only becomes visible again if the user has not hidden its
// legend entry in the meantime - mirroring lazy_labels.js's policy so pan
// does not fight the user's own legend clicks.
if (window._derender_timeout) clearTimeout(window._derender_timeout);

function _derender_hide() {
    for (let i = 0; i < renderers.length; i++) {
        renderers[i].visible = false;
    }
}

function _derender_restore() {
    for (let i = 0; i < renderers.length; i++) {
        const g = groups[i];
        const group_on = (g === null || g === undefined) ? true : g.visible;
        renderers[i].visible = group_on;
    }
}

if (kind === "start") {
    _derender_hide();
} else if (kind === "end") {
    _derender_restore();
} else if (kind === "wheel") {
    _derender_hide();
    window._derender_timeout = setTimeout(_derender_restore, 150);
}

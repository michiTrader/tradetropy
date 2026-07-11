// Hide every indicator text label when the chart is zoomed too far out.
//
// Attached to the shared price x_range (start/end). When the number of candles
// in view exceeds ``zoom_range`` the labels would overlap into an unreadable
// smear, so they are hidden until the user zooms back in - the same lazy policy
// the footprint uses, generalized to all indicator/tool labels (internal and
// external). ``labels`` and ``groups`` are parallel lists: ``groups[i]`` is the
// label's legend group glyph (or null), so a label only shows when both zoomed
// in AND its legend group is enabled.

if (window._lazy_labels_timeout) clearTimeout(window._lazy_labels_timeout);
window._lazy_labels_timeout = setTimeout(function () {
    const range_ms = cb_obj.end - cb_obj.start;
    const n_candles = range_ms / interval_ms;
    const zoomed_in = n_candles <= zoom_range;
    for (let i = 0; i < labels.length; i++) {
        const g = groups[i];
        const group_on = (g === null || g === undefined) ? true : g.visible;
        labels[i].visible = zoomed_in && group_on;
    }
}, 100);

// Per-group lazy labels (live path).
//
// The static chart collects every label into one callback, but the live chart
// creates label glyphs incrementally as data streams in, so each group attaches
// its own x_range callback at creation time. Hides this group's labels when the
// view exceeds ``zoom_range`` candles, gated by the group's legend glyph.

const range_ms = cb_obj.end - cb_obj.start;
const n_candles = range_ms / interval_ms;
const zoomed_in = n_candles <= zoom_range;
const group_on = (group === null || group === undefined) ? true : group.visible;
const v = zoomed_in && group_on;
for (let i = 0; i < labels.length; i++) {
    labels[i].visible = v;
}

// Autoscale a geometric panel's Y axis from draw-primitive sources.
//
// Own-panel geometric indicators (CVD, DeltaBars, ...) render via draw()
// primitives instead of a value series, so their ColumnDataSources carry quad
// columns (left/right/top/bottom) and segment columns (x0/x1/y0/y1) rather than
// a single ts/value pair. This scans those columns for the visible min/max as
// the X range changes and reshapes the Y range with a small padding.

function _ms(v) {
    return v instanceof Date ? v.getTime() : Number(v);
}

if (typeof ylock !== "undefined" && ylock.data['locked'][0]) {
    return;
}

let lo = Infinity, hi = -Infinity;

for (const src of sources) {
    const d = src.data;

    // Quad rectangles: left/right in time, top/bottom in value.
    const left = d['left'], right = d['right'];
    const top = d['top'], bottom = d['bottom'];
    if (left && right && top && bottom) {
        for (let i = 0; i < top.length; i++) {
            const a = _ms(left[i]);
            const b = _ms(right[i]);
            if (b >= x_range.start && a <= x_range.end) {
                if (bottom[i] < lo) lo = bottom[i];
                if (top[i] < lo) lo = top[i];
                if (bottom[i] > hi) hi = bottom[i];
                if (top[i] > hi) hi = top[i];
            }
        }
    }

    // Segments: (x0, y0) -> (x1, y1).
    const x0 = d['x0'], x1 = d['x1'], y0 = d['y0'], y1 = d['y1'];
    if (x0 && y0 && y1) {
        for (let i = 0; i < y0.length; i++) {
            const a = _ms(x0[i]);
            const b = x1 ? _ms(x1[i]) : a;
            if (b >= x_range.start && a <= x_range.end) {
                if (y0[i] < lo) lo = y0[i];
                if (y1[i] < lo) lo = y1[i];
                if (y0[i] > hi) hi = y0[i];
                if (y1[i] > hi) hi = y1[i];
            }
        }
    }
}

if (lo < Infinity && hi > -Infinity) {
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1.0;
    if (typeof ylock !== "undefined") ylock.data['scaling'][0] = true;
    y_range.start = lo - pad;
    y_range.end = hi + pad;
    if (typeof ylock !== "undefined") ylock.data['scaling'][0] = false;
}

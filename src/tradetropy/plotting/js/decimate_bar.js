// Level-of-detail decimation for a dense bar-style indicator panel (e.g. a
// MACD histogram, a per-bar delta/volume panel).
//
// Unlike the marker/label derender (which simply hides glyphs during pan -
// see derender.js), a bar panel should NEVER blank mid-drag: a vanishing
// histogram reads as a bug. Instead this keeps the FULL bar data in ``full``
// and writes only a bucket-aggregated view into ``view`` (the source the vbar
// glyph draws), refined on every pan/zoom - the same strategy
// decimate_candles.js already uses for OHLC. Each bucket keeps the value with
// the LARGEST absolute magnitude (not an average), so a sharp histogram spike
// is never smoothed away by the reduction; ``bar_color`` (for diverging bars)
// is carried from that same picked row so the sign/color stays correct.
const fd = full.data;
const ts = fd['ts'];
const n = ts ? ts.length : 0;
if (n) {
    const s = x_range.start;
    const e = x_range.end;
    const tnum = (v) => (v instanceof Date ? v.getTime() : Number(v));

    let lo = 0;
    while (lo < n && tnum(ts[lo]) < s) lo++;
    let hi = n;
    while (hi > lo && tnum(ts[hi - 1]) > e) hi--;
    const vis = hi - lo;

    if (vis <= max_visible) {
        // Passthrough: rebuild `view` from the visible slice only, so a
        // zoomed-in window always shows every real bar (mirrors
        // decimate_candles.js's passthrough branch).
        const out = {};
        for (const k of Object.keys(fd)) out[k] = fd[k].slice(lo, hi);
        view.data = out;
    } else {
        const nb = max_visible;
        const out = {};
        for (const k of Object.keys(fd)) out[k] = new Array(nb);
        const values = fd['value'];

        for (let b = 0; b < nb; b++) {
            const a = lo + Math.floor((b * vis) / nb);
            const z = lo + Math.floor(((b + 1) * vis) / nb);  // exclusive
            let pick = a;
            let pick_abs = Math.abs(values[a]);
            for (let i = a + 1; i < z; i++) {
                const av = Math.abs(values[i]);
                if (av > pick_abs) { pick = i; pick_abs = av; }
            }
            for (const k of Object.keys(fd)) out[k][b] = fd[k][pick];
        }
        view.data = out;
    }
}

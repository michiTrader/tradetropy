// Level-of-detail decimation for the candlestick / OHLC source.
//
// A backtest can hold tens of thousands of candles. When zoomed out they
// collapse into a wall of 1-px bars at screen resolution, yet the canvas still
// paints every wick + body AND the O(N) Y-autoscale rescans every row on each
// pan/zoom frame - the real cause of sluggish interaction on big datasets.
//
// This callback keeps the FULL candle data in ``full`` and writes only the
// currently-visible window into ``view`` (the source the glyphs + volume draw
// and the autoscale scans). When the visible window fits under ``max_visible``
// every real candle is shown 1:1. When it exceeds the cap the window is split
// into ``max_visible`` equal-time buckets and each bucket is aggregated into a
// single synthetic candle: open = first open, close = last close, high = max
// high, low = min low, volume = sum. Because the bucket keeps the true high/low
// extremes, the Y-autoscale still frames the panel correctly; the reduction is
// a drawing concern only (``full`` stays intact).
const fd = full.data;
const ts = fd['ts'];
const n = ts ? ts.length : 0;
if (n) {
    const s = x_range.start;
    const e = x_range.end;
    const tnum = (v) => (v instanceof Date ? v.getTime() : Number(v));

    // Visible index span [lo, hi) in the (time-ordered) full data.
    let lo = 0;
    while (lo < n && tnum(ts[lo]) < s) lo++;
    let hi = n;
    while (hi > lo && tnum(ts[hi - 1]) > e) hi--;
    const vis = hi - lo;

    // Columns to rebuild; everything else is carried by last-row-in-bucket.
    const O = fd['Open'], H = fd['High'], L = fd['Low'], C = fd['Close'];
    const V = fd['Volume'];
    const passthrough = vis <= max_visible;

    // Bucket boundaries as row indices into [lo, hi).
    const nb = passthrough ? vis : max_visible;
    const out = {};
    for (const k of Object.keys(fd)) out[k] = new Array(nb);

    const half_default = (n > 1 ? (tnum(ts[1]) - tnum(ts[0])) : 60000) * 0.45;

    for (let b = 0; b < nb; b++) {
        const a = lo + Math.floor((b * vis) / nb);
        const z = lo + Math.floor(((b + 1) * vis) / nb);  // exclusive
        const last = Math.max(a, z - 1);

        // Aggregate OHLCV across [a, z).
        let hi_v = H[a], lo_v = L[a], vol = 0;
        for (let i = a; i < z; i++) {
            if (H[i] > hi_v) hi_v = H[i];
            if (L[i] < lo_v) lo_v = L[i];
            vol += (V ? V[i] : 0);
        }
        const open_v = O[a];
        const close_v = C[last];

        // Carry non-OHLC columns from the bucket's last row (ts, fp_*, ...).
        for (const k of Object.keys(fd)) out[k][b] = fd[k][last];

        out['Open'][b] = open_v;
        out['High'][b] = hi_v;
        out['Low'][b] = lo_v;
        out['Close'][b] = close_v;
        if (V) out['Volume'][b] = vol;
        out['inc'][b] = (close_v >= open_v) ? '1' : '0';
        out['top_body'][b] = Math.max(open_v, close_v);
        out['bottom_body'][b] = Math.min(open_v, close_v);

        // Bar width: real interval when 1:1, else 90% of the bucket time span
        // so aggregated bars tile the axis without overlapping.
        let half;
        if (passthrough) {
            half = half_default;
        } else {
            const t0 = tnum(ts[a]);
            const t1 = tnum(ts[Math.min(z, hi) - 1]);
            const span = (z - a) > 1 ? (t1 - t0) : (half_default * 2);
            half = span * 0.45;
        }
        const tc = tnum(out['ts'][b]);
        out['bar_width'][b] = half * 2;
        if ('ts_left' in out) out['ts_left'][b] = new Date(tc - half);
        if ('ts_right' in out) out['ts_right'][b] = new Date(tc + half);
    }

    view.data = out;
}

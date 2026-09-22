if (typeof ylock !== "undefined" && ylock.data['locked'][0]) {
    return;
}

let lo = Infinity, hi = -Infinity;
for (const src of sources) {
    const x = src.data['ts'];
    const y = src.data['value'];
    if (!x || !y) continue;
    for (let i = 0; i < x.length; i++) {
        const xi = x[i] instanceof Date ? x[i].getTime() : Number(x[i]);
        if (xi >= x_range.start && xi <= x_range.end) {
            if (y[i] < lo) lo = y[i];
            if (y[i] > hi) hi = y[i];
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

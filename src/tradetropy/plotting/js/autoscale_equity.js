const x   = source.data['ts'];
const eq  = source.data['equity'];
const s   = x_range.start;
const e   = x_range.end;
let lo = Infinity, hi = -Infinity;
for (let i = 0; i < x.length; i++) {
    const xi = x[i] instanceof Date ? x[i].getTime() : Number(x[i]);
    if (xi >= s && xi <= e) {
        if (eq[i] < lo) lo = eq[i];
        if (eq[i] > hi) hi = eq[i];
    }
}
if (lo < Infinity && hi > -Infinity) {
    lo = Math.min(lo, baseline);
    hi = Math.max(hi, baseline);
    const pad = (hi - lo) * 0.10 || Math.abs(hi) * 0.05 || 1.0;
    y_range.start = lo - pad;
    y_range.end   = hi + pad;
}

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
// Include the trailing-drawdown line (when present) so its floor never falls
// off-screen: its extremes must drive the Y-range exactly like the equity's.
if (typeof tdd_source !== 'undefined' && tdd_source) {
    const tx  = tdd_source.data['ts'];
    const tdd = tdd_source.data['trailing_dd'];
    for (let i = 0; i < tx.length; i++) {
        const xi = tx[i] instanceof Date ? tx[i].getTime() : Number(tx[i]);
        if (xi >= s && xi <= e) {
            if (tdd[i] < lo) lo = tdd[i];
            if (tdd[i] > hi) hi = tdd[i];
        }
    }
}
if (lo < Infinity && hi > -Infinity) {
    lo = Math.min(lo, baseline);
    hi = Math.max(hi, baseline);
    const pad = (hi - lo) * 0.10 || Math.abs(hi) * 0.05 || 1.0;
    y_range.start = lo - pad;
    y_range.end   = hi + pad;
}

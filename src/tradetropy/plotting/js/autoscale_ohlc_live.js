if (typeof ylock !== "undefined" && ylock.data['locked'][0]) {
    return;
}

const data = source.data;
const x    = data['ts'];
const s    = x_range.start;
const e    = x_range.end;
let lo = Infinity, hi = -Infinity;

const high = data['High'];
const low  = data['Low'];
for (let i = 0; i < x.length; i++) {
    if (x[i] >= s && x[i] <= e) {
        if (low[i]  < lo) lo = low[i];
        if (high[i] > hi) hi = high[i];
    }
}

// Include the visible heatmap cells so the autoscale never clips the liquidity
// grid above/below the candles.
if (typeof hm_sources !== "undefined" && hm_sources) {
    for (const hs of hm_sources) {
        const hd = hs.data;
        const hleft = hd['left'], hright = hd['right'];
        const hbottom = hd['bottom'], htop = hd['top'];
        if (!hleft || !hright || !hbottom || !htop) continue;
        for (let i = 0; i < hleft.length; i++) {
            const a = hleft[i] instanceof Date ? hleft[i].getTime() : Number(hleft[i]);
            const b = hright[i] instanceof Date ? hright[i].getTime() : Number(hright[i]);
            if (b >= s && a <= e) {
                if (hbottom[i] < lo) lo = hbottom[i];
                if (htop[i] > hi) hi = htop[i];
            }
        }
    }
}

if (lo < Infinity && hi > -Infinity) {
    const fp_ch = fp_source_bid.data['cell_h'];
    const fp_half = (fp_ch && fp_ch.length > 0) ? fp_ch[0] / 2.0 : 0.0;
    lo -= fp_half;
    hi += fp_half;

    const pad = (hi - lo) * pad_factor;
    const new_lo = lo - pad;
    const new_hi = hi + pad;
    if (typeof ylock !== "undefined") ylock.data['scaling'][0] = true;
    y_range.start = new_lo;
    y_range.end   = new_hi;
    if (typeof ylock !== "undefined") ylock.data['scaling'][0] = false;

    const ts = fp_ch && fp_ch.length > 0 ? fp_ch[0] : 0;
    if (ts > 0 && y_ticker) {
        const rng = new_hi - new_lo;
        const step = Math.max(ts, Math.ceil(rng / 8 / ts) * ts);
        const first = Math.ceil(new_lo / ts) * ts;
        const ticks = [];
        for (let t = first; t <= new_hi && ticks.length < 30; t = Math.round((t + step) * 1e10) / 1e10) {
            ticks.push(t);
        }
        y_ticker.ticks = ticks;
    }
}

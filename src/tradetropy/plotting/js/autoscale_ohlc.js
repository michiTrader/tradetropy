if (typeof ylock !== "undefined" && ylock.data['locked'][0]) {
    return;
}

const data = source.data;
const x    = data['ts'];
const s    = x_range.start;
const e    = x_range.end;
let lo = Infinity, hi = -Infinity;

if (mode === "ohlc") {
    const high = data['High'];
    const low  = data['Low'];
    for (let i = 0; i < x.length; i++) {
        if (x[i] >= s && x[i] <= e) {
            if (low[i]  < lo) lo = low[i];
            if (high[i] > hi) hi = high[i];
        }
    }
} else if (mode === "volume") {
    const vol = data['Volume'];
    lo = 0;
    for (let i = 0; i < x.length; i++) {
        if (x[i] >= s && x[i] <= e) {
            if (vol[i] > hi) hi = vol[i];
        }
    }
}

// Include the visible heatmap cells so the autoscale never clips the liquidity
// grid above/below the candles.
if (typeof hm_sources !== "undefined" && hm_sources) {
    for (const hs of hm_sources) {
        const hd = hs.data;
        const left = hd['left'], right = hd['right'];
        const bottom = hd['bottom'], top = hd['top'];
        if (!left || !right || !bottom || !top) continue;
        for (let i = 0; i < left.length; i++) {
            const a = left[i] instanceof Date ? left[i].getTime() : Number(left[i]);
            const b = right[i] instanceof Date ? right[i].getTime() : Number(right[i]);
            if (b >= s && a <= e) {
                if (bottom[i] < lo) lo = bottom[i];
                if (top[i] > hi) hi = top[i];
            }
        }
    }
}

if (lo < Infinity && hi > -Infinity) {
    const fp_half_tick = fp_tick_size / 2;
    lo -= fp_half_tick;
    hi += fp_half_tick;

    const pad = (hi - lo) * pad_factor;
    if (typeof ylock !== "undefined") ylock.data['scaling'][0] = true;
    y_range.start = lo - pad;
    y_range.end   = hi + pad;
    if (typeof ylock !== "undefined") ylock.data['scaling'][0] = false;
}

const lo = y_range.start;
const hi = y_range.end;
const rng = hi - lo;
const step = Math.max(ts, Math.ceil(rng / 8 / ts) * ts);
const first = Math.ceil(lo / ts) * ts;
const ticks = [];
for (let t = first; t <= hi && ticks.length < 30;
     t = Math.round((t + step) * 1e10) / 1e10) {
    ticks.push(t);
}
ticker.ticks = ticks;

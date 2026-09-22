const data = source.data;
const x    = data['ts'];
const vol  = data['Volume'];
const s    = x_range.start;
const e    = x_range.end;
let hi = 0;
for (let i = 0; i < x.length; i++) {
    if (x[i] >= s && x[i] <= e) {
        if (vol[i] > hi) hi = vol[i];
    }
}
if (hi > 0) {
    y_range.start = 0;
    y_range.end   = hi / 0.20;
}

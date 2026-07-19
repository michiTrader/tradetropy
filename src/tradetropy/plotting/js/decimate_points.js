// Level-of-detail decimation for a large point cloud (scatter).
//
// A dense Points glyph (order-flow bubbles, executed-volume dots, ...) can hold
// tens of thousands of markers. When zoomed out they overlap into a blob at
// screen resolution yet the canvas still paints every one, so pan/zoom turns
// sluggish. This callback keeps the FULL cloud in ``full`` and writes only a
// capped, uniformly-spaced sample of the CURRENTLY VISIBLE points into ``view``
// (the source the glyph draws). Zoom in -> fewer points in range -> all shown;
// zoom out -> many in range -> subsampled to ``max_visible``. The reduction is
// a drawing concern only; the full data stays intact for hover/analysis when
// zoomed in enough that the whole window fits under the cap.
const fd = full.data;
const x  = fd['x'];
if (x && x.length) {
    const s = x_range.start;
    const e = x_range.end;

    // Visible indices in the current x window.
    const idx = [];
    for (let i = 0; i < x.length; i++) {
        const t = x[i] instanceof Date ? x[i].getTime() : Number(x[i]);
        if (t >= s && t <= e) idx.push(i);
    }

    // Uniformly subsample to the cap while preserving time order.
    let sel;
    if (idx.length > max_visible) {
        sel = new Array(max_visible);
        const step = idx.length / max_visible;
        for (let k = 0; k < max_visible; k++) {
            sel[k] = idx[Math.floor(k * step)];
        }
    } else {
        sel = idx;
    }

    // Rebuild every column of the drawn source by picking the selected rows.
    const out = {};
    for (const key of Object.keys(fd)) {
        const col = fd[key];
        const dst = new Array(sel.length);
        for (let k = 0; k < sel.length; k++) dst[k] = col[sel[k]];
        out[key] = dst;
    }
    view.data = out;
}

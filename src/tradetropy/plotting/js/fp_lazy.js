txt_bid.visible = false;
txt_ask.visible = false;
rects_bid.visible = false;
rects_ask.visible = false;
if (ohlc_source) {
    ohlc_source.data['bar_width'] = ohlc_source.data['bar_width'].map(function() { return bar_width_wide; });
    ohlc_source.change.emit();
}
if (window._fp_timeout) clearTimeout(window._fp_timeout);
window._fp_timeout = setTimeout(function() {
    if (dummy !== null && !dummy.visible) { return; }
    const range_ms = cb_obj.end - cb_obj.start;
    const n_candles = range_ms / interval_ms;
    if (n_candles <= zoom_range) {
        txt_bid.visible = true;
        txt_ask.visible = true;
        rects_bid.visible = true;
        rects_ask.visible = true;
        if (ohlc_source) {
            ohlc_source.data['bar_width'] = ohlc_source.data['bar_width'].map(function() { return bar_width_narrow; });
            ohlc_source.change.emit();
        }
    }
}, 100);

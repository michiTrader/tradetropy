const visible = dummy.visible;
fp_renderers.forEach(r => { r.visible = visible; });
if (ohlc_source) {
    ohlc_source.data['bar_width'] = ohlc_source.data['bar_width'].map(
        function() { return visible ? bar_width_narrow : bar_width_wide; }
    );
    ohlc_source.change.emit();
}

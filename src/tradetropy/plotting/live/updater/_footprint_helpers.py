import numpy as np
from tradetropy.plotting._util import _fmt_vol, _fp_level_color


def _build_fp_rows_for_candle(
    candle,
    ts_ms: int,
    interval_ms: int,
    theme: dict,
    tick_size: float = 1.0,
    price_range: tuple[float, float] | None = None,
) -> tuple[dict, dict]:
    bid_scale   = theme["fp_bid_scale"]
    ask_scale   = theme["fp_ask_scale"]
    poc_fill    = theme["fp_poc_fill"]
    poc_border  = theme["fp_poc_text"]
    text_normal = theme["fp_text"]
    poc_text    = theme["fp_poc_text"]
    bg_hex      = theme["bg"]

    half_bar_ms = int(interval_ms * 0.40)
    offset_ms   = int(interval_ms * 0.26)

    if candle.levels == 0:
        if price_range is not None:
            low, high = price_range
            empty_levels = np.arange(low, high + tick_size / 2, tick_size)
            n_empty = len(empty_levels)
            x_bid_arr = np.array(
                [np.datetime64(ts_ms - offset_ms, "ms")] * n_empty,
                dtype="datetime64[ms]",
            )
            x_ask_arr = np.array(
                [np.datetime64(ts_ms + offset_ms, "ms")] * n_empty,
                dtype="datetime64[ms]",
            )
            y_arr   = np.array(empty_levels, dtype=np.float64)
            empty_s = np.full(n_empty, "", dtype=object)
            bg_arr  = np.full(n_empty, bg_hex, dtype=object)
            lw_arr  = np.full(n_empty, 0.5, dtype=np.float64)
            cw_arr  = np.full(n_empty, half_bar_ms, dtype=np.float64)
            ch_arr  = np.full(n_empty, tick_size, dtype=np.float64)

            bid_cols = dict(
                x=x_bid_arr, y=y_arr, text=empty_s.copy(),
                fill=bg_arr.copy(), lc=bg_arr.copy(),
                lw=lw_arr.copy(), tc=bg_arr.copy(),
                cell_w=cw_arr.copy(), cell_h=ch_arr.copy(),
            )
            ask_cols = dict(
                x=x_ask_arr, y=y_arr, text=empty_s.copy(),
                fill=bg_arr.copy(), lc=bg_arr.copy(),
                lw=lw_arr.copy(), tc=bg_arr.copy(),
                cell_w=cw_arr.copy(), cell_h=ch_arr.copy(),
            )
            return bid_cols, ask_cols
        else:
            empty_s = np.array([], dtype=object)
            empty_f = np.array([], dtype=np.float64)
            empty_t = np.array([], dtype="datetime64[ms]")
            empty = dict(
                x=empty_t, y=empty_f, text=empty_s,
                fill=empty_s, lc=empty_s,
                lw=empty_f, tc=empty_s,
                cell_w=empty_f, cell_h=empty_f,
            )
            return dict(empty), dict(empty)

    poc_vol = candle.poc_vol if candle.poc_vol > 0 else 1.0

    bid_x, bid_y, bid_txt = [], [], []
    bid_fill, bid_lc, bid_lw, bid_tc = [], [], [], []

    ask_x, ask_y, ask_txt = [], [], []
    ask_fill, ask_lc, ask_lw, ask_tc = [], [], [], []

    bid_cell_w, bid_cell_h = [], []
    ask_cell_w, ask_cell_h = [], []

    for j in range(candle.levels):
        price = float(candle.price_levels[j, 0])
        vb    = float(candle.price_levels[j, 1])
        va    = float(candle.price_levels[j, 2])
        is_poc = (j == candle.poc_idx)

        x_bid = np.datetime64(ts_ms - offset_ms, "ms")
        x_ask = np.datetime64(ts_ms + offset_ms, "ms")

        if is_poc:
            bid_x.append(x_bid);  bid_y.append(price)
            bid_txt.append(_fmt_vol(vb))
            bid_fill.append(poc_fill); bid_lc.append(poc_border)
            bid_lw.append(0.5);        bid_tc.append(poc_text)
            bid_cell_w.append(half_bar_ms); bid_cell_h.append(tick_size)

            ask_x.append(x_ask);  ask_y.append(price)
            ask_txt.append(_fmt_vol(va))
            ask_fill.append(poc_fill); ask_lc.append(poc_border)
            ask_lw.append(0.5);        ask_tc.append(poc_text)
            ask_cell_w.append(half_bar_ms); ask_cell_h.append(tick_size)
        else:
            if vb > 0:
                intensity_bid = min(1.0, vb / poc_vol)
                bid_x.append(x_bid);  bid_y.append(price)
                bid_txt.append(_fmt_vol(vb))
                bid_fill.append(_fp_level_color(intensity_bid, bid_scale, bg_hex))
                bid_lc.append("white"); bid_lw.append(0)
                bid_tc.append(text_normal)
                bid_cell_w.append(half_bar_ms); bid_cell_h.append(tick_size)

            if va > 0:
                intensity_ask = min(1.0, va / poc_vol)
                ask_x.append(x_ask);  ask_y.append(price)
                ask_txt.append(_fmt_vol(va))
                ask_fill.append(_fp_level_color(intensity_ask, ask_scale, bg_hex))
                ask_lc.append("white"); ask_lw.append(0)
                ask_tc.append(text_normal)
                ask_cell_w.append(half_bar_ms); ask_cell_h.append(tick_size)

    bid_cols = dict(
        x      = np.array(bid_x,       dtype="datetime64[ms]"),
        y      = np.array(bid_y,       dtype=np.float64),
        text   = np.array(bid_txt,     dtype=object),
        fill   = np.array(bid_fill,    dtype=object),
        lc     = np.array(bid_lc,      dtype=object),
        lw     = np.array(bid_lw,      dtype=np.float64),
        tc     = np.array(bid_tc,      dtype=object),
        cell_w = np.array(bid_cell_w,  dtype=np.float64),
        cell_h = np.array(bid_cell_h,  dtype=np.float64),
    )
    ask_cols = dict(
        x      = np.array(ask_x,       dtype="datetime64[ms]"),
        y      = np.array(ask_y,       dtype=np.float64),
        text   = np.array(ask_txt,     dtype=object),
        fill   = np.array(ask_fill,    dtype=object),
        lc     = np.array(ask_lc,      dtype=object),
        lw     = np.array(ask_lw,      dtype=np.float64),
        tc     = np.array(ask_tc,      dtype=object),
        cell_w = np.array(ask_cell_w,  dtype=np.float64),
        cell_h = np.array(ask_cell_h,  dtype=np.float64),
    )
    return bid_cols, ask_cols

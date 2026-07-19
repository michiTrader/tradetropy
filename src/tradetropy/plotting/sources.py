from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .config import IndicatorPlotMeta
from ._util import _ms_to_bokeh_datetime, _fmt_vol, _ohlc_tick_bounds, _align_ts_to_candle

# VPVR (view="visible") histogram geometry now lives in the Bokeh-free core
# module so both render paths share it. Re-exported here for back-compat
# (vp_visible_right_pad_bars is read by the live navigation).
from tradetropy.ta._volume_profile import (
    DEFAULT_VP_BUY,
    DEFAULT_VP_SELL,
    vp_visible_right_pad_bars,
    volume_profile_bar_arrays,
)


# DATA BUILDERS -> ColumnDataSource

def build_ohlc_source(ohlc_array: np.ndarray, interval_ms: int):
    from bokeh.models import ColumnDataSource

    ts = _ms_to_bokeh_datetime(ohlc_array[:, 0])
    open_  = ohlc_array[:, 1]
    high   = ohlc_array[:, 2]
    low    = ohlc_array[:, 3]
    close  = ohlc_array[:, 4]
    volume = ohlc_array[:, 5]

    bar_width_ms = int(interval_ms * 0.9)
    inc = (close >= open_).astype(np.uint8).astype(str)

    ts_left, ts_right = _ohlc_tick_bounds(ohlc_array[:, 0], bar_width_ms)

    return ColumnDataSource(dict(
        ts=ts,
        Open=open_,
        High=high,
        Low=low,
        Close=close,
        Volume=volume,
        inc=inc,
        bar_width=[bar_width_ms] * len(ts),
        top_body=np.maximum(open_, close),
        bottom_body=np.minimum(open_, close),
        ts_left=ts_left,
        ts_right=ts_right,
    ))


def _fp_scalars_from_candle(candle) -> tuple[float, float, float, float, float, float, float]:
    """
    Extract the 7 footprint scalars from an FpCandle for the HoverTool OHLC.

    Returns (bid, ask, delta, poc_price, poc_vol, vah, val). Bid/ask are
    derived from total volume and delta (not stored separately at the candle
    level). Shared between the static plot (_build_footprint_cols) and the
    live updater (_ohlc_mixin) so the tooltip is identical in both modes.
    """
    bid = candle.vol_total * 0.5 + candle.delta_total * 0.5
    ask = candle.vol_total * 0.5 - candle.delta_total * 0.5
    return (bid, ask, candle.delta_total, candle.poc_price,
            candle.poc_vol, candle.vah, candle.val)


def _build_footprint_cols(fp_proxy, n_candles: int) -> dict:
    store = fp_proxy._store
    n_closed = store._n_closed if store is not None else 0

    fp_bid   = [np.nan] * n_candles
    fp_ask   = [np.nan] * n_candles
    fp_delta = [np.nan] * n_candles
    fp_poc_p = [np.nan] * n_candles
    fp_poc_v = [np.nan] * n_candles
    fp_vah   = [np.nan] * n_candles
    fp_val   = [np.nan] * n_candles

    for v_idx in range(n_closed):
        if v_idx >= n_candles:
            break
        vela = store.closed_candle(v_idx)
        if vela is None:
            continue
        (fp_bid[v_idx], fp_ask[v_idx], fp_delta[v_idx], fp_poc_p[v_idx],
         fp_poc_v[v_idx], fp_vah[v_idx], fp_val[v_idx]) = _fp_scalars_from_candle(vela)

    partial_idx = n_closed
    if partial_idx < n_candles:
        vela = store.partial_candle()
        if vela is not None:
            (fp_bid[partial_idx], fp_ask[partial_idx], fp_delta[partial_idx],
             fp_poc_p[partial_idx], fp_poc_v[partial_idx], fp_vah[partial_idx],
             fp_val[partial_idx]) = _fp_scalars_from_candle(vela)

    return dict(
        fp_bid=fp_bid,
        fp_ask=fp_ask,
        fp_delta=fp_delta,
        fp_poc_price=fp_poc_p,
        fp_poc_vol=fp_poc_v,
        fp_vah=fp_vah,
        fp_val=fp_val,
    )


def build_trades_source(trades_df: pd.DataFrame, interval_ms: int = 0, align_to_candle: bool = False, candle_origin_ms: int = 0):
    from bokeh.models import ColumnDataSource

    tc = trades_df[trades_df["exit_time"].notna()].copy()
    if tc.empty:
        return None

    def _ts_to_ms(series: pd.Series) -> np.ndarray:
        return (
            pd.to_datetime(series, utc=True)
            .dt.tz_localize(None)
            .astype("datetime64[ms]")
            .astype("int64")
        )

    entry_ms = _ts_to_ms(tc["entry_time"])
    exit_ms  = _ts_to_ms(tc["exit_time"])

    if align_to_candle and interval_ms > 0:
        entry_ms = _align_ts_to_candle(entry_ms, interval_ms, candle_origin_ms)
        exit_ms = _align_ts_to_candle(exit_ms, interval_ms, candle_origin_ms)

    entry_ts = entry_ms.astype("datetime64[ms]")
    exit_ts  = exit_ms.astype("datetime64[ms]")

    is_win = (tc["pnl_net"] > 0).astype(int).astype(str)

    lines_xs = [[int(e), int(x)] for e, x in zip(entry_ms, exit_ms)]
    lines_ys = list(zip(tc["entry_price"].to_numpy(), tc["exit_price"].to_numpy()))

    # ── Fields for the P&L panel ─────────────────────────────────────────────
    pnl_net_arr     = tc["pnl_net"].to_numpy(dtype=np.float64)
    entry_price_arr = tc["entry_price"].to_numpy(dtype=np.float64)
    sizes_arr       = tc["size"].to_numpy(dtype=np.float64)

    notional = np.where(
        (entry_price_arr > 0) & (np.abs(sizes_arr) > 0),
        np.abs(entry_price_arr * sizes_arr),
        np.nan,
    )
    pnl_pct = np.where(np.isfinite(notional), pnl_net_arr / notional * 100.0, 0.0)

    duration_ms_arr = (exit_ms - entry_ms).astype(np.float64)

    trade_label = [
        f"{'+' if p >= 0 else ''}{p:.2f}%"
        for p in pnl_pct
    ]

    trade_idx = np.arange(len(tc), dtype=np.float64)

    # Signed size for the P&L hover: positive for long, negative for short, so
    # the tooltip alone tells direction without a separate field (mirrors
    # backtesting.py's convention for its trade markers).
    direction_arr = tc["direction"].to_numpy()
    size_signed = np.where(direction_arr == "short", -np.abs(sizes_arr), np.abs(sizes_arr))

    return ColumnDataSource(dict(
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_price=entry_price_arr,
        exit_price=tc["exit_price"].to_numpy(),
        pnl=pnl_net_arr,
        pnl_pct=pnl_pct,
        size=sizes_arr,
        size_signed=size_signed,
        direction=direction_arr,
        is_win=is_win,
        duration_ms=duration_ms_arr,
        trade_label=trade_label,
        trade_idx=trade_idx,
        lines_xs=lines_xs,
        lines_ys=lines_ys,
    ))


def _prepare_equity_series(
    equity_curve: pd.Series,
    initial_balance: float,
    mode: Literal["balance", "return"],
    unit: Literal["currency", "percent"],
) -> tuple[pd.Series, float]:
    if mode == "balance":
        if unit == "currency":
            return equity_curve.copy(), initial_balance
        else:
            return equity_curve / initial_balance, 1.0
    else:
        if unit == "currency":
            return equity_curve - initial_balance, 0.0
        else:
            return equity_curve / initial_balance - 1.0, 0.0


def _downsample_minmax(ts: np.ndarray, vals: np.ndarray, max_points: int = 4000):
    """
    Reduce a long (ts, vals) curve for plotting while preserving its envelope.

    A tick-driven backtest records equity once per tick, so the curve can hold
    hundreds of thousands of points even when the chart shows only a few hundred
    candles. Such a curve is invisible at screen resolution (a 1200 px panel
    cannot show more than ~1200 distinct x positions) yet it makes the line
    heavy AND, worse, makes the O(N) autoscale CustomJS rescan every point on
    every pan/zoom frame - the real cause of the sluggish interaction.

    The curve is split into ``max_points // 2`` equal index buckets and only the
    per-bucket MIN and MAX (in time order) are kept, plus the first and last
    point. This keeps the visible shape and the true extremes - so the equity
    autoscale still frames the panel correctly - at a fraction of the points.
    The reduction is for DRAWING only; stats and the peak/trough markers are
    computed from the full curve upstream.

    Args:
        ts: Timestamps (any dtype indexable in parallel with ``vals``).
        vals (np.ndarray): The curve values.
        max_points (int): Target upper bound on the returned point count.

    Returns:
        tuple: ``(ts, vals)`` unchanged if already small enough, else the
        downsampled pair.
    """
    n = len(vals)
    if n <= max_points or max_points < 4:
        return ts, vals

    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, n_buckets + 1).astype(np.int64)
    keep = [0, n - 1]
    for b in range(n_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        seg = vals[lo:hi]
        finite = np.isfinite(seg)
        if not finite.any():
            keep.append(lo)
            continue
        seg_idx = np.nonzero(finite)[0]
        seg_vals = seg[seg_idx]
        imin = lo + int(seg_idx[np.argmin(seg_vals)])
        imax = lo + int(seg_idx[np.argmax(seg_vals)])
        if imin <= imax:
            keep.append(imin)
            keep.append(imax)
        else:
            keep.append(imax)
            keep.append(imin)

    idx = np.unique(np.asarray(keep, dtype=np.int64))
    return ts[idx], vals[idx]


def build_drawdown_source(equity_curve: pd.Series):
    from bokeh.models import ColumnDataSource
    from tradetropy.stats import drawdown_series, calc_daily_equity

    daily = calc_daily_equity(equity_curve)
    dd    = drawdown_series(daily)
    ts    = dd.index.values.astype("datetime64[ms]")
    vals  = dd.to_numpy(dtype=np.float64) 

    return ColumnDataSource(dict(ts=ts, drawdown=vals))


def build_indicator_source(meta: IndicatorPlotMeta):
    """
    Build the ColumnDataSource for an indicator.

    Cases:
      - 1D [N]    -> list with 1 source.
      - 2D [K x N] -> list with K sources.
      - ts bands (_ts_band_indices) -> empty sources, never plotted.
    """
    from bokeh.models import ColumnDataSource

    ts_default = _ms_to_bokeh_datetime(meta.timestamps)
    values = np.atleast_2d(meta.values)
    n_ts = len(ts_default)
    if values.ndim == 2:
        if values.shape[1] == n_ts and values.shape[0] != n_ts:
            pass
        elif values.shape[0] == n_ts and values.shape[1] != n_ts:
            values = values.T
        elif values.shape[0] > values.shape[1]:
            values = values.T

    ts_band_indices: list[int] = getattr(meta, "_ts_band_indices", [])
    ts_band_set = set(ts_band_indices)
    output_names: list[str] = getattr(meta, "output_names", [])

    _MS_THRESHOLD = 1_000_000_000_000  # year 2001 in ms

    sources = []
    price_band_count = 0

    for i, row in enumerate(values):
        if i in ts_band_set:
            sources.append(ColumnDataSource(dict(
                ts=np.array([], dtype="datetime64[ms]"),
                value=np.array([]),
                tag=np.array([], dtype=object),
            )))
            continue

        ts_row = None
        if ts_band_indices and price_band_count < len(ts_band_indices):
            ts_idx = ts_band_indices[price_band_count]
            if ts_idx < len(values):
                candidate = values[ts_idx]
                finite_vals = candidate[np.isfinite(candidate)]
                if len(finite_vals) > 0 and float(finite_vals[0]) > _MS_THRESHOLD:
                    ts_row = candidate

        tag_label = output_names[price_band_count] if price_band_count < len(output_names) else ""
        price_band_count += 1

        # Align row
        if len(row) != n_ts:
            row = np.r_[row, np.full(n_ts - len(row), np.nan)] if len(row) < n_ts else row[:n_ts]

        if ts_row is not None:
            # Align ts_row
            if len(ts_row) != n_ts:
                ts_row = np.r_[ts_row, np.full(n_ts - len(ts_row), np.nan)] if len(ts_row) < n_ts else ts_row[:n_ts]
            # valid only where BOTH price and ts are finite
            valid = np.isfinite(row) & np.isfinite(ts_row)
            x_vals = ts_row[valid].astype(np.int64).astype("datetime64[ms]")
        else:
            valid = np.isfinite(row)
            x_vals = ts_default[valid]

        y_vals = row[valid]
        n_valid = int(valid.sum())
        sources.append(ColumnDataSource(dict(
            ts=x_vals,
            value=y_vals,
            tag=np.full(n_valid, tag_label, dtype=object),
        )))

    return sources


def trailing_drawdown(balance: pd.Series, max_drawdown: float) -> pd.Series:
    bal = balance.copy()
    bal = bal if isinstance(balance, pd.Series) else pd.Series(balance)
    bal = bal + 1

    trailing_dd = []
    temp_max = 0
    arr = bal.to_numpy(dtype=np.float64)
    for i in range(len(arr)):
        val = arr[i]
        if val > temp_max:
            trailing_dd.append(val - max_drawdown)
            temp_max = val
        else:
            trailing_dd.append(None)

    trailing_dd = pd.Series(trailing_dd, index=bal.index)
    trailing_dd = trailing_dd.ffill()

    return trailing_dd - 1


def build_trailing_dd_source(
    equity_curve: pd.Series,
    initial_balance: float,
    max_trailing_dd: float,
    mode: str,
    unit: str,
):
    from bokeh.models import ColumnDataSource

    mult = equity_curve / initial_balance
    trailing_mult = trailing_drawdown(mult, max_trailing_dd)

    result, _ = _prepare_equity_series(
        trailing_mult * initial_balance, initial_balance, mode, unit
    )

    ts = result.index.values.astype("datetime64[ms]")
    vals = result.to_numpy(dtype=np.float64)

    # Same per-tick length as the equity curve -> decimate for drawing.
    ts, vals = _downsample_minmax(ts, vals)
    return ColumnDataSource(dict(ts=ts, trailing_dd=vals))


def build_footprint_source(fp_proxy, ohlc_array: np.ndarray, interval_ms: int, theme: dict):
    from bokeh.models import ColumnDataSource
    from tradetropy.plotting._util import _fp_level_color

    store = fp_proxy._store
    n_closed = store._n_closed if store is not None else 0
    if n_closed == 0:
        return None

    bid_scale   = theme["fp_bid_scale"]
    ask_scale   = theme["fp_ask_scale"]
    poc_fill    = theme["fp_poc_fill"]
    poc_border  = theme["fp_poc_text"]
    text_normal = theme["fp_text"]
    poc_text    = theme["fp_poc_text"]
    bg_hex      = theme["bg"]

    half_bar_ms = int(interval_ms * 0.40)
    offset_ms   = int(interval_ms * 0.26)

    tick_size = float(fp_proxy.config.tick_size or 1.0)

    x_coords     = []
    y_coords     = []
    texts        = []
    fill_colors  = []
    line_widths  = []
    line_colors  = []
    text_colors  = []
    cell_widths  = []
    cell_heights = []
    side         = []

    for v_idx in range(n_closed):
        candle = store.closed_candle(v_idx)
        if candle is None:
            continue
        if v_idx >= len(ohlc_array):
            continue

        ts_ms = int(ohlc_array[v_idx, 0])
        if candle.levels == 0:
            ohlc_low   = float(ohlc_array[v_idx, 3])
            ohlc_high  = float(ohlc_array[v_idx, 2])
            empty_levels = np.arange(
                ohlc_low, ohlc_high + tick_size / 2, tick_size
            )
            for price in empty_levels:
                x_bid = np.datetime64(ts_ms - offset_ms, "ms")
                x_coords.append(x_bid)
                y_coords.append(price)
                texts.append("")
                fill_colors.append(bg_hex)
                line_colors.append(bg_hex)
                line_widths.append(0)
                text_colors.append(bg_hex)
                cell_widths.append(half_bar_ms)
                cell_heights.append(tick_size)
                side.append("bid")
                x_ask = np.datetime64(ts_ms + offset_ms, "ms")
                x_coords.append(x_ask)
                y_coords.append(price)
                texts.append("")
                fill_colors.append(bg_hex)
                line_colors.append(bg_hex)
                line_widths.append(0)
                text_colors.append(bg_hex)
                cell_widths.append(half_bar_ms)
                cell_heights.append(tick_size)
                side.append("ask")
            continue

        poc_vol = candle.poc_vol if candle.poc_vol > 0 else 1.0

        for j in range(candle.levels):
            price = float(candle.price_levels[j, 0])
            vb    = float(candle.price_levels[j, 1])
            va    = float(candle.price_levels[j, 2])
            is_poc = (j == candle.poc_idx)

            if is_poc:
                x_bid = np.datetime64(ts_ms - offset_ms, "ms")
                x_coords.append(x_bid)
                y_coords.append(price)
                texts.append(_fmt_vol(vb))
                fill_colors.append(poc_fill)
                line_colors.append(poc_border)
                line_widths.append(0.5)
                text_colors.append(poc_text)
                cell_widths.append(half_bar_ms)
                cell_heights.append(tick_size)
                side.append("bid")

                x_ask = np.datetime64(ts_ms + offset_ms, "ms")
                x_coords.append(x_ask)
                y_coords.append(price)
                texts.append(_fmt_vol(va))
                fill_colors.append(poc_fill)
                line_colors.append(poc_border)
                line_widths.append(0.5)
                text_colors.append(poc_text)
                cell_widths.append(half_bar_ms)
                cell_heights.append(tick_size)
                side.append("ask")
            else:
                if vb > 0:
                    x_bid = np.datetime64(ts_ms - offset_ms, "ms")
                    x_coords.append(x_bid)
                    y_coords.append(price)
                    intensity_bid = min(1.0, vb / poc_vol)
                    texts.append(_fmt_vol(vb))
                    fill_colors.append(_fp_level_color(intensity_bid, bid_scale, bg_hex))
                    line_colors.append("white")
                    line_widths.append(0)
                    text_colors.append(text_normal)
                    cell_widths.append(half_bar_ms)
                    cell_heights.append(tick_size)
                    side.append("bid")

                if va > 0:
                    x_ask = np.datetime64(ts_ms + offset_ms, "ms")
                    x_coords.append(x_ask)
                    y_coords.append(price)
                    intensity_ask = min(1.0, va / poc_vol)
                    texts.append(_fmt_vol(va))
                    fill_colors.append(_fp_level_color(intensity_ask, ask_scale, bg_hex))
                    line_colors.append("white")
                    line_widths.append(0)
                    text_colors.append(text_normal)
                    cell_widths.append(half_bar_ms)
                    cell_heights.append(tick_size)
                    side.append("ask")

    partial = store.partial_candle()
    if partial is not None and n_closed < len(ohlc_array):
        ts_ms = int(ohlc_array[n_closed, 0])
        poc_vol = partial.poc_vol if partial.poc_vol > 0 else 1.0
        for j in range(partial.levels):
            price = float(partial.price_levels[j, 0])
            vb = float(partial.price_levels[j, 1])
            va = float(partial.price_levels[j, 2])
            is_poc = (j == partial.poc_idx)

            if is_poc:
                x_coords.append(np.datetime64(ts_ms - offset_ms, "ms"))
                y_coords.append(price)
                texts.append(_fmt_vol(vb))
                fill_colors.append(poc_fill)
                line_colors.append(poc_border)
                line_widths.append(0.5)
                text_colors.append(poc_text)
                cell_widths.append(half_bar_ms)
                cell_heights.append(tick_size)
                side.append("bid")

                x_coords.append(np.datetime64(ts_ms + offset_ms, "ms"))
                y_coords.append(price)
                texts.append(_fmt_vol(va))
                fill_colors.append(poc_fill)
                line_colors.append(poc_border)
                line_widths.append(0.5)
                text_colors.append(poc_text)
                cell_widths.append(half_bar_ms)
                cell_heights.append(tick_size)
                side.append("ask")
            else:
                if vb > 0:
                    x_coords.append(np.datetime64(ts_ms - offset_ms, "ms"))
                    y_coords.append(price)
                    intensity_bid = min(1.0, vb / poc_vol)
                    texts.append(_fmt_vol(vb))
                    fill_colors.append(_fp_level_color(intensity_bid, bid_scale, bg_hex))
                    line_colors.append("white")
                    line_widths.append(0)
                    text_colors.append(text_normal)
                    cell_widths.append(half_bar_ms)
                    cell_heights.append(tick_size)
                    side.append("bid")

                if va > 0:
                    x_coords.append(np.datetime64(ts_ms + offset_ms, "ms"))
                    y_coords.append(price)
                    intensity_ask = min(1.0, va / poc_vol)
                    texts.append(_fmt_vol(va))
                    fill_colors.append(_fp_level_color(intensity_ask, ask_scale, bg_hex))
                    line_colors.append("white")
                    line_widths.append(0)
                    text_colors.append(text_normal)
                    cell_widths.append(half_bar_ms)
                    cell_heights.append(tick_size)
                    side.append("ask")

    if not x_coords:
        return None

    return ColumnDataSource(dict(
        x=np.array(x_coords, dtype="datetime64[ms]"),
        y=np.array(y_coords, dtype=np.float64),
        text=texts,
        fill=fill_colors,
        lw=np.array(line_widths, dtype=np.float64),
        lc=line_colors,
        tc=text_colors,
        cell_w=np.array(cell_widths, dtype=np.float64),
        cell_h=np.array(cell_heights, dtype=np.float64),
        side=side,
    ))


def build_volume_profile_source(vp_data: dict, theme: dict | None = None,
                                width_fraction: float = 0.32,
                                interval_ms: int | None = None):
    """
    Build the ColumnDataSource for the horizontal volume-profile histogram.

    Each price level produces two stacked horizontal segments — sell-aggressor
    (bid) volume and buy-aggressor (ask) volume — so the bar is split in two
    tones (blue/yellow). The combined bar length is proportional to that level's
    total volume relative to the largest level. Levels inside the Value Area
    (VAL..VAH) are drawn with high opacity; levels outside are dimmed.

    Two layouts are supported through ``vp_data["view"]``:

    - "session" (default): one profile per finished period, anchored at the
      period's left edge and growing right (TradingView VPSV).
    - "visible": every level across all periods is merged by price into a single
      profile anchored at the right edge and growing left (TradingView VPVR).
      This is a static approximation of the full data range; it does not
      recompute on zoom.

    Args:
        vp_data (dict): {"profiles": list[dict], "tick_size": float, "view": str,
            "buy_color": str, "sell_color": str, ...} as produced by the volume
            profile indicator (see period_profiles()). The color keys carry the
            indicator's object colors, used in preference to the theme.
        theme (dict | None): Accepted for call-site compatibility but unused for
            VP colors; VP colors are theme-independent (carried in vp_data).
        width_fraction (float): Max fraction of the period width used by a full
            (max-volume) bar. Kept well under 1 so bars do not cover the candles.

    Returns:
        ColumnDataSource | None: Source with columns y, height, left, right,
        color, alpha, side, volume, vol_buy, vol_sell; or None if empty.
    """
    from bokeh.models import ColumnDataSource

    profiles = vp_data.get("profiles", []) if vp_data else []
    if not profiles:
        return None

    tick_size = float(vp_data.get("tick_size", 0.0) or 0.0)
    view = vp_data.get("view", "session")

    # VP colors come from the indicator object (theme-independent). Fall back to
    # the fixed module defaults only; the plot theme never drives VP colors.
    buy_color = vp_data.get("buy_color") or DEFAULT_VP_BUY
    sell_color = vp_data.get("sell_color") or DEFAULT_VP_SELL

    # The histogram geometry is shared (Bokeh-free) with the indicator's draw()
    # so both render paths produce identical bars. Here we wrap the arrays in a
    # ColumnDataSource (left/right -> datetime64[ms]).
    arrays = volume_profile_bar_arrays(
        profiles, tick_size, view,
        buy_color=buy_color, sell_color=sell_color,
        interval_ms=interval_ms, width_fraction=width_fraction,
    )
    if arrays is None:
        return None
    return ColumnDataSource(dict(
        y=arrays["y"],
        height=arrays["height"],
        left=np.asarray(arrays["left"], dtype="datetime64[ms]"),
        right=np.asarray(arrays["right"], dtype="datetime64[ms]"),
        color=arrays["color"],
        alpha=arrays["alpha"],
        side=arrays["side"],
        volume=arrays["volume"],
        vol_buy=arrays["vol_buy"],
        vol_sell=arrays["vol_sell"],
    ))

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

from .config import PlotConfig, IndicatorPlotMeta
from tradetropy.exceptions import DataError
from tradetropy.core.data_types import infer_price_digits


def _resolve_price_digits(digits, tick_size, default: int = 2) -> int:
    """
    Resolve the price decimal places to use for plotting a symbol.

    Prefers an explicit ``digits`` but never shows fewer decimals than the
    tick size implies (guards against feeds that under-report ``digits`` for a
    fine-grained ``tick_size``, which is what truncates small prices like
    0.1844 to 0.18). Falls back to ``default`` when neither is informative.

    Args:
        digits (int | None): Explicit decimal places from the data / config.
        tick_size (float | None): Minimum price step.
        default (int): Floor used when nothing else is informative.

    Returns:
        int: Decimal places to render (>= 0).
    """
    tick_d = infer_price_digits(tick_size, default=0)
    d = int(digits) if digits is not None else 0
    resolved = max(d, tick_d)
    return resolved if resolved > 0 else default


def _price_number_format(digits: int) -> str:
    """
    Build a Bokeh ``NumeralTickFormatter`` pattern with fixed decimals.

    Args:
        digits (int): Decimal places (fixed, so trailing zeros are shown and a
            small price like 0.1844 is not truncated to 0.18).

    Returns:
        str: e.g. ``"0,0.0000"`` for digits=4, ``"0,0"`` for digits<=0.
    """
    d = max(0, int(digits))
    return "0,0." + ("0" * d) if d > 0 else "0,0"

# Themes now live in the tradetropy.plotting.theme package. Re-exported here under
# their historical private names so existing imports keep working unchanged.
from tradetropy.plotting.theme import (
    THEMES as _THEMES,
    INDICATOR_COLORS as _INDICATOR_COLORS,
)

_JS_DIR = pathlib.Path(__file__).parent / "js"
_AUTOSCALE_JS = (_JS_DIR / "autoscale_ohlc.js").read_text()

# -- Bokeh -- deferred imports in _ensure_bokeh() -----------------------------
_BOKEH_AVAILABLE: bool | None = None

def _ensure_bokeh() -> None:
    global _BOKEH_AVAILABLE
    if _BOKEH_AVAILABLE is True:
        return
    try:
        import bokeh  # noqa: F401
        _BOKEH_AVAILABLE = True
    except ImportError as exc:
        raise ImportError(
            "Bokeh is not installed. Install it with:\n"
            "    pip install bokeh\n"
            "or:\n"
            "    conda install bokeh"
        ) from exc


def _to_dt64(ts) -> np.ndarray:
    """Convert pd.Timestamp (with or without tz) to datetime64[ms] for Bokeh."""
    return np.array([pd.Timestamp(ts).tz_localize(None)], dtype="datetime64[ms]")


def _align_ts_to_candle(ts_ms, interval_ms: int, candle_origin_ms: int = 0):
    """
    Snap timestamps to the candle grid they belong to.

    Trades happen at arbitrary intra-candle times; aligning them to the candle
    they fall in places their markers/lines on top of the candle. Candles are
    NOT necessarily on the epoch grid (multiples of interval_ms from epoch 0):
    kline feeds carry their own origin, so the candle phase is
    candle_origin_ms % interval_ms. Flooring with a plain
    (ts // interval) * interval would snap to the wrong grid whenever the candle
    origin is phase-shifted from the epoch (the classic fractional-candle
    offset). This floors relative to the actual candle phase instead.

    Args:
        ts_ms: Timestamps in epoch milliseconds (scalar or array).
        interval_ms (int): Candle interval in milliseconds. Values <= 0 are a
            no-op (returns ts_ms unchanged).
        candle_origin_ms (int): Any real candle-open timestamp (its remainder
            mod interval_ms defines the grid phase). Defaults to 0 (epoch grid).

    Returns:
        The timestamps snapped down to the candle grid, same shape as ts_ms.

    Example:
        aligned = _align_ts_to_candle(entry_ms, 60_000, candle_origin_ms=ts0)
    """
    if interval_ms <= 0:
        return ts_ms
    phase = int(candle_origin_ms) % int(interval_ms)
    arr = np.asarray(ts_ms, dtype=np.int64)
    return ((arr - phase) // interval_ms) * interval_ms + phase


# Point-anchored (Points/Labels) draw primitives whose X is a single instant.
# Kept name-based (not isinstance) so it works regardless of which module
# re-exports the primitive classes, mirroring the duck-typed dispatch the rest
# of the plotting layer uses.
_PRIMITIVE_TIME_FIELDS = {
    "Points": ("x",),
    "Labels": ("x",),
}


def _snap_primitive_groups(groups, interval_ms, candle_origin_ms=0):
    """
    Snap point-anchored draw primitive's X (time) fields to the OHLC candle grid.

    Indicators emit ``draw()`` primitives at the exact event time (a tick's
    millisecond timestamp), which puts e.g. a LargeTrades bubble or a COT label
    at 12:10:17.4 on a chart whose candles are at 12:10:15. This floors the
    event-time field of point-anchored primitives to the candle it belongs
    to, so all indicator geometry shares the OHLC time unit. Span primitives
    (Rects/Segments/HBars/HLines) are left untouched - per-bar order-flow
    boxes are already centered on their candle, and multi-candle spans carry
    real start/end candle timestamps. Tools are intentionally not
    routed through here (they anchor at explicit user ranges), only indicators.

    The primitives are plain (non-frozen) dataclasses, so their arrays are
    rewritten in place. The per-bar value series of indicators is already on the
    OHLC grid (resampled to the candle index), so only ``draw()`` geometry needs
    this. A no-op when ``interval_ms <= 0`` (e.g. tick/range charts).

    Args:
        groups (dict | None): {legend_name: [primitive, ...]} as produced by
            ``_indicator_primitive_groups`` / ``collect_draw_primitives``.
        interval_ms (int): OHLC candle interval in milliseconds.
        candle_origin_ms (int): Any real candle-open timestamp; its remainder mod
            interval_ms sets the grid phase (handles phase-shifted candles).

    Returns:
        dict | None: the same ``groups`` (mutated in place), for convenience.
    """
    if not groups or not interval_ms or interval_ms <= 0:
        return groups
    for prims in groups.values():
        for p in prims:
            fields = _PRIMITIVE_TIME_FIELDS.get(type(p).__name__)
            if not fields:
                continue
            for fname in fields:
                arr = getattr(p, fname, None)
                if arr is None:
                    continue
                vals = np.asarray(list(arr), dtype=np.int64)
                if vals.size == 0:
                    continue
                setattr(p, fname,
                        _align_ts_to_candle(vals, interval_ms, candle_origin_ms))
    return groups


def _ohlc_tick_bounds(ts_ms, bar_width_ms) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute left/right x-bounds for OHLC bar-style open/close ticks.

    A bar-style OHLC glyph draws a vertical High-Low segment with a short
    horizontal tick to the left (open) and to the right (close). This returns
    the timestamps of those tick endpoints, offset by half the bar width.

    Args:
        ts_ms: Candle start timestamps in epoch milliseconds (scalar or array).
        bar_width_ms: Bar width in milliseconds (scalar or array). The tick
            half-length is bar_width_ms / 2.

    Returns:
        tuple[np.ndarray, np.ndarray]: (ts_left, ts_right) as datetime64[ms],
        where ts_left = ts - width/2 and ts_right = ts + width/2.

    Example:
        ts_left, ts_right = _ohlc_tick_bounds(ts_ms_array, 54_000)
    """
    ts_arr = np.asarray(ts_ms, dtype=np.int64)
    half   = (np.asarray(bar_width_ms, dtype=np.float64) / 2.0).astype(np.int64)
    ts_left  = (ts_arr - half).astype("datetime64[ms]")
    ts_right = (ts_arr + half).astype("datetime64[ms]")
    return ts_left, ts_right


def _resolve_dim(value, idx: int):
    """
    Resolves the value of a style field for dimension `idx`.

    If `value` is a list, returns value[idx] (clamped to last if idx exceeds).
    If `value` is scalar, returns it directly.
    """
    if isinstance(value, list):
        return value[idx] if idx < len(value) else value[-1]
    return value



# INTERNAL HELPERS

def _theme(config: PlotConfig) -> dict:
    return _THEMES[config.theme]


def _fp_level_color(intensity: float, scale, fallback: str = "#888888") -> str:
    """
    Resolve a footprint level color from a discrete color scale.

    The scale assigns an explicit color to each volume band, replacing the old
    automatic background-blend ramp. Each level's volume is expressed as a
    fraction of the candle POC volume (``intensity`` in 0..1); the color of the
    first band whose upper bound is greater than or equal to that fraction is
    used.

    Args:
        intensity (float): Level volume as a fraction of POC volume, 0..1.
        scale (list): Ordered list of ``(upper_bound, color)`` tuples, with
            upper_bound ascending in 0..1. Example: ``[(0.20, '#..'), ...]``.
        fallback (str): Color returned when the scale is empty or invalid.

    Returns:
        str: Hex color string for the matching band.

    Example:
        color = _fp_level_color(0.55, theme['fp_bid_scale'])
    """
    if not scale:
        return fallback
    t = min(1.0, max(0.0, float(intensity)))
    for upper, color in scale:
        if t <= float(upper):
            return color
    return scale[-1][1]


def _color_cycle(idx: int) -> str:
    return _INDICATOR_COLORS[idx % len(_INDICATOR_COLORS)]


def _cfg(defn: dict):
    """
    Returns the IndicatorPlotConfig associated with an indicator_def.

    strategy.add_indicator() always stores "plot_config" in the dict.
    If for some reason it does not exist (legacy code), returns an empty config.
    """
    from tradetropy.ta.base import IndicatorPlotConfig
    pc = defn.get("plot_config")
    if pc is None:
        pc = IndicatorPlotConfig()
    return pc


def _infer_overlay(values: np.ndarray, close_prices: np.ndarray) -> bool:
    """
    Automatically detects if an indicator should be overlaid on the OHLC panel.
    Fallback when overlay=None and no known category.
    """
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return False
    close_valid = close_prices[np.isfinite(close_prices)]
    if close_valid.size == 0:
        return False
    with np.errstate(invalid="ignore", divide="ignore"):
        x = valid / np.nanmean(close_valid)
        return bool(((x < 1.4) & (x > 0.6)).mean() > 0.6)


def _ms_to_bokeh_datetime(ts_ms: np.ndarray) -> np.ndarray:
    return ts_ms.astype("datetime64[ms]")


def _resample_indicator_to_index(
    values: np.ndarray,
    src_index: pd.DatetimeIndex,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    s = pd.Series(values, index=src_index)
    # Tick data keeps same-millisecond trades (the streaming path never dedups
    # timestamps), so the source index can carry duplicate labels; pandas
    # refuses to reindex on a duplicated axis. Keep the last observation per
    # timestamp - causally correct since the reindex forward-fills, so the
    # value in effect at a given ms is the latest tick of that ms - and sort so
    # the ffill runs in time order.
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s.reindex(target_index).ffill().to_numpy(dtype=np.float64)


def _extract_volume_profile_data(indicator) -> "dict | None":
    """
    Collect the finalized per-period histograms from a volume profile indicator.

    VolumeProfile / TickVolumeProfile populate ``profiles_`` during
    calculate() and expose a render-ready view via ``period_profiles()``.
    Returns None for any indicator that is not a volume profile.

    Args:
        indicator: The indicator instance from an indicator_def.

    Returns:
        dict | None: {"profiles": list[dict], "tick_size": float} or None.
    """
    if not (hasattr(indicator, "period_profiles") and hasattr(indicator, "profiles_")):
        return None
    profiles = indicator.period_profiles()
    if not profiles:
        return None
    return {
        "profiles": profiles,
        "tick_size": float(getattr(indicator, "tick_size_used_", 0.0) or 0.0),
        "view": getattr(indicator, "view", "session"),
        # Semantic colors carried from the indicator object so the histogram is
        # theme-independent (the plot theme no longer drives VP colors).
        "buy_color": getattr(indicator, "buy_color", None),
        "sell_color": getattr(indicator, "sell_color", None),
        "poc_color": getattr(indicator, "poc_color", None),
        "hvn_color": getattr(indicator, "hvn_color", None),
        "lvn_color": getattr(indicator, "lvn_color", None),
    }


def _indicator_primitive_groups(defn: dict, interval_ms: "int | None") -> "dict | None":
    """
    Collect an indicator's declarative draw primitives grouped by legend name.

    Any indicator may implement ``draw(cfg, *, interval_ms) -> list[Primitive]``
    to render geometry that is not a per-bar series (e.g. the Volume Profile
    histogram). This mirrors ``collect_draw_primitives`` for tools so both feed
    the same generic renderer (``render_tool_groups``). Returns None when the
    indicator emits no primitives (the common case: ordinary series indicators).

    Args:
        defn (dict): An indicator_def from the strategy.
        interval_ms (int | None): OHLC interval, passed to draw() for geometry.

    Returns:
        dict | None: {legend_name: [primitives]} or None.
    """
    indicator = defn["indicator"]
    draw = getattr(indicator, "draw", None)
    if draw is None:
        return None
    pc = _cfg(defn)
    if not pc.plot:
        return None
    try:
        result = draw(pc, interval_ms=interval_ms)
    except TypeError:
        result = draw(pc)
    result = result or []
    # draw() may return either a flat list of primitives (single legend group)
    # or a dict {legend_name: [primitives]} for an indicator that wants several
    # independently toggleable legend entries (e.g. DeltaVolumeInfo: Buy/Sell/
    # Delta/Total). Normalize both to the {legend: [primitives]} shape.
    if isinstance(result, dict):
        groups = {k: list(v) for k, v in result.items() if v}
        return groups or None
    prims = result
    if not prims:
        return None
    name = pc.name
    legend = name if isinstance(name, str) else indicator.display_name()
    return {legend: list(prims)}


def _meta_from_defn(
    defn: dict,
    values: np.ndarray,
    ts_aligned: np.ndarray,
    close_prices: np.ndarray,
    color_idx: int,
) -> "IndicatorPlotMeta | None":
    """
    Builds an IndicatorPlotMeta from a post-run indicator_def.

    Uses the IndicatorPlotConfig from the def to populate all style fields.
    """
    pc = _cfg(defn)
    indicator = defn["indicator"]

    if not pc.plot:
        return None

    # -- Resolve overlay -------------------------------------------------------
    overlay = pc.overlay
    if overlay is None:
        # Try resolution by category
        category = getattr(indicator, "category", None)
        if category:
            overlay = pc.resolve_overlay(category)
        else:
            # Statistical fallback
            vals_1d = values[0] if (isinstance(values, np.ndarray) and values.ndim == 2) else values
            overlay = _infer_overlay(vals_1d, close_prices)

    # -- Name ------------------------------------------------------------------
    name = pc.name
    if name is None:
        if hasattr(indicator, "display_name"):
            name = indicator.display_name()
        else:
            name = indicator.col_name(
                defn["source"].symbol, defn["source"]._col_name
            )

    # -- Color -----------------------------------------------------------------
    color = pc.color
    if color is None:
        color = _color_cycle(color_idx)

    # -- Show legend -----------------------------------------------------------
    # Explicit: always respect.
    # overlay=False (own panel): False -- panel_title identifies it.
    # overlay=True: True -- needs a togglable legend.
    #
    if pc.show_legend is not None:
        show_legend = pc.show_legend
    else:
        show_legend = False if overlay is False else True

    return IndicatorPlotMeta(
        name=name,
        values=values,
        timestamps=ts_aligned,
        overlay=overlay,
        scatter=pc.scatter,
        renderer=pc.renderer,
        plot=True,
        color=color,
        panel_height=pc.panel_height,
        panel_title=pc.panel_title,
        show_legend=show_legend,
        line_width=pc.line_width,
        line_dash=pc.line_dash,
        line_alpha=pc.line_alpha,
        marker=pc.marker,
        marker_size=pc.marker_size,
        marker_alpha=pc.marker_alpha,
        marker_fill=pc.marker_fill,
        marker_line_width=pc.marker_line_width,
        reference_lines=list(pc.reference_lines),
        bar_width_factor=pc.bar_width_factor,
        bar_align=pc.bar_align,
        bar_alpha=pc.bar_alpha,
        bar_color_positive=pc.bar_color_positive,
        bar_color_negative=pc.bar_color_negative,
        _ts_band_indices=list(getattr(indicator, "ts_band_indices", [])),
        autoscale=pc.autoscale,
        exclude_from_autoscale=getattr(pc, "exclude_from_autoscale", False),
        rect_fill_alpha=getattr(pc, "rect_fill_alpha", 0.15),
        rect_line_alpha=getattr(pc, "rect_line_alpha", 0.6),
        rect_line_width=getattr(pc, "rect_line_width", 1.0),
        label_font_size=getattr(pc, "label_font_size", "9pt"),
        label_x_offset=getattr(pc, "label_x_offset", 5),
        label_y_offset=getattr(pc, "label_y_offset", 5),
        label_text_align=getattr(pc, "label_text_align", "left"),
        label_text_baseline=getattr(pc, "label_text_baseline", "bottom"),
        arrow_size=getattr(pc, "arrow_size", 10),
        arrow_length=getattr(pc, "arrow_length", 0),
        _vp_data=_extract_volume_profile_data(indicator),
        output_names=list(getattr(indicator, "output_names", [])),
        front_dim=getattr(pc, "front_dim", None),
    )


# -- Helpers for partial candle in indicators -----------------------------------

def _build_partial_source(store, defn: dict, n_ticks: int, N_OHLC_COLS: int) -> np.ndarray:
    """
    Builds the source array to compute the indicator value on the active
    partial candle: (L-1) closes from closed candles + partial close.

    Returns a 1D array [L] for single-source or 2D [L x K] for multi-source.
    """
    from tradetropy.core.constants import _OHLC_COL

    indicator = defn["indicator"]
    sources   = defn.get("sources", [defn["source"]])
    multi_source = defn.get("multi_source", False)
    L = getattr(indicator, "length", 1)

    # Number of closed candle bars needed (excluding the partial)
    n_closed = store.matrix.shape[0]
    n_prev  = min(L - 1, n_closed)

    # Partial candle -- obtained from tick-to-candle mapping using the last tick
    partial_ohlc = store.partial_tick_candle(n_ticks - 1)   # shape [N_OHLC_COLS]

    if multi_source:
        col_srcs = [_OHLC_COL[f._col_name] for f in sources]
        prev = store.matrix[-n_prev:, col_srcs] if n_prev > 0 else np.empty((0, len(col_srcs)))
        partial_row = partial_ohlc[col_srcs]
        if len(prev) > 0:
            return np.vstack([prev, partial_row])
        else:
            return partial_row[np.newaxis, :]
    else:
        col_src = _OHLC_COL[sources[0]._col_name]
        prev = store.matrix[-n_prev:, col_src] if n_prev > 0 else np.array([])
        close_p = float(partial_ohlc[_OHLC_COL["close"]])
        return np.append(prev, close_p) if len(prev) > 0 else np.array([close_p])


def _append_partial_monoband(
    raw: np.ndarray,
    store,
    defn: dict,
    n_ticks: int,
    N_OHLC_COLS: int,
) -> np.ndarray:
    """
    Appends the indicator value computed on the partial candle to the end
    of the mono-band array. Returns raw with one extra element.
    """
    try:
        indicator = defn["indicator"]
        source_arr = _build_partial_source(store, defn, n_ticks, N_OHLC_COLS)
        result  = indicator.calculate(source_arr)
        val = float(result[-1]) if result.ndim == 1 else float(result[0, -1])
        return np.append(raw, val)
    except Exception:
        return np.append(raw, np.nan)


def _append_partial_multiband(
    raw_multi: np.ndarray,
    store,
    defn: dict,
    n_ticks: int,
    N_OHLC_COLS: int,
) -> np.ndarray:
    """
    Appends a column (partial candle) to the multi-band array [K x N].
    Returns raw_multi with one extra column -> [K x N+1].
    """
    try:
        indicator = defn["indicator"]
        source_arr = _build_partial_source(store, defn, n_ticks, N_OHLC_COLS)
        result  = indicator.calculate(source_arr)   # [K × M]
        K = raw_multi.shape[0]
        if result.ndim == 2:
            col_p = result[:, -1].reshape(K, 1)
        else:
            col_p = np.full((K, 1), float(result[-1]))
        return np.hstack([raw_multi, col_p])
    except Exception:
        K = raw_multi.shape[0]
        return np.hstack([raw_multi, np.full((K, 1), np.nan)])


def _gather_plot_data(bt) -> dict:
    """
    Collects all necessary data from the BacktestEngine post-run.
    """
    strategy = bt.strategy

    # -- Get OHLC data ---------------------------------------------------------
    ohlc_array = None
    interval_ms = None
    symbol = None
    ohlc_timestamps_ms = None

    if strategy._ohlc_proxies:
        op = strategy._ohlc_proxies[0]
        symbol = op.symbol
        interval_ms = op.interval_ms

        if op._ohlc_store is not None:
            store = op._ohlc_store
            from tradetropy.core.constants import N_OHLC_COLS
            ohlc_array = store.matrix[:, :N_OHLC_COLS].copy()
            ohlc_timestamps_ms = ohlc_array[:, 0]

            n_ticks = len(store.tick_to_candle_mapping)
            if n_ticks > 0:
                partial_candle = store.partial_tick_candle(n_ticks - 1)
                ohlc_array = np.vstack([ohlc_array, partial_candle])
                ohlc_timestamps_ms = ohlc_array[:, 0]

    if ohlc_array is None and bt._tick_inputs:
        tick_input = bt._tick_inputs[0]
        symbol = tick_input.symbol
        interval_ms = 60_000
        from tradetropy.core.constants import _TICK_COL
        from tradetropy.data.data import build_candles_from_ticks
        arr = tick_input.data
        result = build_candles_from_ticks(
            arr[:, _TICK_COL["ts"]],
            arr[:, _TICK_COL["price"]],
            arr[:, _TICK_COL["volume"]],
            interval_ms,
        )
        from tradetropy.core.constants import N_OHLC_COLS
        ohlc_array = result["closed_candles"][:, :N_OHLC_COLS]

        # build_candles_from_ticks() returns only CLOSED candles (the last,
        # in-progress candle is excluded). Re-append that developing candle so
        # the static chart shows the same final bar as replay/live (whose OHLC
        # ring keeps a partial candle). Reconstructed from the accumulators the
        # helper already returns, mirroring OhlcDataStore.partial_tick_candle.
        _acc_per_tick = result["accumulated_per_tick"]
        if len(_acc_per_tick) > 0:
            _acc = _acc_per_tick[-1]                       # [open, high, low, vol]
            partial_candle = np.array(
                [
                    result["candle_ts_per_tick"][-1],  # candle open ts
                    _acc[0],                              # open
                    _acc[1],                              # high
                    _acc[2],                              # low
                    result["prices"][-1],              # close (last price)
                    _acc[3],                              # volume
                ],
                dtype=np.float64,
            )
            ohlc_array = np.vstack([ohlc_array, partial_candle])

        ohlc_timestamps_ms = ohlc_array[:, 0]

    if ohlc_array is None or len(ohlc_array) == 0:
        raise DataError(
            "plot() found no OHLC data. "
            "Make sure the strategy uses subscribe_ohlc() "
            "or the engine has tick data."
        )

    close_prices = ohlc_array[:, 4]

    # -- Does OHLC include a partial candle? ------------------------------------
    # ohlc_array has N+1 rows when n_ticks > 0 (the last is the partial
    # candle). For indicators to stay aligned we need to compute their value
    # on that same partial candle and append it to the array.
    # We detect this by inspecting the same store used above.
    _primary_store = None
    _primary_n_ticks = 0
    if strategy._ohlc_proxies:
        _op0 = strategy._ohlc_proxies[0]
        if _op0._ohlc_store is not None:
            _primary_store = _op0._ohlc_store
            _primary_n_ticks = len(_primary_store.tick_to_candle_mapping)

    # -- Collect indicators -----------------------------------------------------
    indicators: list[IndicatorPlotMeta] = []
    color_idx = 0

    for defn in strategy._indicator_defs:
        pc = _cfg(defn)
        if not pc.plot:
            continue

        from tradetropy.data.data import OhlcProxy as _OhlcProxy, TickProxy as _TickProxy
        source = defn["source"]
        col_name = defn.get("col_name")
        if col_name is None:
            continue

        values = None

        if isinstance(source.proxy, _OhlcProxy):
            ohlc_proxy = defn.get("ohlc_proxy")
            if ohlc_proxy is not None and ohlc_proxy._ohlc_store is not None:
                store = ohlc_proxy._ohlc_store
                from tradetropy.core.constants import N_OHLC_COLS, _OHLC_COL
                is_primary = ohlc_proxy is strategy._ohlc_proxies[0]

                # Do we need to add the partial candle value to this indicator?
                # Only if the primary OHLC included a partial candle AND this
                # indicator operates on the same store.
                _need_partial = (
                    is_primary
                    and _primary_n_ticks > 0
                    and store is _primary_store
                    and getattr(defn["indicator"], "use_partial", True)
                )

                if defn.get("multi_band") and defn.get("col_names"):
                    band_arrays = []
                    for cname in defn["col_names"]:
                        cidx = store.col_index.get(cname)
                        if cidx is not None and cidx >= N_OHLC_COLS:
                            band_arrays.append(store.matrix[:, cidx].copy())
                    if band_arrays:
                        raw_multi = np.vstack(band_arrays)

                        # Add partial candle column
                        if _need_partial:
                            raw_multi = _append_partial_multiband(
                                raw_multi, store, defn, _primary_n_ticks, N_OHLC_COLS
                            )

                        if is_primary:
                            values = raw_multi
                        else:
                            src_ts = store.matrix[:, 0]
                            src_idx = pd.to_datetime(src_ts, unit="ms", utc=True)
                            tgt_idx = pd.to_datetime(ohlc_timestamps_ms, unit="ms", utc=True)
                            values = np.vstack([
                                _resample_indicator_to_index(row, src_idx, tgt_idx)
                                for row in raw_multi
                            ])
                else:
                    col_idx = store.col_index.get(col_name)
                    if col_idx is not None and col_idx >= N_OHLC_COLS:
                        raw = store.matrix[:, col_idx].copy()

                        # Add the value computed on the partial candle
                        if _need_partial:
                            raw = _append_partial_monoband(
                                raw, store, defn, _primary_n_ticks, N_OHLC_COLS
                            )

                        if is_primary:
                            values = raw
                        else:
                            src_ts = store.matrix[:, 0]
                            src_idx = pd.to_datetime(src_ts, unit="ms", utc=True)
                            tgt_idx = pd.to_datetime(ohlc_timestamps_ms, unit="ms", utc=True)
                            values = _resample_indicator_to_index(raw, src_idx, tgt_idx)

        elif isinstance(source.proxy, _TickProxy):
            tick_symbol = defn.get("tick_symbol")
            if tick_symbol and tick_symbol in bt._tick_stores:
                tick_store = bt._tick_stores[tick_symbol]
                col_idx = tick_store.col_index.get(col_name)
                if col_idx is not None:
                    raw = tick_store.matrix[:, col_idx].copy()
                    tick_ts = tick_store.matrix[:, 0]
                    src_idx = pd.to_datetime(tick_ts, unit="ms", utc=True)
                    tgt_idx = pd.to_datetime(ohlc_timestamps_ms, unit="ms", utc=True)
                    values = _resample_indicator_to_index(raw, src_idx, tgt_idx)

        if values is None or (hasattr(values, "__len__") and len(values) == 0):
            continue

        is_2d = isinstance(values, np.ndarray) and values.ndim == 2
        n_vals = values.shape[-1] if is_2d else len(values)
        n_ohlc = len(ohlc_array)
        if n_vals != n_ohlc:
            min_len = min(n_vals, n_ohlc)
            values = values[..., -min_len:] if is_2d else values[-min_len:]
            ts_aligned = ohlc_timestamps_ms[-min_len:]
        else:
            ts_aligned = ohlc_timestamps_ms

        meta = _meta_from_defn(defn, values, ts_aligned, close_prices, color_idx)
        if meta is not None:
            meta._draw_primitives = _indicator_primitive_groups(defn, interval_ms)
            indicators.append(meta)
            # Only increment color_idx if it did not have its own color
            if pc.color is None:
                color_idx += 1

    # -- Symbol from trades ----------------------------------------------------
    # `symbol` here is still the one from the primary data (OHLC proxy or tick
    # input); we use it to look up its tick_size before overwriting with the
    # symbols derived from trades.
    primary_symbol = symbol

    price_tick = 0.0
    price_digits = None
    _all_inputs = list(getattr(bt, "_kline_inputs", ()) or ())
    _all_inputs += list(getattr(bt, "_tick_inputs", ()) or ())
    if _all_inputs:
        _match = next(
            (inp for inp in _all_inputs if inp.symbol == primary_symbol),
            _all_inputs[0],
        )
        price_tick = float(getattr(_match, "tick_size", 0.0) or 0.0)
        price_digits = getattr(_match, "digits", None)

    if bt.stats is not None:
        trades_df = bt.stats.trades
        if trades_df is not None and not trades_df.empty:
            if "symbol" in trades_df.columns:
                syms = trades_df["symbol"].dropna().unique()
                syms = [s for s in syms if s and str(s).strip()]
                if syms:
                    symbol = ", ".join(sorted(syms))

    return {
        "ohlc_array": ohlc_array,
        "interval_ms": interval_ms,
        "stats": bt.stats,
        "indicators": indicators,
        "fp_proxies": strategy._fp_proxies,
        "symbol": symbol or "SYMBOL",
        "price_tick": price_tick,
        "price_digits": _resolve_price_digits(price_digits, price_tick),
    }


def _resample_ohlc(
    ohlc_array: np.ndarray,
    source_interval_ms: int,
    target_interval_ms: int,
) -> tuple[np.ndarray, int]:
    # Delegates to shared vectorized implementation. First 6 columns (ts,o,h,l,c,v) are kept
    # to maintain historical signature.
    from tradetropy.data._klines import resample_klines

    result, interval = resample_klines(
        ohlc_array[:, :6], source_interval_ms, target_interval_ms
    )
    return result[:, :6], interval



def _fmt_vol(v: float) -> str:
    a = abs(v)
    if a >= 1_000_000_000: return f"{v/1e9:.2f}B"
    if a >= 1_000_000:     return f"{v/1e6:.2f}M"
    if a >= 10_000:        return f"{v/1e3:.1f}k"
    if a >= 1_000:         return f"{v/1e3:.2f}k"
    return f"{v:.0f}"


def _datetime_format(interval_ms: int) -> str:
    if interval_ms < 60_000:
        return "%Y-%m-%d %H:%M:%S.%f"
    return "%Y-%m-%d %H:%M"


def _interval_label(interval_ms: int) -> str:
    seconds = interval_ms // 1000
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"
    
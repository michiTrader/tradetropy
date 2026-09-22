"""
document.py
===========
build_live_document() -- builds the Bokeh Document for the live chart.

Reuses exactly the same renderers from the backtest system:
    render_ohlc, render_volume, render_indicator, render_reference_lines,
    render_equity, _configure_legends, _configure_margins, _configure_crosshair,
    _configure_autoscale_ohlc, _build_layout, _new_fig, _new_ohlc_fig

The difference from the static pipeline is that the ColumnDataSources are
created empty. Bokeh does not know or care -- it simply renders whatever is
there. When the updater does .stream() or .patch(), Bokeh propagates the
change to the browser via WebSocket automatically.

Supported indicators
--------------------
- Single-band normal   (SMA, EMA, RSI, MACD histogram, ...): is_stateful=False
- Multi-band normal    (MACD line+signal, BB, ...):           is_stateful=False
- Single-band stateful (pivots with use_partial=False):       is_stateful=True
- Multi-band stateful  (ConfirmedPivot with ts_band_indices): is_stateful=True
- Pure annotations     (FVG, OB, Sessions, ...):              skipped in live
  (depend on fully recalculated arrays -- no support in v1)
"""

from __future__ import annotations
import pathlib
from typing import TYPE_CHECKING

import numpy as np

from tradetropy.plotting._fig_factory import _new_fig, _new_ohlc_fig
from tradetropy.plotting._util import _theme, _color_cycle, _cfg, _datetime_format, _interval_label, _ohlc_tick_bounds, _resolve_dim
from tradetropy.plotting._layout import (
    _configure_legends,
    _configure_margins,
    _configure_crosshair,
    _configure_autoscale_ohlc,
    _configure_fp_yaxis,
    _build_layout,
)
from tradetropy.plotting.render import (
    render_indicator,
    render_reference_lines,
    render_equity,
)
from tradetropy.plotting.config import PlotConfig, IndicatorPlotMeta
from tradetropy.plotting.live.updater import (
    LiveSourceUpdater,
    _OhlcSourceRef,
    _BandRef,
    _IndicatorSourceRef,
    _EquitySourceRef,
    _FootprintSourceRef,
    _TradesSourceRef,
    _build_trades_data_live,
)
from tradetropy.plotting.live.navigation import LiveNavigationController

_JS_DIR = pathlib.Path(__file__).resolve().parent.parent / "js"

if TYPE_CHECKING:
    from bokeh.document import Document
    from tradetropy.models.strategy import Strategy


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_initial_ohlc_source(primary_proxy, interval_ms: int, theme: dict, fp_proxy=None):
    """
    Builds the OHLC ColumnDataSource pre-populated with the ring's history.

    In Bokeh 3.x, .data= on a source already added to the Document does not
    propagate to the initial serialized state. The source MUST have the data
    before renderers are created and it is added to the Document.

    If fp_proxy is not None, adds fp_bid, fp_ask, ... columns so the
    HoverTool and updater can read/write them.
    """
    from bokeh.models import ColumnDataSource
    from tradetropy.core.constants import _OHLC_COL

    def _fp_nan_cols(n):
        return dict(
            fp_bid       = np.full(n, np.nan, dtype=np.float64),
            fp_ask       = np.full(n, np.nan, dtype=np.float64),
            fp_delta     = np.full(n, np.nan, dtype=np.float64),
            fp_poc_price = np.full(n, np.nan, dtype=np.float64),
            fp_poc_vol   = np.full(n, np.nan, dtype=np.float64),
            fp_vah       = np.full(n, np.nan, dtype=np.float64),
            fp_val       = np.full(n, np.nan, dtype=np.float64),
        )

    ring = primary_proxy._ohlc_ring

    # If no history, return empty source with typed columns
    if ring is None or ring._n_closed == 0:
        base = dict(
            ts                = np.array([], dtype="datetime64[ms]"),
            Open              = np.array([], dtype=np.float64),
            High              = np.array([], dtype=np.float64),
            Low               = np.array([], dtype=np.float64),
            Close             = np.array([], dtype=np.float64),
            Volume            = np.array([], dtype=np.float64),
            inc               = np.array([], dtype=object),
            bar_width         = np.array([], dtype=np.float64),
            candle_color      = np.array([], dtype=object),
            wick_color        = np.array([], dtype=object),
            body_border_color = np.array([], dtype=object),
            body_border_width = np.array([], dtype=np.float64),
            top_body          = np.array([], dtype=np.float64),
            bottom_body       = np.array([], dtype=np.float64),
            ts_left           = np.array([], dtype="datetime64[ms]"),
            ts_right          = np.array([], dtype="datetime64[ms]"),
        )
        if fp_proxy is not None:
            base.update(_fp_nan_cols(0))
        return ColumnDataSource(base)

    n        = min(ring._n_closed, ring._W)
    ts_col   = ring.closed_window(_OHLC_COL["ts"],     n).copy()
    open_col = ring.closed_window(_OHLC_COL["open"],   n).copy()
    high_col = ring.closed_window(_OHLC_COL["high"],   n).copy()
    low_col  = ring.closed_window(_OHLC_COL["low"],    n).copy()
    close_col= ring.closed_window(_OHLC_COL["close"],  n).copy()
    vol_col  = ring.closed_window(_OHLC_COL["volume"], n).copy()

    actual_n  = len(ts_col)
    color_up  = theme.get("candle_up",   "#2ECC71")
    color_dn  = theme.get("candle_down", "#E74C3C")
    is_up     = close_col >= open_col
    colors    = np.array([color_up if u else color_dn for u in is_up], dtype=object)
    bar_width = int(interval_ms * 0.9)

    is_doji = close_col == open_col
    body_border_color = np.array([
        c if d else None for c, d in zip(colors, is_doji)
    ], dtype=object)
    body_border_width = np.where(is_doji, 2.0, 1.0).astype(np.float64)

    # print(f"[BUILD_SRC] pre-populating source with {actual_n} historical candles")

    ts_left, ts_right = _ohlc_tick_bounds(ts_col, bar_width)

    data = dict(
        ts                = ts_col.astype("datetime64[ms]"),
        Open              = open_col,
        High              = high_col,
        Low               = low_col,
        Close             = close_col,
        Volume            = vol_col,
        inc               = np.where(is_up, "1", "0").astype(object),
        bar_width         = np.full(actual_n, bar_width, dtype=np.float64),
        candle_color      = colors,
        wick_color        = colors,
        body_border_color = body_border_color,
        body_border_width = body_border_width,
        top_body          = np.maximum(open_col, close_col),
        bottom_body       = np.minimum(open_col, close_col),
        ts_left           = ts_left,
        ts_right          = ts_right,
    )

    if fp_proxy is not None:
        data.update(_fp_nan_cols(actual_n))
        # _populate_ohlc_history() (via _populate_ohlc_fp_cols) in the updater
        # fills in the real historical values when binding ref.fp_proxy.

    return ColumnDataSource(data)


def _empty_indicator_source():
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(dict(
        ts    = np.array([], dtype="datetime64[ms]"),
        value = np.array([], dtype=np.float64),
        tag   = np.array([], dtype=object),
    ))


def _empty_equity_source():
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(dict(
        ts     = np.array([], dtype="datetime64[ms]"),
        equity = np.array([], dtype=np.float64),
    ))


def _empty_trailing_dd_source():
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(dict(
        ts          = np.array([], dtype="datetime64[ms]"),
        trailing_dd = np.array([], dtype=np.float64),
    ))


def _empty_trades_source():
    """
    Empty source compatible with render_trades_live().
    Explicitly typed columns so Bokeh does not infer incorrect types
    when doing .data= with the first batch of real trades.
    """
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(dict(
        lines_xs    = [],
        lines_ys    = [],
        trade_color = np.array([], dtype=object),
        entry_ts    = np.array([], dtype="datetime64[ms]"),
        exit_ts     = np.array([], dtype="datetime64[ms]"),
        entry_price = np.array([], dtype=np.float64),
        exit_price  = np.array([], dtype=np.float64),
        pnl         = np.array([], dtype=np.float64),
        direction   = np.array([], dtype=object),
        size        = np.array([], dtype=np.float64),
    ))


def _empty_fp_source():
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(dict(
        x      = np.array([], dtype="datetime64[ms]"),
        y      = np.array([], dtype=np.float64),
        text   = [],
        fill   = [],
        lc     = [],
        lw     = np.array([], dtype=np.float64),
        tc     = [],
        cell_w = np.array([], dtype=np.float64),
        cell_h = np.array([], dtype=np.float64),
    ))


def _meta_from_defn_live(
    defn: dict,
    color_idx: int,
    interval_ms: int,
) -> "IndicatorPlotMeta | None":
    """
    Builds IndicatorPlotMeta from an indicator_def without real data.
    Same logic as _meta_from_defn in _util.py but with empty arrays.
    Pure annotations -> None (not supported in live v1).
    """
    from tradetropy.ta.base import _CATEGORY_OVERLAY_DEFAULTS

    pc        = _cfg(defn)
    indicator = defn["indicator"]

    if not pc.plot:
        return None

    # Geometric / draw-only indicators (rect/span/segment/label/arrow/none) render
    # via Indicator.draw() through the generic primitive path, not as per-bar band
    # series - skip them here so the standard live indicator path does not draw
    # spurious lines. Series renderers (line/scatter/step/bar) proceed normally
    # (e.g. SwingHL keeps its scatter markers).
    _GEOM = {"rect", "span", "label", "arrow", "segment", "none"}
    _r = pc.renderer
    _is_geom = (all(x in _GEOM for x in _r) if isinstance(_r, (list, tuple))
                else _r in _GEOM)
    if _is_geom:
        return None

    # Resolve overlay
    overlay = pc.overlay
    if overlay is None:
        category = getattr(indicator, "category", None)
        if category:
            overlay = pc.resolve_overlay(category)
        else:
            # Without data we cannot do statistical inference --
            # use False (own panel) as conservative default
            overlay = _CATEGORY_OVERLAY_DEFAULTS.get("other", False)

    # Name
    name = pc.name
    if name is None:
        if hasattr(indicator, "display_name"):
            name = indicator.display_name()
        else:
            name = indicator.col_name(
                defn["source"].symbol, defn["source"]._col_name
            )

    # Color
    color = pc.color
    if color is None:
        color = _color_cycle(color_idx)

    # Show legend
    if pc.show_legend is not None:
        show_legend = pc.show_legend
    else:
        show_legend = False if overlay is False else True

    n_outputs = indicator.n_outputs
    # empty values with correct shape so render_indicator does not crash
    if n_outputs > 1:
        empty_values = np.empty((n_outputs, 0), dtype=np.float64)
    else:
        empty_values = np.array([], dtype=np.float64)

    return IndicatorPlotMeta(
        name                 = name,
        values               = empty_values,
        timestamps           = np.array([], dtype=np.float64),
        overlay              = overlay,
        scatter              = pc.scatter,
        renderer             = pc.renderer,
        plot                 = True,
        color                = color,
        panel_height         = pc.panel_height,
        panel_title          = pc.panel_title,
        show_legend          = show_legend,
        line_width           = pc.line_width,
        line_dash            = pc.line_dash,
        line_alpha           = pc.line_alpha,
        marker               = pc.marker,
        marker_size          = pc.marker_size,
        marker_alpha         = pc.marker_alpha,
        marker_fill          = pc.marker_fill,
        marker_line_width    = pc.marker_line_width,
        reference_lines      = list(pc.reference_lines),
        bar_width_factor     = pc.bar_width_factor,
        bar_align            = pc.bar_align,
        bar_alpha            = pc.bar_alpha,
        bar_color_positive   = pc.bar_color_positive,
        bar_color_negative   = pc.bar_color_negative,
        _ts_band_indices     = list(getattr(indicator, "ts_band_indices", [])),
        autoscale            = pc.autoscale,
        exclude_from_autoscale = getattr(pc, "exclude_from_autoscale", False),
        hide_on_pan          = getattr(pc, "hide_on_pan", None),
        rect_fill_alpha      = getattr(pc, "rect_fill_alpha", 0.15),
        rect_line_alpha      = getattr(pc, "rect_line_alpha", 0.6),
        rect_line_width      = getattr(pc, "rect_line_width", 1.0),
        label_font_size      = getattr(pc, "label_font_size", "9pt"),
        label_x_offset       = getattr(pc, "label_x_offset", 5),
        label_y_offset       = getattr(pc, "label_y_offset", 5),
        label_text_align     = getattr(pc, "label_text_align", "left"),
        label_text_baseline  = getattr(pc, "label_text_baseline", "bottom"),
        arrow_size           = getattr(pc, "arrow_size", 10),
        arrow_length         = getattr(pc, "arrow_length", 0),
        output_names         = list(getattr(indicator, "output_names", [])),
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PANEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _build_live_equity_panel(config, theme, broker, all_figs):
    if broker is None or config.equity_mode == "none":
        return None, None, None
    source_eq = _empty_equity_source()
    fig_eq = _new_fig(
        config.equity_height, config.width, theme,
        name=config.equity_mode, output_backend=config.output_backend,
    )
    trailing_source = None
    if config.max_trailing_dd is not None:
        trailing_source = _empty_trailing_dd_source()
    _render_equity_placeholder(fig_eq, source_eq, theme, config,
                               trailing_source=trailing_source)
    all_figs.append(fig_eq)
    eq_ref = _EquitySourceRef(source=source_eq, broker=broker, config=config,
                              trailing_source=trailing_source)
    return fig_eq, source_eq, eq_ref


def _compute_live_initial_ranges(proxy, interval_ms):
    y_start, y_end = 0.0, 1.0
    x_start_init: "float | None" = None
    x_end_init: "float | None" = None
    ring = proxy._ohlc_ring
    if ring is not None and ring._n_closed > 0:
        from tradetropy.core.constants import _OHLC_COL as _OC
        n = min(ring._n_closed, ring._W)
        highs = ring.closed_window(_OC["high"], n)
        lows  = ring.closed_window(_OC["low"],  n)
        hi = float(np.nanmax(highs))
        lo = float(np.nanmin(lows))
        pad = (hi - lo) * 0.05 or abs(hi) * 0.005 or 1.0
        y_start = lo - pad
        y_end   = hi + pad
        ts_hist = ring.closed_window(_OC["ts"], n)
        ts_valid = ts_hist[np.isfinite(ts_hist) & (ts_hist > 0)]
        if len(ts_valid) > 0:
            ts_data_min = float(ts_valid[0])
            ts_data_max = float(ts_valid[-1])
            x_pad_r = interval_ms * 10
            x_pad_l = interval_ms * 5
            x_start_init = ts_data_min - x_pad_l
            x_end_init   = ts_data_max + x_pad_r

    if x_start_init is None or x_end_init is None:
        # Ring without closed history (warmup=0 case): no data to anchor
        # the viewport. Derive a non-null default window so the X axis
        # uses a manual Range1d (see _create_live_ohlc_figure). Anchor to the
        # partial candle if one exists, otherwise to "now" aligned to the
        # interval. The streaming + navigation.follow repositions the view
        # with the first real tick.
        if ring is not None and ring._ts_current_candle >= 0:
            anchor = float(ring._ts_current_candle)
        else:
            import time as _t
            anchor = (int(_t.time() * 1000) // interval_ms) * interval_ms
        x_start_init = anchor - interval_ms * 100
        x_end_init   = anchor + interval_ms * 10

    return y_start, y_end, x_start_init, x_end_init


def _create_live_ohlc_figure(config, theme, y_start, y_end, x_start_init, x_end_init, fig_eq):
    from bokeh.models import Range1d
    # Force ALWAYS an explicit Range1d on the X axis. If _new_ohlc_fig fell into
    # a DataRange1d (auto-range) and then LiveNavigationController wrote
    # .start/.end on it, Bokeh leaves the range inconsistent and does not render
    # the glyphs (blank canvas). _compute_live_initial_ranges guarantees non-null
    # start/end even without history.
    fig_ohlc, x_range = _new_ohlc_fig(
        config.ohlc_height, config.width, theme,
        y_range=Range1d(start=y_start, end=y_end),
        x_range=Range1d(start=x_start_init, end=x_end_init),
        output_backend=config.output_backend,
    )
    if fig_eq is not None:
        fig_eq.x_range = x_range
    return fig_ohlc, x_range


def _render_live_ohlc_bars(fig, source, symbol, interval_ms, theme, fp_proxy=None, ohlc_style="candle", price_digits=2):
    from bokeh.models import HoverTool, NumeralTickFormatter
    from tradetropy.plotting._util import _price_number_format
    dt_fmt = _datetime_format(interval_ms)
    price_fmt = _price_number_format(price_digits)
    ohlc_label = f"{symbol} ({_interval_label(interval_ms)})"
    if ohlc_style == "bar":
        bars = fig.segment(
            x0="ts", y0="Low", x1="ts", y1="High",
            source=source, color="wick_color", line_width=1.5, legend_label=ohlc_label,
        )
        fig.segment(
            x0="ts_left", y0="Open", x1="ts", y1="Open",
            source=source, color="wick_color", line_width=1.5, legend_label=ohlc_label,
        )
        fig.segment(
            x0="ts", y0="Close", x1="ts_right", y1="Close",
            source=source, color="wick_color", line_width=1.5, legend_label=ohlc_label,
        )
    else:
        fig.segment(
            x0="ts", y0="Low", x1="ts", y1="High",
            source=source, color="wick_color", line_width=1.5, legend_label=ohlc_label,
        )
        bars = fig.vbar(
            x="ts", width="bar_width",
            top="top_body", bottom="bottom_body",
            source=source, fill_color="candle_color",
            line_color="body_border_color", line_width="body_border_width",
            legend_label=ohlc_label,
        )
    tooltips = [
        ("Date", f"@ts{{{dt_fmt}}}"),
        ("Open", f"@Open{{{price_fmt}}}"), ("High", f"@High{{{price_fmt}}}"),
        ("Low", f"@Low{{{price_fmt}}}"), ("Close", f"@Close{{{price_fmt}}}"),
        ("Volume", "@Volume{0,0.00}"),
    ]
    if fp_proxy is not None and "fp_delta" in source.data:
        tooltips += [
            ("Bid", "@fp_bid{0,0}"), ("Ask", "@fp_ask{0,0}"),
            ("Delta", "@fp_delta{+0,0}"), ("POC", f"@fp_poc_price{{{price_fmt}}}"),
            ("POC Vol", "@fp_poc_vol{0,0}"), ("VAH", f"@fp_vah{{{price_fmt}}}"),
            ("VAL", f"@fp_val{{{price_fmt}}}"),
        ]
    fig.add_tools(HoverTool(tooltips=tooltips, formatters={"@ts": "datetime"}, mode="mouse", renderers=[bars]))
    # Price (Y) axis: fixed-decimal format so small prices are not truncated. A
    # footprint tick-based y-axis, if enabled, overrides this later.
    fig.yaxis.formatter = NumeralTickFormatter(format=price_fmt)


def _add_live_volume_bars(fig, source, theme):
    from bokeh.models import LinearAxis, Range1d, NumeralTickFormatter
    # Bottom-anchored band like backtest (render/_ohlc_volume.py): a plain
    # Range1d whose peak reaches ~20% of the frame height, re-anchored on every
    # navigation by autoscale_volume.js. A DataRange1d would auto-fit the volume
    # to the whole pane and let the bars drift with the price plane on vertical
    # pan (their base detaching from the bottom); the fixed Range1d + re-anchor
    # keeps them glued to the bottom band. Start neutral (0..1) because the live
    # source is empty at build time; the re-anchor sets the real end once data
    # arrives and on the first y_range change.
    vol_range = Range1d(start=0, end=1)
    fig.extra_y_ranges = {"vol": vol_range}
    ax_vol = LinearAxis(y_range_name="vol", axis_label="Volume")
    ax_vol.formatter = NumeralTickFormatter(format="0.0a")
    ax_vol.axis_line_color = theme["bg"]
    ax_vol.major_tick_line_color = theme["axis"]
    ax_vol.minor_tick_line_color = None
    ax_vol.major_label_text_color = theme["axis"]
    ax_vol.major_label_text_font_size = "8pt"
    ax_vol.axis_label_text_color = theme["axis"]
    ax_vol.axis_label_text_font_size = "9pt"
    ax_vol.axis_label_standoff = 12
    fig.add_layout(ax_vol, "left")
    fig.vbar(
        x="ts", width="bar_width", top="Volume", bottom=0,
        source=source, fill_color="candle_color", line_color="candle_color",
        line_width=0, alpha=0.30, y_range_name="vol", legend_label="Volume",
    )
    _configure_live_autoscale_volume(fig, source, vol_range)
    return vol_range


def _configure_live_autoscale_volume(fig, source, vol_range) -> None:
    """
    Keep the live volume bars pinned to the bottom band of the price pane.

    Mirror of the backtest wiring (_layout.py::_configure_autoscale_volume):
    reuse js/autoscale_volume.js to re-anchor the secondary ``vol`` range to the
    visible volume peak (~20% of the frame) on every x_range/y_range change, so
    Bokeh's pan/zoom tools cannot detach the volume from the pane bottom. The JS
    reads ``source.data['ts']`` and ``source.data['Volume']``, which the live
    OHLC source exposes, so the same callback is shared verbatim.
    """
    from bokeh.models import CustomJS

    cb = CustomJS(
        args=dict(source=source, y_range=vol_range, x_range=fig.x_range),
        code=(_JS_DIR / "autoscale_volume.js").read_text(),
    )
    fig.x_range.js_on_change("start", cb)
    fig.x_range.js_on_change("end", cb)
    fig.y_range.js_on_change("start", cb)
    fig.y_range.js_on_change("end", cb)


def _build_live_indicator_refs(strategy, config, theme, interval_ms, fig_ohlc, x_range):
    indicator_refs: list[_IndicatorSourceRef] = []
    ind_figs: list = []
    overlay_series_sources: list = []
    color_idx = 0
    for defn in strategy._indicator_defs:
        pc = _cfg(defn)
        indicator = defn["indicator"]
        if not pc.plot:
            continue
        from tradetropy.data.data import OhlcProxy as _OhlcProxy
        source = defn["source"]
        if not isinstance(source.proxy, _OhlcProxy):
            color_idx += 1
            continue
        ohlc_proxy_for_ind = defn.get("ohlc_proxy") or strategy._ohlc_proxies[0]
        meta = _meta_from_defn_live(defn, color_idx, interval_ms)
        if meta is None:
            color_idx += 1
            continue
        n_outputs = indicator.n_outputs
        ts_band_set = set(getattr(indicator, "ts_band_indices", []))
        is_stateful = not getattr(indicator, "use_partial", True)
        multi_band = defn.get("multi_band", False)
        if multi_band:
            ring_col_names: list[str] = defn.get("col_names", [])
        else:
            _single = defn.get("col_name", "")
            ring_col_names = [_single] if _single else []
        band_sources = [_empty_indicator_source() for _ in range(n_outputs)]
        bands = []
        _ind_ts_band_indices = list(getattr(indicator, "ts_band_indices", []))
        _output_names = list(getattr(indicator, "output_names", []))
        _price_band_count = 0
        for i in range(n_outputs):
            if multi_band:
                _cname = ring_col_names[i] if i < len(ring_col_names) else ""
            else:
                _cname = ring_col_names[0] if ring_col_names else ""
            _is_price_band = i not in ts_band_set
            _has_real_ts = _is_price_band and (i < len(_ind_ts_band_indices))
            _price_pos = _price_band_count
            _tag = _output_names[_price_band_count] if (_is_price_band and _price_band_count < len(_output_names)) else ""
            if _is_price_band:
                _price_band_count += 1

            # Diverging bar band (e.g. MACD histogram): carry the per-row color so
            # the vbar is colored by value sign from streamed data (WebGL-safe),
            # and seed the source with a matching empty ``bar_color`` column so
            # every stream()/patch() stays column-consistent.
            _bar_pos = _bar_neg = None
            if (
                _is_price_band
                and meta.bar_color_positive is not None
                and meta.bar_color_negative is not None
                and _resolve_dim(meta.renderer, _price_pos) == "bar"
            ):
                _bar_pos = _resolve_dim(meta.bar_color_positive, _price_pos)
                _bar_neg = _resolve_dim(meta.bar_color_negative, _price_pos)
                band_sources[i].data["bar_color"] = np.array([], dtype=object)

            bands.append(_BandRef(
                source=band_sources[i], band_idx=i, is_ts_band=i in ts_band_set,
                is_stateful=is_stateful, col_name=_cname, has_real_ts=_has_real_ts,
                tag_label=_tag, bar_pos_color=_bar_pos, bar_neg_color=_bar_neg,
            ))
        ind_ref = _IndicatorSourceRef(bands=bands, proxy=defn["proxy"], ohlc_proxy=ohlc_proxy_for_ind)
        indicator_refs.append(ind_ref)
        if meta.overlay:
            render_indicator(fig_ohlc, band_sources, meta, interval_ms=interval_ms)
            # Feed the overlay series into the price autoscale unless it opted
            # out, so the Y range frames the indicator (e.g. ZigZag) even where
            # candles rolled off the max_candles window.
            if not getattr(meta, "exclude_from_autoscale", False):
                overlay_series_sources.extend(band_sources)
            if meta.reference_lines:
                render_reference_lines(fig_ohlc, meta)
        else:
            fig_ind = _new_fig(
                meta.panel_height or config.indicator_height, config.width, theme,
                x_range=x_range,
                name=f"ind_{meta.name if isinstance(meta.name, str) else meta.name[0]}",
                output_backend=config.output_backend,
            )
            render_indicator(fig_ind, band_sources, meta, interval_ms=interval_ms)
            if meta.reference_lines:
                render_reference_lines(fig_ind, meta)
            panel_label = (
                meta.panel_title or (meta.name if isinstance(meta.name, str) else
                                     (meta.name[0] if meta.name else ""))
            )
            fig_ind.yaxis.axis_label = panel_label
            ind_figs.append(fig_ind)
        if pc.color is None:
            color_idx += 1
    return indicator_refs, ind_figs, overlay_series_sources


def _build_live_trades_ref(strategy, config, fig_ohlc, theme, interval_ms, trades_poll_interval, symbol):
    if not config.plot_trades:
        return None
    from tradetropy.plotting.render import render_trades_live
    source_trades = _empty_trades_source()
    (trades_renderer, source_markers, source_open, pos_line, pos_label,
     tpsl_source, tpsl_lines, tpsl_labels,
     pending_source, pending_lines, pending_labels) = render_trades_live(
        fig_ohlc, source_trades, theme, interval_ms)
    sesh = getattr(strategy, "_sesh", None)
    if sesh is None:
        return None
    return _TradesSourceRef(
        source=source_trades, sesh=sesh, symbol=symbol, theme=theme,
        interval_ms=interval_ms, poll_interval_s=trades_poll_interval,
        align_to_candle=config.align_trades_to_candle,
        fig=fig_ohlc, renderer=trades_renderer,
        source_markers=source_markers, source_open=source_open,
        pos_line=pos_line, pos_label=pos_label,
        tpsl_source=tpsl_source, tpsl_lines=tpsl_lines, tpsl_labels=tpsl_labels,
        pending_source=pending_source, pending_lines=pending_lines,
        pending_labels=pending_labels,
    )


def _build_live_footprint_refs(strategy, config, fig_ohlc, source_ohlc, theme, max_candles, primary_proxy):
    fp_refs: list[_FootprintSourceRef] = []
    if not config.plot_footprint or not strategy._fp_proxies:
        return fp_refs
    from bokeh.models import CustomJS, ColumnDataSource
    for fp_proxy in strategy._fp_proxies:
        if fp_proxy._ring is None:
            continue
        fp_src_bid = _empty_fp_source()
        fp_src_ask = _empty_fp_source()
        fp_interval = fp_proxy.interval_ms
        fp_zoom = config.footprint_zoom_range
        rects_bid = fig_ohlc.rect(
            x="x", y="y", width="cell_w", height="cell_h",
            width_units="data", height_units="data",
            source=fp_src_bid, fill_color="fill", line_color="lc", line_width="lw",
        )
        rects_ask = fig_ohlc.rect(
            x="x", y="y", width="cell_w", height="cell_h",
            width_units="data", height_units="data",
            source=fp_src_ask, fill_color="fill", line_color="lc", line_width="lw",
        )
        txt_bid = fig_ohlc.text(
            x="x", y="y", text="text",
            source=fp_src_bid, text_align="center", text_baseline="middle",
            text_font_size="6.6pt", text_color="tc",
        )
        txt_ask = fig_ohlc.text(
            x="x", y="y", text="text",
            source=fp_src_ask, text_align="center", text_baseline="middle",
            text_font_size="6.6pt", text_color="tc",
        )
        _fp_lazy_cb = CustomJS(
            args=dict(
                txt_bid=txt_bid, txt_ask=txt_ask, rects_bid=rects_bid, rects_ask=rects_ask,
                dummy=None, interval_ms=fp_interval, zoom_range=fp_zoom,
                ohlc_source=source_ohlc, bar_width_wide=int(fp_interval * 0.9),
                bar_width_narrow=int(fp_interval * 0.10),
            ),
            code=(_JS_DIR / "fp_lazy_live.js").read_text(),
        )
        fig_ohlc.x_range.js_on_change("start", _fp_lazy_cb)
        fig_ohlc.x_range.js_on_change("end", _fp_lazy_cb)
        rects_bid.visible = False
        rects_ask.visible = False
        txt_bid.visible = False
        txt_ask.visible = False
        _dummy_src = ColumnDataSource(dict(x=[], y=[]))
        _dummy = fig_ohlc.scatter(
            x="x", y="y", source=_dummy_src, size=0, alpha=0,
            color="#2ECC71", legend_label="Footprint",
        )
        _fp_lazy_cb.args["dummy"] = _dummy
        _fp_renderers = [rects_bid, rects_ask, txt_bid, txt_ask]
        _cb_legend = CustomJS(
            args=dict(
                dummy=_dummy, fp_renderers=_fp_renderers, ohlc_source=source_ohlc,
                bar_width_wide=int(fp_interval * 0.9), bar_width_narrow=int(fp_interval * 0.3),
            ),
            code=(_JS_DIR / "legend_toggle.js").read_text(),
        )
        _dummy.js_on_change("visible", _cb_legend)
        fp_refs.append(_FootprintSourceRef(
            source_bid=fp_src_bid, source_ask=fp_src_ask, fp_proxy=fp_proxy,
            ohlc_proxy=primary_proxy, interval_ms=fp_interval, theme=theme,
            _n_candles_max=max_candles, renderers=[rects_bid, rects_ask, txt_bid, txt_ask],
            zoom_range=fp_zoom,
        ))
    return fp_refs


def _build_live_vp_refs(strategy, config, fig_ohlc, theme, interval_ms, primary_proxy, x_range=None):
    """
    Build the live refs for declared indicators that emit draw primitives.

    Detects any indicator that overrides ``Indicator.draw`` (e.g. VolumeProfile,
    RollingVolumeProfile, TickVolumeProfile, and the order-flow CVD / DeltaBars /
    DeltaVolumeInfo panels) and creates a ref carrying an empty renderer registry. No
    glyph is created here: the updater calls ``render_tool_groups`` with the
    ref's registry, which creates the glyphs lazily on the first non-empty draw
    and updates them in place afterwards (developing geometry redrawn each tick).

    Overlay draw indicators draw on the price figure; own-panel draw indicators
    (overlay=False, e.g. CVD candles) get their own panel figure, returned in
    ``panel_figs`` so the caller can append them to the layout. Both paths go
    through the same generic renderer - only the target figure differs.

    Returns:
        tuple[list, list]: (vp_refs, panel_figs).
    """
    from tradetropy.ta.base import Indicator, _CATEGORY_OVERLAY_DEFAULTS
    from tradetropy.plotting.live.updater._refs import _VolumeProfileSourceRef

    vp_refs: list[_VolumeProfileSourceRef] = []
    panel_figs: list = []
    if not getattr(config, "plot_volume_profile", True):
        return vp_refs, panel_figs

    # Grid phase for snapping draw() primitives to the candle grid: any real
    # candle-open ts works (its remainder mod interval_ms is the phase). Use the
    # oldest candle currently in the primary OHLC ring; 0 (epoch grid) if none.
    _align_indicators = getattr(config, "align_indicators_to_candle", True)
    _candle_origin_ms = 0
    _store = getattr(primary_proxy, "_ohlc_store", None)
    _matrix = getattr(_store, "matrix", None) if _store is not None else None
    if _matrix is not None and len(_matrix) > 0:
        _candle_origin_ms = int(_matrix[0, 0])

    for defn in strategy._indicator_defs:
        indicator = defn["indicator"]
        # Duck-typed on the METHOD: the indicator implements draw().
        if type(indicator).draw is Indicator.draw:
            continue
        pc = _cfg(defn)
        if not pc.plot:
            continue
        # Resolve overlay (own panel vs price overlay), same rule as the meta.
        overlay = pc.overlay
        if overlay is None:
            category = getattr(indicator, "category", None)
            overlay = (pc.resolve_overlay(category) if category
                       else _CATEGORY_OVERLAY_DEFAULTS.get("other", False))
        if overlay:
            target_fig = fig_ohlc
        else:
            disp = getattr(indicator, "display_name", lambda: "")
            label = (pc.panel_title or (pc.name if isinstance(pc.name, str)
                     else (pc.name[0] if isinstance(pc.name, list) and pc.name
                           else disp())))
            target_fig = _new_fig(
                pc.panel_height or config.indicator_height, config.width, theme,
                x_range=x_range,
                name=f"ind_{label or 'of'}",
                output_backend=config.output_backend,
            )
            target_fig.yaxis.axis_label = label or ""
            panel_figs.append(target_fig)
        src_proxy = None
        source = defn.get("source")
        if source is not None:
            src_proxy = getattr(source, "proxy", None)
        # Overlays that opt into the price autoscale (exclude_from_autoscale=
        # False, e.g. the Heatmap) get a mirror CDS the OHLC autoscale reads.
        # The real quad source is created lazily on the first tick, so the
        # build-time autoscale references this mirror; the updater keeps it in
        # sync each tick.
        autoscale_mirror = None
        if overlay and not getattr(pc, "exclude_from_autoscale", True):
            from bokeh.models import ColumnDataSource
            autoscale_mirror = ColumnDataSource(
                data=dict(left=[], right=[], bottom=[], top=[])
            )
        vp_refs.append(_VolumeProfileSourceRef(
            defn=defn, indicator=indicator, fig=target_fig,
            ohlc_proxy=primary_proxy, interval_ms=interval_ms, theme=theme,
            source_proxy=src_proxy, lazy_zoom_range=config.labels_zoom_range,
            align_to_candle=_align_indicators, candle_origin_ms=_candle_origin_ms,
            autoscale_mirror=autoscale_mirror,
        ))
    return vp_refs, panel_figs


def _create_tool_ref(strategy, fig_ohlc, theme, interval_ms, lazy_zoom_range=40):
    """
    Creates the ref for use_tool() snapshots in live (generic renderer).

    Does not create glyphs yet: the ref stores an empty registry that the
    updater fills by calling ``render_tool_groups`` with each tool's
    primitives. Each tool produces an independent legend entry; since
    fig_ohlc already has a Legend with click_policy="hide", those entries
    inherit the toggle.
    """
    from tradetropy.plotting.live.updater._refs import _ToolSnapshotRef

    return _ToolSnapshotRef(
        strategy=strategy, fig=fig_ohlc,
        interval_ms=interval_ms, theme=theme,
        lazy_zoom_range=lazy_zoom_range,
    )


def _build_live_tool_ref(strategy, config, fig_ohlc, theme, interval_ms):
    """
    Builds the tool ref at build-time ONLY if snapshots are already accumulated.

    Tools are invoked via use_tool() inside on_data(), so when building the
    document there are normally no snapshots yet (the first on_data() has not
    run). In that case it returns None and the updater creates the ref
    lazily on the first tick that has a snapshot (see _update_tool_snapshots).

    If the document is built with snapshots already present (replay, chart
    attached late, or tests), the ref is created immediately so the "Tools"
    legend is not lost.
    """
    if not getattr(config, "plot_volume_profile", True):
        return None
    if not getattr(strategy, "_tool_snapshots", None):
        return None
    return _create_tool_ref(strategy, fig_ohlc, theme, interval_ms,
                            lazy_zoom_range=config.labels_zoom_range)


def _build_live_drawdown_panel(config, theme, broker, x_range):
    if not config.plot_drawdown or broker is None:
        return None, None
    from tradetropy.plotting.render import render_drawdown
    source_dd = _empty_equity_source()
    source_dd.data["drawdown"] = np.array([], dtype=np.float64)
    fig_dd = _new_fig(
        config.drawdown_height, config.width, theme,
        x_range=x_range, name="drawdown", output_backend=config.output_backend,
    )
    render_drawdown(fig_dd, source_dd, theme)
    fig_dd.x_range = x_range
    return fig_dd, source_dd


def _build_live_pl_panel(config, theme, trades_ref, x_range):
    if not config.plot_pl or not config.plot_trades:
        return None, None
    if trades_ref is not None:
        source_pl = trades_ref.source
    else:
        source_pl = _empty_trades_source()
    fig_pl = _new_fig(
        config.pl_height, config.width, theme,
        x_range=x_range, name="pl", output_backend=config.output_backend,
    )
    fig_pl.x_range = x_range
    return fig_pl, source_pl


def _compute_vp_right_padding_candles(vp_refs) -> int:
    """
    Candles to reserve to the right of the live edge so the Volume Profile
    histogram in "visible" view (VPVR) fits in view when following the
    candle. Profiles in "session" view anchor to their own candles and
    do not need extra padding, so they return the default padding.
    """
    from tradetropy.plotting.live.navigation import _RIGHT_PADDING_CANDLES as _DEFAULT
    if not vp_refs:
        return _DEFAULT
    needs_visible = any(
        getattr(ref.indicator, "view", "session") == "visible"
        for ref in vp_refs
    )
    if not needs_visible:
        return _DEFAULT
    from tradetropy.plotting.sources import vp_visible_right_pad_bars
    # Small cushion so the longest bar does not touch the right edge.
    return vp_visible_right_pad_bars() + 2


def _assemble_live_layout_and_updater(
    doc, all_figs, config, theme, fig_dd, fig_pl, fig_ohlc, ind_figs,
    fp_refs, source_ohlc, ohlc_ref, indicator_refs, eq_ref, trades_ref,
    source_dd, source_pl, symbol, interval_ms, broker, replay_controller,
    io_loop, x_start_init, x_end_init, max_candles, vp_refs=None, tool_ref=None,
    strategy=None, engine=None, overlay_series_sources=None,
):
    if fig_dd is not None:
        all_figs.append(fig_dd)
    if fig_pl is not None:
        all_figs.append(fig_pl)
    all_figs.append(fig_ohlc)
    all_figs.extend(ind_figs)
    fp_tick_size = 0.0
    fp_source_bid = None
    if fp_refs:
        ts = fp_refs[0].fp_proxy.config.tick_size
        if ts is not None:
            fp_tick_size = float(ts)
        fp_source_bid = fp_refs[0].source_bid
    # Follow state shared with the client: the autoscale JS reads it to
    # decide whether to rescale on each data change (only in follow mode).
    # Python navigation updates it when follow is toggled on/off.
    from bokeh.models import ColumnDataSource as _CDS
    follow_state = _CDS(data=dict(follow=[1]))
    # Mirror sources of overlays that opt into the price autoscale (Heatmap).
    _hm_mirrors = [
        ref.autoscale_mirror for ref in (vp_refs or [])
        if getattr(ref, "autoscale_mirror", None) is not None
    ]
    _autoscale_cb = _configure_autoscale_ohlc(
        fig_ohlc, source_ohlc, theme,
        fp_tick_size=fp_tick_size, fp_source_bid=fp_source_bid,
        follow_state=follow_state,
        heatmap_sources=_hm_mirrors,
        indicator_sources=overlay_series_sources,
    )
    if fp_tick_size > 0:
        _configure_fp_yaxis(fig_ohlc, fp_tick_size, autoscale_cb=_autoscale_cb)
    _configure_legends(all_figs, theme)
    _configure_margins(all_figs)
    # tick_size of the main symbol (from the broker), with the footprint
    # bucket as fallback when the symbol is not registered.
    price_tick = 0.0
    _symbols = getattr(broker, "symbols", None)
    if _symbols:
        _cfg = _symbols.get(symbol) or next(iter(_symbols.values()), None)
        if _cfg is not None:
            price_tick = float(getattr(_cfg, "tick_size", 0.0) or 0.0)
    _configure_crosshair(all_figs, theme, tick_size=(price_tick or fp_tick_size),
                         ohlc_fig=fig_ohlc)
    # Navigation is created and attach() is called BEFORE doc.add_root() so
    # that the on_event(DoubleTap) Python handler is registered before the
    # figure enters the Document -- a Bokeh 3.x requirement for the handler
    # to connect correctly to the WebSocket session.
    navigation = LiveNavigationController(
        fig_ohlc, interval_ms, follow_state=follow_state,
        right_padding_candles=_compute_vp_right_padding_candles(vp_refs),
    )
    if x_start_init is not None and x_end_init is not None:
        _initial_window = int(x_end_init - x_start_init) - interval_ms * 10
        if _initial_window > interval_ms * 10:
            navigation.follow_window_ms = _initial_window
    navigation.attach(source_ohlc)
    stats_div = None
    replay_row = None
    if replay_controller is not None:
        from tradetropy.replay.controller import build_replay_controls as _brc
        _, _, _, replay_row = _brc(replay_controller, io_loop, doc=doc)
    from bokeh.layouts import column as _bk_column
    from bokeh.layouts import row as _bk_row

    # The replay controls (when present) sit in their OWN row at the very top,
    # stretched full width, so they never share a line with the stats div. The
    # stats div, if enabled, goes directly below them.
    _replay_bar = None
    if replay_row is not None:
        _replay_bar = _bk_row(
            replay_row, sizing_mode="stretch_width",
            styles={"display": "flex", "justify-content": "flex-start",
                    "align-items": "center",
                    "padding": "4px 12px 2px 8px",
                    "background": theme["bg"]},
        )

    if config.plot_stats:
        from tradetropy.plotting._layout import _build_stats_div
        stats_div = _build_stats_div(None, theme)
        stats_div.sizing_mode = "stretch_width"

    header_children = []
    if _replay_bar is not None:
        header_children.append(_replay_bar)
    if stats_div is not None:
        header_children.append(stats_div)

    layout = _build_layout(all_figs)

    if header_children:
        _header = _bk_column(
            *header_children, sizing_mode="stretch_width",
            styles={"background": theme["bg"],
                    "border-bottom": "1px solid rgba(255,255,255,0.06)"},
        )
        main_content = _bk_column(
            _header, layout, sizing_mode="stretch_width",
            styles={"background-color": theme["bg"], "padding": "0", "margin": "0"},
        )
    else:
        main_content = layout

    # Manual paper-trading panels (Order Ticket + Position Modifier) on the
    # right. Built only when an engine that wants them is attached
    # (PaperEngine).
    _panel_refresh = None
    _orders_refresh = None
    order_panel = None
    if engine is not None and getattr(engine, "_WANTS_ORDER_TICKET", False):
        from tradetropy.plotting.live.order_ticket import (
            build_order_ticket, build_position_modifier, build_orders_panel,
        )
        _sym = engine.primary_symbol() or symbol
        _cfg = broker.symbols.get(_sym) if broker else None
        _price_step = float(getattr(_cfg, "tick_size", 0.01) or 0.01) if _cfg else 0.01
        _vol_step = float(getattr(_cfg, "volume_step", 0.01) or 0.01) if _cfg else 0.01
        ticket_panel, _sides = build_order_ticket(
            engine.submit_command, _sym, doc=doc,
            price_step=_price_step, vol_step=_vol_step,
        )
        modifier_panel, _mod_w = build_position_modifier(
            engine.submit_command, _sym,
            fetch_position=lambda: engine.net_position(_sym),
            price_step=_price_step,
        )
        orders_panel, _ord_w = build_orders_panel(
            engine.submit_command, _sym,
            fetch_orders=lambda: (
                broker.get_orders(_sym) if broker else []
            ),
            cancel_order=lambda ticket: engine.cancel_order(ticket),
        )
        _panel_refresh = _mod_w["refresh"]
        _orders_refresh = _ord_w["refresh"]
        order_panel = _bk_column(
            ticket_panel, modifier_panel, orders_panel,
            styles={"gap": "8px", "padding": "6px", "background": theme["bg"]},
        )

    if order_panel is not None:
        root = _bk_row(
            main_content, order_panel, sizing_mode="stretch_width",
            styles={"background-color": theme["bg"]},
        )
    else:
        root = main_content
    doc.add_root(root)
    stats_broker = eq_ref.broker if eq_ref is not None else broker
    trailing_source = eq_ref.trailing_source if eq_ref is not None else None
    updater = LiveSourceUpdater(
        ohlc_refs=[ohlc_ref], indicator_refs=indicator_refs, equity_ref=eq_ref,
        fp_refs=fp_refs, max_candles=max_candles, fig_ohlc=fig_ohlc,
        trades_ref=trades_ref, drawdown_source=source_dd,
        trailing_dd_source=trailing_source, pl_source=source_pl,
        fig_pl=fig_pl, navigation=navigation, stats_div=stats_div,
        stats_broker=stats_broker, stats_symbol=symbol,
        stats_interval_ms=interval_ms, stats_theme=theme,
        vp_refs=vp_refs or [],
        tool_ref=tool_ref,
        strategy=strategy,
        panel_refresh=_panel_refresh,
        orders_refresh=_orders_refresh,
    )
    return doc, updater


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def build_live_document(
    strategy: "Strategy",
    config: PlotConfig,
    max_candles: int,
    broker=None,
    doc: "Document | None" = None,
    trades_poll_interval: float = 5.0,
    replay_controller=None,
    io_loop=None,
    engine=None,
) -> "tuple[Document, LiveSourceUpdater]":
    """
    Builds the Bokeh Document with empty figures and ColumnDataSources.

    Reuses the same render_* functions used by the static backtest.
    Sources start empty and are filled via stream/patch in update().

    Parameters
    ----------
    doc : Document | None
        If provided, builds on that Document (Bokeh session).
        If None, creates a new one (legacy / tests mode).
    trades_poll_interval : float
        How often (in seconds) to query the broker for new trades
        (default 5.0 s). The check runs inside the Bokeh IOLoop, so it
        does not block the engine or the data feed.
    replay_controller : ReplayController | None
        If provided, adds replay control widgets (pause/play, step,
        speed) to the Document layout. Default: None (no controls).
    io_loop : tornado.IOLoop | None
        Bokeh server IOLoop -- needed for the ReplayController's
        on_finished callback to notify the Document. Ignored if
        replay_controller is None.

    Returns (doc, updater) where:
        doc     : Document (the one provided or a new one)
        updater : LiveSourceUpdater with all necessary references
    """
    from bokeh.document import Document

    if doc is None:
        doc = Document()
    theme = _theme(config)

    # ── Determine symbol and interval from the strategy ────────────────────
    if not strategy._ohlc_proxies:
        raise ValueError(
            "LiveChart requires at least one subscribe_ohlc() in the strategy."
        )

    primary_proxy = strategy._ohlc_proxies[0]
    symbol        = primary_proxy.symbol
    interval_ms   = primary_proxy.interval_ms

    # Price decimals for hover / y-axis formatting. Prefer the broker's symbol
    # config (digits + tick_size); fall back to the OHLC proxy config so small
    # prices (e.g. 0.1844) are not truncated to 0.18.
    from tradetropy.plotting._util import _resolve_price_digits as _rpd
    _sym_cfg = None
    _bsyms = getattr(broker, "symbols", None)
    if _bsyms:
        _sym_cfg = _bsyms.get(symbol) or next(iter(_bsyms.values()), None)
    if _sym_cfg is None:
        _sym_cfg = getattr(primary_proxy, "config", None)
    price_digits = _rpd(
        getattr(_sym_cfg, "digits", None),
        getattr(_sym_cfg, "tick_size", None),
    )

    # ── Find the connected FpProxy (live) sharing symbol and interval ──
    primary_fp_proxy = None
    if config.plot_footprint and strategy._fp_proxies:
        for _fp in strategy._fp_proxies:
            if _fp.symbol == symbol and _fp.interval_ms == interval_ms and _fp._ring is not None:
                primary_fp_proxy = _fp
                break

    # ======================================================================
    # 1. Empty sources
    # ======================================================================

    source_ohlc = _build_initial_ohlc_source(primary_proxy, interval_ms, theme, fp_proxy=primary_fp_proxy)

    # ======================================================================
    # 2. Figures -- same hierarchy as plotting.py
    # ======================================================================

    all_figs = []

    fig_eq, source_eq, eq_ref = _build_live_equity_panel(config, theme, broker, all_figs)

    y_start, y_end, x_start_init, x_end_init = _compute_live_initial_ranges(primary_proxy, interval_ms)

    fig_ohlc, x_range = _create_live_ohlc_figure(config, theme, y_start, y_end, x_start_init, x_end_init, fig_eq)

    ohlc_ref = _OhlcSourceRef(
        source=source_ohlc, proxy=primary_proxy, interval_ms=interval_ms,
        theme=theme, fp_proxy=primary_fp_proxy,
    )

    _render_live_ohlc_bars(fig_ohlc, source_ohlc, symbol, interval_ms, theme, fp_proxy=primary_fp_proxy, ohlc_style=config.ohlc_style, price_digits=price_digits)

    if config.plot_volume:
        ohlc_ref.vol_range = _add_live_volume_bars(fig_ohlc, source_ohlc, theme)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Indicators + Trades + Footprint
    # ══════════════════════════════════════════════════════════════════════════

    indicator_refs, ind_figs, overlay_series_sources = _build_live_indicator_refs(
        strategy, config, theme, interval_ms, fig_ohlc, x_range,
    )

    trades_ref = _build_live_trades_ref(strategy, config, fig_ohlc, theme, interval_ms, trades_poll_interval, symbol)

    fp_refs = _build_live_footprint_refs(strategy, config, fig_ohlc, source_ohlc, theme, max_candles, primary_proxy)

    vp_refs, vp_panel_figs = _build_live_vp_refs(strategy, config, fig_ohlc, theme, interval_ms, primary_proxy, x_range)
    ind_figs.extend(vp_panel_figs)

    tool_ref = _build_live_tool_ref(strategy, config, fig_ohlc, theme, interval_ms)

    fig_dd, source_dd = _build_live_drawdown_panel(config, theme, broker, x_range)

    fig_pl, source_pl = _build_live_pl_panel(config, theme, trades_ref, x_range)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Assemble layout + updater
    # ══════════════════════════════════════════════════════════════════════════

    return _assemble_live_layout_and_updater(
        doc, all_figs, config, theme, fig_dd, fig_pl, fig_ohlc, ind_figs,
        fp_refs, source_ohlc, ohlc_ref, indicator_refs, eq_ref, trades_ref,
        source_dd, source_pl, symbol, interval_ms, broker, replay_controller,
        io_loop, x_start_init, x_end_init, max_candles, vp_refs, tool_ref,
        strategy=strategy, engine=engine,
        overlay_series_sources=overlay_series_sources,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_equity_placeholder(fig, source, theme: dict, config: PlotConfig,
                               trailing_source=None) -> None:
    """
    Renders the equity figure with an empty source.
    render_equity() needs numeric values for the Peak/MaxDD/Final dots.
    Here we draw only the line -- dots will be added when real data arrives.
    """
    from bokeh.models import Span, NumeralTickFormatter, HoverTool

    is_percent  = config.equity_unit == "percent"
    mode        = config.equity_mode
    fmt         = "0.[00]%" if is_percent else "$0,0"
    label_prefix = "Return" if mode == "return" else "Equity"
    baseline    = 0.0

    fig.add_layout(Span(
        location   = baseline,
        dimension  = "width",
        line_color = theme["span_zero"],
        line_width = 0.8,
        line_dash  = "dashed",
    ))

    line_glyph = fig.line(
        x="ts", y="equity",
        source=source,
        color=theme["equity_line"],
        line_width=2.5,
        alpha=0.9,
        legend_label=f"{label_prefix}",
    )

    if trailing_source is not None and config.max_trailing_dd is not None:
        tdd_line = fig.line(
            x="ts", y="trailing_dd",
            source=trailing_source,
            color=theme["trailing_dd_line"],
            line_width=1.5,
            line_dash="dashed",
            alpha=0.85,
            legend_label=f"Trailing DD (max {config.max_trailing_dd:.0%})",
        )
        fig.add_tools(HoverTool(
            tooltips=[
                ("Date",       "@ts{%Y-%m-%d %H:%M}"),
                ("Trailing DD", f"@trailing_dd{{{fmt}}}"),
            ],
            formatters={"@ts": "datetime"},
            mode="mouse",
            renderers=[tdd_line],
        ))

    fig.yaxis.formatter   = NumeralTickFormatter(format=fmt)
    fig.yaxis.axis_label  = "Return %" if mode == "return" else "Equity"

    fig.add_tools(HoverTool(
        tooltips=[
            ("Date",       "@ts{%Y-%m-%d %H:%M}"),
            (label_prefix,  f"@equity{{{fmt}}}"),
        ],
        formatters={"@ts": "datetime"},
        mode="mouse",
        renderers=[line_glyph],
    ))
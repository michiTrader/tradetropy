"""
================
Plotting module for backtest results.

MAIN USAGE
──────────
    bt = BacktestEngine.by_ticks(strategy, data=(ticks,), sesh=sesh).run()
    bt.plot()
    bt.plot(PlotConfig(plot_trades=False, plot_volume=False))

    from tradetropy.plotting import plot, PlotConfig
    plot(bt)
    plot(bt, PlotConfig(equity_mode="none", width=1400))
    plot(bt, PlotConfig(equity_mode="balance", equity_unit="currency"))
    plot(bt, PlotConfig(equity_mode="return", equity_unit="percent"))

INDICATORS IN THE PLOT
───────────────────────
    class MyStrategy(Strategy):
        def init(self):
            self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
            self.sma = self.add_indicator(
                self.btc.close_ref, SMA(20),
                name="SMA(20)",
                plot=True,
                overlay=True,
                scatter=False,
                color="#FF6B35",
            )
            self.rsi = self.add_indicator(
                self.btc.close_ref, RSI(14),
                name="RSI(14)",
                plot=True,
                overlay=False,
                color="#7B2FBE",
            )
            # RSI will automatically draw the reference_lines (70/30/50)
            # declared in its plot_config.

ARCHITECTURE
────────────
    Section 1  - Data types and configuration
    Section 2  - Internal helpers (colors, formatting)
    Section 3  - Data builders -> ColumnDataSource
    Section 4  - Renderers (draw on existing figures)
    Section 5  - Figure factory with base styling
    Section 6  - Layout and global configurations (crosshair, legends)
    Section 7  - Public entrypoint: plot()

EXTERNAL DEPENDENCIES
─────────────────────
    bokeh >= 3.0
    numpy, pandas  (already required by the project)
"""

from __future__ import annotations
import pathlib

import numpy as np

from tradetropy.plotting.config import PlotConfig
from tradetropy.plotting._fig_factory import _new_fig, _new_ohlc_fig, _apply_style
from tradetropy.plotting._util import (
    _ensure_bokeh, 
    _theme, 
    _gather_plot_data,
    _resample_ohlc,
    _snap_primitive_groups,
)
from tradetropy.plotting.sources import (
    build_ohlc_source, 
    _build_footprint_cols,
    build_indicator_source,
    build_trades_source,
    build_drawdown_source,
    build_footprint_source,
    build_volume_profile_source,
    _prepare_equity_series,
    build_trailing_dd_source,
)
from tradetropy.plotting.render import (
    render_ohlc,
    render_volume,
    render_trades,
    render_equity,
    render_drawdown,
    render_indicator,
    # ── Resolve renderer type for this dimension ────────────────────────
    render_reference_lines,
    render_footprint,
    render_volume_profile,
    render_pl_bars,
)
from tradetropy.plotting._layout import (
    _configure_autoscale_ohlc,
    _configure_autoscale_volume,
    _configure_autoscale_indicator,
    _configure_autoscale_primitives,
    _configure_lazy_labels,
    _configure_autoscale_equity,
    _configure_fp_yaxis,
    _configure_legends,
    _configure_crosshair,
    _configure_margins,
    _build_layout,
    _show_layout,
    _build_stats_div,
)

_JS_DIR = pathlib.Path(__file__).parent / "js"

# ══════════════════════════════════════════════════════════════════════════════
# PANEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _render_overlay_indicators(fig, indicators: list, interval_ms: int, theme: dict = None,
                               label_sink: list = None, lazy_x_range=None,
                               lazy_zoom_range: int = None) -> list:
    """
    Render every overlay indicator onto the price figure.

    Returns:
        list: ColumnDataSources of overlay draw-primitive groups that opt into
        the price autoscale (``exclude_from_autoscale=False``, e.g. the
        Heatmap's liquidity cells). The caller feeds these to the OHLC autoscale
        so the visible geometry is never clipped.
    """
    from tradetropy.plotting.render._tools import render_tool_groups

    autoscale_sources: list = []
    for meta in indicators:
        if not meta.overlay:
            continue

        if getattr(meta, "_draw_primitives", None):
            # Geometry emitted via Indicator.draw() (VP histogram, zones, ...).
            # Drawn first so it sits behind any series (e.g. VP POC/VAH/VAL).
            registry = render_tool_groups(
                fig, meta._draw_primitives, theme=theme, interval_ms=interval_ms,
                show_legend=meta.show_legend, label_sink=label_sink,
                lazy_x_range=lazy_x_range, lazy_zoom_range=lazy_zoom_range,
            )
            # Overlays that opt into the price autoscale (e.g. the Heatmap)
            # contribute their quad sources so the Y range covers their cells.
            if not getattr(meta, "exclude_from_autoscale", True):
                autoscale_sources.extend(registry.values())
        elif getattr(meta, "_vp_data", None):
            # Legacy VP histogram path (kept until fully removed in cleanup).
            vp_src = build_volume_profile_source(meta._vp_data, interval_ms=interval_ms)
            render_volume_profile(fig, vp_src, interval_ms)

        # Render the per-bar series only for real series renderers. Geometric
        # renderers (rect/span/label/arrow) are fully expressed by draw()
        # primitives, so their bands are data-only (read in on_data()).
        if not _is_geometric_renderer(meta.renderer):
            srcs = build_indicator_source(meta)
            render_indicator(fig, srcs, meta, interval_ms=interval_ms)
            if meta.reference_lines:
                render_reference_lines(fig, meta)

    return autoscale_sources


_GEOMETRIC_RENDERERS = {"rect", "span", "label", "arrow", "segment", "none"}


def _is_geometric_renderer(renderer) -> bool:
    """True when every band uses a geometric renderer (rect/span/label/arrow)."""
    if isinstance(renderer, (list, tuple)):
        return bool(renderer) and all(r in _GEOMETRIC_RENDERERS for r in renderer)
    return renderer in _GEOMETRIC_RENDERERS


def _render_tool_snapshots(fig, bt, interval_ms: int, theme: dict,
                           label_sink: list = None, lazy_x_range=None,
                           lazy_zoom_range: int = None) -> None:
    """
    Draw the snapshots accumulated by use_tool() via the generic tool renderer.

    Each tool (FixedRangeVP, FibRetracement, ...) emits declarative draw
    primitives, grouped by legend name (one independently toggled entry per tool
    type). The generic renderer translates those primitives into Bokeh glyphs.
    Snapshots are pull-computed by the strategy, so this only runs when some were
    stored.

    Args:
        fig: Target OHLC price figure.
        bt: Post-run engine exposing ``strategy._tool_snapshots``.
        interval_ms (int): OHLC interval, used for bar geometry.
        theme (dict): Plot theme for colors.
        label_sink (list): Collects (label, group_glyph) for the lazy callback.
        lazy_x_range: Shared price x_range for the zoom-aware legend link.
        lazy_zoom_range (int): Max candles in view to keep labels visible.
    """
    snapshots = getattr(getattr(bt, "strategy", None), "_tool_snapshots", None)
    if not snapshots:
        return

    from tradetropy.ta.tool import collect_draw_primitives
    from tradetropy.plotting.render._tools import render_tool_groups

    groups = collect_draw_primitives(snapshots)
    if not groups:
        return

    render_tool_groups(
        fig, groups, theme=theme, interval_ms=interval_ms,
        label_sink=label_sink, lazy_x_range=lazy_x_range,
        lazy_zoom_range=lazy_zoom_range,
    )


def _build_equity_panel(config, theme, stats, all_figs):
    if config.equity_mode == "none" or stats is None:
        return None, None, None
    eq_plot, baseline = _prepare_equity_series(
        stats.equity_curve, stats.initial_balance,
        config.equity_mode, config.equity_unit,
    )
    ts = eq_plot.index.values.astype("datetime64[ms]")
    vals = eq_plot.to_numpy(dtype=np.float64)
    # Use positional indexing throughout. A tick-driven backtest records equity
    # per tick, so the curve can carry duplicate timestamps (several ticks in
    # the same millisecond); label-based .loc[ts] would then return a Series,
    # not a scalar. argmax/argmin resolve the first occurrence unambiguously and
    # _eq_orig shares eq_plot's index/order/length, so positions map 1:1.
    peak_i = int(np.nanargmax(vals))
    peak_val = float(vals[peak_i])
    peak_ts = eq_plot.index[peak_i]
    final_val = float(vals[-1])
    final_ts = eq_plot.index[-1]
    _eq_orig = stats.equity_curve
    dd_vals = (_eq_orig / _eq_orig.cummax() - 1.0).to_numpy(dtype=np.float64)
    max_dd_i = int(np.nanargmin(dd_vals))
    max_dd_pct = float(dd_vals[max_dd_i])
    max_dd_ts = eq_plot.index[max_dd_i]
    max_dd_val = float(vals[max_dd_i])
    from bokeh.models import ColumnDataSource
    eq_source = ColumnDataSource(dict(ts=ts, equity=vals))

    trailing_source = None
    if config.max_trailing_dd is not None:
        trailing_source = build_trailing_dd_source(
            stats.equity_curve, stats.initial_balance,
            config.max_trailing_dd, config.equity_mode, config.equity_unit,
        )

    fig_eq = _new_fig(
        config.equity_height, config.width, theme,
        name=config.equity_mode, output_backend=config.output_backend,
    )
    render_equity(fig_eq, eq_source, theme, config,
                  peak_val, final_val, peak_ts, final_ts,
                  max_dd_ts, max_dd_val, max_dd_pct, baseline,
                  trailing_source=trailing_source)
    all_figs.append(fig_eq)
    return fig_eq, eq_source, baseline


def _build_drawdown_panel(config, theme, stats, all_figs):
    if not config.plot_drawdown or stats is None:
        return None
    dd_source = build_drawdown_source(stats.equity_curve)
    fig_dd = _new_fig(
        config.drawdown_height, config.width, theme,
        name="drawdown", output_backend=config.output_backend,
    )
    render_drawdown(fig_dd, dd_source, theme)
    all_figs.append(fig_dd)
    return fig_dd


def _build_pl_panel(config, trades_source, theme, all_figs):
    if not config.plot_pl or trades_source is None:
        return None
    fig_pl = _new_fig(
        config.pl_height, config.width, theme,
        name="pl", output_backend=config.output_backend,
    )
    render_pl_bars(fig_pl, trades_source, theme)
    all_figs.append(fig_pl)
    return fig_pl


def _build_indicator_panels(indicators: list, config: PlotConfig, x_range, theme: dict,
                            interval_ms: int, label_sink: list = None,
                            lazy_zoom_range: int = None) -> list:
    from tradetropy.plotting.render._tools import render_tool_groups

    ind_figs = []
    for meta in indicators:
        if meta.overlay:
            continue
        has_prims = bool(getattr(meta, "_draw_primitives", None))
        is_geom = _is_geometric_renderer(meta.renderer)
        # Own-panel indicator that is geometry-only but emitted no primitives:
        # nothing to draw (e.g. detector with no events in the window).
        if is_geom and not has_prims:
            continue
        fig_ind = _new_fig(
            meta.panel_height or config.indicator_height, config.width, theme,
            x_range=x_range,
            name=f"ind_{meta.name if isinstance(meta.name, str) else meta.name[0]}",
            output_backend=config.output_backend,
        )
        # Geometry emitted via Indicator.draw() (CVD candles, delta bars, the
        # bid/ask/delta/total footprint cells). Same generic renderer used by
        # overlays, now routed into the indicator's own panel.
        if has_prims:
            prim_registry = render_tool_groups(
                fig_ind, meta._draw_primitives, theme=theme, interval_ms=interval_ms,
                show_legend=meta.show_legend, label_sink=label_sink,
                lazy_x_range=x_range, lazy_zoom_range=lazy_zoom_range,
            )
            # Autoscale a geometric panel from its primitive sources (quads /
            # segments), since it has no value series to drive the standard hook.
            if meta.autoscale and x_range is not None:
                prim_sources = list(prim_registry.values())
                if prim_sources:
                    _configure_autoscale_primitives(fig_ind, prim_sources, x_range)
        # Per-bar series only for real series renderers. Geometric renderers are
        # fully expressed by the draw() primitives above.
        if not is_geom:
            srcs = build_indicator_source(meta)
            render_indicator(fig_ind, srcs, meta, interval_ms=interval_ms)
            if meta.autoscale:
                bokeh_srcs = [s for s in srcs if len(s.data.get("ts", [])) > 0]
                if bokeh_srcs:
                    _configure_autoscale_indicator(fig_ind, bokeh_srcs, x_range)
        if meta.reference_lines:
            render_reference_lines(fig_ind, meta)
        if meta.panel_title:
            panel_label = meta.panel_title
        elif isinstance(meta.name, str):
            panel_label = meta.name
        elif isinstance(meta.name, list) and meta.name:
            panel_label = meta.name[0]
        else:
            panel_label = ""
        fig_ind.yaxis.axis_label = panel_label
        ind_figs.append(fig_ind)
    return ind_figs


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def plot(bt, config: PlotConfig | None = None, **kwargs) -> None:
    """
    Generate the interactive backtest results chart.

    Parameters
    ----------
    bt     : BacktestEngine post-run (after calling bt.run())
    config : PlotConfig with display options. If passed along with
             kwargs, kwargs override the config values.
    **kwargs : any PlotConfig field as keyword argument, alternative
               to constructing the PlotConfig object manually.

    Example
    -------
        from tradetropy.plotting import plot

        plot(bt)
        plot(bt, theme="dark", plot_trades=False)
        plot(bt, PlotConfig(theme="dark"))          # old style, still works
        plot(bt, output="file", filename="results.html")
    """
    _ensure_bokeh()

    if config is None:
        config = PlotConfig(**kwargs) if kwargs else PlotConfig()
    elif kwargs:
        # Explicit config + point overrides via kwargs
        import dataclasses
        base = dataclasses.asdict(config)
        base.update(kwargs)
        config = PlotConfig(**base)

    theme = _theme(config)

    # ── 1. Gather data ──────────────────────────────────────────────────
    data = _gather_plot_data(bt)
    ohlc_array   = data["ohlc_array"]
    interval_ms  = data["interval_ms"]
    stats        = data["stats"]
    indicators   = data["indicators"]
    fp_proxies   = data["fp_proxies"]
    symbol       = data["symbol"]
    price_tick   = data["price_tick"]
    price_digits = data["price_digits"]

    # ── 1b. Sync OHLC interval with footprint (if applicable) ───────────
    fp_active = [
        fp for fp in fp_proxies
        if fp._store is not None or fp._ring is not None
    ]

    if config.plot_footprint and fp_active:
        fp_interval = fp_active[0].interval_ms

        if fp_interval != interval_ms:
            op_fp = next(
                (
                    op
                    for op in bt.strategy._ohlc_proxies
                    if op.interval_ms == fp_interval
                    and op._ohlc_store is not None
                ),
                None,
            )

            if op_fp is not None:
                from tradetropy.core.constants import N_OHLC_COLS
                ohlc_array = op_fp._ohlc_store.matrix[:, :N_OHLC_COLS].copy()
                interval_ms = fp_interval
            else:
                if fp_interval > interval_ms:
                    ohlc_array, interval_ms = _resample_ohlc(
                        ohlc_array, interval_ms, fp_interval
                    )
                else:
                    import warnings as _w
                    _w.warn(
                        f"plot_footprint=True requires candles of {fp_interval} ms, "
                        f"but the only available OhlcProxy is {interval_ms} ms and "
                        f"it is not possible to disaggregate candles. Add "
                        f"self.subscribe_ohlc(symbol, timeframe={fp_interval}) "
                        f"in your strategy so that OHLC matches the footprint.",
                        stacklevel=2,
                    )

    elif config.resample_timeframe is not None:
        ohlc_array, interval_ms = _resample_ohlc(
            ohlc_array, interval_ms, config.resample_timeframe
        )

    if len(ohlc_array) > config.max_candles:
        ohlc_array = ohlc_array[-config.max_candles:]

    # ── Align indicator draw() primitives to the candle grid ─────────────
    # interval_ms and ohlc_array are now finalized (after footprint/resample),
    # so the grid phase matches what is drawn. Per-bar series already run on
    # the OHLC grid; this only repositions the draw() geometry (bubbles,
    # histograms, labels) to the start of their candle. use_tool() snapshots
    # do not go through here (anchored to explicit ranges).
    if config.align_indicators_to_candle and interval_ms and len(ohlc_array) > 0:
        _candle_origin_ind = int(ohlc_array[0, 0])
        for _meta in indicators:
            _prims = getattr(_meta, "_draw_primitives", None)
            if _prims:
                _snap_primitive_groups(_prims, interval_ms, _candle_origin_ind)

    # ── 2. Build base sources ────────────────────────────────────────────
    source_ohlc = build_ohlc_source(ohlc_array, interval_ms)

    _fp_active_with_data = [
        fp for fp in fp_proxies
        if fp._store is not None and fp._store._n_closed > 0
    ]
    if _fp_active_with_data:
        _fp_cols = _build_footprint_cols(_fp_active_with_data[0], len(ohlc_array))
        for _k, _v in _fp_cols.items():
            source_ohlc.data[_k] = _v

    trades_source = None
    align_trades = config.align_trades_to_candle
    if config.plot_trades and stats is not None and not stats.trades.empty:
        _candle_origin = int(ohlc_array[0, 0]) if len(ohlc_array) > 0 else 0
        trades_source = build_trades_source(
            stats.trades, interval_ms, align_trades, candle_origin_ms=_candle_origin
        )

    # ── 3. Create figures ────────────────────────────────────────────────
    all_figs = []

    fig_eq, eq_source, baseline = _build_equity_panel(config, theme, stats, all_figs)
    fig_dd = _build_drawdown_panel(config, theme, stats, all_figs)
    fig_pl = _build_pl_panel(config, trades_source, theme, all_figs)

    # ── Main OHLC panel ─────────────────────────────────────────────────
    fig_ohlc, x_range = _new_ohlc_fig(config.ohlc_height, config.width, theme, output_backend=config.output_backend)
    for f in all_figs:
        f.x_range = x_range

    if fig_pl is not None:
        fig_pl.x_range = x_range

    # Autoscale equity now that x_range is available
    if config.equity_mode != "none" and stats is not None:
        _configure_autoscale_equity(fig_eq, eq_source, x_range, baseline)

    _price_lo = float(np.nanmin(ohlc_array[:, 3]))
    _price_hi = float(np.nanmax(ohlc_array[:, 2]))
    _price_pad = (_price_hi - _price_lo) * 0.05
    fig_ohlc.y_range.start = _price_lo - _price_pad
    fig_ohlc.y_range.end   = _price_hi + _price_pad

    render_ohlc(fig_ohlc, source_ohlc, theme, config, interval_ms, symbol, price_digits)

    if config.plot_volume:
        render_volume(fig_ohlc, source_ohlc, theme, config, interval_ms)
        if "vol" in fig_ohlc.extra_y_ranges:
            _configure_autoscale_volume(
                fig_ohlc, source_ohlc, fig_ohlc.extra_y_ranges["vol"]
            )

    if config.plot_trades and trades_source is not None:
        render_trades(fig_ohlc, trades_source, theme, interval_ms)

    # Collected text labels (indicator/tool/panel) for the global lazy-labels
    # callback that hides them when the chart is zoomed too far out.
    lazy_label_entries: list = []
    _lazy_zoom = config.labels_zoom_range

    _heatmap_autoscale_sources = _render_overlay_indicators(
        fig_ohlc, indicators, interval_ms, theme,
        label_sink=lazy_label_entries, lazy_x_range=x_range,
        lazy_zoom_range=_lazy_zoom,
    )

    _render_tool_snapshots(
        fig_ohlc, bt, interval_ms, theme,
        label_sink=lazy_label_entries, lazy_x_range=x_range,
        lazy_zoom_range=_lazy_zoom,
    )

    # ── Footprint ────────────────────────────────────────────────────────
    fp_tick_size = 0.0
    fp_renderers = []
    _fp_lazy_cbs = []   # lazy CustomJS callbacks -- need dummy injected later
    if config.plot_footprint:
        fp_activos_con_datos = [
            fp for fp in fp_proxies
            if fp._store is not None and fp._store._n_closed > 0
        ]
        for fp_proxy in fp_activos_con_datos:
            fp_interval = fp_proxy.interval_ms
            fp_src = build_footprint_source(fp_proxy, ohlc_array, fp_interval, theme)
            if fp_src is not None:
                fp_ret = render_footprint(fig_ohlc, fp_src, fp_interval, config.footprint_zoom_range)
                if fp_ret:
                    fp_tick_size = float(fp_proxy.config.tick_size)
                    fp_renderers.extend([fp_ret[2], fp_ret[3], fp_ret[4], fp_ret[5]])
                    _fp_lazy_cbs.append(fp_ret[6])  # CustomJS lazy callback

        if fp_renderers:
            from bokeh.models import ColumnDataSource, CustomJS
            dummy_src = ColumnDataSource(dict(x=[], y=[]))
            dummy = fig_ohlc.scatter(
                x="x", y="y",
                source=dummy_src,
                size=0,
                alpha=0,
                color=theme["fp_ask_scale"][-1][1],
                legend_label="Footprint",
            )
            # Inject dummy and ohlc_source into the lazy callback so it respects
            # the legend state and adjusts candle width when the range moves.
            for _lazy_cb in _fp_lazy_cbs:
                _lazy_cb.args["dummy"] = dummy
                _lazy_cb.args["ohlc_source"] = source_ohlc

            # On legend click, propagate visibility to all renderers and adjust
            # candle width (thin with FP, wide without FP).
            _bar_width_wide   = int(interval_ms * 0.9)
            _bar_width_narrow = int(interval_ms * 0.3)
            cb = CustomJS(
                args=dict(
                    dummy=dummy,
                    fp_renderers=fp_renderers,
                    ohlc_source=source_ohlc,
                    bar_width_wide=_bar_width_wide,
                    bar_width_narrow=_bar_width_narrow,
                ),
                code=(_JS_DIR / "legend_toggle.js").read_text(),
            )
            dummy.js_on_change("visible", cb)

    _configure_autoscale_ohlc(
        fig_ohlc, source_ohlc, theme, fp_tick_size,
        heatmap_sources=_heatmap_autoscale_sources,
    )
    if fp_tick_size > 0:
        _configure_fp_yaxis(fig_ohlc, fp_tick_size)
    all_figs.append(fig_ohlc)

    ind_figs = _build_indicator_panels(
        indicators, config, x_range, theme, interval_ms,
        label_sink=lazy_label_entries, lazy_zoom_range=_lazy_zoom,
    )
    all_figs.extend(ind_figs)

    # Global lazy labels: hide every collected indicator/tool/panel text label
    # when zoomed past labels_zoom_range candles (internal + external alike).
    _initial_zoomed_in = len(ohlc_array) <= _lazy_zoom
    _configure_lazy_labels(
        x_range, lazy_label_entries, interval_ms, _lazy_zoom, _initial_zoomed_in
    )

    # ── 4. Global configurations ─────────────────────────────────────────
    _configure_legends(all_figs, theme)
    _configure_margins(all_figs)
    _configure_crosshair(all_figs, theme, tick_size=(price_tick or fp_tick_size),
                         show_price_tag=config.show_price_tag, ohlc_fig=fig_ohlc)

    # ── 5. Stats bar ─────────────────────────────────────────────────────
    if config.plot_stats:
        stats_div = _build_stats_div(stats, theme)
        all_figs.insert(0, stats_div)

    # ── 6. Assemble and display ──────────────────────────────────────────
    layout = _build_layout(all_figs)
    _show_layout(layout, config, theme)
    
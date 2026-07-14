from __future__ import annotations
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd

from .config import PlotConfig
from ._util import _AUTOSCALE_JS

_JS_DIR = pathlib.Path(__file__).parent / "js"


def _make_ylock():
    """
    Create the shared Y-lock state for a figure's autoscale.

    The state carries two single-element arrays:
        - 'locked'  : True while the user's manual Y zoom/pan suspends autoscale.
        - 'scaling' : True only while the autoscale itself writes y_range, so the
          watcher can tell autoscale writes apart from user gestures.

    Returns:
        ColumnDataSource: the lock state, passed to the autoscale CustomJS.
    """
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(data=dict(locked=[False], scaling=[False]))


def _wire_ylock(fig, ylock, y_range=None) -> None:
    """
    Wire the Y-lock watcher and the Reset handler onto a figure.

    A user gesture on the Y axis (ywheel_zoom / ypan / box_zoom) changes
    ``y_range`` while ``ylock.scaling`` is False, so the watcher sets
    ``locked=True`` and the autoscale suspends. The ResetTool clears the lock.

    Args:
        fig: Target Bokeh figure (its ``reset`` event clears the lock).
        ylock (ColumnDataSource): state from :func:`_make_ylock`.
        y_range: Range to watch (defaults to ``fig.y_range``).
    """
    from bokeh.models import CustomJS

    yr = y_range if y_range is not None else fig.y_range
    watch = CustomJS(
        args=dict(ylock=ylock),
        code=(_JS_DIR / "ylock_watch.js").read_text(),
    )
    yr.js_on_change("start", watch)
    yr.js_on_change("end", watch)

    reset = CustomJS(
        args=dict(ylock=ylock),
        code=(_JS_DIR / "ylock_reset.js").read_text(),
    )
    fig.js_on_event("reset", reset)


def _configure_autoscale_ohlc(fig, source, theme: dict, fp_tick_size: float = 0, fp_source_bid=None, follow_state=None, heatmap_sources=None):
    """
    Configures the Y-axis autoscale for the OHLC panel.

    Backtest mode (follow_state=None)
    ----------------------------------
    Registers a single CustomJS on x_range.start/end. The Y-axis rescales
    when the user pans/zooms. Returns None.

    Live mode (follow_state provided)
    ---------------------------------
    Autoscale is the ONLY authority on y_range (Python navigation no longer
    writes y_range). Two triggers are registered that share the SAME padding
    formula, so they produce identical ranges and there is no "double
    adjustment":

      - x_range.start/end  -- always rescales (follows the candle in follow
        mode and respects the user's pan when follow is off).
      - source "data"      -- rescales ONLY when follow is active
        (`follow_state.data['follow'][0]`). Covers the forming candle's ticks
        without rescaling when the user has panned (follow off).

    Returns the list of CustomJS [cb_xrange, cb_source] so that
    _configure_fp_yaxis can inject the footprint y_ticker into both.

    If fp_source_bid is passed (live mode with footprint), the JS reads
    tick_size directly from the source at runtime (cell_h[0]) instead of the
    static argument. This ensures correctness even before the ring has
    inferred its tick_size.
    """
    from bokeh.models import CustomJS

    # The ylock was meant to let a manual Y zoom/pan (box_zoom) persist by
    # suspending the autoscale until Reset. It is disabled here (in BOTH
    # backtest and live/replay) because it is fundamentally racy: BokehJS
    # propagates range-change callbacks ASYNCHRONOUSLY, so the autoscale's own
    # y_range write reaches ylock_watch AFTER the callback has reset the
    # `scaling` guard to false. That latches locked=true on the very first
    # wheel/pan/tick, after which every autoscale call returns early and the Y
    # axis freezes:
    #   - backtest: candles look flattened on the next pan (nothing re-runs the
    #     autoscale, so the lock never clears without Reset);
    #   - live/replay: the view keeps following the candle on X but the Y axis
    #     stops repositioning (the follow data-trigger also respects the lock).
    # With ylock left as None, the `typeof ylock === "undefined"` guards in
    # autoscale_ohlc.js / autoscale_ohlc_live.js skip all lock/scaling logic, so
    # the autoscale runs unconditionally on every x-range (and, in live-follow,
    # data) change and always reframes Y to the visible candles. Tradeoff: a
    # manual Y box-zoom no longer persists across a later pan / next tick -
    # acceptable, and what "autoscale just works" implies for both charts.
    ylock = None
    hm_sources = list(heatmap_sources) if heatmap_sources else []

    if fp_source_bid is not None:
        body = (_JS_DIR / "autoscale_ohlc_live.js").read_text()
        args = dict(
            source=source,
            y_range=fig.y_range,
            x_range=fig.x_range,
            pad_factor=0.05,
            fp_source_bid=fp_source_bid,
            y_ticker=None,   # injected in _configure_fp_yaxis
            hm_sources=hm_sources,
        )
    else:
        body = _AUTOSCALE_JS
        args = dict(
            source=source,
            y_range=fig.y_range,
            x_range=fig.x_range,
            mode="ohlc",
            pad_factor=0.05,
            fp_tick_size=fp_tick_size,
            hm_sources=hm_sources,
        )
    if ylock is not None:
        args["ylock"] = ylock

    cb = CustomJS(args=dict(args), code=body)
    fig.x_range.js_on_change("start", cb)
    fig.x_range.js_on_change("end", cb)

    if ylock is not None:
        # Live only: suspend the autoscale on a manual Y gesture; Reset re-enables.
        _wire_ylock(fig, ylock)

    if follow_state is None:
        # Backtest: the OHLC x_range is forced to an explicit Range1d in
        # plotting.py::plot (a DataRange1d does not emit reliable start/end
        # change events, so the callback would never fire). With a Range1d the
        # js_on_change("start"/"end") wiring above fires deterministically on
        # every pan/zoom and, with no ylock, always reframes Y to the visible
        # candles.
        return None

    # Live: add data-change trigger, gated by the follow state.
    # Reuses the same autoscale body wrapped in a follow guard,
    # guaranteeing an identical formula to the x_range trigger.
    src_args = dict(args)
    src_args["follow_state"] = follow_state
    cb_source = CustomJS(
        args=src_args,
        code=(
            "if (follow_state.data['follow'] && follow_state.data['follow'][0]) {\n"
            + body
            + "\n}"
        ),
    )
    source.js_on_change("data", cb_source)
    return [cb, cb_source]


def _configure_fp_yaxis(fig, fp_tick_size: float, autoscale_cb=None) -> None:
    """
    Configures the Y axis of the OHLC panel to display ticks at exact
    multiples of the footprint tick_size.

    In backtest (fp_tick_size known): uses FixedTicker with the grid
    calculated from the current y_range.

    In live (autoscale_cb provided): injects the FixedTicker into the
    autoscale CustomJS so it updates dynamically on each zoom/pan.

    Parameters
    ----------
    fig           : Bokeh OHLC figure
    fp_tick_size  : footprint tick size (e.g. 0.25, 5, 10)
    autoscale_cb  : CustomJS returned by _configure_autoscale_ohlc in live mode
    """
    from bokeh.models import FixedTicker, CustomJS, CustomJSTickFormatter

    if fp_tick_size <= 0:
        return

    ticker = FixedTicker(ticks=[])

    # Determine number of decimals needed for the formatter
    # e.g. tick_size=0.25 -> 2 decimals; tick_size=5 -> 0 decimals
    if fp_tick_size >= 1:
        decimals = 0
    else:
        import math
        decimals = max(0, -int(math.floor(math.log10(fp_tick_size))))

    fmt_str = f"{{0:.{decimals}f}}"  # noqa — kept for clarity, not used directly
    formatter = CustomJSTickFormatter(
        args=dict(dec=decimals),
        code=f"""
            // Format tick to the number of decimals of tick_size
            return tick.toFixed(dec);
        """
    )

    for _ax in fig.yaxis:
        if _ax.y_range_name == "default":
            _ax.ticker    = ticker
            _ax.formatter = formatter
            break

    if autoscale_cb is not None:
        # Live mode: inject the ticker into the autoscale CustomJS(s)
        # so it updates on each zoom/pan and each data rescale.
        _cbs = autoscale_cb if isinstance(autoscale_cb, (list, tuple)) else [autoscale_cb]
        for _cb in _cbs:
            _cb.args["y_ticker"] = ticker
    else:
        # Backtest mode: populate the ticker with the current y_range ticks
        # and update via CustomJS when y_range changes
        _populate_ticker_js = CustomJS(
            args=dict(y_range=fig.y_range, ticker=ticker, ts=fp_tick_size),
            code=(_JS_DIR / "fp_yaxis_populate.js").read_text(),
        )
        fig.y_range.js_on_change("start", _populate_ticker_js)
        fig.y_range.js_on_change("end",   _populate_ticker_js)


def _configure_autoscale_volume(fig, source, vol_range) -> None:
    """
    Keep the volume bars anchored to the bottom band of the price pane.

    The volume lives on a secondary ``vol`` range mapped so its peak reaches
    ~20% of the frame height (bottom band, TradingView style). Bokeh's pan/zoom
    tools scale every range of the frame - including this extra range - which
    would detach the volume from the pane bottom when the user zooms/pans the Y
    axis. Re-anchoring on both x_range AND y_range changes restores the fixed
    bottom band after any navigation, so the volume never "floats".
    """
    from bokeh.models import CustomJS

    cb = CustomJS(
        args=dict(source=source, y_range=vol_range, x_range=fig.x_range),
        code=(_JS_DIR / "autoscale_volume.js").read_text(),
    )
    fig.x_range.js_on_change("start", cb)
    fig.x_range.js_on_change("end", cb)
    # Re-anchor when the price Y axis is panned/zoomed (the tools would otherwise
    # scale the vol range too, leaving the volume bars floating).
    fig.y_range.js_on_change("start", cb)
    fig.y_range.js_on_change("end", cb)


def _configure_autoscale_indicator(fig, sources: list, x_range) -> None:
    """
    Autoscale the Y axis of an indicator panel when the X range changes.
    Iterates over all indicator sources to find the visible min/max.
    """
    from bokeh.models import CustomJS

    ylock = _make_ylock()
    cb = CustomJS(
        args=dict(sources=sources, y_range=fig.y_range, x_range=x_range, ylock=ylock),
        code=(_JS_DIR / "autoscale_indicator.js").read_text(),
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)
    _wire_ylock(fig, ylock)


def _configure_autoscale_primitives(fig, sources: list, x_range) -> None:
    """
    Autoscale a geometric panel's Y axis from draw-primitive sources.

    Own-panel geometric indicators (CVD, DeltaBars) render through draw()
    primitives rather than a value series, so their ColumnDataSources carry quad
    (left/right/top/bottom) and segment (x0/x1/y0/y1) columns instead of a
    ts/value pair. This reads those columns to find the visible min/max as the X
    range changes.

    Args:
        fig: Target indicator panel figure.
        sources (list): Primitive ColumnDataSources (the render_tool_groups
            registry values for this panel).
        x_range: The shared X range whose start/end drive the rescale.
    """
    from bokeh.models import CustomJS

    ylock = _make_ylock()
    cb = CustomJS(
        args=dict(sources=sources, y_range=fig.y_range, x_range=x_range, ylock=ylock),
        code=(_JS_DIR / "autoscale_primitives.js").read_text(),
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)
    _wire_ylock(fig, ylock)


def _configure_lazy_labels(x_range, entries: list, interval_ms: int,
                           zoom_range: int, initial_zoomed_in: bool) -> None:
    """
    Hide all indicator text labels when the chart is zoomed too far out.

    Generalizes the footprint's lazy-label policy to every indicator/tool label
    (internal and external): when more than ``zoom_range`` candles are in view
    the labels would overlap into an unreadable smear, so they are hidden until
    the user zooms back in. A single CustomJS on the shared price ``x_range``
    governs all collected labels.

    Args:
        x_range: The shared price X range whose start/end drive the toggle.
        entries (list): ``(label_renderer, group_glyph)`` tuples; ``group_glyph``
            (or None) gates the label on its legend state.
        interval_ms (int): OHLC interval (to convert the range to candle count).
        zoom_range (int): Max candles in view to keep labels visible.
        initial_zoomed_in (bool): Whether the initial (full) view is within the
            threshold; sets the labels' initial visibility (the callback only
            fires on later range changes).
    """
    from bokeh.models import CustomJS

    if not entries:
        return

    labels = [lbl for lbl, _ in entries]
    groups = [grp for _, grp in entries]

    # Initial visibility: the x_range callback only fires on later changes.
    for lbl, grp in entries:
        group_on = True if grp is None else grp.visible
        lbl.visible = bool(initial_zoomed_in and group_on)

    cb = CustomJS(
        args=dict(labels=labels, groups=groups,
                  interval_ms=interval_ms, zoom_range=zoom_range),
        code=(_JS_DIR / "lazy_labels.js").read_text(),
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)


def _configure_autoscale_equity(fig, source, x_range, baseline: float) -> None:
    """
    Autoscale the equity panel when the X range changes.
    Respects the baseline (0 for return, initial_balance for balance).
    """
    from bokeh.models import CustomJS

    cb = CustomJS(
        args=dict(source=source, y_range=fig.y_range, x_range=x_range, baseline=baseline),
        code=(_JS_DIR / "autoscale_equity.js").read_text(),
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)


def _configure_legends(figs: list, theme: dict) -> None:
    for fig in figs:
        if not getattr(fig, "legend", None):
            continue
        try:
            leg = fig.legend[0]
        except (IndexError, TypeError):
            continue
        leg.visible              = True
        leg.location             = "top_left"
        leg.orientation          = "vertical"
        leg.click_policy         = "hide"
        leg.padding              = 3
        leg.spacing              = 0
        leg.margin               = 1
        leg.label_text_font_size = "8pt"
        leg.label_text_font      = "monospace"
        leg.label_text_color         = theme["text"]
        leg.background_fill_color    = theme["bg_border"]
        leg.background_fill_alpha    = 0.80
        leg.border_line_color        = theme["grid"]
        leg.border_line_alpha        = 0.5
        leg.border_line_width        = 1


def _configure_crosshair(
    figs: list, theme: dict, tick_size: float = 0.0, show_price_tag: bool = True,
    ohlc_fig=None,
) -> None:
    from bokeh.models import CrosshairTool, Span, CustomJS, Label

    color = theme["crosshair"]
    tag_bg = theme.get("price_tag_bg", color)
    tag_text = theme.get("price_tag_text", "#FFFFFF")

    # Decimals to display: derived from tick size (e.g. 0.01 -> 2, 0.5 -> 1).
    if tick_size and tick_size > 0:
        decs = max(0, -int(np.floor(np.log10(tick_size))))
        s = f"{tick_size:.10f}".rstrip("0")
        if "." in s:
            decs = max(decs, len(s.split(".")[1]))
    else:
        decs = 2

    # The price label must ALWAYS be anchored to the OHLC panel (its `y` lives
    # in price coordinates). The caller passes fig_ohlc explicitly; without it
    # we fall back to the first panel with a Y axis, which could be
    # equity/drawdown and would leave the label invisible over the price axis.
    vlines = []
    price_label = None

    for fig in figs:
        if not hasattr(fig, "add_tools"):
            continue
        fig.add_tools(CrosshairTool(
            dimensions="width",
            line_color=color,
            line_width=0.8,
            line_alpha=0.6,
        ))
        v = Span(
            location=0,
            dimension="height",
            line_color=color,
            line_width=0.8,
            line_dash="dashed",
            line_alpha=0.6,
        )
        fig.add_layout(v)
        vlines.append(v)

    if show_price_tag and ohlc_fig is None:
        ohlc_fig = next(
            (f for f in figs
             if hasattr(f, "y_range") and hasattr(f, "yaxis") and hasattr(f, "add_layout")),
            None,
        )

    if show_price_tag and ohlc_fig is not None:
        # The label is drawn INSIDE the data frame (not on the panel axis).
        # Anchoring it to the right panel caused Bokeh to recalculate the panel
        # width on mouse enter/leave, annoyingly rearranging the chart. As a
        # center renderer it does not touch the layout.
        # x in data coordinates = x_range.end (right edge of the frame);
        # text_align=right keeps the label tucked inside the axis.
        price_label = Label(
            x=0, y=0,
            x_units="data",
            y_units="data",
            text="",
            text_font_size="11px",
            text_font="monospace",
            text_color=tag_text,
            text_align="right",
            text_baseline="middle",
            background_fill_color=tag_bg,
            background_fill_alpha=1.0,
            border_line_color=tag_bg,
            border_line_alpha=1.0,
            border_line_width=1,
            x_offset=-2,
            y_offset=0,
            level="overlay",
        )
        ohlc_fig.add_layout(price_label)

    if price_label is not None:
        # The vertical lines update from any panel (shared x axis).
        vline_args = {f"v{i}": v for i, v in enumerate(vlines)}
        vline_code = "\n".join(f"v{i}.location = cb_obj.x;" for i in range(len(vlines)))
        vline_cb = CustomJS(args=vline_args, code=vline_code)
        for fig in figs:
            fig.js_on_event("mousemove", vline_cb)

        # The price label only updates from the OHLC figure: its `y` belongs
        # to the price coordinate system, not to the subpanels.
        # If tick_size is available, the price is snapped to the nearest tick multiple.
        price_cb = CustomJS(
            args={"label": price_label, "ohlc": ohlc_fig,
                  "tick": float(tick_size or 0.0), "decs": decs},
            code=(
                "let y = cb_obj.y;"
                "if (isFinite(y)) {"
                "  if (tick > 0) { y = Math.round(y / tick) * tick; }"
                "  label.x = ohlc.x_range.end;"
                "  label.y = y;"
                "  label.text = y.toFixed(decs);"
                "  label.visible = true;"
                "} else {"
                "  label.visible = false;"
                "}"
            ),
        )
        ohlc_fig.js_on_event("mousemove", price_cb)

        hide_cb = CustomJS(
            args={"label": price_label},
            code="label.visible = false;",
        )
        ohlc_fig.js_on_event("mouseleave", hide_cb)
    else:
        args = {f"v{i}": v for i, v in enumerate(vlines)}
        code = "\n".join(f"v{i}.location = cb_obj.x;" for i in range(len(vlines)))
        callback = CustomJS(args=args, code=code)
        for fig in figs:
            fig.js_on_event("mousemove", callback)


def _configure_margins(figs: list) -> None:
    for fig in figs:
        if not hasattr(fig, "min_border_left"):
            continue
        fig.min_border_left   = 70
        fig.min_border_right  = 70
        fig.min_border_top    = 8
        fig.min_border_bottom = 8


def _build_layout(figs: list):
    from bokeh.layouts import gridplot

    return gridplot(
        [[f] for f in figs],
        toolbar_options=dict(logo=None),
        toolbar_location="right",
        merge_tools=True,
        sizing_mode="stretch_width",
    ) if len(figs) > 1 else figs[0]


def _build_theme_css_div(theme: dict):
    """
    Returns an invisible Div that injects global CSS to style:
      - <body> and .bk-root with the theme background color.
      - The Bokeh toolbar (background, buttons, dividers) with theme colors.

    Needed because Bokeh does not expose these elements as configurable
    properties -- their color comes from the browser CSS or Bokeh.js.
    """
    from bokeh.models import Div

    bg      = theme["bg"]
    bg_brd  = theme["bg_border"]
    text    = theme["text"]
    axis    = theme["axis"]
    grid    = theme["grid"]
    is_dark = bg == "#0E1117"
    icon_filter        = "invert(1) opacity(0.7)" if is_dark else "none"
    icon_filter_hover  = "invert(1) opacity(1.0)" if is_dark else "none"

    css = f"""
    <style>
    html, body {{
        background-color: {bg} !important;
        margin: 0;
        padding: 0;
    }}
    .bk-root {{
        background-color: {bg} !important;
    }}
    /* -- Tooltip -- selectors for Bokeh 2.x / 3.x ----------------------- */
    /* The tooltip renders as a floating element directly in <body>          */
    .bk-tooltip,
    div.bk-tooltip,
    .bk-root .bk-tooltip,
    .bk-context-menu {{
        background-color: {bg_brd} !important;
        color: {text} !important;
        border: 1px solid {grid} !important;
        border-radius: 4px !important;
        font-family: monospace !important;
        font-size: 11px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.45) !important;
    }}
    .bk-tooltip > div:not(:first-child) {{
        border-top: 1px solid {grid} !important;
    }}
    .bk-tooltip th,
    .bk-tooltip-row-label {{
        color: {axis} !important;
        background-color: transparent !important;
        padding-right: 8px !important;
        font-weight: normal !important;
    }}
    .bk-tooltip td,
    .bk-tooltip-row-value {{
        color: {text} !important;
        background-color: transparent !important;
    }}
    /* Tooltip arrow */
    .bk-tooltip-arrow,
    .bk-tooltip::before,
    .bk-tooltip::after {{
        border-top-color: {grid} !important;
        border-bottom-color: {grid} !important;
    }}
    /* Alternating row background (some versions apply it) */
    .bk-tooltip tr:nth-child(even) td,
    .bk-tooltip tr:nth-child(even) th {{
        background-color: transparent !important;
    }}
    /* -- Toolbar container ------------------------------------------------ */
    .bk-toolbar {{
        background-color: {bg_brd} !important;
        border: 1px solid {grid} !important;
    }}
    .bk-toolbar.bk-right {{
        border-left: 1px solid {grid} !important;
    }}
    /* -- Individual buttons ----------------------------------------------- */
    .bk-toolbar-button {{
        background-color: {bg_brd} !important;
        color: {axis} !important;
    }}
    .bk-toolbar-button:hover {{
        background-color: {grid} !important;
        color: {text} !important;
    }}
    .bk-toolbar-button.bk-active {{
        background-color: {grid} !important;
        color: {text} !important;
    }}
    /* -- SVG icons inside buttons ----------------------------------------- */
    .bk-toolbar-button .bk-tool-icon {{
        filter: {icon_filter};
    }}
    .bk-toolbar-button:hover .bk-tool-icon {{
        filter: {icon_filter_hover};
    }}
    /* -- Divider between tool groups -------------------------------------- */
    .bk-toolbar-divider {{
        background-color: {grid} !important;
    }}
    /* -- Outer panel of gridplot ------------------------------------------ */
    .bk-grid-toolbar {{
        background-color: {bg_brd} !important;
    }}
    </style>
    """
    # visible=False makes Bokeh NOT render the element in the DOM,
    # so the <style> never reaches the browser. Instead we use a
    # Div with height=0 and overflow hidden so the CSS IS injected.
    return Div(
        text=css,
        sizing_mode="stretch_width",
        height=0,
        styles={"overflow": "hidden", "padding": "0", "margin": "0", "height": "0px"},
    )


def _show_layout(layout, config: PlotConfig, theme: dict) -> None:
    from bokeh.plotting import show, output_notebook, output_file
    from bokeh.io import reset_output
    from bokeh.layouts import column

    reset_output()

    # Inject global theme CSS (body, toolbar) as an invisible Div
    css_div = _build_theme_css_div(theme)
    full_layout = column(
        css_div, layout,
        sizing_mode="stretch_width",
        styles={"background-color": theme["bg"], "padding": "0", "margin": "0"},
    )

    if config.output == "notebook":
        output_notebook(hide_banner=True)
        show(full_layout)
    elif config.output == "file":
        output_file(config.filename, title="Backtest")
        show(full_layout)
    elif config.output == "server":
        _serve_layout(full_layout, config)
    else:
        show(full_layout)


def _serve_layout(full_layout, config: PlotConfig) -> None:
    """
    Starts a blocking Bokeh Server that serves the backtest layout.

    Useful in headless environments or without an integrated browser
    (Termux, SSH, WSL). The server does not open the browser automatically;
    it prints the URL in the terminal so the user can open it manually
    from any device on the same network.

    The server runs until the user presses Ctrl+C (KeyboardInterrupt),
    which is caught cleanly to avoid propagating the traceback.

    Args:
        full_layout : already assembled Bokeh layout (column with CSS div + figures).
        config      : PlotConfig with server_port and optionally open_browser.
    """
    import asyncio
    import socket

    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from bokeh.server.server import Server
    from tornado.ioloop import IOLoop

    port = config.server_port

    # Get the local IP of the active interface to display the network URL.
    try:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _sock.connect(("8.8.8.8", 80))
        local_ip = _sock.getsockname()[0]
        _sock.close()
    except Exception:
        local_ip = "127.0.0.1"

    def make_doc(doc):
        # Each session receives a copy of the layout (Bokeh requires new roots
        # per document; full_layout is already built so it is reused directly --
        # one active session at a time is the normal case for static backtest).
        from bokeh.layouts import column as bk_column
        # Clone the children of full_layout so each session has its own roots
        # and Bokeh does not complain about shared roots.
        doc.add_root(full_layout)
        doc.title = "Backtest"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    io_loop = IOLoop.current()

    app = Application(FunctionHandler(make_doc))
    server = Server(
        {"/": app},
        port=port,
        io_loop=io_loop,
        allow_websocket_origin=[
            f"localhost:{port}",
            f"127.0.0.1:{port}",
            f"{local_ip}:{port}",
        ],
        num_procs=1,
        session_token_expiration=3_600_000,  # 1 h -- long session for review
    )
    server.start()

    print(
        f"\n  Backtest plot ready at:\n"
        f"    Local   -> http://localhost:{port}\n"
        f"    Network -> http://{local_ip}:{port}\n"
        f"\n  Press Ctrl+C to stop the server.\n"
    )

    try:
        io_loop.start()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    finally:
        server.stop()
        io_loop.close(all_fds=True)


def _build_stats_div(stats, theme: dict) -> object:
    from bokeh.models import Div

    cells = []

    if stats is not None:
        key_labels = OrderedDict([
            # ("Start",                    "Start"),
            # ("End",                      "End"),
            ("Duration",                  "Dur"),
            # ("Exposure Time [%]",         "Exp %"),
            ("Equity Final [$]",         "Eq Final"),
            # ("Equity Peak [$]",          "Eq Peak"),
            ("Return [%]",              "Return"),
            # ("Return (Ann.) [%]",       "Ret (Ann)"),
            # ("Volatility (Ann.) [%]",   "Vol (Ann)"),
            ("Sharpe Ratio",            "Sharpe"),
            ("Sortino Ratio",           "Sortino"),
            ("Calmar Ratio",           "Calmar"),
            # ("Max. Drawdown [%]",       "Max DD"),
            ("Avg. Drawdown [%]",       "Avg DD"),
            # ("Max. Drawdown Duration",  "Max DD Dur"),
            # ("Avg. Drawdown Duration","Avg DD Dur"),
            ("# Trades",             "Trades"),
            ("# Trades Long",        "Long"),
            ("# Trades Short",       "Short"),
            ("Win Rate [%]",          "Win Rate"),
            # ("Best Trade [%]",        "Best"),
            # ("Worst Trade [%]",      "Worst"),
            ("Avg. Trade [%]",       "Avg"),
            # ("Max. Trade Duration",   "Max T Dur"),
            # ("Avg. Trade Duration",  "Avg T Dur"),
            ("Profit Factor",         "PF"),
            ("Expectancy [%]",       "Exp"),
            ("Total Commissions [$]", "Comm"),
            # ("SQN",                 "SQN"),
        ])
        for key, label in key_labels.items():
            val = stats.get(key, None)
            if val is not None and val != "":
                if isinstance(val, float):
                    if pd.isna(val):
                        val_str = "—"
                    elif key == "Total Commissions [$]":
                        sign = "-" if val < 0 else ""
                        val_str = f"{sign}${abs(val):,.2f}"
                    elif "Return" in key or key in (
                        "Max. Drawdown [%]", "Avg. Drawdown [%]", "Volatility (Ann.) [%]", "Expectancy [%]"
                    ):
                        sign = "+" if val >= 0 else ""
                        val_str = f"{sign}{val:.2f}%"
                    else:
                        val_str = f"{val:,.0f}" if abs(val) >= 1 else f"{val:.2f}"
                elif isinstance(val, pd.Timestamp):
                    val_str = val.strftime("%Y-%m-%d %H:%M")
                else:
                    val_str = str(val)
                cells.append(f'<span class="stat-label">{label}</span><span class="stat-val">{val_str}</span>')

    bg = theme["bg"]
    text = theme["text"]
    border = theme.get("grid_line", bg)

    html = f"""
    <style>
    .stats-bar {{
        display: flex;
        flex-wrap: wrap;
        gap: 0px;
        padding: 0 10px;
        background: {bg};
        border-bottom: 1px solid {border};
        font-family: monospace;
        font-size: 10px;
        line-height: 1.6;
    }}
    .stat-label {{
        color: {text};
        margin-right: 3px;
        opacity: 0.6;
    }}
    .stat-val {{
        color: {text};
        margin-right: 14px;
    }}
    </style>
    <div class="stats-bar">{"".join(cells)}</div>
    """

    return Div(text=html, sizing_mode="stretch_width", styles={"min-height": "30px"})
    
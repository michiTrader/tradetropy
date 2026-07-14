from __future__ import annotations


# FIGURE FACTORY WITH BASE STYLE


def _apply_style(fig, theme: dict) -> None:
    from bokeh.models import NumeralTickFormatter

    bg  = theme["bg"]
    brd = theme["bg_border"]
    grd = theme["grid"]
    txt = theme["text"]
    ax  = theme["axis"]

    fig.background_fill_color = bg
    fig.border_fill_color     = brd
    fig.outline_line_color    = brd

    for axis in (fig.xaxis, fig.yaxis):
        axis.axis_line_color        = brd
        axis.major_tick_line_color  = ax
        axis.minor_tick_line_color  = None
        axis.major_label_text_color = txt
        axis.major_label_text_font  = "monospace"
        axis.major_label_text_font_size = "10px"

    fig.xgrid.grid_line_color  = grd
    fig.ygrid.grid_line_color  = grd
    fig.xgrid.grid_line_alpha  = 0.5
    fig.ygrid.grid_line_alpha  = 0.5

    fig.yaxis.formatter = NumeralTickFormatter(format="0,0.[00]")


def _new_fig(
    height: int,
    width: int,
    theme: dict,
    x_range=None,
    hide_x_axis: bool = True,
    y_range=None,
    y_axis_location: str = "left",
    output_backend: str = "canvas",
    **kwargs,
):
    from bokeh.plotting import figure
    from bokeh.models import DataRange1d, WheelZoomTool

    # All panels (equity, drawdown, P&L, indicator studies) share the OHLC
    # x_range, so their scroll wheel must zoom time-only (dimensions="width")
    # exactly like _new_ohlc_fig. Two reasons:
    #  1. gridplot(merge_tools=True) fuses every panel's toolbar into ONE shared
    #     toolbar with ONE active_scroll. If some panels used the mixed
    #     wheel_zoom (dimensions="both") and the OHLC used width-only, Bokeh
    #     warned about "competing values for toolbar.active_scroll" and picked
    #     one arbitrarily. When the both-axes tool won, scrolling over the price
    #     panel moved the Y axis as a user gesture, which the ylock watcher read
    #     as manual and SUSPENDED the OHLC autoscale - leaving the candles
    #     frozen/flattened on the next pan. Keeping every panel width-only makes
    #     the merged active_scroll unambiguous, so scrolling never touches Y and
    #     the autoscale keeps driving it.
    #  2. It matches trading-chart UX: the wheel zooms time everywhere; each
    #     panel reframes its own Y (OHLC/volume/equity autoscale on x change).
    # A real WheelZoomTool instance (not the "wheel_zoom" string) is used so the
    # dimensions restriction survives the toolbar merge. box_zoom/pan stay free
    # on both axes for manual Y framing; ResetTool restores autoscale.
    panel_wheel_zoom = WheelZoomTool(dimensions="width")

    kw = dict(
        x_axis_type="datetime",
        tools=[
            "xpan", "pan", "xwheel_zoom", "xwheel_pan",
            "box_zoom", "undo", "redo", "reset", "save",
        ],
        width=width,
        height=height,
        y_axis_location=y_axis_location,
        background_fill_color=theme["bg"],
        border_fill_color=theme["bg_border"],
        active_drag="xpan",
        active_scroll=panel_wheel_zoom,
        y_range=y_range or DataRange1d(range_padding=0.05),
        output_backend=output_backend,
    )
    if x_range is not None:
        kw["x_range"] = x_range
    kw.update(kwargs)

    fig = figure(**kw)
    fig.add_tools(panel_wheel_zoom)
    _apply_style(fig, theme)

    if hide_x_axis:
        fig.xaxis.visible = False

    return fig


def _new_ohlc_fig(height: int, width: int, theme: dict, y_range=None, x_range=None, output_backend: str = "canvas") -> tuple:
    from bokeh.plotting import figure
    from bokeh.models import Range1d, DataRange1d, WheelZoomTool

    # Unlike _new_fig's panels, the price axis drives the OHLC autoscale
    # (_configure_autoscale_ohlc): it rewrites y_range on every x_range change
    # so the visible candles are always framed. The scroll wheel tool here is
    # deliberately time-only (dimensions="width"), NOT the mixed wheel_zoom
    # kept in _new_fig's panels - scrolling over price would otherwise move
    # y_range as a user gesture, which the ylock watcher (see
    # _layout.py::_wire_ylock) reads as "manual" and suspends the autoscale
    # until Reset is pressed. A real WheelZoomTool instance (rather than the
    # "wheel_zoom" string) is used so the restriction holds even after
    # gridplot(merge_tools=True) fuses this figure's toolbar with panels that
    # keep the free two-axis tool - merging active_scroll strings across
    # figures with different values is ambiguous (Bokeh just picks the last
    # one), but each figure's own tool instance keeps its own `dimensions`.
    # xwheel_zoom is kept as a second explicit tool for parity with the
    # toolbar entry name; box_zoom/pan remain free on both axes for anyone who
    # wants to frame Y manually, and ResetTool always clears the lock.
    price_wheel_zoom = WheelZoomTool(dimensions="width")

    fig = figure(
        x_axis_type="datetime",
        tools=[
            "xpan", "ypan", "pan", "xwheel_zoom", "xwheel_pan",
            "box_zoom", "undo", "redo", "reset", "save",
        ],
        width=width,
        height=height,
        name="ohlc",
        y_axis_location="right",
        background_fill_color=theme["bg"],
        border_fill_color=theme["bg_border"],
        active_drag="xpan",
        active_scroll=price_wheel_zoom,
        y_range=y_range or Range1d(start=0, end=1),
        x_range=x_range or DataRange1d(),
        output_backend=output_backend,
    )
    fig.add_tools(price_wheel_zoom)
    _apply_style(fig, theme)

    fig.xaxis.visible = True
    fig.xaxis.major_label_text_color = theme["text"]
    fig.xaxis.major_label_text_font  = "monospace"
    fig.xaxis.major_label_text_font_size = "10px"

    return fig, fig.x_range
    
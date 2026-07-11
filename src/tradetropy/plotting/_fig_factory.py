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
    from bokeh.models import DataRange1d

    kw = dict(
        x_axis_type="datetime",
        # wheel_zoom has dimensions="both": scrolling over the price (Y) axis
        # scales Y, over the time (X) axis scales X, and over the chart body
        # scales both. xwheel_zoom is kept in the toolbar for those who want the
        # wheel to only ever scale time. 'pan' is the free (both-axes) pan tool;
        # 'xpan' stays the default active drag (horizontal navigation).
        tools="xpan,pan,wheel_zoom,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save",
        width=width,
        height=height,
        y_axis_location=y_axis_location,
        background_fill_color=theme["bg"],
        border_fill_color=theme["bg_border"],
        active_drag="xpan",
        active_scroll="wheel_zoom",
        y_range=y_range or DataRange1d(range_padding=0.05),
        output_backend=output_backend,
    )
    if x_range is not None:
        kw["x_range"] = x_range
    kw.update(kwargs)

    fig = figure(**kw)
    _apply_style(fig, theme)

    if hide_x_axis:
        fig.xaxis.visible = False

    return fig


def _new_ohlc_fig(height: int, width: int, theme: dict, y_range=None, x_range=None, output_backend: str = "canvas") -> tuple:
    from bokeh.plotting import figure
    from bokeh.models import Range1d, DataRange1d

    fig = figure(
        x_axis_type="datetime",
        # See _new_fig: wheel_zoom (dimensions="both") scales Y when scrolling
        # over the price axis, X over the time axis, both over the chart body.
        # 'pan' is the free (both-axes) pan; 'xpan' stays the default active drag.
        tools="xpan,pan,wheel_zoom,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save",
        width=width,
        height=height,
        name="ohlc",
        y_axis_location="right",
        background_fill_color=theme["bg"],
        border_fill_color=theme["bg_border"],
        active_drag="xpan",
        active_scroll="wheel_zoom",
        y_range=y_range or Range1d(start=0, end=1),
        x_range=x_range or DataRange1d(),
        output_backend=output_backend,
    )
    _apply_style(fig, theme)

    fig.xaxis.visible = True
    fig.xaxis.major_label_text_color = theme["text"]
    fig.xaxis.major_label_text_font  = "monospace"
    fig.xaxis.major_label_text_font_size = "10px"

    return fig, fig.x_range
    
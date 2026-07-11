from __future__ import annotations

import numpy as np

from ..config import PlotConfig
from .._util import (
    _datetime_format,
    _interval_label,
)


def render_ohlc(fig, source, theme: dict, config: PlotConfig, interval_ms: int, symbol: str, price_digits: int = 2) -> None:
    from bokeh.models import HoverTool, NumeralTickFormatter
    from bokeh.transform import factor_cmap
    from .._util import _price_number_format

    price_fmt = _price_number_format(price_digits)

    colors = [theme["candle_down"], theme["candle_up"]]
    cmap   = factor_cmap("inc", colors, ["0", "1"])

    ohlc_label = f"{symbol} ({_interval_label(interval_ms)})"

    if config.ohlc_style == "bar":
        hover_renderer = _render_ohlc_bars(fig, source, cmap, ohlc_label)
    else:
        hover_renderer = _render_ohlc_candles(
            fig, source, theme, cmap, ohlc_label
        )

    dt_fmt = _datetime_format(interval_ms)
    tooltips = [
        ("Date",      f"@ts{{{dt_fmt}}}"),
        ("Open",      f"@Open{{{price_fmt}}}"),
        ("High",      f"@High{{{price_fmt}}}"),
        ("Low",       f"@Low{{{price_fmt}}}"),
        ("Close",     f"@Close{{{price_fmt}}}"),
        ("Volume",    "@Volume{0,0.00}"),
    ]

    if "fp_delta" in source.data:
        tooltips += [
            ("Bid",       "@fp_bid{0,0}"),
            ("Ask",       "@fp_ask{0,0}"),
            ("Delta",     "@fp_delta{+0,0}"),
            ("POC",       f"@fp_poc_price{{{price_fmt}}}"),
            ("POC Vol",   "@fp_poc_vol{0,0}"),
            ("VAH",      f"@fp_vah{{{price_fmt}}}"),
            ("VAL",      f"@fp_val{{{price_fmt}}}"),
        ]

    fig.add_tools(HoverTool(
        tooltips=tooltips,
        formatters={"@ts": "datetime"},
        mode="mouse",
        renderers=[hover_renderer],
    ))

    # Price (Y) axis: fixed-decimal format so small prices (e.g. 0.1844) are not
    # truncated to 0.18. A footprint tick-based y-axis, if enabled, overrides
    # this later in the plot builder.
    fig.yaxis.formatter = NumeralTickFormatter(format=price_fmt)


def _render_ohlc_candles(fig, source, theme: dict, cmap, ohlc_label):
    """
    Draw classic Japanese candlesticks: a High-Low wick segment plus an
    Open-Close body bar. Returns the body renderer for HoverTool attachment.
    """
    fig.segment(
        x0="ts", y0="Low", x1="ts", y1="High", source=source,
        color=cmap, line_width=1.5, legend_label=ohlc_label,
    )

    src_data = source.data
    open_a   = src_data["Open"]
    close_a  = src_data["Close"]
    inc_a    = src_data["inc"]
    is_doji  = open_a == close_a
    color_up = theme.get("candle_up",   "#2ECC71")
    color_dn = theme.get("candle_down", "#E74C3C")
    candle_colors = np.array([
        color_up if inc == "1" else color_dn for inc in inc_a
    ], dtype=object)
    body_border_color = np.array([
        c if d else None for c, d in zip(candle_colors, is_doji)
    ], dtype=object)
    body_border_width = np.where(is_doji, 2.0, 1.0).astype(np.float64)
    source.data["body_border_color"] = body_border_color
    source.data["body_border_width"] = body_border_width

    bars = fig.vbar(
        x="ts", width="bar_width",
        top="top_body", bottom="bottom_body",
        source=source,
        fill_color=cmap,
        line_color="body_border_color",
        line_width="body_border_width",
        legend_label=ohlc_label,
    )
    return bars


def _render_ohlc_bars(fig, source, cmap, ohlc_label):
    """
    Draw OHLC bars: a vertical High-Low segment with a left tick at the open
    and a right tick at the close. All glyphs are segments (no body bar), which
    is lighter to prepare and accelerates well under the WebGL backend.

    Returns the vertical High-Low renderer for HoverTool attachment.
    """
    hl = fig.segment(
        x0="ts", y0="Low", x1="ts", y1="High", source=source,
        color=cmap, line_width=1.5, legend_label=ohlc_label,
    )
    fig.segment(
        x0="ts_left", y0="Open", x1="ts", y1="Open", source=source,
        color=cmap, line_width=1.5, legend_label=ohlc_label,
    )
    fig.segment(
        x0="ts", y0="Close", x1="ts_right", y1="Close", source=source,
        color=cmap, line_width=1.5, legend_label=ohlc_label,
    )
    return hl


def render_volume(fig, source, theme: dict, config: PlotConfig, interval_ms: int) -> None:
    from bokeh.models import LinearAxis, Range1d, DataRange1d, NumeralTickFormatter
    from bokeh.transform import factor_cmap

    vol_arr = np.asarray(source.data["Volume"], dtype=np.float64)
    if len(vol_arr) > 0:
        vol_max = float(np.nanmax(vol_arr)) or 1.0
        vol_range = Range1d(start=0, end=vol_max / 0.20)
    else:
        vol_range = DataRange1d(start=0, range_padding=4.0, range_padding_units="percent")
    fig.extra_y_ranges = {"vol": vol_range}

    ax_vol = LinearAxis(
        y_range_name="vol",
        axis_label="Volume",
    )
    ax_vol.formatter = NumeralTickFormatter(format="0.0a")
    ax_vol.axis_line_color        = theme["bg"]
    ax_vol.major_tick_line_color  = theme["axis"]
    ax_vol.minor_tick_line_color  = None
    ax_vol.major_label_text_color = theme["axis"]
    ax_vol.major_label_text_font_size = "8pt"
    ax_vol.axis_label_text_color = theme["axis"]
    ax_vol.axis_label_text_font_size = "9pt"
    ax_vol.axis_label_standoff = 12
    fig.add_layout(ax_vol, "left")

    colors = [theme["candle_down"], theme["candle_up"]]
    cmap   = factor_cmap("inc", colors, ["0", "1"])

    fig.vbar(
        x="ts", width="bar_width",
        top="Volume", bottom=0,
        source=source,
        fill_color=cmap,
        line_color=cmap,
        line_width=0,
        alpha=theme["volume_alpha"],
        y_range_name="vol",
        legend_label="Volume",
    )

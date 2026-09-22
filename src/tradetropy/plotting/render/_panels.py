from __future__ import annotations
import pathlib

import numpy as np

from .._util import _datetime_format


_JS_DIR = pathlib.Path(__file__).resolve().parent.parent / "js"


def render_footprint(
    fig,
    source,
    interval_ms: int,
    zoom_range: int,
) -> None:
    from bokeh.models import ColumnDataSource, CustomJS

    data  = source.data
    side_arr = np.array(data.get("side", []))
    if len(side_arr) == 0:
        n = len(data.get("x", []))
        bid_mask = np.arange(0, n, 2)
        ask_mask = np.arange(1, n, 2)
    else:
        bid_mask = side_arr == "bid"
        ask_mask = side_arr == "ask"

    bid_data = {k: np.asarray(v)[bid_mask] for k, v in data.items()}
    ask_data = {k: np.asarray(v)[ask_mask] for k, v in data.items()}

    source_bid = ColumnDataSource(data=bid_data)
    source_ask = ColumnDataSource(data=ask_data)

    rects_bid = fig.rect(
        x="x", y="y",
        width="cell_w", height="cell_h",
        width_units="data",
        height_units="data",
        source=source_bid,
        fill_color="fill",
        line_color="lc",
        line_width="lw",
    )
    rects_ask = fig.rect(
        x="x", y="y",
        width="cell_w", height="cell_h",
        width_units="data",
        height_units="data",
        source=source_ask,
        fill_color="fill",
        line_color="lc",
        line_width="lw",
    )

    txt_bid = fig.text(
        x="x", y="y",
        text="text",
        source=source_bid,
        text_align="center",
        text_baseline="middle",
        text_font_size="6.6pt",
        text_color="tc",
    )
    txt_ask = fig.text(
        x="x", y="y",
        text="text",
        source=source_ask,
        text_align="center",
        text_baseline="middle",
        text_font_size="6.6pt",
        text_color="tc",
    )

    _fp_lazy_cb = CustomJS(
        args=dict(
            txt_bid=txt_bid, txt_ask=txt_ask,
            rects_bid=rects_bid, rects_ask=rects_ask,
            dummy=None,
            interval_ms=interval_ms,
            zoom_range=zoom_range,
            ohlc_source=None,
            bar_width_wide=interval_ms * 0.9,
            bar_width_narrow=interval_ms * 0.3,
        ),
        code=(_JS_DIR / "fp_lazy.js").read_text(),
    )
    fig.x_range.js_on_change("start", _fp_lazy_cb)
    fig.x_range.js_on_change("end", _fp_lazy_cb)

    rects_bid.visible = False
    rects_ask.visible = False
    txt_bid.visible = False
    txt_ask.visible = False

    return source_bid, source_ask, rects_bid, rects_ask, txt_bid, txt_ask, _fp_lazy_cb


def render_pl_bars(fig, source, theme: dict, interval_ms: int = 0) -> None:
    """
    Render the per-trade P&L panel as connected markers (backtesting.py style).

    Each closed trade draws a segment from its entry to its exit (in
    entry_ts/exit_ts x pnl_pct y-space, entry pinned to 0 so the segment shows
    the trade's outcome path) and a circle marker AT the exit, sized by the
    magnitude of its P&L (|pnl|) and colored win/loss. This replaces the
    previous quad-bar rendering with the entry->exit + exit-circle convention
    used by backtesting.py's trade markers.
    """
    from bokeh.models import HoverTool, NumeralTickFormatter, Span

    pnl_pct      = np.asarray(source.data.get("pnl_pct",      []), dtype=np.float64)
    pnl_arr      = np.asarray(source.data.get("pnl",           []), dtype=np.float64)
    is_win       = source.data.get("is_win",       [])
    entry_ts     = source.data.get("entry_ts",     [])
    exit_ts      = source.data.get("exit_ts",       [])
    n = len(entry_ts)
    if n == 0:
        return

    win_color  = theme["trade_win"]
    loss_color = theme["trade_loss"]
    fill_colors = [win_color if w == "1" else loss_color for w in is_win]

    # Marker size scales with |pnl| (magnitude of money won/lost), the same
    # convention backtesting.py uses for its trade circles: bigger circle,
    # bigger trade outcome. Guards a degenerate all-zero-pnl series so trades
    # stay visible with a floor size instead of collapsing to 0px.
    abs_pnl = np.abs(pnl_arr)
    max_abs_pnl = float(np.nanmax(abs_pnl)) if n > 0 else 0.0
    if max_abs_pnl <= 0:
        marker_sizes = np.full(n, 10.0)
    else:
        marker_sizes = 6.0 + 24.0 * (abs_pnl / max_abs_pnl)
        marker_sizes = np.nan_to_num(marker_sizes, nan=6.0)

    source.data["_pl_marker_size"] = marker_sizes
    source.data["_pl_fill_color"]  = fill_colors
    source.data["_pl_entry_y"]     = np.zeros(n, dtype=np.float64)
    source.data["_pl_exit_y"]      = pnl_pct
    source.data["_pl_seg_x0"]      = np.array(entry_ts, dtype="datetime64[ms]")
    source.data["_pl_seg_x1"]      = np.array(exit_ts,  dtype="datetime64[ms]")

    # Entry -> exit connecting segment (thin, same win/loss color as the
    # marker), drawn first so the circle sits on top of its own line.
    fig.segment(
        x0="_pl_seg_x0", y0="_pl_entry_y",
        x1="_pl_seg_x1", y1="_pl_exit_y",
        source=source,
        line_color="_pl_fill_color",
        line_width=1.5,
        line_alpha=0.6,
    )

    markers = fig.scatter(
        x="_pl_seg_x1", y="_pl_exit_y",
        source=source,
        marker="circle",
        size="_pl_marker_size",
        fill_color="_pl_fill_color",
        line_color="_pl_fill_color",
        fill_alpha=0.75,
        line_alpha=0.9,
        line_width=1.0,
    )

    fig.add_layout(Span(
        location=0,
        dimension="width",
        line_color=theme.get("axis", "#8A9BB0"),
        line_width=1.0,
        line_dash="solid",
        line_alpha=0.6,
    ))

    dt_fmt = "%Y-%m-%d %H:%M"
    fig.add_tools(HoverTool(
        tooltips=[
            ("Dir.",     "@direction"),
            ("Entrada",  f"@entry_ts{{{dt_fmt}}}"),
            ("Salida",   f"@exit_ts{{{dt_fmt}}}"),
            ("P. Entr.", "@entry_price{0,0.00}"),
            ("P. Sal.",  "@exit_price{0,0.00}"),
            ("PnL $",    "@pnl{+0,0.00}"),
            ("PnL %",    "@trade_label"),
            ("Size",     "@size_signed{+0,0.00}"),
        ],
        formatters={"@entry_ts": "datetime", "@exit_ts": "datetime"},
        mode="mouse",
        renderers=[markers],
    ))

    fig.yaxis.axis_label = "PnL %"
    fig.yaxis.formatter  = NumeralTickFormatter(format="0.0")

    max_abs = float(np.nanmax(np.abs(pnl_pct))) if n > 0 else 1.0
    if max_abs == 0:
        max_abs = 1.0
    pad = max_abs * 0.15
    fig.y_range.start = float(np.nanmin(np.minimum(pnl_pct, 0.0))) - pad
    fig.y_range.end   = float(np.nanmax(np.maximum(pnl_pct, 0.0))) + pad


def render_volume_profile(fig, source, interval_ms: int = 0) -> None:
    """
    Render the horizontal volume-by-price histogram on the price figure.

    Draws one horizontal bar segment per (level, side) using the source built by
    build_volume_profile_source(). Sell- and buy-aggressor volume are colored
    distinctly via the per-row 'color' / 'alpha' columns, and the POC level is
    emphasized with a higher alpha.

    Args:
        fig: Target Bokeh figure (the OHLC price panel).
        source: ColumnDataSource with y, height, left, right, color, alpha, side.
        interval_ms (int): Unused; kept for renderer signature symmetry.

    Returns:
        The hbar glyph renderer, or None if the source is empty.
    """
    from bokeh.models import HoverTool

    if source is None or len(source.data.get("y", [])) == 0:
        return None

    bars = fig.hbar(
        y="y",
        height="height",
        left="left",
        right="right",
        source=source,
        fill_color="color",
        line_color="color",
        fill_alpha="alpha",
        line_alpha=0.0,
        legend_label="Volume Profile",
    )

    fig.add_tools(HoverTool(
        tooltips=[
            ("Price", "@y{0,0.00}"),
            ("Buy", "@vol_buy{0,0.00}"),
            ("Sell", "@vol_sell{0,0.00}"),
            ("Total", "@volume{0,0.00}"),
        ],
        mode="mouse",
        renderers=[bars],
    ))
    return bars

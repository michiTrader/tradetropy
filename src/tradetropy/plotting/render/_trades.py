from __future__ import annotations

from .._util import _datetime_format


def render_trades(fig, source, theme: dict, interval_ms: int) -> None:
    from bokeh.models import HoverTool
    from bokeh.transform import factor_cmap

    if source is None:
        return

    n_trades = len(source.data["entry_ts"])
    colors = [theme["trade_loss"], theme["trade_win"]]
    cmap   = factor_cmap("is_win", colors, ["0", "1"])

    lines = fig.multi_line(
        xs="lines_xs", ys="lines_ys",
        source=source,
        line_width=5,
        line_color=cmap,
        line_dash="dotted",
        line_alpha=0.8,
        legend_label=f"Trades ({n_trades})",
    )
    dt_fmt = _datetime_format(interval_ms)
    fig.add_tools(HoverTool(
        tooltips=[
            ("Entry",    f"@entry_ts{{{dt_fmt}}}"),
            ("Exit",     f"@exit_ts{{{dt_fmt}}}"),
            ("Entry P.", "@entry_price{0,0.00}"),
            ("Exit P.",  "@exit_price{0,0.00}"),
            ("PnL",      "@pnl{+0,0.00}"),
            ("Dir.",     "@direction"),
        ],
        formatters={"@entry_ts": "datetime", "@exit_ts": "datetime"},
        mode="mouse",
        renderers=[lines],
    ))


def render_trades_live(fig, source, theme: dict, interval_ms: int) -> object:
    from bokeh.models import HoverTool, ColumnDataSource, Span, Label
    import numpy as _np

    dt_fmt = _datetime_format(interval_ms)

    lines = fig.multi_line(
        xs="lines_xs", ys="lines_ys",
        source=source,
        line_width=2,
        line_color="trade_color",
        line_dash="dotted",
        line_alpha=0.6,
        legend_label="Trades (0)",
    )
    fig.add_tools(HoverTool(
        tooltips=[
            ("Entry",    f"@entry_ts{{{dt_fmt}}}"),
            ("Exit",     f"@exit_ts{{{dt_fmt}}}"),
            ("Entry P.", "@entry_price{0,0.00}"),
            ("Exit P.",  "@exit_price{0,0.00}"),
            ("PnL",      "@pnl{+0,0.00}"),
            ("Dir.",     "@direction"),
        ],
        formatters={"@entry_ts": "datetime", "@exit_ts": "datetime"},
        mode="mouse",
        renderers=[lines],
    ))

    # ── Entry / exit triangles ──────────────────────────────────────────
    source_markers = ColumnDataSource(dict(
        ts            = _np.array([], dtype="datetime64[ms]"),
        price         = _np.array([], dtype=_np.float64),
        color         = _np.array([], dtype=object),
        marker        = _np.array([], dtype=object),
        angle         = _np.array([], dtype=_np.float64),
        tooltip_ts    = _np.array([], dtype="datetime64[ms]"),
        tooltip_price = _np.array([], dtype=_np.float64),
        tooltip_pnl   = _np.array([], dtype=_np.float64),
        tooltip_dir   = _np.array([], dtype=object),
        tooltip_tag   = _np.array([], dtype=object),
    ))

    markers = fig.scatter(
        x="ts", y="price",
        source=source_markers,
        marker="marker",
        angle="angle",
        size=6,
        fill_color="color",
        line_color="color",
        fill_alpha=0.9,
        line_alpha=1.0,
        line_width=1.0,
        legend_label="Trades (0)",
    )
    fig.add_tools(HoverTool(
        tooltips=[
            ("",         "@tooltip_tag"),
            ("Time",     f"@tooltip_ts{{{dt_fmt}}}"),
            ("Price",    "@tooltip_price{0,0.00}"),
            ("PnL",      "@tooltip_pnl{+0,0.00}"),
            ("Dir.",     "@tooltip_dir"),
        ],
        formatters={"@tooltip_ts": "datetime"},
        mode="mouse",
        renderers=[markers],
    ))

    # ── Open position triangles (updated each tick) ─────────────────────
    source_open = ColumnDataSource(dict(
        ts            = _np.array([], dtype="datetime64[ms]"),
        price         = _np.array([], dtype=_np.float64),
        color         = _np.array([], dtype=object),
        marker        = _np.array([], dtype=object),
        angle         = _np.array([], dtype=_np.float64),
        tooltip_ts    = _np.array([], dtype="datetime64[ms]"),
        tooltip_price = _np.array([], dtype=_np.float64),
        tooltip_dir   = _np.array([], dtype=object),
        tooltip_tag   = _np.array([], dtype=object),
    ))

    open_markers = fig.scatter(
        x="ts", y="price",
        source=source_open,
        marker="marker",
        angle="angle",
        size=8,
        fill_color="white",
        line_color="color",
        line_alpha=1.0,
        line_width=1.5,
        legend_label="Trades (0)",
    )
    fig.add_tools(HoverTool(
        tooltips=[
            ("",         "@tooltip_tag"),
            ("Time",     f"@tooltip_ts{{{dt_fmt}}}"),
            ("Price",    "@tooltip_price{0,0.00}"),
            ("Dir.",     "@tooltip_dir"),
        ],
        formatters={"@tooltip_ts": "datetime"},
        mode="mouse",
        renderers=[open_markers],
    ))

    # ── Active position tracker: horizontal avg-price line + label ────────────
    # Tracks the open position's average price in real time (MT5/TV style).
    # Hidden when flat; the updater toggles visibility and sets location/text
    # each tick from the net position.
    pos_line = Span(
        location=0.0,
        dimension="width",
        line_color=theme.get("span_zero", "#94A3B8"),
        line_width=1.5,
        line_dash="dashed",
        line_alpha=0.9,
        visible=False,
    )
    fig.add_layout(pos_line)

    pos_label = Label(
        x=8, y=0.0,
        x_units="screen", y_units="data",
        text="",
        text_color=theme.get("span_zero", "#94A3B8"),
        text_font_size="9pt",
        background_fill_color=theme.get("bg", "#0F172A"),
        background_fill_alpha=0.7,
        visible=False,
    )
    fig.add_layout(pos_label)

    # ── TP/SL lines (data-driven, multi-level) ────────────────────────────────
    # Horizontal dashed lines at each open position's stop-loss (loss color) and
    # take-profit (win color), with a left-edge text label. Driven by a single
    # ColumnDataSource so any number of brackets render; empty when flat. The
    # updater refreshes it each tick from the open positions' sl/tp.
    from bokeh.models import HSpan, LabelSet

    tpsl_source = ColumnDataSource(dict(
        y     = _np.array([], dtype=_np.float64),
        color = _np.array([], dtype=object),
        text  = _np.array([], dtype=object),
    ))
    tpsl_lines = fig.hspan(
        y="y",
        source=tpsl_source,
        line_color="color",
        line_width=1.5,
        line_dash="dashed",
        line_alpha=0.9,
        legend_label="TP/SL",
    )
    tpsl_labels = LabelSet(
        x=8, y="y",
        x_units="screen", y_units="data",
        text="text",
        source=tpsl_source,
        text_color="color",
        text_font_size="9pt",
        background_fill_color=theme.get("bg", "#0F172A"),
        background_fill_alpha=0.7,
    )
    fig.add_layout(tpsl_labels)

    # ── Pending limit/stop order lines ───────────────────────────────────────
    pending_source = ColumnDataSource(dict(
        y     = _np.array([], dtype=_np.float64),
        color = _np.array([], dtype=object),
        text  = _np.array([], dtype=object),
    ))
    pending_lines = fig.hspan(
        y="y",
        source=pending_source,
        line_color="color",
        line_width=1.2,
        line_dash="dotted",
        line_alpha=0.85,
        legend_label="Orders",
    )
    pending_labels = LabelSet(
        x=8, y="y",
        x_units="screen", y_units="data",
        text="text",
        source=pending_source,
        text_color="color",
        text_font_size="9pt",
        background_fill_color=theme.get("bg", "#0F172A"),
        background_fill_alpha=0.7,
    )
    fig.add_layout(pending_labels)

    return (lines, source_markers, source_open, pos_line, pos_label,
            tpsl_source, tpsl_lines, tpsl_labels,
            pending_source, pending_lines, pending_labels)

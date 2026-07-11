from __future__ import annotations

import numpy as np

from ..config import PlotConfig
from .._util import _to_dt64


def render_equity(
    fig,
    source,
    theme: dict,
    config: PlotConfig,
    peak_val: float,
    final_val: float,
    peak_ts,
    final_ts,
    max_dd_ts,
    max_dd_val: float,
    max_dd_pct: float,
    baseline: float,
    trailing_source=None,
) -> None:
    from bokeh.models import HoverTool, Span, NumeralTickFormatter

    is_percent = config.equity_unit == "percent"
    mode = config.equity_mode
    fmt = "0.[00]%" if is_percent else "$0,0"

    final_label = f"{final_val:.2%}" if is_percent else f"${final_val:,.2f}"
    dd_label = f"Max DD ({max_dd_pct:.2%})"
    label_prefix = "Return" if mode == "return" else "Equity"

    fig.add_layout(Span(
        location=baseline,
        dimension="width",
        line_color=theme["span_zero"],
        line_width=0.8,
        line_dash="dashed",
    ))

    line_glyph = fig.line(
        x="ts", y="equity",
        source=source,
        color=theme["equity_line"],
        line_width=2.5,
        alpha=0.9,
        legend_label=f"{label_prefix} ({final_label})",
    )

    eq_vals  = source.data["equity"]
    ts_vals  = source.data["ts"]
    fig.varea(
        x=ts_vals,
        y1=np.full(len(eq_vals), baseline),
        y2=eq_vals,
        color=theme["equity_area"],
        alpha=0.08,
    )

    peak_ts_dt   = _to_dt64(peak_ts)
    final_ts_dt  = _to_dt64(final_ts)
    max_dd_ts_dt = _to_dt64(max_dd_ts)

    peak_label = f"Peak ({peak_val:.2%})" if is_percent else f"Peak (${peak_val:,.2f})"
    fig.scatter(
        x=peak_ts_dt, y=[peak_val],
        size=6, color=theme["peak_dot"],
        marker="circle", legend_label=peak_label,
    )
    fig.scatter(
        x=max_dd_ts_dt, y=[max_dd_val],
        size=7, color=theme["maxdd_dot"],
        marker="circle", legend_label=dd_label,
    )
    fig.scatter(
        x=final_ts_dt, y=[final_val],
        size=5, color=theme["equity_line"],
        marker="circle",
    )

    fig.yaxis.formatter = NumeralTickFormatter(format=fmt)
    fig.yaxis.axis_label = "Return %" if mode == "return" else "Equity"

    fig.add_tools(HoverTool(
        tooltips=[
            ("Fecha", "@ts{%Y-%m-%d}"),
            (label_prefix, f"@equity{{{fmt}}}"),
        ],
        formatters={"@ts": "datetime"},
        mode="mouse",
        renderers=[line_glyph],
    ))

    _eq_lo = float(np.nanmin(eq_vals))
    _eq_hi = float(np.nanmax(eq_vals))

    if trailing_source is not None and len(trailing_source.data["trailing_dd"]) > 0:
        tdd_vals = trailing_source.data["trailing_dd"]
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
                ("Fecha", "@ts{%Y-%m-%d}"),
                ("Trailing DD", f"@trailing_dd{{{fmt}}}"),
            ],
            formatters={"@ts": "datetime"},
            mode="mouse",
            renderers=[tdd_line],
        ))
        _tdd_lo = float(np.nanmin(tdd_vals))
        _tdd_hi = float(np.nanmax(tdd_vals))
        _eq_lo = min(_eq_lo, _tdd_lo)
        _eq_hi = max(_eq_hi, _tdd_hi)

    _eq_pad = (_eq_hi - _eq_lo) * 0.10
    fig.y_range.start = _eq_lo - _eq_pad
    fig.y_range.end   = _eq_hi + _eq_pad


def render_drawdown(fig, source, theme: dict) -> None:
    from bokeh.models import HoverTool, NumeralTickFormatter

    fig.varea(
        x="ts", y1=0, y2="drawdown",
        source=source,
        color=theme["drawdown_fill"],
        alpha=0.5,
    )
    dd_line = fig.line(
        x="ts", y="drawdown",
        source=source,
        color=theme["drawdown_fill"],
        line_width=1.5,
    )
    fig.add_tools(HoverTool(
        tooltips=[
            ("Fecha", "@ts{%Y-%m-%d}"),
            ("DD", "@drawdown{0.[0]%}"),
        ],
        formatters={"@ts": "datetime"},
        mode="mouse",
        renderers=[dd_line],
    ))
    fig.yaxis.formatter = NumeralTickFormatter(format="0.[0]%")
    fig.yaxis.axis_label = "DD %"

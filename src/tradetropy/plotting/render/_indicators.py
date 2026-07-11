from __future__ import annotations

import numpy as np

from ..config import IndicatorPlotMeta
from .._util import (
    _resolve_dim,
    _color_cycle,
    _datetime_format,
)


def _render_line(fig, src, color, meta, dim_idx, render_kwargs):
    lw = _resolve_dim(meta.line_width, dim_idx)
    ld = _resolve_dim(meta.line_dash, dim_idx)
    la = _resolve_dim(meta.line_alpha, dim_idx)
    bokeh_dash = [] if ld == "solid" else ld
    return fig.line(
        x="ts", y="value", source=src,
        color=color, line_width=lw, line_dash=bokeh_dash, alpha=la,
        **render_kwargs,
    )


def _render_scatter(fig, src, color, meta, dim_idx, render_kwargs):
    mk = _resolve_dim(meta.marker, dim_idx)
    ms = _resolve_dim(meta.marker_size, dim_idx)
    ma = _resolve_dim(meta.marker_alpha, dim_idx)
    mf = _resolve_dim(meta.marker_fill, dim_idx)
    mlw = _resolve_dim(meta.marker_line_width, dim_idx)
    fill_color = color if mf else None
    return fig.scatter(
        x="ts", y="value", source=src,
        color=color, fill_color=fill_color,
        size=ms, marker=mk, alpha=ma, line_width=mlw,
        **render_kwargs,
    )


def _render_step(fig, src, color, meta, dim_idx, render_kwargs):
    lw = _resolve_dim(meta.line_width, dim_idx)
    la = _resolve_dim(meta.line_alpha, dim_idx)
    return fig.step(
        x="ts", y="value", source=src,
        color=color, line_width=lw, alpha=la,
        mode="after",
        **render_kwargs,
    )


def _render_bar(fig, src, color, meta, dim_idx, render_kwargs, interval_ms=None):
    bar_w = int((interval_ms or 60_000) * _resolve_dim(meta.bar_width_factor, dim_idx))
    ba = _resolve_dim(meta.bar_alpha, dim_idx)
    col_pos = meta.bar_color_positive
    col_neg = meta.bar_color_negative

    # Diverging bars (one color per sign): bind a single vbar to the SAME source
    # the renderer received and color each bar from a per-row ``bar_color``
    # column carried in the source data. This is the proven, WebGL-safe pattern
    # (the order-flow DeltaBars panel colors its diverging bars the same way) and
    # is live-compatible: the source is filled/streamed by the updater, which
    # writes ``bar_color`` alongside ``value``, so the bars recolor as data
    # arrives without any client-side transform. Splitting the data into derived
    # ColumnDataSources at render time (the original approach) produced zero
    # glyphs in live because the source is empty when render runs.
    if col_pos is not None and col_neg is not None:
        c_pos = _resolve_dim(col_pos, dim_idx)
        c_neg = _resolve_dim(col_neg, dim_idx)
        # Ensure the column exists for the static path (the live empty source and
        # updater populate it on their own). Compute it from the current values.
        if "bar_color" not in src.data:
            vals = np.asarray(src.data.get("value", []), dtype=np.float64)
            src.data["bar_color"] = np.where(vals >= 0, c_pos, c_neg)
        g = fig.vbar(
            x="ts", top="value", bottom=0, width=bar_w,
            source=src, fill_color="bar_color", line_color="bar_color",
            fill_alpha=ba, line_alpha=0,
            **render_kwargs,
        )
    else:
        g = fig.vbar(
            x="ts", top="value", bottom=0, width=bar_w,
            source=src, fill_color=color, line_color=color,
            fill_alpha=ba, line_alpha=0,
            **render_kwargs,
        )
    return [g]


def render_indicator(
    fig,
    sources: list,
    meta: IndicatorPlotMeta,
    color_override: str | None = None,
    interval_ms: int | None = None,
) -> None:
    names_raw = meta.name
    single_legend = isinstance(names_raw, str)
    names = [names_raw] if single_legend else (names_raw or [])

    colors_raw = meta.color
    ts_band_set = set(getattr(meta, "_ts_band_indices", []))

    price_dim = 0

    from collections import defaultdict
    hover_renderers: dict[str, list] = defaultdict(list)

    # Glyph(s) for each price dimension, for z-order reordering if front_dim.
    dim_glyphs: dict[int, list] = defaultdict(list)

    for i, src in enumerate(sources):
        if i in ts_band_set:
            continue

        if isinstance(colors_raw, list):
            color = colors_raw[price_dim] if price_dim < len(colors_raw) else color_override or _color_cycle(price_dim)
        else:
            color = colors_raw or color_override or _color_cycle(price_dim)

        if color is None:
            price_dim += 1
            continue

        if meta.show_legend:
            if single_legend:
                label = names_raw
            else:
                label = names[price_dim] if price_dim < len(names) else f"{names[0] if names else ''}[{price_dim}]"
        else:
            label = None

        r_type = _resolve_dim(meta.renderer, price_dim)
        render_kwargs = {"legend_label": label} if meta.show_legend else {}

        if single_legend:
            dim_label = str(names_raw)
        else:
            dim_label = names[price_dim] if price_dim < len(names) else f"{names[0] if names else 'Indicator'}[{price_dim}]"

        if r_type == "scatter":
            g = _render_scatter(fig, src, color, meta, price_dim, render_kwargs)
            if g is not None:
                hover_renderers[dim_label].append(g)
                dim_glyphs[price_dim].append(g)
        elif r_type == "bar":
            glyphs = _render_bar(fig, src, color, meta, price_dim, render_kwargs, interval_ms)
            for g in glyphs:
                hover_renderers[dim_label].append(g)
                dim_glyphs[price_dim].append(g)
        elif r_type == "step":
            g = _render_step(fig, src, color, meta, price_dim, render_kwargs)
            if g is not None:
                hover_renderers[dim_label].append(g)
                dim_glyphs[price_dim].append(g)
        else:
            if meta.scatter:
                g = _render_scatter(fig, src, color, meta, price_dim, render_kwargs)
                if g is not None:
                    hover_renderers[dim_label].append(g)
                    dim_glyphs[price_dim].append(g)
            else:
                g = _render_line(fig, src, color, meta, price_dim, render_kwargs)
                if g is not None:
                    hover_renderers[dim_label].append(g)
                    dim_glyphs[price_dim].append(g)

        price_dim += 1

    # ── Reorder z-order: front_dim dimension is drawn on top ───────────────────
    front_dim = getattr(meta, "front_dim", None)
    if front_dim is not None and front_dim in dim_glyphs:
        front = dim_glyphs[front_dim]
        rest = [r for r in fig.renderers if r not in front]
        fig.renderers = rest + front

    if hover_renderers:
        from bokeh.models import HoverTool
        dt_fmt = _datetime_format(interval_ms) if interval_ms else "%Y-%m-%d %H:%M"
        for dlabel, renderers in hover_renderers.items():
            # Detect if any renderer uses a source with a 'tag' column
            has_tag = any(
                "tag" in getattr(getattr(r, "data_source", None), "data", {})
                for r in renderers
            )
            tooltips = [(dlabel, "@value{0,0.00}")]
            if has_tag:
                tooltips.append(("Type", "@tag"))
            tooltips.append(("Time", f"@ts{{{dt_fmt}}}"))
            fig.add_tools(HoverTool(
                tooltips=tooltips,
                formatters={"@ts": "datetime"},
                mode="mouse",
                renderers=renderers,
            ))


def render_reference_lines(fig, meta: IndicatorPlotMeta) -> None:
    from bokeh.models import Span, Label

    for ref in (meta.reference_lines or []):
        value = ref.get("value")
        if value is None:
            continue

        color  = ref.get("color", "#888888")
        dash   = ref.get("dash", "dashed")
        width  = ref.get("width", 1.0)
        label  = ref.get("label", "")

        bokeh_dash = [] if dash == "solid" else dash
        fig.add_layout(Span(
            location=value,
            dimension="width",
            line_color=color,
            line_width=width,
            line_dash=bokeh_dash,
            line_alpha=0.7,
        ))

        if label:
            fig.add_layout(Label(
                x=1.0,
                y=value,
                x_units="screen",
                text=f" {label}",
                text_font_size="8pt",
                text_color=color,
                text_alpha=0.8,
            ))

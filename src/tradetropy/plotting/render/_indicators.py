from __future__ import annotations

import numpy as np

from ..config import IndicatorPlotMeta
from .._util import (
    _resolve_dim,
    _color_cycle,
    _datetime_format,
)
from tradetropy.ta.base import resolve_hide_on_pan

# Level-of-detail decimation for dense bar-style indicator panels (e.g. a
# MACD histogram). Panels with more bars than this keep their full data in a
# backing source and draw a bucket-aggregated view refined on pan/zoom (see
# decimate_bar.js); below the threshold bars draw 1:1 with no callback
# overhead. This is the alternative to blanking for renderer="bar": dense
# panels stay visible and legible during pan instead of disappearing.
_BAR_LOD_THRESHOLD = 4000
_BAR_LOD_MAX_VISIBLE = 3000


def _attach_bar_lod(x_range, full_src, view_src, max_visible: int) -> None:
    """Wire decimate_bar.js so the drawn bar view refines on pan/zoom."""
    import pathlib
    from bokeh.models import CustomJS

    js_dir = pathlib.Path(__file__).resolve().parent.parent / "js"
    body = (js_dir / "decimate_bar.js").read_text()
    cb = CustomJS(
        args=dict(full=full_src, view=view_src, x_range=x_range,
                  max_visible=max_visible),
        code=body,
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)


def _decimate_bar_indices(n: int, max_visible: int) -> np.ndarray:
    """Python mirror of decimate_bar.js's passthrough/bucket split, used only
    to seed the initial drawn view before any pan/zoom interaction."""
    if n <= max_visible:
        return np.arange(n, dtype=np.int64)
    edges = np.linspace(0, n, max_visible + 1).astype(np.int64)
    return edges[:-1]


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


def _render_bar(fig, src, color, meta, dim_idx, render_kwargs, interval_ms=None,
                x_range=None):
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

    # Static-only level-of-detail: a dense, already-fixed bar source (a
    # finished backtest's MACD histogram, etc.) is bucket-decimated on
    # pan/zoom instead of blanked (see decimate_bar.js). x_range is only
    # passed by the static plotting.py path; the live path (empty source
    # filled in later by the updater) leaves it None and always draws 1:1,
    # since decimating a source that is about to be live-patched would fight
    # the updater's own writes.
    draw_src = src
    n_rows = len(src.data.get("ts", []))
    if x_range is not None and n_rows > _BAR_LOD_THRESHOLD:
        from bokeh.models import ColumnDataSource
        idx = _decimate_bar_indices(n_rows, _BAR_LOD_MAX_VISIBLE)
        view_data = {k: (np.asarray(v)[idx] if hasattr(v, "__len__") else v)
                     for k, v in src.data.items()}
        draw_src = ColumnDataSource(view_data)
        _attach_bar_lod(x_range, src, draw_src, _BAR_LOD_MAX_VISIBLE)

    if col_pos is not None and col_neg is not None:
        g = fig.vbar(
            x="ts", top="value", bottom=0, width=bar_w,
            source=draw_src, fill_color="bar_color", line_color="bar_color",
            fill_alpha=ba, line_alpha=0,
            **render_kwargs,
        )
    else:
        g = fig.vbar(
            x="ts", top="value", bottom=0, width=bar_w,
            source=draw_src, fill_color=color, line_color=color,
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
    hide_on_pan_sink: list | None = None,
    x_range=None,
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

        _hide_on_pan = resolve_hide_on_pan(r_type, getattr(meta, "hide_on_pan", None))
        # Bar panels never enter the blanking registry, even with an explicit
        # hide_on_pan=True override: dense bars rely on decimate_bar.js's LOD
        # instead (a vanishing histogram mid-drag reads as a bug, unlike a
        # marker/label blink). See _render_bar's own LOD wiring above.
        if r_type == "bar":
            _hide_on_pan = False

        def _register(glyph):
            if hide_on_pan_sink is not None and _hide_on_pan and glyph is not None:
                hide_on_pan_sink.append((glyph, None))

        if r_type == "scatter":
            g = _render_scatter(fig, src, color, meta, price_dim, render_kwargs)
            if g is not None:
                hover_renderers[dim_label].append(g)
                dim_glyphs[price_dim].append(g)
                _register(g)
        elif r_type == "bar":
            glyphs = _render_bar(fig, src, color, meta, price_dim, render_kwargs,
                                interval_ms, x_range=x_range)
            for g in glyphs:
                hover_renderers[dim_label].append(g)
                dim_glyphs[price_dim].append(g)
                _register(g)
        elif r_type == "step":
            g = _render_step(fig, src, color, meta, price_dim, render_kwargs)
            if g is not None:
                hover_renderers[dim_label].append(g)
                dim_glyphs[price_dim].append(g)
                _register(g)
        else:
            if meta.scatter:
                g = _render_scatter(fig, src, color, meta, price_dim, render_kwargs)
                if g is not None:
                    hover_renderers[dim_label].append(g)
                    dim_glyphs[price_dim].append(g)
                    _register(g)
            else:
                g = _render_line(fig, src, color, meta, price_dim, render_kwargs)
                if g is not None:
                    hover_renderers[dim_label].append(g)
                    dim_glyphs[price_dim].append(g)
                    _register(g)

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

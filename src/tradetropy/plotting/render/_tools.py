"""
Generic renderer for tool draw primitives.

Translates the Bokeh-free primitives emitted by tools (:mod:`tradetropy.ta.tool.draw`)
into Bokeh glyphs. This is the single place that knows how to draw a tool, shared
by the static (backtest) and live render paths so neither needs tool-specific
knowledge.

Primitives are grouped by legend name upstream (``collect_draw_primitives``).
Within a group, primitives of the same kind are concatenated into one
ColumnDataSource / glyph, and all glyphs of a group share the same
``legend_label`` so a single legend click toggles the whole tool.

Colors may be theme tokens of the form ``"theme:<key>"`` (e.g. ``"theme:text"``),
resolved here against the active plot theme.
"""

from __future__ import annotations

import numpy as np

from tradetropy.ta.draw import HBars, HLines, Labels, Points, Rects, Segments

_FALLBACK_COLOR = "#888888"

# Level-of-detail decimation for large point clouds (scatter). Clouds larger
# than the threshold keep their full data in a backing source; only a capped,
# uniformly-spaced sample of the visible window is drawn (refined on zoom by
# ``decimate_points.js``). Below the threshold the cloud draws in full with no
# callback overhead, so zoom-in always shows every point.
_POINTS_LOD_THRESHOLD = 6000
_POINTS_LOD_MAX_VISIBLE = 4000


def _decimate_indices(n: int, max_visible: int) -> np.ndarray:
    """Uniformly-spaced row indices to keep, preserving order (Python mirror of
    the JS subsample)."""
    if n <= max_visible:
        return np.arange(n, dtype=np.int64)
    step = n / max_visible
    return (np.arange(max_visible, dtype=np.float64) * step).astype(np.int64)


def _decimate_point_data(data: dict, max_visible: int) -> dict:
    """
    Build the initial drawn view for a point cloud by uniformly subsampling
    every parallel column to at most ``max_visible`` rows.

    The x_range CustomJS refines this on the first pan/zoom; this only seeds a
    representative view so the glyph shows before any interaction. Handles both
    NumPy arrays (y/size/alpha/x) and plain lists (color/fill_color strings).
    """
    n = len(data.get("x", []))
    idx = _decimate_indices(n, max_visible)
    out: dict = {}
    for key, col in data.items():
        if isinstance(col, np.ndarray):
            out[key] = col[idx]
        else:
            out[key] = [col[i] for i in idx]
    return out


def _attach_point_lod(x_range, full_src, view_src, max_visible: int) -> None:
    """Wire the decimate_points.js callback so the drawn view refines on zoom."""
    import pathlib
    from bokeh.models import CustomJS

    js_dir = pathlib.Path(__file__).resolve().parent.parent / "js"
    body = (js_dir / "decimate_points.js").read_text()
    cb = CustomJS(
        args=dict(full=full_src, view=view_src, x_range=x_range,
                  max_visible=max_visible),
        code=body,
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)



def _tok(value, theme: dict | None):
    """Resolve a single color value, expanding 'theme:<key>' tokens."""
    if isinstance(value, str) and value.startswith("theme:"):
        key = value.split(":", 1)[1]
        return (theme or {}).get(key, _FALLBACK_COLOR)
    return value if value is not None else _FALLBACK_COLOR


def _colors(value, n: int, theme: dict | None) -> list:
    """Resolve a scalar or per-row color spec into a list of length n."""
    if isinstance(value, (list, tuple, np.ndarray)):
        out = [_tok(v, theme) for v in value]
        if len(out) < n:
            out += [out[-1] if out else _FALLBACK_COLOR] * (n - len(out))
        return out[:n]
    return [_tok(value, theme)] * n


def _alphas(value, n: int) -> np.ndarray:
    """Resolve a scalar or per-row alpha spec into an array of length n."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64)
        if len(arr) < n:
            arr = np.concatenate([arr, np.full(n - len(arr), arr[-1] if len(arr) else 0.9)])
        return arr[:n]
    return np.full(n, float(value))


def _dt(arr) -> np.ndarray:
    return np.asarray(list(arr), dtype="datetime64[ms]")


def render_tool_groups(
    fig,
    groups: dict[str, list],
    theme: dict | None,
    interval_ms: int | None,
    registry: dict | None = None,
    show_legend: bool = True,
    label_sink: list | None = None,
    lazy_x_range=None,
    lazy_zoom_range: int | None = None,
    enable_point_lod: bool = False,
) -> dict:
    """
    Render (or update) tool primitive groups onto ``fig``.

    Args:
        fig: Target OHLC price figure.
        groups (dict[str, list]): legend name -> primitives, from
            ``collect_draw_primitives``.
        theme (dict | None): Active plot theme (for color tokens).
        interval_ms (int | None): OHLC interval (reserved for geometry).
        registry (dict | None): Persistent map (legend, kind) -> ColumnDataSource.
            None -> static mode (create glyphs). When provided, existing sources
            are updated in place and new ones are created lazily (live mode).
        show_legend (bool): When False, glyphs are created without a
            ``legend_label`` so the panel shows no legend (own-panel indicators
            such as Delta / CVD identify themselves by their panel title).
        label_sink (list | None): When provided, each created text-label glyph is
            appended as a ``(label_renderer, group_glyph)`` tuple so the caller
            can wire one global zoom callback (lazy labels). ``group_glyph`` is
            the group's first non-label glyph (or None) used to gate the label on
            its legend state.
        lazy_x_range: Shared price x_range; when given, the legend toggle re-runs
            the lazy condition so a label only shows when zoomed in AND enabled.
        lazy_zoom_range (int | None): Max candles in view to keep labels visible.

    Returns:
        dict: The registry, so a live ref can reuse it across updates.
    """
    registry = registry if registry is not None else {}

    for legend, prims in groups.items():
        legend_label = legend if show_legend else None
        by_kind: dict[type, list] = {}
        for p in prims:
            by_kind.setdefault(type(p), []).append(p)

        new_glyphs: list = []
        new_labels: list = []
        for kind, plist in by_kind.items():
            data = _build_data(kind, plist, theme)
            if data is None:
                continue
            key = (legend, kind.__name__)
            existing = registry.get(key)
            if existing is not None:
                existing.data = data
            else:
                from bokeh.models import ColumnDataSource
                # Large point clouds: keep the full data in a backing source and
                # draw only a capped, zoom-refined sample (static paths only;
                # the live path bounds points by the max_candles rollover).
                if (enable_point_lod and kind is Points and lazy_x_range is not None
                        and len(data.get("x", [])) > _POINTS_LOD_THRESHOLD):
                    full_src = ColumnDataSource(data)
                    view_src = ColumnDataSource(
                        _decimate_point_data(data, _POINTS_LOD_MAX_VISIBLE)
                    )
                    obj = _create_glyph(fig, kind, view_src, plist[0], legend_label, theme)
                    _attach_point_lod(lazy_x_range, full_src, view_src,
                                      _POINTS_LOD_MAX_VISIBLE)
                    registry[key] = full_src
                else:
                    src = ColumnDataSource(data)
                    obj = _create_glyph(fig, kind, src, plist[0], legend_label, theme)
                    registry[key] = src
                if obj is not None:
                    (new_labels if kind is Labels else new_glyphs).append(obj)

        if not new_glyphs and new_labels and legend_label:
            from bokeh.models import ColumnDataSource
            _dummy_src = ColumnDataSource({"x": [], "y": []})
            _anchor = fig.scatter(
                x="x", y="y", source=_dummy_src, size=1, alpha=0,
                legend_label=legend_label,
            )
            new_glyphs.append(_anchor)

        # Toggling the legend hides the group's glyphs; link any LabelSet so the
        # text hides with them (LabelSets are not legend-driven on their own).
        if new_glyphs and new_labels:
            _link_labels_visibility(
                new_glyphs, new_labels,
                x_range=lazy_x_range, interval_ms=interval_ms,
                zoom_range=lazy_zoom_range,
            )

        # Register labels for the global zoom-based hide (lazy labels).
        if label_sink is not None and new_labels:
            group_glyph = new_glyphs[0] if new_glyphs else None
            for lbl in new_labels:
                label_sink.append((lbl, group_glyph))
        # Live path: no central post-pass, so each group attaches its own
        # x_range callback at creation time (labels stream in incrementally).
        elif label_sink is None and lazy_x_range is not None and new_labels:
            _attach_group_lazy_labels(
                lazy_x_range, new_labels,
                new_glyphs[0] if new_glyphs else None,
                interval_ms, lazy_zoom_range,
            )

    return registry


def _attach_group_lazy_labels(x_range, labels: list, group, interval_ms,
                              zoom_range) -> None:
    """
    Attach a per-group zoom callback hiding a single group's labels (live path).

    Args:
        x_range: The shared price X range driving the toggle.
        labels (list): This group's text-label renderers.
        group: The group's legend glyph (or None) gating the labels.
        interval_ms (int): OHLC interval (to convert the range to candle count).
        zoom_range (int): Max candles in view to keep labels visible.
    """
    if not interval_ms or not zoom_range:
        return
    import pathlib
    from bokeh.models import CustomJS

    js_dir = pathlib.Path(__file__).resolve().parent.parent / "js"
    cb = CustomJS(
        args=dict(labels=labels, group=group,
                  interval_ms=interval_ms, zoom_range=zoom_range),
        code=(js_dir / "lazy_labels_group.js").read_text(),
    )
    x_range.js_on_change("start", cb)
    x_range.js_on_change("end", cb)


def _link_labels_visibility(glyphs: list, labelsets: list, x_range=None,
                            interval_ms: int | None = None,
                            zoom_range: int | None = None) -> None:
    """
    Tie LabelSet visibility to a group's glyph (legend toggle propagates).

    When ``x_range`` / ``interval_ms`` / ``zoom_range`` are provided the toggle is
    also gated by the zoom level, so re-enabling a legend while zoomed out does
    not bring its labels back until the user zooms in (parity with the global
    lazy-labels callback).
    """
    from bokeh.models import CustomJS
    if x_range is not None and interval_ms and zoom_range:
        cb = CustomJS(
            args=dict(labelsets=labelsets, glyph=glyphs[0], x_range=x_range,
                      interval_ms=interval_ms, zoom_range=zoom_range),
            code=(
                "const n = (x_range.end - x_range.start) / interval_ms;"
                "const v = glyph.visible && (n <= zoom_range);"
                "for (const ls of labelsets) { ls.visible = v; }"
            ),
        )
    else:
        cb = CustomJS(
            args=dict(labelsets=labelsets, glyph=glyphs[0]),
            code="const v = glyph.visible; for (const ls of labelsets) { ls.visible = v; }",
        )
    for g in glyphs:
        g.js_on_change("visible", cb)


def _build_data(kind, plist: list, theme: dict | None) -> dict | None:
    """Concatenate same-kind primitives into a single column-data dict."""
    if kind is HLines:
        x0, x1, y, color, alpha = [], [], [], [], []
        for p in plist:
            n = len(list(p.y))
            x0 += list(p.x0); x1 += list(p.x1); y += list(p.y)
            color += _colors(p.color, n, theme)
            alpha += list(_alphas(p.alpha, n))
        if not y:
            return None
        data = dict(x0=_dt(x0), x1=_dt(x1), y=np.asarray(y, dtype=np.float64),
                    color=color, alpha=np.asarray(alpha, dtype=np.float64))
        _merge_extra(data, plist)
        return data

    if kind is Segments:
        x0, y0, x1, y1, color, alpha = [], [], [], [], [], []
        for p in plist:
            n = len(list(p.y0))
            x0 += list(p.x0); y0 += list(p.y0); x1 += list(p.x1); y1 += list(p.y1)
            color += _colors(p.color, n, theme)
            alpha += list(_alphas(p.alpha, n))
        if not y0:
            return None
        data = dict(x0=_dt(x0), y0=np.asarray(y0, dtype=np.float64),
                    x1=_dt(x1), y1=np.asarray(y1, dtype=np.float64),
                    color=color, alpha=np.asarray(alpha, dtype=np.float64))
        _merge_extra(data, plist)
        return data

    if kind is Points:
        x, y, color, alpha, size = [], [], [], [], []
        fill_color, fill_alpha = [], []
        for p in plist:
            n = len(list(p.y))
            x += list(p.x); y += list(p.y)
            color += _colors(p.color, n, theme)
            alpha += list(_alphas(p.alpha, n))
            size += list(_alphas(p.size, n))
            fc = p.fill_color if p.fill_color is not None else p.color
            fill_color += _colors(fc, n, theme)
            if p.fill_alpha is not None:
                fill_alpha += list(_alphas(p.fill_alpha, n))
            else:
                fill_alpha += list(_alphas(p.alpha if p.fill else 0.0, n))
        if not y:
            return None
        data = dict(x=_dt(x), y=np.asarray(y, dtype=np.float64),
                    color=color, alpha=np.asarray(alpha, dtype=np.float64),
                    fill_color=fill_color,
                    fill_alpha=np.asarray(fill_alpha, dtype=np.float64),
                    size=np.asarray(size, dtype=np.float64))
        _merge_extra(data, plist)
        return data

    if kind is HBars:
        y, height, left, right, color, alpha = [], [], [], [], [], []
        for p in plist:
            n = len(list(p.y))
            y += list(p.y); height += list(p.height)
            left += list(p.left); right += list(p.right)
            color += _colors(p.color, n, theme)
            alpha += list(_alphas(p.alpha, n))
        if not y:
            return None
        data = dict(y=np.asarray(y, dtype=np.float64),
                    height=np.asarray(height, dtype=np.float64),
                    left=_dt(left), right=_dt(right),
                    color=color, alpha=np.asarray(alpha, dtype=np.float64))
        _merge_extra(data, plist)
        return data

    if kind is Rects:
        x0, x1, y0, y1, fill, line, fa, la = [], [], [], [], [], [], [], []
        for p in plist:
            n = len(list(p.y0))
            x0 += list(p.x0); x1 += list(p.x1); y0 += list(p.y0); y1 += list(p.y1)
            fill += _colors(p.fill_color, n, theme)
            line += _colors(p.line_color if p.line_color is not None else p.fill_color, n, theme)
            fa += list(_alphas(p.fill_alpha, n))
            la += list(_alphas(p.line_alpha, n))
        if not y0:
            return None
        data = dict(left=_dt(x0), right=_dt(x1),
                    bottom=np.asarray(y0, dtype=np.float64),
                    top=np.asarray(y1, dtype=np.float64),
                    fill=fill, line=line,
                    fill_alpha=np.asarray(fa, dtype=np.float64),
                    line_alpha=np.asarray(la, dtype=np.float64))
        _merge_extra(data, plist)
        return data

    if kind is Labels:
        x, y, text, color = [], [], [], []
        x_off, y_off, t_align, t_base = [], [], [], []
        for p in plist:
            n = len(list(p.y))
            x += list(p.x); y += list(p.y); text += list(p.text)
            color += _colors(p.color, n, theme)
            x_off += [p.x_offset] * n
            y_off += [p.y_offset] * n
            t_align += [p.text_align] * n
            t_base += [p.text_baseline] * n
        if not y:
            return None
        return dict(x=_dt(x), y=np.asarray(y, dtype=np.float64),
                    text=[str(t) for t in text], color=color,
                    x_offset=np.asarray(x_off, dtype=np.int32),
                    y_offset=np.asarray(y_off, dtype=np.int32),
                    text_align=t_align, text_baseline=t_base)

    return None


def _merge_extra(data: dict, plist: list) -> None:
    """Concatenate any ``extra`` hover columns across same-kind primitives."""
    keys: set[str] = set()
    for p in plist:
        keys |= set(getattr(p, "extra", {}) or {})
    for k in keys:
        col: list = []
        for p in plist:
            extra = getattr(p, "extra", {}) or {}
            col += list(extra.get(k, []))
        data[k] = np.asarray(col, dtype=np.float64)


def _create_glyph(fig, kind, src, proto, legend: str | None, theme: dict | None) -> None:
    """
    Create the Bokeh glyph for one primitive kind and wire its HoverTool.

    ``legend`` is the legend label for the group, or None to create the glyph
    without a legend entry (own-panel indicators that show no legend).
    """
    from bokeh.models import HoverTool

    legend_kw = {"legend_label": legend} if legend else {}

    glyph = None
    if kind is HLines:
        glyph = fig.segment(
            x0="x0", y0="y", x1="x1", y1="y", source=src,
            line_color="color", line_alpha="alpha",
            line_width=proto.width, line_dash=proto.dash,
            **legend_kw,
        )
    elif kind is Segments:
        glyph = fig.segment(
            x0="x0", y0="y0", x1="x1", y1="y1", source=src,
            line_color="color", line_alpha="alpha",
            line_width=proto.width, line_dash=proto.dash,
            **legend_kw,
        )
    elif kind is Points:
        glyph = fig.scatter(
            x="x", y="y", source=src, size="size", marker=proto.marker,
            fill_color="fill_color", fill_alpha="fill_alpha",
            line_color="color", line_alpha="alpha",
            line_width=proto.line_width,
            **legend_kw,
        )
    elif kind is HBars:
        glyph = fig.hbar(
            y="y", height="height", left="left", right="right", source=src,
            fill_color="color", line_color="color",
            fill_alpha="alpha", line_alpha=0.0,
            **legend_kw,
        )
    elif kind is Rects:
        glyph = fig.quad(
            left="left", right="right", bottom="bottom", top="top", source=src,
            fill_color="fill", fill_alpha="fill_alpha",
            line_color="line", line_alpha="line_alpha",
            line_width=proto.line_width,
            **legend_kw,
        )
    elif kind is Labels:
        from bokeh.models import LabelSet
        labels = LabelSet(
            x="x", y="y", text="text", source=src,
            x_offset="x_offset", y_offset="y_offset",
            text_font_size=proto.font_size, text_color="color",
            text_align="text_align", text_baseline="text_baseline",
            y_units=getattr(proto, "y_units", "data"),
        )
        fig.add_layout(labels)
        return labels

    if glyph is not None and getattr(proto, "hover", None):
        fig.add_tools(HoverTool(tooltips=list(proto.hover), mode="mouse", renderers=[glyph]))
    return glyph

"""
Monte Carlo result visualization.

Renders two complementary views with Bokeh:

    - Equity cone: the simulated equity paths summarized as percentile bands
      (p5-p95 shaded, median line) with the baseline equity overlaid. Only
      available for the trade-level path, which keeps the per-simulation equity
      curves.
    - Metric histograms: the distribution of each tracked metric across
      simulations, with the baseline value marked.

Bokeh is an optional dependency; importing this module without it raises a
clear error only when ``plot()`` is actually called.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Number of points the equity cone is resampled to (keeps the figure light).
_CONE_POINTS = 200


def _require_bokeh():
    """
    Import Bokeh lazily.

    Returns:
        module: The bokeh.plotting module.

    Raises:
        ImportError: If Bokeh is not installed.
    """
    try:
        import bokeh.plotting as bp  # noqa: F401
        return bp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'Monte Carlo plotting requires bokeh. Install it with '
            "'pip install bokeh'."
        ) from exc


def _resample_equity(equity: pd.Series, n: int) -> np.ndarray:
    """
    Resample an equity curve to a fixed number of points on [0, 1] progress.

    Args:
        equity (pd.Series): Equity curve (any length).
        n (int): Target number of points.

    Returns:
        np.ndarray: Equity values interpolated onto n evenly spaced points.
    """
    values = np.asarray(equity, dtype=float)
    if values.size == 0:
        return np.full(n, np.nan)
    if values.size == 1:
        return np.full(n, values[0])
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, values)


def _build_cone(equity_paths: List[pd.Series]):
    """
    Build percentile bands from a set of equity paths.

    Args:
        equity_paths (list[pd.Series]): Per-simulation equity curves.

    Returns:
        tuple: (x, p5, p50, p95) arrays on normalized progress, or None if
            there are no usable paths.
    """
    if not equity_paths:
        return None
    matrix = np.vstack([_resample_equity(e, _CONE_POINTS) for e in equity_paths])
    x = np.linspace(0.0, 1.0, _CONE_POINTS)
    p5 = np.nanpercentile(matrix, 5, axis=0)
    p50 = np.nanpercentile(matrix, 50, axis=0)
    p95 = np.nanpercentile(matrix, 95, axis=0)
    return x, p5, p50, p95


def plot_result(result, *, theme='light', width=1100, height=320,
                output='show', filename='montecarlo.html'):
    """
    Render the Monte Carlo result as an equity cone plus metric histograms.

    Args:
        result (MonteCarloResult): The result to visualize.
        theme (str): 'light' or 'dark'.
        width (int): Figure width in pixels.
        height (int): Per-panel height in pixels.
        output (str): 'show', 'file' or 'notebook'.
        filename (str): Output HTML path when output == 'file'.

    Returns:
        The Bokeh layout object (also shown/saved per ``output``).
    """
    bp = _require_bokeh()
    from bokeh.layouts import column, gridplot
    from bokeh.models import Span
    from tradetropy.plotting.theme.registry import get_theme

    tokens = get_theme(theme)
    bg = tokens['bg']
    grid = tokens['grid']
    text = tokens['text']
    band_color = tokens['equity_area']
    median_color = tokens['equity_line']
    baseline_color = tokens['peak_dot']
    hist_color = tokens['candle_up']

    def _style(fig):
        fig.background_fill_color = bg
        fig.border_fill_color = bg
        fig.xgrid.grid_line_color = grid
        fig.ygrid.grid_line_color = grid
        fig.axis.axis_line_color = grid
        fig.axis.major_label_text_color = text
        fig.title.text_color = text
        return fig

    panels = []

    cone = _build_cone(result._equity_paths)
    if cone is not None:
        x, p5, p50, p95 = cone
        fig = bp.figure(width=width, height=height,
                        title='Equity cone (p5 - p95, median)',
                        x_axis_label='progress', y_axis_label='equity')
        _style(fig)
        fig.varea(x=x, y1=p5, y2=p95, fill_color=band_color, fill_alpha=0.25,
                  legend_label='p5 - p95')
        fig.line(x, p50, line_color=median_color, line_width=2,
                 legend_label='median')
        base_eq = result.original_stats.equity_curve \
            if result.original_stats is not None else None
        if base_eq is not None and len(base_eq):
            fig.line(x, _resample_equity(base_eq, _CONE_POINTS),
                     line_color=baseline_color, line_width=2,
                     line_dash='dashed', legend_label='baseline')
        fig.legend.label_text_color = text
        fig.legend.background_fill_color = bg
        panels.append(fig)

    hist_figs = []
    sims = result.to_dataframe()
    for metric in sims.columns:
        col = sims[metric].to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        hist, edges = np.histogram(col, bins=30)
        fig = bp.figure(width=width // 2, height=height,
                        title=metric, y_axis_label='count')
        _style(fig)
        fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                 fill_color=hist_color, fill_alpha=0.6, line_color=grid)
        original = result.original_stats.get(metric, None) \
            if result.original_stats is not None else None
        if original is not None and np.isfinite(float(original)):
            fig.add_layout(Span(location=float(original), dimension='height',
                                line_color=baseline_color, line_width=2,
                                line_dash='dashed'))
        hist_figs.append(fig)

    layout_items = list(panels)
    if hist_figs:
        layout_items.append(gridplot(
            [hist_figs[i:i + 2] for i in range(0, len(hist_figs), 2)],
            toolbar_location='right',
        ))
    layout = column(*layout_items) if layout_items else bp.figure()

    if output == 'file':
        bp.output_file(filename)
        bp.save(layout)
    elif output == 'notebook':
        bp.output_notebook()
        bp.show(layout)
    else:
        bp.show(layout)
    return layout

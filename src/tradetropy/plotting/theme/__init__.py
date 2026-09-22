"""
Plot themes for the static (backtest) and live charts.

A *theme* is a flat dict of named color/style tokens (background, grid, candle
colors, equity, footprint scales, ...) consumed by the plotting layer. Themes
are selected by name through ``PlotConfig(theme="light" | "dark")``.

Indicators and tools do NOT read the theme for their own colors: those colors
live on the indicator/tool objects (e.g. ``VolumeProfile(buy_color=...)``,
``FixedRangeVP(poc_color=...)``) so they render the same under any theme.

Public API:
    THEMES            : dict name -> token dict
    INDICATOR_COLORS  : default color cycle for auto-colored indicators
    get_theme(name)   : token dict lookup
    LIGHT, DARK       : the built-in theme dicts
"""

from __future__ import annotations

from tradetropy.plotting.theme.light import LIGHT
from tradetropy.plotting.theme.dark import DARK
from tradetropy.plotting.theme.registry import THEMES, INDICATOR_COLORS, get_theme

__all__ = [
    "THEMES",
    "INDICATOR_COLORS",
    "get_theme",
    "LIGHT",
    "DARK",
]

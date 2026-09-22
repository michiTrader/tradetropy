"""
Tool plot configuration. The declarative draw primitives now live in the neutral
module :mod:`tradetropy.ta.draw` (shared by indicators and tools) and are re-exported
here for back-compatibility.

A *tool* (see :class:`tradetropy.ta.tool.base.Tool`) computes a result inside
``on_data()`` and, when it should be drawn, emits a list of those **draw
primitives**. A single generic renderer (``plotting/render/_tools.py``) translates
them into Bokeh glyphs, shared by the static (backtest) and live paths.

This mirrors how indicators declare an :class:`~tradetropy.ta.base.IndicatorPlotConfig`:
a tool declares all of its visual configuration in :class:`ToolPlotConfig` and the
plotting layer never needs tool-specific knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tradetropy.exceptions import ConfigError
from tradetropy.ta.base import LineDash
from tradetropy.ta.draw import (  # re-exported for back-compatibility
    HBars,
    HLines,
    Labels,
    Points,
    Primitive,
    Rects,
    Segments,
)


# =====
# TOOL PLOT CONFIG
# =====


@dataclass
class ToolPlotConfig:
    """
    Visualization configuration for a tool — the tool analogue of
    :class:`~tradetropy.ta.base.IndicatorPlotConfig`.

    Tools declare their defaults by assigning ``self.plot_config`` in ``__init__``;
    ``use_tool(..., **plot_overrides)`` overrides individual fields via
    :meth:`merged`. Every snapshot sharing the same effective ``name`` (or, when
    None, the tool's ``display_name()``) becomes a single, independently toggled
    legend entry.

    Fields:
        plot: False -> the tool's result is not drawn.
        name: Legend label for this tool's glyphs. None -> tool.display_name().
        zorder: Render order hint within the price panel (higher = on top).
        exclude_from_autoscale: True -> glyphs do not participate in the Y-range
            calculation (useful for wide zones that would distort the view).
        color: Primary color. Falls back to theme / fixed defaults when None.
        alpha: Primary opacity.
        line_width: Line width for HLines-based tools.
        line_dash: Line style for HLines-based tools.
        show_labels: Tools that emit Labels honor this toggle.
    """

    plot: bool = True
    name: str | None = None
    zorder: int = 0
    exclude_from_autoscale: bool = False

    color: str | None = None
    alpha: float = 0.9
    line_width: float = 1.6
    line_dash: LineDash = "solid"
    show_labels: bool = True

    def merged(self, **overrides) -> "ToolPlotConfig":
        valid = {f.name for f in self.__dataclass_fields__.values()}
        bad = set(overrides) - valid
        if bad:
            raise ConfigError(
                f"use_tool() received unknown visualization parameters: {bad}. "
                f"Valid parameters: {sorted(valid)}"
            )
        return replace(self, **overrides)

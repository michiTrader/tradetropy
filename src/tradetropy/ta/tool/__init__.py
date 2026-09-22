"""
On-demand analysis tools — the single access point for tools.

Tools are pull operations invoked from inside ``on_data()`` with
``self.use_tool(source, tool, start=, end=)``. Unlike indicators (declared in
``init()`` and precomputed by the engine), a tool reads the source's current
buffer and computes its result immediately; nothing enters the optimize/pool
precompute.

Import tools from here — ``from tradetropy.ta.tool import FixedRangeVP`` — kept
separate from ``tradetropy.ta`` (indicators) on purpose, to avoid confusion.

Creating a tool
---------------
Subclass :class:`Tool`, implement ``run(source, *, start, end)`` returning a
result snapshot, assign a :class:`ToolPlotConfig` in ``__init__`` and implement
``draw(result, cfg)`` returning declarative draw primitives
(:mod:`tradetropy.ta.tool.draw`). A single generic renderer turns those primitives
into Bokeh glyphs for both the static and live paths. See :class:`FixedRangeVP`
and :class:`FibRetracement` for complete examples.
"""

from tradetropy.ta.tool.base import Tool, collect_draw_primitives
from tradetropy.ta.tool.draw import (
    HBars,
    HLines,
    Labels,
    Points,
    Primitive,
    Rects,
    Segments,
    ToolPlotConfig,
)
from tradetropy.ta.tool.fibonacci import DEFAULT_LEVELS, FibResult, FibRetracement
from tradetropy.ta.tool.volume_profile import FixedRangeVP, VolumeProfileResult

__all__ = [
    "Tool",
    "ToolPlotConfig",
    "collect_draw_primitives",
    # draw primitives
    "Primitive",
    "HLines",
    "Segments",
    "Points",
    "HBars",
    "Rects",
    "Labels",
    # volume profile
    "FixedRangeVP",
    "VolumeProfileResult",
    # fibonacci
    "FibRetracement",
    "FibResult",
    "DEFAULT_LEVELS",
]

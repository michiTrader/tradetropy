"""
Declarative draw primitives shared by indicators and tools.

A *drawable* (an indicator or a tool) describes *what* to render in price x time
space by emitting a list of these lightweight, Bokeh-free dataclasses. A single
generic renderer (``plotting/render/_tools.py``) translates the primitives into
Bokeh glyphs, shared by the static (backtest) and live paths, so the plotting
layer never needs drawable-specific knowledge.

Primitives carry parallel arrays (one primitive == one Bokeh glyph), so a
drawable batches all its segments / points / bars into a single primitive
instead of one per element. Timestamps are plain epoch-ms integers; the renderer
converts them to ``datetime64[ms]``.

This module is intentionally free of any plotting or tool dependency so both
``tradetropy.ta`` (indicators) and ``tradetropy.ta.tool`` (tools) can import it without
creating a cycle. ``tradetropy.ta.tool.draw`` re-exports these names for
back-compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from tradetropy.ta.base import LineDash, MarkerType


# =====
# DRAW PRIMITIVES
# =====
#
# Each primitive is one logical Bokeh glyph backed by parallel arrays. The
# optional ``hover`` is a list of (label, column) tuples; when present the
# renderer attaches a HoverTool reading those columns from the primitive's
# ColumnDataSource. ``extra`` carries any additional columns referenced by hover
# tooltips (e.g. volume figures on VP bars).


@dataclass
class HLines:
    """Horizontal segments: a price level spanning [x0, x1] in time (ms)."""
    x0: Sequence[int]
    x1: Sequence[int]
    y: Sequence[float]
    color: Sequence[str] | str
    alpha: Sequence[float] | float = 0.9
    width: float = 1.6
    dash: LineDash = "solid"
    hover: list[tuple[str, str]] | None = None
    extra: dict[str, Sequence] = field(default_factory=dict)


@dataclass
class Segments:
    """
    Arbitrary line segments: (x0, y0) -> (x1, y1) in time (ms) x price.

    Generalizes :class:`HLines` to sloped lines (e.g. swing lines connecting a
    high to a low). Use HLines when every segment is horizontal (cheaper intent).
    """
    x0: Sequence[int]
    y0: Sequence[float]
    x1: Sequence[int]
    y1: Sequence[float]
    color: Sequence[str] | str
    alpha: Sequence[float] | float = 0.9
    width: float = 1.6
    dash: LineDash = "solid"
    hover: list[tuple[str, str]] | None = None
    extra: dict[str, Sequence] = field(default_factory=dict)


@dataclass
class Points:
    """Scatter markers at (x, y) in time (ms) x price."""
    x: Sequence[int]
    y: Sequence[float]
    color: Sequence[str] | str
    alpha: Sequence[float] | float = 0.85
    size: Sequence[float] | int = 6
    marker: MarkerType = "circle"
    fill: bool = True            # False -> hollow markers (border only)
    fill_color: Sequence[str] | str | None = None  # None -> uses ``color``
    fill_alpha: Sequence[float] | float | None = None  # None -> auto based on fill
    line_width: float = 1.0
    hover: list[tuple[str, str]] | None = None
    extra: dict[str, Sequence] = field(default_factory=dict)


@dataclass
class HBars:
    """Horizontal bars (histogram): a band at y from left..right in time (ms)."""
    y: Sequence[float]
    height: Sequence[float]
    left: Sequence[int]
    right: Sequence[int]
    color: Sequence[str] | str
    alpha: Sequence[float] | float = 0.85
    hover: list[tuple[str, str]] | None = None
    extra: dict[str, Sequence] = field(default_factory=dict)


@dataclass
class Rects:
    """Filled rectangles in price x time space: [x0, x1] x [y0, y1] (x in ms)."""
    x0: Sequence[int]
    x1: Sequence[int]
    y0: Sequence[float]
    y1: Sequence[float]
    fill_color: Sequence[str] | str
    fill_alpha: Sequence[float] | float = 0.15
    line_color: Sequence[str] | str | None = None
    line_alpha: Sequence[float] | float = 0.6
    line_width: float = 1.0
    hover: list[tuple[str, str]] | None = None
    extra: dict[str, Sequence] = field(default_factory=dict)


@dataclass
class Labels:
    """Floating text anchored at (x, y) in time (ms) x price."""
    x: Sequence[int]
    y: Sequence[float]
    text: Sequence[str]
    color: Sequence[str] | str = "#888888"
    font_size: str = "9pt"
    x_offset: int = 5
    y_offset: int = 0
    text_align: str = "left"
    text_baseline: str = "middle"
    y_units: str = "data"   # "data" | "screen" (e.g. session labels at bottom)


# Union of every drawable primitive. New primitives must be handled by the
# generic renderer in plotting/render/_tools.py.
Primitive = HLines | Segments | Points | HBars | Rects | Labels

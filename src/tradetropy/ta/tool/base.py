"""
Base contract for on-demand analysis tools.

A *tool* is the pull-operation counterpart of an :class:`Indicator`. Where an
indicator is declared in ``init()`` and precomputed by the engine over the whole
dataset, a tool is invoked inside ``on_data()`` via
``self.use_tool(source, tool, start=, end=)`` and computes a result immediately
from the source's current buffer. Nothing is wired into the engine compute path,
so tools never enter the optimize/pool precompute.

To create a new tool, subclass :class:`Tool` and implement ``run()``. When the
tool should be drawn, also assign a :class:`ToolPlotConfig` in ``__init__`` and
implement ``draw()`` returning a list of declarative
:mod:`~tradetropy.ta.tool.draw` primitives — a single generic renderer turns those
into Bokeh glyphs for both the static (backtest) and live paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradetropy.ta.tool.draw import Primitive, ToolPlotConfig


def _to_ms(value, *, default: int | None) -> int | None:
    """
    Resolve a range bound to an epoch timestamp in milliseconds.

    Shared by range-based tools (``[start, end]``) so each tool resolves bounds
    the same way.

    Args:
        value (int | float | datetime | str | None): The bound. None falls back
            to ``default``. int/float are treated as already-resolved ms.
            datetime / ISO-like strings are parsed as UTC.
        default (int | None): Value returned when ``value`` is None.

    Returns:
        int | None: Timestamp in ms, or ``default`` when value is None.
    """
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value // 1_000_000)


class Tool:
    """
    Base class for on-demand analysis tools used with ``Strategy.use_tool()``.

    Subclasses implement :meth:`run`, which receives the source proxy plus the
    call-time range and returns a result snapshot. Tools hold only configuration
    (no per-run state), so the same instance can be reused across calls.

    When the result should be drawn, assign ``self.plot_config`` in ``__init__``
    and override :meth:`draw` to emit declarative primitives
    (:mod:`tradetropy.ta.tool.draw`). The generic renderer handles all Bokeh work.

    CLASS ATTRIBUTES (override in subclasses)
    -----------------------------------------
    name : str
        Short lowercase identifier. E.g.: "vp", "fib".
    """

    name: str = ""

    @property
    def plot_config(self) -> ToolPlotConfig:
        """
        Visualization config. Subclasses should assign
        ``self.plot_config = ToolPlotConfig(...)`` in ``__init__`` to declare
        their defaults. ``use_tool()`` overrides fields via ``merged()``.
        """
        return getattr(self, "_plot_config", ToolPlotConfig())

    @plot_config.setter
    def plot_config(self, value: ToolPlotConfig):
        self._plot_config = value

    def run(self, source, *, start=None, end=None):  # pragma: no cover - interface
        """Compute the tool over ``source`` for ``[start, end]`` and return a snapshot."""
        raise NotImplementedError

    def draw(self, result, cfg: ToolPlotConfig) -> list[Primitive]:  # pragma: no cover - interface
        """Return declarative draw primitives for ``result`` (or [] if nothing)."""
        return []

    def display_name(self) -> str:
        """Human-readable label for the legend. Used when plot_config.name is None."""
        cls_name = self.name or type(self).__name__
        return cls_name.upper() if len(cls_name) <= 4 else cls_name.capitalize()


def collect_draw_primitives(snapshots: list[dict]) -> dict[str, list[Primitive]]:
    """
    Aggregate stored tool snapshots into primitives grouped by legend entry.

    Walks the ``strategy._tool_snapshots`` list, applies each snapshot's
    per-call overrides to the tool's plot config, asks the tool to draw its
    result, and groups the resulting primitives by effective legend name
    (``cfg.name`` or ``tool.display_name()``). Shared by the static (backtest)
    and live render paths so both aggregate identically.

    Each group becomes one independently toggled legend entry.

    Args:
        snapshots (list[dict]): Items shaped {"tool", "result", "overrides"}.

    Returns:
        dict[str, list[Primitive]]: legend name -> primitives to draw.
    """
    groups: dict[str, list[Primitive]] = {}
    for snap in snapshots:
        tool = snap.get("tool")
        result = snap.get("result")
        if tool is None or not hasattr(tool, "draw"):
            continue
        cfg = getattr(tool, "plot_config", ToolPlotConfig())
        overrides = snap.get("overrides") or {}
        if overrides:
            cfg = cfg.merged(**overrides)
        if not cfg.plot:
            continue
        prims = tool.draw(result, cfg)
        if not prims:
            continue
        legend = cfg.name or tool.display_name()
        groups.setdefault(legend, []).extend(prims)
    return groups

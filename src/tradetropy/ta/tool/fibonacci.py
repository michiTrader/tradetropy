"""
Fixed-range Fibonacci retracement tool.

:class:`FibRetracement` computes Fibonacci retracement levels over an explicit
``[start, end]`` timestamp range. It auto-detects the swing inside the range
(the high and low of the slice) and infers direction from which extreme occurs
last: if the low is more recent than the high the move is treated as bullish
(retracements measured down from the high), otherwise bearish.

Invoked through ``Strategy.use_tool(source, FibRetracement(...), start=, end=)``
inside ``on_data()``. When drawn, it emits one horizontal segment per level plus
an optional price/ratio label, routed through the generic tool renderer.
"""

from __future__ import annotations

import numpy as np

from tradetropy.core.constants import _TICK_COL
from tradetropy.ta.tool.base import Tool, _to_ms
from tradetropy.ta.tool.draw import HLines, Labels, Primitive, ToolPlotConfig


# Standard retracement ratios (TradingView defaults), no extensions.
DEFAULT_LEVELS: tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

# Palette by ratio — falls back to a neutral color for custom levels.
_LEVEL_COLORS: dict[float, str] = {
    0.0:   "#787B86",
    0.236: "#EF5350",
    0.382: "#FF9800",
    0.5:   "#4CAF50",
    0.618: "#26A69A",
    0.786: "#42A5F5",
    1.0:   "#787B86",
}
_FALLBACK_LEVEL_COLOR = "#9598A1"


class FibResult:
    """
    Immutable snapshot returned by :class:`FibRetracement`.

    Falsy when the range is empty.

    Attributes:
        start (int): Effective range start ts in ms.
        end (int): Effective range end ts in ms.
        hi (float): Swing high price in the range.
        lo (float): Swing low price in the range.
        direction (str): 'up' (bullish swing) or 'down' (bearish swing).
        levels (dict[float, float]): ratio -> price for each requested level.
    """

    __slots__ = ("start", "end", "hi", "lo", "direction", "levels", "_empty")

    def __init__(self, data: dict | None):
        if not data:
            self._empty = True
            self.start = self.end = 0
            self.hi = self.lo = float("nan")
            self.direction = ""
            self.levels = {}
            return
        self._empty = False
        self.start = int(data["start"])
        self.end = int(data["end"])
        self.hi = float(data["hi"])
        self.lo = float(data["lo"])
        self.direction = data["direction"]
        self.levels = dict(data["levels"])

    def __bool__(self) -> bool:
        return not self._empty

    def __repr__(self) -> str:
        if self._empty:
            return "FibResult(empty)"
        return (
            f"FibResult(dir={self.direction} hi={self.hi:.2f} lo={self.lo:.2f} "
            f"levels={len(self.levels)})"
        )

    def price(self, ratio: float) -> float:
        """Return the price for a retracement ratio (nan when not present)."""
        return self.levels.get(ratio, float("nan"))


class FibRetracement(Tool):
    """
    Fibonacci retracement over an explicit [start, end] range, on demand.

    Invoked through ``Strategy.use_tool(source, FibRetracement(...), start=, end=)``
    inside ``on_data()``. The swing is auto-detected from the slice: the high and
    low of the range define the move, and direction is inferred from which
    extreme occurs last.

    Both tick (TickProxy) and kline (OhlcProxy) sources are supported; the tool
    reads only the rows currently held in the source's window buffer.

    Args:
        levels (sequence[float]): Retracement ratios to compute and draw.
            Defaults to the standard 0/0.236/0.382/0.5/0.618/0.786/1.0.
        show_labels (bool): Draw a ratio/price label at each level (default True).
    """

    name = "fib"

    def __init__(
        self,
        *,
        levels=DEFAULT_LEVELS,
        show_labels: bool = True,
    ):
        self.levels = tuple(float(r) for r in levels)
        self.plot_config = ToolPlotConfig(name="Fib", show_labels=show_labels)

    def run(self, source, *, start=None, end=None) -> FibResult:
        """
        Compute retracement levels over the source buffer slice [start, end].

        Args:
            source (TickProxy | OhlcProxy): Subscription the slice is read from.
            start: Range start (epoch ms, datetime, ISO str, or None=oldest row).
            end: Range end (epoch ms, datetime, ISO str, or None=newest row).

        Returns:
            FibResult: The snapshot (falsy when the range is empty).
        """
        is_tick = hasattr(source, "price_ref") and "price" in _TICK_COL
        is_tick = is_tick and not hasattr(source, "open_ref")

        ts = np.asarray(source.ts[:], dtype=np.int64)
        n = len(ts)
        if n == 0:
            return FibResult(None)

        start_ms = _to_ms(start, default=None)
        end_ms = _to_ms(end, default=None)
        lo_i = 0 if start_ms is None else int(np.searchsorted(ts, int(start_ms), side="left"))
        hi_i = n if end_ms is None else int(np.searchsorted(ts, int(end_ms), side="right"))
        if hi_i <= lo_i:
            return FibResult(None)

        ts_slice = ts[lo_i:hi_i]
        if is_tick:
            highs = np.asarray(source.price[:], dtype=np.float64)[lo_i:hi_i]
            lows = highs
        else:
            highs = np.asarray(source.high[:], dtype=np.float64)[lo_i:hi_i]
            lows = np.asarray(source.low[:], dtype=np.float64)[lo_i:hi_i]
        if len(highs) == 0:
            return FibResult(None)

        i_hi = int(np.argmax(highs))
        i_lo = int(np.argmin(lows))
        hi = float(highs[i_hi])
        lo = float(lows[i_lo])
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            return FibResult(None)

        # Bullish swing when the low precedes the high (move went up); the
        # retracement is then measured down from the high.
        direction = "up" if ts_slice[i_lo] <= ts_slice[i_hi] else "down"
        span = hi - lo
        levels: dict[float, float] = {}
        for r in self.levels:
            if direction == "up":
                levels[r] = hi - r * span
            else:
                levels[r] = lo + r * span

        return FibResult({
            "start": int(ts_slice[0]),
            "end": int(ts_slice[-1]),
            "hi": hi,
            "lo": lo,
            "direction": direction,
            "levels": levels,
        })

    def draw(self, result: FibResult, cfg: ToolPlotConfig) -> list[Primitive]:
        """Emit one horizontal segment per level plus optional price labels."""
        if not result:
            return []

        ratios = list(result.levels.keys())
        prices = [result.levels[r] for r in ratios]
        colors = [_LEVEL_COLORS.get(r, _FALLBACK_LEVEL_COLOR) for r in ratios]

        prims: list[Primitive] = [HLines(
            x0=[result.start] * len(ratios),
            x1=[result.end] * len(ratios),
            y=prices,
            color=colors,
            alpha=cfg.alpha,
            width=cfg.line_width,
            dash=cfg.line_dash,
        )]

        if cfg.show_labels:
            prims.append(Labels(
                x=[result.end] * len(ratios),
                y=prices,
                text=[f"{r:.3f} ({p:.2f})" for r, p in zip(ratios, prices)],
                color=colors,
            ))

        return prims

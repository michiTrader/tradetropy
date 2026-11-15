"""Strategy-driven rectangle and text annotations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .base import Indicator, IndicatorPlotConfig
from .draw import Labels, Rects


class _ManualAnnotations(Indicator):
    category = "annotation"
    source_cols = ("ts",)
    output_names = ["clock", "count"]
    exposes_query_api = True

    def __init__(self, name: str, limit: int):
        if limit < 0:
            raise ValueError("The annotation limit must be nonnegative")
        self._limit = limit
        self._items: dict = {}
        self.reset_for_replay()
        self.plot_config = IndicatorPlotConfig(
            overlay=True, exclude_from_autoscale=True, renderer="none", name=name,
        )

    @property
    def min_periods(self) -> int:
        return 1

    def reset_for_replay(self) -> None:
        self._items.clear()
        self._next_id = 1
        self._last_ts: int | None = None

    def calculate(self, source: np.ndarray) -> np.ndarray:
        timestamps = source[:, 0] if source.ndim == 2 else source
        out = np.full((2, len(timestamps)), np.nan, dtype=np.float64)
        if len(timestamps):
            self._last_ts = int(timestamps[-1])
            out[1, -1] = len(self._items)
        return out

    def _add(self, item) -> int:
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = item
        if self._limit and len(self._items) > self._limit:
            del self._items[next(iter(self._items))]
        return item_id


@dataclass(frozen=True)
class _Box:
    price0: float
    price1: float
    ts0: int
    ts1: int | None
    color: str
    fill_alpha: float
    line_alpha: float
    line_width: float
    label: str | None


class ManualBoxes(_ManualAnnotations):
    """Draw strategy-defined price zones between two timestamps in ms.

    Attach with ``ManualBoxes.refs(ohlc)``. The resulting strategy handle
    supports ``add_box``, ``update_box``, ``remove_box`` and ``clear_boxes``.
    An open box extends to the latest chart candle. Set ``max_boxes=0`` for
    unlimited storage.
    """

    name = "manualboxes"

    def __init__(self, max_boxes: int = 500):
        super().__init__("Boxes", max_boxes)

    def add_box(self, price0: float, price1: float, ts0: int, ts1: int | None = None, *,
                color: str = "#F59E0B", fill_alpha: float = 0.18,
                line_alpha: float = 0.95, line_width: float = 1.4,
                label: str | None = None) -> int:
        """Add a rectangle and return its ID."""
        return self._add(_Box(price0, price1, ts0, ts1, color, fill_alpha,
                              line_alpha, line_width, label))

    def update_box(self, box_id: int, **fields) -> None:
        if box_id in self._items:
            self._items[box_id] = replace(self._items[box_id], **fields)

    def remove_box(self, box_id: int) -> None:
        self._items.pop(box_id, None)

    def clear_boxes(self) -> None:
        self._items.clear()

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        boxes = list(self._items.values())
        if not boxes:
            return []
        primitives = [Rects(
            x0=[b.ts0 for b in boxes],
            x1=[b.ts1 if b.ts1 is not None else max(b.ts0, self._last_ts or b.ts0)
                for b in boxes],
            y0=[min(b.price0, b.price1) for b in boxes],
            y1=[max(b.price0, b.price1) for b in boxes],
            fill_color=[b.color for b in boxes], fill_alpha=[b.fill_alpha for b in boxes],
            line_color=[b.color for b in boxes], line_alpha=[b.line_alpha for b in boxes],
            line_width=[b.line_width for b in boxes],
        )]
        labeled = [b for b in boxes if b.label]
        if labeled:
            primitives.append(Labels(
                x=[b.ts0 for b in labeled], y=[max(b.price0, b.price1) for b in labeled],
                text=[b.label for b in labeled], color=[b.color for b in labeled],
                x_offset=4, y_offset=-2, text_baseline="top",
            ))
        return primitives


class ManualLabels(_ManualAnnotations):
    """Draw strategy-defined text at a timestamp in ms and a price.

    Attach with ``ManualLabels.refs(ohlc)``. Each label can set a font size,
    alignment and pixel offsets. Set ``max_labels=0`` for unlimited storage.
    """

    name = "manuallabels"

    def __init__(self, max_labels: int = 4000):
        super().__init__("Labels", max_labels)

    def add_label(self, ts: int, price: float, text: str, *, color: str = "#E6E8EB",
                  dy: int = 0, dx: int = 0, align: str = "center",
                  size: str = "9pt") -> int:
        """Add a label and return its ID."""
        return self._add(Labels(x=[ts], y=[price], text=[text], color=color,
                                font_size=size, x_offset=dx, y_offset=dy, text_align=align))

    def remove_label(self, label_id: int) -> None:
        self._items.pop(label_id, None)

    def clear(self) -> None:
        self._items.clear()

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        return list(self._items.values())

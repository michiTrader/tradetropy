"""Indicator wrappers for the SMC block detectors.

The detectors in this package are pure functions over numpy OHLC (easy to test
and to reuse). This module is the thin bridge to the engine:

  * ``blocks_to_rects`` — draw any ``list[Block]`` as borderless zones (fresh
    lighter, mitigated darker), reusing the house style.
  * ``pack_active_bands`` — pack the most-recent unmitigated zone per side into
    the 8-row band layout tradetropy indicators expose, so a Strategy can read
    the live levels in ``on_data``.
  * ``SmcOrderBlock`` — an ``Indicator`` wrapping ``find_order_blocks``.
"""

from __future__ import annotations

import numpy as np

from ..base import Indicator, IndicatorPlotConfig
from ..draw import Rects

from ._core import BEARISH, BULLISH, Block, Pivot
from .order_block import find_order_blocks

_FILL_ALPHA = 0.25       # same fill for fresh & mitigated (a touch lighter than before)
_BORDER_ALPHA = 0.9      # unmitigated get a visible border; mitigated get none
_BORDER_WIDTH = 1.2


def blocks_to_rects(blocks: list[Block], *, bull_color: str, bear_color: str,
                    interval_ms: int | None, show_mitigated: bool = True,
                    extend_mult: int = 20) -> list:
    """Zone boxes. Fresh and mitigated share the same fill opacity; unmitigated
    zones carry a border and mitigated ones do not (the border is what tells them
    apart). Each box starts at the origin candle; a mitigated one stops at its
    mitigation ts, a fresh one extends right."""
    extend_ms = int((interval_ms or 60_000) * extend_mult)
    x0, x1, y0, y1, fills, fas, lcs, las = [], [], [], [], [], [], [], []
    for b in blocks:
        if b.mitigated and not show_mitigated:
            continue
        color = bull_color if b.kind == BULLISH else bear_color
        x0.append(int(b.origin_ts))
        x1.append(int(b.mitigated_ts) if b.mitigated else int(b.origin_ts) + extend_ms)
        y0.append(float(b.bottom))
        y1.append(float(b.top))
        fills.append(color)
        fas.append(_FILL_ALPHA)
        lcs.append(color)
        las.append(0.0 if b.mitigated else _BORDER_ALPHA)
    if not x0:
        return []
    return [Rects(x0=x0, x1=x1, y0=y0, y1=y1, fill_color=fills, fill_alpha=fas,
                  line_color=lcs, line_alpha=las, line_width=_BORDER_WIDTH)]


def pack_active_bands(blocks: list[Block], ts: np.ndarray, n: int) -> np.ndarray:
    """[8 x N]: rows 0-1 bull top/bot, 2-3 bear top/bot, 4-7 the paired ts bands.
    At each bar, emit the most-recent still-unmitigated zone per side.

    Single forward sweep per side (amortised O(N + blocks)) instead of rescanning
    the whole block list for every bar — that O(N x blocks) version ran on every
    recalculation and dominated the backtest. A block is live on
    ``[confirmed_idx, mitigated_idx)``; once it dies it is never live again, so
    dropping it from the tail of the "started" list is safe.
    """
    out = np.full((8, n), np.nan, dtype=np.float64)
    sides = ((BULLISH, 0, 1, 4, 5), (BEARISH, 2, 3, 6, 7))
    for kind, r_top, r_bot, r_open, r_close in sides:
        seq = sorted((b for b in blocks if b.kind == kind),
                     key=lambda b: (b.confirmed_idx, b.origin_idx))
        live: list[Block] = []
        k, nb = 0, len(seq)
        for i in range(n):
            while k < nb and seq[k].confirmed_idx <= i:
                live.append(seq[k])
                k += 1
            while live:
                last = live[-1]
                if last.mitigated_idx is not None and last.mitigated_idx <= i:
                    live.pop()
                else:
                    break
            if live:
                b = live[-1]
                out[r_top, i], out[r_bot, i] = b.top, b.bottom
                out[r_open, i] = b.origin_ts
                out[r_close, i] = b.mitigated_ts if b.mitigated else np.nan
    return out

class SmcOrderBlock(Indicator):
    """Order Block indicator over ``find_order_blocks`` — selectable anchor / zone
    / validation / mitigation. Supersedes the older ``PivotOrderBlock`` (which is
    just ``anchor="pivot_break"``). Same 8-band output as the native OrderBlock.

    source: [N x 6] — ts, open, high, low, close, volume.
    """

    name = "smc_ob"
    category = "annotation"
    source_cols = ("ts", "open", "high", "low", "close", "volume")
    output_names = ["bull_top", "bull_bot", "bear_top", "bear_bot"]
    ts_band_indices = [4, 5, 6, 7]
    ts_output_names = ["ts_bull_open", "ts_bull_close",
                       "ts_bear_open", "ts_bear_close"]

    # Strategy reads the SAME precomputed block list through these queries,
    # addressed by timestamp so they are correct whatever window the engine
    # precomputed over. No second detector run per bar.
    exposes_query_api = True

    def __init__(self, anchor: str = "pivot_break", zone: str = "body",
                 swing: int = 2, break_on: str = "close",
                 validate: tuple[str, ...] = (), fvg_window: int = 3,
                 atr_period: int = 14, atr_mult: float = 1.5,
                 mitigation: str = "close", scan_back: int = 20,
                 min_body_atr: float = 0.0, extend_to_extreme: bool = True,
                 zone_fallback: str | None = None, min_zone_atr: float = 0.0,
                 require_pivot_anchor: bool = False,
                 show_mitigated: bool = True, bull_color: str = "#0ECB81",
                 bear_color: str = "#F6465D"):
        self.anchor = anchor
        self.zone = zone
        self.swing = swing
        self.break_on = break_on
        self.validate = tuple(validate)
        self.fvg_window = fvg_window
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.mitigation = mitigation
        self.scan_back = scan_back
        self.min_body_atr = min_body_atr
        self.extend_to_extreme = extend_to_extreme
        self.zone_fallback = zone_fallback
        self.min_zone_atr = min_zone_atr
        self.require_pivot_anchor = require_pivot_anchor
        self.show_mitigated = show_mitigated
        self.length = swing
        self._bull_color = bull_color
        self._bear_color = bear_color
        self._blocks: list[Block] = []
        self._confirmed_ts = np.empty(0, dtype=np.float64)
        # Query indexes, rebuilt with the block list (they are the reason a
        # query is O(1) instead of a slice copy + Python scan per bar):
        #   _span_index : (kind, span_ts) -> [block indices] ascending
        #   _kind_conf  : kind -> (confirmed_idx array, block index array)
        self._span_index: dict[tuple[str, int], list[int]] = {}
        self._kind_conf: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.plot_config = IndicatorPlotConfig(
            overlay=True, exclude_from_autoscale=True, renderer="rect",
            color=[bull_color, bull_color, bear_color, bear_color], name="SmcOB")

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 2

    def display_name(self) -> str:
        return f"SmcOB({self.anchor},{self.zone})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"smcob_{self.anchor}_{symbol}"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        if n < self.min_periods:
            self._blocks = []
            return np.full((8, n), np.nan, dtype=np.float64)
        self._blocks = find_order_blocks(
            source, anchor=self.anchor, zone=self.zone, swing=self.swing,
            break_on=self.break_on, validate=self.validate,
            fvg_window=self.fvg_window, atr_period=self.atr_period,
            atr_mult=self.atr_mult, mitigation=self.mitigation,
            scan_back=self.scan_back, min_body_atr=self.min_body_atr,
            extend_to_extreme=self.extend_to_extreme,
            zone_fallback=self.zone_fallback, min_zone_atr=self.min_zone_atr,
            require_pivot_anchor=self.require_pivot_anchor)
        # Sorted by confirmed_idx -> confirmed_ts is monotone: one searchsorted
        # gives the causal prefix for any "as of" timestamp.
        self._confirmed_ts = np.fromiter(
            (b.confirmed_ts for b in self._blocks), dtype=np.float64,
            count=len(self._blocks))
        self._build_indexes()
        return pack_active_bands(self._blocks, source[:, 0].astype(np.float64), n)

    def _build_indexes(self) -> None:
        """Index the block list so every query resolves without scanning.

        ``at_span`` used to slice the whole prefix (a copy of up to a thousand
        blocks) and scan it from the end on EVERY bar; ``newest_live`` did the
        same. With the indexes both are O(1) plus the few blocks sharing the
        key, which is what makes the queries safe to call per bar.
        """
        span_index: dict[tuple[str, int], list[int]] = {}
        for i, b in enumerate(self._blocks):
            key = (b.kind, int(round(float(b.meta.get("span_ts", -1.0)))))
            span_index.setdefault(key, []).append(i)
        self._span_index = span_index
        kind_conf: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for kind in (BULLISH, BEARISH):
            idxs = [i for i, b in enumerate(self._blocks) if b.kind == kind]
            kind_conf[kind] = (
                np.fromiter((self._blocks[i].confirmed_idx for i in idxs),
                            dtype=np.int64, count=len(idxs)),
                np.asarray(idxs, dtype=np.int64),
            )
        self._kind_conf = kind_conf

    # ------------------------------------------------------------------
    # Causal queries (called from Strategy.on_data through the proxy)
    # ------------------------------------------------------------------
    def _prefix(self, closed_ts: float) -> int:
        """Number of blocks already confirmed at ``closed_ts`` (inclusive)."""
        if not self._blocks:
            return 0
        return int(np.searchsorted(self._confirmed_ts, float(closed_ts),
                                   side="right"))

    def blocks_upto(self, closed_ts: float) -> list[Block]:
        """Blocks knowable at ``closed_ts`` (confirmation bar included)."""
        return self._blocks[:self._prefix(closed_ts)]

    def at_span(self, span_ts: float, closed_ts: float,
                kind: str = "bearish") -> Block | None:
        """Newest block of ``kind`` whose originating swing has this ts.

        This is the exact matching the strategy used to redo by running the
        detector over the whole window on every bar: same blocks, zero cost.
        """
        hits = self._span_index.get((kind, int(round(float(span_ts)))))
        if not hits:
            return None
        k = self._prefix(closed_ts)
        for idx in reversed(hits):          # newest first
            if idx < k:
                return self._blocks[idx]
        return None

    def newest_live(self, closed_ts: float, *, kind: str = "bearish",
                    above: float | None = None,
                    below: float | None = None) -> Block | None:
        """Newest block of ``kind`` still unmitigated at ``closed_ts``.

        Mitigation only counts once its bar has CLOSED: ``find_mitigation`` can
        flag the forming bar, which is not knowable yet.
        """
        conf, idxs = self._kind_conf.get(kind, (None, None))
        if conf is None or not conf.size:
            return None
        k = self._prefix(closed_ts)
        pos = int(np.searchsorted(conf, k, side="left"))   # first idx >= k
        for j in range(pos - 1, -1, -1):
            b = self._blocks[int(idxs[j])]
            if b.mitigated_ts is not None and b.mitigated_ts <= closed_ts:
                continue
            if above is not None and b.bottom <= above:
                continue
            if below is not None and b.top >= below:
                continue
            return b
        return None

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        return blocks_to_rects(
            self._blocks, bull_color=self._bull_color, bear_color=self._bear_color,
            interval_ms=interval_ms, show_mitigated=self.show_mitigated)



from __future__ import annotations

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.structure._utils import extract_confirmed_pivots
from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL MAPS
# ══════════════════════════════════════════════════════════════════════════════

_NBS_PRICE_IDX: dict[str, int] = {
    "H-neu": 0, "L-neu": 1,
    "H-boo": 2, "L-boo": 3,
    "H-shk": 4, "L-shk": 5,
    "H-emp": 6, "L-emp": 7,
}
_NBS_TS_IDX: dict[str, int] = {k: v + 8 for k, v in _NBS_PRICE_IDX.items()}

_NBS_TAG_CODE: dict[str, float] = {
    "H-neu": 0.0, "L-neu": 1.0,
    "H-boo": 2.0, "L-boo": 3.0,
    "H-shk": 4.0, "L-shk": 5.0,
    "H-emp": 6.0, "L-emp": 7.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# NBS — Neutralizer / Booster / Shaker
# ══════════════════════════════════════════════════════════════════════════════

class NBS(Indicator, PivotIndicatorMixin):
    """
    Classifies each confirmed pivot as Neutralizer, Booster, Shaker or Empty.

    Output — [17 × N] float64:
        b0..b7   : pivot price by type (on confirmation bar)
                   b0=neu_high  b1=neu_low
                   b2=boo_high  b3=boo_low
                   b4=shk_high  b5=shk_low
                   b6=emp_high  b7=emp_low
        b8..b15  : pivot ts_real by type (timestamp of the real swing bar)
                   same order as b0..b7
        b16      : encoded tag (internal — PatternMatcher / not exposed to user)
                   0=H-neu 1=L-neu 2=H-boo 3=L-boo 4=H-shk 5=L-shk 6=H-emp 7=L-emp

    Access in on_data()
    ────────────────────
        self.nbs.neu_high[-1]   → price of last confirmed NEU-H (or NaN)
        self.nbs.boo_low[-1]    → price of last confirmed BOO-L (or NaN)
        self.nbs[8][-1]         → ts_real of last NEU-H (internal / plotting)

    NBS Algorithm
    ─────────────
    Maintains two pools (pool_H, pool_L) of active points.

    Forced initialization:
        pivot_0 → BOO, enters its type pool
        pivot_1 → NEU, enters its type pool

    For each pivot_i (i >= 2):
        1. Does it surpass any point in its pool?
           H surpasses if new_value >= point_value
           L surpasses if new_value <= point_value
        2. Does not surpass → EMP, does not enter the pool.
        3. Surpasses → NEU:
           a. Clean all surpassed points from the pool.
           b. Promote the immediately preceding opposite pivot:
              - Was NEU → SHK
              - Was EMP → BOO
              - Was BOO or SHK → no change
           c. The new NEU enters the pool.

    source : [N × 3] — high(0), low(1), ts_ms(2)

    PatternMatcher
    ──────────────
    Implements PivotIndicatorMixin. Use as decorator in add_pattern_matcher():

        self.setup = self.add_pattern_matcher(
            base_pivot=self.cpivot,
            decorators=[self.nbs],
            pattern = Pattern([
                PatternNode('H', {'nbs': 'neu'}, []),
                PatternNode('L', {'nbs': 'boo'}, []),
                PatternNode('H', {'nbs': 'neu'}, [Condition('>', NodeRef(0, 'value'))]),
            ], tag="setup")
        )

        Available tags: 'neu' | 'boo' | 'shk' | 'emp'
    """

    name     = "nbs"
    category = "structure"
    source_cols = ("high", "low", "ts")

    use_partial = False

    output_names = [
        "neu_high", "neu_low",
        "boo_high", "boo_low",
        "shk_high", "shk_low",
        "emp_high", "emp_low",
    ]
    ts_band_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    ts_output_names = [
        "ts_neu_high", "ts_neu_low",
        "ts_boo_high", "ts_boo_low",
        "ts_shk_high", "ts_shk_low",
        "ts_emp_high", "ts_emp_low",
        "_tag",
    ]

    tag_name      = "nbs"
    is_base_pivot = False

    TAG_DECODE: dict[float, str] = {
        0.0: "H-neu",  1.0: "L-neu",
        2.0: "H-boo",  3.0: "L-boo",
        4.0: "H-shk",  5.0: "L-shk",
        6.0: "H-emp",  7.0: "L-emp",
    }

    def pivot_col_names(self, symbol: str) -> tuple[str, ...]:
        base = self.col_name(symbol)
        return (f"{base}_b16",)

    def __init__(self, swing: int = 2):
        self.swing  = swing
        self.length = swing

        self.plot_config = IndicatorPlotConfig(
            overlay  = True,
            renderer = "scatter",
            color = [
                "#8B5CF6", "#8B5CF6",
                "#0ECB81", "#0ECB81",
                "#F6465D", "#F6465D",
                "#6B7280", "#6B7280",
            ],
            marker = [
                "triangle",          "inverted_triangle",
                "triangle",          "inverted_triangle",
                "triangle",          "inverted_triangle",
                "triangle",          "inverted_triangle",
            ],
            marker_size  = 7,
            marker_alpha = 0.85,
            exclude_from_autoscale = True,
            name = "NBS",
        )

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 1

    def display_name(self) -> str:
        return f"NBS({self.swing})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"nbs{self.swing}_{symbol}"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n  = len(source)
        out = np.full((16, n), np.nan, dtype=np.float64)

        if n < self.swing * 2 + 1:
            return np.vstack([out, np.full(n, np.nan, dtype=np.float64)])

        pivots = extract_confirmed_pivots(source, self.swing, use_ts_fallback=True)

        if len(pivots) < 2:
            return np.vstack([out, np.full(n, np.nan, dtype=np.float64)])

        tagged = self._run_nbs(pivots)

        TAG_TO_PRICE_BAND = {
            'H-neu': 0, 'L-neu': 1,
            'H-boo': 2, 'L-boo': 3,
            'H-shk': 4, 'L-shk': 5,
            'H-emp': 6, 'L-emp': 7,
        }
        TAG_TO_TS_BAND = {k: v + 8 for k, v in TAG_TO_PRICE_BAND.items()}

        tag_band = np.full(n, np.nan, dtype=np.float64)

        for bar_idx, ptype, value, ts_real, tag in tagged:
            price_band = TAG_TO_PRICE_BAND.get(tag)
            ts_band    = TAG_TO_TS_BAND.get(tag)
            if price_band is None:
                continue
            out[price_band, bar_idx] = value
            out[ts_band,    bar_idx] = ts_real
            tag_band[bar_idx] = _NBS_TAG_CODE.get(tag, np.nan)

        return np.vstack([out, tag_band])

    def _run_nbs(
        self,
        pivots: list[tuple[int, str, float, float]],
    ) -> list[tuple[int, str, float, float, str]]:
        result: dict[int, tuple[int, str, float, float, str]] = {}

        pool_H: list[list] = []
        pool_L: list[list] = []

        def _set_tag(bar_idx: int, new_tag: str) -> None:
            old = result[bar_idx]
            result[bar_idx] = (old[0], old[1], old[2], old[3], new_tag)

        p0, p1 = pivots[0], pivots[1]
        result[p0[0]] = (p0[0], p0[1], p0[2], p0[3], f"{p0[1]}-boo")
        result[p1[0]] = (p1[0], p1[1], p1[2], p1[3], f"{p1[1]}-neu")

        (pool_H if p0[1] == 'H' else pool_L).append([p0[0], p0[2], p0[1], p0[2], p0[3]])
        (pool_H if p1[1] == 'H' else pool_L).append([p1[0], p1[2], p1[1], p1[2], p1[3]])

        for p_idx in range(2, len(pivots)):
            bar_idx, ptype, value, ts_real = pivots[p_idx]
            pool = pool_H if ptype == 'H' else pool_L

            surpassed = (
                [pt for pt in pool if value >= pt[1]]
                if ptype == 'H'
                else [pt for pt in pool if value <= pt[1]]
            )

            if not surpassed:
                result[bar_idx] = (bar_idx, ptype, value, ts_real, f"{ptype}-emp")
                continue

            result[bar_idx] = (bar_idx, ptype, value, ts_real, f"{ptype}-neu")

            if ptype == 'H':
                pool_H[:] = [pt for pt in pool_H if value < pt[1]]
            else:
                pool_L[:] = [pt for pt in pool_L if value > pt[1]]

            opposite = 'L' if ptype == 'H' else 'H'
            for q_idx in range(p_idx - 1, -1, -1):
                if pivots[q_idx][1] == opposite:
                    pc = pivots[q_idx]
                    pc_entry = result.get(pc[0])
                    if pc_entry is None:
                        break
                    pc_tag = pc_entry[4]

                    if pc_tag == f"{opposite}-neu":
                        _set_tag(pc[0], f"{opposite}-shk")
                        pc_pool = pool_H if opposite == 'H' else pool_L
                        if not any(pt[0] == pc[0] for pt in pc_pool):
                            pc_pool.append([pc[0], pc[2], pc[1], pc[2], pc[3]])

                    elif pc_tag == f"{opposite}-emp":
                        _set_tag(pc[0], f"{opposite}-boo")
                        pc_pool = pool_H if opposite == 'H' else pool_L
                        pc_pool.append([pc[0], pc[2], pc[1], pc[2], pc[3]])
                    break

            pool = pool_H if ptype == 'H' else pool_L
            pool.append([bar_idx, value, ptype, value, ts_real])

        return list(result.values())
        

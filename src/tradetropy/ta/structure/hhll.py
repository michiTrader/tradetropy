from __future__ import annotations

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.structure._utils import extract_confirmed_pivots
from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL MAPS
# ══════════════════════════════════════════════════════════════════════════════

_HHLL_PRICE_IDX: dict[str, int] = {"HH": 0, "HL": 1, "LH": 2, "LL": 3}
_HHLL_TS_IDX: dict[str, int] = {k: v + 4 for k, v in _HHLL_PRICE_IDX.items()}

_HHLL_TAG_CODE: dict[str, float] = {
    "HH": 0.0, "HL": 1.0, "LH": 2.0, "LL": 3.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# HHLL — Higher High / Higher Low / Lower High / Lower Low
# ══════════════════════════════════════════════════════════════════════════════

class HHLL(Indicator, PivotIndicatorMixin):
    """
    Classifies each confirmed pivot by comparing it to the previous one of the same type.

    Output — [9 × N] float64:
        b0..b3  : price by type (on confirmation bar)
                  b0=hh  b1=hl  b2=lh  b3=ll
        b4..b7  : ts_real by type (timestamp of the real pivot bar)
                  same order: hh_ts, hl_ts, lh_ts, ll_ts
        b8      : encoded tag (internal — PatternMatcher / not exposed to user)
                  0=HH  1=HL  2=LH  3=LL

    Access in on_data()
    ────────────────────
        self.hhll.hh[-1]   → price of last confirmed HH (or NaN)
        self.hhll.hl[-1]   → price of last confirmed HL (or NaN)
        self.hhll.lh[-1]   → price of last confirmed LH (or NaN)
        self.hhll.ll[-1]   → price of last confirmed LL (or NaN)
        self.hhll[4][-1]   → ts_real of last HH (internal / plotting)
        self.hhll[5][-1]   → ts_real of last HL

    Classifications
    ───────────────
        HH — High >= previous High    LH — High < previous High
        HL — Low  >= previous Low     LL — Low  < previous Low

    source : [N × 3] — high(0), low(1), ts_ms(2)

    PatternMatcher
    ──────────────
    Implements PivotIndicatorMixin. Use as decorator in add_pattern_matcher():

        self.setup = self.add_pattern_matcher(
            pivots  = [self.cpivot, self.hhll],
            pattern = Pattern([
                PatternNode('H', {'hhll': 'HH'}, []),
                PatternNode('L', {'hhll': 'HL'}, []),
                PatternNode('H', {'hhll': 'HH'}, [Condition('>', NodeRef(0, 'value'))]),
            ], tag="hh_hl_hh")
        )

        Available tags: 'HH' | 'HL' | 'LH' | 'LL'
    """

    name     = "hhll"
    category = "structure"

    use_partial = False

    tag_name      = "hhll"
    is_base_pivot = False

    TAG_DECODE: dict[float, str] = {
        0.0: "HH",
        1.0: "HL",
        2.0: "LH",
        3.0: "LL",
    }

    def pivot_col_names(self, symbol: str) -> tuple[str, ...]:
        base = self.col_name(symbol)
        return (f"{base}_b8",)

    def __init__(self, swing: int = 2):
        self.swing  = swing
        self.length = swing

        self.output_names    = ["hh", "hl", "lh", "ll"]
        self.ts_band_indices = [4, 5, 6, 7, 8]
        self.ts_output_names = ["ts_hh", "ts_hl", "ts_lh", "ts_ll", "_tag"]

        self.plot_config = IndicatorPlotConfig(
            overlay  = True,
            renderer = "scatter",
            color = [
                "#0ECB81",
                "#26A69A",
                "#EF5350",
                "#F6465D",
            ],
            marker = [
                "triangle",
                "inverted_triangle",
                "triangle",
                "inverted_triangle",
            ],
            marker_size  = 6,
            marker_alpha = 0.85,
            exclude_from_autoscale = True,
            name = "HHLL",
        )

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 1

    def display_name(self) -> str:
        return f"HHLL({self.swing})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"hhll{self.swing}_{symbol}"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n   = len(source)
        out = np.full((8, n), np.nan, dtype=np.float64)

        if n < self.min_periods:
            return np.vstack([out, np.full(n, np.nan, dtype=np.float64)])

        high = source[:, 0].astype(np.float64)
        low  = source[:, 1].astype(np.float64)

        pivots = extract_confirmed_pivots(source, self.swing, use_ts_fallback=False)

        tag_band = np.full(n, np.nan, dtype=np.float64)

        if not pivots:
            return np.vstack([out, tag_band])

        last_h: float | None = None
        last_l: float | None = None

        for bar_idx, ptype, value, ts_real in pivots:
            if ptype == 'H':
                if last_h is None:
                    last_h = value
                else:
                    tag = "HH" if value >= last_h else "LH"
                    price_idx = _HHLL_PRICE_IDX[tag]
                    ts_idx    = _HHLL_TS_IDX[tag]
                    out[price_idx, bar_idx] = value
                    out[ts_idx,    bar_idx] = ts_real
                    tag_band[bar_idx]       = _HHLL_TAG_CODE[tag]
                    last_h = value
            else:
                if last_l is None:
                    last_l = value
                else:
                    tag = "HL" if value >= last_l else "LL"
                    price_idx = _HHLL_PRICE_IDX[tag]
                    ts_idx    = _HHLL_TS_IDX[tag]
                    out[price_idx, bar_idx] = value
                    out[ts_idx,    bar_idx] = ts_real
                    tag_band[bar_idx]       = _HHLL_TAG_CODE[tag]
                    last_l = value

        return np.vstack([out, tag_band])
        

import numpy as np
from typing import Protocol
from dataclasses import dataclass, field

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin


# ══════════════════════════════════════════════════════════════════════════════
# PivotHighLow  (multi-source, multi-band, lookahead — visual only)
# ══════════════════════════════════════════════════════════════════════════════
class PivotHighLow(Indicator):
    """
    Detects pivots with a symmetric ±n window, CAUSAL: the value appears on
    the confirmation bar (ci + n), not on the candidate bar (ci).

    Returns [4 × N]:
        row 0 : high  — PH price on the confirmation bar (NaN if none)
        row 1 : low   — PL price on the confirmation bar (NaN if none)
        row 2 : _ph_ts  — timestamp of the REAL PH bar (internal — plotting)
        row 3 : _pl_ts  — timestamp of the REAL PL bar (internal — plotting)

    The last n bars are always NaN (insufficient right-side window).

    source : [N × 3] — high(0), low(1), ts_ms(2)

    Behavior in "real time" (bar-by-bar backtest):
        idx 5 (n=2): ci=3 doesn't have 2 closed bars to the right yet → nan
        idx 6 (n=2): ci=3 now has 2 bars → ph[6]=752.0, ph_ts[6]=ts(3)
        idx 7 (n=2): no new pivot → ph[7]=nan

    Usage:
        self.pv = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
            PivotHighLow(n=2),
            plot=True, overlay=True,
            color=["#F6465D", "#0ECB81"],
        )

    Access in on_data():
        self.pv.high[-1]  → price of PH confirmed on this bar (or NaN)
        self.pv.low[-1]   → price of PL confirmed on this bar (or NaN)
    """
    name            = "pivot"
    category        = "structure"
    source_cols     = ("high", "low", "ts")
    output_names    = ["high", "low"]
    ts_band_indices = [2, 3]
    ts_output_names = ["ts_high", "ts_low"]

    def __init__(self, n: int = 3):
        self.n      = n
        self.length = n
        self.plot_config = IndicatorPlotConfig(
            renderer="scatter",
            marker="triangle",
            marker_size=9,
            color=["#F6465D", "#0ECB81"],
        )

    @property
    def min_periods(self) -> int:
        return self.n * 2 + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        source : [N × 3] — high(0), low(1), ts_ms(2). Returns [4 × N].

        The value is placed on the confirmation bar ci+n, not on ci.
        ts_real points to the timestamp of the original candidate bar (ci).
        """
        n     = len(source)
        K     = self.n
        ph    = np.full(n, np.nan, dtype=np.float64)
        pl    = np.full(n, np.nan, dtype=np.float64)
        ph_ts = np.full(n, np.nan, dtype=np.float64)
        pl_ts = np.full(n, np.nan, dtype=np.float64)

        if n < K * 2 + 1:
            return np.vstack([ph, pl, ph_ts, pl_ts])

        high  = source[:, 0].astype(np.float64)
        low   = source[:, 1].astype(np.float64)
        ts_ms = source[:, 2].astype(np.float64)

        for ci in range(K, n - K):
            confirm_idx = ci + K

            window_h = high[ci - K : ci + K + 1]
            if high[ci] == np.max(window_h):
                ph[confirm_idx]    = high[ci]
                ph_ts[confirm_idx] = ts_ms[ci]

            window_l = low[ci - K : ci + K + 1]
            if low[ci] == np.min(window_l):
                pl[confirm_idx]    = low[ci]
                pl_ts[confirm_idx] = ts_ms[ci]

        return np.vstack([ph, pl, ph_ts, pl_ts])


# ══════════════════════════════════════════════════════════════════════════════
# ConfirmedPivot  (multi-source, multi-band, causal — trading)
# ══════════════════════════════════════════════════════════════════════════════
class ConfirmedPivot(Indicator, PivotIndicatorMixin):
    """
    Pivot High/Low confirmed by price break. Strictly causal.

    Returns 4 internal series — the user only accesses the 2 price ones:

        self.pv.high[-1]  → price of the last confirmed PH (or NaN)
        self.pv.low[-1]   → price of the last confirmed PL (or NaN)

    The other 2 series (timestamps of the real pivot bar) are internal:
    plotting uses them to relocate markers visually, but the user
    does not need and should not access them.

    source : [N × 3] — high(0), low(1), ts_ms(2)

    returns : [4 × N]
        row 0 : high     — PH price on the confirmation bar
        row 1 : low      — PL price on the confirmation bar
        row 2 : _high_ts — ts_ms of the real PH bar  [internal]
        row 3 : _low_ts  — ts_ms of the real PL bar  [internal]

    PivotIndicatorMixin
    ───────────────────
    Implements tag_name, is_base_pivot and pivot_col_names() so that
    PatternMatcher can build the FrozenPivotSequence from this indicator
    without direct coupling.
    """

    name            = "cpivot"
    category        = "structure"
    source_cols     = ("high", "low", "ts")
    ts_band_indices = [2, 3]
    ts_output_names = ["ts_high", "ts_low"]
    output_names    = ["high", "low"]
    use_partial = False
    tag_name      = "type"
    is_base_pivot = True

    def __init__(self, swing: int = 2):
        self.swing  = swing
        self.length = swing
        self.plot_config = IndicatorPlotConfig(
            renderer="scatter",
            marker="circle",
            marker_size=6,
            color=["#D39414", "#44D314FF"],
            marker_alpha=0.8,
        )

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 1

    def display_name(self) -> str:
        return f"cpivot"

    def pivot_col_names(self, symbol: str) -> tuple[str, str, str, str]:
        base = self.col_name(symbol)
        return (
            f"{base}_b0",
            f"{base}_b1",
            f"{base}_b2",
            f"{base}_b3",
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n  = len(source)
        S  = self.swing
        ph    = np.full(n, np.nan, dtype=np.float64)
        pl    = np.full(n, np.nan, dtype=np.float64)
        ph_ts = np.full(n, np.nan, dtype=np.float64)
        pl_ts = np.full(n, np.nan, dtype=np.float64)

        if n < S * 2 + 1:
            return np.vstack([ph, pl, ph_ts, pl_ts])

        high  = source[:, 0].astype(np.float64)
        low   = source[:, 1].astype(np.float64)
        ts_ms = source[:, 2].astype(np.float64)

        last_type : str | None = None
        cand_type : str | None = None
        cand_h    : float = np.nan
        cand_l    : float = np.nan
        cand_ts   : float = np.nan
        next_scan : int   = S

        for i in range(S, n):
            if cand_type is not None:
                if cand_type == 'H':
                    if low[i] < cand_l:
                        ph[i]    = cand_h
                        ph_ts[i] = cand_ts
                        last_type = 'H'
                        cand_type = None
                    elif high[i] > cand_h:
                        cand_type = None
                else:
                    if high[i] > cand_h:
                        pl[i]    = cand_l
                        pl_ts[i] = cand_ts
                        last_type = 'L'
                        cand_type = None
                    elif low[i] < cand_l:
                        cand_type = None

            if cand_type is None:
                ci = i - S
                if ci >= next_scan:
                    next_scan = ci + 1
                    if ci >= S:
                        h_ci    = high[ci]
                        l_ci    = low[ci]
                        left_h  = high[ci - S : ci]
                        left_l  = low[ci - S : ci]
                        right_h = high[ci + 1 : i + 1]
                        right_l = low[ci + 1 : i + 1]

                        if len(left_h) >= S and len(right_h) >= S:
                            is_ph = (h_ci >= np.max(left_h)) and (h_ci > np.max(right_h))
                            is_pl = (l_ci <= np.min(left_l)) and (l_ci < np.min(right_l))

                            if is_ph and last_type != 'H':
                                confirmed_at = None
                                cancelled = False
                                for k in range(ci + 1, i + 1):
                                    if low[k] < l_ci:
                                        confirmed_at = k
                                        break
                                    if high[k] > h_ci:
                                        cancelled = True
                                        break
                                if confirmed_at is not None:
                                    # Anchor at i = ci + S (bar that completes the
                                    # swing = where the pivot becomes causally
                                    # knowable), not at the break bar
                                    # (confirmed_at), which would be lookahead.
                                    ph[i]    = h_ci
                                    ph_ts[i] = ts_ms[ci]
                                    last_type = 'H'
                                elif not cancelled:
                                    cand_type = 'H'
                                    cand_h    = h_ci
                                    cand_l    = l_ci
                                    cand_ts   = ts_ms[ci]

                            elif is_pl and last_type != 'L':
                                confirmed_at = None
                                cancelled = False
                                for k in range(ci + 1, i + 1):
                                    if high[k] > h_ci:
                                        confirmed_at = k
                                        break
                                    if low[k] < l_ci:
                                        cancelled = True
                                        break
                                if confirmed_at is not None:
                                    # Anchor at i = ci + S (swing-complete bar),
                                    # not at the break bar (lookahead).
                                    pl[i]    = l_ci
                                    pl_ts[i] = ts_ms[ci]
                                    last_type = 'L'
                                elif not cancelled:
                                    cand_type = 'L'
                                    cand_h    = h_ci
                                    cand_l    = l_ci
                                    cand_ts   = ts_ms[ci]

        return np.vstack([ph, pl, ph_ts, pl_ts])


# ══════════════════════════════════════════════════════════════════════════════
# PIVOT DETECTOR PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
class PivotDetector(Protocol):
    """
    Protocol that any pivot indicator must implement to be used
    as a backend for ZigZag.

    calculate must return [K × N] where:
      - row 0: pivot high prices (NaN where no pivot)
      - row 1: pivot low  prices (NaN where no pivot)
      - rows 2+: internal auxiliary bands (ts, etc.) — ignored by ZigZag
    """

    def calculate(self, source: np.ndarray) -> np.ndarray: ...


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def _collapse_pivots_to_zigzag(
    ph: np.ndarray,
    pl: np.ndarray,
    ph_ts: "np.ndarray | None" = None,
    pl_ts: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray | None]":
    """
    Collapses two pivot high/low series into a single single-band zigzag series,
    forcibly alternating H→L→H→L.
    """
    n      = len(ph)
    prices = np.full(n, np.nan, dtype=np.float64)
    has_ts = ph_ts is not None and pl_ts is not None
    ts_out = np.full(n, np.nan, dtype=np.float64) if has_ts else None
    last: str | None = None

    for i in range(n):
        has_h = not np.isnan(ph[i])
        has_l = not np.isnan(pl[i])

        chosen: str | None = None
        if has_h and has_l:
            chosen = 'H' if last != 'H' else ('L' if last != 'L' else None)
        elif has_h and last != 'H':
            chosen = 'H'
        elif has_l and last != 'L':
            chosen = 'L'

        if chosen == 'H':
            prices[i] = ph[i]
            if has_ts:
                ts_out[i] = ph_ts[i]
            last = 'H'
        elif chosen == 'L':
            prices[i] = pl[i]
            if has_ts:
                ts_out[i] = pl_ts[i]
            last = 'L'

    return prices, ts_out
    

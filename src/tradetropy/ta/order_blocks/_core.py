"""Shared primitives for the SMC block detectors — pure, causal, engine-agnostic.

Everything here operates on plain numpy OHLC arrays and returns structured
``Block`` objects. There is no Strategy/engine dependency (only the shared
``extract_confirmed_pivots`` helper), so the package can be unit tested
directly against plain arrays.

Causality contract
------------------
Every detector stamps each ``Block`` with ``confirmed_idx`` — the bar index at
which the block first becomes knowable from *closed* bars only. No detector uses
information from bars after ``confirmed_idx`` to decide a block's existence or
geometry. Mitigation only *consumes* a block, so it may be stamped on a later
bar; that is the one field allowed to change as new bars close. This is the same
anti-lookahead rule ``MarketStructure`` follows.

The design separates three orthogonal axes (anchor / zone / mitigation) plus a
set of optional validation filters, all on ONE order-block detector, and keeps
the genuinely different families (breaker, rejection, propulsion, vacuum) as
their own detectors that reuse these primitives. See ``README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..structure._utils import extract_confirmed_pivots

BULLISH = "bullish"
BEARISH = "bearish"

# Block families
ORDER = "order"
BREAKER = "breaker"
REJECTION = "rejection"
PROPULSION = "propulsion"
VACUUM = "vacuum"

@dataclass(frozen=True)
class Pivot:
    """A confirmed swing pivot. ``conf_idx`` is where it became knowable;
    ``real_idx`` is the bar of the actual extreme (never used to backdate)."""

    kind: str          # "high" | "low"
    price: float
    conf_idx: int
    real_idx: int
    ts: float


@dataclass(frozen=True)
class Block:
    """A marked zone, shared by every family. ``meta`` carries family-specific
    extras (e.g. the FVG that validated it, the swept level, a mean threshold)."""

    kind: str          # BULLISH | BEARISH
    family: str        # ORDER | BREAKER | REJECTION | PROPULSION | VACUUM
    top: float
    bottom: float
    origin_idx: int
    origin_ts: float
    confirmed_idx: int
    confirmed_ts: float
    mitigated_idx: int | None = None
    mitigated_ts: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def mean_threshold(self) -> float:
        """Consequent-encroachment / 50% level of the zone."""
        return (self.top + self.bottom) / 2.0

    @property
    def proximal(self) -> float:
        """Edge price meets first on return (demand: top, supply: bottom)."""
        return self.bottom if self.kind == BEARISH else self.top

    @property
    def distal(self) -> float:
        return self.top if self.kind == BEARISH else self.bottom

    @property
    def mitigated(self) -> bool:
        return self.mitigated_idx is not None

    def with_mitigation(self, idx: int, ts: float) -> "Block":
        return replace(self, mitigated_idx=idx, mitigated_ts=ts)

def ts_index_map(ts: np.ndarray) -> dict[float, int]:
    return {float(t): i for i, t in enumerate(ts)}


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Wilder ATR. NaN until ``period`` bars exist. Strictly causal."""
    tr = true_range(high, low, close)
    out = np.full(len(tr), np.nan, dtype=np.float64)
    if len(tr) >= period:
        out[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def zone_of(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
            close: np.ndarray, idx: int, mode: str = "body",
            kind: str | None = None) -> tuple[float, float]:
    """(top, bottom) for a single candle.

        body   open..close.
        range  high..low (body + both wicks).
        wick   body on the PROXIMAL side, wick extreme on the DISTAL side —
               needs ``kind`` (demand reaches the lower wick, supply the upper).
    """
    o, c = open_[idx], close[idx]
    body_top, body_bot = float(max(o, c)), float(min(o, c))
    if mode == "range":
        return float(high[idx]), float(low[idx])
    if mode == "wick":
        if kind == BEARISH:                    # supply: distal = top -> upper wick
            return float(high[idx]), body_bot
        return body_top, float(low[idx])       # demand: distal = bottom -> lower wick
    return body_top, body_bot


def extend_to_span_extreme(top: float, bottom: float, kind: str,
                           high: np.ndarray, low: np.ndarray, a: int,
                           b: int) -> tuple[float, float]:
    """Push the DISTAL edge to the extreme of the span ``[a, b]`` (origin ->
    originating swing), so a body-only zone still covers the swing low/high that
    created the move. Demand extends the bottom down; supply extends the top up."""
    lo, hi = (a, b) if a <= b else (b, a)
    lo = max(0, lo)
    hi = min(len(high) - 1, hi)
    if hi < lo:
        return top, bottom
    if kind == BEARISH:
        return max(top, float(high[lo:hi + 1].max())), bottom
    return top, min(bottom, float(low[lo:hi + 1].min()))


def apply_min_height(top: float, bottom: float,
                     min_height: float) -> tuple[float, float]:
    """Inflate a degenerate zone symmetrically around its mid to a floor, so a
    doji-thin OB can never produce a zero-width band / zero stop distance."""
    if min_height <= 0 or (top - bottom) >= min_height:
        return top, bottom
    mid = (top + bottom) / 2.0
    half = min_height / 2.0
    return mid + half, mid - half



def confirmed_pivots(high: np.ndarray, low: np.ndarray, ts: np.ndarray,
                     swing: int = 2) -> list[Pivot]:
    """Wrap tradetropy's causal confirmed-pivot extractor into ``Pivot`` objects
    (same ``swing`` semantics as ConfirmedPivot / MarketStructure)."""
    raw = extract_confirmed_pivots(
        np.column_stack([high, low, ts]), swing, use_ts_fallback=False)
    tsmap = ts_index_map(ts)
    out: list[Pivot] = []
    for bar_idx, ptype, price, ts_real in raw:
        out.append(Pivot(
            kind="high" if ptype == "H" else "low",
            price=float(price), conf_idx=int(bar_idx),
            real_idx=int(tsmap.get(float(ts_real), int(bar_idx))),
            ts=float(ts_real)))
    return out

def find_fvgs(high: np.ndarray, low: np.ndarray, min_gap: float = 0.0) -> list[dict]:
    """3-candle Fair Value Gaps. A bullish FVG is ``low[i+1] > high[i-1]`` (gap
    around the displacement candle ``i``); bearish is the mirror. ``confirmed_idx``
    is ``i + 1`` — the bar that closes the triplet and makes the gap knowable."""
    out: list[dict] = []
    n = len(high)
    for i in range(1, n - 1):
        up = low[i + 1] - high[i - 1]
        dn = low[i - 1] - high[i + 1]
        if up > min_gap:
            out.append({"kind": BULLISH, "mid_idx": i, "top": float(low[i + 1]),
                        "bottom": float(high[i - 1]), "confirmed_idx": i + 1})
        elif dn > min_gap:
            out.append({"kind": BEARISH, "mid_idx": i, "top": float(low[i - 1]),
                        "bottom": float(high[i + 1]), "confirmed_idx": i + 1})
    return out


def fvg_after(fvgs: list[dict], origin_idx: int, kind: str,
              window: int) -> dict | None:
    """First FVG of direction ``kind`` whose displacement candle sits within
    ``[origin_idx, origin_idx + window]`` — i.e. the imbalance belongs to the leg
    leaving the block, not an unrelated gap later in the move. The block candle
    itself can be the ``i-1`` leg, so the range starts at ``origin_idx``."""
    lo, hi = origin_idx, origin_idx + window
    for f in fvgs:
        if f["kind"] == kind and lo <= f["mid_idx"] <= hi:
            return f
    return None


def find_mitigation(top: float, bottom: float, kind: str, high: np.ndarray,
                    low: np.ndarray, close: np.ndarray, start_idx: int,
                    mode: str = "close") -> int | None:
    """First bar >= ``start_idx`` that mitigates the zone.

        wick   any wick overlaps the zone (price touched it).
        close  a body closes past the PROXIMAL edge, i.e. into/through the zone
               (demand: close <= top; supply: close >= bottom). This catches a
               candle that blasts fully through and closes on the far side — a
               "close strictly inside the band" test would miss those.
        mean   price reaches the 50% level (demand: low <= mean; supply:
               high >= mean).
    """
    mean = (top + bottom) / 2.0
    for k in range(start_idx, len(close)):
        if mode == "wick":
            if high[k] >= bottom and low[k] <= top:
                return k
        elif mode == "mean":
            if (kind == BULLISH and low[k] <= mean) or \
               (kind == BEARISH and high[k] >= mean):
                return k
        else:  # "close" — body closes past the proximal edge
            if (kind == BULLISH and close[k] <= top) or \
               (kind == BEARISH and close[k] >= bottom):
                return k
    return None


def last_opposite(open_: np.ndarray, close: np.ndarray, anchor_idx: int,
                  want: str, scan_back: int = 20, atr_arr: np.ndarray | None = None,
                  min_body_atr: float = 0.0) -> int | None:
    """Walk backward from ``anchor_idx`` for the last candle of the ``want``
    direction ("down" = bearish body, "up" = bullish body). This is the literal
    "last opposite candle" rule; the ``pivot_swing`` anchor starts at the swing
    that originated the move, the ``pivot_break`` anchor starts at the break bar.

    When ``min_body_atr > 0`` candles whose body is smaller than
    ``min_body_atr * ATR`` are skipped, so a doji is never chosen as the origin.
    """
    lo = max(0, anchor_idx - scan_back)
    for j in range(anchor_idx, lo - 1, -1):
        down = close[j] < open_[j]
        up = close[j] > open_[j]
        if not ((want == "down" and down) or (want == "up" and up)):
            continue
        if min_body_atr > 0 and atr_arr is not None and j < len(atr_arr) \
                and not np.isnan(atr_arr[j]):
            if abs(close[j] - open_[j]) < min_body_atr * atr_arr[j]:
                continue
        return j
    return None


def has_displacement(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, a: int, b: int, atr_arr: np.ndarray,
                     mult: float) -> bool:
    """True if any bar in ``(a, b]`` is an impulse candle (range > ATR*mult)."""
    for k in range(a + 1, b + 1):
        if k < len(atr_arr) and not np.isnan(atr_arr[k]):
            if (high[k] - low[k]) > atr_arr[k] * mult:
                return True
    return False




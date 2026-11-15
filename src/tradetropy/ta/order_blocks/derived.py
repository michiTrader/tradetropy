"""Additional SMC zones adapted from Pitlane Strategies.

Each detector consumes closed OHLC rows and returns blocks in confirmation
order. The origin identifies the chart geometry; the confirmation identifies
the first bar on which a strategy can act on it.
"""

from __future__ import annotations

import numpy as np

from ._core import (
    BEARISH, BREAKER, BULLISH, PROPULSION, REJECTION, VACUUM, Block,
    confirmed_pivots, find_mitigation,
)
from .order_block import _bos_break_bars, find_order_blocks


def find_breaker_blocks(source: np.ndarray, *, swing: int = 2, zone: str = "body",
                        break_on: str = "close", mitigation: str = "close",
                        mss_window: int = 3, scan_back: int = 20,
                        anchor: str = "pivot_break") -> list[Block]:
    """Flip an order block after a close beyond its distal edge and an opposing
    structure break within ``mss_window`` bars of that failure, inclusive.
    The breaker confirms on the later of the two events.
    """
    if mss_window < 0:
        raise ValueError("mss_window must be nonnegative")
    if len(source) < swing * 2 + 2:
        return []
    ts, _, high, low, close = source[:, :5].T
    orders = find_order_blocks(source, anchor=anchor, zone=zone, swing=swing,
                               break_on=break_on, scan_back=scan_back)
    break_bars = _bos_break_bars(confirmed_pivots(high, low, ts, swing),
                                 high, low, close, break_on)
    structure_breaks = {
        direction: np.fromiter((i for i, sides in break_bars.items() if direction in sides),
                               dtype=np.intp)
        for direction in ("bull", "bear")
    }
    blocks = []
    for order in orders:
        start = order.confirmed_idx + 1
        bearish_order = order.kind == BEARISH
        failed = close[start:] > order.top if bearish_order else close[start:] < order.bottom
        failures = np.flatnonzero(failed)
        if not failures.size:
            continue
        failure = start + int(failures[0])
        opposite = structure_breaks["bull" if bearish_order else "bear"]
        pos = opposite.searchsorted(max(start, failure - mss_window))
        if pos == len(opposite) or opposite[pos] > failure + mss_window:
            continue
        mss = int(opposite[pos])
        confirmation = max(failure, mss)
        kind = BULLISH if bearish_order else BEARISH
        mit = find_mitigation(order.top, order.bottom, kind, high, low, close,
                              confirmation + 1, mitigation)
        blocks.append(Block(
            kind=kind, family=BREAKER, top=order.top, bottom=order.bottom,
            origin_idx=order.origin_idx, origin_ts=order.origin_ts,
            confirmed_idx=confirmation, confirmed_ts=float(ts[confirmation]),
            mitigated_idx=mit, mitigated_ts=float(ts[mit]) if mit is not None else None,
            meta={"from_ob_kind": order.kind, "fail_idx": failure, "mss_idx": mss},
        ))
    return sorted(blocks, key=lambda block: block.confirmed_idx)


def find_rejection_blocks(source: np.ndarray, *, swing: int = 2,
                          min_wick_ratio: float = 0.5,
                          mitigation: str = "close") -> list[Block]:
    """Mark the wick of a bar that sweeps a previously confirmed pivot and
    closes back across it. The wick must span at least ``min_wick_ratio`` of
    the candle's full range.
    """
    if not 0 <= min_wick_ratio <= 1:
        raise ValueError("min_wick_ratio must be between 0 and 1")
    if len(source) < swing * 2 + 2:
        return []
    ts, open_, high, low, close = source[:, :5].T
    pivots = confirmed_pivots(high, low, ts, swing)
    active = {"high": np.nan, "low": np.nan}
    cursor = 0
    blocks = []
    for i in range(len(source)):
        while cursor < len(pivots) and pivots[cursor].conf_idx < i:
            pivot = pivots[cursor]
            active[pivot.kind] = pivot.price
            cursor += 1
        if high[i] <= low[i]:
            continue
        body_top, body_bottom = max(open_[i], close[i]), min(open_[i], close[i])
        wick_min = (high[i] - low[i]) * min_wick_ratio
        for kind, level, top, bottom, swept in (
            (BEARISH, active["high"], high[i], body_top, close[i] < active["high"] < high[i]),
            (BULLISH, active["low"], body_bottom, low[i], low[i] < active["low"] < close[i]),
        ):
            if not swept or top - bottom < wick_min:
                continue
            mit = find_mitigation(top, bottom, kind, high, low, close, i + 1, mitigation)
            blocks.append(Block(
                kind=kind, family=REJECTION, top=float(top), bottom=float(bottom),
                origin_idx=i, origin_ts=float(ts[i]), confirmed_idx=i,
                confirmed_ts=float(ts[i]), mitigated_idx=mit,
                mitigated_ts=float(ts[mit]) if mit is not None else None,
                meta={"swept_level": level},
            ))
    return blocks


def find_propulsion_blocks(source: np.ndarray, *, swing: int = 2,
                           zone: str = "body", break_on: str = "close",
                           mitigation: str = "mean_close", scan_back: int = 20) -> list[Block]:
    """Mark the full range of the first directional candle that retests an
    order block before the parent fails. By default a later close through the
    propulsion candle's midpoint mitigates the zone.
    """
    if len(source) < swing * 2 + 2:
        return []
    ts, open_, high, low, close = source[:, :5].T
    orders = find_order_blocks(source, zone=zone, swing=swing, break_on=break_on,
                               scan_back=scan_back)
    blocks = []
    for order in orders:
        bullish = order.kind == BULLISH
        for i in range(order.confirmed_idx + 1, len(source)):
            if (bullish and close[i] < order.bottom) or (not bullish and close[i] > order.top):
                break
            tapped = high[i] >= order.bottom and low[i] <= order.top
            continued = close[i] > open_[i] if bullish else close[i] < open_[i]
            if not (tapped and continued):
                continue
            top, bottom = float(high[i]), float(low[i])
            mit = find_mitigation(top, bottom, order.kind, high, low, close, i + 1, mitigation)
            blocks.append(Block(
                kind=order.kind, family=PROPULSION, top=top, bottom=bottom,
                origin_idx=i, origin_ts=float(ts[i]), confirmed_idx=i,
                confirmed_ts=float(ts[i]), mitigated_idx=mit,
                mitigated_ts=float(ts[mit]) if mit is not None else None,
                meta={"parent_ob_origin": order.origin_idx, "mean_threshold": (top + bottom) / 2},
            ))
            break
    return sorted(blocks, key=lambda block: block.confirmed_idx)


def find_vacuum_blocks(source: np.ndarray, *, min_gap: float = 0.0,
                       mitigation: str = "mean") -> list[Block]:
    """Mark wick-to-wick gaps between two consecutive closed candles.

    ``min_gap`` is expressed in price units. The default mitigation level is
    the midpoint of the gap.
    """
    if min_gap < 0:
        raise ValueError("min_gap must be nonnegative")
    if len(source) < 2:
        return []
    ts, _, high, low, close = source[:, :5].T
    blocks = []
    for i in range(1, len(source)):
        if low[i] - high[i - 1] > min_gap:
            kind, top, bottom = BULLISH, float(low[i]), float(high[i - 1])
        elif low[i - 1] - high[i] > min_gap:
            kind, top, bottom = BEARISH, float(low[i - 1]), float(high[i])
        else:
            continue
        mit = find_mitigation(top, bottom, kind, high, low, close, i + 1, mitigation)
        blocks.append(Block(
            kind=kind, family=VACUUM, top=top, bottom=bottom,
            origin_idx=i, origin_ts=float(ts[i]), confirmed_idx=i,
            confirmed_ts=float(ts[i]), mitigated_idx=mit,
            mitigated_ts=float(ts[mit]) if mit is not None else None,
            meta={"gap": top - bottom},
        ))
    return blocks

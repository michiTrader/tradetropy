"""Order Block — the base detector, parameterised on three orthogonal axes plus
optional validation filters.

Axes (independent, combine freely)
----------------------------------
anchor      how the origin candle is located:
              "pivot_break" – OB is the last opposite candle before the leg that
                              BROKE a confirmed pivot (BOS-gated). Walk back from
                              the break bar.
              "pivot_swing" – the method that anchors at the swing that ORIGINATED
                              the move: on the break, go to the originating
                              confirmed swing (low for a bullish leg, high for a
                              bearish leg) and walk back from there to the last
                              opposite candle. Confirmed at the break bar.
              "atr_impulse" – classic single big candle (range > ATR*mult); OB is
                              the last opposite candle before it.
zone        "body" (open..close) | "range" (high..low, body + both wicks) |
            "wick" (body on the proximal side, wick extreme on the distal side).
mitigation  "close" | "wick" | "mean" (50% / consequent encroachment).

Degenerate-candle handling (doji origins)
------------------------------------------
min_body_atr      skip candles whose body < this * ATR when walking back, so a
                  doji is never chosen as the origin (0 = off).
extend_to_extreme default TRUE: push the distal edge to the extreme of the span
                  origin->originating-swing, so a body-only zone still covers the
                  swing low/high that created the move.
zone_fallback     if set (e.g. "range") and the zone is still thinner than the
                  floor, rebuild it with this mode.
require_pivot_anchor default FALSE: with it, a block is only emitted when a
                  confirmed swing is in force at the break. The pivot anchors
                  retire a swing once the price breaks it, so a break in the
                  opposite direction can arrive with no swing in force and the
                  zone then ends at the break bar. That is NOT a fake block: in
                  the case that motivated the flag (15-05 13:40, zone
                  7446.50-7460.75) the origin is the 13:25 bar, the last bullish
                  candle before the drop (body 7446.50-7459.75) — a textbook
                  bearish order block. The flag only trims those cases
                  (282 -> 276 blocks in May) and cost one trade in seven months,
                  so it is off and kept as a knob for measuring.
min_zone_atr      floor on zone height as a fraction of ATR at the origin; a
                  thinner zone is inflated symmetrically around its mid (0 = off).

validate    any subset of {"displacement", "fvg", "bos"} ANDed on top:
              displacement – an impulse candle (range > ATR*mult) in the leg.
              fvg          – a same-direction Fair Value Gap within ``fvg_window``
                             bars of the origin (see ``fvg_after``); pushes
                             ``confirmed_idx`` to the FVG's confirmation bar.
              bos          – the leg broke a confirmed pivot (inherent for the
                             pivot anchors; enforced for "atr_impulse").

All modes return a list of ``Block`` objects with ``family == "order"``.
"""

from __future__ import annotations

import numpy as np

from ._core import (BEARISH, BULLISH, ORDER, Block, apply_min_height, atr,
                    confirmed_pivots, extend_to_span_extreme, find_fvgs,
                    find_mitigation, fvg_after, has_displacement, last_opposite,
                    zone_of)

def _pivot_break_events(pivots, high, low, close, break_on):
    """Yield ``(bar_idx, direction, active_high, active_low)`` for every bar that
    breaks a confirmed pivot. ``active_*`` are ``(price, known_from, real_idx)``
    of the pivots in force at the break, so callers can pick either anchor."""
    by_bar: dict[int, list] = {}
    for p in pivots:
        by_bar.setdefault(p.conf_idx, []).append(p)
    up_src = close if break_on == "close" else high
    down_src = close if break_on == "close" else low
    act_h = act_l = None
    bias = None
    for i in range(len(close)):
        for p in by_bar.get(i, ()):
            if p.kind == "high":
                act_h = (p.price, i, p.real_idx)
            else:
                act_l = (p.price, i, p.real_idx)
        broke_up = act_h is not None and act_h[1] < i and up_src[i] > act_h[0]
        broke_down = act_l is not None and act_l[1] < i and down_src[i] < act_l[0]
        if broke_up and broke_down:            # a bar piercing both: bias wins
            if bias == "bear":
                broke_up = False
            else:
                broke_down = False
        if broke_up:
            yield i, "bull", act_h, act_l
            bias = "bull"; act_h = None
        elif broke_down:
            yield i, "bear", act_h, act_l
            bias = "bear"; act_l = None


def _pivot_candidates(pivots, open_, high, low, close, break_on, anchor, scan_back,
                      atr_arr, min_body_atr, require_pivot=False):
    """Yield ``(kind, origin_idx, conf_idx, imp_kind, span_end)``. ``span_end`` is
    the originating swing's real bar (swing low for a bullish leg, swing high for
    a bearish leg) — the far end of the span used to extend the zone to the
    extreme — regardless of which anchor located the origin candle."""
    out = []
    for i, direction, act_h, act_l in _pivot_break_events(
            pivots, high, low, close, break_on):
        if direction == "bull":                # bullish leg -> demand OB
            if require_pivot and act_l is None:
                continue                       # no confirmed swing -> no block
            swing = act_l[2] if act_l is not None else i
            anchor_idx = i if anchor == "pivot_break" else swing
            j = last_opposite(open_, close, anchor_idx, "down", scan_back,
                              atr_arr, min_body_atr)
            if j is not None:
                out.append((BULLISH, j, i, BULLISH, swing))
        else:                                  # bearish leg -> supply OB
            if require_pivot and act_h is None:
                continue                       # no confirmed swing -> no block
            swing = act_h[2] if act_h is not None else i
            anchor_idx = i if anchor == "pivot_break" else swing
            j = last_opposite(open_, close, anchor_idx, "up", scan_back,
                              atr_arr, min_body_atr)
            if j is not None:
                out.append((BEARISH, j, i, BEARISH, swing))
    return out


def _atr_candidates(open_, high, low, close, atr_arr, mult, scan_back, min_body_atr):
    out = []
    for i in range(len(close)):
        if i >= len(atr_arr) or np.isnan(atr_arr[i]):
            continue
        if (high[i] - low[i]) <= atr_arr[i] * mult:
            continue
        if close[i] > open_[i]:                # bullish impulse -> demand OB
            j = last_opposite(open_, close, i - 1, "down", scan_back,
                              atr_arr, min_body_atr)
            if j is not None:
                out.append((BULLISH, j, i, BULLISH, i))
        elif close[i] < open_[i]:              # bearish impulse -> supply OB
            j = last_opposite(open_, close, i - 1, "up", scan_back,
                              atr_arr, min_body_atr)
            if j is not None:
                out.append((BEARISH, j, i, BEARISH, i))
    return out


def _bos_break_bars(pivots, high, low, close, break_on) -> dict[int, set]:
    out: dict[int, set] = {}
    for i, direction, _h, _l in _pivot_break_events(
            pivots, high, low, close, break_on):
        out.setdefault(i, set()).add(direction)
    return out

def find_order_blocks(source: np.ndarray, *, anchor: str = "pivot_break",
                      zone: str = "body", swing: int = 2, break_on: str = "close",
                      validate: tuple[str, ...] = (), fvg_window: int = 3,
                      atr_period: int = 14, atr_mult: float = 1.5,
                      mitigation: str = "close", scan_back: int = 20,
                      min_body_atr: float = 0.0, extend_to_extreme: bool = True,
                      zone_fallback: str | None = None,
                      min_zone_atr: float = 0.0,
                      require_pivot_anchor: bool = False) -> list[Block]:
    """Detect order blocks. ``source`` is ``[N x 6]`` (ts, open, high, low, close,
    volume). See the module docstring for every option. Returns blocks sorted by
    ``confirmed_idx`` (the bar each becomes knowable)."""
    n = len(source)
    if n < swing * 2 + 2:
        return []
    ts = source[:, 0].astype(np.float64)
    open_ = source[:, 1].astype(np.float64)
    high = source[:, 2].astype(np.float64)
    low = source[:, 3].astype(np.float64)
    close = source[:, 4].astype(np.float64)

    need_atr = (anchor == "atr_impulse" or "displacement" in validate
                or min_body_atr > 0 or min_zone_atr > 0)
    atr_arr = atr(high, low, close, atr_period) if need_atr else np.array([])
    fvgs = find_fvgs(high, low) if "fvg" in validate else []

    if anchor == "atr_impulse":
        cands = _atr_candidates(open_, high, low, close, atr_arr, atr_mult,
                                scan_back, min_body_atr)
    else:
        pivots = confirmed_pivots(high, low, ts, swing)
        cands = _pivot_candidates(pivots, open_, high, low, close, break_on,
                                  anchor, scan_back, atr_arr, min_body_atr,
                                  require_pivot_anchor)

    break_bars = None
    if "bos" in validate and anchor == "atr_impulse":
        pivots = confirmed_pivots(high, low, ts, swing)
        break_bars = _bos_break_bars(pivots, high, low, close, break_on)

    blocks: list[Block] = []
    seen: set = set()
    for kind, origin_idx, conf_idx, imp_kind, span_end in cands:
        cidx = conf_idx
        # ``span_ts`` makes a block addressable from any window: the strategy
        # queries the indicator by timestamp instead of by array index, so it
        # never depends on where the precompute window starts.
        meta: dict = {"anchor": anchor, "span_end": span_end,
                      "span_ts": float(ts[span_end]),
                      "origin_idx": origin_idx}

        if "fvg" in validate:
            f = fvg_after(fvgs, origin_idx, imp_kind, fvg_window)
            if f is None:
                continue
            meta["fvg"] = (f["bottom"], f["top"], f["mid_idx"])
            cidx = max(cidx, f["confirmed_idx"])

        if "displacement" in validate and not has_displacement(
                open_, high, low, close, origin_idx, conf_idx, atr_arr, atr_mult):
            continue

        if break_bars is not None:             # bos filter for atr anchor
            want = "bull" if imp_kind == BULLISH else "bear"
            hit = next((b for b in range(origin_idx + 1,
                        min(n, conf_idx + fvg_window + 1))
                        if want in break_bars.get(b, ())), None)
            if hit is None:
                continue
            cidx = max(cidx, hit)

        key = (kind, origin_idx)
        if key in seen:
            continue
        seen.add(key)

        top, bot = zone_of(open_, high, low, close, origin_idx, zone, kind=kind)
        atr_o = (atr_arr[origin_idx] if need_atr and origin_idx < len(atr_arr)
                 and not np.isnan(atr_arr[origin_idx]) else np.nan)
        floor = (min_zone_atr * atr_o) if (min_zone_atr > 0
                                           and not np.isnan(atr_o)) else 0.0
        if zone_fallback and floor > 0 and (top - bot) < floor:
            top, bot = zone_of(open_, high, low, close, origin_idx,
                               zone_fallback, kind=kind)
        if extend_to_extreme:
            top, bot = extend_to_span_extreme(top, bot, kind, high, low,
                                              origin_idx, span_end)
        if floor > 0:
            top, bot = apply_min_height(top, bot, floor)

        mit = find_mitigation(top, bot, kind, high, low, close, cidx + 1, mitigation)
        blocks.append(Block(
            kind=kind, family=ORDER, top=top, bottom=bot,
            origin_idx=origin_idx, origin_ts=float(ts[origin_idx]),
            confirmed_idx=cidx, confirmed_ts=float(ts[cidx]),
            mitigated_idx=mit,
            mitigated_ts=(float(ts[mit]) if mit is not None else None),
            meta=meta))

    blocks.sort(key=lambda b: b.confirmed_idx)
    return blocks



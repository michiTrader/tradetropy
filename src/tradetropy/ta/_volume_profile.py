"""
Core volume-by-price computation shared by the Volume Profile indicators.

This module is intentionally free of any plotting or proxy dependencies. It
reuses the footprint scalar engine (_compute_scalars) so that POC, VAH and VAL
are computed with the exact same value-area logic used by the footprint module.

Two pieces live here:

- Binning helpers
    Group prices into discrete levels of size ``tick_size`` and aggregate
    bid / ask volume per level.

- Developing scan
    Walk source rows in time order, accumulate the per-period histogram and
    emit the *developing* POC / VAH / VAL for every row (no look-ahead). The
    finalized per-period histograms are returned as well for rendering.

The developing scan is what makes the result safe for backtesting: at row ``i``
the emitted values only depend on rows ``<= i`` within the same period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradetropy.models.footprint._compute import _compute_scalars, _dict_to_levels
from tradetropy.models.footprint._types import (
    _FP_SCALAR_COL,
    _FP_LEVEL_COL,
    N_FP_LEVEL_COLS,
)
from tradetropy.models.footprint._config import _round_tick_pretty


# =====
# Default volume-profile colors
# =====
#
# Semantic colors for the volume-by-price histogram and its markers. They live
# on the VP indicators / tools as object attributes (e.g. VolumeProfile(
# buy_color=...), FixedRangeVP(poc_color=...)), NOT in the plot theme, so a VP
# renders identically under any theme. Values match the historical light-theme
# vp_* tokens so the default look is unchanged.
DEFAULT_VP_BUY = "#2EB1E1"    # buy-aggressor (ask) volume segment
DEFAULT_VP_SELL = "#F9A825"   # sell-aggressor (bid) volume segment
DEFAULT_VP_POC = "#EF5350"    # point-of-control line
DEFAULT_VP_HVN = "#80D837"    # high-volume node markers
DEFAULT_VP_LVN = "#4F2472"    # low-volume node markers


# =====
# Volume nodes (HVN / LVN)
# =====
@dataclass(frozen=True)
class VolumeNode:
    """
    A high- or low-volume node detected on a volume-by-price profile.

    High-volume nodes (HVN) are local peaks of the volume histogram - prices
    where the market spent disproportionate volume, acting as acceptance areas
    or magnets. Low-volume nodes (LVN) are local valleys - thin prices the
    market moved through quickly, often acting as rejection levels.

    Attributes:
        price (float): Center price of the node level.
        volume (float): Total volume accumulated at the node level.
        kind (str): 'hvn' (local volume peak) or 'lvn' (local volume valley).
        strength (float): Prominence normalized to 0..1 relative to the largest
            level in the profile. Higher means a more pronounced node.
    """

    price: float
    volume: float
    kind: str
    strength: float


def _local_maxima(values: np.ndarray) -> list[int]:
    """
    Indices of local maxima of a 1D array, plateau- and endpoint-aware.

    A plateau of equal values that is higher than both surrounding values
    yields a single index at the plateau center. Endpoints qualify when they
    are strictly greater than their only inner neighbor (values beyond the
    array are treated as -inf).

    Args:
        values (np.ndarray): 1D array to scan.

    Returns:
        list[int]: Indices of local maxima.
    """
    n = len(values)
    peaks: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[i]:
            j += 1
        left = values[i - 1] if i > 0 else -np.inf
        right = values[j + 1] if j + 1 < n else -np.inf
        if values[i] > left and values[i] > right:
            peaks.append((i + j) // 2)
        i = j + 1
    return peaks


def _peak_prominence(values: np.ndarray, p: int) -> float:
    """
    Topographic prominence of the peak at index ``p``.

    Walks left and right from the peak until a strictly higher value is found,
    tracking the lowest value (valley) on each side. The base is the higher of
    the two valleys; prominence is the peak height above that base. When no
    higher value exists on a side (the peak dominates to that end), the base on
    that side is 0, so an isolated or global peak gets its full height.

    Args:
        values (np.ndarray): 1D array the peak belongs to.
        p (int): Index of the peak.

    Returns:
        float: Absolute prominence (same units as ``values``).
    """
    n = len(values)
    v = float(values[p])

    def _base(step: int) -> float:
        valley = np.inf
        i = p + step
        found_higher = False
        while 0 <= i < n:
            if values[i] > v:
                found_higher = True
                break
            valley = min(valley, float(values[i]))
            i += step
        if not found_higher:
            return 0.0
        return valley if valley != np.inf else 0.0

    base = max(_base(-1), _base(1))
    return v - base


def detect_volume_nodes(
    prices: np.ndarray,
    volumes: np.ndarray,
    kind: str = "both",
    prominence: float = 0.1,
    max_nodes: int | None = None,
) -> list[VolumeNode]:
    """
    Detect high- and/or low-volume nodes on a volume-by-price histogram.

    HVN are local maxima of ``volumes``; LVN are local minima (detected as
    maxima of the inverted histogram). Each node's strength is its topographic
    prominence normalized to the largest level volume, so both kinds share a
    comparable 0..1 scale. Nodes weaker than ``prominence`` are discarded.

    Args:
        prices (np.ndarray): Level prices. Need not be pre-sorted.
        volumes (np.ndarray): Total volume per level, aligned with ``prices``.
        kind (str): 'hvn', 'lvn' or 'both'.
        prominence (float): Minimum strength (0..1) to keep a node.
        max_nodes (int | None): Optional cap per kind, keeping the strongest
            nodes first. The returned list is always ordered by price.

    Returns:
        list[VolumeNode]: Detected nodes ordered ascending by price.

    Raises:
        ValueError: If ``kind`` is not 'hvn', 'lvn' or 'both'.
    """
    if kind not in ("hvn", "lvn", "both"):
        raise ValueError(f"kind must be 'hvn', 'lvn' or 'both', not {kind!r}")

    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if len(prices) == 0 or len(volumes) == 0:
        return []

    order = np.argsort(prices)
    prices = prices[order]
    volumes = volumes[order]

    max_vol = float(volumes.max())
    if max_vol <= 0.0:
        return []

    def _collect(values: np.ndarray, node_kind: str) -> list[VolumeNode]:
        found: list[VolumeNode] = []
        for p in _local_maxima(values):
            strength = _peak_prominence(values, p) / max_vol
            if strength < prominence:
                continue
            found.append(VolumeNode(
                price=float(prices[p]),
                volume=float(volumes[p]),
                kind=node_kind,
                strength=float(min(max(strength, 0.0), 1.0)),
            ))
        if max_nodes is not None and len(found) > max_nodes:
            found = sorted(found, key=lambda n: n.strength, reverse=True)[:max_nodes]
        return found

    nodes: list[VolumeNode] = []
    if kind in ("hvn", "both"):
        nodes.extend(_collect(volumes, "hvn"))
    if kind in ("lvn", "both"):
        nodes.extend(_collect(max_vol - volumes, "lvn"))

    nodes.sort(key=lambda n: n.price)
    return nodes


def resolve_period_ids(
    ts: np.ndarray,
    period_ms: int,
    anchor,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Map timestamps to period ids and their period-start timestamps for an anchor.

    The period id of a row decides which developing profile it belongs to, and
    the returned period-start timestamps let the plotting layer anchor each
    histogram on the x axis. All four anchor modes keep the same period length
    (``period_ms``); they differ only in *where* each period begins.

    Anchor modes:
        'utc'  (default) - Periods start at absolute UTC multiples of period_ms
            (Unix-epoch aligned). This matches the legacy behaviour: intraday
            timeframes ('30m', '1h') already fall on round clock marks, and
            '1d' / '1w' cut at UTC midnight / Thursday-UTC respectively.
        'data' - The first row defines the anchor; periods start at
            ``ts[0] + k * period_ms``. Use when the profile should begin wherever
            the data begins (e.g. 11:11 -> 11:11 next day).
        tzinfo / str - A timezone whose local midnight (for '1d'/'1w') anchors
            the period. Each row is shifted by that zone's offset at its own
            instant, so DST transitions are handled per row.
        (tz, 'HH:MM') / (tz, minutes) - A timezone plus a session open time of
            day. Periods start at that local wall-clock time (e.g. CME Globex
            17:00 America/New_York), again DST-aware per row.

    Args:
        ts (np.ndarray): Timestamps in ms (ascending), int64.
        period_ms (int): Period length in milliseconds (> 0).
        anchor: One of the modes described above. None is treated as 'utc'.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - period_id (int64): Period id per row.
            - period_start (int64): Period-start ts (ms) per row, such that
              ``period_start <= ts < period_start + period_ms``.

    Raises:
        ValueError: If the anchor specification is malformed.
    """
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    offset = _anchor_offset_ms(ts, period_ms, anchor)
    shifted = ts - offset
    period_id = np.floor_divide(shifted, period_ms).astype(np.int64)
    period_start = period_id * period_ms + offset
    return period_id, period_start


def _anchor_offset_ms(ts: np.ndarray, period_ms: int, anchor) -> np.ndarray:
    """
    Per-row offset (ms) subtracted from ts before the period division.

    The offset embeds where a period begins. For 'utc' it is 0 (Unix aligned).
    For 'data' it is the constant ``ts[0] % period_ms`` so the grid starts on the
    first row. For timezone anchors it is the zone's UTC offset at each row
    (DST-aware), optionally minus a session ``time_of_day_ms`` so periods begin
    at a wall-clock session open instead of local midnight.

    Args:
        ts (np.ndarray): Timestamps in ms (ascending), int64.
        period_ms (int): Period length in milliseconds.
        anchor: Anchor spec (see :func:`resolve_period_ids`).

    Returns:
        np.ndarray: int64 offset per row (broadcastable shape (n,)).

    Raises:
        ValueError: If the anchor specification is malformed.
    """
    n = len(ts)
    if anchor is None or anchor == "utc":
        return np.zeros(n, dtype=np.int64)

    if anchor == "data":
        return np.full(n, int(ts[0]) % period_ms, dtype=np.int64)

    tz_spec, time_of_day_ms = _split_anchor(anchor)
    tz = _coerce_anchor_tz(tz_spec)
    tz_off = _tz_offsets_per_row(ts, tz)
    # Local midnight is at ts where (ts + tz_off) % period_ms == 0; shifting by
    # the session open moves the cut to that wall-clock time of day.
    return (tz_off + int(time_of_day_ms)).astype(np.int64)


def _split_anchor(anchor) -> "tuple[object, int]":
    """
    Split an anchor into (timezone spec, session time-of-day in ms).

    Accepts a bare timezone (str or tzinfo) -> time of day 0 (local midnight),
    or a 2-tuple ``(tz, when)`` where ``when`` is 'HH:MM', 'HH:MM:SS', minutes
    past midnight (int), or a ``datetime.time``.

    Args:
        anchor: Anchor spec.

    Returns:
        tuple[object, int]: (tz_spec, time_of_day_ms).

    Raises:
        ValueError: If the tuple form or the time-of-day value is malformed.
    """
    if isinstance(anchor, (tuple, list)):
        if len(anchor) != 2:
            raise ValueError(
                f"anchor tuple must be (tz, time_of_day), received {anchor!r}"
            )
        tz_spec, when = anchor
        return tz_spec, _time_of_day_ms(when)
    return anchor, 0


def _time_of_day_ms(when) -> int:
    """
    Convert a session time-of-day spec to milliseconds past local midnight.

    Args:
        when: 'HH:MM', 'HH:MM:SS', minutes-past-midnight (int/float), or a
            ``datetime.time``.

    Returns:
        int: Milliseconds in ``[0, 86_400_000)``.

    Raises:
        ValueError: If the value cannot be parsed or is out of range.
    """
    from datetime import time as _time

    if isinstance(when, _time):
        ms = ((when.hour * 60 + when.minute) * 60 + when.second) * 1000
    elif isinstance(when, (int, float)) and not isinstance(when, bool):
        ms = int(when) * 60_000
    elif isinstance(when, str):
        parts = when.strip().split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            raise ValueError(f"invalid session time {when!r}, expected 'HH:MM'")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) == 3 else 0
        ms = ((h * 60 + m) * 60 + s) * 1000
    else:
        raise ValueError(f"unsupported session time spec {when!r}")
    if not (0 <= ms < 86_400_000):
        raise ValueError(f"session time {when!r} out of range [00:00, 24:00)")
    return ms


def _coerce_anchor_tz(tz_spec):
    """
    Resolve an anchor timezone spec to a tzinfo, reusing the session coercion.

    Args:
        tz_spec: Timezone string ('America/New_York'), tzinfo, or numeric offset.

    Returns:
        tzinfo: Concrete timezone instance.

    Raises:
        ValueError: If the spec is not a recognised timezone.
    """
    from tradetropy.session.base import _coerce_tz

    if tz_spec is None or tz_spec == "utc":
        from datetime import timezone
        return timezone.utc
    try:
        return _coerce_tz(tz_spec)
    except Exception as exc:  # noqa: BLE001 - surface a clear anchor error
        raise ValueError(f"invalid anchor timezone {tz_spec!r}: {exc}") from exc


def _tz_offsets_per_row(ts: np.ndarray, tz) -> np.ndarray:
    """
    DST-aware UTC offset (ms) for each row, computed without a per-row loop.

    Offsets change only at DST boundaries, so unique day buckets are evaluated
    once via :func:`tradetropy.session.base._tz_offset_ms` and broadcast back to
    rows. This keeps tick-scale inputs fast while staying correct across DST.

    Args:
        ts (np.ndarray): Timestamps in ms (ascending), int64.
        tz: Timezone instance.

    Returns:
        np.ndarray: int64 UTC offset per row.
    """
    from tradetropy.session.base import _tz_offset_ms

    day_ms = 86_400_000
    day_bucket = np.floor_divide(ts, day_ms).astype(np.int64)
    unique_days, inverse = np.unique(day_bucket, return_inverse=True)
    off_by_day = np.array(
        [_tz_offset_ms(tz, int(d) * day_ms) for d in unique_days],
        dtype=np.int64,
    )
    return off_by_day[inverse]


def level_price(price: np.ndarray, tick_size: float) -> np.ndarray:
    """
    Snap raw prices to the center price of their volume-profile level.

    Args:
        price (np.ndarray): Raw price array.
        tick_size (float): Level size (bin width) in price units.

    Returns:
        np.ndarray: Prices rounded to the nearest multiple of tick_size.
    """
    return np.round(price / tick_size) * tick_size


def infer_tick_size(prices: np.ndarray, target_bins: int = 100) -> float:
    """
    Infer a reasonable level size from the global price range.

    Used only when the user does not pass an explicit tick_size. The range is
    divided by the desired number of bins and rounded to a 'nice' value
    (1, 2.5, 5, 10 x 10^k), matching the footprint convention.

    Args:
        prices (np.ndarray): Price array (any subset is fine).
        target_bins (int): Desired number of levels across the full range.

    Returns:
        float: Level size in price units (always > 0).
    """
    if len(prices) == 0:
        return 1.0
    rango = float(np.nanmax(prices) - np.nanmin(prices))
    if rango <= 0:
        return 1.0
    return _round_tick_pretty(rango / max(target_bins, 1))


# =====
# Contribution builders (shared by indicators and the on-demand range tool)
# =====
def kline_contributions(high, low, vol, is_bull, tick_size):
    """
    Spread each candle's volume uniformly across its high-low price levels.

    Returns the contribution arrays consumed by the developing / rolling scans
    and by :func:`compute_range_profile`: (rows, levels, vol_bid, vol_ask,
    counts). A bullish candle contributes ask (buy) volume, a bearish one bid
    (sell) volume.

    Args:
        high (np.ndarray): Candle high per row.
        low (np.ndarray): Candle low per row.
        vol (np.ndarray): Candle volume per row.
        is_bull (np.ndarray): Boolean per row (close >= open).
        tick_size (float): Price level size.

    Returns:
        tuple[np.ndarray, ...]: (rows, levels, vol_bid, vol_ask, counts).
    """
    rows: list[int] = []
    levels: list[float] = []
    v_bid: list[float] = []
    v_ask: list[float] = []
    counts: list[float] = []
    n = len(high)
    for i in range(n):
        lo_k = int(round(low[i] / tick_size))
        hi_k = int(round(high[i] / tick_size))
        if hi_k < lo_k:
            lo_k, hi_k = hi_k, lo_k
        nbins = hi_k - lo_k + 1
        vol_per = vol[i] / nbins if nbins > 0 else 0.0
        for k in range(lo_k, hi_k + 1):
            rows.append(i)
            levels.append(k * tick_size)
            if is_bull[i]:
                v_ask.append(vol_per)
                v_bid.append(0.0)
            else:
                v_ask.append(0.0)
                v_bid.append(vol_per)
            counts.append(0.0)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(levels, dtype=np.float64),
        np.asarray(v_bid, dtype=np.float64),
        np.asarray(v_ask, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def tick_contributions(price, vol, flags, tick_size):
    """
    Bin each trade at its real price level and split it by aggressor side.

    Aggressor side is taken from the tick ``flags`` (bit 32 = buy, bit 64 =
    sell); when neither bit is set a tick-rule fallback is used (price >=
    previous price -> buy). Mirrors the classification in TickVolumeProfile.

    Args:
        price (np.ndarray): Trade price per row.
        vol (np.ndarray): Trade volume per row.
        flags (np.ndarray): Tick flags per row.
        tick_size (float): Price level size.

    Returns:
        tuple[np.ndarray, ...]: (levels, vol_bid, vol_ask, counts).
    """
    n = len(price)
    levels = level_price(price, tick_size)
    is_buy_flag = (flags & 32) != 0
    is_sell_flag = (flags & 64) != 0
    prev = np.roll(price, 1)
    if n:
        prev[0] = price[0]
    tick_buy = price >= prev
    is_buy = np.where(is_buy_flag, True, np.where(is_sell_flag, False, tick_buy))
    vol_ask = np.where(is_buy, vol, 0.0)
    vol_bid = np.where(is_buy, 0.0, vol)
    counts = np.ones(n, dtype=np.float64)
    return levels, vol_bid, vol_ask, counts


def histogram_from_contributions(
    c_level: np.ndarray,
    c_vol_bid: np.ndarray,
    c_vol_ask: np.ndarray,
    c_count: np.ndarray,
) -> np.ndarray:
    """
    Aggregate flat contribution arrays into a sorted footprint level matrix.

    Args:
        c_level (np.ndarray): Level (snapped price) per contribution.
        c_vol_bid (np.ndarray): Bid (sell-aggressor) volume per contribution.
        c_vol_ask (np.ndarray): Ask (buy-aggressor) volume per contribution.
        c_count (np.ndarray): Trade count per contribution.

    Returns:
        np.ndarray: Level matrix with columns
            (price, vol_bid, vol_ask, vol_total, delta, n_trades),
            ordered ascending by price (empty (0, 6) when there are no levels).
    """
    acc: dict[float, list] = {}
    for i in range(len(c_level)):
        lp = float(c_level[i])
        entry = acc.get(lp)
        if entry is None:
            entry = [0.0, 0.0, 0.0]
            acc[lp] = entry
        entry[0] += float(c_vol_bid[i])
        entry[1] += float(c_vol_ask[i])
        entry[2] += float(c_count[i])
    return _dict_to_levels(acc)


def scan_developing_profiles(
    n_rows: int,
    period_id: np.ndarray,
    contrib_row: np.ndarray,
    c_level: np.ndarray,
    c_vol_bid: np.ndarray,
    c_vol_ask: np.ndarray,
    c_count: np.ndarray,
    value_area_pct: float = 0.70,
    va_recompute_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Accumulate per-period volume profiles and emit developing POC/VAH/VAL.

    Rows are processed in order. Each row may receive several contributions
    (e.g. a kline distributes its volume across many price levels), referenced
    through ``contrib_row``. When the period id changes the histogram resets and
    the finished period is snapshotted.

    Args:
        n_rows (int): Number of source rows (klines or ticks).
        period_id (np.ndarray): Period id per row (e.g. ts // period_ms).
        contrib_row (np.ndarray): Row index each contribution belongs to,
            in non-decreasing order.
        c_level (np.ndarray): Level (snapped price) of each contribution.
        c_vol_bid (np.ndarray): Bid (sell-aggressor) volume per contribution.
        c_vol_ask (np.ndarray): Ask (buy-aggressor) volume per contribution.
        c_count (np.ndarray): Trade count per contribution.
        value_area_pct (float): Fraction of volume defining the value area.
        va_recompute_every (int): Recompute the value area every N rows.
            1 = exact developing values every row. Larger values trade some
            VAH/VAL granularity for speed on very large tick datasets.

    Returns:
        tuple:
            - poc (np.ndarray): Developing POC price per row.
            - vah (np.ndarray): Developing value-area high per row.
            - val (np.ndarray): Developing value-area low per row.
            - periods (list[dict]): One dict per finished period with keys
              'pid', 'row_start', 'row_end', 'levels', 'scalars'.
    """
    poc = np.full(n_rows, np.nan, dtype=np.float64)
    vah = np.full(n_rows, np.nan, dtype=np.float64)
    val = np.full(n_rows, np.nan, dtype=np.float64)
    periods: list[dict] = []

    if n_rows == 0:
        return poc, vah, val, periods

    def _finalize(pid, row_start, row_end, acc):
        levels = _dict_to_levels(acc)
        scal = _compute_scalars(levels, value_area_pct, 0.0)
        periods.append({
            "pid": pid,
            "row_start": row_start,
            "row_end": row_end,
            "levels": levels,
            "scalars": scal,
        })

    cur_pid = period_id[0]
    acc: dict[float, list] = {}
    pid_start = 0
    rows_since = va_recompute_every  # force compute on first row
    last_poc = last_vah = last_val = np.nan

    n_contrib = len(contrib_row)
    ci = 0  # contribution cursor

    for i in range(n_rows):
        pid = period_id[i]
        if pid != cur_pid:
            _finalize(cur_pid, pid_start, i - 1, acc)
            acc = {}
            cur_pid = pid
            pid_start = i
            rows_since = va_recompute_every
            last_poc = last_vah = last_val = np.nan

        # Apply every contribution attached to this row.
        while ci < n_contrib and contrib_row[ci] == i:
            lp = float(c_level[ci])
            entry = acc.get(lp)
            if entry is None:
                entry = [0.0, 0.0, 0.0]
                acc[lp] = entry
            entry[0] += float(c_vol_bid[ci])
            entry[1] += float(c_vol_ask[ci])
            entry[2] += float(c_count[ci])
            ci += 1

        rows_since += 1
        if rows_since >= va_recompute_every and acc:
            levels = _dict_to_levels(acc)
            scal = _compute_scalars(levels, value_area_pct, 0.0)
            last_poc = float(scal[_FP_SCALAR_COL["poc_price"]])
            last_vah = float(scal[_FP_SCALAR_COL["vah"]])
            last_val = float(scal[_FP_SCALAR_COL["val"]])
            rows_since = 0

        poc[i] = last_poc
        vah[i] = last_vah
        val[i] = last_val

    _finalize(cur_pid, pid_start, n_rows - 1, acc)
    return poc, vah, val, periods


def scan_rolling_profiles(
    n_rows: int,
    window: int,
    contrib_row: np.ndarray,
    c_level: np.ndarray,
    c_vol_bid: np.ndarray,
    c_vol_ask: np.ndarray,
    c_count: np.ndarray,
    value_area_pct: float = 0.70,
    va_recompute_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Accumulate a sliding-window volume profile and emit continuous POC/VAH/VAL.

    Unlike :func:`scan_developing_profiles`, there are no discrete periods: at
    row ``i`` the histogram covers rows ``[max(0, i - window + 1) .. i]``. As the
    window slides forward the contributions of the row leaving the window are
    subtracted, so the profile is continuous (no per-period reset) and the
    emitted POC/VAH/VAL form a smooth, non-look-ahead series.

    Args:
        n_rows (int): Number of source rows (klines or ticks).
        window (int): Number of trailing rows accumulated into the profile.
        contrib_row (np.ndarray): Row index each contribution belongs to,
            in non-decreasing order.
        c_level (np.ndarray): Level (snapped price) of each contribution.
        c_vol_bid (np.ndarray): Bid (sell-aggressor) volume per contribution.
        c_vol_ask (np.ndarray): Ask (buy-aggressor) volume per contribution.
        c_count (np.ndarray): Trade count per contribution.
        value_area_pct (float): Fraction of volume defining the value area.
        va_recompute_every (int): Recompute the value area every N rows.

    Returns:
        tuple:
            - poc (np.ndarray): Rolling POC price per row.
            - vah (np.ndarray): Rolling value-area high per row.
            - val (np.ndarray): Rolling value-area low per row.
            - final (dict): The finalized last-window histogram with keys
              'levels' and 'scalars' (empty arrays when there is no volume).
    """
    poc = np.full(n_rows, np.nan, dtype=np.float64)
    vah = np.full(n_rows, np.nan, dtype=np.float64)
    val = np.full(n_rows, np.nan, dtype=np.float64)
    empty_final = {
        "levels": np.empty((0, 0), dtype=np.float64),
        "scalars": np.zeros(len(_FP_SCALAR_COL), dtype=np.float64),
    }

    if n_rows == 0:
        return poc, vah, val, empty_final

    window = max(int(window), 1)

    # Contribution slice per row: row_slice[i] = (start, end) into contrib arrays.
    n_contrib = len(contrib_row)
    row_start = np.zeros(n_rows + 1, dtype=np.int64)
    ci = 0
    for i in range(n_rows):
        row_start[i] = ci
        while ci < n_contrib and contrib_row[ci] == i:
            ci += 1
    row_start[n_rows] = ci

    acc: dict[float, list] = {}
    rows_since = va_recompute_every
    last_poc = last_vah = last_val = np.nan

    def _apply(start, end, sign):
        for j in range(start, end):
            lp = float(c_level[j])
            entry = acc.get(lp)
            if entry is None:
                if sign < 0:
                    continue
                entry = [0.0, 0.0, 0.0]
                acc[lp] = entry
            entry[0] += sign * float(c_vol_bid[j])
            entry[1] += sign * float(c_vol_ask[j])
            entry[2] += sign * float(c_count[j])
            if sign < 0 and entry[0] <= 1e-12 and entry[1] <= 1e-12:
                del acc[lp]

    for i in range(n_rows):
        _apply(row_start[i], row_start[i + 1], +1)
        out = i - window
        if out >= 0:
            _apply(row_start[out], row_start[out + 1], -1)

        rows_since += 1
        if rows_since >= va_recompute_every and acc:
            levels = _dict_to_levels(acc)
            scal = _compute_scalars(levels, value_area_pct, 0.0)
            last_poc = float(scal[_FP_SCALAR_COL["poc_price"]])
            last_vah = float(scal[_FP_SCALAR_COL["vah"]])
            last_val = float(scal[_FP_SCALAR_COL["val"]])
            rows_since = 0

        poc[i] = last_poc
        vah[i] = last_vah
        val[i] = last_val

    if acc:
        levels = _dict_to_levels(acc)
        final = {
            "levels": levels,
            "scalars": _compute_scalars(levels, value_area_pct, 0.0),
        }
    else:
        final = empty_final
    return poc, vah, val, final


def _profile_from_levels(
    levels: np.ndarray,
    value_area_pct: float,
    nodes: str | None,
    node_prominence: float,
    max_nodes: int | None,
) -> dict | None:
    """
    Turn a level matrix into a render-ready profile dict with scalars + nodes.

    Args:
        levels (np.ndarray): Level matrix from :func:`histogram_from_contributions`.
        value_area_pct (float): Fraction of volume defining the value area.
        nodes (str | None): None | 'hvn' | 'lvn' | 'both' — which nodes to detect.
        node_prominence (float): Node prominence threshold 0..1.
        max_nodes (int | None): Optional cap per node kind.

    Returns:
        dict | None: Profile dict (poc/vah/val/hvn/lvn/profile fields) or None
        when there is no volume.
    """
    if levels is None or len(levels) == 0:
        return None
    scal = _compute_scalars(levels, value_area_pct, 0.0)
    prices = levels[:, _FP_LEVEL_COL["price"]]
    volumes = levels[:, _FP_LEVEL_COL["vol_total"]]
    if float(volumes.sum()) <= 0.0:
        return None
    vol_bid = levels[:, _FP_LEVEL_COL["vol_bid"]]
    vol_ask = levels[:, _FP_LEVEL_COL["vol_ask"]]
    deltas = levels[:, _FP_LEVEL_COL["delta"]]

    hvn: list[VolumeNode] = []
    lvn: list[VolumeNode] = []
    if nodes is not None:
        detected = detect_volume_nodes(
            prices, volumes, kind=nodes,
            prominence=node_prominence, max_nodes=max_nodes,
        )
        hvn = [n for n in detected if n.kind == "hvn"]
        lvn = [n for n in detected if n.kind == "lvn"]

    return {
        "poc": float(scal[_FP_SCALAR_COL["poc_price"]]),
        "vah": float(scal[_FP_SCALAR_COL["vah"]]),
        "val": float(scal[_FP_SCALAR_COL["val"]]),
        "hvn": hvn,
        "lvn": lvn,
        "max_vol": float(volumes.max()),
        "prices": prices,
        "volumes": volumes,
        "vol_bid": vol_bid,
        "vol_ask": vol_ask,
        "deltas": deltas,
    }


def compute_range_profile(
    ts: np.ndarray,
    *,
    is_tick: bool,
    start: int | None,
    end: int | None,
    tick_size: float | None,
    bins: int,
    value_area_pct: float,
    nodes: str | None = None,
    node_prominence: float = 0.1,
    max_nodes: int | None = None,
    # tick source columns
    price: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    # kline source columns
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
) -> dict | None:
    """
    Compute a full volume-by-price profile over an explicit [start, end] range.

    This is the on-demand counterpart of the developing scans: it reads a slice
    of the source buffer bounded by timestamp and builds a single aggregated
    profile (no per-period reset, no look-ahead handling — the caller is
    responsible for passing only causal data). Used by the FixedRangeVP tool and
    by RollingVolumeProfile's node access.

    Rows with ``start <= ts <= end`` are included (None means open-ended).

    Args:
        ts (np.ndarray): Timestamp per row (ms), ascending.
        is_tick (bool): True for tick sources (price/volume/flags), False for
            kline sources (open/high/low/close/volume).
        start (int | None): Inclusive range start ts. None -> oldest row.
        end (int | None): Inclusive range end ts. None -> newest row.
        tick_size (float | None): Level size; inferred from the slice when None.
        bins (int): Target number of levels for inference when tick_size is None.
        value_area_pct (float): Fraction of volume defining the value area.
        nodes (str | None): None | 'hvn' | 'lvn' | 'both'.
        node_prominence (float): Node prominence threshold 0..1.
        max_nodes (int | None): Optional cap per node kind.
        price, volume, flags (np.ndarray | None): Tick columns (is_tick=True).
        open_, high, low, close, volume (np.ndarray | None): Kline columns.

    Returns:
        dict | None: Profile dict with keys 'start', 'end', 'tick_size', plus
        poc/vah/val/hvn/lvn and histogram arrays; None when the range is empty.
    """
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    if n == 0:
        return None

    lo = 0 if start is None else int(np.searchsorted(ts, int(start), side="left"))
    hi = n if end is None else int(np.searchsorted(ts, int(end), side="right"))
    if hi <= lo:
        return None

    if is_tick:
        p = np.asarray(price, dtype=np.float64)[lo:hi]
        v = np.asarray(volume, dtype=np.float64)[lo:hi]
        f = np.asarray(flags, dtype=np.int64)[lo:hi]
        if len(p) == 0:
            return None
        ts_eff = float(tick_size) if (tick_size and tick_size > 0) else infer_tick_size(p, bins)
        levels, c_bid, c_ask, c_cnt = tick_contributions(p, v, f, ts_eff)
    else:
        o = np.asarray(open_, dtype=np.float64)[lo:hi]
        h = np.asarray(high, dtype=np.float64)[lo:hi]
        l = np.asarray(low, dtype=np.float64)[lo:hi]
        c = np.asarray(close, dtype=np.float64)[lo:hi]
        vol = np.asarray(volume, dtype=np.float64)[lo:hi]
        if len(h) == 0:
            return None
        ts_eff = (
            float(tick_size)
            if (tick_size and tick_size > 0)
            else infer_tick_size(np.concatenate([h, l]), bins)
        )
        is_bull = c >= o
        _rows, levels, c_bid, c_ask, c_cnt = kline_contributions(h, l, vol, is_bull, ts_eff)

    levels = histogram_from_contributions(levels, c_bid, c_ask, c_cnt)
    prof = _profile_from_levels(
        levels, value_area_pct, nodes, node_prominence, max_nodes
    )
    if prof is None:
        return None
    prof["start"] = int(ts[lo])
    prof["end"] = int(ts[hi - 1])
    prof["tick_size"] = ts_eff
    return prof


# =====
# Histogram geometry (Bokeh-free)
# =====
#
# The volume-by-price histogram is geometry, not a time series, so it is emitted
# as draw primitives (HBars) instead of going through the per-bar series
# renderer. The array-building here is shared by both render paths so they stay
# identical:
#   - the indicator's draw() wraps these arrays in an HBars primitive;
#   - plotting.sources.build_volume_profile_source wraps them in a Bokeh source.
#
# Geometry of the VPVR (view="visible") layout: the merged profile is anchored
# to the right of the last candle. _VP_VISIBLE_GAP_BARS candles of separation
# plus _VP_VISIBLE_BARS_WIDE candles of POC width set how far the histogram
# extends to the right of the live edge.
_VP_VISIBLE_BARS_WIDE = 14
_VP_VISIBLE_GAP_BARS = 3


def vp_visible_right_pad_bars() -> int:
    """
    Candles the VPVR (view='visible') histogram occupies to the right of the
    last candle. The live navigation reserves this padding so the profile stays
    in view during follow.
    """
    return _VP_VISIBLE_GAP_BARS + _VP_VISIBLE_BARS_WIDE


def _merge_profiles_by_price(profiles: "list[dict]") -> "dict | None":
    """
    Aggregate every per-period profile into a single price->volume histogram.

    Used by the 'visible' (VPVR) layout. Levels are merged by exact price; the
    POC of the merged profile is the price with the largest total volume.

    Args:
        profiles (list[dict]): Per-period profiles from period_profiles().

    Returns:
        dict | None: A profile-shaped dict (prices, volumes, vol_bid, vol_ask,
        deltas, poc, vah, val, max_vol) or None when there is no volume to draw.
    """
    acc: "dict[float, list[float]]" = {}
    for prof in profiles:
        prices = np.asarray(prof["prices"], dtype=np.float64)
        vols = np.asarray(prof["volumes"], dtype=np.float64)
        v_bid = np.asarray(prof["vol_bid"], dtype=np.float64)
        v_ask = np.asarray(prof["vol_ask"], dtype=np.float64)
        for price, vol, vb, va in zip(prices, vols, v_bid, v_ask):
            entry = acc.get(price)
            if entry is None:
                entry = [0.0, 0.0, 0.0]
                acc[price] = entry
            entry[0] += float(vol)
            entry[1] += float(vb)
            entry[2] += float(va)
    if not acc:
        return None

    prices = np.array(sorted(acc.keys()), dtype=np.float64)
    vols = np.array([acc[p][0] for p in prices], dtype=np.float64)
    v_bid = np.array([acc[p][1] for p in prices], dtype=np.float64)
    v_ask = np.array([acc[p][2] for p in prices], dtype=np.float64)
    poc_idx = int(np.argmax(vols))
    poc_price = float(prices[poc_idx])

    # Value area of the merged profile: expand from the POC until it covers 70%
    # of the volume, same logic as _compute_scalars (footprint).
    va_target = float(vols.sum()) * 0.70
    va_vol = float(vols[poc_idx])
    lo = hi = poc_idx
    while va_vol < va_target:
        can_up = hi + 1 < len(vols)
        can_down = lo - 1 >= 0
        if not can_up and not can_down:
            break
        vol_up = float(vols[hi + 1]) if can_up else -1.0
        vol_down = float(vols[lo - 1]) if can_down else -1.0
        if vol_up >= vol_down:
            hi += 1
            va_vol += vol_up
        else:
            lo -= 1
            va_vol += vol_down

    return {
        "prices": prices,
        "volumes": vols,
        "vol_bid": v_bid,
        "vol_ask": v_ask,
        "deltas": v_ask - v_bid,
        "poc": poc_price,
        "vah": float(prices[hi]),
        "val": float(prices[lo]),
        "max_vol": float(vols.max()) if len(vols) else 0.0,
    }


def volume_profile_bar_arrays(
    profiles: "list[dict]",
    tick_size: float,
    view: str = "session",
    *,
    buy_color: str = DEFAULT_VP_BUY,
    sell_color: str = DEFAULT_VP_SELL,
    interval_ms: "int | None" = None,
    width_fraction: float = 0.32,
) -> "dict | None":
    """
    Build the parallel arrays for the volume-by-price histogram bars.

    Each price level produces two stacked horizontal segments - sell-aggressor
    (bid) volume and buy-aggressor (ask) volume - so the bar is split in two
    tones. The combined bar length is proportional to that level's total volume
    relative to the largest level. Levels inside the value area (VAL..VAH) are
    drawn opaque; levels outside are dimmed. Timestamps are epoch-ms integers
    (the caller converts to datetime64 / wraps in a primitive).

    Two layouts are supported through ``view``:

    - 'session' (default): one profile per finished period, anchored at the
      period's left edge and growing right (TradingView VPSV).
    - 'visible': every level across all periods is merged by price into a single
      profile anchored at the right edge and growing left (TradingView VPVR).

    Args:
        profiles (list[dict]): Per-period profiles from period_profiles().
        tick_size (float): Price level size (0 -> inferred from spacing).
        view (str): 'session' or 'visible'.
        buy_color (str): Color of the buy-aggressor (ask) segment.
        sell_color (str): Color of the sell-aggressor (bid) segment.
        interval_ms (int | None): OHLC interval, used for VPVR anchoring.
        width_fraction (float): Max fraction of the period width used by a full
            (max-volume) bar.

    Returns:
        dict | None: Parallel arrays {y, height, left, right, color, alpha,
        side, volume, vol_buy, vol_sell} (left/right are int epoch-ms), or None
        when there is nothing to draw.
    """
    if not profiles:
        return None
    tick_size = float(tick_size or 0.0)

    if view == "visible":
        merged = _merge_profiles_by_price(profiles)
        if merged is None:
            return None
        last_ts = int(max(int(p["ts_end"]) for p in profiles))
        first_ts = int(min(int(p["ts_start"]) for p in profiles))
        bar = int(interval_ms) if interval_ms and interval_ms > 0 else max(
            1, (last_ts - first_ts) // 100
        )
        prof_real = _VP_VISIBLE_BARS_WIDE * bar
        anchor = last_ts + _VP_VISIBLE_GAP_BARS * bar + prof_real
        vis_width = int(prof_real / width_fraction) if width_fraction else prof_real
        render_units = [(merged, anchor, vis_width, -1)]
    else:
        render_units = [
            (prof, int(prof["ts_start"]),
             int(prof["ts_end"]) - int(prof["ts_start"]), +1)
            for prof in profiles
        ]

    ys: "list[float]" = []
    heights: "list[float]" = []
    lefts: "list[int]" = []
    rights: "list[int]" = []
    colors: "list[str]" = []
    alphas: "list[float]" = []
    sides: "list[str]" = []
    volumes: "list[float]" = []
    vol_buys: "list[float]" = []
    vol_sells: "list[float]" = []

    for prof, anchor, width, direction in render_units:
        prices = np.asarray(prof["prices"], dtype=np.float64)
        if len(prices) == 0:
            continue
        vols = np.asarray(prof["volumes"], dtype=np.float64)
        v_bid = np.asarray(prof["vol_bid"], dtype=np.float64)
        v_ask = np.asarray(prof["vol_ask"], dtype=np.float64)
        max_vol = float(prof.get("max_vol") or (vols.max() if len(vols) else 0.0)) or 1.0
        poc = float(prof["poc"])
        vah = float(prof.get("vah", np.nan))
        val = float(prof.get("val", np.nan))
        profile_width = width * width_fraction

        # Bar height: tick_size with a robust minimum from the real spacing.
        if tick_size > 0:
            bar_h = tick_size * 0.9
            half_h = tick_size / 2.0
        elif len(prices) > 1:
            spacing = float(np.min(np.diff(np.sort(prices))))
            bar_h = spacing * 0.9
            half_h = spacing / 2.0
        else:
            bar_h = max(abs(poc) * 0.001, 1.0)
            half_h = bar_h / 2.0

        has_va = np.isfinite(vah) and np.isfinite(val)
        va_lo = min(val, vah) - half_h
        va_hi = max(val, vah) + half_h

        for price, vol, vb, va in zip(prices, vols, v_bid, v_ask):
            if vol <= 0:
                continue
            bar_len = (vol / max_vol) * profile_width
            sell_len = (vb / vol) * bar_len if vol > 0 else 0.0
            in_va = has_va and (va_lo <= price <= va_hi)
            bar_alpha = 0.85 if (in_va or not has_va) else 0.25

            buy_len = bar_len - sell_len
            buy_left = anchor
            buy_right = anchor + direction * buy_len
            sell_left = buy_right
            sell_right = anchor + direction * bar_len

            for left, right, side, color, raw_vol in (
                (sell_left, sell_right, "sell", sell_color, vb),
                (buy_left, buy_right, "buy", buy_color, va),
            ):
                if raw_vol <= 0:
                    continue
                ys.append(float(price))
                heights.append(bar_h)
                lefts.append(int(min(left, right)))
                rights.append(int(max(left, right)))
                colors.append(color)
                alphas.append(bar_alpha)
                sides.append(side)
                volumes.append(float(vol))
                vol_buys.append(float(va))
                vol_sells.append(float(vb))

    if not ys:
        return None

    return {
        "y": np.asarray(ys, dtype=np.float64),
        "height": np.asarray(heights, dtype=np.float64),
        "left": lefts,
        "right": rights,
        "color": colors,
        "alpha": np.asarray(alphas, dtype=np.float64),
        "side": sides,
        "volume": np.asarray(volumes, dtype=np.float64),
        "vol_buy": np.asarray(vol_buys, dtype=np.float64),
        "vol_sell": np.asarray(vol_sells, dtype=np.float64),
    }

"""
Pure, vectorized core for order-flow trade analysis.

This module is Bokeh-free and engine-free: it only manipulates NumPy arrays so
the same logic can be reused by the precomputed backtest path and by the live
streaming path, guaranteeing identical detection and styling in both.

Provided building blocks:

- ``classify_aggressor`` - per-trade aggressor side (+1 buy / -1 sell) from the
  tick ``flags`` (supporting both the sign encoding ``+1/-1`` used by the
  connectors and the bit encoding ``32/64`` used elsewhere), with a quote-rule
  and a tick-rule fallback.
- ``trade_metric`` - the magnitude used to rank trades: raw ``volume``,
  ``notional`` (price * volume) or net ``delta``.
- ``aggregate_trades`` - merge same-execution-burst trades into synthetic events
  (same-side for volume/notional, net signed for delta) within a time window.
- ``detect_large_trades`` - causal detection of large trades / bursts against an
  absolute, quantile or median-multiple threshold, with optional aggregation and
  anti-clustering.
- ``map_bubble_style`` - map trade magnitude to a bubble size (sqrt-area or
  linear, clamped) and the aggressor side to a color.
- ``format_magnitude`` - compact human-readable label for a magnitude number.
"""

from __future__ import annotations

import numpy as np

# Buy/sell colors are bound to the aggressor side. Defaults requested:
# buy = blue, sell = pink.
DEFAULT_BUY_COLOR = "#359EF5"
DEFAULT_SELL_COLOR = "#F53C79"
_NEUTRAL_COLOR = "#9CA3AF"

# Tick flag bit masks (same convention as the volume profile core).
_FLAG_BIT_BUY = 32
_FLAG_BIT_SELL = 64


def classify_aggressor(
    price: np.ndarray,
    flags: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Classify the aggressor side of each trade.

    Resolution order, per trade:
        1. ``flags`` - sign encoding (+1 buy / -1 sell) or bit encoding
           (bit 32 buy / bit 64 sell).
        2. Quote rule - price at/above ask is a buy, at/below bid is a sell
           (only when bid/ask are finite).
        3. Tick rule - price rising (or unchanged) vs the previous trade is a
           buy, falling is a sell. The first trade defaults to buy.

    Args:
        price (np.ndarray): Trade prices, shape [N].
        flags (np.ndarray | None): Per-trade side flags. None to skip step 1.
        bid (np.ndarray | None): Best bid per trade. None to skip the quote rule.
        ask (np.ndarray | None): Best ask per trade. None to skip the quote rule.

    Returns:
        np.ndarray: int8 array [N] with +1 (buy) or -1 (sell).
    """
    price = np.asarray(price, dtype=np.float64)
    n = price.shape[0]
    side = np.zeros(n, dtype=np.int8)
    if n == 0:
        return side

    if flags is not None:
        f = np.asarray(flags, dtype=np.float64)
        fi = f.astype(np.int64)
        # Sign encoding first so a negative flag is never misread by the bit
        # test (a negative integer has many bits set in two's complement).
        buy_flag = (f == 1.0) | ((fi > 0) & ((fi & _FLAG_BIT_BUY) != 0))
        sell_flag = (f == -1.0) | ((fi > 0) & ((fi & _FLAG_BIT_SELL) != 0))
        side[buy_flag] = 1
        side[sell_flag] = -1

    unknown = side == 0
    if unknown.any() and bid is not None and ask is not None:
        b = np.asarray(bid, dtype=np.float64)
        a = np.asarray(ask, dtype=np.float64)
        # Only apply the quote rule where there is a real spread (ask > bid).
        # A TradeEvent carries no quote, so its row has bid == ask == price;
        # without this gate both `price >= ask` and `price <= bid` are true and
        # the sell branch (applied last) would misclassify every trade as a
        # sell. A degenerate/locked book carries no aggressor information, so
        # such rows fall through to the tick rule instead.
        spread = np.isfinite(a) & np.isfinite(b) & (a > b)
        q_buy = unknown & spread & (price >= a)
        q_sell = unknown & spread & (price <= b)
        side[q_buy] = 1
        side[q_sell] = -1
        unknown = side == 0

    if unknown.any():
        diff = np.empty(n, dtype=np.float64)
        diff[0] = 0.0
        diff[1:] = np.diff(price)
        # Carry the last non-zero direction forward so flat prints inherit the
        # previous trade's sign instead of defaulting to buy.
        last_dir = 0
        tick = np.ones(n, dtype=np.int8)
        for i in range(n):
            if diff[i] > 0:
                last_dir = 1
            elif diff[i] < 0:
                last_dir = -1
            tick[i] = -1 if last_dir < 0 else 1
        side[unknown] = tick[unknown]

    return side


def trade_metric(
    price: np.ndarray,
    volume: np.ndarray,
    by: str = "volume",
    side: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the per-trade magnitude used to rank trades.

    Args:
        price (np.ndarray): Trade prices, shape [N].
        volume (np.ndarray): Trade volumes, shape [N].
        by (str): Magnitude metric:
            - 'volume'   - raw traded size.
            - 'notional' - price * volume.
            - 'delta'    - net aggression magnitude |volume * side|. Per trade
              this equals the raw volume (a single print is fully one-sided);
              the delta metric only becomes distinct from 'volume' once trades
              are aggregated into bursts (see ``aggregate_trades``), where buys
              and sells inside the window cancel into a net delta.
        side (np.ndarray | None): Aggressor side (+1/-1) per trade. Only used by
            'delta'; ignored otherwise.

    Returns:
        np.ndarray: float64 array [N] of magnitudes.

    Raises:
        ValueError: If ``by`` is not 'volume', 'notional' or 'delta'.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if by == "volume":
        return np.abs(volume)
    if by == "notional":
        return np.abs(np.asarray(price, dtype=np.float64) * volume)
    if by == "delta":
        if side is None:
            return np.abs(volume)
        return np.abs(volume * np.asarray(side, dtype=np.float64))
    raise ValueError(f"by must be 'volume', 'notional' or 'delta', not {by!r}")


def _rolling_threshold(metric: np.ndarray, threshold, window: int) -> np.ndarray:
    """
    Build the causal per-trade acceptance threshold array.

    Args:
        metric (np.ndarray): Per-trade magnitudes [N].
        threshold (float | str): Absolute float, 'pXX' quantile, or 'Nx' median
            multiple. Quantile/median are computed over the trailing ``window``
            trades (inclusive), so no future trade influences the threshold.
        window (int): Trailing window length for relative thresholds.

    Returns:
        np.ndarray: float64 threshold per trade [N]; NaN during warmup (when the
            window is not yet full) so early trades are not flagged.

    Raises:
        ValueError: If a string threshold is malformed.
    """
    n = metric.shape[0]

    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        return np.full(n, float(threshold), dtype=np.float64)

    if not isinstance(threshold, str):
        raise ValueError(f"threshold must be float or str, not {type(threshold)!r}")

    spec = threshold.strip().lower()
    win = max(int(window), 1)

    import pandas as pd

    s = pd.Series(metric, dtype="float64")
    roll = s.rolling(window=win, min_periods=win)

    if spec.startswith("p"):
        try:
            q = float(spec[1:]) / 100.0
        except ValueError as exc:
            raise ValueError(
                f"quantile threshold must look like 'p99', got {threshold!r}"
            ) from exc
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile out of range (0,1): {threshold!r}")
        return roll.quantile(q).to_numpy(dtype=np.float64)

    if spec.endswith("x"):
        try:
            mult = float(spec[:-1])
        except ValueError as exc:
            raise ValueError(
                f"multiple threshold must look like '5x', got {threshold!r}"
            ) from exc
        if mult <= 0.0:
            raise ValueError(f"multiple must be > 0: {threshold!r}")
        return (roll.median() * mult).to_numpy(dtype=np.float64)

    raise ValueError(
        f"threshold string must be 'pXX' (quantile) or 'Nx' (median multiple), "
        f"got {threshold!r}"
    )


def _apply_min_gap(idx: np.ndarray, ts: np.ndarray, min_gap_ms: int) -> np.ndarray:
    """
    Suppress detections closer than ``min_gap_ms`` to the previous accepted one.

    Walks detections in time order, keeping a trade only when enough time has
    passed since the last kept trade (causal anti-clustering).

    Args:
        idx (np.ndarray): Indices of candidate detections (ascending).
        ts (np.ndarray): Timestamps (ms) for all trades.
        min_gap_ms (int): Minimum spacing in ms between kept detections.

    Returns:
        np.ndarray: Filtered indices.
    """
    if min_gap_ms <= 0 or idx.size == 0:
        return idx
    kept = []
    last_ts = None
    for i in idx:
        t = int(ts[i])
        if last_ts is None or (t - last_ts) >= min_gap_ms:
            kept.append(i)
            last_ts = t
    return np.asarray(kept, dtype=np.int64)


def aggregate_trades(
    ts: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    side: np.ndarray,
    *,
    aggregate_ms: int = 0,
    by: str = "volume",
) -> dict:
    """
    Merge trades from the same execution burst into synthetic events.

    Institutional orders often arrive as a rapid sequence of child prints in the
    same millisecond range; ranking them one by one dilutes the signal. This
    groups consecutive trades into causal time bursts (a burst spans at most
    ``aggregate_ms`` from its first trade) and emits one synthetic event per
    burst. Grouping depends on the metric:

    - 'volume' / 'notional': only consecutive trades of the SAME aggressor side
      are merged (a split market order), summing their size / notional.
    - 'delta': ALL consecutive trades in the window are merged regardless of
      side into a NET signed delta (buys minus sells); the event side is the
      sign of that net and the ranking metric is its absolute value, so a burst
      that is evenly two-sided ranks low even if its gross volume is large.

    With ``aggregate_ms <= 0`` no merging happens: every trade is its own event
    (identical to the non-aggregated path).

    Note: a burst's membership can grow until ``aggregate_ms`` has elapsed past
    its first trade, so in live mode the most recent (trailing) burst may still
    be incomplete. Closed bursts are causal and stable.

    Args:
        ts (np.ndarray): Trade timestamps in ms [N] (non-decreasing).
        price (np.ndarray): Trade prices [N].
        volume (np.ndarray): Trade volumes [N] (magnitude; sign comes from side).
        side (np.ndarray): Aggressor side per trade [N] (+1 buy / -1 sell).
        aggregate_ms (int): Burst window in ms (0 disables aggregation).
        by (str): 'volume', 'notional' or 'delta' (drives grouping and metric).

    Returns:
        dict: Burst-level arrays, one entry per event:
            - 'rep_idx' (int64 [M]): index of the burst's last trade (the tick
              the event is anchored at).
            - 'ts' (int64 [M]) / 'price' (float64 [M]): anchor tick ts / price.
            - 'volume' (float64 [M]): summed traded size in the burst.
            - 'notional' (float64 [M]): summed price * volume in the burst.
            - 'delta' (float64 [M]): net signed volume (buys - sells).
            - 'side' (int8 [M]): event aggressor side.
            - 'metric' (float64 [M]): ranking magnitude per ``by``.
    """
    ts = np.asarray(ts, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    side = np.asarray(side, dtype=np.int8)
    n = ts.shape[0]

    if n == 0 or aggregate_ms <= 0:
        rep = np.arange(n, dtype=np.int64)
        vol = np.abs(volume)
        notional = np.abs(price * volume)
        delta = vol * side.astype(np.float64)
        if by == "notional":
            metric = notional
        elif by == "delta":
            metric = np.abs(delta)
        else:
            metric = vol
        return {
            "rep_idx": rep,
            "ts": ts.astype(np.int64),
            "price": price,
            "volume": vol,
            "notional": notional,
            "delta": delta,
            "side": side,
            "metric": metric,
        }

    same_side = by != "delta"
    win = float(aggregate_ms)

    rep_idx = []
    b_ts = []
    b_price = []
    b_vol = []
    b_not = []
    b_delta = []
    b_side = []
    b_metric = []

    i = 0
    while i < n:
        t0 = ts[i]
        s0 = int(side[i])
        vol = 0.0
        notl = 0.0
        delta = 0.0
        j = i
        while j < n and (ts[j] - t0) <= win and (not same_side or int(side[j]) == s0):
            v = abs(float(volume[j]))
            vol += v
            notl += abs(float(price[j]) * float(volume[j]))
            delta += v * float(side[j])
            j += 1
        rep = j - 1
        rep_idx.append(rep)
        b_ts.append(int(ts[rep]))
        b_price.append(float(price[rep]))
        b_vol.append(vol)
        b_not.append(notl)
        b_delta.append(delta)
        if by == "delta":
            sd = 1 if delta > 0 else (-1 if delta < 0 else 0)
            met = abs(delta)
        else:
            sd = s0
            met = vol if by == "volume" else notl
        b_side.append(sd)
        b_metric.append(met)
        i = j

    return {
        "rep_idx": np.asarray(rep_idx, dtype=np.int64),
        "ts": np.asarray(b_ts, dtype=np.int64),
        "price": np.asarray(b_price, dtype=np.float64),
        "volume": np.asarray(b_vol, dtype=np.float64),
        "notional": np.asarray(b_not, dtype=np.float64),
        "delta": np.asarray(b_delta, dtype=np.float64),
        "side": np.asarray(b_side, dtype=np.int8),
        "metric": np.asarray(b_metric, dtype=np.float64),
    }


def detect_large_trades(
    ts: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    flags: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    *,
    threshold="p99",
    by: str = "volume",
    window: int = 2000,
    min_gap_ms: int = 0,
    aggregate_ms: int = 0,
) -> dict:
    """
    Detect large trades causally, with optional execution-burst aggregation.

    A trade (or aggregated burst) is large when its magnitude (``by``) meets the
    threshold. Relative thresholds ('pXX', 'Nx') use only the trailing
    ``window`` events, so a detection never depends on future data and the
    result is safe for backtesting and live use alike. When ``aggregate_ms`` is
    set, trades are first merged into bursts (see ``aggregate_trades``) and the
    threshold / window then operate over BURSTS rather than raw trades.

    Args:
        ts (np.ndarray): Trade timestamps in ms [N].
        price (np.ndarray): Trade prices [N].
        volume (np.ndarray): Trade volumes [N].
        flags (np.ndarray | None): Per-trade side flags (see classify_aggressor).
        bid (np.ndarray | None): Best bid per trade (quote-rule fallback).
        ask (np.ndarray | None): Best ask per trade (quote-rule fallback).
        threshold (float | str): Absolute float, 'pXX' quantile, or 'Nx' median
            multiple.
        by (str): 'volume', 'notional' or 'delta' (net aggression).
        window (int): Trailing window (in events) for relative thresholds.
        min_gap_ms (int): Minimum ms spacing between detections (0 disables it).
        aggregate_ms (int): Execution-burst window in ms (0 disables it).

    Returns:
        dict: Detection result with:
            - 'mask' (bool [N]): True at the anchor tick of each detected event.
            - 'metric' (float64 [N]): event magnitude at anchor ticks, NaN
              elsewhere.
            - 'side' (int8 [N]): per-trade aggressor side, overwritten with the
              event side at anchor ticks.
            - 'events' (dict): compact arrays for detected events only, with keys
              'ts', 'price', 'volume', 'notional', 'delta', 'side', 'metric'.
    """
    ts = np.asarray(ts, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = price.shape[0]

    side = classify_aggressor(price, flags=flags, bid=bid, ask=ask)

    empty_events = {
        "ts": np.zeros(0, dtype=np.int64),
        "price": np.zeros(0, dtype=np.float64),
        "volume": np.zeros(0, dtype=np.float64),
        "notional": np.zeros(0, dtype=np.float64),
        "delta": np.zeros(0, dtype=np.float64),
        "side": np.zeros(0, dtype=np.int8),
        "metric": np.zeros(0, dtype=np.float64),
    }
    if n == 0:
        return {
            "mask": np.zeros(0, dtype=bool),
            "metric": np.zeros(0, dtype=np.float64),
            "side": side,
            "events": empty_events,
        }

    bursts = aggregate_trades(
        ts, price, volume, side, aggregate_ms=aggregate_ms, by=by
    )
    b_metric = bursts["metric"]
    b_ts = bursts["ts"].astype(np.float64)

    thr = _rolling_threshold(b_metric, threshold, window)
    with np.errstate(invalid="ignore"):
        accept = np.isfinite(thr) & (b_metric >= thr) & (b_metric > 0.0)

    sel = np.where(accept)[0]
    if min_gap_ms > 0 and sel.size:
        sel = _apply_min_gap(sel, b_ts, int(min_gap_ms))

    metric_n = np.full(n, np.nan, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    side_n = side.copy()

    rep = bursts["rep_idx"]
    if sel.size:
        anchor = rep[sel]
        mask[anchor] = True
        metric_n[anchor] = b_metric[sel]
        side_n[anchor] = bursts["side"][sel]

    events = {
        "ts": bursts["ts"][sel],
        "price": bursts["price"][sel],
        "volume": bursts["volume"][sel],
        "notional": bursts["notional"][sel],
        "delta": bursts["delta"][sel],
        "side": bursts["side"][sel],
        "metric": b_metric[sel],
    } if sel.size else empty_events

    return {"mask": mask, "metric": metric_n, "side": side_n, "events": events}


def map_bubble_style(
    metric: np.ndarray,
    side: np.ndarray,
    *,
    scale: str = "sqrt",
    min_size: float = 6.0,
    max_size: float = 40.0,
    buy_color: str = DEFAULT_BUY_COLOR,
    sell_color: str = DEFAULT_SELL_COLOR,
) -> "tuple[np.ndarray, list]":
    """
    Map trade magnitudes to bubble sizes and aggressor sides to colors.

    With ``scale='sqrt'`` the bubble *area* grows with the magnitude (radius
    proportional to sqrt(magnitude)), which is the perceptually correct mapping
    for circle markers. ``scale='linear'`` maps magnitude directly to diameter.
    Sizes are normalized across the provided magnitudes and clamped to
    ``[min_size, max_size]``.

    Args:
        metric (np.ndarray): Magnitudes of the detected trades [M].
        side (np.ndarray): Aggressor side per detected trade [M] (+1 / -1).
        scale (str): 'sqrt' (area-proportional) or 'linear'.
        min_size (float): Smallest bubble diameter in px.
        max_size (float): Largest bubble diameter in px.
        buy_color (str): Color for buy-aggressor bubbles.
        sell_color (str): Color for sell-aggressor bubbles.

    Returns:
        tuple[np.ndarray, list]:
            - sizes: float64 [M] diameters in px.
            - colors: list[str] [M] per-bubble colors.

    Raises:
        ValueError: If ``scale`` is not 'sqrt' or 'linear'.
    """
    metric = np.asarray(metric, dtype=np.float64)
    side = np.asarray(side)
    m = metric.shape[0]

    if scale not in ("sqrt", "linear"):
        raise ValueError(f"scale must be 'sqrt' or 'linear', not {scale!r}")

    lo = float(min(min_size, max_size))
    hi = float(max(min_size, max_size))

    if m == 0:
        sizes = np.zeros(0, dtype=np.float64)
    else:
        base = np.sqrt(np.clip(metric, 0.0, None)) if scale == "sqrt" else np.clip(metric, 0.0, None)
        bmin = float(np.nanmin(base))
        bmax = float(np.nanmax(base))
        if not np.isfinite(bmin) or not np.isfinite(bmax) or bmax <= bmin:
            # Single trade or constant magnitude: use a mid-size bubble.
            sizes = np.full(m, (lo + hi) / 2.0, dtype=np.float64)
        else:
            norm = (base - bmin) / (bmax - bmin)
            sizes = lo + (hi - lo) * norm

    colors = [
        buy_color if s > 0 else (sell_color if s < 0 else _NEUTRAL_COLOR)
        for s in side
    ]
    return sizes, colors


def format_magnitude(value: float) -> str:
    """
    Format a magnitude as a compact human-readable number.

    Examples:
        1_800_000 -> '1.8M'
        12_500    -> '12.5k'
        7.0       -> '7'
        0.42      -> '0.42'

    Args:
        value (float): The magnitude (volume or notional).

    Returns:
        str: Compact label.
    """
    if value is None or not np.isfinite(value):
        return ""
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.1f}B"
    if a >= 1e6:
        return f"{value / 1e6:.1f}M"
    if a >= 1e3:
        return f"{value / 1e3:.1f}k"
    if a >= 1.0:
        return f"{value:.0f}"
    return f"{value:.2f}"



def build_bubble_columns(events: dict, style: dict, labels=None) -> dict:
    """
    Build the column arrays for the deep-trades bubble glyph.

    Bokeh-free: produces plain arrays/lists so both the static (backtest) and
    live render paths build identical glyphs. The caller converts ``ts`` (epoch
    ms) into ``datetime64[ms]`` for the figure's x axis.

    Args:
        events (dict): Detected events with keys 'ts', 'price', 'side', and the
            magnitude arrays 'volume'/'notional'/'metric'.
        style (dict): Style snapshot (scale, min/max size, colors, label, by).
        labels (list | None): Precomputed label strings. When None they are
            derived from the magnitude selected by ``style['label']``.

    Returns:
        dict: {'ts' (int64 ms), 'price' (float64), 'size' (float64),
        'color' (list[str]), 'text' (list[str])}. Empty arrays when no events.
    """
    ts = np.asarray(events.get("ts", []), dtype=np.int64)
    price = np.asarray(events.get("price", []), dtype=np.float64)
    side = np.asarray(events.get("side", []))
    metric = np.asarray(events.get("metric", []), dtype=np.float64)

    sizes, colors = map_bubble_style(
        metric, side,
        scale=style.get("scale", "sqrt"),
        min_size=style.get("min_size", 6.0),
        max_size=style.get("max_size", 40.0),
        buy_color=style.get("buy_color", DEFAULT_BUY_COLOR),
        sell_color=style.get("sell_color", DEFAULT_SELL_COLOR),
    )

    if labels is None:
        label_key = style.get("label")
        if label_key:
            src = np.asarray(events.get(label_key, []), dtype=np.float64)
            labels = [format_magnitude(float(v)) for v in src]
        else:
            labels = [""] * len(ts)
    else:
        labels = list(labels)
    if len(labels) != len(ts):
        labels = (labels + [""] * len(ts))[: len(ts)]

    return {
        "ts": ts,
        "price": price,
        "size": sizes,
        "color": list(colors),
        "text": labels,
    }


def accumulate_events(prev: dict, new: dict, *, min_ts=None) -> dict:
    """
    Merge two causal event dicts by unique anchor ts (schema-agnostic).

    The live/replay refresh recomputes large-trade detection over the current
    tick-ring window, whose oldest ``window`` events fall in the relative-
    threshold warmup (NaN threshold -> never flagged). A bubble that was flagged
    while its tick was fresh (full trailing window, i.e. the exact same causal
    decision the full-series backtest makes once) would otherwise vanish as soon
    as its tick drifts into that warmup band. Merging freshly-detected events
    into the ones already seen keeps a detection for as long as its tick stays
    in the window, restoring backtest parity within the visible window.

    Every per-event column (``price``, ``side``, ``metric``, and any extra
    columns such as DeepTrades' ``event_type`` / ``resting``) is carried through
    unchanged, so the merge works for LargeTrades, DeepTrades and the Heatmap
    bubbles alike. Events are keyed by integer ts; a ts present in ``new``
    overrides ``prev`` (the latest detection wins). Events older than ``min_ts``
    (the oldest tick still in the window) are pruned so the accumulator stays
    bounded to the visible span. The result is sorted ascending by ts.

    Args:
        prev (dict): Previously accumulated events (same schema), may be empty.
        new (dict): Freshly detected events for the current window.
        min_ts (int | None): Drop events with ts < min_ts (None keeps all).

    Returns:
        dict: Merged events with the same keys as the input dicts, sorted by ts.
    """
    prev = prev or {}
    new = new or {}

    # Resolve the value-column schema from the newest non-empty dict so any
    # extra columns (DeepTrades classes) are honored; 'ts' is the merge key.
    new_has = np.asarray(new.get("ts", [])).size > 0
    schema_src = new if new_has else prev
    value_keys = [k for k in schema_src.keys() if k != "ts"]

    merged: dict = {}

    def _ingest(ev: dict) -> None:
        ts = np.asarray(ev.get("ts", []), dtype=np.int64)
        if ts.size == 0:
            return
        cols = [np.asarray(ev.get(k)) for k in value_keys]
        for i in range(ts.size):
            merged[int(ts[i])] = tuple(col[i] for col in cols)

    _ingest(prev)
    _ingest(new)

    if min_ts is not None:
        cutoff = int(min_ts)
        merged = {t: v for t, v in merged.items() if t >= cutoff}

    order = sorted(merged)
    out = {"ts": np.asarray(order, dtype=np.int64)}
    for j, k in enumerate(value_keys):
        ref = np.asarray(schema_src.get(k))
        dtype = ref.dtype if ref.size else np.float64
        if order:
            out[k] = np.asarray([merged[t][j] for t in order], dtype=dtype)
        else:
            out[k] = np.zeros(0, dtype=dtype)
    return out


def merge_live_events(prev, new, window_ts, last_max_ts):
    """
    Stateful glue for the live/replay bubble accumulator.

    Wraps ``accumulate_events`` with the window bookkeeping the indicators need:
    it prunes to the oldest tick in the current window and detects a replay
    rewind (the window's newest ts moved backwards) to drop the stale
    accumulator so a rewound replay starts clean.

    Args:
        prev (dict | None): The indicator's accumulated events so far.
        new (dict): Events freshly detected over the current window.
        window_ts (np.ndarray): Tick timestamps of the current window (ms).
        last_max_ts (int | None): Newest window ts seen on the previous refresh.

    Returns:
        tuple[dict, int | None]: (merged events, new last_max_ts).
    """
    wts = np.asarray(window_ts, dtype=np.int64)
    if wts.size == 0:
        return (new or {}), last_max_ts
    cur_min = int(wts[0])
    cur_max = int(wts[-1])
    if last_max_ts is not None and cur_max < last_max_ts:
        prev = {}   # time went backwards -> replay rewind, reset accumulator
    merged = accumulate_events(prev or {}, new or {}, min_ts=cur_min)
    return merged, cur_max


# =============================================================================
# DEEP TRADES (L2 order-flow classification)
# =============================================================================
#
# DeepTrades classifies each large aggressive execution against the resting L2
# liquidity it hit, read causally as-of the trade (the book state at or just
# before the trade, never after). Three L2 classes are supported:
#
#   EVENT_LARGE_AGGRESSOR - a big print that neither sweeps the stack nor is
#       absorbed by a wall (the baseline LargeTrades behavior).
#   EVENT_ABSORPTION      - a big print hitting a resting wall that holds: the
#       traded size is a large fraction of a substantial resting level but does
#       not fully clear it (price stalls into the wall).
#   EVENT_SWEEP           - a single aggressive order that fully clears several
#       price levels in one shot (liquidity taken across the stack).
#
# Iceberg / liquidity-grab need L3 (per-order) data and are out of scope here.

EVENT_LARGE_AGGRESSOR = 0
EVENT_ABSORPTION = 1
EVENT_SWEEP = 2

DEEP_TRADE_LABELS = {
    EVENT_LARGE_AGGRESSOR: "aggressor",
    EVENT_ABSORPTION: "absorption",
    EVENT_SWEEP: "sweep",
    # L3 classes (values mirror EVENT_ICEBERG=3 / EVENT_LIQUIDITY_GRAB=4 defined
    # in the L3 detectors section below).
    3: "iceberg",
    4: "liquidity_grab",
}


def _resting_and_consumed(level_px, level_sz, traded: float):
    """
    Walk the touched side of the book to measure the liquidity a trade consumed.

    Args:
        level_px (np.ndarray): Prices of the touched side, best level first.
        level_sz (np.ndarray): Sizes of the touched side, best level first.
        traded (float): Aggregated traded size of the execution.

    Returns:
        tuple[float, int, float]:
            - resting_best: size at the best (touched) level (NaN if no book).
            - levels_cleared: number of levels FULLY consumed by ``traded``.
            - total_depth: total visible resting size across the levels.
    """
    resting_best = float("nan")
    levels_cleared = 0
    total_depth = 0.0
    remaining = float(traded)
    seen_any = False
    for px, sz in zip(level_px, level_sz):
        if not np.isfinite(px) or not np.isfinite(sz) or sz <= 0.0:
            break
        if not seen_any:
            resting_best = float(sz)
            seen_any = True
        total_depth += float(sz)
        if remaining >= sz:
            levels_cleared += 1
            remaining -= float(sz)
        else:
            remaining = 0.0
    return resting_best, levels_cleared, total_depth


def classify_deep_trade(
    side: int,
    traded: float,
    book: "dict | None",
    *,
    min_resting_volume: float,
    absorption_ratio: float,
    stack_depth: int,
) -> "tuple[int, float, int]":
    """
    Classify one large execution against the as-of book state.

    A buy aggressor (side > 0) consumes the ask side; a sell aggressor consumes
    the bid side. All inputs are causal (book is the state at/just before the
    trade).

    Args:
        side (int): Aggressor side (+1 buy / -1 sell).
        traded (float): Aggregated traded size of the execution.
        book (dict | None): Book image from book_as_of (keys bid_px/bid_sz/
            ask_px/ask_sz), or None when no book is available.
        min_resting_volume (float): Minimum resting size for a level to count as
            a wall (absorption candidate).
        absorption_ratio (float): traded / resting_best threshold for absorption.
        stack_depth (int): Levels fully cleared to call a sweep.

    Returns:
        tuple[int, float, int]: (event_type, resting_best, levels_cleared).
    """
    if book is None:
        return EVENT_LARGE_AGGRESSOR, float("nan"), 0

    if side > 0:
        px, sz = book["ask_px"], book["ask_sz"]
    else:
        px, sz = book["bid_px"], book["bid_sz"]

    resting_best, levels_cleared, _ = _resting_and_consumed(px, sz, traded)

    if levels_cleared >= stack_depth:
        return EVENT_SWEEP, resting_best, levels_cleared
    if (
        levels_cleared == 0
        and np.isfinite(resting_best)
        and resting_best >= min_resting_volume
        and traded >= absorption_ratio * resting_best
    ):
        return EVENT_ABSORPTION, resting_best, levels_cleared
    return EVENT_LARGE_AGGRESSOR, resting_best, levels_cleared


def detect_deep_trades(
    ts: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    book_fn,
    flags: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    *,
    threshold="p99",
    by: str = "volume",
    window: int = 2000,
    min_gap_ms: int = 0,
    aggregate_ms: int = 0,
    min_resting_volume: float = 0.0,
    absorption_ratio: float = 0.8,
    stack_depth: int = 3,
) -> dict:
    """
    Detect large trades and classify each against the L2 book (causal).

    Runs the same causal large-trade detection as ``detect_large_trades`` and
    then classifies every detected event by the resting liquidity it hit, read
    via ``book_fn(ts)`` (the book state at or just before the trade).

    Args:
        ts, price, volume: Trade arrays [N].
        book_fn (Callable[[int], dict | None]): Returns the book image as-of a
            timestamp (keys bid_px/bid_sz/ask_px/ask_sz), or None.
        flags, bid, ask: Optional aggressor inputs (see classify_aggressor).
        threshold, by, window, min_gap_ms, aggregate_ms: Large-trade detection
            parameters (identical semantics to detect_large_trades).
        min_resting_volume, absorption_ratio, stack_depth: Classification
            thresholds (see classify_deep_trade).

    Returns:
        dict: Same shape as detect_large_trades' result plus, in ``events``, the
        extra arrays 'event_type' (int8), 'resting' (float64) and 'consumed'
        (int64). 'mask', 'metric', 'side' are over all trades; 'events' holds
        the detected events only.
    """
    base = detect_large_trades(
        ts, price, volume, flags=flags, bid=bid, ask=ask,
        threshold=threshold, by=by, window=window,
        min_gap_ms=min_gap_ms, aggregate_ms=aggregate_ms,
    )
    events = dict(base["events"])
    ev_ts = events["ts"]
    ev_side = events["side"]
    ev_vol = events["volume"]
    m = len(ev_ts)

    event_type = np.zeros(m, dtype=np.int8)
    resting = np.full(m, np.nan, dtype=np.float64)
    consumed = np.zeros(m, dtype=np.int64)

    for i in range(m):
        book = book_fn(int(ev_ts[i]))
        etype, rest, cleared = classify_deep_trade(
            int(ev_side[i]), float(ev_vol[i]), book,
            min_resting_volume=min_resting_volume,
            absorption_ratio=absorption_ratio,
            stack_depth=stack_depth,
        )
        event_type[i] = etype
        resting[i] = rest
        consumed[i] = cleared

    events["event_type"] = event_type
    events["resting"] = resting
    events["consumed"] = consumed
    return {
        "mask": base["mask"],
        "metric": base["metric"],
        "side": base["side"],
        "events": events,
    }



# =============================================================================
# L3 / MBO DETECTORS (iceberg reloads, liquidity grabs)
# =============================================================================
#
# These operate on the per-order (market-by-order) event stream - columns
# ts, order_id, side, price, size, action (action: 0 add / 1 modify / 2 cancel
# / 3 trade). All detection is causal (only events up to each point are used).
# Iceberg / liquidity-grab are L3-specific because they need per-order reloads
# and the swept price path; venues that expose only L2 cannot feed them.

EVENT_ICEBERG = 3
EVENT_LIQUIDITY_GRAB = 4

# Action codes (mirror core.data_types.MBO_*).
_MBO_ADD = 0
_MBO_MODIFY = 1
_MBO_CANCEL = 2
_MBO_TRADE = 3


def detect_iceberg(
    ts: np.ndarray,
    order_id: np.ndarray,
    side: np.ndarray,
    price: np.ndarray,
    size: np.ndarray,
    action: np.ndarray,
    *,
    reload_threshold: int = 3,
    window_ms: int = 2000,
    price_tol: float = 1e-9,
) -> dict:
    """
    Detect iceberg orders from L3 reloads at a price level.

    An iceberg reveals itself when a price level keeps refilling after being
    eaten: a TRADE consumes liquidity at a price, then an ADD/MODIFY restores it
    (a reload). When the reload count at a (side, price) level reaches
    ``reload_threshold`` within ``window_ms`` of the last fill, an iceberg is
    flagged at that event. Causal: only past events are used.

    Args:
        ts, order_id, side, price, size, action: MBO event columns [M].
        reload_threshold (int): Reloads needed at a level to confirm an iceberg.
        window_ms (int): Max gap (ms) between a fill and the reload that counts
            (0 disables the time limit).
        price_tol (float): Price comparison tolerance.

    Returns:
        dict: 'ts', 'price', 'side', 'reloads' arrays for the detected icebergs.
    """
    n = len(ts)
    state: dict = {}
    d_ts, d_px, d_side, d_reloads = [], [], [], []

    for i in range(n):
        a = int(action[i])
        p = float(price[i])
        s = int(side[i])
        t = int(ts[i])
        key = (s, round(p / max(price_tol, 1e-12)) if price_tol > 0 else p)
        st = state.setdefault(key, {"eaten": False, "eaten_ts": -1, "reloads": 0})

        if a == _MBO_TRADE:
            st["eaten"] = True
            st["eaten_ts"] = t
        elif a in (_MBO_ADD, _MBO_MODIFY):
            if st["eaten"] and (window_ms <= 0 or (t - st["eaten_ts"]) <= window_ms):
                st["reloads"] += 1
                st["eaten"] = False
                if st["reloads"] >= reload_threshold:
                    d_ts.append(t)
                    d_px.append(p)
                    d_side.append(s)
                    d_reloads.append(st["reloads"])

    return {
        "ts": np.asarray(d_ts, dtype=np.int64),
        "price": np.asarray(d_px, dtype=np.float64),
        "side": np.asarray(d_side, dtype=np.int8),
        "reloads": np.asarray(d_reloads, dtype=np.int64),
    }


def detect_liquidity_grab(
    ts: np.ndarray,
    price: np.ndarray,
    action: np.ndarray,
    *,
    tick_size: float = 1.0,
    sweep_levels: int = 3,
    sweep_window_ms: int = 1000,
    reversal_window_ms: int = 3000,
) -> dict:
    """
    Detect liquidity grabs: a fast sweep through levels followed by a reversal.

    Operates on the TRADE price path. A sweep is a monotonic run of trades that
    moves price by at least ``sweep_levels`` ticks within ``sweep_window_ms``; a
    grab is confirmed when, within ``reversal_window_ms`` after the sweep
    extreme, price trades back through the sweep's start level (a stop run that
    rejects). Causal: the grab is only emitted once the reversal is observed.

    Args:
        ts, price, action: MBO event columns [M].
        tick_size (float): Price step used to count swept levels.
        sweep_levels (int): Minimum ticks moved to call a sweep.
        sweep_window_ms (int): Max duration of the sweep run.
        reversal_window_ms (int): Window after the extreme to observe a reversal.

    Returns:
        dict: 'ts' (reversal time), 'price' (sweep extreme), 'side' (+1/-1 grab
        direction: +1 = downside sweep then up reversal, -1 = upside then down).
    """
    act = np.asarray(action)
    mask = act == _MBO_TRADE
    t = np.asarray(ts)[mask].astype(np.int64)
    p = np.asarray(price)[mask].astype(np.float64)
    n = len(t)
    ts_step = max(float(tick_size), 1e-12)

    d_ts, d_px, d_side = [], [], []
    i = 0
    while i < n - 1:
        direction = 0
        start = p[i]
        extreme = p[i]
        extreme_idx = i
        j = i
        while j + 1 < n and (t[j + 1] - t[i]) <= sweep_window_ms:
            d = 1 if p[j + 1] > p[j] else (-1 if p[j + 1] < p[j] else 0)
            if d != 0:
                if direction == 0:
                    direction = d
                elif d != direction:
                    break
            j += 1
            if direction > 0 and p[j] > extreme:
                extreme, extreme_idx = p[j], j
            elif direction < 0 and p[j] < extreme:
                extreme, extreme_idx = p[j], j

        levels = abs(extreme - start) / ts_step
        if direction != 0 and levels >= sweep_levels:
            k = extreme_idx + 1
            while k < n and (t[k] - t[extreme_idx]) <= reversal_window_ms:
                if direction > 0 and p[k] <= start:
                    d_ts.append(int(t[k])); d_px.append(float(extreme)); d_side.append(-1)
                    break
                if direction < 0 and p[k] >= start:
                    d_ts.append(int(t[k])); d_px.append(float(extreme)); d_side.append(1)
                    break
                k += 1
            i = extreme_idx + 1
        else:
            i += 1

    return {
        "ts": np.asarray(d_ts, dtype=np.int64),
        "price": np.asarray(d_px, dtype=np.float64),
        "side": np.asarray(d_side, dtype=np.int8),
    }


# =============================================================================
# AUTOFILTER (book-aware class filtering)
# =============================================================================
#
# DeepTrades can optionally drop low-conviction events, keeping only the
# classified order-flow signals. The filter is pure and causal so backtest,
# live and replay filter identically. It is book-aware: an event that could
# not be classified for lack of book depth (resting is NaN, so it fell back to
# Large Aggressor) is kept by default, so a plain backtest with no book still
# shows events.

# Reverse map name -> event code, derived from DEEP_TRADE_LABELS so the two
# never drift apart.
DEEP_TRADE_CODES = {name: code for code, name in DEEP_TRADE_LABELS.items()}

# The 'significant' shortcut: every classified class except the plain Large
# Aggressor baseline (the source of most of the noise).
_SIGNIFICANT_CLASSES = frozenset(
    {EVENT_ABSORPTION, EVENT_SWEEP, EVENT_ICEBERG, EVENT_LIQUIDITY_GRAB}
)


def resolve_autofilter(autofilter) -> "frozenset | None":
    """
    Resolve an autofilter spec into the set of event-type codes to keep.

    Args:
        autofilter: One of:
            - None: no filtering (returns None).
            - 'significant': keep every classified class except the plain Large
              Aggressor baseline (absorption, sweep, iceberg, liquidity_grab).
            - a class name str ('aggressor', 'absorption', 'sweep', 'iceberg',
              'liquidity_grab').
            - an iterable of class names (same valid names as above).

    Returns:
        frozenset[int] | None: Event-type codes to keep, or None for no filter.

    Raises:
        ValueError: If a string spec or any class name is not recognized.
    """
    if autofilter is None:
        return None
    if isinstance(autofilter, str):
        spec = autofilter.strip().lower()
        if spec == "significant":
            return _SIGNIFICANT_CLASSES
        if spec in DEEP_TRADE_CODES:
            return frozenset({DEEP_TRADE_CODES[spec]})
        raise ValueError(
            f"autofilter string must be 'significant' or a class name "
            f"{sorted(DEEP_TRADE_CODES)}, not {autofilter!r}"
        )
    try:
        names = list(autofilter)
    except TypeError as exc:
        raise ValueError(
            f"autofilter must be None, a str or an iterable of class names, "
            f"not {type(autofilter)!r}"
        ) from exc
    codes = set()
    for nm in names:
        key = str(nm).strip().lower()
        if key not in DEEP_TRADE_CODES:
            raise ValueError(
                f"unknown autofilter class {nm!r}; valid classes are "
                f"{sorted(DEEP_TRADE_CODES)}"
            )
        codes.add(DEEP_TRADE_CODES[key])
    return frozenset(codes)


def apply_deep_autofilter(
    event_type: np.ndarray,
    resting: np.ndarray,
    keep_classes: "frozenset | None",
    *,
    keep_unclassified: bool = True,
) -> np.ndarray:
    """
    Build the keep mask for detected deep-trade events (book-aware).

    Args:
        event_type (np.ndarray): Event class codes [M].
        resting (np.ndarray): Resting size at the touched level [M]; NaN means
            the event could not be classified (no book as-of the trade).
        keep_classes (frozenset[int] | None): Codes to keep. None keeps all.
        keep_unclassified (bool): When True, events with NaN resting (no book)
            are always kept regardless of class, so a backtest with no depth is
            not silently emptied. When False, only events whose class is in
            ``keep_classes`` survive (strict mode).

    Returns:
        np.ndarray: Boolean keep mask [M].
    """
    event_type = np.asarray(event_type)
    m = event_type.shape[0]
    if keep_classes is None:
        return np.ones(m, dtype=bool)
    if m == 0:
        return np.zeros(0, dtype=bool)
    codes = np.fromiter(keep_classes, dtype=np.int64, count=len(keep_classes))
    keep = np.isin(event_type.astype(np.int64), codes)
    if keep_unclassified:
        resting = np.asarray(resting, dtype=np.float64)
        keep = keep | ~np.isfinite(resting)
    return keep


def deep_trade_class_name(event_type) -> str:
    """
    Map a deep-trade event code to its human-readable class name.

    NaN-safe so it can be called directly on a band value read in on_data()
    (where most ticks carry NaN because they are not detected events).

    Args:
        event_type: Numeric class code (NaN / None allowed).

    Returns:
        str: Class name ('aggressor', 'absorption', 'sweep', 'iceberg',
        'liquidity_grab'), or '' when the value is NaN / None / unknown (i.e.
        no event at this tick).
    """
    if event_type is None:
        return ""
    try:
        val = float(event_type)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return DEEP_TRADE_LABELS.get(int(val), "")



# =============================================================================
# PER-BAR DELTA (delta bars, CVD candles, bid/ask/delta/total footprint)
# =============================================================================
#
# These helpers aggregate the trade (tick) stream into fixed-interval bars and
# compute the order-flow figures per bar:
#
#   - ask_vol : volume executed against the ask (buy aggressor).
#   - bid_vol : volume executed against the bid (sell aggressor).
#   - delta   : ask_vol - bid_vol (net aggression of the bar).
#   - total   : ask_vol + bid_vol (total traded volume of the bar).
#
# and the cumulative-volume-delta (CVD) OHLC path, where the running cumulative
# delta is sampled per bar into open / high / low / close so it can be drawn as
# a candle series. All functions are pure, vectorized and causal (a bar only
# depends on the trades up to and including it), so backtest, live and replay
# produce identical figures. Aggressor side comes from ``classify_aggressor``.


def bar_index(ts: np.ndarray, interval_ms: int, anchor: int = 0) -> np.ndarray:
    """
    Map each trade timestamp to its fixed-interval bar id.

    Args:
        ts (np.ndarray): Trade timestamps in ms [N] (non-decreasing).
        interval_ms (int): Bar width in ms.
        anchor (int): Epoch-ms origin of the bar grid (default 0 = UTC epoch,
            so bars fall on round clock marks like the OHLC candles).

    Returns:
        np.ndarray: int64 bar id per trade [N] (floor((ts - anchor)/interval)).
    """
    ts = np.asarray(ts, dtype=np.int64)
    iv = max(int(interval_ms), 1)
    return (ts - int(anchor)) // iv


def _empty_bar_delta() -> dict:
    """Empty per-bar delta result (no trades)."""
    return {
        "bar_id": np.zeros(0, dtype=np.int64),
        "bar_ts": np.zeros(0, dtype=np.int64),
        "rep_idx": np.zeros(0, dtype=np.int64),
        "rep_ts": np.zeros(0, dtype=np.int64),
        "ask_vol": np.zeros(0, dtype=np.float64),
        "bid_vol": np.zeros(0, dtype=np.float64),
        "delta": np.zeros(0, dtype=np.float64),
        "total": np.zeros(0, dtype=np.float64),
        "delta_max": np.zeros(0, dtype=np.float64),
        "delta_min": np.zeros(0, dtype=np.float64),
        "cot_high": np.zeros(0, dtype=np.float64),
        "cot_low": np.zeros(0, dtype=np.float64),
        "high_price": np.zeros(0, dtype=np.float64),
        "low_price": np.zeros(0, dtype=np.float64),
    }


def bar_delta(
    ts: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    *,
    interval_ms: int,
    flags: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    anchor: int = 0,
) -> dict:
    """
    Aggregate trades into fixed-interval bars and compute the per-bar figures.

    Each trade is classified buy / sell aggressor (``classify_aggressor``) and
    its volume added to the bar's ask (buy) or bid (sell) bucket. The result is
    one row per bar that holds at least one trade, in ascending time order.

    Args:
        ts (np.ndarray): Trade timestamps in ms [N] (non-decreasing).
        price (np.ndarray): Trade prices [N].
        volume (np.ndarray): Trade volumes [N].
        interval_ms (int): Bar width in ms.
        flags (np.ndarray | None): Per-trade side flags (see classify_aggressor).
        bid (np.ndarray | None): Best bid per trade (quote-rule fallback).
        ask (np.ndarray | None): Best ask per trade (quote-rule fallback).
        anchor (int): Epoch-ms origin of the bar grid.

    Returns:
        dict: Per-bar arrays (one entry per non-empty bar):
            - 'bar_id'    (int64 [B]): bar grid id.
            - 'bar_ts'    (int64 [B]): bar start ts (bar_id * interval + anchor).
            - 'rep_idx'   (int64 [B]): index of the bar's last trade (anchor tick).
            - 'rep_ts'    (int64 [B]): timestamp of that last trade.
            - 'ask_vol'   (float64 [B]): buy-aggressor volume.
            - 'bid_vol'   (float64 [B]): sell-aggressor volume.
            - 'delta'     (float64 [B]): ask_vol - bid_vol.
            - 'total'     (float64 [B]): ask_vol + bid_vol.
            - 'delta_max' (float64 [B]): max intra-bar cumulative delta.
            - 'delta_min' (float64 [B]): min intra-bar cumulative delta.
            - 'cot_high'  (float64 [B]): COT High - intra-bar cumulative delta at
              the last tick where price touched the bar's high (delta committed
              at the price high; GoCharting-style, relative to the bar start).
            - 'cot_low'   (float64 [B]): COT Low - intra-bar cumulative delta at
              the last tick where price touched the bar's low.
            - 'high_price'(float64 [B]): bar high price (label anchor for COTH).
            - 'low_price' (float64 [B]): bar low price (label anchor for COTL).
    """
    ts = np.asarray(ts, dtype=np.int64)
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = ts.shape[0]
    if n == 0:
        return _empty_bar_delta()

    side = classify_aggressor(price, flags=flags, bid=bid, ask=ask)
    bvol = np.abs(volume)
    ask_contrib = np.where(side > 0, bvol, 0.0)
    bid_contrib = np.where(side < 0, bvol, 0.0)

    iv = max(int(interval_ms), 1)
    bid_grid = (ts - int(anchor)) // iv
    uniq, first_idx = np.unique(bid_grid, return_index=True)
    first_idx = first_idx.astype(np.int64)

    ask_vol = np.add.reduceat(ask_contrib, first_idx)
    bid_vol = np.add.reduceat(bid_contrib, first_idx)

    rep_idx = np.empty(first_idx.shape[0], dtype=np.int64)
    rep_idx[:-1] = first_idx[1:] - 1
    rep_idx[-1] = n - 1

    # Intra-bar cumulative delta max/min: for each bar, compute the running
    # cumulative delta (signed volume) and take its max and min over the bar.
    signed = np.where(side > 0, bvol, -bvol)
    cum = np.cumsum(signed)
    bounds = list(first_idx) + [n]
    b = first_idx.shape[0]
    delta_max = np.empty(b, dtype=np.float64)
    delta_min = np.empty(b, dtype=np.float64)
    cot_high = np.empty(b, dtype=np.float64)
    cot_low = np.empty(b, dtype=np.float64)
    high_price = np.empty(b, dtype=np.float64)
    low_price = np.empty(b, dtype=np.float64)
    for k in range(b):
        a_, b_ = int(bounds[k]), int(bounds[k + 1])
        # Intra-bar cumulative delta relative to bar start.
        bar_cum = cum[a_:b_] - (cum[a_ - 1] if a_ > 0 else 0.0)
        delta_max[k] = float(np.max(bar_cum))
        delta_min[k] = float(np.min(bar_cum))
        # COT High / Low: cumulative delta at the last tick where price touched
        # the bar's high / low (delta committed at the price extreme).
        seg_price = price[a_:b_]
        hi = float(np.max(seg_price))
        lo = float(np.min(seg_price))
        high_price[k] = hi
        low_price[k] = lo
        # Last occurrence of the extreme (carries the most committed delta).
        idx_hi = int(np.flatnonzero(seg_price == hi)[-1])
        idx_lo = int(np.flatnonzero(seg_price == lo)[-1])
        cot_high[k] = float(bar_cum[idx_hi])
        cot_low[k] = float(bar_cum[idx_lo])

    return {
        "bar_id": uniq.astype(np.int64),
        "bar_ts": (uniq * iv + int(anchor)).astype(np.int64),
        "rep_idx": rep_idx,
        "rep_ts": ts[rep_idx].astype(np.int64),
        "ask_vol": ask_vol.astype(np.float64),
        "bid_vol": bid_vol.astype(np.float64),
        "delta": (ask_vol - bid_vol).astype(np.float64),
        "total": (ask_vol + bid_vol).astype(np.float64),
        "delta_max": delta_max,
        "delta_min": delta_min,
        "cot_high": cot_high,
        "cot_low": cot_low,
        "high_price": high_price,
        "low_price": low_price,
    }


def cumulative_delta_ohlc(
    ts: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    *,
    interval_ms: int,
    flags: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    anchor: int = 0,
) -> dict:
    """
    Sample the running cumulative volume delta (CVD) into per-bar OHLC.

    The cumulative delta is the running sum of signed trade volume (+volume for
    a buy aggressor, -volume for a sell aggressor). Within each bar it traces a
    path; this samples that path into a candle:

        - open  : cumulative delta just before the bar's first trade (= the
                  previous bar's close; 0 for the first bar).
        - close : cumulative delta after the bar's last trade.
        - high  : max of the path over the bar, never below the open.
        - low   : min of the path over the bar, never above the open.

    So ``open`` chains to the previous ``close`` and ``low <= open/close <=
    high`` always hold, exactly like a price candle.

    Args:
        ts, price, volume: Trade arrays [N] (ts non-decreasing).
        interval_ms (int): Bar width in ms.
        flags, bid, ask: Optional aggressor inputs (see classify_aggressor).
        anchor (int): Epoch-ms origin of the bar grid.

    Returns:
        dict: Per-bar arrays (ascending time):
            - 'bar_ts' (int64 [B]): bar start ts.
            - 'rep_ts' (int64 [B]): last-trade ts of the bar (anchor tick).
            - 'open' / 'high' / 'low' / 'close' (float64 [B]): CVD OHLC.
            - 'delta' (float64 [B]): per-bar delta (close - open).
    """
    ts = np.asarray(ts, dtype=np.int64)
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = ts.shape[0]
    if n == 0:
        return {
            "bar_ts": np.zeros(0, dtype=np.int64),
            "rep_ts": np.zeros(0, dtype=np.int64),
            "open": np.zeros(0, dtype=np.float64),
            "high": np.zeros(0, dtype=np.float64),
            "low": np.zeros(0, dtype=np.float64),
            "close": np.zeros(0, dtype=np.float64),
            "delta": np.zeros(0, dtype=np.float64),
        }

    side = classify_aggressor(price, flags=flags, bid=bid, ask=ask)
    signed = np.abs(volume) * np.where(side >= 0, 1.0, -1.0)
    cum = np.cumsum(signed)

    iv = max(int(interval_ms), 1)
    bid_grid = (ts - int(anchor)) // iv
    uniq, first_idx = np.unique(bid_grid, return_index=True)
    first_idx = first_idx.astype(np.int64)

    rep_idx = np.empty(first_idx.shape[0], dtype=np.int64)
    rep_idx[:-1] = first_idx[1:] - 1
    rep_idx[-1] = n - 1

    seg_max = np.maximum.reduceat(cum, first_idx)
    seg_min = np.minimum.reduceat(cum, first_idx)

    open_cum = np.empty(first_idx.shape[0], dtype=np.float64)
    open_cum[0] = 0.0
    open_cum[1:] = cum[first_idx[1:] - 1]
    close = cum[rep_idx].astype(np.float64)
    high = np.maximum(open_cum, seg_max)
    low = np.minimum(open_cum, seg_min)

    return {
        "bar_ts": (uniq * iv + int(anchor)).astype(np.int64),
        "rep_ts": ts[rep_idx].astype(np.int64),
        "open": open_cum,
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": close,
        "delta": (close - open_cum).astype(np.float64),
    }




# =============================================================================
# L2 BOOK DETECTORS (Deep Wall, Deep Reload, Stop Run)
# =============================================================================
#
# These read the L2 order book's evolution over a window (a sequence of top-K
# snapshots from ``OrderbookProxy.book_window()``), not just a single as-of
# image. They are pure, NumPy-only and causal (a detection only depends on the
# snapshots up to it), so backtest, live and replay produce identical results.
#
# Side convention: +1 = bid side, -1 = ask side (matches classify_aggressor).


def _level_threshold(bid_sz_row, ask_sz_row, min_volume, rel_multiple, top):
    """
    Per-snapshot wall threshold: max(min_volume, rel_multiple * median size).

    The relative term self-adjusts to the book's current depth so a wall is
    "large relative to the rest of the stack". Returns NaN when the snapshot
    has no visible liquidity.
    """
    sizes = []
    for arr in (bid_sz_row[:top], ask_sz_row[:top]):
        for s in arr:
            if np.isfinite(s) and s > 0.0:
                sizes.append(float(s))
    if not sizes:
        return float("nan")
    thr = float(min_volume)
    if rel_multiple:
        thr = max(thr, float(rel_multiple) * float(np.median(sizes)))
    return thr


def detect_walls(
    book: dict,
    *,
    min_volume: float = 0.0,
    rel_multiple: float = 5.0,
    persistence_ms: int = 0,
    top_n: "int | None" = None,
    price_tol: float = 1e-9,
) -> dict:
    """
    Detect resting liquidity walls (Deep Wall) from L2 book snapshots.

    A wall is a price level whose resting size is anomalously large (absolute
    ``min_volume`` and/or ``rel_multiple`` times the snapshot's median level
    size) and that persists for at least ``persistence_ms``. Each detected wall
    is returned as a time span [first_ts, last_ts] at its price, so an overlay
    can draw it as a zone/line. Causal: a wall span only uses the snapshots it
    spans.

    Args:
        book (dict): ``OrderbookProxy.book_window()`` output.
        min_volume (float): Absolute minimum resting size to qualify as a wall.
        rel_multiple (float): Multiple of the per-snapshot median level size
            (0 disables the relative term).
        persistence_ms (int): Minimum lifetime of a wall to be reported.
        top_n (int | None): Levels per side to scan (None -> all retained).
        price_tol (float): Price grouping tolerance.

    Returns:
        dict: 'price', 'side' (+1 bid / -1 ask), 'first_ts', 'last_ts',
        'max_size' arrays for the detected walls.
    """
    ts = np.asarray(book.get("ts", []), dtype=np.int64)
    r = ts.shape[0]
    k = int(book.get("levels") or (book["bid_px"].shape[1] if r else 0))
    if r == 0 or k == 0:
        return _empty_walls()
    top = k if top_n is None else min(int(top_n), k)
    bid_px, bid_sz = book["bid_px"], book["bid_sz"]
    ask_px, ask_sz = book["ask_px"], book["ask_sz"]

    inv = 1.0 / max(price_tol, 1e-12)

    def pkey(p):
        return round(float(p) * inv)

    active: dict = {}
    closed: list = []
    for j in range(r):
        thr = _level_threshold(bid_sz[j], ask_sz[j], min_volume, rel_multiple, top)
        seen = set()
        if np.isfinite(thr):
            for side, px_arr, sz_arr in ((1, bid_px, bid_sz), (-1, ask_px, ask_sz)):
                for i in range(top):
                    p = px_arr[j, i]
                    s = sz_arr[j, i]
                    if not (np.isfinite(p) and np.isfinite(s) and s > 0.0):
                        continue
                    if s >= thr:
                        key = (side, pkey(p))
                        seen.add(key)
                        st = active.get(key)
                        if st is None:
                            active[key] = {
                                "price": float(p), "side": side,
                                "first_ts": int(ts[j]), "last_ts": int(ts[j]),
                                "max_size": float(s),
                            }
                        else:
                            st["last_ts"] = int(ts[j])
                            st["max_size"] = max(st["max_size"], float(s))
        for key in [k_ for k_ in active if k_ not in seen]:
            st = active.pop(key)
            if (st["last_ts"] - st["first_ts"]) >= persistence_ms:
                closed.append(st)
    for st in active.values():
        if (st["last_ts"] - st["first_ts"]) >= persistence_ms:
            closed.append(st)

    if not closed:
        return _empty_walls()
    closed.sort(key=lambda d: (d["first_ts"], d["price"]))
    return {
        "price": np.array([d["price"] for d in closed], dtype=np.float64),
        "side": np.array([d["side"] for d in closed], dtype=np.int8),
        "first_ts": np.array([d["first_ts"] for d in closed], dtype=np.int64),
        "last_ts": np.array([d["last_ts"] for d in closed], dtype=np.int64),
        "max_size": np.array([d["max_size"] for d in closed], dtype=np.float64),
    }


def _empty_walls() -> dict:
    return {
        "price": np.zeros(0, dtype=np.float64),
        "side": np.zeros(0, dtype=np.int8),
        "first_ts": np.zeros(0, dtype=np.int64),
        "last_ts": np.zeros(0, dtype=np.int64),
        "max_size": np.zeros(0, dtype=np.float64),
    }


def detect_reload_l2(
    book: dict,
    *,
    drop_frac: float = 0.5,
    recover_frac: float = 0.8,
    reload_threshold: int = 2,
    window_ms: int = 0,
    top_n: "int | None" = None,
    min_volume: float = 0.0,
    price_tol: float = 1e-9,
) -> dict:
    """
    Detect liquidity replenishment (Deep Reload) from L2 book snapshots.

    A level "reloads" when its resting size is consumed - drops to at most
    ``drop_frac`` of a recent high (including being removed entirely) - and then
    refills back to at least ``recover_frac`` of that high. When a (side, price)
    level reaches ``reload_threshold`` reloads, a detection is emitted at the
    refill snapshot. This is the L2 (snapshot) analogue of the L3 iceberg
    detector, usable on venues that only expose depth. Causal.

    Args:
        book (dict): ``OrderbookProxy.book_window()`` output.
        drop_frac (float): Fraction of the reference high a level must fall to
            (or below) to count as consumed (0.5 = halved).
        recover_frac (float): Fraction of the reference high the refill must
            reach to count as a reload.
        reload_threshold (int): Reloads at a level needed to flag it.
        window_ms (int): Max gap (ms) from the consumption to the refill that
            counts (0 disables the time limit).
        top_n (int | None): Levels per side to scan (None -> all retained).
        min_volume (float): Ignore levels whose reference high is below this
            (filters noise from tiny levels).
        price_tol (float): Price grouping tolerance.

    Returns:
        dict: 'ts' (refill time), 'price', 'side' (+1 bid / -1 ask), 'reloads'.
    """
    ts = np.asarray(book.get("ts", []), dtype=np.int64)
    r = ts.shape[0]
    k = int(book.get("levels") or (book["bid_px"].shape[1] if r else 0))
    if r == 0 or k == 0:
        return _empty_reload()
    top = k if top_n is None else min(int(top_n), k)
    bid_px, bid_sz = book["bid_px"], book["bid_sz"]
    ask_px, ask_sz = book["ask_px"], book["ask_sz"]

    inv = 1.0 / max(price_tol, 1e-12)

    def pkey(p):
        return round(float(p) * inv)

    state: dict = {}
    d_ts, d_px, d_side, d_reloads = [], [], [], []

    for j in range(r):
        cur: dict = {}
        for side, px_arr, sz_arr in ((1, bid_px, bid_sz), (-1, ask_px, ask_sz)):
            for i in range(top):
                p = px_arr[j, i]
                s = sz_arr[j, i]
                if np.isfinite(p) and np.isfinite(s) and s > 0.0:
                    cur[(side, pkey(p), float(p))] = float(s)

        # Keys to evaluate: those present now plus those we are already tracking.
        present_norm = {(sd, pk): (p, s) for (sd, pk, p), s in cur.items()}
        keys = set(state) | set(present_norm)
        for key in keys:
            p_s = present_norm.get(key)
            s = p_s[1] if p_s is not None else 0.0
            price_val = p_s[0] if p_s is not None else (state[key]["price"] if key in state else 0.0)
            st = state.get(key)
            if st is None:
                if s <= 0.0:
                    continue
                state[key] = {"ref": s, "eaten": False, "eaten_ts": -1,
                              "reloads": 0, "price": price_val}
                continue
            st["price"] = price_val or st["price"]
            if not st["eaten"]:
                if st["ref"] >= min_volume and s <= drop_frac * st["ref"]:
                    st["eaten"] = True
                    st["eaten_ts"] = int(ts[j])
                else:
                    if s > st["ref"]:
                        st["ref"] = s
                    if s <= 0.0:
                        # Dead, never-eaten level: stop tracking.
                        state.pop(key, None)
            else:
                in_window = (window_ms <= 0) or (int(ts[j]) - st["eaten_ts"] <= window_ms)
                if s >= recover_frac * st["ref"] and in_window:
                    st["reloads"] += 1
                    st["eaten"] = False
                    if st["reloads"] >= reload_threshold:
                        d_ts.append(int(ts[j]))
                        d_px.append(st["price"])
                        d_side.append(key[0])
                        d_reloads.append(st["reloads"])
                elif not in_window:
                    # Reload window elapsed without a refill: reset baseline.
                    st["eaten"] = False
                    st["ref"] = max(s, min_volume)

    if not d_ts:
        return _empty_reload()
    return {
        "ts": np.asarray(d_ts, dtype=np.int64),
        "price": np.asarray(d_px, dtype=np.float64),
        "side": np.asarray(d_side, dtype=np.int8),
        "reloads": np.asarray(d_reloads, dtype=np.int64),
    }


def _empty_reload() -> dict:
    return {
        "ts": np.zeros(0, dtype=np.int64),
        "price": np.zeros(0, dtype=np.float64),
        "side": np.zeros(0, dtype=np.int8),
        "reloads": np.zeros(0, dtype=np.int64),
    }


def detect_stop_run_l2(
    ts: np.ndarray,
    price: np.ndarray,
    *,
    tick_size: float = 1.0,
    sweep_levels: int = 3,
    sweep_window_ms: int = 1000,
    reversal_window_ms: int = 3000,
) -> dict:
    """
    Detect stop runs (stop sweeps) from the trade price path.

    A stop run is a fast directional sweep - a monotonic run that moves price by
    at least ``sweep_levels`` ticks within ``sweep_window_ms`` (running stops) -
    immediately followed by a reversal back through the sweep's start level
    within ``reversal_window_ms`` (the move rejects). Causal: a run is only
    emitted once the reversal is observed.

    Args:
        ts (np.ndarray): Trade timestamps in ms [N].
        price (np.ndarray): Trade prices [N].
        tick_size (float): Price step used to count swept levels.
        sweep_levels (int): Minimum ticks moved to qualify as a sweep.
        sweep_window_ms (int): Max duration of the sweep run.
        reversal_window_ms (int): Window after the extreme to observe a reversal.

    Returns:
        dict: 'start_ts', 'start_px', 'extreme_ts', 'extreme_px', 'ts'
        (reversal time), 'side' (+1 = down-sweep then up reversal, -1 = up-sweep
        then down reversal). Arrays, one entry per detected stop run.
    """
    t = np.asarray(ts, dtype=np.int64)
    p = np.asarray(price, dtype=np.float64)
    n = len(t)
    step = max(float(tick_size), 1e-12)

    s_ts, s_px, e_ts, e_px, r_ts, side = [], [], [], [], [], []
    i = 0
    while i < n - 1:
        direction = 0
        start = p[i]
        extreme = p[i]
        ext_idx = i
        j = i
        while j + 1 < n and (t[j + 1] - t[i]) <= sweep_window_ms:
            d = 1 if p[j + 1] > p[j] else (-1 if p[j + 1] < p[j] else 0)
            if d != 0:
                if direction == 0:
                    direction = d
                elif d != direction:
                    break
            j += 1
            if direction > 0 and p[j] > extreme:
                extreme, ext_idx = p[j], j
            elif direction < 0 and p[j] < extreme:
                extreme, ext_idx = p[j], j

        levels = abs(extreme - start) / step
        if direction != 0 and levels >= sweep_levels:
            kk = ext_idx + 1
            hit = False
            while kk < n and (t[kk] - t[ext_idx]) <= reversal_window_ms:
                if direction > 0 and p[kk] <= start:
                    side.append(-1); hit = True
                elif direction < 0 and p[kk] >= start:
                    side.append(1); hit = True
                if hit:
                    s_ts.append(int(t[i])); s_px.append(float(start))
                    e_ts.append(int(t[ext_idx])); e_px.append(float(extreme))
                    r_ts.append(int(t[kk]))
                    break
                kk += 1
            i = ext_idx + 1
        else:
            i += 1

    return {
        "start_ts": np.asarray(s_ts, dtype=np.int64),
        "start_px": np.asarray(s_px, dtype=np.float64),
        "extreme_ts": np.asarray(e_ts, dtype=np.int64),
        "extreme_px": np.asarray(e_px, dtype=np.float64),
        "ts": np.asarray(r_ts, dtype=np.int64),
        "side": np.asarray(side, dtype=np.int8),
    }


# =============================================================================
# HEATMAP (Bookmap-style L2 liquidity grid)
# =============================================================================
#
# A heatmap turns the L2 order book's evolution (a sequence of top-K snapshots
# from ``OrderbookProxy.book_window()``) into a regular time x price grid where
# each cell holds the resting liquidity (size) at that price band and time
# slice. Colder cells hold little liquidity, hotter cells hold a lot - the same
# model Bookmap / ATAS heatmaps use. The builder is pure, NumPy-only and causal
# (a column only depends on the snapshots that fall in it), so backtest, live
# and replay produce the identical grid.
#
# The grid is the shared substrate for both the visual layer (colored Rects)
# and the programmatic query API (liquidity_at / hottest / walls / ...).


def infer_price_tick(book: dict, *, top_n: "int | None" = None,
                     default: float = 1.0) -> float:
    """
    Infer the price tick (minimum level spacing) from an L2 book window.

    Scans the smallest positive gap between adjacent price levels across the
    stored snapshots, on both sides. This lets the heatmap pick a sensible
    price-bucket width without needing symbol metadata.

    Args:
        book (dict): ``OrderbookProxy.book_window()`` output.
        top_n (int | None): Levels per side to scan (None -> all retained).
        default (float): Fallback when no spacing can be measured.

    Returns:
        float: The inferred price tick, or ``default`` when undetermined.
    """
    r = int(np.asarray(book.get("ts", [])).shape[0])
    k = int(book.get("levels") or (book["bid_px"].shape[1] if r else 0))
    if r == 0 or k == 0:
        return float(default)
    top = k if top_n is None else min(int(top_n), k)
    best = np.inf
    for arr in (book["bid_px"][:, :top], book["ask_px"][:, :top]):
        d = np.abs(np.diff(arr, axis=1))
        d = d[np.isfinite(d) & (d > 0.0)]
        if d.size:
            best = min(best, float(np.min(d)))
    return float(best) if np.isfinite(best) else float(default)


def _empty_heatmap(price_bucket: float, levels: int) -> dict:
    """Empty heatmap grid (no book / no liquidity)."""
    return {
        "col_ts": np.zeros(0, dtype=np.int64),
        "col_left": np.zeros(0, dtype=np.int64),
        "col_right": np.zeros(0, dtype=np.int64),
        "price_edges": np.zeros(0, dtype=np.float64),
        "price_centers": np.zeros(0, dtype=np.float64),
        "bid": np.zeros((0, 0), dtype=np.float64),
        "ask": np.zeros((0, 0), dtype=np.float64),
        "levels": int(levels),
        "price_bucket": float(price_bucket),
    }


def _accumulate_side(px: np.ndarray, sz: np.ndarray, c0: float,
                     bucket: float, n_rows: int, n_buckets: int) -> np.ndarray:
    """
    Sum resting size into a [R x P] time-by-price-bucket matrix (one side).

    Buckets are centered on price-grid multiples (center of bucket ``i`` is
    ``c0 + i * bucket``), so a level sitting on a grid multiple maps to a bucket
    whose center equals its price - the intuitive mapping for the query API.

    Args:
        px (np.ndarray): Level prices [R x K].
        sz (np.ndarray): Level sizes [R x K].
        c0 (float): Center of bucket 0 (a price-grid multiple).
        bucket (float): Price-bucket width.
        n_rows (int): Number of snapshots R.
        n_buckets (int): Number of price buckets P.

    Returns:
        np.ndarray: float64 [R x P] summed resting size per (snapshot, bucket).
    """
    mat = np.zeros((n_rows, n_buckets), dtype=np.float64)
    finite = np.isfinite(px) & np.isfinite(sz) & (sz > 0.0)
    if not finite.any():
        return mat
    rows, _ = np.nonzero(finite)
    p = px[finite]
    s = sz[finite]
    b = np.round((p - c0) / bucket).astype(np.int64)
    np.clip(b, 0, n_buckets - 1, out=b)
    np.add.at(mat, (rows, b), s)
    return mat


def build_heatmap_grid(
    book: dict,
    *,
    price_bucket: float,
    time_bucket_ms: "int | None" = None,
    max_levels: "int | None" = None,
    min_size: float = 0.0,
) -> dict:
    """
    Build a Bookmap-style time x price liquidity grid from an L2 book window.

    Every book snapshot's resting size is bucketed by price into a fixed grid of
    width ``price_bucket``; snapshots are then grouped into time columns. With
    ``time_bucket_ms=None`` every snapshot is its own column (the recording
    granularity, so a replay reproduces the grid exactly); otherwise snapshots
    that fall in the same ``time_bucket_ms`` window are aggregated by taking the
    peak (max) resting size per price band in that window (the liquidity that
    sat there). Causal: a column only depends on the snapshots it contains.

    Args:
        book (dict): ``OrderbookProxy.book_window()`` output.
        price_bucket (float): Price width of each bucket (multiple of the tick).
        time_bucket_ms (int | None): Time-column width in ms (None -> one column
            per snapshot).
        max_levels (int | None): Levels per side to include (None -> all).
        min_size (float): Zero out cells whose resting size is below this
            (hides small-order noise, like Bookmap's minimum-size filter).

    Returns:
        dict: Grid arrays:
            - 'col_ts'      (int64 [T]): representative timestamp per column.
            - 'col_left'    (int64 [T]): column start ts (ms) for drawing.
            - 'col_right'   (int64 [T]): column end ts (ms) for drawing.
            - 'price_edges' (float64 [P+1]): price-bucket edges.
            - 'price_centers'(float64 [P]): price-bucket centers.
            - 'bid'/'ask'   (float64 [T x P]): resting size per cell.
            - 'levels' (int): book depth K.
            - 'price_bucket' (float): the bucket width used.
    """
    bucket = float(price_bucket)
    ts = np.asarray(book.get("ts", []), dtype=np.int64)
    r = ts.shape[0]
    k = int(book.get("levels") or (book["bid_px"].shape[1] if r else 0))
    if r == 0 or k == 0 or bucket <= 0.0:
        return _empty_heatmap(bucket, k)

    top = k if max_levels is None else min(int(max_levels), k)
    bid_px = book["bid_px"][:, :top]
    bid_sz = book["bid_sz"][:, :top]
    ask_px = book["ask_px"][:, :top]
    ask_sz = book["ask_sz"][:, :top]

    all_px = np.concatenate([bid_px.ravel(), ask_px.ravel()])
    all_px = all_px[np.isfinite(all_px)]
    if all_px.size == 0:
        return _empty_heatmap(bucket, k)

    # Buckets centered on price-grid multiples of ``bucket`` (so a level on the
    # grid maps to a bucket center equal to its price).
    c0 = np.round(float(np.min(all_px)) / bucket) * bucket
    cmax = np.round(float(np.max(all_px)) / bucket) * bucket
    n_buckets = max(int(round((cmax - c0) / bucket)) + 1, 1)
    price_centers = c0 + np.arange(n_buckets, dtype=np.float64) * bucket
    price_edges = np.empty(n_buckets + 1, dtype=np.float64)
    price_edges[:-1] = price_centers - bucket / 2.0
    price_edges[-1] = price_centers[-1] + bucket / 2.0

    bid_row = _accumulate_side(bid_px, bid_sz, c0, bucket, r, n_buckets)
    ask_row = _accumulate_side(ask_px, ask_sz, c0, bucket, r, n_buckets)

    if min_size > 0.0:
        bid_row[bid_row < min_size] = 0.0
        ask_row[ask_row < min_size] = 0.0

    if time_bucket_ms is None or int(time_bucket_ms) <= 0:
        bid = bid_row
        ask = ask_row
        col_ts = ts.copy()
        col_left = ts.copy()
        col_right = np.empty(r, dtype=np.int64)
        if r > 1:
            col_right[:-1] = ts[1:]
            gap = int(np.median(np.diff(ts))) or 1
            col_right[-1] = ts[-1] + gap
        else:
            col_right[0] = ts[0] + 1
    else:
        # Time columns anchored to an ABSOLUTE origin (epoch-aligned buckets:
        # tb = ts // time_bucket_ms), NOT to ts[0] of the passed window. This
        # keeps a column's boundaries and col_ts stable no matter which sliding
        # book window a grid is built from, so live/replay accumulation aligns
        # column-for-column with a single backtest pass (parity).
        tb = ts // int(time_bucket_ms)
        uniq, first = np.unique(tb, return_index=True)
        first = first.astype(np.int64)
        bid = np.maximum.reduceat(bid_row, first)
        ask = np.maximum.reduceat(ask_row, first)
        col_left = (uniq * int(time_bucket_ms)).astype(np.int64)
        col_right = (col_left + int(time_bucket_ms)).astype(np.int64)
        col_ts = col_left.copy()

    return {
        "col_ts": col_ts,
        "col_left": col_left,
        "col_right": col_right,
        "price_edges": price_edges,
        "price_centers": price_centers,
        "bid": bid,
        "ask": ask,
        "levels": k,
        "price_bucket": bucket,
    }


def resolve_persistence(mat: np.ndarray, col_ts: np.ndarray,
                        *, min_active: float = 0.0) -> np.ndarray:
    """
    Per-bucket duration (ms) that liquidity has continuously rested up to now.

    For each price bucket, measures the contiguous run of columns with resting
    size above ``min_active`` that ends at the last column, and returns its
    duration in ms. A bucket whose last column is empty has persistence 0. This
    is what tells real (persistent) liquidity from a level that just appeared.

    Args:
        mat (np.ndarray): One side's grid [T x P] (bid or ask resting size).
        col_ts (np.ndarray): Column timestamps [T] (ms, ascending).
        min_active (float): Minimum size for a cell to count as active.

    Returns:
        np.ndarray: float64 [P] persistence in ms per bucket (0 where inactive).
    """
    mat = np.asarray(mat, dtype=np.float64)
    t, p = mat.shape
    persist = np.zeros(p, dtype=np.float64)
    if t == 0 or p == 0:
        return persist
    col_ts = np.asarray(col_ts, dtype=np.int64)
    active = mat > float(min_active)
    # Contiguous active run length ending at the last column: reverse cumulative
    # AND, then count. run_len[p] trailing columns are active.
    rev_and = np.cumprod(active[::-1], axis=0).astype(bool)
    run_len = rev_and.sum(axis=0).astype(np.int64)
    has = run_len > 0
    first_idx = np.clip(t - run_len, 0, t - 1)
    persist[has] = (col_ts[-1] - col_ts[first_idx[has]]).astype(np.float64)
    return persist


def _empty_heatmap_walls() -> dict:
    return {
        "price": np.zeros(0, dtype=np.float64),
        "size": np.zeros(0, dtype=np.float64),
        "side": np.zeros(0, dtype=np.int8),
        "persistence_ms": np.zeros(0, dtype=np.int64),
    }


def find_heatmap_walls(
    grid: dict,
    *,
    min_size: float = 0.0,
    rel_multiple: float = 0.0,
    persistence_ms: int = 0,
    bid_pers: "np.ndarray | None" = None,
    ask_pers: "np.ndarray | None" = None,
) -> dict:
    """
    Find the persistent resting-liquidity walls in the latest grid column.

    A wall is a price bucket whose current resting size is large (absolute
    ``min_size`` and/or ``rel_multiple`` times the median size of the current
    column) and that has rested for at least ``persistence_ms`` (measured with
    ``resolve_persistence``). Reads the newest column of ``grid`` (the "now"
    state), so it is the causal wall snapshot the query API surfaces.

    Args:
        grid (dict): ``build_heatmap_grid`` output.
        min_size (float): Absolute minimum resting size to qualify.
        rel_multiple (float): Multiple of the current column's median non-empty
            size (0 disables the relative term).
        persistence_ms (int): Minimum resting duration to report.
        bid_pers (np.ndarray | None): Precomputed bid persistence [P] to reuse
            (falls back to ``resolve_persistence`` when None).
        ask_pers (np.ndarray | None): Precomputed ask persistence [P] to reuse.

    Returns:
        dict: 'price', 'size', 'side' (+1 bid / -1 ask), 'persistence_ms' arrays
        for the detected walls, ascending by price.
    """
    bid = np.asarray(grid.get("bid", np.zeros((0, 0))), dtype=np.float64)
    ask = np.asarray(grid.get("ask", np.zeros((0, 0))), dtype=np.float64)
    centers = np.asarray(grid.get("price_centers", []), dtype=np.float64)
    col_ts = np.asarray(grid.get("col_ts", []), dtype=np.int64)
    if bid.shape[0] == 0 or centers.size == 0:
        return _empty_heatmap_walls()

    bid_now = bid[-1]
    ask_now = ask[-1]
    present = np.concatenate([bid_now[bid_now > 0.0], ask_now[ask_now > 0.0]])
    thr = float(min_size)
    if rel_multiple and present.size:
        thr = max(thr, float(rel_multiple) * float(np.median(present)))

    bid_pers = (resolve_persistence(bid, col_ts) if bid_pers is None
                else np.asarray(bid_pers, dtype=np.float64))
    ask_pers = (resolve_persistence(ask, col_ts) if ask_pers is None
                else np.asarray(ask_pers, dtype=np.float64))

    prices, sizes, sides, pers = [], [], [], []
    for now, pmat, sign in ((bid_now, bid_pers, 1), (ask_now, ask_pers, -1)):
        sel = np.nonzero((now >= thr) & (now > 0.0) & (pmat >= persistence_ms))[0]
        for i in sel:
            prices.append(float(centers[i]))
            sizes.append(float(now[i]))
            sides.append(sign)
            pers.append(int(pmat[i]))

    if not prices:
        return _empty_heatmap_walls()
    order = np.argsort(prices)
    return {
        "price": np.asarray(prices, dtype=np.float64)[order],
        "size": np.asarray(sizes, dtype=np.float64)[order],
        "side": np.asarray(sides, dtype=np.int8)[order],
        "persistence_ms": np.asarray(pers, dtype=np.int64)[order],
    }


def heatmap_color_bounds(
    bid: np.ndarray,
    ask: np.ndarray,
    *,
    pct: float = 99.0,
    scale="auto",
    pool: "np.ndarray | None" = None,
) -> tuple:
    """
    Per-column color-scale bounds (lo, hi) for the liquidity heatmap.

    This is the causal (Bookmap-style) calibration. With ``scale='auto'`` the
    hot end of column ``j`` is the ``pct`` percentile of ALL non-empty resting
    sizes seen up to and including column ``j`` (an expanding window), so a
    column's color is fixed by the liquidity known at that moment and is NEVER
    repainted by liquidity that arrives later - the past stays frozen. With an
    explicit ``(lo, hi)`` scale the bounds are constant across every column (a
    fixed manual range, identical to the pre-existing behavior).

    An optional ``pool`` seeds the expanding reservoir with sizes from earlier
    columns that are no longer in ``bid`` / ``ask`` (e.g. pruned history), so
    the calibration of a new column reflects the whole session, not just the
    rows passed in.

    Args:
        bid (np.ndarray): Bid side grid [T x P] resting size.
        ask (np.ndarray): Ask side grid [T x P] resting size.
        pct (float): Percentile of non-empty sizes used as the hot end when
            ``scale='auto'``.
        scale (str | tuple): 'auto' (causal expanding calibration) or an
            explicit ``(lo, hi)`` size range.
        pool (np.ndarray | None): Non-empty sizes from prior columns to seed the
            expanding reservoir (None starts empty).

    Returns:
        tuple[np.ndarray, np.ndarray]: (lo[T], hi[T]) float64 per-column bounds.
        ``hi`` falls back to 1.0 for a column with no liquidity seen yet.

    Note:
        The expanding percentile is recomputed per column, so the worst-case
        cost is O(T * nnz) (T time columns, nnz non-empty cells). This is fine
        for the per-refresh grids used live and for the single backtest pass.
    """
    bid = np.asarray(bid, dtype=np.float64)
    ask = np.asarray(ask, dtype=np.float64)
    t = bid.shape[0]
    lo = np.zeros(t, dtype=np.float64)
    hi = np.ones(t, dtype=np.float64)
    if t == 0:
        return lo, hi

    if isinstance(scale, (tuple, list)) and len(scale) == 2:
        lo[:] = float(scale[0])
        hi[:] = float(scale[1])
        return lo, hi

    # 'auto': expanding percentile of non-empty sizes up to each column.
    seed = np.asarray(pool, dtype=np.float64) if pool is not None else None
    parts = [] if seed is None or seed.size == 0 else [seed]
    for j in range(t):
        row = np.concatenate([bid[j], ask[j]])
        row = row[row > 0.0]
        if row.size:
            parts.append(row)
        if parts:
            allv = np.concatenate(parts)
            h = float(np.percentile(allv, pct))
            hi[j] = h if h > 0.0 else float(allv.max())
        # else: no liquidity seen yet, hi[j] stays 1.0
    return lo, hi


def _union_price_axis(centers_a: np.ndarray, centers_b: np.ndarray,
                      bucket: float) -> np.ndarray:
    """
    Union of two bucket-aligned price-center axes.

    Both axes are anchored to absolute multiples of ``bucket`` (see
    ``build_heatmap_grid``), so their union is the contiguous run of centers
    spanning both ranges. An empty axis contributes nothing.

    Args:
        centers_a (np.ndarray): First price-center axis [Pa] (may be empty).
        centers_b (np.ndarray): Second price-center axis [Pb] (may be empty).
        bucket (float): Bucket width shared by both axes.

    Returns:
        np.ndarray: float64 union price centers, ascending.
    """
    a = np.asarray(centers_a, dtype=np.float64)
    b = np.asarray(centers_b, dtype=np.float64)
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a.copy()
    lo = min(float(a[0]), float(b[0]))
    hi = max(float(a[-1]), float(b[-1]))
    n = int(round((hi - lo) / bucket)) + 1
    return lo + np.arange(max(n, 1), dtype=np.float64) * bucket


def _place_on_axis(mat: np.ndarray, centers: np.ndarray,
                   union_centers: np.ndarray, bucket: float) -> np.ndarray:
    """
    Re-index a [T x P] side matrix onto a wider union price axis (zero-filled).

    Args:
        mat (np.ndarray): Side matrix [T x P] on ``centers``.
        centers (np.ndarray): The matrix's price centers [P].
        union_centers (np.ndarray): Target union axis [Pu] (superset).
        bucket (float): Bucket width.

    Returns:
        np.ndarray: float64 [T x Pu] with ``mat`` placed at its bucket offset.
    """
    t = mat.shape[0]
    pu = union_centers.size
    out = np.zeros((t, pu), dtype=np.float64)
    if mat.size == 0 or centers.size == 0:
        return out
    offset = int(round((float(centers[0]) - float(union_centers[0])) / bucket))
    out[:, offset:offset + centers.size] = mat
    return out


def merge_heatmap_grids(accum, new, *, max_history_columns=None) -> dict:
    """
    Merge a freshly built heatmap grid into an accumulated one (live/replay).

    Keeps the full session history (Bookmap-style) instead of only the sliding
    book window. The merge is a full outer join on ``col_ts`` over the union
    price axis:

    - A column present only in ``accum`` (it has slid out of the book window)
      is preserved untouched - its frozen per-column color is never repainted.
    - A column present only in ``new`` is appended.
    - A column present in BOTH (the same time bucket seen across refreshes) is
      combined by an element-wise MAX of the resting sizes. This is exactly the
      peak aggregation ``build_heatmap_grid`` applies within a time bucket, so a
      time-bucketed column keeps accumulating its true peak even when its
      snapshots are spread across several refreshes / partial windows (a
      snapshot is captured on the refresh where it is the newest, and MAX folds
      those captures together). With one column per book event
      (``time_bucket_ms=None``) every ``col_ts`` is unique and the overlap MAX
      is a no-op, so this reduces to a plain append.

    Buckets are anchored to absolute price-grid multiples, so overlapping price
    buckets align exactly on the union axis.

    Per-column arrays are carried uniformly (``col_ts`` / ``col_left`` /
    ``col_right`` and, when present on both grids, the frozen ``color_lo`` /
    ``color_hi`` - the accumulator's value wins on an overlap so the past is not
    repainted). ``_accumulate_grid`` recomputes the colors over the merged grid
    afterwards; the color carry here is for direct callers.

    Rewind (replay restart) is signalled when the new grid's newest column is
    older than the accumulator's newest; the caller then resets and re-merges.

    Args:
        accum (dict | None): Previously accumulated grid (None/empty to start).
        new (dict): Grid freshly built from the current book window (may carry
            ``color_lo`` / ``color_hi`` per column).
        max_history_columns (int | None): Cap on retained columns (drops the
            oldest beyond the cap; None keeps the whole session).

    Returns:
        dict: The merged grid (same schema as ``build_heatmap_grid``, plus the
        carried ``color_lo`` / ``color_hi`` when provided). Carries a private
        ``_rewind`` bool flag for the caller.
    """
    new = new or {}
    new_ts = np.asarray(new.get("col_ts", []), dtype=np.int64)
    accum = accum or {}
    acc_ts = np.asarray(accum.get("col_ts", []), dtype=np.int64)

    # Rewind: the freshly built grid is older than what we accumulated.
    rewind = bool(acc_ts.size and new_ts.size and new_ts[-1] < acc_ts[-1])
    if rewind or acc_ts.size == 0:
        out = _slice_grid_columns(new, 0, new_ts.size)
        out["_rewind"] = rewind
        return _cap_grid_columns(out, max_history_columns)
    if new_ts.size == 0:
        out = dict(accum)
        out["_rewind"] = False
        return out

    bucket = float(accum.get("price_bucket", new.get("price_bucket", 1.0)))
    acc_centers = np.asarray(accum.get("price_centers", []), dtype=np.float64)
    new_centers = np.asarray(new.get("price_centers", []), dtype=np.float64)
    union = _union_price_axis(acc_centers, new_centers, bucket)
    edges = np.empty(union.size + 1, dtype=np.float64)
    if union.size:
        edges[:-1] = union - bucket / 2.0
        edges[-1] = union[-1] + bucket / 2.0

    # Both grids on the shared price axis.
    acc_bid = _place_on_axis(np.asarray(accum["bid"], dtype=np.float64),
                             acc_centers, union, bucket)
    acc_ask = _place_on_axis(np.asarray(accum["ask"], dtype=np.float64),
                             acc_centers, union, bucket)
    new_bid = _place_on_axis(np.asarray(new["bid"], dtype=np.float64),
                             new_centers, union, bucket)
    new_ask = _place_on_axis(np.asarray(new["ask"], dtype=np.float64),
                             new_centers, union, bucket)

    acc_left = np.asarray(accum["col_left"], dtype=np.int64)
    acc_right = np.asarray(accum["col_right"], dtype=np.int64)
    new_left = np.asarray(new["col_left"], dtype=np.int64)
    new_right = np.asarray(new["col_right"], dtype=np.int64)

    carry_color = all(k in accum and k in new
                      for k in ("color_lo", "color_hi"))
    if carry_color:
        acc_lo = np.asarray(accum["color_lo"], dtype=np.float64)
        acc_hi = np.asarray(accum["color_hi"], dtype=np.float64)
        new_lo = np.asarray(new["color_lo"], dtype=np.float64)
        new_hi = np.asarray(new["color_hi"], dtype=np.float64)

    # Full outer join on col_ts (sorted union), MAX on overlapping sizes.
    all_ts = np.union1d(acc_ts, new_ts).astype(np.int64)
    t, p = all_ts.size, union.size
    bid = np.zeros((t, p), dtype=np.float64)
    ask = np.zeros((t, p), dtype=np.float64)
    col_left = np.zeros(t, dtype=np.int64)
    col_right = np.zeros(t, dtype=np.int64)
    color_lo = np.zeros(t, dtype=np.float64)
    color_hi = np.ones(t, dtype=np.float64)

    ai = np.searchsorted(acc_ts, all_ts)
    ni = np.searchsorted(new_ts, all_ts)
    for i, ts in enumerate(all_ts):
        in_acc = ai[i] < acc_ts.size and acc_ts[ai[i]] == ts
        in_new = ni[i] < new_ts.size and new_ts[ni[i]] == ts
        if in_acc and in_new:
            a, b = ai[i], ni[i]
            bid[i] = np.maximum(acc_bid[a], new_bid[b])
            ask[i] = np.maximum(acc_ask[a], new_ask[b])
            col_left[i], col_right[i] = acc_left[a], acc_right[a]
            if carry_color:
                color_lo[i], color_hi[i] = acc_lo[a], acc_hi[a]
        elif in_acc:
            a = ai[i]
            bid[i], ask[i] = acc_bid[a], acc_ask[a]
            col_left[i], col_right[i] = acc_left[a], acc_right[a]
            if carry_color:
                color_lo[i], color_hi[i] = acc_lo[a], acc_hi[a]
        else:
            b = ni[i]
            bid[i], ask[i] = new_bid[b], new_ask[b]
            col_left[i], col_right[i] = new_left[b], new_right[b]
            if carry_color:
                color_lo[i], color_hi[i] = new_lo[b], new_hi[b]

    merged = {
        "col_ts": all_ts,
        "col_left": col_left,
        "col_right": col_right,
        "price_centers": union,
        "price_edges": edges,
        "bid": bid,
        "ask": ask,
        "levels": int(max(accum.get("levels", 0), new.get("levels", 0))),
        "price_bucket": bucket,
    }
    if carry_color:
        merged["color_lo"] = color_lo
        merged["color_hi"] = color_hi
    merged["_rewind"] = False
    return _cap_grid_columns(merged, max_history_columns)


def _slice_grid_columns(grid: dict, i0: int, i1: int) -> dict:
    """Return a copy of ``grid`` keeping only columns [i0:i1]."""
    per_col = ("col_ts", "col_left", "col_right", "bid", "ask",
               "color_lo", "color_hi")
    out = {}
    for k, v in grid.items():
        if k == "_rewind":
            continue
        if k in per_col:
            out[k] = np.asarray(v)[i0:i1].copy()
        else:
            out[k] = np.asarray(v).copy() if isinstance(v, np.ndarray) else v
    return out


def _cap_grid_columns(grid: dict, max_history_columns: "int | None") -> dict:
    """Drop the oldest columns so at most ``max_history_columns`` remain."""
    if max_history_columns is None:
        return grid
    cap = int(max_history_columns)
    t = np.asarray(grid.get("col_ts", []), dtype=np.int64).size
    if cap <= 0 or t <= cap:
        return grid
    rewind = grid.get("_rewind", False)
    out = _slice_grid_columns(grid, t - cap, t)
    out["_rewind"] = rewind
    return out


# ==============================================================================
# ORDER-BOOK / TICK SYNCHRONIZATION PREFLIGHT
# ==============================================================================
#
# When a recorded L2 order book (BookData) is attached to a tick-driven engine
# (backtest / replay / training), the book and the trade streams are two
# independently timestamped series. If their clocks disagree (different origin
# or a constant lag) or the book does not cover the trade range, the causal
# book_as_of() lookup silently returns stale / None and every book metric is
# NaN. This preflight measures the alignment BEFORE the run so the engine can
# warn and, only when the desync is actually recoverable, opt-in to shift the
# book timestamps into the trade clock.
#
# The estimator is causal-alignment based (never interpolates): for a candidate
# time offset L it aligns each trade to the book image as-of (trade_ts) on the
# shifted book clock and scores how well the book mid tracks the trade price.
# A near-constant clock offset is detected as the L that both maximizes
# coverage and minimizes the median |book_mid - trade_price| residual, gated by
# a price correlation so unrelated series are never spuriously "aligned".

# Default sync thresholds (see analyze_book_tick_sync).
BOOK_SYNC_MIN_COVERAGE = 0.90
BOOK_SYNC_MIN_ROWS = 50
BOOK_SYNC_MAX_OFFSET_MS = 60_000
#: Multiple of the median inter-row book interval used as the default
#: max-staleness tolerance when the caller does not pass one explicitly.
BOOK_SYNC_STALENESS_FACTOR = 5.0
#: Minimum price correlation (book mid vs trade price, at the chosen offset)
#: required to trust an estimated offset as a real clock shift.
BOOK_SYNC_MIN_CORR = 0.90

# Verdict codes.
BOOK_SYNC_SYNCED = "synced"
BOOK_SYNC_RECOVERABLE = "recoverable"
BOOK_SYNC_UNRECOVERABLE = "unrecoverable"


class BookSyncReport:
    """
    Immutable diagnostic of how a recorded book aligns with a trade stream.

    Produced by :func:`analyze_book_tick_sync`. The engine reads ``verdict`` to
    decide the sync policy and ``resolved_offset_ms`` as the shift to apply when
    the desync is recoverable and the caller opted in.

    Attributes:
        verdict (str): One of ``'synced'`` / ``'recoverable'`` /
            ``'unrecoverable'``.
        coverage (float): Fraction of trades that have a causal book image
            within the staleness tolerance, at the resolved offset.
        median_staleness_ms (float): Median age (trade_ts - book_as_of_ts) of
            the book image over covered trades, at the resolved offset (NaN when
            nothing is covered).
        estimated_offset_ms (int): Best time offset found (book_ts + offset ->
            trade clock); 0 when the streams are already aligned.
        resolved_offset_ms (int): Offset the engine should actually apply: the
            estimate when ``recoverable``, otherwise 0.
        offset_score (float): Price correlation (book mid vs trade price) at the
            estimated offset; NaN when it could not be computed.
        n_rows (int): Number of book rows.
        n_trades (int): Number of trades.
        max_staleness_ms (float): Staleness tolerance actually used.
        reason (str): Short human-readable explanation of the verdict.
    """

    __slots__ = (
        "verdict", "coverage", "median_staleness_ms", "estimated_offset_ms",
        "resolved_offset_ms", "offset_score", "n_rows", "n_trades",
        "max_staleness_ms", "reason",
    )

    def __init__(
        self, *, verdict, coverage, median_staleness_ms, estimated_offset_ms,
        resolved_offset_ms, offset_score, n_rows, n_trades, max_staleness_ms,
        reason,
    ):
        self.verdict = verdict
        self.coverage = float(coverage)
        self.median_staleness_ms = float(median_staleness_ms)
        self.estimated_offset_ms = int(estimated_offset_ms)
        self.resolved_offset_ms = int(resolved_offset_ms)
        self.offset_score = float(offset_score)
        self.n_rows = int(n_rows)
        self.n_trades = int(n_trades)
        self.max_staleness_ms = float(max_staleness_ms)
        self.reason = str(reason)

    @property
    def synced(self) -> bool:
        """True when the book is already aligned with the trades."""
        return self.verdict == BOOK_SYNC_SYNCED

    @property
    def recoverable(self) -> bool:
        """True when a shift can align the book (offset is trustworthy)."""
        return self.verdict == BOOK_SYNC_RECOVERABLE

    def __repr__(self) -> str:
        return (
            f"BookSyncReport(verdict={self.verdict!r}, "
            f"coverage={self.coverage:.3f}, "
            f"offset_ms={self.estimated_offset_ms}, "
            f"staleness_ms={self.median_staleness_ms:.0f}, "
            f"corr={self.offset_score:.3f})"
        )


def _asof_indices(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """
    Causal as-of index of each query into a sorted timestamp array.

    Returns, per query, the index of the most recent ``sorted_ts`` value at or
    before the query (``searchsorted(..., 'right') - 1``); -1 where no such
    value exists. Mirrors LiveBookRing.book_as_of so the preflight measures the
    exact lookup the engine will perform.
    """
    return np.searchsorted(sorted_ts, query_ts, side="right").astype(np.int64) - 1


def _coverage_and_staleness(
    book_ts: np.ndarray, trade_ts: np.ndarray, max_staleness_ms: float,
) -> "tuple[float, float, np.ndarray]":
    """
    Coverage and median staleness of a trade stream against a book clock.

    A trade is covered when it has a causal book image (as-of index >= 0) whose
    age does not exceed ``max_staleness_ms``. Returns (coverage, median
    staleness over covered trades, boolean covered mask).
    """
    n = trade_ts.size
    if n == 0 or book_ts.size == 0:
        return 0.0, float("nan"), np.zeros(n, dtype=bool)
    idx = _asof_indices(book_ts, trade_ts)
    has_image = idx >= 0
    staleness = np.full(n, np.inf, dtype=np.float64)
    if has_image.any():
        staleness[has_image] = trade_ts[has_image] - book_ts[idx[has_image]]
    covered = has_image & (staleness <= max_staleness_ms)
    coverage = float(covered.mean())
    med = float(np.median(staleness[covered])) if covered.any() else float("nan")
    return coverage, med, covered


def _alignment_error(
    book_ts: np.ndarray, book_mid: np.ndarray, trade_ts: np.ndarray,
    trade_price: np.ndarray, max_staleness_ms: float,
) -> "tuple[float, float, float]":
    """
    Score a single time alignment of the book against the trades.

    Aligns each trade to the book mid as-of its timestamp (on the given book
    clock) and returns (coverage, median absolute price residual, price
    correlation) over the covered trades with a finite mid. Lower residual and
    higher correlation mean a better alignment. Residual/correlation are NaN
    when too few covered pairs exist.
    """
    coverage, _, covered = _coverage_and_staleness(
        book_ts, trade_ts, max_staleness_ms,
    )
    if not covered.any():
        return coverage, float("inf"), float("nan")
    idx = _asof_indices(book_ts, trade_ts[covered])
    mid = book_mid[idx]
    price = trade_price[covered]
    ok = np.isfinite(mid) & np.isfinite(price)
    if ok.sum() < 3:
        return coverage, float("inf"), float("nan")
    mid = mid[ok]
    price = price[ok]
    med_err = float(np.median(np.abs(mid - price)))
    if np.std(mid) <= 0.0 or np.std(price) <= 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(mid, price)[0, 1])
    return coverage, med_err, corr


def _search_offset(
    book_ts: np.ndarray, book_mid: np.ndarray, trade_ts: np.ndarray,
    trade_price: np.ndarray, max_staleness_ms: float, max_offset_ms: int,
    min_coverage: float,
) -> "tuple[int, float, float, float]":
    """
    Coarse-to-fine search for the best book->trade time offset.

    Evaluates candidate offsets L over ``[-max_offset_ms, +max_offset_ms]``,
    aligning ``book_ts + L`` to the trades. The best offset maximizes coverage
    first and minimizes the median price residual second (with correlation as a
    tie-safe gate applied by the caller). Returns
    (best_offset_ms, coverage, median_error, correlation).
    """
    def score(offset: int):
        cov, err, corr = _alignment_error(
            book_ts + offset, book_mid, trade_ts, trade_price, max_staleness_ms,
        )
        return cov, err, corr

    # Rank candidates: prefer those meeting min_coverage, then lowest price
    # residual, then higher coverage, then the offset closest to zero (so a
    # truly aligned book resolves to offset 0 instead of an equivalent shift).
    def rank_key(cand):
        off, cov, err, _corr = cand
        meets = cov >= min_coverage
        return (0 if meets else 1, err if np.isfinite(err) else float("inf"),
                -cov, abs(off))

    max_offset_ms = int(max_offset_ms)
    coarse_step = max(1, max_offset_ms // 60)
    coarse = list(range(-max_offset_ms, max_offset_ms + 1, coarse_step))
    if 0 not in coarse:
        coarse.append(0)
    candidates = []
    for off in coarse:
        cov, err, corr = score(off)
        candidates.append((off, cov, err, corr))
    best = min(candidates, key=rank_key)

    # Refine around the coarse winner.
    best_off = best[0]
    fine_step = max(1, coarse_step // 50)
    lo = max(-max_offset_ms, best_off - coarse_step)
    hi = min(max_offset_ms, best_off + coarse_step)
    fine = list(range(lo, hi + 1, fine_step))
    for off in fine:
        cov, err, corr = score(off)
        candidates.append((off, cov, err, corr))
    best = min(candidates, key=rank_key)
    return best[0], best[1], best[2], best[3]


def analyze_book_tick_sync(
    trade_ts: np.ndarray,
    book_ts: np.ndarray,
    book_mid: np.ndarray,
    trade_price: np.ndarray,
    *,
    min_coverage: float = BOOK_SYNC_MIN_COVERAGE,
    min_rows: int = BOOK_SYNC_MIN_ROWS,
    max_staleness_ms: "float | None" = None,
    max_offset_ms: int = BOOK_SYNC_MAX_OFFSET_MS,
    min_corr: float = BOOK_SYNC_MIN_CORR,
) -> BookSyncReport:
    """
    Diagnose how a recorded order book aligns with a trade stream.

    Pure and causal: it only reads timestamps and prices and only measures the
    same as-of lookup the engine performs (never interpolates). It decides
    whether a book can be trusted as-is, shifted into the trade clock, or must
    be rejected as too desynchronized to align.

    Args:
        trade_ts (np.ndarray): Trade timestamps [N] (ms), ascending.
        book_ts (np.ndarray): Book event timestamps [M] (ms), ascending.
        book_mid (np.ndarray): Book mid price per event [M] (NaN allowed).
        trade_price (np.ndarray): Trade prices [N].
        min_coverage (float): Minimum fraction of trades that must have a fresh
            causal book image to consider the book usable.
        min_rows (int): Minimum book rows; fewer -> unrecoverable (not enough
            data to align).
        max_staleness_ms (float | None): Max age of a book image for a trade to
            count as covered. None -> BOOK_SYNC_STALENESS_FACTOR x the median
            inter-row book interval.
        max_offset_ms (int): Largest clock offset that may be corrected; a best
            offset beyond this is treated as unrecoverable.
        min_corr (float): Minimum book-mid/trade-price correlation required to
            trust an estimated offset as a real clock shift.

    Returns:
        BookSyncReport: The alignment diagnostic and verdict.
    """
    trade_ts = np.asarray(trade_ts, dtype=np.float64).ravel()
    book_ts = np.asarray(book_ts, dtype=np.float64).ravel()
    book_mid = np.asarray(book_mid, dtype=np.float64).ravel()
    trade_price = np.asarray(trade_price, dtype=np.float64).ravel()

    n_rows = int(book_ts.size)
    n_trades = int(trade_ts.size)

    # Resolve the staleness tolerance from the book's own cadence when not given.
    if max_staleness_ms is None:
        if n_rows >= 2:
            dt = np.median(np.diff(book_ts))
            dt = dt if np.isfinite(dt) and dt > 0 else 1000.0
        else:
            dt = 1000.0
        max_staleness_ms = float(BOOK_SYNC_STALENESS_FACTOR * dt)
    max_staleness_ms = float(max_staleness_ms)

    def report(verdict, coverage, staleness, est_off, resolved_off, corr, reason):
        return BookSyncReport(
            verdict=verdict, coverage=coverage, median_staleness_ms=staleness,
            estimated_offset_ms=est_off, resolved_offset_ms=resolved_off,
            offset_score=corr, n_rows=n_rows, n_trades=n_trades,
            max_staleness_ms=max_staleness_ms, reason=reason,
        )

    # Nothing to align against - treat as trivially synced (no book work).
    if n_trades == 0:
        return report(BOOK_SYNC_SYNCED, 1.0, float("nan"), 0, 0, float("nan"),
                      "no trades to align")

    # Alignment of the book as delivered (zero offset).
    cov0, stale0, _ = _coverage_and_staleness(book_ts, trade_ts, max_staleness_ms)
    _, err0, corr0 = _alignment_error(
        book_ts, book_mid, trade_ts, trade_price, max_staleness_ms,
    )

    # Search for the offset that best aligns the book with the trades by
    # minimizing the price residual (coverage + staleness alone cannot see a
    # constant clock offset when both streams are dense: the as-of image stays
    # "fresh" in time but carries the price of a different real moment). The
    # search includes zero, so a truly aligned book resolves to offset 0.
    est_off, _covL, errL, corrL = _search_offset(
        book_ts, book_mid, trade_ts, trade_price, max_staleness_ms,
        max_offset_ms, min_coverage,
    )
    covL_final, staleL, _ = _coverage_and_staleness(
        book_ts + est_off, trade_ts, max_staleness_ms,
    )

    # An offset smaller than the streams' own sampling resolution is
    # indistinguishable from zero.
    def _median_dt(arr):
        if arr.size >= 2:
            d = np.median(np.diff(arr))
            return float(d) if np.isfinite(d) and d > 0 else 1.0
        return 1.0
    offset_tol = max(_median_dt(trade_ts), _median_dt(book_ts), 1.0)
    price_scale = float(np.median(np.abs(np.diff(trade_price)))) if n_trades >= 2 else 0.0
    if not np.isfinite(price_scale) or price_scale <= 0:
        price_scale = 1e-9

    # A shift is "beneficial" only when it is more than a sampling interval
    # away AND it materially reduces the price residual (the zero-offset
    # residual is real - above one typical price step - and the shift roughly
    # halves it). When the book is already aligned no shift helps, so this is
    # False and the book is accepted as-is regardless of how few rows it has:
    # min_rows / coverage only gate whether a NEEDED shift can be trusted, never
    # the attach of an already-aligned book.
    beneficial = bool(
        abs(est_off) > offset_tol
        and np.isfinite(errL)
        and (
            (not np.isfinite(err0))
            or (err0 > price_scale and errL < 0.5 * err0)
        )
    )

    if beneficial:
        # The book needs a shift; only trust it with enough data, real coverage,
        # a price correlation (guards against aligning unrelated series that
        # merely overlap in time) and an offset within tolerance.
        trustworthy = (
            n_rows >= int(min_rows)
            and covL_final >= min_coverage
            and abs(est_off) <= int(max_offset_ms)
            and np.isfinite(corrL) and corrL >= min_corr
        )
        if trustworthy:
            return report(
                BOOK_SYNC_RECOVERABLE, covL_final, staleL, est_off, est_off,
                corrL,
                f"offset {est_off} ms aligns book (coverage {covL_final:.2%}, "
                f"corr {corrL:.2f})",
            )
        if n_rows < int(min_rows):
            reason = (
                f"book needs a {est_off} ms shift but only {n_rows} rows "
                f"(< min_rows={int(min_rows)}) - not enough data to align"
            )
        elif covL_final < min_coverage:
            reason = (
                f"best coverage {covL_final:.2%} < {min_coverage:.0%} within "
                f"+/-{int(max_offset_ms)} ms (timestamps too far apart)"
            )
        elif not (np.isfinite(corrL) and corrL >= min_corr):
            reason = (
                f"book mid does not track trade price (corr {corrL:.2f} "
                f"< {min_corr:.2f})"
            )
        else:
            reason = f"required offset {est_off} ms exceeds +/-{int(max_offset_ms)} ms"
        return report(
            BOOK_SYNC_UNRECOVERABLE, cov0, stale0, est_off, 0, corrL, reason,
        )

    # No beneficial shift: the book is best used as delivered. Refuse only two
    # cases the user asked to refuse:
    #   - the book never overlaps the trades (disjoint ranges beyond the
    #     correctable offset) - "timestamps too far apart";
    #   - the book mid does not track the trade price where it IS covered (an
    #     unrelated / corrupt book) and no shift fixes it.
    # Otherwise it is aligned (partial coverage is fine - trades outside the
    # book range simply read NaN).
    overlap = max(cov0, covL_final) * n_trades
    if overlap < 1.0:
        return report(
            BOOK_SYNC_UNRECOVERABLE, cov0, stale0, est_off, 0, corrL,
            f"book does not overlap the trades within +/-{int(max_offset_ms)} "
            f"ms (timestamps too far apart)",
        )
    price_consistent = (
        err0 <= price_scale
        or not np.isfinite(corr0)          # flat/degenerate price -> residual is just spread
        or corr0 >= min_corr
    )
    if not price_consistent:
        return report(
            BOOK_SYNC_UNRECOVERABLE, cov0, stale0, est_off, 0, corr0,
            f"book mid does not track the trade price (corr {corr0:.2f} "
            f"< {min_corr:.2f}) and no offset within +/-{int(max_offset_ms)} ms "
            f"fixes it",
        )
    return report(
        BOOK_SYNC_SYNCED, cov0, stale0, 0, 0, corr0,
        f"book aligned as delivered (coverage {cov0:.2%})",
    )

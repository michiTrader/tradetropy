import numpy as np

from tradetropy.core.constants import (
    TICK_COLS,
    N_TICK_COLS,
    N_OHLC_COLS,
    N_OHLCV_TURNOVER_COLS,
    _TICK_COL,
    _OHLC_COL,
    _OHLCV_TURNOVER_COL,
    parse_timeframe,
)
from tradetropy.exceptions import DataError


def build_candles_from_ticks(
    timestamps_ms: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    interval_ms: int,
) -> dict:
    """
    Build OHLC candles from ticks using vectorized operations.

    Constructs candles by grouping ticks into time intervals. Returns a dict
    with vectorized arrays for efficient backtesting and analysis.

    Args:
        timestamps_ms (ndarray): Tick timestamps in milliseconds
        prices (ndarray): Tick prices
        volumes (ndarray): Tick volumes
        interval_ms (int): Candle interval in milliseconds

    Returns:
        dict: Contains the following keys:

        - closed_candles: ndarray [M x 6] - complete candles
          [timestamp, open, high, low, close, volume]
        - tick_to_candle_map: ndarray [N] - for each tick, its candle index
        - accumulated_per_tick: ndarray [N x 4] - accumulated [open, high, low,
          volume] up to each tick within its candle
        - candle_ts_per_tick: ndarray [N] - candle opening timestamp per tick
        - prices: ndarray [N] - original prices array

    Example:
        result = build_candles_from_ticks(ts, prices, volumes, 60000)
        candles = result['closed_candles']
    """
    interval_ms = int(interval_ms)

    candle_ts_per_tick = (timestamps_ms // interval_ms) * interval_ms

    is_new_candle = np.r_[True, candle_ts_per_tick[1:] != candle_ts_per_tick[:-1]]
    start_indices = np.where(is_new_candle)[0]
    end_indices = np.r_[start_indices[1:], len(timestamps_ms)]

    n_candles_total = len(start_indices)

    count_per_candle = np.diff(np.r_[start_indices, len(timestamps_ms)])
    tick_to_candle_map = np.repeat(np.arange(n_candles_total), count_per_candle)

    open_per_tick = np.repeat(prices[start_indices], count_per_candle)

    high_accum = prices.copy()
    low_accum = prices.copy()
    for s, e in zip(start_indices, end_indices):
        np.maximum.accumulate(high_accum[s:e], out=high_accum[s:e])
        np.minimum.accumulate(low_accum[s:e], out=low_accum[s:e])

    cumsum_vol = np.cumsum(volumes)
    vol_at_start = cumsum_vol[start_indices] - volumes[start_indices]
    vol_accum = cumsum_vol - np.repeat(vol_at_start, count_per_candle)

    accumulated_per_tick = np.column_stack([open_per_tick, high_accum, low_accum, vol_accum])

    close_indices = end_indices - 1
    n_closed_candles = n_candles_total - 1

    if n_closed_candles > 0:
        closed_close_indices = close_indices[:n_closed_candles]
        closed_candles = np.column_stack(
            [
                candle_ts_per_tick[closed_close_indices],
                accumulated_per_tick[closed_close_indices, 0],
                accumulated_per_tick[closed_close_indices, 1],
                accumulated_per_tick[closed_close_indices, 2],
                prices[closed_close_indices],
                accumulated_per_tick[closed_close_indices, 3],
            ]
        )
    else:
        closed_candles = np.empty((0, N_OHLC_COLS), dtype=np.float64)

    return {
        "closed_candles": closed_candles.astype(np.float64),
        "tick_to_candle_map": tick_to_candle_map,
        "accumulated_per_tick": accumulated_per_tick.astype(np.float64),
        "candle_ts_per_tick": candle_ts_per_tick,
        "prices": prices,
    }


def normalize_ticks(ticks: np.ndarray) -> np.ndarray:
    """
    Fill missing or invalid columns in a tick array.

    Handles missing values in tick data by filling with sensible defaults.
    Columns processed: ts, bid, ask, volume, flags, volume_real, price.

    Args:
        ticks (ndarray): Tick data with shape [N, 7]

    Returns:
        ndarray: Normalized tick array [N, 7] with no NaN values

    Processing rules:
        - bid, ask: NaN or 0 -> price
        - volume: NaN -> 0
        - flags: NaN -> 0
        - price: NaN or 0 -> (bid + ask) / 2
        - ts, volume_real: unchanged

    Raises:
        DataError: If array shape is not [N, 7]

    Example:
        normalized = normalize_ticks(raw_ticks)
    """
    if ticks.ndim != 2 or ticks.shape[1] != N_TICK_COLS:
        raise DataError(
            f"normalize_ticks() expects an N×{N_TICK_COLS} array, "
            f"but received shape {ticks.shape}. "
            f"Columns must be: {TICK_COLS}."
        )

    out = ticks.astype(np.float64, copy=True)
    C = _TICK_COL
    price = out[:, C["price"]]
    bid = out[:, C["bid"]]
    ask = out[:, C["ask"]]

    bid[np.isnan(bid) | (bid == 0.0)] = price[np.isnan(bid) | (bid == 0.0)]
    ask[np.isnan(ask) | (ask == 0.0)] = price[np.isnan(ask) | (ask == 0.0)]

    vol = out[:, C["volume"]]
    vol[np.isnan(vol)] = 0.0

    flags = out[:, C["flags"]]
    flags[np.isnan(flags)] = 0.0

    mask_price = np.isnan(price) | (price == 0.0)
    if mask_price.any():
        price[mask_price] = (bid[mask_price] + ask[mask_price]) / 2.0

    return out


# =============================================================================
# PUBLIC API -- conversion and resample (vectorized, no Python loops)
# =============================================================================


def ticks_to_klines(
    ticks: np.ndarray,
    interval_ms,
    *,
    include_partial: bool = False,
    price_source: str = 'price',
    volume_source: str = 'volume',
) -> np.ndarray:
    """
    Convert tick array to OHLCV candles. Fully vectorized operation.

    Aggregates ticks into candles at specified interval. Handles partial candles
    (the in-progress candle) and supports custom price/volume sources.

    Args:
        ticks (ndarray): Tick data [N x 7] - ts, bid, ask, volume, flags,
          volume_real, price
        interval_ms: Candle interval in milliseconds or timeframe string
          ('5m', '1h', etc.)
        include_partial (bool): If False (default), omits the last incomplete
          candle. If True, includes it with values computed so far.
        price_source (str): 'price' uses price column, 'mid' uses (bid+ask)/2
        volume_source (str): 'volume' or 'volume_real'

    Returns:
        ndarray: [M x 7] float64 array with columns:
          ts, open, high, low, close, volume, turnover

        Turnover is computed as sum of (price x volume) per candle. Returns
        empty [0 x 7] if input is empty or no closed candles exist.

    Raises:
        DataError: If price_source or volume_source is invalid

    Example:
        klines = ticks_to_klines(ticks, 60000)
        klines_1m = ticks_to_klines(ticks, '1m')
    """
    interval_ms = parse_timeframe(interval_ms)

    empty = np.empty((0, N_OHLCV_TURNOVER_COLS), dtype=np.float64)
    if ticks is None or len(ticks) == 0:
        return empty

    t = normalize_ticks(np.asarray(ticks, dtype=np.float64))

    C = _TICK_COL
    ts_col = t[:, C["ts"]]

    if price_source == "price":
        px = t[:, C["price"]]
    elif price_source == "mid":
        px = (t[:, C["bid"]] + t[:, C["ask"]]) / 2.0
    else:
        raise DataError(
            f"Invalid price_source: {price_source!r}. Use 'price' or 'mid'."
        )

    if volume_source == "volume":
        vol = t[:, C["volume"]]
    elif volume_source == "volume_real":
        vol = t[:, C["volume_real"]]
    else:
        raise DataError(
            f"Invalid volume_source: {volume_source!r}. "
            f"Use 'volume' or 'volume_real'."
        )
    vol = np.nan_to_num(vol, nan=0.0)

    # Opening timestamp for the candle each tick belongs to.
    bar_ts = (ts_col // interval_ms).astype(np.int64) * interval_ms

    # Start indices of each contiguous segment (one candle).
    es_nueva = np.r_[True, bar_ts[1:] != bar_ts[:-1]]
    starts = np.where(es_nueva)[0]
    ends = np.r_[starts[1:], len(ts_col)]

    open_col = px[starts]
    high_col = np.maximum.reduceat(px, starts)
    low_col = np.minimum.reduceat(px, starts)
    close_col = px[ends - 1]
    vol_col = np.add.reduceat(vol, starts)
    turnover_col = np.add.reduceat(px * vol, starts)
    ts_open = bar_ts[starts].astype(np.float64)

    result = np.column_stack(
        [ts_open, open_col, high_col, low_col, close_col, vol_col, turnover_col]
    ).astype(np.float64)

    if not include_partial:
        result = result[:-1]  # discard the last (in-progress) candle

    if len(result) == 0:
        return empty
    return result


def resample_klines(
    ohlc: np.ndarray,
    source_interval_ms: int,
    target_interval_ms,
) -> tuple[np.ndarray, int]:
    """
    Aggregate candles to a higher timeframe. Fully vectorized operation.

    Resamples OHLC candles by grouping them into larger time buckets. Preserves
    the number of columns (6 or 7) from the input array.

    Args:
        ohlc (ndarray): [N x 6] or [N x 7] OHLC data, time-sorted
        source_interval_ms (int): Original candle interval in milliseconds
        target_interval_ms: Target candle interval in milliseconds or timeframe
          string ('5m', '1h', etc.)

    Returns:
        tuple: (ndarray [M x K], effective_target_interval_ms)

        Where M is the number of aggregated candles and K matches the input
        column count (6 or 7). If target interval is smaller than source,
        returns input array unchanged with source interval.

    Aggregation rules per group:
        - ts: Opening timestamp of the group
        - open: First candle's open
        - high: Maximum high
        - low: Minimum low
        - close: Last candle's close
        - volume: Sum of volumes
        - turnover: Sum (if 7-column input)

    Raises:
        DataError: If input shape is not [N, 6] or [N, 7]

    Example:
        klines_5m, interval = resample_klines(klines_1m, 60000, 300000)
    """
    import warnings

    source_interval_ms = int(source_interval_ms)
    target_interval_ms = parse_timeframe(target_interval_ms)

    arr = np.asarray(ohlc, dtype=np.float64)
    ncols = arr.shape[1] if arr.ndim == 2 else 0
    if ncols not in (N_OHLC_COLS, N_OHLCV_TURNOVER_COLS):
        raise DataError(
            f"resample_klines() expects an N×{N_OHLC_COLS} or "
            f"N×{N_OHLCV_TURNOVER_COLS} array, received shape "
            f"{getattr(arr, 'shape', None)}."
        )

    if len(arr) == 0:
        return arr, source_interval_ms

    if target_interval_ms <= source_interval_ms:
        if target_interval_ms < source_interval_ms:
            warnings.warn(
                f"target_interval_ms={target_interval_ms} is less than the "
                f"original interval ({source_interval_ms}). Using the original.",
                stacklevel=2,
            )
        return arr, source_interval_ms

    remainder = target_interval_ms % source_interval_ms
    if remainder != 0:
        adjusted = target_interval_ms + (source_interval_ms - remainder)
        warnings.warn(
            f"target_interval_ms={target_interval_ms} is not a multiple of "
            f"{source_interval_ms}. Adjusted to {adjusted}.",
            stacklevel=2,
        )
        target_interval_ms = adjusted

    has_turnover = ncols == N_OHLCV_TURNOVER_COLS

    ts_col = arr[:, 0]
    open_col = arr[:, 1]
    high_col = arr[:, 2]
    low_col = arr[:, 3]
    close_col = arr[:, 4]
    vol_col = arr[:, 5]

    bar_ts = (ts_col // target_interval_ms).astype(np.int64) * target_interval_ms
    es_nuevo = np.r_[True, bar_ts[1:] != bar_ts[:-1]]
    starts = np.where(es_nuevo)[0]
    ends = np.r_[starts[1:], len(arr)]

    cols = [
        bar_ts[starts].astype(np.float64),
        open_col[starts],
        np.maximum.reduceat(high_col, starts),
        np.minimum.reduceat(low_col, starts),
        close_col[ends - 1],
        np.add.reduceat(vol_col, starts),
    ]
    if has_turnover:
        cols.append(np.add.reduceat(arr[:, 6], starts))

    result = np.column_stack(cols).astype(np.float64)
    return result, target_interval_ms


# =============================================================================
# VALIDATION -- gap detection and non-monotonic timestamps
# =============================================================================


def validate_continuity(candles=None, interval_ms: int | None = None, *, velas=None, intervalo_ms: int | None = None) -> dict:
    """
    Detect gaps and monotonicity violations in candle data.

    Args:
        candles: KlineData object (with .data and .interval_ms) or ndarray [N x K]
        interval_ms (int): Interval in ms. Required if candles is an array.
        velas: Deprecated alias for candles.
        intervalo_ms: Deprecated alias for interval_ms.

    Returns:
        dict: Report with keys:
        ok, n_rows, n_gaps, missing_total, gaps, non_monotonic.

    Raises:
        DataError: If interval_ms <= 0 or not provided.

    Example:
        report = validate_continuity(klines_1m, interval_ms=60000)
        if not report['ok']:
            print(f"Found {report['n_gaps']} gaps")
    """
    # Backward compatibility aliases
    if candles is None and velas is not None:
        candles = velas
    if interval_ms is None and intervalo_ms is not None:
        interval_ms = intervalo_ms

    if hasattr(candles, "data") and (hasattr(candles, "intervalo_ms") or hasattr(candles, "interval_ms")):
        arr = np.asarray(candles.data, dtype=np.float64)
        if interval_ms is None:
            interval_ms = getattr(candles, "interval_ms", None) or getattr(candles, "intervalo_ms", None)
    else:
        arr = np.asarray(candles, dtype=np.float64)

    if interval_ms is None or interval_ms <= 0:
        raise DataError(
            "validate_continuity() requires interval_ms > 0 "
            "(or a KlineData with interval_ms)."
        )
    interval_ms = int(interval_ms)

    n = int(len(arr))
    base = {
        "ok": True,
        "n_rows": n,
        "n_gaps": 0,
        "missing_total": 0,
        "gaps": [],
        "non_monotonic": [],
    }
    if n < 2:
        return base

    ts = arr[:, 0]
    diffs = np.diff(ts)

    non_monotonic = (np.where(diffs <= 0)[0] + 1).tolist()

    gap_idx = np.where(diffs > interval_ms)[0]
    gaps = []
    for i in gap_idx:
        missing = int(diffs[i] // interval_ms) - 1
        if missing <= 0:
            continue
        gaps.append({
            "idx": int(i + 1),
            "ts_prev": float(ts[i]),
            "ts_next": float(ts[i + 1]),
            "missing": missing,
        })

    missing_total = sum(g["missing"] for g in gaps)
    base.update(
        ok=(len(gaps) == 0 and len(non_monotonic) == 0),
        n_gaps=len(gaps),
        missing_total=missing_total,
        gaps=gaps,
        non_monotonic=non_monotonic,
    )
    return base


"""
Pure, causal candlestick pattern detection with adaptive (percentile)
thresholds, a z-score context filter and empirical efficacy tracking.

This module is NumPy/pandas-only and has no plotting or engine dependencies, so
the exact same detection runs identically in backtest, live and replay (the
:class:`~tradetropy.ta.annotations.CandlePatterns` indicator is only a thin
wrapper around it). Two design choices keep it honest:

- **Adaptive size thresholds, not fixed ratios.** What counts as a "small body"
  (doji), a "long body" (marubozu) or a "long wick" (hammer / shooting star) is
  measured against a trailing rolling quantile of the recent distribution of
  that metric (e.g. the lower wick must exceed the ``wick`` percentile of the
  last ``window`` bars). The threshold spec accepts an absolute float, a
  quantile ``'pXX'`` or a median multiple ``'Nx'`` (same convention as
  ``order_flow/_core.py::_rolling_threshold``). The rolling window is trailing
  and inclusive, so no future bar can change a past bar's classification -
  backtest and replay are identical.

- **Context by z-score.** A lower-wick candle is only a *hammer* when price is
  stretched down (z-score of close vs its rolling mean <= ``-zscore_thresh``)
  and a *hanging man* when stretched up; a hammer at the highs is noise. The
  filter is optional (``context_filter``); with it off the shape's classic name
  is used regardless of context.

Patterns are anchored at the LAST bar of the formation (the confirmation bar),
using only that bar and the one or two preceding it, so every code at index
``i`` depends solely on bars ``<= i``.
"""

from __future__ import annotations

import numpy as np

# =====
# PATTERN CODES
# =====
#
# 0 is reserved for "no pattern". Single-candle codes are 1..8, two-candle
# 9..16, three-candle 17..20. Multi-candle patterns take priority over
# single-candle ones at the same bar (see _resolve_priority).
PATTERN_NONE = 0

DOJI = 1
HAMMER = 2
INVERTED_HAMMER = 3
HANGING_MAN = 4
SHOOTING_STAR = 5
MARUBOZU_BULL = 6
MARUBOZU_BEAR = 7
SPINNING_TOP = 8

ENGULFING_BULL = 9
ENGULFING_BEAR = 10
HARAMI_BULL = 11
HARAMI_BEAR = 12
PIERCING = 13
DARK_CLOUD = 14
TWEEZER_BOTTOM = 15
TWEEZER_TOP = 16

MORNING_STAR = 17
EVENING_STAR = 18
THREE_WHITE_SOLDIERS = 19
THREE_BLACK_CROWS = 20

# Human-readable name per code (used for labels and the query API).
PATTERN_LABELS: dict[int, str] = {
    PATTERN_NONE: "none",
    DOJI: "Doji",
    HAMMER: "Hammer",
    INVERTED_HAMMER: "Inverted Hammer",
    HANGING_MAN: "Hanging Man",
    SHOOTING_STAR: "Shooting Star",
    MARUBOZU_BULL: "Bullish Marubozu",
    MARUBOZU_BEAR: "Bearish Marubozu",
    SPINNING_TOP: "Spinning Top",
    ENGULFING_BULL: "Bullish Engulfing",
    ENGULFING_BEAR: "Bearish Engulfing",
    HARAMI_BULL: "Bullish Harami",
    HARAMI_BEAR: "Bearish Harami",
    PIERCING: "Piercing Line",
    DARK_CLOUD: "Dark Cloud Cover",
    TWEEZER_BOTTOM: "Tweezer Bottom",
    TWEEZER_TOP: "Tweezer Top",
    MORNING_STAR: "Morning Star",
    EVENING_STAR: "Evening Star",
    THREE_WHITE_SOLDIERS: "Three White Soldiers",
    THREE_BLACK_CROWS: "Three Black Crows",
}

# Directional bias per code: +1 bullish, -1 bearish, 0 neutral.
PATTERN_DIRECTION: dict[int, float] = {
    PATTERN_NONE: 0.0,
    DOJI: 0.0,
    HAMMER: 1.0,
    INVERTED_HAMMER: 1.0,
    HANGING_MAN: -1.0,
    SHOOTING_STAR: -1.0,
    MARUBOZU_BULL: 1.0,
    MARUBOZU_BEAR: -1.0,
    SPINNING_TOP: 0.0,
    ENGULFING_BULL: 1.0,
    ENGULFING_BEAR: -1.0,
    HARAMI_BULL: 1.0,
    HARAMI_BEAR: -1.0,
    PIERCING: 1.0,
    DARK_CLOUD: -1.0,
    TWEEZER_BOTTOM: 1.0,
    TWEEZER_TOP: -1.0,
    MORNING_STAR: 1.0,
    EVENING_STAR: -1.0,
    THREE_WHITE_SOLDIERS: 1.0,
    THREE_BLACK_CROWS: -1.0,
}

# Priority when several patterns match the same confirmation bar. Higher wins;
# multi-candle reversals dominate single-candle shapes.
_PRIORITY: dict[int, int] = {
    PATTERN_NONE: 0,
    DOJI: 1, SPINNING_TOP: 1,
    MARUBOZU_BULL: 2, MARUBOZU_BEAR: 2,
    HAMMER: 3, INVERTED_HAMMER: 3, HANGING_MAN: 3, SHOOTING_STAR: 3,
    HARAMI_BULL: 4, HARAMI_BEAR: 4,
    ENGULFING_BULL: 5, ENGULFING_BEAR: 5,
    PIERCING: 5, DARK_CLOUD: 5,
    TWEEZER_BOTTOM: 5, TWEEZER_TOP: 5,
    MORNING_STAR: 6, EVENING_STAR: 6,
    THREE_WHITE_SOLDIERS: 6, THREE_BLACK_CROWS: 6,
}


def pattern_name(code) -> str:
    """Return the human-readable label for a pattern code (or 'none')."""
    try:
        return PATTERN_LABELS.get(int(code), "none")
    except (TypeError, ValueError):
        return "none"


def pattern_direction(code) -> float:
    """Return the directional bias for a pattern code (+1 / -1 / 0)."""
    try:
        return PATTERN_DIRECTION.get(int(code), 0.0)
    except (TypeError, ValueError):
        return 0.0


# =====
# Causal rolling helpers
# =====
def _rolling_threshold(metric: np.ndarray, spec, window: int,
                       min_periods: int) -> np.ndarray:
    """
    Per-bar acceptance threshold over a trailing (inclusive) window.

    Args:
        metric (np.ndarray): Per-bar magnitudes [N].
        spec (float | str): Absolute float, quantile ``'pXX'`` or median
            multiple ``'Nx'``. Quantile/median use only the trailing ``window``
            bars, so no future bar influences a past threshold.
        window (int): Trailing window length for the relative specs.
        min_periods (int): Minimum samples before a threshold is produced
            (fewer -> NaN, which downstream treats as "not classifiable yet").

    Returns:
        np.ndarray: float64 threshold per bar [N].

    Raises:
        ValueError: If a string spec is malformed.
    """
    n = metric.shape[0]
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return np.full(n, float(spec), dtype=np.float64)
    if not isinstance(spec, str):
        raise ValueError(f"threshold must be float or str, not {type(spec)!r}")

    import pandas as pd

    s = pd.Series(metric, dtype="float64")
    roll = s.rolling(window=max(int(window), 1), min_periods=max(int(min_periods), 1))
    text = spec.strip().lower()
    if text.startswith("p"):
        try:
            q = float(text[1:]) / 100.0
        except ValueError as exc:
            raise ValueError(f"quantile spec must look like 'p80', got {spec!r}") from exc
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile out of range [0,1]: {spec!r}")
        return roll.quantile(q).to_numpy(dtype=np.float64)
    if text.endswith("x"):
        try:
            mult = float(text[:-1])
        except ValueError as exc:
            raise ValueError(f"multiple spec must look like '2x', got {spec!r}") from exc
        if mult <= 0.0:
            raise ValueError(f"multiple must be > 0: {spec!r}")
        return (roll.median() * mult).to_numpy(dtype=np.float64)
    raise ValueError(f"unrecognized threshold spec: {spec!r}")


def _rolling_zscore(close: np.ndarray, length: int, min_periods: int) -> np.ndarray:
    """
    Causal z-score of close vs its trailing rolling mean / std (population).

    Args:
        close (np.ndarray): Close prices [N].
        length (int): Rolling window.
        min_periods (int): Minimum samples before a value is produced.

    Returns:
        np.ndarray: z-score per bar [N]; 0.0 where std is 0 (flat), NaN during
            warmup.
    """
    import pandas as pd

    s = pd.Series(close, dtype="float64")
    roll = s.rolling(window=max(int(length), 1), min_periods=max(int(min_periods), 1))
    mean = roll.mean()
    std = roll.std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (s - mean) / std
    z = np.array(z.to_numpy(dtype=np.float64), copy=True)
    flat = (std.to_numpy(dtype=np.float64) == 0.0)
    z[flat] = 0.0
    return z


# =====
# Pattern detection
# =====
def detect_candle_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    window: int = 100,
    doji_body: str = "p10",
    small_body: str = "p50",
    long_body: str = "p60",
    wick: str = "p80",
    marubozu_wick: str = "p20",
    doji_frac: float = 0.1,
    zscore_len: int = 50,
    zscore_thresh: float = 0.5,
    context_filter: bool = True,
    eq_tol: float = 0.0015,
    min_periods: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect candlestick patterns causally with adaptive thresholds.

    Every code at bar ``i`` uses only bars ``<= i`` (the pattern is anchored at
    its confirmation bar), so the output is identical in backtest and replay.

    Args:
        open_, high, low, close (np.ndarray): OHLC arrays [N], same length.
        window (int): Trailing window for the rolling size percentiles.
        doji_body (str|float): Threshold for a doji body (very small).
        small_body (str|float): Threshold for a "small" body (harami / star /
            hammer bodies).
        long_body (str|float): Threshold for a "long" body (marubozu / soldiers).
        wick (str|float): Threshold a wick must EXCEED to count as "long"
            (hammer lower wick, shooting-star upper wick).
        marubozu_wick (str|float): Threshold a wick must stay UNDER for a
            marubozu (negligible wicks).
        doji_frac (float): Also require body <= doji_frac * range for a doji, so
            a flat low-volatility stretch does not label everything a doji.
        zscore_len (int): Rolling window for the context z-score.
        zscore_thresh (float): |z| beyond which price is "stretched"
            (down <= -thresh, up >= +thresh).
        context_filter (bool): If True, hammer/hanging-man and inverted-hammer/
            shooting-star are only emitted in the matching context; ambiguous
            context yields no single-wick pattern. If False, the classic name
            (hammer / shooting star) is used regardless.
        eq_tol (float): Fractional tolerance for "equal" highs/lows (tweezers).
        min_periods (int | None): Minimum trailing samples before the rolling
            thresholds produce a value. Defaults to ``min(window, 20)`` so short
            series are still classified (causally) without a full window.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            codes [N] int64 - pattern code per bar (0 = none).
            directions [N] float64 - +1 / -1 / 0 bias of that code.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    codes = np.zeros(n, dtype=np.int64)
    directions = np.zeros(n, dtype=np.float64)
    if n == 0:
        return codes, directions

    if min_periods is None:
        min_periods = min(int(window), 20)
    min_periods = max(int(min_periods), 2)

    body = np.abs(close - open_)
    rng = high - low
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    bullish = close > open_
    bearish = close < open_

    # Causal adaptive thresholds.
    doji_thr = _rolling_threshold(body, doji_body, window, min_periods)
    small_thr = _rolling_threshold(body, small_body, window, min_periods)
    long_thr = _rolling_threshold(body, long_body, window, min_periods)
    lower_wick_thr = _rolling_threshold(lower, wick, window, min_periods)
    upper_wick_thr = _rolling_threshold(upper, wick, window, min_periods)
    lower_maru_thr = _rolling_threshold(lower, marubozu_wick, window, min_periods)
    upper_maru_thr = _rolling_threshold(upper, marubozu_wick, window, min_periods)
    z = _rolling_zscore(close, zscore_len, min_periods=min(zscore_len, min_periods))

    def _le(a, thr):
        return (not np.isnan(thr)) and a <= thr

    def _ge(a, thr):
        return (not np.isnan(thr)) and a >= thr

    for i in range(n):
        r = rng[i]
        if r <= 0.0:
            continue
        b = body[i]
        up = upper[i]
        lo = lower[i]
        z_i = z[i]
        down_ctx = (not np.isnan(z_i)) and z_i <= -zscore_thresh
        up_ctx = (not np.isnan(z_i)) and z_i >= zscore_thresh

        candidates: list[int] = []

        # --- single-candle shapes ---
        is_small = _le(b, small_thr[i])
        is_doji = _le(b, doji_thr[i]) and (b <= doji_frac * r)
        if is_doji:
            candidates.append(DOJI)

        # Marubozu: long body, negligible wicks.
        if _ge(b, long_thr[i]) and _le(up, upper_maru_thr[i]) and _le(lo, lower_maru_thr[i]):
            candidates.append(MARUBOZU_BULL if bullish[i] else MARUBOZU_BEAR)

        # Lower-wick dominant (hammer family): long lower wick, small upper wick,
        # small body sitting in the upper part of the range.
        long_lower = _ge(lo, lower_wick_thr[i]) and lo >= 2.0 * b and up <= b
        long_upper = _ge(up, upper_wick_thr[i]) and up >= 2.0 * b and lo <= b
        if long_lower and not is_doji:
            if context_filter:
                if down_ctx:
                    candidates.append(HAMMER)
                elif up_ctx:
                    candidates.append(HANGING_MAN)
            else:
                candidates.append(HAMMER)
        elif long_upper and not is_doji:
            if context_filter:
                if up_ctx:
                    candidates.append(SHOOTING_STAR)
                elif down_ctx:
                    candidates.append(INVERTED_HAMMER)
            else:
                candidates.append(SHOOTING_STAR)

        # Spinning top: small body with wicks on both sides (not a doji).
        if is_small and not is_doji and up > 0.0 and lo > 0.0 \
                and up >= 0.5 * b and lo >= 0.5 * b and not long_lower and not long_upper:
            candidates.append(SPINNING_TOP)

        # --- two-candle patterns ---
        if i >= 1:
            po, pc = open_[i - 1], close[i - 1]
            ph, pl = high[i - 1], low[i - 1]
            pbody = body[i - 1]
            prev_bull = pc > po
            prev_bear = pc < po
            pmid = (po + pc) / 2.0

            # Engulfing: current body fully engulfs previous body.
            if prev_bear and bullish[i] and close[i] >= po and open_[i] <= pc \
                    and b > pbody:
                candidates.append(ENGULFING_BULL)
            if prev_bull and bearish[i] and open_[i] >= pc and close[i] <= po \
                    and b > pbody:
                candidates.append(ENGULFING_BEAR)

            # Harami: small current body inside the previous (long) body.
            if _ge(pbody, long_thr[i - 1] if not np.isnan(long_thr[i - 1]) else long_thr[i]) \
                    and is_small:
                lo_prev, hi_prev = min(po, pc), max(po, pc)
                lo_cur, hi_cur = min(open_[i], close[i]), max(open_[i], close[i])
                if lo_cur >= lo_prev and hi_cur <= hi_prev:
                    if prev_bear and bullish[i]:
                        candidates.append(HARAMI_BULL)
                    elif prev_bull and bearish[i]:
                        candidates.append(HARAMI_BEAR)

            # Piercing / Dark cloud.
            if prev_bear and bullish[i] and open_[i] < pl and pmid < close[i] < po:
                candidates.append(PIERCING)
            if prev_bull and bearish[i] and open_[i] > ph and po > close[i] > pmid:
                candidates.append(DARK_CLOUD)

            # Tweezers: two adjacent bars sharing (near) the same extreme.
            if ph > 0 and abs(high[i] - ph) <= eq_tol * ph and prev_bull and bearish[i]:
                if (not context_filter) or up_ctx:
                    candidates.append(TWEEZER_TOP)
            if pl > 0 and abs(low[i] - pl) <= eq_tol * pl and prev_bear and bullish[i]:
                if (not context_filter) or down_ctx:
                    candidates.append(TWEEZER_BOTTOM)

        # --- three-candle patterns ---
        if i >= 2:
            o1, c1 = open_[i - 2], close[i - 2]
            b1 = body[i - 2]
            o2, c2 = open_[i - 1], close[i - 1]
            b2 = body[i - 1]
            first_bull = c1 > o1
            first_bear = c1 < o1
            star_small = _le(b2, small_thr[i - 1] if not np.isnan(small_thr[i - 1]) else small_thr[i])
            long1 = _ge(b1, long_thr[i - 2] if not np.isnan(long_thr[i - 2]) else long_thr[i])

            # Morning / Evening star.
            if long1 and star_small and first_bear and bullish[i] \
                    and close[i] > (o1 + c1) / 2.0 and max(o2, c2) < c1 + 0 and b > b2:
                candidates.append(MORNING_STAR)
            if long1 and star_small and first_bull and bearish[i] \
                    and close[i] < (o1 + c1) / 2.0 and min(o2, c2) > c1 - 0 and b > b2:
                candidates.append(EVENING_STAR)

            # Three white soldiers / black crows: three long same-color bodies,
            # each closing beyond the previous close.
            if bullish[i] and (c2 > o2) and first_bull \
                    and _ge(b, long_thr[i]) and _ge(b2, long_thr[i - 1]) and _ge(b1, long_thr[i - 2]) \
                    and c2 > c1 and close[i] > c2:
                candidates.append(THREE_WHITE_SOLDIERS)
            if bearish[i] and (c2 < o2) and first_bear \
                    and _ge(b, long_thr[i]) and _ge(b2, long_thr[i - 1]) and _ge(b1, long_thr[i - 2]) \
                    and c2 < c1 and close[i] < c2:
                candidates.append(THREE_BLACK_CROWS)

        if candidates:
            best = max(candidates, key=lambda c: _PRIORITY.get(c, 0))
            codes[i] = best
            directions[i] = PATTERN_DIRECTION.get(best, 0.0)

    return codes, directions


# =====
# Empirical efficacy (causal)
# =====
def evaluate_efficacy(
    codes: np.ndarray,
    directions: np.ndarray,
    close: np.ndarray,
    *,
    horizon: int = 10,
    as_of: int | None = None,
) -> dict[int, dict]:
    """
    Empirical hit-rate per pattern, counting ONLY signals already resolved.

    A signal at bar ``s`` is "resolved" once ``s + horizon`` bars exist up to
    the evaluation point; it is a hit when the sign of
    ``close[s + horizon] - close[s]`` matches the pattern's directional bias.
    Because only signals with ``s + horizon <= as_of`` are scored, the result
    is strictly causal: querying at bar ``i`` never uses information past ``i``,
    so the running hit-rate is identical in backtest, live and replay.

    Neutral-direction patterns (doji, spinning top) are tracked for count only
    (they have no expected direction, so ``hit_rate`` is NaN).

    Args:
        codes (np.ndarray): Pattern code per bar [N] (0 = none).
        directions (np.ndarray): Directional bias per bar [N].
        close (np.ndarray): Close prices [N].
        horizon (int): Bars ahead used to judge a signal's outcome.
        as_of (int | None): Evaluate as of this bar index (inclusive). None ->
            the last bar (len - 1). Only signals resolved by ``as_of`` count.

    Returns:
        dict[int, dict]: Per pattern code present, a dict with:
            'hit_rate' (float, NaN if no resolved directional signal),
            'sample_size' (int, resolved signals scored),
            'wins' (int), 'name' (str).
    """
    codes = np.asarray(codes)
    directions = np.asarray(directions, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if as_of is None:
        as_of = n - 1
    as_of = min(int(as_of), n - 1)
    h = max(int(horizon), 1)

    stats: dict[int, dict] = {}
    for s in range(0, as_of - h + 1):
        code = int(codes[s])
        if code == PATTERN_NONE:
            continue
        entry = stats.setdefault(
            code, {"wins": 0, "sample_size": 0, "name": pattern_name(code)}
        )
        bias = directions[s]
        move = close[s + h] - close[s]
        entry["sample_size"] += 1
        if bias != 0.0 and np.sign(move) == np.sign(bias):
            entry["wins"] += 1

    for code, entry in stats.items():
        directional = PATTERN_DIRECTION.get(code, 0.0) != 0.0
        ss = entry["sample_size"]
        entry["hit_rate"] = (entry["wins"] / ss) if (directional and ss > 0) else float("nan")
    return stats

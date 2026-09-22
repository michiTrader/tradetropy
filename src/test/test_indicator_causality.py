"""
Causality test (prefix-stability) for indicators.

CONTRACT BEING VERIFIED
───────────────────────
The backtest precalculates each indicator ONCE with calculate() over the
entire closed candle series and then, at each tick, slices that matrix with
`matrix[:n_closed]` (see OhlcIndicatorView / WindowView in data/data.py).

For that slice to NOT leak future information, calculate() must be
**causal**: the value at position i may depend only on source[:i+1],
never on source[i+1:].

Operational form of causality (prefix-stability):

    calculate(series[:k])[i]  ==  calculate(series)[i]    for all i < k

In other words: adding future data must NEVER change a value already
calculated in the past. If they differ, the indicator "looks into the
future" and produces lookahead bias in backtest, optimize and pool.

SCOPE
─────
Non-stateful indicators are tested (use_partial=True), where causality
is the responsibility of calculate itself: SMA, EMA, MACD, RSI,
ATR, BollingerBands.

ta/structure indicators (ConfirmedPivot, NBS, HHLL, ZigZag...) are
deliberately non-causal in bulk (they confirm a pivot using subsequent
bars). Their anti-lookahead protection does NOT live in calculate but in
the `use_partial` branch of OhlcIndicatorView, which applies a +1 offset
to anchor the pivot to the bar where it was actually confirmable. Those
are covered by engine parity tests, not here.
"""

import numpy as np
import pytest

from tradetropy.ta import (
    SMA, EMA, MACD, RSI, ATR, BollingerBands,
)


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA — various regimes to stress indicators
# ══════════════════════════════════════════════════════════════════════════════

def _price_series(n: int = 400, seed: int = 7) -> np.ndarray:
    """Random walk with drift + noise. 1D, serves as price source."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n) * 5.0 + 0.05
    return (50_000.0 + np.cumsum(steps)).astype(np.float64)


def _series_with_reversals(n: int = 400, seed: int = 11) -> np.ndarray:
    """Series with alternating trends and spikes, to stress EMA/RSI/MACD."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wave = 200.0 * np.sin(t / 25.0) + 80.0 * np.sin(t / 7.0)
    noise = rng.standard_normal(n) * 8.0
    return (50_000.0 + wave + np.cumsum(rng.standard_normal(n)) + noise).astype(np.float64)


def _hlc_series(n: int = 400, seed: int = 23) -> np.ndarray:
    """2D source [N×3] = (high, low, close) for ATR and other multi-source indicators."""
    close = _price_series(n, seed)
    rng = np.random.default_rng(seed + 1)
    rng_range = rng.uniform(2.0, 20.0, n)
    high = close + rng_range * rng.uniform(0.0, 1.0, n)
    low = close - rng_range * rng.uniform(0.0, 1.0, n)
    return np.column_stack([high, low, close]).astype(np.float64)


def _hlcv_series(n: int = 400, seed: int = 31) -> np.ndarray:
    """2D source [N×2] = (close, volume) for OBV."""
    close = _price_series(n, seed)
    rng = np.random.default_rng(seed + 2)
    volume = rng.uniform(100.0, 1000.0, n)
    return np.column_stack([close, volume]).astype(np.float64)


def _ohlcv_ts_series(n: int = 400, seed: int = 37) -> np.ndarray:
    """2D source [N×6] = (ts, open, high, low, close, volume) for VWAP."""
    rng = np.random.default_rng(seed)
    base_ts = 1_700_000_000_000
    ts = np.arange(n, dtype=np.float64) * 60_000 + base_ts
    close = _price_series(n, seed)
    rng2 = np.random.default_rng(seed + 3)
    rng_range = rng2.uniform(2.0, 15.0, n)
    high = close + rng_range * rng2.uniform(0.0, 1.0, n)
    low = close - rng_range * rng2.uniform(0.0, 1.0, n)
    op = close + rng2.uniform(-5.0, 5.0, n)
    vol = rng2.uniform(100.0, 1000.0, n)
    return np.column_stack([ts, op, high, low, close, vol]).astype(np.float64)


def _hlcv_mfi_series(n: int = 400, seed: int = 41) -> np.ndarray:
    """2D source [N×4] = (high, low, close, volume) for MFI."""
    hlc = _hlc_series(n, seed)
    rng = np.random.default_rng(seed + 4)
    volume = rng.uniform(100.0, 1000.0, n).reshape(-1, 1)
    return np.column_stack([hlc, volume]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION CORE
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_output(out: np.ndarray) -> np.ndarray:
    """calculate returns [N] (single-band) or [B×N] (multi-band).
    Always normalizes to 2D [B×N] for uniform band iteration."""
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 1:
        return out.reshape(1, -1)
    return out


def assert_causal(indicator, source: np.ndarray, *, rtol=1e-9, atol=1e-9,
                  exclude_bands: list[int] | None = None):
    """
    Verify prefix-stability: for a grid of cuts k, check that
    calculate(source[:k]) matches calculate(source)[:k] at every
    non-NaN position. Any discrepancy = lookahead.

    exclude_bands: bands to exclude from verification (e.g. chikou in Ichimoku
    which by design looks into the future).
    """
    exclude_bands = set(exclude_bands or [])
    n = len(source)
    full = _normalize_output(indicator.calculate(source))
    n_bands = full.shape[0]

    mp = getattr(indicator, "min_periods", 1)
    # Cuts: from the first potentially valid point to N, dense at
    # the start (where recursive seeds are) and sampled afterwards.
    ks = sorted(set(
        list(range(mp, min(mp + 15, n + 1)))
        + list(range(mp, n + 1, max(1, (n - mp) // 25)))
        + [n]
    ))

    for k in ks:
        pref = _normalize_output(indicator.calculate(source[:k]))
        assert pref.shape == (n_bands, k), (
            f"{type(indicator).__name__}: calculate(source[:{k}]) changed shape "
            f"to {pref.shape}, expected {(n_bands, k)}"
        )

        full_k = full[:, :k]
        # Positions where the full calculation already has an established value.
        valid = ~np.isnan(full_k)
        if not valid.any():
            continue

        # Exclude specified bands
        for b in exclude_bands:
            if b < n_bands:
                valid[b, :] = False

        # 1) Value causality: where full has a value, the prefix must
        #    have EXACTLY the same (within numerical tolerance).
        diff_ok = np.isclose(pref, full_k, rtol=rtol, atol=atol, equal_nan=False)
        bad = valid & ~diff_ok
        if bad.any():
            b, i = [int(x[0]) for x in np.where(bad)]
            raise AssertionError(
                f"LOOKAHEAD in {type(indicator).__name__} "
                f"(band {b}, index {i}, cut k={k}): "
                f"prefix={pref[b, i]!r} vs full={full_k[b, i]!r}. "
                f"Adding future data changed a past value → looks into the future."
            )

        # 2) NaN pattern: if the full calculation already validated position i
        #    with data <= i, the prefix (which contains those same data) must
        #    also have validated it. An extra NaN in the prefix would indicate
        #    the indicator needed future data to establish that value.
        nan_extra = valid & np.isnan(pref)
        if nan_extra.any():
            b, i = [int(x[0]) for x in np.where(nan_extra)]
            raise AssertionError(
                f"LOOKAHEAD (NaN) in {type(indicator).__name__} "
                f"(band {b}, index {i}, cut k={k}): the full calculation "
                f"establishes a value that the prefix leaves NaN → needed future."
            )


# ══════════════════════════════════════════════════════════════════════════════
# CASE MATRIX — (id, factory, source)
# ══════════════════════════════════════════════════════════════════════════════

_PRICE_SERIES = _price_series()
_REVERSALS_SERIES = _series_with_reversals()
_HLC = _hlc_series()
_HLCV = _hlcv_series()
_OHLCV_TS = _ohlcv_ts_series()

_CASES = [
    # SMA
    ("SMA(10)/walk", lambda: SMA(10), _PRICE_SERIES),
    ("SMA(50)/giros", lambda: SMA(50), _REVERSALS_SERIES),
    # EMA (recursive with SMA seed)
    ("EMA(10)/walk", lambda: EMA(10), _PRICE_SERIES),
    ("EMA(20)/giros", lambda: EMA(20), _REVERSALS_SERIES),
    # RSI (Wilder, recursive)
    ("RSI(14)/walk", lambda: RSI(14), _PRICE_SERIES),
    ("RSI(14)/giros", lambda: RSI(14), _REVERSALS_SERIES),
    ("RSI(7)/giros", lambda: RSI(7), _REVERSALS_SERIES),
    # MACD (multi-band, double EMA + signal EMA)
    ("MACD(12,26,9)/walk", lambda: MACD(12, 26, 9), _PRICE_SERIES),
    ("MACD(5,13,4)/giros", lambda: MACD(5, 13, 4), _REVERSALS_SERIES),
    # Bollinger (multi-band)
    ("BB(20,2)/walk", lambda: BollingerBands(20, 2.0), _PRICE_SERIES),
    ("BB(14,2.5)/giros", lambda: BollingerBands(14, 2.5), _REVERSALS_SERIES),
    # ATR (multi-source HLC, recursive)
    ("ATR(14)/hlc", lambda: ATR(14), _HLC),
    ("ATR(7)/hlc", lambda: ATR(7), _HLC),
]


def _extract_case_params(case):
    if len(case) == 4:
        return case[0], case[1], case[2], case[3]
    return case[0], case[1], case[2], None


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,factory,source,exclude_bands",
    [_extract_case_params(c) for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_indicator_is_causal(name, factory, source, exclude_bands):
    """Each non-stateful indicator must be prefix-stable (no lookahead)."""
    assert_causal(factory(), source, exclude_bands=exclude_bands)


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL TEST — a deliberately non-causal indicator MUST fail
# ══════════════════════════════════════════════════════════════════════════════

class _DeliberateLookahead(SMA):
    """Centered SMA: uses source[i+1] → non-causal. Used to verify that
    assert_causal actually detects lookahead (test of the test)."""

    def calculate(self, source: np.ndarray) -> np.ndarray:
        # Centered 3-point average: (x[i-1] + x[i] + x[i+1]) / 3 → looks into the future.
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n - 1):
            out[i] = (source[i - 1] + source[i] + source[i + 1]) / 3.0
        return out


@pytest.mark.unit
def test_assert_causal_detects_lookahead():
    """Meta-test: assert_causal must catch a non-causal indicator."""
    ind = _DeliberateLookahead(3)
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        assert_causal(ind, _PRICE_SERIES)

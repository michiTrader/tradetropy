"""
Rolling-mean helpers must survive a NaN warmup prefix (Stochastic, RVI).

Two related defects lived in the private ``_sma`` helpers that smooth an
intermediate series which legitimately starts with NaN:

1. **Stochastic - all-NaN output.** ``_sma`` used ``np.cumsum`` over ``raw_k``,
   whose first ``k_period - 1`` entries are NaN warmup. cumsum propagates the
   first NaN to every later element, so %K and %D were NaN over the *entire*
   series, on any input. The indicator never produced a single usable value.

2. **RVI - silent corruption.** ``_sma`` used ``np.nancumsum``, which dodges
   the all-NaN result but treats NaN as 0. A window straddling the warmup then
   divides a partial sum by the full ``L``, returning a plausible-looking
   number built from fewer observations than requested - e.g. a 3-bar mean of
   ``[NaN, NaN, 30]`` reported ``10.0`` instead of NaN. That is worse than the
   first bug: NaN is visible, a wrong number is not.

Both now accumulate NaN-as-zero alongside a validity counter and emit a value
only where the window is completely populated.

Usage:
    pytest src/test/test_stochastic_nan_warmup.py -v
"""

import numpy as np
import pytest

from tradetropy.ta import RVI, Stochastic


def _series(n=400, seed=3):
    """Synthetic OHLC with a real high/low envelope."""
    base = np.cumsum(np.random.default_rng(seed).normal(0, 1, n)) + 100.0
    return base


def _hlc(n=400, seed=3):
    base = _series(n, seed)
    return np.column_stack([base + 2.0, base - 2.0, base])


def _ohlcv(n=400, seed=3):
    base = _series(n, seed)
    vol = np.abs(np.random.default_rng(seed + 1).normal(100, 10, n))
    return np.column_stack([base - 0.5, base + 2.0, base - 2.0, base, vol])


# =====
# THE HELPER ITSELF
# =====

@pytest.mark.unit
class TestSmaWarmupHelper:
    @pytest.mark.parametrize(
        'indicator', [Stochastic(), RVI()], ids=['Stochastic', 'RVI']
    )
    def test_nan_prefix_does_not_poison_tail(self, indicator):
        """One leading NaN must not blank the rest of the series."""
        arr = np.array([np.nan] * 3 + [30.0, 60.0, 90.0, 120.0])
        out = indicator._sma(arr, 3)
        assert np.isfinite(out[-1]), "a plain cumsum would make this NaN"
        assert np.isclose(out[-1], (60.0 + 90.0 + 120.0) / 3.0)

    @pytest.mark.parametrize(
        'indicator', [Stochastic(), RVI()], ids=['Stochastic', 'RVI']
    )
    def test_partial_window_is_nan_not_a_wrong_number(self, indicator):
        """
        Windows overlapping the warmup must be NaN. np.nancumsum would report
        10.0 and 30.0 here by counting NaN as zero.
        """
        arr = np.array([np.nan] * 3 + [30.0, 60.0, 90.0, 120.0])
        out = indicator._sma(arr, 3)
        assert np.isnan(out[3]), f"partial window leaked value {out[3]}"
        assert np.isnan(out[4]), f"partial window leaked value {out[4]}"
        assert np.isfinite(out[5]), "first complete window must emit"

    @pytest.mark.parametrize(
        'indicator', [Stochastic(), RVI()], ids=['Stochastic', 'RVI']
    )
    def test_matches_naive_rolling_mean(self, indicator):
        """Equivalent to an explicit loop that skips incomplete windows."""
        rng = np.random.default_rng(7)
        arr = np.concatenate([np.full(5, np.nan), rng.normal(50, 10, 120)])
        L = 4
        got = indicator._sma(arr, L)

        want = np.full(len(arr), np.nan)
        for i in range(L - 1, len(arr)):
            window = arr[i - L + 1:i + 1]
            if np.all(np.isfinite(window)):
                want[i] = window.mean()

        assert np.array_equal(np.isnan(got), np.isnan(want))
        mask = ~np.isnan(want)
        assert np.allclose(got[mask], want[mask])

    @pytest.mark.parametrize(
        'indicator', [Stochastic(), RVI()], ids=['Stochastic', 'RVI']
    )
    def test_degenerate_inputs(self, indicator):
        assert indicator._sma(np.array([]), 3).size == 0
        assert np.isnan(indicator._sma(np.array([1.0, 2.0]), 5)).all()
        assert np.isnan(indicator._sma(np.full(10, np.nan), 3)).all()


# =====
# STOCHASTIC END TO END
# =====

@pytest.mark.unit
class TestStochasticOutput:
    def test_produces_finite_values(self):
        """The headline regression: output was 100% NaN on every input."""
        out = Stochastic().calculate(_hlc())
        assert int(np.sum(~np.isnan(out[0]))) > 300, "%K is still empty"
        assert int(np.sum(~np.isnan(out[1]))) > 300, "%D is still empty"

    def test_bounded_zero_to_hundred(self):
        out = Stochastic().calculate(_hlc())
        k = out[0][~np.isnan(out[0])]
        assert k.min() >= 0.0 and k.max() <= 100.0

    def test_warmup_matches_min_periods(self):
        """First finite %D must land exactly on min_periods - 1."""
        s = Stochastic()
        out = s.calculate(_hlc())
        first_d = int(np.argmax(~np.isnan(out[1])))
        assert first_d == s.min_periods - 1
        assert np.isnan(out[1][:first_d]).all()

    def test_matches_independent_implementation(self):
        """Cross-check %K against a from-scratch reference calculation."""
        src = _hlc()
        s = Stochastic()
        n = len(src)
        L, ks = s.k_period, s.k_smooth

        raw = np.full(n, np.nan)
        for i in range(L - 1, n):
            hh = np.max(src[i - L + 1:i + 1, 0])
            ll = np.min(src[i - L + 1:i + 1, 1])
            if hh != ll:
                raw[i] = 100.0 * (src[i, 2] - ll) / (hh - ll)

        want = np.full(n, np.nan)
        for i in range(ks - 1, n):
            window = raw[i - ks + 1:i + 1]
            if np.all(np.isfinite(window)):
                want[i] = window.mean()

        got = s.calculate(src)[0]
        mask = ~np.isnan(want)
        assert mask.sum() > 300
        assert np.allclose(got[mask], want[mask])

    def test_flat_market_stays_nan(self):
        """hh == ll leaves raw_k undefined; it must not fabricate a value."""
        flat = np.column_stack([np.full(60, 10.0)] * 3)
        out = Stochastic().calculate(flat)
        assert np.isnan(out[0]).all()

    def test_causality_prefix_invariance(self):
        """Recomputing over a prefix must reproduce the same values."""
        src = _hlc()
        full = Stochastic().calculate(src)
        for k in (60, 150, 300):
            pref = Stochastic().calculate(src[:k])
            a, b = full[:, :k], pref[:, :k]
            assert np.array_equal(np.isnan(a), np.isnan(b))
            mask = ~np.isnan(a)
            assert np.allclose(a[mask], b[mask])

    def test_empty_input(self):
        out = Stochastic().calculate(np.empty((0, 3)))
        assert out.shape == (2, 0)


# =====
# RVI REGRESSION
# =====

@pytest.mark.unit
class TestRviStillWorks:
    def test_produces_finite_values(self):
        out = np.atleast_2d(RVI().calculate(_ohlcv()))
        assert int(np.sum(~np.isnan(out))) > 300

    def test_warmup_no_longer_leaks_partial_windows(self):
        """
        With nancumsum, RVI emitted values before its warmup completed. The
        first finite output must now respect min_periods.
        """
        r = RVI()
        out = np.atleast_2d(r.calculate(_ohlcv()))
        first = int(np.argmax(~np.isnan(out[0])))
        assert first >= r.length, (
            f"first finite value at {first}, before the {r.length}-bar window"
        )

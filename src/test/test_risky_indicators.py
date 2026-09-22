"""
Unit tests for the riskiest indicators: KAMA, TSI,
SchaffTrendCycle, StochasticRSI, EMV.

Verifies:
- Correct output shape
- Expected output range
- Causality (prefix-stability)
- Consistency with manual calculations in simple cases
- That valid values appear reasonably early
"""

import numpy as np
import pytest

from tradetropy.ta import KAMA, TSI, SchaffTrendCycle, StochasticRSI, EMV


# =====
# SYNTHETIC DATA
# =====

def _price(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return (100.0 + np.cumsum(rng.standard_normal(n) * 0.5)).astype(np.float64)

def _hlc(n=300, seed=42):
    c = _price(n, seed)
    rng = np.random.default_rng(seed + 1)
    h = c + rng.uniform(1.0, 10.0, n)
    l = c - rng.uniform(1.0, 10.0, n)
    return np.column_stack([h, l, c])

def _hlc_vol(n=300, seed=42):
    hlc = _hlc(n, seed)
    rng = np.random.default_rng(seed + 2)
    v = rng.uniform(100.0, 1000.0, n).reshape(-1, 1)
    return np.column_stack([hlc, v])


# =====
# UTILITIES
# =====

def _assert_causal(indicator, source):
    """Verifies prefix-stability: past values do not change with future data."""
    n = len(source)
    full = indicator.calculate(source)
    mp = indicator.min_periods

    cuts = sorted(set(
        list(range(mp, min(mp + 15, n + 1)))
        + list(range(mp, n + 1, max(1, (n - mp) // 10)))
        + [n]
    ))

    full_2d = full.reshape(1, -1) if full.ndim == 1 else full

    for k in cuts:
        prefix = indicator.calculate(source[:k])
        prefix_2d = prefix.reshape(1, -1) if prefix.ndim == 1 else prefix

        full_k = full_2d[:, :k]
        valid = ~np.isnan(full_k)

        if not valid.any():
            continue

        bad = valid & ~np.isclose(prefix_2d, full_k, rtol=1e-7, atol=1e-10, equal_nan=False)

        if bad.any():
            idx = np.argwhere(bad)
            raise AssertionError(
                f"LOOKAHEAD in {type(indicator).__name__} "
                f"(cut k={k}, position {idx[0]}): "
                f"prefix={prefix_2d[tuple(idx[0])]:.6f} vs full={full_k[tuple(idx[0])]:.6f}"
            )


def _assert_has_valid_values(indicator, source, min_valid_fraction=0.3):
    """Verifies that there are valid values in the second half of the series."""
    out = indicator.calculate(source)
    n = len(source)
    second_half = out[..., n // 2:]
    valid = np.count_nonzero(~np.isnan(second_half))
    total = second_half.size
    assert valid > total * min_valid_fraction, (
        f"{type(indicator).__name__}: only {valid}/{total} valid values in second half"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: KAMA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestKAMA:
    def test_shape(self):
        src = _price(200)
        out = KAMA(10).calculate(src)
        assert out.shape == (200,)

    def test_first_value_nan(self):
        src = _price(200)
        out = KAMA(10).calculate(src)
        assert np.isnan(out[0])

    def test_has_valid_values(self):
        _assert_has_valid_values(KAMA(10), _price(200))

    def test_causal(self):
        _assert_causal(KAMA(10), _price(200))

    def test_kama_tracks_price(self):
        """KAMA in a linear trend should converge to the price."""
        n = 100
        src = np.linspace(100, 200, n, dtype=np.float64)
        out = KAMA(10).calculate(src)
        assert np.allclose(out[-20:], src[-20:], rtol=0.05)

    def test_kama_flat_in_range(self):
        """KAMA at a constant price should be the constant value."""
        n = 50
        src = np.full(n, 100.0, dtype=np.float64)
        out = KAMA(10).calculate(src)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 100.0, atol=0.01)

    def test_no_infs(self):
        src = _price(200)
        out = KAMA(10).calculate(src)
        assert not np.any(np.isinf(out))


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: TSI
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTSI:
    def test_shape(self):
        src = _price(300)
        out = TSI().calculate(src)
        assert out.shape == (2, 300)

    def test_first_band_has_valid_values(self):
        src = _price(300)
        out = TSI().calculate(src)
        assert np.sum(~np.isnan(out[0])) > 50

    def test_second_band_has_valid_values(self):
        src = _price(300)
        out = TSI().calculate(src)
        assert np.sum(~np.isnan(out[1])) > 30

    def test_causal(self):
        _assert_causal(TSI(), _price(300))

    def test_tsi_range(self):
        """TSI debe estar entre -100 y +100."""
        src = _price(300)
        out = TSI().calculate(src)
        valid = out[0, ~np.isnan(out[0])]
        assert np.all(valid >= -101)
        assert np.all(valid <= 101)

    def test_signal_band_is_smoothed(self):
        """The second band (signal) must have more NaN than the first."""
        src = _price(300)
        out = TSI().calculate(src)
        nan_tsi = np.sum(np.isnan(out[0]))
        nan_sig = np.sum(np.isnan(out[1]))
        assert nan_sig >= nan_tsi

    def test_no_infs(self):
        src = _price(300)
        out = TSI().calculate(src)
        assert not np.any(np.isinf(out))


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SchaffTrendCycle
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSchaffTrendCycle:
    def test_shape(self):
        src = _price(400)
        out = SchaffTrendCycle().calculate(src)
        assert out.shape == (400,)

    def test_has_valid_values(self):
        _assert_has_valid_values(SchaffTrendCycle(), _price(400))

    def test_causal(self):
        _assert_causal(SchaffTrendCycle(), _price(400))

    def test_stc_range(self):
        """STC debe estar entre ~0 y ~100."""
        src = _price(400)
        out = SchaffTrendCycle().calculate(src)
        valid = out[~np.isnan(out)]
        assert np.all(valid > -5)
        assert np.all(valid < 105)

    def test_stc_bounded_by_stochastic(self):
        """STC is essentially an iterative stochastic, must stay in [0,100]."""
        rng = np.random.default_rng(99)
        src = (100 + np.cumsum(rng.standard_normal(500) * 0.3)).astype(np.float64)
        out = SchaffTrendCycle(23, 50, 10).calculate(src)
        valid = out[~np.isnan(out)]
        assert np.percentile(valid, 1) > -2
        assert np.percentile(valid, 99) < 102

    def test_starts_at_50(self):
        """The first valid value of STC must be ~50 (initial stochastic)."""
        src = _price(400)
        out = SchaffTrendCycle().calculate(src)
        first_valid_idx = np.argmax(~np.isnan(out))
        assert abs(out[first_valid_idx] - 50.0) < 1.0

    def test_no_infs(self):
        src = _price(400)
        out = SchaffTrendCycle().calculate(src)
        assert not np.any(np.isinf(out))


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: StochasticRSI
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStochasticRSI:
    def test_shape(self):
        src = _price(300)
        out = StochasticRSI().calculate(src)
        assert out.shape == (2, 300)

    def test_k_has_valid_values(self):
        src = _price(300)
        out = StochasticRSI().calculate(src)
        assert np.sum(~np.isnan(out[0])) > 100

    def test_d_has_valid_values(self):
        src = _price(300)
        out = StochasticRSI().calculate(src)
        assert np.sum(~np.isnan(out[1])) > 50

    def test_causal(self):
        _assert_causal(StochasticRSI(), _price(300))

    def test_stochrsi_k_range(self):
        """La banda K debe estar entre 0 y 100."""
        src = _price(300)
        out = StochasticRSI().calculate(src)
        valid = out[0, ~np.isnan(out[0])]
        assert np.all(valid >= -1)
        assert np.all(valid <= 101)

    def test_d_is_smoothed_k(self):
        """The D band must be SMA of K, thus smoothed (more NaN)."""
        src = _price(300)
        out = StochasticRSI().calculate(src)
        nan_k = np.sum(np.isnan(out[0]))
        nan_d = np.sum(np.isnan(out[1]))
        assert nan_d >= nan_k

    def test_constant_price(self):
        """Precio constante no debe crashear."""
        n = 100
        src = np.full(n, 100.0, dtype=np.float64)
        out = StochasticRSI().calculate(src)
        assert out.shape == (2, n)

    def test_no_infs(self):
        src = _price(300)
        out = StochasticRSI().calculate(src)
        assert not np.any(np.isinf(out))


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: EMV
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEMV:
    def test_shape(self):
        data = _hlc_vol(200)
        out = EMV(14).calculate(data)
        assert out.shape == (200,)

    def test_has_valid_values(self):
        _assert_has_valid_values(EMV(14), _hlc_vol(200))

    def test_causal(self):
        _assert_causal(EMV(14), _hlc_vol(200))

    def test_emv_zero_range(self):
        """Cuando high == low, EMV debe ser 0 (no se mueve)."""
        n = 50
        c = np.linspace(100, 110, n)
        h = c.copy()
        l = c.copy()
        v = np.full(n, 1000.0)
        data = np.column_stack([h, l, c, v])
        out = EMV(5).calculate(data)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0, atol=0.01)

    def test_no_infs(self):
        data = _hlc_vol(200)
        out = EMV(14).calculate(data)
        assert not np.any(np.isinf(out))

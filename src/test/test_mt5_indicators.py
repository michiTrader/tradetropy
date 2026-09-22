"""
Unit tests for the additional MT5 indicators:

Phase 1 (trivial): Momentum, OsMA, BullsPower, BearsPower, StdDev, Envelopes
Phase 2 (Bill Williams): Alligator, GatorOscillator, AcceleratorOscillator,
    Fractals, MarketFacilitationIndex
Phase 3 (adaptive/oscillators): FRAMA, VIDYA, DeMarker, RVI
Phase 4 (levels): PivotPoints (classic|camarilla|woodie|demark)

Verifies shape, causality (prefix-stability), expected ranges,
valid values and consistency with manual calculations in simple cases.
"""

import numpy as np
import pytest

from tradetropy.ta import (
    Momentum, OsMA, BullsPower, BearsPower, StdDev, Envelopes,
    Alligator, GatorOscillator, AcceleratorOscillator, Fractals,
    MarketFacilitationIndex,
    FRAMA, VIDYA, DeMarker, RVI,
    PivotPoints,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS SINTETICOS
# ══════════════════════════════════════════════════════════════════════════════

def _price(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return (100.0 + np.cumsum(rng.standard_normal(n) * 0.5)).astype(np.float64)


def _hl(n=300, seed=42):
    c = _price(n, seed)
    rng = np.random.default_rng(seed + 1)
    h = c + rng.uniform(0.5, 5.0, n)
    l = c - rng.uniform(0.5, 5.0, n)
    return np.column_stack([h, l])


def _hlc(n=300, seed=42):
    c = _price(n, seed)
    rng = np.random.default_rng(seed + 1)
    h = c + rng.uniform(0.5, 5.0, n)
    l = c - rng.uniform(0.5, 5.0, n)
    return np.column_stack([h, l, c])


def _hl_ts(n=300, seed=42):
    hl = _hl(n, seed)
    ts = (1_700_000_000_000 + np.arange(n) * 3_600_000).astype(np.float64)
    return np.column_stack([hl, ts])


def _hlv(n=300, seed=42):
    hl = _hl(n, seed)
    rng = np.random.default_rng(seed + 2)
    v = rng.uniform(100.0, 1000.0, n)
    return np.column_stack([hl[:, 0], hl[:, 1], v])


def _ohlc(n=300, seed=42):
    c = _price(n, seed)
    rng = np.random.default_rng(seed + 1)
    o = c + rng.uniform(-2.0, 2.0, n)
    h = np.maximum(o, c) + rng.uniform(0.5, 5.0, n)
    l = np.minimum(o, c) - rng.uniform(0.5, 5.0, n)
    return np.column_stack([o, h, l, c])


def _ohlc_ts(n=300, seed=42):
    ohlc = _ohlc(n, seed)
    ts = (1_700_000_000_000 + np.arange(n) * 3_600_000).astype(np.float64)
    return np.column_stack([ts, ohlc])


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _assert_causal(indicator, source):
    """Verifies prefix-stability: past values don't change with future data."""
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

        bad = valid & ~np.isclose(prefix_2d, full_k, rtol=1e-7, atol=1e-9, equal_nan=False)
        if bad.any():
            idx = np.argwhere(bad)
            raise AssertionError(
                f"LOOKAHEAD en {type(indicator).__name__} "
                f"(corte k={k}, posicion {idx[0]}): "
                f"prefijo={prefix_2d[tuple(idx[0])]:.6f} vs full={full_k[tuple(idx[0])]:.6f}"
            )


def _assert_has_valid(indicator, source, frac=0.3):
    out = indicator.calculate(source)
    n = len(source)
    second = out[..., n // 2:]
    valid = np.count_nonzero(~np.isnan(second))
    assert valid > second.size * frac, (
        f"{type(indicator).__name__}: solo {valid}/{second.size} validos en 2a mitad"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMomentum:
    def test_shape(self):
        assert Momentum(14).calculate(_price(200)).shape == (200,)

    def test_causal(self):
        _assert_causal(Momentum(14), _price(200))

    def test_formula(self):
        src = _price(100)
        out = Momentum(10).calculate(src)
        i = 50
        assert np.isclose(out[i], src[i] / src[i - 10] * 100.0)

    def test_flat_price_is_100(self):
        src = np.full(50, 100.0)
        out = Momentum(10).calculate(src)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 100.0)


@pytest.mark.unit
class TestOsMA:
    def test_shape(self):
        assert OsMA().calculate(_price(300)).shape == (300,)

    def test_causal(self):
        _assert_causal(OsMA(), _price(300))

    def test_has_valid(self):
        _assert_has_valid(OsMA(), _price(300))

    def test_no_infs(self):
        assert not np.any(np.isinf(OsMA().calculate(_price(300))))


@pytest.mark.unit
class TestBullsBearsPower:
    def test_shapes(self):
        data = _hlc(200)
        assert BullsPower(13).calculate(data).shape == (200,)
        assert BearsPower(13).calculate(data).shape == (200,)

    def test_causal(self):
        _assert_causal(BullsPower(13), _hlc(200))
        _assert_causal(BearsPower(13), _hlc(200))

    def test_bulls_ge_bears(self):
        """Bulls (high - EMA) siempre >= Bears (low - EMA) porque high >= low."""
        data = _hlc(200)
        bulls = BullsPower(13).calculate(data)
        bears = BearsPower(13).calculate(data)
        v = ~np.isnan(bulls) & ~np.isnan(bears)
        assert np.all(bulls[v] >= bears[v] - 1e-9)


@pytest.mark.unit
class TestStdDev:
    def test_shape(self):
        assert StdDev(20).calculate(_price(200)).shape == (200,)

    def test_causal(self):
        _assert_causal(StdDev(20), _price(200))

    def test_matches_numpy(self):
        src = _price(100)
        out = StdDev(20).calculate(src)
        i = 80
        expected = np.std(src[i - 19:i + 1])
        assert np.isclose(out[i], expected, rtol=1e-9)

    def test_flat_is_zero(self):
        out = StdDev(10).calculate(np.full(50, 100.0))
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-9)


@pytest.mark.unit
class TestEnvelopes:
    def test_shape(self):
        assert Envelopes(20, 0.5).calculate(_price(200)).shape == (3, 200)

    def test_causal(self):
        _assert_causal(Envelopes(20, 0.5), _price(200))

    def test_band_ordering(self):
        out = Envelopes(20, 0.5).calculate(_price(200))
        v = ~np.isnan(out[1])
        assert np.all(out[0, v] >= out[1, v])
        assert np.all(out[1, v] >= out[2, v])

    def test_deviation_pct(self):
        src = _price(100)
        out = Envelopes(20, 1.0).calculate(src)
        v = ~np.isnan(out[1])
        np.testing.assert_allclose(out[0, v], out[1, v] * 1.01, rtol=1e-9)
        np.testing.assert_allclose(out[2, v], out[1, v] * 0.99, rtol=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 - BILL WILLIAMS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAlligator:
    def test_shape(self):
        assert Alligator().calculate(_hl(300)).shape == (3, 300)

    def test_causal(self):
        _assert_causal(Alligator(), _hl(300))

    def test_has_valid(self):
        _assert_has_valid(Alligator(), _hl(300))

    def test_no_infs(self):
        assert not np.any(np.isinf(np.nan_to_num(Alligator().calculate(_hl(300)))))


@pytest.mark.unit
class TestGatorOscillator:
    def test_shape(self):
        assert GatorOscillator().calculate(_hl(300)).shape == (2, 300)

    def test_causal(self):
        _assert_causal(GatorOscillator(), _hl(300))

    def test_signs(self):
        """upper >= 0 and lower <= 0 by construction."""
        out = GatorOscillator().calculate(_hl(300))
        vu = ~np.isnan(out[0])
        vl = ~np.isnan(out[1])
        assert np.all(out[0, vu] >= -1e-9)
        assert np.all(out[1, vl] <= 1e-9)


@pytest.mark.unit
class TestAcceleratorOscillator:
    def test_shape(self):
        assert AcceleratorOscillator().calculate(_hl(300)).shape == (300,)

    def test_causal(self):
        _assert_causal(AcceleratorOscillator(), _hl(300))

    def test_has_valid(self):
        _assert_has_valid(AcceleratorOscillator(), _hl(300))


@pytest.mark.unit
class TestFractals:
    def test_shape(self):
        assert Fractals().calculate(_hl_ts(200)).shape == (4, 200)

    def test_causal(self):
        _assert_causal(Fractals(), _hl_ts(200))

    def test_last_bars_nan(self):
        out = Fractals(n=2).calculate(_hl_ts(200))
        assert np.all(np.isnan(out[0, -2:]))
        assert np.all(np.isnan(out[1, -2:]))

    def test_detects_known_peak(self):
        """An isolated maximum at the center produces an up fractal at center+2."""
        n = 11
        high = np.array([1, 2, 3, 4, 10, 4, 3, 2, 1, 0, 0], dtype=np.float64)
        low = np.full(n, 0.0)
        ts = np.arange(n, dtype=np.float64)
        src = np.column_stack([high, low, ts])
        out = Fractals(n=2).calculate(src)
        # pico en idx 4 -> confirma en idx 6
        assert out[0, 6] == 10.0
        assert out[2, 6] == 4.0  # real ts of the center bar


@pytest.mark.unit
class TestMarketFacilitationIndex:
    def test_shape(self):
        assert MarketFacilitationIndex().calculate(_hlv(200)).shape == (200,)

    def test_formula(self):
        data = _hlv(100)
        out = MarketFacilitationIndex().calculate(data)
        i = 50
        expected = (data[i, 0] - data[i, 1]) / data[i, 2]
        assert np.isclose(out[i], expected)

    def test_zero_volume_nan(self):
        data = _hlv(50)
        data[10, 2] = 0.0
        out = MarketFacilitationIndex().calculate(data)
        assert np.isnan(out[10])


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 - ADAPTATIVOS / OSCILADORES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFRAMA:
    def test_shape(self):
        assert FRAMA(16).calculate(_price(200)).shape == (200,)

    def test_causal(self):
        _assert_causal(FRAMA(16), _price(200))

    def test_has_valid(self):
        _assert_has_valid(FRAMA(16), _price(200))

    def test_tracks_linear_trend(self):
        src = np.linspace(100, 200, 120, dtype=np.float64)
        out = FRAMA(16).calculate(src)
        assert np.allclose(out[-10:], src[-10:], rtol=0.1)

    def test_no_infs(self):
        assert not np.any(np.isinf(np.nan_to_num(FRAMA(16).calculate(_price(200)))))


@pytest.mark.unit
class TestVIDYA:
    def test_shape(self):
        assert VIDYA(9).calculate(_price(200)).shape == (200,)

    def test_causal(self):
        _assert_causal(VIDYA(9), _price(200))

    def test_has_valid(self):
        _assert_has_valid(VIDYA(9), _price(200))

    def test_flat_price(self):
        out = VIDYA(9).calculate(np.full(60, 100.0))
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 100.0, atol=1e-6)


@pytest.mark.unit
class TestDeMarker:
    def test_shape(self):
        assert DeMarker(14).calculate(_hl(200)).shape == (200,)

    def test_causal(self):
        _assert_causal(DeMarker(14), _hl(200))

    def test_range_0_1(self):
        out = DeMarker(14).calculate(_hl(300))
        valid = out[~np.isnan(out)]
        assert np.all(valid >= -1e-9)
        assert np.all(valid <= 1.0 + 1e-9)

    def test_uptrend_high(self):
        """In a monotonic uptrend DeMarker should tend toward 1."""
        n = 60
        c = np.linspace(100, 200, n)
        h = c + 1.0
        l = c - 1.0
        out = DeMarker(14).calculate(np.column_stack([h, l]))
        valid = out[~np.isnan(out)]
        assert valid[-1] > 0.9


@pytest.mark.unit
class TestRVI:
    def test_shape(self):
        assert RVI(10).calculate(_ohlc(300)).shape == (2, 300)

    def test_causal(self):
        _assert_causal(RVI(10), _ohlc(300))

    def test_has_valid(self):
        out = RVI(10).calculate(_ohlc(300))
        assert np.sum(~np.isnan(out[0])) > 100
        assert np.sum(~np.isnan(out[1])) > 50

    def test_signal_more_nan(self):
        out = RVI(10).calculate(_ohlc(300))
        assert np.sum(np.isnan(out[1])) >= np.sum(np.isnan(out[0]))


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 - PIVOT POINTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPivotPoints:
    def test_shapes_per_method(self):
        data = _ohlc_ts(300)
        assert PivotPoints('classic', '1d').calculate(data).shape == (7, 300)
        assert PivotPoints('woodie', '1d').calculate(data).shape == (5, 300)
        assert PivotPoints('camarilla', '1d').calculate(data).shape == (9, 300)
        assert PivotPoints('demark', '1d').calculate(data).shape == (3, 300)

    def test_causal(self):
        _assert_causal(PivotPoints('classic', '1d'), _ohlc_ts(300))
        _assert_causal(PivotPoints('camarilla', '1d'), _ohlc_ts(300))

    def test_first_period_nan(self):
        """The first period has no previous period -> NaN."""
        data = _ohlc_ts(300)
        out = PivotPoints('classic', '1d').calculate(data)
        assert np.isnan(out[0, 0])

    def test_classic_formula(self):
        data = _ohlc_ts(300)
        out = PivotPoints('classic', '1d').calculate(data)
        # Reconstruct the first closed period by hand.
        ts = data[:, 0].astype(np.int64)
        period_ms = 86_400_000
        pid = ts // period_ms
        first = pid[0]
        mask0 = pid == first
        h = data[mask0, 2].max()
        l = data[mask0, 3].min()
        c = data[mask0, 4][-1]
        pp = (h + l + c) / 3.0
        # First index of the second period.
        i2 = np.argmax(pid != first)
        assert np.isclose(out[0, i2], pp)
        assert np.isclose(out[1, i2], 2 * pp - l)   # r1
        assert np.isclose(out[4, i2], 2 * pp - h)   # s1

    def test_levels_constant_within_period(self):
        data = _ohlc_ts(300)
        out = PivotPoints('classic', '1d').calculate(data)
        ts = data[:, 0].astype(np.int64)
        pid = ts // 86_400_000
        # Take the third period (already has a reference) and check constancy.
        uniq = np.unique(pid)
        target = uniq[3]
        idxs = np.where(pid == target)[0]
        seg = out[0, idxs]
        assert np.allclose(seg, seg[0])

    def test_r_above_s(self):
        out = PivotPoints('classic', '1d').calculate(_ohlc_ts(300))
        v = ~np.isnan(out[0])
        assert np.all(out[1, v] >= out[0, v] - 1e-9)   # r1 >= pp
        assert np.all(out[0, v] >= out[4, v] - 1e-9)   # pp >= s1

    def test_bad_method_raises(self):
        from tradetropy.exceptions import ConfigError
        with pytest.raises(ConfigError):
            PivotPoints('nonexistent', '1d')

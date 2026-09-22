"""
Unit tests for session annotations in tradetropy.ta.annotations:

- MarketSessions (regression after the shared-helpers refactor)
- SessionLevels  (per-session OHLC: running + projected previous)
- KillZones      (ICT windows: range + active flag)

Verifies shape, causality (prefix-stability), block coherence per
occurrence, and that projected levels stay frozen until the next
occurrence.
"""

import numpy as np
import pytest

from tradetropy.ta.annotations import MarketSessions, SessionLevels, KillZones


# ══════════════════════════════════════════════════════════════════════════════
# DATOS SINTETICOS
# ══════════════════════════════════════════════════════════════════════════════

def _ohlc_ts(n=2000, interval_ms=60_000, seed=42, start_ts=1_700_000_000_000):
    """OHLC sintetico con ts en ms, un bar por minuto por defecto (~1.4 dias)."""
    rng = np.random.default_rng(seed)
    close = (100.0 + np.cumsum(rng.standard_normal(n) * 0.05)).astype(np.float64)
    open_ = close + rng.uniform(-0.2, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.5, n)
    ts = (start_ts + np.arange(n) * interval_ms).astype(np.float64)
    return np.column_stack([ts, open_, high, low, close])


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
        if k == 0:
            continue
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


# ══════════════════════════════════════════════════════════════════════════════
# MARKETSESSIONS (post-refactor regression)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMarketSessionsRegression:
    def test_shape_and_binary_values(self):
        data = _ohlc_ts()[:, [0]]
        ms = MarketSessions(sessions=["london", "new_york"])
        out = ms.calculate(data)
        assert out.shape == (2, len(data))
        assert set(np.unique(out)) <= {0.0, 1.0}

    def test_zonas_built(self):
        data = _ohlc_ts()[:, [0]]
        ms = MarketSessions(sessions=["london"])
        ms.calculate(data)
        assert len(ms._zonas) >= 1
        assert "color" in ms._zonas[0] and "ts_start" in ms._zonas[0]

    def test_midnight_crossing_session(self):
        """Sydney (22:00 -> 09:00 UTC) cruza medianoche y debe seguir marcando."""
        data = _ohlc_ts()[:, [0]]
        ms = MarketSessions(sessions=["sydney"])
        out = ms.calculate(data)
        assert out.sum() > 0


# ══════════════════════════════════════════════════════════════════════════════
# SESSIONLEVELS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionLevels:
    def test_shape(self):
        data = _ohlc_ts()
        sl = SessionLevels(sessions=["london", "new_york"])
        out = sl.calculate(data)
        assert out.shape == (14, len(data))  # 7 bands x 2 sessions

    def test_output_names(self):
        sl = SessionLevels(sessions=["london", "new_york"])
        expected = [
            f"{s}_{suf}"
            for s in ("london", "new_york")
            for suf in ("open", "high", "low", "prev_open", "prev_high",
                        "prev_low", "prev_close")
        ]
        assert sl.output_names == expected

    def test_causal(self):
        data = _ohlc_ts(n=1500)
        _assert_causal(SessionLevels(sessions=["london"]), data)

    def test_nan_outside_session_running_bands(self):
        data = _ohlc_ts()
        sl = SessionLevels(sessions=["london"])
        out = sl.calculate(data)
        ts = data[:, 0]
        minutes = (ts.astype(np.int64) // 60_000) % 1440
        outside = (minutes < 8 * 60) | (minutes >= 17 * 60)
        # running open/high/low (bands 0,1,2) must be NaN outside the session
        assert np.all(np.isnan(out[0, outside]))
        assert np.all(np.isnan(out[1, outside]))
        assert np.all(np.isnan(out[2, outside]))

    def test_running_high_is_cummax_within_occurrence(self):
        data = _ohlc_ts()
        sl = SessionLevels(sessions=["london"])
        out = sl.calculate(data)
        high = data[:, 2]
        ts = data[:, 0].astype(np.int64)
        minutes = (ts // 60_000) % 1440
        inside = (minutes >= 8 * 60) & (minutes < 17 * 60)

        # Find the first occurrence's index range and check cummax manually.
        idxs = np.where(inside)[0]
        # first contiguous block
        breaks = np.where(np.diff(idxs) > 1)[0]
        first_block_end = breaks[0] if len(breaks) else len(idxs) - 1
        block = idxs[: first_block_end + 1]

        running_high = out[1, block]
        manual_cummax = np.maximum.accumulate(high[block])
        np.testing.assert_allclose(running_high, manual_cummax)

    def test_prev_close_projected_and_constant_until_next_occurrence(self):
        data = _ohlc_ts(n=3000)
        sl = SessionLevels(sessions=["london"])
        out = sl.calculate(data)
        ts = data[:, 0].astype(np.int64)
        minutes = (ts // 60_000) % 1440
        inside = (minutes >= 8 * 60) & (minutes < 17 * 60)

        idxs = np.where(inside)[0]
        breaks = np.where(np.diff(idxs) > 1)[0]
        assert len(breaks) >= 1, "test data must span at least 2 occurrences"

        first_block_end_pos = breaks[0]
        first_block_last_idx = idxs[first_block_end_pos]
        second_block_start_idx = idxs[first_block_end_pos + 1]

        expected_prev_close = data[first_block_last_idx, 4]

        # Right after the first occurrence closes, prev_close should reflect it
        # and remain constant until the second occurrence starts.
        segment = out[6, first_block_last_idx + 1: second_block_start_idx]
        assert len(segment) > 0
        assert np.all(~np.isnan(segment))
        np.testing.assert_allclose(segment, expected_prev_close)

    def test_prev_bands_nan_before_first_occurrence_closes(self):
        data = _ohlc_ts()
        sl = SessionLevels(sessions=["london"])
        out = sl.calculate(data)
        # Very first bars (before London even opens the first time) must have
        # NaN prev_* bands.
        assert np.isnan(out[6, 0])

    def test_refs_builds_expected_columns(self):
        class _FakeProxy:
            ts_ref = "TS"
            open_ref = "O"
            high_ref = "H"
            low_ref = "L"
            close_ref = "C"

        refs = SessionLevels.refs(_FakeProxy())
        assert refs == ["TS", "O", "H", "L", "C"]

    def test_custom_session_dict(self):
        data = _ohlc_ts()
        sl = SessionLevels(sessions=[
            {"name": "asia", "start": 0, "end": 8, "color": "#FF6B35"},
        ])
        out = sl.calculate(data)
        assert sl.output_names[0] == "asia_open"
        assert out.shape == (7, len(data))

    def test_min_periods(self):
        assert SessionLevels().min_periods == 1

    def test_empty_source(self):
        sl = SessionLevels(sessions=["london"])
        out = sl.calculate(np.zeros((0, 5)))
        assert out.shape[1] == 1  # guarded degenerate case, no crash


# ══════════════════════════════════════════════════════════════════════════════
# KILLZONES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestKillZones:
    def test_default_windows_shape(self):
        data = _ohlc_ts()
        kz = KillZones()
        out = kz.calculate(data)
        assert out.shape == (12, len(data))  # 3 bands x 4 default zones

    def test_output_names_default(self):
        kz = KillZones()
        expected = [
            f"{z}_{suf}"
            for z in ("asian", "london_open", "ny_open", "london_close")
            for suf in ("high", "low", "active")
        ]
        assert kz.output_names == expected

    def test_causal(self):
        data = _ohlc_ts(n=1500)
        _assert_causal(KillZones(windows=["london_open"]), data)

    def test_active_matches_window_mask(self):
        data = _ohlc_ts()
        kz = KillZones(windows=["london_open"])
        out = kz.calculate(data)
        ts = data[:, 0].astype(np.int64)
        minutes = (ts // 60_000) % 1440
        expected_active = ((minutes >= 7 * 60) & (minutes < 10 * 60)).astype(np.float64)
        np.testing.assert_array_equal(out[2], expected_active)

    def test_high_low_frozen_after_window_closes(self):
        data = _ohlc_ts()
        kz = KillZones(windows=["london_open"])
        out = kz.calculate(data)
        ts = data[:, 0].astype(np.int64)
        minutes = (ts // 60_000) % 1440
        inside = (minutes >= 7 * 60) & (minutes < 10 * 60)

        idxs = np.where(inside)[0]
        breaks = np.where(np.diff(idxs) > 1)[0]
        assert len(breaks) >= 1, "test data must span at least 2 occurrences"

        first_block_last_idx = idxs[breaks[0]]
        second_block_start_idx = idxs[breaks[0] + 1]

        expected_high = np.max(data[idxs[: breaks[0] + 1], 2])
        segment = out[0, first_block_last_idx + 1: second_block_start_idx]
        assert len(segment) > 0
        np.testing.assert_allclose(segment, expected_high)

    def test_nan_before_first_occurrence(self):
        data = _ohlc_ts()
        kz = KillZones(windows=["london_open"])
        out = kz.calculate(data)
        assert np.isnan(out[0, 0])

    def test_refs_builds_expected_columns(self):
        class _FakeProxy:
            ts_ref = "TS"
            open_ref = "O"
            high_ref = "H"
            low_ref = "L"
            close_ref = "C"

        refs = KillZones.refs(_FakeProxy())
        assert refs == ["TS", "O", "H", "L", "C"]

    def test_custom_window_dict(self):
        data = _ohlc_ts()
        kz = KillZones(windows=[
            {"name": "silver_bullet", "start": 10, "end": 11, "color": "#FF0000"},
        ])
        out = kz.calculate(data)
        assert kz.output_names == [
            "silver_bullet_high", "silver_bullet_low", "silver_bullet_active",
        ]
        assert out.shape == (3, len(data))

    def test_min_periods(self):
        assert KillZones().min_periods == 1

    def test_empty_source(self):
        kz = KillZones(windows=["asian"])
        out = kz.calculate(np.zeros((0, 5)))
        assert out.shape[1] == 1


# ══════════════════════════════════════════════════════════════════════════════
# DRAW() PRIMITIVES - SessionLevels / KillZones
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionLevelsDraw:
    def test_draw_returns_rects_hlines_labels(self):
        from tradetropy.ta.draw import Rects, HLines, Labels

        data = _ohlc_ts(n=3000)
        sl = SessionLevels(sessions=["london"])
        sl.calculate(data)
        prims = sl.draw()

        kinds = [type(p) for p in prims]
        assert Rects in kinds
        assert HLines in kinds
        assert Labels in kinds

    def test_draw_empty_before_calculate(self):
        sl = SessionLevels(sessions=["london"])
        assert sl.draw() == []

    def test_rect_box_matches_occurrence(self):
        data = _ohlc_ts(n=3000)
        sl = SessionLevels(sessions=["london"])
        sl.calculate(data)
        prims = sl.draw()
        rects = prims[0]
        occ_boxes = sl._occurrences[0]
        assert len(rects.x0) == len(occ_boxes)
        assert rects.x0[0] == occ_boxes[0]["ts_start"]
        assert rects.y1[0] == occ_boxes[0]["high"]
        assert rects.y0[0] == occ_boxes[0]["low"]

    def test_build_data_columns_smoke(self):
        from tradetropy.ta.draw import Rects
        from tradetropy.plotting.render._tools import _build_data

        data = _ohlc_ts(n=3000)
        sl = SessionLevels(sessions=["london"])
        sl.calculate(data)
        prims = sl.draw()
        rects = [p for p in prims if isinstance(p, Rects)][0]
        cols = _build_data(Rects, [rects], theme=None)
        assert cols is not None
        assert cols["left"].dtype == np.dtype("datetime64[ms]")

    def test_renderer_creates_glyphs(self):
        from bokeh.plotting import figure
        from tradetropy.plotting.render._tools import render_tool_groups

        data = _ohlc_ts(n=3000)
        sl = SessionLevels(sessions=["london"])
        sl.calculate(data)
        prims = sl.draw()

        fig = figure(x_axis_type="datetime")
        before = len(fig.renderers)
        registry = render_tool_groups(
            fig, {"SessionLevels": prims}, theme=None,
            interval_ms=60_000, registry={},
        )
        assert len(fig.renderers) > before


@pytest.mark.unit
class TestKillZonesDraw:
    def test_draw_returns_rects_hlines_labels(self):
        from tradetropy.ta.draw import Rects, HLines, Labels

        data = _ohlc_ts(n=3000)
        kz = KillZones(windows=["london_open"])
        kz.calculate(data)
        prims = kz.draw()

        kinds = [type(p) for p in prims]
        assert Rects in kinds
        assert HLines in kinds
        assert Labels in kinds

    def test_draw_empty_before_calculate(self):
        kz = KillZones(windows=["london_open"])
        assert kz.draw() == []

    def test_rect_box_matches_occurrence(self):
        data = _ohlc_ts(n=3000)
        kz = KillZones(windows=["london_open"])
        kz.calculate(data)
        prims = kz.draw()
        rects = prims[0]
        occ_boxes = kz._occurrences[0]
        assert len(rects.x0) == len(occ_boxes)
        assert rects.x0[0] == occ_boxes[0]["ts_start"]
        assert rects.y1[0] == occ_boxes[0]["high"]

    def test_renderer_creates_glyphs(self):
        from bokeh.plotting import figure
        from tradetropy.plotting.render._tools import render_tool_groups

        data = _ohlc_ts(n=3000)
        kz = KillZones(windows=["london_open"])
        kz.calculate(data)
        prims = kz.draw()

        fig = figure(x_axis_type="datetime")
        before = len(fig.renderers)
        registry = render_tool_groups(
            fig, {"KillZones": prims}, theme=None,
            interval_ms=60_000, registry={},
        )
        assert len(fig.renderers) > before



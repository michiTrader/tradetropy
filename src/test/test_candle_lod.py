"""Tests for the candlestick level-of-detail (bucket min/max) decimation."""

import numpy as np

from tradetropy.plotting.sources import (
    build_ohlc_lod_sources,
    _aggregate_candle_view,
    _CANDLE_LOD_MAX_VISIBLE,
)


def _synthetic_ohlc(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ts = (np.arange(n) * 60_000).astype(np.float64)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.random(n) * 2
    low = np.minimum(open_, close) - rng.random(n) * 2
    vol = rng.random(n) * 10
    return np.column_stack([ts, open_, high, low, close, vol])


class TestCandleLOD:
    def test_large_dataset_is_capped_and_preserves_extremes(self):
        arr = _synthetic_ohlc(25_000)
        full, view = build_ohlc_lod_sources(arr, 60_000)
        vd = view.data

        # The drawn view is capped; the backing source keeps every candle.
        assert len(full.data["ts"]) == 25_000
        assert len(vd["ts"]) == _CANDLE_LOD_MAX_VISIBLE

        # Min/max bucketing keeps the true extremes so the Y-autoscale, which
        # scans the drawn source, still frames the panel correctly.
        assert np.isclose(np.max(vd["High"]), np.max(arr[:, 2]))
        assert np.isclose(np.min(vd["Low"]), np.min(arr[:, 3]))
        # Volume is summed per bucket, so the total is conserved.
        assert np.isclose(np.sum(vd["Volume"]), np.sum(arr[:, 5]))

        # The view carries the same columns the glyphs read.
        assert set(vd.keys()) == set(full.data.keys())
        # inc is the up/down factor used by factor_cmap ("0"/"1" strings).
        assert set(np.unique(vd["inc"])).issubset({"0", "1"})

    def test_small_dataset_is_passthrough(self):
        arr = _synthetic_ohlc(100)
        out = _aggregate_candle_view(
            {  # minimal dict shaped like build_ohlc_source output
                "ts": arr[:, 0].astype("datetime64[ms]"),
                "Open": arr[:, 1], "High": arr[:, 2], "Low": arr[:, 3],
                "Close": arr[:, 4], "Volume": arr[:, 5],
            },
            _CANDLE_LOD_MAX_VISIBLE,
        )
        # n <= max_visible -> no aggregation, every candle kept.
        assert len(out["ts"]) == 100

    def test_body_columns_consistent_with_open_close(self):
        arr = _synthetic_ohlc(20_000, seed=3)
        _, view = build_ohlc_lod_sources(arr, 60_000)
        vd = view.data
        o = np.asarray(vd["Open"]); c = np.asarray(vd["Close"])
        assert np.allclose(np.asarray(vd["top_body"]), np.maximum(o, c))
        assert np.allclose(np.asarray(vd["bottom_body"]), np.minimum(o, c))
        # High/Low envelope the body of every aggregated candle.
        assert np.all(np.asarray(vd["High"]) >= np.asarray(vd["top_body"]))
        assert np.all(np.asarray(vd["Low"]) <= np.asarray(vd["bottom_body"]))

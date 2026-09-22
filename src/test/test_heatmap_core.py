"""Tests for the pure heatmap core: build_heatmap_grid, resolve_persistence,
find_heatmap_walls and infer_price_tick."""

import numpy as np

from tradetropy.ta.order_flow._core import (
    build_heatmap_grid,
    find_heatmap_walls,
    heatmap_color_bounds,
    infer_price_tick,
    merge_heatmap_grids,
    resolve_persistence,
)


def _book(snapshots, levels):
    """Build a book_window dict from a list of per-snapshot dicts.

    Each snapshot: {'ts': int, 'bids': [(px, sz), ...], 'asks': [(px, sz), ...]}.
    """
    r = len(snapshots)
    ts = np.array([s["ts"] for s in snapshots], dtype=np.int64)
    bid_px = np.full((r, levels), np.nan)
    bid_sz = np.full((r, levels), np.nan)
    ask_px = np.full((r, levels), np.nan)
    ask_sz = np.full((r, levels), np.nan)
    for j, s in enumerate(snapshots):
        for i, (p, sz) in enumerate(s.get("bids", [])[:levels]):
            bid_px[j, i] = p
            bid_sz[j, i] = sz
        for i, (p, sz) in enumerate(s.get("asks", [])[:levels]):
            ask_px[j, i] = p
            ask_sz[j, i] = sz
    return {"ts": ts, "kind": np.zeros(r, dtype=np.int64),
            "bid_px": bid_px, "bid_sz": bid_sz,
            "ask_px": ask_px, "ask_sz": ask_sz, "levels": levels}


class TestBuildHeatmapGrid:
    def test_basic_grid_shape_and_placement(self):
        snaps = [{
            "ts": j * 1000,
            "bids": [(99.0, 5.0), (98.0, 3.0)],
            "asks": [(100.0, 6.0), (101.0, 4.0)],
        } for j in range(3)]
        grid = build_heatmap_grid(_book(snaps, 2), price_bucket=1.0)

        # 3 snapshots -> 3 columns (one per event by default).
        assert grid["bid"].shape[0] == 3
        assert grid["ask"].shape[0] == 3
        # Prices 98..101 on a 1.0 grid -> 4 buckets centered on 98..101.
        centers = grid["price_centers"]
        assert centers[0] == 98.0
        assert centers[-1] == 101.0
        assert grid["bid"].shape[1] == 4

        # Buckets are centered on grid multiples, so price 99 -> center 99.
        b99 = int(np.argmin(np.abs(centers - 99.0)))
        assert grid["bid"][0, b99] == 5.0
        a100 = int(np.argmin(np.abs(centers - 100.0)))
        assert grid["ask"][0, a100] == 6.0

    def test_price_bucket_aggregates_levels(self):
        # bucket of 2.0 merges 98 and 99 into one bid bucket (5 + 3 = 8).
        snaps = [{"ts": 0, "bids": [(99.0, 5.0), (98.0, 3.0)],
                  "asks": [(100.0, 6.0)]}]
        grid = build_heatmap_grid(_book(snaps, 2), price_bucket=2.0)
        centers = grid["price_centers"]
        # The bucket covering 98-99.
        idx = int(np.argmin(np.abs(centers - 99.0)))
        assert grid["bid"][0, idx] == 8.0

    def test_time_bucket_takes_peak(self):
        # Two snapshots 500ms apart, same 1s time bucket -> peak size kept.
        snaps = [
            {"ts": 0, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
            {"ts": 500, "bids": [(99.0, 12.0)], "asks": [(100.0, 2.0)]},
        ]
        grid = build_heatmap_grid(_book(snaps, 1), price_bucket=1.0,
                                  time_bucket_ms=1000)
        assert grid["bid"].shape[0] == 1          # merged into one column
        centers = grid["price_centers"]
        b99 = int(np.argmin(np.abs(centers - 99.0)))
        assert grid["bid"][0, b99] == 12.0        # peak, not last
        assert grid["col_left"][0] == 0
        assert grid["col_right"][0] == 1000

    def test_min_size_filter(self):
        snaps = [{"ts": 0, "bids": [(99.0, 2.0)], "asks": [(100.0, 50.0)]}]
        grid = build_heatmap_grid(_book(snaps, 1), price_bucket=1.0,
                                  min_size=10.0)
        # The 2.0 bid is filtered out; the 50.0 ask survives.
        assert grid["bid"].sum() == 0.0
        assert grid["ask"].sum() == 50.0

    def test_max_levels_limits_depth(self):
        snaps = [{"ts": 0,
                  "bids": [(99.0, 5.0), (98.0, 5.0), (97.0, 5.0)],
                  "asks": [(100.0, 5.0)]}]
        grid = build_heatmap_grid(_book(snaps, 3), price_bucket=1.0,
                                  max_levels=1)
        # Only the best bid level (99) is counted.
        assert grid["bid"].sum() == 5.0

    def test_empty_book(self):
        grid = build_heatmap_grid(_book([], 2), price_bucket=1.0)
        assert grid["bid"].shape == (0, 0)
        assert grid["col_ts"].size == 0


class TestResolvePersistence:
    def test_persistent_vs_pulled(self):
        # Bucket 0 rests across all 3 columns; bucket 1 only appears last.
        mat = np.array([
            [10.0, 0.0],
            [10.0, 0.0],
            [10.0, 7.0],
        ])
        col_ts = np.array([0, 1000, 2000], dtype=np.int64)
        pers = resolve_persistence(mat, col_ts)
        assert pers[0] == 2000       # rested from ts 0 to 2000
        assert pers[1] == 0          # only the last column -> run start == now

    def test_broken_run_counts_only_trailing(self):
        # A gap in the middle: only the trailing contiguous run counts.
        mat = np.array([[5.0], [0.0], [5.0], [5.0]])
        col_ts = np.array([0, 500, 1000, 1500], dtype=np.int64)
        pers = resolve_persistence(mat, col_ts)
        assert pers[0] == 500        # from ts 1000 to 1500


class TestFindHeatmapWalls:
    def _grid(self):
        # Ask wall at 101 (size 500) persists; small levels elsewhere.
        snaps = [{
            "ts": j * 1000,
            "bids": [(99.0, 5.0)],
            "asks": [(100.0, 5.0), (101.0, 500.0)],
        } for j in range(4)]
        return build_heatmap_grid(_book(snaps, 3), price_bucket=1.0)

    def test_wall_detected_with_persistence(self):
        walls = find_heatmap_walls(self._grid(), rel_multiple=5.0,
                                   persistence_ms=1000)
        assert 101.0 in list(walls["price"])
        k = list(walls["price"]).index(101.0)
        assert walls["side"][k] == -1
        assert walls["size"][k] == 500.0
        assert walls["persistence_ms"][k] == 3000

    def test_absolute_threshold(self):
        walls = find_heatmap_walls(self._grid(), min_size=100.0,
                                   rel_multiple=0.0)
        assert list(walls["price"]) == [101.0]

    def test_no_walls_when_uniform(self):
        snaps = [{"ts": j, "bids": [(99.0, 5.0)], "asks": [(100.0, 5.0)]}
                 for j in range(3)]
        grid = build_heatmap_grid(_book(snaps, 2), price_bucket=1.0)
        walls = find_heatmap_walls(grid, rel_multiple=5.0)
        assert len(walls["price"]) == 0


class TestInferPriceTick:
    def test_infer_from_spacing(self):
        snaps = [{"ts": 0, "bids": [(99.5, 5.0), (99.0, 5.0)],
                  "asks": [(100.0, 5.0), (100.5, 5.0)]}]
        assert infer_price_tick(_book(snaps, 2)) == 0.5

    def test_default_when_empty(self):
        assert infer_price_tick(_book([], 2), default=0.25) == 0.25


class TestHeatmapColorBounds:
    def test_causal_past_is_frozen(self):
        # A huge level appears only in the last column. The bounds of the
        # earlier columns must NOT change when it arrives (freeze the past).
        bid = np.array([[5.0], [6.0], [0.0]])
        ask = np.array([[4.0], [3.0], [500.0]])
        lo, hi = heatmap_color_bounds(bid, ask, pct=99.0)

        # Recompute over only the first two columns (the causal prefix).
        lo2, hi2 = heatmap_color_bounds(bid[:2], ask[:2], pct=99.0)
        assert np.allclose(hi[:2], hi2)
        assert np.allclose(lo[:2], lo2)
        # The late 500 lot lifts only the last column's hot end.
        assert hi[2] > hi[1]

    def test_expanding_only_uses_past(self):
        # hi is non-decreasing here because sizes only grow over time.
        bid = np.array([[10.0], [10.0], [10.0]])
        ask = np.array([[0.0], [50.0], [100.0]])
        _, hi = heatmap_color_bounds(bid, ask, pct=100.0)
        assert hi[0] == 10.0            # only 10 seen
        assert hi[1] == 50.0            # max(10, 50)
        assert hi[2] == 100.0           # max(10, 50, 100)

    def test_explicit_scale_is_constant(self):
        bid = np.array([[5.0], [500.0]])
        ask = np.array([[4.0], [3.0]])
        lo, hi = heatmap_color_bounds(bid, ask, scale=(2.0, 900.0))
        assert np.all(lo == 2.0)
        assert np.all(hi == 900.0)

    def test_empty_grid(self):
        lo, hi = heatmap_color_bounds(np.zeros((0, 0)), np.zeros((0, 0)))
        assert lo.size == 0 and hi.size == 0

    def test_empty_column_fallback_hi_one(self):
        # No liquidity anywhere -> hi falls back to 1.0 (never zero span).
        bid = np.array([[0.0], [0.0]])
        ask = np.array([[0.0], [0.0]])
        _, hi = heatmap_color_bounds(bid, ask)
        assert np.all(hi == 1.0)

    def test_pool_seed_reflects_pruned_history(self):
        # A seed pool (older, pruned sizes) participates in the percentile.
        bid = np.array([[10.0]])
        ask = np.array([[0.0]])
        _, hi_no = heatmap_color_bounds(bid, ask, pct=100.0)
        _, hi_seed = heatmap_color_bounds(
            bid, ask, pct=100.0, pool=np.array([999.0]))
        assert hi_no[0] == 10.0
        assert hi_seed[0] == 999.0


class TestMergeHeatmapGrids:
    def _grid(self, snaps, levels=2, bucket=1.0):
        return build_heatmap_grid(_book(snaps, levels), price_bucket=bucket)

    def test_append_only_keeps_old_columns(self):
        g1 = self._grid([
            {"ts": 1000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
            {"ts": 2000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
        ])
        # Second window has slid: it no longer contains ts 1000.
        g2 = self._grid([
            {"ts": 2000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
            {"ts": 3000, "bids": [(99.0, 7.0)], "asks": [(100.0, 6.0)]},
        ])
        merged = merge_heatmap_grids(g1, g2)
        assert list(merged["col_ts"]) == [1000, 2000, 3000]
        assert merged["_rewind"] is False

    def test_union_price_axis(self):
        # Price drifts up: the union axis must span both ranges.
        g1 = self._grid([{"ts": 1000, "bids": [(99.0, 5.0)],
                          "asks": [(100.0, 6.0)]}])
        g2 = self._grid([{"ts": 2000, "bids": [(101.0, 5.0)],
                          "asks": [(102.0, 6.0)]}])
        merged = merge_heatmap_grids(g1, g2)
        centers = merged["price_centers"]
        assert centers[0] == 99.0 and centers[-1] == 102.0
        # Old column keeps its bid at 99; new column its bid at 101.
        b99 = int(np.argmin(np.abs(centers - 99.0)))
        b101 = int(np.argmin(np.abs(centers - 101.0)))
        assert merged["bid"][0, b99] == 5.0
        assert merged["bid"][1, b101] == 5.0

    def test_rewind_resets(self):
        g1 = self._grid([{"ts": 5000, "bids": [(99.0, 5.0)],
                          "asks": [(100.0, 6.0)]}])
        g2 = self._grid([{"ts": 1000, "bids": [(99.0, 5.0)],
                          "asks": [(100.0, 6.0)]}])
        merged = merge_heatmap_grids(g1, g2)
        assert merged["_rewind"] is True
        assert list(merged["col_ts"]) == [1000]

    def test_frozen_colors_carried_and_not_repainted(self):
        g1 = self._grid([{"ts": 1000, "bids": [(99.0, 5.0)],
                          "asks": [(100.0, 6.0)]}])
        g1["color_lo"] = np.array([0.0])
        g1["color_hi"] = np.array([6.0])
        g2 = self._grid([
            {"ts": 1000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
            {"ts": 2000, "bids": [(99.0, 5.0)], "asks": [(100.0, 500.0)]},
        ])
        g2["color_lo"] = np.array([0.0, 0.0])
        g2["color_hi"] = np.array([6.0, 500.0])
        merged = merge_heatmap_grids(g1, g2)
        # Old column keeps its frozen hi (6.0), the appended one gets 500.
        assert list(merged["color_hi"]) == [6.0, 500.0]

    def test_max_history_columns_trims_oldest(self):
        g1 = self._grid([{"ts": 1000, "bids": [(99.0, 5.0)],
                          "asks": [(100.0, 6.0)]}])
        g2 = self._grid([
            {"ts": 2000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
            {"ts": 3000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
        ])
        merged = merge_heatmap_grids(g1, g2, max_history_columns=2)
        assert list(merged["col_ts"]) == [2000, 3000]

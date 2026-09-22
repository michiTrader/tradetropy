"""
Tests for the Heatmap causal color (Bookmap-style) and append-only grid
accumulation.

Two behaviors are covered:

- Draw calibration is causal per column: with ``color_scale='auto'`` a column's
  color is fixed by the liquidity seen up to its own time and is never repainted
  by liquidity that arrives later (the past is frozen).
- In live/replay the drawn grid is accumulated append-only, so columns that
  slide out of the book window are kept with their frozen colors; the same
  per-column calibration is produced by a single backtest pass (parity).
"""

import numpy as np

from tradetropy.ta.order_flow import Heatmap
from tradetropy.ta.order_flow._core import build_heatmap_grid, heatmap_color_bounds
from tradetropy.ta.order_flow.heatmap import _COLORMAPS, _interp_colormap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mk_grid(bid, ask, *, bucket=1.0, base=100.0, t0=1000, dt=1000):
    """Build a minimal grid dict (as build_heatmap_grid would) for draw tests."""
    bid = np.asarray(bid, dtype=np.float64)
    ask = np.asarray(ask, dtype=np.float64)
    t, p = bid.shape
    centers = base + np.arange(p, dtype=np.float64) * bucket
    edges = np.empty(p + 1, dtype=np.float64)
    edges[:-1] = centers - bucket / 2.0
    edges[-1] = centers[-1] + bucket / 2.0
    col_ts = (t0 + np.arange(t) * dt).astype(np.int64)
    return {
        "col_ts": col_ts, "col_left": col_ts.copy(),
        "col_right": (col_ts + dt).astype(np.int64),
        "price_centers": centers, "price_edges": edges,
        "bid": bid, "ask": ask, "levels": p, "price_bucket": bucket,
    }


def _book_window(snaps, levels):
    """book_window() dict from [{'ts', 'bids', 'asks'}] snapshots."""
    r = len(snaps)
    ts = np.array([s["ts"] for s in snaps], dtype=np.int64)
    bid_px = np.full((r, levels), np.nan)
    bid_sz = np.full((r, levels), np.nan)
    ask_px = np.full((r, levels), np.nan)
    ask_sz = np.full((r, levels), np.nan)
    for j, s in enumerate(snaps):
        for i, (p, sz) in enumerate(s.get("bids", [])[:levels]):
            bid_px[j, i] = p
            bid_sz[j, i] = sz
        for i, (p, sz) in enumerate(s.get("asks", [])[:levels]):
            ask_px[j, i] = p
            ask_sz[j, i] = sz
    return {"ts": ts, "kind": np.zeros(r, dtype=np.int64),
            "bid_px": bid_px, "bid_sz": bid_sz,
            "ask_px": ask_px, "ask_sz": ask_sz, "levels": levels}


class _FakeBook:
    """Order book stub whose book_window() returns a sliding window of snaps."""

    def __init__(self, snaps, levels, window):
        self._snaps = snaps
        self._levels = levels
        self._window = window
        self.cursor = 0
        self.stale = False

    def book_window(self):
        lo = max(0, self.cursor - self._window + 1)
        return _book_window(self._snaps[lo:self.cursor + 1], self._levels)


class _Col:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, item):
        return self._arr[item]


class _FakeTickProxy:
    """Minimal tick proxy exposing [ts, price, volume, flags, bid, ask]."""

    def __init__(self, ts):
        ts = np.asarray(ts, dtype=np.float64)
        n = ts.size
        m = np.zeros((n, 6), dtype=np.float64)
        m[:, 0] = ts
        m[:, 1] = 100.0
        m[:, 2] = 1.0
        m[:, 3] = 1.0
        m[:, 4] = 100.0
        m[:, 5] = 100.0
        self._m = m

    def __len__(self):
        return len(self._m)

    def __getattr__(self, name):
        cols = {"ts": 0, "price": 1, "volume": 2, "flags": 3, "bid": 4, "ask": 5}
        if name in cols:
            return _Col(self._m[:, cols[name]])
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Task 2: causal color in the draw path
# ---------------------------------------------------------------------------
class TestCausalDrawColor:
    def test_past_colors_frozen_when_late_liquidity_arrives(self):
        h = Heatmap(orderbook=None)   # color_scale='auto' by default
        bid = np.array([[10.0], [10.0], [10.0]])
        ask = np.array([[0.0], [0.0], [500.0]])
        rects_full = h._liquidity_rects(_mk_grid(bid, ask))
        rects_pre = h._liquidity_rects(_mk_grid(bid[:2], ask[:2]))

        # The bid cells of columns 0 and 1 are emitted first (bid before ask);
        # their colors must be identical with or without the late 500 lot.
        assert rects_full.fill_color[:2] == rects_pre.fill_color[:2]

    def test_early_column_uses_its_own_hot_end(self):
        # Column 0 bid (10) is the max known at t0, so it is the hot end - NOT
        # dimmed by the 500 lot that only arrives later.
        h = Heatmap(orderbook=None)
        bid = np.array([[10.0], [10.0], [10.0]])
        ask = np.array([[0.0], [0.0], [500.0]])
        rects = h._liquidity_rects(_mk_grid(bid, ask))
        top = _interp_colormap(np.array([1.0]), _COLORMAPS["hot"])[0]
        assert rects.fill_color[0] == top

    def test_global_auto_would_dim_the_past(self):
        # Contrast: the OLD global-auto (one range over the whole grid) would
        # color column 0 near the cold end. Confirm the causal color differs.
        bid = np.array([[10.0], [10.0], [10.0]])
        ask = np.array([[0.0], [0.0], [500.0]])
        cold = _interp_colormap(np.array([10.0 / 500.0]), _COLORMAPS["hot"])[0]
        top = _interp_colormap(np.array([1.0]), _COLORMAPS["hot"])[0]
        assert cold != top

    def test_explicit_scale_is_constant_across_columns(self):
        h = Heatmap(orderbook=None, color_scale=(0.0, 500.0))
        bid = np.array([[10.0], [10.0]])
        ask = np.array([[0.0], [500.0]])
        rects = h._liquidity_rects(_mk_grid(bid, ask))
        # Both column-0 and column-1 bids are 10 lots against a fixed 500 range,
        # so they share the identical (dim) color.
        assert rects.fill_color[0] == rects.fill_color[1]


# ---------------------------------------------------------------------------
# Task 3: append-only grid accumulation in live/replay
# ---------------------------------------------------------------------------
class TestGridAccumulation:
    def _snaps(self):
        return [
            {"ts": 1000, "bids": [(99.0, 6.0)], "asks": [(100.0, 6.0)]},
            {"ts": 2000, "bids": [(99.0, 6.0)], "asks": [(100.0, 6.0)]},
            {"ts": 3000, "bids": [(99.0, 6.0)], "asks": [(100.0, 500.0)]},
        ]

    def _replay(self, window=1, max_history=None):
        snaps = self._snaps()
        book = _FakeBook(snaps, levels=1, window=window)
        h = Heatmap(orderbook=book, price_bucket=1.0, show_bubbles=False,
                    max_history_columns=max_history)
        ts_seen = []
        for c in range(len(snaps)):
            book.cursor = c
            ts_seen.append(1000 * (c + 1))
            h.live_refresh(_FakeTickProxy(ts_seen))
        return h

    def test_dropped_columns_are_retained(self):
        # window=1 => each refresh only sees the latest snapshot; without
        # accumulation columns 1000/2000 would be lost.
        h = self._replay(window=1)
        assert list(h._grid_accum["col_ts"]) == [1000, 2000, 3000]

    def test_past_color_frozen_not_repainted(self):
        h = self._replay(window=1)
        # Column 0 was calibrated against {6} -> hi=6, NOT the later 500 lot.
        assert h._grid_accum["color_hi"][0] == 6.0
        # The last column's hot end lifts toward the 500 lot it can see.
        assert h._grid_accum["color_hi"][2] > 100.0

    def test_rewind_resets_accumulator(self):
        h = self._replay(window=1)
        assert list(h._grid_accum["col_ts"]) == [1000, 2000, 3000]
        # Replay restart: cursor goes back, older ts -> accumulator resets.
        h._book.cursor = 0
        h.live_refresh(_FakeTickProxy([500]))
        assert list(h._grid_accum["col_ts"]) == [1000]

    def test_max_history_columns_caps_retained(self):
        h = self._replay(window=1, max_history=2)
        assert list(h._grid_accum["col_ts"]) == [2000, 3000]

    def test_engine_calculate_does_not_clobber_drawn_grid(self):
        # The engine (_feed.py) re-runs calculate() every tick to feed the
        # best_bid/best_ask bands, overwriting self._grid with the window grid.
        # The drawn grid must come from the accumulator, so draw() keeps the
        # full frozen history even right after such an engine overwrite.
        h = self._replay(window=1)
        # Simulate the engine's per-tick recompute (window grid = latest snap).
        h._book.cursor = 2
        h.calculate(_FakeTickProxy([3000])._m)
        assert list(h._grid["col_ts"]) == [3000]            # engine window grid
        groups = h.draw()
        rects = groups.get("Heatmap", [None])[0]
        assert rects is not None
        # Three columns' worth of cells are drawn (full history), not just one.
        assert len(set(rects.x0)) == 3

    def test_backtest_does_not_accumulate_future(self):
        # A single calculate() pass (backtest) leaves self._grid as the window
        # grid with NO frozen colors (draw calibrates causally at draw time);
        # the accumulator is untouched.
        snaps = self._snaps()
        book = _FakeBook(snaps, levels=1, window=len(snaps))
        book.cursor = len(snaps) - 1
        h = Heatmap(orderbook=book, price_bucket=1.0, show_bubbles=False)
        h.calculate(_FakeTickProxy([1000, 2000, 3000])._m)
        assert "color_lo" not in h._grid          # not frozen by accumulation
        assert h._grid_accum == {}                # accumulator never filled


# ---------------------------------------------------------------------------
# Task 4: backtest single-pass == replay accumulated per-column colors
# ---------------------------------------------------------------------------
class TestColorParity:
    def _snaps(self):
        return [
            {"ts": 1000, "bids": [(99.0, 5.0)], "asks": [(100.0, 6.0)]},
            {"ts": 2000, "bids": [(99.0, 8.0)], "asks": [(100.0, 20.0)]},
            {"ts": 3000, "bids": [(99.0, 8.0)], "asks": [(100.0, 500.0)]},
            {"ts": 4000, "bids": [(99.0, 8.0)], "asks": [(100.0, 30.0)]},
        ]

    def _backtest_bounds(self):
        snaps = self._snaps()
        grid = build_heatmap_grid(_book_window(snaps, 1), price_bucket=1.0)
        return heatmap_color_bounds(grid["bid"], grid["ask"], pct=99.0)

    def _replay_bounds(self, window):
        snaps = self._snaps()
        book = _FakeBook(snaps, levels=1, window=window)
        h = Heatmap(orderbook=book, price_bucket=1.0, show_bubbles=False)
        ts_seen = []
        for c in range(len(snaps)):
            book.cursor = c
            ts_seen.append(1000 * (c + 1))
            h.live_refresh(_FakeTickProxy(ts_seen))
        return h._grid_accum["color_lo"], h._grid_accum["color_hi"]

    def test_replay_window1_matches_backtest(self):
        bt_lo, bt_hi = self._backtest_bounds()
        rp_lo, rp_hi = self._replay_bounds(window=1)
        assert np.allclose(rp_lo, bt_lo)
        assert np.allclose(rp_hi, bt_hi)

    def test_replay_window2_matches_backtest(self):
        bt_lo, bt_hi = self._backtest_bounds()
        rp_lo, rp_hi = self._replay_bounds(window=2)
        assert np.allclose(rp_lo, bt_lo)
        assert np.allclose(rp_hi, bt_hi)


# ---------------------------------------------------------------------------
# Time-bucketed parity: the whole DRAWN grid (sizes + colors) accumulated in
# replay must equal a single backtest pass, even when a time bucket spans
# several refreshes / a sliding window (regression for the examples/heatmap_l2
# time_bucket_ms=60_000 divergence).
# ---------------------------------------------------------------------------
class TestTimeBucketParity:
    TB = 2000

    def _snaps(self):
        # Three 2s time buckets, two snapshots each. The peak is placed in the
        # FIRST snapshot of each bucket so a "freeze at the last snapshot in the
        # window" bug would understate every column.
        return [
            {"ts": 1000, "bids": [(99.0, 40.0)], "asks": [(100.0, 50.0)]},
            {"ts": 1500, "bids": [(99.0, 7.0)], "asks": [(100.0, 10.0)]},
            {"ts": 3000, "bids": [(99.0, 30.0)], "asks": [(100.0, 500.0)]},
            {"ts": 3500, "bids": [(99.0, 9.0)], "asks": [(100.0, 20.0)]},
            {"ts": 5000, "bids": [(99.0, 25.0)], "asks": [(100.0, 30.0)]},
            {"ts": 5500, "bids": [(99.0, 8.0)], "asks": [(100.0, 8.0)]},
        ]

    def _backtest(self):
        grid = build_heatmap_grid(_book_window(self._snaps(), 1),
                                  price_bucket=1.0, time_bucket_ms=self.TB)
        lo, hi = heatmap_color_bounds(grid["bid"], grid["ask"], pct=99.0)
        return grid, lo, hi

    def _replay(self, window):
        snaps = self._snaps()
        book = _FakeBook(snaps, levels=1, window=window)
        h = Heatmap(orderbook=book, price_bucket=1.0, time_bucket_ms=self.TB,
                    show_bubbles=False)
        ts_seen = []
        for c in range(len(snaps)):
            book.cursor = c
            ts_seen.append(snaps[c]["ts"])
            h.live_refresh(_FakeTickProxy(ts_seen))
        return h._grid_accum

    def _assert_matches(self, window):
        grid, lo, hi = self._backtest()
        acc = self._replay(window)
        assert np.array_equal(acc["col_ts"], grid["col_ts"])
        assert np.allclose(acc["price_centers"], grid["price_centers"])
        assert np.allclose(acc["bid"], grid["bid"])
        assert np.allclose(acc["ask"], grid["ask"])
        assert np.allclose(acc["color_lo"], lo)
        assert np.allclose(acc["color_hi"], hi)

    def test_full_window_matches_backtest(self):
        # window_size >> bucket (the live default): every bucket is fully in the
        # window. This is the examples/heatmap_l2 case.
        self._assert_matches(window=len(self._snaps()))

    def test_sliding_window_matches_backtest(self):
        # A window that holds only two snapshots: a bucket's snapshots are never
        # all visible together AND a bucket straddles the window edge, yet the
        # element-wise MAX merge reconstructs the true per-bucket peak.
        self._assert_matches(window=2)

    def test_window_one_matches_backtest(self):
        # Worst case: the window shows a single snapshot at a time, so each
        # bucket peak must be folded purely across refreshes by the MAX merge.
        self._assert_matches(window=1)

"""Tests for the pure L2 book detectors (walls, reload, stop-run) and the
LiveBookRing.book_window() accessor."""

import numpy as np

from tradetropy.ta.order_flow._core import (
    detect_walls,
    detect_reload_l2,
    detect_stop_run_l2,
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
    return {
        "ts": ts, "kind": np.zeros(r, dtype=np.int64),
        "bid_px": bid_px, "bid_sz": bid_sz,
        "ask_px": ask_px, "ask_sz": ask_sz, "levels": levels,
    }


class TestDetectWalls:
    def test_persistent_wall(self):
        # A 500-lot ask wall at 101 persists across 3 snapshots; rest is ~5.
        snaps = []
        for j in range(3):
            snaps.append({
                "ts": j * 1000,
                "bids": [(99.0, 5.0), (98.0, 5.0)],
                "asks": [(100.0, 5.0), (101.0, 500.0)],
            })
        book = _book(snaps, levels=3)
        r = detect_walls(book, rel_multiple=5.0, persistence_ms=0)
        assert 101.0 in list(r["price"])
        k = list(r["price"]).index(101.0)
        assert r["side"][k] == -1            # ask side
        assert r["max_size"][k] == 500.0
        assert r["last_ts"][k] - r["first_ts"][k] == 2000

    def test_no_wall_uniform_book(self):
        snaps = [{"ts": j, "bids": [(99.0, 5.0)], "asks": [(100.0, 5.0)]}
                 for j in range(3)]
        r = detect_walls(_book(snaps, 2), rel_multiple=5.0)
        assert len(r["price"]) == 0

    def test_absolute_threshold(self):
        snaps = [{"ts": 0, "bids": [(99.0, 120.0)], "asks": [(100.0, 5.0)]}]
        r = detect_walls(_book(snaps, 2), min_volume=100.0, rel_multiple=0.0)
        assert list(r["price"]) == [99.0]
        assert r["side"][0] == 1


class TestDetectReload:
    def test_reload_detected(self):
        # Bid wall at 99 starts 100, gets eaten to 10, refills to 100 -> reload;
        # repeat to reach threshold 2.
        seq = [100.0, 10.0, 100.0, 8.0, 100.0]
        snaps = [{"ts": j * 500, "bids": [(99.0, sz)], "asks": [(100.0, 5.0)]}
                 for j, sz in enumerate(seq)]
        r = detect_reload_l2(_book(snaps, 2), drop_frac=0.5, recover_frac=0.8,
                             reload_threshold=2)
        assert len(r["ts"]) >= 1
        assert r["price"][0] == 99.0
        assert r["side"][0] == 1
        assert r["reloads"][-1] >= 2

    def test_no_reload_when_stable(self):
        snaps = [{"ts": j, "bids": [(99.0, 100.0)], "asks": [(100.0, 5.0)]}
                 for j in range(5)]
        r = detect_reload_l2(_book(snaps, 2), reload_threshold=2)
        assert len(r["ts"]) == 0


class TestDetectStopRun:
    def test_sweep_and_reversal(self):
        # Price sweeps up 100 -> 105 fast, then reverses back below 100.
        ts = np.array([0, 100, 200, 300, 400, 1500], dtype=np.int64)
        price = np.array([100.0, 102.0, 104.0, 105.0, 105.0, 99.0])
        r = detect_stop_run_l2(ts, price, tick_size=1.0, sweep_levels=3,
                               sweep_window_ms=1000, reversal_window_ms=3000)
        assert len(r["ts"]) == 1
        assert r["extreme_px"][0] == 105.0
        assert r["side"][0] == -1            # up-sweep then down reversal
        assert r["start_px"][0] == 100.0

    def test_no_reversal_no_detection(self):
        ts = np.array([0, 100, 200, 300], dtype=np.int64)
        price = np.array([100.0, 102.0, 104.0, 106.0])   # keeps going up
        r = detect_stop_run_l2(ts, price, tick_size=1.0, sweep_levels=3)
        assert len(r["ts"]) == 0


class TestBookWindow:
    def test_ring_book_window(self):
        from tradetropy.data._ring import LiveBookRing

        ring = LiveBookRing(window_size=10, levels=3)
        ring.apply_snapshot(1000, bids=[(99.0, 5.0), (98.0, 4.0)],
                            asks=[(100.0, 6.0), (101.0, 7.0)])
        ring.apply_snapshot(2000, bids=[(99.0, 8.0)], asks=[(100.0, 9.0)])
        w = ring.book_window()
        assert list(w["ts"]) == [1000, 2000]
        assert w["levels"] == 3
        assert w["bid_sz"][0, 0] == 5.0
        assert w["ask_px"][0, 1] == 101.0
        assert w["bid_sz"][1, 0] == 8.0

    def test_empty_window(self):
        from tradetropy.data._ring import LiveBookRing

        ring = LiveBookRing(window_size=5, levels=2)
        w = ring.book_window()
        assert len(w["ts"]) == 0
        assert w["bid_px"].shape == (0, 2)

    def test_detectors_consume_ring_window(self):
        from tradetropy.data._ring import LiveBookRing

        ring = LiveBookRing(window_size=10, levels=3)
        for j in range(3):
            ring.apply_snapshot(
                j * 1000,
                bids=[(99.0, 5.0)], asks=[(100.0, 5.0), (101.0, 400.0)],
            )
        walls = detect_walls(ring.book_window(), rel_multiple=5.0)
        assert 101.0 in list(walls["price"])

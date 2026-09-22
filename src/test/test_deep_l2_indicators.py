"""Tests for the L2 order-flow indicators (DeepWall, DeepReload, StopRun)."""

import numpy as np
import pytest

from tradetropy.ta.draw import HLines, Labels, Points, Segments
from tradetropy.ta.order_flow import DeepReload, DeepWall, StopRun


class _FakeBook:
    """Stand-in for OrderbookProxy exposing a fixed book_window()."""

    def __init__(self, window):
        self._w = window

    def book_window(self):
        return self._w


def _book(snapshots, levels):
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


def _ticks(ts, price):
    ts = np.asarray(ts, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    n = len(ts)
    vol = np.ones(n)
    flags = np.ones(n)
    return np.column_stack([ts, price, vol, flags, price - 0.5, price + 0.5])


class TestDeepWall:
    def _wall_book(self):
        snaps = [{
            "ts": j * 1000,
            "bids": [(99.0, 5.0)],
            "asks": [(100.0, 5.0), (101.0, 500.0)],
        } for j in range(4)]
        return _FakeBook(_book(snaps, levels=3))

    def test_calculate_and_draw(self):
        book = self._wall_book()
        ind = DeepWall(book, rel_multiple=5.0)
        # ticks spanning the book window
        m = _ticks(np.arange(0, 4000, 250), 100.0 + np.zeros(16))
        out = ind.calculate(m)
        assert out.shape == (3, len(m))
        # A wall at 101 on the ask side was found and anchored at a tick.
        assert 101.0 in list(ind._events["price"])
        assert np.isfinite(out[0]).any()
        prims = ind.draw(ind.plot_config, interval_ms=1000)
        assert any(isinstance(p, HLines) for p in prims)

    def test_n_outputs(self):
        assert DeepWall().n_outputs == 3

    def test_no_book_empty(self):
        ind = DeepWall(None)
        m = _ticks(np.arange(5) * 1000, np.full(5, 100.0))
        out = ind.calculate(m)
        assert out.shape == (3, 5)
        assert not np.isfinite(out).any()
        assert ind.draw(ind.plot_config) == []


class TestDeepReload:
    def _reload_book(self):
        seq = [100.0, 10.0, 100.0, 8.0, 100.0]
        snaps = [{"ts": j * 500, "bids": [(99.0, sz)], "asks": [(100.0, 5.0)]}
                 for j, sz in enumerate(seq)]
        return _FakeBook(_book(snaps, levels=2))

    def test_calculate_and_draw(self):
        book = self._reload_book()
        ind = DeepReload(book, drop_frac=0.5, recover_frac=0.8, reload_threshold=2)
        m = _ticks(np.arange(0, 2500, 250), np.full(10, 99.5))
        out = ind.calculate(m)
        assert out.shape == (3, len(m))
        assert len(ind._events["ts"]) >= 1
        prims = ind.draw(ind.plot_config, interval_ms=500)
        assert any(isinstance(p, Points) for p in prims)
        assert any(isinstance(p, Labels) for p in prims)

    def test_no_book_empty(self):
        ind = DeepReload(None)
        m = _ticks(np.arange(5) * 1000, np.full(5, 100.0))
        out = ind.calculate(m)
        assert not np.isfinite(out).any()


class TestStopRun:
    def test_calculate_and_draw(self):
        # Sweep up 100 -> 105 then reverse below 100.
        ts = np.array([0, 100, 200, 300, 400, 1500], dtype=np.float64)
        price = np.array([100.0, 102.0, 104.0, 105.0, 105.0, 99.0])
        ind = StopRun(tick_size=1.0, sweep_levels=3)
        out = ind.calculate(_ticks(ts, price))
        assert out.shape == (3, len(ts))
        assert len(ind._events["ts"]) == 1
        # anchored at reversal tick (last), extreme 105, side -1
        assert np.isfinite(out[0, -1])
        assert out[0, -1] == 105.0
        assert out[1, -1] == -1.0
        prims = ind.draw(ind.plot_config, interval_ms=1000)
        kinds = {type(p).__name__ for p in prims}
        assert "Segments" in kinds and "Points" in kinds

    def test_no_sweep_empty(self):
        ts = np.array([0, 100, 200, 300], dtype=np.float64)
        price = np.array([100.0, 102.0, 104.0, 106.0])
        ind = StopRun(tick_size=1.0, sweep_levels=3)
        out = ind.calculate(_ticks(ts, price))
        assert not np.isfinite(out).any()
        assert ind.draw(ind.plot_config) == []

    def test_n_outputs(self):
        assert StopRun().n_outputs == 3


class TestLiveRefresh:
    class _Col:
        def __init__(self, a):
            self._a = a

        def __getitem__(self, i):
            return self._a[i]

    class _FakeTickProxy:
        def __init__(self, m):
            self._m = m

        def __len__(self):
            return len(self._m)

        @property
        def ts(self):
            return TestLiveRefresh._Col(self._m[:, 0])

        @property
        def price(self):
            return TestLiveRefresh._Col(self._m[:, 1])

        @property
        def volume(self):
            return TestLiveRefresh._Col(self._m[:, 2])

        @property
        def flags(self):
            return TestLiveRefresh._Col(self._m[:, 3])

        @property
        def bid(self):
            return TestLiveRefresh._Col(self._m[:, 4])

        @property
        def ask(self):
            return TestLiveRefresh._Col(self._m[:, 5])

    def test_stoprun_live_refresh_matches(self):
        ts = np.array([0, 100, 200, 300, 400, 1500], dtype=np.float64)
        price = np.array([100.0, 102.0, 104.0, 105.0, 105.0, 99.0])
        m = _ticks(ts, price)
        ind = StopRun(tick_size=1.0, sweep_levels=3)
        ind.live_refresh(self._FakeTickProxy(m))
        assert len(ind._events["ts"]) == 1

    def test_empty_proxy_noop(self):
        ind = StopRun()
        ind.live_refresh(self._FakeTickProxy(_ticks([], [])))
        assert ind.draw(ind.plot_config) == []

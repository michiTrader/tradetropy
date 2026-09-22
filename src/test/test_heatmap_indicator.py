"""Tests for the Heatmap indicator: bands, public query API, draw() layers and
the MultiBandProxy query-provider delegation."""

import numpy as np
import pytest

from tradetropy.data._proxy import MultiBandProxy
from tradetropy.ta.draw import Points, Rects, Segments
from tradetropy.ta.order_flow import (
    Heatmap,
    HeatmapGrid,
    LiquidityColumn,
    LiquidityLevel,
)


class _FakeBook:
    """Stand-in for OrderbookProxy exposing a fixed book_window() and metrics."""

    def __init__(self, window, *, best_bid=99.0, best_ask=100.0, stale=False):
        self._w = window
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.mid = (best_bid + best_ask) / 2.0
        self.stale = stale

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


def _wall_book():
    # Ask wall at 101 (500 lots) persists; small levels elsewhere.
    snaps = [{
        "ts": j * 1000,
        "bids": [(99.0, 5.0), (98.0, 4.0)],
        "asks": [(100.0, 6.0), (101.0, 500.0)],
    } for j in range(4)]
    return _FakeBook(_book(snaps, levels=3), best_bid=99.0, best_ask=100.0)


class TestBandsAndShape:
    def test_n_outputs(self):
        assert Heatmap().n_outputs == 2

    def test_calculate_bands(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        m = _ticks(np.arange(0, 4000, 250), np.full(16, 99.5))
        out = ind.calculate(m)
        assert out.shape == (2, len(m))
        # Best bid / ask as-of the ticks (book best bid 99, ask 100).
        assert np.nanmax(out[0]) == 99.0
        assert np.nanmax(out[1]) == 100.0

    def test_no_book_empty(self):
        ind = Heatmap(None, price_bucket=1.0)
        out = ind.calculate(_ticks(np.arange(5) * 1000, np.full(5, 100.0)))
        assert out.shape == (2, 5)
        assert not np.isfinite(out).any()
        assert ind.draw(ind.plot_config) == {}


class TestQueryAPI:
    def test_liquidity_at_and_notional(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        assert ind.liquidity_at(101.0) == 500.0        # ask wall
        assert ind.liquidity_at(99.0) == 5.0           # bid level
        assert ind.notional_at(101.0) == 500.0 * 101.0
        # Empty but in-range bucket -> 0.0.
        assert ind.liquidity_at(100.0, side="bid") == 0.0

    def test_hottest(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        top = ind.hottest(n=1)
        assert len(top) == 1
        assert isinstance(top[0], LiquidityLevel)
        assert top[0].price == 101.0
        assert top[0].size == 500.0
        assert top[0].side == "ask"
        bid_top = ind.hottest(side="bid", n=1)
        assert bid_top[0].price == 99.0

    def test_walls_and_nearest(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0, wall_rel_multiple=5.0,
                      persistence_ms=1000)
        walls = ind.walls()
        assert any(w.price == 101.0 and w.side == "ask" for w in walls)
        nw = ind.nearest_wall("ask")
        assert nw is not None and nw.price == 101.0
        assert nw.persistence_ms == 3000
        # No bid wall of that size -> None.
        assert ind.nearest_wall("bid") is None

    def test_persistence(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        assert ind.persistence(101.0) == 3000          # rested 4 snaps (0..3000)
        assert ind.persistence(101.0, side="ask") == 3000

    def test_column_and_grid(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        col = ind.column()
        assert isinstance(col, LiquidityColumn)
        assert col.ts == 3000
        assert col.ask_size.sum() > 0

        grid = ind.grid()
        assert isinstance(grid, HeatmapGrid)
        assert grid.bid.shape[0] == 4                  # one column per snapshot
        assert grid.levels == 3

    def test_stale_returns_nan_empty(self):
        book = _FakeBook(_book([], 2), stale=True)
        ind = Heatmap(book, price_bucket=1.0)
        assert np.isnan(ind.liquidity_at(100.0))
        assert ind.hottest() == []
        assert ind.walls() == []
        assert ind.column() is None
        assert ind.grid() is None
        assert ind.stale is True

    def test_out_of_range_price(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        assert np.isnan(ind.liquidity_at(50.0))        # far outside grid


class TestDrawLayers:
    def _prepared(self, **kwargs):
        ind = Heatmap(_wall_book(), price_bucket=1.0, **kwargs)
        # Ticks include a big trade so the bubble layer has something.
        ts = np.arange(0, 4000, 250)
        price = np.full(len(ts), 99.5)
        m = _ticks(ts, price)
        m[5, 2] = 500.0        # one large trade volume
        ind.calculate(m)
        return ind

    def test_heatmap_layer(self):
        groups = self._prepared(show_bbo=False, show_bubbles=False).draw(
            None, interval_ms=1000)
        assert "Heatmap" in groups
        assert any(isinstance(p, Rects) for p in groups["Heatmap"])
        rects = groups["Heatmap"][0]
        # Per-cell colors (one color per rect).
        assert len(rects.fill_color) == len(rects.x0) > 0

    def test_heatmap_columns_centered_on_col_ts(self):
        # Columns must be drawn CENTERED on their representative timestamp
        # (col_ts), like CVD / DeltaBars center bars on bar_ts - not starting at
        # it (which would shift the whole grid half a column to the right).
        ind = self._prepared(show_bbo=False, show_bubbles=False)
        rects = ind.draw(None, interval_ms=1000)["Heatmap"][0]
        col_ts = set(np.asarray(ind._grid["col_ts"], dtype=np.float64).tolist())
        width = float(ind._grid["col_right"][0] - ind._grid["col_left"][0])
        for a, b in zip(rects.x0, rects.x1):
            mid = (a + b) / 2.0
            assert min(abs(mid - c) for c in col_ts) < 1e-6   # centered on col_ts
            assert abs((b - a) - width) < 1e-6                 # width preserved

    def test_bbo_layer_toggle(self):
        on = self._prepared(show_bbo=True, show_bubbles=False).draw(None)
        assert "BBO" in on
        assert any(isinstance(p, Segments) for p in on["BBO"])
        off = self._prepared(show_bbo=False, show_bubbles=False).draw(None)
        assert "BBO" not in off

    def test_bubble_layer_toggle(self):
        on = self._prepared(show_bbo=False, show_bubbles=True,
                            bubble_threshold=100.0).draw(None)
        assert "Executed Volume" in on
        assert any(isinstance(p, Points) for p in on["Executed Volume"])
        off = self._prepared(show_bbo=False, show_bubbles=False).draw(None)
        assert "Executed Volume" not in off


class TestProxyDelegation:
    def test_query_delegates_band_access_preserved(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        proxy = MultiBandProxy(["best_bid", "best_ask"])
        proxy._set_query_provider(ind)

        # Query method reaches the indicator.
        assert proxy.liquidity_at(101.0) == 500.0
        assert proxy.hottest(n=1)[0].price == 101.0
        assert proxy.mid == 99.5

        # Band attributes still resolve to the proxy's own (not delegated).
        assert proxy.best_bid is None      # not connected yet -> band slot

    def test_unknown_attr_raises(self):
        ind = Heatmap(_wall_book(), price_bucket=1.0)
        proxy = MultiBandProxy(["best_bid", "best_ask"])
        proxy._set_query_provider(ind)
        with pytest.raises(AttributeError):
            _ = proxy.does_not_exist



class TestLiveAutoscaleMirror:
    """The live updater mirrors an overlay's quad cells into its autoscale CDS
    so the OHLC autoscale (which is serialized at build time) can read the
    lazily-created heatmap source through a stable mirror."""

    def _mixin(self):
        from tradetropy.plotting.live.updater._vp_mixin import VolumeProfileUpdateMixin
        return VolumeProfileUpdateMixin()

    def _ref(self, registry, mirror):
        class _Ref:
            pass
        r = _Ref()
        r.registry = registry
        r.autoscale_mirror = mirror
        return r

    def test_mirror_copies_quad_columns(self):
        from bokeh.models import ColumnDataSource

        rects_src = ColumnDataSource(dict(
            left=[0, 1000], right=[1000, 2000],
            bottom=[99.0, 100.0], top=[100.0, 101.0],
            fill=["#111", "#222"],
        ))
        mirror = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        registry = {("Heatmap", "Rects"): rects_src}
        self._mixin()._sync_autoscale_mirror(self._ref(registry, mirror))
        assert list(mirror.data["left"]) == [0, 1000]
        assert list(mirror.data["right"]) == [1000, 2000]
        assert list(mirror.data["bottom"]) == [99.0, 100.0]
        assert list(mirror.data["top"]) == [100.0, 101.0]

    def test_mirror_noop_without_mirror(self):
        # No autoscale_mirror -> silently does nothing (non-participating overlay).
        self._mixin()._sync_autoscale_mirror(self._ref({}, None))

"""Tests for BookData, LiveBookRing (snapshot/delta + causal book_as_of) and book IO."""

import numpy as np
import pytest

from tradetropy.core.data_types import BookData, book_flat_columns, book_row_width
from tradetropy.data.data import LiveBookRing
from tradetropy.io.io import _append_book_hdf5, read_book


# =====
# BookData layout
# =====
class TestBookData:
    def test_column_layout(self):
        assert book_row_width(2) == 10  # 2 + 4*2
        cols = book_flat_columns(2)
        assert cols == [
            "ts", "kind",
            "bid_px_0", "bid_px_1", "bid_sz_0", "bid_sz_1",
            "ask_px_0", "ask_px_1", "ask_sz_0", "ask_sz_1",
        ]

    def test_accessors(self):
        # one row, 2 levels: ts=10 kind=0 bids px[100,99] sz[1,2] asks px[101,102] sz[3,4]
        row = [[10, 0, 100, 99, 1, 2, 101, 102, 3, 4]]
        bd = BookData("BTC", np.array(row, dtype=np.float64), levels=2)
        assert bd.ts[0] == 10
        assert list(bd.bid_px[0]) == [100, 99]
        assert list(bd.bid_sz[0]) == [1, 2]
        assert list(bd.ask_px[0]) == [101, 102]
        assert list(bd.ask_sz[0]) == [3, 4]

    def test_bad_shape_raises(self):
        from tradetropy.exceptions import ConfigError
        with pytest.raises(ConfigError):
            BookData("BTC", np.zeros((1, 5)), levels=2)  # width should be 10


# =====
# LiveBookRing snapshot + delta
# =====
class TestLiveBookRing:
    def test_snapshot_sets_book(self):
        r = LiveBookRing(window_size=10, levels=3)
        assert r.stale is True
        r.apply_snapshot(
            1, bids=[(100.0, 1.0), (99.0, 2.0)], asks=[(101.0, 1.5), (102.0, 2.5)]
        )
        assert r.stale is False
        assert r.best_bid == 100.0
        assert r.best_ask == 101.0
        assert r.mid == 100.5
        assert r.spread == 1.0

    def test_delta_updates_and_removes(self):
        r = LiveBookRing(window_size=10, levels=3)
        r.apply_snapshot(1, bids=[(100.0, 1.0), (99.0, 2.0)], asks=[(101.0, 1.5)])
        # Update best bid size, add a new bid, remove the 99 level.
        r.apply_delta(2, bids=[(100.0, 5.0), (99.5, 3.0), (99.0, 0.0)])
        assert r.best_bid == 100.0
        bk = r.book_as_of(2)
        # bids now: 100->5, 99.5->3 (99 removed), sorted desc.
        assert list(bk["bid_px"][:2]) == [100.0, 99.5]
        assert list(bk["bid_sz"][:2]) == [5.0, 3.0]

    def test_delta_ignored_while_stale(self):
        r = LiveBookRing(window_size=10, levels=3)
        r.apply_delta(1, bids=[(100.0, 1.0)])  # no snapshot yet -> ignored
        assert r.stale is True
        assert r.n_available == 0

    def test_imbalance(self):
        r = LiveBookRing(window_size=10, levels=3)
        r.apply_snapshot(1, bids=[(100.0, 3.0)], asks=[(101.0, 1.0)])
        # bid_vol=3 ask_vol=1 -> 3/4 = 0.75
        assert r.imbalance() == pytest.approx(0.75)

    def test_mark_stale_blanks_metrics(self):
        r = LiveBookRing(window_size=10, levels=3)
        r.apply_snapshot(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
        r.mark_stale()
        assert np.isnan(r.best_bid)
        assert np.isnan(r.mid)
        assert np.isnan(r.imbalance())


# =====
# Causal book_as_of
# =====
class TestBookAsOf:
    def test_returns_state_at_or_before_ts(self):
        r = LiveBookRing(window_size=10, levels=2)
        r.apply_snapshot(100, bids=[(10.0, 1.0)], asks=[(11.0, 1.0)])
        r.apply_delta(200, bids=[(10.0, 2.0)])
        r.apply_delta(300, bids=[(10.0, 3.0)])

        assert r.book_as_of(150)["bid_sz"][0] == 1.0   # only the snapshot applies
        assert r.book_as_of(250)["bid_sz"][0] == 2.0   # up to ts=200
        assert r.book_as_of(300)["bid_sz"][0] == 3.0   # exact match included
        assert r.book_as_of(999)["bid_sz"][0] == 3.0   # latest

    def test_before_first_returns_none(self):
        r = LiveBookRing(window_size=10, levels=2)
        r.apply_snapshot(100, bids=[(10.0, 1.0)], asks=[(11.0, 1.0)])
        assert r.book_as_of(50) is None

    def test_empty_returns_none(self):
        r = LiveBookRing(window_size=10, levels=2)
        assert r.book_as_of(100) is None


# =====
# Book IO round-trip
# =====
class TestBookIO:
    def test_append_and_read_roundtrip(self, tmp_path):
        r = LiveBookRing(window_size=50, levels=3)
        r.apply_snapshot(1, bids=[(100.0, 1.0), (99.0, 2.0)], asks=[(101.0, 1.5)])
        r.apply_delta(2, bids=[(100.0, 5.0)])
        r.apply_delta(3, asks=[(101.0, 0.0), (102.0, 4.0)])
        rows = r.to_rows()
        assert len(rows) == 3

        path = tmp_path / "book_btc.h5"
        _append_book_hdf5(rows, path, levels=3)

        bd = read_book(path, "BTC")  # levels read from metadata
        assert isinstance(bd, BookData)
        assert bd.levels == 3
        assert len(bd.data) == 3
        np.testing.assert_array_equal(bd.ts.astype(int), [1, 2, 3])
        np.testing.assert_allclose(bd.data, rows, equal_nan=True)

"""
Tests for BookData / MboData display and navigation parity with
TickData / KlineData.

Covers:
    - BookData derived [N] properties (best_bid, best_ask, mid, spread)
    - BookData / MboData table __repr__ and _repr_html_
    - __len__, head, tail, __getitem__ (int/slice, negative, out-of-range)
    - filter() with time range, positional index range and mask/callable
    - Empty-data handling for repr / summary
"""

import numpy as np
import pytest

from tradetropy.core.data_types import (
    BOOK_SUMMARY_COLS,
    BookData,
    MBO_COLS,
    MBO_TRADE,
    MboData,
    book_row_width,
)


# -----
# Fixtures
# -----

@pytest.fixture
def book():
    """
    8 synthetic book events, 2 levels per side, ts stepping by 1000 ms.

    Flat layout columns for levels=2:
        ts, kind, bid_px_0, bid_px_1, bid_sz_0, bid_sz_1,
        ask_px_0, ask_px_1, ask_sz_0, ask_sz_1
    """
    base = 1_700_000_000_000
    levels = 2
    n = 8
    width = book_row_width(levels)          # 2 + 4*2 = 10
    data = np.zeros((n, width), dtype=np.float64)
    data[:, 0] = base + np.arange(n) * 1000            # ts
    data[:, 1] = np.arange(n) % 2                      # kind (0/1)
    # bids (level 0 best = higher price)
    data[:, 2] = 100.0 + np.arange(n)                  # bid_px_0
    data[:, 3] = 99.5 + np.arange(n)                   # bid_px_1
    data[:, 4] = 1.0 + np.arange(n)                    # bid_sz_0
    data[:, 5] = 2.0 + np.arange(n)                    # bid_sz_1
    # asks (level 0 best = lower price)
    data[:, 6] = 100.5 + np.arange(n)                  # ask_px_0
    data[:, 7] = 101.0 + np.arange(n)                  # ask_px_1
    data[:, 8] = 0.8 + np.arange(n)                    # ask_sz_0
    data[:, 9] = 1.6 + np.arange(n)                    # ask_sz_1
    return BookData('BTCUSDT', data, levels=levels, tick_size=0.5)


@pytest.fixture
def mbo():
    """6 synthetic MBO events, ts stepping by 500 ms."""
    base = 1_700_000_000_000
    n = 6
    data = np.zeros((n, len(MBO_COLS)), dtype=np.float64)
    data[:, 0] = base + np.arange(n) * 500             # ts
    data[:, 1] = 1000 + np.arange(n)                   # order_id
    data[:, 2] = np.where(np.arange(n) % 2 == 0, 1, -1)  # side
    data[:, 3] = 100.0 + np.arange(n)                  # price
    data[:, 4] = 1.0 + np.arange(n)                    # size
    data[:, 5] = np.arange(n) % 4                      # action
    return MboData('BTCUSDT', data, tick_size=0.5)


# -----
# BookData derived [N] properties
# -----

def test_book_derived_properties_values(book):
    assert np.array_equal(book.best_bid, book.data[:, 2])
    assert np.array_equal(book.best_ask, book.data[:, 6])
    assert np.allclose(book.spread, book.data[:, 6] - book.data[:, 2])
    assert np.allclose(book.mid, (book.data[:, 2] + book.data[:, 6]) / 2.0)


def test_book_derived_properties_shape(book):
    for arr in (book.best_bid, book.best_ask, book.mid, book.spread):
        assert arr.shape == (len(book),)


def test_book_mid_spread_nan_propagation(book):
    book.data[0, 6] = np.nan                # break best ask on first event
    assert np.isnan(book.mid[0])
    assert np.isnan(book.spread[0])
    assert not np.isnan(book.mid[1])


# -----
# BookData repr / html
# -----

def test_book_repr_has_header_and_summary_columns(book):
    r = repr(book)
    assert "BookData(symbol='BTCUSDT'" in r
    assert "levels=2" in r
    for col in ("best_bid", "best_ask", "spread", "mid"):
        assert col in r


def test_book_repr_html_is_table(book):
    html = book._repr_html_()
    assert "<table" in html
    assert "best_bid" in html


def test_book_summary_cols_constant():
    assert BOOK_SUMMARY_COLS[0] == "ts"
    assert "best_bid" in BOOK_SUMMARY_COLS and "mid" in BOOK_SUMMARY_COLS


def test_book_empty_repr():
    empty = BookData('BTCUSDT', np.empty((0, book_row_width(2))), levels=2)
    assert "(empty)" in repr(empty)
    assert len(empty) == 0


# -----
# BookData navigation
# -----

def test_book_len(book):
    assert len(book) == 8


def test_book_getitem_slice_preserves_metadata(book):
    sub = book[2:5]
    assert isinstance(sub, BookData)
    assert len(sub) == 3
    assert sub.levels == book.levels
    assert sub.tick_size == book.tick_size
    assert np.array_equal(sub.data, book.data[2:5])


def test_book_getitem_int_returns_one_row(book):
    one = book[3]
    assert isinstance(one, BookData)
    assert len(one) == 1
    assert np.array_equal(one.data[0], book.data[3])


def test_book_negative_index_matches_tail(book):
    assert np.array_equal(book[-3:].data, book.tail(3).data)


def test_book_getitem_out_of_range(book):
    with pytest.raises(IndexError):
        _ = book[100]


def test_book_head_tail(book):
    assert np.array_equal(book.head(2).data, book.data[:2])
    assert np.array_equal(book.tail(2).data, book.data[-2:])


def test_book_filter_time_range(book):
    ts = book.data[:, 0]
    sub = book.filter(start=int(ts[2]), end=int(ts[5]))
    assert len(sub) == 4
    assert sub.levels == book.levels


def test_book_filter_idx_range(book):
    sub = book.filter(idx_start=1, idx_end=4)
    assert np.array_equal(sub.data, book.data[1:4])


def test_book_filter_mask_callable(book):
    sub = book.filter(lambda d: d[:, 1] == 0)   # snapshots only
    assert np.all(sub.data[:, 1] == 0)


# -----
# MboData repr / html
# -----

def test_mbo_repr_has_header_and_table(mbo):
    r = repr(mbo)
    assert "MboData(symbol='BTCUSDT'" in r
    for col in ("order_id", "side", "price", "size", "action"):
        assert col in r


def test_mbo_repr_html_is_table(mbo):
    html = mbo._repr_html_()
    assert "<table" in html
    assert "price" in html


def test_mbo_empty_repr():
    empty = MboData('BTCUSDT', np.empty((0, len(MBO_COLS))))
    assert "(empty)" in repr(empty)
    assert len(empty) == 0


# -----
# MboData navigation
# -----

def test_mbo_len(mbo):
    assert len(mbo) == 6


def test_mbo_getitem_slice_preserves_metadata(mbo):
    sub = mbo[1:4]
    assert isinstance(sub, MboData)
    assert len(sub) == 3
    assert sub.tick_size == mbo.tick_size
    assert np.array_equal(sub.data, mbo.data[1:4])


def test_mbo_negative_index_matches_tail(mbo):
    assert np.array_equal(mbo[-2:].data, mbo.tail(2).data)


def test_mbo_getitem_out_of_range(mbo):
    with pytest.raises(IndexError):
        _ = mbo[100]


def test_mbo_head_tail(mbo):
    assert np.array_equal(mbo.head(2).data, mbo.data[:2])
    assert np.array_equal(mbo.tail(2).data, mbo.data[-2:])


def test_mbo_filter_mask_trades_only(mbo):
    sub = mbo.filter(lambda d: d[:, 5] == MBO_TRADE)
    assert np.all(sub.data[:, 5] == MBO_TRADE)

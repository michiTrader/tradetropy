"""
Tests for TickData / KlineData positional indexing and filtering.

Covers:
    - __getitem__ with int (1-row object) and slice (ranged object)
    - Negative integer indexing and out-of-range IndexError
    - filter() with the renamed start/end time-range params
    - filter() with the new idx_start/idx_end positional range
    - Combining index range with mask/time in one filter() call
"""

import numpy as np
import pytest

from tradetropy.core.data_types import TickData, KlineData


# -----
# Fixtures
# -----

@pytest.fixture
def ticks():
    """10 synthetic ticks, ts stepping by 1000 ms from base."""
    base = 1_700_000_000_000
    n = 10
    data = np.zeros((n, 7), dtype=np.float64)
    data[:, 0] = base + np.arange(n) * 1000            # ts
    data[:, 1] = 100.0 + np.arange(n)                  # bid
    data[:, 2] = 100.5 + np.arange(n)                  # ask
    data[:, 3] = np.arange(n) + 1                      # volume
    data[:, 6] = 100.25 + np.arange(n)                 # price
    return TickData('BTCUSDT', data, tick_size=0.5, tick_value=1.25, digits=3)


@pytest.fixture
def klines(klines_1m):
    return KlineData('BTCUSDT', klines_1m, timeframe='1m', tick_size=0.5)


# -----
# __getitem__ : slice
# -----

def test_getitem_slice_returns_same_type_and_rows(klines):
    sub = klines[3:7]
    assert isinstance(sub, KlineData)
    assert len(sub) == 4
    assert np.array_equal(sub.data, klines.data[3:7])


def test_getitem_slice_propagates_metadata(klines):
    sub = klines[3:7]
    assert sub.symbol == klines.symbol
    assert sub.tick_size == klines.tick_size
    assert sub.interval_ms == klines.interval_ms


def test_getitem_negative_slice_matches_tail(ticks):
    assert np.array_equal(ticks[-3:].data, ticks.tail(3).data)


def test_getitem_full_slice_matches_head(ticks):
    assert np.array_equal(ticks[:4].data, ticks.head(4).data)


# -----
# __getitem__ : int
# -----

def test_getitem_int_returns_one_row_object(ticks):
    row = ticks[4]
    assert isinstance(row, TickData)
    assert len(row) == 1
    assert np.array_equal(row.data[0], ticks.data[4])


def test_getitem_negative_int(ticks):
    assert np.array_equal(ticks[-1].data[0], ticks.data[-1])


def test_getitem_int_out_of_range_raises(ticks):
    with pytest.raises(IndexError):
        ticks[len(ticks)]
    with pytest.raises(IndexError):
        ticks[-len(ticks) - 1]


def test_getitem_bad_key_type_raises(ticks):
    with pytest.raises(TypeError):
        ticks['a']


# -----
# filter : start / end (renamed)
# -----

def test_filter_start_end_by_timestamp(ticks):
    base = int(ticks.data[0, 0])
    out = ticks.filter(start=base + 2000, end=base + 5000)
    assert isinstance(out, TickData)
    # ts inclusive on both ends -> indices 2..5
    assert np.array_equal(out.data[:, 0], ticks.data[2:6, 0])


def test_filter_start_only(ticks):
    base = int(ticks.data[0, 0])
    out = ticks.filter(start=base + 7000)
    assert len(out) == 3
    assert out.data[0, 0] == base + 7000


def test_filter_start_end_on_klines(klines):
    base = int(klines.data[0, 0])
    out = klines.filter(start=base + 60_000, end=base + 180_000)
    assert isinstance(out, KlineData)
    assert len(out) == 3


# -----
# filter : idx_start / idx_end
# -----

def test_filter_idx_range(klines):
    out = klines.filter(idx_start=2, idx_end=6)
    assert len(out) == 4
    assert np.array_equal(out.data, klines.data[2:6])


def test_filter_idx_start_only(ticks):
    out = ticks.filter(idx_start=7)
    assert np.array_equal(out.data, ticks.data[7:])


def test_filter_idx_end_only(ticks):
    out = ticks.filter(idx_end=3)
    assert np.array_equal(out.data, ticks.data[:3])


# -----
# filter : combined conditions (AND)
# -----

def test_filter_idx_range_with_mask(ticks):
    # volume (col 3) > 5 -> indices 5..9 ; idx range 0..7 -> intersection 5,6
    out = ticks.filter(idx_start=0, idx_end=7, mask=lambda d: d[:, 3] > 5)
    assert np.array_equal(out.data[:, 3], np.array([6.0, 7.0]))


def test_filter_idx_range_with_time(ticks):
    base = int(ticks.data[0, 0])
    out = ticks.filter(idx_start=0, idx_end=6, start=base + 3000)
    # idx 0..5 AND ts >= base+3000 (idx 3..5) -> idx 3,4,5
    assert np.array_equal(out.data[:, 0], ticks.data[3:6, 0])


def test_filter_no_args_returns_all(ticks):
    out = ticks.filter()
    assert np.array_equal(out.data, ticks.data)


# -----
# immutability
# -----

def test_operations_do_not_mutate_original(klines):
    original = klines.data.copy()
    _ = klines[1:3]
    _ = klines.filter(idx_start=1, idx_end=3)
    _ = klines.head(2)
    assert np.array_equal(klines.data, original)

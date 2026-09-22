"""Tests for resolve_book_sync: the sync_book policy layer (warnings + resolved
offset) applied on top of the pure preflight, shared by all engines."""

import warnings

import numpy as np
import pytest

from tradetropy.core.data_types import BookData, book_row_width
from tradetropy.data._book_replay import resolve_book_sync
from tradetropy.ta.order_flow._core import (
    BOOK_SYNC_SYNCED,
    BOOK_SYNC_RECOVERABLE,
    BOOK_SYNC_UNRECOVERABLE,
)


def _price_path(n, start=100.0, step_ms=1000, seed=0):
    rng = np.random.default_rng(seed)
    prices = start + np.cumsum(rng.normal(0.0, 0.5, size=n))
    ts = np.arange(n, dtype=np.float64) * step_ms
    return ts, prices


def _book_from_prices(symbol, ts, prices, levels=2):
    """BookData whose mid == price path at each ts (best bid/ask straddle it)."""
    w = book_row_width(levels)
    rows = np.full((len(ts), w), np.nan, dtype=np.float64)
    rows[:, 0] = ts
    rows[:, 1] = 0
    rows[:, 2] = prices - 0.5              # bid_px_0
    rows[:, 2 + levels] = 5.0             # bid_sz_0
    rows[:, 2 + 2 * levels] = prices + 0.5  # ask_px_0
    rows[:, 2 + 3 * levels] = 5.0         # ask_sz_0
    return BookData(symbol, rows, levels=levels)


def test_policy_synced_no_warning():
    ts, prices = _price_path(300)
    bd = _book_from_prices("BTC", ts, prices)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        res = resolve_book_sync(
            {"BTC": bd}, {"BTC": (ts, prices)}, sync_book=False,
        )
    report, offset = res["BTC"]
    assert report.verdict == BOOK_SYNC_SYNCED
    assert offset == 0


def test_policy_recoverable_opt_in_applies_offset():
    ts, prices = _price_path(400, seed=1)
    offset = 7000
    book_ts = ts - offset
    bd = _book_from_prices("BTC", book_ts, prices)
    with pytest.warns(UserWarning, match="auto-aligned"):
        res = resolve_book_sync(
            {"BTC": bd}, {"BTC": (ts, prices)},
            sync_book=True, max_staleness_ms=3000,
        )
    report, applied = res["BTC"]
    assert report.verdict == BOOK_SYNC_RECOVERABLE
    assert abs(applied - offset) <= 1000


def test_policy_recoverable_no_opt_in_warns_and_keeps_zero():
    ts, prices = _price_path(400, seed=1)
    bd = _book_from_prices("BTC", ts - 7000, prices)
    with pytest.warns(UserWarning, match="sync_book=True"):
        res = resolve_book_sync(
            {"BTC": bd}, {"BTC": (ts, prices)},
            sync_book=False, max_staleness_ms=3000,
        )
    report, applied = res["BTC"]
    assert report.verdict == BOOK_SYNC_RECOVERABLE
    assert applied == 0


def test_policy_unrecoverable_warns_and_keeps_zero():
    ts, prices = _price_path(400, seed=2)
    bd = _book_from_prices("BTC", ts - 500_000, prices)  # far beyond max_offset
    with pytest.warns(UserWarning, match="too desynchronized"):
        res = resolve_book_sync(
            {"BTC": bd}, {"BTC": (ts, prices)},
            sync_book=True, max_staleness_ms=3000, max_offset_ms=60_000,
        )
    report, applied = res["BTC"]
    assert report.verdict == BOOK_SYNC_UNRECOVERABLE
    assert applied == 0


def test_policy_skips_symbol_without_trades():
    ts, prices = _price_path(100)
    bd = _book_from_prices("BTC", ts, prices)
    res = resolve_book_sync({"BTC": bd}, {}, sync_book=False)
    assert res == {}

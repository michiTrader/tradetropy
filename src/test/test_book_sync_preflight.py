"""Tests for the pure order-book / tick synchronization preflight
(analyze_book_tick_sync): coverage, staleness, clock-offset detection and the
synced / recoverable / unrecoverable verdict."""

import numpy as np

from tradetropy.ta.order_flow._core import (
    analyze_book_tick_sync,
    BOOK_SYNC_SYNCED,
    BOOK_SYNC_RECOVERABLE,
    BOOK_SYNC_UNRECOVERABLE,
)


def _price_path(n, start=100.0, step_ms=1000, seed=0):
    """A random-walk price path and its timestamps (ms)."""
    rng = np.random.default_rng(seed)
    prices = start + np.cumsum(rng.normal(0.0, 0.5, size=n))
    ts = np.arange(n, dtype=np.float64) * step_ms
    return ts, prices


def _book_from_prices(ts, prices, spread=0.1):
    """Build (book_ts, book_mid) tracking a price path with a small spread."""
    return ts.copy(), prices.copy()


def test_fully_synced_book():
    """A book that shares the trade clock and covers it -> synced, offset 0."""
    ts, prices = _price_path(300)
    book_ts, book_mid = _book_from_prices(ts, prices)
    rep = analyze_book_tick_sync(ts, book_ts, book_mid, prices)
    assert rep.verdict == BOOK_SYNC_SYNCED
    assert rep.resolved_offset_ms == 0
    assert rep.coverage >= 0.90


def test_too_few_rows_unrecoverable():
    """A shift is needed but too few rows to trust it -> unrecoverable.

    min_rows gates trust in an ESTIMATED offset, not the attach of an aligned
    book: here the book is clearly offset (large zero-offset residual) but has
    fewer than min_rows, so the required shift cannot be trusted and the book is
    refused (never auto-aligned)."""
    ts, prices = _price_path(45, step_ms=1000, seed=7)
    # A dense book (one row per trade) with a clear 7 s offset, but only 45 rows
    # (< min_rows=50): the offset is detectable yet cannot be trusted.
    book_ts = ts.copy() - 7000
    book_mid = prices.copy()
    rep = analyze_book_tick_sync(
        ts, book_ts, book_mid, prices,
        min_rows=50, max_staleness_ms=3000, max_offset_ms=60_000,
    )
    assert rep.verdict == BOOK_SYNC_UNRECOVERABLE
    assert rep.resolved_offset_ms == 0
    assert "min_rows" in rep.reason


def test_small_aligned_book_is_synced():
    """A small but correctly aligned book (partial coverage) is usable, not a
    desync: min_rows must not reject an already-aligned book."""
    ts, prices = _price_path(300)
    # Only 12 rows, aligned, covering a slice of the trades.
    book_ts = ts[100:112].copy()
    book_mid = prices[100:112].copy()
    rep = analyze_book_tick_sync(
        ts, book_ts, book_mid, prices, min_rows=50,
    )
    assert rep.verdict == BOOK_SYNC_SYNCED
    assert rep.resolved_offset_ms == 0


def test_recoverable_constant_offset():
    """A constant clock offset within tolerance is detected and recoverable."""
    ts, prices = _price_path(400, step_ms=1000, seed=1)
    offset = 7000  # book clock lags the trade clock by 7 s
    book_ts = ts.copy() - offset
    book_mid = prices.copy()
    rep = analyze_book_tick_sync(
        ts, book_ts, book_mid, prices,
        max_staleness_ms=3000, max_offset_ms=60_000,
    )
    assert rep.verdict == BOOK_SYNC_RECOVERABLE
    # book_ts + resolved_offset should recover the trade clock (~+7000).
    assert abs(rep.resolved_offset_ms - offset) <= 1000
    assert rep.offset_score >= 0.90


def test_offset_beyond_tolerance_unrecoverable():
    """An offset larger than max_offset_ms cannot be corrected."""
    ts, prices = _price_path(400, step_ms=1000, seed=2)
    offset = 500_000  # far beyond max_offset_ms
    book_ts = ts.copy() - offset
    book_mid = prices.copy()
    rep = analyze_book_tick_sync(
        ts, book_ts, book_mid, prices,
        max_staleness_ms=3000, max_offset_ms=60_000,
    )
    assert rep.verdict == BOOK_SYNC_UNRECOVERABLE
    assert rep.resolved_offset_ms == 0


def test_unrelated_prices_not_aligned():
    """Book mid that does not track the trade price is never trusted for an
    offset correction: coverage misses at zero offset (book sits in the future),
    and the shift that would restore coverage yields garbage correlation."""
    ts, prices = _price_path(400, seed=3)
    rng = np.random.default_rng(99)
    # Book sits 30 s ahead of the trades (zero-offset coverage fails) and
    # carries an unrelated random price series.
    book_ts = ts.copy() + 30_000
    book_mid = 5000.0 + rng.normal(0.0, 50.0, size=prices.size)
    rep = analyze_book_tick_sync(
        ts, book_ts, book_mid, prices,
        max_staleness_ms=3000, max_offset_ms=60_000, min_corr=0.90,
    )
    assert rep.verdict == BOOK_SYNC_UNRECOVERABLE
    assert rep.resolved_offset_ms == 0


def test_partial_coverage_gap_unrecoverable():
    """A book whose timestamps never overlap the trades (beyond the correctable
    offset) -> unrecoverable ('timestamps too far apart')."""
    ts, prices = _price_path(400, seed=4)
    # Book sits ~10 million ms away from the trades: no offset within tolerance
    # can bring them together.
    book_ts = ts.copy() + 10_000_000
    book_mid = prices.copy()
    rep = analyze_book_tick_sync(
        ts, book_ts, book_mid, prices,
        min_rows=50, min_coverage=0.90, max_offset_ms=60_000,
    )
    assert rep.verdict == BOOK_SYNC_UNRECOVERABLE
    assert rep.resolved_offset_ms == 0


def test_no_trades_trivially_synced():
    """No trades to align against -> trivially synced."""
    ts, prices = _price_path(100)
    rep = analyze_book_tick_sync(
        np.array([]), ts, prices, np.array([]),
    )
    assert rep.verdict == BOOK_SYNC_SYNCED
    assert rep.resolved_offset_ms == 0

"""Tests for the shared BookReplayer: causal drain, idempotency, ordering and
the per-symbol timestamp offset used by the sync preflight."""

import numpy as np

from tradetropy.core.data_types import BookData, book_row_width
from tradetropy.data._book_replay import BookReplayer
from tradetropy.data._ring import LiveBookRing


def _book(symbol, tss, levels=2):
    """Build a BookData with one snapshot row per ts; mid tracks ts."""
    w = book_row_width(levels)
    rows = np.full((len(tss), w), np.nan, dtype=np.float64)
    for i, ts in enumerate(tss):
        px = 100.0 + i
        rows[i, 0] = ts
        rows[i, 1] = 0  # snapshot
        rows[i, 2] = px - 0.5           # bid_px_0
        rows[i, 2 + levels] = 5.0       # bid_sz_0
        rows[i, 2 + 2 * levels] = px + 0.5  # ask_px_0
        rows[i, 2 + 3 * levels] = 5.0   # ask_sz_0
    return BookData(symbol, rows, levels=levels)


def test_drain_is_causal_and_monotonic():
    """Only rows with ts <= cursor are applied; repeated calls are idempotent."""
    bd = _book("BTC", [1000, 2000, 3000])
    rep = BookReplayer({"BTC": bd})
    ring = LiveBookRing(window_size=100, levels=2)
    rings = [ring]

    rep.drain_to("BTC", 1500, rings)
    assert ring.n_available == 1          # only ts=1000 applied
    rep.drain_to("BTC", 1500, rings)      # idempotent
    assert ring.n_available == 1

    rep.drain_to("BTC", 3000, rings)      # ts=2000 and ts=3000 applied
    assert ring.n_available == 3
    # book_as_of returns the most recent image at/before the query.
    assert ring.book_as_of(2500)["ts"] == 2000


def test_offset_shifts_book_clock():
    """A positive offset moves the book onto a later (trade) clock."""
    bd = _book("BTC", [0, 1000, 2000])
    rep = BookReplayer({"BTC": bd}, offsets={"BTC": 5000})
    assert rep.offset("BTC") == 5000
    ring = LiveBookRing(window_size=100, levels=2)

    rep.drain_to("BTC", 4000, [ring])     # shifted ts are 5000,6000,7000
    assert ring.n_available == 0          # nothing <= 4000 yet
    rep.drain_to("BTC", 6000, [ring])
    assert ring.n_available == 2          # 5000 and 6000 applied
    img = ring.book_as_of(6000)
    assert img["ts"] == 6000              # stored on the shifted clock


def test_set_offset_and_reset():
    """set_offset updates the shift; reset rewinds the cursor."""
    bd = _book("BTC", [0, 1000, 2000])
    rep = BookReplayer({"BTC": bd})
    ring = LiveBookRing(window_size=100, levels=2)

    rep.drain_to("BTC", 5000, [ring])
    assert ring.n_available == 3

    rep.reset()
    ring.reset()
    rep.set_offset("BTC", 1000)
    rep.drain_to("BTC", 500, [ring])      # shifted 1000,2000,3000 -> none <=500
    assert ring.n_available == 0


def test_missing_symbol_is_noop():
    """Draining a symbol with no recorded book does nothing."""
    rep = BookReplayer({})
    assert not rep.has("BTC")
    ring = LiveBookRing(window_size=10, levels=2)
    rep.drain_to("BTC", 9999, [ring])
    assert ring.n_available == 0

"""
Shared order-book replayer: attach a recorded ``BookData`` to a tick-driven
engine and drain it into the live book rings in lockstep with the tick cursor.

A recorded book and the trade stream are two independently timestamped series.
Both the playback engines (replay / training) and the tick backtest replay the
book identically: each recorded row is a full top-K image applied as a snapshot
to the symbol's ``LiveBookRing`` when the tick cursor reaches its timestamp, so
``book_as_of`` / DeepTrades / the L2 overlays behave the same causal way in
every engine.

An optional per-symbol ``offset_ms`` shifts the book timestamps into the trade
clock. It is 0 by default (book replayed as delivered) and set by the engine
only when the sync preflight (``order_flow/_core.analyze_book_tick_sync``)
finds a recoverable clock offset and the caller opted in with ``sync_book``.
"""

from __future__ import annotations

import warnings

import numpy as np

from tradetropy.ta.order_flow._core import (
    analyze_book_tick_sync,
    BOOK_SYNC_SYNCED,
    BOOK_SYNC_RECOVERABLE,
    BOOK_SYNC_MIN_COVERAGE,
    BOOK_SYNC_MIN_ROWS,
    BOOK_SYNC_MAX_OFFSET_MS,
    BOOK_SYNC_MIN_CORR,
)


class BookReplayer:
    """
    Replays recorded L2 order books into live book rings, causally.

    Holds the recorded ``BookData`` per symbol, a per-symbol replay cursor and
    an optional per-symbol timestamp offset. ``drain_to(symbol, ts, rings)``
    applies every recorded row whose (offset-adjusted) timestamp is at or before
    ``ts`` as a snapshot; it is monotonic and idempotent past the cursor, so it
    is safe to call once per tick.

    Args:
        books (dict | None): ``{symbol: BookData}`` recorded books to replay.
        offsets (dict | None): Optional ``{symbol: int ms}`` shift added to the
            book timestamps to align them with the trade clock.
    """

    __slots__ = ("_books", "_offsets", "_idx")

    def __init__(self, books: "dict | None" = None, offsets: "dict | None" = None):
        self._books: dict = {}
        self._offsets: dict = {}
        self._idx: dict = {}
        for sym, bd in (books or {}).items():
            self._books[sym] = bd
            self._offsets[sym] = int((offsets or {}).get(sym, 0))
            self._idx[sym] = 0

    @property
    def symbols(self) -> list:
        """Symbols with a recorded book attached."""
        return list(self._books)

    def has(self, symbol: str) -> bool:
        """True when a recorded book is attached for ``symbol``."""
        return symbol in self._books

    def book(self, symbol: str):
        """Recorded BookData for ``symbol`` (or None)."""
        return self._books.get(symbol)

    def offset(self, symbol: str) -> int:
        """Timestamp offset (ms) applied to ``symbol``'s book (0 by default)."""
        return int(self._offsets.get(symbol, 0))

    def set_offset(self, symbol: str, offset_ms: int) -> None:
        """
        Set the timestamp offset (ms) added to a symbol's book timestamps.

        Args:
            symbol (str): Trading symbol.
            offset_ms (int): Shift added to each book row timestamp so the book
                is replayed on the trade clock.
        """
        self._offsets[symbol] = int(offset_ms)

    def reset(self) -> None:
        """Rewind every per-symbol cursor to the start of its recorded book."""
        for sym in self._idx:
            self._idx[sym] = 0

    def drain_to(self, symbol: str, ts: int, rings) -> None:
        """
        Apply recorded book rows with (offset-adjusted) ts <= ``ts`` as snapshots.

        Each recorded row is a full top-K image, applied as a snapshot so the
        book seen at each tick matches the original state (causal). Advances the
        per-symbol cursor; safe to call repeatedly (idempotent past the cursor).

        Args:
            symbol (str): Trading symbol.
            ts (int): Current tick timestamp (ms).
            rings: Iterable of LiveBookRing to feed (the symbol's book rings).
        """
        book = self._books.get(symbol)
        if book is None or not rings:
            return
        offset = self._offsets.get(symbol, 0)
        idx = self._idx.get(symbol, 0)
        n = len(book.data)
        k = book.levels
        bts = book.ts
        data = book.data
        while idx < n and int(bts[idx]) + offset <= ts:
            row = data[idx]
            bids = tuple(
                (float(p), float(s))
                for p, s in zip(row[2:2 + k], row[2 + k:2 + 2 * k])
                if not np.isnan(p) and not np.isnan(s)
            )
            asks = tuple(
                (float(p), float(s))
                for p, s in zip(row[2 + 2 * k:2 + 3 * k], row[2 + 3 * k:2 + 4 * k])
                if not np.isnan(p) and not np.isnan(s)
            )
            snap_ts = int(bts[idx]) + offset
            for ring in rings:
                ring.apply_snapshot(snap_ts, bids, asks)
            idx += 1
        self._idx[symbol] = idx


def resolve_book_sync(
    books: dict,
    trades: dict,
    *,
    sync_book: bool = False,
    engine_label: str = "engine",
    min_coverage: float = BOOK_SYNC_MIN_COVERAGE,
    min_rows: int = BOOK_SYNC_MIN_ROWS,
    max_staleness_ms: "float | None" = None,
    max_offset_ms: int = BOOK_SYNC_MAX_OFFSET_MS,
    min_corr: float = BOOK_SYNC_MIN_CORR,
) -> dict:
    """
    Run the sync preflight per symbol and resolve the offset to apply.

    For each symbol with both a recorded book and a trade stream, this measures
    how the book aligns with the trades (``analyze_book_tick_sync``) and applies
    the ``sync_book`` policy, emitting a warning when the book is desynchronized:

    - ``synced``: attached as-is, no warning.
    - ``recoverable`` + ``sync_book=True``: the estimated clock offset is applied
      (a warning documents the shift).
    - ``recoverable`` + ``sync_book=False``: attached as-is with a warning
      suggesting ``sync_book=True`` (book metrics stay NaN outside coverage).
    - ``unrecoverable``: attached as-is with a clear warning (too desynchronized
      or too little book data to align); never shifted.

    Pure except for the warnings: it computes offsets and lets the caller apply
    them to the BookReplayer, so the same decision is made in every engine.

    Args:
        books (dict): ``{symbol: BookData}`` recorded books.
        trades (dict): ``{symbol: (trade_ts, trade_price)}`` NumPy arrays.
        sync_book (bool): Opt-in to auto-align a recoverable clock offset.
        engine_label (str): Engine name used in the warning messages.
        min_coverage (float): See ``analyze_book_tick_sync``.
        min_rows (int): See ``analyze_book_tick_sync``.
        max_staleness_ms (float | None): See ``analyze_book_tick_sync``.
        max_offset_ms (int): See ``analyze_book_tick_sync``.
        min_corr (float): See ``analyze_book_tick_sync``.

    Returns:
        dict: ``{symbol: (BookSyncReport, offset_ms)}`` where ``offset_ms`` is
        the shift the caller should apply (0 unless auto-aligned).
    """
    results: dict = {}
    for sym, bd in (books or {}).items():
        tr = trades.get(sym)
        if tr is None:
            continue
        trade_ts, trade_price = tr
        report = analyze_book_tick_sync(
            trade_ts, bd.ts, bd.mid, trade_price,
            min_coverage=min_coverage, min_rows=min_rows,
            max_staleness_ms=max_staleness_ms, max_offset_ms=max_offset_ms,
            min_corr=min_corr,
        )
        offset = 0
        if report.verdict == BOOK_SYNC_SYNCED:
            pass
        elif report.verdict == BOOK_SYNC_RECOVERABLE:
            if sync_book:
                offset = report.resolved_offset_ms
                warnings.warn(
                    f"[{engine_label}] order book for {sym!r} is desynchronized "
                    f"from the trades; auto-aligned by {offset} ms "
                    f"(sync_book=True). {report.reason}.",
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"[{engine_label}] order book for {sym!r} looks "
                    f"desynchronized from the trades ({report.reason}). It is "
                    f"recoverable: pass sync_book=True to auto-align. Left as-is "
                    f"for now - book metrics will be NaN outside coverage.",
                    stacklevel=3,
                )
        else:  # unrecoverable
            warnings.warn(
                f"[{engine_label}] order book for {sym!r} is too desynchronized "
                f"to align with the trades ({report.reason}). Left as-is - book "
                f"metrics may be NaN. Provide a better-aligned book or more book "
                f"data.",
                stacklevel=3,
            )
        results[sym] = (report, offset)
    return results

import numpy as np

from tradetropy.core.constants import _OHLC_COL
from tradetropy.core.data_types import (
    book_row_width,
    N_MBO_COLS,
    _MBO_COL,
    MBO_ADD,
    MBO_MODIFY,
    MBO_CANCEL,
    MBO_TRADE,
)


class LiveRingBuffer:
    """
    Circular buffer with double-buffer layout for O(1) window access.

    Each tick is written at two positions (p and p+W) so the window of the
    last W ticks is always contiguous at buf[head : head+W] without wraparound
    complications.
    """

    __slots__ = ("_buf", "_W", "_head", "_n_writings", "col_index")

    def __init__(self, window_size: int, col_index: dict):
        """
        Initialize circular buffer.

        Args:
            window_size (int): Maximum number of rows to hold
            col_index (dict): Column name to index mapping
        """
        n_cols = max(col_index.values()) + 1
        self._buf = np.full((window_size * 2, n_cols), np.nan, dtype=np.float64)
        self._W = window_size
        self._head = 0
        self._n_writings = 0
        self.col_index = col_index

    def write(self, row: np.ndarray):
        """
        Write row to the buffer. Double-write for O(1) access.

        Args:
            row (ndarray): Row data to write
        """
        W = self._W
        p = self._head
        self._buf[p] = row
        self._buf[p + W] = row
        self._head = (p + 1) % W
        self._n_writings += 1

    def reset(self) -> None:
        """
        Clear all buffered data, restoring the ring to its just-created state.

        Resets the buffer contents to NaN and zeroes the head/write counters,
        preserving the column mapping and capacity. Used by ReplayEngine to
        rewind a replay in place without recreating the ring (the chart keeps
        its reference to this same object).
        """
        self._buf.fill(np.nan)
        self._head = 0
        self._n_writings = 0

    @property
    def n_available(self) -> int:
        """Number of rows available (at most window_size)."""
        return min(self._n_writings, self._W)

    def window(self, col_idx: int, size: int) -> np.ndarray:
        """
        Return last N values for given column. View without copy - O(1).

        Args:
            col_idx (int): Column index
            size (int): Maximum number of values to return

        Returns:
            ndarray: Last min(size, available) values
        """
        available = min(self._n_writings, self._W)
        real = min(size, available)
        if real == 0:
            return np.array([], dtype=np.float64)
        start = self._head + (self._W - real)
        return self._buf[start : start + real, col_idx]


class LiveOhlcRing:
    """
    Ring buffer for OHLC candles during live trading.

    Maintains the last W candles. The last candle is always partial and updated
    with each tick. On interval change, the partial candle is confirmed to the
    ring and a new partial candle is opened.
    """

    __slots__ = (
        "_buf",
        "_W",
        "_head",
        "_n_closed",
        "_partial_candle",
        "_ts_current_candle",
        "col_index",
        "interval_ms",
    )

    def __init__(self, window_size: int, col_index: dict, interval_ms: int):
        """
        Initialize OHLC ring buffer.

        Args:
            window_size (int): Maximum number of candles to hold
            col_index (dict): Column name to index mapping
            interval_ms (int): Candle interval in milliseconds
        """
        n_cols = max(col_index.values()) + 1
        self._buf = np.full((window_size * 2, n_cols), np.nan, dtype=np.float64)
        self._W = window_size
        self._head = 0
        self._n_closed = 0
        self._partial_candle = np.full(n_cols, np.nan, dtype=np.float64)
        self._ts_current_candle = -1
        self.col_index = col_index
        self.interval_ms = int(interval_ms)

    def reset(self) -> None:
        """
        Clear all candles, restoring the ring to its just-created state.

        Resets the closed-candle buffer to NaN, zeroes the head/closed
        counters and discards the partial candle, preserving the column
        mapping, capacity and interval. Used by ReplayEngine to rewind a
        replay in place without recreating the ring (the chart keeps its
        reference to this same object).
        """
        self._buf.fill(np.nan)
        self._head = 0
        self._n_closed = 0
        self._partial_candle.fill(np.nan)
        self._ts_current_candle = -1

    def process_tick(self, timestamp_ms: int, price: float, volume: float):
        """
        Update OHLC state with a new tick. Close candle if interval changed.

        Args:
            timestamp_ms (int): Tick timestamp in milliseconds
            price (float): Tick price
            volume (float): Tick volume
        """
        candle_ts = (timestamp_ms // self.interval_ms) * self.interval_ms

        if candle_ts != self._ts_current_candle:
            if self._ts_current_candle >= 0:
                self._confirm_partial_candle()
            self._ts_current_candle = candle_ts
            self._partial_candle[_OHLC_COL["ts"]] = candle_ts
            self._partial_candle[_OHLC_COL["open"]] = price
            self._partial_candle[_OHLC_COL["high"]] = price
            self._partial_candle[_OHLC_COL["low"]] = price
            self._partial_candle[_OHLC_COL["close"]] = price
            self._partial_candle[_OHLC_COL["volume"]] = volume
        else:
            col = _OHLC_COL
            if price > self._partial_candle[col["high"]]:
                self._partial_candle[col["high"]] = price
            if price < self._partial_candle[col["low"]]:
                self._partial_candle[col["low"]] = price
            self._partial_candle[col["close"]] = price
            self._partial_candle[col["volume"]] += volume

    def load_kline(self, ts: int, open_: float, high: float, low: float, close: float, volume: float):
        """
        Load a candle directly into the buffer.

        Used to initialize ring with historical or live candle data.

        Args:
            ts (int): Candle timestamp in milliseconds
            open_ (float): Open price
            high (float): High price
            low (float): Low price
            close (float): Close price
            volume (float): Total volume
        """
        self._partial_candle[_OHLC_COL["ts"]] = ts
        self._partial_candle[_OHLC_COL["open"]] = open_
        self._partial_candle[_OHLC_COL["high"]] = high
        self._partial_candle[_OHLC_COL["low"]] = low
        self._partial_candle[_OHLC_COL["close"]] = close
        self._partial_candle[_OHLC_COL["volume"]] = volume
        self._ts_current_candle = ts
        self._confirm_partial_candle()
        self._ts_current_candle = -1

    def _confirm_partial_candle(self):
        p = self._head
        self._buf[p] = self._partial_candle
        self._buf[p + self._W] = self._partial_candle
        self._head = (p + 1) % self._W
        self._n_closed += 1

    def closed_window(self, col_idx: int, size: int) -> np.ndarray:
        """
        Return last N closed candles for given column. View without copy.

        Args:
            col_idx (int): Column index
            size (int): Maximum number of candles to return

        Returns:
            ndarray: Last min(size, available) closed candles
        """
        available = min(self._n_closed, self._W)
        real = min(size, available)
        if real == 0:
            return np.array([], dtype=np.float64)
        start = self._head + (self._W - real)
        return self._buf[start : start + real, col_idx]

    @property
    def current_partial_candle(self) -> np.ndarray:
        """Current partial candle array."""
        return self._partial_candle

    @property
    def n_available_candles(self) -> int:
        """Total available candles (closed + partial, up to window_size)."""
        has_partial = 1 if self._ts_current_candle >= 0 else 0
        return min(self._n_closed, self._W) + has_partial


class LiveBookRing:
    """
    Ring buffer for an L2 order book reconstructed from snapshots and deltas.

    Maintains the live book as two price->size maps (bids, asks), applies
    snapshots (which reset the book) and deltas (which mutate it; a size of 0
    removes a level), and stores each resulting top-K image as a flat row in a
    circular buffer. The stored history lets ``book_as_of(ts)`` return the book
    state at or just before a trade - the causal lookup the DeepTrades detector
    needs (book strictly as-of each trade, never after).

    Row layout matches core.data_types.book_flat_columns:
        [ts, kind, bid_px_0..K-1, bid_sz_0..K-1, ask_px_0..K-1, ask_sz_0..K-1]
    with level 0 = best, bids descending by price, asks ascending.

    A ``stale`` flag marks the window between a detected sequence gap / reconnect
    and the next snapshot; while stale the book is not trusted (metrics return
    NaN) so indicators never act on a half-built book.
    """

    __slots__ = (
        "_buf", "_W", "_head", "_n_writings", "_levels", "_width",
        "_bids", "_asks", "_stale", "_last_row",
    )

    def __init__(self, window_size: int, levels: int = 10):
        """
        Initialize the book ring.

        Args:
            window_size (int): Maximum number of book events retained.
            levels (int): Number of book levels K kept per side.
        """
        self._levels = int(levels)
        self._width = book_row_width(self._levels)
        self._buf = np.full((window_size * 2, self._width), np.nan, dtype=np.float64)
        self._W = window_size
        self._head = 0
        self._n_writings = 0
        self._bids: dict = {}
        self._asks: dict = {}
        self._stale = True
        self._last_row = None

    def reset(self) -> None:
        """Clear the book and history, restoring the just-created state."""
        self._buf.fill(np.nan)
        self._head = 0
        self._n_writings = 0
        self._bids = {}
        self._asks = {}
        self._stale = True
        self._last_row = None

    @property
    def levels(self) -> int:
        """Number of levels K retained per side."""
        return self._levels

    @property
    def stale(self) -> bool:
        """True while the book is unsynced (awaiting a snapshot after a gap)."""
        return self._stale

    def mark_stale(self) -> None:
        """Flag the book as unsynced (e.g. after a reconnect or sequence gap)."""
        self._stale = True

    # ── mutation ──────────────────────────────────────────────────────────────

    def apply_snapshot(self, ts: int, bids, asks) -> None:
        """
        Replace the whole book with a fresh snapshot and store the image.

        Args:
            ts (int): Snapshot timestamp (ms).
            bids: Iterable of (price, size) bid levels.
            asks: Iterable of (price, size) ask levels.
        """
        self._bids = {float(p): float(s) for p, s in bids if float(s) > 0.0}
        self._asks = {float(p): float(s) for p, s in asks if float(s) > 0.0}
        self._stale = False
        self._store_row(ts, kind=0)

    def apply_delta(self, ts: int, bids=(), asks=()) -> None:
        """
        Mutate the book with incremental changes and store the resulting image.

        A size of 0 removes the level. Ignored while stale (a delta cannot be
        applied to an unsynced book; wait for the next snapshot).

        Args:
            ts (int): Delta timestamp (ms).
            bids: Iterable of (price, size) changed bid levels.
            asks: Iterable of (price, size) changed ask levels.
        """
        if self._stale:
            return
        self._apply_side(self._bids, bids)
        self._apply_side(self._asks, asks)
        self._store_row(ts, kind=1)

    @staticmethod
    def _apply_side(book_side: dict, changes) -> None:
        for p, s in changes:
            p = float(p)
            s = float(s)
            if s <= 0.0:
                book_side.pop(p, None)
            else:
                book_side[p] = s

    def _top_levels(self):
        """Return (bid_px, bid_sz, ask_px, ask_sz) arrays of the top K levels."""
        k = self._levels
        bid_px = np.full(k, np.nan, dtype=np.float64)
        bid_sz = np.full(k, np.nan, dtype=np.float64)
        ask_px = np.full(k, np.nan, dtype=np.float64)
        ask_sz = np.full(k, np.nan, dtype=np.float64)

        bids = sorted(self._bids.items(), key=lambda kv: -kv[0])[:k]
        asks = sorted(self._asks.items(), key=lambda kv: kv[0])[:k]
        for i, (p, s) in enumerate(bids):
            bid_px[i] = p
            bid_sz[i] = s
        for i, (p, s) in enumerate(asks):
            ask_px[i] = p
            ask_sz[i] = s
        return bid_px, bid_sz, ask_px, ask_sz

    def _store_row(self, ts: int, kind: int) -> None:
        k = self._levels
        bid_px, bid_sz, ask_px, ask_sz = self._top_levels()
        row = np.empty(self._width, dtype=np.float64)
        row[0] = float(ts)
        row[1] = float(kind)
        row[2 : 2 + k] = bid_px
        row[2 + k : 2 + 2 * k] = bid_sz
        row[2 + 2 * k : 2 + 3 * k] = ask_px
        row[2 + 3 * k : 2 + 4 * k] = ask_sz

        p = self._head
        self._buf[p] = row
        self._buf[p + self._W] = row
        self._head = (p + 1) % self._W
        self._n_writings += 1
        self._last_row = row

    def last_row(self) -> "np.ndarray | None":
        """Return a copy of the most recently stored book row, or None."""
        return None if self._last_row is None else self._last_row.copy()

    # ── current-book metrics ────────────────────────────────────────────────

    @property
    def best_bid(self) -> float:
        """Best (highest) bid price, or NaN when empty/stale."""
        if self._stale or not self._bids:
            return float("nan")
        return max(self._bids)

    @property
    def best_ask(self) -> float:
        """Best (lowest) ask price, or NaN when empty/stale."""
        if self._stale or not self._asks:
            return float("nan")
        return min(self._asks)

    @property
    def mid(self) -> float:
        """Mid price (best_bid + best_ask) / 2, or NaN."""
        b, a = self.best_bid, self.best_ask
        if np.isnan(b) or np.isnan(a):
            return float("nan")
        return (b + a) / 2.0

    @property
    def spread(self) -> float:
        """Best ask - best bid, or NaN."""
        b, a = self.best_bid, self.best_ask
        if np.isnan(b) or np.isnan(a):
            return float("nan")
        return a - b

    def imbalance(self, depth: "int | None" = None) -> float:
        """
        Order-book imbalance: bid_vol / (bid_vol + ask_vol) over top ``depth``.

        Args:
            depth (int | None): Levels to sum (None -> all retained levels).

        Returns:
            float: Imbalance in [0, 1] (>0.5 bid-heavy), or NaN when empty/stale.
        """
        if self._stale or not self._bids or not self._asks:
            return float("nan")
        n = depth if depth is not None else self._levels
        bid_vol = sum(s for _, s in sorted(self._bids.items(), key=lambda kv: -kv[0])[:n])
        ask_vol = sum(s for _, s in sorted(self._asks.items(), key=lambda kv: kv[0])[:n])
        total = bid_vol + ask_vol
        if total <= 0.0:
            return float("nan")
        return bid_vol / total

    # ── history / causal lookup ───────────────────────────────────────────────

    @property
    def n_available(self) -> int:
        """Number of stored book images (at most window_size)."""
        return min(self._n_writings, self._W)

    def _window(self) -> np.ndarray:
        """Contiguous view of the stored rows, oldest first."""
        n = self.n_available
        if n == 0:
            return self._buf[0:0]
        start = self._head + (self._W - n)
        return self._buf[start : start + n]

    def book_as_of(self, ts: int) -> "dict | None":
        """
        Return the book image as of ``ts`` (most recent row with row_ts <= ts).

        Causal: never returns a book state from after ``ts``. Used to read the
        resting liquidity a trade hit, strictly before/at the trade time.

        Args:
            ts (int): Reference timestamp (ms).

        Returns:
            dict | None: {'ts','kind','bid_px','bid_sz','ask_px','ask_sz'} arrays,
            or None when no book exists at or before ``ts``.
        """
        win = self._window()
        if len(win) == 0:
            return None
        idx = int(np.searchsorted(win[:, 0], float(ts), side="right") - 1)
        if idx < 0:
            return None
        row = win[idx]
        k = self._levels
        return {
            "ts": int(row[0]),
            "kind": int(row[1]),
            "bid_px": row[2 : 2 + k].copy(),
            "bid_sz": row[2 + k : 2 + 2 * k].copy(),
            "ask_px": row[2 + 2 * k : 2 + 3 * k].copy(),
            "ask_sz": row[2 + 3 * k : 2 + 4 * k].copy(),
        }

    def to_rows(self) -> np.ndarray:
        """Return a copy of the stored book rows (oldest first) for recording."""
        return self._window().copy()

    def book_window(self) -> dict:
        """
        Structured view of the stored book history (oldest first).

        Splits the flat circular buffer into per-level arrays so the pure L2
        order-flow detectors (walls, reloads, stop runs) can read the book's
        evolution over time without knowing the flat row layout.

        Returns:
            dict: {'ts' (int64 [R]), 'kind' (int64 [R]),
            'bid_px'/'bid_sz'/'ask_px'/'ask_sz' (float64 [R x K]), 'levels' K}.
            Empty arrays when no book has been stored yet.
        """
        win = self._window()
        k = self._levels
        if len(win) == 0:
            empty2 = np.zeros((0, k), dtype=np.float64)
            return {
                "ts": np.zeros(0, dtype=np.int64),
                "kind": np.zeros(0, dtype=np.int64),
                "bid_px": empty2, "bid_sz": empty2.copy(),
                "ask_px": empty2.copy(), "ask_sz": empty2.copy(),
                "levels": k,
            }
        return {
            "ts": win[:, 0].astype(np.int64),
            "kind": win[:, 1].astype(np.int64),
            "bid_px": win[:, 2 : 2 + k].copy(),
            "bid_sz": win[:, 2 + k : 2 + 2 * k].copy(),
            "ask_px": win[:, 2 + 2 * k : 2 + 3 * k].copy(),
            "ask_sz": win[:, 2 + 3 * k : 2 + 4 * k].copy(),
            "levels": k,
        }


class MboRing:
    """
    Ring buffer for L3 / market-by-order events.

    Stores a rolling window of per-order events (flat [M x 6] rows) in a
    double-buffer for O(1) contiguous window access, and reconstructs the live
    per-order resting book (order_id -> [side, price, size]) by applying
    ADD / MODIFY / CANCEL / TRADE actions. The event window powers the causal
    L3 detectors (iceberg reloads, liquidity grabs); the reconstructed orders
    give resting size per price level.
    """

    __slots__ = (
        "_buf", "_W", "_head", "_n_writings", "_orders", "_last_row",
    )

    def __init__(self, window_size: int):
        """
        Args:
            window_size (int): Maximum number of MBO events retained.
        """
        self._buf = np.full((window_size * 2, N_MBO_COLS), np.nan, dtype=np.float64)
        self._W = window_size
        self._head = 0
        self._n_writings = 0
        # order_id -> [side, price, size]
        self._orders: dict = {}
        self._last_row = None

    def reset(self) -> None:
        """Clear the event window and reconstructed orders."""
        self._buf.fill(np.nan)
        self._head = 0
        self._n_writings = 0
        self._orders = {}
        self._last_row = None

    def apply_event(
        self, ts: int, order_id: int, side: int, price: float,
        size: float, action: int,
    ) -> None:
        """
        Apply one MBO event: update the per-order book and store the event.

        Args:
            ts (int): Event timestamp (ms).
            order_id (int): Venue order id.
            side (int): +1 bid / -1 ask.
            price (float): Order price.
            size (float): New resting size (ADD/MODIFY) or remaining (TRADE).
            action (int): MBO_ADD / MBO_MODIFY / MBO_CANCEL / MBO_TRADE.
        """
        oid = int(order_id)
        act = int(action)
        if act in (MBO_ADD, MBO_MODIFY):
            self._orders[oid] = [int(side), float(price), float(size)]
        elif act == MBO_CANCEL:
            self._orders.pop(oid, None)
        elif act == MBO_TRADE:
            o = self._orders.get(oid)
            if o is not None:
                o[2] = float(size)
                if o[2] <= 0.0:
                    self._orders.pop(oid, None)

        row = np.array(
            [float(ts), float(oid), float(side), float(price),
             float(size), float(act)],
            dtype=np.float64,
        )
        p = self._head
        self._buf[p] = row
        self._buf[p + self._W] = row
        self._head = (p + 1) % self._W
        self._n_writings += 1
        self._last_row = row

    def apply_row(self, row) -> None:
        """Apply an MBO event from a flat [6] row (ts,order_id,side,price,size,action)."""
        self.apply_event(
            int(row[_MBO_COL["ts"]]), int(row[_MBO_COL["order_id"]]),
            int(row[_MBO_COL["side"]]), float(row[_MBO_COL["price"]]),
            float(row[_MBO_COL["size"]]), int(row[_MBO_COL["action"]]),
        )

    def last_row(self) -> "np.ndarray | None":
        """Copy of the most recently stored MBO event row, or None."""
        return None if self._last_row is None else self._last_row.copy()

    @property
    def n_available(self) -> int:
        """Number of stored events (at most window_size)."""
        return min(self._n_writings, self._W)

    def window(self) -> np.ndarray:
        """Contiguous view of the stored events, oldest first ([n x 6])."""
        n = self.n_available
        if n == 0:
            return self._buf[0:0]
        start = self._head + (self._W - n)
        return self._buf[start : start + n]

    def resting_size_at(self, side: int, price: float, tol: float = 1e-9) -> float:
        """
        Total reconstructed resting size at a price level on one side.

        Args:
            side (int): +1 bid / -1 ask.
            price (float): Price level.
            tol (float): Float comparison tolerance.

        Returns:
            float: Summed resting size of orders at that price/side.
        """
        total = 0.0
        for s, p, sz in self._orders.values():
            if s == side and abs(p - price) <= tol:
                total += sz
        return total

    def to_rows(self) -> np.ndarray:
        """Copy of the stored MBO event rows (oldest first) for recording."""
        return self.window().copy()

from __future__ import annotations

import numpy as np

# Sentinel: a scalar fast-path could not resolve the access and the caller must
# fall back to the full windowed view. Distinct from None (a legitimate value
# is never None here, but the sentinel keeps the contract explicit).
_FALLBACK = object()


class _Cursor:
    """
    Mutable, shareable cursor position for a group of views.

    Every view of one proxy references the SAME ``_Cursor`` instance, so
    advancing the proxy one bar is a single ``cursor.pos = idx`` write instead
    of one write per view (an OHLC proxy has 6 price views, a tick proxy 7).
    The value is a plain integer position; ``pos == -1`` means "not advanced".
    """

    __slots__ = ("pos",)

    def __init__(self, pos: int = -1):
        self.pos = pos


# -- WindowView ----------------------------------------------------------------
# Index access interface for ticks and candles.
# proxy[-1] = latest value, proxy[-2] = previous, proxy[0] = oldest.
#
# Supported backends:
#   A) TickDataStore  (backtest ticks)
#   B) OhlcDataStore  (backtest candles) -- includes partial candle logic
#   C) LiveRingBuffer (live ticks)
#   D) LiveOhlcRing   (live candles)     -- includes partial candle logic

class WindowView:
    """
    Indexed access interface for ticks and candles.

    Provides indexing where [-1] is the latest value, [-2] is second-to-latest,
    and [0] is the oldest available value. Supports multiple backends:
        - TickDataStore: Backtesting ticks
        - OhlcDataStore: Backtesting candles (includes partial candle logic)
        - LiveRingBuffer: Live ticks
        - LiveOhlcRing: Live candles (includes partial candle logic)

    Example:
        latest_close = ohlc_view[-1]
        last_10_closes = ohlc_view[-10:]
    """
    __slots__ = (
        "_tick_store",
        "_ohlc_store",
        "_tick_ring",
        "_ohlc_ring",
        "_col_idx",
        "_size",
        "_cursor",
        "_is_ohlc",
    )

    def __init__(
        self,
        col_idx: int,
        size: int,
        tick_store=None,
        ohlc_store=None,
        tick_ring=None,
        ohlc_ring=None,
    ):
        """
        Initialize window view.

        Args:
            col_idx (int): Column index to access
            size (int): Maximum window size
            tick_store: TickDataStore (backtest ticks)
            ohlc_store: OhlcDataStore (backtest candles)
            tick_ring: LiveRingBuffer (live ticks)
            ohlc_ring: LiveOhlcRing (live candles)
        """
        self._tick_store = tick_store
        self._ohlc_store = ohlc_store
        self._tick_ring = tick_ring
        self._ohlc_ring = ohlc_ring
        self._col_idx = col_idx
        self._size = size
        self._cursor = _Cursor()
        self._is_ohlc = (ohlc_store is not None) or (ohlc_ring is not None)

    @property
    def _view(self) -> np.ndarray:
        if self._tick_store is not None:
            idx = self._cursor.pos
            start = max(0, idx + 1 - self._size)
            return self._tick_store.matrix[start : idx + 1, self._col_idx]

        elif self._ohlc_store is not None:
            return self._ohlc_backtest_view()

        elif self._tick_ring is not None:
            return self._tick_ring.window(self._col_idx, self._size)

        else:
            return self._ohlc_live_view()

    def _ohlc_backtest_view(self) -> np.ndarray:
        from tradetropy.core.constants import N_OHLC_COLS  # noqa: F401

        ohlc_store = self._ohlc_store
        tick_idx = self._cursor.pos
        if tick_idx < 0:
            return np.array([], dtype=np.float64)

        current_candle_idx = ohlc_store.tick_to_candle_mapping[tick_idx]

        if ohlc_store.kline_mode:
            partial_candle = ohlc_store.partial_tick_candle(tick_idx)
            partial_value = partial_candle[self._col_idx]
            n_closed_to_show = min(current_candle_idx, self._size)
            if n_closed_to_show == 0:
                return np.array([partial_value], dtype=np.float64)
            start = current_candle_idx - n_closed_to_show
            closed_values = ohlc_store.matrix[start:current_candle_idx, self._col_idx]
            result = np.empty(n_closed_to_show + 1, dtype=np.float64)
            result[:n_closed_to_show] = closed_values
            result[-1] = partial_value
            return result

        n_closed_available = current_candle_idx
        n_closed_to_show = min(n_closed_available, self._size - 1)

        partial_candle = ohlc_store.partial_tick_candle(tick_idx)
        partial_value = partial_candle[self._col_idx]

        if n_closed_to_show == 0:
            return np.array([partial_value], dtype=np.float64)

        start_closed = n_closed_available - n_closed_to_show
        closed_values = ohlc_store.matrix[
            start_closed:n_closed_available, self._col_idx
        ]

        result = np.empty(n_closed_to_show + 1, dtype=np.float64)
        result[:n_closed_to_show] = closed_values
        result[-1] = partial_value
        return result

    def _ohlc_live_view(self) -> np.ndarray:
        from tradetropy.core.constants import N_OHLC_COLS

        ohlc_ring = self._ohlc_ring

        if self._col_idx >= N_OHLC_COLS:
            # The developing (partial) indicator value is stored in the SHADOW
            # half of the double buffer (index 2W-1), never in the primary slot
            # W-1. Slot W-1 is a real closed-candle position in the circular
            # buffer that closed_window() can read (its indices span at most
            # [head, head+W) = up to 2W-2, so 2W-1 is never read). Writing the
            # partial to W-1 would corrupt that closed bar's indicator once the
            # ring wraps; the collision-free shadow slot avoids it.
            p_partial = 2 * ohlc_ring._W - 1
            partial_val = ohlc_ring._buf[p_partial, self._col_idx]
            n_closed = min(ohlc_ring._n_closed, self._size - 1)
            if np.isnan(partial_val):
                nc = ohlc_ring._n_closed
                if nc == 0:
                    return np.array([np.nan], dtype=np.float64)
                n_show = min(nc, self._size - 1)
                lo = nc - n_show - 1
                if lo < 0:
                    body = ohlc_ring.closed_window(self._col_idx, nc)
                    result = np.empty(len(body) + 1, dtype=np.float64)
                    result[0] = np.nan
                    result[1:] = body
                    return result
                return ohlc_ring.closed_window(self._col_idx, n_show + 1)
            if n_closed == 0:
                return np.array([partial_val], dtype=np.float64)
            closed = ohlc_ring.closed_window(self._col_idx, n_closed)
            return np.r_[closed, partial_val]

        n_closed_to_show = min(ohlc_ring._n_closed, self._size - 1)
        partial_value = ohlc_ring.current_partial_candle[self._col_idx]
        if np.isnan(partial_value):
            return np.array([], dtype=np.float64)
        if n_closed_to_show == 0:
            return np.array([partial_value], dtype=np.float64)
        closed_values = ohlc_ring.closed_window(self._col_idx, n_closed_to_show)
        return np.r_[closed_values, partial_value]

    def __getitem__(self, item):
        return self._view[item]

    def __len__(self) -> int:
        return len(self._view)

    def __iter__(self):
        return iter(self._view)


# -- OhlcIndicatorView --------------------------------------------------------
# WindowView for indicators over OHLC candles in backtest.
# proxy[-1]  = indicator computed with current partial candle (O(L) recalc)
# proxy[-2:] = pre-calculated values over closed candles
# Result is cached per cursor: O(L) happens only once per tick.

class OhlcIndicatorView:
    """
    Window view for indicators calculated over OHLC candles in backtesting.

    Latest value ([-1]) is the indicator computed with the current partial
    candle (O(L) recalculation per tick). Earlier values are pre-calculated
    over closed candles. Result is cached per cursor: O(L) only happens once
    per tick.

    Example:
        rsi_view[-1]  # Latest RSI with partial candle
        rsi_view[-10]  # Pre-calculated RSI 10 candles ago
    """
    __slots__ = (
        "_ohlc_store",
        "_indicator",
        "_ind_col_idx",
        "_src_col_idx",
        "_size",
        "_cursor",
        "_cache_view",
        "_cache_cursor",
    )

    def __init__(self, ohlc_store, indicator, ind_col_idx: int, src_col_idx, size: int):
        """
        Initialize OHLC indicator view.

        Args:
            ohlc_store: OhlcDataStore backing this view
            indicator: Indicator instance with calculate method
            ind_col_idx (int): Column index for pre-calculated indicator
            src_col_idx: Source column index(es)
            size (int): Window size
        """
        self._ohlc_store = ohlc_store
        self._indicator = indicator
        self._ind_col_idx = ind_col_idx
        self._src_col_idx = src_col_idx
        self._size = size
        self._cursor = _Cursor()
        self._cache_view = None
        self._cache_cursor = -2

    @property
    def _view(self) -> np.ndarray:
        if self._cursor.pos == self._cache_cursor and self._cache_view is not None:
            return self._cache_view

        tick_idx = self._cursor.pos
        if tick_idx < 0:
            return np.array([], dtype=np.float64)

        store = self._ohlc_store
        n_closed = int(store.tick_to_candle_mapping[tick_idx])

        if store.kline_mode:
            # Partial bar uses a min_periods causal window: enough for the
            # recursive indicators to emit a value (no NaN) while staying cheap
            # per tick. Closed bars are read from the full-history precompute.
            L = getattr(self._indicator, "min_periods",
                        getattr(self._indicator, "length", 1))
            partial_value = float(store.partial_tick_candle(tick_idx)[self._src_col_idx])
            closed_src = store.matrix[
                max(0, n_closed - (L - 1)) : n_closed, self._src_col_idx
            ]
            window = np.append(closed_src, partial_value)
            bulk_result = self._indicator.calculate(window)
            if bulk_result.ndim > 1:
                bulk_result = bulk_result.ravel()
            partial_ind_value = float(bulk_result[-1])

            n_show = min(n_closed, self._size)
            start_ind = n_closed - n_show
            if n_show == 0:
                result = np.array([partial_ind_value], dtype=np.float64)
            else:
                closed_values = store.matrix[start_ind:n_closed, self._ind_col_idx]
                result = np.r_[closed_values, partial_ind_value]

            self._cache_view = result
            self._cache_cursor = tick_idx
            return result

        if not getattr(self._indicator, "use_partial", True):
            n_show = min(n_closed, self._size - 1)
            start_ind = n_closed - n_show
            lo, hi = start_ind - 1, n_closed
            if lo < 0:
                body = store.matrix[0:hi, self._ind_col_idx]
                result = np.empty(len(body) + 1, dtype=np.float64)
                result[0] = np.nan
                result[1:] = body
            else:
                result = store.matrix[lo:hi, self._ind_col_idx].astype(np.float64, copy=True)
            self._cache_view = result
            self._cache_cursor = tick_idx
            return result

        # Partial bar uses a min_periods causal window: enough for the recursive
        # indicators (RSI, MACD) to emit a value (no NaN) while staying cheap
        # per tick. Closed bars are read from the full-history precompute.
        L = getattr(self._indicator, "min_periods",
                    getattr(self._indicator, "length", 1))

        if isinstance(self._src_col_idx, list):
            partial_row = store.partial_tick_candle(tick_idx)
            partial_vals = partial_row[self._src_col_idx]
            closed_src = store.matrix[
                max(0, n_closed - (L - 1)) : n_closed
            ][:, self._src_col_idx]
            window = np.vstack([closed_src, partial_vals])
        else:
            partial_value = float(store.partial_tick_candle(tick_idx)[self._src_col_idx])
            closed_src = store.matrix[
                max(0, n_closed - (L - 1)) : n_closed, self._src_col_idx
            ]
            window = np.append(closed_src, partial_value)

        bulk_result = self._indicator.calculate(window)
        if bulk_result.ndim > 1:
            bulk_result = bulk_result.ravel()
        partial_ind_value = float(bulk_result[-1])

        # Cap closed bars at size - 1 so the total window (closed + partial)
        # is exactly `size`, matching the OHLC price view and the live/replay
        # indicator view. Using `size` here would emit `size + 1` values at ring
        # capacity, breaking backtest <-> replay parity of the full series.
        n_show = min(n_closed, self._size - 1)
        start_ind = n_closed - n_show

        if n_show == 0:
            result = np.array([partial_ind_value], dtype=np.float64)
        else:
            closed_values = store.matrix[start_ind:n_closed, self._ind_col_idx]
            result = np.r_[closed_values, partial_ind_value]

        self._cache_view = result
        self._cache_cursor = tick_idx
        return result

    def _scalar(self, item: int):
        """
        Resolve a single negative-integer access without building the window.

        The common strategy read is a scalar - ``self.ma_fast[-1]`` or
        ``self.ma_fast[-2]`` - yet the windowed ``_view`` materializes the
        whole trailing window (an ``np.r_`` of up to ``size`` values) on every
        access. For an integer index that is O(size) work for an O(1) need.
        This resolves the scalar directly:

        - ``[-1]`` recomputes only the developing (partial) bar's value over a
          min_periods window - identical to ``_view[-1]``, but without the
          O(size) concatenation.
        - ``[-k]`` (k >= 2) reads the single pre-calculated closed value.

        Returns the float, or ``_FALLBACK`` for cases the fast path does not
        cover (positive indices, multi-source indicators, out-of-range or the
        ``use_partial=False`` layout), so the caller uses the full ``_view``.

        Args:
            item (int): Negative integer index.

        Returns:
            float | object: The value, or the ``_FALLBACK`` sentinel.
        """
        store = self._ohlc_store
        tick_idx = self._cursor.pos
        if tick_idx < 0 or isinstance(self._src_col_idx, list):
            return _FALLBACK
        if not (store.kline_mode or getattr(self._indicator, "use_partial", True)):
            return _FALLBACK

        n_closed = int(store.tick_to_candle_mapping[tick_idx])

        if item == -1:
            # Windowed indicators (warmup_factor == 1: SMA, Donchian, ...) are
            # NOT path-dependent - the value at a bar depends only on the last
            # `length` sources - so the developing bar's value already equals
            # the full-history precompute (to the same tolerance the closed
            # bars rely on). Read it directly and skip the per-bar recompute.
            # Recursive indicators (warmup_factor > 1: RSI = Wilder, MACD/EMA)
            # are path-dependent, so keep the causal-window recompute that
            # matches live/replay (backtest <-> live parity).
            if (store.kline_mode
                    and getattr(self._indicator, "warmup_factor", 1) == 1
                    and 0 <= n_closed < store.matrix.shape[0]):
                return float(store.matrix[n_closed, self._ind_col_idx])
            L = getattr(self._indicator, "min_periods",
                        getattr(self._indicator, "length", 1))
            partial_value = float(
                store.partial_tick_candle(tick_idx)[self._src_col_idx]
            )
            closed_src = store.matrix[
                max(0, n_closed - (L - 1)) : n_closed, self._src_col_idx
            ]
            window = np.append(closed_src, partial_value)
            bulk_result = self._indicator.calculate(window)
            if bulk_result.ndim > 1:
                bulk_result = bulk_result.ravel()
            return float(bulk_result[-1])

        # item <= -2: the (k-1)-th closed value from the end (pre-calculated).
        size_cap = self._size if store.kline_mode else self._size - 1
        n_show = min(n_closed, size_cap)
        back = -item - 1
        idx = n_closed - back
        if back > n_show or idx < 0 or idx >= store.matrix.shape[0]:
            return _FALLBACK
        return float(store.matrix[idx, self._ind_col_idx])

    def __getitem__(self, item):
        if type(item) is int and item < 0:
            value = self._scalar(item)
            if value is not _FALLBACK:
                return value
        return self._view[item]

    def __len__(self) -> int:
        return len(self._view)

    def __iter__(self):
        return iter(self._view)


# -- MultiOhlcIndicatorView ---------------------------------------------------
# WindowView for band `band_idx` of a multi-band indicator in backtest.
# Cache is shared among all bands of the same indicator: O(L) recalculation
# happens only once per tick even with K bands.

class MultiOhlcIndicatorView:
    """
    Window view for a specific band of a multi-band indicator in backtesting.

    Handles indicators with K outputs (n_outputs > 1). Shares a cache among
    all bands of the same indicator: O(L) recalculation occurs once per tick
    even with multiple bands.

    Example:
        bbands_upper = MultiOhlcIndicatorView(..., band_idx=0)
        bbands_lower = MultiOhlcIndicatorView(..., band_idx=2)
    """
    __slots__ = (
        "_ohlc_store",
        "_indicator",
        "_ind_col_idxs",
        "_src_col_idx",
        "_band_idx",
        "_size",
        "_cursor",
        "_shared_cache",
    )

    def __init__(
        self,
        ohlc_store,
        indicator,
        ind_col_idxs: list,
        src_col_idx,
        band_idx: int,
        size: int,
        shared_cache: dict,
    ):
        """
        Initialize multi-band indicator view.

        Args:
            ohlc_store: OhlcDataStore backing this view
            indicator: Multi-band indicator instance
            ind_col_idxs (list): Column indices for each output band
            src_col_idx: Source column index(es)
            band_idx (int): Index of this band (0-based)
            size (int): Window size
            shared_cache (dict): Shared cache for all bands of this indicator
        """
        self._ohlc_store = ohlc_store
        self._indicator = indicator
        self._ind_col_idxs = ind_col_idxs
        self._src_col_idx = src_col_idx
        self._band_idx = band_idx
        self._size = size
        self._cursor = _Cursor()
        self._shared_cache = shared_cache

    def _recalculate(self):
        tick_idx = self._cursor.pos
        store = self._ohlc_store
        n_closed = int(store.tick_to_candle_mapping[tick_idx])
        K = self._indicator.n_outputs

        if store.kline_mode:
            use_full = not getattr(self._indicator, "use_partial", True)
            L = getattr(self._indicator, "min_periods", getattr(self._indicator, "length", 1))  # partial window

            if isinstance(self._src_col_idx, list):
                partial_row = store.partial_tick_candle(tick_idx)
                partial_vals = partial_row[self._src_col_idx]
                closed_src = store.matrix[
                    max(0, n_closed - (L - 1)) : n_closed
                ][:, self._src_col_idx]
                window = np.vstack([closed_src, partial_vals])
            else:
                partial_value = float(store.partial_tick_candle(tick_idx)[self._src_col_idx])
                closed_src = store.matrix[
                    max(0, n_closed - (L - 1)) : n_closed, self._src_col_idx
                ]
                window = np.append(closed_src, partial_value)

            bulk_result = self._indicator.calculate(window)
            n_show = min(n_closed, self._size)
            start_ind = n_closed - n_show

            bands = []
            for k in range(K):
                col_idx = self._ind_col_idxs[k]
                partial_val = float(bulk_result[k, -1])
                if n_show == 0:
                    bands.append(np.array([partial_val], dtype=np.float64))
                else:
                    closed_k = store.matrix[start_ind:n_closed, col_idx]
                    bands.append(np.r_[closed_k, partial_val])

            self._shared_cache["cursor"] = tick_idx
            self._shared_cache["bands"] = bands
            return

        use_full = not getattr(self._indicator, "use_partial", True)
        L = getattr(self._indicator, "min_periods", getattr(self._indicator, "length", 1))  # partial window

        if use_full:
            n_show = min(n_closed, self._size - 1)
            start_ind = n_closed - n_show
            bands = []
            for k in range(K):
                col_idx = self._ind_col_idxs[k]
                lo, hi = start_ind - 1, n_closed
                if lo < 0:
                    body = store.matrix[0:hi, col_idx]
                    band = np.empty(len(body) + 1, dtype=np.float64)
                    band[0] = np.nan
                    band[1:] = body
                else:
                    band = store.matrix[lo:hi, col_idx].astype(np.float64, copy=True)
                bands.append(band)
        else:
            if isinstance(self._src_col_idx, list):
                partial_row = store.partial_tick_candle(tick_idx)
                partial_vals = partial_row[self._src_col_idx]
                closed_src = store.matrix[
                    max(0, n_closed - (L - 1)) : n_closed
                ][:, self._src_col_idx]
                window = np.vstack([closed_src, partial_vals])
            else:
                partial_value = float(store.partial_tick_candle(tick_idx)[self._src_col_idx])
                closed_src = store.matrix[
                    max(0, n_closed - (L - 1)) : n_closed, self._src_col_idx
                ]
                window = np.append(closed_src, partial_value)

            bulk_result = self._indicator.calculate(window)
            # Cap closed bars at size - 1 so the total window (closed + partial)
            # is exactly `size`, matching the OHLC price view and the
            # live/replay indicator view (parity of the full band series).
            n_show = min(n_closed, self._size - 1)
            start_ind = n_closed - n_show

            bands = []
            for k in range(K):
                col_idx = self._ind_col_idxs[k]
                partial_val = float(bulk_result[k, -1])
                if n_show == 0:
                    band = np.array([partial_val], dtype=np.float64)
                else:
                    closed_k = store.matrix[start_ind:n_closed, col_idx]
                    band = np.r_[closed_k, partial_val]
                bands.append(band)

        self._shared_cache["cursor"] = tick_idx
        self._shared_cache["bands"] = bands

    @property
    def _view(self) -> np.ndarray:
        if self._cursor.pos < 0:
            return np.array([], dtype=np.float64)
        if self._cursor.pos != self._shared_cache.get("cursor", -2):
            self._recalculate()
        return self._shared_cache["bands"][self._band_idx]

    def __getitem__(self, item):
        return self._view[item]

    def __len__(self) -> int:
        return len(self._view)

    def __iter__(self):
        return iter(self._view)

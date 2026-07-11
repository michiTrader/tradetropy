from __future__ import annotations

import numpy as np

from tradetropy.core.constants import TICK_COLS, OHLC_COLS, _TICK_COL, _OHLC_COL
from tradetropy.exceptions import ColumnNotFoundError
from tradetropy.data._views import WindowView, _Cursor
from tradetropy.data._store import OhlcDataStore, TickDataStore
from tradetropy.data._ring import LiveRingBuffer, LiveOhlcRing


class ColumnRef:
    """
    Unified handle to a proxy column: declarative reference AND data accessor.

    A ColumnRef plays two roles depending on where it is used, so a strategy
    needs a single accessor (``proxy.close``) instead of two (``close_ref`` for
    declaration and ``close`` for data):

    - In ``init()`` it is a declarative reference the engine resolves at setup:
      ``self.add_indicator(self.btc.close, SMA(10))`` reads only its metadata
      (``proxy`` / ``col_name``), no data exists yet.
    - In ``on_data()`` it delegates indexing to the connected window view, so
      ``self.btc.close[-1]`` reads the live value exactly like the underlying
      WindowView (int and slice indexing, ``len()`` and iteration).

    The ``*_ref`` accessors (``close_ref``, ...) remain as aliases.
    """

    __slots__ = ("_proxy", "_col_name")

    def __init__(self, proxy, col_name: str):
        """
        Initialize column reference.

        Args:
            proxy: Parent TickProxy or OhlcProxy object
            col_name (str): Column name to reference
        """
        self._proxy = proxy
        self._col_name = col_name

    @property
    def symbol(self) -> str:
        """Symbol of the referenced proxy."""
        return self._proxy.symbol

    @property
    def col_name(self) -> str:
        """Column name being referenced."""
        return self._col_name

    @property
    def proxy(self):
        """Parent proxy object."""
        return self._proxy

    def _resolve_view(self) -> WindowView:
        """
        Return the connected WindowView for this column.

        Raises:
            ColumnNotFoundError: If accessed as data before the proxy is
                connected to a backend (i.e. before ``init()`` completes).
                Column references are only indexable inside ``on_data()``.
        """
        view = self._proxy._views.get(self._col_name)
        if view is None:
            raise ColumnNotFoundError(
                f"Column '{self._col_name}' is not connected yet. Column "
                "references are declarative in init() and only become "
                "indexable (e.g. proxy.close[-1]) inside on_data()."
            )
        return view

    def __getitem__(self, item):
        """Delegate indexing (int or slice) to the connected window view."""
        return self._resolve_view()[item]

    def __len__(self) -> int:
        """Number of available values in the connected window view."""
        view = self._proxy._views.get(self._col_name)
        return len(view) if view is not None else 0

    def __iter__(self):
        """Iterate over the connected window view values."""
        view = self._proxy._views.get(self._col_name)
        return iter(view) if view is not None else iter(())


class DatetimeColumnRef(ColumnRef):
    """
    Column reference that reads the 'ts' millisecond column and yields it as
    numpy ``datetime64[ms]`` instead of raw epoch milliseconds.

    It behaves exactly like the plain ``ts`` ColumnRef (int/slice indexing,
    ``len()`` and iteration) but every value it returns is converted to
    ``datetime64[ms]`` (UTC epoch, timezone-naive - the same convention used by
    the rest of the framework), so a strategy can read wall-clock datetimes
    directly from ``on_data()``:

        self.btc.datetime[-1]    # numpy.datetime64('2024-...T...', 'ms')
        self.btc.datetime[-10:]  # datetime64[ms] array
        self.tp.datetime[:]      # full window as datetime64[ms]

    Use ``.ts`` when you need the raw epoch-millisecond numbers and
    ``.datetime`` when you want ready-to-use datetimes.
    """

    __slots__ = ()

    def __init__(self, proxy):
        """
        Initialize datetime reference bound to the proxy's 'ts' column.

        Args:
            proxy: Parent TickProxy or OhlcProxy object
        """
        super().__init__(proxy, "ts")

    @staticmethod
    def _to_datetime64(raw):
        """
        Convert raw epoch-millisecond value(s) to datetime64[ms].

        Args:
            raw: Scalar or array of epoch milliseconds (float or int).

        Returns:
            numpy.datetime64 for a scalar, or a datetime64[ms] ndarray.
        """
        arr = np.asarray(raw)
        if arr.ndim == 0:
            return np.datetime64(int(arr), "ms")
        return arr.astype(np.int64).astype("datetime64[ms]")

    def __getitem__(self, item):
        """Index the 'ts' view and return the value(s) as datetime64[ms]."""
        return self._to_datetime64(self._resolve_view()[item])

    def __iter__(self):
        """Iterate over the connected window as datetime64[ms] values."""
        view = self._proxy._views.get(self._col_name)
        if view is None:
            return iter(())
        return iter(self._to_datetime64(view[:]))


class TickProxy:
    __slots__ = ("symbol", "_window_size", "_views", "_n_total", "_record_config",
                 "_cursor_ref")

    def __init__(self, symbol: str, window_size: int = 500):
        self.symbol = symbol
        self._window_size = window_size
        self._views: dict[str, WindowView] = {}
        self._n_total: int = 0
        self._record_config = None
        # Shared cursor for all this proxy's views: advancing is O(1).
        self._cursor_ref = _Cursor()

    def __len__(self) -> int:
        return self._n_total

    def _connect(self, **backend):
        self._views = {
            name: WindowView(col_idx=_TICK_COL[name], size=self._window_size, **backend)
            for name in TICK_COLS
        }
        for v in self._views.values():
            v._cursor = self._cursor_ref

    def _connect_backtest(self, store: TickDataStore):
        self._connect(tick_store=store)

    def _connect_live(self, ring: LiveRingBuffer):
        self._connect(tick_ring=ring)
        self._n_total = ring._n_writings

    def _advance(self, cursor: int):
        self._n_total = cursor + 1
        self._cursor_ref.pos = cursor

    @property
    def ts(self) -> ColumnRef:
        return ColumnRef(self, "ts")

    @property
    def datetime(self) -> DatetimeColumnRef:
        """
        The 'ts' column as datetime64[ms] instead of raw epoch milliseconds.

        Same window/indexing semantics as ``ts`` (``self.tp.datetime[-1]``,
        ``self.tp.datetime[-10:]``), but each value is a numpy datetime64[ms].
        """
        return DatetimeColumnRef(self)

    @property
    def bid(self) -> ColumnRef:
        return ColumnRef(self, "bid")

    @property
    def ask(self) -> ColumnRef:
        return ColumnRef(self, "ask")

    @property
    def volume(self) -> ColumnRef:
        return ColumnRef(self, "volume")

    @property
    def flags(self) -> ColumnRef:
        return ColumnRef(self, "flags")

    @property
    def volume_real(self) -> ColumnRef:
        return ColumnRef(self, "volume_real")

    @property
    def price(self) -> ColumnRef:
        return ColumnRef(self, "price")

    # Explicit ``*_ref`` aliases, kept for backward compatibility. They return
    # the same unified ColumnRef as the plain accessors above.
    @property
    def price_ref(self) -> ColumnRef:
        return ColumnRef(self, "price")

    @property
    def bid_ref(self) -> ColumnRef:
        return ColumnRef(self, "bid")

    @property
    def ask_ref(self) -> ColumnRef:
        return ColumnRef(self, "ask")

    @property
    def ts_ref(self) -> ColumnRef:
        return ColumnRef(self, "ts")

    @property
    def datetime_ref(self) -> DatetimeColumnRef:
        """Alias of ``datetime`` (kept symmetric with the ``*_ref`` accessors)."""
        return DatetimeColumnRef(self)

    @property
    def volume_ref(self) -> ColumnRef:
        return ColumnRef(self, "volume")

    @property
    def flags_ref(self) -> ColumnRef:
        return ColumnRef(self, "flags")

    @property
    def volume_real_ref(self) -> ColumnRef:
        return ColumnRef(self, "volume_real")

    def col_ref(self, name: str) -> ColumnRef:
        if name not in _TICK_COL:
            raise ColumnNotFoundError(f"Column '{name}' does not exist. Available: {TICK_COLS}")
        return ColumnRef(self, name)

    @property
    def n_ticks(self) -> int:
        return self._n_total

    @property
    def spread(self) -> float:
        if not self._views:
            return np.nan
        a = self._views["ask"]._view
        b = self._views["bid"]._view
        return float(a[-1] - b[-1]) if len(a) > 0 else np.nan


class OhlcProxy:
    """
    Handle to OHLC data returned by subscribe_ohlc().

    Exposes OHLC columns as WindowView objects where [-1] is always the partial
    candle (in-progress). Used by strategies to access candle data during
    backtesting and live trading.
    """

    __slots__ = (
        "symbol",
        "interval_ms",
        "_window_size",
        "_views",
        "_n_total",
        "_ohlc_store",
        "_ohlc_ring",
        "_record_config",
        "_cursor_ref",
    )

    def __init__(self, symbol: str, interval_ms: int, window_size: int = 200):
        self.symbol = symbol
        self.interval_ms = int(interval_ms)
        self._window_size = window_size
        self._views: dict[str, WindowView] = {}
        self._n_total: int = 0
        self._ohlc_store: OhlcDataStore | None = None
        self._ohlc_ring: LiveOhlcRing | None = None
        self._record_config = None
        # Shared cursor for all this proxy's views: advancing is O(1).
        self._cursor_ref = _Cursor()

    def _connect(self, **backend):
        self._ohlc_store = backend.get("ohlc_store")
        self._ohlc_ring = backend.get("ohlc_ring")
        self._views = {
            name: WindowView(col_idx=_OHLC_COL[name], size=self._window_size, **backend)
            for name in OHLC_COLS
        }
        for v in self._views.values():
            v._cursor = self._cursor_ref

    def _connect_backtest(self, ohlc_store: OhlcDataStore):
        self._connect(ohlc_store=ohlc_store)

    def _connect_live(self, ohlc_ring: LiveOhlcRing):
        self._connect(ohlc_ring=ohlc_ring)

    def _advance(self, cursor: int):
        self._n_total = cursor + 1
        self._cursor_ref.pos = cursor

    @property
    def ts(self) -> ColumnRef:
        return ColumnRef(self, "ts")

    @property
    def datetime(self) -> DatetimeColumnRef:
        """
        The 'ts' column as datetime64[ms] instead of raw epoch milliseconds.

        Same window/indexing semantics as ``ts`` (``self.btc.datetime[-1]``,
        ``self.btc.datetime[-10:]``), but each value is a numpy datetime64[ms].
        """
        return DatetimeColumnRef(self)

    @property
    def open(self) -> ColumnRef:
        return ColumnRef(self, "open")

    @property
    def high(self) -> ColumnRef:
        return ColumnRef(self, "high")

    @property
    def low(self) -> ColumnRef:
        return ColumnRef(self, "low")

    @property
    def close(self) -> ColumnRef:
        return ColumnRef(self, "close")

    @property
    def volume(self) -> ColumnRef:
        return ColumnRef(self, "volume")

    # Explicit ``*_ref`` aliases, kept for backward compatibility. They return
    # the same unified ColumnRef as the plain accessors above.
    @property
    def close_ref(self) -> ColumnRef:
        return ColumnRef(self, "close")

    @property
    def open_ref(self) -> ColumnRef:
        return ColumnRef(self, "open")

    @property
    def high_ref(self) -> ColumnRef:
        return ColumnRef(self, "high")

    @property
    def low_ref(self) -> ColumnRef:
        return ColumnRef(self, "low")

    @property
    def volume_ref(self) -> ColumnRef:
        return ColumnRef(self, "volume")

    @property
    def ts_ref(self) -> ColumnRef:
        return ColumnRef(self, "ts")

    @property
    def datetime_ref(self) -> DatetimeColumnRef:
        """Alias of ``datetime`` (kept symmetric with the ``*_ref`` accessors)."""
        return DatetimeColumnRef(self)

    def col_ref(self, name: str) -> ColumnRef:
        if name not in _OHLC_COL:
            raise ColumnNotFoundError(
                f"Column '{name}' does not exist in OHLC. Available: {OHLC_COLS}"
            )
        return ColumnRef(self, name)

    @property
    def n_klines(self) -> int:
        if self._ohlc_store is not None and self._n_total > 0:
            candle_idx = self._ohlc_store.tick_to_candle_mapping[self._n_total - 1]
            return candle_idx + 1
        if self._ohlc_ring is not None:
            return self._ohlc_ring.n_available_candles
        return 0

    @property
    def n_candles(self) -> int:
        return self.n_klines


class IndicatorProxy:
    __slots__ = (
        "_view",
        "_cursor_ref",
        "_flat_ok",
        "_flat_col",
        "_flat_map",
        "_flat_nrows",
        "_flat_size",
        "_flat_windowed",
    )

    def __init__(self):
        self._view: WindowView | None = None
        # Shared cursor: the connected view references the SAME holder, so
        # _advance is one write and the scalar fast path below reads it too.
        self._cursor_ref = _Cursor()
        # Single-frame scalar read fast path (kline-mode OHLC indicators only).
        self._flat_ok: bool = False
        self._flat_col: np.ndarray | None = None
        self._flat_map: np.ndarray | None = None
        self._flat_nrows: int = 0
        self._flat_size: int = 0
        self._flat_windowed: bool = False

    def _connect(self, view: WindowView):
        self._view = view
        view._cursor = self._cursor_ref

    def _enable_flat_kline(self, ohlc_store, ind_col_idx: int, size: int,
                           windowed: bool) -> None:
        """
        Enable the single-frame scalar read fast path for an OHLC indicator in
        kline mode.

        The common strategy read (``self.ind[-1]`` / ``self.ind[-k]``) then
        resolves directly against the pre-calculated indicator column, skipping
        the ``IndicatorProxy -> OhlcIndicatorView.__getitem__ -> _scalar`` chain
        (three frames plus per-read getattr/min work). Any access this path does
        not cover - the developing bar of a recursive (path-dependent)
        indicator, an out-of-window index, a slice or a positive index - falls
        back to the full windowed view, so the returned values are byte-for-byte
        identical to the view (guarded by test_kline_indicator_fastpath and
        test_indicator_engine_parity).

        Args:
            ohlc_store: OhlcDataStore backing the indicator column.
            ind_col_idx (int): Pre-calculated indicator column index.
            size (int): Window size (closed-bar cap, matches OhlcIndicatorView).
            windowed (bool): True for non-recursive indicators (warmup_factor
                == 1), whose developing bar equals the full-history precompute.
        """
        self._flat_col = ohlc_store.matrix[:, ind_col_idx]
        self._flat_map = ohlc_store.tick_to_candle_mapping
        self._flat_nrows = int(ohlc_store.matrix.shape[0])
        self._flat_size = int(size)
        self._flat_windowed = bool(windowed)
        self._flat_ok = True

    def _advance(self, cursor: int):
        self._cursor_ref.pos = cursor

    def __getitem__(self, item):
        # Scalar fast path: resolve a negative-integer read from the
        # pre-calculated column in one frame. Mirrors OhlcIndicatorView._scalar
        # exactly, falling through to the full view for anything it does not
        # cover so the result is identical.
        if self._flat_ok and type(item) is int and item < 0:
            cur = self._cursor_ref.pos
            if cur >= 0:
                n_closed = int(self._flat_map[cur])
                if item == -1:
                    # Windowed indicators: the developing bar equals the
                    # full-history precompute, so read it directly. Recursive
                    # ones keep the view's causal recompute (live/replay parity).
                    if self._flat_windowed and 0 <= n_closed < self._flat_nrows:
                        return float(self._flat_col[n_closed])
                else:
                    n_show = n_closed if n_closed < self._flat_size else self._flat_size
                    back = -item - 1
                    idx = n_closed - back
                    if back <= n_show and 0 <= idx < self._flat_nrows:
                        return float(self._flat_col[idx])
        if self._view is None:
            return np.nan
        return self._view[item]

    def __len__(self) -> int:
        return len(self._view) if self._view is not None else 0

    def __iter__(self):
        return iter(self._view) if self._view is not None else iter([])


class MultiBandProxy:
    """
    Proxy for indicators producing multiple series (n_outputs > 1).

    Provides indexed and named access to individual output bands.

    Access patterns:
        - mbp[0][-1]: Last value of band 0
        - mbp.upper[-1]: Last value by name (if output_names defined)
        - mbp[-1]: Equivalent to mbp[0][-1]
    """

    def __init__(self, output_names: list):
        self._views: list = []
        self._output_names: list = output_names
        self._ts_output_names: list = []
        self._ts_band_indices: list = []
        # Shared cursor for every band view of this indicator: advancing is one
        # write regardless of the band count.
        self._cursor_ref = _Cursor()
        # Set by add_indicator() for indicators that expose on-demand HVN/LVN
        # nodes (volume profiles). _node_indicator.compute_nodes(_node_source)
        # is called lazily from the .hvn / .lvn / .nodes properties.
        self._node_indicator = None
        self._node_source = None
        # Set by add_indicator() for indicators that expose a public query API
        # (e.g. Heatmap.liquidity_at / hottest / walls). Unknown public
        # attributes are delegated to this indicator so the strategy can query
        # it causally from on_data() through the same handle add_indicator
        # returned. See __getattr__ below.
        self._query_indicator = None
        for name in output_names:
            object.__setattr__(self, name, None)

    def _set_query_provider(self, indicator) -> None:
        """
        Wire on-demand public-query delegation (e.g. Heatmap query API).

        After this call, public attributes not found on the proxy itself are
        looked up on ``indicator`` (its query methods / scalar properties), so
        ``self.heat.liquidity_at(...)`` reaches the indicator while the band
        access (``self.heat.best_bid[-1]``) keeps working unchanged.
        """
        object.__setattr__(self, "_query_indicator", indicator)

    def __getattr__(self, name):
        """
        Delegate unknown PUBLIC attributes to the query-provider indicator.

        Only invoked when normal attribute lookup fails (so it never shadows the
        output-band attributes, which are real instance attributes). Private
        names are never delegated to avoid recursion during pickling / copy.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        prov = self.__dict__.get("_query_indicator")
        if prov is not None:
            try:
                return getattr(prov, name)
            except AttributeError:
                pass
        raise AttributeError(name)

    def _set_node_provider(self, indicator, source) -> None:
        """Wire on-demand node access (used by volume profile indicators)."""
        self._node_indicator = indicator
        self._node_source = source

    @property
    def hvn(self) -> list:
        """
        High-volume nodes of the indicator's causal profile (volume profiles).

        Computed on demand from the current bar; safe for backtesting. Raises
        ConfigError if the indicator was created with ``nodes=None`` or does not
        support node detection.
        """
        return self._compute_nodes()[0]

    @property
    def lvn(self) -> list:
        """Low-volume nodes of the indicator's causal profile (volume profiles)."""
        return self._compute_nodes()[1]

    @property
    def nodes(self) -> list:
        """All detected nodes (hvn + lvn) ordered ascending by price."""
        hvn, lvn = self._compute_nodes()
        return sorted([*hvn, *lvn], key=lambda n: n.price)

    def _compute_nodes(self) -> tuple:
        if self._node_indicator is None or not hasattr(
            self._node_indicator, "compute_nodes"
        ):
            from tradetropy.exceptions import ConfigError
            raise ConfigError(
                "This indicator does not expose HVN/LVN nodes. Only Volume "
                "Profile indicators created with nodes='hvn'|'lvn'|'both' support this."
            )
        return self._node_indicator.compute_nodes(self._node_source)

    def _connect_band(self, idx: int, view):
        while len(self._views) <= idx:
            self._views.append(None)
        self._views[idx] = view
        view._cursor = self._cursor_ref
        if idx < len(self._output_names):
            object.__setattr__(self, self._output_names[idx], view)
        if idx in self._ts_band_indices:
            ts_pos = self._ts_band_indices.index(idx)
            if ts_pos < len(self._ts_output_names):
                object.__setattr__(self, self._ts_output_names[ts_pos], view)

    def _advance(self, cursor: int):
        self._cursor_ref.pos = cursor

    def _connect(self, view):
        self._connect_band(0, view)

    def __getitem__(self, idx):
        if idx == -1 or idx == 0:
            return self._views[0]
        return self._views[idx]

    def __len__(self) -> int:
        return len(self._views)

    def __iter__(self):
        return iter(self._views)

    def __setattr__(self, name, value):
        if name in ("_views", "_output_names", "_ts_output_names", "_ts_band_indices"):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)


class OrderbookProxy:
    """
    Strategy-facing view of a live L2 order book.

    Returned by ``Strategy.subscribe_orderbook()``. Wraps a LiveBookRing that
    the engine feeds from OrderbookSnapshot / OrderbookDelta events (on the
    engine thread only). Exposes top-of-book metrics for use in ``on_data()``
    and a causal ``book_as_of(ts)`` for the order-flow detectors.

    While the book is unsynced (``stale`` - e.g. between a reconnect and the
    next snapshot) the metrics return NaN so a strategy never acts on a
    half-built book.

    Example:
        self.book = self.subscribe_orderbook('BTCUSDT', depth=20)
        def on_data(self):
            if self.book.imbalance(5) > 0.7:   # bid-heavy top 5 levels
                ...
    """

    __slots__ = ("symbol", "depth", "window_size", "_book_ring", "_record_config")

    def __init__(self, symbol: str, depth: int = 20, window_size: int = 5000):
        """
        Initialize the order-book proxy.

        Args:
            symbol (str): Trading symbol.
            depth (int): Number of book levels K to retain per side.
            window_size (int): Max number of book events kept for book_as_of.
        """
        self.symbol = symbol
        self.depth = int(depth)
        self.window_size = int(window_size)
        self._book_ring = None  # LiveBookRing, attached by the engine
        self._record_config = None  # _RecordConfig, set by subscribe_orderbook

    @property
    def stale(self) -> bool:
        """True when the book is not yet synced (no snapshot / after a gap)."""
        return self._book_ring is None or self._book_ring.stale

    @property
    def best_bid(self) -> float:
        """Best bid price, or NaN when unavailable/stale."""
        return float("nan") if self._book_ring is None else self._book_ring.best_bid

    @property
    def best_ask(self) -> float:
        """Best ask price, or NaN when unavailable/stale."""
        return float("nan") if self._book_ring is None else self._book_ring.best_ask

    @property
    def mid(self) -> float:
        """Mid price, or NaN when unavailable/stale."""
        return float("nan") if self._book_ring is None else self._book_ring.mid

    @property
    def spread(self) -> float:
        """Best ask - best bid, or NaN when unavailable/stale."""
        return float("nan") if self._book_ring is None else self._book_ring.spread

    def imbalance(self, depth: "int | None" = None) -> float:
        """
        Bid/total volume imbalance over the top ``depth`` levels (NaN if stale).

        Args:
            depth (int | None): Levels to sum (None -> all retained levels).

        Returns:
            float: Imbalance in [0, 1], or NaN.
        """
        if self._book_ring is None:
            return float("nan")
        return self._book_ring.imbalance(depth)

    def book_as_of(self, ts: int):
        """
        Causal book image as of ``ts`` (state at or just before ``ts``).

        Args:
            ts (int): Reference timestamp (ms).

        Returns:
            dict | None: Book arrays, or None when no book exists at/before ts.
        """
        if self._book_ring is None:
            return None
        return self._book_ring.book_as_of(ts)

    def book_window(self):
        """
        Structured view of the stored book history (oldest first), or None.

        Powers the L2 order-flow detectors (Deep Wall / Deep Reload / Stop Run)
        that read the book's evolution over a window rather than a single
        as-of snapshot. See ``LiveBookRing.book_window``.

        Returns:
            dict | None: Per-level book arrays, or None when no book is attached.
        """
        if self._book_ring is None:
            return None
        return self._book_ring.book_window()

    def __len__(self) -> int:
        return 0 if self._book_ring is None else self._book_ring.n_available

    def __repr__(self) -> str:
        return (
            f"OrderbookProxy(symbol={self.symbol!r}, depth={self.depth}, "
            f"stale={self.stale})"
        )


class MboProxy:
    """
    Strategy-facing view of an L3 / market-by-order stream.

    Returned by ``Strategy.subscribe_mbo()``. Wraps an MboRing the engine feeds
    from MBO events (engine thread only). Exposes the recent event window (for
    the L3 order-flow detectors) and per-level resting size from the
    reconstructed per-order book.
    """

    __slots__ = ("symbol", "window_size", "_mbo_ring", "_record_config")

    def __init__(self, symbol: str, window_size: int = 50_000):
        """
        Args:
            symbol (str): Trading symbol.
            window_size (int): Max number of MBO events retained.
        """
        self.symbol = symbol
        self.window_size = int(window_size)
        self._mbo_ring = None  # MboRing, attached by the engine
        self._record_config = None

    @property
    def ready(self) -> bool:
        """True once the MBO ring is attached and has received events."""
        return self._mbo_ring is not None and self._mbo_ring.n_available > 0

    def events(self):
        """
        Contiguous [n x 6] view of the recent MBO events (oldest first).

        Columns: ts, order_id, side, price, size, action. Empty when unattached.
        """
        if self._mbo_ring is None:
            import numpy as np
            return np.empty((0, 6), dtype=np.float64)
        return self._mbo_ring.window()

    def resting_size_at(self, side: int, price: float) -> float:
        """Reconstructed resting size at a price level (0.0 when unattached)."""
        if self._mbo_ring is None:
            return 0.0
        return self._mbo_ring.resting_size_at(side, price)

    def __len__(self) -> int:
        return 0 if self._mbo_ring is None else self._mbo_ring.n_available

    def __repr__(self) -> str:
        return f"MboProxy(symbol={self.symbol!r}, events={len(self)})"

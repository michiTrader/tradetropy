"""
Heatmap - Bookmap-style L2 liquidity heatmap (order flow).

Turns the live L2 order book's evolution into a time x price grid where each
cell is colored by the resting liquidity (size) at that price band and time
slice - the same visual model Bookmap / ATAS use. On top of the liquidity grid
it can draw the Best Bid / Ask lines and the executed-volume bubbles (reusing
the LargeTrades detection), for the full Bookmap look, each as an independently
toggleable legend group.

Like the other L2 overlays (``DeepWall`` / ``DeepReload``) it is tick-mounted
and takes the ``OrderbookProxy`` in its constructor; detection is pure and
causal (``order_flow/_core.py``), shared identically by backtest, live and
replay. With no book attached the grid is empty (a plain backtest with no depth
yields nothing).

Beyond the visual layer it exposes a public query API so a strategy can read
the heatmap causally from ``on_data()`` (liquidity at a price, the hottest
levels, persistent walls, the full grid). The handle returned by
``add_indicator`` delegates those queries to the indicator, so the same object
serves both the per-tick bands (``self.heat.best_bid[-1]``) and the queries
(``self.heat.hottest()``).

Usage::

    self.ticks = self.subscribe_ticks("BTCUSDT", window_size=5000)
    self.book  = self.subscribe_orderbook("BTCUSDT", depth=50, record=True)
    self.heat  = self.add_indicator(
        Heatmap.refs(self.ticks),
        Heatmap(self.book, price_bucket_ticks=1, colormap="hot"),
    )

    def on_data(self):
        wall = self.heat.nearest_wall("ask")          # resistance liquidity
        if wall is not None and wall.persistence_ms > 3000:
            size_here = self.heat.liquidity_at(self.heat.mid)
            ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradetropy.ta.base import IndicatorPlotConfig
from tradetropy.ta.draw import Points, Rects, Segments
from tradetropy.ta.order_flow._core import (
    DEFAULT_BUY_COLOR,
    DEFAULT_SELL_COLOR,
    build_bubble_columns,
    build_heatmap_grid,
    detect_large_trades,
    find_heatmap_walls,
    heatmap_color_bounds,
    infer_price_tick,
    merge_heatmap_grids,
    merge_live_events,
    resolve_persistence,
)
from tradetropy.ta.order_flow.deep_l2 import _L2BookIndicator

# Best Bid / Ask line colors (object defaults, theme-independent).
DEFAULT_BBO_BID_COLOR = "#0ECB81"   # green - best bid
DEFAULT_BBO_ASK_COLOR = "#F6465D"   # red - best ask

# Built-in colormaps: ordered hex stops from cold (low liquidity) to hot (high).
# 'hot' intentionally avoids green: it runs blue -> yellow -> orange -> red. The
# blue stops keep a low green channel so the blue->yellow interpolation midtone
# reads as tan/amber (R >= G) rather than olive-green.
_COLORMAPS = {
    "hot": [
        "#0A2B4A", "#145A82", "#399BC2", "#36A2E1",
        "#ffd500", "#ff9500", "#ff5a00", "#e00000",
    ],
    "viridis": [
        "#440154", "#443983", "#31688e", "#21918c",
        "#35b779", "#90d743", "#fde725",
    ],
    "greyscale": ["#101010", "#3a3a3a", "#707070", "#a8a8a8", "#e8e8e8"],
}


def _hex_to_rgb(h: str) -> tuple:
    """Convert a '#rrggbb' string to an (r, g, b) int tuple."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interp_colormap(norm: np.ndarray, stops: list) -> list:
    """
    Map normalized values in [0, 1] to hex colors along a stop gradient.

    Args:
        norm (np.ndarray): Values in [0, 1] [M].
        stops (list[str]): Ordered hex color stops (cold -> hot).

    Returns:
        list[str]: One '#rrggbb' color per input value.
    """
    rgb = np.array([_hex_to_rgb(s) for s in stops], dtype=np.float64)
    n_stops = len(stops)
    x = np.clip(np.asarray(norm, dtype=np.float64), 0.0, 1.0) * (n_stops - 1)
    lo = np.floor(x).astype(np.int64)
    hi = np.clip(lo + 1, 0, n_stops - 1)
    frac = (x - lo).reshape(-1, 1)
    col = rgb[lo] * (1.0 - frac) + rgb[hi] * frac
    col = np.clip(np.round(col), 0, 255).astype(np.int64)
    return ["#%02x%02x%02x" % (r, g, b) for r, g, b in col]


@dataclass(frozen=True)
class LiquidityLevel:
    """A single resting-liquidity level surfaced by the heatmap query API."""

    price: float            # bucket center price
    size: float             # resting size at the level
    notional: float         # size * price
    side: str               # 'bid' | 'ask'
    persistence_ms: int     # how long liquidity has continuously rested here
    distance_ticks: int     # signed distance to mid in price buckets


@dataclass(frozen=True)
class LiquidityColumn:
    """The liquidity profile (price -> size) at a single time slice."""

    ts: int                    # column timestamp (ms)
    price_centers: np.ndarray  # [P] bucket center prices
    bid_size: np.ndarray       # [P] bid resting size per bucket
    ask_size: np.ndarray       # [P] ask resting size per bucket


@dataclass(frozen=True)
class HeatmapGrid:
    """The full causal time x price liquidity grid."""

    col_ts: np.ndarray       # [T] column timestamps (ms)
    col_left: np.ndarray     # [T] column start ts (ms)
    col_right: np.ndarray    # [T] column end ts (ms)
    price_edges: np.ndarray  # [P+1] price-bucket edges
    price_centers: np.ndarray  # [P] price-bucket centers
    bid: np.ndarray          # [T x P] bid resting size
    ask: np.ndarray          # [T x P] ask resting size
    levels: int              # book depth K
    price_bucket: float      # bucket width used


class Heatmap(_L2BookIndicator):
    """
    Bookmap-style L2 liquidity heatmap with a public query API.

    Source columns (use ``Heatmap.refs(tick_proxy)``): ``[ts, price, volume,
    flags, bid, ask]``; the order book is supplied to the constructor.

    Outputs - [2 x N] per-tick bands (best bid / ask as-of each tick, NaN before
    the book exists):
        row 0 : best_bid, row 1 : best_ask.

    Visual layers (via ``draw()``, each a toggleable legend group):
        - 'Liquidity' : resting-size grid as colored Rects (the heatmap).
        - 'BBO'       : Best Bid / Ask lines as Segments (``show_bbo``).
        - 'Executed Volume' : executed-volume bubbles as Points
          (``show_bubbles``).

    Args:
        orderbook: OrderbookProxy supplying the L2 book (``book_window``).
        price_bucket (float | None): Absolute price width of a bucket. When
            None it is derived from ``tick_size * price_bucket_ticks``.
        price_bucket_ticks (int): Bucket width in ticks (used when
            ``price_bucket`` is None).
        tick_size (float | None): Price tick. When None it is inferred from the
            book's level spacing.
        time_bucket_ms (int | None): Time-column width in ms (None -> one column
            per book event; the recording granularity, so replay is exact).
        max_levels (int | None): Book levels per side to include (None -> all).
        min_size (float): Hide cells whose resting size is below this.
        color_scale (str | tuple): 'auto' (causal, Bookmap-style calibration:
            each column's color is fixed by the ``color_scale_pct`` percentile
            of the liquidity seen up to its own time and is never repainted by
            later liquidity) or an explicit (lo, hi) size range held constant
            across every column.
        color_scale_pct (float): Percentile of non-empty sizes used as the hot
            end when ``color_scale='auto'`` (applied causally / expanding).
        colormap (str): 'hot', 'viridis' or 'greyscale'.
        heat_alpha (float): Base fill alpha of the liquidity cells.
        max_history_columns (int | None): Cap on the accumulated drawn columns
            in live/replay (drops the oldest beyond the cap; None keeps the
            whole session, like Bookmap). Drawing-only; the query API is
            unaffected.
        show_bbo (bool): Draw the Best Bid / Ask lines.
        bbo_bid_color / bbo_ask_color (str): BBO line colors.
        show_bubbles (bool): Draw executed-volume bubbles.
        bubble_threshold / bubble_by / bubble_window / bubble_aggregate_ms:
            Large-trade detection parameters for the bubbles (see LargeTrades).
        buy_color / sell_color (str): Bubble colors by aggressor side.
        persistence_ms (int): Minimum resting duration for the wall queries.
        wall_min_size (float): Absolute minimum size for the wall queries.
        wall_rel_multiple (float): Relative wall threshold (x median size).
    """

    name = "heatmap"
    output_names = ["best_bid", "best_ask"]

    # Marks this indicator as a query provider so add_indicator wires the
    # returned proxy to delegate public queries here (see MultiBandProxy).
    exposes_query_api = True

    def __init__(
        self,
        orderbook=None,
        *,
        price_bucket: "float | None" = None,
        price_bucket_ticks: int = 1,
        tick_size: "float | None" = None,
        time_bucket_ms: "int | None" = None,
        max_levels: "int | None" = None,
        min_size: float = 0.0,
        color_scale="auto",
        color_scale_pct: float = 99.0,
        colormap: str = "hot",
        heat_alpha: float = 0.85,
        max_history_columns: "int | None" = None,
        show_bbo: bool = True,
        bbo_bid_color: str = DEFAULT_BBO_BID_COLOR,
        bbo_ask_color: str = DEFAULT_BBO_ASK_COLOR,
        show_bubbles: bool = True,
        bubble_threshold="p99",
        bubble_by: str = "volume",
        bubble_window: int = 2000,
        bubble_aggregate_ms: int = 0,
        buy_color: str = DEFAULT_BUY_COLOR,
        sell_color: str = DEFAULT_SELL_COLOR,
        persistence_ms: int = 2000,
        wall_min_size: float = 0.0,
        wall_rel_multiple: float = 4.0,
    ):
        super().__init__(orderbook)
        if colormap not in _COLORMAPS:
            raise ValueError(
                f"colormap must be one of {sorted(_COLORMAPS)}, not {colormap!r}"
            )
        self.price_bucket = None if price_bucket is None else float(price_bucket)
        self.price_bucket_ticks = int(price_bucket_ticks)
        self.tick_size = None if tick_size is None else float(tick_size)
        self.time_bucket_ms = time_bucket_ms
        self.max_levels = max_levels
        self.min_size = float(min_size)
        self.color_scale = color_scale
        self.color_scale_pct = float(color_scale_pct)
        self.colormap = colormap
        self.heat_alpha = float(heat_alpha)
        self.max_history_columns = (
            None if max_history_columns is None else int(max_history_columns))
        self.show_bbo = bool(show_bbo)
        self.bbo_bid_color = bbo_bid_color
        self.bbo_ask_color = bbo_ask_color
        self.show_bubbles = bool(show_bubbles)
        self.bubble_threshold = bubble_threshold
        self.bubble_by = bubble_by
        self.bubble_window = int(bubble_window)
        self.bubble_aggregate_ms = int(bubble_aggregate_ms)
        self.buy_color = buy_color
        self.sell_color = sell_color
        self.persistence_ms = int(persistence_ms)
        self.wall_min_size = float(wall_min_size)
        self.wall_rel_multiple = float(wall_rel_multiple)

        # Populated by calculate() for the plotting layer.
        self._grid: dict = {}
        self._bbo: dict = {}
        self._bubble_events: dict = {}
        # Append-only accumulation of the DRAWN grid for live/replay so the
        # heatmap keeps the full session history (Bookmap-style) and past
        # columns keep their frozen color calibration (never repainted). This is
        # a drawing concern only; the query API stays causal (reads book_window).
        self._grid_accum: dict = {}
        # On-demand query cache (keyed on the book's latest ts). The grid, the
        # per-side persistence and the computed walls are memoized together so a
        # strategy calling several queries per tick recomputes each at most once
        # (the derived caches are invalidated whenever the grid is rebuilt).
        self._q_cache_ts = None
        self._q_cache_grid = None
        self._q_cache_pers = None
        self._q_cache_walls: dict = {}

        self.plot_config = IndicatorPlotConfig(
            overlay=True, renderer="none", exclude_from_autoscale=False,
            name="Heatmap",
        )

    def display_name(self) -> str:
        return "Heatmap"

    # ── grid building ─────────────────────────────────────────────────────────

    def _resolve_bucket(self, book: dict) -> float:
        """Resolve the price-bucket width (explicit, tick-based or inferred)."""
        if self.price_bucket is not None:
            return self.price_bucket
        tick = self.tick_size
        if tick is None:
            tick = infer_price_tick(book, default=1.0)
        return float(tick) * max(int(self.price_bucket_ticks), 1)

    def _grid_from_book(self, book: dict) -> dict:
        """Build a heatmap grid from a book_window dict with this config."""
        return build_heatmap_grid(
            book, price_bucket=self._resolve_bucket(book),
            time_bucket_ms=self.time_bucket_ms, max_levels=self.max_levels,
            min_size=self.min_size,
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Build the heatmap grid and emit the [2 x N] best-bid/ask bands.

        Stores the grid, Best Bid/Ask series and bubble events for ``draw()``;
        the bands expose the causal best bid / ask as-of each tick for
        ``on_data()``.
        """
        n = len(source) if source.ndim == 2 else 0
        out = np.full((2, max(n, 0)), np.nan, dtype=np.float64)
        if n == 0:
            self._grid, self._bbo, self._bubble_events = {}, {}, {}
            return out

        tick_ts = source[:, 0].astype(np.int64)
        book = self._book_window()
        if book is not None and len(book.get("ts", [])):
            self._grid = self._grid_from_book(book)
            book_ts = np.asarray(book["ts"], dtype=np.int64)
            bb = np.asarray(book["bid_px"])[:, 0].astype(np.float64)
            ba = np.asarray(book["ask_px"])[:, 0].astype(np.float64)
            self._bbo = {"ts": book_ts, "bid": bb, "ask": ba}
            # Anchor best bid/ask as-of each tick (causal).
            idx = np.searchsorted(book_ts, tick_ts, side="right") - 1
            valid = idx >= 0
            if valid.any():
                out[0, valid] = bb[idx[valid]]
                out[1, valid] = ba[idx[valid]]
        else:
            self._grid, self._bbo = {}, {}

        self._bubble_events = self._detect_bubbles(source) if self.show_bubbles else {}
        return out

    def _detect_bubbles(self, source: np.ndarray) -> dict:
        """Run the large-trade detection used for the volume bubbles."""
        ts = source[:, 0].astype(np.float64)
        price = source[:, 1].astype(np.float64)
        volume = source[:, 2].astype(np.float64)
        flags = source[:, 3] if source.shape[1] > 3 else None
        bid = source[:, 4] if source.shape[1] > 4 else None
        ask = source[:, 5] if source.shape[1] > 5 else None
        res = detect_large_trades(
            ts, price, volume, flags=flags, bid=bid, ask=ask,
            threshold=self.bubble_threshold, by=self.bubble_by,
            window=self.bubble_window, aggregate_ms=self.bubble_aggregate_ms,
        )
        return res["events"]

    def live_refresh(self, proxy) -> None:
        """
        Re-run calculate over the live tick window, then persist the drawing.

        Two accumulations keep the DRAWN heatmap stable as the tick / book
        windows slide (a drawing concern only; the query API stays causal):

        - The liquidity grid is accumulated append-only (``_accumulate_grid``)
          so the heatmap keeps the full session history instead of only the
          current book window, and each new column's color calibration is frozen
          the moment it is drawn - the past is never repainted (Bookmap-style).
        - The executed-volume bubbles are accumulated exactly like
          ``LargeTrades`` so replay shows the same bubbles as the backtest
          within the visible window (a bubble detected while its tick was fresh
          survives after the tick drifts into the relative-threshold warmup band
          at the ring's left edge). Bubble accumulation is skipped under burst
          aggregation (``bubble_aggregate_ms != 0``), whose in-progress bursts
          would accumulate partial duplicates.
        """
        super().live_refresh(proxy)
        self._accumulate_grid()
        if not self.show_bubbles or self.bubble_aggregate_ms != 0:
            return
        if proxy is None or len(proxy) == 0:
            return
        try:
            window_ts = np.asarray(proxy.ts[:], dtype=np.int64)
        except Exception:
            return
        if window_ts.size == 0:
            return
        merged, self._bubble_accum_ts = merge_live_events(
            getattr(self, "_bubble_accum", None), self._bubble_events, window_ts,
            getattr(self, "_bubble_accum_ts", None),
        )
        self._bubble_events = merged
        self._bubble_accum = merged

    def _explicit_scale(self) -> bool:
        """True when color_scale is an explicit (lo, hi) range (not 'auto')."""
        return (isinstance(self.color_scale, (tuple, list))
                and len(self.color_scale) == 2)

    def _accumulate_grid(self) -> None:
        """
        Merge the window grid into the append-only accumulated grid (live/replay).

        The sizes are merged by ``merge_heatmap_grids`` (full outer join on
        ``col_ts`` with an element-wise MAX on overlapping columns), so a
        time-bucketed column keeps accumulating its true peak across refreshes
        instead of freezing at whatever snapshots one window happened to hold.
        The per-column colors are then recomputed over the WHOLE merged grid
        with the causal expanding calibration (``heatmap_color_bounds``): a past
        column's sizes no longer change (its time bucket is closed), so its
        color is stable and never repainted, while the calibration matches a
        single backtest pass exactly (both expand from column 0). ``draw()``
        reads the accumulator so it shows the full history. Resets on a replay
        rewind; no-op when the current window produced no grid.
        """
        new = getattr(self, "_grid", {}) or {}
        new_ts = np.asarray(new.get("col_ts", []), dtype=np.int64)
        if new_ts.size == 0:
            return

        accum = self._grid_accum or {}
        acc_ts = np.asarray(accum.get("col_ts", []), dtype=np.int64)
        rewind = bool(acc_ts.size and new_ts[-1] < acc_ts[-1])

        merged = merge_heatmap_grids(
            {} if rewind else accum, new,
            max_history_columns=self.max_history_columns,
        )
        merged.pop("_rewind", None)

        # Recompute the causal expanding color calibration over the merged grid.
        # Past columns' sizes are stable (their bucket is closed) so their color
        # is not repainted, and the result equals a single backtest pass.
        lo, hi = heatmap_color_bounds(
            np.asarray(merged["bid"], dtype=np.float64),
            np.asarray(merged["ask"], dtype=np.float64),
            pct=self.color_scale_pct, scale=self.color_scale,
        )
        merged["color_lo"] = lo
        merged["color_hi"] = hi

        # Store ONLY in the accumulator. self._grid stays the engine-owned window
        # grid (rebuilt every tick by _feed.py::calculate to feed the best_bid /
        # best_ask output bands); draw() reads the accumulator so the drawn
        # heatmap is immune to those per-tick overwrites (otherwise the engine
        # thread clobbers the accumulated history between refreshes and old
        # columns flicker in and out).
        self._grid_accum = merged

    # ── public query API (causal, callable from on_data()) ─────────────────────

    def _live_grid(self) -> "dict | None":
        """Build (and cache) the causal grid from the current book window."""
        book = self._book_window()
        if book is None or len(book.get("ts", [])) == 0:
            return None
        last_ts = int(np.asarray(book["ts"], dtype=np.int64)[-1])
        if self._q_cache_ts == last_ts and self._q_cache_grid is not None:
            return self._q_cache_grid
        grid = self._grid_from_book(book)
        self._q_cache_ts = last_ts
        self._q_cache_grid = grid
        # New grid -> invalidate the derived (persistence / walls) caches.
        self._q_cache_pers = None
        self._q_cache_walls = {}
        return grid

    def _live_persistence(self, grid: dict) -> tuple:
        """
        Per-side persistence (bid, ask) for the current grid, memoized per ts.

        Returns:
            tuple[np.ndarray, np.ndarray]: (bid_pers, ask_pers) arrays [P], the
            continuous resting duration in ms per price bucket up to now.
        """
        if self._q_cache_pers is None:
            col_ts = grid["col_ts"]
            self._q_cache_pers = (
                resolve_persistence(grid["bid"], col_ts),
                resolve_persistence(grid["ask"], col_ts),
            )
        return self._q_cache_pers

    @property
    def stale(self) -> bool:
        """True when no synced book is available (queries return NaN/empty)."""
        return self._book is None or getattr(self._book, "stale", True)

    @property
    def best_bid(self) -> float:
        """Current best bid price, or NaN when stale (scalar query)."""
        return float("nan") if self._book is None else float(self._book.best_bid)

    @property
    def best_ask(self) -> float:
        """Current best ask price, or NaN when stale (scalar query)."""
        return float("nan") if self._book is None else float(self._book.best_ask)

    @property
    def mid(self) -> float:
        """Current mid price, or NaN when stale (scalar query)."""
        return float("nan") if self._book is None else float(self._book.mid)

    def _bucket_index(self, grid: dict, price: float) -> int:
        """Bucket index of ``price`` in the grid, or -1 when out of range."""
        centers = grid["price_centers"]
        if centers.size == 0:
            return -1
        bucket = grid["price_bucket"]
        idx = int(round((float(price) - centers[0]) / bucket))
        return idx if 0 <= idx < centers.size else -1

    def _pick_side(self, price: float, side: "str | None") -> str:
        """Resolve the side to query ('bid' below mid, 'ask' above)."""
        if side in ("bid", "ask"):
            return side
        mid = self.mid
        if np.isnan(mid):
            return "bid"
        return "bid" if float(price) <= mid else "ask"

    def liquidity_at(self, price: float, side: "str | None" = None) -> float:
        """
        Resting liquidity size at ``price`` now (0 if empty, NaN if unavailable).

        Args:
            price (float): Query price (mapped to its bucket).
            side (str | None): 'bid' / 'ask', or None to auto-pick by the mid.

        Returns:
            float: Resting size at the price bucket, 0.0 when the bucket is in
            range but empty, or NaN when there is no book / price is out of grid.
        """
        grid = self._live_grid()
        if grid is None:
            return float("nan")
        idx = self._bucket_index(grid, price)
        if idx < 0:
            return float("nan")
        mat = grid[self._pick_side(price, side)]
        return float(mat[-1, idx])

    def notional_at(self, price: float, side: "str | None" = None) -> float:
        """Resting notional (size * price) at ``price`` now (NaN if unavailable)."""
        size = self.liquidity_at(price, side)
        return size * float(price) if np.isfinite(size) else float("nan")

    def _levels_from_column(self, grid: dict, mat: np.ndarray, side: str,
                            pers: np.ndarray) -> list:
        """Build LiquidityLevel objects for the non-empty buckets of a column."""
        centers = grid["price_centers"]
        bucket = grid["price_bucket"]
        mid = self.mid
        out = []
        for i in np.nonzero(mat > 0.0)[0]:
            price = float(centers[i])
            dist = 0 if np.isnan(mid) else int(round((price - mid) / bucket))
            out.append(LiquidityLevel(
                price=price, size=float(mat[i]), notional=float(mat[i]) * price,
                side=side, persistence_ms=int(pers[i]), distance_ticks=dist,
            ))
        return out

    def hottest(self, side: "str | None" = None, n: int = 1) -> list:
        """
        The ``n`` thickest resting levels now, descending by size.

        Args:
            side (str | None): 'bid' / 'ask', or None for both sides.
            n (int): Number of levels to return.

        Returns:
            list[LiquidityLevel]: Up to ``n`` levels (empty when no book).
        """
        grid = self._live_grid()
        if grid is None:
            return []
        bid_pers, ask_pers = self._live_persistence(grid)
        pers_by_side = {"bid": bid_pers, "ask": ask_pers}
        levels = []
        sides = ("bid", "ask") if side is None else (side,)
        for sd in sides:
            mat = grid[sd]
            levels.extend(
                self._levels_from_column(grid, mat[-1], sd, pers_by_side[sd])
            )
        levels.sort(key=lambda lv: lv.size, reverse=True)
        return levels[: max(int(n), 0)]

    def _all_walls(self, min_size: float) -> list:
        """
        All persistent walls now (both sides), memoized per (ts, min_size).

        The public ``walls`` / ``nearest_wall`` filter this by side, so several
        wall queries in one ``on_data()`` share a single detection pass (and the
        already-memoized per-side persistence).

        Args:
            min_size (float): Absolute wall size threshold to use.

        Returns:
            list[LiquidityLevel]: The current walls, ascending by price.
        """
        grid = self._live_grid()
        if grid is None:
            return []
        key = float(min_size)
        cached = self._q_cache_walls.get(key)
        if cached is not None:
            return cached
        bid_pers, ask_pers = self._live_persistence(grid)
        w = find_heatmap_walls(
            grid, min_size=key, rel_multiple=self.wall_rel_multiple,
            persistence_ms=self.persistence_ms,
            bid_pers=bid_pers, ask_pers=ask_pers,
        )
        mid = self.mid
        bucket = grid["price_bucket"]
        out = []
        for p, sz, sd, pers in zip(w["price"], w["size"], w["side"],
                                   w["persistence_ms"]):
            side_name = "bid" if sd > 0 else "ask"
            dist = 0 if np.isnan(mid) else int(round((float(p) - mid) / bucket))
            out.append(LiquidityLevel(
                price=float(p), size=float(sz), notional=float(sz) * float(p),
                side=side_name, persistence_ms=int(pers), distance_ticks=dist,
            ))
        self._q_cache_walls[key] = out
        return out

    def walls(self, side: "str | None" = None,
              min_size: "float | None" = None) -> list:
        """
        Persistent resting-liquidity walls now (see ``find_heatmap_walls``).

        Args:
            side (str | None): Filter to 'bid' / 'ask', or None for both.
            min_size (float | None): Override the absolute wall size threshold.

        Returns:
            list[LiquidityLevel]: The current walls, ascending by price.
        """
        ms = self.wall_min_size if min_size is None else float(min_size)
        allw = self._all_walls(ms)
        if side is None:
            return list(allw)
        return [lv for lv in allw if lv.side == side]

    def nearest_wall(self, side: str,
                     min_size: "float | None" = None) -> "LiquidityLevel | None":
        """
        Nearest persistent wall to the mid on ``side`` ('bid' below / 'ask' above).

        Returns:
            LiquidityLevel | None: The closest wall, or None when there is none.
        """
        mid = self.mid
        candidates = self.walls(side=side, min_size=min_size)
        if not candidates:
            return None
        if np.isnan(mid):
            return candidates[0]
        return min(candidates, key=lambda lv: abs(lv.price - mid))

    def persistence(self, price: float, side: "str | None" = None) -> float:
        """
        Milliseconds liquidity has continuously rested at ``price`` up to now.

        Returns:
            float: Persistence in ms (0 when currently empty, NaN if no book).
        """
        grid = self._live_grid()
        if grid is None:
            return float("nan")
        idx = self._bucket_index(grid, price)
        if idx < 0:
            return float("nan")
        sd = self._pick_side(price, side)
        bid_pers, ask_pers = self._live_persistence(grid)
        pers = bid_pers if sd == "bid" else ask_pers
        return float(pers[idx])

    def column(self, ts: "int | None" = None) -> "LiquidityColumn | None":
        """
        Liquidity profile at ``ts`` (or the latest column when ts is None).

        Returns:
            LiquidityColumn | None: The column, or None when there is no book.
        """
        grid = self._live_grid()
        if grid is None:
            return None
        col_ts = grid["col_ts"]
        j = col_ts.size - 1
        if ts is not None:
            pos = int(np.searchsorted(col_ts, int(ts), side="right") - 1)
            if pos < 0:
                return None
            j = pos
        return LiquidityColumn(
            ts=int(col_ts[j]), price_centers=grid["price_centers"].copy(),
            bid_size=grid["bid"][j].copy(), ask_size=grid["ask"][j].copy(),
        )

    def grid(self) -> "HeatmapGrid | None":
        """
        The full causal time x price liquidity grid, or None when no book.

        Returns:
            HeatmapGrid | None: The current grid snapshot.
        """
        g = self._live_grid()
        if g is None:
            return None
        return HeatmapGrid(
            col_ts=g["col_ts"].copy(), col_left=g["col_left"].copy(),
            col_right=g["col_right"].copy(), price_edges=g["price_edges"].copy(),
            price_centers=g["price_centers"].copy(), bid=g["bid"].copy(),
            ask=g["ask"].copy(), levels=int(g["levels"]),
            price_bucket=float(g["price_bucket"]),
        )

    # ── visual layers ──────────────────────────────────────────────────────────

    def _column_color_bounds(self, grid: dict) -> tuple:
        """
        Resolve per-column (lo, hi) color bounds for the liquidity cells.

        When the grid carries frozen ``color_lo`` / ``color_hi`` (accumulated
        live/replay so the past is never repainted) they are used as-is;
        otherwise the causal calibration is computed from the grid: ``'auto'``
        yields the Bookmap-style expanding percentile per column (a column's
        color depends only on the liquidity seen up to it), and an explicit
        ``(lo, hi)`` yields a constant range across every column.

        Args:
            grid (dict): The grid being drawn.

        Returns:
            tuple[np.ndarray, np.ndarray]: (lo[T], hi[T]) per-column bounds.
        """
        lo = grid.get("color_lo")
        hi = grid.get("color_hi")
        if lo is not None and hi is not None:
            return (np.asarray(lo, dtype=np.float64),
                    np.asarray(hi, dtype=np.float64))
        return heatmap_color_bounds(
            grid["bid"], grid["ask"],
            pct=self.color_scale_pct, scale=self.color_scale,
        )

    def _liquidity_rects(self, grid: dict) -> "Rects | None":
        """Build the colored liquidity-cell Rects for the heatmap layer."""
        bid = grid["bid"]
        ask = grid["ask"]
        if bid.size == 0:
            return None
        edges = grid["price_edges"]
        # Center each column on its representative timestamp (col_ts) instead of
        # letting it start there, mirroring how CVD / DeltaBars center bars on
        # bar_ts (see DeltaBars._half_width). The builder keeps col_left/col_right
        # as the true time bounds (used by the query API and parity tests); only
        # the drawn rectangle is shifted so a column sits centered under the
        # candle at the same time, contiguous with its neighbours.
        col_ts = np.asarray(grid["col_ts"], dtype=np.int64)
        width = (np.asarray(grid["col_right"], dtype=np.int64)
                 - np.asarray(grid["col_left"], dtype=np.int64))
        half = width // 2
        left = col_ts - half
        right = left + width
        # Per-column causal bounds: each column normalizes against the liquidity
        # known at its own time, so the past keeps its color (Bookmap-style).
        lo_col, hi_col = self._column_color_bounds(grid)
        span_col = hi_col - lo_col
        span_col = np.where(span_col > 0.0, span_col, 1.0)

        x0, x1, y0, y1, norms = [], [], [], [], []
        for mat in (bid, ask):
            rows, cols = np.nonzero(mat > 0.0)
            if rows.size == 0:
                continue
            x0.extend(left[rows].tolist())
            x1.extend(right[rows].tolist())
            y0.extend(edges[cols].tolist())
            y1.extend(edges[cols + 1].tolist())
            cell = (mat[rows, cols] - lo_col[rows]) / span_col[rows]
            norms.extend(np.clip(cell, 0.0, 1.0).tolist())
        if not norms:
            return None

        norm = np.asarray(norms, dtype=np.float64)
        colors = _interp_colormap(norm, _COLORMAPS[self.colormap])
        alpha = (self.heat_alpha * (0.25 + 0.75 * norm)).tolist()
        return Rects(
            x0=x0, x1=x1, y0=y0, y1=y1, fill_color=colors,
            fill_alpha=alpha, line_color=None, line_alpha=0.0,
        )

    def _bbo_segments(self) -> list:
        """Build the Best Bid / Ask line Segments."""
        bbo = getattr(self, "_bbo", {}) or {}
        ts = np.asarray(bbo.get("ts", []), dtype=np.int64)
        if ts.size < 2:
            return []
        prims = []
        for key, color in (("bid", self.bbo_bid_color),
                           ("ask", self.bbo_ask_color)):
            y = np.asarray(bbo[key], dtype=np.float64)
            finite = np.isfinite(y[:-1]) & np.isfinite(y[1:])
            if not finite.any():
                continue
            prims.append(Segments(
                x0=ts[:-1][finite].tolist(), y0=y[:-1][finite].tolist(),
                x1=ts[1:][finite].tolist(), y1=y[1:][finite].tolist(),
                color=color, alpha=0.9, width=1.0,
            ))
        return prims

    def _bubble_points(self) -> list:
        """Build the executed-volume bubble Points (reused LargeTrades style)."""
        events = getattr(self, "_bubble_events", {}) or {}
        style = {
            "scale": "sqrt", "min_size": 6.0, "max_size": 34.0,
            "buy_color": self.buy_color, "sell_color": self.sell_color,
            "label": None, "by": self.bubble_by,
        }
        cols = build_bubble_columns(events, style)
        if len(cols["ts"]) == 0:
            return []
        return [Points(
            x=list(cols["ts"]), y=list(cols["price"]),
            color=list(cols["color"]), alpha=0.9,
            size=list(cols["size"]), marker="circle",
            fill=True, fill_alpha=0.55, line_width=0.8,
        )]

    def draw(self, cfg=None, *, interval_ms=None) -> dict:
        """
        Emit the heatmap layers as toggleable legend groups.

        Returns a dict mapping each legend name to its primitives:
            - 'Heatmap' : the colored liquidity grid (Rects).
            - 'BBO'     : Best Bid / Ask lines (Segments), when ``show_bbo``.
            - 'Executed Volume' : executed-volume bubbles (Points), when
              ``show_bubbles``.
        """
        groups: dict[str, list] = {}
        # Prefer the append-only accumulated grid (live/replay, frozen colors and
        # full history); fall back to the window grid in backtest, where the
        # single calculate() pass already holds the whole series and there is no
        # accumulation.
        grid = (getattr(self, "_grid_accum", None)
                or getattr(self, "_grid", {}) or {})
        if grid:
            rects = self._liquidity_rects(grid)
            if rects is not None:
                groups["Heatmap"] = [rects]
        if self.show_bbo:
            seg = self._bbo_segments()
            if seg:
                groups["BBO"] = seg
        if self.show_bubbles:
            pts = self._bubble_points()
            if pts:
                groups["Executed Volume"] = pts
        return groups

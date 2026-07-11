"""
Volume Profile indicators.

Two indicators are provided, both built on the shared core in
``tradetropy.ta._volume_profile``:

- VolumeProfile
    Kline-based. Each candle's volume is distributed uniformly across the price
    levels spanned by its high-low range (the standard approximation used when
    only OHLCV is available). Aggressor side is approximated from candle
    direction (close vs open).

- TickVolumeProfile
    Tick-based and more precise. Each trade is binned at its real price level
    and classified as buy / sell aggressor from the tick ``flags`` (with a
    tick-rule fallback), producing a real bid/ask split and delta.

Both expose POC / VAH / VAL as developing series (no look-ahead) accessible from
``on_data()``, reset every ``period`` (e.g. "1d"). The finalized per-period
histograms are stored on the instance for the plotting layer to render the
horizontal volume-by-price bars.

Usage (klines)::

    self.btc = self.subscribe_ohlc("BTCUSDT", 60_000)
    self.vp  = self.add_indicator(
        VolumeProfile.refs(self.btc),
        VolumeProfile(period="1d"),
    )
    # in on_data():
    poc = self.vp.poc[-1]

Usage (ticks)::

    self.ticks = self.subscribe_ticks("BTCUSDT")
    self.tvp   = self.add_indicator(
        TickVolumeProfile.refs(self.ticks),
        TickVolumeProfile(period="1d"),
    )
"""

from __future__ import annotations

import numpy as np

from tradetropy.core.constants import parse_timeframe
from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.exceptions import ConfigError
from tradetropy.ta._volume_profile import (
    level_price,
    infer_tick_size,
    kline_contributions,
    tick_contributions,
    histogram_from_contributions,
    detect_volume_nodes,
    scan_developing_profiles,
    scan_rolling_profiles,
    resolve_period_ids,
    volume_profile_bar_arrays,
    DEFAULT_VP_BUY,
    DEFAULT_VP_SELL,
    DEFAULT_VP_POC,
    DEFAULT_VP_HVN,
    DEFAULT_VP_LVN,
)
from tradetropy.ta.draw import HBars, HLines, Points


def _kline_contributions(high, low, vol, is_bull, tick_size):
    """Deprecated local alias; use kline_contributions from the core module."""
    return kline_contributions(high, low, vol, is_bull, tick_size)


# =====
# Shared base
# =====
class _VolumeProfileBase(Indicator):
    """
    Common machinery for the Volume Profile indicators.

    Subclasses implement ``_build_contributions(source)`` returning the per-row
    period ids plus the contribution arrays consumed by the developing scan.
    This base runs the scan, stores the finalized per-period histograms on the
    instance (``self.profiles_``) and returns the [3 x N] POC/VAH/VAL series.
    """

    category = "volume"

    #: Whether the source proxy is tick-based (TickProxy) or kline-based.
    #: Drives how node ranges read the source buffer in compute_nodes().
    source_is_tick = False

    def __init__(
        self,
        period="1d",
        tick_size: float | None = None,
        bins: int = 100,
        value_area_pct: float = 0.70,
        va_recompute_every: int = 1,
        view: str = "session",
        nodes: str | None = None,
        node_prominence: float = 0.1,
        max_nodes: int | None = None,
        anchor=None,
        buy_color: str = DEFAULT_VP_BUY,
        sell_color: str = DEFAULT_VP_SELL,
        poc_color: str = DEFAULT_VP_POC,
        hvn_color: str = DEFAULT_VP_HVN,
        lvn_color: str = DEFAULT_VP_LVN,
    ):
        """
        Initialize a volume profile indicator.

        Args:
            period (str or int): Reset period for the profile (e.g. "1d", "1w",
                or milliseconds). Each period accumulates its own histogram.
            tick_size (float or None): Price level size. If None it is inferred
                from the price range and ``bins``.
            bins (int): Target number of levels across the range, used only when
                ``tick_size`` is None.
            value_area_pct (float): Fraction of volume defining the value area.
            va_recompute_every (int): Recompute the value area every N rows.
                1 = exact developing VAH/VAL on every row.
            view (str): How the histogram is laid out when plotting.
                "session" (default) draws one profile per period anchored at the
                period start and growing right (TradingView VPSV). "visible"
                aggregates every level across all periods into a single profile
                anchored at the right edge and growing left (TradingView VPVR).
            nodes (str or None): Enable HVN/LVN node detection accessible from
                on_data() via the proxy (``self.vp.hvn`` / ``.lvn`` / ``.nodes``).
                None (default) disables it; accessing the proxy nodes then raises
                ConfigError. 'hvn', 'lvn' or 'both' select which kinds to detect.
            node_prominence (float): Minimum node strength (0..1) to keep.
            max_nodes (int or None): Optional cap per node kind (strongest first).
            anchor: Where each period begins. None or 'utc' (default) keeps the
                legacy Unix-aligned cut (UTC midnight for '1d', Thursday-UTC for
                '1w'; intraday timeframes already fall on round clock marks).
                'data' starts the grid on the first row (e.g. 11:11 -> 11:11 next
                day). A timezone (str like 'America/New_York' or tzinfo) anchors
                to that zone's local midnight, DST-aware. A 2-tuple
                ``(tz, 'HH:MM')`` anchors to a session open time of day (e.g.
                ``('America/New_York', '17:00')`` for CME Globex). Only affects
                period-based cuts; intraday utc behaviour is unchanged.
            buy_color (str): Color of the buy-aggressor (ask) volume segment in
                the histogram. Fixed default, independent of the plot theme.
            sell_color (str): Color of the sell-aggressor (bid) volume segment.
            poc_color (str): Color of the point-of-control line.
            hvn_color (str): Color of the high-volume node markers.
            lvn_color (str): Color of the low-volume node markers.
        """
        if view not in ("session", "visible"):
            raise ValueError(f"view must be 'session' or 'visible', not {view!r}")
        if nodes not in (None, "hvn", "lvn", "both"):
            raise ConfigError(
                f"nodes must be None, 'hvn', 'lvn' or 'both', not {nodes!r}"
            )
        self.period = period
        self.period_ms = parse_timeframe(period)
        self.tick_size = tick_size
        self.bins = bins
        self.value_area_pct = value_area_pct
        self.va_recompute_every = max(int(va_recompute_every), 1)
        self.view = view
        self.nodes = nodes
        self.node_prominence = node_prominence
        self.max_nodes = max_nodes
        self.anchor = anchor
        # Semantic histogram colors, theme-independent (live on the object).
        self.buy_color = buy_color
        self.sell_color = sell_color
        self.poc_color = poc_color
        self.hvn_color = hvn_color
        self.lvn_color = lvn_color
        # Validate the anchor eagerly so configuration errors surface at
        # construction time, not deep inside calculate. The dummy timestamp
        # only exercises the parsing/tz-resolution path.
        try:
            resolve_period_ids(
                np.array([0], dtype=np.int64), self.period_ms, anchor
            )
        except ValueError as exc:
            raise ConfigError(f"invalid anchor: {exc}") from exc

        self.output_names = ["poc", "vah", "val"]

        # Per-period histograms, filled by calculate for the plotting layer.
        self.profiles_: list[dict] = []
        self.tick_size_used_: float | None = None
        # pid -> period-start ts (ms), filled during calculate so that
        # period_profiles() can reconstruct the anchored x extent even when the
        # period grid is not Unix aligned (data/timezone/session anchors).
        self._pid_start_ms: dict[int, int] = {}

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            renderer="step",
            color=[self.poc_color, "#9CA3AF", "#9CA3AF"],
            line_width=[1.6, 1.0, 1.0],
            line_dash=["solid", "dashed", "dashed"],
            line_alpha=[0.95, 0.7, 0.7],
            front_dim=0,
        )

    @property
    def min_periods(self) -> int:
        return 1

    def default_refs(self, proxy):
        """ColumnRefs resolved from a proxy, so add_indicator(proxy, ind) works."""
        return type(self).refs(proxy)

    def display_name(self) -> str:
        return f"{type(self).__name__}({self.period})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        cls = type(self).__name__.lower()
        return f"{cls}_{self.period}_{symbol}"

    def _resolve_tick_size(self, prices: np.ndarray) -> float:
        if self.tick_size is not None and self.tick_size > 0:
            return float(self.tick_size)
        return infer_tick_size(prices, self.bins)

    def _empty_result(self) -> np.ndarray:
        return np.full((3, 0), np.nan, dtype=np.float64)

    def _run_scan(
        self,
        n_rows: int,
        period_id: np.ndarray,
        contrib_row: np.ndarray,
        c_level: np.ndarray,
        c_vol_bid: np.ndarray,
        c_vol_ask: np.ndarray,
        c_count: np.ndarray,
        period_start: np.ndarray | None = None,
    ) -> np.ndarray:
        poc, vah, val, periods = scan_developing_profiles(
            n_rows,
            period_id,
            contrib_row,
            c_level,
            c_vol_bid,
            c_vol_ask,
            c_count,
            value_area_pct=self.value_area_pct,
            va_recompute_every=self.va_recompute_every,
        )
        self.profiles_ = periods
        # Record the anchored start ts for every period id seen, so the plotting
        # layer can place each histogram at its true left edge under any anchor.
        self._pid_start_ms = {}
        if period_start is not None and len(period_start):
            pid_arr = np.asarray(period_id, dtype=np.int64)
            start_arr = np.asarray(period_start, dtype=np.int64)
            for pid, start in zip(pid_arr, start_arr):
                ipid = int(pid)
                if ipid not in self._pid_start_ms:
                    self._pid_start_ms[ipid] = int(start)
        out = np.full((3, n_rows), np.nan, dtype=np.float64)
        out[0] = poc
        out[1] = vah
        out[2] = val
        return out

    # ── Node access (HVN / LVN), causal and engine-agnostic ──────────────────
    def _node_range(self, ts: np.ndarray) -> "tuple[int | None, int | None]":
        """
        Causal [start, end] ts range whose profile feeds node detection.

        Default: the last *closed* period strictly before the current bar, so
        nodes never use look-ahead data. Subclasses (e.g. RollingVolumeProfile)
        override this with their own window.

        Args:
            ts (np.ndarray): Source timestamps held in the buffer (ascending).

        Returns:
            tuple[int | None, int | None]: (start_ms, end_ms); (None, None) when
            there is no closed period yet.
        """
        if len(ts) == 0:
            return None, None
        ts_now = int(ts[-1])
        # Period boundaries follow the configured anchor, so the "last closed
        # period" is found from the anchored grid rather than the Unix grid.
        pid_arr, start_arr = resolve_period_ids(
            np.array([ts_now], dtype=np.int64), self.period_ms, self.anchor
        )
        period_start = int(start_arr[0])
        prev_start = period_start - self.period_ms
        if prev_start < int(ts[0]):
            # The previous period is not fully in the buffer; fall back to the
            # part of it that is available (still causal — all <= ts_now).
            prev_start = None
        return prev_start, period_start - 1

    def compute_nodes(self, source) -> "tuple[list, list]":
        """
        Detect HVN/LVN over the causal range, reading the source proxy buffer.

        Computed on demand from inside on_data(): only the rows currently held in
        the source window are read, and only up to the current cursor, so the
        result is safe for backtesting. Returns (hvn, lvn) lists of VolumeNode.

        Raises:
            ConfigError: If the indicator was created with ``nodes=None``.
        """
        if self.nodes is None:
            raise ConfigError(
                f"{type(self).__name__} was created with nodes=None: does not expose "
                "HVN/LVN. Pass nodes='hvn'|'lvn'|'both' to enable them."
            )
        from tradetropy.ta._volume_profile import compute_range_profile

        ts = np.asarray(source.ts[:], dtype=np.int64)
        if len(ts) == 0:
            return [], []
        start, end = self._node_range(ts)

        common = dict(
            start=start, end=end,
            tick_size=self.tick_size, bins=self.bins,
            value_area_pct=self.value_area_pct,
            nodes=self.nodes, node_prominence=self.node_prominence,
            max_nodes=self.max_nodes,
        )
        if self.source_is_tick:
            prof = compute_range_profile(
                ts, is_tick=True,
                price=np.asarray(source.price[:], dtype=np.float64),
                volume=np.asarray(source.volume[:], dtype=np.float64),
                flags=np.asarray(source.flags[:], dtype=np.float64).astype(np.int64),
                **common,
            )
        else:
            prof = compute_range_profile(
                ts, is_tick=False,
                open_=np.asarray(source.open[:], dtype=np.float64),
                high=np.asarray(source.high[:], dtype=np.float64),
                low=np.asarray(source.low[:], dtype=np.float64),
                close=np.asarray(source.close[:], dtype=np.float64),
                volume=np.asarray(source.volume[:], dtype=np.float64),
                **common,
            )
        if not prof:
            return [], []
        return prof.get("hvn", []), prof.get("lvn", [])

    def period_profiles(self) -> list[dict]:
        """
        Normalized per-period histograms for the plotting layer.

        Converts the raw per-period scan output into render-ready dicts. Each
        period carries its time extent (derived from the period id and
        ``period_ms``) plus the finalized POC/VAH/VAL and the price-level matrix.

        Returns:
            list[dict]: One dict per period with keys:
                - 'ts_start' (int): Period start timestamp in ms.
                - 'ts_end'   (int): Period end timestamp in ms (exclusive).
                - 'poc'      (float): Final point of control price.
                - 'vah'      (float): Final value-area high.
                - 'val'      (float): Final value-area low.
                - 'max_vol'  (float): Largest level volume (for bar scaling).
                - 'prices'   (np.ndarray): Level prices.
                - 'volumes'  (np.ndarray): Total volume per level.
                - 'vol_bid'  (np.ndarray): Sell-aggressor (bid) volume per level.
                - 'vol_ask'  (np.ndarray): Buy-aggressor (ask) volume per level.
                - 'deltas'   (np.ndarray): Ask-minus-bid volume per level.
        """
        from tradetropy.models.footprint import _FP_LEVEL_COL, _FP_SCALAR_COL

        out: list[dict] = []
        for p in self.profiles_:
            levels = p["levels"]
            scal = p["scalars"]
            if len(levels) == 0:
                continue
            pid = int(p["pid"])
            prices = levels[:, _FP_LEVEL_COL["price"]]
            volumes = levels[:, _FP_LEVEL_COL["vol_total"]]
            vol_bid = levels[:, _FP_LEVEL_COL["vol_bid"]]
            vol_ask = levels[:, _FP_LEVEL_COL["vol_ask"]]
            deltas = levels[:, _FP_LEVEL_COL["delta"]]
            # Anchored start ts recorded during the scan; fall back to the
            # Unix-aligned formula only if absent (e.g. empty pid map).
            ts_start = int(self._pid_start_ms.get(pid, pid * self.period_ms))
            out.append({
                "ts_start": ts_start,
                "ts_end": ts_start + self.period_ms,
                "poc": float(scal[_FP_SCALAR_COL["poc_price"]]),
                "vah": float(scal[_FP_SCALAR_COL["vah"]]),
                "val": float(scal[_FP_SCALAR_COL["val"]]),
                "max_vol": float(volumes.max()) if len(volumes) else 0.0,
                "prices": prices,
                "volumes": volumes,
                "vol_bid": vol_bid,
                "vol_ask": vol_ask,
                "deltas": deltas,
            })
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """
        Emit the volume-by-price histogram as a single HBars primitive.

        The POC / VAH / VAL developing lines keep drawing as a step series from
        ``plot_config``; only the histogram (geometry, not a per-bar series) is
        emitted here. Colors come from the indicator object (buy_color /
        sell_color), so the histogram is independent of the plot theme. Returns
        an empty list when there are no finalized profiles yet.
        """
        profiles = self.period_profiles()
        if not profiles:
            return []
        arrays = volume_profile_bar_arrays(
            profiles,
            float(getattr(self, "tick_size_used_", 0.0) or 0.0),
            getattr(self, "view", "session"),
            buy_color=self.buy_color,
            sell_color=self.sell_color,
            interval_ms=interval_ms,
        )
        if arrays is None:
            return []
        return [HBars(
            y=arrays["y"], height=arrays["height"],
            left=arrays["left"], right=arrays["right"],
            color=arrays["color"], alpha=arrays["alpha"],
            hover=[
                ("Price", "@y{0,0.00}"),
                ("Buy", "@vol_buy{0,0.00}"),
                ("Sell", "@vol_sell{0,0.00}"),
                ("Total", "@volume{0,0.00}"),
            ],
            extra={
                "volume": arrays["volume"],
                "vol_buy": arrays["vol_buy"],
                "vol_sell": arrays["vol_sell"],
            },
        )]


# =====
# VolumeProfile (klines)
# =====
class VolumeProfile(_VolumeProfileBase):
    """
    Kline-based volume profile (uniform high-low distribution).

    Source columns (use ``VolumeProfile.refs(ohlc_proxy)``):
        [ts, open, high, low, close, volume]

    Each candle's volume is spread uniformly across the price levels between its
    low and high. Aggressor side is approximated from candle direction:
    a bullish candle (close >= open) counts as ask (buy) volume, a bearish one
    as bid (sell) volume.
    """

    name = "vp"

    @staticmethod
    def refs(ohlc_proxy):
        """
        Build the ColumnRef list for this indicator in the expected order.

        Args:
            ohlc_proxy (OhlcProxy): Proxy returned by subscribe_ohlc().

        Returns:
            list[ColumnRef]: [ts, open, high, low, close, volume] refs.
        """
        return [
            ohlc_proxy.ts_ref,
            ohlc_proxy.open_ref,
            ohlc_proxy.high_ref,
            ohlc_proxy.low_ref,
            ohlc_proxy.close_ref,
            ohlc_proxy.volume_ref,
        ]

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 6 or len(source) == 0:
            return self._empty_result()

        ts = source[:, 0].astype(np.int64)
        open_ = source[:, 1].astype(np.float64)
        high = source[:, 2].astype(np.float64)
        low = source[:, 3].astype(np.float64)
        close = source[:, 4].astype(np.float64)
        vol = source[:, 5].astype(np.float64)
        n = len(ts)

        tick_size = self._resolve_tick_size(np.concatenate([high, low]))
        self.tick_size_used_ = tick_size
        period_id, period_start = resolve_period_ids(ts, self.period_ms, self.anchor)

        is_bull = close >= open_

        rows, levels, v_bid, v_ask, counts = _kline_contributions(
            high, low, vol, is_bull, tick_size
        )

        return self._run_scan(
            n,
            period_id,
            rows,
            levels,
            v_bid,
            v_ask,
            counts,
            period_start=period_start,
        )


# =====
# RollingVolumeProfile (klines, sliding window)
# =====
class RollingVolumeProfile(VolumeProfile):
    """
    Kline-based volume profile over a sliding window of the last ``length`` bars.
    source: ts, open, high, low, close, volume [N×6]

    Unlike VolumeProfile (which resets every ``period`` and gives a saw-tooth
    developing series), this profile never resets: at each bar it aggregates the
    trailing ``length`` candles, producing *continuous* POC / VAH / VAL lines
    suitable for plotting and for reading in on_data()::

        self.rvp = self.add_indicator(self.btc, RollingVolumeProfile(length=200))
        # in on_data():
        if self.btc.close[-1] > self.rvp.vah[-1]:
            ...

    The single histogram exposed for plotting is the profile of the current
    (most recent) window, drawn at the right edge like TradingView's VPVR.
    """

    name = "rvp"

    def __init__(
        self,
        length: int = 200,
        tick_size: float | None = None,
        bins: int = 100,
        value_area_pct: float = 0.70,
        va_recompute_every: int = 1,
        nodes: str | None = None,
        node_prominence: float = 0.1,
        max_nodes: int | None = None,
        buy_color: str = DEFAULT_VP_BUY,
        sell_color: str = DEFAULT_VP_SELL,
        poc_color: str = DEFAULT_VP_POC,
        hvn_color: str = DEFAULT_VP_HVN,
        lvn_color: str = DEFAULT_VP_LVN,
    ):
        """
        Initialize a rolling volume profile.

        Args:
            length (int): Number of trailing candles aggregated into the profile.
            tick_size (float or None): Price level size; inferred when None.
            bins (int): Target number of levels, used only when tick_size is None.
            value_area_pct (float): Fraction of volume defining the value area.
            va_recompute_every (int): Recompute the value area every N bars.
            nodes (str or None): HVN/LVN detection (see _VolumeProfileBase). The
                nodes cover the trailing ``length`` bars ending at the current
                bar (the same window as the developing profile).
            node_prominence (float): Minimum node strength (0..1) to keep.
            max_nodes (int or None): Optional cap per node kind.
            buy_color (str): Color of the buy-aggressor (ask) volume segment.
            sell_color (str): Color of the sell-aggressor (bid) volume segment.
            poc_color (str): Color of the point-of-control line.
            hvn_color (str): Color of the high-volume node markers.
            lvn_color (str): Color of the low-volume node markers.
        """
        super().__init__(
            period="1d",
            tick_size=tick_size,
            bins=bins,
            value_area_pct=value_area_pct,
            va_recompute_every=va_recompute_every,
            view="visible",
            nodes=nodes,
            node_prominence=node_prominence,
            max_nodes=max_nodes,
            buy_color=buy_color,
            sell_color=sell_color,
            poc_color=poc_color,
            hvn_color=hvn_color,
            lvn_color=lvn_color,
        )
        self.length = max(int(length), 1)
        self._final_profile: dict | None = None
        self._win_ts_start = 0
        self._win_ts_end = 0

    def _node_range(self, ts: np.ndarray) -> "tuple[int | None, int | None]":
        """Rolling nodes cover the trailing ``length`` bars up to the current bar."""
        if len(ts) == 0:
            return None, None
        lo = max(0, len(ts) - self.length)
        return int(ts[lo]), int(ts[-1])

    # Forces the engine to feed calculate with the full window (not just
    # min_periods) so that recomputation on the partial candle covers the
    # `length` complete bars.
    use_partial = False

    # The histogram must evolve within the live bar (developing profile
    # VPVR-style): it is recomputed on every tick over the window of
    # closed candles + the partial candle, refreshing POC/VAH/VAL and
    # _final_profile.
    recompute_on_partial = True

    @property
    def min_periods(self) -> int:
        # A moving window needs its `length` candles for the profile to
        # start complete: the engine's automatic warmup uses this value.
        # (The base returns 1 because VolumeProfile resets per period, not
        # per fixed number of bars.)
        return self.length

    def display_name(self) -> str:
        return f"RollingVP({self.length})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"rvp_{self.length}_{symbol}"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 6 or len(source) == 0:
            self._final_profile = None
            return self._empty_result()

        open_ = source[:, 1].astype(np.float64)
        high = source[:, 2].astype(np.float64)
        low = source[:, 3].astype(np.float64)
        close = source[:, 4].astype(np.float64)
        vol = source[:, 5].astype(np.float64)
        n = len(close)

        ts = source[:, 0].astype(np.int64)
        win_lo = max(0, n - self.length)
        self._win_ts_start = int(ts[win_lo]) if n else 0
        self._win_ts_end = int(ts[-1]) if n else 0

        tick_size = self._resolve_tick_size(np.concatenate([high, low]))
        self.tick_size_used_ = tick_size

        is_bull = close >= open_
        rows, levels, v_bid, v_ask, counts = _kline_contributions(
            high, low, vol, is_bull, tick_size
        )

        poc, vah, val, final = scan_rolling_profiles(
            n,
            self.length,
            rows,
            levels,
            v_bid,
            v_ask,
            counts,
            value_area_pct=self.value_area_pct,
            va_recompute_every=self.va_recompute_every,
        )
        self._final_profile = final

        out = np.full((3, n), np.nan, dtype=np.float64)
        out[0] = poc
        out[1] = vah
        out[2] = val
        return out

    def period_profiles(self) -> list[dict]:
        """
        Single render-ready profile of the most recent window (VPVR-style).

        Returns a one-element list (or empty) shaped like the per-period dicts
        consumed by the plotting layer, so build_volume_profile_source() can draw
        it at the right edge via view="visible".
        """
        from tradetropy.models.footprint import _FP_LEVEL_COL, _FP_SCALAR_COL

        final = self._final_profile
        if not final:
            return []
        levels = final["levels"]
        if levels.size == 0 or len(levels) == 0:
            return []
        scal = final["scalars"]
        volumes = levels[:, _FP_LEVEL_COL["vol_total"]]
        return [{
            "ts_start": self._win_ts_start,
            "ts_end": self._win_ts_end,
            "poc": float(scal[_FP_SCALAR_COL["poc_price"]]),
            "vah": float(scal[_FP_SCALAR_COL["vah"]]),
            "val": float(scal[_FP_SCALAR_COL["val"]]),
            "max_vol": float(volumes.max()) if len(volumes) else 0.0,
            "prices": levels[:, _FP_LEVEL_COL["price"]],
            "volumes": volumes,
            "vol_bid": levels[:, _FP_LEVEL_COL["vol_bid"]],
            "vol_ask": levels[:, _FP_LEVEL_COL["vol_ask"]],
            "deltas": levels[:, _FP_LEVEL_COL["delta"]],
        }]

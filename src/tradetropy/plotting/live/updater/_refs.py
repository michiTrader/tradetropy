from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tradetropy.core.constants import _OHLC_COL, N_OHLC_COLS

if TYPE_CHECKING:
    from tradetropy.data.data import OhlcProxy, IndicatorProxy, MultiBandProxy
    from tradetropy.plotting.live.navigation import LiveNavigationController
    from tradetropy.models.footprint import FpProxy


@dataclass
class _OhlcSourceRef:
    source:      object          # ColumnDataSource
    proxy:       "OhlcProxy"
    interval_ms: int
    theme:       dict = None     # to compute candle_color/wick_color
    fp_proxy:    object = None   # FootprintProxy (optional)
    # Secondary "vol" Range1d of the volume bars, pinned to the bottom band.
    # Kept here so the OHLC updater can re-anchor it when new data arrives
    # (the JS callback only fires on pan/zoom); None when plot_volume is off.
    vol_range:   object = None


@dataclass
class _BandRef:
    """An indicator band -- can be price or auxiliary timestamp."""
    source:       object          # ColumnDataSource  {"ts": [], "value": [], "tag": []}
    band_idx:     int             # index in proxy (0..n_outputs-1)
    is_ts_band:   bool            # True -> always skip (timestamp band)
    is_stateful:  bool            # True -> only update on candle close (pivots)
    col_name:     str = ""        # column name in ring.col_index
    has_real_ts:  bool = False    # True -> has parallel ts band (ZigZag, ConfirmedPivot)
                                  #         sparse values -> use _stream_new_pivots
    tag_label:    str = ""        # short label for pivot type (e.g. "H", "HH", "H-neu")
    # Diverging-bar bands (e.g. MACD histogram) carry a per-row ``bar_color``
    # column so the vbar is colored by the sign of each value without any
    # client-side transform (WebGL-safe). None → not a diverging bar band.
    bar_pos_color: "str | None" = None
    bar_neg_color: "str | None" = None


@dataclass
class _IndicatorSourceRef:
    bands:       list[_BandRef]
    proxy:       "IndicatorProxy | MultiBandProxy"
    ohlc_proxy:  "OhlcProxy"      # to synchronize timestamps


@dataclass
class _EquitySourceRef:
    source:  object               # ColumnDataSource {"ts": [], "equity": []}
    broker:  object               # any broker with .equity_curve
    config:  object               # PlotConfig -- for equity_mode / equity_unit
    trailing_source: object = None  # ColumnDataSource {"ts": [], "trailing_dd": []} | None


@dataclass
class _FootprintSourceRef:
    source_bid:         object      # ColumnDataSource with bid rows
    source_ask:         object      # ColumnDataSource with ask rows
    fp_proxy:           "FpProxy"
    ohlc_proxy:         "OhlcProxy" # to read real timestamps of closed candles
    interval_ms:        int
    theme:              dict        # to color new candles
    _candle_bid_row_counts: list[int] = field(default_factory=list)
    _candle_ask_row_counts: list[int] = field(default_factory=list)
    _n_candles_max:       int         = 50
    _n_partial_bid_rows: int        = 0
    _n_partial_ask_rows: int        = 0
    _last_fp_ts_ms:      int        = -1
    renderers:          list        = field(default_factory=list)
    zoom_range:         int         = 40


@dataclass
class _VolumeProfileSourceRef:
    """
    Live ref for an indicator that emits draw primitives (e.g. the Volume
    Profile histogram).

    The indicator recomputes its primitives each tick (developing profile); the
    generic renderer (render_tool_groups) updates the per-(legend, kind) sources
    in place via ``registry``, so glyphs are created once and only their data
    changes afterwards. This is the same mechanism used for use_tool() snapshots.
    """
    defn:        dict            # indicator_def (carries the indicator + cfg)
    indicator:   object          # VolumeProfile/Rolling/TickVolumeProfile instance
    fig:         object          # OHLC figure where glyphs are drawn
    ohlc_proxy:  "OhlcProxy"      # to know the candle interval
    interval_ms: int
    theme:       dict
    source_proxy: object = None   # indicator source proxy (live_refresh)
    registry:    dict = field(default_factory=dict)  # (legend, kind) -> CDS
    lazy_zoom_range: int = 40     # max visible candles to show labels
    align_to_candle: bool = True  # floor draw() primitives to the candle grid
    candle_origin_ms: int = 0     # grid phase (any real open)
    # Mirror CDS (left/right/bottom/top) kept in sync with this overlay's quad
    # cells so the OHLC autoscale can include them. Set only for overlays that
    # opt into the price autoscale (exclude_from_autoscale=False, e.g. Heatmap);
    # the live source is created lazily at runtime, so a pre-created mirror is
    # what the build-time autoscale CustomJS can reference.
    autoscale_mirror: object = None


@dataclass
class _ToolSnapshotRef:
    """
    Dynamic snapshots from use_tool() (FixedRangeVP, FibRetracement, ...) in live.

    Unlike _VolumeProfileSourceRef (an indicator declared in init()), here the
    profiles are generated inside on_data() via use_tool(); the updater creates
    this ref lazily on the first snapshot and then reconstructs the sources when
    the number of accumulated snapshots in the strategy changes.

    The ref stores the generic renderer's *registry* -- a map
    (legend, kind) -> ColumnDataSource -- to reuse and update the same sources
    on each tick without recreating glyphs.
    """
    strategy:        object       # Strategy with _tool_snapshots
    fig:             object       # OHLC figure where glyphs are drawn
    interval_ms:     int
    theme:           dict
    registry:        dict   = field(default_factory=dict)  # (legend, kind) -> CDS
    _n_drawn:        int    = 0    # number of snapshots already rendered
    lazy_zoom_range: int    = 40   # max visible candles to show labels


@dataclass
class _TradesSourceRef:
    source:           object        # ColumnDataSource -- connection lines
    sesh:             object        # live or sim Sesh
    symbol:           str
    theme:            dict
    interval_ms:      int
    poll_interval_s:  float         = 5.0
    align_to_candle:  bool          = True
    fig:              object        = None
    renderer:         object        = None
    source_markers:   object        = None
    source_open:      object        = None
    pos_line:         object        = None
    pos_label:        object        = None
    tpsl_source:      object        = None
    tpsl_lines:       object        = None
    tpsl_labels:      object        = None
    pending_source:   object        = None
    pending_lines:    object        = None
    pending_labels:   object        = None
    _last_check_s:    float         = field(default=0.0)
    _last_deal_ts_ms: int           = field(default=0)
    _n_known_trades:  int           = field(default=0)

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import numpy as np

@dataclass
class IndicatorPlotMeta:
    """
    Indicator display metadata.

    Fields
    ------
    name         : legend label.
                   str       -> a single entry controlling all dimensions.
                   list[str] -> one entry per dimension (independent clicks).
    panel_title  : Y-axis title when overlay=False.
                   None -> uses name (if str) or the first entry of name.
    values       : array [N] (one series) or [K x N] (K series).
    timestamps   : array of timestamps in ms.
    overlay      : True -> on top of OHLC / False -> own panel.
    scatter      : True -> dots, False -> continuous line.
    plot         : False -> do not render.
    color        : str | list[str] | None - color per dimension.
    panel_height : height in px of own panel (only if overlay=False).

    Line style fields - scalar or list per dimension:
        line_width, line_dash, line_alpha

    Scatter style fields - scalar or list per dimension:
        marker, marker_size, marker_alpha, marker_fill, marker_line_width

    reference_lines : list[dict] with horizontal lines for the panel.
        Format: {"value": float, "color": str, "dash": str, "label": str, "width": float}

    _ts_band_indices : indices of bands containing timestamps in ms.
    """

    name: str | list[str]
    values: np.ndarray
    timestamps: np.ndarray
    overlay: bool | None = None
    scatter: bool = False
    plot: bool = True
    color: str | list[str] | None = None
    panel_height: int = 100
    panel_title: str | None = None
    show_legend: bool | None = None

    # ── Line style - scalar or list per dimension ─────────────────────────────
    line_width: float | list = 1.8
    line_dash: str | list = "solid"
    line_alpha: float | list = 0.9

    # ── Scatter style - scalar or list per dimension ──────────────────────────
    marker: str | list = "circle"
    marker_size: int | list = 8
    marker_alpha: float | list = 0.85
    marker_fill: bool | list = True
    marker_line_width: float | list = 1.0

    # ── Renderer (replaces scatter for new code) ─────────────────────────────
    renderer: str | list[str] = "line"

    # ── Dimension to draw at the front (z-order) ─────────────────────────────
    # Index of the price band whose glyph should be drawn on top of the rest.
    # Useful when several series share price (e.g. POC of Volume Profile
    # coincides with VAH/VAL and would be covered). None = natural order.
    front_dim: int | None = None

    # ── Bar (scalar or list per dimension) ─────────────────────────────────────────────────
    bar_width_factor: float | list = 0.8
    bar_align: str | list = "center"
    bar_alpha: float | list = 0.75
    bar_color_positive: str | list | None = None
    bar_color_negative: str | list | None = None

    # ── Reference lines (RSI 70/30, etc.) ─────────────────────────────────────
    reference_lines: list = field(default_factory=list)

    # ── Internal: timestamp bands for ConfirmedPivot, etc. ───────────────────
    _ts_band_indices: list = field(default_factory=list)

    # ── Indicator output names (for tag in HoverTool scatter) ─────────────────
    # output_names[i] = short label for price band i (e.g. "H", "L", "HH")
    output_names: list = field(default_factory=list)

    # ── Panel autoscale on horizontal zoom ────────────────────────────────────
    autoscale: bool = False

    # ── Exclude from Y-axis autoscale ────────────────────────────────────────
    # True -> the renderer does not participate in the OHLC Y-range calculation.
    # Always True for category="annotation" (rect, span, label, arrow).
    exclude_from_autoscale: bool = False

    # ── Rect styles ───────────────────────────────────────────────────────────
    rect_fill_alpha: float = 0.15
    rect_line_alpha: float = 0.6
    rect_line_width: float = 1.0

    # ── Label styles ──────────────────────────────────────────────────────────
    label_font_size: str = "9pt"
    label_x_offset: int = 5
    label_y_offset: int = 5
    label_text_align: str = "left"
    label_text_baseline: str = "bottom"

    # ── Arrow styles ──────────────────────────────────────────────────────────
    arrow_size: int = 10
    arrow_length: int = 0

    # ── Volume profile data (horizontal histogram per period) ─────────────────
    # VolumeProfile / TickVolumeProfile indicators store their finished
    # histograms per period here so the renderer draws horizontal
    # volume-by-price bars. None if not applicable.
    _vp_data: dict = None

    # ── Declarative drawing primitives (unified contract) ─────────────────────
    # Indicators whose geometry is not 'one value per bar' (VP histogram,
    # price-time zones) emit primitives via Indicator.draw(); they are stored
    # here grouped by legend name and drawn with the generic renderer
    # render_tool_groups, same as tools. None / empty if not applicable.
    _draw_primitives: dict = None


@dataclass
class PlotConfig:
    """
    Backtest chart configuration.

    Parameters
    ----------
    plot_drawdown        : show drawdown panel
    plot_pl              : show P&L per trade panel (horizontal bars)
    pl_height            : P&L panel height in px
    plot_trades          : show trade entry/exit lines
    ohlc_style           : price panel style - "candle" (Japanese candlesticks,
                           default) or "bar" (OHLC bars: vertical High-Low line
                           with open/close ticks). The "bar" mode uses only
                           segments, which renders fastest with
                           output_backend="webgl".
    plot_volume          : show volume bars in the OHLC panel
    plot_footprint       : show footprint (requires FpProxy in strategy)
    plot_volume_profile  : show Volume Profile bar histogram
    show_price_tag       : show the floating price box that follows the
                           crosshair snapped to the Y-axis (TradingView style)
    width                : total width in pixels
    ohlc_height          : main OHLC panel height
    equity_height        : equity/return panel height
    drawdown_height      : drawdown panel height
    indicator_height     : default height for external indicator panels
    footprint_zoom_range : max number of candles to show the footprint
    labels_zoom_range    : max number of visible candles to render indicator
                           text labels; when zooming out above this threshold
                           all labels are hidden (internal and external), same
                           as footprint.
    max_candles          : candle render limit (performance)
    chart_refresh_ms     : minimum interval (ms) between chart refreshes in
                           live/replay. Refresh is decoupled from the engine
                           data rate: the engine produces data and only marks
                           the chart as 'dirty', while a periodic callback in
                           the Bokeh IOLoop redraws at most once per this
                           interval. This way the IOLoop is never saturated with
                           back-to-back refreshes and replay controls
                           (pause/step/speed) respond immediately even at high
                           speeds. Default 66 ms (~15 fps). Increasing reduces
                           render load; decreasing gives a smoother chart at the
                           cost of more IOLoop work.
    heavy_refresh_ms     : minimum interval (ms) between refreshes of heavy
                           drawing layers (Volume Profile / Heatmap and
                           use_tool() snapshots), which rebuild their full
                           geometry via Indicator.draw() on each refresh (for
                           the Heatmap, the entire time x price grid from
                           book_window()). These refresh at this coarser cadence
                           than the rest of the chart (OHLC/indicators/footprint
                           follow chart_refresh_ms), because they recompute from
                           the current causal state and skipping frames loses no
                           information. This way the IOLoop is not kept busy
                           recalculating that geometry every frame and replay
                           controls still respond at high speed. On a candle
                           close they always refresh (to reflect the final
                           profile). Default 200 ms (~5 fps). Must be >= chart_refresh_ms.
    equity_mode          : "none" | "balance" | "return"
    equity_unit          : "currency" | "percent"
    theme                : "light" | "dark"
    output               : "notebook" | "file" | "show" | "server"
                           - "show"     : opens system browser (default).
                           - "notebook" : renders inline in Jupyter.
                           - "file"     : saves HTML to disk (see filename).
                           - "server"   : starts a blocking Bokeh Server on
                             server_port and prints the URL to the terminal
                             without opening the browser. Ideal for Termux, SSH
                             or any headless environment: open
                             http://localhost:5006 (or the printed network IP)
                             from any browser. Stop with Ctrl+C.
    server_port          : Bokeh Server port when output="server" (default 5006)
    output_backend       : "canvas" | "webgl" | "svg"
    filename             : HTML filename (only if output="file")
    resample_timeframe   : Regroup candles to the given interval before
                           drawing. Accepts a timeframe string ('1m', '5m',
                           '1h', '1d', etc.) parsed via parse_timeframe().
    align_trades_to_candle : Align trade timestamps to candle open.
    align_indicators_to_candle : Align (floor) indicator draw() primitives to
                           the OHLC candle grid, so all indicator geometry
                           (LargeTrades bubbles, Volume Profile histogram, COT
                           labels, zones, ...) is drawn in the same time unit
                           as the candles (e.g. to the minute in a 1m OHLC, to
                           :00/:15/:30/:45 in a 15s one), ignoring
                           sub-candle seconds/milliseconds. Respects the real
                           candle phase (origin not necessarily on the epoch
                           grid). Per-bar indicator series are already on the
                           OHLC grid, so this only affects draw() primitives.
                           Default True; set to False to preserve sub-candle tick
                           precision (microstructure analysis).
    """

    plot_drawdown: bool = False
    plot_trades: bool = True
    ohlc_style: Literal["candle", "bar"] = "candle"
    plot_volume: bool = True
    plot_footprint: bool = True
    plot_volume_profile: bool = True
    show_price_tag: bool = True
    plot_stats: bool = False
    plot_pl: bool = False
    pl_height: int = 70
    width: int = 1200
    ohlc_height: int = 420
    equity_height: int = 80
    drawdown_height: int = 70
    indicator_height: int = 100
    footprint_zoom_range: int = 40
    labels_zoom_range: int = 40
    max_candles: int = 10_000
    chart_refresh_ms: int = 66
    heavy_refresh_ms: int = 200
    equity_mode: Literal["none", "balance", "return"] = "return"
    equity_unit: Literal["currency", "percent"] = "percent"
    theme: Literal["light", "dark"] = "light"
    output: Literal["notebook", "file", "show", "server"] = "show"
    server_port: int = 5006
    output_backend: Literal["canvas", "webgl", "svg"] = "canvas"
    filename: str = "backtest.html"
    resample_timeframe: str | None = None
    align_trades_to_candle: bool = True
    align_indicators_to_candle: bool = True
    max_trailing_dd: float | None = None

    def __post_init__(self):
        if self.resample_timeframe is not None:
            from tradetropy.core.constants import parse_timeframe
            self.resample_timeframe = parse_timeframe(self.resample_timeframe)

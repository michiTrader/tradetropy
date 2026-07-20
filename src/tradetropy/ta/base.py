from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal
import numpy as np

from tradetropy.exceptions import ConfigError


# =====
# INDICATOR CATEGORIES
# =====
#
# Available categories:
#   "trend"      - Direction/trend indicators (SMA, EMA, MACD...)
#   "momentum"   - Price movement speed (RSI, Stoch, CCI...)
#   "volatility" - Movement amplitude (ATR, BB, Keltner...)
#   "volume"     - Volume activity (OBV, VWAP, CMF...)
#   "structure"  - Price structure: pivots, support/resistance,
#                  swing highs/lows. May have timestamp bands (_ts)
#                  that plotting treats specially to reposition
#                  markers at the real pivot bar.
#   "other"      - Everything else (custom, experimental...)
#
IndicatorCategory = Literal["trend", "momentum", "volatility", "volume", "structure", "annotation", "other"]

# Valid markers in Bokeh scatter
MarkerType = Literal[
    "circle", "square", "triangle", "inverted_triangle",
    "diamond", "hex", "star", "cross", "x", "dot",
    "circle_x", "circle_cross", "square_x", "square_cross",
    "diamond_cross", "asterisk",
]

# Valid line styles in Bokeh
LineDash = Literal["solid", "dashed", "dotted", "dashdot", "dotdash"]

# Valid renderer types
# Besides series types, annotation renderers are:
#   "rect"  - rectangle with coordinates [x0, x1, y0, y1] in price x time space
#   "span"  - infinite vertical/horizontal zone in Y (sessions, time ranges)
#   "label" - floating text anchored to price x time
#   "arrow" - arrow with direction (buy/sell signals)
RendererType = Literal["line", "scatter", "bar", "step", "rect", "span", "label", "arrow"]


# =====
# INDICATOR PLOT CONFIG
# =====
#
# All visualization configuration for an indicator lives here.
# Indicators declare their defaults in __init__ by assigning self.plot_config.
# add_indicator() only allows overriding specific fields.
#
# Default overlay rules (when overlay=None):
#   "trend"      -> True   (SMA, EMA - on top of price)
#   "volatility" -> True   (BB, ATR - on top of price)
#   "momentum"   -> False  (RSI, Stoch - own panel, 0-100 scale)
#   "volume"     -> False  (OBV, VWAP - own panel, volume scale)
#   "structure"  -> True   (pivots - over price, almost always scatter)
#   "other"      -> False  (conservative: own panel by default)
#
_CATEGORY_OVERLAY_DEFAULTS: dict[str, bool] = {
    "trend":      True,
    "volatility": True,
    "momentum":   False,
    "volume":     False,
    "structure":  True,
    "annotation": True,   # always overlay - lives in price x time space
    "other":      False,
}


def _resolve_dim(value, idx: int):
    """
    Resolve the value of a style field for dimension `idx`.

    If `value` is a list, returns value[idx] (or value[-1] if idx is out of range).
    If `value` is scalar, returns it directly for all dimensions.

    Examples:
        _resolve_dim("solid", 1)            -> "solid"
        _resolve_dim(["solid","dashed"], 1) -> "dashed"
        _resolve_dim([1.5], 2)              -> 1.5   (clamped to last)
    """
    if isinstance(value, list):
        return value[idx] if idx < len(value) else value[-1]
    return value


@dataclass
class IndicatorPlotConfig:
    """
    Complete visualization configuration for an indicator.

    Visibility fields
    -----------------
    plot         : False -> do not render in any panel.
    overlay      : True -> on top of OHLC / False -> own panel.
                   None -> resolved from indicator category
                   (see _CATEGORY_OVERLAY_DEFAULTS).
    panel_height : height in px of own panel (only if overlay=False).
    panel_title  : Y-axis title when indicator is in its own panel.
                   None -> uses name (if str) or display_name() of indicator.
    name         : legend label.
                   - str       -> single legend entry controlling all
                                  dimensions with one click.
                   - list[str] -> one entry per dimension, independent clicks.
                   - None      -> uses display_name() of indicator.
    zorder       : render order within panel (higher = on top).

    Line fields (when scatter=False) - scalar or list per dimension
    ---------------------------------------------------------------
    color        : str | list[str] | None - color per dimension or single for all.
                   None -> automatic color cycle.
    line_width   : float | list[float]    - line width per dimension.
    line_dash    : LineDash | list[LineDash] - line style per dimension.
    line_alpha   : float | list[float]    - opacity per dimension.

    Scatter fields (when scatter=True) - scalar or list per dimension
    -----------------------------------------------------------------
    scatter      : True -> render as points instead of line.
    marker       : MarkerType | list[MarkerType] - marker shape per dimension.
    marker_size  : int | list[int]        - marker size per dimension.
    marker_alpha : float | list[float]    - opacity per dimension.
    marker_fill  : bool | list[bool]      - fill per dimension.
    marker_line_width : float | list[float] - border width per dimension.

    Reference lines
    ---------------
    reference_lines : list of dicts with horizontal lines in the panel.
                      Format: {"value": float, "color": str, "dash": str,
                               "label": str, "width": float}

    Per-dimension style usage
    -------------------------
    Bollinger Bands with differentiated lines:

        self.bb = self.add_indicator(
            self.btc.close_ref, BollingerBands(20, 2.0),
            name="BB(20)",                              # one legend -> one click
            line_dash=["solid", "dashed", "solid"],
            line_width=[1.5, 1.0, 1.5],
            line_alpha=[0.9, 0.6, 0.9],
            color=["#FF6B35", "#888888", "#3B82F6"],
        )

    Legend per dimension (independent clicks):

        self.bb = self.add_indicator(
            self.btc.close_ref, BollingerBands(20),
            name=["BB Upper", "BB Mid", "BB Lower"],
        )

    Explicit panel title:

        self.rsi = self.add_indicator(
            self.btc.close_ref, RSI(14),
            panel_title="RSI",
        )
    """

    # Visibility
    plot: bool = True
    overlay: bool | None = None
    panel_height: int = 100
    panel_title: str | None = None
    name: str | list[str] | None = None
    zorder: int = 0

    # Line (scalar or list per dimension)
    color: str | list[str] | None = None
    line_width: float | list = 1.8
    line_dash: str | list = "solid"
    line_alpha: float | list = 0.9

    # Renderer (replaces scatter: bool)
    renderer: str | list = "line"

    # Dimension drawn on top (z-order). Index of the price band whose glyph
    # must sit above the rest. Useful when several series share a price
    # (e.g. the volume profile POC coincides with VAH/VAL and would be hidden).
    # None = natural order.
    front_dim: int | None = None

    # Scatter (scalar or list per dimension)
    marker: str | list = "circle"
    marker_size: int | list = 8
    marker_alpha: float | list = 0.85
    marker_fill: bool | list = True
    marker_line_width: float | list = 1.0

    # Bar (scalar or list per dimension)
    bar_width_factor: float | list = 0.8
    bar_align: str | list = "center"
    bar_alpha: float | list = 0.75
    bar_color_positive: str | list | None = None
    bar_color_negative: str | list | None = None

    # Scatter (backwards compatibility)
    @property
    def scatter(self) -> bool:
        """Backwards compatibility: True if all dims use renderer="scatter"."""
        r = self.renderer
        if isinstance(r, list):
            return all(x == "scatter" for x in r)
        return r == "scatter"

    # Legend
    show_legend: bool | None = None   # None -> default plotting behavior

    # Reference
    reference_lines: list[dict] = field(default_factory=list)

    # Autoscale panel on horizontal zoom
    autoscale: bool = False

    # Annotations - exclude from Y-axis autoscale
    # True -> renderer does not participate in Y range calculation.
    # Enabled automatically for category="annotation".
    # Also useful for price indicators that draw very wide zones
    # (e.g. session levels) that would distort the visible chart range.
    exclude_from_autoscale: bool = False

    # Rect-specific styles
    # For renderer="rect": fill and border of rectangles.
    rect_fill_alpha: float = 0.15
    rect_line_alpha: float = 0.6
    rect_line_width: float = 1.0

    # Label-specific styles
    # For renderer="label": font, size, alignment.
    label_font_size: str = "9pt"
    label_x_offset: int = 5      # offset in pixels from timestamp
    label_y_offset: int = 5      # offset in pixels from price
    label_text_align: str = "left"  # "left" | "center" | "right"
    label_text_baseline: str = "bottom"  # "top" | "middle" | "bottom"

    # Arrow-specific styles
    # For renderer="arrow": arrowhead size and length.
    arrow_size: int = 10         # arrowhead size in pixels
    arrow_length: int = 0        # shaft length (0 = head only)

    def resolve_overlay(self, category: IndicatorCategory) -> bool:
        """
        Returns the effective overlay value, resolving None from the category.
        """
        if self.overlay is not None:
            return self.overlay
        return _CATEGORY_OVERLAY_DEFAULTS.get(category, False)

    def dim(self, field_name: str, idx: int):
        """
        Returns the field value for dimension `idx`.
        Delegates to _resolve_dim() - supports scalar and list.

        Example:
            pc.dim("line_dash", 0)  -> "solid"
            pc.dim("line_dash", 1)  -> "dashed"  (if line_dash is a list)
        """
        return _resolve_dim(getattr(self, field_name), idx)

    def merged(self, **overrides) -> "IndicatorPlotConfig":
        # Convert scatter=True/False -> renderer (backwards compatibility)
        if "scatter" in overrides:
            scatter_val = overrides.pop("scatter")
            if "renderer" not in overrides:
                overrides["renderer"] = "scatter" if scatter_val else "line"

        valid_fields = {f.name for f in self.__dataclass_fields__.values()}
        # Filter out properties (like scatter) from valid fields
        actual_fields = {f for f in valid_fields if not f.startswith("_")}
        bad = set(overrides) - actual_fields
        if bad:
            raise ConfigError(
                f"add_indicator() received unknown visualization parameters: {bad}. "
                f"Valid parameters: {sorted(actual_fields)}"
            )
        return replace(self, **overrides)


# =====
# BASE CLASS Indicator
# =====
#
# To create a new indicator, inherit from Indicator and implement ONE method:
#
#   calculate(source: np.ndarray) -> np.ndarray
#
#   Receives the full source column array (single) or [N x K] (multi-source).
#   Returns an array [N] (single-band) or [K x N] (multi-band, n_outputs > 1).
#   NaN where the window is not ready.
#
# CLASS ATTRIBUTES - override in subclasses:
#
#   name            str          - short identifier: "sma", "rsi", "bb"
#   category        str          - "trend"|"momentum"|"volatility"|"volume"|"structure"|"other"
#   output_names    list[str]    - names of PRICE output series (no auxiliary ts).
#                                  Can be class attribute (static) or instance
#                                  attribute (assigned in __init__ for dynamic names).
#                                  Examples: ["upper","mid","lower"] for BB,
#                                            ["pivot_high","pivot_low"] for PivotHighLow.
#   ts_band_indices list[int]    - indices of auxiliary timestamp bands (not price).
#                                  Plotting excludes them from rendering and uses them
#                                  to reposition markers at the real pivot bar.
#   ts_output_names list[str]    - names for ts bands (same order as ts_band_indices).
#                                  Enables attribute access in on_data():
#                                    self.cpivot.ts_high[-1]   instead of  self.cpivot[2][-1]
#                                  Opt-in: [] if explicit names are not needed.
#
# AUTOMATICALLY DERIVED - do NOT override in subclasses:
#
#   n_outputs       @property    - len(output_names or [_]) + len(ts_band_indices)
#                                  Single source of truth. Never declare n_outputs=K.
#
# KEY METHODS:
#
#   display_name() -> str
#       Human-readable name for UI: "RSI(14)", "SMA(20)", "BB(20)".
#       Used in legends when plot_config.name is None.
#
#   col_name(symbol, col_source) -> str
#       Internal DataStore key: "rsi14_BTCUSDT". NEVER show to user.
#
class Indicator:
    """
    Base class for indicators. Only requires calculate.

    calculate(source)
        source  : 1D array [N] (single-source) or 2D [N x K] (multi-source)
        returns : array [N] (single-band) or [K x N] (multi-band, n_outputs > 1)
                  NaN where the window is not complete

    CLASS ATTRIBUTES (override in subclasses)
    -----------------------------------------
    name : str
        Short lowercase identifier. E.g.: "sma", "rsi", "bb".

    category : IndicatorCategory
        Functional classification. Determines the default overlay when
        plot_config.overlay is None (see _CATEGORY_OVERLAY_DEFAULTS).

    output_names : list[str]
        Names of PRICE output series accessible by the user.
        Can be class attribute (static) or instance attribute (in __init__
        for indicators where names depend on parameters).
        Do not include timestamp series (those go in ts_band_indices).
        Examples: ["upper","mid","lower"] for BB, ["pivot_high","pivot_low"] for PivotHighLow.

    ts_band_indices : list[int]
        Indices of internal timestamp auxiliary bands. Plotting excludes
        them from rendering and uses them to reposition markers at the real bar.

    ts_output_names : list[str]
        Names for ts bands, in the same order as ts_band_indices.
        Enables attribute access from on_data():
            self.cpivot.ts_high[-1]   ->  real timestamp of last PH
            self.cpivot.ts_low[-1]    ->  real timestamp of last PL
        Opt-in: leave [] if explicit names are not needed.
        Index access (self.cpivot[2][-1]) always works.

    plot_config : IndicatorPlotConfig
        Visualization configuration for the indicator. Indicators should
        assign self.plot_config in __init__ to declare their defaults.
        add_indicator() can override any field.

    METHODS
    -------
    col_name(symbol, col_source) -> str
        Internal DataStore key. Do not show to user.

    display_name() -> str
        Human-readable name for legends. Used when plot_config.name is None.
    """

    # Class attributes - override in subclasses
    name: str = ""
    category: IndicatorCategory = "other"
    output_names: list[str] = []
    ts_band_indices: list[int] = []
    ts_output_names: list[str] = []   # names for ts bands (opt-in)

    # recompute_on_partial: if True, the indicator is recomputed intrabar (on each
    # tick, over the partial candle) EVEN IF use_partial=False. Designed for
    # full-window indicators that still need to evolve within the
    # live bar (e.g. Volume Profile / developing POC), unlike
    # structure detectors (pivots, zigzag) which only change on close.
    # No effect when use_partial=True (those already recompute intrabar).
    recompute_on_partial: bool = False

    # warmup_factor: multiplier over `length` used by the auto-warmup policy.
    # Recursive indicators (RSI = Wilder, MACD = EMA) are path-dependent: their
    # value needs more than `min_periods` bars to converge (the seed influence
    # decays over ~a few times the smoothing period). They set warmup_factor > 1
    # so the auto warmup reserves max(min_periods, warmup_factor * length) bars
    # and the first on_data() bar is already converged. Windowed indicators
    # (SMA, Volume Profile, Donchian, ...) keep 1, so their auto warmup stays at
    # min_periods (no inflation).
    warmup_factor: int = 1

    # source_cols: names of the proxy columns this indicator consumes, IN THE
    # ORDER calculate() expects them. Drives the generic refs()/default_refs()
    # helpers below so a multi-source indicator no longer forces the caller to
    # spell out [proxy.high_ref, proxy.low_ref, proxy.close_ref] by hand: it can
    # pass the proxy directly (add_indicator(self.btc, Stochastic())) or the
    # helper (Stochastic.refs(self.btc)). Single-source indicators keep the
    # default ('close',). Order-flow / tick indicators that build tick-specific
    # refs override refs()/default_refs() themselves. Passing an explicit
    # ColumnRef list stays fully supported.
    source_cols: tuple = ("close",)

    # n_outputs: automatically derived - do NOT override
    @property
    def n_outputs(self) -> int:
        n_price = len(self.output_names) if self.output_names else 1
        return n_price + len(self.ts_band_indices)

    @property
    def min_periods(self) -> int:
        return 1

    @property
    def plot_config(self) -> IndicatorPlotConfig:
        """
        Visualization config. Subclasses should override this by assigning
        self.plot_config = IndicatorPlotConfig(...) in __init__.
        Default here is an empty config - overlay is resolved by category.
        """
        return getattr(self, "_plot_config", IndicatorPlotConfig())

    @plot_config.setter
    def plot_config(self, value: IndicatorPlotConfig):
        self._plot_config = value

    # Methods

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """Calculate the indicator over the full array. Must be vectorized."""
        raise NotImplementedError(f"{type(self).__name__}.calculate not implemented")

    def draw(self, cfg: "IndicatorPlotConfig | None" = None,
             *, interval_ms: "int | None" = None) -> list:
        """
        Optional: emit declarative draw primitives for geometry that is not a
        per-bar series (e.g. a volume-by-price histogram, price-time zones).

        Return a list of primitives from :mod:`tradetropy.ta.draw` (``HBars``,
        ``HLines``, ``Segments``, ``Points``, ``Rects``, ``Labels``); a single
        generic renderer turns them into glyphs for both the static and live
        charts, so no plotting-package code is needed. The default returns no
        primitives: ordinary series indicators draw purely from their
        ``IndicatorPlotConfig``.

        Args:
            cfg (IndicatorPlotConfig | None): The effective plot config for this
                use (after add_indicator() overrides). May be None.
            interval_ms (int | None): OHLC interval, available for geometry that
                needs candle-width anchoring (e.g. VPVR).

        Returns:
            list: Draw primitives, or [] when the indicator draws as series only.
        """
        return []

    @classmethod
    def refs(cls, proxy) -> list:
        """
        Build the ColumnRef list this indicator consumes, in the expected order.

        Generic convenience so a multi-source indicator does not force the
        caller to spell out every column by hand. Reads ``cls.source_cols`` and
        resolves each name against the proxy:

            self.stoch = self.add_indicator(
                Stochastic.refs(self.btc), Stochastic(),
            )

        Equivalent to passing the proxy directly to ``add_indicator`` (which
        calls :meth:`default_refs`). Passing an explicit ColumnRef list stays
        supported. Indicators with tick-specific columns (order flow) override
        this.

        Args:
            proxy (OhlcProxy | TickProxy): An already-subscribed source proxy.

        Returns:
            list[ColumnRef]: One ColumnRef per name in ``source_cols``.
        """
        return [proxy.col_ref(col) for col in cls.source_cols]

    def default_refs(self, proxy) -> list:
        """
        Resolve the source columns when the proxy is passed directly to
        ``add_indicator`` (``add_indicator(self.btc, Stochastic())``).

        Instance-level counterpart of :meth:`refs`; reads ``self.source_cols``
        so subclasses that make ``source_cols`` an instance attribute (dynamic
        columns) resolve correctly. Order-flow indicators override this to build
        tick-specific refs.

        Args:
            proxy (OhlcProxy | TickProxy): An already-subscribed source proxy.

        Returns:
            list[ColumnRef]: One ColumnRef per name in ``source_cols``.
        """
        return [proxy.col_ref(col) for col in self.source_cols]

    def col_name(self, symbol: str, col_source: str = "") -> str:
        """Internal DataStore key. Do not use as display label."""
        cls_name = self.name or type(self).__name__.lower()
        length = getattr(self, "length", "")
        suffix = str(length) if length != "" else ""
        return f"{cls_name}{suffix}_{symbol}"

    def display_name(self) -> str:
        """
        Human-readable name for legends and panel titles.

        Examples:
            RSI(14)          -> "RSI(14)"
            SMA(20)          -> "SMA(20)"
            BollingerBands() -> "BB(20)"
            PivotHighLow(3)  -> "Pivot(3)"
        """
        cls_name = self.name or type(self).__name__
        label = cls_name.upper() if len(cls_name) <= 4 else cls_name.capitalize()
        length = getattr(self, "length", None)
        if length is not None:
            return f"{label}({length})"
        param = getattr(self, "n", None) or getattr(self, "swing", None)
        if param is not None:
            return f"{label}({param})"
        return label

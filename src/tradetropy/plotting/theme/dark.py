"""
Dark plot theme.

A theme is a flat dict of named color/style tokens consumed by the plotting
layer (figures, candles, equity, footprint, ...). Indicators do NOT read the
theme for their own colors - those live on the indicator/tool objects so they
are theme-independent. See tradetropy.plotting.theme for the registry.
"""

from __future__ import annotations

DARK: dict = {
    "bg":           "#0E1117",
    "bg_border":    "#161B22",
    "grid":         "#21262D",
    "text":         "#E6EDF3",
    "axis":         "#8B949E",
    "candle_up":    "#2ECC71",
    "candle_down":  "#E74C3C",
    "volume_alpha": 0.30,
    "equity_line":  "#58A6FF",
    "equity_area":  "#58A6FF",
    "peak_dot":     "#A78BFA",
    "maxdd_dot":    "#FF7B72",
    "trade_win":    "#007F5F",
    "trade_loss":   "#AD1D2B",
    "crosshair":    "#E6EDF3",
    "price_tag_bg":   "#E6EDF3",
    "price_tag_text": "#0E1117",
    "span_zero":    "#8B949E",
    "drawdown_fill": "#E74C3C",
    "trailing_dd_line": "#FBBF24",
    "fp_bid":       "231, 76, 60",
    "fp_ask":       "46, 204, 113",
    "fp_bid_scale": [
        (0.20, "#161213"),
        (0.40, "#533130"),
        (0.60, "#95423C"),
        (0.80, "#EB5240"),
    ],
    "fp_ask_scale": [
        (0.20, "#121615"),
        (0.40, "#27493C"),
        (0.60, "#479973"),
        (0.80, "#4BEF90"),
    ],
    "fp_poc_fill":  "#62697B",
    "fp_poc_text":  "#E6EDF3",
    "fp_text":      "#E6EDF3",
}

"""
Light plot theme.

A theme is a flat dict of named color/style tokens consumed by the plotting
layer (figures, candles, equity, footprint, ...). Indicators do NOT read the
theme for their own colors - those live on the indicator/tool objects so they
are theme-independent. See tradetropy.plotting.theme for the registry.
"""

from __future__ import annotations

LIGHT: dict = {
    "bg":           "#FFFFFF",
    "bg_border":    "#FFFFFF",
    "grid":         "#C8D3DC",
    "text":         "#1A2332",
    "axis":         "#8A9BB0",
    "candle_up":    "#0ECB81",
    "candle_down":  "#F6465D",
    "volume_alpha": 0.35,
    "equity_line":  "#2563EB",
    "equity_area":  "#2563EB",
    "peak_dot":     "#7C3AED",
    "maxdd_dot":    "#DC2626",
    "trade_win":    "#007F5F",
    "trade_loss":   "#AD1D2B",
    "crosshair":    "#1A2332",
    "price_tag_bg":   "#1A2332",
    "price_tag_text": "#F5F7FA",
    "span_zero":    "#8A9BB0",
    "drawdown_fill": "#F6465D",
    "trailing_dd_line": "#F59E0B",
    "fp_bid":       "255, 0, 40",
    "fp_ask":       "20, 235, 180",
    "fp_bid_scale": [
        (0.20, "#F9D1D8"),
        (0.40, "#F4A6B0"),
        (0.60, "#D56978"),
        (0.80, "#BA1A1A"),
    ],
    "fp_ask_scale": [
        (0.20, "#CDF6E8"),
        (0.40, "#8BDDC4"),
        (0.60, "#4CBCAF"),
        (0.80, "#069887"),
    ],
    "fp_poc_fill":  "#1A2332",
    "fp_poc_text":  "#FFFFFF",
    "fp_text":      "#1A2332",
}

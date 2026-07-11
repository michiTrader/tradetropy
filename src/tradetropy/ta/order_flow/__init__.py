"""
Order flow indicators.

This package groups market-microstructure / order-flow indicators that work on
the raw trade (tick) stream. The first member is :class:`LargeTrades`, a
large-trade ("whale") bubble overlay that detects outsized aggressive
executions from the time-and-sales feed (no order-book depth). The name
``DeepTrades`` is reserved for a future depth-of-market (DOM/MBO) detector;
see ``docs/design/deep_trades_dom.md``. Future order-flow indicators
(cumulative delta, trade imbalance, absorption, ...) belong here too.
"""

from tradetropy.ta.order_flow.large_trades import LargeTrades
from tradetropy.ta.order_flow._core import (
    EVENT_LARGE_AGGRESSOR,
    EVENT_ABSORPTION,
    EVENT_SWEEP,
    EVENT_ICEBERG,
    EVENT_LIQUIDITY_GRAB,
    DEEP_TRADE_LABELS,
    aggregate_trades,
    apply_deep_autofilter,
    bar_delta,
    bar_index,
    build_heatmap_grid,
    classify_aggressor,
    cumulative_delta_ohlc,
    deep_trade_class_name,
    detect_large_trades,
    detect_deep_trades,
    detect_iceberg,
    detect_liquidity_grab,
    detect_reload_l2,
    detect_stop_run_l2,
    detect_walls,
    find_heatmap_walls,
    format_magnitude,
    heatmap_color_bounds,
    infer_price_tick,
    map_bubble_style,
    merge_heatmap_grids,
    resolve_autofilter,
    resolve_persistence,
    trade_metric,
)

__all__ = [
    "LargeTrades",
    "EVENT_LARGE_AGGRESSOR",
    "EVENT_ABSORPTION",
    "EVENT_SWEEP",
    "EVENT_ICEBERG",
    "EVENT_LIQUIDITY_GRAB",
    "DEEP_TRADE_LABELS",
    "aggregate_trades",
    "apply_deep_autofilter",
    "bar_delta",
    "bar_index",
    "build_heatmap_grid",
    "classify_aggressor",
    "cumulative_delta_ohlc",
    "deep_trade_class_name",
    "detect_large_trades",
    "detect_deep_trades",
    "detect_iceberg",
    "detect_liquidity_grab",
    "detect_reload_l2",
    "detect_stop_run_l2",
    "detect_walls",
    "find_heatmap_walls",
    "heatmap_color_bounds",
    "merge_heatmap_grids",
    "trade_metric",
    "map_bubble_style",
    "resolve_autofilter",
    "resolve_persistence",
    "infer_price_tick",
    "format_magnitude",
]

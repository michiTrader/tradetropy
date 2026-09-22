import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class FootprintConfig:
    tick_size: float | None = None
    levels: int = 5
    value_area_pct: float = 0.70
    aggressor_col: str | None = "flags"
    vol_col: str = "volume"


def _round_tick_pretty(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    fraction = value / magnitude
    candidates = [1.0, 2.5, 5.0, 10.0]
    rounded = min(candidates, key=lambda c: abs(c - fraction))
    return float(rounded * magnitude)


def _infer_tick_size(
    price_col: np.ndarray,
    tick_to_candle_map: np.ndarray,
    n_candles: int,
    levels: int,
) -> float:
    sample_size = min(n_candles, 200)
    ranges = []
    for v_idx in range(n_candles - sample_size, n_candles):
        mask = tick_to_candle_map == v_idx
        if not np.any(mask):
            continue
        candle_prices = price_col[mask]
        ranges.append(float(candle_prices.max() - candle_prices.min()))

    if not ranges:
        price_range = float(price_col.max() - price_col.min())
    else:
        price_range = float(np.median(ranges))

    if price_range <= 0:
        return 1.0

    raw = price_range / levels
    return _round_tick_pretty(raw)


def _resolve_tick_size(
    config: FootprintConfig,
    price_col: np.ndarray,
    tick_to_candle_map: np.ndarray,
    n_candles: int,
) -> FootprintConfig:
    if config.tick_size is not None:
        return config
    ts = _infer_tick_size(price_col, tick_to_candle_map, n_candles, config.levels)
    from dataclasses import replace
    return replace(config, tick_size=ts)

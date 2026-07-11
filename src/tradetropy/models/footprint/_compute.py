import numpy as np

from ._config import FootprintConfig
from ._types import (
    N_FP_LEVEL_COLS,
    N_FP_SCALAR_COLS,
    _FP_LEVEL_COL,
    _FP_SCALAR_COL,
)


def _price_level(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 10)


def _compute_scalars(
    levels: np.ndarray,
    value_area_pct: float,
    cvd_prev: float,
) -> np.ndarray:
    sc = np.zeros(N_FP_SCALAR_COLS, dtype=np.float64)
    if len(levels) == 0:
        return sc

    vols = levels[:, _FP_LEVEL_COL["vol_total"]]
    deltas = levels[:, _FP_LEVEL_COL["delta"]]
    prices = levels[:, _FP_LEVEL_COL["price"]]

    poc_idx = int(np.argmax(vols))
    vol_total = float(np.sum(vols))
    delta_total = float(np.sum(deltas))

    va_target = vol_total * value_area_pct
    va_vol = float(vols[poc_idx])
    lo, hi = poc_idx, poc_idx

    while va_vol < va_target:
        can_up = hi + 1 < len(levels)
        can_down = lo - 1 >= 0
        if not can_up and not can_down:
            break
        vol_up = float(vols[hi + 1]) if can_up else -1.0
        vol_down = float(vols[lo - 1]) if can_down else -1.0
        if vol_up >= vol_down:
            hi += 1
            va_vol += vol_up
        else:
            lo -= 1
            va_vol += vol_down

    sc[_FP_SCALAR_COL["poc_price"]] = float(prices[poc_idx])
    sc[_FP_SCALAR_COL["poc_vol"]] = float(vols[poc_idx])
    sc[_FP_SCALAR_COL["poc_idx_local"]] = poc_idx
    sc[_FP_SCALAR_COL["vah"]] = float(prices[hi])
    sc[_FP_SCALAR_COL["val"]] = float(prices[lo])
    sc[_FP_SCALAR_COL["delta_total"]] = delta_total
    sc[_FP_SCALAR_COL["vol_total"]] = vol_total
    sc[_FP_SCALAR_COL["cvd"]] = cvd_prev + delta_total
    sc[_FP_SCALAR_COL["levels"]] = len(levels)
    return sc


def _dict_to_levels(partial_dict: dict) -> np.ndarray:
    if not partial_dict:
        return np.empty((0, N_FP_LEVEL_COLS), dtype=np.float64)
    items = sorted(partial_dict.items())
    n = len(items)
    levels = np.zeros((n, N_FP_LEVEL_COLS), dtype=np.float64)
    for i, (price, (vb, va, nt)) in enumerate(items):
        levels[i] = [price, vb, va, vb + va, va - vb, nt]
    return levels


def _accumulate_to_dict(
    partial_dict: dict,
    price: float,
    vol: float,
    bid: float,
    ask: float,
    flag: float,
    config: FootprintConfig,
):
    level = _price_level(price, config.tick_size)

    if config.aggressor_col is not None:
        flag_int = int(flag)
        if flag_int & 32:
            is_ask = True
        elif flag_int & 64:
            is_ask = False
        elif flag_int == 0:
            is_ask = False
        else:
            is_ask = price >= (bid + ask) / 2.0
    else:
        is_ask = price >= (bid + ask) / 2.0

    if level not in partial_dict:
        partial_dict[level] = [0.0, 0.0, 0]
    entry = partial_dict[level]
    if is_ask:
        entry[1] += vol
    else:
        entry[0] += vol
    entry[2] += 1

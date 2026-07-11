import numpy as np


FP_LEVEL_COLS = ("price", "vol_bid", "vol_ask", "vol_total", "delta", "n_trades")
N_FP_LEVEL_COLS = len(FP_LEVEL_COLS)
_FP_LEVEL_COL = {name: i for i, name in enumerate(FP_LEVEL_COLS)}

FP_SCALAR_COLS = (
    "poc_price",
    "poc_vol",
    "poc_idx_local",
    "vah",
    "val",
    "delta_total",
    "vol_total",
    "cvd",
    "levels",
)
N_FP_SCALAR_COLS = len(FP_SCALAR_COLS)
_FP_SCALAR_COL = {name: i for i, name in enumerate(FP_SCALAR_COLS)}


class FpCandle:
    __slots__ = (
        "price_levels",
        "poc_price",
        "poc_vol",
        "poc_idx",
        "vah",
        "val",
        "delta_total",
        "vol_total",
        "cvd",
        "levels",
        "is_partial",
    )

    def __init__(
        self, price_levels: np.ndarray, scalars: np.ndarray, is_partial: bool = False
    ):
        self.price_levels = price_levels
        self.poc_price = float(scalars[_FP_SCALAR_COL["poc_price"]])
        self.poc_vol = float(scalars[_FP_SCALAR_COL["poc_vol"]])
        self.poc_idx = int(scalars[_FP_SCALAR_COL["poc_idx_local"]])
        self.vah = float(scalars[_FP_SCALAR_COL["vah"]])
        self.val = float(scalars[_FP_SCALAR_COL["val"]])
        self.delta_total = float(scalars[_FP_SCALAR_COL["delta_total"]])
        self.vol_total = float(scalars[_FP_SCALAR_COL["vol_total"]])
        self.cvd = float(scalars[_FP_SCALAR_COL["cvd"]])
        self.levels = int(scalars[_FP_SCALAR_COL["levels"]])
        self.is_partial = is_partial

    def __repr__(self) -> str:
        tag = " [partial]" if self.is_partial else ""
        return (
            f"FpCandle{tag}(poc={self.poc_price:.2f} vol={self.poc_vol:.1f} "
            f"delta={self.delta_total:+.1f} levels={self.levels})"
        )

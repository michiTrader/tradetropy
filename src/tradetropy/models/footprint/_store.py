import numpy as np

from ._types import FpCandle
from ._compute import _dict_to_levels, _compute_scalars, _accumulate_to_dict


class FootprintStore:
    __slots__ = (
        "fp_data",
        "fp_idx",
        "fp_scalars",
        "config",
        "_n_closed",
        "_partial_dict",
        "_partial_ts",
        "_cvd_total",
        "_interval_ms",
    )

    def __init__(
        self,
        fp_data: np.ndarray,
        fp_idx: np.ndarray,
        fp_scalars: np.ndarray,
        config,
        cvd_total: float,
        interval_ms: int,
    ):
        self.fp_data = fp_data
        self.fp_idx = fp_idx
        self.fp_scalars = fp_scalars
        self.config = config
        self._n_closed = len(fp_scalars)
        self._partial_dict = {}
        self._partial_ts = -1
        self._cvd_total = cvd_total
        self._interval_ms = int(interval_ms)

    def process_tick(
        self,
        timestamp_ms: int,
        price: float,
        vol: float,
        bid: float,
        ask: float,
        flag: float = 0.0,
    ):
        candle_ts = (timestamp_ms // self._interval_ms) * self._interval_ms
        if candle_ts != self._partial_ts:
            self._partial_dict = {}
            self._partial_ts = candle_ts
        _accumulate_to_dict(self._partial_dict, price, vol, bid, ask, flag, self.config)

    def closed_candle(self, candle_idx: int) -> FpCandle:
        start = int(self.fp_idx[candle_idx])
        end = int(self.fp_idx[candle_idx + 1])
        return FpCandle(
            price_levels=self.fp_data[start:end],
            scalars=self.fp_scalars[candle_idx],
            is_partial=False,
        )

    def partial_candle(self) -> "FpCandle | None":
        levels = _dict_to_levels(self._partial_dict)
        if len(levels) == 0:
            return None
        scalars = _compute_scalars(
            levels, self.config.value_area_pct, self._cvd_total
        )
        return FpCandle(price_levels=levels, scalars=scalars, is_partial=True)

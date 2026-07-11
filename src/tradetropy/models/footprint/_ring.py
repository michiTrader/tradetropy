import numpy as np

from ._config import FootprintConfig, _round_tick_pretty
from ._types import FpCandle, N_FP_SCALAR_COLS, N_FP_LEVEL_COLS, _FP_SCALAR_COL
from ._compute import _price_level, _dict_to_levels, _compute_scalars, _accumulate_to_dict


class LiveFpRing:
    __slots__ = (
        "_buf_scalars",
        "_buf_levels",
        "_W",
        "_head",
        "_n_closed",
        "_partial_dict",
        "_ts_current_candle",
        "_cvd_accumulated",
        "config",
        "interval_ms",
        "_first_candle_prices",
    )

    def __init__(self, window_size: int, config: FootprintConfig, interval_ms: int):
        W = window_size
        self._W = W
        self._buf_scalars = np.full((W * 2, N_FP_SCALAR_COLS), np.nan, dtype=np.float64)
        empty = np.empty((0, N_FP_LEVEL_COLS), dtype=np.float64)
        self._buf_levels = [empty] * (W * 2)
        self._head = 0
        self._n_closed = 0
        self._partial_dict = {}
        self._ts_current_candle = -1
        self._cvd_accumulated = 0.0
        self.config = config
        self.interval_ms = int(interval_ms)
        self._first_candle_prices: "list[float] | None" = [] if config.tick_size is None else None

    def reset(self) -> None:
        """
        Clear all footprint candles, restoring the ring to its created state.

        Resets the scalar/level buffers, head/closed counters, accumulated CVD
        and the partial candle, preserving the config, capacity and interval.
        Used by ReplayEngine to rewind a replay in place without recreating the
        ring (the chart keeps its reference to this same object).
        """
        self._buf_scalars.fill(np.nan)
        empty = np.empty((0, N_FP_LEVEL_COLS), dtype=np.float64)
        self._buf_levels = [empty] * (self._W * 2)
        self._head = 0
        self._n_closed = 0
        self._partial_dict = {}
        self._ts_current_candle = -1
        self._cvd_accumulated = 0.0
        self._first_candle_prices = [] if self.config.tick_size is None else None

    def process_tick(
        self,
        timestamp_ms: int,
        price: float,
        vol: float,
        bid: float,
        ask: float,
        flag: float = 0.0,
    ):
        candle_ts = (timestamp_ms // self.interval_ms) * self.interval_ms
        if candle_ts != self._ts_current_candle:
            if self._ts_current_candle >= 0:
                self._confirm_partial_candle()
            self._ts_current_candle = candle_ts
            self._partial_dict = {}

        if self._first_candle_prices is not None:
            self._first_candle_prices.append(price)

        if self.config.tick_size is None:
            from dataclasses import replace
            provisional_tick_size = _round_tick_pretty(price * 0.0001)
            self.config = replace(self.config, tick_size=provisional_tick_size)

        _accumulate_to_dict(self._partial_dict, price, vol, bid, ask, flag, self.config)

    def _confirm_partial_candle(self):
        if self._first_candle_prices is not None and len(self._first_candle_prices) > 1:
            prices_arr = np.array(self._first_candle_prices, dtype=np.float64)
            price_range = float(prices_arr.max() - prices_arr.min())
            raw = price_range / self.config.levels if price_range > 0 else self.config.tick_size or 1.0
            correct_tick_size = _round_tick_pretty(raw)
            from dataclasses import replace
            self.config = replace(self.config, tick_size=correct_tick_size)
            self._first_candle_prices = None

        levels = _dict_to_levels(self._partial_dict)
        scalars = _compute_scalars(levels, self.config.value_area_pct, self._cvd_accumulated)
        self._cvd_accumulated = float(scalars[_FP_SCALAR_COL["cvd"]])

        p = self._head
        self._buf_scalars[p] = scalars
        self._buf_scalars[p + self._W] = scalars
        self._buf_levels[p] = levels
        self._buf_levels[p + self._W] = levels
        self._head = (p + 1) % self._W
        self._n_closed += 1

    def closed_candle(self, idx_from_end: int) -> "FpCandle | None":
        available = min(self._n_closed, self._W)
        if idx_from_end >= available:
            return None
        pos = self._head + self._W - 1 - idx_from_end
        return FpCandle(
            price_levels=self._buf_levels[pos],
            scalars=self._buf_scalars[pos],
            is_partial=False,
        )

    def partial_candle(self) -> "FpCandle | None":
        if self._ts_current_candle < 0:
            return None
        levels = _dict_to_levels(self._partial_dict)
        if len(levels) == 0:
            return None
        scalars = _compute_scalars(levels, self.config.value_area_pct, self._cvd_accumulated)
        return FpCandle(price_levels=levels, scalars=scalars, is_partial=True)

    @property
    def n_available_candles(self) -> int:
        has_partial = 1 if self._ts_current_candle >= 0 else 0
        return min(self._n_closed, self._W) + has_partial

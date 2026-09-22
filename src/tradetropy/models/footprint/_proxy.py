import numpy as np

from tradetropy.exceptions import DataError

from ._config import FootprintConfig
from ._types import FpCandle
from ._store import FootprintStore
from ._ring import LiveFpRing


class FpProxy:
    __slots__ = (
        "symbol",
        "interval_ms",
        "config",
        "_window_size",
        "_store",
        "_ring",
        "_cursor",
        "_mapping",
    )

    def __init__(
        self,
        symbol: str,
        interval_ms: int,
        window_size: int = 50,
        *,
        tick_size: float | None = None,
        levels: int = 5,
        value_area_pct: float = 0.70,
        aggressor_col: str | None = "flags",
        vol_col: str = "volume",
    ):
        self.symbol = symbol
        self.interval_ms = interval_ms
        self.config = FootprintConfig(
            tick_size=tick_size,
            levels=levels,
            value_area_pct=value_area_pct,
            aggressor_col=aggressor_col,
            vol_col=vol_col,
        )
        self._window_size = window_size
        self._store: "FootprintStore | None" = None
        self._ring:  "LiveFpRing | None"     = None
        self._cursor: int = -1
        self._mapping:  "np.ndarray | None" = None

    def _connect(self, store: FootprintStore):
        self._store  = store
        self.config  = store.config
        self._ring   = None
        self._mapping  = None

    def _connect_live(self, ring: LiveFpRing):
        self._ring = ring
        self._store = None

    def process_tick(
        self,
        timestamp_ms: int,
        price: float,
        vol: float,
        bid: float,
        ask: float,
        flag: float = 0.0,
    ):
        if self._store is not None:
            self._store.process_tick(timestamp_ms, price, vol, bid, ask, flag)
        elif self._ring is not None:
            self._ring.process_tick(timestamp_ms, price, vol, bid, ask, flag)

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor

    def _n_closed_until_cursor(self) -> int:
        if self._mapping is not None and self._cursor >= 0:
            return int(self._mapping[self._cursor])
        return self._store._n_closed

    @property
    def n_candles(self) -> int:
        if self._store is not None:
            n_c = self._n_closed_until_cursor()
            n_partial = 1 if self._store._partial_ts >= 0 else 0
            return n_c + n_partial
        if self._ring is not None:
            return self._ring.n_available_candles
        return 0

    def __getitem__(self, idx: int) -> "FpCandle | None":
        if idx >= 0:
            raise DataError(
                "Use negative indices: fp[-1]=partial, fp[-2]=last closed"
            )
        if self._store is not None:
            return self._getitem_store(idx)
        if self._ring is not None:
            return self._getitem_ring(idx)
        return None

    def _getitem_store(self, idx: int) -> "FpCandle | None":
        store = self._store
        n_closed = self._n_closed_until_cursor()

        if idx == -1:
            return store.partial_candle()

        closed_idx = n_closed + idx + 1
        if closed_idx < 0 or closed_idx >= n_closed:
            return None
        window_start = max(0, n_closed - self._window_size)
        if closed_idx < window_start:
            return None
        return store.closed_candle(closed_idx)

    def _getitem_ring(self, idx: int) -> "FpCandle | None":
        if idx == -1:
            return self._ring.partial_candle()

        idx_from_end = (-idx) - 2
        if idx_from_end < 0:
            return None
        available = min(self._ring._n_closed, self._ring._W)
        if idx_from_end >= min(available, self._window_size):
            return None
        return self._ring.closed_candle(idx_from_end)

    def __len__(self) -> int:
        return self.n_candles

    def __iter__(self):
        n = self.n_candles
        for i in range(-n, 0):
            v = self[i]
            if v is not None:
                yield v

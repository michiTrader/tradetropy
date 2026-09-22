"""
Tick-based volume profile (TickVolumeProfile).

Kept in its own module, separate from ``VolumeProfile`` /
``RollingVolumeProfile`` (which live in ``tradetropy.ta.volume``). The shared
base class and period/level helpers live in ``tradetropy.ta.volume`` and
``tradetropy.ta._volume_profile``.
"""

from __future__ import annotations

import numpy as np

from tradetropy.ta.volume import _VolumeProfileBase
from tradetropy.ta._volume_profile import level_price, resolve_period_ids


# =====
# TickVolumeProfile (ticks)
# =====
class TickVolumeProfile(_VolumeProfileBase):
    """
    Tick-based volume profile with real binning and aggressor delta.

    Source columns (use ``TickVolumeProfile.refs(tick_proxy)``):
        [ts, price, volume, flags]

    Each trade is binned at its real price level. The aggressor side is taken
    from the tick ``flags`` (bit 32 = buy, bit 64 = sell); when neither bit is
    set a tick-rule fallback is used (price >= previous price -> buy).
    """

    name = "tvp"

    source_is_tick = True

    @staticmethod
    def refs(tick_proxy):
        """
        Build the ColumnRef list for this indicator in the expected order.

        Args:
            tick_proxy (TickProxy): Proxy returned by subscribe_ticks().

        Returns:
            list[ColumnRef]: [ts, price, volume, flags] refs.
        """
        return [
            tick_proxy.col_ref("ts"),
            tick_proxy.price_ref,
            tick_proxy.col_ref("volume"),
            tick_proxy.col_ref("flags"),
        ]

    def calculate(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.shape[1] < 4 or len(source) == 0:
            return self._empty_result()

        ts = source[:, 0].astype(np.int64)
        price = source[:, 1].astype(np.float64)
        vol = source[:, 2].astype(np.float64)
        flags = source[:, 3].astype(np.int64)
        n = len(ts)

        tick_size = self._resolve_tick_size(price)
        self.tick_size_used_ = tick_size
        period_id, period_start = resolve_period_ids(ts, self.period_ms, self.anchor)

        levels = level_price(price, tick_size)

        # Aggressor classification: flags first, tick rule as fallback.
        is_buy_flag = (flags & 32) != 0
        is_sell_flag = (flags & 64) != 0
        prev = np.roll(price, 1)
        prev[0] = price[0]
        tick_buy = price >= prev
        is_buy = np.where(is_buy_flag, True, np.where(is_sell_flag, False, tick_buy))

        vol_ask = np.where(is_buy, vol, 0.0)
        vol_bid = np.where(is_buy, 0.0, vol)
        counts = np.ones(n, dtype=np.float64)

        return self._run_scan(
            n,
            period_id,
            np.arange(n, dtype=np.int64),
            levels,
            vol_bid,
            vol_ask,
            counts,
            period_start=period_start,
        )

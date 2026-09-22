"""
Volume Profile on bundled BTCUSDT 1-minute candles.

Uses the developing point of control / value area to fade extensions back into
the value area.

    python examples/volume_profile.py
"""

import numpy as np

from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import VolumeProfile


class VPStrategy(Strategy):
    def init(self):
        self.btc = self.subscribe_ohlc('BTCUSDT', '1m', window_size=500)
        self.vp = self.add_indicator(
            VolumeProfile.refs(self.btc),
            VolumeProfile(period='1d', nodes='both'),
        )

    def on_data(self):
        poc = self.vp.poc[-1]
        vah = self.vp.vah[-1]
        val = self.vp.val[-1]
        price = self.btc.close[-1]

        if np.isnan(poc):
            return
        if price < val and not self.sesh.positions('BTCUSDT'):
            self.sesh.buy('BTCUSDT', volume=1)
        elif price > vah:
            for pos in self.sesh.positions('BTCUSDT'):
                self.sesh.position_close(pos.ticket)


if __name__ == '__main__':
    bt = BacktestEngine.by_klines(VPStrategy(), data=(load_btcusd_1m(),))
    bt.run()
    print(bt.stats)

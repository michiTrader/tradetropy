"""
Simple moving-average crossover on bundled BTCUSDT 1-minute candles.

Run it directly:

    python examples/sma_cross.py
"""

from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import SMA


class SmaCross(Strategy):
    """Go long when the fast SMA is above the slow SMA."""

    def init(self):
        self.btc = self.subscribe_ohlc('BTCUSDT', '1m', window_size=200)
        self.fast = self.add_indicator(self.btc.close, SMA(10))
        self.slow = self.add_indicator(self.btc.close, SMA(30))

    def on_data(self):
        if self.fast[-1] > self.slow[-1]:
            if not self.sesh.positions('BTCUSDT'):
                self.sesh.buy('BTCUSDT', volume=1)
        else:
            for pos in self.sesh.positions('BTCUSDT'):
                self.sesh.position_close(pos.ticket)


if __name__ == '__main__':
    engine = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
    engine.run()
    print(engine.stats)

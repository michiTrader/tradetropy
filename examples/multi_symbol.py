"""
Multi-symbol backtest.

Pass one KlineData per symbol to ``by_klines``. The strategy subscribes to each
symbol independently and trades them side by side.

    python examples/multi_symbol.py
"""

from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_adausd_1m, load_btcusd_1m
from tradetropy.ta import SMA

SYMBOLS = ('BTCUSDT', 'ADAUSDT')


class MultiSymbolSma(Strategy):
    def init(self):
        self.feeds = {}
        self.fast = {}
        self.slow = {}
        for sym in SYMBOLS:
            feed = self.subscribe_ohlc(sym, '1m', window_size=200)
            self.feeds[sym] = feed
            self.fast[sym] = self.add_indicator(feed.close, SMA(10))
            self.slow[sym] = self.add_indicator(feed.close, SMA(30))

    def on_data(self):
        for sym in SYMBOLS:
            if self.fast[sym][-1] > self.slow[sym][-1]:
                if not self.sesh.positions(sym):
                    self.sesh.buy(sym, volume=1)
            else:
                for pos in self.sesh.positions(sym):
                    self.sesh.position_close(pos.ticket)


if __name__ == '__main__':
    data = (load_btcusd_1m(), load_adausd_1m())
    bt = BacktestEngine.by_klines(MultiSymbolSma(), data=data)
    bt.run()
    print(bt.stats)

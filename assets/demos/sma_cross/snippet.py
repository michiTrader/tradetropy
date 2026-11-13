from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_goog_1d
from tradetropy.signal import Signal
from tradetropy.ta import SMA


class SmaCross(Strategy):
    """Go long on a fast/slow SMA crossover, flatten on the crossunder."""

    def init(self):
        self.goog = self.subscribe_ohlc('GOOG', '1d', window_size=200)
        self.fast = self.add_indicator(self.goog.close, SMA(10))
        self.slow = self.add_indicator(self.goog.close, SMA(30))
        self.signal = Signal('partial')

    def on_data(self):
        if self.signal.crossover(self.fast, self.slow):
            if not self.sesh.positions('GOOG'):
                self.sesh.buy('GOOG', volume=10)
        elif self.signal.crossunder(self.fast, self.slow):
            for pos in self.sesh.positions('GOOG'):
                self.sesh.position_close(pos.ticket)


bt = BacktestEngine.by_klines(SmaCross(), data=(load_goog_1d(),))
bt.run()
print(bt.stats)

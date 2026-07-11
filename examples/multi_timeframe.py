"""
Multi-timeframe from a single dataset (automatic internal resampling).

Load ONE base series (here BTCUSDT 1m) and subscribe to as many timeframes as
you want from the same symbol. The engine resamples each higher timeframe
internally from the base candles - you never store or pass one file per
timeframe, and you never call resample() by hand. Subscribing to '1m', '5m'
and '15m' over 1m data gives three synchronized OHLC proxies driven off the
same source.

This strategy trades a classic multi-timeframe confluence: the slow 15m SMA
sets the trend bias, and a fast/slow crossover on the 5m candles times the
entries in the direction of that bias.

    python examples/multi_timeframe.py
"""

from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import SMA


class MultiTimeframe(Strategy):
    def init(self):
        # One symbol, three timeframes - all resampled internally from the
        # single 1m dataset passed to the engine below.
        self.m1 = self.subscribe_ohlc('BTCUSDT', '1m', window_size=200)
        self.m5 = self.subscribe_ohlc('BTCUSDT', '5m', window_size=200)
        self.m15 = self.subscribe_ohlc('BTCUSDT', '15m', window_size=200)

        # Higher-timeframe trend bias (15m) and entry signal (5m).
        self.trend = self.add_indicator(self.m15.close, SMA(20))
        self.fast = self.add_indicator(self.m5.close, SMA(5))
        self.slow = self.add_indicator(self.m5.close, SMA(20))

    def on_data(self):
        uptrend = self.m15.close[-1] > self.trend[-1]
        cross_up = self.fast[-1] > self.slow[-1]

        if uptrend and cross_up:
            if not self.sesh.positions('BTCUSDT'):
                self.sesh.buy('BTCUSDT', volume=1)
        else:
            for pos in self.sesh.positions('BTCUSDT'):
                self.sesh.position_close(pos.ticket)


if __name__ == '__main__':
    btc_1m = load_btcusd_1m()

    # A single base series feeds every subscribed timeframe. The engine reports
    # how many candles each resampled timeframe yields off the 1m source.
    engine = BacktestEngine.by_klines(MultiTimeframe(), data=(btc_1m,))
    engine.run()
    print(engine.stats)

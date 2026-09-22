"""
Order flow L2: Deep Trades over a recorded order book (ADAUSDT).

The bundled ADAUSDT ticks and order book share the same timestamp range, so they
replay together through ReplayEngine. DeepTrades reads the book as-of each trade
and classifies the outsized prints into aggressor / absorption / sweep.

Key point: a plain BacktestEngine has NO order book (its book stays stale), so
the L2 path needs ReplayEngine with book=BookData (or a live session).
The order book is fed to the DeepTrades constructor, not through the column refs.

This opens the interactive replay in your browser with play/pause/step/speed
controls:

    python examples/orderflow_l2.py
"""

import numpy as np

from tradetropy import Strategy
from tradetropy.datasets import load_adausd_book, load_adausd_ticks
from tradetropy.plotting import PlotConfig
from tradetropy.plotting.live import LiveChart
from tradetropy.replay import ReplayEngine
from tradetropy.ta import DeepTrades


class DeepFlow(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks('ADAUSDT', window_size=2000)
        self.book = self.subscribe_orderbook('ADAUSDT', depth=5)
        self.deep = self.add_indicator(
            DeepTrades.refs(self.ticks),                 # [ts, price, volume, flags, bid, ask]
            DeepTrades(self.book, threshold=2000.0, by='volume', window=500),
        )

    def on_data(self):
        et = self.deep.event_type[-1]
        if not np.isnan(et):
            name = DeepTrades.class_name(int(et))
            if name == 'sweep':
                self.sesh.buy('ADAUSDT', volume=1)        # follow the sweep


if __name__ == '__main__':
    engine = ReplayEngine.by_ticks(
        DeepFlow(),
        data=(load_adausd_ticks(),),
        book=load_adausd_book(),   # <- the order book enters here
        speed=20.0,
    )
    chart = LiveChart(config=PlotConfig(theme='dark'), max_candles=200)
    engine.run(live_chart=chart)

"""
Bookmap-style L2 heatmap over a recorded order book (ADAUSDT).

The Heatmap indicator turns the L2 book's evolution into a time x price
liquidity grid (colored cells), overlays the Best Bid / Ask lines and the
executed-volume bubbles, and exposes a public query API so the strategy can
read the liquidity picture causally from ``on_data()``.

Like Deep Trades, the L2 path needs an order book: ReplayEngine with
book=BookData (or a live session). The order book is passed to the
Heatmap constructor, not through the column refs.

This opens the interactive replay in your browser with play/pause/step/speed:

    python examples/heatmap_l2.py
"""

import numpy as np

from tradetropy import Strategy
from tradetropy.datasets import load_adausd_book, load_adausd_ticks
from tradetropy.plotting import PlotConfig
from tradetropy.plotting.live import LiveChart
from tradetropy.replay import ReplayEngine
from tradetropy.backtest import BacktestEngine
from tradetropy.ta import Heatmap


class HeatmapFlowSTGY(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks('ADAUSDT', window_size=8000)
        self.book = self.subscribe_orderbook('ADAUSDT', depth=5)
        self.ohlc = self.subscribe_ohlc('ADAUSDT', window_size=8000, timeframe="1m")
        self.heat = self.add_indicator(
            Heatmap.refs(self.ticks),               # [ts, price, volume, flags, bid, ask]
            Heatmap(
                self.book,
                tick_size=0.0005,
                price_bucket_ticks=1,               # price aggregation (x tick)
                time_bucket_ms=60_000,
                colormap='hot',                     # 'hot' | 'viridis' | 'greyscale'
                show_bbo=True,                      # Best Bid / Ask lines
                show_bubbles=True,                  # executed-volume bubbles
                persistence_ms=1500,                # wall persistence window
                # color_scale=(7_000_000.0, 9_000_000.0),
            ),
        )
        self.last_minute_print = 0

    def on_data(self):
        if self.heat.stale:
            return
        # Read the liquidity picture from the public query API.
        wall = self.heat.nearest_wall('ask')        # nearest resistance wall
        if wall is not None and wall.persistence_ms >= 1500:
            # A persistent ask wall above: liquidity resting into it.
            resting = self.heat.liquidity_at(wall.price, side='ask')
            if resting > 0 and not self.sesh.positions('ADAUSDT'):
                self.sesh.sell('ADAUSDT', volume=1000)  # fade into the wall

        try:
            current_minute = self.sesh.time.minute
            if current_minute != self.last_minute_print:
                self.last_minute_print = current_minute
                print(f"t:{self.sesh.time}, v: {self.ohlc.volume[-2] }")

        except:
            pass

if __name__ == '__main__':
    #Backetst
    bt = BacktestEngine.by_ticks(
        HeatmapFlowSTGY(),
        data=(load_adausd_ticks(),),
        book=load_adausd_book(),
    )

    bt.run()
    bt.plot(theme='dark', max_candles=1000, plot_footprint=False)

    #Replay
    bt = ReplayEngine.by_ticks(
        HeatmapFlowSTGY(),
        data=(load_adausd_ticks(),),
        book=load_adausd_book(),                     # <- the order book enters here
        speed=20.0,
    )
    chart = LiveChart(config=PlotConfig(theme='dark'), max_candles=1000)
    bt.run(live_chart=chart)

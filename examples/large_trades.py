"""
Order flow: Large Trades on bundled MESU26 futures ticks.

Detects outsized aggressive prints ('whale' trades) from the trade stream and
follows the buy side.

    python examples/large_trades.py
"""

import numpy as np

from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_mesu26_ticks
from tradetropy.ta import LargeTrades


class WhaleWatch(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks('MESU26', window_size=2000)
        self.whales = self.add_indicator(
            LargeTrades.refs(self.ticks),
            LargeTrades(threshold='p99', by='notional', window=1000,
                        label='notional'),
        )

    def on_data(self):
        if not np.isnan(self.whales.price[-1]):
            side = self.whales.side[-1]
            if side > 0 and not self.sesh.positions('MESU26'):
                self.sesh.buy('MESU26', volume=1)
            elif side < 0:
                for pos in self.sesh.positions('MESU26'):
                    self.sesh.position_close(pos.ticket)


if __name__ == '__main__':
    engine = BacktestEngine.by_ticks(WhaleWatch(), data=(load_mesu26_ticks(),))
    engine.run()
    print(engine.stats)

"""
Duplicate identical indicators (same class+params on the same source) are
computed ONCE and share a single store column, instead of recomputing and
appending an identical column per duplicate. Values stay correct.
"""

import numpy as np

from tradetropy import BacktestEngine, Strategy
from tradetropy.core.data_types import KlineData
from tradetropy.core.constants import N_OHLC_COLS
from tradetropy.ta import SMA, BollingerBands


def _klines(n=400):
    m = np.zeros((n, 7), dtype=np.float64)
    m[:, 0] = np.arange(n) * 60_000.0
    px = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
    m[:, 1] = px
    m[:, 2] = px + 0.5
    m[:, 3] = px - 0.5
    m[:, 4] = px
    m[:, 5] = 1.0
    return KlineData(symbol='SYM', data=m, timeframe='1m', tick_size=0.01)


class _DupStrat(Strategy):
    def init(self):
        self.px = self.subscribe_ohlc('SYM', '1m', window_size=100)
        self.a = self.add_indicator(self.px.close, SMA(20))
        self.b = self.add_indicator(self.px.close, SMA(20))  # duplicate
        self.c = self.add_indicator(self.px.close, SMA(20))  # duplicate
        self.d = self.add_indicator(self.px.close, SMA(30))  # distinct param
        self.samples = []

    def on_data(self):
        self.samples.append((self.a[-1], self.b[-1], self.c[-1], self.d[-1]))


def test_duplicate_indicators_share_one_column():
    kl = _klines()
    strat = _DupStrat()
    eng = BacktestEngine.by_klines(strat, data=(kl,))
    eng.run()

    # Three identical SMA(20) + one SMA(30) => only TWO indicator columns added
    # beyond the OHLC block (dedup), not four.
    store = next(iter(eng._ohlc_stores.values()))
    n_ind_cols = store.matrix.shape[1] - N_OHLC_COLS
    assert n_ind_cols == 2, f"expected 2 indicator columns after dedup, got {n_ind_cols}"

    # Duplicates read identical values; the distinct param differs eventually.
    a, b, c, d = zip(*strat.samples)
    assert a == b == c, "duplicate SMA(20) proxies must read identical values"

    # Correctness: the shared column equals a straight SMA(20) of close.
    close = kl.data[:, 4]
    ref = np.convolve(close, np.ones(20) / 20, mode='valid')
    # last on_data value corresponds to the last closed bar's SMA
    assert abs(a[-1] - ref[-1]) < 1e-9


def test_duplicate_multiband_indicators_share_columns():
    kl = _klines()

    class _S(Strategy):
        def init(self):
            self.px = self.subscribe_ohlc('SYM', '1m', window_size=100)
            self.bb1 = self.add_indicator(self.px.close, BollingerBands(20, 2.0))
            self.bb2 = self.add_indicator(self.px.close, BollingerBands(20, 2.0))
            self.vals = []

        def on_data(self):
            self.vals.append((self.bb1.upper[-1], self.bb2.upper[-1],
                              self.bb1.lower[-1], self.bb2.lower[-1]))

    strat = _S()
    eng = BacktestEngine.by_klines(strat, data=(kl,))
    eng.run()

    store = next(iter(eng._ohlc_stores.values()))
    n_ind_cols = store.matrix.shape[1] - N_OHLC_COLS
    # One BollingerBands => 3 bands; deduped duplicate adds nothing.
    assert n_ind_cols == 3, f"expected 3 band columns after dedup, got {n_ind_cols}"

    u1, u2, l1, l2 = zip(*strat.vals)
    assert u1 == u2 and l1 == l2

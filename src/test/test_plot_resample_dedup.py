"""
Regression tests for plotting resample with duplicate-ms tick timestamps.

A tick-mounted indicator (e.g. LargeTrades / Heatmap) has its output bands
resampled from the tick timestamps onto the OHLC candle index. The tick stream
deliberately keeps same-millisecond trades (the streaming path never dedups
timestamps), so the source index carries duplicate labels. Pandas refuses to
`reindex` on an axis with duplicate labels, which used to crash `plot()` with:

    ValueError: cannot reindex on an axis with duplicate labels

The fix deduplicates the source index (keep='last', causal with the ffill) in
`_resample_indicator_to_index`. These tests cover both the pure resample helper
and the real `_gather_plot_data` call site.
"""

import numpy as np
import pandas as pd
import pytest

from tradetropy.plotting._util import (
    _resample_indicator_to_index,
    _gather_plot_data,
)


# =====
# Pure resample helper
# =====
class TestResampleDedup:
    def test_duplicate_source_index_does_not_raise(self):
        # Two ticks share ms=0 -> duplicate label in the source index.
        values = np.array([1.0, 2.0, 3.0, 4.0])
        src = pd.to_datetime([0, 0, 1000, 2000], unit="ms", utc=True)
        tgt = pd.to_datetime([0, 500, 1000, 3000], unit="ms", utc=True)

        out = _resample_indicator_to_index(values, src, tgt)

        # keep='last' picks value 2.0 at ms=0 (not 1.0); reindex keeps only the
        # target timestamps and ffills the gaps between them.
        # ms=0    -> 2.0 (last at the duplicate label)
        # ms=500  -> 2.0 (ffill from ms=0)
        # ms=1000 -> 3.0
        # ms=3000 -> 3.0 (ms=2000 is not a target point; ffill from ms=1000)
        assert out.dtype == np.float64
        assert np.allclose(out, [2.0, 2.0, 3.0, 3.0])

    def test_unique_source_index_unchanged(self):
        # No duplicates -> behaves exactly as a plain reindex + ffill.
        values = np.array([10.0, 20.0, 30.0])
        src = pd.to_datetime([0, 1000, 2000], unit="ms", utc=True)
        tgt = pd.to_datetime([0, 500, 2000], unit="ms", utc=True)

        out = _resample_indicator_to_index(values, src, tgt)
        assert np.allclose(out, [10.0, 10.0, 30.0])

    def test_unsorted_duplicate_source_is_ordered(self):
        # Out-of-order timestamps with a duplicate must still align causally.
        values = np.array([5.0, 1.0, 2.0, 9.0])
        src = pd.to_datetime([2000, 0, 0, 1000], unit="ms", utc=True)
        tgt = pd.to_datetime([0, 1000, 2000], unit="ms", utc=True)

        out = _resample_indicator_to_index(values, src, tgt)
        # dedup keep='last' at ms=0 -> 2.0; sorted -> [0:2.0, 1000:9.0, 2000:5.0]
        assert np.allclose(out, [2.0, 9.0, 5.0])


# =====
# Real call site: tick backtest + tick indicator + duplicate-ms ticks
# =====
def _ticks_with_dup_ms(n=600, seed=3):
    """Raw [N x 7] ticks (ts,bid,ask,volume,flags,vreal,price) spanning several
    minutes at 1s spacing, with a few same-ms duplicate timestamps injected."""
    rng = np.random.default_rng(seed)
    ts = (np.arange(n) * 1000).astype(np.float64)     # 1s spacing -> n seconds
    # Inject same-ms duplicates (as the streaming path would deliver them).
    ts[100] = ts[99]
    ts[300] = ts[299]
    ts[301] = ts[299]
    price = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    volume = rng.uniform(1.0, 5.0, n)
    volume[70] = 100.0                                # one large trade to detect
    flags = np.ones(n)
    vreal = volume.copy()
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, bid, ask, volume, flags, vreal, price])


@pytest.mark.unit
class TestGatherPlotDataDupTicks:
    def test_gather_plot_data_with_duplicate_ms_ticks(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.ta.order_flow import LargeTrades

        raw = _ticks_with_dup_ms()

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=800)
                self.deep = self.add_indicator(
                    LargeTrades.refs(self.tk),
                    LargeTrades(threshold=50.0, by="volume"),
                )

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", raw, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()

        # Before the fix this raised:
        #   ValueError: cannot reindex on an axis with duplicate labels
        data = _gather_plot_data(bt)

        assert data["ohlc_array"] is not None
        assert len(data["indicators"]) >= 1

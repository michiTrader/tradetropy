"""
Multi-source indicators on kline-driven engines (partial-bar view).

Regression guard for a crash in the kline branch of ``_IndicatorView._view``:
it built the partial-bar window with the *single-source* code path, so any
indicator whose ``source_cols`` holds more than one column (ATR/HLC,
Stochastic/HLC, MarketStructure/HLC+ts, ...) blew up with

    TypeError: only length-1 arrays can be converted to Python scalars

as soon as a strategy read the indicator inside ``on_data()`` under
``BacktestEngine.by_klines``. The tick branch already handled the list case;
the kline branch now mirrors it.

This was invisible in the existing suite because the multi-source coverage ran
through tick-driven engines, while the kline coverage used single-source
indicators (SMA/RSI over close). The gap sat exactly at their intersection -
which is also the most common real setup: a kline backtest reading ATR.

Usage:
    pytest src/test/test_multisource_kline_partial.py -v
"""

import numpy as np
import pytest

from tradetropy.backtest import BacktestEngine
from tradetropy.datasets import load_btcusd_1m
from tradetropy.models.strategy import Strategy
from tradetropy.ta import ATR, MarketStructure, Stochastic


def _run(indicator_factory, reader, window_size=200):
    """
    Drive a kline backtest that reads a multi-source indicator every bar.

    Returns the number of bars where the reader reported a usable value.
    Before the fix this raised TypeError on the very first bar.
    """
    kl = load_btcusd_1m()

    class _S(Strategy):
        def init(self):
            self.ohlc = self.subscribe_ohlc(
                kl.symbol, '1m', window_size=window_size
            )
            ind = indicator_factory()
            self.ind = self.add_indicator(
                type(ind).refs(self.ohlc), ind, plot=False
            )
            self.hits = 0

        def on_data(self):
            if reader(self.ind):
                self.hits += 1

    bt = BacktestEngine.by_klines(_S(), data=(kl,))
    bt.run(verbose=False)
    return bt.strategy.hits


def _finite(series) -> bool:
    """True when the freshest evaluated slot holds a real number."""
    for off in (-1, -2):
        if len(series) >= abs(off) and not np.isnan(series[off]):
            return True
    return False


@pytest.mark.unit
class TestMultiSourceKlinePartial:
    def test_atr_hlc_readable_on_klines(self):
        """ATR consumes (high, low, close): 3 columns -> list index path."""
        hits = _run(lambda: ATR(14), lambda ind: _finite(ind))
        assert hits > 100, f"ATR produced only {hits} usable bars"

    def test_stochastic_hlc_does_not_crash_on_klines(self):
        """
        Stochastic also consumes (high, low, close), so it exercises the same
        list-index path and must not raise.

        Only the crash is asserted, not the values: Stochastic().calculate()
        currently returns all-NaN even when called directly on clean synthetic
        HLC data, which is a separate pre-existing bug in the indicator
        itself - unrelated to this partial-window fix. Asserting finite values
        here would hide that bug behind an unrelated failure.
        """
        hits = _run(lambda: Stochastic(), lambda ind: len(ind.k) > 0)
        assert hits > 100, "Stochastic view was unreadable under klines"

    def test_market_structure_readable_on_klines(self):
        """MarketStructure consumes (high, low, close, ts): 4 columns."""
        hits = _run(
            lambda: MarketStructure(swing=3),
            lambda ind: len(ind.bos_bull) > 0,
        )
        assert hits > 100

    def test_no_crash_on_first_bars(self):
        """
        The original failure happened on the first on_data() call, before any
        warmup completed. Reading from bar zero must not raise.
        """
        kl = load_btcusd_1m()

        class _S(Strategy):
            def init(self):
                self.ohlc = self.subscribe_ohlc(kl.symbol, '1m', window_size=50)
                self.atr = self.add_indicator(
                    ATR.refs(self.ohlc), ATR(14), plot=False
                )
                self.calls = 0

            def on_data(self):
                _ = self.atr[-1]      # must not raise, NaN is fine
                self.calls += 1

        bt = BacktestEngine.by_klines(_S(), data=(kl,))
        bt.run(verbose=False)
        assert bt.strategy.calls > 400

    def test_partial_window_matches_direct_calculate(self):
        """
        The value read on the last CLOSED bar must equal a direct calculate()
        over the same history: the partial-bar window must not distort
        closed-bar values.
        """
        kl = load_btcusd_1m()
        d = kl.data
        expected = ATR(14).calculate(
            np.column_stack([d[:, 2], d[:, 3], d[:, 4]])
        )

        class _S(Strategy):
            def init(self):
                self.ohlc = self.subscribe_ohlc(
                    kl.symbol, '1m', window_size=len(d)
                )
                self.atr = self.add_indicator(
                    ATR.refs(self.ohlc), ATR(14), plot=False
                )
                self.samples = []

            def on_data(self):
                n_closed = len(self.ohlc.ts) - 1
                if n_closed >= 20 and len(self.atr) >= 2:
                    self.samples.append((n_closed, float(self.atr[-2])))

        bt = BacktestEngine.by_klines(_S(), data=(kl,))
        bt.run(verbose=False)
        samples = bt.strategy.samples
        assert samples, "no samples captured"

        mismatches = [
            (idx, got, float(expected[idx - 1]))
            for idx, got in samples
            if not np.isnan(expected[idx - 1])
            and not np.isclose(got, expected[idx - 1], rtol=1e-6)
        ]
        assert not mismatches, (
            f"{len(mismatches)} closed-bar values differ from calculate(); "
            f"first: {mismatches[:3]}"
        )

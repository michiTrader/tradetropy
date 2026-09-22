"""
Fast-path correctness for indicator access in candle (kline_mode) backtests.

BacktestEngine.by_klines exposes each indicator's developing-bar value through
OhlcIndicatorView. A scalar read (self.sma[-1] / self.sma[-2]) resolves the
single value directly instead of materializing the whole trailing window, and
for WINDOWED indicators (warmup_factor == 1) the developing value is read from
the full-history precompute (it is not path-dependent). This guards that the
optimization returns the correct values:

    1. Windowed (SMA): the developing self.sma[-1] equals the plain mean of the
       last `length` closes at every bar (independent NumPy ground truth), and
       self.sma[-2] equals the previous closed bar's SMA.
    2. Recursive (RSI, warmup_factor > 1): keeps the causal-window recompute
       path - values stay finite and bounded in (0, 100) after warmup (the
       cross-engine numeric parity is covered by test_indicator_engine_parity).

Usage:
    pytest src/test/test_kline_indicator_fastpath.py -v
"""

import numpy as np

from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import SMA, RSI

_L = 5


def _run_capture():
    """Run a candle backtest, capturing per-bar developing indicator reads."""
    kl = load_btcusd_1m()
    ts_to_idx = {int(t): i for i, t in enumerate(kl.data[:, 0])}
    captured = {}

    class _S(Strategy):
        warmup = 0

        def init(self):
            self.px = self.subscribe_ohlc(
                kl.symbol, kl.timeframe, window_size=300
            )
            self.sma = self.add_indicator(self.px.close, SMA(_L))
            self.rsi = self.add_indicator(self.px.close, RSI(14))
            self._n = 0

        def on_data(self):
            self._n += 1
            ts = int(self.px.ts[-1])
            snap = {
                "sma_last": float(self.sma[-1]),
                "rsi_last": float(self.rsi[-1]),
            }
            if self._n >= 2:
                snap["sma_prev"] = float(self.sma[-2])
            captured[ts] = snap

    BacktestEngine.by_klines(_S(), data=(kl,)).run()
    return kl, ts_to_idx, captured


def test_windowed_developing_value_matches_numpy():
    """Developing SMA[-1] equals the NumPy mean of the last `length` closes."""
    kl, ts_to_idx, captured = _run_capture()
    closes = kl.data[:, 4]

    checked = 0
    for ts, snap in captured.items():
        i = ts_to_idx[ts]                # developing candle index (0-based)
        if i + 1 < _L:
            continue                     # window not full yet
        ref = float(np.mean(closes[i - _L + 1 : i + 1]))
        assert abs(snap["sma_last"] - ref) < 1e-6, (i, snap["sma_last"], ref)
        checked += 1
    assert checked > 100                 # actually exercised the fast path


def test_windowed_previous_value_matches_numpy():
    """SMA[-2] equals the SMA of the previous (closed) bar."""
    kl, ts_to_idx, captured = _run_capture()
    closes = kl.data[:, 4]

    checked = 0
    for ts, snap in captured.items():
        i = ts_to_idx[ts]
        if i < _L or "sma_prev" not in snap:  # need a full window one bar back
            continue
        ref = float(np.mean(closes[i - _L : i]))
        assert abs(snap["sma_prev"] - ref) < 1e-6, (i, snap["sma_prev"], ref)
        checked += 1
    assert checked > 100


def test_recursive_values_stay_bounded():
    """Recursive RSI keeps the recompute path: finite and within (0, 100)."""
    kl, ts_to_idx, captured = _run_capture()

    seen_valid = 0
    for snap in captured.values():
        v = snap["rsi_last"]
        if np.isnan(v):
            continue
        assert 0.0 <= v <= 100.0, v
        seen_valid += 1
    assert seen_valid > 100

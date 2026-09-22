"""
Regression tests for the developing (partial) candle in the static backtest
chart when the strategy has NO subscribe_ohlc().

A tick-only strategy (order flow / L2, e.g. examples/heatmap_l2.py) never
declares subscribe_ohlc(), so `_gather_plot_data` synthesizes the candles from
the raw ticks via `build_candles_from_ticks`. That helper returns only the
CLOSED candles (n_candles_total - 1); the last, in-progress candle was dropped,
so `bt.plot()` did not draw the final bar even though the tick-mounted overlays
(Heatmap BBO / bubbles) reach the last tick. Replay/live show it because they
keep a developing candle in the OHLC ring.

The fix re-appends the developing candle in the tick fallback of
`_gather_plot_data`, reconstructed from the arrays `build_candles_from_ticks`
already returns (parity with `OhlcDataStore.partial_tick_candle`).
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL
from tradetropy.core import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.session import SeshSimulatorBase
from tradetropy.backtest.engine import BacktestEngine
from tradetropy.plotting._util import _gather_plot_data


_INTERVAL_MS = 60_000


def _ticks_ending_mid_candle(n_full_minutes=2, tail_ticks=30, seed=7):
    """
    Raw [N x 7] ticks (ts,bid,ask,volume,flags,vreal,price) at 1s spacing that
    span `n_full_minutes` complete 1m candles plus a partial last candle made of
    `tail_ticks` ticks (< 60), so the last minute is still developing.
    """
    rng = np.random.default_rng(seed)
    n_full = n_full_minutes * 60
    n = n_full + tail_ticks
    ts = (np.arange(n) * 1000).astype(np.float64)      # 1s spacing
    price = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    volume = rng.uniform(1.0, 5.0, n)
    flags = np.ones(n)
    vreal = volume.copy()
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, bid, ask, volume, flags, vreal, price])


def _expected_last_candle(raw):
    """Compute the developing (last-bucket) candle straight from the ticks."""
    ts = raw[:, _TICK_COL["ts"]]
    price = raw[:, _TICK_COL["price"]]
    volume = raw[:, _TICK_COL["volume"]]
    bucket = (ts // _INTERVAL_MS) * _INTERVAL_MS
    last_bucket = bucket[-1]
    mask = bucket == last_bucket
    return {
        "ts": float(last_bucket),
        "open": float(price[mask][0]),
        "high": float(price[mask].max()),
        "low": float(price[mask].min()),
        "close": float(price[mask][-1]),
        "volume": float(volume[mask].sum()),
    }


def _n_buckets(raw):
    ts = raw[:, _TICK_COL["ts"]]
    bucket = (ts // _INTERVAL_MS) * _INTERVAL_MS
    return len(np.unique(bucket))


class _TickOnly(Strategy):
    """Order-flow style: ticks only, no subscribe_ohlc()."""

    def init(self):
        self.tk = self.subscribe_ticks("SYM", window_size=800)

    def on_data(self):
        pass


def _run_bt(raw):
    bt = BacktestEngine.by_ticks(
        _TickOnly(),
        data=(TickData("SYM", raw, tick_size=0.01),),
        sesh=SeshSimulatorBase("tick"),
    )
    bt.run()
    return bt


@pytest.mark.unit
class TestBacktestPartialCandle:
    def test_tick_only_chart_includes_developing_candle(self):
        raw = _ticks_ending_mid_candle(n_full_minutes=2, tail_ticks=30)
        bt = _run_bt(raw)

        data = _gather_plot_data(bt)
        ohlc = data["ohlc_array"]

        # closed candles + 1 developing candle == number of distinct buckets.
        assert len(ohlc) == _n_buckets(raw)

        exp = _expected_last_candle(raw)
        last = ohlc[-1]
        assert last[0] == pytest.approx(exp["ts"])
        assert last[1] == pytest.approx(exp["open"])
        assert last[2] == pytest.approx(exp["high"])
        assert last[3] == pytest.approx(exp["low"])
        assert last[4] == pytest.approx(exp["close"])
        assert last[5] == pytest.approx(exp["volume"])

    def test_single_developing_interval_yields_one_candle(self):
        # All ticks fall inside one 1m bucket: zero closed candles, so the only
        # bar is the developing one. It must still be drawn (not an empty chart).
        raw = _ticks_ending_mid_candle(n_full_minutes=0, tail_ticks=40)
        assert _n_buckets(raw) == 1

        data = _gather_plot_data(_run_bt(raw))
        ohlc = data["ohlc_array"]

        assert len(ohlc) == 1
        exp = _expected_last_candle(raw)
        assert ohlc[-1][1] == pytest.approx(exp["open"])
        assert ohlc[-1][4] == pytest.approx(exp["close"])
        assert ohlc[-1][5] == pytest.approx(exp["volume"])

    def test_tick_indicator_overlay_aligns_to_developing_candle(self):
        # A tick-mounted indicator (LargeTrades) must still align to the OHLC
        # grid once the developing candle is appended (no length mismatch).
        from tradetropy.ta.order_flow import LargeTrades

        raw = _ticks_ending_mid_candle(n_full_minutes=2, tail_ticks=30)

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=800)
                self.deep = self.add_indicator(
                    LargeTrades.refs(self.tk),
                    LargeTrades(threshold=1e9, by="volume"),
                )

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", raw, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()

        data = _gather_plot_data(bt)
        assert len(data["ohlc_array"]) == _n_buckets(raw)
        assert len(data["indicators"]) >= 1

    def test_subscribe_ohlc_path_matches_tick_fallback_count(self):
        # With an explicit subscribe_ohlc the OHLC-store branch already appends
        # the partial candle. Both paths must yield the same number of bars for
        # identical ticks (no double count, no missing bar).
        raw = _ticks_ending_mid_candle(n_full_minutes=2, tail_ticks=30)

        class SWithOhlc(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=800)
                self.bars = self.subscribe_ohlc("SYM", _INTERVAL_MS, window_size=800)

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            SWithOhlc(),
            data=(TickData("SYM", raw, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()

        data = _gather_plot_data(bt)
        assert len(data["ohlc_array"]) == _n_buckets(raw)
        exp = _expected_last_candle(raw)
        last = data["ohlc_array"][-1]
        assert last[0] == pytest.approx(exp["ts"])
        assert last[4] == pytest.approx(exp["close"])

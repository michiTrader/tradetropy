"""Tests for the FibRetracement on-demand tool."""

from __future__ import annotations

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, N_OHLC_COLS, _TICK_COL, _OHLC_COL


def _make_tick_source(ts, price, volume=None):
    """Minimal source proxy exposing ts/price columns (tick-like)."""
    class _Src:
        price_ref = object()
        def __init__(self, ts, price, vol):
            self.ts = np.asarray(ts, dtype=np.int64)
            self.price = np.asarray(price, dtype=np.float64)
            self.volume = np.asarray(vol, dtype=np.float64)
            self.flags = np.full(len(ts), 32, dtype=np.float64)
    return _Src(ts, price, volume if volume is not None else np.ones(len(ts)))


def _make_kline_source(ts, o, h, l, c, v):
    class _Src:
        def __init__(self):
            self.ts = np.asarray(ts, dtype=np.int64)
            self.open = np.asarray(o, dtype=np.float64)
            self.high = np.asarray(h, dtype=np.float64)
            self.low = np.asarray(l, dtype=np.float64)
            self.close = np.asarray(c, dtype=np.float64)
            self.volume = np.asarray(v, dtype=np.float64)
        open_ref = object()
    return _Src()


class TestFibRun:
    def test_bullish_swing_levels(self):
        from tradetropy.ta.tool import FibRetracement
        # Low first (older), high later → bullish swing.
        ts = np.array([0, 1000, 2000, 3000, 4000], dtype=np.int64)
        # low=100 at t=1000, high=200 at t=4000
        price = np.array([150.0, 100.0, 130.0, 170.0, 200.0])
        res = FibRetracement().run(_make_tick_source(ts, price), start=None, end=None)
        assert res
        assert res.direction == "up"
        assert res.hi == pytest.approx(200.0)
        assert res.lo == pytest.approx(100.0)
        # Retracement measured down from the high: 0.0 -> hi, 1.0 -> lo.
        assert res.levels[0.0] == pytest.approx(200.0)
        assert res.levels[1.0] == pytest.approx(100.0)
        assert res.levels[0.5] == pytest.approx(150.0)
        assert res.levels[0.618] == pytest.approx(200.0 - 0.618 * 100.0)

    def test_bearish_swing_levels(self):
        from tradetropy.ta.tool import FibRetracement
        # High first (older), low later → bearish swing.
        ts = np.array([0, 1000, 2000, 3000, 4000], dtype=np.int64)
        price = np.array([150.0, 200.0, 170.0, 130.0, 100.0])
        res = FibRetracement().run(_make_tick_source(ts, price), start=None, end=None)
        assert res
        assert res.direction == "down"
        # Measured up from the low: 0.0 -> lo, 1.0 -> hi.
        assert res.levels[0.0] == pytest.approx(100.0)
        assert res.levels[1.0] == pytest.approx(200.0)
        assert res.levels[0.5] == pytest.approx(150.0)

    def test_range_slice_respected(self):
        from tradetropy.ta.tool import FibRetracement
        ts = np.arange(10, dtype=np.int64) * 1000
        price = np.array([10, 20, 100, 5, 50, 60, 70, 80, 90, 95], dtype=np.float64)
        # Restrict to ts in [3000, 6000] → prices [5, 50, 60, 70].
        res = FibRetracement().run(_make_tick_source(ts, price), start=3000, end=6000)
        assert res
        assert res.hi == pytest.approx(70.0)
        assert res.lo == pytest.approx(5.0)

    def test_empty_source_is_falsy(self):
        from tradetropy.ta.tool import FibRetracement
        res = FibRetracement().run(
            _make_tick_source(np.array([], dtype=np.int64), np.array([])),
            start=None, end=None,
        )
        assert not res

    def test_flat_range_is_falsy(self):
        from tradetropy.ta.tool import FibRetracement
        ts = np.array([0, 1000, 2000], dtype=np.int64)
        price = np.array([100.0, 100.0, 100.0])
        res = FibRetracement().run(_make_tick_source(ts, price), start=None, end=None)
        assert not res

    def test_kline_source(self):
        from tradetropy.ta.tool import FibRetracement
        ts = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        o = np.array([100, 105, 110, 115], dtype=np.float64)
        h = np.array([105, 112, 130, 120], dtype=np.float64)
        l = np.array([95, 100, 108, 90], dtype=np.float64)
        c = np.array([102, 110, 115, 95], dtype=np.float64)
        v = np.ones(4)
        res = FibRetracement().run(_make_kline_source(ts, o, h, l, c, v),
                                   start=None, end=None)
        assert res
        assert res.hi == pytest.approx(130.0)
        assert res.lo == pytest.approx(90.0)


class TestFibDraw:
    def test_draw_emits_hlines_per_level(self):
        from tradetropy.ta.tool import FibRetracement, HLines, Labels
        ts = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        price = np.array([100.0, 110.0, 150.0, 200.0])
        tool = FibRetracement()
        res = tool.run(_make_tick_source(ts, price), start=None, end=None)
        prims = tool.draw(res, tool.plot_config)
        hlines = [p for p in prims if isinstance(p, HLines)]
        labels = [p for p in prims if isinstance(p, Labels)]
        assert len(hlines) == 1
        assert len(list(hlines[0].y)) == 7  # default 7 levels
        assert len(labels) == 1             # labels on by default

    def test_draw_labels_off(self):
        from tradetropy.ta.tool import FibRetracement, Labels
        ts = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        price = np.array([100.0, 110.0, 150.0, 200.0])
        tool = FibRetracement(show_labels=False)
        res = tool.run(_make_tick_source(ts, price), start=None, end=None)
        prims = tool.draw(res, tool.plot_config)
        assert not any(isinstance(p, Labels) for p in prims)

    def test_draw_empty_returns_nothing(self):
        from tradetropy.ta.tool import FibRetracement
        from tradetropy.ta.tool.fibonacci import FibResult
        tool = FibRetracement()
        assert tool.draw(FibResult(None), tool.plot_config) == []


class TestFibInBacktest:
    def test_use_tool_stores_snapshot_with_own_legend(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.session.base import SeshSimulatorBase
        from tradetropy.core.data_types import TickData
        from tradetropy.ta.tool import FibRetracement, FixedRangeVP

        rng = np.random.default_rng(5)
        n = 400
        ts = np.arange(n) * 1000
        price = 100 + np.cumsum(rng.normal(0, 0.2, n))
        ticks = np.zeros((n, N_TICK_COLS))
        ticks[:, _TICK_COL["ts"]] = ts
        ticks[:, _TICK_COL["bid"]] = price - 0.05
        ticks[:, _TICK_COL["ask"]] = price + 0.05
        ticks[:, _TICK_COL["volume"]] = rng.uniform(1, 5, n)
        ticks[:, _TICK_COL["flags"]] = 32.0
        ticks[:, _TICK_COL["price"]] = price

        class S(Strategy):
            warmup = 0

            def init(self):
                self.tk = self.subscribe_ticks("BTCUSDT", window_size=1000)
                self.done = False

            def on_data(self):
                if not self.done and len(self.tk.ts) >= 250:
                    self.use_tool(self.tk, FibRetracement(),
                                  start=self.ts - 120_000, end=self.ts)
                    self.use_tool(self.tk, FixedRangeVP(nodes="both", tick_size=0.5),
                                  start=self.ts - 120_000, end=self.ts)
                    self.done = True

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        assert len(bt.strategy._tool_snapshots) == 2

        # Two independent legend groups: "Fib" and "VP".
        from tradetropy.ta.tool import collect_draw_primitives
        groups = collect_draw_primitives(bt.strategy._tool_snapshots)
        assert set(groups.keys()) == {"Fib", "VP"}

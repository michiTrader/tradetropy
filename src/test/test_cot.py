"""Tests for the per-bar COT (GoCharting-style) order-flow indicator."""

import numpy as np
import pytest

from tradetropy.ta.draw import Labels
from tradetropy.ta.order_flow import COT, bar_delta


def _tick_matrix(n=240, seed=11):
    """[N x 6] columns [ts, price, volume, flags, bid, ask], 1s spacing."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * 1000
    price = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    volume = rng.uniform(1.0, 5.0, n)
    flags = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, price, volume, flags, bid, ask])


class _Col:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, item):
        return self._arr[item]


class _FakeTickProxy:
    def __init__(self, matrix):
        self._m = matrix

    def __len__(self):
        return len(self._m)

    @property
    def ts(self):
        return _Col(self._m[:, 0])

    @property
    def price(self):
        return _Col(self._m[:, 1])

    @property
    def volume(self):
        return _Col(self._m[:, 2])

    @property
    def flags(self):
        return _Col(self._m[:, 3])

    @property
    def bid(self):
        return _Col(self._m[:, 4])

    @property
    def ask(self):
        return _Col(self._m[:, 5])


# Worked example (single bar):
#   tick  price  side  vol  signed  cum
#    1    100.0  buy   3    +3      +3
#    2    100.2  sell  1    -1      +2
#    3    100.1  sell  4    -4      -2
#    4     99.8  sell  3    -3      -5   <- bar low  99.8 -> COTL = -5
#    5    100.5  buy   9    +9      +4   <- bar high 100.5 -> COTH = +4
#   delta = (3 + 9) - (1 + 4 + 3) = 4
def _worked_example():
    ts = np.array([1_000, 2_000, 3_000, 4_000, 5_000], dtype=np.int64)
    price = np.array([100.0, 100.2, 100.1, 99.8, 100.5])
    volume = np.array([3.0, 1.0, 4.0, 3.0, 9.0])
    flags = np.array([1.0, -1.0, -1.0, -1.0, 1.0])
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, price, volume, flags, bid, ask])


class TestCore:
    def test_worked_example_single_bar(self):
        m = _worked_example()
        ts, price, volume, flags, bid, ask = (m[:, i] for i in range(6))
        r = bar_delta(ts, price, volume, interval_ms=1_000_000,
                      flags=flags, bid=bid, ask=ask)
        np.testing.assert_allclose(r["cot_high"], [4.0])
        np.testing.assert_allclose(r["cot_low"], [-5.0])
        np.testing.assert_allclose(r["high_price"], [100.5])
        np.testing.assert_allclose(r["low_price"], [99.8])
        np.testing.assert_allclose(r["delta"], [4.0])

    def test_two_bars_period_reset(self):
        # Bar 0 [0,60s): buy 2 @100, sell 5 @99   -> cum [2, -3]
        # Bar 1 [60s+): sell 1 @101, buy 4 @102   -> cum [-1, 3]
        ts = np.array([1_000, 2_000, 61_000, 62_000], dtype=np.int64)
        price = np.array([100.0, 99.0, 101.0, 102.0])
        volume = np.array([2.0, 5.0, 1.0, 4.0])
        flags = np.array([1.0, -1.0, -1.0, 1.0])
        r = bar_delta(ts, price, volume, interval_ms=60_000,
                      flags=flags, bid=price - 0.05, ask=price + 0.05, anchor=0)
        np.testing.assert_allclose(r["cot_high"], [2.0, 3.0])
        np.testing.assert_allclose(r["cot_low"], [-3.0, -1.0])
        np.testing.assert_allclose(r["high_price"], [100.0, 102.0])
        np.testing.assert_allclose(r["low_price"], [99.0, 101.0])
        np.testing.assert_allclose(r["delta"], [-3.0, 3.0])

    def test_extreme_uses_last_touch(self):
        # Price hits the high (101) twice; the LAST touch carries the delta.
        #   price 101 100 101, signed +1 -2 +5, cum 1 -1 4
        #   high 101 last at idx 2 -> COTH = 4 (not 1)
        ts = np.array([1_000, 2_000, 3_000], dtype=np.int64)
        price = np.array([101.0, 100.0, 101.0])
        volume = np.array([1.0, 2.0, 5.0])
        flags = np.array([1.0, -1.0, 1.0])
        r = bar_delta(ts, price, volume, interval_ms=1_000_000,
                      flags=flags, bid=price - 0.05, ask=price + 0.05)
        np.testing.assert_allclose(r["cot_high"], [4.0])
        np.testing.assert_allclose(r["cot_low"], [-1.0])

    def test_empty(self):
        r = bar_delta(np.array([], dtype=np.int64), np.array([]), np.array([]),
                      interval_ms=60_000)
        assert r["cot_high"].size == 0
        assert r["low_price"].size == 0


class TestIndicator:
    def test_output_names(self):
        ind = COT()
        assert ind.output_names == ["cot_high", "cot_low", "delta"]
        assert ind.n_outputs == 3

    def test_overlay_config(self):
        ind = COT()
        assert ind.plot_config.overlay is True
        assert ind.plot_config.renderer == "none"
        assert ind.plot_config.exclude_from_autoscale is True

    def test_period_parsing(self):
        assert COT(period="1m").interval_ms == 60_000
        with pytest.raises(ValueError):
            COT(period=0)

    def test_calculate_shape_and_bands(self):
        m = _worked_example()
        ind = COT(period="1m")
        out = ind.calculate(m)
        assert out.shape == (3, len(m))
        # Single bar -> figures anchored at the bar's last tick only.
        assert out[0, -1] == 4.0 and out[1, -1] == -5.0 and out[2, -1] == 4.0
        # All earlier ticks are NaN (developing figure on the rep tick).
        assert np.isnan(out[0, :-1]).all()

    def test_empty_source(self):
        ind = COT()
        out = ind.calculate(np.empty((0, 6)))
        assert out.shape == (3, 0)
        assert ind.draw(ind.plot_config) == {}


class TestDraw:
    def test_groups_and_positions(self):
        m = _worked_example()
        ind = COT(period="1m")
        ind.calculate(m)
        g = ind.draw(ind.plot_config)
        assert set(g) == {"COT"}

        labels = g["COT"]
        assert len(labels) == 3
        coth, cotl, dlt = labels
        assert isinstance(coth, Labels) and isinstance(cotl, Labels)
        # COTH anchored at the bar high, COTL at the bar low.
        np.testing.assert_allclose(list(coth.y), [100.5])
        np.testing.assert_allclose(list(cotl.y), [99.8])
        assert coth.text == ["COTH:+4"]
        assert cotl.text == ["COTL:-5"]
        assert coth.color == ind.up_color
        assert cotl.color == ind.down_color
        # COTH sits above the high, COTL below the low.
        assert coth.y_offset > 0 and coth.text_baseline == "bottom"
        assert cotl.y_offset < 0 and cotl.text_baseline == "top"

    def test_delta_sign_color(self):
        # Two bars with opposite delta -> per-bar green/red coloring.
        ts = np.array([1_000, 2_000, 61_000, 62_000], dtype=np.int64)
        price = np.array([100.0, 99.0, 101.0, 102.0])
        volume = np.array([2.0, 5.0, 1.0, 4.0])
        flags = np.array([1.0, -1.0, -1.0, 1.0])
        m = np.column_stack([ts, price, volume, flags, price - 0.05, price + 0.05])
        ind = COT(period="1m", delta_sign_color=True)
        ind.calculate(m)
        dlt = ind.draw(ind.plot_config)["COT"][2]
        assert dlt.color == [ind.down_color, ind.up_color]   # delta -3, +3

    def test_delta_fixed_color(self):
        m = _worked_example()
        ind = COT(period="1m", delta_sign_color=False, delta_color="#123456")
        ind.calculate(m)
        dlt = ind.draw(ind.plot_config)["COT"][2]
        assert dlt.color == "#123456"


class TestLiveParity:
    def test_live_refresh_matches_calculate(self):
        m = _tick_matrix()
        ind = COT(period="1m")
        ind.calculate(m)
        bars_calc = dict(ind._bars)

        ind2 = COT(period="1m")
        ind2.live_refresh(_FakeTickProxy(m))
        bars_live = ind2._bars
        for key in ("cot_high", "cot_low", "delta", "high_price", "low_price"):
            np.testing.assert_array_equal(bars_calc[key], bars_live[key])

    def test_empty_proxy_noop(self):
        ind = COT(period="1m")
        ind.live_refresh(_FakeTickProxy(_tick_matrix(n=0)))
        assert ind.draw(ind.plot_config) == {}


class TestEngineIntegration:
    def _raw(self, n=600, seed=3):
        # TickData raw layout: [ts, bid, ask, last(volume), flags, volume, price]
        rng = np.random.default_rng(seed)
        ts = np.arange(n) * 1000
        price = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
        volume = rng.uniform(1.0, 5.0, n)
        flags = np.where(rng.random(n) > 0.5, 1.0, -1.0)
        bid = price - 0.05
        ask = price + 0.05
        return np.column_stack([ts, bid, ask, volume, flags, volume.copy(), price])

    def test_on_data_reads_figures(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        seen = {"finite": 0}

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=700)
                self.cot = self.add_indicator(COT.refs(self.tk), COT(period="1m"))

            def on_data(self):
                if np.isfinite(self.cot.cot_high[-1]):
                    seen["finite"] += 1

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", self._raw(), tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        assert seen["finite"] > 0

    def test_plot_smoke(self, tmp_path):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=700)
                self.cot = self.add_indicator(COT.refs(self.tk), COT(period="1m"))

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", self._raw(), tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        out = tmp_path / "cot.html"
        bt.plot(output="file", filename=str(out), plot_stats=False)
        assert out.exists() and out.stat().st_size > 0

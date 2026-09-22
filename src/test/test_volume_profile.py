"""
Tests for the Volume Profile indicators and their shared core.

Covers:
  - Core binning + developing scan (tradetropy.ta._volume_profile)
  - VolumeProfile indicator (kline-based, uniform high-low distribution)
  - TickVolumeProfile indicator (tick-based, real binning + aggressor delta)

Usage:
    pytest src/test/test_volume_profile.py -v
"""

import numpy as np
import pytest

from tradetropy.ta._volume_profile import (
    level_price,
    infer_tick_size,
    scan_developing_profiles,
)
from tradetropy.ta.volume import VolumeProfile, RollingVolumeProfile
from tradetropy.ta.tick_volume_profile import TickVolumeProfile
from tradetropy.data.data import OhlcProxy


# =====
# CORE: binning helpers
# =====
class TestBinning:
    def test_level_price_snaps_to_tick(self):
        prices = np.array([100.4, 100.6, 101.0, 99.9])
        snapped = level_price(prices, 1.0)
        assert np.allclose(snapped, [100.0, 101.0, 101.0, 100.0])

    def test_infer_tick_size_positive(self):
        prices = np.linspace(100.0, 200.0, 500)
        ts = infer_tick_size(prices, target_bins=100)
        assert ts > 0
        # range 100 / 100 bins = 1.0 -> nice value 1.0
        assert ts == pytest.approx(1.0)

    def test_infer_tick_size_empty(self):
        assert infer_tick_size(np.array([])) == 1.0

    def test_infer_tick_size_flat(self):
        assert infer_tick_size(np.array([50.0, 50.0, 50.0])) == 1.0


# =====
# CORE: developing scan
# =====
class TestDevelopingScan:
    def test_empty(self):
        poc, vah, val, periods = scan_developing_profiles(
            0, np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]),
        )
        assert len(poc) == 0
        assert periods == []

    def test_single_period_poc_at_max_volume(self):
        # Three rows, same period. Level 100 gets the most volume -> POC=100.
        period_id = np.array([0, 0, 0])
        contrib_row = np.array([0, 1, 2])
        c_level = np.array([100.0, 101.0, 100.0])
        c_vol_bid = np.array([0.0, 0.0, 0.0])
        c_vol_ask = np.array([5.0, 1.0, 5.0])
        c_count = np.array([1.0, 1.0, 1.0])

        poc, vah, val, periods = scan_developing_profiles(
            3, period_id, contrib_row, c_level,
            c_vol_bid, c_vol_ask, c_count,
        )
        # final POC over the period is 100 (10 vol vs 1 vol)
        assert poc[-1] == pytest.approx(100.0)
        assert len(periods) == 1
        assert periods[0]["scalars"] is not None

    def test_developing_no_lookahead(self):
        # Row 0 only sees level 101; POC at row 0 must be 101 even though
        # level 100 dominates later.
        period_id = np.array([0, 0, 0])
        contrib_row = np.array([0, 1, 2])
        c_level = np.array([101.0, 100.0, 100.0])
        c_vol_bid = np.zeros(3)
        c_vol_ask = np.array([1.0, 5.0, 5.0])
        c_count = np.ones(3)

        poc, vah, val, periods = scan_developing_profiles(
            3, period_id, contrib_row, c_level,
            c_vol_bid, c_vol_ask, c_count,
        )
        assert poc[0] == pytest.approx(101.0)   # only data so far
        assert poc[-1] == pytest.approx(100.0)  # final POC

    def test_period_reset(self):
        # Two periods. Each finalized separately.
        period_id = np.array([0, 0, 1, 1])
        contrib_row = np.array([0, 1, 2, 3])
        c_level = np.array([100.0, 100.0, 200.0, 200.0])
        c_vol_bid = np.zeros(4)
        c_vol_ask = np.array([3.0, 3.0, 7.0, 7.0])
        c_count = np.ones(4)

        poc, vah, val, periods = scan_developing_profiles(
            4, period_id, contrib_row, c_level,
            c_vol_bid, c_vol_ask, c_count,
        )
        assert len(periods) == 2
        assert poc[1] == pytest.approx(100.0)
        assert poc[3] == pytest.approx(200.0)
        # row 2 (first of period 1) must not carry period 0's POC
        assert poc[2] == pytest.approx(200.0)


DAY_MS = 86_400_000


# =====
# VolumeProfile (klines)
# =====
class TestVolumeProfile:
    def _klines(self):
        # 4 candles in one day. ts in ms. cols: ts, open, high, low, close, vol.
        base = 0
        return np.array([
            [base + 0 * 60_000, 100.0, 101.0, 99.0, 100.5, 10.0],
            [base + 1 * 60_000, 100.5, 102.0, 100.0, 101.5, 20.0],
            [base + 2 * 60_000, 101.5, 101.8, 100.2, 100.3, 5.0],
            [base + 3 * 60_000, 100.3, 100.6, 99.5, 100.0, 8.0],
        ], dtype=np.float64)

    def test_shape_3xN(self):
        vp = VolumeProfile(period="1d", tick_size=0.5)
        out = vp.calculate(self._klines())
        assert out.shape == (3, 4)

    def test_output_names(self):
        vp = VolumeProfile()
        assert vp.output_names == ["poc", "vah", "val"]
        assert vp.n_outputs == 3

    def test_poc_within_range(self):
        vp = VolumeProfile(period="1d", tick_size=0.5)
        out = vp.calculate(self._klines())
        poc_final = out[0, -1]
        # POC must fall inside the overall traded range.
        assert 99.0 <= poc_final <= 102.0

    def test_period_reset_across_days(self):
        k = self._klines()
        # push last two candles to next day
        k2 = k.copy()
        k2[2, 0] += DAY_MS
        k2[3, 0] += DAY_MS
        vp = VolumeProfile(period="1d", tick_size=0.5)
        out = vp.calculate(k2)
        # Two finalized periods expected.
        assert len(vp.profiles_) == 2

    def test_empty(self):
        vp = VolumeProfile()
        out = vp.calculate(np.empty((0, 6), dtype=np.float64))
        assert out.shape == (3, 0)

    def test_refs_order(self):
        class _FakeProxy:
            ts_ref = "ts"; open_ref = "open"; high_ref = "high"
            low_ref = "low"; close_ref = "close"; volume_ref = "volume"
        refs = VolumeProfile.refs(_FakeProxy())
        assert refs == ["ts", "open", "high", "low", "close", "volume"]


# =====
# TickVolumeProfile (ticks)
# =====
class TestTickVolumeProfile:
    def _ticks(self):
        # cols: ts, price, volume, flags. flag 32=buy, 64=sell.
        return np.array([
            [0, 100.0, 3.0, 32],
            [1_000, 100.0, 5.0, 32],
            [2_000, 101.0, 2.0, 64],
            [3_000, 100.0, 4.0, 64],
        ], dtype=np.float64)

    def test_shape_3xN(self):
        tvp = TickVolumeProfile(period="1d", tick_size=1.0)
        out = tvp.calculate(self._ticks())
        assert out.shape == (3, 4)

    def test_poc_at_heaviest_level(self):
        tvp = TickVolumeProfile(period="1d", tick_size=1.0)
        out = tvp.calculate(self._ticks())
        # level 100 has 3+5+4=12 vol, level 101 has 2 -> POC=100
        assert out[0, -1] == pytest.approx(100.0)

    def test_aggressor_delta_sign(self):
        # buys at 100 (8 vol), sell at 100 (4) -> delta at 100 = +4 (ask-bid)
        tvp = TickVolumeProfile(period="1d", tick_size=1.0)
        tvp.calculate(self._ticks())
        period = tvp.profiles_[-1]
        levels = period["levels"]
        # find level 100 row
        from tradetropy.models.footprint import _FP_LEVEL_COL
        idx = int(np.argmin(np.abs(levels[:, _FP_LEVEL_COL["price"]] - 100.0)))
        delta = levels[idx, _FP_LEVEL_COL["delta"]]
        assert delta == pytest.approx(4.0)

    def test_empty(self):
        tvp = TickVolumeProfile()
        out = tvp.calculate(np.empty((0, 4), dtype=np.float64))
        assert out.shape == (3, 0)

    def test_tick_rule_fallback(self):
        # flags=0 -> tick rule. Rising price = buy.
        ticks = np.array([
            [0, 100.0, 1.0, 0],
            [1_000, 100.5, 1.0, 0],   # up -> buy
            [2_000, 100.0, 1.0, 0],   # down -> sell
        ], dtype=np.float64)
        tvp = TickVolumeProfile(period="1d", tick_size=0.5)
        out = tvp.calculate(ticks)
        assert out.shape == (3, 3)
        assert len(tvp.profiles_) == 1


def _make_ticks(n=120, base_price=100.0, start_ts=0, step_ms=1000):
    """Build a synthetic [N x 7] tick matrix: ts,bid,ask,volume,flags,vreal,price."""
    rng = np.random.default_rng(42)
    ts = start_ts + np.arange(n) * step_ms
    walk = np.cumsum(rng.normal(0, 0.2, n))
    price = base_price + walk
    bid = price - 0.05
    ask = price + 0.05
    volume = rng.uniform(1.0, 5.0, n)
    # alternate buy/sell flags
    flags = np.where(rng.random(n) > 0.5, 32, 64).astype(np.float64)
    vreal = volume.copy()
    return np.column_stack([ts, bid, ask, volume, flags, vreal, price])


# =====
# INTEGRATION: through BacktestEngine
# =====
class TestEngineIntegration:
    def test_tick_volume_profile_in_backtest(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ticks = _make_ticks(n=120)
        captured = []

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=10)
                self.tvp = self.add_indicator(
                    TickVolumeProfile.refs(self.tk),
                    TickVolumeProfile(period="1d", tick_size=0.5),
                )

            def on_data(self):
                captured.append(self.tvp.poc[-1])

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        assert len(captured) == 120
        # last POC must be a finite price inside the traded range
        last = captured[-1]
        assert np.isfinite(last)
        assert ticks[:, 6].min() - 1 <= last <= ticks[:, 6].max() + 1

    def test_volume_profile_klines_in_backtest(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ticks = _make_ticks(n=300)
        captured = []

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.vp = self.add_indicator(
                    VolumeProfile.refs(self.btc),
                    VolumeProfile(period="1d", tick_size=0.5),
                )

            def on_data(self):
                captured.append(self.vp.poc[-1])

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        assert len(captured) > 0  # warmup trims early ticks
        finite = [v for v in captured if np.isfinite(v)]
        assert len(finite) > 0  # POC becomes available once a candle closes


# =====
# Task 4: per-period histogram artifact
# =====
class TestPeriodProfiles:
    def test_period_profiles_structure(self):
        ticks = np.array([
            [0, 100.0, 3.0, 32],
            [1_000, 100.0, 5.0, 32],
            [2_000, 101.0, 2.0, 64],
            [3_000, 100.0, 4.0, 64],
        ], dtype=np.float64)
        tvp = TickVolumeProfile(period="1d", tick_size=1.0)
        tvp.calculate(ticks)
        profs = tvp.period_profiles()
        assert len(profs) == 1
        p = profs[0]
        assert p["ts_start"] == 0
        assert p["ts_end"] == 86_400_000
        assert p["poc"] == pytest.approx(100.0)
        assert p["max_vol"] == pytest.approx(12.0)  # level 100: 3+5+4
        assert len(p["prices"]) == len(p["volumes"]) == len(p["deltas"])
        # total volume conserved
        assert p["volumes"].sum() == pytest.approx(14.0)

    def test_period_profiles_two_days(self):
        ticks = np.array([
            [0, 100.0, 3.0, 32],
            [1_000, 100.0, 5.0, 32],
            [86_400_000, 200.0, 2.0, 32],
            [86_401_000, 200.0, 4.0, 64],
        ], dtype=np.float64)
        tvp = TickVolumeProfile(period="1d", tick_size=1.0)
        tvp.calculate(ticks)
        profs = tvp.period_profiles()
        assert len(profs) == 2
        assert profs[0]["poc"] == pytest.approx(100.0)
        assert profs[1]["poc"] == pytest.approx(200.0)
        assert profs[1]["ts_start"] == 86_400_000


# =====
# Task 5: histogram source + plotting data flow
# =====
class TestVolumeProfilePlotting:
    def _vp_data(self, view="session"):
        return {
            "tick_size": 1.0,
            "view": view,
            "profiles": [{
                "ts_start": 0, "ts_end": 86_400_000,
                "poc": 100.0, "vah": 101.0, "val": 99.0, "max_vol": 12.0,
                "prices": np.array([99.0, 100.0, 101.0]),
                "volumes": np.array([3.0, 12.0, 2.0]),
                "vol_bid": np.array([1.0, 5.0, 2.0]),
                "vol_ask": np.array([2.0, 7.0, 0.0]),
                "deltas": np.array([1.0, 2.0, -2.0]),
            }],
        }

    def test_source_builder(self):
        from tradetropy.plotting.sources import build_volume_profile_source
        src = build_volume_profile_source(self._vp_data())
        assert src is not None
        # Two-tone: buy + sell segments per level (level 101 has no buy volume).
        sides = list(src.data["side"])
        assert "buy" in sides and "sell" in sides
        # POC level (price 100, max volume) spans the widest total bar.
        y = src.data["y"]
        rights = src.data["right"].astype("int64")
        lefts = src.data["left"].astype("int64")
        per_level = {}
        for price, l, r in zip(y, lefts, rights):
            lo, hi = per_level.get(price, (l, r))
            per_level[price] = (min(lo, l), max(hi, r))
        spans = {p: hi - lo for p, (lo, hi) in per_level.items()}
        assert spans[100.0] == max(spans.values())
        # POC bars are inside the value area with standard VA alpha.
        poc_alphas = [a for yy, a in zip(y, src.data["alpha"]) if yy == 100.0]
        assert all(a == pytest.approx(0.85) for a in poc_alphas)

    def test_source_builder_visible_single_anchor(self):
        from tradetropy.plotting.sources import build_volume_profile_source
        vp_data = self._vp_data(view="visible")
        # Add a second period; visible view merges both into one profile.
        vp_data["profiles"].append({
            "ts_start": 86_400_000, "ts_end": 172_800_000,
            "poc": 100.0, "vah": 101.0, "val": 99.0, "max_vol": 6.0,
            "prices": np.array([100.0, 101.0]),
            "volumes": np.array([6.0, 1.0]),
            "vol_bid": np.array([3.0, 1.0]),
            "vol_ask": np.array([3.0, 0.0]),
            "deltas": np.array([0.0, -1.0]),
        })
        src = build_volume_profile_source(vp_data)
        assert src is not None
        # Whole histogram sits to the right of the last candle (max ts_end),
        # bars grow left (VPVR) but even the longest (POC) keeps a gap so it
        # never overlaps the candles.
        lefts = src.data["left"].astype("int64")
        last_ts = 172_800_000
        bar = last_ts // 100
        assert lefts.min() >= last_ts + 3 * bar

    def test_source_builder_empty(self):
        from tradetropy.plotting.sources import build_volume_profile_source
        assert build_volume_profile_source({"profiles": [], "tick_size": 1.0}) is None

    def test_value_area_bars_emphasized(self):
        from tradetropy.plotting.sources import build_volume_profile_source
        # VAL=99, VAH=101 -> all three levels inside VA with high alpha.
        src = build_volume_profile_source(self._vp_data())
        alpha_by_price = {}
        for price, a in zip(src.data["y"], src.data["alpha"]):
            alpha_by_price[price] = max(alpha_by_price.get(price, 0.0), a)
        assert alpha_by_price[100.0] == pytest.approx(0.85)   # in value area
        assert alpha_by_price[99.0] == pytest.approx(0.85)    # in value area
        assert alpha_by_price[101.0] == pytest.approx(0.85)   # in value area

    def test_outside_value_area_bars_dimmed(self):
        from tradetropy.plotting.sources import build_volume_profile_source
        vp_data = self._vp_data()
        # Add a level well outside a tightened value area so it is dimmed.
        prof = vp_data["profiles"][0]
        prof["vah"] = 100.0
        prof["val"] = 100.0   # value area collapses to the POC level only
        src = build_volume_profile_source(vp_data)
        alpha_by_price = {}
        for price, a in zip(src.data["y"], src.data["alpha"]):
            alpha_by_price[price] = max(alpha_by_price.get(price, 0.0), a)
        assert alpha_by_price[100.0] == pytest.approx(0.85)   # in value area
        assert alpha_by_price[99.0] == pytest.approx(0.25)    # outside VA
        assert alpha_by_price[101.0] == pytest.approx(0.25)   # outside VA

    def test_renderer_adds_glyph(self):
        from tradetropy.plotting.sources import build_volume_profile_source
        from tradetropy.plotting.render import render_volume_profile
        from bokeh.plotting import figure
        src = build_volume_profile_source(self._vp_data())
        fig = figure(x_axis_type="datetime")
        glyph = render_volume_profile(fig, src)
        assert glyph is not None

    def test_plot_data_flow_end_to_end(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.plotting._util import _gather_plot_data

        ticks = _make_ticks(n=300)

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.vp = self.add_indicator(
                    VolumeProfile.refs(self.btc),
                    VolumeProfile(period="1d", tick_size=0.5),
                )

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        data = _gather_plot_data(bt)
        vp_metas = [m for m in data["indicators"] if getattr(m, "_vp_data", None)]
        assert len(vp_metas) == 1
        from tradetropy.plotting.sources import build_volume_profile_source
        src = build_volume_profile_source(vp_metas[0]._vp_data)
        assert src is not None
        assert len(src.data["y"]) > 0

    def test_draw_primitives_match_legacy_source(self):
        """The VP draw() HBars must match the legacy build_volume_profile_source."""
        import numpy as np
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.plotting._util import _gather_plot_data
        from tradetropy.plotting.sources import build_volume_profile_source
        from tradetropy.ta.draw import HBars

        ticks = _make_ticks(n=300)

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.vp = self.add_indicator(
                    VolumeProfile.refs(self.btc),
                    VolumeProfile(period="1d", tick_size=0.5),
                )

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        data = _gather_plot_data(bt)
        meta = next(m for m in data["indicators"] if getattr(m, "_draw_primitives", None))
        groups = meta._draw_primitives
        # One legend group, one HBars primitive.
        prims = next(iter(groups.values()))
        hbars = [p for p in prims if isinstance(p, HBars)]
        assert len(hbars) == 1

        # Legacy source, same vp_data -> identical bar geometry/colors.
        legacy = build_volume_profile_source(
            meta._vp_data, interval_ms=data["interval_ms"]
        )
        hb = hbars[0]
        np.testing.assert_allclose(np.asarray(hb.y), np.asarray(legacy.data["y"]))
        np.testing.assert_allclose(np.asarray(hb.height), np.asarray(legacy.data["height"]))
        assert list(hb.color) == list(legacy.data["color"])
        # left/right in the primitive are int ms; legacy is datetime64[ms].
        np.testing.assert_array_equal(
            np.asarray(hb.left, dtype="datetime64[ms]"), legacy.data["left"]
        )
        np.testing.assert_array_equal(
            np.asarray(hb.right, dtype="datetime64[ms]"), legacy.data["right"]
        )


# =====
# Rolling volume profile (sliding window)
# =====
class TestScanRollingProfiles:
    def _contribs(self, levels, vols):
        # one contribution per row, ask side, count 1.
        rows = np.arange(len(levels), dtype=np.int64)
        c_level = np.asarray(levels, dtype=np.float64)
        c_bid = np.zeros(len(levels), dtype=np.float64)
        c_ask = np.asarray(vols, dtype=np.float64)
        c_cnt = np.ones(len(levels), dtype=np.float64)
        return rows, c_level, c_bid, c_ask, c_cnt

    def test_no_reset_continuous(self):
        from tradetropy.ta._volume_profile import scan_rolling_profiles
        # 6 rows, all volume piled at level 100 -> POC stays 100 throughout.
        rows, lv, cb, ca, cc = self._contribs([100.0] * 6, [1.0] * 6)
        poc, vah, val, final = scan_rolling_profiles(6, 3, rows, lv, cb, ca, cc)
        assert np.all(poc == 100.0)
        # rolling fills immediately: every row has a finite POC
        assert np.all(np.isfinite(poc))

    def test_window_slides_out_old_levels(self):
        from tradetropy.ta._volume_profile import scan_rolling_profiles
        from tradetropy.models.footprint import _FP_LEVEL_COL
        # rows 0-2 at price 100 (heavy), rows 3-5 at price 200.
        # With window=3, by the last row only price-200 levels remain.
        levels = [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
        vols = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
        rows, lv, cb, ca, cc = self._contribs(levels, vols)
        poc, vah, val, final = scan_rolling_profiles(6, 3, rows, lv, cb, ca, cc)
        assert poc[0] == pytest.approx(100.0)
        assert poc[-1] == pytest.approx(200.0)
        prices = final["levels"][:, _FP_LEVEL_COL["price"]]
        assert set(np.round(prices)) == {200.0}

    def test_empty(self):
        from tradetropy.ta._volume_profile import scan_rolling_profiles
        poc, vah, val, final = scan_rolling_profiles(
            0, 3,
            np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0), np.empty(0),
        )
        assert poc.shape == (0,)
        assert final["levels"].size == 0


class TestRollingVolumeProfile:
    def _klines(self, n=10):
        # n candles drifting up. cols: ts, open, high, low, close, vol.
        ts = np.arange(n, dtype=np.float64) * 60_000
        base = 100.0 + np.arange(n)
        return np.column_stack([
            ts, base, base + 1.0, base - 1.0, base + 0.5, np.full(n, 10.0),
        ])

    def test_shape_3xN(self):
        rvp = RollingVolumeProfile(length=4, tick_size=0.5)
        out = rvp.calculate(self._klines(10))
        assert out.shape == (3, 10)
        assert rvp.n_outputs == 3
        assert rvp.output_names == ["poc", "vah", "val"]

    def test_continuous_no_nan(self):
        rvp = RollingVolumeProfile(length=4, tick_size=0.5)
        out = rvp.calculate(self._klines(10))
        assert np.all(np.isfinite(out[0]))

    def test_use_partial_flag(self):
        # The engine relies on this to feed calculate the whole window.
        assert RollingVolumeProfile(length=50).use_partial is False

    def test_period_profiles_single_window(self):
        rvp = RollingVolumeProfile(length=4, tick_size=0.5)
        rvp.calculate(self._klines(10))
        profs = rvp.period_profiles()
        assert len(profs) == 1
        p = profs[0]
        # histogram anchored at the real last-window timestamps (not epoch)
        assert p["ts_start"] > 0
        assert p["ts_end"] > p["ts_start"]
        assert len(p["prices"]) == len(p["volumes"])

    def test_empty(self):
        rvp = RollingVolumeProfile()
        out = rvp.calculate(np.empty((0, 6), dtype=np.float64))
        assert out.shape == (3, 0)
        assert rvp.period_profiles() == []

    def test_in_backtest_proxy_direct(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ticks = _make_ticks(n=1560)
        captured = []

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                # proxy-direct API: no .refs() needed
                self.rvp = self.add_indicator(
                    self.btc, RollingVolumeProfile(length=20, tick_size=0.5),
                )

            def on_data(self):
                captured.append(self.rvp.poc[-1])

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        assert len(captured) > 0
        assert len([p for p in captured if np.isfinite(p)]) > 0


# =====
# Proxy-direct add_indicator API
# =====
class TestProxyDirectApi:
    def test_proxy_direct_resolves_refs(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ticks = _make_ticks(n=200)

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.vp = self.add_indicator(
                    self.btc, VolumeProfile(period="1d", tick_size=0.5),
                )

            def on_data(self):
                pass

        bt = BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()
        assert bt is not None

    def test_proxy_direct_resolves_via_generic_default_refs(self):
        """
        Every indicator now inherits a generic default_refs() driven by
        source_cols, so passing the proxy directly is valid. A single-source
        indicator (SMA, source_cols=('close',)) resolves to the close column,
        exactly like passing proxy.close explicitly.
        """
        from tradetropy.ta import SMA
        from tradetropy.models.strategy import Strategy

        s = Strategy.__new__(Strategy)
        s._indicator_defs = []
        proxy = OhlcProxy("SYM", 60_000)
        direct = s.add_indicator(proxy, SMA(10))

        s2 = Strategy.__new__(Strategy)
        s2._indicator_defs = []
        explicit = s2.add_indicator(proxy.close, SMA(10))

        # Same resolved source column in both indicator definitions.
        assert s._indicator_defs[0]["sources"][0].col_name == "close"
        assert s2._indicator_defs[0]["sources"][0].col_name == "close"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))



# =====
# CORE: volume node detection (HVN / LVN)
# =====
class TestVolumeNodes:
    def test_empty_returns_no_nodes(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        assert detect_volume_nodes(np.array([]), np.array([])) == []

    def test_all_zero_volume_returns_no_nodes(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        prices = np.array([100.0, 101.0, 102.0])
        volumes = np.zeros(3)
        assert detect_volume_nodes(prices, volumes) == []

    def test_single_peak_is_hvn(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volumes = np.array([1.0, 4.0, 10.0, 4.0, 1.0])
        nodes = detect_volume_nodes(prices, volumes, kind="hvn", prominence=0.1)
        assert len(nodes) == 1
        assert nodes[0].kind == "hvn"
        assert nodes[0].price == pytest.approx(102.0)
        assert nodes[0].strength == pytest.approx(1.0)  # global peak, full height

    def test_multipeak_detects_two_hvn_and_lvn_between(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        prices = np.arange(100.0, 107.0)
        #                 100  101  102  103  104  105  106
        volumes = np.array([2.0, 9.0, 3.0, 1.0, 3.0, 8.0, 2.0])
        nodes = detect_volume_nodes(prices, volumes, kind="both", prominence=0.1)
        hvn = [n for n in nodes if n.kind == "hvn"]
        lvn = [n for n in nodes if n.kind == "lvn"]
        assert {n.price for n in hvn} == {101.0, 105.0}
        # The valley between the two peaks is at price 103 (volume 1)
        assert any(n.price == pytest.approx(103.0) for n in lvn)
        # Nodes overall are returned ordered by price
        assert [n.price for n in nodes] == sorted(n.price for n in nodes)

    def test_prominence_filters_weak_nodes(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        prices = np.arange(100.0, 107.0)
        volumes = np.array([2.0, 9.0, 8.0, 8.5, 3.0, 8.0, 2.0])
        # A high threshold keeps only the dominant peak
        strong = detect_volume_nodes(prices, volumes, kind="hvn", prominence=0.6)
        assert all(n.strength >= 0.6 for n in strong)
        assert any(n.price == pytest.approx(101.0) for n in strong)

    def test_max_nodes_keeps_strongest_but_orders_by_price(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        prices = np.arange(100.0, 109.0)
        volumes = np.array([1.0, 10.0, 1.0, 6.0, 1.0, 8.0, 1.0, 4.0, 1.0])
        nodes = detect_volume_nodes(
            prices, volumes, kind="hvn", prominence=0.05, max_nodes=2
        )
        assert len(nodes) == 2
        # Strongest two peaks are at 101 (10) and 105 (8)
        assert {n.price for n in nodes} == {101.0, 105.0}
        # Still ordered by price ascending
        assert nodes[0].price < nodes[1].price

    def test_invalid_kind_raises(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        with pytest.raises(ValueError):
            detect_volume_nodes(np.array([1.0]), np.array([1.0]), kind="bad")

    def test_unsorted_prices_are_handled(self):
        from tradetropy.ta._volume_profile import detect_volume_nodes
        prices = np.array([102.0, 100.0, 101.0, 104.0, 103.0])
        volumes = np.array([10.0, 1.0, 4.0, 1.0, 4.0])
        nodes = detect_volume_nodes(prices, volumes, kind="hvn", prominence=0.1)
        # Peak at price 102 (volume 10) must be detected regardless of input order
        assert any(n.price == pytest.approx(102.0) for n in nodes)


# =====
# CORE: range profile (FixedRangeVP backend)
# =====
def _make_ticks_for_range(n=400, base_price=100.0, start_ts=0, step_ms=1000):
    rng = np.random.default_rng(7)
    ts = start_ts + np.arange(n) * step_ms
    price = base_price + np.cumsum(rng.normal(0, 0.2, n))
    vol = rng.uniform(1.0, 5.0, n)
    flags = np.where(rng.random(n) > 0.5, 32, 64).astype(np.float64)
    return ts, price, vol, flags


class TestComputeRangeProfile:
    def test_kline_range_poc_vah_val(self):
        from tradetropy.ta._volume_profile import compute_range_profile
        ts = np.array([0, 60_000, 120_000, 180_000, 240_000], dtype=np.int64)
        o = np.array([100, 101, 102, 101, 100], float)
        h = np.array([101, 102, 103, 102, 101], float)
        l = np.array([99, 100, 101, 100, 99], float)
        c = np.array([101, 102, 101, 100, 100], float)
        v = np.array([10, 50, 20, 30, 40], float)
        prof = compute_range_profile(
            ts, is_tick=False, start=60_000, end=180_000,
            tick_size=1.0, bins=100, value_area_pct=0.7, nodes="both",
            open_=o, high=h, low=l, close=c, volume=v,
        )
        assert prof is not None
        assert prof["start"] == 60_000 and prof["end"] == 180_000
        assert prof["val"] <= prof["poc"] <= prof["vah"]
        assert prof["tick_size"] == 1.0

    def test_empty_range_returns_none(self):
        from tradetropy.ta._volume_profile import compute_range_profile
        ts, price, vol, flags = _make_ticks_for_range()
        prof = compute_range_profile(
            ts, is_tick=True, start=10**12, end=10**13,
            tick_size=0.5, bins=100, value_area_pct=0.7,
            price=price, volume=vol, flags=flags.astype(np.int64),
        )
        assert prof is None

    def test_tick_range_open_ended_bounds(self):
        from tradetropy.ta._volume_profile import compute_range_profile
        ts, price, vol, flags = _make_ticks_for_range(n=200)
        prof = compute_range_profile(
            ts, is_tick=True, start=None, end=None,
            tick_size=0.5, bins=100, value_area_pct=0.7,
            price=price, volume=vol, flags=flags.astype(np.int64),
        )
        assert prof is not None
        assert prof["start"] == int(ts[0])
        assert prof["end"] == int(ts[-1])


# =====
# Tool: FixedRangeVP + VolumeProfileResult
# =====
class TestVolumeProfileResult:
    def test_empty_is_falsy(self):
        from tradetropy.ta.tool import VolumeProfileResult
        r = VolumeProfileResult(None)
        assert not r
        assert r.hvn == [] and r.lvn == [] and r.nodes == []

    def test_nodes_merged_and_sorted_by_price(self):
        from tradetropy.ta._volume_profile import VolumeNode
        from tradetropy.ta.tool import VolumeProfileResult
        prof = {
            "start": 0, "end": 1, "poc": 100.0, "vah": 101.0, "val": 99.0,
            "tick_size": 1.0,
            "hvn": [VolumeNode(102.0, 9, "hvn", 0.9)],
            "lvn": [VolumeNode(100.5, 1, "lvn", 0.4)],
            "prices": np.array([99.0, 100.0]), "volumes": np.array([1.0, 9.0]),
            "vol_bid": np.array([0.5, 4.0]), "vol_ask": np.array([0.5, 5.0]),
            "deltas": np.array([0.0, 1.0]),
        }
        r = VolumeProfileResult(prof)
        assert r
        assert [n.price for n in r.nodes] == [100.5, 102.0]


class TestFixedRangeVPTool:
    def test_invalid_nodes_raises(self):
        from tradetropy.ta.tool import FixedRangeVP
        from tradetropy.exceptions import ConfigError
        with pytest.raises(ConfigError):
            FixedRangeVP(nodes="bad")

    def test_use_tool_in_backtest_returns_result(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.ta.tool import FixedRangeVP

        ts, price, vol, flags = _make_ticks_for_range(n=400)
        bid = price - 0.05
        ask = price + 0.05
        ticks = np.column_stack([ts, bid, ask, vol, flags, vol.copy(), price])
        captured = {}

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=1000)
                self.done = False

            def on_data(self):
                if not self.done and len(self.tk.ts) >= 200:
                    captured["r"] = self.use_tool(
                        self.tk, FixedRangeVP(nodes="both", tick_size=0.5),
                        start=self.ts - 100_000, end=self.ts,
                    )
                    self.done = True

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        r = captured["r"]
        assert r
        assert np.isfinite(r.poc)
        assert r.val <= r.poc <= r.vah
        assert len(bt.strategy._tool_snapshots) == 1

    def test_use_tool_plot_false_does_not_store(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.ta.tool import FixedRangeVP

        ts, price, vol, flags = _make_ticks_for_range(n=300)
        bid = price - 0.05
        ask = price + 0.05
        ticks = np.column_stack([ts, bid, ask, vol, flags, vol.copy(), price])

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=1000)
                self.done = False

            def on_data(self):
                if not self.done and len(self.tk.ts) >= 150:
                    self.use_tool(
                        self.tk, FixedRangeVP(tick_size=0.5),
                        end=self.ts, plot=False,
                    )
                    self.done = True

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        assert len(bt.strategy._tool_snapshots) == 0


# =====
# Nodes on continuous VP indicators via the proxy
# =====
class TestProxyNodeAccess:
    def test_nodes_none_raises_on_access(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.exceptions import ConfigError

        ts, price, vol, flags = _make_ticks_for_range(n=300)
        bid = price - 0.05
        ask = price + 0.05
        ticks = np.column_stack([ts, bid, ask, vol, flags, vol.copy(), price])
        errors = []

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.vp = self.add_indicator(
                    VolumeProfile.refs(self.btc),
                    VolumeProfile(period="5m", tick_size=0.5),  # nodes=None
                )

            def on_data(self):
                if not errors:
                    try:
                        _ = self.vp.hvn
                    except ConfigError:
                        errors.append("raised")

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        assert errors == ["raised"]

    def test_nodes_enabled_returns_lists(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ts, price, vol, flags = _make_ticks_for_range(n=600)
        bid = price - 0.05
        ask = price + 0.05
        ticks = np.column_stack([ts, bid, ask, vol, flags, vol.copy(), price])
        seen = {"hvn": 0, "lvn": 0, "calls": 0}

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.vp = self.add_indicator(
                    VolumeProfile.refs(self.btc),
                    VolumeProfile(period="5m", tick_size=0.25, nodes="both"),
                )

            def on_data(self):
                if self.ts > 400_000:
                    seen["calls"] += 1
                    seen["hvn"] = max(seen["hvn"], len(self.vp.hvn))
                    seen["lvn"] = max(seen["lvn"], len(self.vp.lvn))

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        assert seen["calls"] > 0
        # At least some nodes detected over a multi-level developing profile.
        assert seen["hvn"] + seen["lvn"] >= 1

    def test_rolling_nodes_window_stays_in_band(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        ts, price, vol, flags = _make_ticks_for_range(n=600)
        bid = price - 0.05
        ask = price + 0.05
        ticks = np.column_stack([ts, bid, ask, vol, flags, vol.copy(), price])
        node_prices_in_range = []

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=5)
                self.btc = self.subscribe_ohlc("SYM", 60_000, window_size=50)
                self.rvp = self.add_indicator(
                    self.btc,
                    RollingVolumeProfile(length=5, tick_size=0.25, nodes="both"),
                )

            def on_data(self):
                for node in self.rvp.nodes:
                    node_prices_in_range.append(
                        price.min() - 5 <= node.price <= price.max() + 5
                    )

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("SYM", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        assert all(node_prices_in_range)


# =====
# README smoke test (use_tool + nodes end to end)
# =====
class TestReadmeExample:
    def test_readme_volume_profile_flow(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase
        from tradetropy.ta import VolumeProfile, RollingVolumeProfile
        from tradetropy.ta.tool import FixedRangeVP

        ts, price, vol, flags = _make_ticks_for_range(n=800)
        bid = price - 0.05
        ask = price + 0.05
        ticks = np.column_stack([ts, bid, ask, vol, flags, vol.copy(), price])

        class S(Strategy):
            def init(self):
                self.ticks = self.subscribe_ticks("BTCUSDT", window_size=1000)
                self.btc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=200)
                self.vp = self.add_indicator(
                    VolumeProfile.refs(self.btc),
                    VolumeProfile(period="5m", nodes="both", tick_size=0.5),
                )
                self.rvp = self.add_indicator(
                    self.btc,
                    RollingVolumeProfile(length=5, nodes="both", tick_size=0.5),
                )
                self.done = False

            def on_data(self):
                _ = self.vp.hvn
                _ = self.rvp.lvn
                _ = self.rvp.nodes
                if not self.done and len(self.ticks.ts) >= 300:
                    vp = self.use_tool(
                        self.ticks, FixedRangeVP(nodes="both", tick_size=0.5),
                        start=self.ts - 200_000, end=self.ts,
                    )
                    if vp and self.ticks.price[-1] > vp.vah:
                        self.sesh.buy("BTCUSDT", volume=1)
                    self.done = True

        bt = BacktestEngine.by_ticks(
            S(), data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        )
        bt.run()
        assert len(bt.strategy._tool_snapshots) == 1


# =====
# Draw primitives (VP tool) + live snapshot rendering
# =====
class TestVPDrawPrimitives:
    def _vp_result(self):
        from tradetropy.ta._volume_profile import compute_range_profile
        from tradetropy.ta.tool.volume_profile import VolumeProfileResult
        rng = np.random.default_rng(7)
        n = 400
        ts = np.arange(n, dtype=np.int64) * 1000
        price = 100 + np.cumsum(rng.normal(0, 0.1, n))
        prof = compute_range_profile(
            ts, is_tick=True, start=None, end=None,
            tick_size=0.25, bins=100, value_area_pct=0.7, nodes="both",
            price=price, volume=rng.uniform(1, 5, n),
            flags=np.full(n, 32, dtype=np.int64),
        )
        return VolumeProfileResult(prof)

    def test_draw_emits_bars_poc_and_nodes(self):
        from tradetropy.ta.tool import FixedRangeVP, HBars, HLines, Points
        tool = FixedRangeVP(nodes="both", tick_size=0.25)
        res = self._vp_result()
        prims = tool.draw(res, tool.plot_config)
        kinds = {type(p) for p in prims}
        assert HBars in kinds
        assert HLines in kinds   # POC line
        assert Points in kinds   # HVN/LVN dots

    def test_draw_empty_result_returns_nothing(self):
        from tradetropy.ta.tool import FixedRangeVP
        from tradetropy.ta.tool.volume_profile import VolumeProfileResult
        tool = FixedRangeVP()
        assert tool.draw(VolumeProfileResult(None), tool.plot_config) == []


class TestLiveToolSnapshot:
    def _make_strategy_engine(self, seed):
        from tradetropy.models.strategy import Strategy
        from tradetropy.live.engine import LiveEngine
        from tradetropy.session.base import SeshSimulatorBase
        from tradetropy.ta.tool import FixedRangeVP
        from tradetropy.core.constants import N_TICK_COLS, _TICK_COL

        rng = np.random.default_rng(seed)
        n = 500
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
                self.btc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=200)
                self.done = False

            def on_data(self):
                if not self.done and len(self.tk.ts) >= 250:
                    self.use_tool(
                        self.tk, FixedRangeVP(nodes="both", tick_size=0.25),
                        start=self.ts - 120_000, end=self.ts,
                    )
                    self.done = True

        eng = LiveEngine.by_ticks(S(), sesh=SeshSimulatorBase("tick"))
        return eng, ticks

    def test_use_tool_snapshot_drawn_in_live_document(self):
        from tradetropy.plotting.live.document import build_live_document
        from tradetropy.plotting.config import PlotConfig

        eng, ticks = self._make_strategy_engine(2)
        eng.prepare({"BTCUSDT": ticks[:50]})
        for i in range(50, 350):
            eng.on_tick("BTCUSDT", ticks[i])
        assert len(eng.strategy._tool_snapshots) == 1

        doc, updater = build_live_document(
            strategy=eng.strategy, config=PlotConfig(),
            max_candles=500, broker=getattr(eng, "_broker", None),
        )
        # Snapshot already present at build time → ref created eagerly.
        assert updater._tool_ref is not None
        updater._history_loaded = True
        updater._update_tool_snapshots(updater._tool_ref)
        # Registry has VP bars + POC + node points populated from the snapshot.
        reg = updater._tool_ref.registry
        bars = reg.get(("VP", "HBars"))
        assert bars is not None and len(bars.data["y"]) > 0
        assert ("VP", "Points") in reg

    def test_use_tool_snapshot_drawn_lazily_in_live_document(self):
        """Tool ref + legend are created lazily on the first snapshot."""
        from tradetropy.plotting.live.document import build_live_document
        from tradetropy.plotting.config import PlotConfig

        eng, ticks = self._make_strategy_engine(3)
        eng.prepare({"BTCUSDT": ticks[:50]})

        # Document built BEFORE any on_data() runs → no snapshot yet, no ref.
        doc, updater = build_live_document(
            strategy=eng.strategy, config=PlotConfig(),
            max_candles=500, broker=getattr(eng, "_broker", None),
        )
        assert updater._tool_ref is None
        updater._history_loaded = True

        # Feed ticks so the strategy calls use_tool(), then update the chart.
        for i in range(50, 350):
            eng.on_tick("BTCUSDT", ticks[i])
        assert len(eng.strategy._tool_snapshots) == 1

        updater.update()
        # Ref + glyphs created lazily and populated from the snapshot.
        assert updater._tool_ref is not None
        bars = updater._tool_ref.registry.get(("VP", "HBars"))
        assert bars is not None and len(bars.data["y"]) > 0

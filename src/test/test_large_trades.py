"""Tests for the LargeTrades order-flow indicator and its pure core."""

import numpy as np
import pytest

from tradetropy.ta.order_flow._core import (
    classify_aggressor,
    trade_metric,
    detect_large_trades,
    aggregate_trades,
    map_bubble_style,
    format_magnitude,
    build_bubble_columns,
    DEFAULT_BUY_COLOR,
    DEFAULT_SELL_COLOR,
)
from tradetropy.ta.order_flow import LargeTrades


def detect3(ts, price, volume, **kw):
    """Adapt detect_large_trades' dict result to the (mask, metric, side) tuple
    used by the pure-detection tests below."""
    r = detect_large_trades(ts, price, volume, **kw)
    return r["mask"], r["metric"], r["side"]


# =====
# Aggressor classification
# =====
class TestClassifyAggressor:
    def test_sign_encoding(self):
        price = np.array([10.0, 10.0, 10.0])
        flags = np.array([1.0, -1.0, 0.0])
        side = classify_aggressor(price, flags=flags)
        assert side[0] == 1
        assert side[1] == -1
        # flag 0 -> tick rule (flat price inherits buy default)
        assert side[2] in (1, -1)

    def test_bit_encoding(self):
        price = np.array([10.0, 10.0])
        flags = np.array([32.0, 64.0])
        side = classify_aggressor(price, flags=flags)
        assert side[0] == 1
        assert side[1] == -1

    def test_negative_flag_not_misread_as_bit(self):
        # -1 (sell) must not be read as a buy via the 32-bit test.
        price = np.array([10.0])
        side = classify_aggressor(price, flags=np.array([-1.0]))
        assert side[0] == -1

    def test_tick_rule_fallback(self):
        price = np.array([10.0, 11.0, 10.5, 10.5])
        side = classify_aggressor(price, flags=np.zeros(4))
        assert side[0] == 1     # first defaults buy
        assert side[1] == 1     # rising
        assert side[2] == -1    # falling
        assert side[3] == -1    # flat inherits previous down

    def test_quote_rule(self):
        price = np.array([10.0, 9.0])
        bid = np.array([9.5, 9.0])
        ask = np.array([10.0, 9.5])
        side = classify_aggressor(price, flags=np.zeros(2), bid=bid, ask=ask)
        assert side[0] == 1     # price >= ask
        assert side[1] == -1    # price <= bid

    def test_quote_rule_locked_book_falls_through_to_tick_rule(self):
        # A TradeEvent carries no quote, so its tick row has bid == ask == price.
        # With no aggressor flag the quote rule must NOT fire (there is no
        # spread), or every trade would be misclassified as a sell. Such rows
        # fall through to the tick rule, yielding a proper buy/sell mix.
        price = np.array([10.0, 11.0, 10.5, 10.5])
        side = classify_aggressor(
            price, flags=np.zeros(4), bid=price.copy(), ask=price.copy()
        )
        assert side[0] == 1     # first defaults buy
        assert side[1] == 1     # rising -> buy
        assert side[2] == -1    # falling -> sell
        assert side[3] == -1    # flat inherits previous down
        assert not np.all(side == -1)   # regression: not all sells (was all red)


# =====
# Metric
# =====
class TestTradeMetric:
    def test_volume(self):
        m = trade_metric(np.array([10.0, 20.0]), np.array([2.0, 3.0]), by="volume")
        assert np.allclose(m, [2.0, 3.0])

    def test_notional(self):
        m = trade_metric(np.array([10.0, 20.0]), np.array([2.0, 3.0]), by="notional")
        assert np.allclose(m, [20.0, 60.0])

    def test_invalid(self):
        with pytest.raises(ValueError):
            trade_metric(np.array([1.0]), np.array([1.0]), by="bogus")

    def test_delta_per_trade_equals_volume(self):
        # A single print is fully one-sided, so |delta| == raw volume.
        price = np.array([10.0, 20.0])
        volume = np.array([2.0, 3.0])
        side = np.array([1, -1])
        m = trade_metric(price, volume, by="delta", side=side)
        assert np.allclose(m, [2.0, 3.0])

    def test_delta_without_side_falls_back_to_volume(self):
        m = trade_metric(np.array([10.0]), np.array([4.0]), by="delta")
        assert np.allclose(m, [4.0])


# =====
# Aggregation
# =====
class TestAggregateTrades:
    def test_no_aggregation_is_per_trade(self):
        ts = np.array([0, 10, 20], dtype=float)
        price = np.array([10.0, 10.0, 10.0])
        volume = np.array([2.0, 3.0, 4.0])
        side = np.array([1, 1, -1], dtype=np.int8)
        agg = aggregate_trades(ts, price, volume, side, aggregate_ms=0, by="volume")
        assert list(agg["rep_idx"]) == [0, 1, 2]
        assert np.allclose(agg["volume"], [2.0, 3.0, 4.0])
        assert np.allclose(agg["metric"], [2.0, 3.0, 4.0])

    def test_same_side_merge_volume(self):
        # Three same-side buys within 50 ms merge into one event of summed size.
        ts = np.array([0, 20, 40, 1000], dtype=float)
        price = np.array([10.0, 10.0, 10.0, 10.0])
        volume = np.array([50.0, 60.0, 40.0, 5.0])
        side = np.array([1, 1, 1, 1], dtype=np.int8)
        agg = aggregate_trades(ts, price, volume, side, aggregate_ms=50, by="volume")
        assert list(agg["rep_idx"]) == [2, 3]
        assert np.allclose(agg["volume"], [150.0, 5.0])
        assert agg["side"][0] == 1

    def test_opposite_side_breaks_burst_for_volume(self):
        ts = np.array([0, 10, 20], dtype=float)
        price = np.array([10.0, 10.0, 10.0])
        volume = np.array([50.0, 60.0, 40.0])
        side = np.array([1, -1, 1], dtype=np.int8)
        agg = aggregate_trades(ts, price, volume, side, aggregate_ms=100, by="volume")
        # side flip splits into separate bursts despite being inside the window.
        assert list(agg["rep_idx"]) == [0, 1, 2]

    def test_delta_nets_both_sides(self):
        # Buys and sells in the same window cancel into a net delta.
        ts = np.array([0, 10, 20, 30], dtype=float)
        price = np.array([10.0, 10.0, 10.0, 10.0])
        volume = np.array([100.0, 90.0, 30.0, 0.0])
        side = np.array([1, -1, 1, 1], dtype=np.int8)
        agg = aggregate_trades(ts, price, volume, side, aggregate_ms=100, by="delta")
        # one burst: net = 100 - 90 + 30 = 40, side +1, metric |40|
        assert len(agg["rep_idx"]) == 1
        assert agg["delta"][0] == pytest.approx(40.0)
        assert agg["side"][0] == 1
        assert agg["metric"][0] == pytest.approx(40.0)


# =====
# Delta detection + aggregated detection
# =====
class TestDeltaAndAggregatedDetection:
    def _flat(self, n=200):
        ts = (np.arange(n) * 1000).astype(float)
        price = np.full(n, 100.0)
        volume = np.full(n, 2.0)
        side = np.ones(n, dtype=np.int8)
        return ts, price, volume, side

    def test_aggregated_detection_marks_anchor_tick(self):
        ts, price, volume, side = self._flat()
        # A burst of three same-side prints in the same 50 ms window.
        ts[100], ts[101], ts[102] = 100_000.0, 100_010.0, 100_020.0
        volume[100] = volume[101] = volume[102] = 40.0  # sum 120 >> threshold
        flags = side.astype(float)
        res = detect_large_trades(
            ts, price, volume, flags=flags,
            threshold=100.0, by="volume", aggregate_ms=50,
        )
        # Only the anchor (last trade of the burst) is flagged, carrying the sum.
        assert res["mask"][102]
        assert not res["mask"][100] and not res["mask"][101]
        assert res["metric"][102] == pytest.approx(120.0)
        ev = res["events"]
        assert ev["volume"][0] == pytest.approx(120.0)

    def test_delta_two_sided_burst_ranks_low(self):
        ts, price, volume, side = self._flat(n=200)
        flags = np.ones(200)
        # Large but balanced burst: huge gross volume, tiny net delta.
        ts[80], ts[81] = 80_000.0, 80_010.0
        volume[80], volume[81] = 500.0, 480.0
        flags[80], flags[81] = 1.0, -1.0   # buy then sell
        res = detect_large_trades(
            ts, price, volume, flags=flags,
            threshold=200.0, by="delta", aggregate_ms=50,
        )
        # Net delta is only 20 -> below the 200 threshold -> not flagged.
        assert not res["mask"][80] and not res["mask"][81]

    def test_aggregate_zero_matches_per_trade(self):
        ts, price, volume, side = self._flat()
        volume[120] = 100.0
        flags = side.astype(float)
        a = detect3(
            ts, price, volume, flags=flags, threshold=50.0, by="volume",
        )
        b = detect3(
            ts, price, volume, flags=flags, threshold=50.0, by="volume",
            aggregate_ms=0,
        )
        assert np.array_equal(a[0], b[0])
        assert a[0][120]


# =====
# Detection
# =====
class TestDetect:
    def _data(self, n=200, seed=7):
        rng = np.random.default_rng(seed)
        ts = np.arange(n) * 1000
        price = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
        volume = rng.uniform(1.0, 5.0, n)
        flags = np.zeros(n)
        return ts, price, volume, flags

    def test_absolute(self):
        ts, price, volume, flags = self._data()
        volume[50] = 100.0
        volume[120] = 80.0
        mask, metric, side = detect3(
            ts, price, volume, flags=flags, threshold=50.0, by="volume",
        )
        assert mask[50] and mask[120]
        assert mask.sum() == 2

    def test_quantile_detects_spike(self):
        ts, price, volume, flags = self._data()
        volume[80] = 200.0
        mask, _, _ = detect3(
            ts, price, volume, flags=flags,
            threshold="p99", by="volume", window=50,
        )
        assert mask[80]

    def test_quantile_is_causal(self):
        # Detection on a prefix must equal detection on the full array over the
        # overlapping range: no future trade may change a past decision.
        ts, price, volume, flags = self._data(n=300)
        volume[250] = 500.0   # a big future spike
        full, _, _ = detect3(
            ts, price, volume, flags=flags,
            threshold="p95", by="volume", window=40,
        )
        for k in (60, 120, 200):
            pre, _, _ = detect3(
                ts[: k + 1], price[: k + 1], volume[: k + 1], flags=flags[: k + 1],
                threshold="p95", by="volume", window=40,
            )
            assert np.array_equal(full[: k + 1], pre)

    def test_multiple(self):
        ts, price, volume, flags = self._data()
        volume[:] = 2.0
        volume[60] = 30.0
        mask, _, _ = detect3(
            ts, price, volume, flags=flags,
            threshold="5x", by="volume", window=30,
        )
        assert mask[60]

    def test_warmup_suppressed(self):
        # Before the window is full no relative detection happens.
        ts, price, volume, flags = self._data(n=20)
        mask, _, _ = detect3(
            ts, price, volume, flags=flags,
            threshold="p90", by="volume", window=50,
        )
        assert not mask.any()

    def test_min_gap(self):
        ts, price, volume, flags = self._data()
        volume[50] = 100.0
        volume[51] = 100.0   # adjacent spike, same gap window
        mask, _, _ = detect3(
            ts, price, volume, flags=flags,
            threshold=50.0, by="volume", min_gap_ms=5000,
        )
        assert mask[50]
        assert not mask[51]   # suppressed by anti-clustering


# =====
# Style mapping
# =====
class TestBubbleStyle:
    def test_monotonic_and_clamped(self):
        metric = np.array([1.0, 4.0, 9.0, 16.0])
        side = np.array([1, 1, -1, -1])
        sizes, colors = map_bubble_style(
            metric, side, scale="sqrt", min_size=6, max_size=40,
        )
        assert sizes[0] == pytest.approx(6.0)
        assert sizes[-1] == pytest.approx(40.0)
        assert np.all(np.diff(sizes) > 0)
        assert (sizes >= 6.0).all() and (sizes <= 40.0).all()

    def test_colors_by_side(self):
        sizes, colors = map_bubble_style(
            np.array([1.0, 2.0]), np.array([1, -1]),
            buy_color=DEFAULT_BUY_COLOR, sell_color=DEFAULT_SELL_COLOR,
        )
        assert colors[0] == DEFAULT_BUY_COLOR
        assert colors[1] == DEFAULT_SELL_COLOR

    def test_single_event_midsize(self):
        sizes, _ = map_bubble_style(np.array([5.0]), np.array([1]),
                                    min_size=6, max_size=40)
        assert sizes[0] == pytest.approx(23.0)

    def test_empty(self):
        sizes, colors = map_bubble_style(np.array([]), np.array([]))
        assert len(sizes) == 0 and len(colors) == 0


# =====
# Formatting
# =====
class TestFormat:
    @pytest.mark.parametrize("val,expected", [
        (1_800_000, "1.8M"),
        (12_500, "12.5k"),
        (7.0, "7"),
        (0.42, "0.42"),
        (2_500_000_000, "2.5B"),
    ])
    def test_format(self, val, expected):
        assert format_magnitude(val) == expected

    def test_nan(self):
        assert format_magnitude(np.nan) == ""


# =====
# Indicator
# =====
def _tick_matrix(n=200, seed=11):
    """[N x 6] columns [ts, price, volume, flags, bid, ask]."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * 1000
    price = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    volume = rng.uniform(1.0, 5.0, n)
    flags = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, price, volume, flags, bid, ask])


class TestIndicator:
    def test_calculate_shape_and_bands(self):
        m = _tick_matrix()
        m[40, 2] = 100.0   # big volume
        ind = LargeTrades(threshold=50.0, by="volume")
        out = ind.calculate(m)
        assert out.shape == (4, len(m))
        # row 0 price only set at detection
        assert np.isfinite(out[0, 40])
        assert np.isnan(out[0, 0])
        assert out[1, 40] == pytest.approx(100.0)         # volume band
        assert out[2, 40] == pytest.approx(100.0 * m[40, 1])  # notional band
        assert out[3, 40] in (1.0, -1.0)                   # side band

    def test_n_outputs(self):
        assert LargeTrades().n_outputs == 4

    def test_deep_events_and_labels(self):
        m = _tick_matrix()
        m[40, 2] = 100.0
        m[90, 2] = 70.0
        ind = LargeTrades(threshold=50.0, by="volume", label="volume")
        ind.calculate(m)
        ev = ind._deep_events
        assert list(ev["ts"]) == [m[40, 0], m[90, 0]]
        labels = ind.event_labels()
        assert labels == ["100", "70"]

    def test_label_none(self):
        m = _tick_matrix()
        m[40, 2] = 100.0
        ind = LargeTrades(threshold=50.0, label=None)
        ind.calculate(m)
        assert ind.event_labels() == []

    def test_build_bubble_columns(self):
        m = _tick_matrix()
        m[40, 2] = 100.0
        ind = LargeTrades(threshold=50.0, by="volume", label="volume")
        ind.calculate(m)
        cols = build_bubble_columns(ind._deep_events, ind._deep_style)
        assert len(cols["ts"]) == 1
        assert len(cols["size"]) == 1
        assert len(cols["color"]) == 1
        assert cols["text"] == ["100"]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            LargeTrades(by="bogus")
        with pytest.raises(ValueError):
            LargeTrades(label="bogus")
        with pytest.raises(ValueError):
            LargeTrades(scale="bogus")


# =====
# Indicator: aggregation + delta mode
# =====
def _burst_matrix():
    """[N x 6] with a tight same-side 3-print burst around index 100."""
    n = 200
    rng = np.random.default_rng(5)
    ts = (np.arange(n) * 1000).astype(float)
    price = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    volume = rng.uniform(1.0, 3.0, n)
    flags = np.ones(n)               # all buys
    # Tight burst: three same-ms-range buys summing to 150.
    ts[100], ts[101], ts[102] = 100_000.0, 100_010.0, 100_020.0
    volume[100] = volume[101] = volume[102] = 50.0
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, price, volume, flags, bid, ask])


class TestIndicatorAggregationDelta:
    def test_aggregated_band_carries_burst_sum(self):
        m = _burst_matrix()
        ind = LargeTrades(threshold=100.0, by="volume", aggregate_ms=50)
        out = ind.calculate(m)
        # Single event anchored at the last print of the burst (index 102).
        finite = np.where(np.isfinite(out[0]))[0]
        assert list(finite) == [102]
        assert out[1, 102] == pytest.approx(150.0)        # summed volume
        assert ind._deep_events["volume"][0] == pytest.approx(150.0)

    def test_delta_label_and_mode(self):
        m = _burst_matrix()
        ind = LargeTrades(threshold=100.0, by="delta", aggregate_ms=50,
                         label="delta")
        ind.calculate(m)
        ev = ind._deep_events
        assert ev["delta"][0] == pytest.approx(150.0)     # net buy delta
        assert ind.event_labels() == ["150"]

    def test_delta_balanced_burst_not_flagged(self):
        m = _burst_matrix()
        m[101, 3] = -1.0          # flip the middle print to a sell
        ind = LargeTrades(threshold=100.0, by="delta", aggregate_ms=50)
        out = ind.calculate(m)
        # Net delta = 50 - 50 + 50 = 50 < 100 -> below threshold -> no event.
        assert not np.isfinite(out[0]).any()

    def test_aggregation_live_backtest_parity(self):
        from tradetropy.ta.draw import Points

        m = _burst_matrix()
        ind = LargeTrades(threshold=100.0, by="volume", aggregate_ms=50,
                         label="volume")
        ind.live_refresh(_FakeTickProxy(m))
        prims = ind.draw(ind.plot_config)
        pts = next(p for p in prims["Large Trades"] if isinstance(p, Points))
        assert len(pts.x) == 1                            # one aggregated bubble
        out = ind.calculate(m)
        assert int(np.isfinite(out[0]).sum()) == 1


# =====
# Backtest integration: data access in on_data()
# =====
class TestEngineIntegration:
    def test_data_access_in_backtest(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        # TickData expects the raw [N x 7] layout: ts,bid,ask,volume,flags,vreal,price
        n = 150
        rng = np.random.default_rng(3)
        ts = np.arange(n) * 1000
        price = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
        volume = rng.uniform(1.0, 5.0, n)
        volume[70] = 100.0
        flags = np.ones(n)
        vreal = volume.copy()
        bid = price - 0.05
        ask = price + 0.05
        raw = np.column_stack([ts, bid, ask, volume, flags, vreal, price])

        seen = []

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=200)
                self.deep = self.add_indicator(
                    LargeTrades.refs(self.tk),
                    LargeTrades(threshold=50.0, by="volume"),
                )

            def on_data(self):
                if not np.isnan(self.deep.price[-1]):
                    seen.append((
                        float(self.deep.volume[-1]),
                        float(self.deep.side[-1]),
                    ))

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", raw, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        assert len(seen) == 1
        assert seen[0][0] == pytest.approx(100.0)
        assert seen[0][1] in (1.0, -1.0)


# =====
# Plotting smoke
# =====
class TestPlotting:
    def test_draw_emits_hollow_points_and_labels(self):
        from tradetropy.ta.draw import Points, Labels

        m = _tick_matrix()
        m[40, 2] = 100.0
        m[90, 2] = 70.0
        ind = LargeTrades(threshold=50.0, by="volume", label="volume")
        ind.calculate(m)

        prims = ind.draw(ind.plot_config)
        all_prims = [p for group in prims.values() for p in group]
        kinds = [type(p).__name__ for p in all_prims]
        assert "Points" in kinds and "Labels" in kinds
        pts = next(p for p in all_prims if isinstance(p, Points))
        assert pts.fill is False                 # hollow bubbles
        assert len(pts.x) == 2

    def test_renderer_draws_points_glyph(self):
        from bokeh.plotting import figure
        from tradetropy.plotting.render._tools import render_tool_groups

        m = _tick_matrix()
        m[40, 2] = 100.0
        ind = LargeTrades(threshold=50.0, by="volume", label="volume")
        ind.calculate(m)

        groups = ind.draw(ind.plot_config)
        fig = figure(x_axis_type="datetime")
        n_before = len(fig.renderers)
        render_tool_groups(fig, groups, theme=None, interval_ms=60_000, registry={})
        assert len(fig.renderers) > n_before



# =====
# Live streaming
# =====
class _Col:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, item):
        return self._arr[item]


class _FakeTickProxy:
    """Minimal stand-in for TickProxy exposing column window views."""

    def __init__(self, matrix):
        # matrix columns: [ts, price, volume, flags, bid, ask]
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


class TestLiveStreaming:
    def test_live_refresh_detects_over_window(self):
        from tradetropy.ta.draw import Points

        m = _tick_matrix(n=120)
        m[40, 2] = 100.0
        m[90, 2] = 80.0
        ind = LargeTrades(threshold=50.0, by="volume", label="volume")

        # live_refresh rebuilds the source matrix from the tick proxy window and
        # re-runs the exact causal detection used by the backtest.
        ind.live_refresh(_FakeTickProxy(m))
        prims = ind.draw(ind.plot_config)
        pts = next(p for p in prims["Large Trades"] if isinstance(p, Points))
        assert len(pts.x) == 2
        assert list(pts.size) and pts.fill is False

        # Parity with the backtest detection over the same data.
        out = ind.calculate(m)
        assert int(np.isfinite(out[0]).sum()) == 2

    def test_live_refresh_empty_proxy_is_noop(self):
        ind = LargeTrades(threshold=50.0, by="volume")
        ind.live_refresh(_FakeTickProxy(_tick_matrix(n=0)))
        assert ind.draw(ind.plot_config) == {}

    def test_live_document_module_imports(self):
        import tradetropy.plotting.live.document as _doc
        # LargeTrades now flows through the generic primitive live path.
        assert hasattr(_doc, "_build_live_vp_refs")
        assert not hasattr(_doc, "_build_live_deep_trades_refs")

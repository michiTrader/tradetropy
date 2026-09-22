"""Tests for the per-bar order-flow indicators (DeltaBars, CVD, VolumeInfo)
and the own-panel draw() primitive rendering enhancement."""

import numpy as np
import pytest

from tradetropy.ta.draw import Labels, Rects, Segments
from tradetropy.ta.order_flow import CVD, DeltaBars, VolumeInfo


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


# A clean two-bar tick set (60s bars): bar0 net buy +2, bar1 net sell -3.
def _clean_two_bars():
    ts = np.array([1_000, 2_000, 61_000, 62_000], dtype=np.int64)
    price = np.array([100.0, 100.0, 100.0, 100.0])
    volume = np.array([3.0, 1.0, 2.0, 5.0])
    flags = np.array([1.0, -1.0, 1.0, -1.0])
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, price, volume, flags, bid, ask])


class TestDeltaBars:
    def test_calculate_bands(self):
        m = _clean_two_bars()
        ind = DeltaBars(period="1m")
        out = ind.calculate(m)
        assert out.shape == (4, len(m))
        # Bands anchored at each bar's last tick (idx 1 and 3).
        np.testing.assert_allclose(out[0, [1, 3]], [2.0, -3.0])    # delta
        np.testing.assert_allclose(out[1, [1, 3]], [3.0, 2.0])     # ask_vol
        np.testing.assert_allclose(out[2, [1, 3]], [1.0, 5.0])     # bid_vol
        np.testing.assert_allclose(out[3, [1, 3]], [4.0, 7.0])     # total
        assert np.isnan(out[0, 0]) and np.isnan(out[0, 2])

    def test_n_outputs(self):
        assert DeltaBars().n_outputs == 4

    def test_draw_diverging_rects(self):
        m = _clean_two_bars()
        ind = DeltaBars(period="1m")
        ind.calculate(m)
        prims = ind.draw(ind.plot_config, interval_ms=60_000)
        rects = [p for p in prims if isinstance(p, Rects)]
        assert len(rects) == 1
        r = rects[0]
        assert list(r.y1) == [2.0, -3.0]
        assert r.fill_color[0] == ind.up_color      # +delta green
        assert r.fill_color[1] == ind.down_color    # -delta red
        # Bars are centered on bar_ts (0 and 60_000) to align with candles.
        centers = [(x0 + x1) / 2 for x0, x1 in zip(r.x0, r.x1)]
        np.testing.assert_allclose(centers, ind._bars["bar_ts"])


class TestCVD:
    def test_calculate_ohlc(self):
        m = _clean_two_bars()
        ind = CVD(period="1m")
        out = ind.calculate(m)
        assert out.shape == (4, len(m))
        # open/high/low/close at rep ticks (1 and 3).
        np.testing.assert_allclose(out[0, [1, 3]], [0.0, 2.0])     # open
        np.testing.assert_allclose(out[3, [1, 3]], [2.0, -1.0])    # close
        np.testing.assert_allclose(out[1, [1, 3]], [3.0, 4.0])     # high
        np.testing.assert_allclose(out[2, [1, 3]], [0.0, -1.0])    # low

    def test_draw_candle_default(self):
        m = _clean_two_bars()
        ind = CVD(period="1m")
        assert ind.style == "candle"
        ind.calculate(m)
        prims = ind.draw(ind.plot_config, interval_ms=60_000)
        # Classic candles by default: wick (Segments) plus body (Rects).
        kinds = {type(p).__name__ for p in prims}
        assert "Segments" in kinds and "Rects" in kinds
        # Wick segments are centered on bar_ts to align with candles.
        seg = next(p for p in prims if isinstance(p, Segments))
        np.testing.assert_allclose(list(seg.x0), ind._bars["bar_ts"])

    def test_draw_bar_style(self):
        m = _clean_two_bars()
        ind = CVD(period="1m", style="bar")
        ind.calculate(m)
        prims = ind.draw(ind.plot_config, interval_ms=60_000)
        # Single diverging bar per period (no candle wick).
        rects = [p for p in prims if isinstance(p, Rects)]
        assert len(prims) == 1 and len(rects) == 1
        assert not any(isinstance(p, Segments) for p in prims)
        r = rects[0]
        assert list(r.y1) == [2.0, -1.0]            # height = cumulative close
        assert r.fill_color[0] == ind.up_color      # rose (open 0 -> close 2)
        assert r.fill_color[1] == ind.down_color    # fell (open 2 -> close -1)
        centers = [(x0 + x1) / 2 for x0, x1 in zip(r.x0, r.x1)]
        np.testing.assert_allclose(centers, ind._bars["bar_ts"])

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError):
            CVD(period="1m", style="line")

    def test_n_outputs(self):
        assert CVD().n_outputs == 4


class TestVolumeInfo:
    def test_calculate_bands(self):
        m = _clean_two_bars()
        ind = VolumeInfo(period="1m")
        out = ind.calculate(m)
        assert out.shape == (6, len(m))
        np.testing.assert_allclose(out[0, [1, 3]], [2.0, -3.0])    # delta
        np.testing.assert_allclose(out[1, [1, 3]], [3.0, 2.0])     # ask_vol (buy)
        np.testing.assert_allclose(out[2, [1, 3]], [1.0, 5.0])     # bid_vol (sell)
        np.testing.assert_allclose(out[3, [1, 3]], [4.0, 7.0])     # total
        # delta_max and delta_min must be present and finite at rep ticks.
        assert np.isfinite(out[4, 1]) and np.isfinite(out[5, 1])

    def test_default_rows(self):
        ind = VolumeInfo(period="1m")
        assert ind.rows == ["delta_max", "delta_min", "delta"]

    def test_custom_rows(self):
        ind = VolumeInfo(period="1m", rows=["buy", "sell", "total"])
        assert ind.rows == ["buy", "sell", "total"]

    def test_invalid_row_raises(self):
        with pytest.raises(ValueError):
            VolumeInfo(rows=["bogus"])

    def test_draw_default_groups(self):
        m = _clean_two_bars()
        ind = VolumeInfo(period="1m")
        ind.calculate(m)
        groups = ind.draw(ind.plot_config, interval_ms=60_000)
        assert set(groups) == {"Delta Max", "Delta Min", "Delta"}

    def test_draw_all_rows(self):
        m = _clean_two_bars()
        ind = VolumeInfo(period="1m",
                         rows=["delta_max", "delta_min", "delta", "buy", "sell", "total"])
        ind.calculate(m)
        groups = ind.draw(ind.plot_config, interval_ms=60_000)
        assert set(groups) == {"Delta Max", "Delta Min", "Delta", "Buy", "Sell", "Total"}

    def test_draw_cells_and_labels(self):
        m = _clean_two_bars()
        ind = VolumeInfo(period="1m")
        ind.calculate(m)
        groups = ind.draw(ind.plot_config, interval_ms=60_000)
        for name, prims in groups.items():
            cells = [p for p in prims if isinstance(p, Rects)]
            labels = [p for p in prims if isinstance(p, Labels)]
            assert len(cells) == 1 and len(labels) == 1
            assert cells[0].fill_alpha == ind.cell_alpha
            r = cells[0]
            centers = [(x0 + x1) / 2 for x0, x1 in zip(r.x0, r.x1)]
            np.testing.assert_allclose(centers, ind._bars["bar_ts"])

    def test_row_colors_coherent(self):
        m = _clean_two_bars()
        ind = VolumeInfo(period="1m")
        ind.calculate(m)
        groups = ind.draw(ind.plot_config, interval_ms=60_000)
        dmax_cell = next(p for p in groups["Delta Max"] if isinstance(p, Rects))
        dmin_cell = next(p for p in groups["Delta Min"] if isinstance(p, Rects))
        delta_cell = next(p for p in groups["Delta"] if isinstance(p, Rects))
        assert dmax_cell.fill_color == ind.delta_max_color  # green single color
        assert dmin_cell.fill_color == ind.delta_min_color  # red single color
        assert delta_cell.fill_color == ind.delta_color     # gray single color

    def test_cell_alpha_default_opaque(self):
        assert VolumeInfo().cell_alpha >= 0.8

    def test_empty_returns_empty_dict(self):
        ind = VolumeInfo(period="1m")
        assert ind.draw(ind.plot_config) == {}


class TestLiveParity:
    @pytest.mark.parametrize("cls", [DeltaBars, CVD, VolumeInfo])
    def test_live_refresh_matches_calculate(self, cls):
        m = _tick_matrix()
        ind = cls(period="1m")
        out = ind.calculate(m)
        bars_calc = dict(ind._bars)

        ind2 = cls(period="1m")
        ind2.live_refresh(_FakeTickProxy(m))
        bars_live = ind2._bars
        # Same bars detected over the same window in both paths.
        np.testing.assert_array_equal(bars_calc["bar_ts"], bars_live["bar_ts"])

    def test_empty_proxy_noop(self):
        ind = CVD(period="1m")
        ind.live_refresh(_FakeTickProxy(_tick_matrix(n=0)))
        assert ind.draw(ind.plot_config) == []


class TestEngineIntegration:
    def test_on_data_reads_bands(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest.engine import BacktestEngine
        from tradetropy.core import TickData
        from tradetropy.session import SeshSimulatorBase

        n = 600
        rng = np.random.default_rng(3)
        ts = np.arange(n) * 1000               # 1s ticks over 10 minutes
        price = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
        volume = rng.uniform(1.0, 5.0, n)
        flags = np.where(rng.random(n) > 0.5, 1.0, -1.0)
        vreal = volume.copy()
        bid = price - 0.05
        ask = price + 0.05
        raw = np.column_stack([ts, bid, ask, volume, flags, vreal, price])

        seen = {"delta": 0, "cvd": 0}

        class S(Strategy):
            def init(self):
                self.tk = self.subscribe_ticks("SYM", window_size=700)
                self.delta = self.add_indicator(
                    DeltaBars.refs(self.tk), DeltaBars(period="1m"))
                self.cvd = self.add_indicator(
                    CVD.refs(self.tk), CVD(period="1m"))

            def on_data(self):
                if not np.isnan(self.delta.delta[-1]):
                    seen["delta"] += 1
                if not np.isnan(self.cvd.close[-1]):
                    seen["cvd"] += 1

        BacktestEngine.by_ticks(
            S(),
            data=(TickData("SYM", raw, tick_size=0.01),),
            sesh=SeshSimulatorBase("tick"),
        ).run()

        # At least a few closed 1m bars should have been observed.
        assert seen["delta"] > 0
        assert seen["cvd"] > 0


class TestOwnPanelRendering:
    """Task 3: own-panel draw() primitives must build a panel figure."""

    def _meta(self, prims, **kw):
        from tradetropy.plotting.config import IndicatorPlotMeta
        meta = IndicatorPlotMeta(
            name="X", values=np.empty((0,)), timestamps=np.empty((0,)),
            overlay=False, renderer="none", panel_title="X", **kw,
        )
        meta._draw_primitives = {"X": prims}
        return meta

    def test_panel_built_for_draw_primitives(self):
        from tradetropy.plotting.plotting import _build_indicator_panels
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme

        m = _clean_two_bars()
        ind = CVD(period="1m")
        ind.calculate(m)
        prims = ind.draw(ind.plot_config, interval_ms=60_000)

        config = PlotConfig()
        theme = _theme(config)
        figs = _build_indicator_panels([self._meta(prims)], config, None, theme, 60_000)
        assert len(figs) == 1
        # The panel figure carries glyph renderers from the primitives.
        assert len(figs[0].renderers) > 0

    def test_empty_primitives_skipped(self):
        from tradetropy.plotting.plotting import _build_indicator_panels
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme

        config = PlotConfig()
        theme = _theme(config)
        # Geometry-only (renderer none) but no primitives -> no panel.
        from tradetropy.plotting.config import IndicatorPlotMeta
        meta = IndicatorPlotMeta(
            name="Y", values=np.empty((0,)), timestamps=np.empty((0,)),
            overlay=False, renderer="none", panel_title="Y",
        )
        meta._draw_primitives = None
        figs = _build_indicator_panels([meta], config, None, theme, 60_000)
        assert len(figs) == 0


class TestLegends:
    """Tasks 4 & 6: legend suppression for Delta/CVD, per-row legends for Numbers."""

    def _build_panel(self, meta):
        from tradetropy.plotting.plotting import _build_indicator_panels
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme

        config = PlotConfig()
        theme = _theme(config)
        figs = _build_indicator_panels([meta], config, None, theme, 60_000)
        assert len(figs) == 1
        return figs[0]

    def _meta(self, prims_or_groups, show_legend, **kw):
        from tradetropy.plotting.config import IndicatorPlotMeta
        meta = IndicatorPlotMeta(
            name="P", values=np.empty((0,)), timestamps=np.empty((0,)),
            overlay=False, renderer="none", panel_title="P",
            show_legend=show_legend, **kw,
        )
        if isinstance(prims_or_groups, dict):
            meta._draw_primitives = prims_or_groups
        else:
            meta._draw_primitives = {"P": prims_or_groups}
        return meta

    def _legend_items(self, fig):
        legends = list(getattr(fig, "legend", []) or [])
        return legends[0].items if legends else []

    def test_cvd_panel_has_no_legend(self):
        m = _clean_two_bars()
        ind = CVD(period="1m")
        ind.calculate(m)
        prims = ind.draw(ind.plot_config, interval_ms=60_000)
        # CVD resolves show_legend False (own panel, not overridden).
        fig = self._build_panel(self._meta(prims, show_legend=False))
        assert len(self._legend_items(fig)) == 0

    def test_numbers_panel_has_three_default_legends(self):
        m = _clean_two_bars()
        ind = VolumeInfo(period="1m")
        ind.calculate(m)
        groups = ind.draw(ind.plot_config, interval_ms=60_000)
        # VolumeInfo default rows -> three toggleable legend entries.
        fig = self._build_panel(self._meta(groups, show_legend=True))
        labels = {item.label["value"] for item in self._legend_items(fig)}
        assert labels == {"Delta Max", "Delta Min", "Delta"}


class TestLazyLabels:
    """Task 4: indicator labels are collected and hidden when zoomed out."""

    def _panel_with_sink(self, groups, sink, xr):
        from tradetropy.plotting.plotting import _build_indicator_panels
        from tradetropy.plotting.config import PlotConfig, IndicatorPlotMeta
        from tradetropy.plotting._util import _theme

        meta = IndicatorPlotMeta(
            name="VI", values=np.empty((0,)), timestamps=np.empty((0,)),
            overlay=False, renderer="none", panel_title="VI", show_legend=True,
        )
        meta._draw_primitives = groups
        config = PlotConfig()
        theme = _theme(config)
        figs = _build_indicator_panels(
            [meta], config, xr, theme, 60_000,
            label_sink=sink, lazy_zoom_range=config.labels_zoom_range,
        )
        return figs, config

    def test_labels_collected_and_zoom_gated(self):
        from bokeh.models import Range1d
        from tradetropy.plotting._layout import _configure_lazy_labels

        m = _clean_two_bars()
        ind = VolumeInfo(period="1m")
        ind.calculate(m)
        groups = ind.draw(ind.plot_config, interval_ms=60_000)

        sink: list = []
        xr = Range1d(start=0.0, end=1.0)
        figs, config = self._panel_with_sink(groups, sink, xr)
        assert len(figs) == 1
        # One label renderer per row; default 3 rows -> 3 entries.
        assert len(sink) == 3
        assert all(grp is not None for _, grp in sink)

        # Zoomed out -> labels start hidden.
        _configure_lazy_labels(xr, sink, 60_000, config.labels_zoom_range,
                               initial_zoomed_in=False)
        assert all(not lbl.visible for lbl, _ in sink)
        # Zoomed in -> labels start visible.
        _configure_lazy_labels(xr, sink, 60_000, config.labels_zoom_range,
                               initial_zoomed_in=True)
        assert all(lbl.visible for lbl, _ in sink)

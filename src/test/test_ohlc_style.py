"""
Tests for the ohlc_style ("candle" | "bar") plotting option.

Coverage:
  - PlotConfig exposes ohlc_style with default "candle".
  - _ohlc_tick_bounds computes the left/right tick endpoints (ts -/+ width/2).
  - build_ohlc_source includes the ts_left / ts_right columns.
  - The static render_ohlc draws candlesticks (segment + vbar) in "candle" mode
    and OHLC bars (3 segments, no vbar) in "bar" mode.
  - The live document builder accepts ohlc_style="bar" and produces a source
    carrying the tick columns plus the bar glyphs.

These run headless (no Bokeh server, no browser).
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine
from tradetropy.ta import SMA


def _count_glyphs(fig):
    """Count Segment and VBar glyph renderers on a Bokeh figure."""
    from bokeh.models import Segment, VBar
    segs = vbars = 0
    for r in fig.renderers:
        glyph = getattr(r, "glyph", None)
        if isinstance(glyph, Segment):
            segs += 1
        elif isinstance(glyph, VBar):
            vbars += 1
    return segs, vbars


@pytest.mark.unit
class TestOhlcStyleConfig:
    def test_default_is_candle(self):
        from tradetropy.plotting.config import PlotConfig
        assert PlotConfig().ohlc_style == "candle"

    def test_bar_value(self):
        from tradetropy.plotting.config import PlotConfig
        assert PlotConfig(ohlc_style="bar").ohlc_style == "bar"


@pytest.mark.unit
class TestTickBounds:
    def test_bounds_offset_by_half_width(self):
        from tradetropy.plotting._util import _ohlc_tick_bounds
        ts_ms = np.array([60_000, 120_000], dtype=np.int64)
        width = 54_000
        left, right = _ohlc_tick_bounds(ts_ms, width)
        left_ms = left.astype("datetime64[ms]").astype(np.int64)
        right_ms = right.astype("datetime64[ms]").astype(np.int64)
        assert list(left_ms) == [60_000 - 27_000, 120_000 - 27_000]
        assert list(right_ms) == [60_000 + 27_000, 120_000 + 27_000]

    def test_returns_datetime64(self):
        from tradetropy.plotting._util import _ohlc_tick_bounds
        left, right = _ohlc_tick_bounds(np.array([60_000]), 54_000)
        assert left.dtype == np.dtype("datetime64[ms]")
        assert right.dtype == np.dtype("datetime64[ms]")


@pytest.mark.unit
class TestStaticSource:
    def test_source_has_tick_columns(self):
        from tradetropy.plotting.sources import build_ohlc_source
        ohlc = np.array([
            [60_000, 10.0, 12.0, 9.0, 11.0, 100.0, 0.0],
            [120_000, 11.0, 11.5, 10.0, 10.5, 80.0, 0.0],
        ], dtype=np.float64)
        src = build_ohlc_source(ohlc, interval_ms=60_000)
        assert "ts_left" in src.data
        assert "ts_right" in src.data
        assert len(src.data["ts_left"]) == 2
        assert len(src.data["ts_right"]) == 2


@pytest.mark.unit
class TestStaticRender:
    def _fig_and_source(self):
        from bokeh.plotting import figure
        from tradetropy.plotting.sources import build_ohlc_source
        ohlc = np.array([
            [60_000, 10.0, 12.0, 9.0, 11.0, 100.0, 0.0],
            [120_000, 11.0, 11.5, 10.0, 10.5, 80.0, 0.0],
            [180_000, 10.5, 13.0, 10.0, 12.5, 90.0, 0.0],
        ], dtype=np.float64)
        src = build_ohlc_source(ohlc, interval_ms=60_000)
        fig = figure(x_axis_type="datetime")
        return fig, src

    def test_candle_mode_draws_segment_and_vbar(self):
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme
        from tradetropy.plotting.render._ohlc_volume import render_ohlc
        fig, src = self._fig_and_source()
        cfg = PlotConfig(ohlc_style="candle")
        render_ohlc(fig, src, _theme(cfg), cfg, 60_000, "BTCUSDT")
        segs, vbars = _count_glyphs(fig)
        assert segs == 1
        assert vbars == 1

    def test_bar_mode_draws_three_segments_no_vbar(self):
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting._util import _theme
        from tradetropy.plotting.render._ohlc_volume import render_ohlc
        fig, src = self._fig_and_source()
        cfg = PlotConfig(ohlc_style="bar")
        render_ohlc(fig, src, _theme(cfg), cfg, 60_000, "BTCUSDT")
        segs, vbars = _count_glyphs(fig)
        assert segs == 3
        assert vbars == 0


def _klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.zeros((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["price"]] = close
    return out


class _SmaStrategy(Strategy):
    def init(self):
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=2000)
        self.fast = self.add_indicator(self.ohlc.close_ref, SMA(3))

    def on_data(self):
        pass


@pytest.fixture
def _replay_engine(klines_20k):
    ticks = _klines_to_ticks(klines_20k[:300])
    engine = ReplayEngine.by_ticks(
        _SmaStrategy(),
        data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"),
    )
    engine.prepare()
    return engine


@pytest.mark.unit
class TestLiveDocument:
    def _build(self, engine, ohlc_style):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document
        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark", ohlc_style=ohlc_style),
            max_candles=200,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=engine._ctrl,
            io_loop=None,
        )
        return updater

    def test_bar_mode_source_has_tick_columns(self, _replay_engine):
        updater = self._build(_replay_engine, "bar")
        src = updater._ohlc_refs[0].source
        assert "ts_left" in src.data
        assert "ts_right" in src.data
        assert len(src.data["ts_left"]) == len(src.data["ts"])

    def test_candle_mode_source_has_tick_columns(self, _replay_engine):
        updater = self._build(_replay_engine, "candle")
        src = updater._ohlc_refs[0].source
        assert "ts_left" in src.data
        assert "ts_right" in src.data

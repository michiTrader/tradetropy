"""
Regression: the live/replay volume bars must be pinned to the bottom band of
the price pane (TradingView style), like the backtest chart already is.

Backtest anchors the volume on a secondary ``vol`` Range1d mapped so the peak
reaches ~20% of the frame and re-anchors it on every pan/zoom. Live used to use
a DataRange1d with no re-anchor, so the bars drifted with the price plane on
vertical pan (their base detaching from the bottom). These tests lock in the
ported behavior:

- the live ``vol`` range is a fixed Range1d (not a DataRange1d),
- a re-anchor CustomJS is wired on both x_range and y_range,
- the Python re-anchor keeps the band mapped to ~20% of the visible peak as
  live data streams in,
- no volume range is created when plot_volume is off.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS, _OHLC_COL
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine


def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.zeros((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["price"]] = close
    return out


class _VolStrategy(Strategy):
    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=5000)
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=5000)

    def on_data(self):
        pass


@pytest.fixture
def ticks(klines_20k):
    return klines_to_ticks(klines_20k[:300])


def _build(ticks, *, plot_volume=True):
    from bokeh.document import Document
    from tradetropy.plotting.config import PlotConfig
    from tradetropy.plotting.live.document import build_live_document

    eng = ReplayEngine.by_ticks(
        _VolStrategy(), data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"),
    )
    eng.prepare()
    doc = Document()
    _doc, updater = build_live_document(
        strategy=eng.strategy,
        config=PlotConfig(theme="dark", plot_volume=plot_volume),
        max_candles=5000, broker=eng._sesh._broker, doc=doc,
    )
    updater.populate_history(max_age_minutes=0)
    return eng, updater


def _feed_all(eng, updater, sym="BTCUSDT"):
    """Replay every live tick through the engine and refresh the chart once."""
    ds = eng._sesh._datasets[sym]
    n_warmup = eng._sesh._warmup_ticks[sym]
    for i in range(n_warmup, len(ds)):
        eng._sesh._cursor[sym] = i
        eng._apply_tick(sym, ds[i], with_broker=True)
    updater.update(bar_closed=True)


@pytest.mark.unit
class TestLiveVolumeAnchor:
    def test_vol_range_is_fixed_range1d(self, ticks):
        from bokeh.models import Range1d, DataRange1d

        _eng, updater = _build(ticks)
        fig = updater._fig_ohlc
        assert "vol" in fig.extra_y_ranges
        vol_range = fig.extra_y_ranges["vol"]
        # Must be a fixed Range1d (bottom-band), never an auto-fitting DataRange1d
        # (which let the bars drift with the price plane on vertical pan).
        assert isinstance(vol_range, Range1d)
        assert not isinstance(vol_range, DataRange1d)

    def test_reanchor_callback_wired_on_both_axes(self, ticks):
        from bokeh.models import CustomJS

        _eng, updater = _build(ticks)
        fig = updater._fig_ohlc

        def _has_vol_cb(range_obj):
            for cbs in range_obj.js_property_callbacks.values():
                for cb in cbs:
                    if isinstance(cb, CustomJS) and "Volume" in cb.code:
                        return True
            return False

        # The re-anchor must fire on horizontal AND vertical navigation, so the
        # band stays glued to the pane bottom instead of floating.
        assert _has_vol_cb(fig.x_range), "vol re-anchor not wired on x_range"
        assert _has_vol_cb(fig.y_range), "vol re-anchor not wired on y_range"

    def test_ohlc_ref_carries_vol_range(self, ticks):
        _eng, updater = _build(ticks)
        ref = updater._ohlc_refs[0]
        assert ref.vol_range is not None
        assert ref.vol_range is updater._fig_ohlc.extra_y_ranges["vol"]

    def test_band_maps_peak_to_20pct_after_history(self, ticks):
        eng, updater = _build(ticks)
        _feed_all(eng, updater)
        ref = updater._ohlc_refs[0]
        vols = np.asarray(ref.source.data["Volume"], dtype=np.float64)
        assert len(vols) > 0

        # With the full history in view (x_range spans all candles), the band
        # end must map the visible peak to ~20% of the frame: end == peak / 0.20.
        ts_ms = np.asarray(ref.source.data["ts"], dtype="datetime64[ms]").astype(np.int64)
        updater._fig_ohlc.x_range.start = float(ts_ms.min())
        updater._fig_ohlc.x_range.end = float(ts_ms.max())
        updater._reanchor_volume(ref)

        peak = float(vols[np.isfinite(vols)].max())
        assert ref.vol_range.start == 0
        assert ref.vol_range.end == pytest.approx(peak / 0.20, rel=1e-9)

    def test_band_reanchors_as_volume_grows(self, ticks):
        eng, updater = _build(ticks)
        _feed_all(eng, updater)
        ref = updater._ohlc_refs[0]

        ts_ms = np.asarray(ref.source.data["ts"], dtype="datetime64[ms]").astype(np.int64)
        updater._fig_ohlc.x_range.start = float(ts_ms.min())
        updater._fig_ohlc.x_range.end = float(ts_ms.max())
        updater._reanchor_volume(ref)
        end_before = ref.vol_range.end

        # Simulate a spike on the last visible bar; the band must grow to keep
        # the (now larger) peak at ~20% of the frame.
        vols = np.asarray(ref.source.data["Volume"], dtype=np.float64)
        spike = float(np.nanmax(vols)) * 5.0
        last = len(vols) - 1
        ref.source.patch({"Volume": [(last, spike)]})
        updater._reanchor_volume(ref)

        assert ref.vol_range.end > end_before
        assert ref.vol_range.end == pytest.approx(spike / 0.20, rel=1e-9)

    def test_no_vol_range_when_plot_volume_off(self, ticks):
        _eng, updater = _build(ticks, plot_volume=False)
        fig = updater._fig_ohlc
        assert "vol" not in (fig.extra_y_ranges or {})
        assert updater._ohlc_refs[0].vol_range is None

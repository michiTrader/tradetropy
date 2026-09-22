"""
An EVENT indicator mounted on a secondary (higher) timeframe must reach the
chart with one point per event, not forward-filled.

A non-primary indicator is realigned onto the primary candle index before
plotting. For a continuous series (an HTF moving average) that alignment is a
reindex + ffill, which is what makes it a step line on the finer chart. For an
event series that carries its own per-point timestamps (ConfirmedPivot, NBS,
ZigZag, HHLL...) the ffill was wrong: every 5m pivot was repeated on all the 1m
bars up to the next event of that band and, because the marker's x comes from
the indicator's ts band, all those copies landed on the same pixel. The result
was hundreds of identical stacked glyphs per pivot and a HoverTool that listed
the same value once per copy.

``_align_events_to_index`` now places each event on the primary bar that
contains it and leaves the rest NaN, so the plotted point count equals the real
event count. Continuous HTF indicators keep the ffill.
"""

import numpy as np
import pytest

from tradetropy import BacktestEngine, Strategy
from tradetropy.core.data_types import KlineData
from tradetropy.plotting._util import (
    _align_events_to_index,
    _gather_plot_data,
    _has_real_ts_bands,
)
from tradetropy.plotting.sources import build_indicator_source
from tradetropy.ta import NBS, SMA

_T0 = 1_700_000_000_000  # realistic epoch ms: ts bands must look like epoch ms


def _klines(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    m = np.zeros((n, 7), dtype=np.float64)
    m[:, 0] = _T0 + np.arange(n) * 60_000.0
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    m[:, 1] = open_
    m[:, 2] = np.maximum(open_, close) + rng.uniform(0, 0.4, n)
    m[:, 3] = np.minimum(open_, close) - rng.uniform(0, 0.4, n)
    m[:, 4] = close
    m[:, 5] = rng.uniform(1, 10, n)
    return KlineData(symbol="SYM", data=m, timeframe="1m", tick_size=0.01)


class _HtfStrat(Strategy):
    """1m chart with a 5m NBS (event) and a 5m SMA (continuous)."""

    def init(self):
        self.o1 = self.subscribe_ohlc("SYM", "1m", window_size=300)
        self.o5 = self.subscribe_ohlc("SYM", "5m", window_size=300)
        self.nbs_5m = self.add_indicator(
            [self.o5.high_ref, self.o5.low_ref, self.o5.ts_ref],
            NBS(swing=2),
            name="nbs_5m",
        )
        self.sma_5m = self.add_indicator(self.o5.close_ref, SMA(10), name="sma_5m")

    def on_data(self):
        pass


def _run():
    strat = _HtfStrat()
    eng = BacktestEngine.by_klines(strat, data=(_klines(),))
    eng.run()
    return strat, eng


@pytest.mark.unit
class TestAlignEventsToIndex:
    def test_events_land_on_their_own_bar_only(self):
        target = np.array([0.0, 60_000.0, 120_000.0, 180_000.0])
        # Two events, on the 1st and 3rd target bars.
        out = _align_events_to_index(
            np.array([10.0, np.nan, 30.0]),
            np.array([0.0, 60_000.0, 120_000.0]),
            target,
        )
        assert np.array_equal(np.isfinite(out), [True, False, True, False])
        assert out[0] == 10.0 and out[2] == 30.0

    def test_coarse_source_is_not_forward_filled(self):
        # One 5-bar-spaced event: only its own slot is filled.
        target = _T0 + np.arange(10) * 60_000.0
        out = _align_events_to_index(
            np.array([7.0]), np.array([_T0 + 5 * 60_000.0]), target
        )
        assert int(np.isfinite(out).sum()) == 1
        assert out[5] == 7.0

    def test_event_before_first_target_bar_is_dropped(self):
        out = _align_events_to_index(
            np.array([1.0]), np.array([0.0]), np.array([60_000.0, 120_000.0])
        )
        assert not np.isfinite(out).any()

    def test_empty_inputs(self):
        assert len(_align_events_to_index(np.array([]), np.array([]),
                                          np.array([1.0]))) == 1
        assert len(_align_events_to_index(np.array([1.0]), np.array([1.0]),
                                          np.array([]))) == 0


@pytest.mark.unit
class TestHasRealTsBands:
    def test_epoch_ms_band_detected(self):
        class _Ind:
            ts_band_indices = [1]

        bands = np.array([[1.0, 2.0], [_T0, _T0 + 1000.0]])
        assert _has_real_ts_bands(_Ind(), bands) is True

    def test_plain_data_band_is_not_a_ts_band(self):
        # Some indicators reuse ts_band_indices for extra data bands (volume,
        # notional, side...). Those must keep the continuous ffill path.
        class _Ind:
            ts_band_indices = [1]

        bands = np.array([[1.0, 2.0], [125.0, 340.0]])
        assert _has_real_ts_bands(_Ind(), bands) is False

    def test_no_ts_bands(self):
        class _Ind:
            ts_band_indices = []

        assert _has_real_ts_bands(_Ind(), np.zeros((2, 2))) is False


@pytest.mark.unit
class TestHtfEventIndicatorPlot:
    def test_no_duplicated_points_and_no_lost_events(self):
        strat, eng = _run()
        data = _gather_plot_data(eng)

        meta = next(m for m in data["indicators"] if m.name == "nbs_5m")
        defn = next(
            d for d in strat._indicator_defs
            if isinstance(d["indicator"], NBS)
        )
        store = defn["ohlc_proxy"]._ohlc_store
        assert defn["ohlc_proxy"].interval_ms == 300_000

        srcs = build_indicator_source(meta)
        total_events = 0
        for band, name in enumerate(NBS.output_names):
            price = store.matrix[:, store.col_index[defn["col_names"][band]]]
            ts = store.matrix[:, store.col_index[defn["col_names"][band + 8]]]
            ok = np.isfinite(price) & np.isfinite(ts)
            raw = set(zip(ts[ok].astype(np.int64).tolist(), price[ok].tolist()))
            total_events += len(raw)

            d = srcs[band].data
            plotted_ts = np.asarray(d["ts"]).astype("datetime64[ms]").astype(np.int64)
            plotted = set(zip(plotted_ts.tolist(), np.asarray(d["value"]).tolist()))

            # One glyph per event: no ffill duplicates, no lost pivots.
            assert len(d["value"]) == len(raw), (
                f"band {name}: {len(d['value'])} glyph points for {len(raw)} events"
            )
            assert plotted == raw, f"band {name}: plotted coords != store coords"

        assert total_events > 0, "the 5m NBS produced no pivots to check"

    def test_continuous_htf_indicator_is_still_forward_filled(self):
        _, eng = _run()
        data = _gather_plot_data(eng)
        meta = next(m for m in data["indicators"] if m.name == "sma_5m")
        vals = np.asarray(meta.values, dtype=np.float64).ravel()
        finite = np.isfinite(vals)
        n_bars = len(data["ohlc_array"])

        # An HTF average must exist on (nearly) every 1m bar after warmup: the
        # ffill path is unchanged for continuous series.
        assert finite.sum() > n_bars * 0.9
        # And it must be a step: consecutive equal values inside each 5m bucket.
        assert (np.diff(vals[finite]) == 0).sum() > finite.sum() * 0.5

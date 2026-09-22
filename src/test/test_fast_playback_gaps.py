"""
Regression: fast playback (e.g. 64x or speed-scaled Step) closes many candles
between two coalesced chart frames. A single update() must backfill ALL closed
candles (and indicator points) so the chart has no gaps.

Reproduces the bug by feeding a batch of candle-ticks into the engine WITHOUT a
chart attached (so no per-tick update fires), then calling updater.update() once
and asserting the OHLC + indicator sources contain every closed candle.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS, _OHLC_COL
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.ta import SMA
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


class _SmaStrategy(Strategy):
    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=5000)
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=5000)
        self.sma = self.add_indicator(self.ohlc.close_ref, SMA(5))

    def on_data(self):
        pass


@pytest.fixture
def ticks(klines_20k):
    return klines_to_ticks(klines_20k[:300])


def _build(ticks):
    from bokeh.document import Document
    from tradetropy.plotting.config import PlotConfig
    from tradetropy.plotting.live.document import build_live_document

    eng = ReplayEngine.by_ticks(
        _SmaStrategy(), data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"),
    )
    eng.prepare()
    doc = Document()
    _doc, updater = build_live_document(
        strategy=eng.strategy, config=PlotConfig(theme="dark"),
        max_candles=5000, broker=eng._sesh._broker, doc=doc,
    )
    updater.populate_history(max_age_minutes=0)
    return eng, updater


def _ring_closed_ts(ring):
    n = min(ring._n_closed, ring._W)
    return set(int(t) for t in ring.closed_window(_OHLC_COL["ts"], n))


@pytest.mark.unit
class TestFastPlaybackNoGaps:
    def test_ohlc_no_candle_gaps_after_batch(self, ticks):
        eng, updater = _build(ticks)
        sym = "BTCUSDT"
        ds = eng._sesh._datasets[sym]
        n_warmup = eng._sesh._warmup_ticks[sym]

        # Feed the whole live region in ONE batch (no chart updates per tick),
        # simulating a fast step where the IOLoop coalesces everything.
        for i in range(n_warmup, len(ds)):
            eng._sesh._cursor[sym] = i
            eng._apply_tick(sym, ds[i], with_broker=True)

        # A single coalesced chart frame.
        updater.update(bar_closed=True)

        ring = eng.strategy.ohlc._ohlc_ring
        ring_ts = _ring_closed_ts(ring)
        src_ts = set(int(np.datetime64(t, "ms").astype(np.int64))
                     for t in updater._ohlc_refs[0].source.data["ts"])

        # Every closed candle in the ring must be present in the chart source.
        missing = ring_ts - src_ts
        assert not missing, f"{len(missing)} candles missing from the chart"

    def test_indicator_no_gaps_after_batch(self, ticks):
        eng, updater = _build(ticks)
        sym = "BTCUSDT"
        ds = eng._sesh._datasets[sym]
        n_warmup = eng._sesh._warmup_ticks[sym]
        for i in range(n_warmup, len(ds)):
            eng._sesh._cursor[sym] = i
            eng._apply_tick(sym, ds[i], with_broker=True)
        updater.update(bar_closed=True)

        # The SMA band should have roughly as many points as closed candles
        # (minus warmup where SMA is NaN), i.e. no large gap.
        band = updater._indicator_refs[0].bands[0]
        n_pts = len(band.source.data["ts"])
        ring = eng.strategy.ohlc._ohlc_ring
        n_closed = min(ring._n_closed, ring._W)
        # Allow the SMA warmup (length-1) of missing leading points.
        assert n_pts >= n_closed - 6, (
            f"indicator has gaps: {n_pts} points vs {n_closed} closed candles"
        )


# =====
# Footprint + candle precision under coalesced fast frames
# =====

def _varied_ticks(n_candles=30, ticks_per=8, base=1_700_000_040_000):
    """Ticks across many candles with varied price + aggressor side."""
    rng = np.random.default_rng(7)
    rows = []
    for cdl in range(n_candles):
        p0 = 50_000 + cdl * 10
        for k in range(ticks_per):
            ts = base + cdl * 60_000 + k * 7_000
            price = p0 + int(rng.integers(-5, 6))
            flags = 32 if rng.random() > 0.5 else 64  # buy/sell aggressor bits
            r = np.zeros(N_TICK_COLS, dtype=np.float64)
            r[_TICK_COL["ts"]] = ts
            r[_TICK_COL["bid"]] = price - 0.5
            r[_TICK_COL["ask"]] = price + 0.5
            r[_TICK_COL["price"]] = price
            r[_TICK_COL["volume"]] = float(rng.uniform(1, 5))
            r[_TICK_COL["flags"]] = flags
            rows.append(r)
    return np.array(rows)


class _FpStrategy(Strategy):
    def init(self):
        self.tp = self.subscribe_ticks("SYM", window_size=5000)
        self.ohlc = self.subscribe_ohlc("SYM", 60_000, window_size=5000)
        self.fp = self.subscribe_footprint(
            "SYM", timeframe=60_000, window_size=200,
            tick_size=1.0, aggressor_col="flags",
        )

    def on_data(self):
        pass


def _build_fp(ticks):
    from bokeh.document import Document
    from tradetropy.plotting.config import PlotConfig
    from tradetropy.plotting.live.document import build_live_document

    eng = ReplayEngine.by_ticks(
        _FpStrategy(), data=(TickData("SYM", ticks, tick_size=1.0),),
        speed=float("inf"),
    )
    eng.prepare()
    doc = Document()
    _doc, updater = build_live_document(
        strategy=eng.strategy,
        config=PlotConfig(theme="dark", plot_footprint=True),
        max_candles=5000, broker=eng._sesh._broker, doc=doc,
    )
    updater.populate_history(max_age_minutes=0)
    return eng, updater


@pytest.mark.unit
class TestFastPlaybackFootprint:
    def test_footprint_no_candle_gaps_after_batch(self):
        ticks = _varied_ticks()
        eng, updater = _build_fp(ticks)
        sym = "SYM"
        ds = eng._sesh._datasets[sym]
        nw = eng._sesh._warmup_ticks[sym]
        for i in range(nw, len(ds)):
            eng._sesh._cursor[sym] = i
            eng._apply_tick(sym, ds[i], with_broker=True)

        updater.update(bar_closed=True)

        ring = eng.strategy.ohlc._ohlc_ring
        n_closed = min(ring._n_closed, ring._W)
        fpref = updater._fp_refs[0]
        # One row-count entry per drawn footprint candle: must cover (almost)
        # every closed candle, not just the latest one (the pre-fix bug).
        assert len(fpref._candle_bid_row_counts) >= n_closed - 1, (
            f"footprint drew {len(fpref._candle_bid_row_counts)} candles "
            f"of {n_closed} closed"
        )

    def test_footprint_x_positions_match_candles(self):
        ticks = _varied_ticks()
        eng, updater = _build_fp(ticks)
        sym = "SYM"
        ds = eng._sesh._datasets[sym]
        nw = eng._sesh._warmup_ticks[sym]
        for i in range(nw, len(ds)):
            eng._sesh._cursor[sym] = i
            eng._apply_tick(sym, ds[i], with_broker=True)
        updater.update(bar_closed=True)

        xs = np.asarray(updater._fp_refs[0].source_bid.data["x"])
        fp_ms = np.unique(xs.astype("datetime64[ms]").astype(np.int64))
        # Each drawn candle contributes one distinct bid-column x (open - offset).
        # The distinct columns must be evenly spaced by the candle interval, i.e.
        # contiguous with no gaps and no two candles collapsed onto one x
        # (the desync/overlap symptom). Allow the trailing partial column.
        diffs = np.diff(fp_ms)
        interval = 60_000
        closed_diffs = diffs[diffs > 0]
        assert np.all(closed_diffs % interval == 0), (
            f"footprint columns not on the candle grid: {set(closed_diffs.tolist())}"
        )
        # No big gaps: every step is exactly one candle (contiguous coverage).
        assert np.all(closed_diffs == interval), (
            f"gaps between footprint candles: {sorted(set(closed_diffs.tolist()))}"
        )
        ring = eng.strategy.ohlc._ohlc_ring
        n_closed = min(ring._n_closed, ring._W)
        assert len(fp_ms) >= n_closed - 1


@pytest.mark.unit
class TestFastPlaybackCandlePrecision:
    def test_partial_candle_finalized_after_jump(self):
        # A candle streamed as a partial frame and then closed during a coalesced
        # jump must end with its FINAL OHLC, not stale intrabar values.
        base = 1_700_000_040_000
        rows = []
        for cdl in range(20):
            for k in range(10):
                ts = base + cdl * 60_000 + k * 5_000
                price = 50_000 + cdl * 20 + k * 3  # rising within each candle
                r = np.zeros(N_TICK_COLS, dtype=np.float64)
                r[_TICK_COL["ts"]] = ts
                r[_TICK_COL["bid"]] = price - 0.5
                r[_TICK_COL["ask"]] = price + 0.5
                r[_TICK_COL["price"]] = price
                r[_TICK_COL["volume"]] = 2.0
                r[_TICK_COL["flags"]] = 32
                rows.append(r)
        ticks = np.array(rows)

        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        sym = "SYM"
        eng = ReplayEngine.by_ticks(
            _OhlcOnlyStrategy(), data=(TickData("SYM", ticks, tick_size=1.0),),
            speed=float("inf"),
        )
        eng.prepare()
        doc = Document()
        _doc, updater = build_live_document(
            strategy=eng.strategy, config=PlotConfig(theme="dark"),
            max_candles=5000, broker=eng._sesh._broker, doc=doc,
        )
        updater.populate_history(max_age_minutes=0)

        ds = eng._sesh._datasets[sym]
        nw = eng._sesh._warmup_ticks[sym]
        ring = eng.strategy.ohlc._ohlc_ring
        i = nw
        # Feed until mid-way through a forming candle, then render a partial frame.
        while i < len(ds):
            eng._sesh._cursor[sym] = i
            eng._apply_tick(sym, ds[i], with_broker=True)
            i += 1
            if (ring._ts_current_candle > 0 and i > nw + 25
                    and (int(ds[i - 1, _TICK_COL["ts"]]) % 60_000)
                    == base % 60_000 + 25_000):
                break
        updater.update(bar_closed=False)

        src = updater._ohlc_refs[0].source
        partial_ts = int(np.datetime64(src.data["ts"][-1], "ms").astype(np.int64))

        # Feed the rest in one batch and coalesce into a single frame.
        while i < len(ds):
            eng._sesh._cursor[sym] = i
            eng._apply_tick(sym, ds[i], with_broker=True)
            i += 1
        updater.update(bar_closed=True)

        ts_arr = np.asarray(src.data["ts"]).astype("datetime64[ms]").astype(np.int64)
        row = int(np.where(ts_arr == partial_ts)[0][0])
        src_close = float(src.data["Close"][row])
        src_high = float(src.data["High"][row])

        n = min(ring._n_closed, ring._W)
        tsw = ring.closed_window(_OHLC_COL["ts"], n)
        cw = ring.closed_window(_OHLC_COL["close"], n)
        hw = ring.closed_window(_OHLC_COL["high"], n)
        j = int(np.where(tsw == partial_ts)[0][0])
        assert src_close == pytest.approx(float(cw[j]))
        assert src_high == pytest.approx(float(hw[j]))


class _OhlcOnlyStrategy(Strategy):
    def init(self):
        self.tp = self.subscribe_ticks("SYM", window_size=5000)
        self.ohlc = self.subscribe_ohlc("SYM", 60_000, window_size=5000)

    def on_data(self):
        pass

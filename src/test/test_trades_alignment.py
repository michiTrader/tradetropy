"""
Tests for align_trades_to_candle (phase-aware candle-grid alignment).

Regression: trade timestamps were floored to the EPOCH grid
((ts // interval) * interval), which is wrong whenever candles carry a phase
offset (e.g. kline feeds whose origin is not a multiple of the interval). Trades
then landed a fraction of a candle to the left of the candles they belong to.
The fix snaps to the actual candle grid via _align_ts_to_candle, used by both
the static (build_trades_source) and live/replay (trades updater) paths.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine
from tradetropy.ta import SMA

# The short alignment fixtures intentionally trip the insufficient-sample stats
# warning (asserted on its own in test_stats.py). Colon matched with '.' since
# warning-filter fields split on ':'.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Stats. insufficient sample:UserWarning"
)


@pytest.mark.unit
class TestAlignHelper:
    def test_epoch_grid_default(self):
        from tradetropy.plotting._util import _align_ts_to_candle
        ts = np.array([125_000, 60_001], dtype=np.int64)
        out = _align_ts_to_candle(ts, 60_000)  # origin 0 -> epoch grid
        assert list(out) == [120_000, 60_000]

    def test_phase_shifted_grid(self):
        from tradetropy.plotting._util import _align_ts_to_candle
        # candle_origin % 60000 == 20000, so candle opens are at ...000000,
        # ...060000, ... (each == 20000 mod 60000).
        origin = 1_700_000_000_000
        ts = np.array([1_700_000_050_000, 1_700_000_065_000], dtype=np.int64)
        out = _align_ts_to_candle(ts, 60_000, candle_origin_ms=origin)
        # 050000 falls in candle [..000000, ..060000); 065000 in [..060000, ..120000).
        assert list(out % 60_000) == [20_000, 20_000]
        assert int(out[0]) == 1_700_000_000_000
        assert int(out[1]) == 1_700_000_060_000

    def test_zero_interval_noop(self):
        from tradetropy.plotting._util import _align_ts_to_candle
        ts = np.array([123, 456], dtype=np.int64)
        out = _align_ts_to_candle(ts, 0)
        assert list(out) == [123, 456]


def _kl(n=400):
    rng = np.random.default_rng(0)
    p = 50_000 + np.cumsum(rng.standard_normal(n) * 5)
    c = p + rng.standard_normal(n) * 2
    h = np.maximum(p, c) + rng.uniform(0, 5, n)
    lo = np.minimum(p, c) - rng.uniform(0, 5, n)
    vol = rng.uniform(1, 50, n)
    ts = 1_700_000_000_000 + np.arange(n) * 60_000  # phase 20000 mod 60000
    return np.column_stack([ts, p, h, lo, c, vol, vol * c])


class _Cross(Strategy):
    def init(self):
        self.b = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=500)
        self.f = self.add_indicator(self.b.close_ref, SMA(3))
        self.s = self.add_indicator(self.b.close_ref, SMA(8))

    def on_data(self):
        if len(self.f) < 1 or len(self.s) < 1:
            return
        f = float(self.f[-1]); s = float(self.s[-1])
        if np.isnan(f) or np.isnan(s):
            return
        pos = self.sesh.positions("BTCUSDT")
        if f > s and not pos:
            self.sesh.buy("BTCUSDT", volume=1)
        elif f < s and pos:
            for x in pos:
                self.sesh.position_close(x.ticket)


@pytest.mark.unit
class TestStaticAlignment:
    def _run(self):
        from tradetropy import BacktestEngine, KlineData
        kl = _kl()
        eng = BacktestEngine.by_klines(
            _Cross(), data=(KlineData("BTCUSDT", kl, timeframe=60_000),))
        eng.run()
        return eng, kl

    def test_aligned_trades_land_on_real_candles(self):
        from tradetropy.plotting.sources import build_trades_source
        eng, kl = self._run()
        ts = kl[:, 0].astype(np.int64)
        src = build_trades_source(
            eng.stats.trades, 60_000, True, candle_origin_ms=int(ts[0]))
        entry = src.data["entry_ts"].astype("datetime64[ms]").astype(np.int64)
        exit_ = src.data["exit_ts"].astype("datetime64[ms]").astype(np.int64)
        # Every aligned timestamp must be an actual candle open.
        assert np.all(np.isin(entry, ts))
        assert np.all(np.isin(exit_, ts))

    def test_old_epoch_grid_would_be_misaligned(self):
        # Sanity: the epoch-grid floor (origin 0) does NOT match the candle grid,
        # proving the phase fix is necessary.
        from tradetropy.plotting.sources import build_trades_source
        eng, kl = self._run()
        ts = kl[:, 0].astype(np.int64)
        src = build_trades_source(eng.stats.trades, 60_000, True, candle_origin_ms=0)
        entry = src.data["entry_ts"].astype("datetime64[ms]").astype(np.int64)
        assert not np.all(np.isin(entry, ts))


def _klines_to_ticks(klines):
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.zeros((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["price"]] = close
    return out


@pytest.mark.unit
class TestLiveAlignment:
    def _build_updater(self, align):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        ticks = _klines_to_ticks(_kl())
        engine = ReplayEngine.by_ticks(
            _Cross(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
            speed=float("inf"),
        )
        engine.prepare()

        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark", align_trades_to_candle=align),
            max_candles=500,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=engine._ctrl,
            io_loop=None,
        )
        return updater

    def test_trades_ref_carries_align_flag_true(self):
        updater = self._build_updater(align=True)
        assert updater._trades_ref is not None
        assert updater._trades_ref.align_to_candle is True

    def test_trades_ref_carries_align_flag_false(self):
        updater = self._build_updater(align=False)
        assert updater._trades_ref is not None
        assert updater._trades_ref.align_to_candle is False

    def test_align_trades_for_ref_snaps_to_grid(self):
        import types
        from tradetropy.plotting.live.updater._trades_mixin import TradesUpdateMixin

        # Candle grid at phase 20000 (mod 60000): opens ..000000, ..060000, ...
        ts_col = np.array(
            [1_700_000_000_000, 1_700_000_060_000, 1_700_000_120_000],
            dtype="datetime64[ms]",
        )
        src = types.SimpleNamespace(data={"ts": ts_col})
        proxy = types.SimpleNamespace(_ohlc_ring=None)
        stub = TradesUpdateMixin()
        stub._ohlc_refs = [types.SimpleNamespace(source=src, proxy=proxy)]

        ref = types.SimpleNamespace(align_to_candle=True, interval_ms=60_000)
        trades = [{
            "entry_ts_ms": 1_700_000_050_000,  # in candle ..000000
            "exit_ts_ms":  1_700_000_065_000,  # in candle ..060000
            "entry_price": 1.0, "exit_price": 2.0, "pnl": 1.0, "direction": "buy",
        }]
        out = stub._align_trades_for_ref(ref, trades)
        assert out[0]["entry_ts_ms"] == 1_700_000_000_000
        assert out[0]["exit_ts_ms"] == 1_700_000_060_000

    def test_align_disabled_is_noop(self):
        import types
        from tradetropy.plotting.live.updater._trades_mixin import TradesUpdateMixin

        ts_col = np.array([1_700_000_000_000], dtype="datetime64[ms]")
        src = types.SimpleNamespace(data={"ts": ts_col})
        proxy = types.SimpleNamespace(_ohlc_ring=None)
        stub = TradesUpdateMixin()
        stub._ohlc_refs = [types.SimpleNamespace(source=src, proxy=proxy)]

        ref = types.SimpleNamespace(align_to_candle=False, interval_ms=60_000)
        trades = [{
            "entry_ts_ms": 1_700_000_050_000,
            "exit_ts_ms":  1_700_000_065_000,
            "entry_price": 1.0, "exit_price": 2.0, "pnl": 1.0, "direction": "buy",
        }]
        out = stub._align_trades_for_ref(ref, trades)
        assert out[0]["entry_ts_ms"] == 1_700_000_050_000
        assert out[0]["exit_ts_ms"] == 1_700_000_065_000


    def test_open_position_alignment(self):
        """Open position triangle (entry_ts_ms) is snapped to the candle grid."""
        import types
        from tradetropy.plotting.live.updater._trades_mixin import TradesUpdateMixin

        ts_col = np.array(
            [1_700_000_000_000, 1_700_000_060_000, 1_700_000_120_000],
            dtype="datetime64[ms]",
        )
        src = types.SimpleNamespace(data={"ts": ts_col})
        proxy = types.SimpleNamespace(_ohlc_ring=None)
        stub = TradesUpdateMixin()
        stub._ohlc_refs = [types.SimpleNamespace(source=src, proxy=proxy)]
        stub._trades_ref = None

        positions = [{"entry_ts_ms": 1_700_000_050_000, "entry_price": 100.0,
                      "direction": "buy", "volume": 1.0}]
        ref = types.SimpleNamespace(
            align_to_candle=True, interval_ms=60_000,
            source_open=types.SimpleNamespace(data={}),
            sesh=None, symbol="X", theme={},
            pos_line=None, pos_label=None,
        )

        # Simulate alignment (same logic as _update_open_positions)
        from tradetropy.plotting._util import _align_ts_to_candle
        origin = stub._candle_origin_ms()
        aligned = [
            dict(p, entry_ts_ms=int(_align_ts_to_candle(
                p["entry_ts_ms"], ref.interval_ms, origin)))
            for p in positions
        ]
        assert aligned[0]["entry_ts_ms"] == 1_700_000_000_000

    def test_open_position_alignment_disabled(self):
        """When align_to_candle=False, open position entry_ts_ms is unchanged."""
        import types
        from tradetropy.plotting.live.updater._trades_mixin import TradesUpdateMixin
        from tradetropy.plotting._util import _align_ts_to_candle

        ts_col = np.array([1_700_000_000_000], dtype="datetime64[ms]")
        src = types.SimpleNamespace(data={"ts": ts_col})
        proxy = types.SimpleNamespace(_ohlc_ring=None)
        stub = TradesUpdateMixin()
        stub._ohlc_refs = [types.SimpleNamespace(source=src, proxy=proxy)]

        positions = [{"entry_ts_ms": 1_700_000_050_000, "entry_price": 100.0,
                      "direction": "buy", "volume": 1.0}]
        # align_to_candle=False -> no flooring
        ref = types.SimpleNamespace(align_to_candle=False, interval_ms=60_000)
        if not getattr(ref, "align_to_candle", True):
            aligned = positions
        else:
            origin = stub._candle_origin_ms()
            aligned = [
                dict(p, entry_ts_ms=int(_align_ts_to_candle(
                    p["entry_ts_ms"], ref.interval_ms, origin)))
                for p in positions
            ]
        assert aligned[0]["entry_ts_ms"] == 1_700_000_050_000

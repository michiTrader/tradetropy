"""
Tests for ReplayEngine in-place rewind (restart / step-back).

Coverage:
  - restart() rewinds the cursor to the start (before the first live tick),
    resets the broker (balance, positions, deals) and clears the rings down to
    the warmup region, WITHOUT recreating proxies/rings/broker (object identity
    preserved so an attached chart keeps working).
  - A rewind that replays up to a cursor reproduces EXACTLY the same engine
    state as feeding those ticks live (determinism / path independence).
  - step_back(k) lands the cursor k units back and yields the same state as a
    direct rewind to that cursor.
  - build_live_document() wires the replay controls (finish + rewind hooks) and
    builds a layout with the controls above the stats div.

These run headless (no Bokeh server, no browser).
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS, _OHLC_COL
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine
from tradetropy.ta import SMA
from tradetropy.ta.volatility import BollingerBands
from tradetropy.live.engine import LiveEngine


def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    """Convert klines [N x 6/7] to a tick matrix [N x 7] using close as price."""
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.zeros((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["flags"]] = 0.0
    out[:, _TICK_COL["volume_real"]] = 0.0
    out[:, _TICK_COL["price"]] = close
    return out


class _CrossStrategy(Strategy):
    """SMA(3)/SMA(8) crossover that opens/closes a single long position."""

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=2000)
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=2000)
        self.fast = self.add_indicator(self.ohlc.close_ref, SMA(3))
        self.slow = self.add_indicator(self.ohlc.close_ref, SMA(8))

    def on_data(self):
        if len(self.fast) < 1 or len(self.slow) < 1:
            return
        f = float(self.fast[-1])
        s = float(self.slow[-1])
        if np.isnan(f) or np.isnan(s):
            return
        pos = self.sesh.positions("BTCUSDT")
        if f > s and not pos:
            self.sesh.buy("BTCUSDT", volume=1)
        elif f < s and pos:
            for p in pos:
                self.sesh.position_close(p.ticket)


@pytest.fixture
def ticks(klines_20k):
    # 400 candles is plenty for several SMA crossings and trades while staying
    # fast for a unit test (each rewind re-feeds the whole range).
    return klines_to_ticks(klines_20k[:400])


def _build_engine(ticks):
    engine = ReplayEngine.by_ticks(
        _CrossStrategy(),
        data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"),
    )
    engine.prepare()
    return engine


def _state(engine):
    """Snapshot the engine state that a rewind must reproduce deterministically."""
    sym = "BTCUSDT"
    ring = engine.strategy.ohlc._ohlc_ring
    acc = engine._sesh.account_info()
    last_close = (
        float(ring.current_partial_candle[_OHLC_COL["close"]])
        if ring._ts_current_candle >= 0
        else np.nan
    )
    return {
        "cursor": engine._sesh._cursor[sym],
        "n_closed": ring._n_closed,
        "balance": round(acc.balance, 6),
        "n_deals": len(engine._sesh.deals()),
        "last_close": round(last_close, 6) if not np.isnan(last_close) else None,
    }


def _play_live(engine, end_idx):
    """
    Feed live ticks [n_warmup .. end_idx] exactly as the feed loop would.

    Mirrors ReplayEngine._loop_tick: advance the cursor and apply each tick with
    the broker. This is the canonical "play" that a rewind must reproduce.
    """
    sym = "BTCUSDT"
    ds = engine._sesh._datasets[sym]
    n_warmup = engine._sesh._warmup_ticks[sym]
    for i in range(n_warmup, end_idx + 1):
        engine._sesh._cursor[sym] = i
        engine._apply_tick(sym, ds[i], with_broker=True)


@pytest.mark.unit
class TestReplayRewind:
    def test_restart_resets_cursor_broker_and_rings(self, ticks):
        engine = _build_engine(ticks)
        sym = "BTCUSDT"
        n_warmup = engine._sesh._warmup_ticks[sym]
        warmup_closed = engine._historical_candles

        # Object identities that MUST survive a rewind (chart binds to these).
        ring_id = id(engine.strategy.ohlc._ohlc_ring)
        broker_id = id(engine._sesh._broker)
        proxy_id = id(engine.strategy.ohlc)

        # Play to the end so there are trades and many candles.
        _play_live(engine, len(ticks) - 1)
        played = _state(engine)
        assert played["n_deals"] > 0, "test data should produce trades"
        assert played["n_closed"] > warmup_closed

        # Restart -> back to the very start.
        engine._rebuild_restart()

        assert engine._sesh._cursor[sym] == n_warmup - 1
        assert engine._sesh.account_info().balance == pytest.approx(10000.0)
        assert len(engine._sesh.deals()) == 0
        assert len(engine._sesh.positions("BTCUSDT")) == 0
        assert engine.strategy.ohlc._ohlc_ring._n_closed == warmup_closed

        # Identity preserved (no recreation).
        assert id(engine.strategy.ohlc._ohlc_ring) == ring_id
        assert id(engine._sesh._broker) == broker_id
        assert id(engine.strategy.ohlc) == proxy_id

    def test_rewind_to_end_reproduces_live_state(self, ticks):
        # Canonical: feed live tick-by-tick.
        canon = _build_engine(ticks)
        _play_live(canon, len(ticks) - 1)
        expected = _state(canon)

        # Rewind path: a fresh engine rebuilt straight to the end.
        engine = _build_engine(ticks)
        targets = {"BTCUSDT": len(ticks) - 1}
        engine._rebuild_to_cursor(targets)
        got = _state(engine)

        assert got == expected

    def test_step_back_matches_direct_rewind(self, ticks):
        engine = _build_engine(ticks)
        end = len(ticks) - 1
        engine._rebuild_to_cursor({"BTCUSDT": end})

        k = 20
        engine._rebuild_step_back(k)
        after_back = _state(engine)
        assert after_back["cursor"] == end - k

        # A direct rewind to the same cursor must yield identical state.
        ref = _build_engine(ticks)
        ref._rebuild_to_cursor({"BTCUSDT": end - k})
        assert after_back == _state(ref)

    def test_restart_then_replay_is_deterministic(self, ticks):
        engine = _build_engine(ticks)
        end = len(ticks) - 1
        engine._rebuild_to_cursor({"BTCUSDT": end})
        first = _state(engine)

        # Full round trip: restart, then replay to the end again.
        engine._rebuild_restart()
        engine._rebuild_to_cursor({"BTCUSDT": end})
        second = _state(engine)

        assert first == second


@pytest.mark.unit
class TestReplayControlsDocument:
    def test_build_document_wires_controls_and_layout(self, ticks):
        from bokeh.document import Document
        from bokeh.layouts import Column
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        engine = _build_engine(ticks)
        controller = engine._ctrl

        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark"),
            max_candles=200,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=controller,
            io_loop=None,
        )

        # The controller's finish + rewind hooks were wired by the controls.
        assert controller._on_finished is not None
        assert controller._on_seek_done is not None
        assert controller._engine is engine

        # A layout root exists; the top of the document is a column whose first
        # child is the header column (replay bar above the stats div).
        assert len(doc.roots) == 1
        root = doc.roots[0]
        assert isinstance(root, Column)
        header = root.children[0]
        assert isinstance(header, Column)
        # Header has at least the replay bar; with stats enabled it has 2 rows
        # (replay bar first, stats div second).
        assert len(header.children) >= 1


def _max_indicator_ts(updater) -> int:
    """Largest timestamp (ms) present in any indicator band source."""
    mx = -1
    for ref in updater._indicator_refs:
        for band in ref.bands:
            if band.is_ts_band:
                continue
            ts = band.source.data.get("ts", [])
            if len(ts) > 0:
                last = int(np.datetime64(ts[-1], "ms").astype(np.int64))
                mx = max(mx, last)
    return mx


@pytest.mark.unit
class TestIndicatorLookAheadCleanup:
    """
    Going backward in time must strip out future indicator plots (no look-ahead
    bias). After a Step Back / Restart the chart sources are repopulated from the
    rewound (causal) rings and then truncated to ts <= the engine clock.
    """

    def _build_doc(self, ticks):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        engine = _build_engine(ticks)
        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark"),
            max_candles=2000,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=engine._ctrl,
            io_loop=None,
        )
        updater.populate_history(max_age_minutes=0)
        return engine, updater

    def test_step_back_strips_future_indicator_points(self, ticks):
        engine, updater = self._build_doc(ticks)
        end = len(ticks) - 1

        # Play to the end: indicators span the whole range.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        updater.repopulate_after_topup()
        full_max = _max_indicator_ts(updater)
        full_engine_ts = updater._engine_ts()
        assert full_max <= full_engine_ts            # invariant holds at the end
        assert full_max > 0

        # Step back a long way: the engine clock moves into the past.
        engine._rebuild_step_back(200)
        updater.repopulate_after_topup()
        back_engine_ts = updater._engine_ts()
        back_max = _max_indicator_ts(updater)

        # The engine clock actually moved back.
        assert back_engine_ts < full_engine_ts
        # No indicator point survives past the (earlier) engine clock.
        assert back_max <= back_engine_ts
        # Future indicator plots were actually stripped.
        assert back_max < full_max

    def test_restart_clears_to_warmup_boundary(self, ticks):
        engine, updater = self._build_doc(ticks)
        end = len(ticks) - 1
        engine._rebuild_to_cursor({"BTCUSDT": end})
        updater.repopulate_after_topup()

        engine._rebuild_restart()
        updater.repopulate_after_topup()

        eng_ts = updater._engine_ts()
        # OHLC source must not carry any candle past the engine clock.
        ohlc_ts = updater._ohlc_refs[0].source.data.get("ts", [])
        if len(ohlc_ts) > 0:
            last = int(np.datetime64(ohlc_ts[-1], "ms").astype(np.int64))
            assert last <= eng_ts
        # And no future indicator points either.
        assert _max_indicator_ts(updater) <= eng_ts


class _BBTickStrategy(Strategy):
    """Multi-band indicator (Bollinger Bands) mounted on the tick price."""

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=2000)
        self.bb = self.add_indicator(self.tp.price_ref, BollingerBands(5, 2.0))
        self.upper = []
        self.mid = []
        self.lower = []

    def on_data(self):
        self.upper.append(float(self.bb.upper[-1]))
        self.mid.append(float(self.bb.mid[-1]))
        self.lower.append(float(self.bb.lower[-1]))


@pytest.mark.unit
class TestMultiBandTickIndicator:
    """
    Regression: a multi-band indicator on a tick stream used to crash the tick
    hot-path with 'only 0-dimensional arrays can be converted to Python scalars'
    because calculate() returns [K x N] and the code did float(result[-1]).
    """

    def test_live_feed_multiband_tick_matches_bulk(self, ticks):
        engine = LiveEngine.by_ticks(_BBTickStrategy())
        engine.prepare()
        for row in ticks:
            engine.on_tick("BTCUSDT", row)  # must not raise

        prices = ticks[:, _TICK_COL["price"]]
        bulk = BollingerBands(5, 2.0).calculate(prices)  # [3 x N]
        mask = ~np.isnan(bulk[1])

        # Incremental windowed compute vs whole-series bulk differ only by
        # floating-point accumulation (rolling cumsum over L points vs over N);
        # compare with a relative tolerance, not bit-for-bit.
        np.testing.assert_allclose(
            np.array(engine.strategy.upper)[mask], bulk[0][mask], rtol=1e-6)
        np.testing.assert_allclose(
            np.array(engine.strategy.mid)[mask], bulk[1][mask], rtol=1e-6)
        np.testing.assert_allclose(
            np.array(engine.strategy.lower)[mask], bulk[2][mask], rtol=1e-6)

    def test_replay_rewind_with_multiband_tick(self, ticks):
        engine = ReplayEngine.by_ticks(
            _BBTickStrategy(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
            speed=float("inf"),
        )
        engine.prepare()
        end = len(ticks) - 1

        # Rewind to the end (must not raise), then restart and replay again:
        # the final band values must be identical (determinism).
        engine._rebuild_to_cursor({"BTCUSDT": end})
        last_upper_1 = float(engine.strategy.bb.upper[-1])

        engine._rebuild_restart()
        engine._rebuild_to_cursor({"BTCUSDT": end})
        last_upper_2 = float(engine.strategy.bb.upper[-1])

        assert last_upper_1 == pytest.approx(last_upper_2)
        assert not np.isnan(last_upper_2)


class _FpStrategy(Strategy):
    """Ticks + OHLC + footprint, so the OHLC source carries fp_* hover columns."""

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=2000)
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=2000)
        self.fp = self.subscribe_footprint(
            "BTCUSDT", 60_000, window_size=2000, tick_size=1.0,
        )

    def on_data(self):
        pass


@pytest.mark.unit
class TestResyncDocumentLock:
    """
    Regression: a rewind resync must mutate Bokeh sources under the document
    lock. Scheduling via io_loop.add_callback ran without the lock, raising
    '_pending_writes should be non-None ...' half-way through the repopulate and
    leaving the OHLC source without its fp_* columns -> the next stream failed
    with 'Must stream updates to all existing columns (extra: fp_ask, ...)'.
    """

    def test_resync_schedules_via_document_next_tick(self, ticks):
        engine = _build_engine(ticks)

        calls = {"doc": 0, "io": 0}

        class _FakeDoc:
            def add_next_tick_callback(self, cb):
                calls["doc"] += 1

        class _FakeIoLoop:
            def add_callback(self, cb):
                calls["io"] += 1

        class _FakeUpdater:
            def repopulate_after_topup(self):
                pass

        class _FakeChart:
            _updater = _FakeUpdater()
            _doc = _FakeDoc()
            _io_loop = _FakeIoLoop()

        engine._chart = _FakeChart()
        engine._resync_chart_sources()

        # Must go through the document (which holds the lock), never the bare
        # IOLoop.
        assert calls["doc"] == 1
        assert calls["io"] == 0

    def test_repopulate_keeps_fp_columns(self):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document
        from tradetropy.plotting.live.updater._ohlc_mixin import _FP_COL_NAMES

        # Footprint replay with enough ticks to close several candles.
        fp_ticks = klines_to_ticks(
            np.column_stack([
                1_700_000_000_000 + np.arange(200) * 15_000,  # 4 ticks / 1m candle
                np.full(200, 0.0), np.full(200, 0.0), np.full(200, 0.0),
                50_000 + np.cumsum(np.random.default_rng(3).standard_normal(200)),
                np.full(200, 1.0), np.full(200, 0.0),
            ])
        )
        engine = ReplayEngine.by_ticks(
            _FpStrategy(),
            data=(TickData("BTCUSDT", fp_ticks, tick_size=1.0),),
            speed=float("inf"),
        )
        engine.prepare()

        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark"),
            max_candles=500,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=engine._ctrl,
            io_loop=None,
        )
        updater.populate_history(max_age_minutes=0)

        # Play to the end so closed candles + footprint exist.
        _play_live(engine, len(fp_ticks) - 1)

        # The resync repopulate (run directly here, as it would run under the
        # document lock) must NOT drop the fp_* columns from the OHLC source.
        updater.repopulate_after_topup()

        ohlc_src = updater._ohlc_refs[0].source
        for col in _FP_COL_NAMES:
            assert col in ohlc_src.data, f"OHLC source lost column {col}"

        # All OHLC columns share the same length -> a stream over them is valid.
        lengths = {len(v) for v in ohlc_src.data.values()}
        assert len(lengths) == 1, f"ragged OHLC source columns: {lengths}"




@pytest.mark.unit
class TestTradesDisplayAfterRestart:
    """
    Regression: after a Restart the broker is reset to zero trades. The trades
    overlay's change-detection counters (_n_known_trades / _last_deal_ts_ms)
    MUST be reset too. Otherwise the deterministic replay re-creates the
    identical trade set and _update_trades' dedup
    (n_new == _n_known_trades and latest_ts == _last_deal_ts_ms) suppresses the
    refresh permanently, so executed trades never re-appear on the chart even
    though the engine ran them correctly.
    """

    def _build_doc(self, ticks):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        engine = _build_engine(ticks)
        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark", align_trades_to_candle=True),
            max_candles=2000,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=engine._ctrl,
            io_loop=None,
        )
        updater._history_loaded = True
        return engine, updater

    @staticmethod
    def _n_trade_rows(ref) -> int:
        data = ref.source.data
        return len(data.get("lines_xs", [])) if data else 0

    def test_restart_resets_trades_overlay_and_replay_redisplays(self, ticks):
        engine, updater = self._build_doc(ticks)
        ref = updater._trades_ref
        assert ref is not None
        end = len(ticks) - 1

        # Play to the end: broker accumulates trades, overlay shows them.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        updater._populate_trades_history(ref)
        n_first = self._n_trade_rows(ref)
        assert n_first > 0, "test data should produce trades"
        assert ref._n_known_trades == n_first

        # Restart -> broker reset to zero trades; the overlay (and counters)
        # MUST reset, not freeze at the pre-restart values.
        engine._rebuild_restart()
        assert len(engine._sesh.deals()) == 0
        updater._populate_trades_history(ref)
        assert self._n_trade_rows(ref) == 0
        assert ref._n_known_trades == 0
        assert ref._last_deal_ts_ms == 0

        # Replay forward to the same end: the identical trade set re-accumulates.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        expected = len(updater._fetch_closed_trades(ref))
        assert expected == n_first

        # A poll must now re-display the trades (dedup not suppressed by stale
        # counters). Bypass the poll-interval throttle.
        ref._last_check_s = 0.0
        updater._update_trades(ref)
        assert self._n_trade_rows(ref) == expected
        assert ref._n_known_trades == expected



class _HoldStrategy(Strategy):
    """Opens a single long with SL/TP after a few bars and holds it open."""

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=2000)
        self.ohlc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=2000)
        self._opened = False
        self._bars = 0

    def on_data(self):
        if self._opened:
            return
        self._bars += 1
        # Wait a few bars so the broker has a valid price, then buy and hold.
        if self._bars < 10:
            return
        if not self.sesh.positions("BTCUSDT"):
            px = float(self.ohlc.close[-1])
            if np.isnan(px) or px <= 0:
                return
            # Brackets well clear of price so the position stays open.
            r = self.sesh.buy("BTCUSDT", volume=1, sl=px * 0.5, tp=px * 2.0)
            if getattr(r, "retcode", 0) == 10009 or getattr(r, "deal", 0):
                self._opened = True


@pytest.mark.unit
class TestOpenPositionDisplayAfterRestart:
    """
    Regression: after a Restart the open-position overlay (entry markers, the
    average-price line/label and the TP/SL lines) must be cleared to a flat
    state, then re-created as the replay advances and the position re-opens.
    Without the reset the pre-restart position glyphs survive the rewind and the
    reset is not a true 'start from scratch'.
    """

    def _build_doc(self, ticks):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        engine = ReplayEngine.by_ticks(
            _HoldStrategy(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
            speed=float("inf"),
        )
        engine.prepare()
        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark", align_trades_to_candle=True),
            max_candles=2000,
            broker=engine._sesh._broker,
            doc=doc,
            replay_controller=engine._ctrl,
            io_loop=None,
        )
        updater._history_loaded = True
        return engine, updater

    @staticmethod
    def _n_open_rows(ref) -> int:
        data = ref.source_open.data
        return len(data.get("ts", [])) if data else 0

    def test_restart_clears_open_position_then_replay_redisplays(self, ticks):
        engine, updater = self._build_doc(ticks)
        ref = updater._trades_ref
        assert ref is not None
        end = len(ticks) - 1

        # Play to the end: a position is open and the overlay shows it.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        assert len(engine._sesh.positions("BTCUSDT")) > 0, "should hold a position"
        ref._last_check_s = 0.0
        updater._update_open_positions(ref)
        assert self._n_open_rows(ref) > 0
        assert ref.pos_line is None or ref.pos_line.visible is True

        # Restart -> the rewind repopulate must clear the open overlay flat.
        engine._rebuild_restart()
        assert len(engine._sesh.positions("BTCUSDT")) == 0
        updater.repopulate_after_topup()
        assert self._n_open_rows(ref) == 0
        if ref.pos_line is not None:
            assert ref.pos_line.visible is False
        if ref.pos_label is not None:
            assert ref.pos_label.visible is False

        # Replay forward again: the position re-opens and the overlay re-appears.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        assert len(engine._sesh.positions("BTCUSDT")) > 0
        ref._last_check_s = 0.0
        updater._update_open_positions(ref)
        assert self._n_open_rows(ref) > 0

    def test_tpsl_lines_populate_clear_and_reset(self, ticks):
        engine, updater = self._build_doc(ticks)
        ref = updater._trades_ref
        assert ref is not None and ref.tpsl_source is not None
        end = len(ticks) - 1

        # Play to the end: a position with SL/TP is open -> two TP/SL lines.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        pos = engine._sesh.positions("BTCUSDT")[0]
        ref._last_check_s = 0.0
        updater._update_open_positions(ref)
        ys = list(np.asarray(ref.tpsl_source.data.get("y", [])))
        assert len(ys) == 2, "expected one SL line + one TP line"
        assert pos.sl in ys and pos.tp in ys

        # Restart -> the TP/SL lines must clear to empty.
        engine._rebuild_restart()
        updater.repopulate_after_topup()
        assert len(np.asarray(ref.tpsl_source.data.get("y", []))) == 0

        # Replay forward again -> TP/SL lines re-appear.
        engine._rebuild_to_cursor({"BTCUSDT": end})
        ref._last_check_s = 0.0
        updater._update_open_positions(ref)
        assert len(np.asarray(ref.tpsl_source.data.get("y", []))) == 2

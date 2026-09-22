"""
Tests for the manual, forward-only PaperEngine.

Coverage:
  - PaperConfig synthesizes a working no-op strategy (subscriptions +
    indicators) and the engine prepares + feeds forward.
  - Forward-only enforcement: the controller forbids backward navigation and
    _rebuild_step_back raises.
  - Manual orders placed on the session are reflected on the engine; Restart
    wipes the ledger, restores the baseline balance and seeks to index 0.

Headless: no Bokeh server, no browser.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.exceptions import ConfigError
from tradetropy.ta import SMA
from tradetropy.ta.volatility import BollingerBands
from tradetropy.paper import (
    PaperConfig, PaperIndicator, PaperEngine, build_manual_strategy_class,
)
from tradetropy.playback import BaseEngine


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


@pytest.fixture
def ticks(klines_20k):
    return klines_to_ticks(klines_20k[:300])


def _config():
    return PaperConfig(
        symbol="BTCUSDT",
        interval_ms=60_000,
        ticks=True,
        indicators=[
            SMA(5),
            PaperIndicator(SMA(10), source="close", overrides={"color": "#F59E0B"}),
            PaperIndicator(BollingerBands(5, 2.0), source="close"),
        ],
        initial_balance=10_000.0,
    )


def _build(ticks, **kw):
    engine = PaperEngine.by_ticks(
        _config(), data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"), **kw,
    )
    engine.prepare()
    return engine


@pytest.mark.unit
class TestPaperEngineBasics:
    def test_is_base_engine_forward_only(self, ticks):
        engine = _build(ticks)
        assert isinstance(engine, BaseEngine)
        assert engine._ALLOW_BACKWARD is False
        assert engine._ctrl.allow_backward is False

    def test_synthesized_strategy_has_subscriptions_and_indicators(self, ticks):
        engine = _build(ticks)
        strat = engine.strategy
        assert strat._ohlc_proxies, "must subscribe OHLC"
        assert strat._tick_proxies, "ticks=True must subscribe ticks"
        # 3 indicator specs: SMA(5), SMA(10), BollingerBands (multi-band).
        assert len(strat._indicator_defs) == 3
        # No-op on_data must not raise.
        strat.on_data()

    def test_feeds_forward(self, ticks):
        engine = _build(ticks)
        sym = "BTCUSDT"
        ds = engine._sesh._datasets[sym]
        n_warmup = engine._sesh._warmup_ticks[sym]
        for i in range(n_warmup, len(ticks)):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)
        ring = engine.strategy.ohlc._ohlc_ring
        assert ring._n_closed > 0


@pytest.mark.unit
class TestForwardOnly:
    def test_rebuild_step_back_raises(self, ticks):
        engine = _build(ticks)
        with pytest.raises(ConfigError):
            engine._rebuild_step_back(1)

    def test_controller_backward_is_blocked(self, ticks):
        engine = _build(ticks)
        ctrl = engine._ctrl
        # Forward-only: backward navigation is disabled.
        assert ctrl.allow_backward is False
        # step_back is a guarded no-op (does not call the raising engine method).
        ctrl.step_back(5)                      # no exception


@pytest.mark.unit
class TestRestartWipesLedger:
    def test_restart_resets_balance_positions_and_cursor(self, ticks):
        engine = _build(ticks)
        sym = "BTCUSDT"
        ds = engine._sesh._datasets[sym]
        n_warmup = engine._sesh._warmup_ticks[sym]

        # Feed forward into the live region, then place a manual order.
        mid = (n_warmup + len(ticks)) // 2
        for i in range(n_warmup, mid + 1):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)
        engine._sesh.buy(sym, volume=1)
        # Feed a few more so the position is marked.
        for i in range(mid + 1, mid + 6):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)

        assert len(engine._sesh.positions(sym)) >= 1

        # Restart: wipe ledger, reset balance baseline, seek to index 0.
        engine._rebuild_restart()

        assert engine._sesh._cursor[sym] == n_warmup - 1
        assert engine._sesh.account_info().balance == pytest.approx(10_000.0)
        assert len(engine._sesh.positions(sym)) == 0
        assert len(engine._sesh.deals()) == 0

    def test_submit_command_routes_order_on_engine(self, ticks):
        engine = _build(ticks)
        sym = "BTCUSDT"
        ds = engine._sesh._datasets[sym]
        n_warmup = engine._sesh._warmup_ticks[sym]
        for i in range(n_warmup, n_warmup + 5):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)

        # Enqueue a buy as the UI would; drain on the engine thread.
        engine.submit_command(lambda eng: eng._sesh.buy(sym, volume=1))
        engine._drain_commands()
        # A market order fills on the next tick(s); feed a few more.
        for i in range(n_warmup + 5, n_warmup + 9):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)
        assert len(engine._sesh.positions(sym)) >= 1


@pytest.mark.unit
class TestPaperDocumentWiring:
    def _build_doc(self, engine):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document
        doc = Document()
        _doc, updater = build_live_document(
            strategy=engine.strategy,
            config=PlotConfig(theme="dark"),
            max_candles=500,
            broker=engine._sesh._broker,
            doc=doc,
            engine=engine,
        )
        return doc, updater

    def test_paper_doc_renders_order_ticket(self, ticks):
        from bokeh.models import Select, Row
        engine = _build(ticks)
        doc, updater = self._build_doc(engine)

        # The paper-trading layout puts the chart + order panel side by side.
        assert isinstance(doc.roots[0], Row)
        # The Order Ticket's Type Select with the 4 order types is present.
        selects = list(doc.select(dict(type=Select)))
        opts = [s.options for s in selects]
        assert ["Market", "Limit", "Stop", "Stop Limit"] in opts
        # Modifier refresh hook wired into the updater.
        assert updater._panel_refresh is not None

    def test_updater_update_refreshes_panel(self, ticks):
        engine = _build(ticks)
        doc, updater = self._build_doc(engine)
        updater.populate_history(max_age_minutes=0)

        sym = "BTCUSDT"
        ds = engine._sesh._datasets[sym]
        n_warmup = engine._sesh._warmup_ticks[sym]
        for i in range(n_warmup, n_warmup + 6):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)
        engine.place_order("buy", "market", volume=1)
        for i in range(n_warmup + 6, n_warmup + 10):
            engine._sesh._cursor[sym] = i
            engine._apply_tick(sym, ds[i], with_broker=True)

        # update() must run the panel refresh without raising and light up the
        # active-position tracker line.
        updater.update(bar_closed=True)
        assert updater._trades_ref.pos_line.visible is True

    def test_replay_doc_has_no_order_panel(self, ticks):
        from bokeh.models import Row
        from tradetropy.replay.engine import ReplayEngine
        from tradetropy.core.data_types import TickData
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document

        # A replay engine (no order ticket) keeps the chart full-width (Column).
        eng = ReplayEngine.by_ticks(
            build_manual_strategy_class(_config())(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
            speed=float("inf"),
        )
        eng.prepare()
        doc = Document()
        build_live_document(
            strategy=eng.strategy, config=PlotConfig(theme="dark"),
            max_candles=500, broker=eng._sesh._broker, doc=doc, engine=eng,
        )
        assert not isinstance(doc.roots[0], Row)

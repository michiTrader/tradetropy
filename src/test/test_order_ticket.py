"""
Tests for manual order routing (place_order) and the UI Order Ticket panel.

  - BaseEngine.place_order routes Market / Limit / Stop / Stop-Limit to the
    session with the correct order type, on the engine thread.
  - The Order Ticket widget reads its fields and enqueues a place_order command
    via submit_command; draining it opens the position / pending order.
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.core.broker import OrderType
from tradetropy.paper import PaperConfig, PaperEngine


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
    return klines_to_ticks(klines_20k[:200])


def _engine(ticks):
    eng = PaperEngine.by_ticks(
        PaperConfig(symbol="BTCUSDT", interval_ms=60_000, ticks=True),
        data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"),
    )
    eng.prepare()
    return eng


def _feed(engine, n):
    sym = "BTCUSDT"
    ds = engine._sesh._datasets[sym]
    n_warmup = engine._sesh._warmup_ticks[sym]
    end = min(n_warmup + n, len(ds))
    for i in range(n_warmup, end):
        engine._sesh._cursor[sym] = i
        engine._apply_tick(sym, ds[i], with_broker=True)
    return float(ds[end - 1][_TICK_COL["price"]])


@pytest.mark.unit
class TestPlaceOrder:
    def test_primary_symbol(self, ticks):
        eng = _engine(ticks)
        assert eng.primary_symbol() == "BTCUSDT"

    def test_market_buy_opens_position(self, ticks):
        eng = _engine(ticks)
        _feed(eng, 5)
        eng.place_order("buy", "market", volume=1)
        _feed(eng, 5)
        assert len(eng._sesh.positions("BTCUSDT")) >= 1

    def test_limit_order_places_pending(self, ticks):
        eng = _engine(ticks)
        last = _feed(eng, 5)
        # A buy limit sits below market.
        eng.place_order("buy", "limit", volume=1, price=last - 50)
        orders = eng._sesh.orders("BTCUSDT")
        assert len(orders) >= 1
        assert orders[0].type == OrderType.ORDER_TYPE_BUY_LIMIT

    def test_stop_order_places_pending_with_explicit_type(self, ticks):
        eng = _engine(ticks)
        last = _feed(eng, 5)
        # Explicit BUY STOP even though price is below market (honor user choice).
        eng.place_order("buy", "stop", volume=1, price=last - 50)
        orders = eng._sesh.orders("BTCUSDT")
        assert any(o.type == OrderType.ORDER_TYPE_BUY_STOP for o in orders)

    def test_stop_limit_uses_price_limit(self, ticks):
        eng = _engine(ticks)
        last = _feed(eng, 5)
        eng.place_order("sell", "stop_limit", volume=1,
                        price=last - 10, limit_price=last - 12)
        orders = eng._sesh.orders("BTCUSDT")
        assert any(o.type == OrderType.ORDER_TYPE_SELL_STOP_LIMIT for o in orders)

    def test_invalid_volume_rejected(self, ticks):
        eng = _engine(ticks)
        _feed(eng, 5)
        with pytest.warns(UserWarning, match="volume must be > 0"):
            assert eng.place_order("buy", "market", volume=0) is None
        assert len(eng._sesh.positions("BTCUSDT")) == 0


@pytest.mark.unit
class TestOrderTicketWidget:
    def test_panel_builds_with_both_sides(self, ticks):
        from tradetropy.plotting.live.order_ticket import build_order_ticket
        eng = _engine(ticks)
        panel, sides = build_order_ticket(eng.submit_command, "BTCUSDT")
        assert "buy" in sides and "sell" in sides
        assert sides["buy"]["type"].options == ["Market", "Limit", "Stop", "Stop Limit"]

    def test_type_change_toggles_price_visibility(self, ticks):
        from tradetropy.plotting.live.order_ticket import build_order_ticket
        eng = _engine(ticks)
        _panel, sides = build_order_ticket(eng.submit_command, "BTCUSDT")
        buy = sides["buy"]
        assert buy["price"].visible is False
        buy["on_type_change"](None, "Market", "Stop Limit")
        assert buy["price"].visible is True
        assert buy["limit"].visible is True
        buy["on_type_change"](None, "Stop Limit", "Market")
        assert buy["price"].visible is False
        assert buy["limit"].visible is False

    def test_buy_button_enqueues_and_opens_position(self, ticks):
        from tradetropy.plotting.live.order_ticket import build_order_ticket
        eng = _engine(ticks)
        _feed(eng, 5)
        _panel, sides = build_order_ticket(eng.submit_command, "BTCUSDT")
        buy = sides["buy"]
        buy["size"].value = 1.0
        buy["type"].value = "Market"
        # Simulate the button click (runs on the IOLoop -> only enqueues).
        buy["on_submit"]()
        assert len(eng._sesh.positions("BTCUSDT")) == 0   # not executed yet
        eng._drain_commands()                              # engine thread
        _feed(eng, 5)
        assert len(eng._sesh.positions("BTCUSDT")) >= 1


@pytest.mark.unit
class TestPositionModifier:
    def _open_long(self, eng):
        _feed(eng, 5)
        eng.place_order("buy", "market", volume=2)
        _feed(eng, 5)

    def test_net_position_flat_then_long(self, ticks):
        eng = _engine(ticks)
        _feed(eng, 5)
        assert eng.net_position("BTCUSDT") is None
        self._open_long(eng)
        pos = eng.net_position("BTCUSDT")
        assert pos is not None
        assert pos["side"] == "buy"
        assert pos["size"] == pytest.approx(2.0)
        assert pos["avg_price"] > 0

    def test_modifier_refresh_reflects_position(self, ticks):
        from tradetropy.plotting.live.order_ticket import build_position_modifier
        eng = _engine(ticks)
        _panel, w = build_position_modifier(
            eng.submit_command, "BTCUSDT", lambda: eng.net_position("BTCUSDT")
        )
        # Flat initially.
        assert w["btn_all"].disabled is True
        assert "Flat" in w["info"].text
        # Open a position and refresh.
        self._open_long(eng)
        w["refresh"]()
        assert w["btn_all"].disabled is False
        assert "BUY" in w["info"].text

    def test_set_tp_sl_routes_to_engine(self, ticks):
        from tradetropy.plotting.live.order_ticket import build_position_modifier
        eng = _engine(ticks)
        self._open_long(eng)
        _panel, w = build_position_modifier(
            eng.submit_command, "BTCUSDT", lambda: eng.net_position("BTCUSDT")
        )
        w["refresh"]()
        pos = eng.net_position("BTCUSDT")
        avg = pos["avg_price"]
        w["tp"].value = avg + 100
        w["sl"].value = avg - 100
        w["on_set"]()
        eng._drain_commands()
        live = eng._sesh.positions("BTCUSDT")[0]
        assert live.tp == pytest.approx(avg + 100)
        assert live.sl == pytest.approx(avg - 100)

    def test_close_all_routes_to_engine(self, ticks):
        from tradetropy.plotting.live.order_ticket import build_position_modifier
        eng = _engine(ticks)
        self._open_long(eng)
        _panel, w = build_position_modifier(
            eng.submit_command, "BTCUSDT", lambda: eng.net_position("BTCUSDT")
        )
        w["refresh"]()
        w["on_close_all"]()
        eng._drain_commands()
        _feed(eng, 3)
        assert len(eng._sesh.positions("BTCUSDT")) == 0


@pytest.mark.unit
class TestPositionTrackerLine:
    def _build_doc(self, eng):
        from bokeh.document import Document
        from tradetropy.plotting.config import PlotConfig
        from tradetropy.plotting.live.document import build_live_document
        doc = Document()
        _doc, updater = build_live_document(
            strategy=eng.strategy,
            config=PlotConfig(theme="dark"),
            max_candles=500,
            broker=eng._sesh._broker,
            doc=doc,
        )
        updater.populate_history(max_age_minutes=0)
        return updater

    def test_line_tracks_avg_and_clears_when_flat(self, ticks):
        eng = _engine(ticks)
        updater = self._build_doc(eng)
        ref = updater._trades_ref
        assert ref is not None and ref.pos_line is not None

        # Flat: line hidden.
        updater._update_open_positions(ref)
        assert ref.pos_line.visible is False
        assert ref.pos_label.visible is False

        # Open a long, then refresh the tracker.
        _feed(eng, 5)
        eng.place_order("buy", "market", volume=2)
        _feed(eng, 5)
        updater._update_open_positions(ref)

        assert ref.pos_line.visible is True
        pos = eng.net_position("BTCUSDT")
        assert ref.pos_line.location == pytest.approx(pos["avg_price"], rel=1e-6)
        assert "Avg" in ref.pos_label.text
        assert "BUY" in ref.pos_label.text

        # Close everything: line hidden again.
        eng.place_order("sell", "market", volume=2)   # net flat
        _feed(eng, 5)
        updater._update_open_positions(ref)
        assert ref.pos_line.visible is False

"""Tests for the user-data session hooks (apply_order_event / apply_fill_event)."""

import pytest

from tradetropy.connectors.ccxt import SeshCCXTLive
from tradetropy.core.broker import OrderType, OrderState
from tradetropy.streaming import FillEvent, OrderEvent


class _MiniEx:
    id = "mockex"
    has: dict = {}
    apiKey = None

    def load_markets(self, *a, **k):
        return {}


class _AuthEx(_MiniEx):
    apiKey = "key123"


# =====
# supports_user_stream
# =====
class TestSupportsUserStream:
    def test_unauthenticated_false(self):
        assert SeshCCXTLive(_MiniEx()).supports_user_stream is False

    def test_authenticated_true(self):
        assert SeshCCXTLive(_AuthEx()).supports_user_stream is True

    def test_apikey_via_config(self):
        sesh = SeshCCXTLive(_MiniEx(), config={"apiKey": "k", "secret": "s"})
        assert sesh.supports_user_stream is True


# =====
# apply_order_event -> order caches
# =====
class TestOrderEvents:
    def test_open_order_cached(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_order_event(
            OrderEvent("BTC/USDT", 1, order_id=10, status="open", side=1,
                       price=100.0, volume=2.0)
        )
        assert 10 in sesh._cache_orders
        orders = sesh.orders() if not sesh._user_stream_active else None
        # _user_stream_active True now; orders() in SeshCCXTLive still REST -
        # but the cache itself is populated.
        assert sesh._cache_orders[10]["symbol"] == "BTC/USDT"
        assert sesh._cache_orders[10]["state"] == int(OrderState.ORDER_STATE_PLACED)

    def test_filled_order_moves_to_history(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_order_event(
            OrderEvent("BTC/USDT", 1, order_id=10, status="open", side=1,
                       price=100.0, volume=2.0)
        )
        sesh.apply_order_event(
            OrderEvent("BTC/USDT", 2, order_id=10, status="filled", side=1,
                       price=100.0, volume=2.0, filled=2.0)
        )
        assert 10 not in sesh._cache_orders
        assert 10 in sesh._cache_orders_hist
        hist = sesh.order_history()
        assert any(o.ticket == 10 for o in hist)


# =====
# apply_fill_event -> deals + net positions
# =====
class TestFillEvents:
    def test_fill_appends_deal(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 1, order_id=10, trade_id=5, side=1,
                      price=100.0, volume=2.0, fee=0.1)
        )
        deals = sesh.deals()
        assert len(deals) == 1
        assert deals[0].symbol == "BTC/USDT"
        assert deals[0].volume == 2.0
        assert deals[0].commission == 0.1

    def test_net_position_long(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 1, order_id=1, side=1, price=100.0, volume=2.0)
        )
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 2, order_id=2, side=1, price=110.0, volume=2.0)
        )
        # positions() prefers streamed state once active.
        pos = sesh.positions("BTC/USDT")
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_BUY
        assert pos[0].volume == pytest.approx(4.0)
        assert pos[0].price_open == pytest.approx(105.0)  # vwap of 100 and 110

    def test_partial_close_keeps_avg(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 1, order_id=1, side=1, price=100.0, volume=4.0)
        )
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 2, order_id=2, side=-1, price=120.0, volume=1.0)
        )
        pos = sesh.positions("BTC/USDT")
        assert pos[0].volume == pytest.approx(3.0)
        assert pos[0].price_open == pytest.approx(100.0)  # reduce keeps avg

    def test_full_close_flat(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 1, order_id=1, side=1, price=100.0, volume=2.0)
        )
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 2, order_id=2, side=-1, price=110.0, volume=2.0)
        )
        assert sesh.positions("BTC/USDT") == []

    def test_flip_to_short(self):
        sesh = SeshCCXTLive(_MiniEx())
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 1, order_id=1, side=1, price=100.0, volume=2.0)
        )
        sesh.apply_fill_event(
            FillEvent("BTC/USDT", 2, order_id=2, side=-1, price=110.0, volume=5.0)
        )
        pos = sesh.positions("BTC/USDT")
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_SELL
        assert pos[0].volume == pytest.approx(3.0)
        assert pos[0].price_open == pytest.approx(110.0)  # flip price


# =====
# Base session is a no-op
# =====
class TestBaseNoOp:
    def test_base_hooks_noop(self):
        from tradetropy.connectors.ccxt import SeshCCXTSim
        sesh = SeshCCXTSim(feed_type="tick")
        assert sesh.supports_user_stream is False
        # No-op base implementation must not raise.
        sesh.apply_order_event(OrderEvent("BTC", 1, order_id=1, status="open"))
        sesh.apply_fill_event(FillEvent("BTC", 1, order_id=1))


# =====
# Engine end-to-end: ORDER/FILL events update session state via streaming
# =====
class TestEngineUserStream:
    def test_fills_update_positions_through_engine(self):
        import numpy as np
        from tradetropy.core.broker import AccountInfo, OrderType
        from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
        from tradetropy.live.engine import LiveEngine
        from tradetropy.models.strategy import Strategy
        from tradetropy.session.base import SeshLiveBase
        from tradetropy.streaming import FakeFeed, TradeEvent

        class _UserStreamSesh(SeshLiveBase):
            def __init__(self, events):
                super().__init__()
                self._events = events

            @property
            def supports_streaming(self):
                return True

            @property
            def supports_user_stream(self):
                return True

            def create_feed(self, **kwargs):
                return FakeFeed(self._events)

            def buy(self, *a, **k):
                return None

            def sell(self, *a, **k):
                return None

            def positions(self, symbol=None):
                return self._streamed_positions(symbol)

            def account_info(self):
                return AccountInfo(0.0, 0.0, 0.0, 0.0, 0.0)

        class _Strat(Strategy):
            warmup = 0

            def init(self):
                self.tp = self.subscribe_ticks("BTC/USDT", window_size=50)

            def on_data(self):
                pass

        events = [
            TradeEvent("BTC/USDT", 1_000, 100.0, 1.0, side=1),
            FillEvent("BTC/USDT", 1_001, order_id=1, trade_id=1, side=1,
                      price=100.0, volume=2.0),
            TradeEvent("BTC/USDT", 1_002, 110.0, 1.0, side=1),
            FillEvent("BTC/USDT", 1_003, order_id=2, trade_id=2, side=1,
                      price=110.0, volume=2.0),
            TradeEvent("BTC/USDT", 1_004, 110.0, 1.0, side=1),
        ]
        hist = np.zeros((3, N_TICK_COLS), dtype=np.float64)
        for i in range(3):
            hist[i, _TICK_COL["ts"]] = 1 + i
            for c in ("bid", "ask", "price"):
                hist[i, _TICK_COL[c]] = 100.0
            hist[i, _TICK_COL["volume"]] = 1.0

        sesh = _UserStreamSesh(events)
        engine = LiveEngine.by_ticks(_Strat(), sesh=sesh)
        engine.prepare({"BTC/USDT": hist})
        engine.run(blocking=True)

        # Fills routed to apply_fill_event -> net position 4 @ vwap 105, 2 deals.
        pos = sesh.positions("BTC/USDT")
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_BUY
        assert pos[0].volume == pytest.approx(4.0)
        assert pos[0].price_open == pytest.approx(105.0)
        assert len(sesh.deals()) == 2

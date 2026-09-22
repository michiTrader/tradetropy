"""
Tests for the Bybit connector (SeshBybitLive + SeshBybitSim).

No network: SeshBybitLive receives an injected MockHTTP via the `http` parameter.
SeshBybitSim uses the internal broker and is fed with update_tick/update_bar.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from tradetropy.connectors.bybit import (
    SeshBybitLive,
    SeshBybitSim,
    _MS_TO_BYBIT_INTERVAL,
    _bybit_side_to_order_type,
    _order_type_to_bybit_side,
    _format_order_to_order,
)
from tradetropy.core.broker import (
    OrderType,
    OrderState,
    SymbolConfig,
    TradeResult,
    PositionMode,
)
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.exceptions import ConfigError, TradingError


# ══════════════════════════════════════════════════════════════════════════════
# MOCK HTTP (mimics pybit.unified_trading.HTTP)
# ══════════════════════════════════════════════════════════════════════════════


class MockHTTP:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls: list[tuple] = []
        # configurable responses
        self.positions_list: list = []
        self.wallet: dict = {
            "walletBalance": "10000",
            "equity": "10250",
            "unrealisedPnl": "250",
            "totalPositionIM": "500",
        }
        self.kline_list: list = []
        self.tickers_list: list = []
        self.trade_list: list = []
        self.orders_list: list = []
        self.order_hist_list: list = []
        self.closed_pnl_list: list = []
        self.instruments_list: list = []
        self.place_order_resp: dict = {"retCode": 0, "result": {"orderId": "12345"}}

    def _rec(self, name, kw):
        self.calls.append((name, kw))

    def place_order(self, **kw):
        self._rec("place_order", kw)
        return self.place_order_resp

    def get_positions(self, **kw):
        self._rec("get_positions", kw)
        return {"retCode": 0, "result": {"list": self.positions_list}}

    def get_wallet_balance(self, **kw):
        self._rec("get_wallet_balance", kw)
        return {"retCode": 0, "result": {"list": [{"coin": [self.wallet]}]}}

    def get_kline(self, **kw):
        self._rec("get_kline", kw)
        return {"retCode": 0, "result": {"list": self.kline_list}}

    def get_tickers(self, **kw):
        self._rec("get_tickers", kw)
        return {"retCode": 0, "result": {"list": self.tickers_list}}

    def get_public_trade_history(self, **kw):
        self._rec("get_public_trade_history", kw)
        return {"retCode": 0, "result": {"list": self.trade_list}}

    def get_open_orders(self, **kw):
        self._rec("get_open_orders", kw)
        return {"retCode": 0, "result": {"list": self.orders_list}}

    def get_order_history(self, **kw):
        self._rec("get_order_history", kw)
        return {"retCode": 0, "result": {"list": self.order_hist_list}}

    def get_closed_pnl(self, **kw):
        self._rec("get_closed_pnl", kw)
        return {"retCode": 0, "result": {"list": self.closed_pnl_list}}

    def get_instrument_info(self, **kw):
        self._rec("get_instrument_info", kw)
        return {"retCode": 0, "result": {"list": self.instruments_list}}

    def cancel_order(self, **kw):
        self._rec("cancel_order", kw)
        return {"retCode": 0}

    def cancel_all_orders(self, **kw):
        self._rec("cancel_all_orders", kw)
        return {"retCode": 0}

    def set_trading_stop(self, **kw):
        self._rec("set_trading_stop", kw)
        return {"retCode": 0}

    def set_leverage(self, **kw):
        self._rec("set_leverage", kw)
        return {"retCode": 0}

    def set_margin_mode(self, **kw):
        self._rec("set_margin_mode", kw)
        return {"retCode": 0}


def make_live(mock: MockHTTP, **kw) -> SeshBybitLive:
    """Creates a SeshBybitLive using a pre-configured MockHTTP."""
    return SeshBybitLive(
        api_key="x", api_secret="y", http=lambda **_: mock, **kw
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHelpers:
    def test_interval_map_complete(self):
        assert _MS_TO_BYBIT_INTERVAL[60_000] == "1"
        assert _MS_TO_BYBIT_INTERVAL[300_000] == "5"
        assert _MS_TO_BYBIT_INTERVAL[3_600_000] == "60"
        assert _MS_TO_BYBIT_INTERVAL[86_400_000] == "D"
        assert _MS_TO_BYBIT_INTERVAL[604_800_000] == "W"
        assert _MS_TO_BYBIT_INTERVAL[2_592_000_000] == "M"
        assert len(_MS_TO_BYBIT_INTERVAL) == 13

    def test_side_to_order_type(self):
        assert _bybit_side_to_order_type("Buy") == OrderType.ORDER_TYPE_BUY
        assert _bybit_side_to_order_type("Sell") == OrderType.ORDER_TYPE_SELL
        assert _bybit_side_to_order_type("Buy", is_pending=True) == OrderType.ORDER_TYPE_BUY_LIMIT
        assert _bybit_side_to_order_type("Sell", is_pending=True) == OrderType.ORDER_TYPE_SELL_LIMIT

    def test_order_type_to_side(self):
        assert _order_type_to_bybit_side(OrderType.ORDER_TYPE_BUY) == "Buy"
        assert _order_type_to_bybit_side(OrderType.ORDER_TYPE_BUY_LIMIT) == "Buy"
        assert _order_type_to_bybit_side(OrderType.ORDER_TYPE_SELL) == "Sell"
        assert _order_type_to_bybit_side(OrderType.ORDER_TYPE_SELL_STOP) == "Sell"

    def test_format_order_market(self):
        raw = {
            "orderId": "999", "symbol": "BTCUSDT", "side": "Buy",
            "orderType": "Market", "qty": "0.01", "avgPrice": "65000",
            "orderStatus": "Filled", "createdTime": "1700000000000",
            "orderLinkId": "mi-orden",
        }
        o = _format_order_to_order(raw)
        assert o.symbol == "BTCUSDT"
        assert o.type == OrderType.ORDER_TYPE_BUY
        assert o.volume == 0.01
        assert o.state == OrderState.ORDER_STATE_FILLED
        assert o.comment == "mi-orden"

    def test_format_order_limit_pending(self):
        raw = {
            "orderId": "abc-uuid", "symbol": "ETHUSDT", "side": "Sell",
            "orderType": "Limit", "qty": "1", "price": "3500",
            "orderStatus": "New",
        }
        o = _format_order_to_order(raw)
        assert o.type == OrderType.ORDER_TYPE_SELL_LIMIT
        assert o.price == 3500.0
        assert o.state == OrderState.ORDER_STATE_PLACED
        # non-numeric orderId → deterministic positive ticket hash
        assert o.ticket >= 0


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — LIVE: 4 MANDATORY
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveMandatory:
    def test_buy_market_done(self):
        mock = MockHTTP()
        sesh = make_live(mock)
        r = sesh.buy("BTCUSDT", 0.01)
        assert isinstance(r, TradeResult)
        assert r.retcode == TradeResult.TRADE_RETCODE_DONE
        assert r.order == 12345
        # verify order signature
        po = [c for c in mock.calls if c[0] == "place_order"][-1][1]
        assert po["side"] == "Buy"
        assert po["orderType"] == "Market"
        assert po["qty"] == "0.01"
        assert po["category"] == "linear"

    def test_sell_limit_placed(self):
        mock = MockHTTP()
        sesh = make_live(mock)
        r = sesh.sell("BTCUSDT", 0.01, price=70000.0, sl=72000.0, tp=65000.0)
        assert r.retcode == TradeResult.TRADE_RETCODE_PLACED
        po = [c for c in mock.calls if c[0] == "place_order"][-1][1]
        assert po["side"] == "Sell"
        assert po["orderType"] == "Limit"
        assert po["price"] == "70000.0"
        assert po["stopLoss"] == "72000.0"
        assert po["takeProfit"] == "65000.0"

    def test_buy_error_retcode(self):
        mock = MockHTTP()
        mock.place_order_resp = {"retCode": 110007, "retMsg": "insufficient balance"}
        sesh = make_live(mock)
        r = sesh.buy("BTCUSDT", 999)
        assert r.retcode == TradeResult.TRADE_RETCODE_INVALID
        assert "insufficient" in r.comment

    def test_positions_parsing(self):
        mock = MockHTTP()
        mock.positions_list = [
            {"symbol": "BTCUSDT", "side": "Buy", "size": "0.5",
             "avgPrice": "60000", "unrealisedPnl": "120.5",
             "positionIdx": 0, "stopLoss": "0", "takeProfit": "0"},
            {"symbol": "BTCUSDT", "side": "Sell", "size": "0",  # closed → ignore
             "avgPrice": "0", "positionIdx": 2},
        ]
        sesh = make_live(mock)
        pos = sesh.positions("BTCUSDT")
        assert len(pos) == 1
        assert pos[0].symbol == "BTCUSDT"
        assert pos[0].type == OrderType.ORDER_TYPE_BUY
        assert pos[0].volume == 0.5
        assert pos[0].price_open == 60000.0
        assert pos[0].profit == 120.5

    def test_account_info_parsing(self):
        mock = MockHTTP()
        sesh = make_live(mock)
        acc = sesh.account_info()
        assert acc.balance == 10000.0
        assert acc.equity == 10250.0
        assert acc.profit == 250.0
        assert acc.margin == 500.0
        assert acc.margin_free == 9500.0


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — LIVE: OPTIONAL AND CUSTOM
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveOptionalCustom:
    def _sesh_with_position(self):
        mock = MockHTTP()
        mock.positions_list = [
            {"symbol": "BTCUSDT", "side": "Buy", "size": "0.5",
             "avgPrice": "60000", "unrealisedPnl": "0", "positionIdx": 0},
        ]
        sesh = make_live(mock)
        return mock, sesh

    def test_position_close_reduce_only_opuesto(self):
        mock, sesh = self._sesh_with_position()
        pos = sesh.positions("BTCUSDT")
        ok = sesh.position_close(pos[0].ticket)
        assert ok is True
        po = [c for c in mock.calls if c[0] == "place_order"][-1][1]
        assert po["side"] == "Sell"          # opposite to the Buy position
        assert po["reduceOnly"] is True
        assert po["qty"] == "0.5"

    def test_position_modify_set_trading_stop(self):
        mock, sesh = self._sesh_with_position()
        pos = sesh.positions("BTCUSDT")
        ok = sesh.position_modify(pos[0].ticket, sl=55000.0, tp=70000.0)
        assert ok is True
        ts = [c for c in mock.calls if c[0] == "set_trading_stop"][-1][1]
        assert ts["stopLoss"] == "55000.0"
        assert ts["takeProfit"] == "70000.0"

    def test_close_all(self):
        mock, sesh = self._sesh_with_position()
        res = sesh.close_all("BTCUSDT")
        assert len(res) == 1
        assert res[0]["ok"] is True

    def test_get_last_price(self):
        mock = MockHTTP()
        mock.tickers_list = [{"lastPrice": "63500.5", "bid1Price": "63500", "ask1Price": "63501"}]
        sesh = make_live(mock)
        assert sesh.get_last_price("BTCUSDT") == 63500.5

    def test_get_instrument_info(self):
        mock = MockHTTP()
        mock.instruments_list = [{
            "symbol": "BTCUSDT", "baseCoin": "BTC",
            "lotSizeFilter": {"minOrderQty": "0.001", "maxMktOrderQty": "100", "qtyStep": "0.001"},
            "priceFilter": {"tickSize": "0.1"},
        }]
        sesh = make_live(mock)
        info = sesh.get_instrument_info("BTCUSDT")
        assert info["symbol"] == "BTCUSDT"
        assert info["volume_min"] == 0.001
        assert info["volume_step"] == 0.001
        assert info["tick_step"] == 0.1
        assert info["digits"] == 3
        assert info["currency_base"] == "BTC"

    def test_orders_history(self):
        mock = MockHTTP()
        mock.order_hist_list = [
            {"orderId": "1", "symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit",
             "qty": "0.01", "price": "60000", "orderStatus": "Filled"},
        ]
        sesh = make_live(mock)
        hist = sesh.orders_history("BTCUSDT")
        assert len(hist) == 1
        assert hist[0].state == OrderState.ORDER_STATE_FILLED

    def test_deals_closed_pnl(self):
        mock = MockHTTP()
        mock.closed_pnl_list = [
            {"symbol": "BTCUSDT", "side": "Buy", "qty": "0.01",
             "avgExitPrice": "61000", "closedPnl": "10.0", "fees": "0.5",
             "updatedTime": "1700000000000"},
        ]
        sesh = make_live(mock)
        deals = sesh.deals("BTCUSDT")
        assert len(deals) == 1
        assert deals[0].profit == 10.0
        assert deals[0].commission == 0.5

    def test_set_leverage(self):
        mock = MockHTTP()
        mock.positions_list = [{"symbol": "BTCUSDT", "side": "Buy", "size": "0.1",
                                "avgPrice": "60000", "leverage": "5", "positionIdx": 0}]
        sesh = make_live(mock)
        # leverage actual = 5; requesting 10 must call set_leverage
        ok = sesh.set_leverage("BTCUSDT", 10)
        assert ok is True
        sl = [c for c in mock.calls if c[0] == "set_leverage"][-1][1]
        assert sl["buyLeverage"] == "10"

    def test_set_margin_mode_invalid(self):
        mock = MockHTTP()
        sesh = make_live(mock)
        with pytest.raises(ConfigError):
            sesh.set_margin_mode("inexistente")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — LIVE: DATA PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLiveData:
    def test_fetch_klines_reverses_and_maps(self):
        mock = MockHTTP()
        # Bybit returns newest-first: [start, open, high, low, close, volume, turnover]
        mock.kline_list = [
            ["1700000120000", "102", "103", "101", "102.5", "5", "510"],
            ["1700000060000", "101", "102", "100", "101.5", "4", "406"],
            ["1700000000000", "100", "101", "99", "100.5", "3", "301"],
        ]
        sesh = make_live(mock)
        arr = sesh._fetch_klines_history("BTCUSDT", 60_000, limit=3)
        assert arr.shape == (3, 6)
        # ascending by ts after reversal
        assert arr[0, 0] == 1700000000000
        assert arr[-1, 0] == 1700000120000
        assert arr[0, 1] == 100.0   # open of the oldest
        assert arr[-1, 4] == 102.5  # close of the newest
        gk = [c for c in mock.calls if c[0] == "get_kline"][-1][1]
        assert gk["interval"] == "1"

    def test_fetch_klines_invalid_interval(self):
        mock = MockHTTP()
        sesh = make_live(mock)
        with pytest.raises(ConfigError):
            sesh._fetch_klines_history("BTCUSDT", 12345, limit=3)

    def test_fetch_last_tick(self):
        mock = MockHTTP()
        mock.tickers_list = [{"lastPrice": "63500", "bid1Price": "63499", "ask1Price": "63501"}]
        sesh = make_live(mock)
        t = sesh._fetch_last_tick("BTCUSDT")
        assert t.shape == (N_TICK_COLS,)
        assert t[_TICK_COL["price"]] == 63500.0
        assert t[_TICK_COL["bid"]] == 63499.0
        assert t[_TICK_COL["ask"]] == 63501.0

    def test_fetch_ticks_history_flags(self):
        mock = MockHTTP()
        # newest-first
        mock.trade_list = [
            {"price": "100.2", "size": "2", "side": "Sell", "time": "1700000002000"},
            {"price": "100.1", "size": "1", "side": "Buy", "time": "1700000001000"},
        ]
        sesh = make_live(mock)
        arr = sesh._fetch_ticks_history("BTCUSDT", limit=2)
        assert arr.shape == (2, N_TICK_COLS)
        # reversed → ascending
        assert arr[0, _TICK_COL["ts"]] == 1700000001000
        assert arr[0, _TICK_COL["flags"]] == 1.0    # Buy
        assert arr[1, _TICK_COL["flags"]] == -1.0   # Sell
        assert arr[0, _TICK_COL["price"]] == 100.1


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — SIM (simulated mirror)
# ══════════════════════════════════════════════════════════════════════════════


def _btc_cfg():
    return SymbolConfig(
        name="BTCUSDT",
        tick_size=0.1,
        tick_value=0.1,
        contract_size=1.0,
        digits=1,
        avg_spread=0.0,
        volume_min=0.001,
        volume_max=100.0,
        volume_step=0.001,
    )


@pytest.mark.unit
class TestSim:
    def _sesh_tick_con_precio(self, price=60000.0):
        s = SeshBybitSim(
            "tick",
            initial_balance=10_000.0,
            position_mode=PositionMode.POSITION_MODE_NETTING,
        )
        s.add_symbol(_btc_cfg())
        self._t = [datetime(2024, 1, 1, tzinfo=timezone.utc)]
        s._broker.update_tick("BTCUSDT", self._t[0], price, price)
        self._price = price
        return s

    def _tick(self, s, price=None):
        """Feeds an additional tick (triggers execution of queued market orders)."""
        from datetime import timedelta
        self._t[0] = self._t[0] + timedelta(seconds=1)
        px = price if price is not None else self._price
        s._broker.update_tick("BTCUSDT", self._t[0], px, px)

    def test_get_last_price_sim(self):
        s = self._sesh_tick_con_precio(60000.0)
        assert s.get_last_price("BTCUSDT") == 60000.0

    def test_buy_and_positions_sim(self):
        s = self._sesh_tick_con_precio(60000.0)
        # market in simulator queues and executes on next tick (retcode PLACED)
        r = s.buy("BTCUSDT", 0.01)
        assert r.retcode == TradeResult.TRADE_RETCODE_PLACED
        self._tick(s)  # triggers execution
        pos = s.positions("BTCUSDT")
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_BUY

    def test_close_all_sim(self):
        s = self._sesh_tick_con_precio(60000.0)
        s.buy("BTCUSDT", 0.01)
        self._tick(s)  # opens the position
        assert len(s.positions("BTCUSDT")) == 1
        res = s.close_all("BTCUSDT")
        assert len(res) == 1
        assert res[0]["ok"] is True
        self._tick(s)  # executes the queued close
        assert len(s.positions("BTCUSDT")) == 0

    def test_account_info_sim(self):
        s = self._sesh_tick_con_precio()
        info = s.get_account_info()
        assert info["balance"] == 10_000.0
        assert info["company"] == "Simulator"
        assert info["marketType"] == "linear"

    def test_instruments_info_sim(self):
        s = self._sesh_tick_con_precio()
        info = s.get_instrument_info("BTCUSDT")
        assert info["symbol"] == "BTCUSDT"
        assert info["tick_step"] == 0.1
        assert info["volume_min"] == 0.001

    def test_calculate_profit_sim(self):
        s = self._sesh_tick_con_precio()
        # long 0.1 from 60000 to 61000 with tick_size=0.1, tick_value=0.1 → 1000 ticks * 0.1 * 0.1
        p = s.calculate_profit("BTCUSDT", 0.1, 60000.0, 61000.0, side="Buy")
        assert p == pytest.approx((1000.0 / 0.1) * 0.1 * 0.1)

    def test_set_leverage_and_margin_mode_noop(self):
        s = self._sesh_tick_con_precio()
        assert s.set_leverage("BTCUSDT", 10) is True
        assert s.set_margin_mode("cross") is True

    def test_is_market_open_sim(self):
        s = self._sesh_tick_con_precio()
        assert s.is_market_open("BTCUSDT") is True


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE SYMMETRY: LIVE vs SIM
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_public_method_symmetry():
    """Live and Sim must expose the same custom Bybit methods."""
    methods = [
        "buy", "sell", "positions", "account_info",
        "position_close", "position_modify", "close_all",
        "get_last_price", "calculate_margin", "calculate_profit",
        "get_account_info", "get_instrument_info", "is_market_open",
        "set_leverage", "set_margin_mode", "cancel_all_orders", "set_trading_stop",
    ]
    for m in methods:
        assert hasattr(SeshBybitLive, m), f"SeshBybitLive missing method {m}"
        assert hasattr(SeshBybitSim, m), f"SeshBybitSim missing method {m}"

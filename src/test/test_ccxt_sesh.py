"""
Tests for ccxt sessions (SeshCCXTLive + SeshCCXTSim) and the disclaimer.

No network: SeshCCXTLive receives a MockExchange that mimics ccxt's unified API.
SeshCCXTSim uses the internal broker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from tradetropy.connectors.ccxt import (
    SeshCCXTLive,
    SeshCCXTSim,
    _MS_TO_CCXT_TF,
    _ccxt_side_to_order_type,
)
from tradetropy.connectors import _disclaimer
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
# MOCK EXCHANGE (mimics ccxt's unified API)
# ══════════════════════════════════════════════════════════════════════════════


class MockExchange:
    id = "mockex"

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls: list = []
        self.has = {
            "createOrder": True,
            "fetchPositions": True,
            "fetchBalance": True,
            "fetchOpenOrders": True,
            "fetchClosedOrders": True,
            "fetchMyTrades": True,
            "fetchOHLCV": True,
            "fetchTrades": True,
            "fetchTicker": True,
            "setLeverage": True,
            "loadMarkets": True,
        }
        self.positions_data: list = []
        self.balance_data = {
            "total": {"USDT": 10000.0},
            "free": {"USDT": 9500.0},
            "used": {"USDT": 500.0},
        }
        self.ohlcv_data: list = []
        self.ticker_data = {"last": 61000.0, "bid": 60999.0, "ask": 61001.0,
                            "timestamp": 1700000000000, "baseVolume": 123.0}
        self.trades_data: list = []
        self.open_orders_data: list = []
        self.closed_orders_data: list = []
        self.my_trades_data: list = []
        self.markets_data = {}
        self.create_order_resp = {"id": "1001", "status": "closed",
                                  "amount": 0.01, "price": 0.0}
        self.sandbox = False
        self.demo = False

    def _rec(self, name, kw):
        self.calls.append((name, kw))

    def set_sandbox_mode(self, flag):
        self.sandbox = flag

    def enable_demo_trading(self, flag=True):
        self.demo = flag

    def load_markets(self, *a, **k):
        return self.markets_data

    def market(self, symbol):
        if symbol not in self.markets_data:
            raise ValueError(f"market {symbol} not loaded")
        return self.markets_data[symbol]

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self._rec("create_order", dict(symbol=symbol, type=type, side=side,
                                       amount=amount, price=price, params=params or {}))
        return dict(self.create_order_resp)

    def fetch_positions(self, symbols=None, params=None):
        self._rec("fetch_positions", dict(symbols=symbols))
        return self.positions_data

    def fetch_balance(self, params=None):
        return self.balance_data

    def fetch_open_orders(self, symbol=None, *a, **k):
        return self.open_orders_data

    def fetch_closed_orders(self, symbol=None, *a, **k):
        return self.closed_orders_data

    def fetch_my_trades(self, symbol=None, *a, **k):
        return self.my_trades_data

    def cancel_order(self, id, symbol=None, *a, **k):
        self._rec("cancel_order", dict(id=id, symbol=symbol))
        return {"id": id, "status": "canceled"}

    def fetch_ticker(self, symbol, *a, **k):
        return self.ticker_data

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self._rec("fetch_ohlcv", dict(symbol=symbol, timeframe=timeframe, limit=limit))
        return self.ohlcv_data

    def fetch_trades(self, symbol, since=None, limit=None):
        return self.trades_data

    def set_leverage(self, leverage, symbol=None, *a, **k):
        self._rec("set_leverage", dict(leverage=leverage, symbol=symbol))
        return {}


@pytest.fixture(autouse=True)
def _reset_disclaimer():
    _disclaimer._reset_disclaimer_flag()
    yield
    _disclaimer._reset_disclaimer_flag()


def make_live(mock: MockExchange, **kw) -> SeshCCXTLive:
    return SeshCCXTLive(exchange=mock, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDisclaimer:
    def test_exported_constant(self):
        import tradetropy
        assert "RISK DISCLAIMER" in tradetropy.LIVE_DISCLAIMER
        assert "DEMO" in tradetropy.LIVE_DISCLAIMER

    def test_emits_once(self, caplog):
        _disclaimer._reset_disclaimer_flag()
        with caplog.at_level(logging.WARNING, logger="tradetropy.connectors"):
            _disclaimer.emit_live_disclaimer()
            _disclaimer.emit_live_disclaimer()
            _disclaimer.emit_live_disclaimer()
        warnings = [r for r in caplog.records if "RISK DISCLAIMER" in r.getMessage()]
        assert len(warnings) == 1

    def test_live_logs_disclaimer(self, caplog):
        _disclaimer._reset_disclaimer_flag()
        with caplog.at_level(logging.WARNING, logger="tradetropy.connectors"):
            make_live(MockExchange())
        assert any("RISK DISCLAIMER" in r.getMessage() for r in caplog.records)

# ══════════════════════════════════════════════════════════════════════════════
# SANDBOX / DEMO MODE
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSandboxDemo:
    def test_default_no_sandbox_no_demo(self):
        ex = MockExchange()
        make_live(ex)
        assert ex.sandbox is False
        assert ex.demo is False

    def test_sandbox_enables_testnet(self):
        ex = MockExchange()
        sesh = make_live(ex, sandbox=True)
        assert ex.sandbox is True
        assert ex.demo is False
        assert sesh._sandbox is True
        assert sesh._demo is False

    def test_demo_enables_demo_trading(self):
        ex = MockExchange()
        sesh = make_live(ex, demo=True)
        assert ex.demo is True
        assert ex.sandbox is False
        assert sesh._demo is True
        assert sesh._sandbox is False

    def test_sandbox_and_demo_mutually_exclusive(self, caplog):
        ex = MockExchange()
        with caplog.at_level(logging.WARNING, logger="tradetropy.connectors.ccxt"):
            sesh = make_live(ex, sandbox=True, demo=True)
        # sandbox wins, demo is skipped
        assert ex.sandbox is True
        assert ex.demo is False
        assert sesh._demo is False
        assert any("mutually exclusive" in r.getMessage() for r in caplog.records)

    def test_demo_unsupported_exchange_is_ignored(self, caplog):
        class NoDemoEx(MockExchange):
            enable_demo_trading = None  # exchange lacks the method

        ex = NoDemoEx()
        with caplog.at_level(logging.WARNING, logger="tradetropy.connectors.ccxt"):
            sesh = make_live(ex, demo=True)
        assert sesh._demo is False
        assert any("does not support demo-trading" in r.getMessage()
                   for r in caplog.records)




# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHelpers:
    def test_interval_map(self):
        assert _MS_TO_CCXT_TF[60_000] == "1m"
        assert _MS_TO_CCXT_TF[3_600_000] == "1h"
        assert _MS_TO_CCXT_TF[86_400_000] == "1d"
        assert len(_MS_TO_CCXT_TF) == 13

    def test_side_to_order_type(self):
        assert _ccxt_side_to_order_type("long") == OrderType.ORDER_TYPE_BUY
        assert _ccxt_side_to_order_type("buy") == OrderType.ORDER_TYPE_BUY
        assert _ccxt_side_to_order_type("short") == OrderType.ORDER_TYPE_SELL
        assert _ccxt_side_to_order_type("sell") == OrderType.ORDER_TYPE_SELL


# ══════════════════════════════════════════════════════════════════════════════
# LIVE
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLive:
    def test_buy_market_done(self):
        mock = MockExchange()
        sesh = make_live(mock)
        r = sesh.buy("BTC/USDT", 0.01)
        assert r.retcode == TradeResult.TRADE_RETCODE_DONE
        assert r.order == 1001
        co = [c for c in mock.calls if c[0] == "create_order"][-1][1]
        assert co["side"] == "buy"
        assert co["type"] == "market"
        assert co["amount"] == 0.01

    def test_sell_limit_placed(self):
        mock = MockExchange()
        mock.create_order_resp = {"id": "2002", "status": "open", "amount": 0.01, "price": 70000.0}
        sesh = make_live(mock)
        r = sesh.sell("BTC/USDT", 0.01, price=70000.0, sl=72000.0, tp=65000.0)
        assert r.retcode == TradeResult.TRADE_RETCODE_PLACED
        co = [c for c in mock.calls if c[0] == "create_order"][-1][1]
        assert co["type"] == "limit"
        assert co["params"]["stopLossPrice"] == 72000.0
        assert co["params"]["takeProfitPrice"] == 65000.0

    def test_buy_error(self):
        mock = MockExchange()

        def boom(*a, **k):
            raise RuntimeError("insufficient funds")
        mock.create_order = boom
        sesh = make_live(mock)
        r = sesh.buy("BTC/USDT", 999)
        assert r.retcode == TradeResult.TRADE_RETCODE_INVALID
        assert "insufficient" in r.comment

    def test_positions(self):
        mock = MockExchange()
        mock.positions_data = [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.5,
             "entryPrice": 60000.0, "unrealizedPnl": 120.0, "timestamp": 1700000000000},
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 0},  # closed
        ]
        sesh = make_live(mock)
        pos = sesh.positions()
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_BUY
        assert pos[0].volume == 0.5
        assert pos[0].profit == 120.0

    def test_account_info(self):
        mock = MockExchange()
        sesh = make_live(mock)
        acc = sesh.account_info()
        assert acc.balance == 10000.0
        assert acc.margin_free == 9500.0
        assert acc.margin == 500.0

    def test_position_close(self):
        mock = MockExchange()
        mock.positions_data = [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.5,
             "entryPrice": 60000.0, "unrealizedPnl": 0.0},
        ]
        sesh = make_live(mock)
        pos = sesh.positions()
        ok = sesh.position_close(pos[0].ticket)
        assert ok is True
        co = [c for c in mock.calls if c[0] == "create_order"][-1][1]
        assert co["side"] == "sell"
        assert co["params"]["reduceOnly"] is True

    def test_set_leverage(self):
        mock = MockExchange()
        sesh = make_live(mock)
        assert sesh.set_leverage("BTC/USDT", 5) is True
        sl = [c for c in mock.calls if c[0] == "set_leverage"][-1][1]
        assert sl["leverage"] == 5

    def test_unsupported_capability(self):
        mock = MockExchange()
        mock.has["fetchPositions"] = False
        sesh = make_live(mock)  # sync does not fail because it checks has
        with pytest.raises(TradingError):
            sesh.positions()

    def test_get_instrument_info(self):
        mock = MockExchange()
        mock.markets_data = {
            "BTC/USDT": {
                "symbol": "BTC/USDT", "base": "BTC", "contractSize": 1.0,
                "precision": {"price": 0.1, "amount": 0.001},
                "limits": {"amount": {"min": 0.001, "max": 100.0}},
            }
        }
        sesh = make_live(mock)
        info = sesh.get_instrument_info("BTC/USDT")
        assert info["symbol"] == "BTC/USDT"
        assert info["tick_step"] == 0.1
        assert info["volume_min"] == 0.001
        assert info["currency_base"] == "BTC"


@pytest.mark.unit
class TestLiveData:
    def test_fetch_klines(self):
        mock = MockExchange()
        mock.ohlcv_data = [
            [1700000000000, 100, 101, 99, 100.5, 3],
            [1700000060000, 101, 102, 100, 101.5, 4],
        ]
        sesh = make_live(mock)
        arr = sesh._fetch_klines_history("BTC/USDT", 60_000, limit=2)
        assert arr.shape == (2, 6)
        assert arr[0, 0] == 1700000000000
        assert arr[-1, 4] == 101.5
        fo = [c for c in mock.calls if c[0] == "fetch_ohlcv"][-1][1]
        assert fo["timeframe"] == "1m"

    def test_fetch_klines_invalid_interval(self):
        mock = MockExchange()
        sesh = make_live(mock)
        with pytest.raises(ConfigError):
            sesh._fetch_klines_history("BTC/USDT", 99999, limit=2)

    def test_fetch_last_tick(self):
        mock = MockExchange()
        sesh = make_live(mock)
        t = sesh._fetch_last_tick("BTC/USDT")
        assert t.shape == (N_TICK_COLS,)
        assert t[_TICK_COL["price"]] == 61000.0
        assert t[_TICK_COL["bid"]] == 60999.0

    def test_fetch_ticks_history_flags(self):
        mock = MockExchange()
        mock.trades_data = [
            {"price": 100.1, "amount": 1, "side": "buy", "timestamp": 1700000001000},
            {"price": 100.2, "amount": 2, "side": "sell", "timestamp": 1700000002000},
        ]
        sesh = make_live(mock)
        arr = sesh._fetch_ticks_history("BTC/USDT", limit=2)
        assert arr.shape == (2, N_TICK_COLS)
        assert arr[0, _TICK_COL["flags"]] == 1.0
        assert arr[1, _TICK_COL["flags"]] == -1.0


# ══════════════════════════════════════════════════════════════════════════════
# SIM
# ══════════════════════════════════════════════════════════════════════════════


def _cfg():
    return SymbolConfig(
        name="BTC/USDT", tick_size=0.1, tick_value=0.1, contract_size=1.0,
        digits=1, volume_min=0.001, volume_max=100.0, volume_step=0.001,
    )


@pytest.mark.unit
class TestSim:
    def _sesh(self, price=60000.0):
        s = SeshCCXTSim("tick", initial_balance=10_000.0,
                        position_mode=PositionMode.POSITION_MODE_NETTING)
        s.add_symbol(_cfg())
        self._t = [datetime(2024, 1, 1, tzinfo=timezone.utc)]
        s._broker.update_tick("BTC/USDT", self._t[0], price, price)
        self._price = price
        return s

    def _tick(self, s, price=None):
        self._t[0] = self._t[0] + timedelta(seconds=1)
        px = price if price is not None else self._price
        s._broker.update_tick("BTC/USDT", self._t[0], px, px)

    def test_get_last_price(self):
        s = self._sesh(60000.0)
        assert s.get_last_price("BTC/USDT") == 60000.0

    def test_buy_and_positions(self):
        s = self._sesh(60000.0)
        r = s.buy("BTC/USDT", 0.01)
        assert r.retcode == TradeResult.TRADE_RETCODE_PLACED
        self._tick(s)
        pos = s.positions("BTC/USDT")
        assert len(pos) == 1

    def test_close_all(self):
        s = self._sesh(60000.0)
        s.buy("BTC/USDT", 0.01)
        self._tick(s)
        res = s.close_all("BTC/USDT")
        assert len(res) == 1 and res[0]["ok"] is True
        self._tick(s)
        assert len(s.positions("BTC/USDT")) == 0

    def test_account_info(self):
        s = self._sesh()
        info = s.get_account_info()
        assert info["balance"] == 10_000.0
        assert info["company"] == "Simulator"

    def test_no_logs_disclaimer(self, caplog):
        # Sim must NOT emit the risk disclaimer (not live).
        _disclaimer._reset_disclaimer_flag()
        with caplog.at_level(logging.WARNING, logger="tradetropy.connectors"):
            self._sesh()
        assert not any("RISK DISCLAIMER" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_method_symmetry():
    methods = [
        "buy", "sell", "positions", "account_info",
        "position_close", "close_all", "get_last_price",
        "calculate_margin", "calculate_profit",
        "get_account_info", "get_instrument_info", "is_market_open",
        "set_leverage",
    ]
    for m in methods:
        assert hasattr(SeshCCXTLive, m), f"SeshCCXTLive missing {m}"
        assert hasattr(SeshCCXTSim, m), f"SeshCCXTSim missing {m}"

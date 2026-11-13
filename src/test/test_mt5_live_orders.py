"""Exercise live trade requests with the MT5 SDK's keyword-only calling convention."""

from types import SimpleNamespace
from unittest.mock import Mock

from tradetropy.connectors.mt5 import SeshMT5Live


def test_mt5_live_trade_requests_use_named_fields():
    requests = []

    def order_send(**request):
        requests.append(request)
        return SimpleNamespace(
            retcode=10008 if request["action"] == 5 else 10009,
            order=42,
            deal=0,
            volume=request.get("volume", 0.0),
            price=request.get("price", 0.0),
            comment="",
        )

    session = SeshMT5Live.__new__(SeshMT5Live)
    session._mt5 = SimpleNamespace(
        order_send=order_send,
        symbol_info_tick=lambda symbol: SimpleNamespace(bid=1.10600, ask=1.10602),
        history_deals_get=lambda **kwargs: (),
        TRADE_ACTION_DEAL=1,
        TRADE_ACTION_PENDING=5,
        TRADE_ACTION_SLTP=6,
        TRADE_ACTION_REMOVE=8,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TIME_GTC=0,
        TRADE_RETCODE_PLACED=10008,
        TRADE_RETCODE_DONE=10009,
    )
    session._default_magic = 7
    session._deviation = 20
    session._cache_positions = {}
    session._cache_orders = {}
    session._normalize_volume = Mock(return_value=0.24)
    session._get_filling_mode = Mock(return_value=2)
    session._register_close = Mock()

    pending = session.sell("EURUSD", 0.24, price=1.10605, sl=1.10613, tp=1.10599)
    opened = session.buy("EURUSD", 0.24)
    assert pending.retcode == 10008
    assert opened.retcode == 10009
    assert session.position_modify(opened.order, sl=1.10590, tp=1.10610)
    assert session.position_close(opened.order)
    assert session.order_delete(pending.order)

    assert requests[0] == {
        "action": 5, "symbol": "EURUSD", "volume": 0.24, "type": 3,
        "price": 1.10605, "sl": 1.10613, "tp": 1.10599, "deviation": 20,
        "magic": 7, "comment": "", "type_time": 0, "type_filling": 2,
    }
    assert requests[1]["action"] == 1
    assert requests[1]["type"] == 0
    assert requests[2] == {"action": 6, "position": opened.order, "sl": 1.10590, "tp": 1.10610}
    assert requests[3]["action"] == 1
    assert requests[3]["position"] == opened.order
    assert requests[3]["type"] == 1
    assert requests[4] == {"action": 8, "order": pending.order}

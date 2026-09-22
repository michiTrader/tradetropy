# region test_broker_v3.0
"""
Tests for KlineBroker and TickBroker — Suite v3.0

Migration from v2.2:
- BrokerSimulator removed → KlineBroker and TickBroker
- Fixtures renamed: broker/broker_realistic → bar_broker_toc/kline_broker
- broker_tick / broker_tick_toc unified into tick_broker
  (trade_on_close does not exist in TickBroker — equivalence test removed
   and replaced by direct behavior test)
- No changes to the logic of any test

Usage:
    pytest test_bbroker.py -v
    pytest test_bbroker.py::TestOnTick -v
"""

import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import List

from tradetropy.core.broker import (
    KlineBroker,
    TickBroker,
    SymbolConfig,
    OrderType,
    OrderTypeTime,
    TradeRequest,
    TradeRequestActions,
    CommissionType,
    PositionMode,
    OrderState,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Bar:
    time: str
    open: float
    high: float
    low: float
    close: float

    @property
    def dt(self):
        return datetime.strptime(self.time, "%Y-%m-%d %H:%M:%S")


@dataclass
class Tick:
    time: str
    bid: float
    ask: float

    @property
    def dt(self):
        return datetime.strptime(
            self.time,
            "%Y-%m-%d %H:%M:%S.%f" if "." in self.time else "%Y-%m-%d %H:%M:%S",
        )


def feed_bars(broker, bars: List[Bar], symbol="MES"):
    for bar in bars:
        broker.update_kline(symbol, bar.dt, bar.open, bar.high, bar.low, bar.close)


def feed_ticks(broker, ticks: List[Tick], symbol="MES"):
    for tick in ticks:
        broker.update_tick(symbol, tick.dt, tick.bid, tick.ask)


# ══════════════════════════════════════════════════════════════════════════════
# ORDER HELPERS — BARS
# ══════════════════════════════════════════════════════════════════════════════


def buy(broker, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_DEAL,
            type=OrderType.ORDER_TYPE_BUY,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def sell(broker, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_DEAL,
            type=OrderType.ORDER_TYPE_SELL,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def buy_limit(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_BUY_LIMIT,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def sell_limit(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_SELL_LIMIT,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def sell_stop(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_SELL_STOP,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORDER HELPERS — TICKS
# ══════════════════════════════════════════════════════════════════════════════


def buy_tick(broker, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_DEAL,
            type=OrderType.ORDER_TYPE_BUY,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def sell_tick(broker, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_DEAL,
            type=OrderType.ORDER_TYPE_SELL,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def buy_limit_tick(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_BUY_LIMIT,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def sell_limit_tick(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_SELL_LIMIT,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def buy_stop_tick(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_BUY_STOP,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


def sell_stop_tick(broker, price, volume=1, tp=None, sl=None, symbol="MES"):
    return broker.order_send(
        TradeRequest(
            symbol=symbol,
            action=TradeRequestActions.TRADE_ACTION_PENDING,
            type=OrderType.ORDER_TYPE_SELL_STOP,
            price=price,
            volume=volume,
            tp=tp,
            sl=sl,
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mes():
    return SymbolConfig(
        name="MES",
        tick_size=0.25,
        tick_value=1.25,
        contract_size=1,
        digits=2,
        avg_spread=0.0,
    )


@pytest.fixture
def bar_broker_toc(mes):
    """KlineBroker with trade_on_close=True — immediate execution at close."""
    b = KlineBroker(
        initial_balance=10_000.0,
        commission=0.6,
        use_spread=False,
        trade_on_close=True,
        position_mode=PositionMode.POSITION_MODE_NETTING,
    )
    b.add_symbol(mes)
    return b


@pytest.fixture
def kline_broker(mes):
    """KlineBroker with trade_on_close=False — execution at next open."""
    b = KlineBroker(
        initial_balance=10_000.0,
        commission=0.6,
        use_spread=False,
        trade_on_close=False,
        position_mode=PositionMode.POSITION_MODE_NETTING,
    )
    b.add_symbol(mes)
    return b


@pytest.fixture
def tick_broker(mes):
    """TickBroker — orders always enqueued for the next tick."""
    b = TickBroker(
        initial_balance=10_000.0,
        commission=0.6,
        position_mode=PositionMode.POSITION_MODE_NETTING,
    )
    b.add_symbol(mes)
    return b


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: TRADE ON CLOSE = TRUE  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTradeOnCloseTrue:
    def test_market_buy_executes_immediately(self, bar_broker_toc):
        feed_bars(
            bar_broker_toc,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(bar_broker_toc, volume=1)

        positions = bar_broker_toc.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].price_open == 6920.00
        assert positions[0].volume == 1.0

    def test_sltp_executes_same_bar(self, bar_broker_toc):
        feed_bars(
            bar_broker_toc,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(bar_broker_toc, volume=1, tp=6925.00, sl=6915.00)

        feed_bars(
            bar_broker_toc,
            [
                Bar("2026-02-11 10:01:00", 6922.00, 6926.00, 6921.00, 6925.00),
            ],
        )

        assert len(bar_broker_toc.get_positions("MES")) == 0
        deals = bar_broker_toc.get_deal_history("MES")
        assert len(deals) == 2
        assert "[tp" in deals[1].comment.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: TRADE ON CLOSE = FALSE  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTradeOnCloseFalse:
    def test_market_order_executes_next_bar_open(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        result = buy(kline_broker, volume=1)

        assert result.retcode == result.TRADE_RETCODE_PLACED
        assert len(kline_broker.get_positions("MES")) == 0

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6924.00, 6926.00, 6923.00, 6925.00),
            ],
        )

        positions = kline_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].price_open == 6924.00

    def test_sltp_can_execute_same_bar_as_entry(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1, tp=6925.00, sl=6916.00)

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6922.00, 6926.00, 6920.00, 6925.00),
            ],
        )

        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[0].price == 6922.00
        assert deals[1].price == 6925.00
        assert "[tp" in deals[1].comment.lower()
        assert kline_broker.account_info().balance == pytest.approx(10_013.80, abs=0.01)

    def test_multiple_orders_same_bar_execute_next_bar(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        buy(kline_broker, volume=2)
        buy(kline_broker, volume=1)

        assert len(kline_broker.get_positions("MES")) == 0

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6924.00, 6926.00, 6923.00, 6925.00),
            ],
        )

        positions = kline_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].volume == 4.0
        assert positions[0].price_open == 6924.00

    def test_gap_up_execution(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6930.00, 6932.00, 6929.00, 6931.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].price_open == 6930.00

    def test_gap_down_execution(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        sell(kline_broker, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6910.00, 6912.00, 6908.00, 6911.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].price_open == 6910.00

    def test_pending_order_executes_intrabar(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy_limit(kline_broker, price=6915.00, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6918.00, 6920.00, 6914.00, 6916.00),
            ],
        )
        positions = kline_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].price_open == 6915.00

    def test_realistic_trading_sequence(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1, tp=6928.00, sl=6916.00)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6922.00, 6924.00, 6921.00, 6923.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].price_open == 6922.00

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6924.00, 6929.00, 6923.00, 6928.00),
            ],
        )
        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[1].price == 6928.00
        assert "[tp" in deals[1].comment.lower()
        assert kline_broker.account_info().balance == pytest.approx(10_028.80, abs=0.01)

    def test_order_sent_on_last_bar_never_executes(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        assert len(kline_broker.get_positions("MES")) == 0
        history = kline_broker.get_order_history("MES")
        assert history[0].state == OrderState.ORDER_STATE_PLACED


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: COMPARISON trade_on_close=True vs False  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestModeComparison:
    def test_same_profit_different_timing(self, mes):
        bars = [
            Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            Bar("2026-02-11 10:01:00", 6924.00, 6926.00, 6923.00, 6925.00),
            Bar("2026-02-11 10:02:00", 6925.00, 6928.00, 6924.00, 6927.00),
        ]

        broker_a = KlineBroker(
            initial_balance=10_000.0, commission=0.6, trade_on_close=True
        )
        broker_a.add_symbol(mes)
        broker_b = KlineBroker(
            initial_balance=10_000.0, commission=0.6, trade_on_close=False
        )
        broker_b.add_symbol(mes)

        feed_bars(broker_a, bars[:1])
        feed_bars(broker_b, bars[:1])
        buy(broker_a, volume=1)
        buy(broker_b, volume=1)

        assert len(broker_a.get_positions("MES")) == 1
        assert len(broker_b.get_positions("MES")) == 0

        feed_bars(broker_a, bars[1:2])
        feed_bars(broker_b, bars[1:2])
        assert broker_a.get_positions("MES")[0].price_open == 6920.00
        assert broker_b.get_positions("MES")[0].price_open == 6924.00

        broker_a.position_close(broker_a.get_positions("MES")[0].ticket)
        feed_bars(broker_b, bars[2:3])
        broker_b.position_close(broker_b.get_positions("MES")[0].ticket)

        assert broker_a.account_info().balance > broker_b.account_info().balance


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: EDGE CASES  (KlineBroker trade_on_close=False)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRealisticEdgeCases:
    def test_close_position_before_entry_executes(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        assert kline_broker.position_close(1) == False

    def test_modify_sltp_before_entry(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1, tp=6930.00, sl=6915.00)
        assert len(kline_broker.get_positions("MES")) == 0

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6924.00, 6926.00, 6923.00, 6925.00),
            ],
        )
        positions = kline_broker.get_positions("MES")
        assert len(positions) == 1

        result = kline_broker.position_modify(positions[0].ticket, sl=6920.00, tp=6935.00)
        assert result == True
        assert positions[0].sl == 6920.00
        assert positions[0].tp == 6935.00

    def test_rapid_reversals(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6922.00, 6924.00, 6921.00, 6923.00),
            ],
        )
        sell(kline_broker, volume=2)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6920.00, 6922.00, 6919.00, 6921.00),
            ],
        )
        positions = kline_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].type == OrderType.ORDER_TYPE_SELL
        assert positions[0].volume == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SL/TP WORST CASE  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSLTPWorstCase:
    def test_worst_case_long(self, kline_broker):
        """
    BUY: bar touches SL and TP → SL wins (worst case).
    """
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6918.00, 6922.00, 6916.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1, tp=6927.00, sl=6915.00)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6919.00, 6921.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].price_open == 6920.00

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6921.00, 6928.00, 6914.00, 6920.00),
            ],
        )
        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[1].price == 6915.00
        assert "[sl" in deals[1].comment.lower()
        assert kline_broker.account_info().balance == pytest.approx(9_973.80, abs=0.01)

    def test_worst_case_short(self, kline_broker):
        """
    SELL: bar touches SL and TP → SL wins (worst case).
    """
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6922.00, 6924.00, 6918.00, 6920.00),
            ],
        )
        sell(kline_broker, volume=1, tp=6913.00, sl=6927.00)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6919.00, 6921.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].price_open == 6920.00

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6921.00, 6928.00, 6912.00, 6918.00),
            ],
        )
        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[1].price == 6927.00
        assert "[sl" in deals[1].comment.lower()
        assert kline_broker.account_info().balance == pytest.approx(9_963.80, abs=0.01)

    def test_only_tp_long(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6918.00, 6922.00, 6916.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1, tp=6927.00, sl=6915.00)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6919.00, 6921.00),
            ],
        )
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6921.00, 6928.00, 6919.00, 6925.00),
            ],
        )
        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[1].price == 6927.00
        assert "[tp" in deals[1].comment.lower()

    def test_only_sl_short(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6922.00, 6924.00, 6918.00, 6920.00),
            ],
        )
        sell(kline_broker, volume=1, tp=6913.00, sl=6927.00)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6919.00, 6921.00),
            ],
        )
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6921.00, 6928.00, 6916.00, 6922.00),
            ],
        )
        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[1].price == 6927.00
        assert "[sl" in deals[1].comment.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PENDING ORDERS  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPendingOrdersRealistic:
    def test_limit_execution_same_bar_as_placement(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy_limit(kline_broker, price=6919.00, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        positions = kline_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].price_open == 6919.00

    def test_stop_loss_hit_after_limit_execution(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy_limit(kline_broker, price=6919.00, volume=1, sl=6916.00)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].price_open == 6919.00

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6918.00, 6920.00, 6915.00, 6917.00),
            ],
        )
        assert len(kline_broker.get_positions("MES")) == 0
        deals = kline_broker.get_deal_history("MES")
        assert deals[0].price == 6919.00
        assert deals[1].price == 6916.00


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: COMPLEX SCENARIOS  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestComplexRealisticScenarios:
    def test_full_trading_day_realistic(self, kline_broker):
        bars = [
            Bar("2026-02-11 09:30:00", 6920.00, 6922.00, 6918.00, 6920.00),
            Bar("2026-02-11 09:31:00", 6921.00, 6924.00, 6920.00, 6923.00),
            Bar("2026-02-11 09:32:00", 6923.00, 6928.00, 6922.00, 6927.00),
            Bar("2026-02-11 09:33:00", 6927.00, 6929.00, 6925.00, 6926.00),
            Bar("2026-02-11 09:34:00", 6926.00, 6926.00, 6920.00, 6922.00),
            Bar("2026-02-11 09:35:00", 6922.00, 6924.00, 6920.00, 6923.00),
        ]

        feed_bars(kline_broker, bars[0:1])
        buy(kline_broker, volume=1, tp=6928.00, sl=6918.00)

        feed_bars(kline_broker, bars[1:2])
        assert kline_broker.get_positions("MES")[0].price_open == 6921.00

        feed_bars(kline_broker, bars[2:3])
        assert len(kline_broker.get_positions("MES")) == 0

        sell(kline_broker, volume=2, tp=6920.00, sl=6930.00)
        feed_bars(kline_broker, bars[3:4])
        assert kline_broker.get_positions("MES")[0].type == OrderType.ORDER_TYPE_SELL

        feed_bars(kline_broker, bars[4:5])
        assert len(kline_broker.get_positions("MES")) == 0
        assert kline_broker.account_info().balance > 10_000.0

    def test_complex_trading_session_exact_calculations(self, bar_broker_toc):
        broker = bar_broker_toc

        # ── Op 1: Market BUY with SL hit ──
        feed_bars(
            broker, [Bar("2026-02-02 02:05:00", 6923.75, 6924.00, 6920.75, 6922.00)]
        )
        buy(broker, volume=1, tp=6926.50, sl=6918.75)
        assert broker.get_positions("MES")[0].price_open == 6922.00
        assert broker.get_positions("MES")[0].commission == -0.6
        assert broker.account_info().balance == 9999.4

        feed_bars(
            broker,
            [
                Bar("2026-02-02 02:06:00", 6921.50, 6924.00, 6920.25, 6923.00),
                Bar("2026-02-02 02:07:00", 6921.50, 6922.00, 6913.50, 6914.75),
            ],
        )
        assert len(broker.get_positions("MES")) == 0
        assert broker.account_info().balance == pytest.approx(9982.55, abs=0.01)

        # ── Op 2: Buy Limit with TP ──
        feed_bars(
            broker, [Bar("2026-02-03 16:48:00", 6960.25, 6960.25, 6957.00, 6957.50)]
        )
        buy_limit(broker, price=6956.25, volume=2, tp=6966.75, sl=6947.25)
        feed_bars(
            broker, [Bar("2026-02-03 02:49:00", 6957.50, 6960.55, 6955.50, 6955.50)]
        )
        assert broker.get_positions("MES")[0].price_open == 6956.25
        assert broker.account_info().balance == pytest.approx(9981.35, abs=0.01)

        feed_bars(
            broker, [Bar("2026-02-03 02:50:00", 6955.50, 6955.75, 6950.75, 6952.00)]
        )
        broker.position_close(broker.get_positions("MES")[0].ticket)
        assert broker.account_info().balance == pytest.approx(9_937.65, abs=0.01)

        # ── Op 3: Buy Limit + manual close ──
        feed_bars(
            broker, [Bar("2026-02-03 17:06:00", 6947.50, 6950.25, 6946.25, 6950.25)]
        )
        buy_limit(broker, price=6949.00, volume=2, tp=6953.50, sl=6948.25)
        assert len(broker.get_orders("MES")) == 1

        feed_bars(
            broker, [Bar("2026-02-03 17:07:00", 6950.00, 6950.75, 6948.75, 6948.75)]
        )
        assert broker.get_positions("MES")[0].price_open == 6949.00
        assert broker.account_info().balance == pytest.approx(9_936.45, abs=0.01)

        feed_bars(
            broker, [Bar("2026-02-03 17:08:00", 6948.75, 6951.00, 6948.75, 6951.00)]
        )
        broker.position_close(broker.get_positions("MES")[-1].ticket)
        assert broker.account_info().balance == pytest.approx(9_955.25, abs=0.01)

        # ── Op 4: Multiple + consolidation ──
        feed_bars(
            broker,
            [
                Bar("2026-02-09 07:46:00", 6951.75, 6953.25, 6951.75, 6953.00),
                Bar("2026-02-09 07:47:00", 6953.00, 6953.75, 6952.75, 6953.00),
            ],
        )
        buy(broker, volume=2)
        buy_limit(broker, price=6952.50, volume=2)
        assert broker.account_info().balance == pytest.approx(9_954.05)

        feed_bars(
            broker, [Bar("2026-02-09 07:48:00", 6953.75, 6954.50, 6952.00, 6952.25)]
        )
        result = buy_limit(broker, price=6951.25, volume=3)
        order_to_cancel = result.order

        positions = broker.get_positions("MES")
        assert positions[0].volume == 4.0
        assert positions[0].price_open == 6952.75
        assert positions[0].profit == -10.0
        assert broker.account_info().balance == pytest.approx(9_952.85, abs=0.01)
        assert broker.account_info().equity == pytest.approx(9_942.85, abs=0.01)

        # ── Op 5: Cancel order ──
        feed_bars(
            broker, [Bar("2026-02-09 07:50:00", 6954.00, 6955.50, 6953.75, 6955.50)]
        )
        broker.order_send(
            TradeRequest(
                symbol="MES",
                action=TradeRequestActions.TRADE_ACTION_REMOVE,
                order=order_to_cancel,
            )
        )
        canceled = [
            o for o in broker.get_order_history("MES") if o.ticket == order_to_cancel
        ]
        assert canceled[0].state == OrderState.ORDER_STATE_CANCELED
        assert broker.account_info().balance == pytest.approx(9_952.85, abs=0.01)
        assert broker.account_info().equity == pytest.approx(10_007.85, abs=0.01)

        # ── Op 6: SELL reversal ──
        feed_bars(
            broker, [Bar("2026-02-09 07:51:00", 6955.50, 6955.75, 6954.25, 6954.50)]
        )
        sell(broker, volume=2)
        sell_stop(broker, price=6953.75, volume=2)

        positions = broker.get_positions("MES")
        assert positions[0].type == OrderType.ORDER_TYPE_BUY
        assert positions[0].volume == 2.0
        assert broker.account_info().balance == pytest.approx(9_969.15, abs=0.5)
        assert broker.account_info().equity == pytest.approx(9_986.65, abs=0.5)

        feed_bars(
            broker,
            [
                Bar("2026-02-09 07:55:00", 6954.50, 6954.75, 6952.50, 6953.00),
                Bar("2026-02-10 04:46:00", 6980.75, 6981.00, 6980.75, 6981.00),
            ],
        )
        sell_limit(broker, price=6981.50, volume=3)
        sell_limit(broker, price=6982.00, volume=2, sl=6983.00)
        assert len(broker.get_orders("MES")) == 2

        feed_bars(
            broker,
            [
                Bar("2026-02-10 04:47:00", 6981.00, 6981.75, 6980.75, 6981.50),
                Bar("2026-02-10 04:48:00", 6981.75, 6982.75, 6981.75, 6982.75),
            ],
        )
        assert broker.get_positions("MES")[0].type == OrderType.ORDER_TYPE_SELL

        feed_bars(
            broker,
            [
                Bar("2026-02-10 04:49:00", 6982.75, 6983.25, 6981.75, 6982.25),
                Bar("2026-02-10 08:02:00", 6981.75, 6981.75, 6979.50, 6980.80),
            ],
        )

        deals = broker.get_deal_history("MES")
        close_deals = [d for d in deals if d.profit != 0]
        wins = [d for d in close_deals if d.profit > 0]
        losses = [d for d in close_deals if d.profit < 0]

        assert len(close_deals) == 6
        assert len(wins) == 3
        assert len(losses) == 3
        assert sum(d.profit for d in wins) / len(wins) == pytest.approx(15.83, abs=0.01)
        assert max(d.profit for d in wins) == pytest.approx(20.00, abs=0.01)
        assert sum(d.profit for d in losses) / len(losses) == pytest.approx(
            -30.42, abs=0.01
        )
        assert min(d.profit for d in losses) == pytest.approx(-42.50, abs=0.01)

        total_profit = sum(d.profit for d in close_deals)
        total_commission = sum(d.commission for d in deals)
        assert total_profit == pytest.approx(-43.75, abs=0.01)
        assert total_commission == pytest.approx(-16.80, abs=0.01)
        assert total_profit + total_commission == pytest.approx(-60.55, abs=0.01)

        roi = (
            (broker.account_info().equity - broker.initial_balance)
            / broker.initial_balance
        ) * 100
        assert roi == pytest.approx(-0.61, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: VALIDATIONS  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestValidations:
    def test_invalid_sltp_direction(self, bar_broker_toc):
        feed_bars(
            bar_broker_toc,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        assert (
            buy(bar_broker_toc, volume=1, sl=6925.00).retcode
            == TradeResult_INVALID_STOPS()
        )
        assert (
            buy(bar_broker_toc, volume=1, tp=6915.00).retcode
            == TradeResult_INVALID_STOPS()
        )
        assert (
            sell(bar_broker_toc, volume=1, sl=6915.00).retcode
            == TradeResult_INVALID_STOPS()
        )
        assert (
            sell(bar_broker_toc, volume=1, tp=6925.00).retcode
            == TradeResult_INVALID_STOPS()
        )

    def test_valid_sltp(self, bar_broker_toc):
        feed_bars(
            bar_broker_toc,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        from tradetropy.core.broker import TradeResult

        assert (
            buy(bar_broker_toc, volume=1, tp=6930.00, sl=6910.00).retcode
            == TradeResult.TRADE_RETCODE_DONE
        )

        bar_broker_toc.reset()
        feed_bars(
            bar_broker_toc,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        assert (
            buy(bar_broker_toc, volume=1, tp=6930.00, sl=6910.00).retcode
            == TradeResult.TRADE_RETCODE_DONE
        )
        assert (
            sell(bar_broker_toc, volume=1, tp=6910.00, sl=6930.00).retcode
            == TradeResult.TRADE_RETCODE_DONE
        )


# helper for the invalid stops retcode
def TradeResult_INVALID_STOPS():
    from tradetropy.core.broker import TradeResult

    return TradeResult.TRADE_RETCODE_INVALID_STOPS


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PERFORMANCE METRICS  (KlineBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPerformanceMetrics:
    def test_profit_calculation_accuracy(self, kline_broker):
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6924.00, 6926.00, 6923.00, 6925.00),
            ],
        )
        assert kline_broker.get_positions("MES")[0].profit == pytest.approx(
            5.00, abs=0.01
        )

        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:02:00", 6926.00, 6928.00, 6925.00, 6927.00),
            ],
        )
        kline_broker.position_close(kline_broker.get_positions("MES")[0].ticket)
        assert kline_broker.account_info().balance == pytest.approx(10_013.80, abs=0.01)

    def test_commission_deduction(self, kline_broker):
        initial = kline_broker.account_info().balance
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:00:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        buy(kline_broker, volume=1)
        feed_bars(
            kline_broker,
            [
                Bar("2026-02-11 10:01:00", 6920.00, 6922.00, 6918.00, 6920.00),
            ],
        )
        assert kline_broker.account_info().balance == initial - 0.6

        kline_broker.position_close(kline_broker.get_positions("MES")[0].ticket)
        assert kline_broker.account_info().balance == initial - 1.2


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: ON_TICK  (TickBroker)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestOnTick:
    """
    Conventions:
    - spread=0 → bid == ask unless stated otherwise
    - MES: tick_size=0.25, tick_value=1.25, commission=0.6/lot
    """

    # ── Execution on the next tick ─────────────────────────────────

    def test_market_order_not_executed_same_tick(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        result = buy_tick(tick_broker, volume=1)
        assert result.retcode == result.TRADE_RETCODE_PLACED
        assert len(tick_broker.get_positions("MES")) == 0

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.25, 6920.25)])
        positions = tick_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].price_open == 6920.25

    def test_market_buy_executes_at_ask(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6919.75, 6920.00)])
        buy_tick(tick_broker, volume=1)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.50)])
        assert tick_broker.get_positions("MES")[0].price_open == 6920.50

    def test_market_sell_executes_at_bid(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.25)])
        sell_tick(tick_broker, volume=1)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6919.50, 6920.00)])
        assert tick_broker.get_positions("MES")[0].price_open == 6919.50

    def test_order_sent_between_ticks_executes_first_tick(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_tick(tick_broker, volume=1)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6921.00, 6921.00)])
        assert tick_broker.get_positions("MES")[0].price_open == 6921.00

    # ── TickBroker never executes on the same tick ─────────────────────────────

    def test_tick_broker_always_enqueues(self, tick_broker):
        """
        TickBroker has no trade_on_close — market orders are always
        enqueued regardless of how the broker was constructed.
        """
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        result = buy_tick(tick_broker, volume=1)
        assert result.retcode == result.TRADE_RETCODE_PLACED
        assert len(tick_broker.get_positions("MES")) == 0

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.50, 6920.50)])
        assert tick_broker.get_positions("MES")[0].price_open == 6920.50

    # ── SL/TP tick a tick ─────────────────────────────────────────────────────

    def test_sl_hit_long(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_tick(tick_broker, volume=1, sl=6915.00, tp=6930.00)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        assert len(tick_broker.get_positions("MES")) == 1

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6914.75, 6915.00)])
        assert len(tick_broker.get_positions("MES")) == 0
        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6915.00
        assert "[sl" in deals[1].comment.lower()

    def test_tp_hit_long(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_tick(tick_broker, volume=1, sl=6910.00, tp=6928.00)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6928.25, 6928.50)])

        assert len(tick_broker.get_positions("MES")) == 0
        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6928.00
        assert "[tp" in deals[1].comment.lower()
        assert tick_broker.account_info().balance == pytest.approx(10_038.80, abs=0.01)

    def test_sl_hit_short(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        sell_tick(tick_broker, volume=1, sl=6928.00, tp=6910.00)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6927.75, 6928.25)])

        assert len(tick_broker.get_positions("MES")) == 0
        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6928.00
        assert "[sl" in deals[1].comment.lower()
        assert tick_broker.account_info().balance == pytest.approx(9_958.80, abs=0.01)

    def test_tp_hit_short(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        sell_tick(tick_broker, volume=1, sl=6930.00, tp=6912.00)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6911.50, 6911.75)])

        assert len(tick_broker.get_positions("MES")) == 0
        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6912.00
        assert "[tp" in deals[1].comment.lower()
        assert tick_broker.account_info().balance == pytest.approx(10_038.80, abs=0.01)

    def test_sl_priority_over_tp_same_tick_long(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_tick(tick_broker, volume=1, sl=6915.00, tp=6925.00)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6914.75, 6926.00)])

        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6915.00
        assert "[sl" in deals[1].comment.lower()

    def test_sl_priority_over_tp_same_tick_short(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        sell_tick(tick_broker, volume=1, sl=6928.00, tp=6912.00)
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6911.00, 6928.25)])

        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6928.00
        assert "[sl" in deals[1].comment.lower()

    def test_sltp_evaluated_chronologically(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_tick(tick_broker, volume=1, sl=6915.00, tp=6928.00)
        feed_ticks(
            tick_broker,
            [
                Tick("2026-02-11 10:00:01", 6920.00, 6920.00),
                Tick("2026-02-11 10:00:02", 6922.00, 6922.00),
                Tick("2026-02-11 10:00:03", 6928.00, 6928.00),
                Tick("2026-02-11 10:00:04", 6914.00, 6914.00),
            ],
        )
        assert len(tick_broker.get_positions("MES")) == 0
        deals = tick_broker.get_deal_history("MES")
        assert len(deals) == 2
        assert deals[1].price == 6928.00
        assert "[tp" in deals[1].comment.lower()

    # ── Pending orders con bid/ask reales ────────────────────────────────────

    def test_buy_limit_triggers_on_ask(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_limit_tick(tick_broker, price=6918.00, volume=1)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6918.50, 6919.00)])
        assert len(tick_broker.get_positions("MES")) == 0

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6917.75, 6918.00)])
        positions = tick_broker.get_positions("MES")
        assert len(positions) == 1
        assert positions[0].price_open == 6918.00

    def test_sell_limit_triggers_on_bid(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        sell_limit_tick(tick_broker, price=6924.00, volume=1)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6922.00, 6922.50)])
        assert len(tick_broker.get_positions("MES")) == 0

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6924.00, 6924.25)])
        assert tick_broker.get_positions("MES")[0].price_open == 6924.00

    def test_buy_stop_triggers_on_ask(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_stop_tick(tick_broker, price=6925.00, volume=1)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6923.00, 6924.75)])
        assert len(tick_broker.get_positions("MES")) == 0

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6924.75, 6925.25)])
        assert tick_broker.get_positions("MES")[0].price_open == 6925.25

    def test_sell_stop_triggers_on_bid(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        sell_stop_tick(tick_broker, price=6916.00, volume=1)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6917.00, 6917.50)])
        assert len(tick_broker.get_positions("MES")) == 0

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6915.75, 6916.25)])
        assert tick_broker.get_positions("MES")[0].price_open == 6915.75

    def test_pending_order_with_sl_tp(self, tick_broker):
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_limit_tick(tick_broker, price=6916.00, volume=1, sl=6912.00, tp=6924.00)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6915.75, 6916.00)])
        pos = tick_broker.get_positions("MES")
        assert pos[0].sl == 6912.00
        assert pos[0].tp == 6924.00

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6924.00, 6924.25)])
        assert len(tick_broker.get_positions("MES")) == 0
        deals = tick_broker.get_deal_history("MES")
        assert deals[1].price == 6924.00
        assert "[tp" in deals[1].comment.lower()

    # ── Secuencia completa con balance exacto ─────────────────────────────────

    def test_full_tick_sequence_exact_balance(self, tick_broker):
        """
        Op 1 BUY: entry ask=6920 → TP bid=6928 → +40.00 - 1.2 = +38.80
        Op 2 SELL: entry bid=6925.75 → SL ask=6930 → -21.25 - 0.6 = -21.85
        Final balance: 10000 + 38.80 - 21.85 = 10016.95 … see inline calculation
        """
        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:00", 6920.00, 6920.00)])
        buy_tick(tick_broker, volume=1, sl=6912.00, tp=6928.00)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:01", 6920.00, 6920.00)])
        assert tick_broker.get_positions("MES")[0].price_open == 6920.00
        assert tick_broker.account_info().balance == pytest.approx(9_999.40, abs=0.01)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:02", 6928.00, 6928.25)])
        assert len(tick_broker.get_positions("MES")) == 0
        assert tick_broker.account_info().balance == pytest.approx(10_038.80, abs=0.01)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:03", 6926.00, 6926.25)])
        sell_tick(tick_broker, volume=1, sl=6930.00, tp=6918.00)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:04", 6925.75, 6926.00)])
        assert tick_broker.get_positions("MES")[0].price_open == 6925.75
        assert tick_broker.account_info().balance == pytest.approx(10_038.20, abs=0.01)

        feed_ticks(tick_broker, [Tick("2026-02-11 10:00:05", 6929.75, 6930.25)])
        assert len(tick_broker.get_positions("MES")) == 0
        # Loss: (6925.75-6930)/0.25*1.25 = -17 ticks * 1.25 = -21.25 - 0.6 = -21.85
        assert tick_broker.account_info().balance == pytest.approx(10_016.35, abs=0.01)


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "--tb=short"])

# endregion

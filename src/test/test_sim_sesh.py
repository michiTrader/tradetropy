# region test_sim_sesh.py v1.0
"""
Tests for SeshSimulator — Suite v1.0

Coverage:
  · Construction (tick / kline) and broker parameters
  · SeshSimulator.buy / sell — market orders (tick and kline)
  · SeshSimulator.buy / sell — pending orders with explicit price
      — automatic detection of BUY_LIMIT vs BUY_STOP
      — automatic detection of SELL_LIMIT vs SELL_STOP
  · SeshSimulator.position_close / position_modify / order_delete
  · SeshSimulator.account / positions / orders / deals / orders_history
  · Strategy.buy / Strategy.sell (shortcuts)
  · Strategy.sesh (property) injected by the engine
  · Integration BacktestEngine.by_ticks + SeshSimulator
  · Integration BacktestEngine.by_klines + SeshSimulator
  · Alias TickBacktestEngine / BarBacktestEngine — sesh also available
  · Error when there is no sesh (no engine)
  · Commission and exact balance
  · SL/TP propagated correctly to broker
  · reset()

Conventions:
  - MES: tick_size=0.25, tick_value=1.25, commission=0.6/lot
  - Spread=0, trade_on_close=True in KlineBroker for simplicity

Usage:
    pytest test_sim_sesh.py -v
    pytest test_sim_sesh.py::TestSeshMarketOrders -v
"""

import pytest
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

from tradetropy.session.base import SeshSimulatorBase
from tradetropy.core.broker import (
    KlineBroker,
    TickBroker,
    SymbolConfig,
    OrderType,
    OrderState,
    CommissionType,
    PositionMode,
    TradeResult,
)
from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.core.data_types import TickData, KlineData
from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.exceptions import ConfigError

# Tiny integration fixtures intentionally trip the insufficient-sample stats
# warning (asserted on its own in test_stats.py). Colon matched with '.' since
# warning-filter fields split on ':'.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Stats. insufficient sample:UserWarning"
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
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in self.time else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(self.time, fmt)


def feed_bars(broker_or_sesh, bars: List[Bar], symbol="MES"):
    """Feed bars to a KlineBroker or the internal broker of a SeshSimulator."""
    b = (
        broker_or_sesh._broker
        if isinstance(broker_or_sesh, SeshSimulatorBase)
        else broker_or_sesh
    )
    for bar in bars:
        b.update_kline(symbol, bar.dt, bar.open, bar.high, bar.low, bar.close)


def feed_ticks(broker_or_sesh, ticks: List[Tick], symbol="MES"):
    b = (
        broker_or_sesh._broker
        if isinstance(broker_or_sesh, SeshSimulatorBase)
        else broker_or_sesh
    )
    for tick in ticks:
        b.update_tick(symbol, tick.dt, tick.bid, tick.ask)


def make_tick_matrix(rows, symbol_ts_offset=0):
    """
    rows: list of (ts_ms, price, volume)
    Returns a synthetic ndarray [N × N_TICK_COLS] with bid==ask==price.
    """
    out = np.zeros((len(rows), N_TICK_COLS), dtype=np.float64)
    for i, (ts, price, vol) in enumerate(rows):
        out[i, _TICK_COL["ts"]] = ts + symbol_ts_offset
        out[i, _TICK_COL["bid"]] = price
        out[i, _TICK_COL["ask"]] = price
        out[i, _TICK_COL["price"]] = price
        out[i, _TICK_COL["volume"]] = vol
        out[i, _TICK_COL["flags"]] = 0.0
        out[i, _TICK_COL["volume_real"]] = np.nan
    return out


BASE_TS = 1_700_000_000_000  # ms
MIN_MS = 60_000


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mes_cfg():
    return SymbolConfig(
        name="MES",
        tick_size=0.25,
        tick_value=1.25,
        contract_size=1,
        digits=2,
        avg_spread=0.0,
    )


@pytest.fixture
def sesh_tick(mes_cfg):
    s = SeshSimulatorBase(
        "tick",
        initial_balance=10_000.0,
        commission=0.6,
        position_mode=PositionMode.POSITION_MODE_NETTING,
    )
    s.add_symbol(mes_cfg)
    return s


@pytest.fixture
def sesh_kline(mes_cfg):
    s = SeshSimulatorBase(
        "kline",
        initial_balance=10_000.0,
        commission=0.6,
        use_spread=False,
        trade_on_close=True,
        position_mode=PositionMode.POSITION_MODE_NETTING,
    )
    s.add_symbol(mes_cfg)
    return s


@pytest.fixture
def ticks_simple():
    """5 synthetic ticks in the same 5-minute interval."""
    rows = [
        (BASE_TS + 0 * 1000, 6920.00, 1.0),
        (BASE_TS + 1 * 1000, 6921.00, 1.0),
        (BASE_TS + 2 * 1000, 6922.00, 1.0),
        (BASE_TS + 3 * 1000, 6923.00, 1.0),
        (BASE_TS + 4 * 1000, 6924.00, 1.0),
    ]
    return make_tick_matrix(rows)


@pytest.fixture
def klines_simple():
    """3 one-minute klines [ts, open, high, low, close, volume]."""
    return np.array(
        [
            [BASE_TS + 0 * MIN_MS, 6920.00, 6922.00, 6918.00, 6920.00, 10.0],
            [BASE_TS + 1 * MIN_MS, 6921.00, 6924.00, 6920.00, 6923.00, 12.0],
            [BASE_TS + 2 * MIN_MS, 6924.00, 6928.00, 6923.00, 6927.00, 8.0],
        ],
        dtype=np.float64,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSeshConstruction:
    def test_tick_mode_creates_tick_broker(self):
        s = SeshSimulatorBase("tick")
        assert isinstance(s._broker, TickBroker)
        assert s._feed_type == "tick"

    def test_kline_mode_creates_bar_broker(self):
        s = SeshSimulatorBase("kline")
        assert isinstance(s._broker, KlineBroker)
        assert s._feed_type == "kline"

    def test_invalid_mode_raises(self):
        with pytest.raises(ConfigError, match="tick.*kline"):
            SeshSimulatorBase("ohlc")

    def test_initial_balance_propagated(self):
        s = SeshSimulatorBase("tick", initial_balance=25_000.0)
        assert s.account_info().balance == pytest.approx(25_000.0)

    def test_commission_propagated(self):
        """Commission is applied on the first order."""
        s = SeshSimulatorBase("tick", initial_balance=10_000.0, commission=1.5)
        feed_ticks(s, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)], "MES")
        s.add_symbol(SymbolConfig("MES", 0.25, 1.25, 1, 2))
        feed_ticks(s, [Tick("2026-01-01 10:00:01", 6920.0, 6920.0)], "MES")
        r = s.buy("MES", volume=1)
        feed_ticks(s, [Tick("2026-01-01 10:00:02", 6920.0, 6920.0)], "MES")
        assert s.account_info().balance == pytest.approx(10_000.0 - 1.5)

    def test_add_symbol_delegates_to_broker(self, mes_cfg):
        s = SeshSimulatorBase("tick")
        s.add_symbol(mes_cfg)
        assert "MES" in s._broker.symbols

    def test_repr_no_raise(self, sesh_tick):
        assert "SeshSimulatorBase" in repr(sesh_tick)
        assert "tick" in repr(sesh_tick)

    def test_reset_restores_balance(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6920.0, 6920.0)])
        sesh_tick.reset()
        assert sesh_tick.account_info().balance == pytest.approx(10_000.0)
        assert sesh_tick.positions() == []


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: AUTO MODE — EAGER BROKER & PRE-ENGINE INSPECTION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSeshAutoMode:
    """Session created without feed_type (auto mode): must be inspectable before
    connecting to an engine, and the engine rebuilds the broker if it infers
    'kline'."""

    def test_auto_creates_tick_broker_default(self):
        s = SeshSimulatorBase()  # feed_type=None
        assert isinstance(s._broker, TickBroker)
        assert s._feed_type is None          # auto-detect sigue vivo
        assert s._broker_es_default is True

    def test_auto_account_info_pre_engine(self):
        s = SeshSimulatorBase(initial_balance=25_000.0)
        # Doesn't crash even without an engine.
        assert s.account_info().balance == pytest.approx(25_000.0)
        assert s.positions() == []

    def test_auto_add_symbol_pre_engine(self, mes_cfg):
        s = SeshSimulatorBase()
        s.add_symbol(mes_cfg)
        assert "MES" in s._broker.symbols

    def test_bind_tick_preserves_broker(self, mes_cfg):
        s = SeshSimulatorBase()
        s.add_symbol(mes_cfg)
        broker_before = s._broker
        s._bind_feed_type("tick")
        assert s._broker is broker_before      # same object, no rebuild
        assert isinstance(s._broker, TickBroker)
        assert s._feed_type == "tick"
        assert s._broker_es_default is False

    def test_bind_kline_rebuilds_preserves_state(self, mes_cfg):
        s = SeshSimulatorBase(initial_balance=25_000.0)
        s.add_symbol(mes_cfg)
        s._bind_feed_type("kline")
        assert isinstance(s._broker, KlineBroker)
        assert s._feed_type == "kline"
        assert s.account_info().balance == pytest.approx(25_000.0)
        assert "MES" in s._broker.symbols     # symbols preserved
        assert s._broker_es_default is False

    def test_bind_kline_after_trading_raises(self):
        s = SeshSimulatorBase()
        s.add_symbol(SymbolConfig("MES", 0.25, 1.25, 1, 2))
        feed_ticks(s, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)], "MES")
        s.buy("MES", volume=1)
        feed_ticks(s, [Tick("2026-01-01 10:00:01", 6920.0, 6920.0)], "MES")
        with pytest.raises(ConfigError, match="trading state"):
            s._bind_feed_type("kline")

    def test_explicit_feed_type_preserves_conflict(self):
        s = SeshSimulatorBase("tick")
        assert s._broker_es_default is False
        with pytest.raises(ConfigError, match="feed_type conflict"):
            s._bind_feed_type("kline")


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: MARKET ORDERS (TICK MODE)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSeshMarketOrders:
    def test_buy_market_returns_placed(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        r = sesh_tick.buy("MES", volume=1)
        assert r.retcode == TradeResult.TRADE_RETCODE_PLACED

    def test_buy_market_executes_next_tick(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        assert sesh_tick.positions("MES") == []  # not executed yet

        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        pos = sesh_tick.positions("MES")
        assert len(pos) == 1
        assert pos[0].price_open == pytest.approx(6921.0)

    def test_sell_market_executes_next_tick(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.sell("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6919.0, 6919.0)])
        pos = sesh_tick.positions("MES")
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_SELL
        assert pos[0].price_open == pytest.approx(6919.0)

    def test_buy_market_with_sltp(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1, sl=6910.0, tp=6930.0)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        pos = sesh_tick.positions("MES")
        assert pos[0].sl == pytest.approx(6910.0)
        assert pos[0].tp == pytest.approx(6930.0)

    def test_sell_market_with_sltp(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.sell("MES", volume=1, sl=6930.0, tp=6910.0)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6919.0, 6919.0)])
        pos = sesh_tick.positions("MES")
        assert pos[0].sl == pytest.approx(6930.0)
        assert pos[0].tp == pytest.approx(6910.0)

    def test_balance_after_commission(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        assert sesh_tick.account_info().balance == pytest.approx(10_000.0 - 0.6)

    def test_buy_market_kline_executes_at_close(self, sesh_kline):
        """trade_on_close=True → market order executes at the close of the same bar."""
        feed_bars(
            sesh_kline, [Bar("2026-01-01 10:00:00", 6920.0, 6922.0, 6918.0, 6920.0)]
        )
        r = sesh_kline.buy("MES", volume=1)
        assert r.retcode == TradeResult.TRADE_RETCODE_DONE
        pos = sesh_kline.positions("MES")
        assert len(pos) == 1
        assert pos[0].price_open == pytest.approx(6920.0)

    def test_sell_market_kline_executes_at_close(self, sesh_kline):
        feed_bars(
            sesh_kline, [Bar("2026-01-01 10:00:00", 6920.0, 6922.0, 6918.0, 6920.0)]
        )
        r = sesh_kline.sell("MES", volume=1)
        assert r.retcode == TradeResult.TRADE_RETCODE_DONE
        pos = sesh_kline.positions("MES")
        assert len(pos) == 1
        assert pos[0].type == OrderType.ORDER_TYPE_SELL


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PENDING ORDERS — AUTOMATIC BUY_LIMIT / BUY_STOP DETECTION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSeshPendingOrders:
    """
    SeshSimulator infers the pending order type:
        buy  + price < ask  → BUY_LIMIT   (we want to buy cheaper)
        buy  + price > ask  → BUY_STOP    (we want to buy on bullish breakout)
        sell + price > bid  → SELL_LIMIT  (we want to sell more expensive)
        sell + price < bid  → SELL_STOP   (we want to sell on bearish breakout)
    """

    def test_buy_limit_detected_correctly(self, sesh_tick):
        """price < current ask → ORDER_TYPE_BUY_LIMIT."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1, price=6915.0)  # 6915 < 6920 (ask)
        orders = sesh_tick.orders("MES")
        assert len(orders) == 1
        assert orders[0].type == OrderType.ORDER_TYPE_BUY_LIMIT
        assert orders[0].price == pytest.approx(6915.0)

    def test_buy_stop_detected_correctly(self, sesh_tick):
        """price > current ask → ORDER_TYPE_BUY_STOP."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1, price=6930.0)  # 6930 > 6920 (ask)
        orders = sesh_tick.orders("MES")
        assert len(orders) == 1
        assert orders[0].type == OrderType.ORDER_TYPE_BUY_STOP

    def test_sell_limit_detected_correctly(self, sesh_tick):
        """price > current bid → ORDER_TYPE_SELL_LIMIT."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.sell("MES", volume=1, price=6930.0)  # 6930 > 6920 (bid)
        orders = sesh_tick.orders("MES")
        assert len(orders) == 1
        assert orders[0].type == OrderType.ORDER_TYPE_SELL_LIMIT

    def test_sell_stop_detected_correctly(self, sesh_tick):
        """price < current bid → ORDER_TYPE_SELL_STOP."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.sell("MES", volume=1, price=6910.0)  # 6910 < 6920 (bid)
        orders = sesh_tick.orders("MES")
        assert len(orders) == 1
        assert orders[0].type == OrderType.ORDER_TYPE_SELL_STOP

    def test_buy_limit_executes_when_ask_drops(self, sesh_tick):
        """BUY_LIMIT at 6915 executes when the ask reaches ≤ 6915."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1, price=6915.0)

        # Tick with ask=6918 → doesn't execute yet
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6917.5, 6918.0)])
        assert sesh_tick.positions("MES") == []

        # Tick with ask=6915 → executes
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:02", 6914.75, 6915.0)])
        pos = sesh_tick.positions("MES")
        assert len(pos) == 1
        assert pos[0].price_open == pytest.approx(6915.0)

    def test_sell_limit_executes_when_bid_rises(self, sesh_tick):
        """SELL_LIMIT at 6930 executes when the bid reaches ≥ 6930."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.sell("MES", volume=1, price=6930.0)

        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6925.0, 6925.25)])
        assert sesh_tick.positions("MES") == []

        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:02", 6930.0, 6930.25)])
        pos = sesh_tick.positions("MES")
        assert len(pos) == 1
        assert pos[0].price_open == pytest.approx(6930.0)

    def test_buy_limit_kline(self, sesh_kline):
        """BUY_LIMIT in kline mode executes intrabar."""
        feed_bars(
            sesh_kline, [Bar("2026-01-01 10:00:00", 6920.0, 6922.0, 6918.0, 6920.0)]
        )
        sesh_kline.buy("MES", volume=1, price=6915.0)  # BUY_LIMIT

        feed_bars(
            sesh_kline, [Bar("2026-01-01 10:01:00", 6918.0, 6920.0, 6914.0, 6916.0)]
        )
        pos = sesh_kline.positions("MES")
        assert len(pos) == 1
        assert pos[0].price_open == pytest.approx(6915.0)

    def test_sell_limit_kline(self, sesh_kline):
        """SELL_LIMIT in kline mode executes intrabar."""
        feed_bars(
            sesh_kline, [Bar("2026-01-01 10:00:00", 6920.0, 6922.0, 6918.0, 6920.0)]
        )
        sesh_kline.sell("MES", volume=1, price=6930.0)  # SELL_LIMIT

        feed_bars(
            sesh_kline, [Bar("2026-01-01 10:01:00", 6920.0, 6931.0, 6919.0, 6929.0)]
        )
        pos = sesh_kline.positions("MES")
        assert len(pos) == 1
        assert pos[0].price_open == pytest.approx(6930.0)

    def test_pending_with_sltp_propagated(self, sesh_tick):
        """SL and TP are transferred to the position opened by the pending order."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1, price=6915.0, sl=6910.0, tp=6925.0)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6914.75, 6915.0)])
        pos = sesh_tick.positions("MES")
        assert pos[0].sl == pytest.approx(6910.0)
        assert pos[0].tp == pytest.approx(6925.0)

    def test_cancel_removes_order(self, sesh_tick):
        """sesh.order_delete(ticket) deletes the pending order."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        r = sesh_tick.buy("MES", volume=1, price=6915.0)
        ticket = r.order
        assert sesh_tick.order_delete(ticket) == True
        assert sesh_tick.orders("MES") == []
        hist = sesh_tick.order_history("MES")
        canceled = [o for o in hist if o.ticket == ticket]
        assert canceled[0].state == OrderState.ORDER_STATE_CANCELED


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CLOSE / MODIFY
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSeshPositionManagement:
    def test_close_closes_position(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        ticket = sesh_tick.positions("MES")[0].ticket

        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:02", 6925.0, 6925.0)])
        assert sesh_tick.position_close(ticket) == True
        assert sesh_tick.positions("MES") == []

    def test_close_nonexistent_returns_false(self, sesh_tick):
        assert sesh_tick.position_close(9999) == False

    def test_partial_close(self, sesh_tick):
        """close with volume less than total → position is reduced."""
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=4)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        ticket = sesh_tick.positions("MES")[0].ticket

        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:02", 6925.0, 6925.0)])
        sesh_tick.position_close(ticket, volume=1)
        pos = sesh_tick.positions("MES")
        assert len(pos) == 1
        assert pos[0].volume == pytest.approx(3.0)

    def test_modify_changes_sl_tp(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        ticket = sesh_tick.positions("MES")[0].ticket

        assert sesh_tick.position_modify(ticket, sl=6910.0, tp=6940.0) == True
        pos = sesh_tick.positions("MES")[0]
        assert pos.sl == pytest.approx(6910.0)
        assert pos.tp == pytest.approx(6940.0)

    def test_modify_nonexistent_returns_false(self, sesh_tick):
        assert sesh_tick.position_modify(9999, sl=6910.0) == False

    def test_deals_records_opening_and_closing(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])
        ticket = sesh_tick.positions("MES")[0].ticket
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:02", 6925.0, 6925.0)])
        sesh_tick.position_close(ticket)

        deals = sesh_tick.deals("MES")
        assert len(deals) == 2  # opening + closing

    def test_account_equity_with_open_position(self, sesh_tick):
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:00", 6920.0, 6920.0)])
        sesh_tick.buy("MES", volume=1)
        feed_ticks(sesh_tick, [Tick("2026-01-01 10:00:01", 6921.0, 6921.0)])

        # Position opened at 6921, current price 6921 → profit = 0
        acc = sesh_tick.account_info()
        assert acc.balance == pytest.approx(10_000.0 - 0.6)
        assert acc.equity == pytest.approx(acc.balance)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: STRATEGY SESH
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestStrategySesh:
    def test_sesh_none_before_engine(self):
        s = Strategy()
        assert s.sesh is None

    def test_sesh_available_in_on_data(self, ticks_simple):
        """self.sesh is injected and available from the first on_data."""
        sesh_ref = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                if not sesh_ref:
                    sesh_ref.append(self.sesh)

        sesh = SeshSimulatorBase("tick")
        BacktestEngine.by_ticks(
            S(), data=(TickData("MES", ticks_simple, tick_size=0.01),), sesh=sesh
        ).run()
        assert sesh_ref[0] is not None
        assert isinstance(sesh_ref[0], SeshSimulatorBase)

    def test_sesh_buy_market_from_on_data(self, ticks_simple):
        """self.sesh.buy() in on_data sends a market order to the broker."""
        captures = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                if self._tp.n_ticks == 1:
                    r = self.sesh.buy("MES", volume=1)
                    captures.append(r.retcode)

        sesh = SeshSimulatorBase("tick")
        BacktestEngine.by_ticks(
            S(), data=(TickData("MES", ticks_simple, tick_size=0.01),), sesh=sesh
        ).run()
        assert captures[0] == TradeResult.TRADE_RETCODE_PLACED

    def test_sesh_sell_market_from_on_data(self, ticks_simple):
        """self.sesh.sell() in on_data sends a market order to the broker."""
        captures = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                if self._tp.n_ticks == 1:
                    r = self.sesh.sell("MES", volume=1)
                    captures.append(r.retcode)

        sesh = SeshSimulatorBase("tick")
        BacktestEngine.by_ticks(
            S(), data=(TickData("MES", ticks_simple, tick_size=0.01),), sesh=sesh
        ).run()
        assert captures[0] == TradeResult.TRADE_RETCODE_PLACED

    def test_sesh_buy_with_pending_price(self, ticks_simple):
        """self.sesh.buy(price=...) generates a BUY_LIMIT pending order."""
        captures = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                if self._tp.n_ticks == 1:
                    r = self.sesh.buy("MES", volume=1, price=6900.0)
                    captures.append(r)

        sesh = SeshSimulatorBase("tick")
        engine = BacktestEngine.by_ticks(
            S(), data=(TickData("MES", ticks_simple, tick_size=0.01),), sesh=sesh
        ).run()
        r = captures[0]
        assert r.retcode == TradeResult.TRADE_RETCODE_PLACED
        orders = engine.strategy.sesh.orders("MES")
        assert len(orders) == 1
        assert orders[0].type == OrderType.ORDER_TYPE_BUY_LIMIT


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: INTEGRATION BacktestEngine.by_ticks + SeshSimulator
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestIntegrationFromTicksWithBroker:
    def test_sesh_available_in_strategy(self, ticks_simple):
        sesh_ref = []

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                if not sesh_ref:
                    sesh_ref.append(self.sesh)

        sesh = SeshSimulatorBase("tick")
        BacktestEngine.by_ticks(
            S(), data=(TickData("MES", ticks_simple, tick_size=0.01),), sesh=sesh
        ).run()
        assert sesh_ref[0] is not None
        assert isinstance(sesh_ref[0], SeshSimulatorBase)
        assert sesh_ref[0]._feed_type == "tick"

    def test_sesh_broker_is_tick_broker(self, ticks_simple):
        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                pass

        sesh = SeshSimulatorBase("tick")
        engine = BacktestEngine.by_ticks(
            S(), data=(TickData("MES", ticks_simple, tick_size=0.01),), sesh=sesh
        ).run()
        assert isinstance(engine.strategy.sesh._broker, TickBroker)

    def test_buy_from_strategy_and_open_position(self, mes_cfg):
        """
        Complete tick-to-tick flow:
          tick 1 → on_data calls self.sesh.buy(market)
          tick 2 → position executed at tick 2 price
        """
        rows = [
            (BASE_TS + 0, 6920.0, 1.0),
            (BASE_TS + 1000, 6921.0, 1.0),
            (BASE_TS + 2000, 6925.0, 1.0),
        ]
        ticks = make_tick_matrix(rows)

        state = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                n = self._tp.n_ticks
                if n == 1:
                    self.sesh.buy("MES", volume=1)
                if n == 2:
                    pos = self.sesh.positions("MES")
                    if pos:
                        state["open_price"] = pos[0].price_open
                if n == 3:
                    pos = self.sesh.positions("MES")
                    if pos:
                        state["profit_n3"] = pos[0].profit

        sesh = SeshSimulatorBase(
            "tick",
            commission=0.6,
            position_mode=PositionMode.POSITION_MODE_NETTING,
        )
        engine = BacktestEngine.by_ticks(
            S(),
            data=(TickData("MES", ticks, tick_size=0.25),),
            sesh=sesh,
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert state.get("open_price") == pytest.approx(6921.0)

    def test_sell_and_close_within_strategy(self, mes_cfg):
        rows = [
            (BASE_TS + 0, 6920.0, 1.0),
            (BASE_TS + 1_000, 6919.0, 1.0),
            (BASE_TS + 2_000, 6915.0, 1.0),
            (BASE_TS + 3_000, 6910.0, 1.0),
        ]
        ticks = make_tick_matrix(rows)
        state = {}

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=10)

            def on_data(self):
                n = self._tp.n_ticks
                if n == 1:
                    self.sesh.sell("MES", volume=1)
                if n == 3:
                    pos = self.sesh.positions("MES")
                    if pos:
                        state["ticket"] = pos[0].ticket
                        self.sesh.position_close(pos[0].ticket)
                if n == 4:
                    state["pos_count"] = len(self.sesh.positions("MES"))
                    state["balance"] = self.sesh.account_info().balance

        engine = BacktestEngine.by_ticks(
            S(),
            data=(TickData("MES", ticks, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "tick",
                commission=0.6,
                position_mode=PositionMode.POSITION_MODE_NETTING,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert state.get("pos_count") == 0

    def test_sesh_account_balance_exact(self, mes_cfg):
        """
        Buy at 6921, close at 6925.
        Profit = (6925 - 6921) / 0.25 * 1.25 * 1 = 20.00
        Commissions = 0.6 * 2 = 1.20
        Final balance = 10_000 + 20.00 - 1.20 = 10_018.80

        The broker is pre-configured with mes_cfg BEFORE running the engine.
        """
        rows = [
            (BASE_TS + 0, 6920.0, 1.0),
            (BASE_TS + 1_000, 6921.0, 1.0),
            (BASE_TS + 2_000, 6925.0, 1.0),
        ]
        ticks = make_tick_matrix(rows)

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=5)

            def on_data(self):
                n = self._tp.n_ticks
                if n == 1:
                    self.sesh.buy("MES", volume=1)
                if n == 3:
                    pos = self.sesh.positions("MES")
                    if pos:
                        self.sesh.position_close(pos[0].ticket)

        engine = BacktestEngine.by_ticks(
            S(),
            data=(TickData("MES", ticks, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "tick",
                initial_balance=10_000.0,
                commission=0.6,
                position_mode=PositionMode.POSITION_MODE_NETTING,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert engine.strategy.sesh.account_info().balance == pytest.approx(
            10_018.80, abs=0.01
        )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: INTEGRATION BacktestEngine.by_klines + SeshSimulator
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestIntegrationFromKlines:
    def test_sesh_available_in_strategy_kline(self, klines_simple):
        sesh_ref = []

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=10)

            def on_data(self):
                if not sesh_ref:
                    sesh_ref.append(self.sesh)

        BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines_simple, timeframe=60_000, tick_size=0.01),),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert sesh_ref[0] is not None
        assert isinstance(sesh_ref[0], SeshSimulatorBase)
        assert sesh_ref[0]._feed_type == "kline"

    def test_sesh_broker_is_bar_broker(self, klines_simple):
        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=10)

            def on_data(self):
                pass

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines_simple, timeframe=60_000, tick_size=0.01),),
            sesh=SeshSimulatorBase("kline"),
        ).run()
        assert isinstance(engine.strategy.sesh._broker, KlineBroker)

    def test_buy_kline_market_in_strategy(self, klines_simple, mes_cfg):
        """trade_on_close=True → buy executes at the close of the same bar."""
        state = {}

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=10)

            def on_data(self):
                n = self._ohlc.n_klines
                if n == 1:
                    self.sesh.buy("MES", volume=1)
                if n == 2:
                    pos = self.sesh.positions("MES")
                    if pos:
                        state["open"] = pos[0].price_open

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines_simple, timeframe=60_000, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "kline",
                trade_on_close=True,
                commission=0.6,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        # Bar 1: close=6920 → open=6920
        assert state.get("open") == pytest.approx(6920.0)

    def test_sell_limit_kline_in_strategy(self, mes_cfg):
        """SELL_LIMIT placed on bar 1 executes when the high reaches the price."""
        klines = np.array(
            [
                [BASE_TS + 0 * MIN_MS, 6920.0, 6922.0, 6918.0, 6920.0, 10.0],
                [BASE_TS + 1 * MIN_MS, 6920.0, 6933.0, 6919.0, 6930.0, 12.0],
            ],
            dtype=np.float64,
        )

        state = {}

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=5)

            def on_data(self):
                n = self._ohlc.n_klines
                if n == 1:
                    # sell price > current price → SELL_LIMIT
                    self.sesh.sell("MES", volume=1, price=6930.0)
                if n == 2:
                    state["pos"] = self.sesh.positions("MES")
                    state["balance"] = self.sesh.account_info().balance

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines, timeframe=60_000, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "kline",
                trade_on_close=True,
                commission=0.6,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert len(state["pos"]) == 1
        assert state["pos"][0].price_open == pytest.approx(6930.0)

    def test_buy_limit_kline_in_strategy(self, mes_cfg):
        """BUY_LIMIT placed on bar 1 executes when the low reaches the price."""
        klines = np.array(
            [
                [BASE_TS + 0 * MIN_MS, 6920.0, 6922.0, 6918.0, 6920.0, 10.0],
                [BASE_TS + 1 * MIN_MS, 6918.0, 6920.0, 6913.0, 6915.0, 12.0],
            ],
            dtype=np.float64,
        )

        state = {}

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=5)

            def on_data(self):
                n = self._ohlc.n_klines
                if n == 1:
                    # price < current price → BUY_LIMIT
                    self.sesh.buy("MES", volume=1, price=6915.0)
                if n == 2:
                    state["pos"] = self.sesh.positions("MES")

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines, timeframe=60_000, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "kline",
                trade_on_close=True,
                commission=0.6,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert len(state["pos"]) == 1
        assert state["pos"][0].price_open == pytest.approx(6915.0)

    def test_balance_kline_buy_and_close(self, mes_cfg):
        """
        Buy at bar 1 close=6920, close at bar 3 close=6927.
        Profit = (6927 - 6920) / 0.25 * 1.25 = 35.00
        Commissions = 0.6 * 2 = 1.20
        Balance = 10_000 + 35.00 - 1.20 = 10_033.80
        """
        klines = np.array(
            [
                [BASE_TS + 0 * MIN_MS, 6920.0, 6922.0, 6918.0, 6920.0, 10.0],
                [BASE_TS + 1 * MIN_MS, 6921.0, 6924.0, 6920.0, 6923.0, 12.0],
                [BASE_TS + 2 * MIN_MS, 6924.0, 6928.0, 6923.0, 6927.0, 8.0],
            ],
            dtype=np.float64,
        )

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=5)

            def on_data(self):
                n = self._ohlc.n_klines
                if n == 1:
                    self.sesh.buy("MES", volume=1)
                if n == 3:
                    pos = self.sesh.positions("MES")
                    if pos:
                        self.sesh.position_close(pos[0].ticket)

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines, timeframe=60_000, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "kline",
                initial_balance=10_000.0,
                commission=0.6,
                trade_on_close=True,
                position_mode=PositionMode.POSITION_MODE_NETTING,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert engine.strategy.sesh.account_info().balance == pytest.approx(
            10_033.80, abs=0.01
        )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SL / TP HIT IN INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSlTpInIntegration:
    def test_balance_kline_buy_and_close(self, mes_cfg):
        """
        Buy at 6921, close at 6927.
        Profit = (6927 - 6921) / 0.25 * 1.25 * 1 = 30.00
        Commissions = 0.6 * 2 = 1.20
        Final balance = 10_000 + 30.00 - 1.20 = 10_028.80
        """
        MIN_MS = 60_000
        klines = np.array(
            [
                [BASE_TS + 0 * MIN_MS, 6920.0, 6922.0, 6918.0, 6920.0, 10.0],
                [BASE_TS + 1 * MIN_MS, 6921.0, 6924.0, 6920.0, 6923.0, 12.0],
                [BASE_TS + 2 * MIN_MS, 6924.0, 6928.0, 6923.0, 6927.0, 8.0],
            ],
            dtype=np.float64,
        )

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=5)

            def on_data(self):
                n = self._ohlc.n_klines
                if n == 1:
                    self.sesh.buy("MES", volume=1)
                if n == 3:
                    pos = self.sesh.positions("MES")
                    if pos:
                        self.sesh.position_close(pos[0].ticket)

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines, timeframe=60_000, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "kline",
                initial_balance=10_000.0,
                commission=0.6,
                trade_on_close=True,
                position_mode=PositionMode.POSITION_MODE_NETTING,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        assert engine.strategy.sesh.account_info().balance == pytest.approx(
            10_028.80, abs=0.01
        )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SL / TP HIT IN INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSlTpInIntegration:
    def test_sl_hit_closes_position_tick(self, mes_cfg):
        """SL triggered by the broker in tick mode closes the position."""
        rows = [
            (BASE_TS + 0, 6920.0, 1.0),
            (BASE_TS + 1_000, 6921.0, 1.0),  # opening
            (BASE_TS + 2_000, 6914.75, 1.0),  # SL hit (sl=6915)
        ]
        ticks = make_tick_matrix(rows)

        class S(Strategy):
            def init(self):
                self._tp = self.subscribe_ticks("MES", window_size=5)

            def on_data(self):
                if self._tp.n_ticks == 1:
                    self.sesh.buy("MES", volume=1, sl=6915.0, tp=6940.0)

        engine = BacktestEngine.by_ticks(
            S(),
            data=(TickData("MES", ticks, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "tick",
                commission=0.6,
                position_mode=PositionMode.POSITION_MODE_NETTING,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert engine.strategy.sesh.positions("MES") == []
        deals = engine.strategy.sesh.deals("MES")
        assert "[sl" in deals[-1].comment.lower()

    def test_tp_hit_closes_position_kline(self, mes_cfg):
        """TP triggered in kline mode closes the position with profit."""
        klines = np.array(
            [
                [BASE_TS + 0 * MIN_MS, 6920.0, 6922.0, 6918.0, 6920.0, 10.0],
                [BASE_TS + 1 * MIN_MS, 6921.0, 6924.0, 6920.0, 6923.0, 12.0],
                [
                    BASE_TS + 2 * MIN_MS,
                    6924.0,
                    6932.0,
                    6923.0,
                    6930.0,
                    8.0,
                ],  # TP=6930 hit
            ],
            dtype=np.float64,
        )

        class S(Strategy):
            def init(self):
                self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=5)

            def on_data(self):
                if self._ohlc.n_klines == 1:
                    self.sesh.buy("MES", volume=1, sl=6910.0, tp=6930.0)

        engine = BacktestEngine.by_klines(
            S(),
            data=(KlineData("MES", klines, timeframe=60_000, tick_size=0.25),),
            sesh=SeshSimulatorBase(
                "kline",
                trade_on_close=True,
                commission=0.6,
            ),
        )
        engine.sesh.add_symbol(mes_cfg)
        engine.run()
        assert engine.strategy.sesh.positions("MES") == []
        deals = engine.strategy.sesh.deals("MES")
        assert "[tp" in deals[-1].comment.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: BUY_LIMIT/STOP INFERENCE WITHOUT CURRENT PRICE
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestInferenceWithoutCurrentPrice:
    """
    If _current_prices is empty (ask=0), SeshSimulatorBase must
    fall back to safe behavior: price < 0 or ask==0 → BUY_LIMIT by default.
    """

    def test_buy_without_current_price_uses_buy_limit(self, mes_cfg):
        """No previous ticks (ask=0) → BUY_LIMIT by default."""
        s = SeshSimulatorBase("tick", initial_balance=10_000.0)
        s.add_symbol(mes_cfg)
        # No ticks fed → ask=0
        assert s._buy_pending_type("MES", 6920.0) == OrderType.ORDER_TYPE_BUY_LIMIT

    def test_sell_without_current_price_uses_sell_limit(self, mes_cfg):
        """No previous ticks (bid=0) → SELL_LIMIT by default."""
        s = SeshSimulatorBase("tick", initial_balance=10_000.0)
        s.add_symbol(mes_cfg)
        assert s._sell_pending_type("MES", 6920.0) == OrderType.ORDER_TYPE_SELL_LIMIT


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "--tb=short"])

# endregion

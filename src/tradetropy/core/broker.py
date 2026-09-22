# region v12.0
"""
Broker simulator for backtesting - MetaTrader 5 style (Version 12.0).

ARCHITECTURE v12.0
===================
Clean separation in three classes with unique responsibilities:

  BrokerCore        - state, positions, orders, public API.
                      No knowledge of candles or ticks.

  KlineBroker       - extends BrokerCore for OHLC data.
                      Handles simulated spread and SL/TP worst-case intrabar.

  TickBroker        - extends BrokerCore for real tick data.
                      Direct bid/ask, SL/TP tick by tick, no look-ahead.

CHANGES vs v11.3
================
- BrokerSimulator removed - clean break, no compatibility alias.
- _tick_mode removed - each subclass knows what it is.
- trade_on_close exists only in KlineBroker.
- _simulate_spread exists only in KlineBroker.
- Duplicate method pairs (_check_sl_tp_intrabar/_check_sl_tp_tick, etc.)
  now live exclusively in the corresponding subclass.
- _should_execute_immediately() abstract hook in BrokerCore, implemented
  by each subclass according to its execution semantics.
- PendingMarketOrder, OrderTrigger, and all enums/dataclasses unchanged.
"""

from __future__ import annotations

import copy
import math
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple

from tradetropy.exceptions import MarginCallError


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS MT5
# ══════════════════════════════════════════════════════════════════════════════


class OrderType(IntEnum):
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7


class OrderTypeTime(IntEnum):
    ORDER_TIME_GTC = 0
    ORDER_TIME_DAY = 1
    ORDER_TIME_SPECIFIED = 2
    ORDER_TIME_SPECIFIED_DAY = 3


class OrderTypeFilling(IntEnum):
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2


class TradeRequestActions(IntEnum):
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_REMOVE = 8
    TRADE_ACTION_CLOSE_BY = 10


class CommissionType(IntEnum):
    COMMISSION_TYPE_MONEY = 0
    COMMISSION_TYPE_PERCENT = 1
    COMMISSION_TYPE_PER_MILLION = 2


class OrderState(IntEnum):
    ORDER_STATE_STARTED = 0
    ORDER_STATE_PLACED = 1
    ORDER_STATE_CANCELED = 2
    ORDER_STATE_PARTIAL = 3
    ORDER_STATE_FILLED = 4
    ORDER_STATE_REJECTED = 5
    ORDER_STATE_EXPIRED = 6


class PositionMode(IntEnum):
    POSITION_MODE_HEDGING = 0
    POSITION_MODE_NETTING = 1


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SymbolConfig:
    name: str
    tick_size: float
    tick_value: float
    contract_size: float
    digits: int
    avg_spread: float = 0.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01

    def normalize_price(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def calculate_profit(
        self, volume: float, price_open: float, price_close: float, is_buy: bool
    ) -> float:
        price_diff = (
            (price_close - price_open) if is_buy else (price_open - price_close)
        )
        return (price_diff / self.tick_size) * self.tick_value * volume


@dataclass
class Position:
    ticket: int
    symbol: str
    type: OrderType
    volume: float
    price_open: float
    time: datetime
    magic: int = 0
    comment: str = ""
    tp: float = 0.0
    sl: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0


@dataclass
class Order:
    ticket: int
    symbol: str
    type: OrderType
    volume: float
    price: float
    time: datetime
    state: OrderState = OrderState.ORDER_STATE_PLACED
    magic: int = 0
    comment: str = ""
    tp: float = 0.0
    sl: float = 0.0
    price_limit: float = 0.0
    type_time: OrderTypeTime = OrderTypeTime.ORDER_TIME_GTC
    type_filling: OrderTypeFilling = OrderTypeFilling.ORDER_FILLING_RETURN
    expiration: Optional[datetime] = None


@dataclass
class Deal:
    ticket: int
    position_id: int
    symbol: str
    type: OrderType
    volume: float
    price: float
    time: datetime
    commission: float
    profit: float
    order_id: int = 0
    swap: float = 0.0
    magic: int = 0
    comment: str = ""


@dataclass
class Trade(Deal):
    price_close: float = 0.0
    time_close: Optional[datetime] = None

    @property
    def pnl_net(self) -> float:
        return self.profit + self.commission


@dataclass
class TradeRequest:
    action: TradeRequestActions
    symbol: str
    volume: float = 0.0
    type: Optional[OrderType] = None
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 0
    magic: int = 0
    comment: str = ""
    type_time: OrderTypeTime = OrderTypeTime.ORDER_TIME_GTC
    type_filling: OrderTypeFilling = OrderTypeFilling.ORDER_FILLING_RETURN
    position: int = 0
    order: int = 0
    price_limit: float = 0.0
    expiration: Optional[datetime] = None


@dataclass
class TradeResult:
    retcode: int
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    comment: str = ""
    request_id: int = 0

    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_MARKET_CLOSED = 10018


@dataclass
class AccountInfo:
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    margin_free: float = 0.0
    margin_level: float = 0.0
    profit: float = 0.0


class PendingMarketOrder(NamedTuple):
    """
    Market order queued for execution on next candle/tick. Immutable.

    Attributes:
        request (TradeRequest): Original trade request
        order_ticket (int): Order ticket number
        close_ticket (int): Close position ticket (>0 for specific position close in hedging)
    """

    request: TradeRequest
    order_ticket: int
    close_ticket: int = 0



# ══════════════════════════════════════════════════════════════════════════════
# ORDER TRIGGER
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OrderTrigger:
    """
    Activation logic and execution price for pending orders.

    Attributes:
        condition (callable): Function that returns True when order should trigger
        execution_price (callable): Function that returns execution price
        position_type (OrderType): Type of position when triggered
    """

    condition: Callable[..., bool]
    execution_price: Callable[..., float]
    position_type: OrderType


# ══════════════════════════════════════════════════════════════════════════════
# BROKER CORE
# ══════════════════════════════════════════════════════════════════════════════


class BrokerCore:
    """
    Broker core: state, positions, orders, and public API.

    Manages account info, positions, orders, and deal execution.
    No knowledge of candles or ticks - that responsibility belongs to
    KlineBroker and TickBroker respectively.

    Subclasses must implement:
        _should_execute_immediately() -> bool
            Determines if market order executes at current price
            or is queued for next update.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission: float = 0.0,
        commission_type: CommissionType = CommissionType.COMMISSION_TYPE_MONEY,
        use_margin: bool = False,
        margin_rate: float = 0.01,
        position_mode: PositionMode = PositionMode.POSITION_MODE_NETTING,
        slippage_points: int = 0,
        stop_out_of_money: bool = True,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_amount = commission
        self.commission_type = commission_type
        self.use_margin = use_margin
        self.margin_rate = margin_rate
        self.position_mode = position_mode
        self.slippage_points = slippage_points
        self.stop_out_of_money = stop_out_of_money
        # Close still-open positions at the end of the backtest so they enter
        # trade-based stats (set by the session; mirrors backtesting.py's
        # finalize_trades). Read by engine.run() after the loop.
        self.finalize_trades: bool = False

        self.pending_market_orders: List[PendingMarketOrder] = []
        self.symbols: Dict[str, SymbolConfig] = {}
        self.positions: Dict[int, Position] = {}
        self.orders: Dict[int, Order] = {}
        self.orders_history: Dict[int, Order] = {}
        self.deals: List[Deal] = []
        self.trades: List[Trade] = []

        self._next_ticket: int = 1
        # Materialized current-bar datetime (explicit in live/replay). In
        # backtest it stays None and is built lazily from _current_ts_ms only
        # when something reads _current_time (a trade/order), so the per-bar
        # loop avoids a datetime.fromtimestamp on every bar.
        self._ct_dt: Optional[datetime] = None
        # Epoch-ms of the current bar/tick when the caller can supply it cheaply
        # (the backtest loop already has it as an int). Lets _record_equity skip
        # the datetime.timestamp()*1000 round-trip on the per-bar hot path and
        # is the source for lazily materializing _current_time.
        self._current_ts_ms: Optional[float] = None
        self._current_prices: Dict[str, Dict[str, float]] = {}

        # Equity curve - dynamic growth lists, converted to numpy in the property
        self._eq_ts: list = []  # timestamps in ms (float)
        self._eq_vals: list = []  # equity at each tick/bar (float)

        # Margin call / stop out levels (percentage of margin_level)
        self.margin_call_level: float = 100.0
        self.stop_out_level: float = 50.0

        # Backtest termination state. When the account is wiped (equity <= 0
        # with stop_out_of_money on) or a margin stop-out fires, the broker
        # liquidates open positions and sets _stopped so the engine loop ends
        # cleanly (no exception). _out_of_money marks the equity<=0 case.
        self._stopped: bool = False
        self._out_of_money: bool = False
        self._stop_ts_ms: "float | None" = None

        # True while engine runs on_data() - allows KlineBroker to distinguish
        # manual closes from on_data() (enqueue) vs from tests/scripts (immediate)
        self._in_on_data: bool = False

    @property
    def _current_time(self) -> "Optional[datetime]":
        """
        Current bar/tick time as a datetime.

        Returns the explicit datetime when set (live/replay), else materializes
        it lazily from ``_current_ts_ms`` (backtest) and caches it. This keeps
        the per-bar loop free of a datetime.fromtimestamp when nothing reads the
        time; a trade/order read pays for it only then.
        """
        dt = self._ct_dt
        if dt is None and self._current_ts_ms is not None:
            dt = datetime.fromtimestamp(self._current_ts_ms / 1000, tz=timezone.utc)
            self._ct_dt = dt
        return dt

    @_current_time.setter
    def _current_time(self, value: "Optional[datetime]") -> None:
        # An explicit datetime (or None) becomes authoritative; clear any stale
        # ms so it never overrides the just-set value. update_kline sets the ms
        # again right after when it has one.
        self._ct_dt = value
        self._current_ts_ms = None

    def _record_equity(self):
        """
        Record current equity (balance + unrealized P&L) to history curve.

        Called after each candle/tick update to track equity progression.
        Checks for margin call conditions if margin mode is enabled.
        """
        # Read the raw fields (not the _current_time property) so the per-bar
        # equity record never materializes the datetime.
        if self._ct_dt is None and self._current_ts_ms is None:
            return
        ts_ms = (
            self._current_ts_ms
            if self._current_ts_ms is not None
            else float(self._current_time.timestamp() * 1000)
        )
        # When flat (the common bar-to-bar case) equity is just the balance;
        # skip building a generator + sum() over an empty position book.
        if self.positions:
            equity = self.balance + sum(
                p.profit for p in self.positions.values()
            )
        else:
            equity = self.balance
        self._eq_ts.append(ts_ms)
        self._eq_vals.append(equity)

        # Out-of-money guard (on by default): a wiped account (equity <= 0)
        # liquidates all positions and stops the backtest cleanly, mirroring
        # backtesting.py. Only fires in the degenerate blow-up case, so
        # well-behaved strategies are unaffected.
        if self.stop_out_of_money and not self._stopped and equity <= 0:
            self._liquidate_and_stop(out_of_money=True)
            return

        # Margin call - only if use_margin active and positions open
        if self.use_margin and self.positions:
            info = self.account_info()
            if info.margin > 0 and info.margin_level <= self.stop_out_level:
                self._execute_stop_out()

    def finalize(self) -> None:
        """
        Close all open positions at the current price (end-of-backtest).

        Used by finalize_trades: the closed positions become Trades in
        get_trades() and enter the trade-based stats, mirroring backtesting.py's
        finalize_trades=True. Does not flag the run as stopped.
        """
        for ticket in list(self.positions.keys()):
            self._close_position_at_current_price(ticket)

    def _liquidate_and_stop(self, *, out_of_money: bool):
        """
        Close all open positions at current price and flag the backtest to end.

        Used by the out-of-money guard and the margin stop-out. Does NOT raise:
        the engine loop checks ``_stopped`` and terminates cleanly, so results
        (equity curve, trades, stats) are produced normally for the run up to
        the liquidation - just like backtesting.py's out-of-money handling.

        Args:
            out_of_money (bool): True for the equity<=0 wipe, False for a
                margin stop-out.
        """
        for ticket in list(self.positions.keys()):
            self._close_position_at_current_price(ticket)
        self._stopped = True
        self._out_of_money = out_of_money
        if self._current_ts_ms is not None:
            self._stop_ts_ms = float(self._current_ts_ms)
        elif self._eq_ts:
            self._stop_ts_ms = float(self._eq_ts[-1])
        # Freeze the recorded equity at zero for the liquidation bar (the
        # account is considered wiped), matching backtesting.py.
        if self._eq_vals:
            self._eq_vals[-1] = 0.0

    def _execute_stop_out(self):
        """
        Liquidate all open positions due to a margin stop out.

        Closes all positions at current prices and stops the backtest cleanly
        (no exception), so a stop-out during optimize/pool cannot kill a worker.
        """
        self._liquidate_and_stop(out_of_money=False)

    @property
    def equity_curve(self) -> "pd.Series":
        """
        Equity curve as pd.Series with DatetimeIndex UTC.
        Built in O(N) the first time it is accessed after the backtest.
        Returns a view of the data - no copy of the underlying arrays.
        """
        import pandas as pd

        if not self._eq_ts:
            return pd.Series([], dtype=float, name="equity")
        ts = pd.to_datetime(self._eq_ts, unit="ms", utc=True)
        return pd.Series(self._eq_vals, index=ts, name="equity", dtype=float)

    # ── Configuration ─────────────────────────────────────────────────────────

    def add_symbol(self, config: SymbolConfig):
        self.symbols[config.name] = config

    # ── Subclass hook ─────────────────────────────────────────────────────────

    @abstractmethod
    def _should_execute_immediately(self) -> bool:
        """
        True  -> the market order executes at current price (trade_on_close in KlineBroker).
        False -> queued for execution on the next update().
        """

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_ticket(self) -> int:
        t = self._next_ticket
        self._next_ticket += 1
        return t

    def _get_symbol_config(self, symbol: str) -> SymbolConfig:
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolConfig(
                name=symbol,
                tick_size=0.01,
                tick_value=0.01,
                contract_size=1.0,
                digits=2,
                avg_spread=0.0,
            )
        return self.symbols[symbol]

    def _apply_slippage(self, symbol: str, price: float, is_buy: bool) -> float:
        if self.slippage_points <= 0:
            return price
        config = self._get_symbol_config(symbol)
        delta = self.slippage_points * config.tick_size
        return config.normalize_price(price + delta if is_buy else price - delta)

    def _calculate_commission(self, symbol: str, volume: float, price: float) -> float:
        config = self._get_symbol_config(symbol)
        if self.commission_type == CommissionType.COMMISSION_TYPE_MONEY:
            return -abs(self.commission_amount * volume)
        if self.commission_type == CommissionType.COMMISSION_TYPE_PERCENT:
            return -abs(
                volume * price * config.contract_size * (self.commission_amount / 100.0)
            )
        if self.commission_type == CommissionType.COMMISSION_TYPE_PER_MILLION:
            return -abs(
                (volume * price * config.contract_size / 1_000_000.0)
                * self.commission_amount
            )
        return 0.0

    def _calculate_required_margin(
        self, symbol: str, volume: float, price: float
    ) -> float:
        if not self.use_margin:
            return 0.0
        config = self._get_symbol_config(symbol)
        return volume * price * config.contract_size * self.margin_rate

    def _update_balance(self, amount: float):
        self.balance += amount

    def _create_deal(
        self,
        position_id: int,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: float,
        commission: float,
        profit: float,
        magic: int,
        comment: str,
        order_id: int = 0,
    ):
        self.deals.append(
            Deal(
                ticket=self._get_ticket(),
                position_id=position_id,
                symbol=symbol,
                type=order_type,
                volume=volume,
                price=price,
                time=self._current_time,
                commission=commission,
                profit=profit,
                order_id=order_id,
                magic=magic,
                comment=comment,
            )
        )

    def _move_order_to_history(self, order: Order):
        self.orders_history[order.ticket] = copy.copy(order)

    def _update_order_state(self, ticket: int, state: OrderState):
        if ticket in self.orders:
            self.orders[ticket].state = state
        if ticket in self.orders_history:
            self.orders_history[ticket].state = state

    def _validate_sl_tp(
        self,
        symbol: str,
        order_type: OrderType,
        current_price: float,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> Tuple[bool, str]:
        is_buy = order_type in (
            OrderType.ORDER_TYPE_BUY,
            OrderType.ORDER_TYPE_BUY_LIMIT,
            OrderType.ORDER_TYPE_BUY_STOP,
            OrderType.ORDER_TYPE_BUY_STOP_LIMIT,
        )
        if sl is not None and sl > 0.0:
            if is_buy and sl >= current_price:
                return False, "SL for BUY must be below current price"
            if not is_buy and sl <= current_price:
                return False, "SL for SELL must be above current price"
        if tp is not None and tp > 0.0:
            if is_buy and tp <= current_price:
                return False, "TP for BUY must be above current price"
            if not is_buy and tp >= current_price:
                return False, "TP for SELL must be below current price"
        return True, ""

    def _calculate_position_profit(
        self, position: Position, config: "SymbolConfig | None" = None
    ) -> float:
        if position.symbol not in self._current_prices:
            return 0.0
        if config is None:
            config = self._get_symbol_config(position.symbol)
        current = self._current_prices[position.symbol]
        price = (
            current["bid"]
            if position.type == OrderType.ORDER_TYPE_BUY
            else current["ask"]
        )
        return config.calculate_profit(
            position.volume,
            position.price_open,
            price,
            is_buy=(position.type == OrderType.ORDER_TYPE_BUY),
        )

    # ── Position management ───────────────────────────────────────────────────

    def _process_position(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: float,
        commission: float,
        magic: int = 0,
        comment: str = "",
        tp: float = 0.0,
        sl: float = 0.0,
        order_id: int = 0,
    ) -> Optional[Position]:
        tp = tp or 0.0
        sl = sl or 0.0

        if self.position_mode == PositionMode.POSITION_MODE_HEDGING:
            return self._create_new_position(
                symbol,
                order_type,
                volume,
                price,
                commission,
                magic,
                comment,
                tp,
                sl,
                order_id,
            )

        existing = next(
            (p for p in self.positions.values() if p.symbol == symbol), None
        )
        if existing is None:
            return self._create_new_position(
                symbol,
                order_type,
                volume,
                price,
                commission,
                magic,
                comment,
                tp,
                sl,
                order_id,
            )
        return self._consolidate_netting_position(
            existing, order_type, volume, price, commission, sl, tp
        )

    def _create_new_position(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: float,
        commission: float,
        magic: int,
        comment: str,
        tp: float,
        sl: float,
        order_id: int = 0,
    ) -> Position:
        ticket = self._get_ticket()
        self._update_balance(commission)
        position = Position(
            ticket=ticket,
            symbol=symbol,
            type=order_type,
            volume=volume,
            price_open=price,
            time=self._current_time,
            magic=magic,
            comment=comment,
            tp=tp,
            sl=sl,
            commission=commission,
        )
        self.positions[ticket] = position
        self._create_deal(
            ticket,
            symbol,
            order_type,
            volume,
            price,
            commission,
            0.0,
            magic,
            comment,
            order_id,
        )
        return position

    def _consolidate_netting_position(
        self,
        existing: Position,
        new_type: OrderType,
        new_volume: float,
        new_price: float,
        new_commission: float,
        new_sl: float = 0.0,
        new_tp: float = 0.0,
    ) -> Optional[Position]:
        config = self._get_symbol_config(existing.symbol)

        if existing.type == new_type:
            self._update_balance(new_commission)
            total = existing.volume + new_volume
            existing.price_open = (
                existing.price_open * existing.volume + new_price * new_volume
            ) / total
            existing.volume = total
            existing.commission += new_commission
            if new_sl > 0.0:
                existing.sl = config.normalize_price(new_sl)
            if new_tp > 0.0:
                existing.tp = config.normalize_price(new_tp)
            self._create_deal(
                existing.ticket,
                existing.symbol,
                new_type,
                new_volume,
                new_price,
                new_commission,
                0.0,
                existing.magic,
                "Netting increase",
            )
            return existing

        return self._handle_opposite_direction(
            existing, new_type, new_volume, new_price, new_commission, new_sl, new_tp
        )

    def _handle_opposite_direction(
        self,
        existing: Position,
        new_type: OrderType,
        new_volume: float,
        new_price: float,
        new_commission: float,
        new_sl: float,
        new_tp: float,
    ) -> Optional[Position]:
        config = self._get_symbol_config(existing.symbol)
        is_buy_existing = existing.type == OrderType.ORDER_TYPE_BUY

        if new_volume <= existing.volume:
            close_vol = new_volume
            profit_bruto = config.calculate_profit(
                close_vol, existing.price_open, new_price, is_buy=is_buy_existing
            )
            self._update_balance(profit_bruto + new_commission)
            label = (
                "Netting close" if new_volume == existing.volume else "Netting partial"
            )
            self._create_deal(
                existing.ticket,
                existing.symbol,
                new_type,
                close_vol,
                new_price,
                new_commission,
                profit_bruto,
                existing.magic,
                label,
            )

            # Register closed trade
            opening_commission = existing.commission * (close_vol / existing.volume)
            self.trades.append(Trade(
                ticket      = existing.ticket,
                position_id = existing.ticket,
                symbol      = existing.symbol,
                type        = existing.type,
                volume      = close_vol,
                price       = existing.price_open,
                time        = existing.time,
                commission  = opening_commission + new_commission,
                profit      = profit_bruto,
                magic       = existing.magic,
                comment     = label,
                price_close = new_price,
                time_close  = self._current_time,
            ))

            if new_volume == existing.volume:
                del self.positions[existing.ticket]
                return None

            existing.commission -= opening_commission
            existing.volume -= close_vol
            return existing

        close_vol = existing.volume
        rev_vol = new_volume - existing.volume
        profit_bruto = config.calculate_profit(
            close_vol, existing.price_open, new_price, is_buy=is_buy_existing
        )
        close_comm = new_commission * (close_vol / new_volume)
        self._update_balance(profit_bruto + close_comm)
        self._create_deal(
            existing.ticket,
            existing.symbol,
            new_type,
            close_vol,
            new_price,
            close_comm,
            profit_bruto,
            existing.magic,
            "Netting close",
        )

        # Register closed trade (full position reversed)
        self.trades.append(Trade(
            ticket      = existing.ticket,
            position_id = existing.ticket,
            symbol      = existing.symbol,
            type        = existing.type,
            volume      = close_vol,
            price       = existing.price_open,
            time        = existing.time,
            commission  = existing.commission + close_comm,
            profit      = profit_bruto,
            magic       = existing.magic,
            comment     = "Netting close",
            price_close = new_price,
            time_close  = self._current_time,
        ))

        old_symbol, old_magic = existing.symbol, existing.magic
        del self.positions[existing.ticket]

        rev_comm = new_commission * (rev_vol / new_volume)
        return self._create_new_position(
            old_symbol,
            new_type,
            rev_vol,
            new_price,
            rev_comm,
            old_magic,
            "Netting reverse",
            config.normalize_price(new_tp) if new_tp > 0.0 else 0.0,
            config.normalize_price(new_sl) if new_sl > 0.0 else 0.0,
        )

    def _close_position_internal(
        self,
        ticket: int,
        volume: float,
        close_price: float,
        comment: str = "",
        order_ticket: Optional[int] = None,
        create_close_order: bool = True,
    ):
        if ticket not in self.positions:
            return
        position = self.positions[ticket]
        config = self._get_symbol_config(position.symbol)
        volume = min(volume, position.volume)
        close_price = config.normalize_price(close_price)

        if create_close_order and order_ticket is None:
            order_ticket = self._get_ticket()
            close_type = (
                OrderType.ORDER_TYPE_SELL
                if position.type == OrderType.ORDER_TYPE_BUY
                else OrderType.ORDER_TYPE_BUY
            )
            self._move_order_to_history(
                Order(
                    ticket=order_ticket,
                    symbol=position.symbol,
                    type=close_type,
                    volume=volume,
                    price=close_price,
                    time=self._current_time,
                    state=OrderState.ORDER_STATE_FILLED,
                    magic=position.magic,
                    comment=comment,
                )
            )

        commission = self._calculate_commission(position.symbol, volume, close_price)
        profit_bruto = config.calculate_profit(
            volume,
            position.price_open,
            close_price,
            is_buy=(position.type == OrderType.ORDER_TYPE_BUY),
        )
        self._update_balance(profit_bruto + commission)
        self._create_deal(
            ticket,
            position.symbol,
            OrderType.ORDER_TYPE_SELL
            if position.type == OrderType.ORDER_TYPE_BUY
            else OrderType.ORDER_TYPE_BUY,
            volume,
            close_price,
            commission,
            profit_bruto,
            position.magic,
            comment,
            order_ticket or 0,
        )

        # ── Register closed trade ────────────────────────────────────────────
        opening_commission    = position.commission * (volume / position.volume)
        commission_total = opening_commission + commission
        self.trades.append(Trade(
            ticket      = ticket,
            position_id = ticket,
            symbol      = position.symbol,
            type        = position.type,
            volume      = volume,
            price       = position.price_open,
            time        = position.time,
            commission  = commission_total,         
            profit      = profit_bruto,
            order_id    = order_ticket or 0,
            magic       = position.magic,
            comment     = comment,
            price_close = close_price,
            time_close  = self._current_time,
        ))

        if volume >= position.volume:
            del self.positions[ticket]
        else:
            position.commission -= opening_commission
            position.volume -= volume

    def _close_position_at_current_price(
        self,
        ticket: int,
        volume: Optional[float] = None,
    ) -> bool:
        if ticket not in self.positions:
            return False
        position = self.positions[ticket]
        if position.symbol not in self._current_prices:
            return False

        vol = min(volume or position.volume, position.volume)
        current = self._current_prices[position.symbol]
        price = (
            current["bid"]
            if position.type == OrderType.ORDER_TYPE_BUY
            else current["ask"]
        )
        price = self._apply_slippage(
            position.symbol, price, position.type == OrderType.ORDER_TYPE_SELL
        )

        order_ticket = self._get_ticket()
        self._move_order_to_history(
            Order(
                ticket=order_ticket,
                symbol=position.symbol,
                type=(
                    OrderType.ORDER_TYPE_SELL
                    if position.type == OrderType.ORDER_TYPE_BUY
                    else OrderType.ORDER_TYPE_BUY
                ),
                volume=vol,
                price=price,
                time=self._current_time,
                state=OrderState.ORDER_STATE_FILLED,
                magic=position.magic,
                comment=f"close position #{ticket}",
            )
        )
        self._close_position_internal(
            ticket, vol, price, "Manual close", order_ticket, create_close_order=False
        )
        return True

    def _expire_pending_orders(self):
        if self._current_time is None:
            return
        for ticket, order in list(self.orders.items()):
            expired = False
            if order.type_time == OrderTypeTime.ORDER_TIME_DAY:
                expired = self._current_time.date() > order.time.date()
            elif order.type_time == OrderTypeTime.ORDER_TIME_SPECIFIED:
                expired = bool(
                    order.expiration and self._current_time >= order.expiration
                )
            elif order.type_time == OrderTypeTime.ORDER_TIME_SPECIFIED_DAY:
                expired = bool(
                    order.expiration
                    and self._current_time.date() > order.expiration.date()
                )
            if expired:
                order.state = OrderState.ORDER_STATE_EXPIRED
                self._update_order_state(ticket, OrderState.ORDER_STATE_EXPIRED)
                del self.orders[ticket]

    # ── order_send and handlers ───────────────────────────────────────────────

    def order_send(self, request: TradeRequest) -> TradeResult:
        if request.action not in (
            TradeRequestActions.TRADE_ACTION_SLTP,
            TradeRequestActions.TRADE_ACTION_MODIFY,
            TradeRequestActions.TRADE_ACTION_REMOVE,
        ):
            if request.volume <= 0:
                return TradeResult(
                    retcode=TradeResult.TRADE_RETCODE_INVALID_VOLUME,
                    comment="Invalid volume",
                )

        handlers = {
            TradeRequestActions.TRADE_ACTION_DEAL: self._handle_market_order,
            TradeRequestActions.TRADE_ACTION_PENDING: self._handle_pending_order,
            TradeRequestActions.TRADE_ACTION_SLTP: self._handle_modify_sltp,
            TradeRequestActions.TRADE_ACTION_MODIFY: self._handle_modify_order,
            TradeRequestActions.TRADE_ACTION_REMOVE: self._handle_remove_order,
        }
        handler = handlers.get(request.action)
        if handler:
            return handler(request)
        return TradeResult(
            retcode=TradeResult.TRADE_RETCODE_INVALID, comment="Unknown action"
        )

    def _handle_market_order(self, request: TradeRequest) -> TradeResult:
        request.sl = request.sl or 0.0
        request.tp = request.tp or 0.0

        if request.type not in (OrderType.ORDER_TYPE_BUY, OrderType.ORDER_TYPE_SELL):
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID,
                comment="Invalid order type for DEAL",
            )
        if request.symbol not in self._current_prices:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_MARKET_CLOSED, comment="No price data"
            )

        current = self._current_prices[request.symbol]
        ref_price = (
            current["ask"]
            if request.type == OrderType.ORDER_TYPE_BUY
            else current["bid"]
        )
        config = self._get_symbol_config(request.symbol)
        ref_price = config.normalize_price(ref_price)

        valid, msg = self._validate_sl_tp(
            request.symbol, request.type, ref_price, request.sl, request.tp
        )
        if not valid:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID_STOPS, comment=msg
            )

        if self.use_margin:
            req_margin = self._calculate_required_margin(
                request.symbol, request.volume, ref_price
            )
            if req_margin > self.account_info().margin_free:
                return TradeResult(
                    retcode=TradeResult.TRADE_RETCODE_NO_MONEY,
                    comment="Not enough margin",
                )

        order_ticket = self._get_ticket()
        market_order = Order(
            ticket=order_ticket,
            symbol=request.symbol,
            type=request.type,
            volume=request.volume,
            price=ref_price,
            time=self._current_time,
            state=OrderState.ORDER_STATE_PLACED,
            magic=request.magic,
            comment=request.comment,
            tp=request.tp,
            sl=request.sl,
        )
        self._move_order_to_history(market_order)

        if self._should_execute_immediately():
            exec_price = self._apply_slippage(
                request.symbol, ref_price, request.type == OrderType.ORDER_TYPE_BUY
            )
            commission = self._calculate_commission(
                request.symbol, request.volume, exec_price
            )
            self.orders_history[order_ticket].price = exec_price
            self.orders_history[order_ticket].state = OrderState.ORDER_STATE_FILLED

            position = self._process_position(
                request.symbol,
                request.type,
                request.volume,
                exec_price,
                commission,
                request.magic,
                request.comment,
                request.tp,
                request.sl,
                order_ticket,
            )
            if position:
                position.profit = self._calculate_position_profit(position)

            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_DONE,
                deal=self._next_ticket - 1,
                order=order_ticket,
                volume=request.volume,
                price=exec_price,
                comment="Done",
            )

        pmo = PendingMarketOrder(request=request, order_ticket=order_ticket)
        self.pending_market_orders.append(pmo)
        self._on_market_order_enqueued(pmo)

        return TradeResult(
            retcode=TradeResult.TRADE_RETCODE_PLACED,
            order=order_ticket,
            volume=request.volume,
            price=ref_price,
            comment="Placed (will execute at next bar open or next tick)",
        )

    def _on_market_order_enqueued(self, pmo: PendingMarketOrder):
        """
        Hook called when a market order is enqueued.
        TickBroker overrides this to handle the 'ready on next tick' logic.
        KlineBroker does not need to do anything special.
        """

    def _handle_pending_order(self, request: TradeRequest) -> TradeResult:
        request.sl = request.sl or 0.0
        request.tp = request.tp or 0.0

        if request.type in (OrderType.ORDER_TYPE_BUY, OrderType.ORDER_TYPE_SELL):
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID,
                comment="Use DEAL action for market orders",
            )

        config = self._get_symbol_config(request.symbol)
        price = config.normalize_price(request.price)
        price_limit = (
            config.normalize_price(request.price_limit)
            if request.price_limit > 0
            else 0.0
        )

        valid, msg = self._validate_sl_tp(
            request.symbol, request.type, price, request.sl, request.tp
        )
        if not valid:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID_STOPS, comment=msg
            )

        ticket = self._get_ticket()
        order = Order(
            ticket=ticket,
            symbol=request.symbol,
            type=request.type,
            volume=request.volume,
            price=price,
            time=self._current_time,
            magic=request.magic,
            comment=request.comment,
            tp=request.tp,
            sl=request.sl,
            price_limit=price_limit,
            type_time=request.type_time,
            type_filling=request.type_filling,
            expiration=request.expiration,
        )
        self.orders[ticket] = order
        self._move_order_to_history(order)
        return TradeResult(
            retcode=TradeResult.TRADE_RETCODE_PLACED,
            order=ticket,
            volume=request.volume,
            price=price,
            comment="Placed",
        )

    def _handle_modify_sltp(self, request: TradeRequest) -> TradeResult:
        if request.position not in self.positions:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID, comment="Position not found"
            )
        position = self.positions[request.position]
        config = self._get_symbol_config(position.symbol)

        if position.symbol in self._current_prices:
            current = self._current_prices[position.symbol]
            price = (
                current["bid"]
                if position.type == OrderType.ORDER_TYPE_BUY
                else current["ask"]
            )
            valid, msg = self._validate_sl_tp(
                position.symbol,
                position.type,
                price,
                request.sl if request.sl > 0 else position.sl,
                request.tp if request.tp > 0 else position.tp,
            )
            if not valid:
                return TradeResult(
                    retcode=TradeResult.TRADE_RETCODE_INVALID_STOPS, comment=msg
                )

        if request.sl > 0:
            position.sl = config.normalize_price(request.sl)
        elif request.sl == 0:
            position.sl = 0.0
        if request.tp > 0:
            position.tp = config.normalize_price(request.tp)
        elif request.tp == 0:
            position.tp = 0.0

        return TradeResult(
            retcode=TradeResult.TRADE_RETCODE_DONE, comment="SL/TP modified"
        )

    def _handle_modify_order(self, request: TradeRequest) -> TradeResult:
        if request.order not in self.orders:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID, comment="Order not found"
            )
        order = self.orders[request.order]
        config = self._get_symbol_config(order.symbol)

        new_price = (
            config.normalize_price(request.price) if request.price > 0 else order.price
        )
        valid, msg = self._validate_sl_tp(
            order.symbol,
            order.type,
            new_price,
            request.sl if request.sl > 0 else order.sl,
            request.tp if request.tp > 0 else order.tp,
        )
        if not valid:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID_STOPS, comment=msg
            )

        if request.price > 0:
            order.price = new_price
        if request.price_limit > 0:
            order.price_limit = config.normalize_price(request.price_limit)
        if request.sl > 0:
            order.sl = config.normalize_price(request.sl)
        if request.tp > 0:
            order.tp = config.normalize_price(request.tp)
        if request.volume > 0:
            order.volume = request.volume

        return TradeResult(
            retcode=TradeResult.TRADE_RETCODE_DONE, comment="Order modified"
        )

    def _handle_remove_order(self, request: TradeRequest) -> TradeResult:
        if request.order not in self.orders:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID, comment="Order not found"
            )
        self._update_order_state(request.order, OrderState.ORDER_STATE_CANCELED)
        del self.orders[request.order]
        return TradeResult(
            retcode=TradeResult.TRADE_RETCODE_DONE, comment="Order removed"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def position_close(self, ticket: int, volume: Optional[float] = None) -> bool:
        return self._close_position_at_current_price(ticket, volume)

    def position_modify(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        if ticket not in self.positions:
            return False
        position = self.positions[ticket]
        config = self._get_symbol_config(position.symbol)

        if position.symbol in self._current_prices:
            current = self._current_prices[position.symbol]
            price = (
                current["bid"]
                if position.type == OrderType.ORDER_TYPE_BUY
                else current["ask"]
            )
            valid, _ = self._validate_sl_tp(
                position.symbol,
                position.type,
                price,
                sl if sl > 0 else position.sl,
                tp if tp > 0 else position.tp,
            )
            if not valid:
                return False

        if sl > 0:
            position.sl = config.normalize_price(sl)
        elif sl == 0:
            position.sl = 0.0
        if tp > 0:
            position.tp = config.normalize_price(tp)
        elif tp == 0:
            position.tp = 0.0
        return True

    def order_delete(self, ticket: int) -> bool:
        if ticket in self.orders:
            order = self.orders[ticket]
            canceled_order = copy.copy(order)
            canceled_order.state = OrderState.ORDER_STATE_CANCELED
            self.orders_history[ticket] = canceled_order
            del self.orders[ticket]
            return True
        return False

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if symbol is None:
            return list(self.positions.values())
        return [p for p in self.positions.values() if p.symbol == symbol]

    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        if symbol is None:
            return list(self.orders.values())
        return [o for o in self.orders.values() if o.symbol == symbol]

    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        if symbol is None:
            return list(self.orders_history.values())
        return [o for o in self.orders_history.values() if o.symbol == symbol]

    def get_deal_history(self, symbol: Optional[str] = None) -> List[Deal]:
        if symbol is None:
            return self.deals.copy()
        return [d for d in self.deals if d.symbol == symbol]

    def get_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        if symbol is None:
            return self.trades.copy()
        return [t for t in self.trades if t.symbol == symbol]

    # Legacy alias
    trades_get = get_trades

    def account_info(self) -> AccountInfo:
        unrealized = sum(p.profit for p in self.positions.values())
        equity = self.balance + unrealized
        margin = (
            sum(
                self._calculate_required_margin(p.symbol, p.volume, p.price_open)
                for p in self.positions.values()
            )
            if self.use_margin
            else 0.0
        )
        margin_free = equity - margin
        margin_level = (equity / margin * 100.0) if margin > 0 else 0.0
        return AccountInfo(
            balance=round(self.balance, 2),
            equity=round(equity, 2),
            margin=round(margin, 2),
            margin_free=round(margin_free, 2),
            margin_level=round(margin_level, 2),
            profit=round(unrealized, 2),
        )

    def reset(self):
        self.balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.orders_history.clear()
        self.deals.clear()
        self.trades.clear()
        self._next_ticket = 1
        self._current_time = None
        self._current_ts_ms = None
        self._current_prices.clear()
        self.pending_market_orders.clear()
        self._eq_ts.clear()
        self._eq_vals.clear()
        self._in_on_data = False
        self._stopped = False
        self._out_of_money = False
        self._stop_ts_ms = None
        self._on_reset()

    def _on_reset(self):
        """Hook for subclasses to clean their own state in reset()."""


# ══════════════════════════════════════════════════════════════════════════════
# BAR BROKER
# ══════════════════════════════════════════════════════════════════════════════


class KlineBroker(BrokerCore):
    """
    Broker for OHLC (bar) data.

    - Simulated spread from the bar base price.
    - SL/TP evaluated worst-case: if both are touched in the same bar,
      SL is assumed to have occurred first (more conservative result).
    - trade_on_close=True  -> market orders execute at the current bar close.
    - trade_on_close=False -> market orders execute at the next bar open.

    Known limitation: positions opened by pending orders in a bar
    do not have their SL/TP evaluated until the next bar, because with OHLC
    data it is not possible to know the exact intrabar chronological order.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission: float = 0.0,
        commission_type: CommissionType = CommissionType.COMMISSION_TYPE_MONEY,
        use_spread: bool = False,
        use_margin: bool = False,
        margin_rate: float = 0.01,
        trade_on_close: bool = False,
        execution_price_source: Literal["close", "mid"] = "close",
        position_mode: PositionMode = PositionMode.POSITION_MODE_NETTING,
        slippage_points: int = 0,
        stop_out_of_money: bool = True,
    ):
        super().__init__(
            initial_balance=initial_balance,
            commission=commission,
            commission_type=commission_type,
            use_margin=use_margin,
            margin_rate=margin_rate,
            position_mode=position_mode,
            slippage_points=slippage_points,
            stop_out_of_money=stop_out_of_money,
        )
        self.use_spread = use_spread
        self.trade_on_close = trade_on_close
        self.execution_price_source = execution_price_source

    # ── Hook ──────────────────────────────────────────────────────────────────

    def _should_execute_immediately(self) -> bool:
        return self.trade_on_close

    # ── Simulated spread ──────────────────────────────────────────────────────

    def _simulate_spread(self, symbol: str, base_price: float) -> Tuple[float, float]:
        config = self._get_symbol_config(symbol)
        if self.use_spread and config.avg_spread > 0:
            half = config.avg_spread / 2.0
            ask = config.normalize_price(
                base_price + math.ceil(half) * config.tick_size
            )
            bid = config.normalize_price(
                base_price - math.floor(half) * config.tick_size
            )
        else:
            bid = ask = config.normalize_price(base_price)
        return bid, ask

    # ── update_kline ──────────────────────────────────────────────────────────

    def update_kline(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0,
        tick_volume: int = 0,
        *,
        _prenormalized: bool = False,
        _ts_ms: "float | None" = None,
    ):
        self._current_time = timestamp
        # Backtest supplies the epoch-ms directly (it already has it as an int),
        # so _record_equity avoids a datetime.timestamp()*1000 round-trip per
        # bar. None (live/replay) keeps the previous behaviour.
        self._current_ts_ms = _ts_ms
        config = self._get_symbol_config(symbol)

        # ``_prenormalized`` lets the backtest loop round OHLC to tick_size ONCE
        # (vectorized over the whole matrix) instead of 4 normalize_price calls
        # per bar. normalize_price is idempotent and deterministic, so the
        # pre-rounded values are bit-identical to normalizing here (guarded by
        # test_bbroker / test_parity_engines).
        if not _prenormalized:
            open_price = config.normalize_price(open_price)
            high = config.normalize_price(high)
            low = config.normalize_price(low)
            close = config.normalize_price(close)

        if not self.trade_on_close:
            if self.pending_market_orders:
                self._execute_pending_market_orders(symbol, open_price)

        if self.orders:
            self._expire_pending_orders()

        base = close if self.execution_price_source == "close" else (high + low) / 2.0
        # Inline the spread from the config already fetched above, instead of
        # calling _simulate_spread (which would re-run _get_symbol_config and
        # add a frame every bar). Short-circuit the no-spread case (the common
        # backtest default) to a single normalize_price. Identical result to
        # _simulate_spread (guarded by test_bbroker / test_parity_engines).
        if self.use_spread and config.avg_spread > 0:
            half = config.avg_spread / 2.0
            ask = config.normalize_price(base + math.ceil(half) * config.tick_size)
            bid = config.normalize_price(base - math.floor(half) * config.tick_size)
        else:
            bid = ask = config.normalize_price(base)
        # Reuse the per-symbol price dict (mutate in place) instead of building
        # two fresh dicts (outer + nested ohlc) every bar. Every reader
        # (_calculate_position_profit, order handlers, session._current_price,
        # _check_sl_tp_intrabar) fetches the fields immediately and never holds
        # the dict across bars, so overwriting in place is behavior-identical
        # while cutting two per-bar allocations (guarded by test_bbroker /
        # test_parity_engines).
        cp = self._current_prices.get(symbol)
        if cp is None:
            self._current_prices[symbol] = {
                "bid": bid,
                "ask": ask,
                "ohlc": {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                },
            }
        else:
            cp["bid"] = bid
            cp["ask"] = ask
            ohlc = cp["ohlc"]
            ohlc["open"] = open_price
            ohlc["high"] = high
            ohlc["low"] = low
            ohlc["close"] = close

        # Order / position machinery only runs when there is something to do.
        # When flat with no working orders (the common case bar-to-bar) all of
        # these are no-ops, so guarding avoids building trigger tables and
        # scanning empty collections every bar. Behavior is unchanged: each
        # guarded call does nothing when its collection is empty.
        if self.positions:
            self._check_sl_tp_intrabar(symbol, high, low)

        if self.orders:
            nuevas = self._process_pending_orders_intrabar(
                symbol, open_price, high, low
            )
            if nuevas:
                # Positions opened by pending orders in this same bar
                # also need their SL/TP evaluated intrabar (e.g. limit order
                # executed at the bar low, and the TP is also touched).
                self._check_sl_tp_intrabar(symbol, high, low)

        if self.positions:
            for p in self.positions.values():
                if p.symbol == symbol:
                    p.profit = self._calculate_position_profit(p, config)

        self._record_equity()

    # ── Execution of enqueued market orders ───────────────────────────────────

    def _execute_pending_market_orders(self, symbol: str, open_price: float):
        orders = [
            pmo for pmo in self.pending_market_orders if pmo.request.symbol == symbol
        ]
        self.pending_market_orders = [
            pmo for pmo in self.pending_market_orders if pmo.request.symbol != symbol
        ]

        for pmo in orders:
            req = pmo.request
            bid, ask = self._simulate_spread(symbol, open_price)
            config = self._get_symbol_config(symbol)
            exec_price = config.normalize_price(
                ask if req.type == OrderType.ORDER_TYPE_BUY else bid
            )
            exec_price = self._apply_slippage(
                req.symbol, exec_price, req.type == OrderType.ORDER_TYPE_BUY
            )

            if pmo.order_ticket in self.orders_history:
                self.orders_history[pmo.order_ticket].price = exec_price
                self.orders_history[
                    pmo.order_ticket
                ].state = OrderState.ORDER_STATE_FILLED

            if pmo.close_ticket > 0:
                # Enqueued close from on_data(): close the original position
                # directly using _close_position_internal to preserve the
                # original ticket and prevent _process_position from opening an
                # opposite position in hedging mode.
                # If the position was already closed by SL/TP in the previous bar,
                # it is simply ignored without error.
                if pmo.close_ticket in self.positions:
                    vol = min(req.volume, self.positions[pmo.close_ticket].volume)
                    self._close_position_internal(
                        pmo.close_ticket,
                        vol,
                        exec_price,
                        req.comment,
                        pmo.order_ticket,
                        create_close_order=False,
                    )
            else:
                commission = self._calculate_commission(req.symbol, req.volume, exec_price)
                position = self._process_position(
                    req.symbol,
                    req.type,
                    req.volume,
                    exec_price,
                    commission,
                    req.magic,
                    req.comment,
                    req.tp,
                    req.sl,
                    pmo.order_ticket,
                )
                if position:
                    position.profit = self._calculate_position_profit(position)

    # ── SL/TP intrabar (worst case with open tiebreak) ────────────────────────

    def _check_sl_tp_intrabar(self, symbol: str, high: float, low: float) -> List[int]:
        closed = []
        if not self.positions:
            return closed
        config = self._get_symbol_config(symbol)

        # FIX bug-6: when both SL and TP are touched in the same bar, use
        # the open to estimate which occurred first. The level closer to the
        # open executed first with higher probability.
        # If the open is not available (symbol not yet updated), fall back
        # to the previous conservative behavior (SL always wins).
        _ohlc = self._current_prices.get(symbol, {}).get("ohlc", {})
        open_price: float | None = _ohlc.get("open")

        for ticket, pos in list(self.positions.items()):
            if pos.symbol != symbol or ticket in closed:
                continue

            # No protective levels -> nothing to trigger. Skip before computing
            # spreads (equivalent to the sl_hit/tp_hit both being False below).
            if pos.sl <= 0 and pos.tp <= 0:
                continue

            sl_hit = tp_hit = False

            if pos.type == OrderType.ORDER_TYPE_BUY:
                bid_low, _ = self._simulate_spread(symbol, low)
                bid_high, _ = self._simulate_spread(symbol, high)
                sl_hit = pos.sl > 0 and bid_low <= pos.sl
                tp_hit = pos.tp > 0 and bid_high >= pos.tp
            else:
                _, ask_high = self._simulate_spread(symbol, high)
                _, ask_low = self._simulate_spread(symbol, low)
                sl_hit = pos.sl > 0 and ask_high >= pos.sl
                tp_hit = pos.tp > 0 and ask_low <= pos.tp

            if sl_hit and tp_hit:
                # Both touched - pick the one closer to the bar open
                if open_price is not None:
                    dist_sl = abs(open_price - pos.sl)
                    dist_tp = abs(open_price - pos.tp)
                    # On exact tie keep SL (conservative)
                    if dist_tp < dist_sl:
                        sl_hit = False   # TP won
                    else:
                        tp_hit = False   # SL won (conservative default)
                else:
                    # No open available: conservative fallback - SL wins
                    tp_hit = False

            if sl_hit:
                level, label = pos.sl, "sl"
            elif tp_hit:
                level, label = pos.tp, "tp"
            else:
                continue

            self._close_position_internal(
                ticket,
                pos.volume,
                level,
                f"[{label} {level:.{config.digits}f}]",
            )
            closed.append(ticket)

        return closed

    # ── Pending orders intrabar ───────────────────────────────────────────────

    def _build_order_triggers_kline(
        self,
        open_price: float,
        high: float,
        low: float,
    ) -> Dict[OrderType, OrderTrigger]:
        return {
            OrderType.ORDER_TYPE_BUY_LIMIT: OrderTrigger(
                condition=lambda o: low <= o.price <= high,
                execution_price=lambda o: o.price,
                position_type=OrderType.ORDER_TYPE_BUY,
            ),
            OrderType.ORDER_TYPE_SELL_LIMIT: OrderTrigger(
                condition=lambda o: low <= o.price <= high,
                execution_price=lambda o: o.price,
                position_type=OrderType.ORDER_TYPE_SELL,
            ),
            OrderType.ORDER_TYPE_BUY_STOP: OrderTrigger(
                condition=lambda o: high >= o.price,
                execution_price=lambda o: max(o.price, open_price),
                position_type=OrderType.ORDER_TYPE_BUY,
            ),
            OrderType.ORDER_TYPE_SELL_STOP: OrderTrigger(
                condition=lambda o: low <= o.price,
                execution_price=lambda o: min(o.price, open_price),
                position_type=OrderType.ORDER_TYPE_SELL,
            ),
            OrderType.ORDER_TYPE_BUY_STOP_LIMIT: OrderTrigger(
                condition=lambda o: high >= o.price and low <= o.price_limit <= high,
                execution_price=lambda o: o.price_limit,
                position_type=OrderType.ORDER_TYPE_BUY,
            ),
            OrderType.ORDER_TYPE_SELL_STOP_LIMIT: OrderTrigger(
                condition=lambda o: low <= o.price and low <= o.price_limit <= high,
                execution_price=lambda o: o.price_limit,
                position_type=OrderType.ORDER_TYPE_SELL,
            ),
        }

    def _process_pending_orders_intrabar(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
    ) -> List[int]:
        executed = []
        if not self.orders:
            return executed
        triggers = self._build_order_triggers_kline(open_price, high, low)

        for ticket, order in list(self.orders.items()):
            if order.symbol != symbol or order.type not in triggers:
                continue
            trigger = triggers[order.type]
            if not trigger.condition(order):
                continue

            config = self._get_symbol_config(symbol)
            exec_price = config.normalize_price(trigger.execution_price(order))

            if self.use_spread and config.avg_spread > 0:
                half = config.avg_spread / 2.0
                if trigger.position_type == OrderType.ORDER_TYPE_BUY:
                    exec_price = config.normalize_price(
                        exec_price + math.ceil(half) * config.tick_step
                    )
                else:
                    exec_price = config.normalize_price(
                        exec_price - math.floor(half) * config.tick_step
                    )

            exec_price = self._apply_slippage(
                order.symbol,
                exec_price,
                trigger.position_type == OrderType.ORDER_TYPE_BUY,
            )
            commission = self._calculate_commission(
                order.symbol, order.volume, exec_price
            )
            order.state = OrderState.ORDER_STATE_FILLED
            self._update_order_state(order.ticket, OrderState.ORDER_STATE_FILLED)

            self._process_position(
                order.symbol,
                trigger.position_type,
                order.volume,
                exec_price,
                commission,
                order.magic,
                f"from order {order.ticket}",
                order.tp,
                order.sl,
                order.ticket,
            )
            del self.orders[ticket]
            executed.append(ticket)

        return executed

    def _on_reset(self):
        pass  # KlineBroker has no extra state

    def _close_position_at_current_price(
        self,
        ticket: int,
        volume: Optional[float] = None,
    ) -> bool:
        """
        Enqueues the close for the next bar open when:
          - trade_on_close=False  (execution at next open)
          - _in_on_data=True      (called from on_data() in the backtest loop)

        In any other case (trade_on_close=True, or called directly from
        tests/scripts outside the loop) delegates to the immediate behavior
        of BrokerCore.
        This ensures position_close() remains synchronous in tests and scripts,
        with no lookahead in the backtest.
        """
        if not self.trade_on_close and self._in_on_data:
            if ticket not in self.positions:
                return False
            position = self.positions[ticket]
            if position.symbol not in self._current_prices:
                return False

            vol = min(volume or position.volume, position.volume)
            close_type = (
                OrderType.ORDER_TYPE_SELL
                if position.type == OrderType.ORDER_TYPE_BUY
                else OrderType.ORDER_TYPE_BUY
            )
            req = TradeRequest(
                action=TradeRequestActions.TRADE_ACTION_DEAL,
                symbol=position.symbol,
                type=close_type,
                volume=vol,
                magic=position.magic,
                comment=f"close position #{ticket}",
            )
            order_ticket = self._get_ticket()
            self._move_order_to_history(
                Order(
                    ticket=order_ticket,
                    symbol=position.symbol,
                    type=close_type,
                    volume=vol,
                    price=0.0,  # set when executing at the next open
                    time=self._current_time,
                    state=OrderState.ORDER_STATE_PLACED,
                    magic=position.magic,
                    comment=f"close position #{ticket}",
                )
            )
            pmo = PendingMarketOrder(
                request=req,
                order_ticket=order_ticket,
                close_ticket=ticket,
            )
            self.pending_market_orders.append(pmo)
            return True

        return super()._close_position_at_current_price(ticket, volume)


# ══════════════════════════════════════════════════════════════════════
# TICK BROKER
# ══════════════════════════════════════════════════════════════════════


class TickBroker(BrokerCore):
    """
    Broker for real tick data (direct bid/ask).

    - bid/ask used directly - no simulated spread.
    - SL/TP evaluated tick by tick in chronological order.
    - Market orders are always enqueued for the first next tick,
      regardless of any configuration.
      This eliminates look-ahead bias: executing at the current tick price
      is impossible in real markets where there is always minimum latency.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission: float = 0.0,
        commission_type: CommissionType = CommissionType.COMMISSION_TYPE_MONEY,
        use_margin: bool = False,
        margin_rate: float = 0.01,
        position_mode: PositionMode = PositionMode.POSITION_MODE_NETTING,
        slippage_points: int = 0,
        stop_out_of_money: bool = True,
    ):
        super().__init__(
            initial_balance=initial_balance,
            commission=commission,
            commission_type=commission_type,
            use_margin=use_margin,
            margin_rate=margin_rate,
            position_mode=position_mode,
            slippage_points=slippage_points,
            stop_out_of_money=stop_out_of_money,
        )
        self._pending_ready: set = set()  # tickets ready to execute
        self._enqueued_this_tick: set = set()  # enqueued during the current tick
        self._processing_tick: bool = False

    # ── Hook ──────────────────────────────────────────────────────────────────

    def _should_execute_immediately(self) -> bool:
        # In TickBroker orders are never executed immediately
        return False

    def _on_market_order_enqueued(self, pmo: PendingMarketOrder):
        """
        Marks the order as 'ready' for the next tick.
        If it was enqueued during processing of the current tick, it is removed
        from 'ready' until the tick finishes - so it does not execute in the same tick.
        """
        self._pending_ready.add(pmo.order_ticket)
        if self._processing_tick:
            self._enqueued_this_tick.add(pmo.order_ticket)

    # ── update_tick ───────────────────────────────────────────────────────────

    def update_tick(
        self,
        symbol: str,
        timestamp: datetime,
        bid: float,
        ask: float,
        volume: float = 0.0,
        flags: int = 0,
        volume_real: float = 0.0,
        price: float = 0.0,
    ):
        self._current_time = timestamp
        self._current_ts_ms = None
        self._processing_tick = True
        self._enqueued_this_tick.clear()

        config = self._get_symbol_config(symbol)
        bid = config.normalize_price(bid)
        ask = config.normalize_price(ask)

        # 1. Execute ready market orders (enqueued in previous ticks)
        if self.pending_market_orders:
            self._execute_pending_market_orders_tick(symbol, bid, ask)

        # 2. Expire pending orders
        self._expire_pending_orders()

        # 3. Update prices
        self._current_prices[symbol] = {
            "bid": bid,
            "ask": ask,
            "last": price if price > 0.0 else (bid + ask) / 2.0,
        }

        # 4. Evaluate SL/TP
        self._check_sl_tp_tick(symbol, bid, ask)

        # 5. Process pending orders
        if self.orders:
            self._process_pending_orders_tick(symbol, bid, ask)

        # 6. Update profit of open positions
        for p in self.positions.values():
            if p.symbol == symbol:
                p.profit = self._calculate_position_profit(p)

        # 7. Orders enqueued THIS tick must not execute until the next one
        self._pending_ready -= self._enqueued_this_tick
        self._enqueued_this_tick.clear()
        self._processing_tick = False

        self._record_equity()

    # ── Execution of enqueued market orders ───────────────────────────────────

    def _execute_pending_market_orders_tick(self, symbol: str, bid: float, ask: float):
        to_execute = [
            pmo
            for pmo in self.pending_market_orders
            if pmo.request.symbol == symbol and pmo.order_ticket in self._pending_ready
        ]
        tickets = {pmo.order_ticket for pmo in to_execute}
        self.pending_market_orders = [
            pmo for pmo in self.pending_market_orders if pmo.order_ticket not in tickets
        ]
        self._pending_ready -= tickets

        config = self._get_symbol_config(symbol)
        for pmo in to_execute:
            req = pmo.request
            is_buy = req.type == OrderType.ORDER_TYPE_BUY
            exec_price = config.normalize_price(ask if is_buy else bid)
            exec_price = self._apply_slippage(req.symbol, exec_price, is_buy)
            commission = self._calculate_commission(req.symbol, req.volume, exec_price)

            if pmo.order_ticket in self.orders_history:
                self.orders_history[pmo.order_ticket].price = exec_price
                self.orders_history[
                    pmo.order_ticket
                ].state = OrderState.ORDER_STATE_FILLED

            position = self._process_position(
                req.symbol,
                req.type,
                req.volume,
                exec_price,
                commission,
                req.magic,
                req.comment,
                req.tp,
                req.sl,
                pmo.order_ticket,
            )
            if position:
                position.profit = self._calculate_position_profit(position)

    # ── SL/TP tick by tick ────────────────────────────────────────────────────

    def _check_sl_tp_tick(self, symbol: str, bid: float, ask: float) -> List[int]:
        closed = []
        config = self._get_symbol_config(symbol)

        for ticket, pos in list(self.positions.items()):
            if pos.symbol != symbol or ticket in closed:
                continue

            close_price = None
            comment = ""

            if pos.type == OrderType.ORDER_TYPE_BUY:
                if pos.sl > 0 and bid <= pos.sl:
                    close_price, comment = pos.sl, f"[sl {pos.sl:.{config.digits}f}]"
                elif pos.tp > 0 and bid >= pos.tp:
                    close_price, comment = pos.tp, f"[tp {pos.tp:.{config.digits}f}]"
            else:
                if pos.sl > 0 and ask >= pos.sl:
                    close_price, comment = pos.sl, f"[sl {pos.sl:.{config.digits}f}]"
                elif pos.tp > 0 and ask <= pos.tp:
                    close_price, comment = pos.tp, f"[tp {pos.tp:.{config.digits}f}]"

            if close_price is not None:
                self._close_position_internal(ticket, pos.volume, close_price, comment)
                closed.append(ticket)

        return closed

    # ── Pending orders tick ───────────────────────────────────────────────────

    def _build_order_triggers_tick(
        self,
        bid: float,
        ask: float,
    ) -> Dict[OrderType, OrderTrigger]:
        return {
            OrderType.ORDER_TYPE_BUY_LIMIT: OrderTrigger(
                condition=lambda o: ask <= o.price,
                execution_price=lambda o: ask,
                position_type=OrderType.ORDER_TYPE_BUY,
            ),
            OrderType.ORDER_TYPE_SELL_LIMIT: OrderTrigger(
                condition=lambda o: bid >= o.price,
                execution_price=lambda o: bid,
                position_type=OrderType.ORDER_TYPE_SELL,
            ),
            OrderType.ORDER_TYPE_BUY_STOP: OrderTrigger(
                condition=lambda o: ask >= o.price,
                execution_price=lambda o: ask,
                position_type=OrderType.ORDER_TYPE_BUY,
            ),
            OrderType.ORDER_TYPE_SELL_STOP: OrderTrigger(
                condition=lambda o: bid <= o.price,
                execution_price=lambda o: bid,
                position_type=OrderType.ORDER_TYPE_SELL,
            ),
            OrderType.ORDER_TYPE_BUY_STOP_LIMIT: OrderTrigger(
                condition=lambda o: ask >= o.price and ask <= o.price_limit,
                execution_price=lambda o: ask,
                position_type=OrderType.ORDER_TYPE_BUY,
            ),
            OrderType.ORDER_TYPE_SELL_STOP_LIMIT: OrderTrigger(
                condition=lambda o: bid <= o.price and bid >= o.price_limit,
                execution_price=lambda o: bid,
                position_type=OrderType.ORDER_TYPE_SELL,
            ),
        }

    def _process_pending_orders_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
    ) -> List[int]:
        executed = []
        config = self._get_symbol_config(symbol)
        triggers = self._build_order_triggers_tick(bid, ask)

        for ticket, order in list(self.orders.items()):
            if order.symbol != symbol or order.type not in triggers:
                continue
            trigger = triggers[order.type]
            if not trigger.condition(order):
                continue

            exec_price = config.normalize_price(trigger.execution_price(order))
            exec_price = self._apply_slippage(
                order.symbol,
                exec_price,
                trigger.position_type == OrderType.ORDER_TYPE_BUY,
            )
            commission = self._calculate_commission(
                order.symbol, order.volume, exec_price
            )
            order.state = OrderState.ORDER_STATE_FILLED
            self._update_order_state(order.ticket, OrderState.ORDER_STATE_FILLED)

            self._process_position(
                order.symbol,
                trigger.position_type,
                order.volume,
                exec_price,
                commission,
                order.magic,
                f"from order {order.ticket}",
                order.tp,
                order.sl,
                order.ticket,
            )
            del self.orders[ticket]
            executed.append(ticket)

        return executed

    def _on_reset(self):
        self._pending_ready.clear()
        self._enqueued_this_tick.clear()
        self._processing_tick = False


# endregion

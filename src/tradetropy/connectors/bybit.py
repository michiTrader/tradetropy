"""
bybit.py
========
Two symmetric implementations for Bybit (linear USDT perp by default):

  SeshBybitLive  - connects to real Bybit via the pybit SDK (unified_trading.HTTP).
  SeshBybitSim   - uses the internal simulated broker (TickBroker/KlineBroker).

GOAL
----
That a strategy works identically in backtest and live without changing
a single line - same contract as SeshMT5Live / SeshMT5Sim:

    class MyStrategy(Strategy):
        def on_data(self):
            pos = self.sesh.positions("BTCUSDT")
            if not pos:
                self.sesh.buy("BTCUSDT", volume=0.01, sl=60000.0, tp=70000.0)
            acc = self.sesh.account_info()
            if acc.equity < acc.balance * 0.95:
                self.sesh.close_all("BTCUSDT")

    # Backtest:
    sesh = SeshBybitSim(initial_balance=10_000, commission=0.0006,
                        commission_type=CommissionType.COMMISSION_TYPE_PERCENT)
    engine = BacktestEngine.by_klines(MyStrategy(), data=(klines,), sesh=sesh)

    # Live:
    sesh = SeshBybitLive(api_key="...", api_secret="...", category="linear")
    engine = LiveEngine.by_klines(MyStrategy(), sesh=sesh)

METHODS AVAILABLE IN BOTH CLASSES (identical)
----------------------------------------------
Required (Sesh):
    buy(symbol, volume, price?, sl?, tp?, comment?, magic?) → TradeResult
    sell(symbol, volume, price?, sl?, tp?, comment?, magic?) → TradeResult
    positions(symbol?)  → list[Position]
    account_info()      → AccountInfo

Optional (implemented in both):
    position_close(ticket, volume?)   → bool
    position_modify(ticket, sl?, tp?) → bool
    order_delete(ticket)              → bool
    orders(symbol?)                   → list[Order]
    order_history(symbol?)           → list[Order]
    deals(symbol?)                    → list[Deal]
    sync()                            → None
    reset()                           → None  (SeshBybitSim only)

Custom Bybit methods (implemented in both with equivalent logic):
    close_all(symbol)                          → list[dict]
    get_last_price(symbol)                     → float
    calculate_margin(symbol, volume, side?)    → float
    calculate_profit(symbol, volume, open, close, side?) → float
    get_account_info()                         → dict
    get_instrument_info(symbol)               → dict
    is_market_open(symbol)                     → bool

Bybit-specific extras (SeshBybitLive only; in SeshBybitSim they are no-op/safe):
    set_leverage(symbol, leverage)
    set_margin_mode(margin_mode)
    cancel_all_orders(symbol, order_filter?)
    set_trading_stop(symbol, tp_price?, sl_price?, tp_volume?, sl_volume?)

OFFLINE TESTING
---------------
The `http` constructor parameter is injectable: pass it a mock class that
mimics the pybit.unified_trading.HTTP API and no network call will be made.

    class MockHTTP:
        def __init__(self, **kw): ...
        def place_order(self, **kw): return {"retCode": 0, "result": {"orderId": "1"}}
        ...
    sesh = SeshBybitLive(api_key="x", api_secret="y", http=MockHTTP)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np

from tradetropy.core.broker import (
    AccountInfo,
    Deal,
    Order,
    OrderState,
    OrderType,
    Position,
    TradeResult,
)
from tradetropy.exceptions import ConfigError, TradingError, ConnectionError
from tradetropy.session.base import (
    Sesh,
    SeshSimulatorBase,
    SeshLiveBase,
)
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.connectors._disclaimer import emit_live_disclaimer

_logger = logging.getLogger("tradetropy.connectors.bybit")

# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ms -> Bybit interval (kline `interval`). Bybit uses minutes as string for
# intraday and "D"/"W"/"M" for daily/weekly/monthly.
_MS_TO_BYBIT_INTERVAL = {
    60_000:        "1",
    180_000:       "3",
    300_000:       "5",
    900_000:       "15",
    1_800_000:     "30",
    3_600_000:     "60",
    7_200_000:     "120",
    14_400_000:    "240",
    21_600_000:    "360",
    43_200_000:    "720",
    86_400_000:    "D",
    604_800_000:   "W",
    2_592_000_000: "M",
}

# raw Bybit orderType/status -> typed OrderState
_BYBIT_STATUS_TO_STATE = {
    "New":             OrderState.ORDER_STATE_PLACED,
    "Created":         OrderState.ORDER_STATE_PLACED,
    "Untriggered":     OrderState.ORDER_STATE_PLACED,
    "Triggered":       OrderState.ORDER_STATE_PLACED,
    "PartiallyFilled": OrderState.ORDER_STATE_PARTIAL,
    "Filled":          OrderState.ORDER_STATE_FILLED,
    "Cancelled":       OrderState.ORDER_STATE_CANCELED,
    "Deactivated":     OrderState.ORDER_STATE_CANCELED,
    "Rejected":        OrderState.ORDER_STATE_REJECTED,
}

_ticket_counter = [0]


def _next_ticket() -> int:
    _ticket_counter[0] += 1
    return _ticket_counter[0]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _bybit_side_to_order_type(side: str, *, is_pending: bool = False) -> OrderType:
    """
    Convert 'Buy'/'Sell' to OrderType.

    Args:
        side: 'Buy' or 'Sell'.
        is_pending: If True, return limit type; else market type.

    Returns:
        OrderType: Market or limit buy/sell order type.
    """
    buy = str(side).capitalize() == "Buy"
    if is_pending:
        return OrderType.ORDER_TYPE_BUY_LIMIT if buy else OrderType.ORDER_TYPE_SELL_LIMIT
    return OrderType.ORDER_TYPE_BUY if buy else OrderType.ORDER_TYPE_SELL


def _order_type_to_bybit_side(ot: OrderType) -> str:
    """
    Convert OrderType to Bybit side string.

    Args:
        ot: OrderType enum value.

    Returns:
        'Buy' or 'Sell' string representing the base side.
    """
    return "Buy" if ot in (
        OrderType.ORDER_TYPE_BUY,
        OrderType.ORDER_TYPE_BUY_LIMIT,
        OrderType.ORDER_TYPE_BUY_STOP,
        OrderType.ORDER_TYPE_BUY_STOP_LIMIT,
    ) else "Sell"


def _safe_float(v, default: float = 0.0) -> float:
    """
    Safely convert value to float with default fallback.

    Args:
        v: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float value or default.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _format_order_to_order(raw: dict) -> Order:
    """Converts a raw Bybit order to a typed Order object."""
    side = raw.get("side", "Buy")
    order_type = str(raw.get("orderType") or "").lower()
    is_pending = order_type == "limit" or bool(raw.get("triggerPrice"))
    ot = _bybit_side_to_order_type(side, is_pending=is_pending)
    state = _BYBIT_STATUS_TO_STATE.get(raw.get("orderStatus"), OrderState.ORDER_STATE_PLACED)

    created_ms = _safe_int(raw.get("createdTime") or raw.get("updatedTime") or 0)
    t = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc) if created_ms else _now()

    return Order(
        ticket=_safe_int(raw.get("orderId"), 0) if str(raw.get("orderId", "")).isdigit()
               else (abs(hash(str(raw.get("orderId")))) % 1_000_000_000),
        symbol=str(raw.get("symbol", "")),
        type=ot,
        volume=_safe_float(raw.get("qty") or raw.get("size") or 0),
        price=_safe_float(raw.get("price") or raw.get("avgPrice") or 0),
        time=t,
        sl=_safe_float(raw.get("stopLoss") or 0),
        tp=_safe_float(raw.get("takeProfit") or 0),
        magic=0,
        comment=str(raw.get("orderLinkId") or ""),
        state=state,
    )


def _import_pybit():
    """Lazy import of pybit HTTP with clear message if missing."""
    try:
        from pybit.unified_trading import HTTP
        return HTTP
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "pybit is not installed. Install it with: pip install pybit\n"
            "(or install the project extra that includes it)."
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 - SESH BYBIT LIVE
# ══════════════════════════════════════════════════════════════════════════════


class SeshBybitLive(SeshLiveBase):
    """
    Live Sesh on Bybit (pybit unified_trading).

    Implements all Sesh methods + custom Bybit methods, with the same
    signature as SeshMT5Live so the strategy doesn't change when moving
    from backtest to production.

    Requires: pip install pybit

    Parameters
    ----------
    api_key, api_secret : Bybit API credentials
    category     : "linear" (USDT perp, default) | "inverse" | "spot"
    account_type : "UNIFIED" (default) | "CONTRACT" | "SPOT"
    base_coin    : account currency for balance (default "USDT")
    demo         : use Bybit demo trading (default False)
    http         : HTTP class to instantiate (injectable for tests). If None
                   pybit.unified_trading.HTTP is imported lazily.
    default_magic: default magic number (Bybit doesn't use it; kept for
                   interface symmetry)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        category: str = "linear",
        account_type: str = "UNIFIED",
        base_coin: str = "USDT",
        demo: bool = False,
        http=None,
        default_magic: int = 0,
    ):
        super().__init__()
        emit_live_disclaimer()
        self._category = category
        self._account_type = account_type
        self._base_coin = base_coin
        self._default_magic = default_magic

        http_cls = http if http is not None else _import_pybit()
        try:
            self._session = http_cls(api_key=api_key, api_secret=api_secret, demo=demo)
        except Exception as exc:
            raise ConnectionError(
                f"Could not create Bybit HTTP session: {exc}"
            ) from exc

        # stable synthetic ticket by (symbol, positionIdx)
        self._ticket_by_key: dict[tuple, int] = {}

        _logger.info(
            "SeshBybitLive conectado — category=%s account_type=%s base_coin=%s demo=%s",
            self._category, self._account_type, self._base_coin, demo,
        )
        self.sync()

    # -- Connection ---------------------------------------------------------------

    def disconnect(self) -> None:
        # pybit HTTP does not maintain a persistent connection; no-op.
        pass

    def reconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return self._session is not None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self) -> str:
        return (
            f"SeshBybitLive(category={self._category!r}, "
            f"account_type={self._account_type!r}, "
            f"positions={len(self._cache_positions)})"
        )

    # -- synthetic ticket helpers --------------------------------------------------

    def _ticket_for(self, symbol: str, position_idx: int) -> int:
        key = (symbol, int(position_idx))
        t = self._ticket_by_key.get(key)
        if t is None:
            t = _next_ticket()
            self._ticket_by_key[key] = t
        return t

    # -- 4 required methods -------------------------------------------------------

    def buy(
        self,
        symbol: str,
        volume: float,
        price: Optional[float] = None,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
        magic: int = 0,
    ) -> TradeResult:
        return self._place_order(symbol, "Buy", volume, price, sl, tp, comment)

    def sell(
        self,
        symbol: str,
        volume: float,
        price: Optional[float] = None,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
        magic: int = 0,
    ) -> TradeResult:
        return self._place_order(symbol, "Sell", volume, price, sl, tp, comment)

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        kwargs = {"category": self._category}
        if symbol:
            kwargs["symbol"] = symbol
        else:
            kwargs["settleCoin"] = self._base_coin
        raw = self._session.get_positions(**kwargs)
        lst = (raw.get("result") or {}).get("list") or []

        out: List[Position] = []
        self._cache_positions = {}
        for p in lst:
            size = _safe_float(p.get("size"))
            if size <= 0:
                continue
            sym = str(p.get("symbol", ""))
            idx = _safe_int(p.get("positionIdx"), 0)
            ticket = self._ticket_for(sym, idx)
            side = p.get("side", "Buy")
            self._cache_positions[ticket] = {
                "symbol": sym,
                "side": side,
                "size": size,
                "positionIdx": idx,
            }
            out.append(Position(
                ticket=ticket,
                symbol=sym,
                type=_bybit_side_to_order_type(side),
                volume=size,
                price_open=_safe_float(p.get("avgPrice")),
                time=_now(),
                sl=_safe_float(p.get("stopLoss")),
                tp=_safe_float(p.get("takeProfit")),
                profit=_safe_float(p.get("unrealisedPnl")),
                magic=0,
                comment="",
            ))
        return out

    def account_info(self) -> AccountInfo:
        raw = self._session.get_wallet_balance(
            accountType=self._account_type, coin=self._base_coin
        )
        lst = (raw.get("result") or {}).get("list") or []
        if not lst:
            return AccountInfo(balance=self._balance, equity=self._equity)
        acc = lst[0]
        coins = acc.get("coin") or []
        coin0 = coins[0] if coins else {}

        wallet = _safe_float(coin0.get("walletBalance"))
        equity = _safe_float(coin0.get("equity") or acc.get("totalEquity"))
        pnl = _safe_float(coin0.get("unrealisedPnl") or acc.get("totalPerpUPL"))
        position_im = _safe_float(coin0.get("totalPositionIM") or acc.get("totalInitialMargin"))

        return AccountInfo(
            balance=wallet,
            equity=equity if equity else wallet,
            margin=position_im,
            margin_free=max(wallet - position_im, 0.0),
            profit=pnl,
        )

    # -- Optional methods ---------------------------------------------------------

    def position_close(self, ticket: int, volume: Optional[float] = None) -> bool:
        pos = self._cache_positions.get(ticket)
        if pos is None:
            # refresh in case cache is cold
            self.positions()
            pos = self._cache_positions.get(ticket)
        if pos is None:
            return False

        close_side = "Sell" if str(pos["side"]).capitalize() == "Buy" else "Buy"
        qty = float(volume) if volume else float(pos["size"])
        resp = self._session.place_order(
            category=self._category,
            symbol=pos["symbol"],
            side=close_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
        )
        ok = _safe_int(resp.get("retCode"), -1) == 0
        if ok:
            if qty >= float(pos["size"]):
                self._cache_positions.pop(ticket, None)
            else:
                pos["size"] -= qty
        return ok

    def position_modify(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        pos = self._cache_positions.get(ticket)
        if pos is None:
            self.positions()
            pos = self._cache_positions.get(ticket)
        if pos is None:
            return False
        resp = self._session.set_trading_stop(
            category=self._category,
            symbol=pos["symbol"],
            takeProfit=str(tp) if tp else None,
            stopLoss=str(sl) if sl else None,
            positionIdx=pos.get("positionIdx", 0),
        )
        return _safe_int(resp.get("retCode"), -1) == 0

    def order_delete(self, ticket: int) -> bool:
        order_id = self._cache_orders.get(ticket, {}).get("orderId", ticket)
        # search for order symbol in cache; otherwise trying without symbol fails on bybit
        symbol = self._cache_orders.get(ticket, {}).get("symbol")
        if symbol is None:
            return False
        resp = self._session.cancel_order(
            category=self._category, symbol=symbol, orderId=str(order_id)
        )
        return _safe_int(resp.get("retCode"), -1) == 0

    def orders(self, symbol: Optional[str] = None) -> List[Order]:
        kwargs = {"category": self._category, "openOnly": 0}
        if symbol:
            kwargs["symbol"] = symbol
        else:
            kwargs["settleCoin"] = self._base_coin
        raw = self._session.get_open_orders(**kwargs)
        lst = (raw.get("result") or {}).get("list") or []
        result = [_format_order_to_order(o) for o in lst]
        # populate cache for order_delete
        for o, raw_o in zip(result, lst):
            self._cache_orders[o.ticket] = {
                "orderId": raw_o.get("orderId"),
                "symbol": o.symbol,
            }
        return result

    def order_history(self, symbol: Optional[str] = None) -> List[Order]:
        kwargs = {"category": self._category}
        if symbol:
            kwargs["symbol"] = symbol
        else:
            kwargs["settleCoin"] = self._base_coin
        raw = self._session.get_order_history(**kwargs)
        lst = (raw.get("result") or {}).get("list") or []
        return [_format_order_to_order(o) for o in lst]

    # Legacy alias
    orders_history = order_history

    def deals(self, symbol: Optional[str] = None) -> List[Deal]:
        kwargs = {"category": self._category}
        if symbol:
            kwargs["symbol"] = symbol
        raw = self._session.get_closed_pnl(**kwargs)
        lst = (raw.get("result") or {}).get("list") or []
        out: List[Deal] = []
        for it in lst:
            ts = _safe_int(it.get("updatedTime") or it.get("createdTime") or 0)
            t = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else _now()
            side = str(it.get("side") or "").capitalize()
            out.append(Deal(
                ticket=_next_ticket(),
                position_id=0,
                symbol=str(it.get("symbol", "")),
                type=OrderType.ORDER_TYPE_BUY if side == "Buy" else OrderType.ORDER_TYPE_SELL,
                volume=_safe_float(it.get("qty") or it.get("size")),
                price=_safe_float(it.get("avgExitPrice") or it.get("avgPrice")),
                time=t,
                commission=_safe_float(it.get("fees") or it.get("fee")),
                profit=_safe_float(it.get("closedPnl") or it.get("pnl")),
            ))
        return out

    def sync(self) -> None:
        # Positions: critical state (see Sesh.sync). Log at error + retry;
        # never swallow the failure silently.
        for attempt in (1, 2):
            try:
                self.positions()  # fills self._cache_positions
                break
            except Exception:
                _logger.error(
                    "SeshBybitLive.sync: failed to get positions (attempt %d/2). "
                    "Position cache may be outdated.",
                    attempt, exc_info=True,
                )
        try:
            acc = self.account_info()
            self._balance = acc.balance
            self._equity = acc.equity
        except Exception:
            _logger.error(
                "SeshBybitLive.sync: failed to get account info. "
                "balance/equity may be outdated.",
                exc_info=True,
            )

    # -- Data primitives ----------------------------------------------------------

    def _fetch_klines_history(
        self, symbol: str, interval_ms: int, limit: int = 200
    ) -> np.ndarray:
        interval = _MS_TO_BYBIT_INTERVAL.get(interval_ms)
        if interval is None:
            raise ConfigError(
                f"interval_ms={interval_ms} has no equivalent Bybit interval. "
                f"Supported: {sorted(_MS_TO_BYBIT_INTERVAL.keys())}"
            )
        raw = self._session.get_kline(
            category=self._category, symbol=symbol, interval=interval, limit=limit
        )
        lst = (raw.get("result") or {}).get("list") or []
        if not lst:
            return np.empty((0, 6), dtype=np.float64)
        # Bybit returns newest-first: [start, open, high, low, close, volume, turnover]
        rows = lst[::-1]
        out = np.empty((len(rows), 6), dtype=np.float64)
        for i, r in enumerate(rows):
            out[i, 0] = _safe_float(r[0])   # ts (ms)
            out[i, 1] = _safe_float(r[1])   # open
            out[i, 2] = _safe_float(r[2])   # high
            out[i, 3] = _safe_float(r[3])   # low
            out[i, 4] = _safe_float(r[4])   # close
            out[i, 5] = _safe_float(r[5])   # volume
        return out

    def _fetch_last_kline(self, symbol: str, interval_ms: int) -> np.ndarray:
        klines = self._fetch_klines_history(symbol, interval_ms, limit=2)
        if len(klines) < 1:
            raise TradingError(
                f"Could not get candles for '{symbol}' "
                f"(interval_ms={interval_ms})."
            )
        return klines[-1]

    def _fetch_last_tick(self, symbol: str) -> np.ndarray:
        raw = self._session.get_tickers(category=self._category, symbol=symbol)
        lst = (raw.get("result") or {}).get("list") or []
        if not lst:
            raise TradingError(f"Could not get tick for '{symbol}'.")
        t = lst[0]
        out = np.empty(N_TICK_COLS, dtype=np.float64)
        bid = _safe_float(t.get("bid1Price"))
        ask = _safe_float(t.get("ask1Price"))
        last = _safe_float(t.get("lastPrice"))
        out[_TICK_COL["ts"]]          = float(int(time.time() * 1000))
        out[_TICK_COL["bid"]]         = bid
        out[_TICK_COL["ask"]]         = ask
        out[_TICK_COL["volume"]]      = 0.0
        out[_TICK_COL["flags"]]       = 0.0
        out[_TICK_COL["volume_real"]] = np.nan
        if last > 0:
            out[_TICK_COL["price"]] = last
        elif bid > 0 and ask > 0:
            out[_TICK_COL["price"]] = (bid + ask) / 2.0
        else:
            out[_TICK_COL["price"]] = last
        return out

    def _fetch_ticks_history(self, symbol: str, limit: int = 500) -> np.ndarray:
        raw = self._session.get_public_trade_history(
            category=self._category, symbol=symbol, limit=limit
        )
        lst = (raw.get("result") or {}).get("list") or []
        if not lst:
            return np.empty((0, N_TICK_COLS), dtype=np.float64)
        # Bybit returns trades from newest to oldest
        rows = lst[::-1]
        out = np.empty((len(rows), N_TICK_COLS), dtype=np.float64)
        for i, r in enumerate(rows):
            price = _safe_float(r.get("price"))
            size = _safe_float(r.get("size"))
            side = str(r.get("side") or "").capitalize()
            out[i, _TICK_COL["ts"]]          = float(_safe_int(r.get("time")))
            out[i, _TICK_COL["bid"]]         = price
            out[i, _TICK_COL["ask"]]         = price
            out[i, _TICK_COL["volume"]]      = size
            out[i, _TICK_COL["flags"]]       = 1.0 if side == "Buy" else -1.0
            out[i, _TICK_COL["volume_real"]] = size
            out[i, _TICK_COL["price"]]       = price
        return out

    # Custom Bybit methods
    # =====

    def close_all(self, symbol: str) -> list:
        """
        Close all open positions for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            List of dicts with 'ticket' and 'ok' status for each closed position.
        """
        results = []
        for p in self.positions(symbol):
            ok = self.position_close(p.ticket)
            results.append({"ticket": p.ticket, "ok": ok})
        return results

    def get_last_price(self, symbol: str) -> float:
        raw = self._session.get_tickers(category=self._category, symbol=symbol)
        lst = (raw.get("result") or {}).get("list") or []
        if not lst:
            raise TradingError(f"No price for '{symbol}'.")
        t = lst[0]
        last = _safe_float(t.get("lastPrice"))
        if last > 0:
            return last
        bid = _safe_float(t.get("bid1Price"))
        ask = _safe_float(t.get("ask1Price"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        raise TradingError(f"No valid price for '{symbol}'.")

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        """Approximate margin = volume * price / leverage (current leverage of the symbol)."""
        price = self.get_last_price(symbol)
        lev = max(self.get_leverage(symbol), 1)
        return (volume * price) / lev

    def calculate_profit(
        self,
        symbol: str,
        volume: float,
        price_open: float,
        price_close: float,
        side: str = "Buy",
    ) -> float:
        """Linear PnL: (close-open)*volume for long, inverted for short."""
        is_buy = str(side).capitalize() == "Buy"
        diff = (price_close - price_open) if is_buy else (price_open - price_close)
        return diff * volume

    def get_account_info(self) -> dict:
        a = self.account_info()
        return {
            "login":       None,
            "tradeMode":   self._account_type,
            "leverage":    None,
            "balance":     a.balance,
            "equity":      a.equity,
            "margin":      a.margin,
            "marginFree":  a.margin_free,
            "profit":      a.profit,
            "currency":    self._base_coin,
            "company":     "Bybit",
            "server":      "Bybit",
            "marketType":  self._category,
        }

    def get_instrument_info(self, symbol: str) -> dict:
        raw = self._session.get_instrument_info(category=self._category, symbol=symbol)
        lst = (raw.get("result") or {}).get("list") or []
        if not lst:
            raise TradingError(f"No info for '{symbol}'.")
        item = lst[0]
        lot = item.get("lotSizeFilter") or {}
        price_filter = item.get("priceFilter") or {}

        qty_step = str(lot.get("qtyStep", "0"))
        digits = len(qty_step.split(".")[-1]) if "." in qty_step else 0
        tick_size = _safe_float(price_filter.get("tickSize"), 0.0)

        return {
            "symbol":              item.get("symbol"),
            "digits":              digits,
            "tick_step":           tick_size,
            "tick_value":          1.0,
            "trade_contract_size": 1.0,
            "volume_min":          _safe_float(lot.get("minOrderQty")),
            "volume_max":          _safe_float(lot.get("maxMktOrderQty") or lot.get("maxOrderQty")),
            "volume_step":         _safe_float(lot.get("qtyStep")),
            "currency_base":       item.get("baseCoin"),
        }

    def is_market_open(self, symbol: str) -> bool:
        # Crypto operates 24/7; validate there is a ticker with valid price.
        try:
            return self.get_last_price(symbol) > 0
        except Exception:
            return False

    # -- Bybit-specific extras -----------------------------------------------------

    def get_leverage(self, symbol: str) -> int:
        try:
            raw = self._session.get_positions(category=self._category, symbol=symbol)
            lst = (raw.get("result") or {}).get("list") or []
            if lst:
                return _safe_int(lst[0].get("leverage"), 1)
        except Exception:
            _logger.warning("get_leverage: could not get leverage for %s", symbol)
        return 1

    def set_leverage(self, symbol: str, leverage: int = 1) -> bool:
        if leverage <= 0:
            raise ConfigError("Leverage must be greater than 0.")
        if self.get_leverage(symbol) == int(leverage):
            return True
        try:
            self._session.set_leverage(
                category=self._category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            return True
        except Exception as exc:
            raise TradingError(f"Could not set leverage: {exc}") from exc

    def set_margin_mode(self, margin_mode: str = "isolated") -> bool:
        modes = {
            "isolated": "ISOLATED_MARGIN",
            "cross": "REGULAR_MARGIN",
            "portfolio": "PORTFOLIO_MARGIN",
        }
        if margin_mode not in modes:
            raise ConfigError(
                f"Invalid margin_mode: {margin_mode!r}. Use {list(modes.keys())}."
            )
        resp = self._session.set_margin_mode(setMarginMode=modes[margin_mode])
        return _safe_int(resp.get("retCode"), -1) == 0

    def cancel_all_orders(self, symbol: str, order_filter: Optional[str] = None) -> bool:
        resp = self._session.cancel_all_orders(
            category=self._category, symbol=symbol, orderFilter=order_filter
        )
        return _safe_int(resp.get("retCode"), -1) == 0

    def set_trading_stop(
        self,
        symbol: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_volume: Optional[float] = None,
        sl_volume: Optional[float] = None,
    ) -> bool:
        """Sets TP/SL on the open position of the symbol (reduce-only)."""
        resp = self._session.set_trading_stop(
            category=self._category,
            symbol=symbol,
            takeProfit=str(tp_price) if tp_price else None,
            stopLoss=str(sl_price) if sl_price else None,
            tpSize=str(tp_volume) if tp_volume else None,
            slSize=str(sl_volume) if sl_volume else None,
            positionIdx=0,
        )
        return _safe_int(resp.get("retCode"), -1) == 0

    # -- Internal logic -----------------------------------------------------------

    def _place_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: Optional[float],
        sl: float,
        tp: float,
        comment: str,
    ) -> TradeResult:
        is_limit = price is not None
        kwargs = dict(
            category=self._category,
            symbol=symbol,
            side=side,
            orderType="Limit" if is_limit else "Market",
            qty=str(volume),
            timeInForce="GTC",
        )
        if is_limit:
            kwargs["price"] = str(price)
        if sl:
            kwargs["stopLoss"] = str(sl)
        if tp:
            kwargs["takeProfit"] = str(tp)
        if comment:
            kwargs["orderLinkId"] = comment

        try:
            resp = self._session.place_order(**kwargs)
        except Exception as exc:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID,
                comment=f"place_order failed: {exc}",
            )

        ret_code = _safe_int(resp.get("retCode"), -1)
        result = resp.get("result") or {}
        order_id_raw = str(result.get("orderId", "0"))
        order_id = _safe_int(order_id_raw, 0) if order_id_raw.isdigit() \
            else (abs(hash(order_id_raw)) % 1_000_000_000)

        if ret_code != 0:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID,
                comment=str(resp.get("retMsg", "error")),
            )

        retcode = (TradeResult.TRADE_RETCODE_PLACED if is_limit
                   else TradeResult.TRADE_RETCODE_DONE)
        return TradeResult(
            retcode=retcode,
            order=order_id,
            volume=float(volume),
            price=float(price) if is_limit else 0.0,
            comment=str(result.get("orderLinkId", comment)),
        )

    # SeshLiveBase hooks (used by default sync if desired)
    def _get_positions_raw(self) -> Optional[list]:
        return None

    def _get_account_raw(self) -> Optional[dict]:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - SESH BYBIT SIM (simulated - symmetric to SeshBybitLive)
# ══════════════════════════════════════════════════════════════════════════════


class SeshBybitSim(SeshSimulatorBase):
    """
    Simulator symmetric to SeshBybitLive.

    Exposes exactly the same custom methods as SeshBybitLive with simulated
    logic using the internal broker. The strategy doesn't need to know which
    one is active.

    Parameters - identical to SeshSimulatorBase:
        feed_type, initial_balance, commission, commission_type,
        position_mode, use_margin, margin_rate, slippage_points,
        use_spread, trade_on_close, execution_price_source.

    USAGE
    -----
        # Backtest
        sesh = SeshBybitSim(initial_balance=10_000, commission=0.0006,
                            commission_type=CommissionType.COMMISSION_TYPE_PERCENT)
        engine = BacktestEngine.by_klines(MyStrategy(), data=(klines,), sesh=sesh)

        # Live (same strategy code)
        sesh = SeshBybitLive(api_key="...", api_secret="...", category="linear")
        engine = LiveEngine.by_klines(MyStrategy(), sesh=sesh)
    """

    # -- Custom Bybit methods - simulated logic -----------------------------------

    def close_all(self, symbol: str) -> list:
        """Closes all open positions for a symbol (simulated version)."""
        results = []
        for p in self.positions(symbol):
            ok = self.position_close(p.ticket)
            results.append({"ticket": p.ticket, "ok": ok})
        return results

    def get_last_price(self, symbol: str) -> float:
        """Returns the current average price of the symbol (simulated version)."""
        price = self._current_price(symbol)
        if price == 0.0:
            raise TradingError(
                f"No price for '{symbol}'. "
                "Make sure you have fed at least one tick/bar."
            )
        return price

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        """Calculates the required margin using the symbol config (simulated version)."""
        cfg = self._broker._get_symbol_config(symbol)
        price = self._current_price(symbol)
        if not self._broker.use_margin:
            return 0.0
        return volume * cfg.contract_size * price * self._broker.margin_rate

    def calculate_profit(
        self,
        symbol: str,
        volume: float,
        price_open: float,
        price_close: float,
        side: str = "Buy",
    ) -> float:
        """Calculates potential profit (simulated version)."""
        cfg = self._broker._get_symbol_config(symbol)
        is_buy = str(side).lower() == "buy"
        return cfg.calculate_profit(volume, price_open, price_close, is_buy)

    def get_account_info(self) -> dict:
        """Returns extended info from the simulated account (same keys as live)."""
        a = self._broker.account_info()
        return {
            "login":      0,
            "tradeMode":  "Simulator",
            "leverage":   1,
            "balance":    a.balance,
            "equity":     a.equity,
            "margin":     a.margin,
            "marginFree": a.margin_free,
            "profit":     a.profit,
            "currency":   "USDT",
            "company":    "Simulator",
            "server":     "Backtest",
            "marketType": "linear",
        }

    def get_instrument_info(self, symbol: str) -> dict:
        """Returns symbol info from the simulated broker config."""
        cfg = self._broker._get_symbol_config(symbol)
        return {
            "symbol":              symbol,
            "digits":              cfg.digits,
            "tick_step":           cfg.tick_size,
            "tick_value":          cfg.tick_value,
            "trade_contract_size": cfg.contract_size,
            "volume_min":          cfg.volume_min,
            "volume_max":          cfg.volume_max,
            "volume_step":         cfg.volume_step,
            "currency_base":       "",
        }

    def get_leverage(self, symbol: str) -> int:
        return 1

    def set_leverage(self, symbol: str, leverage: int = 1) -> bool:
        return True  # no-op in backtest

    def set_margin_mode(self, margin_mode: str = "isolated") -> bool:
        return True  # no-op in backtest

    def cancel_all_orders(self, symbol: str, order_filter: Optional[str] = None) -> bool:
        for o in self.orders(symbol):
            self.order_delete(o.ticket)
        return True

    def set_trading_stop(
        self,
        symbol: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_volume: Optional[float] = None,
        sl_volume: Optional[float] = None,
    ) -> bool:
        """Applies TP/SL to open positions for the symbol (simulated version)."""
        ok = True
        for p in self.positions(symbol):
            ok = self.position_modify(
                p.ticket,
                sl=float(sl_price) if sl_price else 0.0,
                tp=float(tp_price) if tp_price else 0.0,
            ) and ok
        return ok

    def is_market_open(self, symbol: str) -> bool:
        return True

    def __repr__(self) -> str:
        if self._broker is None:
            return (
                f"SeshBybitSim(feed_type=<auto>, "
                f"pending_symbols={len(self._pending_symbols)})"
            )
        a = self._broker.account_info()
        return (
            f"SeshBybitSim(feed_type={self._feed_type!r}, "
            f"balance={a.balance:.2f}, equity={a.equity:.2f}, "
            f"positions={len(self._broker.get_positions())})"
        )

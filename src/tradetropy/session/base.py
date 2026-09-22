from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta, tzinfo
from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np

from tradetropy.core.broker import (
    OrderType,
    OrderState,
    SymbolConfig,
    KlineBroker,
    TickBroker,
    CommissionType,
    PositionMode,
    TradeRequestActions,
    TradeRequest,
    TradeResult,
    AccountInfo,
    Position,
    Order,
    Deal,
    Trade,
)

if TYPE_CHECKING:
    pass

from tradetropy.exceptions import ConfigError, DataError

from typing import Literal, List, Optional
FeedType = Literal["tick", "kline"]

_logger = logging.getLogger("tradetropy.session")


_ORDER_TYPE_STR_MAP: dict[str, OrderType] = {
    "buy":        OrderType.ORDER_TYPE_BUY,
    "sell":      OrderType.ORDER_TYPE_SELL,
    "buy_limit": OrderType.ORDER_TYPE_BUY_LIMIT,
    "sell_limit": OrderType.ORDER_TYPE_SELL_LIMIT,
    "buy_stop":  OrderType.ORDER_TYPE_BUY_STOP,
    "sell_stop": OrderType.ORDER_TYPE_SELL_STOP,
}

_ORDER_TYPE_TO_STR: dict[OrderType, str] = {
    v: k for k, v in _ORDER_TYPE_STR_MAP.items()
}


TzLike = Union[tzinfo, str, int, float, None]


def _coerce_tz(value: "TzLike") -> tzinfo:
    """
    Normalize timezone representations to tzinfo instance.

    Accepts multiple formats:
        - None -> UTC
        - tzinfo -> returned as-is
        - str -> IANA name (e.g. 'America/New_York') or 'UTC'
        - int/float -> fixed UTC offset in hours (e.g. 3 -> UTC+3)

    Args:
        value (TzLike): Timezone value in various formats.

    Returns:
        tzinfo: Normalized timezone instance.

    Raises:
        ConfigError: If timezone string not recognized or type invalid.
    """
    if value is None:
        return timezone.utc
    if isinstance(value, tzinfo):
        return value
    if isinstance(value, (int, float)):
        return timezone(timedelta(hours=float(value)))
    if isinstance(value, str):
        if value.upper() == "UTC":
            return timezone.utc
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(value)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(
                f"Timezone '{value}' not recognized. Use an IANA name "
                f"(e.g. 'America/New_York'), a numeric hour offset, or a tzinfo."
            ) from exc
    raise ConfigError(
        f"Unsupported timezone type: {type(value).__name__}. "
        f"Use tzinfo, str (IANA), int/float (hour offset) or None."
    )


def _tz_offset_ms(tz: tzinfo, ref_ms: int) -> int:
    """
    Get timezone offset relative to UTC in milliseconds.

    Exact for fixed offsets. For DST zones, evaluated at reference instant
    (sufficient to correct feeds whose epoch is in local time).

    Args:
        tz (tzinfo): Timezone instance.
        ref_ms (int): Reference timestamp in milliseconds.

    Returns:
        int: Offset in milliseconds.
    """
    dt = datetime.fromtimestamp(ref_ms / 1000, tz=timezone.utc)
    off = tz.utcoffset(dt.replace(tzinfo=None))
    if off is None:
        return 0
    return int(off.total_seconds() * 1000)


def _dict_to_position(d: dict) -> Position:
    """
    Convert dict to Position data type.

    Args:
        d (dict): Dictionary with position fields (type, symbol, volume, etc).

    Returns:
        Position: Typed position object.
    """
    ot_raw = d.get("type", "buy")
    ot = (
        _ORDER_TYPE_STR_MAP.get(str(ot_raw).lower(), OrderType.ORDER_TYPE_BUY)
        if isinstance(ot_raw, str)
        else OrderType(int(ot_raw))
    )
    return Position(
        ticket=int(d.get("ticket", 0)),
        symbol=str(d.get("symbol", "")),
        type=ot,
        volume=float(d.get("volume", 0.0)),
        price_open=float(d.get("price_open", 0.0)),
        time=d.get("time", datetime.now(tz=timezone.utc)),
        sl=float(d.get("sl", 0.0)),
        tp=float(d.get("tp", 0.0)),
        profit=float(d.get("profit", 0.0)),
        magic=int(d.get("magic", 0)),
        comment=str(d.get("comment", "")),
    )


def _dict_to_order(d: dict) -> Order:
    """
    Convert dict to Order data type.

    Args:
        d (dict): Dictionary with order fields (type, symbol, price, etc).

    Returns:
        Order: Typed order object.
    """
    ot_raw = d.get("type", "buy_limit")
    ot = (
        _ORDER_TYPE_STR_MAP.get(str(ot_raw).lower(), OrderType.ORDER_TYPE_BUY_LIMIT)
        if isinstance(ot_raw, str)
        else OrderType(int(ot_raw))
    )
    state_raw = d.get("state")
    state = (
        OrderState(int(state_raw))
        if state_raw is not None
        else OrderState.ORDER_STATE_PLACED
    )
    return Order(
        ticket=int(d.get("ticket", 0)),
        symbol=str(d.get("symbol", "")),
        type=ot,
        volume=float(d.get("volume", 0.0)),
        price=float(d.get("price", 0.0)),
        time=d.get("time", datetime.now(tz=timezone.utc)),
        sl=float(d.get("sl", 0.0)),
        tp=float(d.get("tp", 0.0)),
        magic=int(d.get("magic", 0)),
        comment=str(d.get("comment", "")),
        state=state,
    )


def _dict_to_account(d: dict) -> AccountInfo:
    """
    Convert dict to AccountInfo data type.

    Args:
        d (dict): Dictionary with account fields (balance, equity, margin, etc).

    Returns:
        AccountInfo: Typed account info object.
    """
    return AccountInfo(
        balance=float(d.get("balance", 0.0)),
        equity=float(d.get("equity", 0.0)),
        margin=float(d.get("margin", 0.0)),
        margin_free=float(d.get("margin_free", 0.0)),
        profit=float(d.get("profit", 0.0)),
    )


def _dict_to_deal(d: dict) -> Deal:
    ot_raw = d.get("type", "buy")
    ot = (
        _ORDER_TYPE_STR_MAP.get(str(ot_raw).lower(), OrderType.ORDER_TYPE_BUY)
        if isinstance(ot_raw, str)
        else OrderType(int(ot_raw))
    )
    return Deal(
        ticket=int(d.get("ticket", 0)),
        position_id=int(d.get("position_id", 0)),
        symbol=str(d.get("symbol", "")),
        type=ot,
        volume=float(d.get("volume", 0.0)),
        price=float(d.get("price", 0.0)),
        time=d.get("time", datetime.now(tz=timezone.utc)),
        commission=float(d.get("commission", 0.0)),
        profit=float(d.get("profit", 0.0)),
        swap=float(d.get("swap", 0.0)),
        magic=int(d.get("magic", 0)),
        comment=str(d.get("comment", "")),
    )


class Sesh(ABC):
    """
    Minimum contract that any session must fulfill.

    Works with simulated or live brokers. Only 4 methods are mandatory.
    Everything else has default implementations raising NotImplementedError,
    allowing users to implement only what their broker supports.

    Mandatory Methods:
        buy() -> TradeResult
        sell() -> TradeResult
        positions() -> list[Position]
        account_info() -> AccountInfo

    Optional Methods (default: NotImplementedError):
        add_symbol(config) -> None
        position_close(ticket, volume?) -> bool
        position_modify(ticket, sl, tp) -> bool
        order_delete(ticket) -> bool
        orders(symbol?) -> list[Order]
        order_history(symbol?) -> list[Order]
        deals(symbol?) -> list[Deal]
        sync() -> None
        reset() -> None
    """

    @abstractmethod
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
        """Market buy (price=None) or pending buy (price given)."""

    @abstractmethod
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
        """Market sell (price=None) or pending sell (price given)."""

    @abstractmethod
    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Return open positions as typed Position objects."""

    @abstractmethod
    def account_info(self) -> AccountInfo:
        """Current account state."""

    def add_symbol(self, config: SymbolConfig) -> None:
        pass

    def position_close(self, ticket: int, volume: Optional[float] = None) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__}.position_close() not implemented."
        )

    def position_modify(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__}.position_modify() not implemented."
        )

    def order_delete(self, ticket: int) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__}.order_delete() not implemented."
        )

    def orders(self, symbol: Optional[str] = None) -> List[Order]:
        raise NotImplementedError(
            f"{type(self).__name__}.orders() not implemented."
        )

    def order_history(self, symbol: Optional[str] = None) -> List[Order]:
        raise NotImplementedError(
            f"{type(self).__name__}.order_history() not implemented."
        )

    def deals(self, symbol: Optional[str] = None) -> List[Deal]:
        raise NotImplementedError(
            f"{type(self).__name__}.deals() not implemented."
        )

    def trades(self, symbol: Optional[str] = None) -> List[Trade]:
        raise NotImplementedError(
            f"{type(self).__name__}.trades() not implemented."
        )

    def sync(self) -> None:
        """Sync local state with the broker/exchange."""

    def reset(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}.reset() not implemented."
        )

    # ── Streaming (live WebSocket) hooks ──────────────────────────────────────
    # Optional: sessions backed by a WebSocket-capable venue override these so
    # the LiveEngine drives an event-driven feed instead of polling. Polling
    # (and backtest/replay) sessions keep the defaults below.

    @property
    def supports_streaming(self) -> bool:
        """
        Whether this session can provide a real-time WebSocketFeed.

        Returns:
            bool: True if create_feed() returns a usable feed. Default False
            (the engine falls back to polling _fetch_last_tick/_fetch_last_kline).
        """
        return False

    @property
    def supports_user_stream(self) -> bool:
        """
        Whether the feed delivers private user-data events (orders / fills).

        When True, the engine subscribes the ORDER and FILL channels and routes
        them to apply_order_event / apply_fill_event, keeping local order/deal/
        position state current without polling sync(). Default False.
        """
        return False

    @property
    def supports_data_fetch(self) -> bool:
        """
        Whether this session can fetch market data on demand from the venue.

        True only for live sessions backed by a real data source (e.g.
        SeshCCXTLive, via SeshLiveBase). The public fetch_klines / fetch_ticks /
        fetch_orderbook methods check this flag and raise ConfigError when it is
        False, so a backtest / optimize / replay session can never pull live
        ("future") data on demand - the guarantee that keeps backtests free of
        lookahead bias. Default False.
        """
        return False

    def apply_order_event(self, event) -> None:
        """
        Apply a private OrderEvent. No-op by default.

        Streaming-capable live sessions override this to update their local
        order cache from the user-data stream.

        Args:
            event: An OrderEvent from the authenticated stream.
        """
        pass

    def apply_fill_event(self, event) -> None:
        """
        Apply a private FillEvent. No-op by default.

        Streaming-capable live sessions override this to update their local deal
        history and positions from the user-data stream.

        Args:
            event: A FillEvent from the authenticated stream.
        """
        pass

    def create_feed(self, **kwargs):
        """
        Build a WebSocketFeed for real-time streaming. No-op in polling sessions.

        Implemented by streaming-capable live sessions (e.g. SeshCCXTLive). The
        engine passes feed parameters (e.g. timeframe_ms, book_limit) as keyword
        arguments. Warmup still uses the REST _fetch_* methods; the feed only
        carries real-time data after warmup completes.

        Returns:
            WebSocketFeed: A feed ready to be driven by a FeedRunner.

        Raises:
            NotImplementedError: If the session does not support streaming.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.create_feed() not implemented "
            f"(supports_streaming is False)."
        )

    def _fetch_ticks_history(self, symbol: str, limit: int = 500) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}._fetch_ticks_history() not implemented."
        )

    def _fetch_last_tick(self, symbol: str) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}._fetch_last_tick() not implemented."
        )

    def _fetch_klines_history(
        self, symbol: str, interval_ms: int, limit: int = 200
    ) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}._fetch_klines_history() not implemented."
        )

    def _fetch_last_kline(self, symbol: str, interval_ms: int) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}._fetch_last_kline() not implemented."
        )

    def _fetch_orderbook(self, symbol: str, depth: int = 20) -> np.ndarray:
        """
        Fetch a single L2 order-book image as a flat book row (venue hook).

        Implemented by data-capable live sessions (e.g. SeshCCXTLive). Returns
        a [N x (2 + 4*depth)] matrix in the layout of
        core.data_types.book_flat_columns(depth): a REST snapshot is one row
        with kind=0. The public fetch_orderbook() wraps it into a BookData.

        Args:
            symbol (str): Trading symbol.
            depth (int): Number of book levels K per side.

        Returns:
            np.ndarray: Flat book rows [N x (2 + 4*depth)].

        Raises:
            NotImplementedError: If the session cannot fetch a book.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._fetch_orderbook() not implemented."
        )

    # ── Public on-demand data fetch (live only, anti-lookahead) ───────────────
    # A single generic API implemented once here on the base. Connectors only
    # provide the venue-specific _fetch_* primitives above; the normalization
    # and the anti-lookahead guard live here and are shared by every session.

    def _require_data_fetch(self, op: str) -> None:
        """
        Guard the on-demand fetch API: allowed only when supports_data_fetch.

        Raises a clear ConfigError otherwise, so a backtest / optimize / replay
        session can never pull live ("future") data on demand (lookahead bias).

        Args:
            op (str): The public method name, for the error message.

        Raises:
            ConfigError: If this session does not support on-demand data fetch.
        """
        if not self.supports_data_fetch:
            raise ConfigError(
                f"{op}() is not available on {type(self).__name__}: on-demand "
                f"market-data fetch is only supported on live sessions. It is "
                f"disabled in backtest/optimize/replay to prevent lookahead "
                f"bias - a backtest must never pull current 'future' data on "
                f"demand. Preload historical data instead (module-level "
                f"tradetropy.connectors.ccxt.fetch_klines/fetch_ticks, or a data "
                f"file via tradetropy.io)."
            )

    def fetch_klines(
        self, symbol: str, timeframe, *, limit: int = 200
    ) -> "KlineData":
        """
        Fetch recent OHLC candles on demand from the venue (live only).

        Reuses the session's _fetch_klines_history primitive and wraps the
        result into a KlineData. Raises ConfigError outside live mode
        (backtest/optimize/replay) to prevent lookahead bias.

        Args:
            symbol (str): Trading symbol (venue native, e.g. 'BTC/USDT').
            timeframe (int | str): Candle interval ('1m', '5m', '1h', ... or ms).
            limit (int): Maximum number of candles to fetch. Default 200.

        Returns:
            KlineData: [N x 7] candles (turnover column is NaN).

        Raises:
            ConfigError: If the session does not support on-demand fetch.

        Example:
            klines = sesh.fetch_klines('BTC/USDT', '1h', limit=500)
        """
        self._require_data_fetch("fetch_klines")
        from tradetropy.core.constants import parse_timeframe
        from tradetropy.core.data_types import KlineData

        interval_ms = parse_timeframe(timeframe)
        arr = np.asarray(
            self._fetch_klines_history(symbol, interval_ms, limit=limit),
            dtype=np.float64,
        )
        if arr.size == 0:
            arr = np.empty((0, 7), dtype=np.float64)
        elif arr.shape[1] < 7:
            turnover = np.full((len(arr), 1), np.nan, dtype=np.float64)
            arr = np.column_stack([arr[:, :6], turnover])
        return KlineData(symbol=symbol, data=arr, timeframe=interval_ms)

    def fetch_ticks(self, symbol: str, *, limit: int = 500) -> "TickData":
        """
        Fetch recent public trades (ticks) on demand from the venue (live only).

        Reuses the session's _fetch_ticks_history primitive and wraps the
        result into a TickData. Raises ConfigError outside live mode to prevent
        lookahead bias.

        Args:
            symbol (str): Trading symbol.
            limit (int): Maximum number of trades to fetch. Default 500.

        Returns:
            TickData: [N x 7] ticks.

        Raises:
            ConfigError: If the session does not support on-demand fetch.

        Example:
            ticks = sesh.fetch_ticks('BTC/USDT', limit=1000)
        """
        self._require_data_fetch("fetch_ticks")
        from tradetropy.core.data_types import TickData

        arr = np.asarray(
            self._fetch_ticks_history(symbol, limit=limit), dtype=np.float64
        )
        return TickData(symbol=symbol, data=arr)

    def fetch_orderbook(
        self, symbol: str, *, depth: int = 20, tick_size: float = 0.01
    ) -> "BookData":
        """
        Fetch a single L2 order-book image on demand from the venue (live only).

        Reuses the session's _fetch_orderbook primitive and wraps the flat book
        row into a BookData (levels=depth), so the result round-trips through
        the same IO as ticks/klines: ``sesh.fetch_orderbook(...).save(path)``
        then ``tradetropy.io.read_book(path, symbol)`` returns an equivalent
        BookData. Raises ConfigError outside live mode to prevent lookahead
        bias.

        Args:
            symbol (str): Trading symbol.
            depth (int): Number of book levels K per side. Default 20.
            tick_size (float): Minimum price step propagated to BookData.

        Returns:
            BookData: One book event (a REST snapshot is a single row).

        Raises:
            ConfigError: If the session does not support on-demand fetch.

        Example:
            book = sesh.fetch_orderbook('BTC/USDT', depth=20)
            book.bid_px[-1]        # best-N bid prices of the fetched image
        """
        self._require_data_fetch("fetch_orderbook")
        from tradetropy.core.data_types import BookData

        arr = np.asarray(self._fetch_orderbook(symbol, depth), dtype=np.float64)
        return BookData(
            symbol=symbol, data=arr, levels=depth, tick_size=tick_size
        )

    _ultimo_ts: int = 0

    # Source timezone of the feed (informational). Documents what zone the raw
    # timestamps came from. Internal `ts` values are ALWAYS normalized to UTC.
    _data_tz: tzinfo = timezone.utc
    # Timezone used to FORMAT/DISPLAY (`time`, `str_time`). Default UTC.
    _display_tz: tzinfo = timezone.utc

    @property
    def tz(self) -> tzinfo:
        """
        Source timezone of the feed data (informational).

        Indicates what zone the raw timestamps from the broker/exchange came in.
        Internal timestamps (`ts`) are always exposed normalized to UTC, so
        this attribute serves to document "where the data came from".

        - CCXT / Binance / Bybit: UTC (their epochs are real UTC).
        - MT5: the broker server's timezone if `server_tz` was configured,
          in which case `ts` values are corrected to UTC on ingestion.
        """
        return self._data_tz

    @property
    def display_tz(self) -> tzinfo:
        """Timezone used to format `time` and `str_time` (default UTC)."""
        return self._display_tz

    @display_tz.setter
    def display_tz(self, value: "TzLike") -> None:
        self._display_tz = _coerce_tz(value)

    @property
    def ts(self) -> int:
        """Timestamp in milliseconds (UTC) of the last processed data point."""
        if self._ultimo_ts == 0:
            raise DataError("No data processed yet")
        return self._ultimo_ts

    @property
    def time(self) -> datetime:
        """datetime of the last data point (timezone-aware, in `display_tz`)."""
        return datetime.fromtimestamp(
            self.ts / 1000, tz=timezone.utc
        ).astimezone(self._display_tz)

    @property
    def utc_time(self) -> datetime:
        """datetime of the last data point always in UTC (timezone-aware)."""
        return datetime.fromtimestamp(self.ts / 1000, tz=timezone.utc)

    @property
    def str_time(self) -> str:
        """Formatted string '%Y-%m-%d %H:%M:%S' in `display_tz`."""
        return self.time.strftime("%Y-%m-%d %H:%M:%S")


class SeshLiveBase(Sesh):
    """Base class for live sessions."""

    @property
    def supports_data_fetch(self) -> bool:
        """
        Live sessions support on-demand market-data fetch.

        The public fetch_klines / fetch_ticks / fetch_orderbook API delegates to
        the venue-specific _fetch_* primitives. A live session that does not
        implement a given primitive still raises a clear NotImplementedError
        for that specific call.
        """
        return True

    def __init__(self):
        self._cache_positions: dict[int, dict] = {}
        self._cache_orders: dict[int, dict] = {}
        self._cache_orders_hist: dict[int, dict] = {}
        self._deals_hist: list[Deal] = []
        self._balance: float = 0.0
        self._equity: float = 0.0
        # Net position per symbol maintained from the user-data fill stream:
        # {symbol: {"net": float, "avg": float}} (net > 0 long, < 0 short).
        self._stream_positions: dict[str, dict] = {}
        # True once the user-data stream has delivered any event, so position
        # reads can prefer the streamed state over polling.
        self._user_stream_active: bool = False

    def order_history(self, symbol: Optional[str] = None) -> List[Order]:
        items = list(self._cache_orders_hist.values())
        if symbol:
            items = [o for o in items if o.get("symbol") == symbol]
        return [_dict_to_order(d) for d in items]

    def deals(self, symbol: Optional[str] = None) -> List[Deal]:
        if symbol is None:
            return self._deals_hist.copy()
        return [d for d in self._deals_hist if d.symbol == symbol]

    def sync(self) -> None:
        # Positions: critical state. A failure leaves _cache_positions stale and the
        # strategy would operate on positions that no longer reflect the broker. The
        # session is not killed over a transient glitch (one retry), but errors are
        # NEVER silently swallowed - logged at error level with context.
        for intento in (1, 2):
            try:
                raw = self._get_positions_raw()
                if raw is not None:
                    self._cache_positions = {int(d["ticket"]): d for d in raw}
                break
            except Exception:
                _logger.error(
                    "Sesh.sync: failed to refresh positions (attempt %d/2). "
                    "Position cache may be stale.",
                    intento,
                    exc_info=True,
                )
        try:
            acc = self._get_account_raw()
            if acc is not None:
                self._balance = float(acc.get("balance", self._balance))
                self._equity = float(acc.get("equity", self._equity))
        except Exception:
            _logger.error(
                "Sesh.sync: failed to refresh account info. "
                "balance/equity may be stale.",
                exc_info=True,
            )

    def reset(self) -> None:
        raise NotImplementedError(
            "reset() is not supported in live sessions."
        )

    def _get_positions_raw(self) -> Optional[list]:
        return None

    def _get_account_raw(self) -> Optional[dict]:
        return None

    # ── User-data stream hooks (orders / fills) ───────────────────────────────

    def apply_order_event(self, event) -> None:
        """
        Update the local order cache from a streamed OrderEvent.

        Open orders go into ``_cache_orders``; terminal states (filled, closed,
        canceled, rejected, expired) move to ``_cache_orders_hist``. This keeps
        orders() / order_history() current without polling.

        Args:
            event: An OrderEvent from the authenticated user-data stream.
        """
        self._user_stream_active = True
        oid = int(event.order_id)
        ot = (
            OrderType.ORDER_TYPE_BUY if event.side > 0
            else OrderType.ORDER_TYPE_SELL
        )
        d = {
            "ticket": oid,
            "symbol": event.symbol,
            "type": int(ot),
            "volume": float(event.volume),
            "price": float(event.price),
            "time": datetime.fromtimestamp(event.ts / 1000, tz=timezone.utc),
            "state": int(self._order_state_from_status(event.status)),
            "comment": event.client_id,
        }
        terminal = {"closed", "filled", "canceled", "cancelled", "rejected",
                    "expired"}
        if str(event.status).lower() in terminal:
            self._cache_orders.pop(oid, None)
            self._cache_orders_hist[oid] = d
        else:
            self._cache_orders[oid] = d

    def apply_fill_event(self, event) -> None:
        """
        Update local deal history and net positions from a streamed FillEvent.

        Appends a Deal to ``_deals_hist`` and updates the per-symbol net
        position (volume-weighted average price for the open side), so deals()
        and the streamed positions stay current without polling.

        Args:
            event: A FillEvent from the authenticated user-data stream.
        """
        self._user_stream_active = True
        ot = (
            OrderType.ORDER_TYPE_BUY if event.side > 0
            else OrderType.ORDER_TYPE_SELL
        )
        self._deals_hist.append(Deal(
            ticket=int(event.trade_id) if event.trade_id >= 0 else int(event.order_id),
            position_id=int(event.position_id) if event.position_id >= 0 else 0,
            symbol=event.symbol,
            type=ot,
            volume=float(event.volume),
            price=float(event.price),
            time=datetime.fromtimestamp(event.ts / 1000, tz=timezone.utc),
            commission=float(event.fee),
            profit=0.0,
        ))
        self._update_net_position(event.symbol, event.side, event.volume, event.price)

    def _update_net_position(
        self, symbol: str, side: int, volume: float, price: float
    ) -> None:
        """Maintain a net position with a volume-weighted open average price."""
        signed = float(volume) * (1.0 if side > 0 else -1.0)
        pos = self._stream_positions.setdefault(symbol, {"net": 0.0, "avg": 0.0})
        old_net = pos["net"]
        old_avg = pos["avg"]
        new_net = old_net + signed

        if old_net == 0.0 or (old_net > 0) == (signed > 0):
            # Opening or increasing the position: blend the average price.
            denom = abs(old_net) + abs(signed)
            pos["avg"] = (
                (abs(old_net) * old_avg + abs(signed) * float(price)) / denom
                if denom > 0 else float(price)
            )
        elif abs(signed) >= abs(old_net):
            # Closed or flipped: average resets to the flip price (or 0 if flat).
            pos["avg"] = float(price) if new_net != 0.0 else 0.0
        # else: partial reduction - average of the remaining side is unchanged.

        pos["net"] = new_net
        if abs(new_net) < 1e-12:
            self._stream_positions.pop(symbol, None)

    def _streamed_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Build Position objects from the streamed net-position map."""
        out: List[Position] = []
        for sym, pos in self._stream_positions.items():
            if symbol is not None and sym != symbol:
                continue
            net = pos["net"]
            if abs(net) < 1e-12:
                continue
            out.append(Position(
                ticket=abs(hash(sym)) % 1_000_000_000,
                symbol=sym,
                type=(OrderType.ORDER_TYPE_BUY if net > 0
                      else OrderType.ORDER_TYPE_SELL),
                volume=abs(net),
                price_open=pos["avg"],
                time=datetime.now(tz=timezone.utc),
                profit=0.0,
            ))
        return out

    @staticmethod
    def _order_state_from_status(status: str) -> OrderState:
        """Map a venue order status string to an OrderState."""
        return {
            "open": OrderState.ORDER_STATE_PLACED,
            "placed": OrderState.ORDER_STATE_PLACED,
            "closed": OrderState.ORDER_STATE_FILLED,
            "filled": OrderState.ORDER_STATE_FILLED,
            "canceled": OrderState.ORDER_STATE_CANCELED,
            "cancelled": OrderState.ORDER_STATE_CANCELED,
            "rejected": OrderState.ORDER_STATE_REJECTED,
            "expired": OrderState.ORDER_STATE_EXPIRED,
        }.get(str(status).lower(), OrderState.ORDER_STATE_PLACED)

    def __repr__(self) -> str:
        n = len(self._cache_positions)
        return f"{type(self).__name__}(positions={n}, balance={self._balance:.2f})"


class SeshSimulatorBase(Sesh):
    """
    Full Sesh implementation with an internal simulated broker.

    Uses TickBroker (feed_type="tick") or KlineBroker (feed_type="kline").
    """

    def __init__(
        self,
        feed_type: "Literal['tick', 'kline'] | None" = None,
        initial_balance: float = 10000.0,
        commission: float = 0.0,
        commission_type: CommissionType = CommissionType.COMMISSION_TYPE_MONEY,
        position_mode: PositionMode = PositionMode.POSITION_MODE_NETTING,
        use_margin: bool = False,
        margin_rate: float = 0.01,
        slippage_points: int = 0,
        use_spread: bool = False,
        trade_on_close: bool = False,
        execution_price_source: str = "close",
        stop_out_of_money: bool = True,
        finalize_trades: bool = False,
    ):
        """
        Args:
            feed_type: "tick" | "kline" | None
                If None (recommended), the broker type is inferred automatically
                by the engine when calling BacktestEngine.by_ticks() /
                by_klines() (or the equivalent constructor in LiveEngine /
                ReplayEngine). In that case, internal broker creation is
                deferred until the engine injects the feed_type via
                _bind_feed_type().

                If "tick" or "kline" is passed explicitly, the broker is built
                immediately and the engine validates that it matches its mode.

            stop_out_of_money: When True (default), a wiped account (equity <= 0)
                liquidates all open positions and stops the backtest cleanly,
                mirroring backtesting.py. Set False to let equity go negative
                and run to the end.
            finalize_trades: When True, open positions are closed at the last
                bar so they enter the trade-based stats (like backtesting.py's
                finalize_trades=True). Default False, which warns if trades
                remain open at the end.
        """
        self._init_params = {
            'feed_type': feed_type,
            'initial_balance': initial_balance,
            'commission': commission,
            'commission_type': commission_type,
            'position_mode': position_mode,
            'use_margin': use_margin,
            'margin_rate': margin_rate,
            'slippage_points': slippage_points,
            'use_spread': use_spread,
            'trade_on_close': trade_on_close,
            'execution_price_source': execution_price_source,
            'stop_out_of_money': stop_out_of_money,
            'finalize_trades': finalize_trades,
        }

        self._feed_type: "FeedType | None" = feed_type
        self._broker: "KlineBroker | TickBroker | None" = None
        # Symbols registered before feed_type is known. Flushed to the broker
        # as soon as it is constructed (_build_broker).
        self._pending_symbols: list[SymbolConfig] = []

        # _broker_es_default=True indicates the broker was provisionally built
        # as a TickBroker because feed_type was not specified. Account state
        # (balance/positions) is feed-agnostic so the session is fully
        # inspectable before being connected to an engine. If the engine later
        # infers "kline", _bind_feed_type rebuilds it as a KlineBroker with
        # no data loss.
        self._broker_es_default: bool = False

        if feed_type is not None:
            self._build_broker()
        else:
            # Auto mode: build a tick broker as default so the session is
            # immediately inspectable, without polluting self._feed_type
            # (remains None for the engine's auto-detect).
            self._broker_es_default = True
            self._build_broker(feed_type_override="tick")

    def _build_broker(self, feed_type_override: "FeedType | None" = None) -> None:
        """
        Build the internal broker (TickBroker or KlineBroker).

        Uses feed_type_override if provided (provisional broker in auto mode);
        otherwise reads self._init_params['feed_type']. The override does NOT
        mutate _init_params, preserving the engine's auto-detect.
        """
        p = self._init_params
        feed_type = feed_type_override if feed_type_override is not None else p['feed_type']

        if feed_type == "tick":
            self._broker = TickBroker(
                initial_balance=p['initial_balance'],
                commission=p['commission'],
                commission_type=p['commission_type'],
                use_margin=p['use_margin'],
                margin_rate=p['margin_rate'],
                position_mode=p['position_mode'],
                slippage_points=p['slippage_points'],
                stop_out_of_money=p['stop_out_of_money'],
            )
        elif feed_type == "kline":
            self._broker = KlineBroker(
                initial_balance=p['initial_balance'],
                commission=p['commission'],
                commission_type=p['commission_type'],
                use_margin=p['use_margin'],
                margin_rate=p['margin_rate'],
                position_mode=p['position_mode'],
                slippage_points=p['slippage_points'],
                use_spread=p['use_spread'],
                trade_on_close=p['trade_on_close'],
                execution_price_source=p['execution_price_source'],
                stop_out_of_money=p['stop_out_of_money'],
            )
        else:
            raise ConfigError(
                f"feed_type must be 'tick' or 'kline', got: {feed_type!r}"
            )

        # Backtest-end trade finalization policy (read by engine.run()).
        self._broker.finalize_trades = p['finalize_trades']

        # Flush symbols registered before the broker was built.
        for cfg in self._pending_symbols:
            self._broker.add_symbol(cfg)
        self._pending_symbols.clear()

    def _bind_feed_type(self, feed_type: FeedType) -> None:
        """
        Bind the feed_type injected by the engine (by_ticks/by_klines).

        - If the session was created without feed_type (None), it already has a
          provisional tick broker. If the engine operates in "tick" mode it is
          kept; if in "kline" mode it is rebuilt as a KlineBroker transferring
          balance and symbols (no data loss, since no trading should have
          occurred before the engine is attached).
        - If the session already had an explicit feed_type, validates it matches
          and raises ConfigError on conflict (e.g. "tick" session + by_klines()).
        """
        if feed_type not in ("tick", "kline"):
            raise ConfigError(
                f"feed_type must be 'tick' or 'kline', got: {feed_type!r}"
            )

        if self._feed_type is None:
            # Auto mode: a provisional tick broker already exists.
            self._feed_type = feed_type
            self._init_params['feed_type'] = feed_type

            if feed_type == "tick":
                # Provisional broker is already the right type.
                self._broker_es_default = False
                return

            # feed_type == "kline": rebuild. No trading state should exist
            # before connecting the session to an engine.
            b = self._broker
            if b is not None and (
                b.positions or b.orders or b.orders_history or b.deals or b.trades
            ):
                raise ConfigError(
                    "Cannot change feed_type to 'kline' because the session "
                    "already has trading state (positions/orders/deals) created "
                    "before being connected to an engine. Create the session with "
                    "feed_type='kline' explicitly, or do not trade before passing "
                    "it to the engine."
                )

            # Preserve symbols registered in the provisional broker.
            if b is not None:
                self._pending_symbols = list(b.symbols.values())
            self._broker_es_default = False
            self._build_broker()
        elif self._feed_type != feed_type:
            raise ConfigError(
                f"feed_type conflict: session was created with "
                f"feed_type={self._feed_type!r} but the engine operates in "
                f"{feed_type!r} mode. Omit feed_type when creating the session "
                f"so it is inferred automatically, or use the matching engine "
                f"constructor ('{self._feed_type}' -> "
                f"by_{'ticks' if self._feed_type == 'tick' else 'klines'}())."
            )

    def add_symbol(self, config: SymbolConfig) -> None:
        if self._broker is None:
            # feed_type not yet known: queue until the broker is built.
            self._pending_symbols.append(config)
        else:
            self._broker.add_symbol(config)

    def configure_symbol(self, spec) -> None:
        """
        Register a symbol in the internal broker.

        Accepts a SymbolConfig directly or any object with a .symbol_config
        property (e.g. _SymbolSpec from FakeLiveSesh).
        """
        if isinstance(spec, SymbolConfig):
            cfg = spec
        elif hasattr(spec, "symbol_config"):
            cfg = spec.symbol_config
        else:
            raise ConfigError(
                f"configure_symbol() expects SymbolConfig or an object with "
                f"a .symbol_config property, got {type(spec).__name__}"
            )
        self.add_symbol(cfg)

    def clone_config(self) -> dict:
        """
        Return reusable kwargs to recreate this session in another context
        (replay, backtest, etc.).

        Includes _init_params and SymbolConfig per registered symbol.
        """
        if self._broker is not None:
            symbol_configs = dict(self._broker.symbols)
        else:
            # Broker not yet built: use queued symbols.
            symbol_configs = {c.symbol: c for c in self._pending_symbols}
        return {
            **self._init_params,
            "_symbol_configs": symbol_configs,
        }

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
        if price is None:
            r = self._broker.order_send(TradeRequest(
                symbol=symbol,
                action=TradeRequestActions.TRADE_ACTION_DEAL,
                type=OrderType.ORDER_TYPE_BUY,
                volume=volume,
                sl=sl, tp=tp,
                comment=comment, magic=magic,
            ))
        else:
            ot = self._buy_pending_type(symbol, price)
            r = self._broker.order_send(TradeRequest(
                symbol=symbol,
                action=TradeRequestActions.TRADE_ACTION_PENDING,
                type=ot,
                volume=volume,
                price=price,
                sl=sl, tp=tp,
                comment=comment, magic=magic,
            ))
        return r

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
        if price is None:
            r = self._broker.order_send(TradeRequest(
                symbol=symbol,
                action=TradeRequestActions.TRADE_ACTION_DEAL,
                type=OrderType.ORDER_TYPE_SELL,
                volume=volume,
                sl=sl, tp=tp,
                comment=comment, magic=magic,
            ))
        else:
            ot = self._sell_pending_type(symbol, price)
            r = self._broker.order_send(TradeRequest(
                symbol=symbol,
                action=TradeRequestActions.TRADE_ACTION_PENDING,
                type=ot,
                volume=volume,
                price=price,
                sl=sl, tp=tp,
                comment=comment, magic=magic,
            ))
        return r

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        return self._broker.get_positions(symbol)

    def account_info(self) -> AccountInfo:
        return self._broker.account_info()

    def position_close(self, ticket: int, volume: Optional[float] = None) -> bool:
        return self._broker.position_close(ticket, volume)

    def position_modify(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        return self._broker.position_modify(ticket, sl=sl, tp=tp)

    def order_delete(self, ticket: int) -> bool:
        return self._broker.order_delete(ticket)

    def orders(self, symbol: Optional[str] = None) -> List[Order]:
        return self._broker.get_orders(symbol)

    def order_history(self, symbol: Optional[str] = None) -> List[Order]:
        return self._broker.get_order_history(symbol)

    def deals(self, symbol: Optional[str] = None) -> List[Deal]:
        return self._broker.get_deal_history(symbol)

    def trades(self, symbol: Optional[str] = None) -> List[Trade]:
        return self._broker.get_trades(symbol)

    def sync(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def reconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def is_market_open(self, symbol: str) -> bool:
        return True

    def reset(self) -> None:
        self._broker.reset()

    def clone(self) -> "SeshSimulatorBase":
        return self.__class__(**self._init_params)

    def _current_price(self, symbol: str) -> float:
        p = self._broker._current_prices.get(symbol)
        if p:
            bid = p.get("bid", 0.0)
            ask = p.get("ask", 0.0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
        return 0.0

    def _current_ask(self, symbol: str) -> float:
        p = self._broker._current_prices.get(symbol)
        return float(p.get("ask", 0.0)) if p else 0.0

    def _current_bid(self, symbol: str) -> float:
        p = self._broker._current_prices.get(symbol)
        return float(p.get("bid", 0.0)) if p else 0.0

    def _buy_pending_type(self, symbol: str, price: float) -> OrderType:
        ask = self._current_ask(symbol)
        return (
            OrderType.ORDER_TYPE_BUY_LIMIT
            if ask == 0.0 or price < ask
            else OrderType.ORDER_TYPE_BUY_STOP
        )

    def _sell_pending_type(self, symbol: str, price: float) -> OrderType:
        bid = self._current_bid(symbol)
        return (
            OrderType.ORDER_TYPE_SELL_LIMIT
            if bid == 0.0 or price > bid
            else OrderType.ORDER_TYPE_SELL_STOP
        )

    def __repr__(self) -> str:
        if self._broker is None:
            return (
                f"{type(self).__name__}(feed_type=<auto>, "
                f"pending_symbols={len(self._pending_symbols)})"
            )
        a = self._broker.account_info()
        return (
            f"SeshSimulatorBase(feed_type={self._feed_type!r}, "
            f"balance={a.balance:.2f}, equity={a.equity:.2f}, "
            f"positions={len(self._broker.get_positions())})"
        )

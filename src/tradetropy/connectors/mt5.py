"""
=================
Two symmetrical implementations for MetaTrader 5:

  SeshMT5Live  - connects to the real MT5 terminal via the MetaTrader5 SDK.
  SeshMT5Sim   - uses the internal simulated broker (TickBroker/KlineBroker).

GOAL
----
A strategy runs identically in backtest and live without changing
a single line:

    class MyStrategy(Strategy):
        def on_data(self):
            pos = self.sesh.positions("MES")
            if not pos:
                r = self.sesh.buy("MES", volume=1, sl=6910.0, tp=6940.0)
            acc = self.sesh.account_info()
            if acc.equity < acc.balance * 0.95:
                self.sesh.close_all("MES")

    # Backtest:
    sesh = SeshMT5Sim(initial_balance=10_000, commission=0.6)
    engine = BacktestEngine.by_ticks(MyStrategy(), data=(ticks,), sesh=sesh)

    # Live:
    sesh = SeshMT5Live(login=123456, password="pass", server="BrokerDemo")
    engine = LiveEngine.by_ticks(MyStrategy(), sesh=sesh)

METHODS AVAILABLE IN BOTH CLASSES (identical)
----------------------------------------------
Mandatory (Sesh):
    buy(symbol, volume, price?, sl?, tp?, comment?, magic?) -> TradeResult
    sell(symbol, volume, price?, sl?, tp?, comment?, magic?) -> TradeResult
    positions(symbol?)  -> list[Position]
    account_info()      -> AccountInfo

Optional (implemented in both):
    position_close(ticket, volume?)   -> bool
    position_modify(ticket, sl?, tp?) -> bool
    order_delete(ticket)              -> bool
    orders(symbol?)                   -> list[Order]
    order_history(symbol?)            -> list[Order]
    deals(symbol?)                    -> list[Deal]
    sync()                            -> None
    reset()                           -> None  (SeshMT5Sim only)

Custom MT5 methods (implemented in both with equivalent logic):
    close_all(symbol)                          -> list[dict]
    get_last_price(symbol)                     -> float
    calculate_margin(symbol, volume, side?)    -> float
    calculate_profit(symbol, volume, open, close, side?) -> float
    get_account_info()                         -> dict
    get_instrument_info(symbol)                -> dict
    is_market_open(symbol)                     -> bool   (SeshMT5Live only)

ADDING NEW CUSTOM METHODS
--------------------------
If you need an extra method (e.g. get_funding_rate), add it in both classes
with the same signature. SeshMT5Sim can return a fixed or computed value:

    class SeshMT5Live(...):
        def get_funding_rate(self, symbol: str) -> float:
            return float(self._mt5.symbol_info(symbol).swap_long)

    class SeshMT5Sim(...):
        def get_funding_rate(self, symbol: str) -> float:
            return 0.0001  # fixed value for backtests

CREATING A SESH FOR ANOTHER BROKER
------------------------------------
Inherit directly from Sesh (or SeshLiveBase for live) and implement
the 4 mandatory methods. Optional methods are added incrementally:

    class SeshBybitLive(Sesh):
        def __init__(self, api_key, secret):
            from pybit.unified_trading import HTTP
            self._client = HTTP(api_key=api_key, api_secret=secret)

        def buy(self, symbol, volume, price=None, sl=0, tp=0, comment="", magic=0):
            r = self._client.place_order(
                category="linear", symbol=symbol, side="Buy",
                orderType="Market" if price is None else "Limit",
                qty=str(volume),
            )
            ok = r["retCode"] == 0
            retcode = (TradeResult.TRADE_RETCODE_DONE if ok and price is None
                       else TradeResult.TRADE_RETCODE_PLACED if ok
                       else TradeResult.TRADE_RETCODE_INVALID)
            return TradeResult(retcode=retcode,
                               order=int(r["result"].get("orderId", 0)))

        def sell(self, symbol, volume, price=None, sl=0, tp=0, comment="", magic=0):
            ...

        def positions(self, symbol=None):
            raw = self._client.get_positions(category="linear", symbol=symbol)
            return [Position(ticket=int(p["positionIdx"]), symbol=p["symbol"],
                             type=OrderType.ORDER_TYPE_BUY if p["side"]=="Buy"
                                  else OrderType.ORDER_TYPE_SELL,
                             volume=float(p["size"]),
                             price_open=float(p["avgPrice"]),
                             time=datetime.now(tz=timezone.utc))
                    for p in raw["result"]["list"] if float(p["size"]) > 0]

        @property
        def account(self):
            r = self._client.get_wallet_balance(accountType="UNIFIED")
            a = r["result"]["list"][0]
            return AccountInfo(balance=float(a["totalWalletBalance"]),
                               equity=float(a["totalEquity"]),
                               profit=float(a["totalUnrealisedPnl"]))
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from tradetropy.core.broker import (
    AccountInfo,
    CommissionType,
    Deal,
    Order,
    OrderState,
    OrderType,
    Position,
    PositionMode,
    TradeResult,
    SymbolConfig,
)
from tradetropy.exceptions import ConfigError, TradingError, ConnectionError
from tradetropy.session.base import (
    Sesh,
    SeshSimulatorBase,
    SeshLiveBase,
    TzLike,
    _coerce_tz,
    _tz_offset_ms,
    _ORDER_TYPE_TO_STR,
    _dict_to_position,
    _dict_to_order,
    _dict_to_account,
    _dict_to_deal,
)
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.connectors._disclaimer import emit_live_disclaimer

_logger = logging.getLogger("tradetropy.connectors.mt5")

# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_MS_TO_TF_KEY = {
    60_000:        "1m",
    300_000:       "5m",
    900_000:       "15m",
    1_800_000:     "30m",
    3_600_000:     "1h",
    7_200_000:     "2h",
    14_400_000:    "4h",
    86_400_000:    "1D",
    604_800_000:   "1W",
    # MT5's TIMEFRAME_MN1 corresponds to the public parse_timeframe('1mo')
    # unit (fixed 30-day duration, see core/constants.py::_TF_UNIT_MS).
    2_592_000_000: "1M",
}

_TF_MAP_KEYS = {
    "1m":     "TIMEFRAME_M1",
    "5m":     "TIMEFRAME_M5",
    "15m":    "TIMEFRAME_M15",
    "30m":    "TIMEFRAME_M30",
    "1h":     "TIMEFRAME_H1",
    "2h":     "TIMEFRAME_H2",
    "4h":     "TIMEFRAME_H4",
    "1D":     "TIMEFRAME_D1",
    "1W":     "TIMEFRAME_W1",
    "1M":     "TIMEFRAME_MN1",
}

_MT5_TYPE_TO_STR = {
    0: "buy",
    1: "sell",
    2: "buy_limit",
    3: "sell_limit",
    4: "buy_stop",
    5: "sell_stop",
}

_ticket_counter = [0]


def _next_ticket() -> int:
    _ticket_counter[0] += 1
    return _ticket_counter[0]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 - SESH MT5 LIVE
# ══════════════════════════════════════════════════════════════════════════════


class SeshMT5Live(SeshLiveBase):
    """
    Live trading session on MetaTrader 5.

    Implements all Sesh methods plus MT5-specific methods. The strategy
    can call exactly the same methods as in SeshMT5Sim - zero changes to
    switch from backtest to production.

    Requires: pip install MetaTrader5

    Args:
        login: MT5 account number.
        password: Account password.
        server: Server name (e.g. 'AMPGlobalUSA-Demo').
        path: Path to MT5 terminal (optional).
        timeout: Connection timeout in milliseconds (default 60000).
        deviation: Max deviation in points for market orders (default 20).
        magic: Default magic number (default 0).
        market_type: 'forex' or 'futures' - affects fill mode (default 'futures').
        server_tz: Broker server timezone. MT5 returns timestamps in server
                   time (not UTC), typically UTC+2 or UTC+3. Set this to
                   normalize internal timestamps to real UTC.
                   Accepts IANA name ('Europe/Helsinki'), offset hours (e.g. 3),
                   tzinfo object, or None (default - assumes UTC).
        display_tz: Timezone for formatting sesh.time / sesh.str_time.
                    Default UTC.

    Example:
        sesh = SeshMT5Live(login=123456, password='pass', server='Demo')
        pos = sesh.positions('EURUSD')
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str = "",
        timeout: int = 60_000,
        deviation: int = 20,
        magic: int = 0,
        market_type: str = "futures",
        server_tz: "TzLike" = None,
        display_tz: "TzLike" = None,
    ):
        super().__init__()
        emit_live_disclaimer()
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout = timeout
        self._deviation = deviation
        self._default_magic = magic
        self._market_type = market_type.lower()
        self._mt5 = None
        self._TIMEFRAME_MAP: dict = {}

        # Broker server timezone. If specified, raw MT5 timestamps (in server
        # time) are corrected to UTC on ingestion.
        self._data_tz = _coerce_tz(server_tz)
        self._display_tz = _coerce_tz(display_tz)

        self._connect()

    def _to_utc_ms(self, ts_ms: float) -> float:
        """
        Convert timestamp from MT5 server time (ms) to real UTC (ms).

        MT5 stores time as if it were UTC but it is actually the broker server
        time. Subtract the server offset to get true UTC. If server_tz is not
        set (data_tz == UTC), this is a no-op.

        Args:
            ts_ms: Timestamp in milliseconds (server time).

        Returns:
            Timestamp in milliseconds (UTC).
        """
        if self._data_tz is timezone.utc:
            return ts_ms
        offset_ms = _tz_offset_ms(self._data_tz, int(ts_ms))
        return ts_ms - offset_ms

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise ImportError(
                "MetaTrader5 is not installed. Install it with: pip install MetaTrader5"
            ) from exc

        self._mt5 = mt5
        self._TIMEFRAME_MAP = {
            k: getattr(mt5, v)
            for k, v in _TF_MAP_KEYS.items()
            if hasattr(mt5, v)
        }
        kwargs = dict(
            login=self._login,
            password=self._password,
            server=self._server,
            timeout=self._timeout,
        )
        if self._path:
            kwargs["path"] = self._path

        if not mt5.initialize(**kwargs):
            raise ConnectionError(
                f"Could not connect to MT5: {mt5.last_error()}\n"
                "Verify that MetaTrader 5 is open and the credentials are correct."
            )
        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise ConnectionError("Could not retrieve account information.")

        _logger.info(
            "SeshMT5Live connected - Account: %s | Broker: %s | Server: %s",
            info.login, info.company, info.server,
        )
        self.sync()

    def disconnect(self):
        if self._mt5:
            self._mt5.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self) -> str:
        n = len(self._cache_positions)
        return (
            f"SeshMT5Live(login={self._login}, server={self._server!r}, "
            f"positions={n}, market={self._market_type})"
        )

    # ── 4 mandatory methods ────────────────────────────────────────────────────

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
        return self._send_order_mt5(
            symbol, volume, price, sl, tp, comment, magic, is_buy=True
        )

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
        return self._send_order_mt5(
            symbol, volume, price, sl, tp, comment, magic, is_buy=False
        )

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        mt5 = self._mt5
        raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not raw:
            return []
        return [
            Position(
                ticket=int(p.ticket),
                symbol=p.symbol,
                type=OrderType.ORDER_TYPE_BUY if p.type == 0 else OrderType.ORDER_TYPE_SELL,
                volume=float(p.volume),
                price_open=float(p.price_open),
                time=datetime.fromtimestamp(p.time, tz=timezone.utc),
                sl=float(p.sl),
                tp=float(p.tp),
                profit=float(p.profit),
                magic=int(p.magic),
                comment=str(p.comment),
            )
            for p in raw
        ]

    def account_info(self) -> AccountInfo:
        mt5 = self._mt5
        info = mt5.account_info() if mt5 else None
        if info is None:
            return AccountInfo(balance=self._balance, equity=self._equity)
        return AccountInfo(
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            margin_free=float(info.margin_free),
            profit=float(info.profit),
        )

    # ── Optional methods ───────────────────────────────────────────────────────

    def position_close(self, ticket: int, volume: Optional[float] = None) -> bool:
        mt5 = self._mt5
        # Find position in cache or in real-time
        pos_dict = self._cache_positions.get(ticket)
        if pos_dict is None:
            raw = mt5.positions_get()
            if raw:
                for p in raw:
                    if p.ticket == ticket:
                        pos_dict = {
                            "symbol": p.symbol,
                            "type": p.type,
                            "volume": p.volume,
                            "magic": p.magic,
                        }
                        break
        if pos_dict is None:
            return False

        symbol = pos_dict["symbol"]
        pos_type = pos_dict["type"]  # 0=buy, 1=sell
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        close_px = tick.bid if pos_type == 0 else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
        close_vol = float(volume) if volume else float(pos_dict["volume"])

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_vol,
            "type": close_type,
            "position": ticket,
            "price": close_px,
            "deviation": self._deviation,
            "magic": pos_dict.get("magic", self._default_magic),
            "comment": f"close #{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }
        result = mt5.order_send(req)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return False

        profit = 0.0
        history = mt5.history_deals_get(position=ticket)
        if history:
            profit = sum(
                d.profit for d in history if d.entry == mt5.DEAL_ENTRY_OUT
            )
        self._register_close(ticket, close_vol, close_px, profit, f"close #{ticket}")
        return True

    def position_modify(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        mt5 = self._mt5
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket,
               "sl": float(sl), "tp": float(tp)}
        result = mt5.order_send(req)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok and ticket in self._cache_positions:
            if sl:
                self._cache_positions[ticket]["sl"] = sl
            if tp:
                self._cache_positions[ticket]["tp"] = tp
        return ok

    def order_delete(self, ticket: int) -> bool:
        mt5 = self._mt5
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
        result = mt5.order_send(req)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok and ticket in self._cache_orders:
            d = self._cache_orders.pop(ticket)
            d["state"] = "canceled"
            self._cache_orders_hist[ticket] = d
        return ok

    def orders(self, symbol: Optional[str] = None) -> List[Order]:
        from tradetropy.sim_sesh import _ORDER_TYPE_STR_MAP
        mt5 = self._mt5
        raw = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if not raw:
            return []
        result = []
        for o in raw:
            ot_str = _MT5_TYPE_TO_STR.get(o.type, "buy_limit")
            ot = _ORDER_TYPE_STR_MAP.get(ot_str, OrderType.ORDER_TYPE_BUY_LIMIT)
            result.append(Order(
                ticket=int(o.ticket),
                symbol=o.symbol,
                type=ot,
                volume=float(o.volume_current),
                price=float(o.price_open),
                time=datetime.fromtimestamp(o.time_setup, tz=timezone.utc),
                sl=float(o.sl),
                tp=float(o.tp),
                magic=int(o.magic),
                comment=str(o.comment),
            ))
        return result

    def order_history(self, symbol: Optional[str] = None) -> List[Order]:
        mt5 = self._mt5
        from datetime import timedelta
        now = datetime.now(tz=timezone.utc)
        until = now - timedelta(days=30)
        raw = mt5.history_orders_get(datetime_from=until, datetime_to=now)
        if not raw:
            return []
        result = []
        for o in raw:
            if symbol is not None and o.symbol != symbol:
                continue
            from tradetropy.sim_sesh import _ORDER_TYPE_STR_MAP
            ot = _MT5_TYPE_TO_STR.get(o.type, "buy")
            result.append(Order(
                ticket=int(o.ticket),
                symbol=o.symbol,
                type=_ORDER_TYPE_STR_MAP.get(ot, OrderType.ORDER_TYPE_BUY),
                volume=float(o.volume_current),
                price=float(o.price_open),
                time=datetime.fromtimestamp(o.time_setup, tz=timezone.utc),
                sl=float(o.sl),
                tp=float(o.tp),
                magic=int(o.magic),
                comment=str(o.comment),
                state=OrderState.ORDER_STATE_FILLED,
            ))
        return result

    def deals(self, symbol: Optional[str] = None) -> List[Deal]:
        mt5 = self._mt5
        from datetime import timedelta
        now = datetime.now(tz=timezone.utc)
        until = now - timedelta(days=30)
        raw = mt5.history_deals_get(datetime_from=until, datetime_to=now)
        if not raw:
            return []
        result = []
        for d in raw:
            if symbol is not None and d.symbol != symbol:
                continue
            result.append(Deal(
                ticket=int(d.ticket),
                position_id=int(d.position),
                symbol=d.symbol,
                type=OrderType.ORDER_TYPE_BUY if d.entry == mt5.DEAL_ENTRY_IN
                     else OrderType.ORDER_TYPE_SELL,
                volume=float(d.volume),
                price=float(d.price),
                time=datetime.fromtimestamp(d.time, tz=timezone.utc),
                commission=float(d.commission),
                profit=float(d.profit),
                swap=float(d.swap),
                magic=int(d.magic),
                comment=str(d.comment),
            ))
        return result

    def sync(self) -> None:
        # Positions: critical state (see Sesh.sync). Log at error + retry;
        # never swallow the failure silently.
        for attempt in (1, 2):
            try:
                raw = self._mt5.positions_get()
                self._cache_positions = {}
                if raw:
                    for p in raw:
                        self._cache_positions[int(p.ticket)] = {
                            "symbol": p.symbol,
                            "type": p.type,
                            "volume": p.volume,
                            "magic": p.magic,
                        }
                break
            except Exception:
                _logger.error(
                    "SeshMT5Live.sync: failed to get positions from terminal "
                    "(attempt %d/2). Position cache may be stale.",
                    attempt,
                    exc_info=True,
                )
        try:
            info = self._mt5.account_info()
            if info:
                self._balance = float(info.balance)
                self._equity = float(info.equity)
        except Exception:
            _logger.error(
                "SeshMT5Live.sync: failed to get account info from terminal. "
                "balance/equity may be stale.",
                exc_info=True,
            )

    # ── Data primitives ────────────────────────────────────────────────────────

    def _fetch_ticks_history(self, symbol: str, limit: int = 500) -> np.ndarray:
        mt5 = self._mt5
        ticks = mt5.copy_ticks_from(
            symbol, datetime.now(tz=timezone.utc), limit, mt5.COPY_TICKS_ALL
        )
        if ticks is None or len(ticks) == 0:
            return np.empty((0, N_TICK_COLS), dtype=np.float64)

        names = ticks.dtype.names or ()
        out = np.empty((len(ticks), N_TICK_COLS), dtype=np.float64)

        # time_msc (ms) exists in forex/CFD; CME futures only have time (s)
        if "time_msc" in names:
            out[:, _TICK_COL["ts"]] = ticks["time_msc"].astype(np.float64)
        elif "time" in names:
            out[:, _TICK_COL["ts"]] = ticks["time"].astype(np.float64) * 1000.0
        else:
            out[:, _TICK_COL["ts"]] = 0.0

        # Normalize MT5 server time to real UTC (no-op if server_tz is not set)
        if self._data_tz is not timezone.utc and len(out):
            ref = next((t for t in out[:, _TICK_COL["ts"]] if t > 0), 0.0)
            offset_ms = _tz_offset_ms(self._data_tz, int(ref))
            mask = out[:, _TICK_COL["ts"]] > 0
            out[mask, _TICK_COL["ts"]] -= offset_ms

        out[:, _TICK_COL["bid"]]         = ticks["bid"].astype(np.float64) if "bid" in names else 0.0
        out[:, _TICK_COL["ask"]]         = ticks["ask"].astype(np.float64) if "ask" in names else 0.0
        out[:, _TICK_COL["volume"]]      = ticks["volume"].astype(np.float64) if "volume" in names else 0.0
        out[:, _TICK_COL["flags"]]       = ticks["flags"].astype(np.float64) if "flags" in names else 0.0
        out[:, _TICK_COL["volume_real"]] = ticks["volume_real"].astype(np.float64) if "volume_real" in names else np.nan

        # price: prefer "last" (futures), fall back to mid bid/ask if 0 or absent
        if "last" in names:
            last = ticks["last"].astype(np.float64)
        else:
            last = np.zeros(len(ticks), dtype=np.float64)
        bid = out[:, _TICK_COL["bid"]]
        ask = out[:, _TICK_COL["ask"]]
        mid = np.where(bid + ask > 0, (bid + ask) / 2.0, last)
        out[:, _TICK_COL["price"]] = np.where(last > 0, last, mid)
        return out

    def _fetch_last_tick(self, symbol: str) -> np.ndarray:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise TradingError(f"Could not get tick for '{symbol}'.")

        out = np.empty(N_TICK_COLS, dtype=np.float64)

        # time_msc exists in forex/CFD; CME futures sometimes only have time (s).
        # If time_msc is 0 (field absent or unsupported) we use time * 1000.
        time_msc = float(getattr(tick, "time_msc", 0))
        if time_msc <= 0:
            time_msc = float(getattr(tick, "time", 0)) * 1000.0
        out[_TICK_COL["ts"]] = self._to_utc_ms(time_msc)

        out[_TICK_COL["bid"]]         = float(tick.bid)
        out[_TICK_COL["ask"]]         = float(tick.ask)
        out[_TICK_COL["volume"]]      = float(getattr(tick, "volume", 0))
        out[_TICK_COL["flags"]]       = float(getattr(tick, "flags", 0))
        out[_TICK_COL["volume_real"]] = float(getattr(tick, "volume_real", np.nan))

        # price: prefer "last" (futures), fall back to mid if 0 or absent
        last = float(getattr(tick, "last", 0.0))
        if last > 0:
            out[_TICK_COL["price"]] = last
        elif tick.bid > 0 and tick.ask > 0:
            out[_TICK_COL["price"]] = (tick.bid + tick.ask) / 2.0
        else:
            out[_TICK_COL["price"]] = last  # 0.0, invalid data signal
        return out

    def _fetch_klines_history(
        self, symbol: str, interval_ms: int, limit: int = 200
    ) -> np.ndarray:
        import datetime as _dt
        mt5 = self._mt5
        tf_key = _MS_TO_TF_KEY.get(interval_ms)
        if tf_key is None:
            raise ConfigError(
                f"interval_ms={interval_ms} has no equivalent MT5 timeframe. "
                f"Supported: {list(_MS_TO_TF_KEY.keys())}"
            )
        tf = self._TIMEFRAME_MAP.get(tf_key)
        if tf is None:
            raise ConfigError(f"Timeframe '{tf_key}' not available in this version of MT5.")

        rates = mt5.copy_rates_from(symbol, tf, _dt.datetime.now(), limit)
        if rates is None or len(rates) == 0:
            return np.empty((0, 6), dtype=np.float64)

        ts_ms = rates["time"].astype(np.float64) * 1000
        # Normalize MT5 server time to real UTC (no-op if server_tz is not set)
        if self._data_tz is not timezone.utc and len(ts_ms):
            ref = next((t for t in ts_ms if t > 0), 0.0)
            offset_ms = _tz_offset_ms(self._data_tz, int(ref))
            ts_ms = np.where(ts_ms > 0, ts_ms - offset_ms, ts_ms)

        return np.column_stack([
            ts_ms,
            rates["open"].astype(np.float64),
            rates["high"].astype(np.float64),
            rates["low"].astype(np.float64),
            rates["close"].astype(np.float64),
            rates["tick_volume"].astype(np.float64),
        ])

    def _fetch_last_kline(self, symbol: str, interval_ms: int) -> np.ndarray:
        klines = self._fetch_klines_history(symbol, interval_ms, limit=2)
        if len(klines) < 1:
            raise TradingError(
                f"Could not get candles for '{symbol}' "
                f"(interval_ms={interval_ms})."
            )
        return klines[-1]

    # ── Custom MT5 methods ────────────────────────────────────────────────────

    def close_all(self, symbol: str) -> list:
        """Close all open positions for a symbol."""
        results = []
        for p in (self._mt5.positions_get(symbol=symbol) or []):
            ok = self.position_close(p.ticket)
            results.append({"ticket": p.ticket, "ok": ok})
        return results

    def get_last_price(self, symbol: str) -> float:
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise TradingError(f"No price for '{symbol}'.")
        if tick.last > 0:
            return float(tick.last)
        if tick.bid > 0 and tick.ask > 0:
            return float((tick.bid + tick.ask) / 2)
        raise TradingError(f"No valid price for '{symbol}'.")

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        mt5 = self._mt5
        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if side == "Buy" else tick.bid
        margin = mt5.order_calc_margin(
            order_type, symbol, self._normalize_volume(symbol, volume), price
        )
        return float(margin) if margin is not None else 0.0

    def calculate_profit(
        self,
        symbol: str,
        volume: float,
        price_open: float,
        price_close: float,
        side: str = "Buy",
    ) -> float:
        mt5 = self._mt5
        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        profit = mt5.order_calc_profit(
            order_type, symbol,
            self._normalize_volume(symbol, volume),
            self._normalize_price(symbol, price_open),
            self._normalize_price(symbol, price_close),
        )
        return float(profit) if profit is not None else 0.0

    def get_account_info(self) -> dict:
        info = self._mt5.account_info()
        if info is None:
            raise TradingError("Could not retrieve account information.")
        return {
            "login":       info.login,
            "leverage":    info.leverage,
            "balance":     float(info.balance),
            "equity":      float(info.equity),
            "margin":      float(info.margin),
            "margin_free": float(info.margin_free),
            "profit":      float(info.profit),
            "currency":    info.currency,
            "company":     info.company,
            "server":      info.server,
        }

    def get_instrument_info(self, symbol: str) -> dict:
        self._ensure_symbol_visible(symbol)
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise TradingError(f"No info for '{symbol}'.")
        return {
            "symbol":              info.name,
            "digits":              info.digits,
            "volume_min":          float(info.volume_min),
            "volume_max":          float(info.volume_max),
            "volume_step":         float(info.volume_step),
            "tick_step":           float(info.trade_tick_size),
            "tick_value":          float(info.trade_tick_value),
            "trade_contract_size": float(info.trade_contract_size),
        }

    def is_market_open(self, symbol: str) -> bool:
        import datetime as _dt
        mt5 = self._mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        if info.trade_mode not in (
            mt5.SYMBOL_TRADE_MODE_FULL,
            mt5.SYMBOL_TRADE_MODE_LONGONLY,
            mt5.SYMBOL_TRADE_MODE_SHORTONLY,
        ):
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        return _dt.datetime.now() - _dt.datetime.fromtimestamp(tick.time) < \
               _dt.timedelta(minutes=5)

    # ── Internal logic ────────────────────────────────────────────────────────

    def _send_order_mt5(
        self,
        symbol: str,
        volume: float,
        price: Optional[float],
        sl: float,
        tp: float,
        comment: str,
        magic: int,
        is_buy: bool,
    ) -> TradeResult:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_MARKET_CLOSED,
                comment=f"No price for {symbol}",
            )

        vol_norm = self._normalize_volume(symbol, volume)
        magic_real = magic or self._default_magic

        if price is None:
            action = mt5.TRADE_ACTION_DEAL
            exec_price = tick.ask if is_buy else tick.bid
            ot = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        else:
            action = mt5.TRADE_ACTION_PENDING
            exec_price = price
            current = tick.ask if is_buy else tick.bid
            if is_buy:
                ot = (mt5.ORDER_TYPE_BUY_LIMIT if price < current
                      else mt5.ORDER_TYPE_BUY_STOP)
            else:
                ot = (mt5.ORDER_TYPE_SELL_LIMIT if price > current
                      else mt5.ORDER_TYPE_SELL_STOP)

        req = {
            "action":       action,
            "symbol":       symbol,
            "volume":       float(vol_norm),
            "type":         ot,
            "price":        exec_price,
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    self._deviation,
            "magic":        magic_real,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }
        result = mt5.order_send(req)
        if result is None:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID,
                comment=f"order_send returned None: {mt5.last_error()}",
            )

        ok = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)

        if ok and price is None:
            # Record opening in cache
            self._cache_positions[result.order] = {
                "symbol": symbol, "type": 0 if is_buy else 1,
                "volume": result.volume, "magic": magic_real,
            }

        return TradeResult(
            retcode=result.retcode,
            order=result.order,
            deal=result.deal,
            volume=result.volume,
            price=result.price,
            comment=result.comment,
        )

    def _register_close(
        self, ticket: int, volume: float, price: float, profit: float, comment: str
    ):
        pos = self._cache_positions.get(ticket)
        if pos is None:
            return
        self._deals_hist.append(
            Deal(
                ticket=_next_ticket(),
                position_id=ticket,
                symbol=pos["symbol"],
                type=OrderType.ORDER_TYPE_SELL if pos["type"] == 0
                     else OrderType.ORDER_TYPE_BUY,
                volume=volume,
                price=price,
                time=_now(),
                commission=0.0,
                profit=profit,
                comment=comment,
                magic=pos.get("magic", 0),
            )
        )
        self._balance += profit
        if volume >= pos["volume"]:
            del self._cache_positions[ticket]
        else:
            pos["volume"] -= volume

    def _get_filling_mode(self, symbol: str) -> int:
        mt5 = self._mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_FOK
        filling = info.filling_mode
        if self._market_type == "futures":
            if filling & 1:
                return mt5.ORDER_FILLING_FOK
            if filling & 2:
                return mt5.ORDER_FILLING_IOC
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        if filling & 4:
            return mt5.ORDER_FILLING_RETURN
        return mt5.ORDER_FILLING_FOK

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        info = self._mt5.symbol_info(symbol)
        if info is None:
            return volume
        step = info.volume_step
        vol = round(volume / step) * step
        return max(info.volume_min, min(vol, info.volume_max))

    def _normalize_price(self, symbol: str, price: float) -> float:
        info = self._mt5.symbol_info(symbol)
        return round(price, info.digits) if info else price

    def _ensure_symbol_visible(self, symbol: str) -> bool:
        mt5 = self._mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        return True if info.visible else mt5.symbol_select(symbol, True)

    def initialize(self):
        # Initialize MT5
        if not mt5.initialize():
            if path:
                if not mt5.initialize(path=path):
                    raise ConnectionError(
                        f"Error initializing MT5: {mt5.last_error()}\n"
                        f"Verify that MetaTrader 5 is installed and the path is correct.\n"
                        f"Path attempted: {path}"
                    )
            else:
                raise ConnectionError(
                    f"Error initializing MT5: {mt5.last_error()}\n"
                    "Verify that MetaTrader 5 is installed.\n"
                    "If it is in a non-standard location, provide the 'path' parameter."
                )
        

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - SESH MT5 SIM (simulated - symmetrical to SeshMT5Live)
# ══════════════════════════════════════════════════════════════════════════════


class SeshMT5Sim(SeshSimulatorBase):
    """
    Symmetrical simulator to SeshMT5Live.

    Exposes exactly the same custom methods as SeshMT5Live with simulated
    logic using the internal broker. The strategy does not need to know
    which one is active.

    Parameters - identical to SeshSimulatorBase plus those of SeshMT5Live:
        feed_type, initial_balance, commission, commission_type,
        position_mode, use_margin, margin_rate, slippage_points,
        use_spread, trade_on_close, execution_price_source.

    USAGE
    -----
        # Backtest
        sesh = SeshMT5Sim(initial_balance=10_000, commission=0.6)
        engine = BacktestEngine.by_ticks(MyStrategy(), data=(ticks,), sesh=sesh)

        # Live (same strategy code)
        sesh = SeshMT5Live(login=123456, password="pass", server="BrokerDemo")
        engine = LiveEngine.by_ticks(MyStrategy(), sesh=sesh)
    """

    # ── Custom MT5 methods - simulated logic ──────────────────────────────────

    def close_all(self, symbol: str) -> list:
        """Close all open positions for a symbol (simulated version)."""
        results = []
        for p in self.positions(symbol):
            ok = self.position_close(p.ticket)
            results.append({"ticket": p.ticket, "ok": ok})
        return results

    def get_last_price(self, symbol: str) -> float:
        """Return the current mid price for the symbol (simulated version)."""
        price = self._current_price(symbol)
        if price == 0.0:
            raise TradingError(
                f"No price for '{symbol}'. "
                "Make sure at least one tick/bar has been fed."
            )
        return price

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        """Calculate required margin using the symbol config (simulated version)."""
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
        """Calculate potential profit (simulated version)."""
        cfg = self._broker._get_symbol_config(symbol)
        is_buy = side.lower() == "buy"
        price_diff = (price_close - price_open) if is_buy else (price_open - price_close)
        return (price_diff / cfg.tick_step) * cfg.tick_value * volume

    def get_account_info(self) -> dict:
        """Return extended info for the simulated account."""
        a = self._broker.account_info()
        return {
            "balance":     a.balance,
            "equity":      a.equity,
            "margin":      a.margin,
            "margin_free": a.margin_free,
            "profit":      a.profit,
            # Extra MT5 fields - fixed values in simulation
            "login":    0,
            "leverage": 1,
            "currency": "USD",
            "company":  "Simulator",
            "server":   "Backtest",
        }

    def get_instrument_info(self, symbol: str) -> dict:
        """Return symbol info from the simulated broker config."""
        cfg = self._broker._get_symbol_config(symbol)
        return {
            "symbol":              symbol,
            "digits":              cfg.digits,
            "volume_min":          cfg.volume_min,
            "volume_max":          cfg.volume_max,
            "volume_step":         cfg.volume_step,
            "tick_step":           cfg.tick_step,
            "tick_value":          cfg.tick_value,
            "trade_contract_size": cfg.contract_size,
        }

    def __repr__(self) -> str:
        a = self._broker.account_info()
        return (
            f"SeshMT5Sim(feed_type={self._feed_type!r}, "
            f"balance={a.balance:.2f}, equity={a.equity:.2f}, "
            f"positions={len(self._broker.positions_get())})"
        )

    # dummy methods for backtest

    def disconnect(self) -> None:
        pass  # no-op in backtest

    def reconnect(self) -> None:
        pass
    
    def is_connected(self) -> bool:
        return True  # in backtest always "connected"

    def is_market_open(self, symbol: str) -> bool:
        return True

# ══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY ALIASES v4.0
# ══════════════════════════════════════════════════════════════════════════════

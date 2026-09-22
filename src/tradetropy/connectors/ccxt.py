"""
ccxt.py
=======
Load market data from public exchanges via the `ccxt` library,
returning `KlineData` / `TickData` objects from tradetropy directly.

`ccxt` is an OPTIONAL dependency. Install with:

    pip install "tradetropy[ccxt]"      # or:  pip install ccxt

USAGE
─────
    from tradetropy.connectors.ccxt import fetch_klines, fetch_ticks

    # 5m candles from Binance (public data, no API key):
    klines = fetch_klines("binance", "BTC/USDT", "5m", limit=500)

    # Resample to 1h using tradetropy's API:
    klines_1h = klines.resample("1h")

    # Ticks (trades) and conversion to candles:
    ticks  = fetch_ticks("binance", "BTC/USDT", limit=1000)
    candles = ticks.to_klines("1m")

NOTE
────
ccxt's `fetch_ohlcv` returns [ts, open, high, low, close, volume] (volume in
base currency). It does not include turnover, so the 7th column is filled with
NaN by default. Use `turnover_mode="approx"` to estimate it as close×volume.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from tradetropy.core.constants import parse_timeframe, N_OHLCV_TURNOVER_COLS, N_TICK_COLS, _TICK_COL
from tradetropy.core.data_types import TickData, KlineData
from tradetropy.core.broker import (
    AccountInfo,
    Deal,
    Order,
    OrderState,
    OrderType,
    Position,
    TradeResult,
)
from tradetropy.exceptions import ConnectionError, ConfigError, TradingError
from tradetropy.session.base import Sesh, SeshLiveBase, SeshSimulatorBase
from tradetropy.connectors._disclaimer import emit_live_disclaimer

_logger = logging.getLogger("tradetropy.connectors.ccxt")


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _import_ccxt():
    """
    Lazily import ccxt with clear error message if missing.

    Returns:
        ccxt module.

    Raises:
        ImportError: If ccxt is not installed.
    """
    try:
        import ccxt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'ccxt' library is not installed. "
            'Install with: pip install "tradetropy[ccxt]"  (or: pip install ccxt).'
        ) from exc
    return ccxt


def _resolve_exchange(exchange, config: dict | None = None):
    """
    Resolve exchange parameter to ccxt instance.

    Args:
        exchange: Exchange ID as string (e.g. 'binance', 'bybit', 'okx')
                  or ccxt instance or mock with compatible API.
        config: Configuration dict for exchange construction (optional).

    Returns:
        ccxt exchange instance ready for use.

    Raises:
        ConnectionError: If exchange ID is unknown.
    """
    if isinstance(exchange, str):
        ccxt = _import_ccxt()
        if not hasattr(ccxt, exchange):
            raise ConnectionError(
                f"Unknown ccxt exchange: {exchange!r}. "
                f"Check the id in ccxt.exchanges."
            )
        return getattr(ccxt, exchange)(config or {})
    # Already an instance (or mock): use directly.
    return exchange


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def fetch_klines(
    exchange,
    symbol: str,
    timeframe: str,
    *,
    since: int | None = None,
    limit: int | None = None,
    config: dict | None = None,
    turnover_mode: str = "nan",
    tick_size: float = 0.01,
    tick_value: float = 0.01,
    contract_size: float = 1.0,
    digits: int = 2,
    avg_spread: float = 0.0,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> KlineData:
    """
    Download OHLCV candles from ccxt exchange and return KlineData.

    Args:
        exchange: Exchange ID (str) or ccxt instance.
        symbol: Market pair in ccxt format (e.g. 'BTC/USDT').
        timeframe: ccxt timeframe (e.g. '1m', '5m', '1h', '1d'). Standard
                   recommended set: '1m', '15m', '1h', '4h', '1d', '1w',
                   '1mo' ('min'/'wk' accepted as aliases for 'm'/'w'; 'mo'
                   is a fixed 30-day month).
        since: Timestamp in ms from which to download (optional).
        limit: Maximum number of candles (optional).
        config: Config dict for exchange construction (optional).
        turnover_mode: 'nan' (default) leaves turnover as NaN;
                       'approx' estimates it as close x volume.
        tick_size ... volume_step: Symbol metadata propagated to KlineData.

    Returns:
        KlineData [N x 7] with interval_ms derived from timeframe.

    Raises:
        ConnectionError: If exchange ID is unknown or fetch fails.
        ConfigError: If turnover_mode is invalid.

    Example:
        klines = fetch_klines('binance', 'BTC/USDT', '5m', limit=500)
        klines_1h = klines.resample('1h')
    """
    interval_ms = parse_timeframe(timeframe)
    ex = _resolve_exchange(exchange, config)

    raw = ex.fetch_ohlcv(symbol, timeframe, since, limit)
    arr = np.asarray(raw, dtype=np.float64)

    if arr.size == 0:
        data = np.empty((0, N_OHLCV_TURNOVER_COLS), dtype=np.float64)
    else:
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ConnectionError(
                f"Unexpected OHLCV response from ccxt: shape {arr.shape}. "
                f"Expected ≥6 columns [ts,o,h,l,c,v]."
            )
        ohlcv = arr[:, :6]
        if turnover_mode == "approx":
            turnover = ohlcv[:, 4] * ohlcv[:, 5]  # close × volume (approximate)
        elif turnover_mode == "nan":
            turnover = np.full(len(ohlcv), np.nan, dtype=np.float64)
        else:
            raise ConfigError(
                f"Invalid turnover_mode: {turnover_mode!r}. Use 'nan' or 'approx'."
            )
        data = np.column_stack([ohlcv, turnover]).astype(np.float64)

    return KlineData(
        symbol=symbol,
        data=data,
        timeframe=interval_ms,
        tick_size=tick_size,
        tick_value=tick_value,
        contract_size=contract_size,
        digits=digits,
        avg_spread=avg_spread,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
    )


#: Tick `flags` column encoding based on trade side.
_SIDE_FLAG = {"buy": 1.0, "sell": -1.0}


def fetch_ticks(
    exchange,
    symbol: str,
    *,
    since: int | None = None,
    limit: int | None = None,
    config: dict | None = None,
    tick_size: float = 0.01,
    tick_value: float = 0.01,
    contract_size: float = 1.0,
    digits: int = 2,
    avg_spread: float = 0.0,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> TickData:
    """
    Download public trades from ccxt exchange and return TickData.

    Maps each trade to [ts, bid, ask, volume, flags, volume_real, price]:
        - price: trade price
        - volume: trade amount
        - volume_real: trade amount
        - bid/ask: price (no order book in fetch_trades)
        - flags: +1 buy / -1 sell / 0 unknown

    Args:
        exchange: Exchange ID (str) or ccxt instance.
        symbol: Market pair in ccxt format (e.g. 'BTC/USDT').
        since: Timestamp in ms from which to download (optional).
        limit: Maximum number of trades (optional).
        config: Config dict for exchange construction (optional).
        tick_size ... volume_step: Symbol metadata.

    Returns:
        TickData [N x 7]. Convert to candles with tick_data.to_klines('1m').

    Raises:
        ConnectionError: If exchange ID is unknown or fetch fails.

    Example:
        ticks = fetch_ticks('binance', 'BTC/USDT', limit=1000)
        candles = ticks.to_klines('1m')
    """
    ex = _resolve_exchange(exchange, config)
    trades = ex.fetch_trades(symbol, since, limit)

    C = _TICK_COL
    if not trades:
        data = np.empty((0, N_TICK_COLS), dtype=np.float64)
        return TickData(
            symbol=symbol, data=data,
            tick_size=tick_size, tick_value=tick_value,
            contract_size=contract_size, digits=digits, avg_spread=avg_spread,
            volume_min=volume_min, volume_max=volume_max, volume_step=volume_step,
        )

    n = len(trades)
    data = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i, tr in enumerate(trades):
        price = float(tr.get("price") or 0.0)
        amount = float(tr.get("amount") or 0.0)
        ts = tr.get("timestamp")
        data[i, C["ts"]] = float(ts) if ts is not None else 0.0
        data[i, C["bid"]] = price
        data[i, C["ask"]] = price
        data[i, C["price"]] = price
        data[i, C["volume"]] = amount
        data[i, C["volume_real"]] = amount
        data[i, C["flags"]] = _SIDE_FLAG.get(tr.get("side"), 0.0)

    return TickData(
        symbol=symbol,
        data=data,
        tick_size=tick_size,
        tick_value=tick_value,
        contract_size=contract_size,
        digits=digits,
        avg_spread=avg_spread,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
    )



# ══════════════════════════════════════════════════════════════════════════════
# CCXT TRADING SESSIONS (Live + Sim) — mirror of MT5 / Bybit
# ══════════════════════════════════════════════════════════════════════════════
#
# Built on ccxt's UNIFIED API: a single interface for 100+ exchanges.
# Not all exchanges support every method; we check `exchange.has[...]`
# and raise a clear error when something is not available.

# ms → ccxt timeframe (string). ccxt uses '1m','1h','1d','1w','1M'…
_MS_TO_CCXT_TF = {
    60_000:        "1m",
    180_000:       "3m",
    300_000:       "5m",
    900_000:       "15m",
    1_800_000:     "30m",
    3_600_000:     "1h",
    7_200_000:     "2h",
    14_400_000:    "4h",
    21_600_000:    "6h",
    43_200_000:    "12h",
    86_400_000:    "1d",
    604_800_000:   "1w",
    # ccxt's monthly "1M" corresponds to the public parse_timeframe('1mo')
    # unit (fixed 30-day duration, see core/constants.py::_TF_UNIT_MS).
    2_592_000_000: "1M",
}

_ticket_counter_ccxt = [0]


def _next_ticket_ccxt() -> int:
    _ticket_counter_ccxt[0] += 1
    return _ticket_counter_ccxt[0]


def _ccxt_side_to_order_type(side: str) -> OrderType:
    """'long'/'buy' → BUY, 'short'/'sell' → SELL."""
    s = str(side).lower()
    return OrderType.ORDER_TYPE_BUY if s in ("long", "buy") else OrderType.ORDER_TYPE_SELL


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class SeshCCXTLive(SeshLiveBase):
    """
    Generic live session over ccxt's unified API (100+ exchanges).

    Same interface as SeshMT5Live / SeshBybitLive: the strategy does not change
    when moving from backtest to production.

    Requires: pip install "tradetropy[ccxt]"  (or pip install ccxt)

    ⚠️  Always test on testnet/sandbox/demo before trading real money. Exchange
    APIs may change and break this connector. Use at your own risk — see
    `tradetropy.LIVE_DISCLAIMER`.

    Parameters
    ----------
    exchange   : ccxt id (str, e.g. "binance", "okx", "bybit") or an already
                 built ccxt instance (or a compatible mock for tests).
    config     : Exchange configuration dict (apiKey, secret, options…).
    base_coin  : Settlement currency for the balance (default "USDT").
    sandbox    : If True, enable the exchange's testnet/sandbox mode. On Bybit
                 this is the SEPARATE testnet platform (testnet.bybit.com).
    demo       : If True, enable the venue's demo-trading mode (Bybit's demo
                 account on the main platform, via ccxt.enable_demo_trading).
                 Distinct from `sandbox`/testnet; the two are mutually
                 exclusive. Ignored if the exchange does not support it.
    market_type: Optional, sets defaultType ("spot"/"swap"/"future") via options.
    display_tz : Timezone for formatting `sesh.time` / `sesh.str_time`.
                 ccxt data is already in UTC, so this only affects display.
                 Default UTC.
    """

    def __init__(
        self,
        exchange,
        config: Optional[dict] = None,
        base_coin: str = "USDT",
        sandbox: bool = False,
        demo: bool = False,
        market_type: Optional[str] = None,
        display_tz=None,
    ):
        super().__init__()
        emit_live_disclaimer()

        # ccxt epochs are already real UTC → data_tz UTC (default).
        from tradetropy.session.base import _coerce_tz
        self._display_tz = _coerce_tz(display_tz)

        cfg = dict(config or {})
        if market_type:
            opts = dict(cfg.get("options") or {})
            opts["defaultType"] = market_type
            cfg["options"] = opts

        self._ex = _resolve_exchange(exchange, cfg)
        self._base_coin = base_coin
        self._market_type = market_type
        self._ticket_by_symbol: dict[str, int] = {}

        # Stored so create_feed() can build a matching ccxt.pro instance for
        # real-time streaming (mirrors this session's exchange/config/sandbox).
        self._feed_config = cfg
        self._sandbox = bool(sandbox)
        self._demo = bool(demo)

        if sandbox and demo:
            _logger.warning(
                "sandbox and demo are mutually exclusive; both were requested. "
                "Applying testnet (sandbox) and skipping demo-trading."
            )
            self._demo = False

        if sandbox:
            try:
                self._ex.set_sandbox_mode(True)
            except Exception as exc:  # pragma: no cover - depends on exchange
                _logger.warning("Could not enable sandbox_mode: %s", exc)
        elif self._demo:
            enable = getattr(self._ex, "enable_demo_trading", None)
            if enable is None:
                _logger.warning(
                    "Exchange %s does not support demo-trading "
                    "(enable_demo_trading); ignoring demo=True.",
                    getattr(self._ex, "id", "?"),
                )
                self._demo = False
            else:
                try:
                    enable(True)
                except Exception as exc:  # pragma: no cover - depends on exchange
                    _logger.warning("Could not enable demo-trading: %s", exc)
                    self._demo = False

        # load_markets is useful for get_instrument_info; best-effort.
        try:
            if getattr(self._ex, "has", {}).get("loadMarkets", True):
                self._ex.load_markets()
        except Exception as exc:  # pragma: no cover
            _logger.warning("load_markets() failed: %s", exc)

        _logger.info(
            "SeshCCXTLive connected — exchange=%s base_coin=%s sandbox=%s "
            "demo=%s type=%s",
            getattr(self._ex, "id", type(self._ex).__name__),
            base_coin, sandbox, self._demo, market_type,
        )
        self.sync()

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _require(self, capability: str) -> None:
        """Check exchange.has[capability]; raise TradingError if missing."""
        has = getattr(self._ex, "has", {}) or {}
        if not has.get(capability, False):
            raise TradingError(
                f"Exchange {getattr(self._ex, 'id', '?')!r} does not support "
                f"'{capability}' according to ccxt (exchange.has)."
            )

    def _ticket_for(self, symbol: str) -> int:
        t = self._ticket_by_symbol.get(symbol)
        if t is None:
            t = _next_ticket_ccxt()
            self._ticket_by_symbol[symbol] = t
        return t

    # ── Connection ──────────────────────────────────────────────────────────────

    def disconnect(self) -> None:
        pass

    def reconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return self._ex is not None

    # ── Streaming (CCXT Pro WebSocket) ────────────────────────────────────────

    @property
    def supports_streaming(self) -> bool:
        """SeshCCXTLive streams via CCXT Pro WebSockets."""
        return True

    @property
    def supports_user_stream(self) -> bool:
        """
        True when the session is authenticated (an API key is configured).

        With a key the engine subscribes the private ORDER / FILL channels and
        keeps orders / deals / net positions current from the user-data stream,
        so orders() / order_history() / deals() / positions() read local state
        instead of polling REST. The key may come from the ccxt instance
        (``self._ex.apiKey``) or from the config passed to this session.
        """
        if getattr(self._ex, "apiKey", None):
            return True
        return bool((self._feed_config or {}).get("apiKey"))

    def create_feed(self, *, timeframe_ms: int = 60_000, book_limit: int = 20):
        """
        Build a CCXTProFeed mirroring this session's exchange/config/sandbox.

        A fresh ccxt.pro instance is created from the exchange id (public
        market-data streaming), independent of this session's REST instance.

        Args:
            timeframe_ms (int): Candle interval for the KLINE channel.
            book_limit (int): Order-book depth for the book channels.

        Returns:
            CCXTProFeed: A feed ready to be driven by a FeedRunner.
        """
        from tradetropy.streaming.ccxt_pro import CCXTProFeed

        exchange_id = getattr(self._ex, "id", None)
        if exchange_id is None:
            raise TradingError(
                "Cannot create a streaming feed: the exchange has no 'id'."
            )
        return CCXTProFeed(
            exchange_id,
            config=self._feed_config,
            timeframe_ms=timeframe_ms,
            book_limit=book_limit,
            sandbox=self._sandbox,
            demo=self._demo,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self) -> str:
        return (
            f"SeshCCXTLive(exchange={getattr(self._ex, 'id', '?')!r}, "
            f"base_coin={self._base_coin!r}, positions={len(self._cache_positions)})"
        )

    # ── 4 required ─────────────────────────────────────────────────────────────

    def buy(
        self, symbol: str, volume: float, price: Optional[float] = None,
        sl: float = 0.0, tp: float = 0.0, comment: str = "", magic: int = 0,
    ) -> TradeResult:
        return self._create_order(symbol, "buy", volume, price, sl, tp, comment)

    def sell(
        self, symbol: str, volume: float, price: Optional[float] = None,
        sl: float = 0.0, tp: float = 0.0, comment: str = "", magic: int = 0,
    ) -> TradeResult:
        return self._create_order(symbol, "sell", volume, price, sl, tp, comment)

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        if self._user_stream_active:
            return self._streamed_positions(symbol)
        self._require("fetchPositions")
        raw = self._ex.fetch_positions([symbol] if symbol else None)
        out: List[Position] = []
        self._cache_positions = {}
        for p in raw or []:
            contracts = _f(p.get("contracts") or p.get("contractSize"))
            if contracts <= 0:
                continue
            sym = str(p.get("symbol", ""))
            ticket = self._ticket_for(sym)
            side = p.get("side", "long")
            self._cache_positions[ticket] = {
                "symbol": sym, "side": side, "contracts": contracts,
            }
            out.append(Position(
                ticket=ticket,
                symbol=sym,
                type=_ccxt_side_to_order_type(side),
                volume=contracts,
                price_open=_f(p.get("entryPrice")),
                time=_now_ccxt(p.get("timestamp")),
                profit=_f(p.get("unrealizedPnl")),
                magic=0,
                comment="",
            ))
        return out

    def account_info(self) -> AccountInfo:
        self._require("fetchBalance")
        bal = self._ex.fetch_balance()
        code = self._base_coin
        total = (bal.get("total") or {}).get(code)
        free = (bal.get("free") or {}).get(code)
        used = (bal.get("used") or {}).get(code)
        balance = _f(total)
        return AccountInfo(
            balance=balance,
            equity=balance,
            margin=_f(used),
            margin_free=_f(free),
            profit=0.0,
        )

    # ── Optional ────────────────────────────────────────────────────────────────

    def position_close(self, ticket: int, volume: Optional[float] = None) -> bool:
        pos = self._cache_positions.get(ticket)
        if pos is None:
            self.positions()
            pos = self._cache_positions.get(ticket)
        if pos is None:
            return False
        close_side = "sell" if _ccxt_side_to_order_type(pos["side"]) == OrderType.ORDER_TYPE_BUY else "buy"
        qty = float(volume) if volume else float(pos["contracts"])
        try:
            self._ex.create_order(
                pos["symbol"], "market", close_side, qty, None, {"reduceOnly": True}
            )
        except Exception as exc:
            _logger.error("position_close failed: %s", exc)
            return False
        if qty >= float(pos["contracts"]):
            self._cache_positions.pop(ticket, None)
        else:
            pos["contracts"] -= qty
        return True

    def order_delete(self, ticket: int) -> bool:
        info = self._cache_orders.get(ticket)
        if not info:
            return False
        try:
            self._ex.cancel_order(info["id"], info["symbol"])
            return True
        except Exception as exc:
            _logger.error("order_delete failed: %s", exc)
            return False

    def orders(self, symbol: Optional[str] = None) -> List[Order]:
        if self._user_stream_active:
            from tradetropy.session.base import _dict_to_order
            items = list(self._cache_orders.values())
            if symbol:
                items = [o for o in items if o.get("symbol") == symbol]
            return [_dict_to_order(d) for d in items]
        self._require("fetchOpenOrders")
        raw = self._ex.fetch_open_orders(symbol)
        out = []
        for o in raw or []:
            order = _ccxt_order_to_order(o)
            self._cache_orders[order.ticket] = {"id": o.get("id"), "symbol": o.get("symbol")}
            out.append(order)
        return out

    def order_history(self, symbol: Optional[str] = None) -> List[Order]:
        if self._user_stream_active:
            return super().order_history(symbol)
        self._require("fetchClosedOrders")
        raw = self._ex.fetch_closed_orders(symbol)
        return [_ccxt_order_to_order(o) for o in (raw or [])]

    def deals(self, symbol: Optional[str] = None) -> List[Deal]:
        if self._user_stream_active:
            return super().deals(symbol)
        self._require("fetchMyTrades")
        raw = self._ex.fetch_my_trades(symbol)
        out = []
        for tr in raw or []:
            out.append(Deal(
                ticket=_next_ticket_ccxt(),
                position_id=0,
                symbol=str(tr.get("symbol", "")),
                type=_ccxt_side_to_order_type(tr.get("side", "buy")),
                volume=_f(tr.get("amount")),
                price=_f(tr.get("price")),
                time=_now_ccxt(tr.get("timestamp")),
                commission=_f((tr.get("fee") or {}).get("cost")),
                profit=0.0,
            ))
        return out

    def sync(self) -> None:
        for attempt in (1, 2):
            try:
                if getattr(self._ex, "has", {}).get("fetchPositions", False):
                    self.positions()
                break
            except Exception:
                _logger.error(
                    "SeshCCXTLive.sync: failed to fetch positions (attempt %d/2).",
                    attempt, exc_info=True,
                )
        try:
            if getattr(self._ex, "has", {}).get("fetchBalance", False):
                acc = self.account_info()
                self._balance = acc.balance
                self._equity = acc.equity
        except Exception:
            _logger.error(
                "SeshCCXTLive.sync: failed to fetch balance.", exc_info=True
            )

    # ── Data primitives ─────────────────────────────────────────────────────────

    def _fetch_klines_history(
        self, symbol: str, interval_ms: int, limit: int = 200
    ) -> np.ndarray:
        tf = _MS_TO_CCXT_TF.get(interval_ms)
        if tf is None:
            raise ConfigError(
                f"interval_ms={interval_ms} has no equivalent ccxt timeframe. "
                f"Supported: {sorted(_MS_TO_CCXT_TF.keys())}"
            )
        raw = self._ex.fetch_ohlcv(symbol, tf, None, limit)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 6), dtype=np.float64)
        return arr[:, :6]

    def _fetch_last_kline(self, symbol: str, interval_ms: int) -> np.ndarray:
        klines = self._fetch_klines_history(symbol, interval_ms, limit=2)
        if len(klines) < 1:
            raise TradingError(
                f"Could not fetch candles for '{symbol}' "
                f"(interval_ms={interval_ms})."
            )
        return klines[-1]

    def _fetch_last_tick(self, symbol: str) -> np.ndarray:
        t = self._ex.fetch_ticker(symbol)
        out = np.empty(N_TICK_COLS, dtype=np.float64)
        bid = _f(t.get("bid"))
        ask = _f(t.get("ask"))
        last = _f(t.get("last"))
        ts = t.get("timestamp")
        out[_TICK_COL["ts"]]          = float(ts) if ts is not None else float(int(time.time() * 1000))
        out[_TICK_COL["bid"]]         = bid
        out[_TICK_COL["ask"]]         = ask
        out[_TICK_COL["volume"]]      = _f(t.get("baseVolume"))
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
        trades = self._ex.fetch_trades(symbol, None, limit)
        C = _TICK_COL
        if not trades:
            return np.empty((0, N_TICK_COLS), dtype=np.float64)
        out = np.zeros((len(trades), N_TICK_COLS), dtype=np.float64)
        for i, tr in enumerate(trades):
            price = _f(tr.get("price"))
            amount = _f(tr.get("amount"))
            ts = tr.get("timestamp")
            out[i, C["ts"]]          = float(ts) if ts is not None else 0.0
            out[i, C["bid"]]         = price
            out[i, C["ask"]]         = price
            out[i, C["price"]]       = price
            out[i, C["volume"]]      = amount
            out[i, C["volume_real"]] = amount
            out[i, C["flags"]]       = _SIDE_FLAG.get(tr.get("side"), 0.0)
        return out

    def _fetch_orderbook(self, symbol: str, depth: int = 20) -> np.ndarray:
        """
        Fetch a single L2 order-book snapshot as a flat book row.

        Wraps ccxt's fetch_order_book into the flat layout of
        core.data_types.book_flat_columns(depth):
        [ts, kind, bid_px_0..K-1, bid_sz_0..K-1, ask_px_0..K-1, ask_sz_0..K-1]
        with kind=0 (snapshot). Missing levels are padded with zeros.

        Args:
            symbol (str): Trading symbol.
            depth (int): Number of book levels K per side.

        Returns:
            np.ndarray: A single flat book row [1 x (2 + 4*depth)].

        Raises:
            TradingError: If the exchange does not support fetchOrderBook.
        """
        self._require("fetchOrderBook")
        from tradetropy.core.data_types import book_row_width

        ob = self._ex.fetch_order_book(symbol, depth)
        k = int(depth)
        width = book_row_width(k)
        row = np.zeros(width, dtype=np.float64)
        ts = ob.get("timestamp")
        row[0] = float(ts) if ts is not None else float(int(time.time() * 1000))
        row[1] = 0.0  # kind = snapshot
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        for i in range(min(k, len(bids))):
            row[2 + i]         = _f(bids[i][0])
            row[2 + k + i]     = _f(bids[i][1])
        for i in range(min(k, len(asks))):
            row[2 + 2 * k + i] = _f(asks[i][0])
            row[2 + 3 * k + i] = _f(asks[i][1])
        return row.reshape(1, width)

    # ── Custom ───────────────────────────────────────────────────────────────────

    def close_all(self, symbol: str) -> list:
        results = []
        for p in self.positions(symbol):
            ok = self.position_close(p.ticket)
            results.append({"ticket": p.ticket, "ok": ok})
        return results

    def get_last_price(self, symbol: str) -> float:
        t = self._ex.fetch_ticker(symbol)
        last = _f(t.get("last"))
        if last > 0:
            return last
        bid = _f(t.get("bid"))
        ask = _f(t.get("ask"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        raise TradingError(f"No valid price for '{symbol}'.")

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        price = self.get_last_price(symbol)
        return volume * price

    def calculate_profit(
        self, symbol: str, volume: float, price_open: float,
        price_close: float, side: str = "Buy",
    ) -> float:
        is_buy = str(side).lower() in ("buy", "long")
        diff = (price_close - price_open) if is_buy else (price_open - price_close)
        return diff * volume

    def get_account_info(self) -> dict:
        a = self.account_info()
        return {
            "login":      None,
            "tradeMode":  self._market_type or "default",
            "leverage":   None,
            "balance":    a.balance,
            "equity":     a.equity,
            "margin":     a.margin,
            "marginFree": a.margin_free,
            "profit":     a.profit,
            "currency":   self._base_coin,
            "company":    getattr(self._ex, "id", "ccxt"),
            "server":     getattr(self._ex, "id", "ccxt"),
            "marketType": self._market_type or "default",
        }

    def get_instrument_info(self, symbol: str) -> dict:
        try:
            m = self._ex.market(symbol)
        except Exception as exc:
            raise TradingError(f"No info for '{symbol}': {exc}") from exc
        precision = m.get("precision") or {}
        limits = m.get("limits") or {}
        amount_lim = limits.get("amount") or {}
        price_prec = precision.get("price")
        # ccxt precision can be number of decimals or tick size depending on exchange
        if isinstance(price_prec, float) and price_prec < 1:
            tick_size = price_prec
            digits = max(0, len(str(price_prec).split(".")[-1]))
        else:
            digits = int(price_prec) if price_prec is not None else 8
            tick_size = 10 ** (-digits) if digits else 1.0
        return {
            "symbol":              m.get("symbol", symbol),
            "digits":              digits,
            "tick_step":           tick_size,
            "tick_value":          1.0,
            "trade_contract_size": _f(m.get("contractSize"), 1.0),
            "volume_min":          _f(amount_lim.get("min")),
            "volume_max":          _f(amount_lim.get("max")),
            "volume_step":         _f((precision.get("amount") if isinstance(precision.get("amount"), float) else None)),
            "currency_base":       m.get("base"),
        }

    def is_market_open(self, symbol: str) -> bool:
        try:
            return self.get_last_price(symbol) > 0
        except Exception:
            return False

    def set_leverage(self, symbol: str, leverage: int = 1) -> bool:
        if not getattr(self._ex, "has", {}).get("setLeverage", False):
            _logger.warning(
                "Exchange %s does not support setLeverage.", getattr(self._ex, "id", "?")
            )
            return False
        try:
            self._ex.set_leverage(int(leverage), symbol)
            return True
        except Exception as exc:
            raise TradingError(f"Could not set leverage: {exc}") from exc

    # ── Internal logic ─────────────────────────────────────────────────────────

    def _create_order(
        self, symbol: str, side: str, volume: float,
        price: Optional[float], sl: float, tp: float, comment: str,
    ) -> TradeResult:
        self._require("createOrder")
        is_limit = price is not None
        order_type = "limit" if is_limit else "market"
        params: dict = {}
        if sl:
            params["stopLossPrice"] = sl
        if tp:
            params["takeProfitPrice"] = tp
        if comment:
            params["clientOrderId"] = comment
        try:
            o = self._ex.create_order(symbol, order_type, side, volume, price, params)
        except Exception as exc:
            return TradeResult(
                retcode=TradeResult.TRADE_RETCODE_INVALID,
                comment=f"create_order failed: {exc}",
            )
        oid_raw = str(o.get("id", "0"))
        order_id = int(oid_raw) if oid_raw.isdigit() else (abs(hash(oid_raw)) % 1_000_000_000)
        status = str(o.get("status", "")).lower()
        # 'closed'/'filled' → executed; 'open' → placed (pending)
        if status in ("closed", "filled"):
            retcode = TradeResult.TRADE_RETCODE_DONE
        elif is_limit:
            retcode = TradeResult.TRADE_RETCODE_PLACED
        else:
            retcode = TradeResult.TRADE_RETCODE_DONE
        return TradeResult(
            retcode=retcode,
            order=order_id,
            volume=_f(o.get("amount"), float(volume)),
            price=_f(o.get("price"), float(price) if is_limit else 0.0),
            comment=str(o.get("clientOrderId") or comment),
        )


def _now_ccxt(ts_ms) -> datetime:
    try:
        if ts_ms:
            return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        pass
    return datetime.now(tz=timezone.utc)


def _ccxt_order_to_order(o: dict) -> Order:
    side = o.get("side", "buy")
    is_limit = str(o.get("type", "")).lower() == "limit"
    ot = (
        (OrderType.ORDER_TYPE_BUY_LIMIT if is_limit else OrderType.ORDER_TYPE_BUY)
        if str(side).lower() == "buy"
        else (OrderType.ORDER_TYPE_SELL_LIMIT if is_limit else OrderType.ORDER_TYPE_SELL)
    )
    status = str(o.get("status", "")).lower()
    state = {
        "open":     OrderState.ORDER_STATE_PLACED,
        "closed":   OrderState.ORDER_STATE_FILLED,
        "filled":   OrderState.ORDER_STATE_FILLED,
        "canceled": OrderState.ORDER_STATE_CANCELED,
        "cancelled": OrderState.ORDER_STATE_CANCELED,
        "rejected": OrderState.ORDER_STATE_REJECTED,
        "expired":  OrderState.ORDER_STATE_EXPIRED,
    }.get(status, OrderState.ORDER_STATE_PLACED)
    oid_raw = str(o.get("id", "0"))
    ticket = int(oid_raw) if oid_raw.isdigit() else (abs(hash(oid_raw)) % 1_000_000_000)
    return Order(
        ticket=ticket,
        symbol=str(o.get("symbol", "")),
        type=ot,
        volume=_f(o.get("amount")),
        price=_f(o.get("price")),
        time=_now_ccxt(o.get("timestamp")),
        sl=_f((o.get("stopLossPrice") or 0)),
        tp=_f((o.get("takeProfitPrice") or 0)),
        magic=0,
        comment=str(o.get("clientOrderId") or ""),
        state=state,
    )


class SeshCCXTSim(SeshSimulatorBase):
    """
    Symmetric simulator to SeshCCXTLive.

    Exposes the same custom methods with simulated logic on the internal broker.
    The strategy does not need to know which one is active.

    Identical parameters to SeshSimulatorBase (feed_type, initial_balance,
    commission, …).
    """

    def close_all(self, symbol: str) -> list:
        results = []
        for p in self.positions(symbol):
            ok = self.position_close(p.ticket)
            results.append({"ticket": p.ticket, "ok": ok})
        return results

    def get_last_price(self, symbol: str) -> float:
        price = self._current_price(symbol)
        if price == 0.0:
            raise TradingError(
                f"No price for '{symbol}'. "
                "Make sure at least one tick/bar has been fed."
            )
        return price

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        cfg = self._broker._get_symbol_config(symbol)
        price = self._current_price(symbol)
        if not self._broker.use_margin:
            return 0.0
        return volume * cfg.contract_size * price * self._broker.margin_rate

    def calculate_profit(
        self, symbol: str, volume: float, price_open: float,
        price_close: float, side: str = "Buy",
    ) -> float:
        cfg = self._broker._get_symbol_config(symbol)
        is_buy = str(side).lower() in ("buy", "long")
        return cfg.calculate_profit(volume, price_open, price_close, is_buy)

    def get_account_info(self) -> dict:
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
            "marketType": "default",
        }

    def get_instrument_info(self, symbol: str) -> dict:
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

    def set_leverage(self, symbol: str, leverage: int = 1) -> bool:
        return True  # no-op in backtest

    def is_market_open(self, symbol: str) -> bool:
        return True

    def __repr__(self) -> str:
        if self._broker is None:
            return (
                f"SeshCCXTSim(feed_type=<auto>, "
                f"pending_symbols={len(self._pending_symbols)})"
            )
        a = self._broker.account_info()
        return (
            f"SeshCCXTSim(feed_type={self._feed_type!r}, "
            f"balance={a.balance:.2f}, equity={a.equity:.2f}, "
            f"positions={len(self._broker.get_positions())})"
        )

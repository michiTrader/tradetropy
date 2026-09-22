"""
binance.py
==========
Public data loader from Binance with NO external dependencies (uses only the
stdlib `urllib`). No API key required for market data.

Returns `KlineData` / `TickData` objects from tradetropy directly.

USAGE
-----
    from tradetropy.connectors.binance import fetch_klines, fetch_ticks

    # 5m candles (up to 1000 per request):
    klines = fetch_klines("BTCUSDT", "5m", limit=500)
    klines_1h = klines.resample("1h")        # resample to 1h

    # Aggregated trades -> TickData -> candles:
    ticks = fetch_ticks("BTCUSDT", limit=1000)
    candles = ticks.to_klines("1m")

ENDPOINTS (Binance Spot REST)
-----------------------------
    GET /api/v3/klines       -- OHLCV candles + quoteAssetVolume (real turnover)
    GET /api/v3/aggTrades    -- aggregated trades (price, quantity, side)

NOTE
----
Unlike ccxt, Binance klines include `quoteAssetVolume`, which maps directly
to the `turnover` column (REAL turnover, not estimated).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import numpy as np

from tradetropy.core.constants import (
    parse_timeframe,
    to_binance_interval,
    N_OHLCV_TURNOVER_COLS,
    N_TICK_COLS,
    _TICK_COL,
)
from tradetropy.core.data_types import TickData, KlineData
from tradetropy.exceptions import ConnectionError as TrConnectionError


_BASE_URL = "https://api.binance.com"
_USER_AGENT = "tradetropy/0.1 (+https://github.com/michiTrader)"


# ══════════════════════════════════════════════════════════════════════════════
# HTTP (stdlib) -- isolated for mocking in tests
# ══════════════════════════════════════════════════════════════════════════════


def _http_get_json(url: str, timeout: float = 10.0):
    """
    GET endpoint and return parsed JSON using only stdlib urllib.

    Args:
        url: Endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        TrConnectionError: If request fails.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            payload = resp.read()
    except Exception as exc:  # pragma: no cover - depends on network
        raise TrConnectionError(f"Failed to query Binance: {url} -> {exc}") from exc
    return json.loads(payload)


def _build_url(path: str, params: dict, base_url: str) -> str:
    """
    Build query string from params and combine with base URL.

    Args:
        path: Endpoint path.
        params: Query parameters dict.
        base_url: Base URL.

    Returns:
        Complete URL with query string.
    """
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{base_url.rstrip('/')}{path}?{query}"


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol format: 'BTC/USDT' - > 'BTCUSDT'.

    Binance format does not use separators.

    Args:
        symbol: Trading pair with optional separators.

    Returns:
        Normalized symbol (uppercase, no separators).
    """
    return symbol.replace("/", "").replace("-", "").upper()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def fetch_klines(
    symbol: str,
    interval: str,
    *,
    limit: int = 500,
    start_ms: int | None = None,
    end_ms: int | None = None,
    base_url: str = _BASE_URL,
    timeout: float = 10.0,
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
    Download OHLCV candles from Binance Spot and return KlineData.

    Args:
        symbol: Trading pair (e.g. 'BTCUSDT' or 'BTC/USDT').
        interval: Timeframe as string ('5m','1h',...) or int milliseconds.
                  Must be supported by Binance (see to_binance_interval).
        limit: Number of candles (max 1000 per request).
        start_ms: Start timestamp in milliseconds (optional).
        end_ms: End timestamp in milliseconds (optional).
        base_url: Base endpoint URL (allows mock/testnet).
        timeout: HTTP timeout in seconds.
        tick_size ... volume_step: Symbol metadata propagated to KlineData.

    Returns:
        KlineData [N x 7] with turnover column filled from quoteAssetVolume.

    Raises:
        TrConnectionError: If request fails.

    Example:
        klines = fetch_klines('BTCUSDT', '5m', limit=500)
        klines_1h = klines.resample('1h')
    """
    interval_ms = parse_timeframe(interval)
    binance_interval = to_binance_interval(interval)
    sym = _normalize_symbol(symbol)

    url = _build_url(
        "/api/v3/klines",
        {
            "symbol": sym,
            "interval": binance_interval,
            "limit": limit,
            "startTime": start_ms,
            "endTime": end_ms,
        },
        base_url,
    )
    raw = _http_get_json(url, timeout)

    if not raw:
        data = np.empty((0, N_OHLCV_TURNOVER_COLS), dtype=np.float64)
    else:
        # Each candle: [openTime, open, high, low, close, volume, closeTime,
        #               quoteAssetVolume, numTrades, takerBuyBase, takerBuyQuote, ignore]
        arr = np.asarray(raw, dtype=np.float64)
        data = np.column_stack([
            arr[:, 0],   # ts (openTime)
            arr[:, 1],   # open
            arr[:, 2],   # high
            arr[:, 3],   # low
            arr[:, 4],   # close
            arr[:, 5],   # volume (base)
            arr[:, 7],   # turnover (quoteAssetVolume)
        ]).astype(np.float64)

    return KlineData(
        symbol=sym,
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


def fetch_ticks(
    symbol: str,
    *,
    limit: int = 500,
    from_id: int | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    base_url: str = _BASE_URL,
    timeout: float = 10.0,
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
    Download aggregated trades (aggTrades) from Binance and return TickData.

    Each aggTrade is mapped to [ts, bid, ask, volume, flags, volume_real, price]:
        - price: trade price
        - volume: trade quantity
        - volume_real: trade quantity
        - bid/ask: price (aggTrades don't include order book)
        - flags: +1 if buyer is not maker (buy aggressor), -1 if buyer is
                 maker (sell aggressor)

    Args:
        symbol: Trading pair (e.g. 'BTCUSDT' or 'BTC/USDT').
        limit: Number of trades (max per Binance request).
        from_id: Trade ID for pagination (optional).
        start_ms: Start timestamp in milliseconds (optional).
        end_ms: End timestamp in milliseconds (optional).
        base_url: Base endpoint URL.
        timeout: HTTP timeout in seconds.
        tick_size ... volume_step: Symbol metadata.

    Returns:
        TickData [N x 7].

    Raises:
        TrConnectionError: If request fails.

    Example:
        ticks = fetch_ticks('BTCUSDT', limit=1000)
        velas = ticks.to_klines('1m')
    """
    sym = _normalize_symbol(symbol)
    url = _build_url(
        "/api/v3/aggTrades",
        {
            "symbol": sym,
            "limit": limit,
            "fromId": from_id,
            "startTime": start_ms,
            "endTime": end_ms,
        },
        base_url,
    )
    raw = _http_get_json(url, timeout)

    C = _TICK_COL
    if not raw:
        data = np.empty((0, N_TICK_COLS), dtype=np.float64)
    else:
        n = len(raw)
        data = np.zeros((n, N_TICK_COLS), dtype=np.float64)
        for i, tr in enumerate(raw):
            price = float(tr["p"])
            qty = float(tr["q"])
            data[i, C["ts"]] = float(tr["T"])
            data[i, C["bid"]] = price
            data[i, C["ask"]] = price
            data[i, C["price"]] = price
            data[i, C["volume"]] = qty
            data[i, C["volume_real"]] = qty
            # m=True -> buyer is maker -> seller is aggressor -> -1
            data[i, C["flags"]] = -1.0 if tr.get("m") else 1.0

    return TickData(
        symbol=sym,
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

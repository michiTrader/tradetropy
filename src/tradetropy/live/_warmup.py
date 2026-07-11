"""
Automatic warmup limit calculation and historical data fetching.
All functions receive self (LiveEngine) as the first argument
and are assigned as methods at the end of engine.py.
"""

from __future__ import annotations

import warnings

import numpy as np

from tradetropy.data.data import TickProxy, OhlcProxy
from tradetropy.exceptions import DataError


def _compute_required_ticks(self) -> int:
    """
    Calculate how many historical ticks to request for warming tick rings.

    Respects strategy.warmup if set. Otherwise falls back to max(min_periods)
    from all tick-based indicators.

    Unit: ticks (consistent with by_ticks warmup semantics).

    Returns:
        int: Number of ticks to fetch (minimum 50).
    """
    MIN_TICKS = 50

    warmup = getattr(self.strategy, "warmup", None)
    if warmup is not None:
        return max(int(warmup), MIN_TICKS)

    n = 0
    for defn in self.strategy._indicator_defs:
        if isinstance(defn["source"].proxy, TickProxy):
            n = max(n, getattr(defn["indicator"], "min_periods", 1))

    if n == 0 and (self.strategy._tick_proxies or self.strategy._fp_proxies):
        n = MIN_TICKS

    return max(n, MIN_TICKS)


def _compute_required_klines(self, ohlc_proxy: "OhlcProxy") -> int:
    """
    Calculate how many historical klines to request for a specific OhlcProxy.

    In by_klines mode: respects strategy.warmup (if set) or falls back
    to max(min_periods) from OHLC indicators.

    In by_ticks mode: klines are calculated from min_periods of OHLC
    indicators only (warmup does not apply here).

    Args:
        ohlc_proxy: OhlcProxy instance to calculate for.

    Returns:
        int: Number of klines to fetch (minimum 1 or 50 depending on mode).
    """
    MIN_KLINES = 50

    if self._feed_type == "kline":
        warmup = getattr(self.strategy, "warmup", None)
        if warmup is not None:
            return int(warmup)

    proxy_id = id(ohlc_proxy)
    n = 0
    for defn in self.strategy._indicator_defs:
        if (
            isinstance(defn["source"].proxy, OhlcProxy)
            and id(defn["source"].proxy) == proxy_id
        ):
            n = max(n, getattr(defn["indicator"], "min_periods", 1))

    if n == 0:
        base = 1 if self._feed_type == "tick" else MIN_KLINES
    else:
        base = max(n, 1)

    # Tick mode: warm-fill the OHLC ring up to its window so recursive
    # indicators (RSI = Wilder, MACD = EMA) seed from a full causal window and
    # match the backtest's trailing-window seed (backtest/live parity).
    if self._feed_type == "tick":
        return max(base, int(getattr(ohlc_proxy, "_window_size", base)))
    return base


def _fetch_auto_history(self) -> dict:
    """
    Fetch historical data needed for warmup based on feed_type.

    In tick mode: fetches ticks (data[symbol]) and klines
    (data[(symbol, interval_ms)]).

    In kline mode: only fetches klines per OhlcProxy.

    Returns:
        dict: Historical data with format {symbol: tick_array} or
              {(symbol, interval_ms): kline_array}.

    Raises:
        DataError: If historical data cannot be fetched or is empty.
    """
    data: dict = {}

    if self._feed_type == "tick":
        _fetch_ticks_for_warmup(self, data)
        _fetch_klines_for_warmup(self, data)
    else:
        _fetch_klines_for_warmup(self, data)

    return data


def _fetch_ticks_for_warmup(self, data: dict) -> None:
    """
    Fetch historical ticks for all symbols in the engine.

    If session implements _fetch_coherent_historical (e.g., FakeLiveSesh),
    delegates to that method to obtain ticks synthesized from klines.
    Otherwise fetches directly via _fetch_ticks_history.

    Args:
        data: Dict to populate with {symbol: tick_array}.

    Raises:
        DataError: If no historical ticks available.
    """
    n_ticks = _compute_required_ticks(self)

    simbolos: set[str] = set()
    for tp in self.strategy._tick_proxies:
        simbolos.add(tp.symbol)
    for op in self.strategy._ohlc_proxies:
        simbolos.add(op.symbol)
    for fp in self.strategy._fp_proxies:
        simbolos.add(fp.symbol)

    for sym in simbolos:
        if hasattr(self.sesh, "_fetch_historico_coherente"):
            proxies_del_sym = [
                op for op in self.strategy._ohlc_proxies if op.symbol == sym
            ]
            coherente = self.sesh._fetch_historico_coherente(
                sym, n_ticks, proxies_del_sym
            )
            for k, v in coherente.items():
                if k not in data:
                    data[k] = v
            continue

        try:
            result = self.sesh._fetch_ticks_history(sym, n_ticks)
        except Exception as e:
            # Abort startup: without historical data we cannot operate blind.
            raise DataError(
                f"LiveEngine: could not fetch historical ticks for "
                f"'{sym}' (requested: {n_ticks}). Warmup cannot be completed."
            ) from e

        if result is None or len(result) == 0:
            if not self._require_warmup:
                warnings.warn(
                    f"LiveEngine: broker returned no historical ticks for "
                    f"'{sym}' (requested: {n_ticks}). Starting with an empty "
                    f"tick ring (require_warmup=False)."
                )
                continue
            raise DataError(
                f"LiveEngine: broker returned no historical ticks for "
                f"'{sym}' (requested: {n_ticks}). Warmup cannot be completed."
            )
        data[sym] = result
        if len(result) < n_ticks:
            warnings.warn(
                f"LiveEngine: broker returned {len(result)} ticks for "
                f"'{sym}' (requested: {n_ticks})."
            )


def _fetch_klines_for_warmup(self, data: dict) -> None:
    """
    Fetch historical klines for each OhlcProxy.

    Stores fetched klines in data[(symbol, interval_ms)].

    Args:
        data: Dict to populate with {(symbol, interval_ms): kline_array}.

    Raises:
        DataError: If no historical klines available.
    """
    for op in self.strategy._ohlc_proxies:
        sym       = op.symbol
        intervalo = op.interval_ms
        key       = (sym, intervalo)
        if key in data:
            continue

        n_klines = _compute_required_klines(self, op)
        try:
            klines = self.sesh._fetch_klines_history(sym, intervalo, n_klines)
        except Exception as e:
            raise DataError(
                f"LiveEngine: could not fetch klines for '{sym}' "
                f"interval={intervalo}ms (requested: {n_klines}). "
                f"Warmup cannot be completed."
            ) from e

        if klines is None or len(klines) == 0:
            if not self._require_warmup:
                warnings.warn(
                    f"LiveEngine: broker returned no klines for '{sym}' "
                    f"interval={intervalo}ms (requested: {n_klines}). Starting "
                    f"with an empty OHLC ring (require_warmup=False)."
                )
                continue
            raise DataError(
                f"LiveEngine: broker returned no klines for '{sym}' "
                f"interval={intervalo}ms (requested: {n_klines}). "
                f"Warmup cannot be completed."
            )
        data[key] = klines
        if len(klines) < n_klines:
            warnings.warn(
                f"LiveEngine: broker returned {len(klines)} klines for "
                f"'{sym}' interval={intervalo}ms (requested: {n_klines})."
            )

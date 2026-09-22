"""
replay_sesh.py
==============
ReplaySesh -- replay session over historical data that simulates the feed
of a live broker, with support for manual cursor control from a
ReplayController.

The cursor ONLY advances when someone calls sesh.step(). This mode is used
by ReplayController to precisely control speed, pause and manual step
from the Bokeh buttons.

PUBLIC API
==========
    sesh = ReplaySesh(data={"BTCUSDT": tick_array})

    # Control methods (used by ReplayController)
    sesh.step("BTCUSDT")   # advance exactly one tick for that symbol
    sesh.step()            # advance one tick for all symbols

    # Broker API called internally by LiveEngine
    sesh._fetch_last_tick(symbol)
    sesh._fetch_ticks_history(symbol, n)
    sesh._fetch_klines_history(symbol, interval_ms, n)
    sesh._fetch_last_kline(symbol, interval_ms)

CURSOR MANAGEMENT
=================
- The cursor starts at -1 (no data).
  The first call to step() moves it to 0 (first available tick).
  _fetch_last_tick() returns the tick at _cursor[symbol].
  If the cursor has not been initialized yet (cursor < 0) returns None
  and the engine processes no data -- warmup does not start until the
  first ticks arrive via step().

- If warmup data (history) is passed separately from the replay data
  (live), the cursor is automatically placed at the first live data tick.

COMPATIBILITY WITH LIVEENGINE
==============================
ReplaySesh is a drop-in for any real sesh. It only needs to implement
the _fetch_* methods that LiveEngine calls. Other broker methods
(orders, positions, deals...) can be added or inherited from a base
class if needed by the strategy.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from tradetropy.core.broker import CommissionType, PositionMode
from tradetropy.core.constants import TICK_COLS, OHLC_COLS, N_TICK_COLS, N_OHLC_COLS, _TICK_COL
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.exceptions import TradingError

# =============================================================================
# KLINE HELPERS
# =============================================================================

def _ticks_to_klines(
    ticks: np.ndarray,
    interval_ms: int,
) -> np.ndarray:
    """
    Convert tick array to OHLCV klines.

    Tick columns (fixed indices):
        0: ts (timestamp), 1: bid, 2: ask, 3: volume, 4: flags,
        5: volume_real, 6: price

    Output kline columns:
        0: ts_ms (candle open), 1: open, 2: high, 3: low, 4: close, 5: volume

    Only includes candles with at least one tick. Uses vectorized
    implementation from tradetropy.data._klines and trims to 6 columns
    (excludes turnover).

    Args:
        ticks (np.ndarray): Tick array of shape [N, N_TICK_COLS].
        interval_ms (int): Candle duration in milliseconds.

    Returns:
        np.ndarray: OHLCV array of shape [M, N_OHLC_COLS].
    """
    if ticks is None or len(ticks) == 0:
        return np.empty((0, N_OHLC_COLS), dtype=np.float64)

    from tradetropy.data._klines import ticks_to_klines

    klines = ticks_to_klines(ticks, interval_ms, include_partial=True)
    if len(klines) == 0:
        return np.empty((0, N_OHLC_COLS), dtype=np.float64)
    return np.ascontiguousarray(klines[:, :N_OHLC_COLS])


# =============================================================================
# REPLAY SESH
# =============================================================================

class ReplaySesh(SeshSimulatorBase):
    """
    Replay session that feeds LiveEngine with historical data.

    Simulates a live broker feed using historical tick or kline data.
    Inherits from SeshSimulatorBase (no GBM or dynamic symbol setup).
    Data is loaded via _bind_data() after ReplayEngine builds the session.

    Supports manual cursor control through step() for pause/play/speed
    features accessible from chart buttons.

    Args:
        feed_type (str | None): 'tick' or 'kline' internal broker type.
        initial_balance (float): Initial account balance.
        commission (float): Commission per operation.
        commission_type (CommissionType): Money or percentage based.
        position_mode (PositionMode): Netting or hedging positions.
        use_margin (bool): Enable margin trading.
        margin_rate (float): Margin rate for calculations.
        slippage_points (int): Slippage in points.
        use_spread (bool): Apply bid-ask spread.
        trade_on_close (bool): Execute orders at candle close.
        execution_price_source (str): Price source for execution.
    """

    _ultimo_ts: int = 0   # compatibility with LiveEngine

    def __init__(
        self,
        feed_type: "str | None" = None,
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
        stop_out_of_money: bool = False,
        finalize_trades: bool = False,
        **extra,  # only _symbol_configs from clone_config()
    ):
        _sym_configs: "dict | None" = extra.pop("_symbol_configs", None)
        super().__init__(
            feed_type=feed_type,
            initial_balance=initial_balance,
            commission=commission,
            commission_type=commission_type,
            position_mode=position_mode,
            use_margin=use_margin,
            margin_rate=margin_rate,
            slippage_points=slippage_points,
            use_spread=use_spread,
            trade_on_close=trade_on_close,
            execution_price_source=execution_price_source,
            stop_out_of_money=stop_out_of_money,
            finalize_trades=finalize_trades,
        )

        # Re-apply symbol configurations from the original session
        if _sym_configs:
            self._broker.symbols.clear()
            for cfg in _sym_configs.values():
                self._broker.add_symbol(cfg)

        self._datasets:       dict[str, np.ndarray] = {}
        self._kline_cache:    dict[tuple, np.ndarray] = {}
        self._warmup_ticks:   "dict[str, int]" = {}
        self._cursor:         "dict[str, int]" = {}
        self._history:        dict = {}

    # -- Bind data (called by ReplayEngine) ------------------------------------

    def _bind_data(
        self,
        data: dict,
        history: "dict | None" = None,
        warmup_ticks: "int | None" = None,
    ) -> None:
        """
        Bind historical data to session.

        Called by ReplayEngine after building the session.
        Populates internal _datasets, _cursor, _warmup_ticks and _history.

        If history is not provided separately, it is inferred from warmup_ticks.
        With n_warmup=0, history is empty and the entire dataset is replayed
        progressively via cursor.

        Args:
            data (dict): {symbol: tick_array} or {(symbol, ms): kline_array}.
            history (dict | None): Optional pre-loaded history data.
            warmup_ticks (int | None): Number of ticks to reserve as warmup.
        """
        self._datasets.clear()
        self._kline_cache.clear()
        self._warmup_ticks.clear()
        self._cursor.clear()
        self._history.clear()

        for key, arr in data.items():
            if isinstance(key, tuple):
                self._kline_cache[key] = np.asarray(arr, dtype=np.float64)
            else:
                symbol = key
                if hasattr(arr, 'data') and hasattr(arr, 'symbol'):
                    arr = arr.data
                ticks  = np.asarray(arr, dtype=np.float64)
                self._datasets[symbol] = ticks

                n_warmup = warmup_ticks if warmup_ticks is not None else 0
                self._warmup_ticks[symbol] = int(n_warmup)
                self._cursor[symbol] = n_warmup - 1

        if history is not None:
            self._history.update(history)
        else:
            # Infer history from data using warmup_ticks. If n_warmup == 0
            # the history is EMPTY (nothing preloaded) -- the entire dataset is
            # replayed progressively via cursor. Assigning the full dataset here
            # would preload it into the rings and replay would not advance.
            for sym, ticks in self._datasets.items():
                n_w = self._warmup_ticks.get(sym, 0)
                self._history[sym] = ticks[:n_w]

    # =========================================================================
    # MANUAL CONTROL
    # =========================================================================

    def step(self, symbol: "str | None" = None) -> None:
        """
        Advance cursor by one for specified symbol or all symbols.

        Does not fall below n_warmup boundary. Respects dataset length.

        Args:
            symbol (str | None): If provided, advance only this symbol.
                If None, advance all registered symbols.
        """
        syms = [symbol] if symbol else list(self._datasets)
        for s in syms:
            ticks = self._datasets.get(s)
            if ticks is None:
                continue
            cur = self._cursor.get(s, -1)
            n_warmup = self._warmup_ticks.get(s, 0)
            max_idx  = len(ticks) - 1
            new_cur  = min(cur + 1, max_idx)
            # Do not fall below n_warmup
            if new_cur < n_warmup:
                new_cur = n_warmup
            self._cursor[s] = new_cur



    @property
    def finished(self) -> bool:
        """True if all cursors have reached the end of their data."""
        for sym, ticks in self._datasets.items():
            cur = self._cursor.get(sym, -1)
            if cur < len(ticks) - 1:
                return False
        return True

    def peek_next_interval_s(self, symbol: str) -> float:
        """
        Get interval in seconds to next tick without advancing cursor.

        Returns 0.0 if no next tick exists. No side effects - does not
        change cursor position.

        Note: ReplayController no longer uses this (uses fixed rate).
        Kept for backward compatibility with external code.

        Args:
            symbol (str): Symbol to peek.

        Returns:
            float: Interval in seconds to next tick.
        """
        ticks = self._datasets.get(symbol)
        if ticks is None or len(ticks) == 0:
            return 0.0

        cur = self._cursor.get(symbol, -1)
        if cur < 0 or cur >= len(ticks) - 1:
            return 0.0

        ts_cur  = ticks[cur][0]
        ts_next = ticks[cur + 1][0]
        diff    = ts_next - ts_cur
        if diff <= 0:
            return 0.0

        return diff / 1000.0

    def _assert_has_data(self, symbol: str) -> None:
        """
        Raise TradingError if symbol has no data loaded.

        Args:
            symbol (str): Symbol to check.

        Raises:
            TradingError: If data not loaded for symbol.
        """
        if symbol not in self._datasets:
            raise TradingError(
                f"ReplaySesh has no data for '{symbol}'. "
                "Did you forget to pass data= to ReplayEngine?"
            )

    def _fetch_pending_ticks(
        self,
        symbol: str,
        last_idx: int,
        up_to: "int | None" = None,
    ) -> list:
        """
        Get ticks in range (last_idx, up_to] in chronological order.

        Read-only operation - does not advance cursor.

        Useful when controller runs at high speed or turbo mode to drain
        multiple accumulated ticks in one call.

        The end of the range is ``up_to`` when provided, otherwise the current
        cursor. The caller (BaseEngine._drain_pending_ticks) passes a cursor
        SNAPSHOT taken once per drain so a concurrent step() (the controller
        loop and the Step button both advance the cursor) cannot move the end of
        the range between the fetch and the drain bookkeeping - which would
        otherwise skip the ticks that landed in between and shrink candle volume.

        Args:
            symbol (str): Symbol to fetch ticks for.
            last_idx (int): Last processed tick index (exclusive start).
            up_to (int | None): Inclusive end index. Defaults to the current
                cursor when None (backward-compatible standalone use).

        Returns:
            list: List of tick arrays in chronological order.
        """
        self._assert_has_data(symbol)
        ticks = self._datasets.get(symbol)
        cur = self._cursor.get(symbol, -1) if up_to is None else int(up_to)
        # Never index past the dataset even if a stale snapshot is passed.
        cur = min(cur, len(ticks) - 1)
        n_warmup = self._warmup_ticks.get(symbol, 0)
        start = max(last_idx + 1, n_warmup)
        if cur < start:
            return []
        return [ticks[i].copy() for i in range(start, cur + 1)]

    # =========================================================================
    # BROKER API -- called by LiveEngine
    # =========================================================================

    def _fetch_last_tick(self, symbol: str) -> "np.ndarray | None":
        """
        Get tick at current cursor position.

        Returns None if cursor not ready or data unavailable.

        Args:
            symbol (str): Symbol to fetch.

        Returns:
            np.ndarray | None: Tick array or None if cursor not ready.
        """
        self._assert_has_data(symbol)
        ticks = self._datasets[symbol]
        if len(ticks) == 0:
            return None

        cur = self._cursor.get(symbol, -1)
        n_warmup = self._warmup_ticks.get(symbol, 0)

        if cur < n_warmup:
            return None
        if cur >= len(ticks):
            cur = len(ticks) - 1
        return ticks[cur].copy()

    def _fetch_ticks_history(
        self,
        symbol: str,
        n: int,
    ) -> "np.ndarray | None":
        """
        Get last n ticks from warmup history.

        Called by LiveEngine.prepare() to warm up the tick_ring.

        Args:
            symbol (str): Symbol to fetch history for.
            n (int): Number of ticks to return.

        Returns:
            np.ndarray: Tick history array.
        """
        self._assert_has_data(symbol)
        hist = self._history.get(symbol)
        if hist is None or len(hist) == 0:
            return np.empty((0, N_TICK_COLS), dtype=np.float64)
        hist = np.asarray(hist, dtype=np.float64)
        return hist[-n:].copy() if len(hist) > n else hist.copy()

    def _fetch_historico_coherente(
        self,
        symbol: str,
        n_ticks: int,
        ohlc_proxies: list,
    ) -> dict:
        d = {}
        d[symbol] = self._fetch_ticks_history(symbol, n_ticks)
        if ohlc_proxies:
            for intv in sorted(set(op.interval_ms for op in ohlc_proxies)):
                # Warm-fill a full ring window of klines so recursive indicators
                # (RSI = Wilder, MACD = EMA) seed from the same trailing causal
                # window as the backtest. _fetch_klines_history caps to the
                # available warmup history (no look-ahead into the live region),
                # so this is bounded by the reserved warmup ticks.
                nk = max(50, int(max(
                    (op._window_size for op in ohlc_proxies
                     if op.interval_ms == intv),
                    default=50,
                )))
                d[(symbol, intv)] = self._fetch_klines_history(symbol, intv, nk)
        return d

    def _fetch_klines_history(
        self,
        symbol: str,
        interval_ms: int,
        n: int,
    ) -> "np.ndarray | None":
        """
        Get last n klines from history calculated from warmup ticks.

        Uses pre-calculated klines from _kline_cache if available.

        Args:
            symbol (str): Symbol to fetch kline history for.
            interval_ms (int): Candle interval in milliseconds.
            n (int): Number of klines to return.

        Returns:
            np.ndarray: Kline history array.
        """
        self._assert_has_data(symbol)
        key = (symbol, interval_ms)

        # Pre-calculated klines for history
        hist_key = ("_hist", symbol, interval_ms)
        if hist_key not in self._kline_cache:
            hist_ticks = self._history.get(symbol)
            if hist_ticks is not None and len(hist_ticks) > 0:
                self._kline_cache[hist_key] = _ticks_to_klines(
                    hist_ticks, interval_ms
                )
            elif key in self._kline_cache:
                self._kline_cache[hist_key] = self._kline_cache[key]
            else:
                self._kline_cache[hist_key] = np.empty(
                    (0, N_OHLC_COLS), dtype=np.float64
                )

        klines = self._kline_cache[hist_key]
        if len(klines) == 0:
            return np.empty((0, N_OHLC_COLS), dtype=np.float64)
        return klines[-n:].copy() if len(klines) > n else klines.copy()

    def _fetch_last_kline(
        self,
        symbol: str,
        interval_ms: int,
    ) -> "np.ndarray | None":
        """
        Get most recent kline for by_klines mode.

        Generates kline from ticks up to current cursor position.

        Args:
            symbol (str): Symbol to fetch.
            interval_ms (int): Candle interval in milliseconds.

        Returns:
            np.ndarray | None: Last kline array or None if unavailable.
        """
        self._assert_has_data(symbol)
        ticks = self._datasets[symbol]
        if len(ticks) == 0:
            return None

        cur = self._cursor.get(symbol, -1)
        n_warmup = self._warmup_ticks.get(symbol, 0)
        end = max(cur + 1, n_warmup)
        visible = ticks[:end]

        if len(visible) == 0:
            return None

        klines = _ticks_to_klines(visible, interval_ms)
        if len(klines) == 0:
            return None

        return klines[-1].copy()

    # =========================================================================
    # BROKER STUBS (compatibility with LiveEngine / strategies)
    # =========================================================================

    def deals(self, symbol: "str | None" = None) -> list:
        return super().deals(symbol)

    def positions(self, symbol: "str | None" = None) -> list:
        return super().positions(symbol)

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        syms = list(self._datasets)
        cursors = {s: self._cursor.get(s, "n/a") for s in syms}
        return (
            f"ReplaySesh(symbols={syms}, "
            f"cursors={cursors})"
        )

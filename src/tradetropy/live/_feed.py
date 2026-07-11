"""
Hot processing of ticks and bars (on_tick / on_bar),
including OHLC indicator updates, pattern stores,
strategy_ready check and recording to disk.
"""

from __future__ import annotations

import warnings

import numpy as np

from tradetropy.core.constants import (
    N_TICK_COLS,
    N_OHLC_COLS,
    _TICK_COL,
    _OHLC_COL,
)
from tradetropy.exceptions import TradingError


def _count_closed_candles_rings(self) -> int:
    """
    Count minimum number of closed candles across all active OhlcRings.

    Returns:
        int: Minimum number of closed candles. Returns 0 if no rings.
    """
    if not self.strategy._ohlc_proxies:
        return 0
    counts = []
    for op in self.strategy._ohlc_proxies:
        ring = op._ohlc_ring
        if ring is not None:
            counts.append(ring._n_closed)
    return min(counts) if counts else 0


def _strategy_ready(self) -> bool:
    """
    Decide whether on_data() should run on this tick or bar.

    Checks warmup conditions:
    - by_klines: warmup in closed candles
    - by_ticks: hybrid warmup (ticks + historical candles)

    Returns:
        bool: True if warmup complete and strategy ready for on_data() calls.
    """
    for op in self.strategy._ohlc_proxies:
        ring = op._ohlc_ring
        if ring is None or ring._ts_current_candle < 0:
            return False

    warmup = getattr(self.strategy, "warmup", None)
    if warmup is None:
        from tradetropy.models._warmup_policy import auto_warmup_candles
        warmup = auto_warmup_candles(self.strategy)
    if warmup == 0:
        return True

    if self._feed_type == "tick":
        if self._tick_rings:
            n_escritos = min(
                ring._n_writings for ring in self._tick_rings.values()
            )
            n_ohlc_hist = self._historical_candles

            if n_escritos < warmup and (n_escritos + n_ohlc_hist) >= warmup:
                if not getattr(self, '_warmup_hibrido_avisado', False):
                    warnings.warn(
                        f"Tick warmup ({warmup}) completed with help from "
                        f"{n_ohlc_hist} historical candles. "
                        f"Real ticks: {n_escritos}. "
                        f"This is normal if the broker does not deliver enough "
                        f"historical ticks."
                    )
                    self._warmup_hibrido_avisado = True

            return (n_escritos + n_ohlc_hist) >= warmup

        live_candles = _count_closed_candles_rings(self) - self._historical_candles
        has_partial = any(
            op._ohlc_ring._ts_current_candle >= 0
            for op in self.strategy._ohlc_proxies
            if op._ohlc_ring is not None
        )
        return (self._historical_candles + max(0, live_candles) + has_partial) >= warmup

    else:
        live_candles = _count_closed_candles_rings(self) - self._historical_candles
        has_partial = any(
            op._ohlc_ring._ts_current_candle >= 0
            for op in self.strategy._ohlc_proxies
            if op._ohlc_ring is not None
        )
        return (self._historical_candles + max(0, live_candles) + has_partial) >= warmup



def _update_ohlc_indicators_on_close(self, symbol: str, ohlc_ring) -> None:
    """
    Recalculate OHLC indicators when a candle closes.

    Handles both regular and multi-band indicators, with full window
    or trailing-window computation modes.

    Args:
        symbol: Trading symbol.
        ohlc_ring: OhlcRing that closed a candle.
    """
    for estado in self._ind_state:
        if estado["symbol"] != symbol or not estado["is_ohlc"]:
            continue
        if estado["ohlc_ring"] is not ohlc_ring:
            continue
        ind = estado["indicator"]
        multi_source_bc  = estado.get("multi_source", False)
        multi_band_bc   = estado.get("multi_band", False)
        use_full_win     = not getattr(ind, "use_partial", True)
        W                = ohlc_ring._W

        if use_full_win:
            n_closed = min(ohlc_ring._n_closed, W)
            if n_closed == 0:
                continue
            if multi_source_bc:
                vals = np.column_stack([
                    ohlc_ring.closed_window(c, n_closed)
                    for c in estado["col_srcs"]
                ])
            else:
                vals = ohlc_ring.closed_window(estado["col_src"], n_closed)
            if (vals.shape[0] if vals.ndim > 1 else len(vals)) == 0:
                continue
            result = ind.calculate(vals)
            n_vals = result.shape[1] if (multi_band_bc and result.ndim > 1) else len(result)
            for i in range(n_vals):
                p = (ohlc_ring._head - n_vals + i) % W
                if multi_band_bc:
                    for k, ci in enumerate(estado["col_inds"]):
                        v = float(result[k, i]) if result.ndim > 1 else float(result[i])
                        ohlc_ring._buf[p,     ci] = v
                        ohlc_ring._buf[p + W, ci] = v
                else:
                    ci = estado["col_ind"]
                    v  = float(result[i]) if result.ndim == 1 else float(result[0, i])
                    ohlc_ring._buf[p,     ci] = v
                    ohlc_ring._buf[p + W, ci] = v
        else:
            # Recursive indicators (RSI, MACD) must seed from the same finite
            # causal window as the backtest precompute (the ring window_size),
            # not just min_periods, so the just-closed bar matches across
            # engines. closed_window() clamps to the available closed candles.
            L: int = W
            if multi_source_bc:
                vals = np.column_stack([
                    ohlc_ring.closed_window(c, L)
                    for c in estado["col_srcs"]
                ])
                n_vals = vals.shape[0]
            else:
                vals = ohlc_ring.closed_window(estado["col_src"], L)
                n_vals = len(vals)
            if n_vals > 0:
                result = ind.calculate(vals)
                p_closed = (ohlc_ring._head - 1) % W
                if multi_band_bc:
                    for k, ci in enumerate(estado["col_inds"]):
                        res_k = float(result[k, -1]) if result.ndim > 1 else float(result[-1])
                        ohlc_ring._buf[p_closed,     ci] = res_k
                        ohlc_ring._buf[p_closed + W, ci] = res_k
                else:
                    res = result[-1] if result.ndim == 1 else result[0, -1]
                    ohlc_ring._buf[p_closed,     estado["col_ind"]] = float(res)
                    ohlc_ring._buf[p_closed + W, estado["col_ind"]] = float(res)


def _update_ohlc_indicators_partial(self, symbol: str, ohlc_ring) -> None:
    """
    Update OHLC indicators on the partial candle (intrabar update).

    Called on each new tick to keep partial candle indicators current.

    Args:
        symbol: Trading symbol.
        ohlc_ring: OhlcRing with partial candle.
    """
    for estado in self._ind_state:
        if estado["symbol"] != symbol or not estado["is_ohlc"]:
            continue
        if estado["ohlc_ring"] is not ohlc_ring:
            continue
        ind = estado["indicator"]
        use_partial = getattr(ind, "use_partial", True)
        # Full-window indicators (use_partial=False) normally only change on
        # closes; but those marked recompute_on_partial (e.g. Volume Profile)
        # must evolve within the bar. In that case, the full window of closed
        # candles + the partial candle is fed.
        full_window = (not use_partial) and getattr(ind, "recompute_on_partial", False)
        if not use_partial and not full_window:
            continue

        # Number of closed candles to prepend to the partial. In full-window
        # mode, the entire available window is used (calculate trims to its
        # `length`); in normal mode, L-1 closed candles suffice for the causal
        # calculation.
        if full_window:
            n_closed = getattr(ind, "length", ohlc_ring._W)
        else:
            # Partial bar uses a min_periods causal window (cheap per tick) and
            # matches the backtest partial. Closed bars are recomputed over the
            # full ring window (see _update_ohlc_indicators_on_close).
            L: int = getattr(ind, "min_periods", getattr(ind, "length", 1))
            n_closed = L - 1

        if estado.get("multi_source"):
            col_srcs = estado["col_srcs"]
            vals_closed = np.column_stack([
                ohlc_ring.closed_window(c, n_closed) for c in col_srcs
            ]) if n_closed > 0 else np.empty((0, len(col_srcs)), dtype=np.float64)
            partial_row = ohlc_ring.current_partial_candle[col_srcs]
            window = (
                np.vstack([vals_closed, partial_row])
                if len(vals_closed) > 0
                else partial_row[np.newaxis, :]
            )
        else:
            col_src = estado["col_src"]
            vals = ohlc_ring.closed_window(col_src, n_closed)
            close_partial = float(ohlc_ring.current_partial_candle[_OHLC_COL["close"]])
            window = np.append(vals, close_partial) if len(vals) > 0 else np.array([close_partial])

        if (window.shape[0] if window.ndim > 1 else len(window)) < 1:
            continue

        result = ind.calculate(window)
        # Store the partial indicator value ONLY in the shadow slot (2W-1),
        # never in the primary slot W-1. W-1 is a real closed-candle position in
        # the circular buffer and closed_window() reads it (but never 2W-1), so
        # writing the partial to W-1 corrupts a closed bar's indicator once the
        # ring wraps. See WindowView._ohlc_live_view (the matching reader).
        p_partial = 2 * ohlc_ring._W - 1

        multi_band = estado.get("multi_band", False)
        if multi_band:
            col_inds = estado["col_inds"]
            for k, ci in enumerate(col_inds):
                res_k = float(result[k, -1]) if result.ndim > 1 else float(result[-1])
                ohlc_ring._buf[p_partial, ci] = res_k
        else:
            res = result[-1] if result.ndim == 1 else result[0, -1]
            ci  = estado["col_ind"]
            ohlc_ring._buf[p_partial, ci] = float(res)



def _update_pattern_stores_live(self, symbol: str) -> None:
    """
    Rebuild the PatternStore for a symbol when one of its candles closes.

    A full rebuild from the ring is used instead of an incremental append
    because some pivot decorators are RETROACTIVE: classifying a new pivot
    can re-tag an earlier one. NBS, for instance, promotes the immediately
    preceding contrary pivot (EMP -> BOO, NEU -> SHK) when a new pivot breaks
    structure, so a pivot's final tag depends on a LATER pivot.

    The on-close indicator pass (``_update_ohlc_indicators_on_close``)
    recomputes those decorators over the whole closed window and writes the
    final, fully-resolved tags back into the ring. Snapshotting a pivot's tag
    once at confirmation time would freeze the pre-promotion tag and diverge
    from the backtest (which always sees the final tags), breaking pattern
    matches in live and replay. Rebuilding from the ring reuses the SAME
    ``FrozenPivotSequence.from_arrays`` + ``PatternStore`` construction as the
    backtest and warmup paths, so live and replay match the backtest exactly.

    Complexity is O(n_pivots x L) per close (n_pivots bounded by the ring
    window); it runs once per candle close, not per tick.

    Args:
        symbol: Trading symbol whose candle just closed.
    """
    if not getattr(self.strategy, "_pattern_matcher_defs", None):
        return
    from tradetropy.ta.pattern.sequence import FrozenPivotSequence
    from tradetropy.ta.pattern.store    import PatternStore
    from tradetropy.ta.pattern.types    import PivotPoint
    from tradetropy.ta.pattern._builder import iter_pattern_matcher_defs

    for pm_def, _, ohlc_proxy, sym, base_cols, decorator_cols, tag_decoders \
        in iter_pattern_matcher_defs(self.strategy, self._get_ind_def_for_proxy):

        if sym != symbol:
            continue

        ohlc_ring = ohlc_proxy._ohlc_ring
        col       = ohlc_ring.col_index

        if not all(c in col for c in base_cols):
            continue

        n = min(ohlc_ring._n_closed, ohlc_ring._W)
        if n <= 0:
            continue

        # Absolute bar index of the oldest candle still visible in the ring.
        # closed_window() returns the last n closed candles in chronological
        # order, so window position i maps to absolute bar window_start + i.
        window_start = ohlc_ring._n_closed - n

        ph    = ohlc_ring.closed_window(col[base_cols[0]], n)
        pl    = ohlc_ring.closed_window(col[base_cols[1]], n)
        ph_ts = ohlc_ring.closed_window(col[base_cols[2]], n)
        pl_ts = ohlc_ring.closed_window(col[base_cols[3]], n)

        decorator_arrays: dict[str, np.ndarray] = {}
        for tag_name, dec_col in decorator_cols.items():
            if dec_col in col:
                decorator_arrays[tag_name] = ohlc_ring.closed_window(col[dec_col], n)

        # Re-read the pivots currently in the window with their FINAL,
        # fully-resolved decorator tags (the on-close indicator pass already
        # wrote any retroactive promotions back into the ring).
        win_seq = FrozenPivotSequence.from_arrays(
            ph_array         = ph,
            pl_array         = pl,
            ph_ts_array      = ph_ts,
            pl_ts_array      = pl_ts,
            decorator_arrays = decorator_arrays,
            tag_decoders     = tag_decoders,
        )
        win_pivots = [
            PivotPoint(
                index     = p.index + window_start,
                timestamp = p.timestamp,
                value     = p.value,
                type      = p.type,
                tags      = p.tags,
            )
            for p in win_seq.pivots
        ]

        # Preserve pivots that have already scrolled out of the ring window:
        # their tags were finalized while still in-window (NBS only re-tags the
        # immediately preceding contrary pivot, so a tag is final within a few
        # pivots, long before it leaves the window). This keeps the live store
        # history unbounded by the ring, matching the backtest sequence.
        prev_store = pm_def.proxy._store
        old_pivots = (
            [p for p in prev_store._sequence.pivots if p.index < window_start]
            if prev_store is not None
            else []
        )

        merged      = old_pivots + win_pivots
        bar_indices = np.array([p.index for p in merged], dtype=np.int64)
        sequence    = FrozenPivotSequence(pivots=merged, _pivot_bar_indices=bar_indices)
        pm_def.proxy._connect_live(PatternStore(sequence, pm_def.pattern))


def _process_tick(self, symbol: str, tick: np.ndarray):
    """
    Process a tick in tick mode.

    Updates ring buffers, calculates indicators, updates footprints,
    refreshes chart and calls strategy.on_data() if warmup is complete.

    Args:
        symbol: Trading symbol.
        tick: Tick array [ts_ms, bid, ask, volume, flags, volume_real, price].

    Raises:
        TradingError: If symbol not prepared in prepare().
    """
    tick_ring = self._tick_rings.get(symbol)
    if tick_ring is None:
        raise TradingError(
            f"Symbol '{symbol}' not prepared. Call prepare() first."
        )

    if not isinstance(tick, np.ndarray) or tick.dtype.names is not None:
        tick = np.array([float(v) for v in tick], dtype=np.float64)
    tick = np.asarray(tick, dtype=np.float64).ravel()
    if len(tick) < N_TICK_COLS:
        pad = np.zeros(N_TICK_COLS, dtype=np.float64)
        pad[:len(tick)] = tick
        tick = pad

    price  = float(tick[_TICK_COL["price"]])
    volume = float(tick[_TICK_COL["volume"]])
    ts_ms   = int(tick[_TICK_COL["ts"]])

    if self.sesh is not None:
        self.sesh._ultimo_ts = ts_ms

    n_cols = tick_ring._buf.shape[1]
    tick_row = np.empty(n_cols, dtype=np.float64)
    tick_row[:N_TICK_COLS] = tick[:N_TICK_COLS]

    for estado in self._ind_state:
        if estado["symbol"] != symbol or estado["is_ohlc"]:
            continue
        ind = estado["indicator"]
        # Feed the full causal window the indicator needs, not just min_periods.
        # Relative-threshold tick indicators (LargeTrades / DeepTrades) declare
        # ``length`` = their detection window; feeding only ``min_periods`` (1)
        # leaves a 'pXX' / 'Nx' threshold in perpetual warmup, so the per-tick
        # bands come out NaN in live/replay while the backtest (full-series
        # precompute) detects them - a parity break for any strategy reading
        # ``self.whales.price[-1]``. Taking the max keeps windowed indicators
        # (length == min_periods) unchanged; extra history is always safe for a
        # causal indicator (only the last value is used).
        L = max(
            int(getattr(ind, "min_periods", 1) or 1),
            int(getattr(ind, "length", 1) or 1),
        )
        multi_source = estado.get("multi_source", False)
        multi_band   = estado.get("multi_band", False)

        # Build the causal window ending at the current tick. Multi-source
        # indicators consume a 2D [N x K] window (one column per source);
        # single-source ones a 1D [N] window.
        if multi_source:
            col_srcs = estado["col_srcs"]
            prev = (
                np.column_stack([tick_ring.window(c, L - 1) for c in col_srcs])
                if L > 1 else np.empty((0, len(col_srcs)), dtype=np.float64)
            )
            cur = np.array([float(tick[c]) for c in col_srcs], dtype=np.float64)
            window = np.vstack([prev, cur]) if len(prev) > 0 else cur[np.newaxis, :]
        else:
            prev = tick_ring.window(estado["col_src"], L - 1)
            window = np.append(prev, float(tick[estado["col_src"]]))

        result = ind.calculate(window)

        # Write each output band. Multi-band indicators return [K x N]; the
        # last column holds the value for the current tick. Single-band
        # indicators return [N] (or [1 x N] for some implementations).
        if multi_band:
            for k, ci in enumerate(estado["col_inds"]):
                tick_row[ci] = (
                    float(result[k, -1]) if result.ndim > 1
                    else float(result[-1])
                )
        else:
            tick_row[estado["col_ind"]] = (
                float(result[-1]) if result.ndim == 1
                else float(result[0, -1])
            )

    tick_ring.write(tick_row)

    _any_bar_closed = False
    for ohlc_ring in self._ohlc_rings.get(symbol, []):
        bar_closed = (
            ohlc_ring._ts_current_candle >= 0
            and (ts_ms // ohlc_ring.interval_ms) * ohlc_ring.interval_ms
            != ohlc_ring._ts_current_candle
        )
        if bar_closed:
            _any_bar_closed = True
        ohlc_ring.process_tick(ts_ms, price, volume)

        if bar_closed:
            _update_ohlc_indicators_on_close(self, symbol, ohlc_ring)
            _update_pattern_stores_live(self, symbol)

        _update_ohlc_indicators_partial(self, symbol, ohlc_ring)

    for fp_ring in self._fp_rings.get(symbol, []):
        fp_ring.process_tick(
            timestamp_ms=ts_ms,
            price=price,
            vol=volume,
            bid=float(tick[_TICK_COL["bid"]]),
            ask=float(tick[_TICK_COL["ask"]]),
            flag=float(tick[_TICK_COL["flags"]]),
        )

    for tp in self.strategy._tick_proxies:
        if tp.symbol == symbol:
            tp._n_total = tick_ring._n_writings

    for pm_def in self.strategy._pattern_matcher_defs:
        pm_def.proxy._advance_live()

    if self._chart is not None and not self._suppress_chart:
        self._chart._on_new_data(bar_closed=_any_bar_closed)

    if self._suppress_on_data or not _strategy_ready(self):
        return

    try:
        self.strategy._in_on_data = True
        self.strategy.on_data()
    finally:
        self.strategy._in_on_data = False

    if not self._is_simulated:
        for tp in self.strategy._tick_proxies:
            if tp.symbol == symbol and tp._record_config is not None:
                tp._record_config._buffer.append(tick[:N_TICK_COLS].copy())
                if len(tp._record_config._buffer) >= tp._record_config.flush_every:
                    self._flush_tick_proxy(tp)


def _process_bar(self, symbol: str, kline: np.ndarray):
    """
    Process a candle (kline) in kline mode.

    Updates ring buffers, calculates indicators, refreshes chart
    and calls strategy.on_data() if warmup is complete.

    Args:
        symbol: Trading symbol.
        kline: Kline array [ts_ms, open, high, low, close, volume, turnover].
    """
    ts_ms   = int(kline[0])
    close   = float(kline[4])
    volume = float(kline[5])

    if self.sesh is not None:
        self.sesh._ultimo_ts = ts_ms

    _any_bar_closed = False
    for ohlc_ring in self._ohlc_rings.get(symbol, []):
        bar_closed = (
            ohlc_ring._ts_current_candle >= 0
            and (ts_ms // ohlc_ring.interval_ms) * ohlc_ring.interval_ms
            != ohlc_ring._ts_current_candle
        )
        if bar_closed:
            _any_bar_closed = True
        ohlc_ring.process_tick(ts_ms, close, volume)

        if bar_closed:
            _update_ohlc_indicators_on_close(self, symbol, ohlc_ring)
            _update_pattern_stores_live(self, symbol)

            if not self._is_simulated:
                for op in self.strategy._ohlc_proxies:
                    if op.symbol == symbol and op._record_config is not None and op._ohlc_ring is ohlc_ring:
                        p_closed = (ohlc_ring._head - 1) % ohlc_ring._W
                        candle = ohlc_ring._buf[p_closed, :N_OHLC_COLS].copy()
                        op._record_config._buffer.append(candle)
                        if len(op._record_config._buffer) >= op._record_config.flush_every:
                            self._flush_ohlc_proxy(op)

        _update_ohlc_indicators_partial(self, symbol, ohlc_ring)

    for pm_def in self.strategy._pattern_matcher_defs:
        pm_def.proxy._advance_live()

    if self._chart is not None and not self._suppress_chart:
        self._chart._on_new_data(bar_closed=_any_bar_closed)

    if self._suppress_on_data or not _strategy_ready(self):
        return

    self.strategy.on_data()


# =============================================================================
# STREAMING EVENT DISPATCH
# =============================================================================
#
# The streaming path (LiveEngine._loop_streaming) drains normalized events from
# the EventBus and routes them here. Unlike the polling tick loop, there is NO
# `ts > last` de-duplication: every trade is processed in order, so same-
# millisecond trades (routine in crypto) are never dropped. on_data() fires per
# event, which is exactly what the replay path reproduces for parity.


def _event_to_tick_row(self, event) -> np.ndarray:
    """
    Build a raw tick row [N_TICK_COLS] from a TradeEvent or TickEvent.

    A TradeEvent carries no quote, so bid/ask default to the trade price and
    `flags` encodes the aggressor side (+1 buy / -1 sell) for the order-flow
    indicators. A TickEvent carries the L1 quote directly.

    Args:
        event: TradeEvent or TickEvent.

    Returns:
        np.ndarray: Tick row [ts, bid, ask, volume, flags, volume_real, price].
    """
    row = np.zeros(N_TICK_COLS, dtype=np.float64)
    row[_TICK_COL["ts"]] = float(event.ts)
    if event.channel == "trade":
        price = float(event.price)
        row[_TICK_COL["price"]] = price
        row[_TICK_COL["bid"]] = price
        row[_TICK_COL["ask"]] = price
        row[_TICK_COL["volume"]] = float(event.volume)
        row[_TICK_COL["volume_real"]] = float(event.volume)
        row[_TICK_COL["flags"]] = float(event.side)
    else:  # tick (L1 quote)
        row[_TICK_COL["bid"]] = float(event.bid)
        row[_TICK_COL["ask"]] = float(event.ask)
        row[_TICK_COL["price"]] = float(event.price)
        row[_TICK_COL["volume"]] = float(event.volume)
        row[_TICK_COL["flags"]] = float(event.flags)
        row[_TICK_COL["volume_real"]] = np.nan
    return row


def _event_to_kline_row(self, event) -> np.ndarray:
    """
    Build a kline row [ts, open, high, low, close, volume, turnover] from a
    KlineEvent.

    Args:
        event: KlineEvent.

    Returns:
        np.ndarray: Kline row consumed by _process_bar.
    """
    return np.array(
        [
            float(event.ts),
            float(event.open),
            float(event.high),
            float(event.low),
            float(event.close),
            float(event.volume),
            float(event.turnover),
        ],
        dtype=np.float64,
    )


def _process_event(self, event) -> None:
    """
    Dispatch one normalized streaming event to the right processing path.

    - TRADE / TICK  -> _process_tick (updates tick + OHLC rings, indicators,
      footprints, chart, on_data).
    - KLINE         -> _process_bar (partial or closed candle update).
    - Order-book events are handled once the OrderbookProxy lands (Phase B).

    Args:
        event: A normalized FeedEvent.
    """
    ch = event.channel
    if ch in ("trade", "tick"):
        _process_tick(self, event.symbol, _event_to_tick_row(self, event))
    elif ch == "kline":
        _process_bar(self, event.symbol, _event_to_kline_row(self, event))
    elif ch == "book_snapshot":
        for ring in self._book_rings.get(event.symbol, []):
            ring.apply_snapshot(event.ts, event.bids, event.asks)
        _record_book(self, event.symbol)
    elif ch == "book_delta":
        for ring in self._book_rings.get(event.symbol, []):
            ring.apply_delta(event.ts, event.bids, event.asks)
        _record_book(self, event.symbol)
    elif ch == "order":
        if self._sesh is not None:
            self._sesh.apply_order_event(event)
    elif ch == "fill":
        if self._sesh is not None:
            self._sesh.apply_fill_event(event)
    elif ch == "mbo":
        for ring in self._mbo_rings.get(event.symbol, []):
            ring.apply_event(event.ts, event.order_id, event.side,
                             event.price, event.size, event.action)
        _record_mbo(self, event.symbol)


def _record_mbo(self, symbol: str) -> None:
    """Record the most recent MBO event row (live sessions only)."""
    if self._is_simulated:
        return
    for mp in getattr(self.strategy, "_mbo_proxies", []):
        if mp.symbol != symbol or mp._record_config is None or mp._mbo_ring is None:
            continue
        row = mp._mbo_ring.last_row()
        if row is None:
            continue
        mp._record_config._buffer.append(row)
        if len(mp._record_config._buffer) >= mp._record_config.flush_every:
            self._flush_mbo_proxy(mp)


def _record_book(self, symbol: str) -> None:
    """
    Record the reconstructed top-K book image after a book event.

    Each recorded row is the full top-K image (not the incremental change), so
    replaying it as a snapshot reproduces the exact book state. Recorded from
    the first event (no warmup gating) and skipped for simulated sessions
    (replay must not re-record).

    Args:
        symbol (str): Symbol whose book just updated.
    """
    if self._is_simulated:
        return
    for bp in self.strategy._book_proxies:
        if (
            bp.symbol != symbol
            or bp._record_config is None
            or bp._book_ring is None
        ):
            continue
        row = bp._book_ring.last_row()
        if row is None:
            continue
        bp._record_config._buffer.append(row)
        if len(bp._record_config._buffer) >= bp._record_config.flush_every:
            self._flush_book_proxy(bp)

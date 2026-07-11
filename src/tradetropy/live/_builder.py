"""
Construction of rings (circular buffers), indicator warmup, topup,
pattern stores and footprint rings for LiveEngine.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from tradetropy.core.constants import (
    TICK_COLS,
    OHLC_COLS,
    N_TICK_COLS,
    N_OHLC_COLS,
    _TICK_COL,
    _OHLC_COL,
)
from tradetropy.data.data import (
    TickProxy,
    OhlcProxy,
    ColumnRef,
    WindowView,
    LiveRingBuffer,
    LiveOhlcRing,
    LiveBookRing,
    MboRing,
)
from tradetropy.ta.base import Indicator


def _build_rings(self, historical_ticks: dict):
    """
    Build LiveRingBuffers (tick) and LiveOhlcRings (OHLC) from historical data.

    Creates circular buffers for each subscribed symbol and loads historical
    data if available. Attaches rings to proxies.

    Args:
        historical_ticks: Dict with historical data. Format:
                         {symbol: tick_array} and/or
                         {(symbol, interval_ms): kline_array}
    """
    for tp in self.strategy._tick_proxies:
        symbol = tp.symbol
        col_index = dict(zip(TICK_COLS, range(N_TICK_COLS)))
        col_offset = N_TICK_COLS

        for defn in self.strategy._indicator_defs:
            if (
                isinstance(defn["source"].proxy, TickProxy)
                and defn["source"].symbol == symbol
            ):
                ind = defn["indicator"]
                multi_band = defn.get("multi_band", False)
                if multi_band:
                    K = ind.n_outputs
                    col_names = []
                    for k in range(K):
                        cname = f"{ind.col_name(symbol, defn['source']._col_name)}_b{k}"
                        col_index[cname] = col_offset
                        col_offset += 1
                        col_names.append(cname)
                    defn["col_names"] = col_names
                    defn["col_name"]  = col_names[0]
                else:
                    col_name = ind.col_name(symbol, defn["source"]._col_name)
                    col_index[col_name] = col_offset
                    defn["col_name"] = col_name
                    col_offset += 1
                defn["en_tick_store"] = True

        ring = LiveRingBuffer(tp._window_size, col_index)

        hist = historical_ticks.get(symbol)
        if hist is not None and len(hist) > 0:
            table = hist.astype(np.float64)
            W = tp._window_size
            start = max(0, len(table) - W)
            empty_row = np.full(col_offset, np.nan, dtype=np.float64)
            for i in range(start, len(table)):
                row = empty_row.copy()
                row[:N_TICK_COLS] = table[i, :N_TICK_COLS]
                ring.write(row)

        self._tick_rings[symbol] = ring
        tp._connect_live(ring)
        tp._n_total = ring._n_writings

    for op in self.strategy._ohlc_proxies:
        symbol = op.symbol
        col_index_ohlc = dict(zip(OHLC_COLS, range(N_OHLC_COLS)))
        col_offset = N_OHLC_COLS

        for defn in self.strategy._indicator_defs:
            if (
                isinstance(defn["source"].proxy, OhlcProxy)
                and defn["source"].proxy is op
            ):
                ind = defn["indicator"]
                multi_band = defn.get("multi_band", False)
                if multi_band:
                    K = ind.n_outputs
                    col_names = []
                    for k in range(K):
                        cname = f"{ind.col_name(symbol, defn['source']._col_name)}_b{k}"
                        col_index_ohlc[cname] = col_offset
                        col_offset += 1
                        col_names.append(cname)
                    defn["col_names"] = col_names
                    defn["col_name"]  = col_names[0]
                else:
                    col_name = ind.col_name(symbol, defn["source"]._col_name)
                    col_index_ohlc[col_name] = col_offset
                    defn["col_name"] = col_name
                    col_offset += 1
                defn["en_tick_store"] = False
                defn["ohlc_proxy"] = op

        ohlc_ring = LiveOhlcRing(
            op._window_size, col_index_ohlc, op.interval_ms
        )

        hist_ticks  = historical_ticks.get(symbol)
        hist_klines = historical_ticks.get((symbol, op.interval_ms))

        n_ticks_closed  = 0
        n_klines_closed = 0
        if hist_ticks is not None and len(hist_ticks) > 0:
            t = hist_ticks.astype(np.float64)
            if len(t) >= 2:
                ts0 = int(t[0,  _TICK_COL["ts"]])
                ts1 = int(t[-1, _TICK_COL["ts"]])
                n_ticks_closed = (ts1 - ts0) // op.interval_ms
        if hist_klines is not None:
            n_klines_closed = len(hist_klines)

        if n_klines_closed > n_ticks_closed and hist_klines is not None and len(hist_klines) > 0:
            klines = hist_klines.astype(np.float64)
            # The last warmup kline is the border candle (potentially partial):
            # seeding it as CLOSED and then letting the live stream reopen it
            # would duplicate the candle. So all but the last are seeded as
            # closed, and the last is left as the active partial candle
            # (same as the tick-based branch does with process_tick). The first
            # live tick continues or closes it without duplication.
            for row in klines[:-1]:
                ohlc_ring.load_kline(
                    ts     = int(row[0]),
                    open_  = float(row[1]),
                    high   = float(row[2]),
                    low    = float(row[3]),
                    close  = float(row[4]),
                    volume = float(row[5]),
                )
            last = klines[-1]
            col = ohlc_ring.col_index
            ohlc_ring._partial_candle[col["ts"]]     = float(last[0])
            ohlc_ring._partial_candle[col["open"]]   = float(last[1])
            ohlc_ring._partial_candle[col["high"]]   = float(last[2])
            ohlc_ring._partial_candle[col["low"]]    = float(last[3])
            ohlc_ring._partial_candle[col["close"]]  = float(last[4])
            ohlc_ring._partial_candle[col["volume"]] = float(last[5])
            ohlc_ring._ts_current_candle = int(last[0])
        elif hist_ticks is not None and len(hist_ticks) > 0:
            table = hist_ticks.astype(np.float64)
            for row in table:
                ohlc_ring.process_tick(
                    int(row[_TICK_COL["ts"]]),
                    float(row[_TICK_COL["price"]]),
                    float(row[_TICK_COL["volume"]]),
                )

        if symbol not in self._ohlc_rings:
            self._ohlc_rings[symbol] = []
        self._ohlc_rings[symbol].append(ohlc_ring)
        op._connect_live(ohlc_ring)



def _warm(self, historical_ticks: dict):
    """
    Warm indicators with historical data.

    Pre-computes initial indicator values using historical data before
    the live loop starts.

    Args:
        historical_ticks: Dict with historical data (same format as _build_rings).
    """
    for defn in self.strategy._indicator_defs:
        source: ColumnRef = defn["source"]
        sources = defn.get("sources", [source])
        indicator: Indicator = defn["indicator"]
        symbol = source.symbol
        multi_source = defn.get("multi_source", False)
        multi_band  = defn.get("multi_band", False)

        if isinstance(source.proxy, TickProxy):
            ring = self._tick_rings.get(symbol)

            if multi_source:
                col_srcs = [_TICK_COL[f._col_name] for f in sources]
            else:
                col_srcs = [_TICK_COL[source._col_name]]
            col_src = col_srcs[0]

            if multi_band:
                col_inds = [ring.col_index[n] for n in defn["col_names"]] if ring else []
                col_ind  = col_inds[0] if col_inds else None
            else:
                col_ind = ring.col_index[defn["col_name"]] if ring else None
                col_inds = [col_ind] if col_ind is not None else []

            hist = historical_ticks.get(symbol)
            if hist is not None and ring is not None and len(hist) > 0:
                table = hist.astype(np.float64)
                if multi_source:
                    source_arr = table[:, col_srcs]
                else:
                    source_arr = table[:, col_src]
                bulk_vals = indicator.calculate(source_arr)
                W = ring._W
                start = max(0, len(table) - W)
                if multi_band:
                    for k, ci in enumerate(col_inds):
                        for i_ring, i_hist in enumerate(range(start, len(table))):
                            pos = i_ring % W
                            v = float(bulk_vals[k, i_hist])
                            ring._buf[pos, ci] = v
                            ring._buf[pos + W, ci] = v
                else:
                    for i_ring, i_hist in enumerate(range(start, len(table))):
                        pos = i_ring % W
                        v = bulk_vals[i_hist]
                        ring._buf[pos, col_ind] = v
                        ring._buf[pos + W, col_ind] = v

            self._ind_state.append({
                "indicator":    indicator,
                "col_src":      col_src,
                "col_srcs":     col_srcs,
                "col_ind":      col_ind,
                "col_inds":     col_inds,
                "symbol":       symbol,
                "is_ohlc":      False,
                "multi_source": multi_source,
                "multi_band":  multi_band,
                "ring":         ring,
            })

            if ring is not None:
                if multi_band:
                    for k, ci in enumerate(col_inds):
                        view = WindowView(
                            col_idx=ci,
                            size=source.proxy._window_size,
                            tick_ring=ring,
                        )
                        defn["proxy"]._connect_band(k, view)
                else:
                    view = WindowView(
                        col_idx=col_ind,
                        size=source.proxy._window_size,
                        tick_ring=ring,
                    )
                    defn["proxy"]._connect(view)

        else:
            ohlc_ring = defn["ohlc_proxy"]._ohlc_ring

            if multi_source:
                col_srcs = [_OHLC_COL[f._col_name] for f in sources]
            else:
                col_srcs = [_OHLC_COL[source._col_name]]
            col_src = col_srcs[0]

            if multi_band:
                col_inds = [ohlc_ring.col_index[n] for n in defn["col_names"]]
                col_ind  = col_inds[0]
            else:
                col_ind  = ohlc_ring.col_index[defn["col_name"]]
                col_inds = [col_ind]

            hist_ticks  = historical_ticks.get(symbol)
            hist_klines = historical_ticks.get((symbol, defn["ohlc_proxy"].interval_ms))

            if hist_ticks is not None or hist_klines is not None:
                W = ohlc_ring._W
                if multi_source:
                    closed = np.column_stack([
                        ohlc_ring.closed_window(c, W) for c in col_srcs
                    ])
                else:
                    closed = ohlc_ring.closed_window(col_src, W)
                if len(closed) > 0:
                    bulk_vals = indicator.calculate(closed)
                    if multi_band:
                        for k, ci in enumerate(col_inds):
                            n = bulk_vals.shape[1] if bulk_vals.ndim > 1 else len(bulk_vals)
                            for i in range(n):
                                pos = i % W
                                v = float(bulk_vals[k, i]) if bulk_vals.ndim > 1 else float(bulk_vals[i])
                                ohlc_ring._buf[pos, ci] = v
                                ohlc_ring._buf[pos + W, ci] = v
                    else:
                        n = len(bulk_vals)
                        for i in range(n):
                            pos = i % W
                            v = bulk_vals[i]
                            ohlc_ring._buf[pos, col_ind] = v
                            ohlc_ring._buf[pos + W, col_ind] = v

            self._ind_state.append({
                "indicator":    indicator,
                "col_src":      col_src,
                "col_srcs":     col_srcs,
                "col_ind":      col_ind,
                "col_inds":     col_inds,
                "symbol":       symbol,
                "is_ohlc":      True,
                "multi_source": multi_source,
                "multi_band":  multi_band,
                "ohlc_ring":    ohlc_ring,
            })

            if multi_band:
                for k, ci in enumerate(col_inds):
                    view = WindowView(
                        col_idx=ci,
                        size=source.proxy._window_size,
                        ohlc_ring=ohlc_ring,
                    )
                    defn["proxy"]._connect_band(k, view)
            else:
                view = WindowView(
                    col_idx=col_ind,
                    size=source.proxy._window_size,
                    ohlc_ring=ohlc_ring,
                )
                defn["proxy"]._connect(view)

            if getattr(indicator, "use_partial", True) and ohlc_ring._ts_current_candle >= 0:
                # Partial bar uses a min_periods causal window (cheap, no NaN);
                # closed bars use the full ring window.
                L = getattr(indicator, "min_periods", getattr(indicator, "length", 1))
                if multi_source:
                    vals = np.column_stack([
                        ohlc_ring.closed_window(c, L - 1) for c in col_srcs
                    ]) if L > 1 else np.empty((0, len(col_srcs)), dtype=np.float64)
                    partial_row = ohlc_ring.current_partial_candle[col_srcs]
                    window_p = (
                        np.vstack([vals, partial_row])
                        if len(vals) > 0
                        else partial_row[np.newaxis, :]
                    )
                else:
                    vals = ohlc_ring.closed_window(col_src, L - 1)
                    close_p = float(ohlc_ring.current_partial_candle[_OHLC_COL["close"]])
                    window_p = np.append(vals, close_p) if len(vals) > 0 else np.array([close_p])

                n_pts = window_p.shape[0] if window_p.ndim > 1 else len(window_p)
                if n_pts >= 1:
                    result_p = indicator.calculate(window_p)
                    W           = ohlc_ring._W
                    # Partial stored in the shadow slot (2W-1) only, never the
                    # primary slot W-1 (a real closed-candle position read by
                    # closed_window). See WindowView._ohlc_live_view.
                    p_partial   = 2 * W - 1
                    if multi_band:
                        for k, ci in enumerate(col_inds):
                            res_k = float(result_p[k, -1]) if result_p.ndim > 1 else float(result_p[-1])
                            ohlc_ring._buf[p_partial, ci] = res_k
                    else:
                        res = float(result_p[-1]) if result_p.ndim == 1 else float(result_p[0, -1])
                        ohlc_ring._buf[p_partial, col_ind] = res



def _resync_chart_sources(self) -> None:
    """
    Notify chart (if attached) to re-sync ColumnDataSources with current ring state.

    No-op if no chart is attached.
    """
    if self._chart is None:
        return
    if self._chart._updater is None:
        return
    _updater = self._chart._updater
    def _cb():
        try:
            _updater.repopulate_after_topup()
        except Exception as _exc:
            warnings.warn(f"LiveEngine resync chart error: {_exc}", stacklevel=2)

    # repopulate_after_topup() mutates Bokeh ColumnDataSources, which is only
    # legal while holding the document lock. doc.add_next_tick_callback runs the
    # callback under that lock (same path the live updater uses); io_loop.add_
    # callback does NOT, so it raises "_pending_writes should be non-None ..."
    # and aborts the repopulate half-way (leaving the OHLC source without its
    # fp_* columns, which then breaks the next stream). Prefer the document.
    _doc = getattr(self._chart, "_doc", None)
    if _doc is not None:
        try:
            _doc.add_next_tick_callback(_cb)
            return
        except Exception:
            pass
    if self._chart._io_loop is not None:
        self._chart._io_loop.add_callback(_cb)


def _topup_ohlc_rings(self) -> None:
    """
    Fill the gap between the last closed kline in the ring and the current partial candle.

    Called in prepare() and at the start of _loop_tick(). Ensures OHLC rings
    are fully populated with historical candles before live trading starts.
    """
    now_ms = int(time.time() * 1000)

    _auto = getattr(self, "_auto_ohlc_proxies", ())

    for op in self.strategy._ohlc_proxies:
        # Chart-only proxies auto-injected for tick strategies are fed purely
        # from the tick stream; skip the REST top-up so a headless tick
        # session triggers no network fetch.
        if op in _auto:
            continue
        ring = op._ohlc_ring
        if ring is None:
            continue
        if ring._n_closed == 0:
            continue

        p_last = (ring._head - 1) % ring._W
        last_ts_ring = int(ring._buf[p_last, ring.col_index["ts"]])
        if last_ts_ring <= 0:
            continue

        gap_ms       = max(0, now_ms - last_ts_ring)
        n_needed = max(int(gap_ms // op.interval_ms) + 2, 2)

        try:
            klines = self.sesh._fetch_klines_history(
                op.symbol, op.interval_ms, n_needed
            )
        except Exception as exc:
            warnings.warn(
                f"LiveEngine top-up: could not fetch klines for "
                f"'{op.symbol}' interval={op.interval_ms}ms: {exc}",
                stacklevel=2,
            )
            continue

        if klines is None or len(klines) == 0:
            continue

        klines = np.asarray(klines, dtype=np.float64)
        mask_new = klines[:, 0] > last_ts_ring
        new = klines[mask_new]

        if len(new) == 0:
            continue

        ts_candle_in_progress = (now_ms // op.interval_ms) * op.interval_ms
        n_closed_new = int(np.sum(new[:, 0] < ts_candle_in_progress))
        n_partial_new   = int(np.sum(new[:, 0] >= ts_candle_in_progress))

        warnings.warn(
            f"LiveEngine top-up '{op.symbol}' ({op.interval_ms}ms): "
            f"{n_closed_new} closed candles + {n_partial_new} partial "
            f"(gap: {gap_ms // 1000}s, requested: {n_needed})"
            f" | ts=[{int(new[0, 0])}, {int(new[-1, 0])}]",
            stacklevel=2,
        )

        for row in new:
            ts_kline = int(row[0])
            if ts_kline < ts_candle_in_progress:
                ring.load_kline(
                    ts     = ts_kline,
                    open_  = float(row[1]),
                    high   = float(row[2]),
                    low    = float(row[3]),
                    close  = float(row[4]),
                    volume = float(row[5]),
                )
            else:
                ring.process_tick(
                    timestamp_ms = ts_kline,
                    price        = float(row[4]),
                    volume       = float(row[5]),
                )
                col = ring.col_index
                ring._partial_candle[col["open"]]   = float(row[1])
                ring._partial_candle[col["high"]]   = float(row[2])
                ring._partial_candle[col["low"]]    = float(row[3])

        _rewarm_ohlc_indicators(self, op, ring)


def _rewarm_ohlc_indicators(self, ohlc_proxy, ring) -> None:
    """
    Recalculate OHLC indicators after a top-up.

    Used after filling gap between historical and live candles.

    Args:
        ohlc_proxy: OhlcProxy instance.
        ring: OhlcRing that was topped up.
    """
    W = ring._W

    for state in self._ind_state:
        if not state["is_ohlc"]:
            continue
        if state["ohlc_ring"] is not ring:
            continue

        ind         = state["indicator"]
        multi_band = state.get("multi_band", False)
        multi_source = state.get("multi_source", False)

        n_closed = min(ring._n_closed, W)
        if n_closed == 0:
            continue

        if multi_source:
            closed = np.column_stack([
                ring.closed_window(c, n_closed)
                for c in state["col_srcs"]
            ])
        else:
            closed = ring.closed_window(state["col_src"], n_closed)

        if (closed.shape[0] if closed.ndim > 1 else len(closed)) == 0:
            continue

        bulk_vals = ind.calculate(closed)
        n_vals = bulk_vals.shape[1] if (multi_band and bulk_vals.ndim > 1) else len(bulk_vals)

        for i in range(n_vals):
            p = (ring._head - n_vals + i) % ring._W
            if multi_band:
                for k, ci in enumerate(state["col_inds"]):
                    v = float(bulk_vals[k, i]) if bulk_vals.ndim > 1 else float(bulk_vals[i])
                    ring._buf[p,     ci] = v
                    ring._buf[p + W, ci] = v
            else:
                ci = state["col_ind"]
                v  = float(bulk_vals[i]) if bulk_vals.ndim == 1 else float(bulk_vals[0, i])
                ring._buf[p,     ci] = v
                ring._buf[p + W, ci] = v

        if not getattr(ind, "use_partial", True):
            continue
        if ring._ts_current_candle < 0:
            continue

        # Partial bar uses a min_periods causal window (cheap, no NaN); closed
        # bars are recomputed over the full ring window above.
        L = getattr(ind, "min_periods", getattr(ind, "length", 1))

        if multi_source:
            vals_prev = np.column_stack([
                ring.closed_window(c, L - 1) for c in state["col_srcs"]
            ]) if L > 1 else np.empty((0, len(state["col_srcs"])), dtype=np.float64)
            partial_row = ring.current_partial_candle[state["col_srcs"]]
            window_p = (
                np.vstack([vals_prev, partial_row])
                if len(vals_prev) > 0
                else partial_row[np.newaxis, :]
            )
        else:
            vals_prev = ring.closed_window(state["col_src"], L - 1)
            close_p   = float(ring.current_partial_candle[_OHLC_COL["close"]])
            window_p = (
                np.append(vals_prev, close_p)
                if len(vals_prev) > 0
                else np.array([close_p])
            )

        n_pts = window_p.shape[0] if window_p.ndim > 1 else len(window_p)
        if n_pts < 1:
            continue

        result_p = ind.calculate(window_p)
        # Partial stored in the shadow slot (2W-1) only, never the primary slot
        # W-1 (a real closed-candle position read by closed_window). See
        # WindowView._ohlc_live_view.
        p_partial   = 2 * W - 1

        if multi_band:
            for k, ci in enumerate(state["col_inds"]):
                res_k = (
                    float(result_p[k, -1])
                    if result_p.ndim > 1
                    else float(result_p[-1])
                )
                ring._buf[p_partial, ci] = res_k
        else:
            ci  = state["col_ind"]
            res = (
                float(result_p[-1])
                if result_p.ndim == 1
                else float(result_p[0, -1])
            )
            ring._buf[p_partial, ci] = res



def _build_fp_rings(self, historical_ticks: dict):
    """
    Build footprint rings from declared proxies.

    Creates footprint rings and pre-loads them with historical tick data.

    Args:
        historical_ticks: Dict with historical tick data.
    """
    from tradetropy.models.footprint import connect_fp_proxies_live

    rings_by_id = connect_fp_proxies_live(self.strategy)

    for fp_proxy in self.strategy._fp_proxies:
        symbol = fp_proxy.symbol
        ring = rings_by_id[id(fp_proxy)]

        hist = historical_ticks.get(symbol)
        if hist is not None and len(hist) > 0:
            table = hist.astype(np.float64)
            for row in table:
                ring.process_tick(
                    timestamp_ms=int(row[_TICK_COL["ts"]]),
                    price=float(row[_TICK_COL["price"]]),
                    vol=float(row[_TICK_COL["volume"]]),
                    bid=float(row[_TICK_COL["bid"]]),
                    ask=float(row[_TICK_COL["ask"]]),
                    flag=float(row[_TICK_COL["flags"]]),
                )

        if symbol not in self._fp_rings:
            self._fp_rings[symbol] = []
        self._fp_rings[symbol].append(ring)


def _build_pattern_stores_live(self, historical_ticks: dict) -> None:
    """
    Build PatternStores from historical data.

    Creates pattern stores for all pattern matchers and pre-loads them
    with pivot sequences from historical candles.

    Args:
        historical_ticks: Dict with historical data.
    """
    if not getattr(self.strategy, "_pattern_matcher_defs", None):
        return
    from tradetropy.ta.pattern.sequence import FrozenPivotSequence
    from tradetropy.ta.pattern.store    import PatternStore
    from tradetropy.ta.pattern._builder import iter_pattern_matcher_defs

    for pm_def, _, ohlc_proxy, symbol, base_cols, decorator_cols, tag_decoders \
        in iter_pattern_matcher_defs(self.strategy, self._get_ind_def_for_proxy):

        ohlc_ring = ohlc_proxy._ohlc_ring
        col       = ohlc_ring.col_index
        W         = ohlc_ring._W

        if not all(c in col for c in base_cols):
            warnings.warn(
                f"PatternMatcher: base columns {base_cols} not found "
                f"in ohlc_ring for '{symbol}'. Make sure to declare "
                f"ConfirmedPivot with add_indicator() before add_pattern_matcher()."
            )
            continue

        n = min(ohlc_ring._n_closed, W)
        ph    = ohlc_ring.closed_window(col[base_cols[0]], n)
        pl    = ohlc_ring.closed_window(col[base_cols[1]], n)
        ph_ts = ohlc_ring.closed_window(col[base_cols[2]], n)
        pl_ts = ohlc_ring.closed_window(col[base_cols[3]], n)

        decorator_arrays: dict[str, np.ndarray] = {}
        for tag_name, dec_col in decorator_cols.items():
            if dec_col in col:
                decorator_arrays[tag_name] = ohlc_ring.closed_window(col[dec_col], n)

        sequence = FrozenPivotSequence.from_arrays(
            ph_array         = ph,
            pl_array         = pl,
            ph_ts_array      = ph_ts,
            pl_ts_array      = pl_ts,
            decorator_arrays = decorator_arrays,
            tag_decoders     = tag_decoders,
        )
        store = PatternStore(sequence, pm_def.pattern)
        pm_def.proxy._connect_live(store)


def _build_book_rings(self, historical) -> None:
    """
    Build a LiveBookRing for each subscribed order book and attach it to its
    OrderbookProxy.

    Book rings start empty and stale: the live feed warms them with the first
    OrderbookSnapshot (and replay re-feeds a recorded book in Task 10). There
    is no REST warmup for depth, so ``historical`` is currently unused here.

    Args:
        historical: Prepared history dict (unused for book rings).
    """
    self._book_rings = {}
    for bp in getattr(self.strategy, "_book_proxies", []):
        ring = LiveBookRing(window_size=bp.window_size, levels=bp.depth)
        bp._book_ring = ring
        self._book_rings.setdefault(bp.symbol, []).append(ring)


def _build_mbo_rings(self, historical) -> None:
    """
    Build an MboRing for each subscribed L3 / MBO stream and attach it to its
    MboProxy.

    Rings start empty; the live feed (or replay of a recorded MBO log) warms
    them. ``historical`` is unused here (no REST warmup for L3).

    Args:
        historical: Prepared history dict (unused for MBO rings).
    """
    self._mbo_rings = {}
    for mp in getattr(self.strategy, "_mbo_proxies", []):
        ring = MboRing(window_size=mp.window_size)
        mp._mbo_ring = ring
        self._mbo_rings.setdefault(mp.symbol, []).append(ring)

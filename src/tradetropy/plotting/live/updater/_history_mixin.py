from __future__ import annotations

import warnings

import numpy as np

from tradetropy.core.constants import _OHLC_COL, N_OHLC_COLS


class HistoryPopulateMixin:
    def repopulate_after_topup(self) -> None:
        if not self._history_loaded:
            return
        self._populate_ohlc_history()
        self._populate_indicator_history()
        self._populate_fp_history()
        self._populate_vp_history()
        if self._trades_ref is not None:
            self._populate_trades_history(self._trades_ref)
            self._reset_open_positions_overlay(self._trades_ref)
        if self._drawdown_source is not None and self._equity_ref is not None:
            self._update_drawdown()
        if self._pl_source is not None and self._trades_ref is not None:
            self._update_pl()
        # Rewind-aware cleanup: after an in-place rewind (Restart / Step Back)
        # the engine re-fed the dataset causally, but some chart sources are
        # repopulated conditionally (e.g. indicator bands only replace their
        # data when the rewound window has valid values), so stale FUTURE glyphs
        # could survive and create an indicator look-ahead bias. Strip every
        # time-keyed source down to ts <= the current engine timestamp so going
        # backward in time cleanly removes any plot ahead of the clock.
        self.truncate_all_sources_to(self._engine_ts())
        self._autoscale_ohlc_yrange()

    def _engine_ts(self) -> int:
        """
        Current engine timestamp (ms) used as the visual boundary for cleanup.

        Uses the primary OHLC ring's partial-candle timestamp (the clock the
        engine is at), falling back to the last closed candle, then 0. Anything
        plotted strictly after this instant is future data and must be stripped.
        """
        ts = 0
        for ref in self._ohlc_refs:
            ring = ref.proxy._ohlc_ring
            if ring is None:
                continue
            if ring._ts_current_candle >= 0:
                ts = max(ts, int(ring._ts_current_candle))
            elif ring._n_closed > 0:
                cand = self._ts_last_closed(ref.proxy)
                if cand:
                    ts = max(ts, int(cand))
        return ts

    def truncate_all_sources_to(self, ts_ms: int) -> None:
        """
        Remove rows newer than ``ts_ms`` from every time-keyed chart source.

        Purely subtractive and structure-preserving (keeps all columns, only
        drops future rows), so it never breaks the stream invariants (e.g. the
        OHLC source keeps its fp_* columns and uniform lengths). Enforces the
        strict visual boundary that prevents indicator look-ahead on rewind.

        Args:
            ts_ms (int): Inclusive upper bound; rows with time > ts_ms are cut.
        """
        if ts_ms <= 0:
            return

        sources = []
        for ref in self._ohlc_refs:
            sources.append(ref.source)
        for ref in self._indicator_refs:
            for band in ref.bands:
                sources.append(band.source)
        if self._equity_ref is not None:
            sources.append(self._equity_ref.source)
        if self._trailing_dd_source is not None:
            sources.append(self._trailing_dd_source)
        if self._drawdown_source is not None:
            sources.append(self._drawdown_source)
        if self._trades_ref is not None:
            sources.append(self._trades_ref.source)
            sources.append(getattr(self._trades_ref, "source_markers", None))
            sources.append(getattr(self._trades_ref, "source_open", None))
        for ref in self._fp_refs:
            sources.append(getattr(ref, "source_bid", None))
            sources.append(getattr(ref, "source_ask", None))

        for src in sources:
            if src is not None:
                self._truncate_source(src, ts_ms)

    @staticmethod
    def _truncate_source(src, ts_ms: int) -> None:
        """
        Filter a single ColumnDataSource to rows whose time column <= ts_ms.

        Picks the first available time column among 'ts', 'entry_ts', 'x'
        (datetime64[ms]). Columns of matching length are filtered with the same
        boolean mask; columns of a different length (defensive) are left intact.
        """
        data = src.data
        time_key = None
        for key in ("ts", "entry_ts", "x"):
            col = data.get(key)
            if col is not None and len(col) > 0:
                time_key = key
                break
        if time_key is None:
            return

        times = np.asarray(data[time_key])
        try:
            t_ms = times.astype("datetime64[ms]").astype(np.int64)
        except (TypeError, ValueError):
            try:
                t_ms = times.astype(np.int64)
            except Exception:
                return

        mask = t_ms <= int(ts_ms)
        if bool(mask.all()):
            return

        n = len(times)
        new_data = {}
        for col, vals in data.items():
            if len(vals) == n:
                if isinstance(vals, np.ndarray):
                    new_data[col] = vals[mask]
                else:
                    new_data[col] = [v for v, keep in zip(vals, mask) if keep]
            else:
                new_data[col] = vals
        src.data = new_data

    def populate_history(self, max_age_minutes: int = 30) -> None:
        if max_age_minutes > 0 and self._ring_data_is_stale(max_age_minutes):
            warnings.warn(
                f"LiveChart populate_history skipped: ring data is older than "
                f"{max_age_minutes} min (probably historical klines). "
                f"The chart will fill with real ticks.",
                stacklevel=2,
            )
            if self._trades_ref is not None:
                self._populate_trades_history(self._trades_ref)
            self._autoscale_ohlc_yrange()
            self._history_loaded = True
            return

        self._populate_ohlc_history()
        self._populate_indicator_history()
        self._populate_fp_history()
        self._populate_vp_history()
        if self._trades_ref is not None:
            self._populate_trades_history(self._trades_ref)
        self._autoscale_ohlc_yrange()
        self._history_loaded = True

    def _autoscale_ohlc_yrange(self) -> None:
        if self._fig_ohlc is None or not self._ohlc_refs:
            return

        ring = self._ohlc_refs[0].proxy._ohlc_ring
        if ring is None or ring._n_closed == 0:
            return

        n = min(ring._n_closed, ring._W)
        high_arr = ring.closed_window(_OHLC_COL["high"], n)
        low_arr  = ring.closed_window(_OHLC_COL["low"],  n)

        valid_high = high_arr[np.isfinite(high_arr)]
        valid_low  = low_arr[np.isfinite(low_arr)]

        if len(valid_high) == 0 or len(valid_low) == 0:
            return

        hi  = float(valid_high.max())
        lo  = float(valid_low.min())

        # Same padding as the JS steady-state autoscale (5% + half footprint
        # tick) so the initial range matches the tracking range and there is no
        # jump on the first real tick.
        fp_half = 0.0
        if self._fp_refs:
            _ts = self._fp_refs[0].fp_proxy.config.tick_size
            if _ts:
                fp_half = float(_ts) / 2.0
        lo -= fp_half
        hi += fp_half

        pad = (hi - lo) * 0.05 or abs(hi) * 0.005 or 1.0

        self._fig_ohlc.y_range.start = lo - pad
        self._fig_ohlc.y_range.end   = hi + pad

        self._sync_xrange_to_source()

    def _sync_xrange_to_source(self) -> None:
        if self._fig_ohlc is None or not self._ohlc_refs:
            return

        src = self._ohlc_refs[0].source
        ts_arr = src.data.get("ts", [])
        if len(ts_arr) == 0:
            return

        interval_ms = self._ohlc_refs[0].interval_ms

        try:
            ts_ms = np.asarray(ts_arr, dtype="datetime64[ms]").astype(np.int64)
            ts_valid = ts_ms[ts_ms > 0]
            if len(ts_valid) == 0:
                return

            ts_min = int(ts_valid[0])
            ts_max = int(ts_valid[-1])

            # Reserve on the right at least the follow padding (which already
            # accounts for the VPVR histogram) so the Volume Profile is not
            # clipped during initial load, before the first live tick.
            right_candles = 10
            if self._navigation is not None:
                right_candles = max(right_candles, self._navigation.right_padding_candles)
            x_pad_r = interval_ms * right_candles
            x_pad_l = interval_ms * 5
            new_start = ts_min - x_pad_l
            new_end   = ts_max + x_pad_r

            if self._navigation is not None:
                self._navigation._updating_range = True
            try:
                self._fig_ohlc.x_range.start = new_start
                self._fig_ohlc.x_range.end   = new_end
            finally:
                if self._navigation is not None:
                    self._navigation._updating_range = False

            if self._navigation is not None:
                window_ms = (ts_max - ts_min) + x_pad_l
                if window_ms > interval_ms * 10:
                    self._navigation.follow_window_ms = window_ms
                if self._navigation._last_ts_ms == 0 and ts_max > 0:
                    self._navigation._last_ts_ms = ts_max

        except Exception:
            pass

    def _ring_data_is_stale(self, max_age_minutes: int) -> bool:
        import time as _time

        now_ms = int(_time.time() * 1000)
        threshold_ms = max_age_minutes * 60 * 1000

        for ref in self._ohlc_refs:
            ring = ref.proxy._ohlc_ring
            if ring is None or ring._n_closed == 0:
                return False
            p_last = (ring._head - 1) % ring._W
            last_ts = int(ring._buf[p_last, _OHLC_COL["ts"]])
            if last_ts <= 0:
                return False
            age_ms = now_ms - last_ts
            if age_ms > threshold_ms:
                return True
        return False

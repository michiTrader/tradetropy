from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from tradetropy.core.constants import _OHLC_COL, N_OHLC_COLS
from tradetropy.plotting.live.updater._footprint_helpers import _build_fp_rows_for_candle

if TYPE_CHECKING:
    from tradetropy.plotting.live.updater._refs import _FootprintSourceRef


class FootprintUpdateMixin:
    def _update_footprint(self, ref: "_FootprintSourceRef") -> None:
        """
        Finalize every footprint candle closed since the last chart frame.

        Fast playback (or a speed-scaled Step) coalesces many closed candles
        into a single update(bar_closed=True) frame (the IOLoop rate-limits the
        chart callback). The OHLC source backfills those candles via
        _backfill_ohlc_gap; the footprint must do the same or it desyncs - only
        the latest closed candle would be drawn and the stale partial rows from
        the previous frame would be left overlapping the candles. This streams
        every closed footprint candle whose timestamp is newer than the last one
        already drawn, regardless of how many candles advanced.
        """
        fp_proxy    = ref.fp_proxy
        ring        = fp_proxy._ring
        if ring is None or ring._n_closed == 0:
            return

        ohlc_ring = ref.ohlc_proxy._ohlc_ring
        if ohlc_ring is None or ohlc_ring._n_closed == 0:
            return

        # Drop the previous frame's partial rows: the candle they belonged to has
        # now closed and is re-added below as a finalized candle (avoids the
        # partial/closed duplicate that shows up as overlapping numbers).
        self._strip_partial_fp_rows(ref)

        n = min(ring._n_closed, ring._W,
                ohlc_ring._n_closed, ohlc_ring._W, ref._n_candles_max)
        if n == 0:
            return

        ts_win   = ohlc_ring.closed_window(_OHLC_COL["ts"],   n)
        high_win = ohlc_ring.closed_window(_OHLC_COL["high"], n)
        low_win  = ohlc_ring.closed_window(_OHLC_COL["low"],  n)

        last_ts = ref._last_fp_ts_ms
        # Window is chronological (0 = oldest). Footprint closed-candle index from
        # the end: window i -> ring.closed_candle(n - 1 - i) (lockstep with OHLC).
        for i in range(len(ts_win)):
            ts_ms = int(ts_win[i])
            if ts_ms <= 0 or ts_ms <= last_ts:
                continue
            vela = ring.closed_candle(n - 1 - i)
            if vela is None:
                continue
            price_range = None
            if vela.levels == 0:
                ohlc_high = float(high_win[i])
                ohlc_low  = float(low_win[i])
                if np.isfinite(ohlc_high) and np.isfinite(ohlc_low):
                    price_range = (ohlc_low, ohlc_high)
            self._append_fp_closed_candle(ref, vela, ts_ms, price_range)
            ref._last_fp_ts_ms = ts_ms

        ref._n_partial_bid_rows = 0
        ref._n_partial_ask_rows = 0

    def _strip_partial_fp_rows(self, ref: "_FootprintSourceRef") -> None:
        """Remove the trailing partial-candle rows from both fp sources."""
        if ref._n_partial_bid_rows > 0:
            n_total = len(ref.source_bid.data.get("x", []))
            keep = max(0, n_total - ref._n_partial_bid_rows)
            ref.source_bid.data = {
                k: np.asarray(v)[:keep] for k, v in ref.source_bid.data.items()
            }
        if ref._n_partial_ask_rows > 0:
            n_total = len(ref.source_ask.data.get("x", []))
            keep = max(0, n_total - ref._n_partial_ask_rows)
            ref.source_ask.data = {
                k: np.asarray(v)[:keep] for k, v in ref.source_ask.data.items()
            }
        ref._n_partial_bid_rows = 0
        ref._n_partial_ask_rows = 0

    def _append_fp_closed_candle(
        self, ref: "_FootprintSourceRef", vela, ts_ms: int, price_range
    ) -> None:
        """Build and stream the footprint rows for one closed candle."""
        ring        = ref.fp_proxy._ring
        interval_ms = ref.interval_ms
        tick_size   = float(ring.config.tick_size or 1.0)

        bid_cols, ask_cols = _build_fp_rows_for_candle(
            vela, ts_ms, interval_ms, ref.theme,
            tick_size=tick_size, price_range=price_range
        )
        n_bid_rows = len(bid_cols.get("x", []))
        n_ask_rows = len(ask_cols.get("x", []))

        if vela.levels >= 2:
            candle_prices = vela.price_levels[:, 0]
            diffs = np.diff(np.sort(candle_prices))
            diffs_pos = diffs[diffs > 0]
            vela_tick_size = float(np.min(diffs_pos)) if len(diffs_pos) > 0 else tick_size
        else:
            vela_tick_size = tick_size
        if vela.levels > 0:
            bid_cols["cell_h"] = np.where(bid_cols["cell_h"] > 0, vela_tick_size, 0).astype(np.float64)
            ask_cols["cell_h"] = np.where(ask_cols["cell_h"] > 0, vela_tick_size, 0).astype(np.float64)

        if len(ref._candle_bid_row_counts) >= ref._n_candles_max:
            n_drop_bid = ref._candle_bid_row_counts.pop(0)
            n_drop_ask = ref._candle_ask_row_counts.pop(0)
            new_bid_data = {k: np.asarray(v)[n_drop_bid:] for k, v in ref.source_bid.data.items()}
            new_ask_data = {k: np.asarray(v)[n_drop_ask:] for k, v in ref.source_ask.data.items()}
            ref.source_bid.data = new_bid_data
            ref.source_ask.data = new_ask_data

        if n_bid_rows > 0:
            ref.source_bid.stream(bid_cols)
        if n_ask_rows > 0:
            ref.source_ask.stream(ask_cols)
        ref._candle_bid_row_counts.append(n_bid_rows)
        ref._candle_ask_row_counts.append(n_ask_rows)

    def _update_footprint_partial(self, ref: "_FootprintSourceRef") -> None:
        ring = ref.fp_proxy._ring
        if ring is None or ring._ts_current_candle < 0:
            return

        vela = ring.partial_candle()
        if vela is None or vela.levels == 0:
            if ref._n_partial_bid_rows > 0 or ref._n_partial_ask_rows > 0:
                if ref._n_partial_bid_rows > 0:
                    n_total = len(ref.source_bid.data.get("x", []))
                    new_data = {k: np.asarray(v)[:n_total - ref._n_partial_bid_rows]
                                for k, v in ref.source_bid.data.items()}
                    ref.source_bid.data = new_data
                if ref._n_partial_ask_rows > 0:
                    n_total = len(ref.source_ask.data.get("x", []))
                    new_data = {k: np.asarray(v)[:n_total - ref._n_partial_ask_rows]
                                for k, v in ref.source_ask.data.items()}
                    ref.source_ask.data = new_data
                ref._n_partial_bid_rows = 0
                ref._n_partial_ask_rows = 0
            return

        ts_ms = int(ring._ts_current_candle)
        interval_ms = ref.interval_ms
        tick_size   = float(ring.config.tick_size or 1.0)

        bid_cols, ask_cols = _build_fp_rows_for_candle(
            vela, ts_ms, interval_ms, ref.theme
        )
        n_bid = len(bid_cols.get("x", []))
        n_ask = len(ask_cols.get("x", []))
        if n_bid + n_ask >= 2:
            candle_prices = vela.price_levels[:, 0]
            diffs = np.diff(np.sort(candle_prices))
            diffs_pos = diffs[diffs > 0]
            vela_tick_size = float(np.min(diffs_pos)) if len(diffs_pos) > 0 else tick_size
        else:
            vela_tick_size = tick_size
        if n_bid > 0:
            bid_cols["cell_h"] = np.where(bid_cols["cell_h"] > 0, vela_tick_size, 0).astype(np.float64)
        if n_ask > 0:
            ask_cols["cell_h"] = np.where(ask_cols["cell_h"] > 0, vela_tick_size, 0).astype(np.float64)

        def _concat_col(existing, new_vals, n_keep: int):
            if isinstance(existing, (list, tuple)):
                kept = list(existing)[:n_keep]
                added = list(new_vals) if not isinstance(new_vals, (list, tuple)) else list(new_vals)
                return kept + added
            else:
                arr_keep = np.asarray(existing)[:n_keep]
                arr_new  = np.asarray(new_vals)
                return np.concatenate([arr_keep, arr_new])

        if ref._n_partial_bid_rows > 0 or ref._n_partial_ask_rows > 0:
            n_bid_total = len(ref.source_bid.data.get("x", []))
            n_ask_total = len(ref.source_ask.data.get("x", []))
            n_keep_bid  = max(0, n_bid_total - ref._n_partial_bid_rows)
            n_keep_ask  = max(0, n_ask_total - ref._n_partial_ask_rows)

            merged_bid: dict = {}
            merged_ask: dict = {}
            for k in ref.source_bid.data:
                merged_bid[k] = _concat_col(
                    ref.source_bid.data[k], bid_cols.get(k, []), n_keep_bid
                )
                merged_ask[k] = _concat_col(
                    ref.source_ask.data[k], ask_cols.get(k, []), n_keep_ask
                )

            ref.source_bid.data = merged_bid
            ref.source_ask.data = merged_ask
        else:
            if n_bid > 0:
                ref.source_bid.stream(bid_cols)
            if n_ask > 0:
                ref.source_ask.stream(ask_cols)

        ref._n_partial_bid_rows = n_bid
        ref._n_partial_ask_rows = n_ask

    def _fp_ts_desde_ohlc(self, ref: "_FootprintSourceRef") -> "int | None":
        ohlc_ring = ref.ohlc_proxy._ohlc_ring
        if ohlc_ring is None or ohlc_ring._n_closed == 0:
            return None
        p_last = (ohlc_ring._head - 1) % ohlc_ring._W
        ts = int(ohlc_ring._buf[p_last, _OHLC_COL["ts"]])
        return ts if ts > 0 else None

    def _populate_fp_history(self) -> None:
        for ref in self._fp_refs:
            ring = ref.fp_proxy._ring
            if ring is None or ring._n_closed == 0:
                continue

            ref._candle_bid_row_counts.clear()
            ref._candle_ask_row_counts.clear()
            ref._n_partial_bid_rows = 0
            ref._n_partial_ask_rows = 0
            ref._last_fp_ts_ms = -1

            _empty = {
                "x": np.array([], dtype="datetime64[ms]"),
                "y": np.array([], dtype=np.float64),
                "text": [],
                "fill": [],
                "lc":   [],
                "lw":   np.array([], dtype=np.float64),
                "tc":   [],
                "cell_w": np.array([], dtype=np.float64),
                "cell_h": np.array([], dtype=np.float64),
            }
            ref.source_bid.data = dict(_empty)
            ref.source_ask.data = dict(_empty)

            ohlc_ring = ref.ohlc_proxy._ohlc_ring
            if ohlc_ring is None or ohlc_ring._n_closed == 0:
                continue

            n_candles     = min(ring._n_closed, ring._W, ohlc_ring._n_closed, ohlc_ring._W, ref._n_candles_max)
            interval_ms = ref.interval_ms
            tick_size   = float(ring.config.tick_size or 1.0)

            ts_col   = ohlc_ring.closed_window(_OHLC_COL["ts"],   n_candles).copy()
            high_col = ohlc_ring.closed_window(_OHLC_COL["high"], n_candles).copy()
            low_col  = ohlc_ring.closed_window(_OHLC_COL["low"],  n_candles).copy()

            n_candles = len(ts_col)

            for i in range(n_candles):
                idx_from_end = n_candles - 1 - i
                vela = ring.closed_candle(idx_from_end)
                if vela is None:
                    continue

                ts_ms = int(ts_col[i]) if i < len(ts_col) else 0
                if ts_ms <= 0:
                    continue

                price_range = None
                if vela.levels == 0:
                    if i < len(high_col) and i < len(low_col):
                        ohlc_high = float(high_col[i])
                        ohlc_low  = float(low_col[i])
                        if np.isfinite(ohlc_high) and np.isfinite(ohlc_low):
                            price_range = (ohlc_low, ohlc_high)

                bid_cols, ask_cols = _build_fp_rows_for_candle(
                    vela, ts_ms, interval_ms, ref.theme,
                    tick_size=tick_size, price_range=price_range
                )
                n_bid_rows = len(bid_cols.get("x", []))
                n_ask_rows = len(ask_cols.get("x", []))

                if vela.levels >= 2:
                    candle_prices = vela.price_levels[:, 0]
                    diffs = np.diff(np.sort(candle_prices))
                    diffs_pos = diffs[diffs > 0]
                    vela_tick_size = float(np.min(diffs_pos)) if len(diffs_pos) > 0 else tick_size
                else:
                    vela_tick_size = tick_size
                if vela.levels > 0:
                    if n_bid_rows > 0:
                        bid_cols["cell_h"] = np.where(bid_cols["cell_h"] > 0, vela_tick_size, 0).astype(np.float64)
                    if n_ask_rows > 0:
                        ask_cols["cell_h"] = np.where(ask_cols["cell_h"] > 0, vela_tick_size, 0).astype(np.float64)

                if n_bid_rows > 0:
                    ref.source_bid.stream(bid_cols)
                if n_ask_rows > 0:
                    ref.source_ask.stream(ask_cols)
                ref._candle_bid_row_counts.append(n_bid_rows)
                ref._candle_ask_row_counts.append(n_ask_rows)

            if n_candles > 0:
                ref._last_fp_ts_ms = int(ts_col[n_candles - 1])

            if ref.renderers and max(len(ref._candle_bid_row_counts), len(ref._candle_ask_row_counts)) <= ref.zoom_range:
                for r in ref.renderers:
                    r.visible = True

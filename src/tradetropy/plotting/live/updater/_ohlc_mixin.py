from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from tradetropy.core.constants import _OHLC_COL, N_OHLC_COLS
from tradetropy.plotting.sources import _fp_scalars_from_candle
from tradetropy.plotting._util import _ohlc_tick_bounds

if TYPE_CHECKING:
    from tradetropy.plotting.live.updater._refs import _OhlcSourceRef

_FP_COL_NAMES = ("fp_bid", "fp_ask", "fp_delta", "fp_poc_price",
                 "fp_poc_vol", "fp_vah", "fp_val")


class OhlcUpdateMixin:
    def _candle_colors(self, ref: "_OhlcSourceRef", is_up: bool) -> tuple[str, str]:
        theme = ref.theme or {}
        c = theme.get("candle_up", "#2ECC71") if is_up else theme.get("candle_down", "#E74C3C")
        return c, c

    def _fp_scalars_actuales(self, ref: "_OhlcSourceRef") -> "tuple | None":
        """
        Footprint scalars (bid/ask/delta/poc/vah/val) of the current candle,
        for the HoverTool OHLC in live/replay. Uses the partial candle if it
        exists (the one still forming), otherwise the last closed one. Returns
        None if no footprint is linked or no data yet.
        """
        fp_proxy = getattr(ref, "fp_proxy", None)
        if fp_proxy is None:
            return None
        ring = getattr(fp_proxy, "_ring", None)
        if ring is None:
            return None
        candle = ring.partial_candle()
        if candle is None and ring._n_closed > 0:
            candle = ring.closed_candle(0)
        if candle is None:
            return None
        return _fp_scalars_from_candle(candle)

    def _update_ohlc(self) -> None:
        for ref in self._ohlc_refs:
            proxy = ref.proxy
            ring  = proxy._ohlc_ring
            if ring is None or ring._ts_current_candle < 0:
                continue

            partial   = ring.current_partial_candle
            ts_ms     = int(partial[_OHLC_COL["ts"]])
            open_val  = float(partial[_OHLC_COL["open"]])
            high_val  = float(partial[_OHLC_COL["high"]])
            low_val   = float(partial[_OHLC_COL["low"]])
            close_val = float(partial[_OHLC_COL["close"]])
            vol_val   = float(partial[_OHLC_COL["volume"]])
            # A non-finite OHLC/volume pushed via stream/patch serializes to the
            # websocket as JSON, where NaN is not compliant and raises in Bokeh's
            # send path. Skip such a (degenerate) candle rather than crash the
            # whole chart update.
            if not (np.isfinite(open_val) and np.isfinite(high_val)
                    and np.isfinite(low_val) and np.isfinite(close_val)
                    and np.isfinite(vol_val)):
                continue
            inc       = "1" if close_val >= open_val else "0"
            is_up     = close_val >= open_val
            bar_width = int(ref.interval_ms * 0.9)
            color, wcolor = self._candle_colors(ref, is_up)

            src    = ref.source
            ts_dt  = np.datetime64(ts_ms, "ms")
            ex_ts  = src.data["ts"]

            n_en_source = len(ex_ts)
            last_ts_en_source = int(np.datetime64(ex_ts[-1], "ms").astype(np.int64)) if n_en_source > 0 else -1

            if len(ex_ts) > 0:
                last_ts_ms = int(np.datetime64(ex_ts[-1], "ms").astype(np.int64))
                if ts_ms < last_ts_ms:
                    # print(f"[UPD_OHLC] SKIP -- partial ({ts_ms}) < last_source ({last_ts_ms})")
                    if self._navigation is not None:
                        self._navigation.update_view(ts_ms)
                    continue

            is_doji  = (close_val == open_val)
            bb_color = color if is_doji else None
            bb_width = 2.0 if is_doji else 1.0

            tick_left, tick_right = _ohlc_tick_bounds(np.array([ts_ms]), bar_width)

            fp_scalars = self._fp_scalars_actuales(ref)

            if len(ex_ts) > 0 and ex_ts[-1] == ts_dt:
                # print(f"[UPD_OHLC] -> PATCH  ts={ts_ms}  close={close_val:.4f}")
                last = len(ex_ts) - 1
                patch = {
                    "High":              [(last, high_val)],
                    "Low":               [(last, low_val)],
                    "Close":             [(last, close_val)],
                    "Volume":            [(last, vol_val)],
                    "inc":               [(last, inc)],
                    "candle_color":      [(last, color)],
                    "wick_color":        [(last, wcolor)],
                    "body_border_color": [(last, bb_color)],
                    "body_border_width": [(last, bb_width)],
                    "top_body":          [(last, max(open_val, close_val))],
                    "bottom_body":       [(last, min(open_val, close_val))],
                    "ts_left":           [(last, tick_left[0])],
                    "ts_right":          [(last, tick_right[0])],
                }
                if fp_scalars is not None:
                    for col, val in zip(_FP_COL_NAMES, fp_scalars):
                        if col in src.data:
                            patch[col] = [(last, float(val) if np.isfinite(val) else None)]
                src.patch(patch)
            else:
                # New-candle territory. If several candles closed since the last
                # chart frame (fast playback / Step at high speed, where the
                # IOLoop coalesces many ticks into one update), the candles
                # between the last plotted one and the current partial were
                # never streamed -> the chart shows gaps. Backfill those closed
                # candles from the ring before streaming the new partial.
                last_src_ts = int(np.datetime64(ex_ts[-1], "ms").astype(np.int64)) if len(ex_ts) > 0 else -1
                # The last plotted candle may have been streamed while still
                # forming (a partial frame) and has since closed during the
                # coalesced jump. Patch it to its final OHLC from the ring before
                # backfilling, or it stays frozen at stale intrabar values.
                self._finalize_last_source_candle(ref, ring, last_src_ts)
                self._backfill_ohlc_gap(ref, ring, last_src_ts, ts_ms)
                # print(f"[UPD_OHLC] -> STREAM ts={ts_ms}  close={close_val:.4f}  "
                #       f"(source had {n_en_source} rows, last={last_ts_en_source})")
                stream_data = {
                    "ts":                [ts_dt],
                    "Open":              [open_val],
                    "High":              [high_val],
                    "Low":               [low_val],
                    "Close":             [close_val],
                    "Volume":            [vol_val],
                    "inc":               np.array([inc],   dtype=object),
                    "bar_width":         [bar_width],
                    "candle_color":      np.array([color],  dtype=object),
                    "wick_color":        np.array([wcolor], dtype=object),
                    "body_border_color": np.array([bb_color], dtype=object),
                    "body_border_width": [bb_width],
                    "top_body":          [max(open_val, close_val)],
                    "bottom_body":       [min(open_val, close_val)],
                    "ts_left":           tick_left,
                    "ts_right":          tick_right,
                }
                if fp_scalars is not None:
                    for col, val in zip(_FP_COL_NAMES, fp_scalars):
                        stream_data[col] = np.array([float(val)], dtype=np.float64)
                # Bokeh requires streaming ALL existing columns of the source.
                # If the source has extra columns (fp_bid/fp_ask/... from
                # footprint, only for the HoverTool) that we do not produce here,
                # fill them with NaN to not break the stream. Without this,
                # _safe_update raises ValueError and the whole chart stops
                # updating when the ring starts empty.
                for col in src.data:
                    if col not in stream_data:
                        stream_data[col] = np.array([np.nan], dtype=np.float64)
                src.stream(stream_data, rollover=self._max_candles)

            self._reanchor_volume(ref)

            if self._navigation is not None:
                self._navigation.update_view(ts_ms)

    def _reanchor_volume(self, ref: "_OhlcSourceRef") -> None:
        """
        Re-anchor the volume band to the visible peak (~20% of the frame).

        The JS callback (autoscale_volume.js) only fires on pan/zoom; between
        those, live data (a growing partial candle or freshly streamed closes)
        would let a tall volume bar overflow the fixed bottom band. This mirrors
        the JS calculation from Python after each source refresh: find the max
        Volume among the currently visible candles and map its peak to ~20% of
        the pane height, so the band tracks the data at the streaming rate.

        No-op when the chart has no volume band (plot_volume off) or no data.

        Args:
            ref (_OhlcSourceRef): OHLC source reference carrying vol_range.
        """
        vol_range = getattr(ref, "vol_range", None)
        if vol_range is None or self._fig_ohlc is None:
            return
        data = ref.source.data
        vols = data.get("Volume")
        ts = data.get("ts")
        if vols is None or ts is None or len(vols) == 0:
            return
        vols = np.asarray(vols, dtype=np.float64)
        ts_ms = np.asarray(ts, dtype="datetime64[ms]").astype(np.int64)

        x_range = self._fig_ohlc.x_range
        s = x_range.start
        e = x_range.end
        finite = np.isfinite(vols)
        if s is None or e is None:
            visible = finite
        else:
            visible = finite & (ts_ms >= float(s)) & (ts_ms <= float(e))
        if not bool(np.any(visible)):
            return
        hi = float(vols[visible].max())
        if hi > 0:
            vol_range.start = 0
            vol_range.end = hi / 0.20

    def _current_ts_ms(self, ohlc_proxy: "OhlcProxy") -> int:
        ring = ohlc_proxy._ohlc_ring
        if ring is not None and ring._ts_current_candle >= 0:
            return int(ring._ts_current_candle)
        return 0

    def _backfill_ohlc_gap(self, ref: "_OhlcSourceRef", ring, last_ts_ms: int, cur_ts_ms: int) -> None:
        """
        Stream closed candles missing between the last plotted candle and now.

        Fast playback (e.g. 64x) or a speed-scaled Step closes many candles
        between two chart frames; the IOLoop coalesces those into a single
        update, so only the latest partial candle would be streamed and the
        intermediate closed candles would be lost (visible gaps). This emits
        every closed candle in the ring with last_ts_ms < ts < cur_ts_ms so the
        chart stays contiguous, regardless of how many candles advanced.

        Args:
            ref (_OhlcSourceRef): OHLC source reference.
            ring: The OHLC ring (already advanced by the engine).
            last_ts_ms (int): Timestamp of the last candle already in the source.
            cur_ts_ms (int): Timestamp of the current partial candle.
        """
        if last_ts_ms is None or last_ts_ms <= 0:
            return
        n = min(ring._n_closed, self._max_candles)
        if n == 0:
            return

        ts_col = ring.closed_window(_OHLC_COL["ts"], n)
        mask = (ts_col > last_ts_ms) & (ts_col < cur_ts_ms)
        if not bool(np.any(mask)):
            return

        open_col  = ring.closed_window(_OHLC_COL["open"],   n)[mask]
        high_col  = ring.closed_window(_OHLC_COL["high"],   n)[mask]
        low_col   = ring.closed_window(_OHLC_COL["low"],    n)[mask]
        close_col = ring.closed_window(_OHLC_COL["close"],  n)[mask]
        vol_col   = ring.closed_window(_OHLC_COL["volume"], n)[mask]
        ts_sel    = ts_col[mask]
        m         = len(ts_sel)

        theme    = ref.theme or {}
        color_up = theme.get("candle_up",   "#2ECC71")
        color_dn = theme.get("candle_down", "#E74C3C")
        is_up    = close_col >= open_col
        colors   = np.array([color_up if u else color_dn for u in is_up], dtype=object)
        is_doji  = close_col == open_col
        bb_color = np.array([c if d else None for c, d in zip(colors, is_doji)], dtype=object)
        bb_width = np.where(is_doji, 2.0, 1.0).astype(np.float64)
        bar_width = int(ref.interval_ms * 0.9)

        bf_ts_left, bf_ts_right = _ohlc_tick_bounds(ts_sel, bar_width)

        stream_data = {
            "ts":                ts_sel.astype("datetime64[ms]"),
            "Open":              open_col,
            "High":              high_col,
            "Low":               low_col,
            "Close":             close_col,
            "Volume":            vol_col,
            "inc":               np.where(is_up, "1", "0").astype(object),
            "bar_width":         np.full(m, bar_width, dtype=np.float64),
            "candle_color":      colors,
            "wick_color":        colors,
            "body_border_color": bb_color,
            "body_border_width": bb_width,
            "top_body":          np.maximum(open_col, close_col),
            "bottom_body":       np.minimum(open_col, close_col),
            "ts_left":           bf_ts_left,
            "ts_right":          bf_ts_right,
        }

        # Footprint hover columns (per candle), aligned by index-from-end like
        # _populate_ohlc_fp_cols. Window index i -> closed_candle(n-1-i).
        src = ref.source
        if any(c in src.data for c in _FP_COL_NAMES):
            fp_cols = {c: np.full(m, np.nan, dtype=np.float64) for c in _FP_COL_NAMES}
            fp_proxy = getattr(ref, "fp_proxy", None)
            fp_ring = getattr(fp_proxy, "_ring", None) if fp_proxy is not None else None
            if fp_ring is not None and fp_ring._n_closed > 0:
                idxs = np.nonzero(mask)[0]
                for j, win_i in enumerate(idxs):
                    candle = fp_ring.closed_candle(n - 1 - int(win_i))
                    if candle is None:
                        continue
                    vals = _fp_scalars_from_candle(candle)
                    for c, v in zip(_FP_COL_NAMES, vals):
                        fp_cols[c][j] = v
            for c in _FP_COL_NAMES:
                if c in src.data:
                    stream_data[c] = fp_cols[c]

        # Fill any remaining source columns with NaN so the stream is valid.
        for col in src.data:
            if col not in stream_data:
                stream_data[col] = np.full(m, np.nan, dtype=np.float64)

        src.stream(stream_data, rollover=self._max_candles)

    def _finalize_last_source_candle(self, ref: "_OhlcSourceRef", ring, last_ts_ms: int) -> None:
        """
        Patch the last plotted candle to its final OHLC from the ring.

        During fast playback a candle can be streamed as a partial (still
        forming) frame and then close before the next chart frame, which would
        leave it frozen at stale intrabar High/Low/Close/Volume. This rewrites
        that last source row from the ring's now-closed candle of the same
        timestamp so the plotted candle matches the real one.

        Args:
            ref (_OhlcSourceRef): OHLC source reference.
            ring: The OHLC ring (already advanced by the engine).
            last_ts_ms (int): Timestamp of the last candle already in the source.
        """
        if last_ts_ms is None or last_ts_ms <= 0:
            return
        src = ref.source
        ex_ts = src.data["ts"]
        if len(ex_ts) == 0:
            return

        n = min(ring._n_closed, self._max_candles)
        if n == 0:
            return
        ts_col = ring.closed_window(_OHLC_COL["ts"], n)
        idx = np.nonzero(ts_col == last_ts_ms)[0]
        if len(idx) == 0:
            return
        w = int(idx[0])

        open_val  = float(ring.closed_window(_OHLC_COL["open"],   n)[w])
        high_val  = float(ring.closed_window(_OHLC_COL["high"],   n)[w])
        low_val   = float(ring.closed_window(_OHLC_COL["low"],    n)[w])
        close_val = float(ring.closed_window(_OHLC_COL["close"],  n)[w])
        vol_val   = float(ring.closed_window(_OHLC_COL["volume"], n)[w])

        is_up = close_val >= open_val
        color, wcolor = self._candle_colors(ref, is_up)
        is_doji  = (close_val == open_val)
        bb_color = color if is_doji else None
        bb_width = 2.0 if is_doji else 1.0
        bar_width = int(ref.interval_ms * 0.9)
        tick_left, tick_right = _ohlc_tick_bounds(np.array([last_ts_ms]), bar_width)

        last = len(ex_ts) - 1
        patch = {
            "Open":              [(last, open_val)],
            "High":              [(last, high_val)],
            "Low":               [(last, low_val)],
            "Close":             [(last, close_val)],
            "Volume":            [(last, vol_val)],
            "inc":               [(last, "1" if is_up else "0")],
            "candle_color":      [(last, color)],
            "wick_color":        [(last, wcolor)],
            "body_border_color": [(last, bb_color)],
            "body_border_width": [(last, bb_width)],
            "top_body":          [(last, max(open_val, close_val))],
            "bottom_body":       [(last, min(open_val, close_val))],
            "ts_left":           [(last, tick_left[0])],
            "ts_right":          [(last, tick_right[0])],
        }
        fp_proxy = getattr(ref, "fp_proxy", None)
        fp_ring = getattr(fp_proxy, "_ring", None) if fp_proxy is not None else None
        if fp_ring is not None and fp_ring._n_closed > 0:
            # Footprint window aligns by index-from-end: window w -> closed(n-1-w).
            candle = fp_ring.closed_candle(n - 1 - w)
            if candle is not None:
                vals = _fp_scalars_from_candle(candle)
                for col, val in zip(_FP_COL_NAMES, vals):
                    if col in src.data:
                        patch[col] = [(last, float(val))]
        src.patch(patch)


    def _populate_ohlc_history(self) -> None:
        for ref in self._ohlc_refs:
            ring = ref.proxy._ohlc_ring
            if ring is None:
                continue

            n = min(ring._n_closed, self._max_candles)

            if n == 0:
                continue

            ts_col    = ring.closed_window(_OHLC_COL["ts"],     n).copy()
            open_col  = ring.closed_window(_OHLC_COL["open"],   n).copy()
            high_col  = ring.closed_window(_OHLC_COL["high"],   n).copy()
            low_col   = ring.closed_window(_OHLC_COL["low"],    n).copy()
            close_col = ring.closed_window(_OHLC_COL["close"],  n).copy()
            vol_col   = ring.closed_window(_OHLC_COL["volume"], n).copy()

            actual_n  = len(ts_col)
            if actual_n == 0:
                continue

            is_up     = close_col >= open_col
            bar_width = int(ref.interval_ms * 0.9)

            theme     = ref.theme or {}
            color_up  = theme.get("candle_up",   "#2ECC71")
            color_dn  = theme.get("candle_down",  "#E74C3C")
            candle_colors = np.array([color_up if u else color_dn for u in is_up], dtype=object)

            is_doji = close_col == open_col
            body_border_color = np.array([
                c if d else None for c, d in zip(candle_colors, is_doji)
            ], dtype=object)
            body_border_width = np.where(is_doji, 2.0, 1.0).astype(np.float64)

            hist_ts_left, hist_ts_right = _ohlc_tick_bounds(ts_col, bar_width)

            ref.source.data = dict(
                ts                = ts_col.astype("datetime64[ms]"),
                Open              = open_col,
                High              = high_col,
                Low               = low_col,
                Close             = close_col,
                Volume            = vol_col,
                inc               = np.where(is_up, "1", "0").astype(object),
                bar_width         = np.full(actual_n, bar_width, dtype=np.float64),
                candle_color      = candle_colors,
                wick_color        = candle_colors,
                body_border_color = body_border_color,
                body_border_width = body_border_width,
                top_body          = np.maximum(open_col, close_col),
                bottom_body       = np.minimum(open_col, close_col),
                ts_left           = hist_ts_left,
                ts_right          = hist_ts_right,
            )

            self._populate_ohlc_fp_cols(ref, actual_n)
            self._reanchor_volume(ref)

    def _populate_ohlc_fp_cols(self, ref: "_OhlcSourceRef", actual_n: int) -> None:
        """
        Fill the fp_* columns of the historical OHLC source (HoverTool only).

        The footprint ring and the OHLC ring close in lockstep (same ticks,
        same interval), so they align by index-from-end: row i of the window
        (0 = oldest) corresponds to closed_candle(actual_n-1-i). If no
        footprint is linked, leaves the columns as NaN.
        """
        fp_proxy = getattr(ref, "fp_proxy", None)
        if fp_proxy is None:
            return
        ring = getattr(fp_proxy, "_ring", None)
        if ring is None or ring._n_closed == 0:
            for col in _FP_COL_NAMES:
                ref.source.data[col] = np.full(actual_n, np.nan, dtype=np.float64)
            return

        cols = {col: np.full(actual_n, np.nan, dtype=np.float64) for col in _FP_COL_NAMES}
        for i in range(actual_n):
            idx_from_end = actual_n - 1 - i
            candle = ring.closed_candle(idx_from_end)
            if candle is None:
                continue
            vals = _fp_scalars_from_candle(candle)
            for col, val in zip(_FP_COL_NAMES, vals):
                cols[col][i] = val

        for col in _FP_COL_NAMES:
            ref.source.data[col] = cols[col]

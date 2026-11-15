from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from tradetropy.core.constants import _OHLC_COL
from tradetropy.plotting.live.updater._refs import _BandRef

if TYPE_CHECKING:
    from tradetropy.data.data import IndicatorProxy, MultiBandProxy
    from tradetropy.plotting.live.updater._refs import _IndicatorSourceRef


class IndicatorUpdateMixin:
    def _update_indicators(self, bar_closed: bool) -> None:
        for ref in self._indicator_refs:
            ring = ref.ohlc_proxy._ohlc_ring
            if ring is None or ring._ts_current_candle < 0:
                continue

            ts_ms_candle = int(ring._ts_current_candle)

            for band in ref.bands:
                if band.is_ts_band:
                    continue

                if band.is_stateful and not bar_closed:
                    continue

                col_idx = ring.col_index.get(band.col_name) if band.col_name else None

                if band.has_real_ts:
                    if col_idx is None or ring._n_closed == 0:
                        continue
                    self._stream_new_pivots(ref, band, ring, col_idx)
                else:
                    val = self._read_band_value(
                        ref.proxy, band.band_idx, is_stateful=False
                    )
                    if val is None:
                        continue

                    ts_dt    = np.datetime64(ts_ms_candle, "ms")
                    src      = band.source
                    existing = src.data["ts"]

                    n_ind = len(existing)
                    last_ind_ts = int(np.datetime64(existing[-1], "ms").astype(np.int64)) if n_ind > 0 else -1

                    # Fast playback can close several candles between two chart
                    # frames; backfill the indicator points for the closed
                    # candles in between so the line stays contiguous (no gaps).
                    if col_idx is not None and last_ind_ts > 0 and ts_ms_candle > last_ind_ts:
                        self._backfill_indicator_gap(
                            band, ring, col_idx, last_ind_ts, ts_ms_candle
                        )
                        existing = src.data["ts"]

                    if len(existing) > 0 and existing[-1] == ts_dt:
                        patch = {"value": [(len(existing) - 1, float(val))]}
                        if band.bar_pos_color is not None:
                            c = band.bar_pos_color if val >= 0 else band.bar_neg_color
                            patch["bar_color"] = [(len(existing) - 1, c)]
                        src.patch(patch)
                    else:
                        d = {"ts": [ts_dt], "value": [float(val)], "tag": [band.tag_label]}
                        if band.bar_pos_color is not None:
                            d["bar_color"] = [
                                band.bar_pos_color if val >= 0 else band.bar_neg_color
                            ]
                        src.stream(d, rollover=self._max_candles)

    def _bar_color_arr(self, band, values) -> "np.ndarray | None":
        """
        Per-row colors for a diverging bar band (None for non-bar bands).

        Returns an object array coloring each value by its sign with the band's
        positive/negative colors, so a vbar bound to a ``bar_color`` data column
        recolors from streamed data without any client-side transform.
        """
        if band.bar_pos_color is None:
            return None
        v = np.asarray(values, dtype=np.float64)
        return np.where(v >= 0, band.bar_pos_color, band.bar_neg_color).astype(object)

    def _backfill_indicator_gap(self, band, ring, col_idx, last_ts_ms, cur_ts_ms):
        """
        Stream indicator points for closed candles missing between two frames.

        Mirrors the OHLC gap backfill: when fast playback closes several candles
        per coalesced chart update, the intermediate indicator values would be
        skipped, leaving a broken line. Emits the indicator value for every
        closed candle with last_ts_ms < ts < cur_ts_ms.
        """
        n = min(ring._n_closed, self._max_candles)
        if n == 0:
            return
        ts_col = ring.closed_window(_OHLC_COL["ts"], n)
        vals   = ring.closed_window(col_idx, n)
        mask = (ts_col > last_ts_ms) & (ts_col < cur_ts_ms) & np.isfinite(vals)
        if not bool(np.any(mask)):
            return
        ts_sel = ts_col[mask].astype("datetime64[ms]")
        v_sel  = vals[mask]
        m = len(ts_sel)
        d = {"ts": ts_sel, "value": v_sel,
             "tag": np.full(m, band.tag_label, dtype=object)}
        bc = self._bar_color_arr(band, v_sel)
        if bc is not None:
            d["bar_color"] = bc
        band.source.stream(d, rollover=self._max_candles)

    def _stream_new_pivots(
        self,
        ref: "_IndicatorSourceRef",
        band: "_BandRef",
        ring,
        col_idx: int,
    ) -> None:
        n = min(ring._n_closed, ring._W)
        if n == 0:
            return

        prices = ring.closed_window(col_idx, n)
        closed_ts = ring.closed_window(_OHLC_COL["ts"], n)
        timestamps = self._event_timestamps(ref, band, ring, closed_ts)
        mask = np.isfinite(prices) & np.isfinite(timestamps)
        existing = band.source.data["ts"]
        if len(existing):
            last = np.datetime64(existing[-1], "ms").astype(np.int64)
            mask &= timestamps > last
        if np.any(mask):
            values = prices[mask]
            data = {
                "ts": timestamps[mask].astype("datetime64[ms]"),
                "value": values,
                "tag": np.full(len(values), band.tag_label, dtype=object),
            }
            colors = self._bar_color_arr(band, values)
            if colors is not None:
                data["bar_color"] = colors
            band.source.stream(data, rollover=self._max_candles)

    def _read_band_value(
        self,
        proxy: "IndicatorProxy | MultiBandProxy",
        band_idx: int,
        is_stateful: bool = False,
    ) -> "float | None":
        from tradetropy.data.data import MultiBandProxy as _MBP
        read_idx = -2 if is_stateful else -1
        try:
            if isinstance(proxy, _MBP):
                view = proxy[band_idx]
            else:
                view = proxy
            if view is None or len(view) < abs(read_idx):
                return None
            val = float(view[read_idx])
            return None if np.isnan(val) else val
        except Exception:
            return None

    def _populate_indicator_history(self) -> None:
        for ref in self._indicator_refs:
            ring = ref.ohlc_proxy._ohlc_ring
            if ring is None:
                continue

            n = min(ring._n_closed, self._max_candles)
            closed_ts = (
                ring.closed_window(_OHLC_COL["ts"], n) if n else np.empty(0)
            )

            for band in ref.bands:
                if band.is_ts_band:
                    continue

                col_idx = self._find_ind_col_idx(ref.proxy, band, ring)
                values = np.empty(0)
                timestamps = np.empty(0)
                if n and col_idx is not None:
                    vals = ring.closed_window(col_idx, n)
                    ts = self._event_timestamps(ref, band, ring, closed_ts)
                    valid = np.isfinite(vals) & np.isfinite(ts)
                    values, timestamps = vals[valid], ts[valid]

                if not (band.is_stateful or band.has_real_ts) and ring._ts_current_candle >= 0:
                    value = self._read_band_value(ref.proxy, band.band_idx)
                    if value is not None:
                        timestamps = np.append(timestamps, ring._ts_current_candle)
                        values = np.append(values, value)

                # Replace even empty histories: a pivot may be plotted before
                # the cursor but only become confirmed after it.
                data = {
                    "ts": timestamps.astype("datetime64[ms]"),
                    "value": values,
                    "tag": np.full(len(values), band.tag_label, dtype=object),
                }
                colors = self._bar_color_arr(band, values)
                if colors is not None:
                    data["bar_color"] = colors
                band.source.data = data

    def _event_timestamps(self, ref, band, ring, closed_ts: np.ndarray) -> np.ndarray:
        if band.has_real_ts:
            event_ts = self._ts_band_history_for_pivot(ref, band, ring, len(closed_ts))
            if event_ts is not None:
                return np.where(np.isfinite(event_ts), event_ts, closed_ts)
        return closed_ts

    def _ts_band_history_for_pivot(
        self,
        ref: "_IndicatorSourceRef",
        band: "_BandRef",
        ring,
        n: int,
    ) -> "np.ndarray | None":
        from tradetropy.data.data import MultiBandProxy as _MBP
        proxy = ref.proxy
        if not isinstance(proxy, _MBP):
            return None

        ts_band_indices = getattr(proxy, "_ts_band_indices", [])
        if not ts_band_indices or band.band_idx >= len(ts_band_indices):
            return None

        ts_band_idx  = ts_band_indices[band.band_idx]
        price_col_name = band.col_name
        if not price_col_name:
            return None

        suffix_price = f"_b{band.band_idx}"
        suffix_ts    = f"_b{ts_band_idx}"
        if not price_col_name.endswith(suffix_price):
            return None

        ts_col_name = price_col_name[:-len(suffix_price)] + suffix_ts
        ts_col_idx  = ring.col_index.get(ts_col_name)
        if ts_col_idx is None:
            return None

        ts_vals = ring.closed_window(ts_col_idx, n).copy()
        return ts_vals.astype(np.float64)

    def _find_ind_col_idx(
        self,
        proxy: "IndicatorProxy | MultiBandProxy",
        band: "_BandRef",
        ring,
    ) -> "int | None":
        if band.col_name and hasattr(ring, "col_index"):
            idx = ring.col_index.get(band.col_name)
            if idx is not None:
                return idx

        from tradetropy.data.data import MultiBandProxy as _MBP, IndicatorProxy as _IP
        try:
            if isinstance(proxy, _MBP):
                view = proxy[band.band_idx]
            elif isinstance(proxy, _IP):
                view = proxy._view
            else:
                return None

            if view is None:
                return None

            for attr in ("_col_idx", "col_idx"):
                col_idx = getattr(view, attr, None)
                if col_idx is not None:
                    return col_idx

            col_name = getattr(view, "_col_name", None) or getattr(view, "col_name", None)
            if col_name is not None and hasattr(ring, "col_index"):
                return ring.col_index.get(col_name)

        except Exception:
            pass
        return None

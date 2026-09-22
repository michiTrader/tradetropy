from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from tradetropy.core.constants import _OHLC_COL, N_OHLC_COLS
from tradetropy.plotting.live.updater._refs import _BandRef

if TYPE_CHECKING:
    from tradetropy.data.data import OhlcProxy, IndicatorProxy, MultiBandProxy
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

    def _resolve_ts_for_band(
        self,
        proxy: "IndicatorProxy | MultiBandProxy",
        band: _BandRef,
        ohlc_proxy: "OhlcProxy",
        ts_ms_fallback: int,
    ) -> int:
        if not band.is_stateful:
            return ts_ms_fallback

        from tradetropy.data.data import MultiBandProxy as _MBP

        try:
            if not isinstance(proxy, _MBP):
                return self._ts_ultima_cerrada(ohlc_proxy) or ts_ms_fallback

            ts_band_indices = getattr(proxy, "_ts_band_indices", [])
            if not ts_band_indices:
                return self._ts_ultima_cerrada(ohlc_proxy) or ts_ms_fallback

            output_names = getattr(proxy, "_output_names", [])
            price_band_pos = band.band_idx
            if price_band_pos < len(ts_band_indices):
                ts_band_idx = ts_band_indices[price_band_pos]
                ts_view = proxy[ts_band_idx]
                if ts_view is not None and len(ts_view) >= 2:
                    ts_raw = float(ts_view[-2])
                    if np.isfinite(ts_raw) and ts_raw > 1_000_000_000_000:
                        return int(ts_raw)

        except Exception:
            pass

        return self._ts_ultima_cerrada(ohlc_proxy) or ts_ms_fallback

    def _stream_new_pivots(
        self,
        ref: "_IndicatorSourceRef",
        band: "_BandRef",
        ring,
        col_idx: int,
    ) -> None:
        from tradetropy.data.data import MultiBandProxy as _MBP

        proxy = ref.proxy
        W = ring._W
        n = min(ring._n_closed, W)
        if n == 0:
            return

        ts_col_idx: "int | None" = None
        if isinstance(proxy, _MBP):
            ts_band_indices = getattr(proxy, "_ts_band_indices", [])
            if ts_band_indices and band.band_idx < len(ts_band_indices):
                ts_band_idx  = ts_band_indices[band.band_idx]
                suffix_price = f"_b{band.band_idx}"
                suffix_ts    = f"_b{ts_band_idx}"
                if band.col_name.endswith(suffix_price):
                    ts_col_name = band.col_name[:-len(suffix_price)] + suffix_ts
                    ts_col_idx  = ring.col_index.get(ts_col_name)

        price_vals = ring.closed_window(col_idx, n)
        if ts_col_idx is not None:
            ts_vals = ring.closed_window(ts_col_idx, n).astype(np.float64)
        else:
            ts_vals = ring.closed_window(_OHLC_COL["ts"], n).astype(np.float64)

        src = band.source
        existing_ts = src.data["ts"]
        if len(existing_ts) > 0:
            last_graficado = int(np.datetime64(existing_ts[-1], "ms").astype(np.int64))
        else:
            last_graficado = 0

        for i in range(n):
            pval = float(price_vals[i])
            if not np.isfinite(pval):
                continue

            ts_raw = float(ts_vals[i])
            if not np.isfinite(ts_raw) or ts_raw <= 0:
                continue

            ts_int = int(ts_raw)

            if ts_int <= last_graficado:
                continue

            ts_dt = np.datetime64(ts_int, "ms")

            if len(src.data["ts"]) > 0 and src.data["ts"][-1] == ts_dt:
                src.patch({"value": [(len(src.data["ts"]) - 1, pval)]})
            else:
                src.stream(
                    {"ts": [ts_dt], "value": [pval], "tag": [band.tag_label]},
                    rollover=self._max_candles,
                )
            last_graficado = ts_int

    def _ts_ultima_cerrada(self, ohlc_proxy: "OhlcProxy") -> "int | None":
        ring = ohlc_proxy._ohlc_ring
        if ring is None or ring._n_closed == 0:
            return None
        p_last = (ring._head - 1) % ring._W
        ts = int(ring._buf[p_last, _OHLC_COL["ts"]])
        return ts if ts > 0 else None

    def _ts_para_pivot(
        self,
        ref: "_IndicatorSourceRef",
        band: "_BandRef",
        ring,
    ) -> "int | None":
        from tradetropy.data.data import MultiBandProxy as _MBP
        proxy = ref.proxy
        if not isinstance(proxy, _MBP):
            return None

        ts_band_indices = getattr(proxy, "_ts_band_indices", [])
        if not ts_band_indices or band.band_idx >= len(ts_band_indices):
            return None

        ts_band_idx = ts_band_indices[band.band_idx]

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

        if ring._n_closed == 0:
            return None

        p = (ring._head - 1) % ring._W
        raw = float(ring._buf[p, ts_col_idx])

        if np.isfinite(raw) and raw > 1_000_000_000_000:
            return int(raw)
        return None

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
        for i_ref, ref in enumerate(self._indicator_refs):
            ring = ref.ohlc_proxy._ohlc_ring
            if ring is None:
                continue

            n = min(ring._n_closed, self._max_candles)

            ts_arr = np.array([], dtype="datetime64[ms]")
            if n > 0:
                ts_col = ring.closed_window(_OHLC_COL["ts"], n).copy()
                ts_arr = ts_col.astype("datetime64[ms]")
                n = len(ts_arr)

            for band in ref.bands:
                if band.is_ts_band:
                    continue

                col_idx = self._find_ind_col_idx(ref.proxy, band, ring)

                if n > 0:
                    if col_idx is not None:
                        vals = ring.closed_window(col_idx, n).copy()
                        min_len = min(len(vals), len(ts_arr))
                        vals    = vals[:min_len]
                        ts_sub  = ts_arr[:min_len]
                        valid = np.isfinite(vals)

                        if np.any(valid) and band.is_stateful:
                            ts_pivot = self._ts_band_history_for_pivot(
                                ref, band, ring, n
                            )
                            if ts_pivot is not None:
                                valid_ts = np.isfinite(ts_pivot) & (ts_pivot > 1_000_000_000_000)
                                mask = valid & valid_ts
                                if np.any(mask):
                                    n_mask = int(mask.sum())
                                    band.source.data = {
                                        "ts":    ts_pivot[mask].astype("datetime64[ms]"),
                                        "value": vals[mask],
                                        "tag":   np.full(n_mask, band.tag_label, dtype=object),
                                    }
                            else:
                                if np.any(valid):
                                    n_valid = int(valid.sum())
                                    band.source.data = {
                                        "ts":    ts_sub[valid],
                                        "value": vals[valid],
                                        "tag":   np.full(n_valid, band.tag_label, dtype=object),
                                    }
                        elif np.any(valid):
                            n_valid = int(valid.sum())
                            _data = {
                                "ts":    ts_sub[valid],
                                "value": vals[valid],
                                "tag":   np.full(n_valid, band.tag_label, dtype=object),
                            }
                            _bc = self._bar_color_arr(band, vals[valid])
                            if _bc is not None:
                                _data["bar_color"] = _bc
                            band.source.data = _data

                if not band.is_stateful and ring._ts_current_candle >= 0:
                    ts_partial = int(ring._ts_current_candle)
                    val_p = self._read_band_value(ref.proxy, band.band_idx, is_stateful=False)
                    if val_p is not None:
                        ts_dt = np.datetime64(ts_partial, "ms")
                        src = band.source
                        existing = src.data["ts"]

                        n_antes = len(existing)
                        last_ts_antes = int(np.datetime64(existing[-1], "ms").astype(np.int64)) if n_antes > 0 else -1

                        def _partial_stream_dict():
                            d = {"ts": [ts_dt], "value": [float(val_p)],
                                 "tag": [band.tag_label]}
                            if band.bar_pos_color is not None:
                                d["bar_color"] = [
                                    band.bar_pos_color if val_p >= 0 else band.bar_neg_color
                                ]
                            return d

                        if len(existing) == 0:
                            src.stream(_partial_stream_dict(), rollover=self._max_candles)
                        elif existing[-1] == ts_dt:
                            _patch = {"value": [(len(existing) - 1, float(val_p))]}
                            if band.bar_pos_color is not None:
                                c = band.bar_pos_color if val_p >= 0 else band.bar_neg_color
                                _patch["bar_color"] = [(len(existing) - 1, c)]
                            src.patch(_patch)
                        else:
                            last_ms = int(np.datetime64(existing[-1], "ms").astype(np.int64))
                            gap_ms = ts_partial - last_ms
                            if 0 < gap_ms <= self._ohlc_refs[0].interval_ms * 2:
                                src.stream(_partial_stream_dict(), rollover=self._max_candles)

            # if i_ref < 3:
            #     for band in ref.bands:
            #         if band.is_ts_band:
            #             continue
            #         ts_b = band.source.data.get("ts", [])
            #         print(f"[DBG IND] ref[{i_ref}] band[{band.band_idx}] "
            #               f"n_source={len(ts_b)}", end="")
            #         if len(ts_b) > 0:
            #             t0 = int(np.datetime64(ts_b[0],  "ms").astype(np.int64))
            #             t1 = int(np.datetime64(ts_b[-1], "ms").astype(np.int64))
            #             print(f"  ts[0]={t0}  ts[-1]={t1}", end="")
            #         print()

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

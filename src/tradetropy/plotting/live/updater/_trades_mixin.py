from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tradetropy.plotting._util import _align_ts_to_candle
from tradetropy.plotting.live.updater._trades_helpers import (
    _build_trades_data_live,
    _build_markers_data_live,
    _build_open_markers_data_live,
    _build_tpsl_data_live,
    _build_pending_orders_data_live,
    _empty_tpsl_data,
    _empty_pending_orders_data,
    _trades_df_to_list,
    _deals_to_trades,
)

if TYPE_CHECKING:
    from tradetropy.plotting.live.updater._refs import _TradesSourceRef


class TradesUpdateMixin:
    def _candle_origin_ms(self) -> int:
        """
        A representative candle-open timestamp (ms) defining the chart's candle
        grid phase, read from the OHLC source actually plotted. Used to snap
        trade timestamps to the same grid the candles sit on. Returns 0 when no
        candles are available yet (falls back to the epoch grid).
        """
        refs = getattr(self, "_ohlc_refs", None)
        if not refs:
            return 0
        ref = refs[0]
        ts_col = ref.source.data.get("ts") if ref.source is not None else None
        if ts_col is not None and len(ts_col) > 0:
            return int(np.datetime64(ts_col[0], "ms").astype(np.int64))
        ring = getattr(ref.proxy, "_ohlc_ring", None)
        if ring is not None and getattr(ring, "_ts_current_candle", -1) >= 0:
            return int(ring._ts_current_candle)
        return 0

    def _align_trades_for_ref(self, ref: "_TradesSourceRef", trades: list) -> list:
        """
        Return a copy of the trade list with entry/exit timestamps snapped to
        the candle grid, so live/replay trade markers and lines sit on top of
        the candles (matching the static chart's align_trades_to_candle). No-op
        when the ref has alignment disabled or the interval is unknown.
        """
        if not trades or not getattr(ref, "align_to_candle", True) or ref.interval_ms <= 0:
            return trades
        origin = self._candle_origin_ms()
        aligned = []
        for t in trades:
            t2 = dict(t)
            t2["entry_ts_ms"] = int(_align_ts_to_candle(
                t["entry_ts_ms"], ref.interval_ms, origin))
            t2["exit_ts_ms"] = int(_align_ts_to_candle(
                t["exit_ts_ms"], ref.interval_ms, origin))
            aligned.append(t2)
        return aligned

    def _update_trades(self, ref: "_TradesSourceRef") -> None:
        import time as _time

        now_s = _time.time()
        if (now_s - ref._last_check_s) < ref.poll_interval_s:
            return

        ref._last_check_s = now_s

        trades = self._fetch_closed_trades(ref)
        if trades is None:
            return

        n_new = len(trades)
        if n_new == 0 and ref._n_known_trades == 0:
            return

        latest_ts = self._latest_trade_ts(trades)
        if latest_ts == ref._last_deal_ts_ms and n_new == ref._n_known_trades:
            return

        ref._last_deal_ts_ms = latest_ts
        ref._n_known_trades  = n_new

        trades = self._align_trades_for_ref(ref, trades)
        new_data = _build_trades_data_live(trades, ref.theme, ref.interval_ms)
        ref.source.data = new_data

        if ref.source_markers is not None:
            ref.source_markers.data = _build_markers_data_live(trades, ref.theme)

        if ref.fig is not None:
            try:
                label = f"Trades ({n_new})"
                for legend in ref.fig.legend:
                    for item in legend.items:
                        if hasattr(item, "label") and item.label is not None and "Trades" in str(item.label):
                            item.label = {"value": label}
            except Exception:
                pass

    def _fetch_closed_trades(self, ref: "_TradesSourceRef") -> "list | None":
        sesh = ref.sesh
        if sesh is None:
            return None

        broker = getattr(sesh, "_broker", None)
        if broker is not None:
            try:
                stats = getattr(broker, "stats", None)
                if stats is not None:
                    trades_df = getattr(stats, "trades", None)
                    if trades_df is not None and not trades_df.empty:
                        return _trades_df_to_list(trades_df)
            except Exception:
                pass

        if broker is not None:
            try:
                deals = broker.get_deal_history(ref.symbol)
                if deals:
                    return _deals_to_trades(deals)
                return []
            except Exception:
                pass

        try:
            deals = sesh.deals(ref.symbol)
        except Exception:
            return None

        if not deals:
            return []

        return _deals_to_trades(deals)

    @staticmethod
    def _latest_trade_ts(trades: list) -> int:
        if not trades:
            return 0
        return max(t.get("exit_ts_ms", 0) for t in trades)

    def _populate_trades_history(self, ref: "_TradesSourceRef") -> None:
        trades = self._fetch_closed_trades(ref)
        if not trades:
            # After an in-place rewind (Restart / Step Back) the broker is reset
            # to zero closed trades. Clear the overlay AND reset the change-
            # detection counters; otherwise they keep the pre-rewind values
            # (e.g. 31 / last_ts) and the deterministic replay that re-creates
            # the identical trade set would satisfy _update_trades' dedup
            # (n_new == _n_known_trades and latest_ts == _last_deal_ts_ms),
            # permanently suppressing the refresh so trades never re-appear.
            ref._n_known_trades  = 0
            ref._last_deal_ts_ms = 0
            ref.source.data = _build_trades_data_live([], ref.theme, ref.interval_ms)
            if ref.source_markers is not None:
                ref.source_markers.data = _build_markers_data_live([], ref.theme)
            return

        ref._n_known_trades  = len(trades)
        ref._last_deal_ts_ms = self._latest_trade_ts(trades)

        trades = self._align_trades_for_ref(ref, trades)
        new_data = _build_trades_data_live(trades, ref.theme, ref.interval_ms)
        ref.source.data = new_data

        if ref.source_markers is not None:
            ref.source_markers.data = _build_markers_data_live(trades, ref.theme)

        if ref.fig is not None:
            try:
                label = f"Trades ({len(trades)})"
                for legend in ref.fig.legend:
                    for item in legend.items:
                        if hasattr(item, "label") and item.label is not None and "Trades" in str(item.label):
                            item.label = {"value": label}
            except Exception:
                pass

    def _fetch_open_positions(self, ref: "_TradesSourceRef") -> list:
        sesh = ref.sesh
        if sesh is None:
            return []

        broker = getattr(sesh, "_broker", None)
        if broker is not None:
            open_trades = getattr(broker, "open_trades", None)
            if open_trades is not None:
                result = []
                for ot in (open_trades if hasattr(open_trades, "__iter__") else []):
                    try:
                        ts   = getattr(ot, "entry_time", None) or getattr(ot, "open_time", None)
                        px   = float(getattr(ot, "entry_price", 0) or getattr(ot, "open_price", 0))
                        typ  = getattr(ot, "direction", None) or getattr(ot, "type", None)
                        if ts is None or px == 0:
                            continue
                        import datetime as _dt, pandas as _pd
                        if isinstance(ts, (int, float)):
                            ts_ms = int(ts * 1000) if ts < 1e12 else int(ts)
                        elif isinstance(ts, _dt.datetime):
                            ts_ms = int(ts.timestamp() * 1000)
                        elif isinstance(ts, _pd.Timestamp):
                            ts_ms = int(ts.timestamp() * 1000)
                        else:
                            ts_ms = 0
                        direction = "buy" if str(typ).lower() in ("buy", "0", "order_type_buy") else "sell"
                        vol = float(
                            getattr(ot, "volume", 0) or getattr(ot, "size", 0)
                            or getattr(ot, "lots", 0) or 1.0
                        )
                        sl = float(getattr(ot, "sl", 0.0) or 0.0)
                        tp = float(getattr(ot, "tp", 0.0) or 0.0)
                        result.append({"entry_ts_ms": ts_ms, "entry_price": px,
                                       "direction": direction, "volume": vol,
                                       "sl": sl, "tp": tp})
                    except Exception:
                        continue
                return result

            try:
                stats = getattr(broker, "stats", None)
                if stats is not None:
                    trades_df = getattr(stats, "trades", None)
                    if trades_df is not None and not trades_df.empty:
                        import pandas as _pd
                        open_df = trades_df[trades_df["exit_time"].isna()]
                        result = []
                        for _, row in open_df.iterrows():
                            try:
                                ts_ms = int(_pd.Timestamp(row["entry_time"]).timestamp() * 1000)
                                px    = float(row["entry_price"])
                                direction = str(row.get("direction", "buy")).lower()
                                vol = float(row.get("size", row.get("volume", 1.0)) or 1.0)
                                sl = float(row.get("sl", 0.0) or 0.0)
                                tp = float(row.get("tp", 0.0) or 0.0)
                                result.append({"entry_ts_ms": ts_ms, "entry_price": px,
                                               "direction": direction, "volume": vol,
                                               "sl": sl, "tp": tp})
                            except Exception:
                                continue
                        return result
            except Exception:
                pass

        try:
            positions = sesh.positions(ref.symbol)
        except Exception:
            return []

        if not positions:
            return []

        result = []
        for pos in positions:
            try:
                ts_raw = getattr(pos, "time", None) or getattr(pos, "open_time", None)
                px     = float(getattr(pos, "price_open", 0) or getattr(pos, "open", 0) or 0)
                typ    = getattr(pos, "type", None)
                if ts_raw is None or px == 0:
                    continue
                import datetime as _dt
                if isinstance(ts_raw, (int, float)):
                    ts_ms = int(ts_raw * 1000) if ts_raw < 1e12 else int(ts_raw)
                elif isinstance(ts_raw, _dt.datetime):
                    ts_ms = int(ts_raw.timestamp() * 1000)
                else:
                    ts_ms = 0
                direction = "buy" if str(typ).lower() in ("buy", "0", "order_type_buy") else "sell"
                vol = float(getattr(pos, "volume", 0) or 1.0)
                sl = float(getattr(pos, "sl", 0.0) or 0.0)
                tp = float(getattr(pos, "tp", 0.0) or 0.0)
                result.append({"entry_ts_ms": ts_ms, "entry_price": px,
                               "direction": direction, "volume": vol,
                               "sl": sl, "tp": tp})
            except Exception:
                continue
        return result

    def _update_open_positions(self, ref: "_TradesSourceRef") -> None:
        if ref.source_open is None:
            return

        broker = getattr(ref.sesh, "_broker", None)
        if broker is None:
            import time as _time
            now_s = _time.time()
            if (now_s - ref._last_check_s) < ref.poll_interval_s:
                return

        positions = self._fetch_open_positions(ref)
        if getattr(ref, "align_to_candle", True) and ref.interval_ms > 0 and positions:
            origin = self._candle_origin_ms()
            positions = [
                dict(p, entry_ts_ms=int(_align_ts_to_candle(
                    p["entry_ts_ms"], ref.interval_ms, origin)))
                for p in positions
            ]
        new_data = _build_open_markers_data_live(positions, ref.theme)
        ref.source_open.data = new_data
        self._update_position_line(ref, positions)
        self._update_tpsl_lines(ref, positions)

    def _update_tpsl_lines(self, ref: "_TradesSourceRef", positions: list) -> None:
        """
        Refresh the data-driven TP/SL lines from the open positions.

        Populates the TP/SL ColumnDataSource with one horizontal line per
        non-zero stop-loss / take-profit level (SL loss color, TP win color),
        each with a left-edge label. Cleared to empty when flat or when no
        brackets are set. The price levels are not time-keyed, so they need no
        candle alignment.

        Args:
            ref (_TradesSourceRef): The trades source reference.
            positions (list): Open-position dicts (carry 'sl' / 'tp').
        """
        tpsl_source = getattr(ref, "tpsl_source", None)
        if tpsl_source is None:
            return
        tpsl_source.data = _build_tpsl_data_live(positions, ref.theme)

    def _reset_open_positions_overlay(self, ref: "_TradesSourceRef") -> None:
        """
        Clear the open-position overlay to a flat state on an in-place rewind.

        Called from repopulate_after_topup (Restart / Step Back) so a true reset
        starts from an empty overlay: the open-position markers, the average-
        price line/label and the TP/SL lines are all cleared, and the poll
        throttle is reset so the very next forward tick repopulates them from the
        (already reset) broker state. Without this the pre-rewind position
        glyphs survive the rewind and never get cleanly re-created.

        Args:
            ref (_TradesSourceRef): The trades source reference to reset.
        """
        if ref.source_open is not None:
            ref.source_open.data = _build_open_markers_data_live([], ref.theme)

        pos_line = getattr(ref, "pos_line", None)
        if pos_line is not None:
            pos_line.visible = False
        pos_label = getattr(ref, "pos_label", None)
        if pos_label is not None:
            pos_label.visible = False

        # Clear the data-driven TP/SL lines (wired in render_trades_live).
        tpsl_source = getattr(ref, "tpsl_source", None)
        if tpsl_source is not None:
            tpsl_source.data = _empty_tpsl_data()

        pending_source = getattr(ref, "pending_source", None)
        if pending_source is not None:
            pending_source.data = _empty_pending_orders_data()

        # Reset the poll throttle so the next update refreshes immediately.
        ref._last_check_s = 0.0

    def _update_position_line(self, ref: "_TradesSourceRef", positions: list) -> None:
        """
        Drive the active-position tracker line (avg price + size label).

        Renders a horizontal reference line at the open position's
        volume-weighted average price with a label 'Avg <price> | Size <n>'.
        Hidden when flat. ``positions`` is the list already fetched by
        _update_open_positions (each item: entry_price/direction; size summed).
        """
        pos_line = getattr(ref, "pos_line", None)
        pos_label = getattr(ref, "pos_label", None)
        if pos_line is None and pos_label is None:
            return

        if not positions:
            if pos_line is not None:
                pos_line.visible = False
            if pos_label is not None:
                pos_label.visible = False
            return

        # Volume-weighted average price; size = total open volume; side from the
        # net signed volume.
        total_vol = 0.0
        notional = 0.0
        net = 0.0
        for p in positions:
            px = float(p.get("entry_price", 0.0) or 0.0)
            vol = float(p.get("volume", 1.0) or 1.0)
            side = str(p.get("direction", "buy")).lower()
            total_vol += abs(vol)
            notional += abs(vol) * px
            net += vol if side == "buy" else -vol
        if total_vol <= 0:
            if pos_line is not None:
                pos_line.visible = False
            if pos_label is not None:
                pos_label.visible = False
            return

        avg = notional / total_vol
        size = abs(net) if net != 0 else total_vol
        side_txt = "BUY" if net >= 0 else "SELL"

        if pos_line is not None:
            pos_line.location = avg
            pos_line.visible = True
        if pos_label is not None:
            pos_label.y = avg
            pos_label.text = f"{side_txt}  Avg {avg:.2f} | Size {size:g}"
            pos_label.visible = True

    def _update_pending_orders_lines(self, ref: "_TradesSourceRef") -> None:
        """
        Refresh horizontal lines for pending limit/stop orders on the chart.

        Reads pending orders from the broker and draws one horizontal dotted
        line per order at its price level with a left-edge label.
        """
        pending_source = getattr(ref, "pending_source", None)
        if pending_source is None:
            return
        sesh = ref.sesh
        if sesh is None:
            return
        try:
            orders = sesh.orders(ref.symbol)
        except Exception:
            return
        from tradetropy.core.broker import OrderState
        pending = [
            o for o in orders
            if getattr(o, "state", None) == OrderState.ORDER_STATE_PLACED
        ]
        new_data = _build_pending_orders_data_live(pending, ref.theme)
        pending_source.data = new_data

from __future__ import annotations

import numpy as np


class EquityStatsMixin:
    def _update_equity(self) -> None:
        from tradetropy.plotting.sources import _prepare_equity_series

        ref    = self._equity_ref
        broker = ref.broker
        if broker is None:
            return

        try:
            eq_curve = broker.equity_curve
        except Exception:
            return

        if eq_curve is None or eq_curve.empty:
            return

        eq_plot, _ = _prepare_equity_series(
            eq_curve,
            broker.initial_balance,
            ref.config.equity_mode,
            ref.config.equity_unit,
        )

        ts_dt = eq_plot.index[-1].to_datetime64().astype("datetime64[ms]")
        val   = float(eq_plot.iloc[-1])
        # A non-finite value pushed through stream/patch is serialized to the
        # websocket as plain JSON, where NaN/Inf are not compliant and raise in
        # Bokeh's send path. Skip the point instead of crashing the update.
        # The same applies to the ts: a NaT index serializes to a non-finite
        # float, so guard it too.
        if np.isnat(ts_dt) or not np.isfinite(val):
            return
        src   = ref.source
        existing = src.data["ts"]

        if len(existing) > 0 and existing[-1] == ts_dt:
            src.patch({"equity": [(len(existing) - 1, val)]})
        else:
            src.stream(
                {"ts": [ts_dt], "equity": [val]},
                rollover=self._max_candles * 20,
            )

    def _update_trailing_dd(self) -> None:
        from tradetropy.plotting.sources import trailing_drawdown, _prepare_equity_series

        ref    = self._equity_ref
        broker = ref.broker
        if broker is None:
            return
        try:
            eq_curve = broker.equity_curve
        except Exception:
            return
        if eq_curve is None or eq_curve.empty:
            return

        max_td = ref.config.max_trailing_dd
        if max_td is None:
            return

        src = ref.trailing_source
        if src is None:
            return

        mult = eq_curve / broker.initial_balance
        trailing_mult = trailing_drawdown(mult, max_td)
        result, _ = _prepare_equity_series(
            trailing_mult * broker.initial_balance,
            broker.initial_balance,
            ref.config.equity_mode,
            ref.config.equity_unit,
        )

        ts_dt = result.index[-1].to_datetime64().astype("datetime64[ms]")
        val   = float(result.iloc[-1])
        if np.isnat(ts_dt) or not np.isfinite(val):
            return
        existing = src.data["ts"]
        if len(existing) > 0 and existing[-1] == ts_dt:
            src.patch({"trailing_dd": [(len(existing) - 1, val)]})
        else:
            src.stream(
                {"ts": [ts_dt], "trailing_dd": [val]},
                rollover=self._max_candles * 20,
            )

    def _update_drawdown(self) -> None:
        from tradetropy.plotting.sources import _prepare_equity_series

        ref    = self._equity_ref
        broker = ref.broker
        if broker is None:
            return

        try:
            eq_curve = broker.equity_curve
        except Exception:
            return

        if eq_curve is None or eq_curve.empty:
            return

        dd_series = eq_curve / eq_curve.cummax() - 1.0
        ts_dt  = dd_series.index[-1].to_datetime64().astype("datetime64[ms]")
        dd_val = float(dd_series.iloc[-1])
        if np.isnat(ts_dt) or not np.isfinite(dd_val):
            return

        src      = self._drawdown_source
        existing = src.data.get("ts", [])

        if len(existing) > 0 and existing[-1] == ts_dt:
            src.patch({"drawdown": [(len(existing) - 1, dd_val)]})
        else:
            src.stream(
                {"ts": [ts_dt], "equity": [dd_val], "drawdown": [dd_val]},
                rollover=self._max_candles * 20,
            )

    def _update_pl(self) -> None:
        from tradetropy.plotting.render import render_pl_bars as _render_pl_bars

        src = self._pl_source
        fig = self._fig_pl
        if src is None or fig is None:
            return

        pnl_arr = src.data.get("pnl", [])
        if len(pnl_arr) == 0:
            return

        n_current = len(src.data.get("_pl_top", []))
        if n_current == len(pnl_arr):
            return

        if "pnl_pct" not in src.data or len(src.data["pnl_pct"]) != len(pnl_arr):
            entry_price_arr = np.asarray(src.data.get("entry_price", []), dtype=np.float64)
            size_arr        = np.ones(len(pnl_arr))
            notional = np.where(entry_price_arr > 0, np.abs(entry_price_arr), np.nan)
            pnl_pct = np.where(
                np.isfinite(notional),
                np.asarray(pnl_arr, dtype=np.float64) / notional * 100.0,
                0.0,
            )
            src.data["pnl_pct"] = pnl_pct

        if "is_win" not in src.data:
            src.data["is_win"] = np.where(
                np.asarray(src.data.get("pnl", []), dtype=np.float64) > 0,
                "1", "0",
            ).tolist()

        if "size_signed" not in src.data or len(src.data["size_signed"]) != len(pnl_arr):
            size_arr = np.abs(np.asarray(src.data.get("size", []), dtype=np.float64))
            direction_arr = np.asarray(src.data.get("direction", []), dtype=object)
            is_short = np.array(
                [str(d).lower() in ("short", "sell") for d in direction_arr],
                dtype=bool,
            )
            src.data["size_signed"] = np.where(is_short, -size_arr, size_arr)

        if "duration_ms" not in src.data:
            entry_ms = np.asarray(src.data.get("entry_ts", []), dtype="datetime64[ms]").astype(np.int64)
            exit_ms  = np.asarray(src.data.get("exit_ts",  []), dtype="datetime64[ms]").astype(np.int64)
            src.data["duration_ms"]  = (exit_ms - entry_ms).astype(np.float64)
            src.data["trade_label"]  = [
                f"{'+' if p >= 0 else ''}{p:.2f}%"
                for p in src.data["pnl_pct"]
            ]
            src.data["trade_idx"]    = list(range(len(pnl_arr)))

        fig.renderers = []
        fig.tools = []

        _theme_dict = {}
        if self._trades_ref is not None:
            _theme_dict = self._trades_ref.theme
        _render_pl_bars(fig, src, _theme_dict)

    def _update_stats(self) -> None:
        broker = self._stats_broker
        if broker is None or self._stats_div is None:
            return

        try:
            eq_curve = broker.equity_curve
        except Exception:
            return

        if eq_curve is None or eq_curve.empty:
            return

        eq_len = len(eq_curve)

        try:
            trades = broker.get_trades()
        except Exception:
            return

        n_trades = len(trades) if trades is not None else 0

        if eq_len == self._stats_last_eq_len and n_trades == self._stats_last_n_trades:
            return

        self._stats_last_eq_len   = eq_len
        self._stats_last_n_trades = n_trades

        try:
            import warnings
            from tradetropy.stats.stats import compute_stats
            from tradetropy.plotting._layout import _build_stats_div

            initial_balance = float(getattr(broker, "initial_balance", 0.0))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                stats = compute_stats(eq_curve, trades, initial_balance)
            new_div = _build_stats_div(
                stats,
                self._stats_theme,
            )
            self._stats_div.text = new_div.text
        except Exception:
            pass

"""
updater package
===============
LiveSourceUpdater -- updates Bokeh ColumnDataSources on each tick/bar.

Split into submodules to keep each file focused and readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradetropy.plotting.live.updater._refs import (
    _OhlcSourceRef,
    _BandRef,
    _IndicatorSourceRef,
    _EquitySourceRef,
    _FootprintSourceRef,
    _TradesSourceRef,
    _VolumeProfileSourceRef,
)
from tradetropy.plotting.live.updater._footprint_helpers import (
    _build_fp_rows_for_candle,
)
from tradetropy.plotting.live.updater._trades_helpers import (
    _deals_to_trades,
    _deal_ts_ms,
    _trades_df_to_list,
    _empty_markers_source,
    _build_open_markers_data_live,
    _build_markers_data_live,
    _build_trades_data_live,
)
from tradetropy.plotting.live.updater._ohlc_mixin import OhlcUpdateMixin
from tradetropy.plotting.live.updater._indicator_mixin import IndicatorUpdateMixin
from tradetropy.plotting.live.updater._trades_mixin import TradesUpdateMixin
from tradetropy.plotting.live.updater._equity_mixin import EquityStatsMixin
from tradetropy.plotting.live.updater._footprint_mixin import FootprintUpdateMixin
from tradetropy.plotting.live.updater._history_mixin import HistoryPopulateMixin
from tradetropy.plotting.live.updater._vp_mixin import VolumeProfileUpdateMixin

if TYPE_CHECKING:
    from tradetropy.data.data import OhlcProxy, IndicatorProxy, MultiBandProxy
    from tradetropy.plotting.live.navigation import LiveNavigationController


# ══════════════════════════════════════════════════════════════════════════════
# LIVE SOURCE UPDATER -- main class
# ══════════════════════════════════════════════════════════════════════════════

class LiveSourceUpdater(
    OhlcUpdateMixin,
    IndicatorUpdateMixin,
    TradesUpdateMixin,
    EquityStatsMixin,
    FootprintUpdateMixin,
    HistoryPopulateMixin,
    VolumeProfileUpdateMixin,
):
    """
    Update all Bokeh ColumnDataSources in response to new data.

    Instantiated once from build_live_document() and receives all the
    references to sources and proxies it needs. update() is the only public
    method and is called from doc.add_next_tick_callback() -- always in the
    Bokeh IO loop, never directly from the engine thread.

    bar_closed parameter
    --------------------
    update(bar_closed=True) when the current tick closed a candle.
    This triggers the update of stateful indicators (pivots) and
    streams the newly closed candle to the footprint source.
    On all ticks, the footprint partial candle is updated.
    """

    __slots__ = (
        "_ohlc_refs",
        "_indicator_refs",
        "_equity_ref",
        "_fp_refs",
        "_vp_refs",
        "_tool_ref",
        "_strategy",
        "_trades_ref",
        "_drawdown_source",
        "_trailing_dd_source",
        "_pl_source",
        "_fig_pl",
        "_max_candles",
        "_last_ohlc_ts",
        "_fig_ohlc",
        "_history_loaded",
        "_navigation",
        "_stats_div",
        "_stats_broker",
        "_stats_symbol",
        "_stats_interval_ms",
        "_stats_theme",
        "_stats_last_n_trades",
        "_stats_last_eq_len",
        "_panel_refresh",
        "_orders_refresh",
    )

    def __init__(
        self,
        ohlc_refs:      list[_OhlcSourceRef],
        indicator_refs: list[_IndicatorSourceRef],
        equity_ref:     "_EquitySourceRef | None",
        fp_refs:        list[_FootprintSourceRef],
        max_candles:    int,
        fig_ohlc:       object = None,
        trades_ref:     "_TradesSourceRef | None" = None,
        drawdown_source:    object = None,
        trailing_dd_source: object = None,
        pl_source:          object = None,
        fig_pl:             object = None,
        navigation:      "'LiveNavigationController | None'" = None,
        stats_div:       object = None,
        stats_broker:    object = None,
        stats_symbol:    str    = "",
        stats_interval_ms: int  = 0,
        stats_theme:     object = None,
        vp_refs:         "list[_VolumeProfileSourceRef] | None" = None,
        tool_ref:        "object | None" = None,
        strategy:        object = None,
        panel_refresh:   "object | None" = None,
        orders_refresh:  "object | None" = None,
    ):
        self._ohlc_refs       = ohlc_refs
        self._indicator_refs  = indicator_refs
        self._equity_ref      = equity_ref
        self._fp_refs         = fp_refs
        self._vp_refs         = vp_refs or []
        self._tool_ref        = tool_ref
        self._strategy        = strategy
        self._trades_ref      = trades_ref
        self._drawdown_source    = drawdown_source
        self._trailing_dd_source = trailing_dd_source
        self._pl_source          = pl_source
        self._fig_pl          = fig_pl
        self._max_candles     = max_candles
        self._last_ohlc_ts: dict[int, int] = {}
        self._fig_ohlc        = fig_ohlc
        self._history_loaded  = False
        self._navigation      = navigation
        self._stats_div          = stats_div
        self._stats_broker       = stats_broker
        self._stats_symbol       = stats_symbol
        self._stats_interval_ms  = stats_interval_ms
        self._stats_theme        = stats_theme or {}
        self._stats_last_n_trades = 0
        self._stats_last_eq_len   = 0
        self._panel_refresh       = panel_refresh
        self._orders_refresh      = orders_refresh

    # ── Entrypoint ───────────────────────────────────────────────────────

    def update(self, bar_closed: bool = False, heavy: bool = True) -> None:
        """
        Refresh all ColumnDataSources with the current state.

        Args:
            bar_closed (bool): True when the current tick closed a candle.
            heavy (bool): If True (default), also refresh the heavy draw layers
                based on Indicator.draw() (Volume Profile / Heatmap and
                use_tool() snapshots), which rebuild their full geometry each
                time. The chart calls them with heavy=False for most frames and
                heavy=True at a coarser cadence (heavy_refresh_ms) or on candle
                close, to avoid occupying the IOLoop recomputing that geometry
                on every frame. Since these layers recompute from the current
                causal state, skipping frames loses no information.
        """
        if not self._history_loaded:
            return
        # On candle close, heavy layers must reflect the final state.
        heavy = heavy or bar_closed
        self._update_ohlc()
        self._update_indicators(bar_closed=bar_closed)
        if self._equity_ref is not None:
            self._update_equity()
        if self._trailing_dd_source is not None and self._equity_ref is not None:
            self._update_trailing_dd()
        if self._drawdown_source is not None and self._equity_ref is not None:
            self._update_drawdown()
        for ref in self._fp_refs:
            if bar_closed:
                self._update_footprint(ref)
            self._update_footprint_partial(ref)
        # The Volume Profile / Heatmap is redrawn by recomputing its entire
        # geometry (draw()). It is the most expensive layer per frame, so it is
        # refreshed at the coarse cadence (heavy) instead of every tick: the
        # developing profile still reflects fluidly (~5 fps) without saturating
        # the IOLoop.
        if heavy:
            for ref in self._vp_refs:
                self._update_volume_profile(ref)
            # The tool is drawn when the ref already exists, or when it does not
            # exist yet but use_tool() has already accumulated some snapshot
            # (deferred creation).
            if self._tool_ref is not None or getattr(self._strategy, "_tool_snapshots", None):
                self._update_tool_snapshots(self._tool_ref)
        if self._trades_ref is not None:
            self._update_trades(self._trades_ref)
            self._update_open_positions(self._trades_ref)
            self._update_pending_orders_lines(self._trades_ref)
        if self._pl_source is not None and self._trades_ref is not None:
            self._update_pl()
        if bar_closed and self._stats_div is not None:
            self._update_stats()
        # Refresh the manual Order Ticket / Position Modifier panels (training
        # mode) so the position widget reflects the live state each tick.
        if self._panel_refresh is not None:
            try:
                self._panel_refresh()
            except Exception:
                pass
        if self._orders_refresh is not None:
            try:
                self._orders_refresh()
            except Exception:
                pass


__all__ = [
    "LiveSourceUpdater",
    "_OhlcSourceRef",
    "_BandRef",
    "_IndicatorSourceRef",
    "_EquitySourceRef",
    "_FootprintSourceRef",
    "_TradesSourceRef",
    "_VolumeProfileSourceRef",
    "_build_fp_rows_for_candle",
    "_deals_to_trades",
    "_deal_ts_ms",
    "_trades_df_to_list",
    "_empty_markers_source",
    "_build_open_markers_data_live",
    "_build_markers_data_live",
    "_build_trades_data_live",
]

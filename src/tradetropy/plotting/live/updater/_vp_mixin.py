from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradetropy.plotting.live.updater._refs import _VolumeProfileSourceRef


def _resolve_show_legend(defn: dict) -> bool:
    """
    Resolve whether an indicator's draw primitives show a legend in live.

    Mirrors the static-chart resolution (``_util._meta_from_defn``): an explicit
    ``show_legend`` wins; otherwise own-panel indicators (overlay False) default
    to no legend and overlays default to a legend. Keeps Delta / CVD legend-free
    and DeltaVolumeInfo's per-row legend identical to the backtest chart.

    Args:
        defn (dict): The indicator_def carrying the indicator and plot_config.

    Returns:
        bool: True to render legend labels for the indicator's primitives.
    """
    from tradetropy.plotting._util import _cfg

    pc = _cfg(defn)
    if pc.show_legend is not None:
        return bool(pc.show_legend)

    overlay = pc.overlay
    if overlay is None:
        indicator = defn.get("indicator")
        category = getattr(indicator, "category", None)
        if category is not None:
            overlay = pc.resolve_overlay(category)
        else:
            overlay = True
    return False if overlay is False else True


class VolumeProfileUpdateMixin:
    """
    Updates the draw primitives emitted by declared indicators (e.g. the Volume
    Profile histogram) on the price panel.

    The indicator (VolumeProfile / RollingVolumeProfile / TickVolumeProfile, or
    any indicator implementing draw()) recomputes its histograms in
    calculate() - invoked by the feed on every candle close / tick. Here we
    re-collect its draw primitives and feed them to the generic renderer
    (render_tool_groups), which updates the per-(legend, kind) sources in place
    via the ref's registry. This is the exact same path used for the static
    chart and for use_tool() snapshots, so live and backtest stay identical.
    """

    def _update_volume_profile(self, ref: "_VolumeProfileSourceRef") -> None:
        from tradetropy.plotting._util import (
            _indicator_primitive_groups, _snap_primitive_groups,
        )
        from tradetropy.plotting.render._tools import render_tool_groups

        # Tick-mounted indicators (e.g. LargeTrades) are not recomputed by the
        # standard live indicator path; let them refresh from their source window
        # before we re-collect their primitives. OHLC-mounted indicators (VP) are
        # already kept fresh by the engine, so they have no live_refresh hook.
        refresh = getattr(ref.indicator, "live_refresh", None)
        if refresh is not None:
            refresh(ref.source_proxy)

        groups = _indicator_primitive_groups(ref.defn, ref.interval_ms)
        if not groups:
            return
        # Floor draw() geometry to the candle grid (parity with the static plot,
        # gated by the same PlotConfig flag carried on the ref).
        if getattr(ref, "align_to_candle", True):
            _snap_primitive_groups(
                groups, ref.interval_ms,
                getattr(ref, "candle_origin_ms", 0),
            )
        render_tool_groups(
            ref.fig, groups, theme=ref.theme,
            interval_ms=ref.interval_ms, registry=ref.registry,
            show_legend=_resolve_show_legend(ref.defn),
            lazy_x_range=getattr(ref.fig, "x_range", None),
            lazy_zoom_range=ref.lazy_zoom_range,
        )
        self._sync_autoscale_mirror(ref)

    def _sync_autoscale_mirror(self, ref: "_VolumeProfileSourceRef") -> None:
        """
        Mirror this overlay's quad cells into its autoscale source.

        Overlays that opt into the price autoscale (e.g. the Heatmap) carry a
        pre-created ``autoscale_mirror`` CDS that the OHLC autoscale reads. The
        real quad source is created lazily by ``render_tool_groups`` on the first
        non-empty draw, so after each render we copy its left/right/bottom/top
        columns into the mirror. No-op for overlays without a mirror.
        """
        mirror = getattr(ref, "autoscale_mirror", None)
        if mirror is None:
            return
        left, right, bottom, top = [], [], [], []
        for (_, kind), src in ref.registry.items():
            d = src.data
            if "left" in d and "right" in d and "bottom" in d and "top" in d:
                left.extend(list(d["left"]))
                right.extend(list(d["right"]))
                bottom.extend(list(d["bottom"]))
                top.extend(list(d["top"]))
        mirror.data = dict(left=left, right=right, bottom=bottom, top=top)

    def _populate_vp_history(self) -> None:
        for ref in self._vp_refs:
            self._update_volume_profile(ref)

    def _update_tool_snapshots(self, ref) -> None:
        """
        Rebuild the use_tool() snapshot glyphs from the strategy's accumulated list.

        Snapshots are created dynamically inside on_data() via use_tool(); each
        tick we check whether new ones appeared and, if so, re-render every tool's
        draw primitives through the generic renderer (render_tool_groups), which
        updates the existing per-(legend, kind) sources or creates them lazily.

        The tool ref is created lazily the first time a snapshot appears: at
        document build time there is usually no snapshot yet (the first on_data()
        has not run), so ``ref`` is None until then. New glyphs attach to the OHLC
        figure whose Legend already has click_policy="hide", so each tool's legend
        entry inherits the toggle.
        """
        from tradetropy.ta.tool import collect_draw_primitives
        from tradetropy.plotting.render._tools import render_tool_groups

        if ref is None:
            strategy = self._strategy
            snapshots = getattr(strategy, "_tool_snapshots", None)
            if not snapshots or self._fig_ohlc is None:
                return
            from tradetropy.plotting.live.document import _create_tool_ref
            ref = _create_tool_ref(
                strategy, self._fig_ohlc, self._stats_theme, self._stats_interval_ms,
            )
            self._tool_ref = ref

        snapshots = getattr(ref.strategy, "_tool_snapshots", None)
        if not snapshots:
            return
        if len(snapshots) == ref._n_drawn:
            return  # nothing new since last tick

        groups = collect_draw_primitives(snapshots)
        if not groups:
            ref._n_drawn = len(snapshots)
            return

        render_tool_groups(
            ref.fig, groups, theme=ref.theme,
            interval_ms=ref.interval_ms, registry=ref.registry,
            lazy_x_range=getattr(ref.fig, "x_range", None),
            lazy_zoom_range=ref.lazy_zoom_range,
        )
        ref._n_drawn = len(snapshots)

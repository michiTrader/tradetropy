"""
Fixed-range volume profile tool.

:class:`FixedRangeVP` computes a volume-by-price profile over an explicit
``[start, end]`` timestamp range and returns an immutable
:class:`VolumeProfileResult`. Invoked through
``Strategy.use_tool(source, FixedRangeVP(...), start=, end=)`` inside
``on_data()``.

When drawn, the tool emits declarative primitives (see
:mod:`tradetropy.ta.tool.draw`): a horizontal-bar histogram split into buy/sell
tones, a red POC line, and HVN/LVN node dots — the same look the static and live
renderers produced before, now routed through the generic tool renderer.
"""

from __future__ import annotations

import numpy as np

from tradetropy.core.constants import _TICK_COL
from tradetropy.exceptions import ConfigError
from tradetropy.ta._volume_profile import (
    VolumeNode,
    compute_range_profile,
    DEFAULT_VP_BUY,
    DEFAULT_VP_SELL,
    DEFAULT_VP_POC,
    DEFAULT_VP_HVN,
    DEFAULT_VP_LVN,
)
from tradetropy.ta.tool.base import Tool, _to_ms
from tradetropy.ta.tool.draw import HBars, HLines, Points, Primitive, ToolPlotConfig


class VolumeProfileResult:
    """
    Immutable snapshot returned by a volume profile tool.

    Falsy (``bool()`` is False) when empty - i.e. when the requested range has
    no data in the source buffer.

    Attributes:
        start (int): Effective range start ts in ms.
        end (int): Effective range end ts in ms.
        poc (float): Point of control price.
        vah (float): Value-area high.
        val (float): Value-area low.
        hvn (list[VolumeNode]): High-volume nodes (empty if not requested).
        lvn (list[VolumeNode]): Low-volume nodes (empty if not requested).
        nodes (list[VolumeNode]): hvn + lvn, ordered by price.
        tick_size (float): Effective level size used.
        profile (dict): Raw histogram with keys 'prices', 'volumes', 'vol_bid',
            'vol_ask', 'deltas'.
    """

    __slots__ = (
        "start", "end", "poc", "vah", "val",
        "hvn", "lvn", "tick_size", "profile", "_empty",
    )

    def __init__(self, prof: dict | None):
        if not prof:
            self._empty = True
            self.start = self.end = 0
            self.poc = self.vah = self.val = float("nan")
            self.hvn = []
            self.lvn = []
            self.tick_size = 0.0
            self.profile = {}
            return
        self._empty = False
        self.start = int(prof["start"])
        self.end = int(prof["end"])
        self.poc = float(prof["poc"])
        self.vah = float(prof["vah"])
        self.val = float(prof["val"])
        self.hvn = list(prof.get("hvn", []))
        self.lvn = list(prof.get("lvn", []))
        self.tick_size = float(prof.get("tick_size", 0.0))
        self.profile = {
            "prices": prof["prices"],
            "volumes": prof["volumes"],
            "vol_bid": prof["vol_bid"],
            "vol_ask": prof["vol_ask"],
            "deltas": prof["deltas"],
        }

    @property
    def nodes(self) -> list[VolumeNode]:
        """All detected nodes (hvn + lvn) ordered ascending by price."""
        return sorted([*self.hvn, *self.lvn], key=lambda n: n.price)

    def __bool__(self) -> bool:
        return not self._empty

    def __repr__(self) -> str:
        if self._empty:
            return "VolumeProfileResult(empty)"
        return (
            f"VolumeProfileResult(poc={self.poc:.2f} vah={self.vah:.2f} "
            f"val={self.val:.2f} hvn={len(self.hvn)} lvn={len(self.lvn)})"
        )


class FixedRangeVP(Tool):
    """
    Volume profile over an explicit [start, end] range, computed on demand.

    Invoked through ``Strategy.use_tool(source, FixedRangeVP(...), start=, end=)``
    inside ``on_data()``. The range is chosen at call time, so the same tool
    covers fixed historical ranges and 'anchored to now' usage (pass
    ``start=<event ts>``, ``end=self.ts``).

    Both tick (TickProxy) and kline (OhlcProxy) sources are supported; the tool
    reads only the rows currently held in the source's window buffer.

    Args:
        tick_size (float | None): Price level size; inferred when None.
        bins (int): Target number of levels when tick_size is None.
        value_area_pct (float): Fraction of volume defining the value area.
        nodes (str | None): None | 'hvn' | 'lvn' | 'both'.
        node_prominence (float): Node prominence threshold 0..1.
        max_nodes (int | None): Optional cap per node kind (strongest first).
        width_fraction (float): Max fraction of the range width used by a full
            (max-volume) bar when drawing. Kept well under 1 so bars do not cover
            the candles.
        buy_color (str): Color of the buy-aggressor (ask) volume segment.
            Fixed default, independent of the plot theme.
        sell_color (str): Color of the sell-aggressor (bid) volume segment.
        poc_color (str): Color of the point-of-control line.
        hvn_color (str): Color of the high-volume node markers.
        lvn_color (str): Color of the low-volume node markers.
    """

    name = "vp"

    def __init__(
        self,
        *,
        tick_size: float | None = None,
        bins: int = 100,
        value_area_pct: float = 0.70,
        nodes: str | None = None,
        node_prominence: float = 0.1,
        max_nodes: int | None = None,
        width_fraction: float = 0.32,
        buy_color: str = DEFAULT_VP_BUY,
        sell_color: str = DEFAULT_VP_SELL,
        poc_color: str = DEFAULT_VP_POC,
        hvn_color: str = DEFAULT_VP_HVN,
        lvn_color: str = DEFAULT_VP_LVN,
    ):
        if nodes not in (None, "hvn", "lvn", "both"):
            raise ConfigError(
                f"nodes must be None, 'hvn', 'lvn' or 'both', not {nodes!r}"
            )
        self.tick_size = tick_size
        self.bins = bins
        self.value_area_pct = value_area_pct
        self.nodes = nodes
        self.node_prominence = node_prominence
        self.max_nodes = max_nodes
        self.width_fraction = width_fraction
        # Semantic colors, theme-independent (live on the object).
        self.buy_color = buy_color
        self.sell_color = sell_color
        self.poc_color = poc_color
        self.hvn_color = hvn_color
        self.lvn_color = lvn_color
        self.plot_config = ToolPlotConfig(name="VP")

    def run(self, source, *, start=None, end=None) -> VolumeProfileResult:
        """
        Compute the profile over the source buffer slice [start, end].

        Args:
            source (TickProxy | OhlcProxy): Subscription the slice is read from.
            start: Range start (epoch ms, datetime, ISO str, or None=oldest row).
            end: Range end (epoch ms, datetime, ISO str, or None=newest row).

        Returns:
            VolumeProfileResult: The snapshot (falsy when the range is empty).
        """
        # Tick vs kline source: decided by which columns the proxy exposes.
        is_tick = hasattr(source, "price_ref") and "price" in _TICK_COL
        is_tick = is_tick and not hasattr(source, "open_ref")

        ts = np.asarray(source.ts[:], dtype=np.int64)
        start_ms = _to_ms(start, default=None)
        end_ms = _to_ms(end, default=None)

        if is_tick:
            prof = compute_range_profile(
                ts,
                is_tick=True,
                start=start_ms,
                end=end_ms,
                tick_size=self.tick_size,
                bins=self.bins,
                value_area_pct=self.value_area_pct,
                nodes=self.nodes,
                node_prominence=self.node_prominence,
                max_nodes=self.max_nodes,
                price=np.asarray(source.price[:], dtype=np.float64),
                volume=np.asarray(source.volume[:], dtype=np.float64),
                flags=np.asarray(source.flags[:], dtype=np.float64).astype(np.int64),
            )
        else:
            prof = compute_range_profile(
                ts,
                is_tick=False,
                start=start_ms,
                end=end_ms,
                tick_size=self.tick_size,
                bins=self.bins,
                value_area_pct=self.value_area_pct,
                nodes=self.nodes,
                node_prominence=self.node_prominence,
                max_nodes=self.max_nodes,
                open_=np.asarray(source.open[:], dtype=np.float64),
                high=np.asarray(source.high[:], dtype=np.float64),
                low=np.asarray(source.low[:], dtype=np.float64),
                close=np.asarray(source.close[:], dtype=np.float64),
                volume=np.asarray(source.volume[:], dtype=np.float64),
            )
        return VolumeProfileResult(prof)

    def draw(self, result: VolumeProfileResult, cfg: ToolPlotConfig) -> list[Primitive]:
        """
        Emit draw primitives for the snapshot: histogram bars, POC line, nodes.

        Returns an empty list when the range was empty. Colors come from the
        tool's object attributes (buy_color / sell_color / poc_color /
        hvn_color / lvn_color), so they are independent of the plot theme.
        """
        if not result:
            return []

        prims: list[Primitive] = []

        # ── Stacked bar histogram (buy/sell stacked) ──────────────────────────
        bars = _vp_bar_arrays(
            result, self.width_fraction, self.buy_color, self.sell_color
        )
        if bars is not None:
            ys, heights, lefts, rights, colors, alphas, vols, vbuys, vsells = bars
            prims.append(HBars(
                y=ys, height=heights, left=lefts, right=rights,
                color=colors, alpha=alphas,
                hover=[
                    ("Price", "@y{0,0.00}"),
                    ("Buy", "@vol_buy{0,0.00}"),
                    ("Sell", "@vol_sell{0,0.00}"),
                    ("Total", "@volume{0,0.00}"),
                ],
                extra={"volume": vols, "vol_buy": vbuys, "vol_sell": vsells},
            ))

        # -- POC line -----------------------------------------------------------
        if np.isfinite(result.poc):
            prims.append(HLines(
                x0=[result.start], x1=[result.end], y=[result.poc],
                color=self.poc_color, alpha=0.9, width=1.6,
            ))

        # -- HVN / LVN nodes at the right edge of the profile --------------------
        node_x, node_y, node_c = [], [], []
        for kind, color in (("hvn", self.hvn_color), ("lvn", self.lvn_color)):
            for node in getattr(result, kind):
                node_x.append(result.end)
                node_y.append(float(node.price))
                node_c.append(color)
        if node_y:
            prims.append(Points(
                x=node_x, y=node_y, color=node_c, alpha=0.6, size=5,
            ))

        return prims


def _vp_bar_arrays(result: VolumeProfileResult, width_fraction: float,
                   buy_color: str, sell_color: str):
    """
    Build the parallel arrays for the VP histogram bars (session layout).

    Ports the per-profile bar geometry of ``build_volume_profile_source`` for a
    single profile anchored at ``start`` and growing right. Each price level
    yields a sell (bid) and a buy (ask) segment stacked from the anchor; levels
    inside the value area are opaque, levels outside are dimmed.

    Args:
        result (VolumeProfileResult): The computed profile snapshot.
        width_fraction (float): Max fraction of the range width used by a full
            (max-volume) bar.
        buy_color (str): Color for the buy-aggressor (ask) segment.
        sell_color (str): Color for the sell-aggressor (bid) segment.

    Returns:
        tuple of arrays (y, height, left, right, color, alpha, volume, vol_buy,
        vol_sell) in epoch-ms for left/right, or None when there is nothing to
        draw.
    """
    p = result.profile
    prices = np.asarray(p.get("prices", []), dtype=np.float64)
    if len(prices) == 0:
        return None
    vols = np.asarray(p["volumes"], dtype=np.float64)
    v_bid = np.asarray(p["vol_bid"], dtype=np.float64)
    v_ask = np.asarray(p["vol_ask"], dtype=np.float64)
    max_vol = float(vols.max()) if len(vols) else 0.0
    max_vol = max_vol or 1.0

    anchor = int(result.start)
    width = int(result.end) - int(result.start)
    profile_width = width * width_fraction
    tick_size = float(result.tick_size or 0.0)

    if tick_size > 0:
        bar_h = tick_size * 0.9
        half_h = tick_size / 2.0
    elif len(prices) > 1:
        spacing = float(np.min(np.diff(np.sort(prices))))
        bar_h = spacing * 0.9
        half_h = spacing / 2.0
    else:
        bar_h = max(abs(result.poc) * 0.001, 1.0)
        half_h = bar_h / 2.0

    has_va = np.isfinite(result.vah) and np.isfinite(result.val)
    va_lo = min(result.val, result.vah) - half_h
    va_hi = max(result.val, result.vah) + half_h

    ys, heights, lefts, rights = [], [], [], []
    colors, alphas, volumes, vol_buys, vol_sells = [], [], [], [], []

    for price, vol, vb, va in zip(prices, vols, v_bid, v_ask):
        if vol <= 0:
            continue
        bar_len = (vol / max_vol) * profile_width
        sell_len = (vb / vol) * bar_len if vol > 0 else 0.0
        in_va = has_va and (va_lo <= price <= va_hi)
        bar_alpha = 0.85 if (in_va or not has_va) else 0.25

        buy_len = bar_len - sell_len
        buy_left = anchor
        buy_right = anchor + buy_len
        sell_left = buy_right
        sell_right = anchor + bar_len

        for left, right, color, raw_vol in (
            (sell_left, sell_right, sell_color, vb),
            (buy_left, buy_right, buy_color, va),
        ):
            if raw_vol <= 0:
                continue
            ys.append(float(price))
            heights.append(bar_h)
            lefts.append(int(min(left, right)))
            rights.append(int(max(left, right)))
            colors.append(color)
            alphas.append(bar_alpha)
            volumes.append(float(vol))
            vol_buys.append(float(va))
            vol_sells.append(float(vb))

    if not ys:
        return None
    return ys, heights, lefts, rights, colors, alphas, volumes, vol_buys, vol_sells

"""
tradetropy.paper.config
=====================
Declarative configuration for the manual PaperEngine.

The PaperEngine has no programmatic strategy: the user only declares WHAT to
show (symbol, interval, optional ticks/footprint/order-book, and any indicators
they want visible while trading by hand) through a ``PaperConfig``. The
engine synthesizes an internal strategy from it (subscriptions + indicators,
empty ``on_data``) and reuses the entire live charting pipeline unchanged.

Example
-------
    from tradetropy.paper import PaperConfig, PaperIndicator, PaperEngine
    from tradetropy.ta import SMA, BollingerBands

    config = PaperConfig(
        symbol      = "BTCUSDT",
        interval_ms = 60_000,
        ticks       = True,
        footprint   = False,
        initial_balance = 10_000.0,
        indicators  = [
            SMA(20),                                   # mounted on close by default
            PaperIndicator(SMA(50), source="close", color="#F59E0B"),
            PaperIndicator(BollingerBands(20, 2.0), source="close"),
        ],
    )

    engine = PaperEngine.by_ticks(config, data=(ticks,))
    engine.run(live_chart=chart)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union

from tradetropy.ta.base import Indicator


@dataclass
class PaperIndicator:
    """
    One indicator to display in manual paper-trading mode.

    Attributes:
        indicator (Indicator): The indicator instance to plot.
        source: Where to mount it. A string token resolved against the synthesized
            strategy's proxies:
                'close' | 'open' | 'high' | 'low' | 'volume' -> the OHLC column ref
                'price'                                        -> the tick price ref
                'ohlc'                                         -> the OHLC proxy
                'ticks'                                        -> the tick proxy
            Or a ColumnRef / list[ColumnRef] passed straight through, or a
            callable ``fn(strategy) -> source`` for full control (e.g.
            ``lambda s: VolumeProfile.refs(s.ohlc)``).
        overrides (dict): Plot overrides forwarded to ``add_indicator`` (name,
            color, overlay, panel_title, ...).
    """

    indicator: Indicator
    source: Union[str, Callable, Any] = "close"
    overrides: dict = field(default_factory=dict)


# A declared indicator may be given as a bare Indicator (mounted on close), a
# PaperIndicator, or a (indicator, source[, overrides]) tuple.
IndicatorSpec = Union[Indicator, PaperIndicator, tuple]


@dataclass
class PaperConfig:
    """
    Declarative spec for a manual paper-trading session.

    Attributes:
        symbol (str): Trading symbol (e.g. "BTCUSDT").
        interval_ms (int): OHLC candle interval in milliseconds.
        ohlc_window (int): OHLC ring window size.
        ticks (bool): Subscribe the tick stream (needed for footprint / tick
            indicators and tick-precise fills).
        tick_window (int): Tick ring window size when ticks=True.
        footprint (bool | dict): Subscribe footprint. True for defaults, or a
            dict of subscribe_footprint kwargs (tick_size, levels, ...).
        orderbook_depth (int | None): If set, subscribe the L2 order book at this
            depth (enables book metrics / DeepTrades when a recorded book is fed).
        indicators (list[IndicatorSpec]): Indicators to display.
        warmup (int | None): Warmup ticks/candles. None = auto from indicators.
        initial_balance (float): Starting account balance (baseline restored on
            Restart).
        commission (float): Commission per operation for the simulated broker.
        sesh_kwargs (dict): Extra keyword args forwarded to ReplaySesh when the
            engine builds the session (e.g. use_spread, slippage_points).
    """

    symbol: str
    interval_ms: int
    ohlc_window: int = 1000
    ticks: bool = True
    tick_window: int = 5000
    footprint: Union[bool, dict] = False
    orderbook_depth: "int | None" = None
    indicators: list = field(default_factory=list)
    warmup: "int | None" = None
    initial_balance: float = 10_000.0
    commission: float = 0.0
    sesh_kwargs: dict = field(default_factory=dict)

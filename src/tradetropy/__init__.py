"""
tradetropy - professional backtesting and live trading framework.

The public API is exposed lazily (PEP 562). Importing ``tradetropy`` itself is
cheap: heavy subsystems (pandas via ``stats``/``io``, ``live``, ``plotting``,
``robustness``) are only imported the first time one of their names is
accessed. This keeps the import cost of a pure backtest/optimize workflow -
and of every spawned optimize/pool worker on Windows/macOS - close to the
numpy floor, instead of paying for plotting, live and robustness that a
worker never touches.

The exported surface is unchanged; ``from tradetropy import BacktestEngine``,
``tradetropy.SMA`` and ``from tradetropy import *`` all keep working exactly as
before.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import TYPE_CHECKING

# ``exceptions`` is tiny (no third-party imports) and is referenced as the
# submodule ``tradetropy.exceptions`` throughout the codebase, so it stays eager.
from tradetropy import exceptions

try:
    __version__ = importlib.metadata.version("tradetropy")
except importlib.metadata.PackageNotFoundError:
    # Running from a source checkout without an installed/editable metadata
    # entry (e.g. a fresh `uv run` before `uv sync`). Never let a version
    # lookup crash the import of the whole package.
    __version__ = "0.0.0"

# ══════════════════════════════════════════════════════════════════════════════
# Lazy public-symbol registry:  name -> (submodule, attribute)
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry maps a public name to the module that defines it. Nothing here is
# imported until the name is first accessed (see ``__getattr__`` below), so a
# process that only runs backtests never imports ``live``/``plotting``/
# ``robustness`` and only pays for ``pandas`` when it actually computes stats.
#
# The ``# >>> tier:N`` / ``# <<< tier:N`` marker blocks are consumed by
# tools/generate_tier.py to gate optional names out of lower-tier builds; keep
# each gated name's registry entry AND its ``__all__`` entry inside the markers.

_LAZY: dict[str, tuple[str, str]] = {
    # -- Engines ---------------------------------------------------------------
    "BacktestEngine":     ("tradetropy.backtest", "BacktestEngine"),
    "PoolBacktestEngine": ("tradetropy.backtest", "PoolBacktestEngine"),
    "LiveEngine":         ("tradetropy.live", "LiveEngine"),

    # -- Disclaimer ------------------------------------------------------------
    "LIVE_DISCLAIMER":    ("tradetropy.connectors._disclaimer", "LIVE_DISCLAIMER"),

    # -- Strategy / sessions ---------------------------------------------------
    "Strategy":           ("tradetropy.models", "Strategy"),
    "FootprintConfig":    ("tradetropy.models", "FootprintConfig"),
    "SeshSimulatorBase":  ("tradetropy.session", "SeshSimulatorBase"),

    # -- Core data types -------------------------------------------------------
    "TickData":           ("tradetropy.core", "TickData"),
    "KlineData":          ("tradetropy.core", "KlineData"),
    "parse_timeframe":    ("tradetropy.core", "parse_timeframe"),
    "TIMEFRAME_PRESETS":  ("tradetropy.core", "TIMEFRAME_PRESETS"),

    # -- Stats -----------------------------------------------------------------
    "Stats":              ("tradetropy.stats", "Stats"),
    "compute_stats":      ("tradetropy.stats", "compute_stats"),

    # -- Robustness ------------------------------------------------------------

    # -- Plotting --------------------------------------------------------------
    "plot":               ("tradetropy.plotting", "plot"),
    "PlotConfig":         ("tradetropy.plotting", "PlotConfig"),

    # -- Indicators (tradetropy.ta) ----------------------------------------------
    "Indicator":           ("tradetropy.ta", "Indicator"),
    "IndicatorPlotConfig": ("tradetropy.ta", "IndicatorPlotConfig"),
    "SMA":                 ("tradetropy.ta", "SMA"),
    "EMA":                 ("tradetropy.ta", "EMA"),
    "MACD":                ("tradetropy.ta", "MACD"),
    "RSI":                 ("tradetropy.ta", "RSI"),
    "ATR":                 ("tradetropy.ta", "ATR"),
    "BollingerBands":      ("tradetropy.ta", "BollingerBands"),
    "VolumeProfile":       ("tradetropy.ta", "VolumeProfile"),
    "RollingVolumeProfile": ("tradetropy.ta", "RollingVolumeProfile"),
    "VolumeNode":          ("tradetropy.ta", "VolumeNode"),
    "detect_volume_nodes": ("tradetropy.ta", "detect_volume_nodes"),
    "ZigZag":              ("tradetropy.ta", "ZigZag"),
    "ConfirmedPivot":      ("tradetropy.ta", "ConfirmedPivot"),
    "PivotHighLow":        ("tradetropy.ta", "PivotHighLow"),
    "SwingHL":             ("tradetropy.ta", "SwingHL"),
    "EqualHL":             ("tradetropy.ta", "EqualHL"),
    "FairValueGap":        ("tradetropy.ta", "FairValueGap"),
    "OrderBlock":          ("tradetropy.ta", "OrderBlock"),
    "MarketSessions":      ("tradetropy.ta", "MarketSessions"),
    "SessionLevels":       ("tradetropy.ta", "SessionLevels"),
    "KillZones":           ("tradetropy.ta", "KillZones"),
    "LargeTrades":         ("tradetropy.ta", "LargeTrades"),
}

__all__ = [
    "__version__",
    "exceptions",
    "BacktestEngine",
    "PoolBacktestEngine",
    "LiveEngine",
    "LIVE_DISCLAIMER",
    "Strategy",
    "FootprintConfig",
    "SeshSimulatorBase",
    "TickData",
    "KlineData",
    "parse_timeframe",
    "TIMEFRAME_PRESETS",
    "Stats",
    "compute_stats",
    "plot",
    "PlotConfig",
    "Indicator",
    "IndicatorPlotConfig",
    "SMA",
    "EMA",
    "MACD",
    "RSI",
    "ATR",
    "BollingerBands",
    "VolumeProfile",
    "RollingVolumeProfile",
    "VolumeNode",
    "detect_volume_nodes",
    "ZigZag",
    "ConfirmedPivot",
    "PivotHighLow",
    "SwingHL",
    "EqualHL",
    "FairValueGap",
    "OrderBlock",
    "MarketSessions",
    "SessionLevels",
    "KillZones",
    "LargeTrades",
]


def __getattr__(name: str):
    """
    Resolve a public name lazily (PEP 562).

    Looks the name up in the lazy registry, imports the owning submodule on
    first access, caches the resolved object in the module globals (so later
    lookups skip this path), and returns it. Falls back to importing a
    same-named submodule, then raises AttributeError.
    """
    entry = _LAZY.get(name)
    if entry is not None:
        module_path, attr = entry
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value

    # Fall back to a real submodule access (e.g. ``tradetropy.core``).
    try:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    except ImportError:
        pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the full public surface to ``dir()`` and tab completion."""
    return sorted(set(globals()) | set(_LAZY))


# Give static analysers / IDEs the real symbols without any runtime import cost.
if TYPE_CHECKING:  # pragma: no cover
    from tradetropy.backtest import BacktestEngine, PoolBacktestEngine
    from tradetropy.live import LiveEngine, LivePool, Recorder
    from tradetropy.connectors._disclaimer import LIVE_DISCLAIMER
    from tradetropy.models import Strategy, FootprintConfig
    from tradetropy.session import SeshSimulatorBase
    from tradetropy.core import TickData, KlineData, parse_timeframe, TIMEFRAME_PRESETS
    from tradetropy.stats import Stats, compute_stats
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig, MonteCarloResult
    from tradetropy.plotting import plot, PlotConfig
    from tradetropy.ta import (
        Indicator, IndicatorPlotConfig, SMA, EMA, MACD, RSI, ATR,
        BollingerBands, VolumeProfile, TickVolumeProfile, RollingVolumeProfile,
        VolumeNode, detect_volume_nodes, ZigZag, ConfirmedPivot, PivotHighLow,
        SwingHL, EqualHL, FairValueGap, OrderBlock, MarketSessions,
        SessionLevels, KillZones, LargeTrades,
    )

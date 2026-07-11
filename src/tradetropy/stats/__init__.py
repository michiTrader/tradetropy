"""
tradetropy.stats - backtest performance metrics.

The public names (``Stats``, ``compute_stats``, ...) live in ``stats.py`` which
imports pandas. They are exposed lazily (PEP 562) so that importing a
pandas-free sibling - notably ``tradetropy.stats._fast.compute_stats_fast`` used by
the optimize/pool worker hot path - does NOT drag pandas into the process just
by triggering this package's ``__init__``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY = {
    "Stats": "stats",
    "compute_stats": "stats",
    "drawdown_series": "stats",
    "calc_daily_equity": "stats",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is not None:
        mod = importlib.import_module(f"{__name__}.{module}")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # pragma: no cover
    from tradetropy.stats.stats import (
        Stats, compute_stats, drawdown_series, calc_daily_equity,
    )

"""
tradetropy.paper
=============
Manual, forward-only market simulation (paper trading / discretionary practice).

    from tradetropy.paper import PaperConfig, PaperIndicator, PaperEngine

    config = PaperConfig(symbol="BTCUSDT", interval_ms=60_000, ticks=True)
    engine = PaperEngine.by_ticks(config, data=(ticks,))
    engine.run(live_chart=chart)
"""

from tradetropy.paper.config import PaperConfig, PaperIndicator
from tradetropy.paper.engine import PaperEngine, build_manual_strategy_class

__all__ = [
    "PaperConfig",
    "PaperIndicator",
    "PaperEngine",
    "build_manual_strategy_class",
]

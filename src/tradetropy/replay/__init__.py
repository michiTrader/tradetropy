"""
tradetropy.replay
==============
Historical data replay system with real-time charting.

Main public API
---------------
    from tradetropy.replay import ReplayEngine

    engine = ReplayEngine.by_ticks(strategy, data=(ticks,))
    engine = ReplayEngine.by_klines(strategy, data=(klines,))

Low-level API (advanced cases)
------------------------------
    from tradetropy.replay import ReplaySesh, ReplayController
"""

from tradetropy.replay.engine import ReplayEngine
from tradetropy.replay.replay_sesh import ReplaySesh
from tradetropy.replay.controller import ReplayController, PlaybackController

__all__ = [
    "ReplayEngine",
    "ReplaySesh",
    "ReplayController",
    "PlaybackController",
]
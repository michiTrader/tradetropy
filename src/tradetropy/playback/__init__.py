"""
tradetropy.playback
================
Shared foundation for the cursor-driven playback engines.

    from tradetropy.playback import BaseEngine

Concrete engines:
    - tradetropy.replay.ReplayEngine (automated, bi-directional)
    - tradetropy.paper.PaperEngine   (manual, forward-only)
"""

from tradetropy.playback.base import BaseEngine

__all__ = ["BaseEngine"]

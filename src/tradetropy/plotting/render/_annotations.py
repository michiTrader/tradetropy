"""
(Removed) Per-type annotation renderers.

Every annotation - FairValueGap, OrderBlock, MarketSessions, SwingHL, EqualHL
and LargeTrades - now emits declarative draw primitives via ``Indicator.draw()``
and is drawn by the generic primitive renderer (``render/_tools.py``). There is
no per-type annotation rendering code anymore; this module is intentionally
empty and kept only to avoid breaking stale imports during transition.
"""

from __future__ import annotations

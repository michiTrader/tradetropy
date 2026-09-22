"""
Registry of documentation demos.

A documentation demo is a small, self-contained backtest that doubles as the
interactive example shown on the docs site and README. Each demo is declared
ONCE in ``registry.py`` as a canonical, runnable snippet (the single source of
truth, shown verbatim in the docs) plus metadata (id, title). Building a
demo executes that exact snippet and returns the finished engine, so the
generated artifacts (chart, stats, snippet - see ``tools/gen_docs_demo.py``)
can never drift from one another or from what the user reads.
"""

from __future__ import annotations

from .registry import (
    DEMOS,
    DemoResult,
    DemoSpec,
    get_demo,
)

__all__ = [
    "DEMOS",
    "DemoResult",
    "DemoSpec",
    "get_demo",
]

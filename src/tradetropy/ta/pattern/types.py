"""
types.py
========
Base types for the pivot pattern matching system.

PivotPoint — a confirmed pivot with all its decorator indicator tags.
MatchResult — the result of a successful Pattern match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING


# ══════════════════════════════════════════════════════════════════════════════
# PIVOT POINT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PivotPoint:
    """
    Represents a confirmed pivot with all its tags.

    Attributes
    ──────────
    index     : bar index in the OhlcStore (confirmation bar)
    timestamp : ts_ms of the REAL pivot bar (not the confirmation bar)
    value     : pivot price
    type      : 'H' or 'L'
    tags      : dict with tags from each decorator indicator
                {
                    "type": "H",        ← always present (from ConfirmedPivot base)
                    "nbs":  "neu",      ← from NBS indicator
                    "hhll": "HH",       ← from HHLL indicator
                }

    Notes
    ─────
    · frozen=True → immutable, hashable, safe for sharing between workers.
    · tags["type"] == self.type always — intentionally duplicated
      so PatternNode conditions can reference another node's type
      via tags without special cases.
    """

    index:     int
    timestamp: float
    value:     float
    type:      Literal['H', 'L']
    tags:      dict[str, str]

    def __repr__(self) -> str:
        tags_str = ", ".join(f"{k}={v}" for k, v in self.tags.items() if k != "type")
        return (
            f"PivotPoint({self.type} @ bar={self.index} "
            f"val={self.value:.2f}"
            + (f" [{tags_str}]" if tags_str else "")
            + ")"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MATCH RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MatchResult:
    """
    Result of a successful Pattern match on a pivot sequence.

    Attributes
    ──────────
    tag   : tag of the Pattern that produced this match
    nodes : tuple of PivotPoint in pattern order

    Access
    ──────
    match[0]         → first node (PivotPoint)
    match[-1]        → last node
    match.first      → first node
    match.last_node  → last node
    match.indices    → [45, 48, 52]
    match.values     → [3500.0, 3200.0, 3600.0]
    match.timestamps → [t1, t2, t3]

    Example
    ───────
        match = self.setup.last
        if match:
            n0 = match[0]
            n0.value        # 3500.0
            n0.index        # 45
            n0.timestamp    # 1234567890000
            n0.type         # "H"
            n0.tags         # {"type": "H", "nbs": "neu"}

            match.tag           # "strong_setup"
            match.indices       # [45, 48, 52]
            match.values        # [3500.0, 3200.0, 3600.0]
            match.timestamps    # [t1, t2, t3]
    """

    tag:   str
    nodes: tuple[PivotPoint, ...]
    node_map: dict[int, PivotPoint | None] | None = None

    # ── Access by position in the original pattern ──────────────────────────────

    def node_at(self, pattern_idx: int) -> 'PivotPoint | None':
        """
        Returns the node at position *pattern_idx* in the original pattern.

        Works for both captured and non-captured nodes.
        Returns None if the node was optional and is absent.
        Raises IndexError if *pattern_idx* is out of range.
        """
        if self.node_map is None:
            return self.nodes[pattern_idx]
        return self.node_map[pattern_idx]

    @property
    def all_nodes(self) -> 'tuple[PivotPoint | None, ...]':
        """
        All nodes of the original pattern in order, including non-captured
        and absent optionals (as None).
        """
        if self.node_map is None:
            return self.nodes
        n = max(self.node_map.keys()) + 1
        return tuple(self.node_map.get(i) for i in range(n))

    # ── Optional node properties ────────────────────────────────────────────────

    @property
    def matched_optional(self) -> dict[int, bool] | None:
        """Dict {node_idx: bool} for each optional node, or None if no optionals."""
        if self.node_map is None:
            return None
        return {k: v is not None for k, v in self.node_map.items()}

    # ── Access by index (captured nodes only) ──────────────────────────────────

    def __getitem__(self, idx: int) -> PivotPoint:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    # ── Convenience properties (over captured nodes) ──────────────────────────

    @property
    def first(self) -> PivotPoint:
        """First captured node of the match."""
        return self.nodes[0]

    @property
    def last_node(self) -> PivotPoint:
        """Last captured node of the match."""
        return self.nodes[-1]

    @property
    def indices(self) -> list[int]:
        """Bar indices of each captured node."""
        return [n.index for n in self.nodes]

    @property
    def values(self) -> list[float]:
        """Values (prices) of each captured node."""
        return [n.value for n in self.nodes]

    @property
    def timestamps(self) -> list[float]:
        """Timestamps (ms) of each captured node."""
        return [n.timestamp for n in self.nodes]

    def __repr__(self) -> str:
        nodes_str = " >> ".join(
            f"{n.type}({n.value:.2f})" for n in self.nodes
        )
        parts = [f"tag={self.tag!r}", nodes_str]
        if self.matched_optional is not None:
            absent = [str(i) for i, present in self.matched_optional.items() if not present]
            if absent:
                parts.append(f"absent_optionals=[{','.join(absent)}]")
        if self.node_map is not None:
            nc_keys = [str(i) for i, n in self.node_map.items() if n is not None and n not in self.nodes]
            if nc_keys:
                parts.append(f"non_captured=[{','.join(nc_keys)}]")
        return f"MatchResult({', '.join(parts)})"

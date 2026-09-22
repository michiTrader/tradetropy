"""
store.py
========
PatternStore — precalculates all matches over the complete pivot sequence
once. Read-only after construction.

In backtest and optimization it is built once and shared between
workers (fork on Linux, serialization on Windows/macOS).

In live it is built from history and updated incrementally
when a new pivot is confirmed.

Complexity
──────────
Construction : O(n_pivots × L) where L = pattern length (typically 2-8)
Lookup       : O(log n_matches) via binary search
Live update  : O(L) per new confirmed pivot
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tradetropy.ta.pattern.types    import MatchResult, PivotPoint
from tradetropy.ta.pattern.sequence import FrozenPivotSequence

if TYPE_CHECKING:
    from tradetropy.ta.pattern.pattern import Pattern


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN STORE
# ══════════════════════════════════════════════════════════════════════════════

class PatternStore:
    """
    Precalculates and stores all matches of a Pattern over a sequence.

    Internal attributes
    ───────────────────
    _sequence      : FrozenPivotSequence — the pivot sequence
    _pattern       : Pattern — the pattern to search for
    _matches       : list[MatchResult | None] — result per pivot
                     _matches[i] = MatchResult if the pattern ENDS at pivot i
    _match_bar_indices : array of bar indices where there is a match
                         for O(log n) binary search
    """

    __slots__ = (
        '_sequence',
        '_pattern',
        '_matches',
        '_match_bar_indices',
    )

    def __init__(self, sequence: FrozenPivotSequence, pattern: 'Pattern'):
        self._sequence = sequence
        self._pattern  = pattern
        self._matches: list[MatchResult | None] = self._scan_all()
        self._match_bar_indices: np.ndarray = self._build_match_index()

    # ── Construction ──────────────────────────────────────────────────────────

    def _scan_all(self) -> list[MatchResult | None]:
        """
        Scans the entire sequence looking for the pattern.
        O(n_pivots × L) — runs once.
        """
        pivots  = self._sequence.pivots
        pattern = self._pattern
        n       = len(pivots)
        L       = pattern.length
        min_len = pattern.min_length
        results: list[MatchResult | None] = [None] * n

        if n < min_len:
            return results

        for i in range(min_len - 1, n):
            # Try windows from longest to shortest — prefer more complete match
            for w_len in range(L, min_len - 1, -1):
                start = i - w_len + 1
                if start < 0:
                    continue
                window = pivots[start : i + 1]
                match  = pattern.try_match(window)
                if match is not None:
                    results[i] = match
                    break

        return results

    def _build_match_index(self) -> np.ndarray:
        """
        Builds array of bar indices where there is a match.
        Used for binary search in last_match_at().
        """
        indices = [
            m.last_node.index
            for m in self._matches
            if m is not None
        ]
        return np.array(indices, dtype=np.int64)

    # ── Lookup (backtest) ─────────────────────────────────────────────────────

    def last_match_at(self, bar_index: int) -> 'MatchResult | None':
        """
        Returns the most recent match whose last node is at a bar
        <= bar_index.

        O(log n_matches) via binary search over _match_bar_indices.
        Returns None if there is no match up to bar_index.
        """
        if len(self._match_bar_indices) == 0:
            return None

        # bisect_right(arr, bar_index) - 1 → index of last <= bar_index
        pos = bisect.bisect_right(self._match_bar_indices, bar_index) - 1
        if pos < 0:
            return None

        # find the MatchResult corresponding to that bar index
        target_bar = int(self._match_bar_indices[pos])

        # look in _matches for the result with last_node.index == target_bar
        # pivot_idx in _sequence gives us the direct position in _matches
        pivot_idx = self._sequence.pivot_idx_at_bar(target_bar)
        if pivot_idx < 0 or pivot_idx >= len(self._matches):
            return None

        match = self._matches[pivot_idx]
        if match is not None and self._pattern.anchor_end:
            last_pivot_idx = self._sequence.pivot_idx_at_bar(bar_index)
            if pivot_idx != last_pivot_idx:
                return None
        return match

    # ── Incremental update (live) ─────────────────────────────────────────────

    def append_pivot(self, pivot: PivotPoint) -> None:
        """
        Adds a new confirmed pivot and evaluates if it completes the pattern.
        Called by LiveEngine when a new pivot is confirmed.

        O(L) — only evaluates the window ending at the new pivot.
        """
        self._sequence.pivots.append(pivot)

        # update the sequence's bar index array
        new_bar_idx = np.array([pivot.index], dtype=np.int64)
        self._sequence._pivot_bar_indices = np.concatenate([
            self._sequence._pivot_bar_indices,
            new_bar_idx,
        ])

        n = len(self._sequence.pivots)
        L = self._pattern.length
        min_len = self._pattern.min_length

        if n < min_len:
            self._matches.append(None)
            return

        # Try windows from longest to shortest
        for w_len in range(L, min_len - 1, -1):
            if n < w_len:
                continue
            window = self._sequence.pivots[n - w_len : n]
            match  = self._pattern.try_match(window)
            if match is not None:
                self._matches.append(match)
                self._match_bar_indices = np.append(
                    self._match_bar_indices,
                    np.int64(pivot.index),
                )
                return

        self._matches.append(None)

    @property
    def last_match(self) -> 'MatchResult | None':
        """
        Last match of the entire sequence.
        Used in live — always reflects the most recent state.

        Respects anchor_end like last_match_at (backtest): if the pattern
        is anchored to the end, the match is only valid when it ENDS at the
        last confirmed pivot. In live the sequence only contains pivots
        up to "now", so that pivot is self._matches[-1]. Without this
        symmetry the anchored match would stick to the first result
        even if new pivots are confirmed that don't complete the pattern.
        """
        if not self._matches:
            return None
        if self._pattern.anchor_end:
            return self._matches[-1]
        for m in reversed(self._matches):
            if m is not None:
                return m
        return None

    def __repr__(self) -> str:
        n_matches = sum(1 for m in self._matches if m is not None)
        return (
            f"PatternStore("
            f"pattern={self._pattern.tag!r}, "
            f"n_pivots={len(self._sequence)}, "
            f"n_matches={n_matches})"
        )
        

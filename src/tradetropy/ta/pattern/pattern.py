"""
pattern.py
==========
PatternNode and Pattern — the declarative description of a pivot pattern.

A pattern is a sequence of PatternNode. Each node describes what type
of pivot is expected, what tags it must have and what numeric conditions
its value must satisfy.

Optional nodes
──────────────
A PatternNode can be marked with ``optional=True``. An optional node
may be present or absent in the match; the pattern is considered valid
in both cases, as long as all mandatory nodes match.

    Pattern([
        PatternNode('L', {'nbs': 'boo'}, []),
        PatternNode('H', {'nbs': 'neu'}, [], optional=True),   # ← may not exist
        PatternNode('L', {'nbs': 'boo'}, [Condition('>', NodeRef(0, 'value'))]),
        PatternNode('H', {'nbs': 'neu'}, [Condition('>', NodeRef(2, 'value'))]),
    ], tag="setup_flexible")

Subsequence semantics
─────────────────────
``try_match`` always receives a window of size between
``n_mandatory`` and ``len(nodes)``. Internally it tries all possible
assignments of pivots from the window to pattern nodes that
respect the original order and keep all mandatory nodes present.

Conditions using NodeRef ($N) reference positions in the
original pattern, not in the compressed window. When node N
is optional and absent, the NodeRef safely returns False.

Conditions with NodeRef on absent optional nodes
────────────────────────────────────────────────
If node A references ``NodeRef(B, 'value')`` and B is optional
and absent, the condition returns False. This means that if a
mandatory node has a condition depending on an absent optional
node, that match attempt fails and the next one is tried.

Example
───────
    from tradetropy.ta.pattern import Pattern, PatternNode, Condition, NodeRef, ConditionOr, ConditionAnd, NodeTypeRef

    # simple pattern — bullish impulse
    Pattern([
        PatternNode('L', {}, []),
        PatternNode('H', {}, [Condition('>', NodeRef(0, 'value'))]),
    ], tag="impulse")

    # pattern with optional central node (the pullback may or may not exist)
    Pattern([
        PatternNode('H', {'nbs': 'neu'}, []),
        PatternNode('L', {'nbs': 'boo'}, [], optional=True),   # optional pullback
        PatternNode('H', {'nbs': 'neu'}, [Condition('>', NodeRef(0, 'value'))]),
    ], tag="hh_with_optional_pullback")

    # pattern with OR — any neutralizer in any direction
    Pattern([
        PatternNode('any', {'nbs': 'neu'}, []),
        PatternNode('any', {}, []),
        PatternNode('any', {'nbs': 'neu'}, [
            ConditionOr([
                ConditionAnd([
                    Condition('==', NodeTypeRef(0)),
                    Condition('<',  NodeRef(0, 'value')),
                ]),
                ConditionAnd([
                    Condition('!=', NodeTypeRef(0)),
                    Condition('>',  NodeRef(0, 'value')),
                ]),
            ])
        ]),
    ], tag="reversal")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, TYPE_CHECKING

from tradetropy.exceptions import PatternSyntaxError
from tradetropy.ta.pattern.conditions import (
    ConditionExpr,
    evaluate_condition,
    Condition,
    RangeCondition,
    TimeCondition,
    TagCondition,
    ConditionAnd,
    ConditionOr,
    NodeRef,
    NodeTypeRef,
    CurrentRef,
    BinaryExpr,
    AbsExpr,
)

if TYPE_CHECKING:
    from tradetropy.ta.pattern.types import PivotPoint, MatchResult


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE VALIDATION (recursive collectors)
# ══════════════════════════════════════════════════════════════════════════════

def _iter_value_positions(expr) -> 'list[int]':
    """Positions of NodeRef contained in a value expression."""
    out: list[int] = []
    if isinstance(expr, NodeRef):
        out.append(expr.position)
    elif isinstance(expr, AbsExpr):
        out.extend(_iter_value_positions(expr.operand))
    elif isinstance(expr, BinaryExpr):
        out.extend(_iter_value_positions(expr.left))
        out.extend(_iter_value_positions(expr.right))
    # float, int, CurrentRef → no referenced position
    return out


def _iter_ref_positions(expr: 'ConditionExpr') -> 'list[int]':
    """
    Recursively traverses a condition expression and returns all
    referenced node positions (NodeRef, NodeTypeRef, CurrentRef does not
    count as it points to the current node).

    Covers lhs/operand of Condition, the three operands of RangeCondition,
    the ref of TimeCondition('hours_since') and And/Or nesting.
    """
    out: list[int] = []

    if isinstance(expr, Condition):
        if expr.lhs is not None:
            out.extend(_iter_value_positions(expr.lhs))
        if isinstance(expr.operand, NodeTypeRef):
            out.append(expr.operand.position)
        else:
            out.extend(_iter_value_positions(expr.operand))

    elif isinstance(expr, RangeCondition):
        out.extend(_iter_value_positions(expr.expr))
        out.extend(_iter_value_positions(expr.low))
        out.extend(_iter_value_positions(expr.high))

    elif isinstance(expr, TimeCondition):
        if expr.ref is not None:
            out.append(expr.ref.position)

    elif isinstance(expr, (ConditionAnd, ConditionOr)):
        for c in expr.conditions:
            out.extend(_iter_ref_positions(c))

    # TagCondition → no references to other nodes
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN NODE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatternNode:
    """
    Describes a pivot within a pattern.

    Parameters
    ──────────
    type        : expected pivot type.
                  · 'H'   → highs only
                  · 'L'   → lows only
                  · 'any' → any type (H or L)
                  Note: H/L alternation is guaranteed by ConfirmedPivot,
                  so 'any' is only useful when the type doesn't matter but
                  tags or conditions do.

    tag_filters : dict of tags the pivot must satisfy.
                  Each key is the decorator indicator name and the value
                  can be:
                  · str        — single exact value:      {'nbs': 'neu'}
                  · set[str]   — multiple alternative values: {'nbs': {'neu', 'shk'}}
                  Examples:
                    {'nbs': 'neu'}                → NBS neutralizers only
                    {'nbs': {'neu', 'shk'}}       → NBS neutralizer OR shocker
                    {'nbs': 'neu', 'hhll': 'HH'}  → AND of two decorators
                    {}                            → no tag filter

    conditions  : list of ConditionExpr evaluated with implicit AND.
                  Can be Condition, ConditionAnd, ConditionOr or nested.
                  Examples:
                    [Condition('>', 3500)]                    → absolute value
                    [Condition('>', NodeRef(0, 'value'))]     → relative to node $0
                    [ConditionOr([...])]                      → explicit OR

    optional    : if True, this node may be present or absent in the match.
                  The pattern is considered valid in both cases.
                  Default False (mandatory node).

    Notes
    ─────
    · The conditions list is implicit AND — all must be satisfied.
    · For OR between conditions, use explicit ConditionOr.
    · Conditions are evaluated AFTER type and tag filters.
      If type or tags don't match, conditions are not evaluated.
    · NodeRef($N) in conditions always references position N of the
      original pattern. If node N is optional and absent, the condition
      safely returns False (the match attempt fails).
    """

    type:        Literal['H', 'L', 'any']
    tag_filters: 'dict[str, str | set[str]]'
    conditions:  list[ConditionExpr] = field(default_factory=list)
    optional:    bool = False
    captured:    bool = True

    def matches(
        self,
        pivot:       'PivotPoint',
        node_idx:    int,
        full_window: 'list[PivotPoint | None]',
    ) -> bool:
        """
        Evaluates whether a PivotPoint satisfies this pattern node.

        Parameters
        ──────────
        pivot       : the candidate pivot
        node_idx    : position of this node in the full pattern (pattern index,
                      not compressed window index).
        full_window : list of size == len(pattern.nodes). Each position is the
                      PivotPoint assigned to that pattern node, or None if the
                      corresponding optional node is absent.

        Returns True if type, tags and all conditions are satisfied.
        """
        # ── 1. type filter ──────────────────────────────────────────────────
        if self.type != 'any' and pivot.type != self.type:
            return False

        # ── 2. tag filter ──────────────────────────────────────────────────
        for tag_key, tag_val in self.tag_filters.items():
            actual = pivot.tags.get(tag_key)
            if isinstance(tag_val, (set, frozenset)):
                if actual not in tag_val:
                    return False
            else:
                if actual != tag_val:
                    return False

        # ── 3. conditions — implicit AND ────────────────────────────────────
        for cond in self.conditions:
            if not evaluate_condition(cond, node_idx, full_window):
                return False

        return True

    def __repr__(self) -> str:
        parts = [f"type={self.type!r}"]
        if not self.captured:
            parts.append("captured=False")
        if self.optional:
            parts.append("optional=True")
        if self.tag_filters:
            tags_str = ", ".join(f"{k}={v!r}" for k, v in self.tag_filters.items())
            parts.append(f"tags={{{tags_str}}}")
        if self.conditions:
            parts.append(f"conditions=[{len(self.conditions)}]")
        return f"PatternNode({', '.join(parts)})"


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pattern:
    """
    Ordered sequence of PatternNode with an identifying tag.

    Parameters
    ──────────
    nodes : list of PatternNode in chronological order.
            The first node is the oldest, the last the most recent.
            Some nodes may have ``optional=True``.
    tag   : name assigned to the MatchResult when the pattern matches.

    Properties
    ──────────
    length       : total number of nodes (mandatory + optional).
    min_length   : number of mandatory nodes — minimum window size.
    has_optional : True if at least one node is optional.

    Notes on optional nodes
    ───────────────────────
    · A Pattern with N nodes and K optionals searches windows of size
      between N-K and N over the pivot sequence.
    · ``try_match`` tries all valid assignments of pivots to pattern nodes
      that respect the original order. Returns the first match found
      (more covered optionals has priority: iterates from longer to shorter windows).
    · NodeRef($i) in conditions always references position ``i`` of the
      full pattern. If node ``i`` is optional and absent in a candidate
      assignment, any condition referencing it safely returns False
      (that candidate is discarded).

    Examples
    ────────
        # minimum — 2 nodes
        Pattern([
            PatternNode('L', {}, []),
            PatternNode('H', {}, [Condition('>', NodeRef(0, 'value'))]),
        ], tag="basic_impulse")

        # with optional center node
        Pattern([
            PatternNode('H', {'nbs': 'neu'}, []),
            PatternNode('L', {'nbs': 'boo'}, [], optional=True),
            PatternNode('H', {'nbs': 'neu'}, [Condition('>', NodeRef(0, 'value'))]),
        ], tag="hh_optional_pullback")
    """

    nodes: list[PatternNode]
    tag:   str
    anchor_end: bool = False
    active_start: int = 0
    active_end: 'int | None' = None

    def __post_init__(self):
        if len(self.nodes) < 2:
            raise PatternSyntaxError(
                f"Pattern '{self.tag}' must have at least 2 nodes, "
                f"received {len(self.nodes)}."
            )
        if self.min_length < 2:
            raise PatternSyntaxError(
                f"Pattern '{self.tag}' must have at least 2 mandatory nodes, "
                f"but has {self.min_length}.  Cannot mark all nodes as optional."
            )
        if not any(n.captured for n in self.nodes):
            raise PatternSyntaxError(
                f"Pattern '{self.tag}': all nodes are non-captured (~). "
                f"At least one node must be captured."
            )

        self._validate_active_window()

        # Early validation of NodeRef/NodeTypeRef references: detects typos
        # like $5 in a 3-node pattern before running the matching (which
        # would silently fail).
        self._validate_node_refs()

    # ── Active window ─────────────────────────────────────────────────────────

    def _validate_active_window(self) -> None:
        """
        Validate the ``@active_bars(start, end)`` activation window.

        The window is measured in OHLC bars elapsed since the confirmation
        bar of the last matched pivot (age 0 == confirmation bar), so both
        bounds must be non-negative and ``end`` cannot precede ``start``.
        """
        if self.active_start < 0:
            raise PatternSyntaxError(
                f"Pattern '{self.tag}': @active_bars start must be >= 0, "
                f"received {self.active_start}."
            )
        if self.active_end is not None:
            if self.active_end < 0:
                raise PatternSyntaxError(
                    f"Pattern '{self.tag}': @active_bars end must be >= 0, "
                    f"received {self.active_end}."
                )
            if self.active_end < self.active_start:
                raise PatternSyntaxError(
                    f"Pattern '{self.tag}': @active_bars end ({self.active_end}) "
                    f"cannot be smaller than start ({self.active_start})."
                )

    @property
    def has_active_window(self) -> bool:
        """True if an ``@active_bars`` directive narrows the activation window."""
        return self.active_start != 0 or self.active_end is not None

    def active_age_ok(self, age: int) -> bool:
        """
        Whether a match is active at the given bar ``age``.

        ``age`` is the number of OHLC bars elapsed since the confirmation
        bar of the pattern's last matched pivot (age 0 == confirmation bar).
        With the default window (start 0, end None) every non-negative age
        is active, so patterns without an ``@active_bars`` directive are
        unaffected.

        Args:
            age (int): Bars elapsed since the last pivot's confirmation bar.

        Returns:
            bool: True if the match is active at that age.
        """
        if age < self.active_start:
            return False
        if self.active_end is not None and age > self.active_end:
            return False
        return True

    # ── Reference validation ──────────────────────────────────────────────────

    def _validate_node_refs(self) -> None:
        """
        Traverses all conditions and verifies that each NodeRef/NodeTypeRef
        points to an existing pattern position.

        · Absolute position (>= 0): must be < len(nodes).
        · Relative position (< 0): resolved from the containing node
          (node_idx + position) cannot fall before the start (< 0).

        Raises PatternSyntaxError with a clear message if an invalid
        reference is found. This eliminates an entire class of silent bugs
        (pattern never matches with no explanation).
        """
        n = len(self.nodes)
        for node_idx, node in enumerate(self.nodes):
            for cond in node.conditions:
                for position in _iter_ref_positions(cond):
                    if position >= 0:
                        if position >= n:
                            raise PatternSyntaxError(
                                f"Pattern '{self.tag}': node {node_idx} references "
                                f"${position}, but the pattern only has {n} nodes "
                                f"(valid indices 0..{n - 1}).  Possible typo in $N?"
                            )
                    else:
                        resolved = node_idx + position
                        if resolved < 0:
                            raise PatternSyntaxError(
                                f"Pattern '{self.tag}': node {node_idx} uses the "
                                f"relative reference ${position}, which points before "
                                f"the start of the pattern (position {resolved}).  "
                                f"From node {node_idx} the minimum valid relative "
                                f"reference is ${-node_idx}."
                            )

    @property
    def length(self) -> int:
        """Total number of nodes (mandatory + optional)."""
        return len(self.nodes)

    @property
    def min_length(self) -> int:
        """Number of mandatory nodes — minimum valid window size."""
        return sum(1 for n in self.nodes if not n.optional)

    @property
    def has_optional(self) -> bool:
        """True if at least one node is optional."""
        return any(n.optional for n in self.nodes)

    # ── Optional node indices (computed once) ─────────────────────────────────

    @property
    def _optional_indices(self) -> tuple[int, ...]:
        """Indices (in self.nodes) of optional nodes."""
        return tuple(i for i, n in enumerate(self.nodes) if n.optional)

    # ── Matching ──────────────────────────────────────────────────────────────

    def try_match(self, window: 'list[PivotPoint]') -> 'MatchResult | None':
        """
        Attempts to match against a window of pivots.

        Parameters
        ──────────
        window : list of PivotPoint of size ``w`` where
                 ``min_length <= w <= length``.

        Returns MatchResult if any valid assignment matches, None otherwise.

        Strategy
        ────────
        If the pattern has no optionals, behavior identical to the original:
        the window must have exactly ``length`` pivots.

        If the pattern has optionals, iterates over all subsets of optional
        nodes that could be OMITTED, starting with the empty subset (no
        optional omitted — most complete match first). For each subset
        checks if a window of size ``w`` can be assigned to the non-omitted
        nodes.

        The window passed to ``try_match`` always ends at the most recent
        pivot of the candidate. ``PatternStore._scan_all`` handles passing
        the correct window size for each attempt.
        """
        from tradetropy.ta.pattern.types import MatchResult

        w = len(window)
        L = self.length

        if not self.has_optional:
            # Fast path — no optionals
            if w != L:
                return None
            full_window: list[PivotPoint | None] = list(window)  # type: ignore[assignment]
            for node_idx, (node_def, pivot) in enumerate(zip(self.nodes, window)):
                if not node_def.matches(pivot, node_idx, full_window):
                    return None

            has_nc = any(not n.captured for n in self.nodes)
            if not has_nc:
                # No optionals or non-captured: exact original path
                return MatchResult(tag=self.tag, nodes=tuple(window))

            # No optionals but with non-captured
            captured_nodes = tuple(
                pivot for node_def, pivot in zip(self.nodes, window)
                if node_def.captured
            )
            node_map = {i: window[i] for i in range(L)}
            return MatchResult(tag=self.tag, nodes=captured_nodes, node_map=node_map)

        # ── Path with optionals ──────────────────────────────────────────────
        #
        # The window has ``w`` pivots and we want to assign them to ``w`` nodes
        # of the pattern (the ``L - (L - w)`` that we don't omit).
        #
        # Strategy:
        #   · n_omit = L - w  → how many optionals we omit.
        #   · Iterate over the C(n_optional, n_omit) subsets of
        #     optionals to omit.
        #   · For each subset, the active nodes (not omitted) must
        #     be exactly ``w``.
        #   · Assign window[0..w-1] to the active nodes in order.
        #   · Build full_window of size L with None at omitted positions
        #     and the corresponding pivot at active positions.
        #   · Evaluate each active node with full_window.

        optional_idxs = self._optional_indices
        n_omit = L - w

        if n_omit < 0 or n_omit > len(optional_idxs):
            # Window too long or more omissions than possible
            return None

        # Iterate subsets of optionals to OMIT, from fewer to more omitted
        # (since we call try_match with the exact window, n_omit is fixed here)
        for omit_subset in combinations(optional_idxs, n_omit):
            omit_set = set(omit_subset)

            # Active nodes in pattern order
            active_node_idxs = [i for i in range(L) if i not in omit_set]
            # Must match the window count
            if len(active_node_idxs) != w:
                continue

            # Build full_window: L slots, None where omitted
            full_window_opt: list[PivotPoint | None] = [None] * L
            for slot, node_idx in enumerate(active_node_idxs):
                full_window_opt[node_idx] = window[slot]

            # Evaluate each active node
            ok = True
            for slot, node_idx in enumerate(active_node_idxs):
                node_def = self.nodes[node_idx]
                pivot    = window[slot]
                if not node_def.matches(pivot, node_idx, full_window_opt):
                    ok = False
                    break

            if ok:
                # node_map: covers all nodes if there are non-captured,
                # or only optionals if all are captured
                has_nc = any(not n.captured for n in self.nodes)
                if has_nc or optional_idxs:
                    node_map: dict[int, PivotPoint | None] = {
                        i: full_window_opt[i] for i in range(L)
                    }
                else:
                    node_map = {}

                # nodes of MatchResult: only captured participants (no None)
                matched_nodes = tuple(
                    full_window_opt[i]
                    for i in range(L)
                    if full_window_opt[i] is not None and self.nodes[i].captured
                )
                return MatchResult(
                    tag=self.tag,
                    nodes=matched_nodes,
                    node_map=node_map if node_map else None,
                )

        return None

    def __repr__(self) -> str:
        nodes_str = " >> ".join(repr(n) for n in self.nodes)
        return f"Pattern(tag={self.tag!r}, {nodes_str})"
        

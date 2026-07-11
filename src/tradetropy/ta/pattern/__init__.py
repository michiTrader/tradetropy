"""
tradetropy.ta.pattern
==================
Pivot sequence pattern matching system.

Public exports
──────────────
    from tradetropy.ta.pattern import (
        Pattern,
        PatternNode,
        Condition,
        ConditionAnd,
        ConditionOr,
        NodeRef,
        NodeTypeRef,
        TagCondition,
        TimeCondition,
        parse_pattern,
    )

Minimal usage
─────────────
    from tradetropy.ta.pattern import Pattern, PatternNode, Condition, NodeRef

    pattern = Pattern([
        PatternNode('L', {}, []),
        PatternNode('H', {}, [Condition('>', NodeRef(0, 'value'))]),
    ], tag="impulse")

Usage with optional nodes (Python API)
──────────────────────────────────────
    from tradetropy.ta.pattern import Pattern, PatternNode, Condition, NodeRef

    pattern = Pattern([
        PatternNode('H', {'nbs': 'neu'}, []),
        PatternNode('L', {'nbs': 'boo'}, [], optional=True),   # may not exist
        PatternNode('H', {'nbs': 'neu'}, [Condition('>', NodeRef(0, 'value'))]),
    ], tag="hh_optional_pullback")

    # In on_data():
    match = self.setup.last
    if match:
        # Check if node 1 (optional) was present
        if match.matched_optional and match.matched_optional.get(1):
            pullback = match.node_map[1]   # PivotPoint of the pullback

Usage with optional nodes (DSL)
───────────────────────────────
    from tradetropy.ta.pattern import parse_pattern

    pattern = parse_pattern(\"\"\"
        H[nbs=neu]
        L?[nbs=boo]   > $0*0.97       # optional pullback, minimum 97% of H
        H[nbs=neu]    > $0             # Higher High (pattern index $0)
    \"\"\", tag="hh_optional_pullback_dsl")

Full usage
──────────
    from tradetropy.ta.pattern import (
        Pattern, PatternNode,
        Condition, ConditionAnd, ConditionOr,
        NodeRef, NodeTypeRef,
    )

    pattern = Pattern([
        PatternNode('H', {'nbs': 'neu', 'hhll': 'HH'}, [
            Condition('>', 3500),
        ]),
        PatternNode('L', {'nbs': 'boo'}, [
            Condition('>', NodeRef(0, 'value', 0.95)),
        ]),
        PatternNode('H', {'nbs': 'neu'}, [
            Condition('>', NodeRef(0, 'value')),
            ConditionOr([
                Condition('>', 3600),
                Condition('==', NodeTypeRef(0)),
            ]),
        ]),
    ], tag="strong_setup")
"""

from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin

__all__ = [
    # infra used by the free-tier pivot indicators
    "PivotIndicatorMixin",
]

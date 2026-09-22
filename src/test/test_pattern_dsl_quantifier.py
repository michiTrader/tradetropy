"""
test_pattern_dsl_quantifier.py
==============================
Tests for the ``{min,max}`` repetition quantifier of the pattern DSL:

  · Expansion into mandatory + optional PatternNode copies.
  · Equivalence with the hand-written (desugared) form.
  · Remapping of $N references across the expansion.
  · Elastic-gap matching over contiguous pivot windows.
  · Validation errors (bad ranges, '?' mixing, refs to min-0 groups,
    relative refs inside a quantified node, optional-node cap).
"""

import pytest

from tradetropy.exceptions import PatternSyntaxError
from tradetropy.ta.pattern import parse_pattern
from tradetropy.ta.pattern.conditions import Condition, NodeRef
from tradetropy.ta.pattern.types import PivotPoint


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pivot(value, ptype="H", index=0, **tags):
    base_tags = {"type": ptype}
    base_tags.update({k: str(v) for k, v in tags.items()})
    return PivotPoint(
        index=index, timestamp=float(index), value=value, type=ptype, tags=base_tags
    )


def _alternating(values, first="L", **tags_per_node):
    """Build an alternating L/H pivot run from ``values``."""
    out = []
    for i, v in enumerate(values):
        ptype = first if i % 2 == 0 else ("H" if first == "L" else "L")
        out.append(_pivot(v, ptype=ptype, index=i, **tags_per_node.get(i, {})))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuantifierExpansion:

    def test_exact_count_expands_to_mandatory_nodes(self):
        p = parse_pattern("L\nH{3}\n", tag="t")
        assert p.length == 4
        assert p.min_length == 4
        assert [n.optional for n in p.nodes] == [False, False, False, False]

    def test_range_expands_to_mandatory_plus_optional(self):
        p = parse_pattern("L\n~any{1,3}\nH\n", tag="t")
        # 1 + (1 mandatory + 2 optional) + 1
        assert p.length == 5
        assert p.min_length == 3
        assert [n.optional for n in p.nodes] == [False, False, True, True, False]

    def test_zero_min_expands_to_all_optional(self):
        p = parse_pattern("L\n~any{0,2}\nH\n", tag="t")
        assert p.length == 4
        assert p.min_length == 2
        assert [n.optional for n in p.nodes] == [False, True, True, False]

    def test_copies_inherit_type_tags_conditions_and_capture(self):
        p = parse_pattern("L[nbs=neu]\n~any{2} tag(nbs)!=neu\nH\n", tag="t")
        gap = p.nodes[1:3]
        assert all(n.type == "any" for n in gap)
        assert all(n.captured is False for n in gap)
        assert all(len(n.conditions) == 1 for n in gap)

    def test_no_quantifier_is_unchanged(self):
        p = parse_pattern("L\nH\n", tag="t")
        assert p.length == 2
        assert p.min_length == 2


# ══════════════════════════════════════════════════════════════════════════════
# EQUIVALENCE WITH THE DESUGARED FORM
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDesugaredEquivalence:

    def test_quantifier_matches_hand_written_optionals(self):
        sugar = parse_pattern("L[nbs=neu]\n~any{1,3}\nL[nbs=neu]\n", tag="t")
        desugared = parse_pattern(
            "L[nbs=neu]\n~any\n~any?\n~any?\nL[nbs=neu]\n", tag="t"
        )
        assert sugar.length == desugared.length
        assert sugar.min_length == desugared.min_length
        assert [n.optional for n in sugar.nodes] == [
            n.optional for n in desugared.nodes
        ]
        assert [n.captured for n in sugar.nodes] == [
            n.captured for n in desugared.nodes
        ]
        assert [n.type for n in sugar.nodes] == [n.type for n in desugared.nodes]


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE REMAPPING
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestReferenceRemapping:

    def test_absolute_ref_after_quantifier_is_shifted(self):
        # $0 is the first L; the quantifier in between expands to 3 nodes, so
        # the final H physically sits at index 4 and must still see $0 as 0.
        p = parse_pattern("L\n~any{1,3}\nH > $0\n", tag="t")
        cond = p.nodes[4].conditions[0]
        assert isinstance(cond, Condition)
        assert isinstance(cond.operand, NodeRef)
        assert cond.operand.position == 0

    def test_absolute_ref_to_node_after_a_quantifier(self):
        # $2 as written is the second L, which lands at physical index 4.
        p = parse_pattern("L\n~any{1,3}\nL\nH > $2\n", tag="t")
        cond = p.nodes[5].conditions[0]
        assert cond.operand.position == 4

    def test_ref_to_quantified_group_resolves_to_first_copy(self):
        # $1 is the quantified node; its first copy is physical index 1.
        p = parse_pattern("L\n~any{1,3}\nH > $1\n", tag="t")
        cond = p.nodes[4].conditions[0]
        assert cond.operand.position == 1

    def test_relative_ref_is_recomputed_across_expansion(self):
        # $-1 from the final H means "the previous logical node" (the second L
        # at physical 4). From physical 5 that is an offset of -1 as well, but
        # the point is that it must resolve to physical 4, not to a gap copy.
        p = parse_pattern("L\n~any{1,3}\nL\nH > $-1\n", tag="t")
        cond = p.nodes[5].conditions[0]
        assert 5 + cond.operand.position == 4

    def test_relative_ref_counts_logical_nodes_not_expanded_ones(self):
        # Logical nodes: 0=L, 1=gap{1,3}, 2=L, 3=H.
        # From the H, $-2 is the GAP (logical 1), whose first copy is physical 1
        # — the quantifier must not make the gap "invisible" to relative refs.
        p = parse_pattern("L\n~any{1,3}\nL\nH > $-2\n", tag="t")
        cond = p.nodes[5].conditions[0]
        assert 5 + cond.operand.position == 1

        # Reaching the first L therefore takes $-3.
        p2 = parse_pattern("L\n~any{1,3}\nL\nH > $-3\n", tag="t")
        cond2 = p2.nodes[5].conditions[0]
        assert 5 + cond2.operand.position == 0

    def test_arithmetic_expression_refs_are_remapped(self):
        p = parse_pattern("L\n~any{1,2}\nL\nH > ($2 - $0) * 2\n", tag="t")
        cond = p.nodes[4].conditions[0]
        positions = sorted(
            [cond.operand.left.left.position, cond.operand.left.right.position]
        )
        assert positions == [0, 3]


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING (elastic gap over contiguous windows)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuantifierMatching:

    def test_gap_absorbs_variable_number_of_pivots(self):
        p = parse_pattern("L[nbs=neu]\n~any{1,3}\nL[nbs=neu]\n", tag="t")

        # gap of 1: L[neu] H L[neu]
        win3 = [
            _pivot(10, "L", 0, nbs="neu"),
            _pivot(20, "H", 1, nbs="boo"),
            _pivot(11, "L", 2, nbs="neu"),
        ]
        assert p.try_match(win3) is not None

        # gap of 3: L[neu] H L H L[neu]
        win5 = [
            _pivot(10, "L", 0, nbs="neu"),
            _pivot(20, "H", 1, nbs="boo"),
            _pivot(12, "L", 2, nbs="boo"),
            _pivot(21, "H", 3, nbs="boo"),
            _pivot(11, "L", 4, nbs="neu"),
        ]
        assert p.try_match(win5) is not None

    def test_gap_conditions_filter_intermediate_pivots(self):
        # The gap forbids neutralizer highs, so this window must NOT match.
        p = parse_pattern(
            "L[nbs=neu]\n"
            "~any{1,3} tag(type)!=H | (tag(nbs)!=neu & tag(nbs)!=shk)\n"
            "L[nbs=neu]\n",
            tag="t",
        )
        blocked = [
            _pivot(10, "L", 0, nbs="neu"),
            _pivot(20, "H", 1, nbs="neu"),   # neutralizer high → interrupts
            _pivot(11, "L", 2, nbs="neu"),
        ]
        assert p.try_match(blocked) is None

        allowed = [
            _pivot(10, "L", 0, nbs="neu"),
            _pivot(20, "H", 1, nbs="boo"),
            _pivot(11, "L", 2, nbs="neu"),
        ]
        assert p.try_match(allowed) is not None

    def test_non_captured_copies_absent_from_result_but_in_node_map(self):
        p = parse_pattern("L[nbs=neu]\n~any{1,2}\nL[nbs=neu]\n", tag="t")
        win = [
            _pivot(10, "L", 0, nbs="neu"),
            _pivot(20, "H", 1, nbs="boo"),
            _pivot(11, "L", 2, nbs="neu"),
        ]
        match = p.try_match(win)
        assert match is not None
        # only the two captured L pivots
        assert len(match) == 2
        # the gap copy is still reachable through node_map
        assert match.node_map is not None
        assert match.node_map[1] is not None
        assert match.node_map[1].value == 20


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuantifierValidation:

    def test_min_greater_than_max_is_rejected(self):
        with pytest.raises(PatternSyntaxError, match="cannot be greater"):
            parse_pattern("L\n~any{3,1}\nH\n", tag="t")

    def test_zero_max_is_rejected(self):
        with pytest.raises(PatternSyntaxError, match="must be >= 1"):
            parse_pattern("L\n~any{0,0}\nH\n", tag="t")

    def test_optional_suffix_mixed_with_quantifier_is_rejected(self):
        with pytest.raises(PatternSyntaxError, match="equivalent to"):
            parse_pattern("L\n~any?{0,3}\nH\n", tag="t")

    def test_reference_to_min_zero_group_is_rejected(self):
        with pytest.raises(PatternSyntaxError, match="min 0"):
            parse_pattern("L\n~any{0,3}\nH > $1\n", tag="t")

    def test_relative_reference_inside_quantified_node_is_rejected(self):
        with pytest.raises(PatternSyntaxError, match="ambiguous"):
            parse_pattern("L\n~any{1,3} > $-1\nH\n", tag="t")

    def test_out_of_range_reference_is_rejected(self):
        with pytest.raises(PatternSyntaxError, match=r"\$9"):
            parse_pattern("L\n~any{1,3}\nH > $9\n", tag="t")

    def test_optional_node_cap_is_enforced(self):
        with pytest.raises(PatternSyntaxError, match="optional nodes"):
            parse_pattern("L\n~any{0,40}\nH\n", tag="t")

    def test_all_optional_pattern_still_rejected_by_pattern(self):
        # Pattern requires >= 2 mandatory nodes; a {0,N} gap alone cannot
        # satisfy that.
        with pytest.raises(PatternSyntaxError):
            parse_pattern("~any{0,2}\n~any{0,2}\n", tag="t")

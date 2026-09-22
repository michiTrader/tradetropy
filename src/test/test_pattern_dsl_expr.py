"""
test_pattern_dsl_expr.py
========================
Tests for the extended pattern matcher DSL:

  · Backward compatibility with the previous syntax.
  · Arithmetic expressions between pivots (+ - * /, parentheses).
  · Current node ($., $.index, $.timestamp) and explicit lhs.
  · Absolute value abs(...).
  · Ranges  expr in [low, high].
  · Precautions: division by zero, references to missing slots.
  · Early validation of NodeRef/NodeTypeRef out of range.
"""

import pytest

from tradetropy.exceptions import PatternSyntaxError
from tradetropy.ta.pattern import (
    Pattern,
    PatternNode,
    Condition,
    NodeRef,
    NodeTypeRef,
    CurrentRef,
    BinaryExpr,
    AbsExpr,
    RangeCondition,
    parse_pattern,
)
from tradetropy.ta.pattern.conditions import evaluate_condition, ConditionOr
from tradetropy.ta.pattern.types import PivotPoint


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pivot(value, ptype="H", index=0, timestamp=0.0, **tags):
    base_tags = {"type": ptype}
    base_tags.update({k: str(v) for k, v in tags.items()})
    return PivotPoint(index=index, timestamp=timestamp, value=value, type=ptype, tags=base_tags)


def _eval(cond, window, idx=None):
    """Evaluates a condition; idx defaults to the last node in the window."""
    if idx is None:
        idx = len(window) - 1
    return evaluate_condition(cond, idx, window)


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_simple_comparison_noderef(self):
        p = parse_pattern("L\nH > $0", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond, Condition)
        assert cond.lhs is None
        assert isinstance(cond.operand, NodeRef)
        assert cond.operand.position == 0
        assert cond.operand.attribute == "value"

    def test_classic_multiplier(self):
        # "$0*1.02" pasted → single NODEREF with multiplier
        p = parse_pattern("L\nH > $0*1.02", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.operand, NodeRef)
        assert cond.operand.multiplier == pytest.approx(1.02)
        assert cond.lhs is None  # classic path (same attribute)

    def test_index_attribute(self):
        p = parse_pattern("L\nH > $0.index", tag="t")
        cond = p.nodes[1].conditions[0]
        assert cond.operand.attribute == "index"

    def test_eval_noderef_value(self):
        # current node (idx 1) value=110 > node0 value=100
        w = [_pivot(100, "L"), _pivot(110, "H")]
        cond = Condition(">", NodeRef(0, "value"))
        assert _eval(cond, w) is True
        w2 = [_pivot(100, "L"), _pivot(90, "H")]
        assert _eval(cond, w2) is False

    def test_eval_index_same_attribute(self):
        # index comparison: current.index vs node0.index
        w = [_pivot(100, "L", index=5), _pivot(110, "H", index=8)]
        cond = Condition(">", NodeRef(0, "index"))
        assert _eval(cond, w) is True

    def test_numeric_absolute(self):
        w = [_pivot(100, "L"), _pivot(110, "H")]
        assert _eval(Condition(">", 105), w) is True
        assert _eval(Condition(">", 120), w) is False

    def test_type_ref_still_works(self):
        p = parse_pattern("H\nany == type(0)\nH > $0", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.operand, NodeTypeRef)
        # eval: current node type H == node0 type H
        w = [_pivot(100, "H"), _pivot(90, "H"), _pivot(110, "H")]
        assert _eval(cond, w, idx=1) is True


# ══════════════════════════════════════════════════════════════════════════════
# ARITHMETICS (rhs with implicit lhs)
# ══════════════════════════════════════════════════════════════════════════════

class TestArithmeticRhs:

    def test_parse_extension_fib(self):
        p = parse_pattern("L\nH > $1 + ($1 - $0) * 0.618", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.operand, BinaryExpr)
        assert cond.operand.op == "+"
        assert cond.lhs is None  # implicit lhs = current node

    def test_eval_constant_sum(self):
        # current.value > $0 + 50
        w = [_pivot(100, "L"), _pivot(160, "H")]
        cond = Condition(">", BinaryExpr("+", NodeRef(0, "value"), 50.0))
        assert _eval(cond, w) is True       # 160 > 150
        w2 = [_pivot(100, "L"), _pivot(140, "H")]
        assert _eval(cond, w2) is False     # 140 > 150 → no

    def test_eval_extension_fib(self):
        # leg $0=100 → $1=200, extension 0.618 → 200 + 100*0.618 = 261.8
        cond_str = "L\nH\nH > $1 + ($1 - $0) * 0.618"
        p = parse_pattern(cond_str, tag="t")
        cond = p.nodes[2].conditions[0]
        w_ok = [_pivot(100, "L"), _pivot(200, "H"), _pivot(270, "H")]
        assert _eval(cond, w_ok, idx=2) is True
        w_no = [_pivot(100, "L"), _pivot(200, "H"), _pivot(250, "H")]
        assert _eval(cond, w_no, idx=2) is False

    def test_multiplication_over_addition_precedence(self):
        # $0 + $1 * 2  →  $0 + ($1 * 2)
        p = parse_pattern("L\nH > $0 + $1 * 2", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.operand, BinaryExpr)
        assert cond.operand.op == "+"
        assert isinstance(cond.operand.right, BinaryExpr)
        assert cond.operand.right.op == "*"


# ══════════════════════════════════════════════════════════════════════════════
# EXPLICIT LHS AND CURRENT NODE
# ══════════════════════════════════════════════════════════════════════════════

class TestExplicitLhs:

    def test_parse_explicit_lhs(self):
        p = parse_pattern("L\nH\nH ($. - $1) > ($1 - $0)", tag="t")
        cond = p.nodes[2].conditions[0]
        assert isinstance(cond, Condition)
        assert cond.lhs is not None
        assert isinstance(cond.lhs, BinaryExpr)

    def test_eval_second_leg_greater(self):
        # leg1 = $1-$0 = 50 ; leg2 = $.-$1 = 80 → 80 > 50
        p = parse_pattern("L\nH\nH ($. - $1) > ($1 - $0)", tag="t")
        cond = p.nodes[2].conditions[0]
        w = [_pivot(100, "L"), _pivot(150, "H"), _pivot(230, "H")]
        assert _eval(cond, w, idx=2) is True
        # leg2 = 30 < 50 → False
        w2 = [_pivot(100, "L"), _pivot(150, "H"), _pivot(180, "H")]
        assert _eval(cond, w2, idx=2) is False

    def test_current_ref_index(self):
        p = parse_pattern("L\nH $.index > $0.index", tag="t")
        cond = p.nodes[1].conditions[0]
        assert cond.lhs is not None
        assert isinstance(cond.lhs, CurrentRef)
        assert cond.lhs.attribute == "index"
        w = [_pivot(100, "L", index=3), _pivot(110, "H", index=9)]
        assert _eval(cond, w) is True


# ══════════════════════════════════════════════════════════════════════════════
# ABS
# ══════════════════════════════════════════════════════════════════════════════

class TestAbs:

    def test_parse_abs(self):
        p = parse_pattern("L\nH abs($. - $0) > 100", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.lhs, AbsExpr)

    def test_eval_abs(self):
        p = parse_pattern("L\nH abs($. - $0) > 100", tag="t")
        cond = p.nodes[1].conditions[0]
        # |90 - 200| = 110 > 100
        w = [_pivot(200, "L"), _pivot(90, "H")]
        assert _eval(cond, w) is True
        # |150 - 200| = 50 < 100
        w2 = [_pivot(200, "L"), _pivot(150, "H")]
        assert _eval(cond, w2) is False


# ══════════════════════════════════════════════════════════════════════════════
# RANGES  in [a, b]
# ══════════════════════════════════════════════════════════════════════════════

class TestRanges:

    # Retracement pattern: L($0) → H($1) → L($2 pullback).
    # At node 2 ($.==pullback) the retracement of leg $0→$1 is
    #   ($1 - $.) / ($1 - $0)
    _DSL = "L\nH > $0\nL ($1 - $.) / ($1 - $0) in [0.382, 0.618]"

    def test_parse_range(self):
        p = parse_pattern(self._DSL, tag="t")
        cond = p.nodes[2].conditions[0]
        assert isinstance(cond, RangeCondition)
        assert isinstance(cond.expr, BinaryExpr)
        assert cond.low == pytest.approx(0.382)
        assert cond.high == pytest.approx(0.618)

    def test_eval_retracement_inside(self):
        # $0=100, $1=200, pullback $.=150 → (200-150)/(200-100) = 0.5 ∈ range
        p = parse_pattern(self._DSL, tag="t")
        cond = p.nodes[2].conditions[0]
        w = [_pivot(100, "L"), _pivot(200, "H"), _pivot(150, "L")]
        assert _eval(cond, w, idx=2) is True

    def test_eval_retracement_outside(self):
        # pullback $.=180 → (200-180)/100 = 0.2 < 0.382
        p = parse_pattern(self._DSL, tag="t")
        cond = p.nodes[2].conditions[0]
        w = [_pivot(100, "L"), _pivot(200, "H"), _pivot(180, "L")]
        assert _eval(cond, w, idx=2) is False


# ══════════════════════════════════════════════════════════════════════════════
# PRECAUTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestPrecautions:

    def test_division_by_zero_no_raise(self):
        # $0 - $1 = 0 → undefined division → condition False, no exception
        cond = RangeCondition(
            BinaryExpr("/", BinaryExpr("-", NodeRef(0, "value"), CurrentRef()),
                       BinaryExpr("-", NodeRef(0, "value"), NodeRef(1, "value"))),
            0.0, 1.0,
        )
        w = [_pivot(100, "H"), _pivot(100, "L"), _pivot(50, "L")]  # $0==$1 → den=0
        assert _eval(cond, w, idx=2) is False

    def test_ref_to_optional_missing_is_false(self):
        # window with None in slot 1 (missing optional)
        cond = Condition(">", NodeRef(1, "value"))
        w = [_pivot(100, "L"), None, _pivot(110, "H")]
        assert _eval(cond, w, idx=2) is False

    def test_binary_with_missing_ref_is_false(self):
        cond = Condition(">", BinaryExpr("+", NodeRef(1, "value"), 10.0))
        w = [_pivot(100, "L"), None, _pivot(110, "H")]
        assert _eval(cond, w, idx=2) is False


# ══════════════════════════════════════════════════════════════════════════════
# EARLY VALIDATION OF REFERENCES
# ══════════════════════════════════════════════════════════════════════════════

class TestRefValidation:

    def test_absolute_out_of_range_raises(self):
        with pytest.raises(PatternSyntaxError, match=r"\$5"):
            parse_pattern("L\nH > $5", tag="t")  # solo 2 nodos

    def test_absolute_out_of_range_in_arithmetic(self):
        with pytest.raises(PatternSyntaxError):
            parse_pattern("L\nH > $0 + ($9 - $0)", tag="t")

    def test_relative_before_start_raises(self):
        # node 0 with $-1 → position -1 → invalid
        with pytest.raises(PatternSyntaxError, match=r"antes del inicio|before"):
            Pattern(
                [
                    PatternNode("L", {}, [Condition(">", NodeRef(-1, "value"))]),
                    PatternNode("H", {}, []),
                ],
                tag="t",
            )

    def test_relative_valid_no_raise(self):
        # node 1 with $-1 → position 0 → valid
        p = parse_pattern("L\nH > $-1", tag="t")
        assert len(p.nodes) == 2

    def test_forward_ref_valid_no_raise(self):
        # node 0 references $2 (future within pattern) → valid (3 nodes)
        p = parse_pattern("L < $2\nH > $0\nH > $1", tag="t")
        assert len(p.nodes) == 3

    def test_typeref_out_of_range_raises(self):
        with pytest.raises(PatternSyntaxError):
            parse_pattern("L\nH == type(7)", tag="t")

    def test_range_with_out_of_range_ref_raises(self):
        with pytest.raises(PatternSyntaxError):
            parse_pattern("H\nL $0 in [$1, $9]\nH > $0", tag="t")

    def test_timecond_hours_since_out_of_range_raises(self):
        with pytest.raises(PatternSyntaxError):
            parse_pattern(
                "L\nH > $0 & @hours_since($5.timestamp, min=0, max=48)",
                tag="t",
            )


# ══════════════════════════════════════════════════════════════════════════════
# MATCH END-TO-END (try_match con expresiones)
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchEndToEnd:

    def test_try_match_extension_fib(self):
        p = parse_pattern("L\nH\nH > $1 + ($1 - $0) * 0.5", tag="fib")
        # $0=100, $1=200, extension 0.5 → 250.  node2=260 passes.
        w = [_pivot(100, "L"), _pivot(200, "H"), _pivot(260, "H")]
        m = p.try_match(w)
        assert m is not None
        assert m.tag == "fib"

    def test_try_match_extension_fib_fails(self):
        p = parse_pattern("L\nH\nH > $1 + ($1 - $0) * 0.5", tag="fib")
        w = [_pivot(100, "L"), _pivot(200, "H"), _pivot(240, "H")]  # 240 < 250
        assert p.try_match(w) is None

    def test_try_match_retracement_range(self):
        # L($0) → H($1) → L($2 pullback) with retracement 0.382..0.618
        p = parse_pattern(
            "L\nH > $0\nL ($1 - $.) / ($1 - $0) in [0.382, 0.618]",
            tag="retr",
        )
        # $0=100, $1=200, pullback=150 → retracement 0.5 ∈ range → match
        w_ok = [_pivot(100, "L"), _pivot(200, "H"), _pivot(150, "L")]
        assert p.try_match(w_ok) is not None
        # pullback=190 → retracement 0.1 → outside → no match
        w_no = [_pivot(100, "L"), _pivot(200, "H"), _pivot(190, "L")]
        assert p.try_match(w_no) is None


# ══════════════════════════════════════════════════════════════════════════════
# TIME_SINCE  (unit=h|m|s) and hours_since alias
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeSince:

    def test_parse_default_unit_hours(self):
        from tradetropy.ta.pattern.conditions import TimeCondition
        p = parse_pattern("L\nH @time_since($0.timestamp, min=2, max=48)", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond, TimeCondition)
        assert cond.kind == "hours_since"          # internal kind unchanged
        assert cond.min_hours == pytest.approx(2.0)
        assert cond.max_hours == pytest.approx(48.0)

    def test_parse_unit_minutes(self):
        p = parse_pattern(
            "L\nH @time_since($0.timestamp, min=5, max=15, unit=m)", tag="t"
        )
        cond = p.nodes[1].conditions[0]
        assert cond.min_hours == pytest.approx(5 / 60)
        assert cond.max_hours == pytest.approx(15 / 60)

    def test_parse_unit_seconds(self):
        p = parse_pattern("L\nH @time_since($0.timestamp, min=30, unit=s)", tag="t")
        cond = p.nodes[1].conditions[0]
        assert cond.min_hours == pytest.approx(30 / 3600)
        assert cond.max_hours is None

    def test_hours_since_alias_still_works(self):
        p = parse_pattern("L\nH @hours_since($0.timestamp, min=2, max=48)", tag="t")
        cond = p.nodes[1].conditions[0]
        assert cond.kind == "hours_since"
        assert cond.min_hours == pytest.approx(2.0)
        assert cond.max_hours == pytest.approx(48.0)

    def test_hours_since_rejects_unit(self):
        with pytest.raises(PatternSyntaxError, match="unit"):
            parse_pattern("L\nH @hours_since($0.timestamp, min=2, unit=m)", tag="t")

    def test_unknown_unit_raises(self):
        with pytest.raises(PatternSyntaxError, match="unit"):
            parse_pattern("L\nH @time_since($0.timestamp, min=2, unit=days)", tag="t")

    def test_missing_timestamp_attr_raises(self):
        with pytest.raises(PatternSyntaxError):
            parse_pattern("L\nH @time_since($0, min=2)", tag="t")

    def test_eval_minutes_window(self):
        # between 5 and 15 minutes after node 0
        p = parse_pattern(
            "L\nH @time_since($0.timestamp, min=5, max=15, unit=m)", tag="t"
        )
        cond = p.nodes[1].conditions[0]
        # 10 min → inside
        w_ok = [_pivot(100, "L", index=0, timestamp=0.0),
                _pivot(110, "H", index=1, timestamp=10 * 60 * 1000)]
        assert _eval(cond, w_ok, idx=1) is True
        # 2 min → below min
        w_lo = [_pivot(100, "L", index=0, timestamp=0.0),
                _pivot(110, "H", index=1, timestamp=2 * 60 * 1000)]
        assert _eval(cond, w_lo, idx=1) is False
        # 20 min → above max
        w_hi = [_pivot(100, "L", index=0, timestamp=0.0),
                _pivot(110, "H", index=1, timestamp=20 * 60 * 1000)]
        assert _eval(cond, w_hi, idx=1) is False

    def test_eval_seconds_window(self):
        # at least 30 seconds after node 0, no upper bound
        p = parse_pattern("L\nH @time_since($0.timestamp, min=30, unit=s)", tag="t")
        cond = p.nodes[1].conditions[0]
        # 45 s → inside
        w_ok = [_pivot(100, "L", index=0, timestamp=0.0),
                _pivot(110, "H", index=1, timestamp=45 * 1000)]
        assert _eval(cond, w_ok, idx=1) is True
        # 10 s → below min
        w_lo = [_pivot(100, "L", index=0, timestamp=0.0),
                _pivot(110, "H", index=1, timestamp=10 * 1000)]
        assert _eval(cond, w_lo, idx=1) is False

    def test_time_since_matches_hours_since_numerically(self):
        # @time_since(unit=h) and @hours_since produce the same bounds
        a = parse_pattern(
            "L\nH @time_since($0.timestamp, min=2, max=48, unit=h)", tag="t"
        ).nodes[1].conditions[0]
        b = parse_pattern(
            "L\nH @hours_since($0.timestamp, min=2, max=48)", tag="t"
        ).nodes[1].conditions[0]
        assert a.min_hours == pytest.approx(b.min_hours)
        assert a.max_hours == pytest.approx(b.max_hours)


# ══════════════════════════════════════════════════════════════════════════════
# @active_bars  (activation window since last pivot confirmation bar)
# ══════════════════════════════════════════════════════════════════════════════

class TestActiveBars:

    def test_default_no_active_window(self):
        p = parse_pattern("L\nH > $0", tag="t")
        assert p.active_start == 0
        assert p.active_end is None
        assert p.has_active_window is False

    def test_parse_two_args(self):
        p = parse_pattern("L\nH > $0\n@active_bars(2, 10)", tag="t")
        assert (p.active_start, p.active_end) == (2, 10)
        assert p.has_active_window is True

    def test_parse_single_arg_is_start_zero(self):
        p = parse_pattern("L\nH > $0\n@active_bars(10)", tag="t")
        assert (p.active_start, p.active_end) == (0, 10)
        assert p.has_active_window is True

    def test_parse_with_anchor_end_any_order(self):
        p = parse_pattern("L\nH > $0\n$\n@active_bars(0, 5)", tag="t")
        assert p.anchor_end is True
        assert (p.active_start, p.active_end) == (0, 5)
        # nodes must exclude the directive lines
        assert len(p.nodes) == 2

    def test_parse_active_bars_before_anchor(self):
        p = parse_pattern("L\nH > $0\n@active_bars(0, 5)\n$", tag="t")
        assert p.anchor_end is True
        assert (p.active_start, p.active_end) == (0, 5)
        assert len(p.nodes) == 2

    def test_invalid_directive_raises(self):
        with pytest.raises(PatternSyntaxError, match="active_bars"):
            parse_pattern("L\nH > $0\n@active_bars(a, b)", tag="t")

    def test_end_before_start_raises(self):
        with pytest.raises(PatternSyntaxError, match="cannot be smaller"):
            parse_pattern("L\nH > $0\n@active_bars(10, 2)", tag="t")

    def test_active_age_ok_bounds(self):
        p = parse_pattern("L\nH > $0\n@active_bars(2, 10)", tag="t")
        assert p.active_age_ok(1) is False   # before start
        assert p.active_age_ok(2) is True    # start inclusive
        assert p.active_age_ok(6) is True
        assert p.active_age_ok(10) is True   # end inclusive
        assert p.active_age_ok(11) is False  # after end

    def test_active_age_ok_open_ended(self):
        p = parse_pattern("L\nH > $0\n@active_bars(3)", tag="t")
        # start 0, end 3
        assert p.active_age_ok(0) is True
        assert p.active_age_ok(3) is True
        assert p.active_age_ok(4) is False

    def test_default_pattern_always_active(self):
        p = parse_pattern("L\nH > $0", tag="t")
        assert p.active_age_ok(0) is True
        assert p.active_age_ok(9999) is True


# ══════════════════════════════════════════════════════════════════════════════
# SPACE-OPTIONAL ARITHMETIC  ('-' disambiguation, no spaces required)
# ══════════════════════════════════════════════════════════════════════════════

class TestSpaceOptionalArithmetic:

    def test_subtraction_without_spaces_equals_spaced(self):
        tight   = parse_pattern("L\nH\nL ($1-$0) >= ($2-$1)", tag="t")
        spaced  = parse_pattern("L\nH\nL ($1 - $0) >= ($2 - $1)", tag="t")
        assert tight == spaced

    def test_number_minus_number_without_spaces(self):
        # (5-6) must be subtraction, not NUM(5) NUM(-6)
        p = parse_pattern("L\nH (5-6) < $0", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.lhs, BinaryExpr)
        assert cond.lhs.op == "-"
        assert cond.lhs.left == pytest.approx(5.0)
        assert cond.lhs.right == pytest.approx(6.0)

    def test_noderef_minus_number_without_spaces(self):
        # "$0-5" → subtraction ($0 - 5), not NODEREF then NUM(-5)
        p = parse_pattern("L\nH $0-5 > $1", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.lhs, BinaryExpr)
        assert cond.lhs.op == "-"
        assert isinstance(cond.lhs.left, NodeRef)
        assert cond.lhs.right == pytest.approx(5.0)

    def test_multiplier_after_paren_without_spaces(self):
        # user's real case: ($1-$0)*2
        tight  = parse_pattern("L\nH\nL ($2-$1) <= ($1-$0)*2", tag="t")
        spaced = parse_pattern("L\nH\nL ($2 - $1) <= ($1 - $0) * 2", tag="t")
        assert tight == spaced

    def test_negative_literal_still_works_in_prefix_position(self):
        # "> -5" stays "greater than minus five" (prefix position after OP)
        p = parse_pattern("L\nH > -5", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond, Condition)
        assert cond.operand == pytest.approx(-5.0)

    def test_negative_literal_inside_arithmetic_prefix(self):
        # "$0 + -3" → $0 + (-3); '-3' is prefix (after ARITH '+')
        p = parse_pattern("L\nH > $0 + -3", tag="t")
        cond = p.nodes[1].conditions[0]
        assert isinstance(cond.operand, BinaryExpr)
        assert cond.operand.op == "+"
        assert cond.operand.right == pytest.approx(-3.0)

    def test_eval_tight_subtraction_matches(self):
        # ($1-$0) >= ($2-$1): leg1=100, leg2=100 → equal → True
        p = parse_pattern("L\nH\nL ($2-$1) <= ($1-$0)", tag="t")
        cond = p.nodes[2].conditions[0]
        w = [_pivot(100, "L"), _pivot(200, "H"), _pivot(150, "L")]
        # ($2-$1) = 150-200 = -50 ; ($1-$0) = 200-100 = 100 ; -50 <= 100 → True
        assert _eval(cond, w, idx=2) is True


# ══════════════════════════════════════════════════════════════════════════════
# LINE CONTINUATION  (folding of '&' / '|' lines)
# ══════════════════════════════════════════════════════════════════════════════

class TestLineContinuation:

    def test_multiline_equals_singleline(self):
        # A folded multi-line pattern must produce the SAME Pattern object as
        # its single-line equivalent (Pattern/PatternNode/Condition are
        # dataclasses, so == compares structurally).
        multiline = parse_pattern(
            """
            L[nbs=boo]
            H[nbs=neu|shk] > $1
                & @between(22:30, 20:30)
                & @between(22:30, 16:30)
            """,
            tag="t",
        )
        singleline = parse_pattern(
            "L[nbs=boo]\n"
            "H[nbs=neu|shk] > $1 & @between(22:30, 20:30) & @between(22:30, 16:30)",
            tag="t",
        )
        assert multiline == singleline

    def test_folding_produces_three_conditions(self):
        p = parse_pattern(
            """
            L
            H > $0
                & > $0*1.01
                & @after(09:30)
            """,
            tag="t",
        )
        assert len(p.nodes) == 2
        assert len(p.nodes[1].conditions) == 3

    def test_or_connector_continuation(self):
        # A '|' continuation folds and builds an OR chain, identical to the
        # single-line form.
        folded = parse_pattern(
            """
            L
            H > $0
                | > 50000
            """,
            tag="t",
        )
        inline = parse_pattern("L\nH > $0 | > 50000", tag="t")
        assert folded == inline
        assert isinstance(folded.nodes[1].conditions[0], ConditionOr)

    def test_continuation_across_multiple_nodes(self):
        # Folding is per logical line: each node keeps its own continuations.
        folded = parse_pattern(
            """
            L @after(09:00)
                & @before(16:00)
            H > $0
                & @after(10:00)
            """,
            tag="t",
        )
        inline = parse_pattern(
            "L @after(09:00) & @before(16:00)\n"
            "H > $0 & @after(10:00)",
            tag="t",
        )
        assert folded == inline

    def test_continuation_with_interleaved_comment(self):
        # Comment lines are skipped and do not break the fold.
        folded = parse_pattern(
            """
            L
            H > $0
                # extra guard
                & > $0*1.02
            """,
            tag="t",
        )
        inline = parse_pattern("L\nH > $0 & > $0*1.02", tag="t")
        assert folded == inline

    def test_continuation_before_directives(self):
        # $ and @active_bars directives still parse when a folded node
        # precedes them.
        p = parse_pattern(
            """
            L
            H > $0
                & @after(09:30)
            $
            @active_bars(0, 5)
            """,
            tag="t",
        )
        assert len(p.nodes) == 2
        assert p.anchor_end is True
        assert (p.active_start, p.active_end) == (0, 5)

    def test_continuation_without_previous_node_raises(self):
        with pytest.raises(PatternSyntaxError, match="continuation line"):
            parse_pattern("& > $0\nL\nH > $0", tag="t")

    def test_bare_node_with_condition_on_next_line(self):
        # Node "pelado" arriba, primera condición (bare '>') abajo.
        folded = parse_pattern(
            """
            L
            H
                > $0
            """,
            tag="t",
        )
        inline = parse_pattern("L\nH > $0", tag="t")
        assert folded == inline

    def test_bare_node_multi_condition_chain(self):
        # Caso real reportado: nodo pelado, condicion bare '>' y luego
        # comparaciones aritmeticas con parentesis, todo indentado abajo.
        folded = parse_pattern(
            """
            ~L
            ~H
                > $0
                & $2 >= $1
            ~L[nbs=boo]
                > $0
                & ($3-$2) >= ($1-$0)
            H[nbs=neu|shk] > $1
                & @between(22:30, 20:30)
                & @between(22:30, 16:30)
            """,
            tag="t",
        )
        inline = parse_pattern(
            "~L\n"
            "~H > $0 & $2 >= $1\n"
            "~L[nbs=boo] > $0 & ($3-$2) >= ($1-$0)\n"
            "H[nbs=neu|shk] > $1 & @between(22:30, 20:30) & @between(22:30, 16:30)",
            tag="t",
        )
        assert folded == inline

    def test_leading_lt_and_le_ge_ne_are_continuations(self):
        for op in ("<", ">=", "<=", "==", "!="):
            operand = "type(0)" if op in ("==", "!=") else "$0"
            folded = parse_pattern(f"L\nH\n    {op} {operand}", tag="t")
            inline = parse_pattern(f"L\nH {op} {operand}", tag="t")
            assert folded == inline, f"failed for operator {op!r}"

    def test_leading_parenthesis_is_continuation(self):
        folded = parse_pattern(
            """
            L
            H
                (> $0 & > 50000) | @after(20:00)
            """,
            tag="t",
        )
        inline = parse_pattern(
            "L\nH (> $0 & > 50000) | @after(20:00)", tag="t",
        )
        assert folded == inline

    def test_bare_at_is_not_a_continuation_marker(self):
        # A leading '@' is NOT treated as a continuation: it is parsed as
        # its own (invalid, since it lacks TYPE) node line, to avoid
        # ambiguity with the standalone '@active_bars(...)' directive.
        with pytest.raises(PatternSyntaxError):
            parse_pattern(
                """
                L
                H > $0
                    @after(20:00)
                """,
                tag="t",
            )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

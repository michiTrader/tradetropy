"""
conditions.py
=============
Condition system for PatternNode.

Condition types
───────────────
Condition      — atomic price/index/timestamp condition (operator + operand)
TimeCondition  — time condition: hour of day, weekday, time distance
TagCondition   — condition on current node tags (tag(key) == value)
ConditionAnd   — all must be satisfied
ConditionOr    — at least one must be satisfied

Operand types (Condition)
─────────────────────────
float / int   — absolute value:            Condition('>', 3500)
NodeRef       — reference to another node: Condition('>', NodeRef(0, 'value'))
               with multiplier:            Condition('>', NodeRef(0, 'value', 0.98))
               relative to current:        Condition('>', NodeRef(-1, 'value'))
               by bar index:              Condition('>', NodeRef(0, 'index'))
               by timestamp:              Condition('>', NodeRef(0, 'timestamp'))
NodeTypeRef   — reference to type:         Condition('==', NodeTypeRef(0))

Note on NodeRef and lhs
───────────────────────
The lhs of the comparison is derived from the SAME attribute as the rhs.
That is:
    Condition('>', NodeRef(0, 'value'))      → current_price     > node0_price
    Condition('>', NodeRef(0, 'index'))      → current_index     > node0_index
    Condition('>', NodeRef(0, 'timestamp'))  → current_timestamp > node0_timestamp

The multiplier only makes sense on 'value' — it applies equally
to 'index' and 'timestamp' but the user should be aware of this.

TimeCondition — time conditions
───────────────────────────────
Operates on the timestamp (ts_ms, UTC epoch) of the current node or another node.

    # hour of day of current pivot between 09:30 and 16:00 NY
    TimeCondition('between', start='09:30', end='16:00', tz='America/New_York')

    # pivot occurs after 14:00 UTC
    TimeCondition('after', start='14:00')

    # pivot occurs before 22:00 UTC
    TimeCondition('before', end='22:00')

    # pivot occurs on Monday, Tuesday or Wednesday
    TimeCondition('weekday', days=['monday', 'tuesday', 'wednesday'])

    # current pivot confirmed between 2 and 48 hours after node 0
    TimeCondition('hours_since', ref=NodeRef(0, 'timestamp'), min_hours=2, max_hours=48)

Combination
───────────
The condition list in PatternNode is an implicit AND.
ConditionOr and ConditionAnd are used for more complex cases:

    # (A AND B) OR C
    ConditionOr([
        ConditionAnd([Condition('>', 3500), TimeCondition('between', '09:00', '10:00')]),
        Condition('==', NodeTypeRef(0))
    ])

Optional nodes and NodeRef
──────────────────────────
When a pattern has optional nodes (``PatternNode.optional=True``),
the window received by the evaluator has size ``len(pattern.nodes)`` with
``None`` in positions of absent optional nodes.

· If a NodeRef or NodeTypeRef points to a ``None`` slot (absent node),
  the condition safely returns ``False`` — the match candidate
  is discarded and the next one is tried.
· TimeCondition also returns ``False`` if the current node is None
  (a situation that shouldn't occur, but is protected against).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union, TYPE_CHECKING

from tradetropy.exceptions import PatternSyntaxError, TimeConditionError

if TYPE_CHECKING:
    from tradetropy.ta.pattern.types import PivotPoint


# ══════════════════════════════════════════════════════════════════════════════
# OPERANDS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NodeRef:
    """
    Reference to an attribute of another node in the pattern window.

    Parameters
    ──────────
    position  : position of the referenced node.
                · >= 0 → absolute from the start of the pattern ($0, $1, $2...)
                · < 0  → relative to the current node ($-1 = previous node)
    attribute : which attribute to read from the referenced node.
                · 'value'     → pivot price
                · 'index'     → confirmation bar index
                · 'timestamp' → ts_ms of the REAL pivot bar
    multiplier: factor to apply on the referenced value.
                Useful for conditions like "> $0.value * 1.02" (2% above).
                Default 1.0 (no modification).
                Note: on 'index' and 'timestamp' the multiplier applies
                arithmetically — use with caution.

    The lhs of the comparison is derived from the SAME attribute, so units
    are always coherent:
        NodeRef(0, 'value')     → current_price     vs node0_price
        NodeRef(0, 'index')     → current_index     vs node0_index
        NodeRef(0, 'timestamp') → current_timestamp vs node0_timestamp

    Examples
    ────────
        NodeRef(0, 'value')              # price of first node
        NodeRef(0, 'value', 1.02)        # 2% above the first node
        NodeRef(-1, 'value')             # price of node before current
        NodeRef(0, 'index')              # bar index of first node
        NodeRef(0, 'timestamp')          # timestamp of first node
        NodeRef(-1, 'timestamp')         # timestamp of previous node
    """
    position:   int
    attribute:  Literal['value', 'index', 'timestamp']
    multiplier: float = 1.0


@dataclass(frozen=True)
class NodeTypeRef:
    """
    Reference to the type ('H' or 'L') of another node in the window.

    Parameters
    ──────────
    position : position of the referenced node (same as NodeRef).

    Typical usage
    ─────────────
        # same type as the first node
        Condition('==', NodeTypeRef(0))

        # opposite type to the previous node
        Condition('!=', NodeTypeRef(-1))
    """
    position: int


@dataclass(frozen=True)
class CurrentRef:
    """
    Reference to an attribute of the CURRENT node (the one being evaluated).

    In the DSL written as ``$.`` (value), ``$.value``, ``$.index`` or
    ``$.timestamp``. Its utility is being able to use the current node as
    an operand within arithmetic expressions, since in "legacy" conditions
    the left side (the current node) was always implicit.

    Examples
    ────────
        # pullback retracement relative to the $0→$1 leg
        ($0 - $.) / ($0 - $1)        →  BinaryExpr('/',
                                            BinaryExpr('-', NodeRef(0), CurrentRef()),
                                            BinaryExpr('-', NodeRef(0), NodeRef(1)))
    """
    attribute: Literal['value', 'index', 'timestamp'] = 'value'


@dataclass(frozen=True)
class AbsExpr:
    """Absolute value of an expression: ``abs(expr)``."""
    operand: 'ValueExpr'


@dataclass(frozen=True)
class BinaryExpr:
    """
    Binary arithmetic operation between two value expressions.

    Allows building expressions like ``$1 + ($1 - $0) * 0.618`` (Fibonacci
    extension) or ``($0 - $.) / ($0 - $1)`` (retracement).

    Parameters
    ──────────
    op    : '+', '-', '*', '/'
    left  : ValueExpr
    right : ValueExpr

    Division by zero is evaluated safely: the expression returns
    ``None`` and the condition containing it is considered unsatisfied.
    """
    op:    Literal['+', '-', '*', '/']
    left:  'ValueExpr'
    right: 'ValueExpr'


# Union type of all valid operands for Condition (compatibility)
Operand = Union[float, int, NodeRef, NodeTypeRef, 'CurrentRef', 'BinaryExpr', 'AbsExpr']

# Expression that evaluates to a number (lhs/rhs of comparisons and ranges).
# Does not include NodeTypeRef (which is a type comparison, not numeric).
ValueExpr = Union[float, int, NodeRef, 'CurrentRef', 'BinaryExpr', 'AbsExpr']


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Condition:
    """
    Atomic condition: compares an attribute of the current node against an operand.

    The attribute compared on the lhs depends on the operand:
    · Numeric operand  → lhs = current.value
    · NodeRef          → lhs = getattr(current, operand.attribute)
                         (same units as the rhs)
    · NodeTypeRef      → lhs = current.type  (string 'H'/'L')

    Parameters
    ──────────
    operator : '>', '<', '>=', '<=', '==', '!='
    operand  : float | int | NodeRef | NodeTypeRef

    Examples
    ────────
        Condition('>', 3500)                          # absolute price
        Condition('<', NodeRef(0, 'value'))            # relative to first node
        Condition('>', NodeRef(0, 'value', 1.02))     # > 2% above node 0
        Condition('>', NodeRef(-1, 'value'))           # > previous node
        Condition('>', NodeRef(0, 'index'))            # current index > node 0 index
        Condition('<', NodeRef(-1, 'timestamp'))       # current ts < previous node ts
        Condition('==', NodeTypeRef(0))                # same type as first node
        Condition('!=', NodeTypeRef(-1))               # opposite type to previous

    Explicit lhs (arithmetic expressions)
    ─────────────────────────────────────
    By default ``lhs is None`` and the left side is the current node
    (classic behavior). If a ValueExpr is passed in ``lhs``, both sides
    are evaluated as numeric expressions:

        # current leg greater than the $0→$1 leg
        Condition('>', BinaryExpr('-', NodeRef(1), NodeRef(0)),
                  lhs=BinaryExpr('-', CurrentRef(), NodeRef(2)))
    """
    operator: Literal['>', '<', '>=', '<=', '==', '!=']
    operand:  Operand
    lhs:      'ValueExpr | None' = None


@dataclass(frozen=True)
class RangeCondition:
    """
    Checks that an expression falls within an inclusive range ``[low, high]``.

    In the DSL written as ``expr in [low, high]``. Equivalent to
    ``low <= expr <= high``. Useful for Fibonacci retracements/extensions:

        ($0 - $.) / ($0 - $1) in [0.382, 0.618]

    Parameters
    ──────────
    expr : ValueExpr to evaluate.
    low  : ValueExpr — lower bound (inclusive).
    high : ValueExpr — upper bound (inclusive).

    If any of the three expressions cannot be evaluated (out-of-range
    reference, absent optional node or division by zero), the condition
    returns ``False``.
    """
    expr: 'ValueExpr'
    low:  'ValueExpr'
    high: 'ValueExpr'


# ── Valid weekdays ────────────────────────────────────────────────────────────
_WEEKDAY_MAP: dict[str, int] = {
    "monday":    0,
    "tuesday":   1,
    "wednesday": 2,
    "thursday":  3,
    "friday":    4,
    "saturday":  5,
    "sunday":    6,
    # abbreviations
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

# TimeCondition kinds
TimeConditionKind = Literal['between', 'after', 'before', 'weekday', 'hours_since']


@dataclass(frozen=True)
class TimeCondition:
    """
    Time condition on the current node's timestamp.

    The timestamp used is always from the REAL pivot bar (ts_ms, UTC epoch).

    Types (kind)
    ────────────
    'between'
        The pivot occurs between start and end (hour of day, inclusive bounds).
        start and end : "HH:MM" in 24h.
        tz            : timezone to interpret the hour. Default "UTC".
        Supports overnight ranges: start="22:00", end="04:00" works.

        TimeCondition('between', start='09:30', end='16:00', tz='America/New_York')
        TimeCondition('between', start='22:00', end='04:00')   # overnight range

    'after'
        The pivot occurs after the start hour (inclusive) on the pivot's day.
        start : "HH:MM" in 24h.
        tz    : timezone. Default "UTC".

        TimeCondition('after', start='14:00')
        TimeCondition('after', start='09:30', tz='America/New_York')

    'before'
        The pivot occurs before the end hour (exclusive) on the pivot's day.
        end : "HH:MM" in 24h.
        tz  : timezone. Default "UTC".

        TimeCondition('before', end='22:00')
        TimeCondition('before', end='16:00', tz='Europe/London')

    'weekday'
        The pivot occurs on one of the given weekdays.
        days : list of strings — "monday", "tuesday", ..., "sunday"
               or abbreviations "mon", "tue", "wed", "thu", "fri", "sat", "sun".
        tz   : timezone to determine the day. Default "UTC".

        TimeCondition('weekday', days=['monday', 'tuesday', 'wednesday'])
        TimeCondition('weekday', days=['friday'], tz='America/New_York')

    'hours_since'
        The current node confirmed between min_hours and max_hours after the ref node.
        ref       : NodeRef with attribute='timestamp' pointing to another node.
        min_hours : minimum hours between ref and current node (inclusive). Default 0.
        max_hours : maximum hours between ref and current node (inclusive). None = no limit.

        TimeCondition('hours_since', ref=NodeRef(0, 'timestamp'), min_hours=2, max_hours=48)
        TimeCondition('hours_since', ref=NodeRef(-1, 'timestamp'), min_hours=0, max_hours=24)

        Note: the internal kind is always 'hours_since' and the bounds are
        always in HOURS. In the DSL the general spelling is
        @time_since($N.timestamp, min=..., max=..., unit=h|m|s): the parser
        converts min/max from the given unit into hours and builds this same
        kind. @hours_since is the DSL alias that always uses hours.

    Combination with price conditions
    ──────────────────────────────────
    TimeCondition are ConditionExpr — they can be combined with & and | in the DSL
    and with ConditionAnd/ConditionOr in the Python API:

        # price > node 0 AND occurs during NY session
        ConditionAnd([
            Condition('>', NodeRef(0, 'value')),
            TimeCondition('between', start='09:30', end='16:00', tz='America/New_York'),
        ])
    """

    kind:      TimeConditionKind
    start:     str | None = None       # "HH:MM" for between / after
    end:       str | None = None       # "HH:MM" for between / before
    tz:        str        = "UTC"
    days:      tuple[str, ...] = field(default_factory=tuple)
    ref:       NodeRef | None = None   # for hours_since
    min_hours: float = 0.0
    max_hours: float | None = None

    def __init__(
        self,
        kind:      TimeConditionKind,
        start:     str | None = None,
        end:       str | None = None,
        tz:        str        = "UTC",
        days:      "list[str] | tuple[str, ...] | None" = None,
        ref:       NodeRef | None = None,
        min_hours: float = 0.0,
        max_hours: float | None = None,
    ):
        object.__setattr__(self, 'kind',      kind)
        object.__setattr__(self, 'start',     start)
        object.__setattr__(self, 'end',       end)
        object.__setattr__(self, 'tz',        tz)
        object.__setattr__(self, 'days',      tuple(days) if days else ())
        object.__setattr__(self, 'ref',       ref)
        object.__setattr__(self, 'min_hours', float(min_hours))
        object.__setattr__(self, 'max_hours', float(max_hours) if max_hours is not None else None)

        # Validation at construction
        if kind == 'between':
            if start is None or end is None:
                raise TimeConditionError("TimeCondition('between') requires start and end.")
        elif kind == 'after':
            if start is None:
                raise TimeConditionError("TimeCondition('after') requires start.")
        elif kind == 'before':
            if end is None:
                raise TimeConditionError("TimeCondition('before') requires end.")
        elif kind == 'weekday':
            if not days:
                raise TimeConditionError("TimeCondition('weekday') requires at least one day in days.")
            unknown = [d for d in (days or []) if d.lower() not in _WEEKDAY_MAP]
            if unknown:
                raise TimeConditionError(
                    f"TimeCondition('weekday'): unknown days {unknown}. "
                    f"Valid: {list(_WEEKDAY_MAP.keys())}"
                )
        elif kind == 'hours_since':
            if ref is None:
                raise TimeConditionError(
                    "TimeCondition('hours_since') requires ref=NodeRef(N, 'timestamp')."
                )
            if ref.attribute != 'timestamp':
                raise TimeConditionError(
                    f"TimeCondition('hours_since'): ref.attribute must be 'timestamp', "
                    f"received '{ref.attribute}'. Use NodeRef(N, 'timestamp')."
                )
        else:
            raise TimeConditionError(f"Unknown TimeCondition kind: {kind!r}")


@dataclass(frozen=True)
class TagCondition:
    """
    Condition on the value of a tag on the current node.

    Compares ``pivot.tags[tag_key]`` against a literal value using
    '==' or '!='.

    Useful for expressing OR between tags from different keys that cannot
    be expressed with tag_filters (where all conditions are AND):

        tag(key1)==val1 | tag(key2)==val2

    Parameters
    ──────────
    tag_key  : tag name ('nbs', 'hhll', 'type', ...)
    operator : '==' or '!='
    value    : string value to compare against

    Examples
    ────────
        TagCondition('nbs', '==', 'neu')
        TagCondition('hhll', '==', 'HH')
        TagCondition('nbs', '!=', 'shk')

        # Combined with OR across different keys
        ConditionOr([
            TagCondition('nbs',  '==', 'neu'),
            TagCondition('hhll', '==', 'HH'),
        ])
    """
    tag_key:  str
    operator: Literal['==', '!=']
    value:    str


@dataclass(frozen=True)
class ConditionAnd:
    """
    All conditions must be satisfied (logical AND).

    Example
    ───────
        ConditionAnd([
            Condition('>', 3500),
            Condition('<', NodeRef(0, 'value')),
            TimeCondition('between', start='09:00', end='17:00'),
        ])
    """
    conditions: tuple['ConditionExpr', ...]

    def __init__(self, conditions: list['ConditionExpr']):
        object.__setattr__(self, 'conditions', tuple(conditions))


@dataclass(frozen=True)
class ConditionOr:
    """
    At least one condition must be satisfied (logical OR).

    Example
    ───────
        ConditionOr([
            Condition('>', NodeRef(0, 'value', 1.02)),
            TimeCondition('after', start='14:00'),
        ])
    """
    conditions: tuple['ConditionExpr', ...]

    def __init__(self, conditions: list['ConditionExpr']):
        object.__setattr__(self, 'conditions', tuple(conditions))


# Union type of any valid condition expression
ConditionExpr = Union[Condition, TimeCondition, TagCondition, ConditionAnd, ConditionOr, RangeCondition]


# ══════════════════════════════════════════════════════════════════════════════
# TIME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_hhmm(s: str) -> tuple[int, int]:
    """Parses "HH:MM" → (hour, minute). Raises ValueError if format is invalid."""
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise PatternSyntaxError(f"Invalid time format: {s!r}. Expected 'HH:MM'.")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise PatternSyntaxError(f"Time out of range: {s!r}.")
    return h, m


def _ts_to_datetime(ts_ms: float, tz: str):
    """
    Converts a timestamp in ms (UTC epoch) to datetime with timezone.
    Uses zoneinfo (Python 3.9+) without external dependencies.
    """
    from datetime import datetime, timezone
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        # Python < 3.9 — fallback to UTC ignoring requested tz
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

    zone = ZoneInfo("UTC") if tz == "UTC" else ZoneInfo(tz)
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=zone)


def _eval_time_condition(
    tc:               TimeCondition,
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> bool:
    """
    Evaluates a TimeCondition on the current node.

    Parameters
    ──────────
    window : list of size ``len(pattern.nodes)`` where positions
             of absent optional nodes contain ``None``.
             If the current node (``window[current_node_idx]``) is ``None``,
             returns ``False`` safely.
    """
    current = window[current_node_idx]
    # Guard: absent current node (shouldn't occur, but for safety)
    if current is None:
        return False
    ts_ms = current.timestamp   # ts_ms of the REAL pivot bar

    # ── between ──────────────────────────────────────────────────────────────
    if tc.kind == 'between':
        dt = _ts_to_datetime(ts_ms, tc.tz)
        sh, sm = _parse_hhmm(tc.start)
        eh, em = _parse_hhmm(tc.end)
        cur_minutes = dt.hour * 60 + dt.minute
        start_min   = sh * 60 + sm
        end_min     = eh * 60 + em

        if start_min <= end_min:
            # normal range: 09:30 → 16:00
            return start_min <= cur_minutes <= end_min
        else:
            # overnight range: 22:00 → 04:00
            return cur_minutes >= start_min or cur_minutes <= end_min

    # ── after ─────────────────────────────────────────────────────────────────
    if tc.kind == 'after':
        dt = _ts_to_datetime(ts_ms, tc.tz)
        sh, sm = _parse_hhmm(tc.start)
        cur_minutes   = dt.hour * 60 + dt.minute
        start_minutes = sh * 60 + sm
        return cur_minutes >= start_minutes

    # ── before ────────────────────────────────────────────────────────────────
    if tc.kind == 'before':
        dt = _ts_to_datetime(ts_ms, tc.tz)
        eh, em = _parse_hhmm(tc.end)
        cur_minutes = dt.hour * 60 + dt.minute
        end_minutes = eh * 60 + em
        return cur_minutes < end_minutes

    # ── weekday ───────────────────────────────────────────────────────────────
    if tc.kind == 'weekday':
        dt = _ts_to_datetime(ts_ms, tc.tz)
        current_weekday = dt.weekday()   # 0=Monday ... 6=Sunday
        valid_weekdays  = {_WEEKDAY_MAP[d.lower()] for d in tc.days}
        return current_weekday in valid_weekdays

    # ── hours_since ───────────────────────────────────────────────────────────
    if tc.kind == 'hours_since':
        ref_pos = _resolve_position(tc.ref.position, current_node_idx, window)
        if ref_pos is None:
            return False
        ref_node = window[ref_pos]
        if ref_node is None:
            # The referenced node is an absent optional → condition not satisfied
            return False
        ref_ts_ms   = ref_node.timestamp
        delta_hours = (ts_ms - ref_ts_ms) / (1000.0 * 3600.0)
        if delta_hours < tc.min_hours:
            return False
        if tc.max_hours is not None and delta_hours > tc.max_hours:
            return False
        return True

    return False   # unknown kind — shouldn't reach here (validated in __init__)


# ══════════════════════════════════════════════════════════════════════════════
# TAG CONDITION EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════


def _eval_tag_condition(
    tc:               'TagCondition',
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> bool:
    """
    Evaluates a TagCondition on the current node.

    Compares ``pivot.tags[tc.tag_key]`` (of the current node) against
    ``tc.value`` using ``tc.operator``.

    Returns False if the current node is None (absent optional node).
    """
    current = window[current_node_idx]
    if current is None:
        return False

    actual = current.tags.get(tc.tag_key)
    if actual is None:
        return False   # tag not present

    if tc.operator == '==':
        return actual == tc.value
    else:  # '!='
        return actual != tc.value


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_condition(
    expr:             'ConditionExpr',
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> bool:
    """
    Evaluates a condition expression on the current pattern window.

    Parameters
    ──────────
    expr             : the expression to evaluate
    current_node_idx : position of the node being evaluated within the
                       window (index in the full pattern, not in the
                       compressed pivot window).
    window           : list of size ``len(pattern.nodes)``. Each position
                       corresponds to a pattern node; contains the PivotPoint
                       assigned to that node, or ``None`` if the node is
                       optional and absent in the current match candidate.

    Returns True if the condition is satisfied, False otherwise.
    Returns False if a position reference is out of range or points to
    an absent optional node (slot None).

    Recursive for ConditionAnd / ConditionOr.
    """
    if isinstance(expr, Condition):
        return _eval_atomic(expr, current_node_idx, window)
    elif isinstance(expr, RangeCondition):
        return _eval_range_condition(expr, current_node_idx, window)
    elif isinstance(expr, TimeCondition):
        return _eval_time_condition(expr, current_node_idx, window)
    elif isinstance(expr, TagCondition):
        return _eval_tag_condition(expr, current_node_idx, window)
    elif isinstance(expr, ConditionAnd):
        return all(
            evaluate_condition(c, current_node_idx, window)
            for c in expr.conditions
        )
    elif isinstance(expr, ConditionOr):
        return any(
            evaluate_condition(c, current_node_idx, window)
            for c in expr.conditions
        )
    else:
        raise PatternSyntaxError(f"Unknown condition type: {type(expr)}")


def _eval_value(
    expr:             'ValueExpr',
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> 'float | None':
    """
    Evaluates a value expression to a number.

    Returns ``None`` (safely) if:
      · a NodeRef/CurrentRef points to an absent or out-of-range slot,
      · there is a division by zero in a BinaryExpr.

    Callers treat ``None`` as "condition not satisfied".
    """
    # ── Numeric constant ─────────────────────────────────────────────────────
    if isinstance(expr, (int, float)):
        return float(expr)

    # ── Current node ─────────────────────────────────────────────────────────
    if isinstance(expr, CurrentRef):
        current = window[current_node_idx]
        if current is None:
            return None
        return float(getattr(current, expr.attribute))

    # ── Reference to another node ────────────────────────────────────────────
    if isinstance(expr, NodeRef):
        ref_idx = _resolve_position(expr.position, current_node_idx, window)
        if ref_idx is None:
            return None
        ref_node = window[ref_idx]
        if ref_node is None:
            return None
        return float(getattr(ref_node, expr.attribute)) * expr.multiplier

    # ── Absolute value ───────────────────────────────────────────────────────
    if isinstance(expr, AbsExpr):
        v = _eval_value(expr.operand, current_node_idx, window)
        return None if v is None else abs(v)

    # ── Binary operation ─────────────────────────────────────────────────────
    if isinstance(expr, BinaryExpr):
        left  = _eval_value(expr.left,  current_node_idx, window)
        right = _eval_value(expr.right, current_node_idx, window)
        if left is None or right is None:
            return None
        if expr.op == '+':
            return left + right
        if expr.op == '-':
            return left - right
        if expr.op == '*':
            return left * right
        if expr.op == '/':
            if right == 0.0:
                return None          # division by zero → undefined expression
            return left / right
        raise PatternSyntaxError(f"Unknown arithmetic operator: {expr.op!r}")

    raise PatternSyntaxError(
        f"Unknown value expression: {type(expr)}. "
        f"(NodeTypeRef is not numeric; use it only with '==' / '!=')."
    )


def _eval_range_condition(
    rc:               RangeCondition,
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> bool:
    """
    Evaluates ``low <= expr <= high``. Returns False if any part is None.
    """
    val = _eval_value(rc.expr, current_node_idx, window)
    if val is None:
        return False
    low = _eval_value(rc.low, current_node_idx, window)
    if low is None:
        return False
    high = _eval_value(rc.high, current_node_idx, window)
    if high is None:
        return False
    return low <= val <= high


def _eval_atomic(
    cond:             Condition,
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> bool:
    """
    Evaluates an atomic Condition.

    ``window`` may contain ``None`` at positions of absent optional nodes.
    If a reference points to a ``None`` slot, returns ``False``.

    Two modes:
      · explicit lhs (``cond.lhs is not None``): both sides are evaluated as
        numeric expressions and compared.
      · implicit lhs (classic): the left side is the current node and the
        compared attribute is derived from the operand (unit coherence).
    """
    current = window[current_node_idx]
    # The current node must always be present (it's the one calling matches())
    if current is None:
        return False
    op = cond.operator

    # ── explicit lhs → numeric expression comparison ─────────────────────────
    if cond.lhs is not None:
        if isinstance(cond.operand, NodeTypeRef):
            raise PatternSyntaxError(
                "NodeTypeRef cannot be compared against an arithmetic expression. "
                "Use 'type(N)' only in the classic form (== type(N) / != type(N))."
            )
        lhs_val = _eval_value(cond.lhs, current_node_idx, window)
        if lhs_val is None:
            return False
        rhs_val = _eval_value(cond.operand, current_node_idx, window)  # type: ignore[arg-type]
        if rhs_val is None:
            return False
        return _compare(lhs_val, op, rhs_val)

    # ── Absolute numeric operand ─────────────────────────────────────────────
    if isinstance(cond.operand, (float, int)):
        lhs = current.value
        rhs = float(cond.operand)
        return _compare(lhs, op, rhs)

    # ── Reference to another node's attribute (classic: same attribute) ──────
    if isinstance(cond.operand, NodeRef):
        ref_idx = _resolve_position(cond.operand.position, current_node_idx, window)
        if ref_idx is None:
            return False
        ref_node = window[ref_idx]
        if ref_node is None:
            # The referenced node is an absent optional → condition not satisfied
            return False
        attribute = cond.operand.attribute
        rhs = getattr(ref_node, attribute) * cond.operand.multiplier
        lhs = getattr(current,  attribute)   # same attribute → coherent units
        return _compare(lhs, op, rhs)

    # ── Reference to another node's type ─────────────────────────────────────
    if isinstance(cond.operand, NodeTypeRef):
        if op not in ('==', '!='):
            raise PatternSyntaxError(
                f"NodeTypeRef only supports '==' and '!=' as operator, "
                f"received '{op}'"
            )
        ref_idx = _resolve_position(cond.operand.position, current_node_idx, window)
        if ref_idx is None:
            return False
        ref_node = window[ref_idx]
        if ref_node is None:
            # The referenced node is an absent optional → condition not satisfied
            return False
        ref_type = ref_node.type
        lhs = current.type
        return (lhs == ref_type) if op == '==' else (lhs != ref_type)

    # ── Expression operand (CurrentRef / BinaryExpr / AbsExpr) ───────────────
    # implicit lhs = current node's value.
    if isinstance(cond.operand, (CurrentRef, BinaryExpr, AbsExpr)):
        rhs_val = _eval_value(cond.operand, current_node_idx, window)
        if rhs_val is None:
            return False
        return _compare(current.value, op, rhs_val)

    raise PatternSyntaxError(f"Unknown operand type: {type(cond.operand)}")


def _resolve_position(
    position:         int,
    current_node_idx: int,
    window:           'list[PivotPoint | None]',
) -> 'int | None':
    """
    Resolves the position of a referenced node within the window.

    · position >= 0 → absolute from the start of the pattern ($0, $1, $2...)
    · position < 0  → relative to the current node ($-1 = previous in pattern)

    Returns None if the position is out of range.
    Note: does not check if ``window[idx]`` is None (absent optional slot);
    callers (_eval_atomic, _eval_time_condition) check that.
    """
    if position >= 0:
        idx = position
    else:
        idx = current_node_idx + position

    if 0 <= idx < len(window):
        return idx
    return None


def _compare(lhs: float, op: str, rhs: float) -> bool:
    """Applies the comparison operator."""
    match op:
        case '>':  return lhs > rhs
        case '<':  return lhs < rhs
        case '>=': return lhs >= rhs
        case '<=': return lhs <= rhs
        case '==': return lhs == rhs
        case '!=': return lhs != rhs
        case _:    raise PatternSyntaxError(f"Unknown operator: {op!r}")

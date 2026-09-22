"""
pattern_dsl.py
==============
DSL string parser → Pattern / PatternNode / Condition*.

Converts readable strings into pattern matching system objects
without needing to instantiate PatternNode, Condition, NodeRef, etc. manually.

Usage
─────
    from tradetropy.ta.pattern.dsl import parse_pattern

    p = parse_pattern(\"\"\"
        L[nbs=boo]
        H[nbs=neu]  > $0
        L[nbs=boo]  > $0
        H[nbs=neu]  > $1
    \"\"\", tag="bullish_impulse")

    self.setup = self.add_pattern_matcher(
        base_pivot=self.cpivot,
        decorators=[self.nbs],
        pattern = p,
    )

Grammar
───────
Each non-empty (and non-comment) line is a node:

    [~] TYPE[?][{min,max}] [TAGS] [CONDITIONS]

TYPE
    H | L | any

~  (non-captured, immediate prefix before TYPE)
    Marks the node as non-captured. Participates in matching (filters,
    tags, conditions, NodeRef references) but does NOT appear in
    ``MatchResult.nodes``. Useful for intermediate pivots that serve
    as a reference but are not of interest in the result.
    Compatible with ``?``: ``~L?`` is both optional and non-captured.
    Examples: ~H  ~L?  ~any  ~H[nbs=neu]  ~L?[nbs=boo] > $0

?  (optional, immediate suffix to TYPE)
    Marks the node as optional. The pattern matches whether or not
    the corresponding pivot exists.
    Examples: H?  L?  any?  H?[nbs=neu]  L?[nbs=boo] > $0

{min,max}  (repetition, immediate suffix to TYPE)
    Repeats the node: ``min`` mandatory copies plus ``max - min`` optional
    ones. ``{n}`` is shorthand for exactly n copies. Cannot be combined
    with ``?`` (which already means {0,1}).

    Because pattern windows are contiguous runs of pivots, a repeated node
    is an ELASTIC GAP of between ``min`` and ``max`` pivots, which is how a
    pattern expresses "any number of intermediate pivots" without knowing
    their count in advance:

        L[nbs=neu]
        ~any{1,3}  tag(type)!=H | (tag(nbs)!=neu & tag(nbs)!=shk)
        L[nbs=neu]

    reads as "two neutralizer lows with 1 to 3 pivots between them, none of
    which is a neutralizer/shocker high".

    Cost note: matching enumerates subsets of optional nodes, so work grows
    as 2^(optional nodes). Keep ranges tight; the parser rejects patterns
    expanding past ``_MAX_OPTIONAL_NODES`` optional nodes.

    References ($N) count LOGICAL lines as written, and are remapped to the
    expanded positions automatically. An absolute ``$k`` pointing at a
    quantified node resolves to its first copy; pointing at a node whose
    minimum is 0 is rejected (it may be absent). Relative references
    (``$-1``) are rejected inside a quantified node, where each copy sits at
    a different distance.

    Examples: ~any{1,3}  ~L{0,4} tag(nbs)!=neu  H{2}

TAGS  (optional)
    [key=val, key=val, ...]
    Alternative values can be separated by ``|``:
        [nbs=neu|shk]      → tag 'nbs' must be 'neu' OR 'shk'
        [nbs=neu, hhll=HH] → tag 'nbs' = 'neu' AND tag 'hhll' = 'HH'
    Example: [nbs=neu, hhll=HH]

CONDITIONS  (optional)
    Sequence of EXPR separated by & (implicit AND) or | (OR).
    Groups with parentheses for precedence control.

EXPR
    OP OPERAND
    @TIME_EXPR
    tag(KEY)==VAL   |  tag(KEY)!=VAL
    ( EXPR (&|) EXPR ... )

OP
    >  <  >=  <=  ==  !=

OPERAND  (can also be combined in arithmetic EXPRESSIONS)
    50000           → absolute (float/int)
    $0              → NodeRef(0, 'value')           price of node 0
    $0.value        → NodeRef(0, 'value')           same as $0
    $0.index        → NodeRef(0, 'index')           bar index of node 0
    $0.timestamp    → NodeRef(0, 'timestamp')       timestamp of node 0
    $0*1.02         → NodeRef(0, 'value', 1.02)     price of node 0 × 1.02
    $-1             → NodeRef(-1, 'value')          price of previous node
    $-1*0.98        → NodeRef(-1, 'value', 0.98)
    $.              → CurrentRef('value')           price of CURRENT node
    $.index         → CurrentRef('index')           index of current node
    $.timestamp     → CurrentRef('timestamp')       timestamp of current node
    type(0)         → NodeTypeRef(0)                type ('H'/'L') of node 0
    type(-1)        → NodeTypeRef(-1)               type of previous node

ARITHMETIC EXPRESSIONS  (between value operands)
    Operators  + - * /  with parentheses and abs(...). Standard precedence
    (* / before + -). Spaces around operators are optional: `$1 - $0` and
    `$1-$0` both parse as subtraction, and likewise `(5-6)`, `($1-$0)*2`.
    (A leading '-' stays a negative literal only in prefix position, e.g.
    `> -5` means "greater than minus five".)

    > $0 + 50                       current price > node0 + 50
    > $1 + ($1 - $0) * 0.618        Fibonacci extension of the $0→$1 leg
    abs($. - $0) > 100              absolute distance to node 0 greater than 100

EXPLICIT LHS  (left side of the comparison)
    By default the lhs is the current node. You can write a complete
    expression on the left to compare legs or arbitrary relationships:

    ($. - $1) > ($1 - $0)           second leg greater than the first
    $.index > $0.index              current index greater than node 0's

RANGES  (operator 'in')
    EXPR in [LOW, HIGH]   →   LOW <= EXPR <= HIGH   (inclusive bounds)
    Ideal for retracements/extensions:

    ($1 - $.) / ($1 - $0) in [0.382, 0.618]    Fibonacci pullback retracement

    Note: division by zero makes the expression undefined and the
    condition is considered NOT satisfied (no error raised).

TAG_EXPR  (tag conditions, prefix tag() — no @)
    tag(nbs)==neu            → current.tags['nbs'] == 'neu'
    tag(nbs)!=shk            → current.tags['nbs'] != 'shk'
    tag(hhll)==HH            → current.tags['hhll'] == 'HH'

    Combinable with price and time conditions via & and |:
        (tag(nbs)==neu | tag(hhll)==HH) & > $0

TIME_EXPR  (time conditions, prefix @)
    @between(09:30, 16:00)                         hour of day in UTC, between 09:30 and 16:00
    @between(09:30, 16:00, tz=America/New_York)    same in NY timezone
    @after(14:00)                                  after 14:00 UTC
    @after(09:30, tz=America/New_York)             after 09:30 NY
    @before(22:00)                                 before 22:00 UTC
    @before(16:00, tz=Europe/London)               before 16:00 London
    @weekday(monday, tuesday, wednesday)            weekdays in UTC
    @weekday(friday, tz=America/New_York)           Friday in NY
    @time_since($0.timestamp, min=2, max=168)      between 2 and 168h since node 0 (unit=h default)
    @time_since($0.timestamp, min=5, max=15, unit=m) between 5 and 15 minutes since node 0
    @time_since($0.timestamp, min=30, unit=s)      at least 30 seconds since node 0
    @hours_since($0.timestamp, min=2, max=168)     alias of @time_since with unit=h (always hours)
    @hours_since($-1.timestamp, min=0, max=24)     up to 24h since previous node

    Units (unit=, @time_since only): 'h' hours (default), 'm' minutes, 's' seconds.
    min/max are given in the chosen unit. @hours_since keeps the legacy
    hours-only behavior and rejects unit=.

Precedence and grouping
───────────────────────
    > $0 & > 50000                        AND of two price conditions
    > $0 | > 55000                        OR of two price conditions
    > $0 & @between(09:00, 17:00)         price AND hour of day
    @between(09:00, 17:00) | @after(20:00) OR between time conditions
    (> $0 & == type(0)) | @weekday(monday) AND grouped, then OR with time

Note on NodeRef and lhs
───────────────────────
The lhs of the comparison uses the SAME attribute as the rhs:
    > $0.value     → current_price     > node0_price      ✓
    > $0.index     → current_index     > node0_index      ✓
    > $0.timestamp → current_timestamp > node0_timestamp   ✓

Note on NodeRef with optional nodes
────────────────────────────────────
Indices in NodeRef ($0, $1, ...) always reference the position in the
ORIGINAL pattern, including optional nodes. If the referenced node is
optional and absent in a match candidate, the condition returns False
and that candidate is discarded.

Comments
────────
    Lines starting with '#' are ignored.

Line continuation
─────────────────
    A long condition chain can be split across several physical lines. A
    line is treated as a CONTINUATION of the previous line (glued onto it)
    when it starts with one of these markers:

        &  |                       logical connectors
        > < >= <= == !=            comparison operators (bare condition)
        (                          a parenthesized group

    This is purely cosmetic (folded once at parse time) and produces
    exactly the same Pattern as the single-line form. It also allows
    writing the node "bare" on its own line, with the entire condition
    chain — including the first condition — indented below it:

        ~L[nbs=boo]
            > $0
            & ($3-$2) >= ($1-$0)

        H[nbs=neu|shk] > $1
            & @between(22:30, 20:30)
            & @between(22:30, 16:30)

    are respectively equivalent to:

        ~L[nbs=boo] > $0 & ($3-$2) >= ($1-$0)
        H[nbs=neu|shk] > $1 & @between(22:30, 20:30) & @between(22:30, 16:30)

    Note: a bare leading '@' is NOT a continuation marker (it would be
    ambiguous with the standalone '@active_bars(...)' directive); an
    '@time_expr' condition can still be chained after an explicit '&'/'|'.

    Node lines always start with '~', 'H', 'L' or 'any', and directives with
    '$' or '@active_bars', so none of the continuation markers above can
    ever collide with the start of a node or directive.

Complete examples
─────────────────
    # minimum — 2 nodes
    parse_pattern(\"\"\"
        L
        H > $0
    \"\"\", tag="basic")

    # with optional node — the pullback may or may not exist
    parse_pattern(\"\"\"
        H[nbs=neu]
        L?[nbs=boo]           # this pivot may not be there
        H[nbs=neu]  > $0
    \"\"\", tag="hh_optional_pullback")

    # with NBS tags
    parse_pattern(\"\"\"
        L[nbs=boo]
        H[nbs=neu]  > $0
        L[nbs=boo]  > $0
        H[nbs=neu]  > $1
    \"\"\", tag="bullish_impulse")

    # with time conditions
    parse_pattern(\"\"\"
        L[nbs=boo]  @between(09:30, 16:00, tz=America/New_York)
        H[nbs=neu]  > $0 & @between(09:30, 16:00, tz=America/New_York)
    \"\"\", tag="ny_session")

    # with weekdays
    parse_pattern(\"\"\"
        L[nbs=boo]  @weekday(monday, tuesday, wednesday)
        H[nbs=neu]  > $0 & @weekday(monday, tuesday, wednesday)
    \"\"\", tag="week_start")

    # with time distance between pivots (hours)
    parse_pattern(\"\"\"
        L[nbs=boo]
        H[nbs=neu]  > $0 & @time_since($0.timestamp, min=2, max=72)
    \"\"\", tag="fast_impulse")

    # with time distance in minutes (unit=m) — intraday scalping window
    parse_pattern(\"\"\"
        L[nbs=boo]
        H[nbs=neu]  > $0 & @time_since($0.timestamp, min=5, max=30, unit=m)
    \"\"\", tag="minutes_window")

    # OR between time conditions
    parse_pattern(\"\"\"
        L
        H > $0 & (@between(09:30, 16:00, tz=America/New_York) | @after(20:00))
    \"\"\", tag="or_schedule")

    # with NBS + HHLL tags and multiplier
    parse_pattern(\"\"\"
        H[nbs=neu, hhll=HH]
        L[nbs=boo, hhll=HL]  > $0*0.95
        H[nbs=neu, hhll=HH]  > $1
    \"\"\", tag="hh_hl_hh")

    # NodeRef by bar index
    parse_pattern(\"\"\"
        H
        L > $-1
        H > $0.index & > $0
    \"\"\", tag="with_index")

    # with comments
    parse_pattern(\"\"\"
        # Base support
        L[nbs=boo]  @between(09:30, 16:00, tz=America/New_York)
        # First impulse — above support, same schedule
        H[nbs=neu]  > $0 & @between(09:30, 16:00, tz=America/New_York)
    \"\"\", tag="with_comments")

    # optional center node — the pullback confirms if it happens, but is not required
    parse_pattern(\"\"\"
        L[nbs=boo]
        H?[nbs=neu]  > $0         # optional impulse
        L[nbs=boo]   > $0         # second support
        H[nbs=neu]   > $1         # Higher High over first impulse (if it existed)
    \"\"\", tag="flexible_double_bottom")

    # OR inline in tag_filters — nbs=neu OR nbs=shk
    parse_pattern(\"\"\"
        L[nbs=boo]
        H[nbs=neu|shk]  > $0      # neutralizer OR shocker
        H[nbs=neu]      > $1
    \"\"\", tag="neu_or_shk")

    # TagCondition — OR between different keys
    parse_pattern(\"\"\"
        H  (tag(nbs)==neu | tag(hhll)==HH)
        L   > $0
        H   > $1
    \"\"\", tag="tag_or_between_keys")

    # TagCondition combined with price conditions
    parse_pattern(\"\"\"
        L
        L  > $0 & (tag(nbs)==boo | tag(nbs)==emp)
        H  > $1
    \"\"\", tag="tag_or_with_price")
"""

from __future__ import annotations

import re
from typing import Union

from tradetropy.exceptions import PatternSyntaxError
from tradetropy.ta.pattern.pattern import Pattern, PatternNode
from tradetropy.ta.pattern.conditions import (
    Condition,
    TimeCondition,
    TagCondition,
    ConditionAnd,
    ConditionOr,
    ConditionExpr,
    RangeCondition,
    NodeRef,
    NodeTypeRef,
    CurrentRef,
    BinaryExpr,
    AbsExpr,
    Operand,
    ValueExpr,
)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

# Internal token types
_TOK_OP       = "OP"        # > < >= <= == !=
_TOK_NUM      = "NUM"       # 50000  0.98  -5
_TOK_NODEREF  = "NODEREF"   # $0  $-1  $0.value  $0*1.02  $0.timestamp
_TOK_CURREF   = "CURREF"    # $.  $.value  $.index  $.timestamp
_TOK_TYPEREF  = "TYPEREF"   # type(0)  type(-1)
_TOK_TIMECOND = "TIMECOND"  # @between(...)  @after(...)  @before(...)  @weekday(...)  @time_since(...)  @hours_since(...)
_TOK_TAGCOND  = "TAGCOND"   # tag(key)==val  tag(key)!=val
_TOK_ABS      = "ABS"       # abs
_TOK_IN       = "IN"        # in   (range operator)
_TOK_ARITH    = "ARITH"     # + - * /   (arithmetic operators)
_TOK_AND      = "AND"       # &
_TOK_OR       = "OR"        # |
_TOK_LPAREN   = "LPAREN"    # (
_TOK_RPAREN   = "RPAREN"    # )
_TOK_LBRACKET = "LBRACKET"  # [
_TOK_RBRACKET = "RBRACKET"  # ]
_TOK_COMMA    = "COMMA"     # ,

# Order matters: 2+ char tokens before 1 char tokens.
# TIMECOND captures @word(...) with full content including nested parentheses.
# The TIMECOND regex is greedy and captures until the closing ) of @expr.
#
# Disambiguation notes:
#   · NUM may carry a leading '-' ("-5"). This is ambiguous with binary
#     subtraction. It is resolved by CONTEXT in _tokenize(): a "-<digits>"
#     token is split back into ARITH('-') + NUM when it directly follows a
#     value token (NUM, NODEREF, CURREF or ')'). So both "$0 - 5" and
#     "$0-5" tokenize as subtraction, while "> -5" (after an operator) stays
#     a negative literal. The other operators (+, *, /) never collide with
#     NUM, so they already work with or without surrounding spaces.
#   · NODEREF captures "$0*1.02" as a single token (classic multiplier);
#     "$0 * 1.02" (with spaces) tokenizes as NODEREF ARITH NUM.
#   · CURREF ("$.") comes BEFORE NODEREF.
#   · ABS and IN are reserved keywords within conditions.
_TOKEN_RE = re.compile(
    r"(?P<TIMECOND>@\w+\([^)]*\))"         # @between(...)  @after(...)  etc.
    r"|(?P<TAGCOND>tag\(\s*\w+\s*\)\s*(?:==|!=)\s*\w+)"  # tag(key)==val  tag(key)!=val
    r"|(?P<ABS>\babs\b)"
    r"|(?P<IN>\bin\b)"
    r"|(?P<TYPEREF>type\(\s*-?\d+\s*\))"
    r"|(?P<CURREF>\$\.(?:value|index|timestamp)?)"
    r"|(?P<NODEREF>\$-?\d+(?:\.\w+)?(?:\*[\d.]+)?)"
    r"|(?P<OP>>=|<=|!=|==|>|<)"
    r"|(?P<NUM>-?[\d]+(?:\.[\d]+)?)"
    r"|(?P<ARITH>[+\-*/])"
    r"|(?P<AND>&)"
    r"|(?P<OR>\|)"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<LBRACKET>\[)"
    r"|(?P<RBRACKET>\])"
    r"|(?P<COMMA>,)"
    r"|(?P<SPACE>\s+)"
)

# Token kinds after which a leading-'-' NUM is really binary subtraction of a
# positive number, not a negative literal (see _tokenize).
_VALUE_TOKENS = frozenset({_TOK_NUM, _TOK_NODEREF, _TOK_CURREF, _TOK_RPAREN})


def _tokenize(s: str) -> list[tuple[str, str]]:
    """Converts a conditions string into a list of (type, value) pairs."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if m is None:
            raise PatternSyntaxError(
                f"Unexpected character at position {pos}: {s[pos:pos+10]!r}\n"
                f"Full string: {s!r}"
            )
        kind = m.lastgroup
        val  = m.group()
        pos  = m.end()
        if kind == "SPACE":
            continue
        # Context-sensitive '-': a NUM with a leading minus that directly
        # follows a value token (a number, node ref, current ref or a closing
        # ')') is subtraction of a positive number, not a negative literal.
        # This lets tight arithmetic like "(5-6)" or "$0-5" parse without
        # requiring spaces around '-'. After an operator / '(' / start
        # (i.e. a prefix position) the '-' stays part of the negative number,
        # so "> -5" still means "greater than minus five".
        if (
            kind == _TOK_NUM
            and val[0] == "-"
            and tokens
            and tokens[-1][0] in _VALUE_TOKENS
        ):
            tokens.append((_TOK_ARITH, "-"))
            tokens.append((_TOK_NUM, val[1:]))
            continue
        tokens.append((kind, val))
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# OPERAND PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_operand(kind: str, val: str) -> Operand:
    """Converts a NUM, NODEREF or TYPEREF token into the correct operand."""

    if kind == _TOK_NUM:
        return float(val)

    if kind == _TOK_CURREF:
        # "$."  "$.value"  "$.index"  "$.timestamp"
        attr_part = val[2:]  # remove "$."
        attribute = attr_part if attr_part else "value"
        if attribute not in ("value", "index", "timestamp"):
            raise PatternSyntaxError(
                f"Unknown attribute in CurrentRef: {attribute!r}. "
                f"Valid options: 'value', 'index', 'timestamp'."
            )
        return CurrentRef(attribute)  # type: ignore[arg-type]

    if kind == _TOK_TYPEREF:
        # "type(-1)" → NodeTypeRef(-1)
        inner = val[4:].strip()       # remove "type"
        inner = inner.strip("()")     # remove parentheses
        return NodeTypeRef(int(inner.strip()))

    if kind == _TOK_NODEREF:
        # "$0"  "$-1"  "$0.value"  "$0.index"  "$0.timestamp"  "$0*1.02"  "$-1*0.98"
        s = val[1:]  # remove the "$"

        # separate multiplier  →  "$0*1.02"  →  s="0"  mult=1.02
        multiplier = 1.0
        if "*" in s:
            s, mult_str = s.split("*", 1)
            multiplier = float(mult_str)

        # separate attribute  →  "$0.index"  →  s="0"  attribute="index"
        attribute = "value"
        if "." in s:
            s, attribute = s.rsplit(".", 1)
            if attribute not in ("value", "index", "timestamp"):
                raise PatternSyntaxError(
                    f"Unknown attribute in NodeRef: {attribute!r}. "
                    f"Valid options: 'value', 'index', 'timestamp'."
                )

        position = int(s)
        return NodeRef(position, attribute, multiplier)  # type: ignore[arg-type]

    raise PatternSyntaxError(
        f"Unexpected token as operand: kind={kind!r} val={val!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TIME CONDITION PARSER
# ══════════════════════════════════════════════════════════════════════════════

# Regex to extract @name(content)
_TIMECOND_RE = re.compile(r"^@(\w+)\((.*)\)\s*$", re.DOTALL)

# Valid days (for the parser)
_VALID_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

# Time units accepted by @time_since (unit=), mapped to a hours factor.
# min/max are given in the chosen unit and converted to hours internally so
# the TimeCondition evaluator stays unit-agnostic (it always compares hours).
_TIME_UNIT_TO_HOURS = {
    "h": 1.0,
    "m": 1.0 / 60.0,
    "s": 1.0 / 3600.0,
}


def _parse_time_condition(val: str) -> TimeCondition:
    """
    Parses a TIMECOND token like "@name(args)" → TimeCondition.

    Supported formats:
        @between(09:30, 16:00)
        @between(09:30, 16:00, tz=America/New_York)
        @after(14:00)
        @after(09:30, tz=America/New_York)
        @before(22:00)
        @before(16:00, tz=Europe/London)
        @weekday(monday, tuesday, wednesday)
        @weekday(friday, tz=America/New_York)
        @time_since($0.timestamp, min=2, max=168)          # unit=h (default)
        @time_since($0.timestamp, min=5, max=15, unit=m)   # minutes
        @time_since($0.timestamp, min=30, unit=s)          # seconds
        @hours_since($0.timestamp, min=2, max=168)         # alias, always hours
        @hours_since($-1.timestamp, min=0, max=24)
        @hours_since($0.timestamp, min=4)
    """
    m = _TIMECOND_RE.match(val)
    if m is None:
        raise PatternSyntaxError(f"Invalid time condition format: {val!r}")

    name    = m.group(1).lower()
    content = m.group(2).strip()

    # Split arguments by comma, respecting no nesting
    raw_args = [a.strip() for a in content.split(",") if a.strip()]

    # Separate positional and keyword (key=value) args
    pos_args: list[str] = []
    kw_args:  dict[str, str] = {}
    for arg in raw_args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kw_args[k.strip().lower()] = v.strip()
        else:
            pos_args.append(arg)

    tz = kw_args.get("tz", "UTC")

    # ── @between ─────────────────────────────────────────────────────────────
    if name == "between":
        if len(pos_args) < 2:
            raise PatternSyntaxError(
                f"@between requires start and end: @between(09:30, 16:00) "
                f"or @between(09:30, 16:00, tz=America/New_York). "
                f"Received: {val!r}"
            )
        return TimeCondition('between', start=pos_args[0], end=pos_args[1], tz=tz)

    # ── @after ───────────────────────────────────────────────────────────────
    if name == "after":
        if not pos_args:
            raise PatternSyntaxError(
                f"@after requires a time: @after(14:00). Received: {val!r}"
            )
        return TimeCondition('after', start=pos_args[0], tz=tz)

    # ── @before ──────────────────────────────────────────────────────────────
    if name == "before":
        if not pos_args:
            raise PatternSyntaxError(
                f"@before requires a time: @before(22:00). Received: {val!r}"
            )
        return TimeCondition('before', end=pos_args[0], tz=tz)

    # ── @weekday ─────────────────────────────────────────────────────────────
    if name == "weekday":
        # Days are pos_args; tz is in kw_args
        days = [d.lower() for d in pos_args]
        if not days:
            raise PatternSyntaxError(
                f"@weekday requires at least one day: @weekday(monday, tuesday). "
                f"Received: {val!r}"
            )
        unknown = [d for d in days if d not in _VALID_DAYS]
        if unknown:
            raise PatternSyntaxError(
                f"@weekday: unknown days {unknown}. "
                f"Valid: monday tuesday wednesday thursday friday saturday sunday "
                f"(or abbreviations mon tue wed thu fri sat sun). "
                f"Received: {val!r}"
            )
        return TimeCondition('weekday', days=days, tz=tz)

    # ── @time_since / @hours_since ────────────────────────────────────────────
    #
    # @time_since is the general time-distance condition: min/max are expressed
    # in the unit given by unit=h|m|s (hours, minutes, seconds; default hours)
    # and converted to hours internally, so the evaluator stays unit-agnostic.
    # @hours_since is the backward-compatible alias — always hours, no unit=.
    if name in ("time_since", "hours_since"):
        # First positional arg: $N.timestamp
        if not pos_args:
            raise PatternSyntaxError(
                f"@{name} requires a node reference: "
                f"@{name}($0.timestamp, min=2, max=48). Received: {val!r}"
            )
        ref_str = pos_args[0].strip()
        if not ref_str.startswith("$"):
            raise PatternSyntaxError(
                f"@{name}: the first argument must be a NodeRef "
                f"($0.timestamp, $-1.timestamp, etc.). Received: {ref_str!r}"
            )

        # Parse the NodeRef manually (without going through the full tokenizer)
        ref_inner = ref_str[1:]   # remove "$"
        if "." not in ref_inner:
            raise PatternSyntaxError(
                f"@{name}: you must specify '.timestamp' explicitly. "
                f"Use $0.timestamp or $-1.timestamp, not $0 or $-1. "
                f"Received: {ref_str!r}"
            )
        pos_str, attr = ref_inner.rsplit(".", 1)
        if attr != "timestamp":
            raise PatternSyntaxError(
                f"@{name}: the NodeRef attribute must be 'timestamp', "
                f"received '{attr}'. Use $0.timestamp or $-1.timestamp."
            )

        try:
            position = int(pos_str)
        except ValueError:
            raise PatternSyntaxError(
                f"@{name}: invalid node position: {pos_str!r}. "
                f"Must be an integer ($0, $-1, $2, etc.)."
            )

        ref = NodeRef(position, 'timestamp')

        # ── unit → hours factor ──────────────────────────────────────────────
        # @hours_since is always hours and rejects unit= (use @time_since for
        # minutes/seconds); @time_since defaults to hours when unit= is omitted.
        if name == "hours_since":
            if "unit" in kw_args:
                raise PatternSyntaxError(
                    f"@hours_since does not accept unit= (it is always hours). "
                    f"Use @time_since($..., unit=m) or @time_since($..., unit=s) "
                    f"for minutes/seconds. Received: {val!r}"
                )
            unit = "h"
        else:
            unit = kw_args.get("unit", "h").strip().lower()

        if unit not in _TIME_UNIT_TO_HOURS:
            raise PatternSyntaxError(
                f"@{name}: unknown unit {unit!r}. "
                f"Valid units: 'h' (hours), 'm' (minutes), 's' (seconds). "
                f"Received: {val!r}"
            )
        factor = _TIME_UNIT_TO_HOURS[unit]

        # Extract min and max from kw_args, in the given unit, then to hours
        min_hours: float = float(kw_args.get("min", 0)) * factor
        max_hours: float | None = (
            float(kw_args["max"]) * factor if "max" in kw_args else None
        )

        return TimeCondition('hours_since', ref=ref, min_hours=min_hours, max_hours=max_hours)

    raise PatternSyntaxError(
        f"Unknown time condition: @{name}. "
        f"Valid: @between, @after, @before, @weekday, @time_since, @hours_since."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAG CONDITION PARSER
# ══════════════════════════════════════════════════════════════════════════════

# Regex to extract tag(key)==val or tag(key)!=val
_TAGCOND_RE = re.compile(
    r"^tag\(\s*(?P<key>\w+)\s*\)\s*(?P<op>==|!=)\s*(?P<val>\w+)\s*$"
)


def _parse_tag_condition(val: str) -> TagCondition:
    """
    Parses a TAGCOND token like "tag(key)==val" → TagCondition.

    Supported formats:
        tag(nbs)==neu
        tag(nbs)!=shk
        tag(hhll)==HH
        tag(type)==H
    """
    m = _TAGCOND_RE.match(val)
    if m is None:
        raise PatternSyntaxError(
            f"Invalid tag condition format: {val!r}.\n"
            f"Expected format: tag(key)==val or tag(key)!=val.\n"
            f"Example: tag(nbs)==neu"
        )

    key = m.group("key")
    op  = m.group("op")
    val_str = m.group("val")

    return TagCondition(tag_key=key, operator=op, value=val_str)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION PARSER  (recursive descent)
# ══════════════════════════════════════════════════════════════════════════════

class _CondParser:
    """
    Recursive descent parser for the conditions part of a line.

    Implemented grammar:
        top_level  := item ( '&' item | '|' item )*
        item       := atom ( '|' atom )*          ← OR chain
        atom       := OP OPERAND
                    | TIMECOND
                    | '(' group_body ')'
        group_body := atom ( ('&'|'|') atom )*

    The root level produces a list of ConditionExpr with implicit AND.
    Within that list there may be ConditionOr if there is '|' at the same level.
    """

    def __init__(self, tokens: list[tuple[str, str]]):
        self._tokens = tokens
        self._pos    = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _peek(self) -> tuple[str, str] | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> tuple[str, str]:
        tok = self._peek()
        if tok is None or tok[0] != kind:
            raise PatternSyntaxError(
                f"Expected token {kind!r}, "
                f"found: {tok!r}"
            )
        return self._consume()

    # ── main production ──────────────────────────────────────────────────────

    def parse(self) -> list[ConditionExpr]:
        """
        Parses the complete sequence and returns the list that goes into
        PatternNode.conditions (implicit AND between elements).
        """
        if not self._tokens:
            return []
        return self._parse_top_level()

    def _parse_top_level(self) -> list[ConditionExpr]:
        """
        Root level: list of items with implicit AND between them.
        Each item can be an atomic or an OR-chain.
        """
        items: list[ConditionExpr] = []

        while self._peek() is not None:
            kind = self._peek()[0]

            # End of group (shouldn't occur here, but protective)
            if kind == _TOK_RPAREN:
                break

            # Read an atomic
            left = self._parse_atom()

            # Is there an immediate OR? → OR-chain
            or_chain: list[ConditionExpr] = [left]
            while self._peek() and self._peek()[0] == _TOK_OR:
                self._consume()            # consume '|'
                or_chain.append(self._parse_atom())

            if len(or_chain) == 1:
                items.append(or_chain[0])
            else:
                items.append(ConditionOr(or_chain))

            # Is there a '&'? → next item of root level
            if self._peek() and self._peek()[0] == _TOK_AND:
                self._consume()            # consume '&'

        return items

    def _parse_atom(self) -> ConditionExpr:
        """
        Atomic:
            OP value_expr                         (implicit lhs = current node)
            OP TYPEREF                             (classic type comparison)
            value_expr OP value_expr               (explicit lhs)
            value_expr 'in' '[' value_expr ',' value_expr ']'   (range)
            TIMECOND  |  TAGCOND
            '(' group_body ')'                     (logical group)
        """
        tok = self._peek()
        if tok is None:
            raise PatternSyntaxError(
                "Expected a condition (operator, @time or parenthesis) "
                "but the conditions string ended."
            )

        kind, val = tok

        # ── Time condition @name(...) ────────────────────────────────────────
        if kind == _TOK_TIMECOND:
            self._consume()
            return _parse_time_condition(val)

        # ── Tag condition tag(key)==val / tag(key)!=val ──────────────────────
        if kind == _TOK_TAGCOND:
            self._consume()
            return _parse_tag_condition(val)

        # ── Condition with implicit lhs: OP OPERAND ──────────────────────────
        if kind == _TOK_OP:
            self._consume()
            op = val

            nxt = self._peek()
            if nxt is None:
                raise PatternSyntaxError(
                    f"Expected an operand after operator {op!r}, "
                    f"but the conditions string ended."
                )

            # Classic type comparison: OP type(N)
            if nxt[0] == _TOK_TYPEREF:
                self._consume()
                return Condition(op, _parse_operand(nxt[0], nxt[1]))  # type: ignore[arg-type]

            # rhs as value expression (includes arithmetic).
            rhs = self._parse_value_expr()
            return Condition(op, rhs)  # type: ignore[arg-type]  # implicit lhs

        # ── Parenthesized group: logical or arithmetic (explicit lhs) ────────
        if kind == _TOK_LPAREN:
            if self._paren_is_logical():
                self._consume()                # consume '('
                inner = self._parse_group_body()
                self._expect(_TOK_RPAREN)      # consume ')'
                return inner
            # Arithmetic paren → start of explicit lhs
            lhs = self._parse_value_expr()
            return self._parse_comparison_from_lhs(lhs)

        # ── Explicit lhs starting with a value operand ───────────────────────
        if kind in (_TOK_NUM, _TOK_NODEREF, _TOK_CURREF, _TOK_ABS):
            lhs = self._parse_value_expr()
            return self._parse_comparison_from_lhs(lhs)

        # ── Standalone TYPEREF (without preceding OP) is invalid ─────────────
        if kind == _TOK_TYPEREF:
            raise PatternSyntaxError(
                f"'type(N)' is only valid as an operand of a type comparison: "
                f"'== type(N)' or '!= type(N)'.  Received standalone: {val!r}"
            )

        raise PatternSyntaxError(
            f"Unexpected token when parsing condition: {tok!r}.\n"
            f"Expected an operator (>, <, >=, <=, ==, !=), "
            f"a value expression ($0, $., 50000, abs(...), '('...), "
            f"a time condition (@between, @after, @before, @weekday, @time_since, @hours_since), "
            f"a tag condition (tag(key)==val, tag(key)!=val) "
            f"or a parenthesis '('."
        )

    # ── Comparison / range from an already-parsed lhs ────────────────────────

    def _parse_comparison_from_lhs(self, lhs: 'ValueExpr') -> ConditionExpr:
        """
        After parsing an explicit lhs, expects:
            OP value_expr                          → Condition(op, rhs, lhs=lhs)
            'in' '[' value_expr ',' value_expr ']' → RangeCondition(lhs, low, high)
        """
        tok = self._peek()
        if tok is None:
            raise PatternSyntaxError(
                "Value expression without comparison operator.  "
                "Expected an operator (>, <, ==, ...) or 'in [a, b]' after the lhs."
            )

        # ── Range: expr in [low, high] ───────────────────────────────────────
        if tok[0] == _TOK_IN:
            self._consume()                       # consume 'in'
            self._expect(_TOK_LBRACKET)           # consume '['
            low = self._parse_value_expr()
            self._expect(_TOK_COMMA)              # consume ','
            high = self._parse_value_expr()
            self._expect(_TOK_RBRACKET)           # consume ']'
            return RangeCondition(lhs, low, high)

        # ── Comparison: lhs OP rhs ───────────────────────────────────────────
        if tok[0] == _TOK_OP:
            self._consume()
            op = tok[1]
            nxt = self._peek()
            if nxt is not None and nxt[0] == _TOK_TYPEREF:
                raise PatternSyntaxError(
                    "Cannot compare an arithmetic expression against 'type(N)'. "
                    "Type comparisons use the form '== type(N)' with implicit lhs."
                )
            rhs = self._parse_value_expr()
            return Condition(op, rhs, lhs=lhs)  # type: ignore[arg-type]

        raise PatternSyntaxError(
            f"Expected a comparison operator (>, <, >=, <=, ==, !=) "
            f"or 'in [a, b]' after the expression, found: {tok!r}"
        )

    # ── Value expression parser (arithmetic) ─────────────────────────────────
    #
    #   value_expr := term ( ('+'|'-') term )*
    #   term       := factor ( ('*'|'/') factor )*
    #   factor     := NUM | NODEREF | CURREF | abs '(' value_expr ')' | '(' value_expr ')'

    def _parse_value_expr(self) -> 'ValueExpr':
        node = self._parse_term()
        while self._peek() and self._peek()[0] == _TOK_ARITH and self._peek()[1] in ('+', '-'):
            op = self._consume()[1]
            right = self._parse_term()
            node = BinaryExpr(op, node, right)  # type: ignore[arg-type]
        return node

    def _parse_term(self) -> 'ValueExpr':
        node = self._parse_factor()
        while self._peek() and self._peek()[0] == _TOK_ARITH and self._peek()[1] in ('*', '/'):
            op = self._consume()[1]
            right = self._parse_factor()
            node = BinaryExpr(op, node, right)  # type: ignore[arg-type]
        return node

    def _parse_factor(self) -> 'ValueExpr':
        tok = self._peek()
        if tok is None:
            raise PatternSyntaxError(
                "Expected an expression operand (number, $0, $., abs(...) or '(') "
                "but the conditions string ended."
            )
        kind, val = tok

        # abs( value_expr )
        if kind == _TOK_ABS:
            self._consume()
            self._expect(_TOK_LPAREN)
            inner = self._parse_value_expr()
            self._expect(_TOK_RPAREN)
            return AbsExpr(inner)

        # ( value_expr )
        if kind == _TOK_LPAREN:
            self._consume()
            inner = self._parse_value_expr()
            self._expect(_TOK_RPAREN)
            return inner

        # leaf operands
        if kind in (_TOK_NUM, _TOK_NODEREF, _TOK_CURREF):
            self._consume()
            return _parse_operand(kind, val)  # type: ignore[return-value]

        raise PatternSyntaxError(
            f"Invalid expression operand: {tok!r}.\n"
            f"Valid: number (50000), node ($0, $-1, $0*1.02, $0.index, "
            f"$0.timestamp), current node ($., $.index, $.timestamp), "
            f"absolute value (abs(...)) or parenthesis '('."
        )

    def _paren_is_logical(self) -> bool:
        """
        Looking from a '(' (without consuming it), determines whether the group
        is LOGICAL (contains comparison / time / tag / & / | / 'in' at its
        immediate level) or ARITHMETIC (only + - * / and operands).

        Allows distinguishing '(> $0 & > $1)'  (logical)  from  '($1 - $0)'  (arithmetic).
        """
        depth = 0
        i = self._pos
        n = len(self._tokens)
        while i < n:
            kind = self._tokens[i][0]
            if kind == _TOK_LPAREN:
                depth += 1
            elif kind == _TOK_RPAREN:
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and kind in (
                _TOK_OP, _TOK_AND, _TOK_OR, _TOK_TIMECOND, _TOK_TAGCOND, _TOK_IN
            ):
                return True
            i += 1
        return False

    def _parse_group_body(self) -> ConditionExpr:
        """
        Body of a '(...)' group: one or more atomics with '&' or '|' between them.

        Precedence within the group:
            a & b | c & d  →  OR( AND(a, b), AND(c, d) )
        """
        items:     list[ConditionExpr] = []
        operators: list[str]           = []

        items.append(self._parse_atom())

        while self._peek() and self._peek()[0] in (_TOK_AND, _TOK_OR):
            op_tok = self._consume()
            operators.append(op_tok[0])
            items.append(self._parse_atom())

        # Trivial case: single atomic
        if not operators:
            return items[0]

        # All AND → ConditionAnd
        if all(o == _TOK_AND for o in operators):
            return ConditionAnd(items)

        # All OR → ConditionOr
        if all(o == _TOK_OR for o in operators):
            return ConditionOr(items)

        # Mixed AND/OR: group AND first, then OR between groups
        # Example: a & b | c & d  →  OR( AND(a,b), AND(c,d) )
        and_groups: list[list[ConditionExpr]] = []
        current:    list[ConditionExpr]       = [items[0]]

        for i, op in enumerate(operators):
            if op == _TOK_AND:
                current.append(items[i + 1])
            else:   # OR — close current group, open a new one
                and_groups.append(current)
                current = [items[i + 1]]
        and_groups.append(current)

        or_items: list[ConditionExpr] = [
            group[0] if len(group) == 1 else ConditionAnd(group)
            for group in and_groups
        ]
        return ConditionOr(or_items)


# ══════════════════════════════════════════════════════════════════════════════
# NODE LINE PARSER
# ══════════════════════════════════════════════════════════════════════════════

# [~] TYPE[?][{min,max}]  [TAGS]  rest_of_conditions
# The ~ prefix (immediately before the type, no space) marks the node
# as non-captured: participates in matching but doesn't appear in MatchResult.nodes.
# The ? suffix (immediately after the type, no space) marks the node
# as optional.  Examples: "~H?", "~L[nbs=boo]", "~any? > $0"
# The {min,max} (or {n}) suffix repeats the node, expanding at parse time
# into ``min`` mandatory copies plus ``max - min`` optional ones.
# Examples: "~any{1,3}", "~L{0,4} tag(nbs)!=neu", "H{2}"
_NODE_RE = re.compile(
    r"^(?P<nc>~)?"                  # non-captured ~ prefix (no space)
    r"(?P<type>H|L|any)"            # required type
    r"(?P<optional>\?)?"            # optional ? suffix (no space)
    r"(?:\{\s*(?P<qmin>\d+)\s*"     # repetition {min[,max]} (no space before {)
    r"(?:,\s*(?P<qmax>\d+)\s*)?\})?"
    r"(?:\[(?P<tags>[^\]]*)\])?"    # optional [tags]
    r"(?P<conds>.*)$"               # conditions (may be empty)
)

# Maximum number of optional nodes allowed in one pattern (after expanding
# every {min,max} quantifier).
#
# ``Pattern.try_match`` enumerates ``combinations(optional_idxs, n_omit)``, and
# ``PatternStore._scan_all`` calls it once per window size, so the total work
# per pivot grows as 2^K with K = number of optional nodes. Measured on 172k
# 15s MES bars (~4s baseline backtest): K=4 -> 4.6s, K=6 -> 5.8s, K=8 -> 10.3s,
# K=10 -> 28.8s, K=12 -> 117s. The cap keeps a typo like ``{0,30}`` from
# silently turning a 4-second backtest into an unusable one.
_MAX_OPTIONAL_NODES = 12

_TAG_RE = re.compile(r"(\w+)\s*=\s*([\w|]+)")

# Standalone global directive: @active_bars(end) or @active_bars(start, end).
# Bounds are non-negative integers (bars since the last pivot's confirmation
# bar). Parsed in parse_pattern(), not attached to any node.
_ACTIVE_BARS_RE = re.compile(
    r"^@active_bars\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)$"
)


def _parse_node_line(line: str) -> 'tuple[PatternNode, int, int]':
    """
    Parses a node line and returns ``(node, rep_min, rep_max)``.

    ``rep_min``/``rep_max`` come from the ``{min,max}`` quantifier and are
    both 1 when the node carries no quantifier. Expanding the repetition
    into concrete nodes is ``parse_pattern``'s job, since it also has to
    remap the ``$N`` references to the shifted positions.

    Format:
        [~] TYPE[?][{min,max}] [TAGS] [CONDITIONS]

    The ``~`` prefix immediately before the type (no space) marks the
    node as non-captured: participates in matching but doesn't appear in
    ``MatchResult.nodes`` (``PatternNode.captured=False``).

    The ``?`` suffix immediately after the type (no space) marks the
    node as optional (``PatternNode.optional=True``).

    Examples:
        "H"
        "L?"                                          ← optional node without tags or conds
        "~L?"                                         ← optional + non-captured
        "L[nbs=boo]"
        "~L[nbs=boo]"                                 ← non-captured node with tag
        "H?[nbs=neu]"                                 ← optional node with tag
        "~H?[nbs=neu]"                                ← optional + non-captured with tag
        "H[nbs=neu, hhll=HH]  > $0 & > 50000"
        "H?[nbs=neu]  > $0 & @between(09:30, 16:00, tz=America/New_York)"
        "L?[nbs=boo]  @weekday(monday, tuesday)"
        "H  > $0 & @hours_since($0.timestamp, min=2, max=48)"
        "any[nbs=neu]  (> $0 & == type(0)) | (!= type(0) & < $0*0.98)"
        "any?  > $0"                                  ← optional node without tag
    """
    line = line.strip()
    if not line:
        raise PatternSyntaxError("Empty node line.")

    m = _NODE_RE.match(line)
    if m is None:
        raise PatternSyntaxError(
            f"Could not parse: {line!r}\n"
            f"Expected format: [~] TYPE[?] [TAGS] [CONDITIONS]\n"
            f"  ~   : non-captured prefix (optional)\n"
            f"  TYPE: H | L | any\n"
            f"  ?   : optional suffix — marks node as optional\n"
            f"  TAGS: [nbs=neu, hhll=HH]   (optional)\n"
            f"  CONDITIONS: > $0 & > 50000  (optional)\n"
            f"              @between(09:30, 16:00, tz=America/New_York)"
        )

    # ── Non-captured ───────────────────────────────────────────────────────────
    nc_flag = m.group("nc") is not None

    # ── Type ──────────────────────────────────────────────────────────────────
    ptype = m.group("type")          # "H" | "L" | "any"

    # ── Optional ──────────────────────────────────────────────────────────────
    optional = m.group("optional") == "?"

    # ── Repetition {min,max} ──────────────────────────────────────────────────
    # ``{n}``     → exactly n copies
    # ``{n,m}``   → n mandatory copies + (m - n) optional ones
    # Mixing with ``?`` is rejected: ``?`` already means {0,1}, so
    # ``any?{0,3}`` would express the same idea twice, ambiguously.
    qmin_raw = m.group("qmin")
    if qmin_raw is None:
        rep_min = rep_max = 1
    else:
        if optional:
            raise PatternSyntaxError(
                f"Node {line!r} combines the optional suffix '?' with the "
                f"repetition {{{qmin_raw}...}}. Use only the quantifier: "
                f"'?' is equivalent to {{0,1}}."
            )
        rep_min = int(qmin_raw)
        qmax_raw = m.group("qmax")
        rep_max = rep_min if qmax_raw is None else int(qmax_raw)
        if rep_max < 1:
            raise PatternSyntaxError(
                f"Node {line!r}: repetition max must be >= 1, got {rep_max}."
            )
        if rep_min > rep_max:
            raise PatternSyntaxError(
                f"Node {line!r}: repetition min ({rep_min}) cannot be greater "
                f"than max ({rep_max})."
            )

    # ── Tags ──────────────────────────────────────────────────────────────────
    tag_filters: 'dict[str, str | set[str]]' = {}
    tags_str = m.group("tags")
    if tags_str:
        for tm in _TAG_RE.finditer(tags_str):
            key   = tm.group(1)
            raw   = tm.group(2)
            parts = raw.split("|")
            if len(parts) == 1:
                tag_filters[key] = raw
            else:
                tag_filters[key] = set(parts)

    # ── Conditions ────────────────────────────────────────────────────────────
    conds_str = m.group("conds").strip()
    conditions: list[ConditionExpr] = []
    if conds_str:
        tokens     = _tokenize(conds_str)
        parser     = _CondParser(tokens)
        conditions = parser.parse()

        # Verify all tokens were consumed
        if parser._pos < len(tokens):
            remaining = tokens[parser._pos:]
            raise PatternSyntaxError(
                f"Unprocessed tokens in conditions of {line!r}: "
                f"{remaining!r}"
            )

    return (
        PatternNode(
            type=ptype,             # type: ignore[arg-type]
            tag_filters=tag_filters,
            conditions=conditions,
            optional=optional,
            captured=not nc_flag,
        ),
        rep_min,
        rep_max,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE REMAPPING (for {min,max} expansion)
# ══════════════════════════════════════════════════════════════════════════════
#
# ``$N`` written in the DSL counts LOGICAL lines, but expanding a quantifier
# turns one line into several physical nodes, shifting every position after it.
# These helpers rebuild each condition expression with the positions the
# expanded pattern actually uses.
#
# The condition dataclasses are all frozen, so nothing is mutated: every node
# is reconstructed. ``remap`` is a callable that receives the position written
# by the user and returns the physical one (raising PatternSyntaxError when the
# reference is not expressible after expansion).

def _remap_value_expr(expr: 'ValueExpr', remap) -> 'ValueExpr':
    """Rebuild a value expression with remapped NodeRef positions."""
    if isinstance(expr, NodeRef):
        return NodeRef(
            position=remap(expr.position),
            attribute=expr.attribute,
            multiplier=expr.multiplier,
        )
    if isinstance(expr, AbsExpr):
        return AbsExpr(operand=_remap_value_expr(expr.operand, remap))
    if isinstance(expr, BinaryExpr):
        return BinaryExpr(
            op=expr.op,
            left=_remap_value_expr(expr.left, remap),
            right=_remap_value_expr(expr.right, remap),
        )
    # float / int / CurrentRef → no position to remap
    return expr


def _remap_condition(expr: 'ConditionExpr', remap) -> 'ConditionExpr':
    """Rebuild a condition expression with remapped node positions."""
    if isinstance(expr, Condition):
        new_lhs = None if expr.lhs is None else _remap_value_expr(expr.lhs, remap)
        if isinstance(expr.operand, NodeTypeRef):
            new_operand: 'Operand' = NodeTypeRef(position=remap(expr.operand.position))
        else:
            new_operand = _remap_value_expr(expr.operand, remap)
        return Condition(operator=expr.operator, operand=new_operand, lhs=new_lhs)

    if isinstance(expr, RangeCondition):
        return RangeCondition(
            expr=_remap_value_expr(expr.expr, remap),
            low=_remap_value_expr(expr.low, remap),
            high=_remap_value_expr(expr.high, remap),
        )

    if isinstance(expr, TimeCondition):
        if expr.ref is None:
            return expr
        return TimeCondition(
            kind=expr.kind,
            start=expr.start,
            end=expr.end,
            tz=expr.tz,
            days=expr.days,
            ref=NodeRef(
                position=remap(expr.ref.position),
                attribute=expr.ref.attribute,
                multiplier=expr.ref.multiplier,
            ),
            min_hours=expr.min_hours,
            max_hours=expr.max_hours,
        )

    if isinstance(expr, (ConditionAnd, ConditionOr)):
        return type(expr)(
            conditions=tuple(_remap_condition(c, remap) for c in expr.conditions)
        )

    # TagCondition → references no other node
    return expr


def _expand_quantifiers(
    parsed: 'list[tuple[PatternNode, int, int]]',
    tag: str,
) -> 'list[PatternNode]':
    """
    Expand ``{min,max}`` quantifiers into concrete PatternNode copies.

    Each logical node ``i`` becomes ``rep_min`` mandatory copies followed by
    ``rep_max - rep_min`` optional ones, and every ``$N`` reference is remapped
    to the resulting physical positions:

    * absolute ``$k``  → the FIRST copy of logical node ``k``. That copy is
      mandatory whenever ``rep_min >= 1``, so the reference always resolves to
      a pivot that is present. Referencing a group whose ``rep_min`` is 0 is
      rejected, because the target may be absent and every condition using it
      would silently evaluate to False.
    * relative ``$-p`` → recomputed so it still points at the same logical
      node. Rejected inside a quantified node, where each copy sits at a
      different distance and the intent would be ambiguous.
    """
    n_logical = len(parsed)

    # Physical start index of each logical node, and its repetition bounds.
    starts: list[int] = []
    rep_mins: list[int] = []
    cursor = 0
    for node, rep_min, rep_max in parsed:
        starts.append(cursor)
        rep_mins.append(rep_min)
        cursor += rep_max

    # Guard against the 2^K blow-up in Pattern.try_match (see _MAX_OPTIONAL_NODES).
    n_optional = sum(
        (rep_max - rep_min) + (1 if node.optional else 0)
        for node, rep_min, rep_max in parsed
    )
    if n_optional > _MAX_OPTIONAL_NODES:
        raise PatternSyntaxError(
            f"parse_pattern(tag={tag!r}): the pattern expands to {n_optional} "
            f"optional nodes, over the limit of {_MAX_OPTIONAL_NODES}.\n"
            f"Matching cost grows as 2^(optional nodes), so a wider range makes "
            f"the pattern store unusably slow. Narrow the {{min,max}} ranges."
        )

    expanded: list[PatternNode] = []
    for i, (node, rep_min, rep_max) in enumerate(parsed):
        is_quantified = rep_max > 1

        def remap(position: int, _i: int = i, _quant: bool = is_quantified) -> int:
            if position >= 0:
                if position >= n_logical:
                    raise PatternSyntaxError(
                        f"parse_pattern(tag={tag!r}): node {_i} references "
                        f"${position}, but the pattern only has {n_logical} "
                        f"nodes (valid indices 0..{n_logical - 1})."
                    )
                if rep_mins[position] == 0:
                    raise PatternSyntaxError(
                        f"parse_pattern(tag={tag!r}): node {_i} references "
                        f"${position}, a node quantified with min 0 that may be "
                        f"absent from the match. Reference a mandatory node, or "
                        f"raise that node's minimum to 1."
                    )
                return starts[position]

            if _quant:
                raise PatternSyntaxError(
                    f"parse_pattern(tag={tag!r}): node {_i} is quantified and "
                    f"uses the relative reference ${position}. Each copy sits at "
                    f"a different distance, so the target is ambiguous. Use an "
                    f"absolute reference ($0, $1, ...)."
                )
            target = _i + position
            if target < 0:
                raise PatternSyntaxError(
                    f"parse_pattern(tag={tag!r}): node {_i} uses the relative "
                    f"reference ${position}, which points before the start of "
                    f"the pattern."
                )
            if rep_mins[target] == 0:
                raise PatternSyntaxError(
                    f"parse_pattern(tag={tag!r}): node {_i} relatively references "
                    f"${position} (node {target}), quantified with min 0 and thus "
                    f"possibly absent. Reference a mandatory node instead."
                )
            # Keep it relative, but measured between physical positions.
            return starts[target] - starts[_i]

        if node.conditions:
            conditions = [_remap_condition(c, remap) for c in node.conditions]
        else:
            conditions = []

        for copy_idx in range(rep_max):
            expanded.append(
                PatternNode(
                    type=node.type,
                    tag_filters=dict(node.tag_filters),
                    conditions=list(conditions),
                    # Copies beyond rep_min are what makes the gap elastic.
                    optional=node.optional or copy_idx >= rep_min,
                    captured=node.captured,
                )
            )

    return expanded


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def parse_pattern(dsl: str, tag: str) -> Pattern:
    """
    Converts a DSL string into a Pattern ready for use in add_pattern_matcher().

    Parameters
    ──────────
    dsl : multi-line string. Each non-empty line is a pattern node.
          Lines starting with '#' are comments and are ignored.
          If the last line is a "$", enables anchor_end: the match is only
          valid if it ends at the last confirmed pivot in the sequence.

          Optional nodes: add ``?`` immediately after the type (no space)
          to mark the node as optional. The pattern matches with or without
          that pivot.
          Example: ``H?[nbs=neu]  > $0``

          Non-captured nodes: add ``~`` immediately before the type (no space)
          to mark the node as non-captured. Participates in matching but
          does not appear in ``MatchResult.nodes``.
          Compatible with optional nodes: ``~H?[nbs=neu]``.
          Example: ``~L?  > $0``

          Active window: a standalone ``@active_bars(end)`` or
          ``@active_bars(start, end)`` line bounds how long a completed match
          stays reported by the proxy's ``.last``. The age is counted in OHLC
          bars since the confirmation bar of the last matched pivot (age 0 ==
          confirmation bar). Outside ``start <= age <= end`` ``.last`` is None.
          Example: ``@active_bars(0, 10)``.

    tag : pattern name. Appears in MatchResult.tag when there is a match.

    Returns
    ───────
    Pattern — pass directly to add_pattern_matcher(pattern=...).

    Raises
    ──────
    PatternSyntaxError — if any line cannot be parsed.
    PatternSyntaxError — if there are fewer than 2 mandatory nodes (minimum required).

    Complete usage example
    ──────────────────────
        from tradetropy.ta.pattern.dsl import parse_pattern

        class MyStrategy(Strategy):

            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.cpivot = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    ConfirmedPivot(swing=3),
                )
                self.nbs = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    NBS(swing=3),
                )
                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                    decorators=[self.nbs],
                    pattern = parse_pattern(\"\"\"
                        # Base support — Booster in NY session
                        L[nbs=boo]  @between(09:30, 16:00, tz=America/New_York)
                        # Impulse — Neutralizer, above support, same schedule
                        H[nbs=neu]  > $0 & @between(09:30, 16:00, tz=America/New_York)
                        # Second support — Higher Low, confirmed within 48h
                        L[nbs=boo]  > $0 & @hours_since($1.timestamp, min=0, max=48)
                        # Second impulse — Higher High
                        H[nbs=neu]  > $1
                    \"\"\", tag="bullish_impulse_ny"),
                )

            def on_data(self):
                match = self.setup.last
                if match:
                    self.log.signal(
                        "NY Impulse: L=%.2f H=%.2f HL=%.2f HH=%.2f",
                        match[0].value, match[1].value,
                        match[2].value, match[3].value,
                    )

    Example with optional node
    ──────────────────────────
        self.setup = self.add_pattern_matcher(
            base_pivot=self.cpivot,
            decorators=[self.nbs],
            pattern = parse_pattern(\"\"\"
                # Base support
                L[nbs=boo]
                # Optional impulse — may or may not exist
                H?[nbs=neu]  > $0
                # Second support (always required)
                L[nbs=boo]   > $0
                # Higher High over the impulse if it existed
                H[nbs=neu]   > $1
            \"\"\", tag="double_bottom"),
        )

        def on_data(self):
            match = self.setup.last
            if match:
                # Was the optional impulse (node 1) present?
                if match.matched_optional and match.matched_optional.get(1):
                    impulse = match.node_map[1]
                    self.log.info("Impulse at %.2f", impulse.value)
    """
    # Filter lines: ignore empty and comments, then FOLD continuation lines.
    #
    # A physical line is a CONTINUATION of the previous logical line (glued
    # onto it, separated by a space) when its first non-space character(s)
    # unambiguously mark it as "more conditions for the node above" rather
    # than a new node or a new directive. Recognized continuation markers:
    #
    #   & |                              logical connectors
    #   > < >= <= == !=                  comparison operators (bare condition)
    #   (                                a parenthesized group
    #
    # This lets a node be written "bare" on its own line, with the whole
    # condition chain — including its very first condition — indented below:
    #
    #     ~L[nbs=boo]
    #         > $0
    #         & ($3-$2) >= ($1-$0)
    #
    # which is exactly equivalent to the single physical line:
    #
    #     ~L[nbs=boo] > $0 & ($3-$2) >= ($1-$0)
    #
    # '@' is intentionally NOT a bare continuation marker: a leading '@'
    # is ambiguous between an inline time condition on a new node
    # (rare/unused in practice) and the standalone '@active_bars(...)'
    # directive, so '@...' lines are only folded when written after an
    # explicit '&'/'|' (as already supported), never as a bare prefix.
    #
    # Node lines always start with '~', 'H', 'L' or 'any', and directives
    # with '$' or '@active_bars', so none of the markers above can ever be
    # confused with the start of a node or a directive.
    #
    # This runs once, at compile time (parse_pattern is called when the
    # strategy is built, not per bar), and is O(number of physical lines).
    # The tokenizer and recursive condition parser are untouched: after
    # folding, each element of ``lines`` is again "one logical line = one
    # node (or directive)", the invariant the rest of this function assumes.
    _CONTINUATION_RE = re.compile(r"^(&|\||>=|<=|==|!=|>|<|\()")

    lines: list[str] = []
    for raw in dsl.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if _CONTINUATION_RE.match(s):
            if not lines:
                raise PatternSyntaxError(
                    f"parse_pattern(tag={tag!r}): continuation line {s!r} "
                    f"starts with a continuation marker ('&', '|', a "
                    f"comparison operator or '(') but there is no previous "
                    f"node line to attach it to."
                )
            lines[-1] = f"{lines[-1]} {s}"
            continue
        lines.append(s)

    # Extract global (standalone) directives from node lines:
    #   · "$"                      -> anchor_end
    #   · "@active_bars(a, b)"     -> activation window in bars since the
    #                                 confirmation bar of the last pivot
    # Node lines never start with '@' or equal "$", so the separation is
    # unambiguous. Directives may appear in any trailing position.
    anchor_end   = False
    active_start = 0
    active_end: 'int | None' = None
    node_lines: list[str] = []

    for l in lines:
        if l == "$":
            anchor_end = True
            continue
        if l.startswith("@active_bars"):
            m = _ACTIVE_BARS_RE.match(l)
            if m is None:
                raise PatternSyntaxError(
                    f"parse_pattern(tag={tag!r}): invalid @active_bars directive "
                    f"{l!r}.\n"
                    f"Expected @active_bars(end) or @active_bars(start, end) "
                    f"with non-negative integers, e.g. @active_bars(10) or "
                    f"@active_bars(0, 10)."
                )
            first  = int(m.group(1))
            second = m.group(2)
            if second is None:
                active_start, active_end = 0, first
            else:
                active_start, active_end = first, int(second)
            continue
        node_lines.append(l)

    lines = node_lines

    if len(lines) < 2:
        raise PatternSyntaxError(
            f"parse_pattern(tag={tag!r}): pattern needs at least 2 nodes, "
            f"found {len(lines)}.\n"
            f"Received string:\n{dsl}"
        )

    parsed: list[tuple[PatternNode, int, int]] = []
    for i, line in enumerate(lines):
        try:
            parsed.append(_parse_node_line(line))
        except PatternSyntaxError as exc:
            raise PatternSyntaxError(
                f"parse_pattern(tag={tag!r}): error in node {i} "
                f"(line: {line!r})\n  → {exc}"
            ) from exc

    # Expand {min,max} quantifiers and remap $N to the shifted positions.
    nodes = _expand_quantifiers(parsed, tag)

    return Pattern(
        nodes=nodes,
        tag=tag,
        anchor_end=anchor_end,
        active_start=active_start,
        active_end=active_end,
    )
    

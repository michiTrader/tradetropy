# Raw tick columns
TICK_COLS   = ("ts", "bid", "ask", "volume", "flags", "volume_real", "price")
N_TICK_COLS = len(TICK_COLS)
_TICK_COL   = {name: i for i, name in enumerate(TICK_COLS)}

# OHLC candle columns
OHLC_COLS   = ("ts", "open", "high", "low", "close", "volume")
N_OHLC_COLS = len(OHLC_COLS)
_OHLC_COL   = {name: i for i, name in enumerate(OHLC_COLS)}

# OHLC candle columns + turnover (KlineData.data format, N x 7)
OHLCV_TURNOVER_COLS = ("ts", "open", "high", "low", "close", "volume", "turnover")
N_OHLCV_TURNOVER_COLS = len(OHLCV_TURNOVER_COLS)
_OHLCV_TURNOVER_COL = {name: i for i, name in enumerate(OHLCV_TURNOVER_COLS)}


# =============================================================================
# TIMEFRAMES
# =============================================================================

#: Timeframe unit multipliers -> milliseconds.
#:
#: 'mo' (month) is a FIXED 30-day duration, not a real calendar month (which
#: varies 28-31 days). This codebase always represents timeframes as a fixed
#: millisecond duration, so 'mo' is an approximation kept consistent with the
#: same convention used internally by the ccxt and MT5 connectors for the
#: monthly candle (see connectors/ccxt.py::_MS_TO_CCXT_TF and
#: connectors/mt5.py::_MS_TO_TF_KEY, both keyed on 2_592_000_000 ms).
#: 'min' and 'wk' are input-only aliases for 'm' (minute) and 'w' (week),
#: added so a user cannot confuse minute and month: 'mo' never collides with
#: 'm' regardless of case, unlike some venues' 'M' == month convention.
_TF_UNIT_MS = {
    "ms": 1,
    "s": 1_000,
    "min": 60_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "wk": 604_800_000,
    "w": 604_800_000,
    "mo": 2_592_000_000,
}

#: Suffix-detection order for parse_timeframe. Longer / more specific
#: suffixes must be checked before shorter ones they could otherwise be
#: mistaken for via a naive endswith scan (e.g. 'min' before 'm', 'ms'
#: before 's', 'wk' before 'w'). 'mo' has no collision risk with any
#: single-letter unit but is listed early for clarity.
_TF_UNIT_SUFFIXES = ("ms", "min", "mo", "wk", "s", "m", "h", "d", "w")


def parse_timeframe(tf) -> int:
    """
    Parse timeframe to milliseconds.

    Accepts int, float, or string format. Strings must be number + unit,
    e.g. '1m', '5m', '15m', '1h', '4h', '1d', '1w', '1mo', '30s', '500ms'.
    Unit is case-insensitive in general ('1H' == '1h', '1MO' == '1mo'),
    with ONE deliberate exception: a bare uppercase 'M' (minute) is
    rejected rather than silently treated as minute, because some venues
    (Binance, ccxt, MT5) use '1M' to mean month. Write minute lowercase
    ('1m') or use the explicit '1mo' for month; 'mo'/'MO'/'Mo' and
    'min'/'MIN' are unaffected (already unambiguous). 'min' is an alias
    for 'm' (minute) and 'wk' is an alias for 'w' (week), e.g. '15min'
    == '15m' and '1wk' == '1w'. 'mo' (month) is a fixed 30-day duration,
    not a real calendar month. Int/float are interpreted as already-resolved
    milliseconds (must be > 0).

    Note: only the bare 'm'/'M' unit is guarded; every other unit letter
    stays fully case-insensitive as before.

    Args:
        tf (int, float, or str): Timeframe value

    Returns:
        int: Duration in milliseconds

    Raises:
        ConfigError: If format is invalid, unit is unknown, value is not
            positive, or the timeframe uses the ambiguous bare 'M' unit
            (use '1m' for minute or '1mo' for month instead)

    Example:
        parse_timeframe('5m')    # 300_000
        parse_timeframe('1h')    # 3_600_000
        parse_timeframe('1mo')   # 2_592_000_000 (fixed 30 days)
        parse_timeframe('15min') # 900_000 (alias for '15m')
        parse_timeframe('1wk')   # 604_800_000 (alias for '1w')
        parse_timeframe(60_000)  # 60_000
        parse_timeframe('1M')    # raises ConfigError (ambiguous, use '1m' or '1mo')
    """
    from tradetropy.exceptions import ConfigError

    if isinstance(tf, bool):  # bool is a subclass of int -- reject explicitly
        raise ConfigError(f"parse_timeframe() does not accept bool, received {tf!r}.")

    if isinstance(tf, (int, float)):
        ms = int(tf)
        if ms <= 0:
            raise ConfigError(
                f"timeframe in ms must be a positive integer, received {tf!r}."
            )
        return ms

    if not isinstance(tf, str):
        raise ConfigError(
            f"timeframe must be str (e.g. '5m') or int in ms, "
            f"received {type(tf).__name__}."
        )

    s_stripped = tf.strip()
    if not s_stripped:
        raise ConfigError("empty timeframe.")

    # Reject a bare uppercase 'M' unit before case-folding. Some venues
    # (Binance, ccxt, MT5) use '1M' to mean month, which would otherwise
    # silently resolve to 1 minute here (this parser is case-insensitive
    # everywhere else). This is the one deliberate exception: 'M'/'m' as
    # the FINAL unit letter must be written lowercase; 'mo'/'MO'/'Mo' (month)
    # and 'min'/'MIN' (explicit minute alias) are unaffected since they are
    # not the ambiguous bare form.
    if s_stripped.endswith("M") and not s_stripped.lower().endswith(("mo", "min")):
        raise ConfigError(
            f"timeframe '{tf}' is ambiguous: bare uppercase 'M' is reserved "
            f"to avoid confusion with month (as used by some venues, e.g. "
            f"Binance/ccxt/MT5's '1M'). Use '1m' for 1 minute or '1mo' for "
            f"1 month (fixed 30 days)."
        )

    s = s_stripped.lower()

    # Split the unit suffix. Longer / more specific suffixes are checked
    # first (see _TF_UNIT_SUFFIXES) so 'min' is matched before 'm', 'ms'
    # before 's', and 'wk' before 'w'.
    unit = None
    for u in _TF_UNIT_SUFFIXES:
        if s.endswith(u):
            unit = u
            num_part = s[: -len(u)]
            break

    if unit is None:
        raise ConfigError(
            f"timeframe '{tf}' has no recognized unit. "
            f"Valid units: {', '.join(_TF_UNIT_SUFFIXES)}. "
            f"Example: '5m', '1h', '1d', '1mo', '15min', '1wk'."
        )

    if not num_part:
        num_part = "1"  # allows "h" == "1h"

    try:
        value = float(num_part)
    except ValueError:
        raise ConfigError(
            f"timeframe '{tf}' has an invalid quantity ('{num_part}')."
        )

    if value <= 0:
        raise ConfigError(f"timeframe '{tf}' must be positive.")

    ms = int(round(value * _TF_UNIT_MS[unit]))
    if ms <= 0:
        raise ConfigError(f"timeframe '{tf}' resolves to {ms} ms (not positive).")
    return ms


#: Units ordered largest-first for format_timeframe (inverse of _TF_UNIT_MS).
#: Only canonical output units are listed here: 'min'/'wk' are input-only
#: aliases (see _TF_UNIT_MS) and never produced as output.
_TF_UNIT_ORDER = (
    ("mo", 2_592_000_000),
    ("w", 604_800_000),
    ("d", 86_400_000),
    ("h", 3_600_000),
    ("m", 60_000),
    ("s", 1_000),
    ("ms", 1),
)


def format_timeframe(ms) -> str:
    """
    Format a duration in milliseconds as a readable timeframe string.

    Inverse of parse_timeframe: picks the largest unit that divides the
    duration exactly, so 86_400_000 -> '1d', 3_600_000 -> '1h',
    900_000 -> '15m', 2_592_000_000 -> '1mo'. Durations that are not an
    exact multiple of any unit fall back to the largest unit that fits
    plus the remainder in ms (e.g. 90_000 -> '1m30s'). The canonical output
    for minute and week is always 'm' / 'w' ('min' and 'wk' are accepted on
    input but never produced by this function).

    Args:
        ms (int): Duration in milliseconds (must be positive).

    Returns:
        str: Readable timeframe label, e.g. '1d', '1h', '15m', '1mo', '30s'.

    Example:
        format_timeframe(86_400_000)    # '1d'
        format_timeframe(3_600_000)     # '1h'
        format_timeframe(900_000)       # '15m'
        format_timeframe(2_592_000_000) # '1mo'
    """
    ms = int(ms)
    if ms <= 0:
        return f'{ms}ms'

    # Exact single-unit match: the common case for standard timeframes.
    for unit, size in _TF_UNIT_ORDER:
        if ms % size == 0:
            return f'{ms // size}{unit}'

    # Non-exact duration: largest unit that fits, plus remainder in ms.
    for unit, size in _TF_UNIT_ORDER[:-1]:
        if ms >= size:
            whole, rem = divmod(ms, size)
            return f'{whole}{unit}{rem}ms' if rem else f'{whole}{unit}'
    return f'{ms}ms'


# =============================================================================
# TIMEFRAME PRESETS
# =============================================================================

#: Readable timeframe presets -> milliseconds. Covers standard intervals from
#: most exchanges (compatible with parse_timeframe).
TIMEFRAME_PRESETS = {
    "1s":  1_000,
    "5s":  5_000,
    "15s": 15_000,
    "30s": 30_000,
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
    "6h":  21_600_000,
    "8h":  28_800_000,
    "12h": 43_200_000,
    "1d":  86_400_000,
    "3d":  259_200_000,
    "1w":  604_800_000,
    "1mo": 2_592_000_000,
}

#: ms -> Binance interval label (subset supported by the REST API).
_MS_TO_BINANCE = {
    60_000: "1m",
    180_000: "3m",
    300_000: "5m",
    900_000: "15m",
    1_800_000: "30m",
    3_600_000: "1h",
    7_200_000: "2h",
    14_400_000: "4h",
    21_600_000: "6h",
    28_800_000: "8h",
    43_200_000: "12h",
    86_400_000: "1d",
    259_200_000: "3d",
    604_800_000: "1w",
}


def to_binance_interval(tf) -> str:
    """
    Convert timeframe to Binance REST API interval format.

    Args:
        tf (str or int): Timeframe as string (e.g. '5m') or milliseconds

    Returns:
        str: Binance interval format (e.g. '1m', '5m', '1h', '1d')

    Raises:
        ConfigError: If the interval is not supported by Binance

    Example:
        to_binance_interval('5m')   # '5m'
        to_binance_interval(300_000) # '5m'
    """
    from tradetropy.exceptions import ConfigError

    ms = parse_timeframe(tf)
    label = _MS_TO_BINANCE.get(ms)
    if label is None:
        supported = ", ".join(_MS_TO_BINANCE.values())
        raise ConfigError(
            f"Interval {tf!r} ({ms} ms) not supported by Binance. "
            f"Supported: {supported}."
        )
    return label

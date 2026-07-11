"""
logging.py  v1.0
========================
Strategy logger for the tradetropy trading system.

Extends the base pintar (logging.py) system with:
  - 3 additional trading-specific levels:
        PERF    (5)  - performance metrics, tick-by-tick, very verbose
        SIGNAL  (15) - entry/exit signals detected by the strategy
        TRADING (25) - executed orders, fills, position closes

  - Default trading-specific color palette for the 3 new levels.
  - get_strategy_logger() - creates a logger ready for Strategy with console
    and file handler, a per-strategy name color, and custom levels.
  - Optional integration in Strategy via lazy property `log`.

LEVELS (ascending order)
------------------------
    PERF     =  5   (below DEBUG)
    DEBUG    = 10
    SIGNAL   = 15   (between DEBUG and INFO)
    INFO     = 20
    TRADING  = 25   (between INFO and WARNING)
    WARNING  = 30
    ERROR    = 40
    CRITICAL = 50

QUICK USAGE IN STRATEGY
-----------------------
    # Option A - lazy property (zero-config, name = class)
    class MyStrategy(Strategy):
        def on_data(self):
            self.log.signal("bullish cross detected -> price=%.2f", price)
            self.log.trading("BUY BTCUSDT 0.01 @ %.2f", fill_price)
            self.log.perf("tick processed in %.3f ms", elapsed)

    # Option B - explicit logger with custom name color
    from logging import get_strategy_logger
    log = get_strategy_logger(
        name       = "MyStrategy",
        name_color = "#BFD8FF",          # color of the {name} field
        log_file   = "logs/bt.log",      # None for console only
        level      = SIGNAL,             # filter by minimum level
    )
    log.signal("signal detected")
    log.trading("order sent")

EXPORTED LEVELS
---------------
    from logging import PERF, SIGNAL, TRADING

INTEGRATION WITH STRATEGY (snippet for strategy.py)
----------------------------------------------------
    from logging import get_strategy_logger, PERF

    class Strategy:
        _logger: logging.Logger | None = None

        @property
        def log(self) -> logging.Logger:
            if self._logger is None:
                self._logger = get_strategy_logger(
                    name       = self.__class__.__name__,
                    name_color = getattr(self, "log_color", "#BFD8FF"),
                    log_file   = getattr(self, "log_file",  None),
                    level      = getattr(self, "log_level", PERF),
                )
            return self._logger

        # The strategy can override these attributes in __init__:
        #   self.log_color = "#FF6B35"
        #   self.log_file  = "logs/my_bt.log"
        #   self.log_level = SIGNAL

ARCHITECTURE
------------
    Section 1 - Custom levels (registration in logging + methods in Logger)
    Section 2 - Extended default palette with the 3 new levels
    Section 3 - get_strategy_logger() - main API
    Section 4 - Demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import re

# Import the base pintar system
from pintar.logging import (
    Theme,
    FieldDef,
    PintarStreamHandler,
    PintarFileHandler,
    PintarFormatter,
    _ColorSpec,
    _ColorInput,
    _DEFAULT_FMT,
    _DEFAULT_DATEFMT,
    _MAX_LEVEL_LEN,
    _DEFAULT_PALETTE,
)
from pintar.colors import RGB, HEX, HSL

import time as _time


_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

# ==========================================================================
# Custom levels
# ==========================================================================

#: PERF level - tick-by-tick, performance metrics. Very verbose.
PERF: int = 5

#: SIGNAL level - trading signals detected by the strategy.
SIGNAL: int = 15

#: TRADING level - executed orders, fills, closes.
TRADING: int = 25


def _register_levels() -> None:
    """
    Register custom logging levels and add methods to Logger.

    Registers PERF, SIGNAL, and TRADING levels in the logging module,
    and adds perf(), signal(), and trading() methods to logging.Logger.

    This function is idempotent - calling multiple times has no side effects.
    """
    logging.addLevelName(PERF,    "PERF")
    logging.addLevelName(SIGNAL,  "SIGNAL")
    logging.addLevelName(TRADING, "TRADING")

    if not hasattr(logging.Logger, "perf"):
        def perf(self, msg, *args, **kwargs):
            if self.isEnabledFor(PERF):
                self._log(PERF, msg, args, **kwargs)
        logging.Logger.perf = perf

    if not hasattr(logging.Logger, "signal"):
        def signal(self, msg, *args, **kwargs):
            if self.isEnabledFor(SIGNAL):
                self._log(SIGNAL, msg, args, **kwargs)
        logging.Logger.signal = signal

    if not hasattr(logging.Logger, "trading"):
        def trading(self, msg, *args, **kwargs):
            if self.isEnabledFor(TRADING):
                self._log(TRADING, msg, args, **kwargs)
        logging.Logger.trading = trading


# Register on module import
_register_levels()

# After registration, _MAX_LEVEL_LEN may be too short.
# Recalculate so ljust() pads correctly with the new names.
_TRADING_MAX_LEN: int = max(len(k) for k in logging._nameToLevel)


# ==========================================================================
# Trading palette
# ==========================================================================
#
# Extends _DEFAULT_PALETTE with the 3 new levels.
# The `name` field receives the strategy color in get_strategy_logger().
#
# Color scheme:
#   PERF    - muted gray, "dim" - does not distract from normal flow
#   SIGNAL  - ice blue - signal detected, not yet an order
#   TRADING - bright white + bold - real action on the market
#

_TRADING_PALETTE_EXTRA: dict[str, dict[str, _ColorSpec]] = {
    "PERF": {
        "asctime":   ("#2D3748", None, None),
        "bar":       ("#2D3748", None, None),
        "levelname": ("#4A5568", None, "dim"),
        "name":      ("#4A5568", None, "dim"),
        "message":   ("#4A5568", None, "dim"),
    },
    "SIGNAL": {
        "asctime":   ("#4A5568", None, None),
        "bar":       ("#4A5568", None, None),
        "levelname": ("#63B3ED", None, "bold"),
        "name":      ("#63B3ED", None, None),
        "message":   ("#BEE3F8", None, None),
    },
    "TRADING": {
        "asctime":   ("#718096", None, None),
        "bar":       ("#718096", None, None),
        "levelname": ("#EAEFFF", None, "bold"),
        "name":      ("#EAEFFF", None, "bold"),
        "message":   ("#D1D1D1", None, None),
    },
}


def _build_trading_palette(name_color: _ColorInput | None = None) -> dict[str, dict[str, _ColorSpec]]:
    """
    Build the complete logging palette with trading levels.

    Merges the base palette with trading-specific levels (PERF, SIGNAL,
    TRADING). If name_color is provided, overrides the 'name' field color
    in all levels to allow unique strategy colors.

    Args:
        name_color: Optional color for the 'name' field. Accepts hex string,
                    RGB, HEX, HSL, tuple, ANSI-256 int, or None.

    Returns:
        dict: Complete palette mapping level names to field color specs.
    """
    import copy
    palette: dict[str, dict[str, _ColorSpec]] = copy.deepcopy(_DEFAULT_PALETTE)

    # Add the 3 new levels
    for level_name, fields in _TRADING_PALETTE_EXTRA.items():
        palette[level_name] = dict(fields)

    # Override `name` color in all levels if specified
    if name_color is not None:
        for level_fields in palette.values():
            if "name" in level_fields:
                fore, bg, style = level_fields["name"]
                level_fields["name"] = (name_color, bg, style)

    return palette


# ==========================================================================
# Formatter and main API
# ==========================================================================


class TradingFormatter(PintarFormatter):
    """
    Formatter extending PintarFormatter with trading level support.

    Supports PERF, SIGNAL, and TRADING levels. Recalculates the max level
    name length to ensure proper padding with custom level names.

    Built from a Theme that already contains the trading palette - no
    additional logic is needed.
    """

    def format(self, record: logging.LogRecord) -> str:
        if not self._theme.dye:
            record = logging.makeLogRecord(record.__dict__)  # copy - do not mutate original
            record.msg = _ANSI_RE.sub('', str(record.msg))

        # Ensure the record has the bar field
        if not hasattr(record, "bar"):
            record.bar = "│"

        # Populate custom fields from FieldDef
        for field_name, fdef in self._theme.fields.items():
            setattr(record, field_name, fdef.resolve_value(record))

        # Use the extended max len with custom levels
        record.levelname = record.levelname.ljust(_TRADING_MAX_LEN)
        record.asctime   = self.formatTime(record, self.datefmt)
        record.message   = record.getMessage()

        fmt = self._level_fmts.get(record.levelno, self._level_fmts.get(logging.INFO, self._theme.fmt))
        result = fmt.format_map(record.__dict__)

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result = f"{result}\n{record.exc_text}"
        if record.stack_info:
            result = f"{result}\n{self.formatStack(record.stack_info)}"

        return result


def _make_trading_theme(
    name_color: _ColorInput | None,
    fmt: str,
    datefmt: str,
    dye: bool,
    extra_overrides: dict | None,
    fields: dict | None,
) -> Theme:
    """
    Build a Theme with trading palette incorporated in overrides.

    Constructs the trading palette and merges user-provided overrides on top.

    Args:
        name_color: Optional color for the 'name' field across all levels.
        fmt (str): Message format string.
        datefmt (str): Date format for timestamps.
        dye (bool): Whether to apply colors (True) or plain text (False).
        extra_overrides (dict): User-provided palette overrides.
        fields (dict): Custom field definitions.

    Returns:
        Theme: Configured theme with trading levels and user overrides.
    """
    palette = _build_trading_palette(name_color)

    # Merge user's extra overrides on top
    if extra_overrides:
        for level, field_map in extra_overrides.items():
            if level not in palette:
                palette[level] = {}
            palette[level].update(field_map)

    return Theme(
        overrides=palette,
        fmt=fmt,
        datefmt=datefmt,
        dye=dye,
        fields=fields or {},
    )


def get_strategy_logger(
    name: str,
    name_color: _ColorInput | None = "#BFD8FF",
    log_file: str | Path | None = None,
    level: int = PERF,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    overrides: dict | None = None,
    fields: dict | None = None,
    stream=None,
) -> logging.Logger:
    """
    Create a complete logger for a trading strategy.

    Automatically registers PERF, SIGNAL, and TRADING levels if not already
    registered, and adds the corresponding methods to Logger.

    Args:
        name (str): Logger name. Recommended: Strategy class name.
        name_color: Color of the {name} field. Accepts hex string, RGB, HEX,
                    HSL, tuple, ANSI-256 int, or None. Default "#BFD8FF"
                    (ice blue). Useful to distinguish different strategies.
        log_file: Path to log file. If None, console only. Directory is
                  created automatically if needed.
        level (int): Minimum logging level. Default PERF (captures all).
                     Use SIGNAL to filter performance metrics, TRADING to
                     see only executed orders.
        fmt (str): Message format. Default shows asctime, levelname, name,
                   and message.
        datefmt (str): Date format for {asctime} field.
        overrides (dict): Additional palette overrides. Uses same schema as
                          Theme.overrides.
        fields (dict): Extra custom field definitions. Uses same schema as
                       Theme.fields.
        stream: Console stream (default sys.stdout).

    Returns:
        logging.Logger: Logger with methods perf(), signal(), trading().

    Example:
        log = get_strategy_logger('MyStrategy', name_color='#FF6B35')
        log.perf('tick: ts=%d price=%.2f', ts, price)
        log.signal('SMA cross -> LONG price=%.2f', price)
        log.trading('BUY BTCUSDT 0.01 @ %.2f fill=%.2f', price, fill)
    """
    # Avoid duplicating handlers if the logger already exists
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Theme with trading palette
    theme_color = _make_trading_theme(
        name_color=name_color,
        fmt=fmt,
        datefmt=datefmt,
        dye=True,
        extra_overrides=overrides,
        fields=fields,
    )
    theme_plain = _make_trading_theme(
        name_color=None,
        fmt=fmt,
        datefmt=datefmt,
        dye=False,
        extra_overrides=overrides,
        fields=fields,
    )

    # Console handler
    sh = logging.StreamHandler(stream or sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(TradingFormatter(theme_color))
    logger.addHandler(sh)

    # File handler (optional, no color)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(TradingFormatter(theme_plain))
        logger.addHandler(fh)

    return logger


# Adapter for backtest (data timestamp, not system clock)

class _BacktestRecord(logging.LogRecord):
    """
    LogRecord that uses backtest data timestamp instead of system clock.

    Allows logging to show the timestamp from backtesting data rather than
    wall-clock time, useful for synchronizing logs with market events.
    """
    _bt_ts_ms: int = 0

    def getMessage(self):
        return super().getMessage()


class BacktestLoggerWrapper(logging.Logger):
    """
    Logger wrapper that uses backtest data timestamp.

    Replaces record.created and record.msecs with the backtest data
    timestamp. Useful for log synchronization during backtesting.

    Internal use - created via wrap_backtest_logger().
    """
    def __init__(self, logger: logging.Logger, ts_provider):
        # Call Logger's __init__ so it creates _cache, manager, etc.
        super().__init__(logger.name, logger.level)
        # Copy state from the original logger
        self.handlers  = list(logger.handlers)
        self.propagate = logger.propagate
        self.disabled  = logger.disabled
        self.parent    = logger.parent
        self._ts_provider = ts_provider
        self._wrapped     = logger

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        record = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )
        try:
            ts_ms = self._ts_provider()
            if ts_ms and ts_ms > 0:
                record.created = ts_ms / 1000.0
                record.msecs   = ts_ms % 1000
        except Exception:
            pass  # If it fails, use the real timestamp - never break the log
        return record


def wrap_backtest_logger(
    logger: logging.Logger,
    ts_provider,
) -> "BacktestLoggerWrapper":
    """
    Wrap a logger to use backtest data timestamp instead of system clock.

    Allows log timestamps to reflect the backtest data timeline rather
    than wall-clock time, enabling proper log synchronization with market
    events during backtesting.

    Args:
        logger (logging.Logger): The logger to wrap.
        ts_provider: Callable () -> int that returns the timestamp in ms.
                     Example: lambda: engine._sesh._ultimo_ts

    Returns:
        BacktestLoggerWrapper: Wrapper with same configuration as original
                               logger but with backtest timestamps.
    """
    wrapper = BacktestLoggerWrapper.__new__(BacktestLoggerWrapper)
    BacktestLoggerWrapper.__init__(wrapper, logger, ts_provider)
    return wrapper

# ==========================================================================
# DEMO
# ==========================================================================

if __name__ == "__main__":
    from pintar.colors import HEX as PintarHEX

    print("--- Strategy logger - default color ---")
    log = get_strategy_logger("BtcStrategy", log_file="logs/btc_strategy.log")
    log.perf("tick processed: ts=1_700_000_000  price=44230.5")
    log.debug("SMA(20)=44100.0  EMA(9)=44250.0")
    log.signal("bullish SMA/EMA cross -> LONG  price=44230.5")
    log.info("backtest started - period 2024-01-01..2024-12-31")
    log.trading("BUY BTCUSDT 0.01 @ 44230.5  fill=44232.0  commission=0.044")
    log.warning("drawdown > 5%%  equity=9450.0")
    log.error("order rejected: insufficient funds")
    log.critical("broker disconnected - loop stopped")

    print("\n--- Custom name color per strategy ---")
    log_eth = get_strategy_logger("EthStrategy", name_color="#A78BFA")
    log_eth.signal("SHORT signal detected -> price=2450.0")
    log_eth.trading("SELL ETHUSDT 0.1 @ 2450.0  fill=2449.5")

    log_mes = get_strategy_logger("MesStrategy", name_color="#F6AD55")
    log_mes.signal("RSI divergence on 1h -> LONG")
    log_mes.trading("BUY MES 2 @ 4820.25  fill=4820.50")

    print("\n--- Minimum level TRADING (filters PERF, DEBUG, SIGNAL, INFO) ---")
    log_prod = get_strategy_logger(
        "ProdStrategy",
        name_color="#68D391",
        level=TRADING,
    )
    log_prod.perf("this does NOT appear")
    log_prod.signal("neither does this")
    log_prod.trading("BUY BTCUSDT 0.01 @ 44000.0  <- this DOES appear")
    log_prod.warning("high drawdown  <- this too")

    print("\n--- With log file ---")
    log_file = get_strategy_logger(
        "FileStrategy",
        name_color="#FC8181",
        log_file="/tmp/tradetropy_demo.log",
        level=SIGNAL,
    )
    log_file.signal("signal saved to console and file")
    log_file.trading("order saved to console and file")
    print("  -> file written to /tmp/tradetropy_demo.log")

    print("\n--- Palette overrides ---")
    log_custom = get_strategy_logger(
        "CustomStrategy",
        name_color="#B794F4",
        overrides={
            "TRADING": {
                "levelname": ("#000000", "#68D391", "bold"),
                "message":   ("#FFFFFF", None, "bold"),
            },
        },
    )
    log_custom.trading("order with green background on levelname")
    log_custom.signal("normal signal")

    print("\n--- With FieldDef (custom field) ---")
    log_field = get_strategy_logger(
        "FieldStrategy",
        name_color="#63B3ED",
        fmt="{asctime} {bar} {levelname} {arrow} {name} - {message}",
        fields={
            "arrow": FieldDef(
                value="->",
                palette={
                    "DEFAULT":  ("#4A5568", None, None),
                    "PERF":     ("#2D3748", None, "dim"),
                    "SIGNAL":   ("#63B3ED", None, "bold"),
                    "TRADING":  ("#EAEFFF", None, "bold"),
                    "INFO":     ("#0ECB81", None, None),
                    "WARNING":  ("#F59E0B", None, None),
                    "ERROR":    ("#F6465D", None, "bold"),
                    "CRITICAL": ("#FFFFFF", "#9B2335", "bold"),
                }
            ),
        }
    )
    log_field.signal("signal with custom arrow")
    log_field.trading("order with custom arrow")

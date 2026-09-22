"""Tests for the strategy logger's configurable display_tz."""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from tradetropy.logger import TradingFormatter, get_strategy_logger


class TestDisplayTz:
    def test_default_display_tz_is_utc(self):
        fmt = TradingFormatter()
        record = logging.LogRecord(
            "x", logging.INFO, "", 0, "msg", None, None,
        )
        # A known epoch instant: 2024-01-01T00:00:00 UTC.
        record.created = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        assert fmt.formatTime(record) == "2024-01-01 00:00:00"

    def test_custom_display_tz_shifts_the_rendered_time(self):
        fmt = TradingFormatter(display_tz=ZoneInfo("America/New_York"))
        record = logging.LogRecord(
            "x", logging.INFO, "", 0, "msg", None, None,
        )
        # 2024-01-01T00:00:00 UTC -> 2023-12-31T19:00:00 in New York (UTC-5).
        record.created = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        assert fmt.formatTime(record) == "2023-12-31 19:00:00"

    def test_display_tz_does_not_change_the_logged_instant(self):
        # Same epoch instant rendered in two zones differs only in presentation,
        # not in the underlying record.created value.
        record = logging.LogRecord(
            "x", logging.INFO, "", 0, "msg", None, None,
        )
        record.created = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()

        utc_fmt = TradingFormatter(display_tz=timezone.utc)
        tokyo_fmt = TradingFormatter(display_tz=ZoneInfo("Asia/Tokyo"))
        assert utc_fmt.formatTime(record) != tokyo_fmt.formatTime(record)
        # Tokyo is UTC+9, so its rendered hour is 9 ahead of the UTC rendering.
        utc_dt = datetime.strptime(utc_fmt.formatTime(record), "%Y-%m-%d %H:%M:%S")
        tokyo_dt = datetime.strptime(tokyo_fmt.formatTime(record), "%Y-%m-%d %H:%M:%S")
        assert (tokyo_dt - utc_dt).total_seconds() == 9 * 3600

    def test_get_strategy_logger_wires_display_tz_into_handlers(self):
        logger = get_strategy_logger(
            "TestDisplayTzLogger", display_tz=ZoneInfo("America/New_York"),
        )
        formatters = [h.formatter for h in logger.handlers
                     if isinstance(h.formatter, TradingFormatter)]
        assert formatters
        for f in formatters:
            assert f._display_tz == ZoneInfo("America/New_York")

    def test_get_strategy_logger_updates_display_tz_on_cached_logger(self):
        # Calling again with a different display_tz for the SAME name updates
        # the existing handlers instead of duplicating them.
        name = "TestDisplayTzLoggerCached"
        get_strategy_logger(name, display_tz=timezone.utc)
        logger = get_strategy_logger(name, display_tz=ZoneInfo("Asia/Tokyo"))

        n_handlers = len(logger.handlers)
        assert n_handlers > 0
        formatters = [h.formatter for h in logger.handlers
                     if isinstance(h.formatter, TradingFormatter)]
        for f in formatters:
            assert f._display_tz == ZoneInfo("Asia/Tokyo")

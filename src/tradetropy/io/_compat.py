"""
Backward-compatible aliases for the pre-rename IO functions.

New code should use save_ticks/save_klines/read_ticks/read_klines/save_book/
read_book directly; these thin wrappers exist only so older call sites keep
working.
"""

from __future__ import annotations

from tradetropy.io._ticks import save_ticks, read_ticks
from tradetropy.io._klines import save_klines, read_klines
from tradetropy.io._book import save_book, read_book


def ticks_to_file(data, path, **kwargs):
    """Compatibility alias for save_ticks(). Use save_ticks() for new code."""
    return save_ticks(data, path, **kwargs)


def klines_to_file(data, path, **kwargs):
    """Compatibility alias for save_klines(). Use save_klines() for new code."""
    return save_klines(data, path, **kwargs)


def ticks_from_file(path, symbol, **kwargs):
    """Compatibility alias for read_ticks(). Use read_ticks() for new code."""
    return read_ticks(path, symbol, **kwargs)


def klines_from_file(path, symbol, timeframe=None, **kwargs):
    """Compatibility alias for read_klines(). Use read_klines() for new code."""
    return read_klines(path, symbol, timeframe, **kwargs)


def read_klines_csv(path, symbol, timeframe=None, **kwargs):
    """Compatibility alias for read_klines(format='csv'). Use read_klines() for new code."""
    return read_klines(path, symbol, timeframe, format="csv", **kwargs)


def read_ticks_csv(path, symbol, **kwargs):
    """Compatibility alias for read_ticks(format='csv'). Use read_ticks() for new code."""
    return read_ticks(path, symbol, format="csv", **kwargs)


def book_to_file(data, path, **kwargs):
    """Compatibility alias for save_book(). Use save_book() for new code."""
    return save_book(data, path, **kwargs)


def book_from_file(path, symbol=None, **kwargs):
    """Compatibility alias for read_book(). Use read_book() for new code."""
    return read_book(path, symbol, **kwargs)


def save_proxy(proxy, path, format="parquet", **kwargs):
    """
    Save TickProxy or OhlcProxy to file.

    Automatically dispatches to save_ticks() or save_klines() based on proxy type.

    Args:
        proxy: TickProxy or OhlcProxy instance.
        path: Destination file path.
        format: Output format ('csv', 'parquet', 'hdf5').
        **kwargs: Additional arguments passed to save_ticks() or save_klines().

    Raises:
        TradingError: If proxy type is not TickProxy or OhlcProxy.
    """

"""
Bundled sample datasets for Tradetropy.

This package ships a small, curated collection of market data so the examples,
the user guide and the test suite all run out of the box - no downloads, no API
keys. Every loader returns a first-class Tradetropy domain object (``KlineData``,
``TickData`` or ``BookData``) ready to feed an engine.

The datasets are intentionally small and independent samples from different
instruments and periods:

======================  =========  ========  =======  =====================
Loader                  Type       Symbol    Rows     Notes
======================  =========  ========  =======  =====================
``load_btcusd_1m()``    KlineData  BTCUSDT   500      1-minute crypto (npz)
``load_adausd_1m()``    KlineData  ADAUSDT   500      1-minute crypto (npz)
``load_aapl_1d()``      KlineData  AAPL      300      daily stock (csv)
``load_goog_1d()``      KlineData  GOOG      300      daily stock (csv)
``load_mesu26_ticks()`` TickData   MESU26    2000     futures ticks (npz)
``load_mnqu26_ticks()`` TickData   MNQU26    2000     futures ticks (npz)
``load_adausd_ticks()`` TickData   ADAUSDT   43629    ticks paired with book
``load_adausd_book()``  BookData   ADAUSDT   240      L2 depth, 6 levels
======================  =========  ========  =======  =====================

Multi-timeframe
    Load a 1-minute series and call ``.resample()`` to build any higher
    timeframe from a single dataset::

        from tradetropy.datasets import load_btcusd_1m

        btc_1m = load_btcusd_1m()
        btc_1h = btc_1m.resample('1h')
        btc_1d = btc_1m.resample('1d')

Multi-symbol
    Every loader is independent, so combine them freely::

        from tradetropy.datasets import load_btcusd_1m, load_adausd_1m

        data = (load_btcusd_1m(), load_adausd_1m())

L2 order book (paired)
    ``load_adausd_ticks()`` and ``load_adausd_book()`` share the same timestamp
    range, so they replay together for the order-flow L2 indicators
    (``DeepTrades``, ``DeepWall``, ...). The order book is fed to the engine
    through ``ReplayEngine`` (a plain backtest has no book)::

        from tradetropy.datasets import load_adausd_ticks, load_adausd_book
        from tradetropy.replay import ReplayEngine

        ticks = load_adausd_ticks()
        book  = load_adausd_book()
        engine = ReplayEngine.by_ticks(
            MyStrategy(), data=(ticks,), book=book,
        )
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Iterator, List

from tradetropy.io import read_book, read_klines, read_ticks

if TYPE_CHECKING:
    from pathlib import Path

    from tradetropy.core.data_types import BookData, KlineData, TickData

__all__ = [
    "load_btcusd_1m",
    "load_adausd_1m",
    "load_aapl_1d",
    "load_goog_1d",
    "load_mesu26_ticks",
    "load_mnqu26_ticks",
    "load_adausd_ticks",
    "load_adausd_book",
    "list_datasets",
    "dataset_path",
    "DATASETS",
]

# Registry of the bundled files with the metadata each loader needs. Keeping it
# declarative lets list_datasets() describe the collection without loading it.
DATASETS = {
    "btcusd_1m": {
        "file": "BTCUSDT_1m.npz", "kind": "klines", "symbol": "BTCUSDT",
        "timeframe": "1m", "rows": 500,
        "tick_size": 0.1, "digits": 4,
        "description": "500 one-minute BTCUSDT candles (crypto, npz).",
    },
    "adausd_1m": {
        "file": "ADAUSDT_1m.npz", "kind": "klines", "symbol": "ADAUSDT",
        "timeframe": "1m", "rows": 500,
        "tick_size": 0.0001, "digits": 4,
        "description": "500 one-minute ADAUSDT candles (crypto, npz).",
    },
    "aapl_1d": {
        "file": "AAPL_1d.csv", "kind": "klines", "symbol": "AAPL",
        "timeframe": "1d", "rows": 300,
        "tick_size": 0.01, "digits": 2,
        "description": "300 daily AAPL candles (stock, csv).",
    },
    "goog_1d": {
        "file": "GOOG_1d.csv", "kind": "klines", "symbol": "GOOG",
        "timeframe": "1d", "rows": 300,
        "tick_size": 0.01, "digits": 2,
        "description": "300 daily GOOG candles (stock, csv).",
    },
    "mesu26_ticks": {
        "file": "MESU26_ticks.npz", "kind": "ticks", "symbol": "MESU26",
        "rows": 2000, "tick_size": 0.25, "tick_value": 1.25, "digits": 2,
        "description": "2000 Micro E-mini S&P 500 futures ticks (npz).",
    },
    "mnqu26_ticks": {
        "file": "MNQU26_ticks.npz", "kind": "ticks", "symbol": "MNQU26",
        "rows": 2000, "tick_size": 0.25, "tick_value": 0.5, "digits": 2,
        "description": "2000 Micro E-mini Nasdaq-100 futures ticks (npz).",
    },
    "adausd_ticks": {
        "file": "ADAUSDT_ticks.npz", "kind": "ticks", "symbol": "ADAUSDT",
        "rows": 43629, "tick_size": 0.0001, "digits": 4,
        "description": "43629 ADAUSDT trades (times and sales) paired in time "
                       "with adausd_book (for the L2 / DeepTrades example).",
    },
    "adausd_book": {
        "file": "ADAUSDT_book.npz", "kind": "book", "symbol": "ADAUSDT",
        "rows": 240, "levels": 6, "tick_size": 0.0001,
        "description": "240 rows of ADAUSDT L2 order-book depth, 6 levels "
                       "(paired with adausd_ticks).",
    },
}


@contextmanager
def _resolved(file_name: str) -> "Iterator[Path]":
    """
    Yield a real filesystem path to a bundled data file.

    Uses importlib.resources so it works whether the package is installed as a
    plain directory or extracted from a wheel/zip.

    Args:
        file_name (str): File name inside the datasets/_data folder.

    Yields:
        Path: A concrete path valid for the duration of the context.
    """
    resource = files(__package__).joinpath("_data", file_name)
    with as_file(resource) as path:
        yield path


def dataset_path(name: str):
    """
    Return a context manager yielding the on-disk path of a bundled dataset.

    Handy when you want to read a file directly (e.g. with pandas) instead of
    through the typed loaders.

    Args:
        name (str): Dataset key (see ``DATASETS`` / ``list_datasets()``).

    Returns:
        contextmanager: Yields a ``pathlib.Path`` to the data file.

    Raises:
        KeyError: If the dataset name is unknown.

    Example:
        with dataset_path('btcusd_1m') as p:
            print(p, p.stat().st_size)
    """
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}"
        )
    return _resolved(DATASETS[name]["file"])


def list_datasets() -> "List[dict]":
    """
    List the bundled datasets and their metadata.

    Returns:
        list[dict]: One entry per dataset with keys ``name``, ``kind``,
        ``symbol``, ``rows``, ``timeframe`` (klines only) and ``description``.

    Example:
        for d in list_datasets():
            print(d['name'], d['kind'], d['symbol'], d['rows'])
    """
    out: List[dict] = []
    for name, meta in DATASETS.items():
        entry = {
            "name": name,
            "kind": meta["kind"],
            "symbol": meta["symbol"],
            "rows": meta["rows"],
            "description": meta["description"],
        }
        if "timeframe" in meta:
            entry["timeframe"] = meta["timeframe"]
        out.append(entry)
    return out


# -----
# Kline loaders
# -----

def _load_klines(name: str) -> "KlineData":
    meta = DATASETS[name]
    with _resolved(meta["file"]) as path:
        return read_klines(
            str(path), meta["symbol"], meta["timeframe"],
            tick_size=meta["tick_size"], digits=meta["digits"],
        )


def load_btcusd_1m() -> "KlineData":
    """
    Load 500 one-minute BTCUSDT candles (crypto).

    Returns:
        KlineData: 1-minute OHLCV+turnover candles for BTCUSDT.

    Example:
        from tradetropy import BacktestEngine
        from tradetropy.datasets import load_btcusd_1m

        btc = load_btcusd_1m()
        engine = BacktestEngine.by_klines(MyStrategy(), data=(btc,))
    """
    return _load_klines("btcusd_1m")


def load_adausd_1m() -> "KlineData":
    """
    Load 500 one-minute ADAUSDT candles (crypto).

    Returns:
        KlineData: 1-minute OHLCV+turnover candles for ADAUSDT.
    """
    return _load_klines("adausd_1m")


def load_aapl_1d() -> "KlineData":
    """
    Load 300 daily AAPL candles (stock, from CSV).

    Returns:
        KlineData: Daily OHLCV candles for AAPL.
    """
    return _load_klines("aapl_1d")


def load_goog_1d() -> "KlineData":
    """
    Load 300 daily GOOG candles (stock, from CSV).

    Returns:
        KlineData: Daily OHLCV candles for GOOG.
    """
    return _load_klines("goog_1d")


# -----
# Tick loaders
# -----

def _load_ticks(name: str) -> "TickData":
    meta = DATASETS[name]
    kwargs = {"tick_size": meta["tick_size"], "digits": meta["digits"]}
    if "tick_value" in meta:
        kwargs["tick_value"] = meta["tick_value"]
    with _resolved(meta["file"]) as path:
        return read_ticks(str(path), meta["symbol"], **kwargs)


def load_mesu26_ticks() -> "TickData":
    """
    Load 2000 Micro E-mini S&P 500 (MESU26) futures ticks.

    Returns:
        TickData: Trade ticks with bid/ask/volume/flags for MESU26.

    Example:
        from tradetropy import BacktestEngine
        from tradetropy.datasets import load_mesu26_ticks

        ticks = load_mesu26_ticks()
        engine = BacktestEngine.by_ticks(MyStrategy(), data=(ticks,))
    """
    return _load_ticks("mesu26_ticks")


def load_mnqu26_ticks() -> "TickData":
    """
    Load 2000 Micro E-mini Nasdaq-100 (MNQU26) futures ticks.

    Returns:
        TickData: Trade ticks with bid/ask/volume/flags for MNQU26.
    """
    return _load_ticks("mnqu26_ticks")


def load_adausd_ticks() -> "TickData":
    """
    Load 43629 ADAUSDT ticks paired in time with ``load_adausd_book()``.

    The tick timestamps span exactly the order book's range, so the two replay
    together and the L2 order-flow indicators (``DeepTrades``, ``DeepWall``,
    ...) have a book to read as-of each trade.

    Returns:
        TickData: Trade ticks for ADAUSDT overlapping the bundled book.

    Example:
        from tradetropy.replay import ReplayEngine
        from tradetropy.datasets import load_adausd_ticks, load_adausd_book

        ticks = load_adausd_ticks()
        book  = load_adausd_book()
        engine = ReplayEngine.by_ticks(
            MyStrategy(), data=(ticks,), book=book,
        )
    """
    return _load_ticks("adausd_ticks")


# -----
# Order-book loader
# -----

def load_adausd_book() -> "BookData":
    """
    Load 240 rows of ADAUSDT L2 order-book depth (6 levels per side).

    Paired in time with ``load_adausd_ticks()`` for the L2 / order-flow
    examples. Feed it to the engine through ``ReplayEngine(book=...)``; a plain
    backtest has no order book.

    Returns:
        BookData: Wide L2 book snapshots (bid/ask price and size per level).
    """
    meta = DATASETS["adausd_book"]
    with _resolved(meta["file"]) as path:
        return read_book(
            str(path), meta["symbol"], levels=meta["levels"],
            tick_size=meta["tick_size"],
        )

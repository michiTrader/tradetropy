"""
Tests for the live-only public data-fetch API and symmetric book IO.

No network: SeshCCXTLive receives a small mock exchange. Covers:
  - supports_data_fetch capability flag (live True, simulator False).
  - Sesh.fetch_klines/fetch_ticks/fetch_orderbook (types + anti-lookahead guard).
  - Strategy.fetch_* run_mode guard (only 'live' allowed).
  - save_book / read_book / BookData.save round-trip (hdf5, parquet, csv).
"""

from __future__ import annotations

import numpy as np
import pytest

from tradetropy.connectors.ccxt import SeshCCXTLive, SeshCCXTSim
from tradetropy.core.data_types import BookData, KlineData, TickData, book_row_width
from tradetropy.exceptions import ConfigError
from tradetropy.io import read_book, save_book
from tradetropy.models.strategy import Strategy


# ══════════════════════════════════════════════════════════════════════════════
# Mock exchange (subset of ccxt's unified API used by the fetch primitives)
# ══════════════════════════════════════════════════════════════════════════════


class FetchMock:
    id = "mockex"

    def __init__(self, **kwargs):
        self.has = {
            "fetchOrderBook": True,
            "fetchOHLCV": True,
            "fetchTrades": True,
            "fetchPositions": False,   # skip sync() position polling
            "fetchBalance": False,     # skip sync() balance polling
            "loadMarkets": False,      # skip load_markets in __init__
        }

    def fetch_ohlcv(self, symbol, tf, since, limit):
        base = 1_700_000_000_000
        n = limit or 2
        return [
            [base + i * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
            for i in range(n)
        ]

    def fetch_trades(self, symbol, since, limit):
        base = 1_700_000_000_000
        n = limit or 3
        return [
            {
                "price": 100.0 + i,
                "amount": 1.0 + i,
                "timestamp": base + i,
                "side": "buy" if i % 2 == 0 else "sell",
            }
            for i in range(n)
        ]

    def fetch_order_book(self, symbol, depth=None):
        k = depth or 20
        return {
            "timestamp": 1_700_000_000_000,
            "bids": [[100.0 - i * 0.1, 5.0 + i] for i in range(k)],
            "asks": [[100.1 + i * 0.1, 6.0 + i] for i in range(k)],
        }


@pytest.fixture
def live():
    return SeshCCXTLive(FetchMock())


# ══════════════════════════════════════════════════════════════════════════════
# Capability flag
# ══════════════════════════════════════════════════════════════════════════════


def test_supports_data_fetch_flag(live):
    assert live.supports_data_fetch is True
    assert SeshCCXTSim().supports_data_fetch is False


# ══════════════════════════════════════════════════════════════════════════════
# Sesh public API - live
# ══════════════════════════════════════════════════════════════════════════════


def test_sesh_fetch_klines(live):
    kl = live.fetch_klines("BTC/USDT", "1m", limit=4)
    assert isinstance(kl, KlineData)
    assert kl.data.shape == (4, 7)
    assert np.isnan(kl.data[:, 6]).all()          # turnover is NaN


def test_sesh_fetch_ticks(live):
    tk = live.fetch_ticks("BTC/USDT", limit=3)
    assert isinstance(tk, TickData)
    assert tk.data.shape == (3, 7)


def test_sesh_fetch_orderbook(live):
    bk = live.fetch_orderbook("BTC/USDT", depth=3, tick_size=0.5)
    assert isinstance(bk, BookData)
    assert bk.levels == 3
    assert bk.data.shape == (1, book_row_width(3))
    assert bk.kind[-1] == 0.0                      # snapshot
    assert bk.bid_px[-1, 0] == pytest.approx(100.0)
    assert bk.ask_px[-1, 0] == pytest.approx(100.1)
    assert bk.bid_sz[-1, 0] == pytest.approx(5.0)
    assert bk.ask_sz[-1, 0] == pytest.approx(6.0)


# ══════════════════════════════════════════════════════════════════════════════
# Anti-lookahead guard on the simulator session
# ══════════════════════════════════════════════════════════════════════════════


def test_simulator_fetch_raises():
    sim = SeshCCXTSim()
    with pytest.raises(ConfigError):
        sim.fetch_klines("BTC/USDT", "1m")
    with pytest.raises(ConfigError):
        sim.fetch_ticks("BTC/USDT")
    with pytest.raises(ConfigError):
        sim.fetch_orderbook("BTC/USDT")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy run_mode guard
# ══════════════════════════════════════════════════════════════════════════════


class _Strat(Strategy):
    def init(self):
        pass

    def on_data(self):
        pass


@pytest.mark.parametrize("mode", ["backtest", "optimize", "pool"])
def test_strategy_fetch_blocked_off_live(live, mode):
    s = _Strat()
    s._sesh = live
    s._set_run_mode(mode)
    with pytest.raises(ConfigError):
        s.fetch_klines("BTC/USDT", "1m")
    with pytest.raises(ConfigError):
        s.fetch_ticks("BTC/USDT")
    with pytest.raises(ConfigError):
        s.fetch_orderbook("BTC/USDT")


def test_strategy_fetch_delegates_in_live(live):
    s = _Strat()
    s._sesh = live
    s._set_run_mode("live")
    assert isinstance(s.fetch_klines("BTC/USDT", "1m", limit=2), KlineData)
    assert isinstance(s.fetch_ticks("BTC/USDT", limit=2), TickData)
    bk = s.fetch_orderbook("BTC/USDT", depth=3)
    assert isinstance(bk, BookData) and bk.levels == 3


# ══════════════════════════════════════════════════════════════════════════════
# Book IO round-trip (fetch -> save -> read -> BookData)
# ══════════════════════════════════════════════════════════════════════════════


def _make_book(live) -> BookData:
    # Two stacked snapshots to exercise multi-row round-trip.
    r1 = live._fetch_orderbook("BTC/USDT", 3)
    r2 = live._fetch_orderbook("BTC/USDT", 3)
    r2 = r2.copy()
    r2[0, 0] += 1000.0                              # distinct ts
    data = np.vstack([r1, r2])
    return BookData("BTC/USDT", data, levels=3, tick_size=0.5)


def test_book_io_hdf5_roundtrip(tmp_path, live):
    pytest.importorskip("tables")
    book = _make_book(live)
    path = tmp_path / "book.h5"
    book.save(str(path))                            # hdf5 via BookData.save()
    # symbol/levels/tick_size recovered from HDF5 attrs (no args needed).
    back = read_book(str(path))
    assert back.symbol == "BTC/USDT"
    assert back.levels == 3
    assert back.tick_size == pytest.approx(0.5)
    assert np.allclose(back.bid_px, book.bid_px)
    assert np.allclose(back.ask_sz, book.ask_sz)
    assert np.allclose(back.ts, book.ts)


def test_book_io_parquet_roundtrip(tmp_path, live):
    pytest.importorskip("pyarrow")
    book = _make_book(live)
    path = tmp_path / "book.parquet"
    save_book(book, str(path), format="parquet")
    # levels inferred from bid_px_* columns; symbol explicit.
    back = read_book(str(path), "BTC/USDT")
    assert back.levels == 3
    assert np.allclose(back.bid_px, book.bid_px)
    assert np.allclose(back.ts, book.ts)


def test_book_io_csv_roundtrip(tmp_path, live):
    book = _make_book(live)
    path = tmp_path / "book.csv"
    save_book(book, str(path), format="csv")
    back = read_book(str(path), "BTC/USDT")
    assert back.levels == 3
    assert np.allclose(back.bid_px, book.bid_px)
    assert np.allclose(back.ask_px, book.ask_px)
    assert np.allclose(back.ts, book.ts)

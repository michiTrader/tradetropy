"""
Tests for the unified column accessor (ColumnRef).

A proxy column accessor (e.g. ``proxy.close``) now plays two roles with a
single name, removing the need for the ``_ref`` suffix:

- In ``init()`` it is a declarative ColumnRef consumed by ``add_indicator``.
- In ``on_data()`` it delegates indexing to the connected window view, so
  ``proxy.close[-1]`` reads the live value like the underlying WindowView.

The explicit ``*_ref`` accessors remain as backward-compatible aliases.
"""

import numpy as np
import pytest

from tradetropy import Strategy, BacktestEngine, KlineData
from tradetropy.ta import SMA
from tradetropy.data._proxy import ColumnRef, OhlcProxy, TickProxy
from tradetropy.exceptions import ColumnNotFoundError


def _make_klines(n: int = 120, interval_ms: int = 60_000) -> np.ndarray:
    """Build a deterministic OHLC matrix [N x 7]."""
    ts = np.arange(n, dtype=np.float64) * interval_ms
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.full(n, 10.0)
    turnover = volume * close
    return np.column_stack([ts, open_, high, low, close, volume, turnover])


# ==============================================================================
# Unit level: ColumnRef unified behavior
# ==============================================================================

class TestColumnRefUnit:
    def test_accessor_returns_columnref(self):
        """Plain accessors and *_ref aliases both return a ColumnRef."""
        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        assert isinstance(p.close, ColumnRef)
        assert isinstance(p.close_ref, ColumnRef)
        assert p.close.col_name == "close" == p.close_ref.col_name
        assert p.close.proxy is p and p.close_ref.proxy is p

        t = TickProxy("BTCUSDT", window_size=50)
        assert isinstance(t.price, ColumnRef)
        assert isinstance(t.price_ref, ColumnRef)
        assert t.price.col_name == "price" == t.price_ref.col_name

    def test_index_before_connect_raises(self):
        """Indexing a ref before the proxy is connected is a clear error."""
        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        with pytest.raises(ColumnNotFoundError):
            _ = p.close[-1]
        # len/iter degrade gracefully to empty (no data yet)
        assert len(p.close) == 0
        assert list(p.close) == []


# ==============================================================================
# Integration: unified accessor works in add_indicator AND on_data
# ==============================================================================

class TestColumnRefIntegration:
    def test_close_works_as_ref_and_as_data(self):
        klines = _make_klines()
        captured = {"checks": 0}

        class Strat(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", 60_000, window_size=100)
                # Declarative use of the plain accessor (no _ref suffix)
                self.sma_plain = self.add_indicator(self.btc.close, SMA(10))
                # Alias still accepted and equivalent
                self.sma_ref = self.add_indicator(self.btc.close_ref, SMA(10))

            def on_data(self):
                # Data use of the same plain accessor
                c_plain = self.btc.close[-1]
                c_ref = self.btc.close_ref[-1]
                assert c_plain == c_ref
                # Slice, len and iter all delegate to the live window view
                sl = self.btc.close[-3:]
                assert isinstance(sl, np.ndarray)
                assert sl[-1] == c_plain
                assert len(self.btc.close) == len(self.btc.close_ref)
                assert list(self.btc.close) == list(self.btc.close_ref)
                # Both indicators (declared via plain vs _ref) match
                if not np.isnan(self.sma_plain[-1]):
                    assert self.sma_plain[-1] == self.sma_ref[-1]
                captured["checks"] += 1

        engine = BacktestEngine.by_klines(
            Strat(),
            data=(KlineData(symbol="BTCUSDT", data=klines, timeframe=60_000),),
        )
        engine.run()
        assert captured["checks"] > 0

    def test_tick_price_accessor_unified(self):
        from tradetropy.core.data_types import TickData
        from tradetropy.core.constants import _TICK_COL, N_TICK_COLS

        n = 60
        ticks = np.zeros((n, N_TICK_COLS), dtype=np.float64)
        price = 100.0 + np.cumsum(np.sin(np.arange(n) / 4.0))
        ticks[:, _TICK_COL["ts"]] = np.arange(n, dtype=np.float64) * 1000.0
        ticks[:, _TICK_COL["bid"]] = price - 0.05
        ticks[:, _TICK_COL["ask"]] = price + 0.05
        ticks[:, _TICK_COL["volume"]] = 1.0
        ticks[:, _TICK_COL["price"]] = price
        captured = {"ok": False}

        class Strat(Strategy):
            def init(self):
                self.ticks = self.subscribe_ticks("BTCUSDT", window_size=100)

            def on_data(self):
                # Plain tick accessor delegates to data, identical to _ref alias
                if len(self.ticks.price) > 0:
                    assert self.ticks.price[-1] == self.ticks.price_ref[-1]
                    assert self.ticks.bid[-1] == self.ticks.bid_ref[-1]
                    captured["ok"] = True

        engine = BacktestEngine.by_ticks(
            Strat(),
            data=(TickData("BTCUSDT", ticks, tick_size=0.01),),
        )
        engine.run()
        assert captured["ok"]

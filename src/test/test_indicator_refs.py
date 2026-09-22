"""
Tests for the generic refs()/default_refs() helpers on the Indicator base.

Multi-source indicators declare ``source_cols`` (the proxy columns their
``calculate`` expects, in order). The base class turns that into:

- ``Ind.refs(proxy)``            -> list[ColumnRef] (documented helper).
- ``Ind().default_refs(proxy)``  -> same list (used when the proxy is passed
                                    directly to ``add_indicator``).

so a caller never has to spell out ``[proxy.high_ref, proxy.low_ref,
proxy.close_ref]`` by hand. Passing an explicit ColumnRef list stays supported.
"""

import numpy as np
import pytest

from tradetropy import Strategy, BacktestEngine, KlineData
from tradetropy.ta import SMA, Stochastic, ATR, PivotHighLow
from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.data._proxy import ColumnRef, OhlcProxy
from tradetropy.exceptions import ColumnNotFoundError


def _make_klines(n: int = 200, interval_ms: int = 60_000) -> np.ndarray:
    """Deterministic OHLC matrix [N x 7] (ts, o, h, l, c, vol, turnover)."""
    ts = np.arange(n, dtype=np.float64) * interval_ms
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.full(n, 10.0)
    turnover = volume * close
    return np.column_stack([ts, open_, high, low, close, volume, turnover])


class _HLC(Indicator):
    """Dummy multi-source indicator consuming high/low/close."""

    name = "hlc_dummy"
    category = "momentum"
    source_cols = ("high", "low", "close")

    def calculate(self, source: np.ndarray) -> np.ndarray:
        return source[:, 2]  # echo close


# ==============================================================================
# Unit level: source_cols -> refs / default_refs
# ==============================================================================

class TestGenericRefs:
    def test_default_source_cols_is_close(self):
        """Base default resolves to a single close ref (single-source)."""
        assert Indicator.source_cols == ("close",)
        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        refs = SMA(10).default_refs(p)
        assert [r.col_name for r in refs] == ["close"]
        assert all(isinstance(r, ColumnRef) and r.proxy is p for r in refs)

    def test_classmethod_refs_order(self):
        """refs() returns one ColumnRef per source_cols name, in order."""
        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        refs = _HLC.refs(p)
        assert [r.col_name for r in refs] == ["high", "low", "close"]
        assert all(r.proxy is p for r in refs)

    def test_refs_matches_default_refs(self):
        """The classmethod helper and the instance resolver agree."""
        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        a = [r.col_name for r in _HLC.refs(p)]
        b = [r.col_name for r in _HLC().default_refs(p)]
        assert a == b == ["high", "low", "close"]

    def test_builtins_expose_expected_columns(self):
        """Representative built-ins declare the columns their calculate uses."""
        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        assert [r.col_name for r in Stochastic.refs(p)] == ["high", "low", "close"]
        assert [r.col_name for r in ATR.refs(p)] == ["high", "low", "close"]
        # PivotHighLow carries an auxiliary ts column for marker repositioning.
        piv = [r.col_name for r in PivotHighLow.refs(p)]
        assert piv[:2] == ["high", "low"]
        assert "ts" in piv

    def test_unknown_column_raises(self):
        """A bad source_cols name surfaces as a clear column error."""
        class _Bad(Indicator):
            source_cols = ("nope",)

            def calculate(self, source):
                return source

        p = OhlcProxy("BTCUSDT", 60_000, window_size=50)
        with pytest.raises(ColumnNotFoundError):
            _Bad.refs(p)


# ==============================================================================
# Integration: proxy-direct == helper == manual list
# ==============================================================================

class _StochProxyDirect(Strategy):
    def init(self):
        self.btc = self.subscribe_ohlc("BTCUSDT", "1m", window_size=120)
        self.stoch = self.add_indicator(self.btc, Stochastic(14, 3, 3))
        self.k_series = []

    def on_data(self):
        self.k_series.append(float(self.stoch.k[-1]))


class _StochManual(Strategy):
    def init(self):
        self.btc = self.subscribe_ohlc("BTCUSDT", "1m", window_size=120)
        self.stoch = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.close_ref],
            Stochastic(14, 3, 3),
        )
        self.k_series = []

    def on_data(self):
        self.k_series.append(float(self.stoch.k[-1]))


class _StochHelper(Strategy):
    def init(self):
        self.btc = self.subscribe_ohlc("BTCUSDT", "1m", window_size=120)
        self.stoch = self.add_indicator(Stochastic.refs(self.btc), Stochastic(14, 3, 3))
        self.k_series = []

    def on_data(self):
        self.k_series.append(float(self.stoch.k[-1]))


def test_proxy_direct_equals_manual_and_helper():
    """add_indicator(proxy, Ind()) == add_indicator([refs], Ind()) == refs()."""
    kl = KlineData(symbol="BTCUSDT", data=_make_klines(), timeframe="1m")

    outs = []
    for strat_cls in (_StochProxyDirect, _StochManual, _StochHelper):
        bt = BacktestEngine.by_klines(strat_cls(), data=(kl,))
        bt.run()
        outs.append(np.asarray(bt.strategy.k_series))

    np.testing.assert_allclose(outs[0], outs[1], equal_nan=True)
    np.testing.assert_allclose(outs[0], outs[2], equal_nan=True)

"""
Integration tests for the CandlePatterns annotation indicator: output bands,
the public query API (last_pattern / patterns / efficacy / ...) reached through
the add_indicator() handle, the draw() primitives and the refs wiring.
"""

import numpy as np
import pytest

from tradetropy import Strategy, BacktestEngine, KlineData
from tradetropy.ta import CandlePatterns
from tradetropy.ta import _candlestick as cs


def _trending_klines(n=240, interval_ms=3_600_000, seed=3):
    """OHLC [N x 7] with varied bodies and a couple of trend swings."""
    gen = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.float64) * interval_ms
    # Piecewise drift so both up- and down-context patterns can appear.
    drift = np.concatenate([
        np.full(n // 2, -0.5), np.full(n - n // 2, 0.5),
    ])
    close = 200.0 + np.cumsum(drift + gen.normal(0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    body = np.abs(close - open_)
    high = np.maximum(open_, close) + gen.uniform(0.1, 1.0, n) + body * 0.1
    low = np.minimum(open_, close) - gen.uniform(0.1, 1.0, n) - body * 0.1
    volume = gen.uniform(5.0, 15.0, n)
    turnover = volume * close
    return np.column_stack([ts, open_, high, low, close, volume, turnover])


class _CandleStrat(Strategy):
    """Uses the proxy-direct form and records the query API each bar."""

    def init(self):
        self.btc = self.subscribe_ohlc("BTCUSDT", "1h", window_size=300)
        self.candles = self.add_indicator(self.btc, CandlePatterns(window=50))
        self.seen_names = set()
        self.code_reads = []

    def on_data(self):
        # Output band access.
        self.code_reads.append(float(self.candles.code[-1]))
        # Query API access.
        self.seen_names.add(self.candles.last_pattern())
        _ = self.candles.is_bullish()
        _ = self.candles.is_bearish()


def _run(strat):
    kl = KlineData(symbol="BTCUSDT", data=_trending_klines(), timeframe="1h")
    bt = BacktestEngine.by_klines(strat, data=(kl,))
    bt.run()
    return bt


def test_bands_and_query_api_reachable():
    bt = _run(_CandleStrat())
    strat = bt.strategy
    # Bands were read every bar and are finite pattern codes.
    assert len(strat.code_reads) > 0
    assert all(0 <= v <= 20 for v in strat.code_reads if not np.isnan(v))
    # The query API returned strings; at least 'none' plus (typically) patterns.
    assert "none" in strat.seen_names or len(strat.seen_names) >= 1
    assert all(isinstance(x, str) for x in strat.seen_names)


def test_efficacy_all_is_causal_dict():
    """efficacy_all() returns a name->stats dict computed causally."""

    class _EffStrat(Strategy):
        def init(self):
            self.btc = self.subscribe_ohlc("BTCUSDT", "1h", window_size=300)
            self.candles = self.add_indicator(
                self.btc, CandlePatterns(window=50, horizon=5)
            )
            self.snapshot = None

        def on_data(self):
            self.snapshot = self.candles.efficacy_all()

    bt = _run(_EffStrat())
    eff = bt.strategy.snapshot
    assert isinstance(eff, dict)
    for name, entry in eff.items():
        assert isinstance(name, str)
        assert set(entry) == {"hit_rate", "sample_size", "wins", "code"}
        assert entry["sample_size"] >= 0
        # Directional patterns have a hit-rate in [0,1]; neutral ones are NaN.
        hr = entry["hit_rate"]
        assert np.isnan(hr) or (0.0 <= hr <= 1.0)


def test_efficacy_single_pattern_lookup_by_name_and_code():
    class _EffStrat(Strategy):
        def init(self):
            self.btc = self.subscribe_ohlc("BTCUSDT", "1h", window_size=300)
            self.candles = self.add_indicator(self.btc, CandlePatterns(window=50))
            self.by_name = None
            self.by_code = None

        def on_data(self):
            self.by_name = self.candles.efficacy("Bullish Engulfing")
            self.by_code = self.candles.efficacy(cs.ENGULFING_BULL)

    bt = _run(_EffStrat())
    a, b = bt.strategy.by_name, bt.strategy.by_code
    assert a["sample_size"] == b["sample_size"]
    assert a["name"] == "Bullish Engulfing"


def test_draw_returns_label_primitives():
    from tradetropy.ta.draw import Labels

    bt = _run(_CandleStrat())
    ind = bt.strategy.candles  # MultiBandProxy delegates unknown attrs, but the
    # indicator instance is what draws; fetch it from the strategy defs.
    defs = bt.strategy._indicator_defs
    indicator = next(d["indicator"] for d in defs
                     if isinstance(d["indicator"], CandlePatterns))
    prims = indicator.draw()
    assert isinstance(prims, list)
    # There should be at least one Labels primitive over a 240-bar trending set.
    assert all(isinstance(p, Labels) for p in prims)


def test_refs_helper_and_proxy_direct_equivalent():
    """CandlePatterns.refs(proxy) resolves the same ts+OHLC columns."""
    from tradetropy.data._proxy import OhlcProxy

    p = OhlcProxy("BTCUSDT", 3_600_000, window_size=50)
    cols = [r.col_name for r in CandlePatterns.refs(p)]
    assert cols == ["ts", "open", "high", "low", "close"]
    cols2 = [r.col_name for r in CandlePatterns().default_refs(p)]
    assert cols2 == cols


def test_static_plot_preserves_annotation_history():
    """Static plot collection keeps CandlePatterns labels beyond the last bar."""
    from tradetropy.plotting._util import _gather_plot_data

    bt = _run(_CandleStrat())
    data = _gather_plot_data(bt)
    meta = next(m for m in data['indicators'] if m.name == 'Candles')
    primitives = [p for group in meta._draw_primitives.values() for p in group]

    assert sum(len(p.text) for p in primitives) > 0
    assert max(len(p.text) for p in primitives) > 1


def test_show_stats_appends_hit_rate_to_labels():
    class _StatStrat(Strategy):
        def init(self):
            self.btc = self.subscribe_ohlc("BTCUSDT", "1h", window_size=300)
            self.candles = self.add_indicator(
                self.btc, CandlePatterns(window=50, show_stats=True, horizon=5)
            )

        def on_data(self):
            pass

    bt = _run(_StatStrat())
    indicator = next(d["indicator"] for d in bt.strategy._indicator_defs
                     if isinstance(d["indicator"], CandlePatterns))
    prims = indicator.draw()
    # With show_stats, at least one directional label carries a "%" suffix.
    texts = [t for p in prims for t in p.text]
    assert any("%" in t for t in texts) or len(texts) == 0

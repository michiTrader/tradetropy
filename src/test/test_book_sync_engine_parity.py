"""End-to-end parity: a recorded L2 book replayed through BacktestEngine.by_ticks
produces the SAME DeepTrades decision stream as through ReplayEngine, and the
sync preflight reaches the same verdict. Also covers the by_klines(book=) guard.
"""

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.core.data_types import (
    BookData, TickData, KlineData, book_row_width,
)
from tradetropy.backtest import BacktestEngine
from tradetropy.replay.engine import ReplayEngine
from tradetropy.models.strategy import Strategy
from tradetropy.ta import DeepTrades
from tradetropy.ta.order_flow._core import EVENT_ABSORPTION, EVENT_SWEEP
from tradetropy.exceptions import ConfigError


def _ticks():
    n = 10
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    vols = [1, 1, 1, 1, 1, 16, 1, 90, 1, 1]  # large buys at index 5 and 7
    for i in range(n):
        out[i, _TICK_COL["ts"]] = 1000 + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = 100.0
        out[i, _TICK_COL["volume"]] = float(vols[i])
        out[i, _TICK_COL["volume_real"]] = float(vols[i])
        out[i, _TICK_COL["flags"]] = 1.0  # buy aggressor
    return out


def _book():
    levels = 4
    w = book_row_width(levels)

    def row(ts, bids, asks):
        r = np.full(w, np.nan, dtype=np.float64)
        r[0] = ts
        r[1] = 0
        for j, (p, s) in enumerate(bids):
            r[2 + j] = p
            r[2 + levels + j] = s
        for j, (p, s) in enumerate(asks):
            r[2 + 2 * levels + j] = p
            r[2 + 3 * levels + j] = s
        return r

    rows = np.array([
        row(1004, [(99.0, 5.0)],
            [(100.0, 5.0), (101.0, 5.0), (102.0, 5.0), (103.0, 5.0)]),
        row(1006, [(99.0, 5.0)], [(100.0, 100.0), (101.0, 5.0)]),
    ], dtype=np.float64)
    return BookData("BTCUSDT", rows, levels=levels)


class _DeepStrat(Strategy):
    warmup = 2

    def init(self):
        self.ticks = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.book = self.subscribe_orderbook("BTCUSDT", depth=4)
        self.deep = self.add_indicator(
            DeepTrades.refs(self.ticks),
            DeepTrades(self.book, threshold=5.0, by="volume", window=5,
                       min_resting_volume=50.0, absorption_ratio=0.8,
                       stack_depth=3),
        )
        self.events = []

    def on_data(self):
        et = self.deep.event_type[-1]
        if not np.isnan(et):
            self.events.append((int(self.ticks.ts[-1]), int(et)))


def _run_backtest():
    strat = _DeepStrat()
    engine = BacktestEngine.by_ticks(
        strat,
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
    )
    engine.run(verbose=False)
    return dict(strat.events)


def _run_replay():
    strat = _DeepStrat()
    engine = ReplayEngine.by_ticks(
        strat,
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
        warmup_ticks=2,
        speed=float("inf"),
    )
    engine.prepare()
    engine._rebuild_to_cursor({"BTCUSDT": 9})
    return dict(engine.strategy.events)


def test_backtest_with_book_classifies_deeptrades():
    """A backtest over ticks WITH a book classifies DeepTrades (sweep/absorption)."""
    bt = _run_backtest()
    assert bt.get(1005) == EVENT_SWEEP
    assert bt.get(1007) == EVENT_ABSORPTION


def test_backtest_replay_book_parity():
    """Same ticks + book -> identical DeepTrades decision stream in both engines."""
    bt = _run_backtest()
    rp = _run_replay()
    assert bt == rp


def test_by_klines_rejects_book():
    """BacktestEngine.by_klines(book=) is a ConfigError (needs per-trade ts)."""
    klines = KlineData(
        "BTCUSDT",
        np.array([[0, 1, 2, 0.5, 1.5, 10, 0]], dtype=np.float64),
        timeframe="1m",
    )
    with pytest.raises(ConfigError, match="by_klines"):
        BacktestEngine.by_klines(
            _DeepStrat(), data=(klines,), book=_book(),
        )


def test_book_accepts_single_and_tuple():
    """book= accepts a single BookData and a 1-tuple identically (symbol from
    BookData.symbol), both classifying the same DeepTrades stream."""
    def run(book_arg):
        strat = _DeepStrat()
        BacktestEngine.by_ticks(
            strat,
            data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
            book=book_arg,
        ).run(verbose=False)
        return dict(strat.events)

    single = run(_book())
    as_tuple = run((_book(),))
    assert single == as_tuple
    assert single.get(1005) == EVENT_SWEEP
    assert single.get(1007) == EVENT_ABSORPTION


def test_book_dict_rejected():
    """The removed {symbol: BookData} mapping raises ConfigError."""
    with pytest.raises(ConfigError, match="Dict-based `book`"):
        BacktestEngine.by_ticks(
            _DeepStrat(),
            data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
            book={"BTCUSDT": _book()},
        )


class _BookMetricStratPool(Strategy):
    """Reads the L2 book directly in on_data (best_bid) so the pool book replay
    can be validated without the multi-band DeepTrades precompute (unsupported
    in the pool tick worker). Returns the best_bid seen at each tick."""

    warmup = 0

    def init(self):
        self.ticks = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.book = self.subscribe_orderbook("BTCUSDT", depth=4)
        self.seen = []

    def on_data(self):
        self.seen.append((int(self.ticks.ts[-1]), float(self.book.best_bid)))

    def result(self):
        return dict(self.seen)


def test_pool_with_book_populates_book():
    """PoolBacktestEngine.by_ticks(book=) replays the book so a strategy reads
    live book metrics in on_data (best_bid becomes 99.0 once the book lands)."""
    from tradetropy.backtest.pool import PoolBacktestEngine

    results = PoolBacktestEngine.by_ticks(
        [_BookMetricStratPool],
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
        workers=1,
    )
    assert len(results) == 1
    seen = results[0]
    # Before the first book row (ts 1004) the book is stale -> NaN.
    assert np.isnan(seen[1000])
    # From ts 1004 on, the book is live: best bid is 99.0.
    assert seen[1005] == 99.0
    assert seen[1007] == 99.0


def test_pool_book_matches_single_backtest():
    """The pool's book-metric stream matches a single BacktestEngine run."""
    from tradetropy.backtest.pool import PoolBacktestEngine

    strat = _BookMetricStratPool()
    bt = BacktestEngine.by_ticks(
        strat,
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
    )
    bt.run(verbose=False)
    single = dict(strat.seen)

    pooled = PoolBacktestEngine.by_ticks(
        [_BookMetricStratPool],
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
        workers=1,
    )[0]
    # Compare on finite values (NaN != NaN) and NaN positions.
    assert single.keys() == pooled.keys()
    for k in single:
        a, b = single[k], pooled[k]
        assert (np.isnan(a) and np.isnan(b)) or a == b


def test_pool_by_klines_rejects_book():
    """PoolBacktestEngine.by_klines(book=) raises ConfigError."""
    from tradetropy.backtest.pool import PoolBacktestEngine

    klines = KlineData(
        "BTCUSDT",
        np.array([[0, 1, 2, 0.5, 1.5, 10, 0]], dtype=np.float64),
        timeframe="1m",
    )
    with pytest.raises(ConfigError, match="by_ticks"):
        PoolBacktestEngine.by_klines(
            [_BookMetricStratPool], data=(klines,), book=_book(),
        )


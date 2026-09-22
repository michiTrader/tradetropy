"""
Replay parity (trades-only): the backtest == live invariant for the streaming
path.

A strategy with a tick-mounted indicator runs live over a scripted trade feed
while recording the stream to disk; replaying that recording through the SAME
engine event path must reproduce the identical on_data() decision stream.
"""

import math

import numpy as np
import pytest

from tradetropy.core.broker import AccountInfo
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.io.io import read_ticks, read_book
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import SeshLiveBase, SeshSimulatorBase
from tradetropy.streaming import (
    FakeFeed,
    OrderbookSnapshot,
    ReplayFeed,
    TradeEvent,
)
from tradetropy.ta import SMA


class _RecordStreamSesh(SeshLiveBase):
    """Non-simulated streaming session (recording is active) over a FakeFeed."""

    def __init__(self, events):
        super().__init__()
        self._events = events

    @property
    def supports_streaming(self):
        return True

    def create_feed(self, **kwargs):
        return FakeFeed(self._events)

    def buy(self, *a, **k):
        return None

    def sell(self, *a, **k):
        return None

    def positions(self, symbol=None):
        return []

    def account_info(self):
        return AccountInfo(0.0, 0.0, 0.0, 0.0, 0.0)


class _ReplayStreamSesh(SeshSimulatorBase):
    """Simulated streaming session that replays recorded ticks via ReplayFeed."""

    def __init__(self, ticks_by_symbol, **kw):
        super().__init__(**kw)
        self._ticks_by_symbol = ticks_by_symbol

    @property
    def supports_streaming(self):
        return True

    def create_feed(self, **kwargs):
        return ReplayFeed.from_ticks(self._ticks_by_symbol)


class _ParityStrat(Strategy):
    warmup = 0
    rec_path = None

    def init(self):
        self.tp = self.subscribe_ticks(
            "BTCUSDT", window_size=300, record=self.rec_path
        )
        self.sma = self.add_indicator(self.tp.price_ref, SMA(5))
        self.decisions = []

    def on_data(self):
        sma = float(self.sma[-1])
        price = float(self.tp.price[-1])
        self.decisions.append(
            (
                int(self.tp.ts[-1]),
                round(price, 8),
                None if np.isnan(sma) else round(sma, 8),
                bool(price > sma) if not np.isnan(sma) else None,
            )
        )


def _hist_ticks(n=3, base_ts=1_000_000, price=100.0):
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i in range(n):
        out[i, _TICK_COL["ts"]] = base_ts + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = price
        out[i, _TICK_COL["volume"]] = 1.0
    return out


def test_record_then_replay_parity(tmp_path):
    rec = tmp_path / "parity_btc.h5"
    rng = np.random.default_rng(7)
    prices = 100.0 + np.cumsum(rng.standard_normal(40))
    events = [
        TradeEvent("BTCUSDT", 2_000_000 + i * 10, float(prices[i]), 1.0,
                   side=(1 if i % 2 == 0 else -1))
        for i in range(40)
    ]
    hist = _hist_ticks()

    # --- Live run: stream + record, capture decisions ---
    live_strat = _ParityStrat()
    live_strat.rec_path = str(rec)
    live_engine = LiveEngine.by_ticks(live_strat, sesh=_RecordStreamSesh(events))
    live_engine.prepare({"BTCUSDT": hist})
    live_engine.run(blocking=True)
    live_engine.stop()
    live_decisions = list(live_strat.decisions)

    assert len(live_decisions) == 40  # one on_data per streamed trade

    # --- Replay run: same warmup history, replay recorded stream ---
    recorded = read_ticks(rec, "BTCUSDT").data
    replay_strat = _ParityStrat()  # no recording this time
    replay_engine = LiveEngine.by_ticks(
        replay_strat, sesh=_ReplayStreamSesh({"BTCUSDT": recorded}, feed_type="tick")
    )
    replay_engine.prepare({"BTCUSDT": hist})
    replay_engine.run(blocking=True)
    replay_decisions = list(replay_strat.decisions)

    # The decision streams must be byte-identical.
    assert replay_decisions == live_decisions


# =====
# Book replay parity (trades + L2 order book)
# =====
class _ReplayBookSesh(SeshSimulatorBase):
    """Replays recorded trades + book via ReplayFeed.from_records."""

    def __init__(self, ticks_by_symbol, books_by_symbol, **kw):
        super().__init__(**kw)
        self._ticks = ticks_by_symbol
        self._books = books_by_symbol

    @property
    def supports_streaming(self):
        return True

    def create_feed(self, **kwargs):
        return ReplayFeed.from_records(self._ticks, self._books)


class _BookParityStrat(Strategy):
    warmup = 0
    tick_rec = None
    book_rec = None

    def init(self):
        self.tp = self.subscribe_ticks(
            "BTCUSDT", window_size=300, record=self.tick_rec
        )
        self.book = self.subscribe_orderbook(
            "BTCUSDT", depth=5, record=self.book_rec
        )
        self.sma = self.add_indicator(self.tp.price_ref, SMA(5))
        self.decisions = []

    def on_data(self):
        imb = self.book.imbalance()
        mid = self.book.mid
        self.decisions.append(
            (
                int(self.tp.ts[-1]),
                round(float(self.tp.price[-1]), 8),
                None if math.isnan(imb) else round(imb, 8),
                None if math.isnan(mid) else round(mid, 8),
            )
        )


def test_record_then_replay_book_parity(tmp_path):
    tick_rec = tmp_path / "bp_ticks.h5"
    book_rec = tmp_path / "bp_book.h5"
    rng = np.random.default_rng(11)

    # Interleave book then trade with strictly increasing distinct ts.
    events = []
    ts = 2_000_000
    price = 100.0
    for i in range(20):
        ts += 1
        bid_top = float(1 + (i % 3))
        events.append(
            OrderbookSnapshot(
                "BTCUSDT", ts,
                ((100.0, bid_top), (99.5, 2.0)),
                ((100.5, 1.0), (101.0, 1.5)),
            )
        )
        ts += 1
        price += float(rng.standard_normal())
        events.append(
            TradeEvent("BTCUSDT", ts, price, 1.0, side=(1 if i % 2 == 0 else -1))
        )

    hist = _hist_ticks()

    # --- Live: stream + record trades and book ---
    live_strat = _BookParityStrat()
    live_strat.tick_rec = str(tick_rec)
    live_strat.book_rec = str(book_rec)
    live_engine = LiveEngine.by_ticks(live_strat, sesh=_RecordStreamSesh(events))
    live_engine.prepare({"BTCUSDT": hist})
    live_engine.run(blocking=True)
    live_engine.stop()
    live_decisions = list(live_strat.decisions)
    assert len(live_decisions) == 20

    # The live book metrics actually varied (not all NaN/constant imbalance).
    imbs = {d[2] for d in live_decisions}
    assert len([x for x in imbs if x is not None]) >= 2

    # --- Replay: merge recorded trades + book, same warmup ---
    ticks = read_ticks(tick_rec, "BTCUSDT").data
    book = read_book(book_rec, "BTCUSDT")
    replay_strat = _BookParityStrat()
    replay_engine = LiveEngine.by_ticks(
        replay_strat,
        sesh=_ReplayBookSesh(
            {"BTCUSDT": ticks}, {"BTCUSDT": book}, feed_type="tick"
        ),
    )
    replay_engine.prepare({"BTCUSDT": hist})
    replay_engine.run(blocking=True)
    replay_decisions = list(replay_strat.decisions)

    assert replay_decisions == live_decisions

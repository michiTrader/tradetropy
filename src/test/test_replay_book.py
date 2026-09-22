"""
ReplayEngine replay of a recorded L2 book alongside ticks.

A recorded book is replayed time-aligned with the ticks through the same
ReplayEngine path that powers the pause/play UI, so DeepTrades / book metrics
behave identically to the live streaming replay. Driven headless via
_rebuild_to_cursor (the same mechanism the rewind controls use).
"""

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.core.data_types import BookData, TickData, book_flat_columns, book_row_width
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine
from tradetropy.ta import DeepTrades
from tradetropy.ta.order_flow._core import EVENT_ABSORPTION, EVENT_SWEEP


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
        r[1] = 0  # snapshot
        for j, (p, s) in enumerate(bids):
            r[2 + j] = p
            r[2 + levels + j] = s
        for j, (p, s) in enumerate(asks):
            r[2 + 2 * levels + j] = p
            r[2 + 3 * levels + j] = s
        return r

    rows = np.array([
        # ts=1004 sweep book (thin stacked asks): a 16-lot buy clears 3 levels.
        row(1004, [(99.0, 5.0)],
            [(100.0, 5.0), (101.0, 5.0), (102.0, 5.0), (103.0, 5.0)]),
        # ts=1006 absorption book (big ask wall): a 90-lot buy is absorbed.
        row(1006, [(99.0, 5.0)], [(100.0, 100.0), (101.0, 5.0)]),
    ], dtype=np.float64)
    return BookData("BTCUSDT", rows, levels=levels)


class _DeepReplayStrat(Strategy):
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


def test_replay_engine_replays_book_for_deeptrades():
    engine = ReplayEngine.by_ticks(
        _DeepReplayStrat(),
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
        warmup_ticks=2,
        speed=float("inf"),
    )
    engine.prepare()

    # Drive the replay headless to the end (same mechanism as the rewind UI).
    engine._rebuild_to_cursor({"BTCUSDT": 9})

    by_ts = dict(engine.strategy.events)
    assert by_ts.get(1005) == EVENT_SWEEP
    assert by_ts.get(1007) == EVENT_ABSORPTION


def test_replay_book_resets_on_rewind():
    engine = ReplayEngine.by_ticks(
        _DeepReplayStrat(),
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book(),
        warmup_ticks=2,
        speed=float("inf"),
    )
    engine.prepare()
    engine._rebuild_to_cursor({"BTCUSDT": 9})
    first = dict(engine.strategy.events)
    # Rewind to the start and replay again: book cursor resets, same result.
    engine._rebuild_to_cursor({"BTCUSDT": 9})
    second = dict(engine.strategy.events)
    assert first == second
    assert second.get(1005) == EVENT_SWEEP
    assert second.get(1007) == EVENT_ABSORPTION

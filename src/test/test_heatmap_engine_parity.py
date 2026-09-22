"""
Engine parity for the Heatmap indicator: replaying a recorded L2 book through
ReplayEngine reproduces the identical liquidity grid and query-API answers that
the pure ``build_heatmap_grid`` produces over the same book, and the replay is
deterministic across a rewind.

This is the heatmap analogue of the DeepTrades replay-book test: the order book
is fed to the Heatmap constructor and the strategy reads it causally via the
public query API (grid / hottest / liquidity_at / walls).
"""

import numpy as np

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.core.data_types import BookData, TickData, book_row_width
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine
from tradetropy.ta import Heatmap
from tradetropy.ta.order_flow._core import build_heatmap_grid

LEVELS = 4
PRICE_BUCKET = 1.0


def _ticks():
    n = 10
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i in range(n):
        out[i, _TICK_COL["ts"]] = 1000 + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = 100.0
        out[i, _TICK_COL["volume"]] = 1.0
        out[i, _TICK_COL["volume_real"]] = 1.0
        out[i, _TICK_COL["flags"]] = 1.0
    return out


# The two book snapshots fed during replay (ask wall at 101, persists).
_SNAPS = [
    {"ts": 1002, "bids": [(99.0, 5.0), (98.0, 4.0)],
     "asks": [(100.0, 6.0), (101.0, 500.0)]},
    {"ts": 1005, "bids": [(99.0, 7.0), (98.0, 4.0)],
     "asks": [(100.0, 6.0), (101.0, 500.0)]},
]


def _book_data():
    w = book_row_width(LEVELS)

    def row(ts, bids, asks):
        r = np.full(w, np.nan, dtype=np.float64)
        r[0] = ts
        r[1] = 0  # snapshot
        for j, (p, s) in enumerate(bids):
            r[2 + j] = p
            r[2 + LEVELS + j] = s
        for j, (p, s) in enumerate(asks):
            r[2 + 2 * LEVELS + j] = p
            r[2 + 3 * LEVELS + j] = s
        return r

    rows = np.array([row(s["ts"], s["bids"], s["asks"]) for s in _SNAPS],
                    dtype=np.float64)
    return BookData("BTCUSDT", rows, levels=LEVELS)


def _expected_book_window():
    """Independent book_window dict matching what the ring reconstructs."""
    r = len(_SNAPS)
    bid_px = np.full((r, LEVELS), np.nan)
    bid_sz = np.full((r, LEVELS), np.nan)
    ask_px = np.full((r, LEVELS), np.nan)
    ask_sz = np.full((r, LEVELS), np.nan)
    for j, s in enumerate(_SNAPS):
        for i, (p, sz) in enumerate(s["bids"]):
            bid_px[j, i] = p
            bid_sz[j, i] = sz
        for i, (p, sz) in enumerate(s["asks"]):
            ask_px[j, i] = p
            ask_sz[j, i] = sz
    return {"ts": np.array([s["ts"] for s in _SNAPS], dtype=np.int64),
            "kind": np.zeros(r, dtype=np.int64),
            "bid_px": bid_px, "bid_sz": bid_sz,
            "ask_px": ask_px, "ask_sz": ask_sz, "levels": LEVELS}


class _HeatStrat(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks("BTCUSDT", window_size=50)
        self.book = self.subscribe_orderbook("BTCUSDT", depth=LEVELS)
        self.heat = self.add_indicator(
            Heatmap.refs(self.ticks),
            Heatmap(self.book, price_bucket=PRICE_BUCKET, persistence_ms=0,
                    wall_rel_multiple=5.0, show_bubbles=False),
        )


def _run():
    engine = ReplayEngine.by_ticks(
        _HeatStrat(),
        data=(TickData("BTCUSDT", _ticks(), tick_size=0.01),),
        book=_book_data(),
        warmup_ticks=2,
        speed=float("inf"),
    )
    engine.prepare()
    engine._rebuild_to_cursor({"BTCUSDT": 9})
    return engine


def test_replay_grid_matches_pure_builder():
    engine = _run()
    heat = engine.strategy.heat

    exp = build_heatmap_grid(_expected_book_window(), price_bucket=PRICE_BUCKET)
    got = heat.grid()
    assert got is not None

    assert np.array_equal(got.col_ts, exp["col_ts"])
    assert np.allclose(got.price_centers, exp["price_centers"])
    assert np.allclose(got.bid, exp["bid"])
    assert np.allclose(got.ask, exp["ask"])


def test_replay_query_api_answers():
    heat = _run().strategy.heat

    # The 500-lot ask wall at 101 is the hottest level and a persistent wall.
    top = heat.hottest(n=1)
    assert top and top[0].price == 101.0 and top[0].size == 500.0
    assert top[0].side == "ask"

    assert heat.liquidity_at(101.0, side="ask") == 500.0
    assert heat.liquidity_at(99.0, side="bid") == 7.0     # last snapshot bid

    nw = heat.nearest_wall("ask")
    assert nw is not None and nw.price == 101.0
    assert nw.persistence_ms == 3  # rested from ts 1002 to 1005


def test_replay_is_deterministic_across_rewind():
    engine = _run()
    first = engine.strategy.heat.grid()
    # Rewind and replay again: the book cursor resets, same grid.
    engine._rebuild_to_cursor({"BTCUSDT": 9})
    second = engine.strategy.heat.grid()
    assert np.array_equal(first.bid, second.bid)
    assert np.array_equal(first.ask, second.ask)
    assert np.array_equal(first.col_ts, second.col_ts)


def test_query_grid_stays_causal_no_frozen_colors():
    # The append-only color accumulation is a DRAWING concern (live_refresh).
    # The query API must keep reading the causal book window, never the
    # accumulated / frozen-color grid, so a strategy cannot see future columns.
    proxy = _run().strategy.heat
    heat = proxy._query_indicator          # the underlying Heatmap indicator
    causal = heat._live_grid()
    assert causal is not None
    assert "color_lo" not in causal
    assert "color_hi" not in causal
    # The accumulator (drawing) is not populated by the headless replay path.
    assert heat._grid_accum == {}

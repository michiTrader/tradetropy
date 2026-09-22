"""Tests for the pure detect_deep_trades L2 classifier (_core.py)."""

import numpy as np
import pytest

from tradetropy.ta.order_flow._core import (
    EVENT_ABSORPTION,
    EVENT_LARGE_AGGRESSOR,
    EVENT_SWEEP,
    classify_deep_trade,
    detect_deep_trades,
)


def _book(bid_px, bid_sz, ask_px, ask_sz):
    return {
        "bid_px": np.array(bid_px, dtype=np.float64),
        "bid_sz": np.array(bid_sz, dtype=np.float64),
        "ask_px": np.array(ask_px, dtype=np.float64),
        "ask_sz": np.array(ask_sz, dtype=np.float64),
    }


# =====
# classify_deep_trade (single event)
# =====
class TestClassify:
    def test_large_aggressor_small_trade(self):
        # Buy hits asks; small trade vs a thin level, no wall -> aggressor.
        book = _book([99.0], [5.0], [100.0, 101.0, 102.0], [5.0, 5.0, 5.0])
        etype, resting, cleared = classify_deep_trade(
            side=1, traded=2.0, book=book,
            min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        )
        assert etype == EVENT_LARGE_AGGRESSOR
        assert resting == 5.0 and cleared == 0

    def test_absorption_into_wall(self):
        # Buy of 90 into a resting ask wall of 100: big fraction, not cleared.
        book = _book([99.0], [10.0], [100.0, 101.0], [100.0, 10.0])
        etype, resting, cleared = classify_deep_trade(
            side=1, traded=90.0, book=book,
            min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        )
        assert etype == EVENT_ABSORPTION
        assert resting == 100.0 and cleared == 0

    def test_sweep_clears_levels(self):
        # Buy of 16 clears asks 5+5+5=15 fully (3 levels) -> sweep.
        book = _book([99.0], [5.0], [100.0, 101.0, 102.0, 103.0],
                     [5.0, 5.0, 5.0, 5.0])
        etype, resting, cleared = classify_deep_trade(
            side=1, traded=16.0, book=book,
            min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        )
        assert etype == EVENT_SWEEP
        assert cleared >= 3

    def test_sell_consumes_bids(self):
        # Sell hits bids; big fraction of a bid wall, not cleared -> absorption.
        book = _book([100.0, 99.0], [200.0, 10.0], [101.0], [10.0])
        etype, resting, cleared = classify_deep_trade(
            side=-1, traded=180.0, book=book,
            min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        )
        assert etype == EVENT_ABSORPTION
        assert resting == 200.0

    def test_no_book_is_large_aggressor(self):
        etype, resting, cleared = classify_deep_trade(
            side=1, traded=10.0, book=None,
            min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        )
        assert etype == EVENT_LARGE_AGGRESSOR
        assert np.isnan(resting) and cleared == 0


# =====
# detect_deep_trades (end to end over arrays)
# =====
class TestDetectDeepTrades:
    def test_classifies_detected_events(self):
        # 6 trades; one huge sweep-sized buy, one absorption-sized buy.
        ts = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        price = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        volume = np.array([1.0, 1.0, 16.0, 1.0, 90.0, 1.0])

        def book_fn(t):
            if t == 3:  # sweep book: thin stacked asks
                return _book([99.0], [5.0],
                             [100.0, 101.0, 102.0, 103.0], [5.0, 5.0, 5.0, 5.0])
            if t == 5:  # absorption book: big ask wall
                return _book([99.0], [10.0], [100.0, 101.0], [100.0, 10.0])
            return _book([99.0], [5.0], [100.0], [5.0])

        res = detect_deep_trades(
            ts, price, volume, book_fn,
            threshold=5.0, by="volume", window=3,
            min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        )
        ev = res["events"]
        # Absolute threshold 5.0 catches the 16 and 90 lots.
        types = {int(t): et for t, et in zip(ev["ts"], ev["event_type"])}
        assert types.get(3) == EVENT_SWEEP
        assert types.get(5) == EVENT_ABSORPTION
        # consumed/resting populated.
        idx = list(ev["ts"].astype(int)).index(3)
        assert ev["consumed"][idx] >= 3

    def test_no_events_when_below_threshold(self):
        ts = np.arange(1, 6, dtype=np.float64)
        price = np.full(5, 100.0)
        volume = np.ones(5)
        res = detect_deep_trades(
            ts, price, volume, lambda t: None,
            threshold=1000.0, window=3,
        )
        assert len(res["events"]["ts"]) == 0
        assert res["events"]["event_type"].shape == (0,)

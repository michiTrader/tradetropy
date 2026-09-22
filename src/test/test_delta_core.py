"""Tests for the pure per-bar delta / CVD core (_core.py)."""

import numpy as np

from tradetropy.ta.order_flow._core import (
    bar_index,
    bar_delta,
    cumulative_delta_ohlc,
)


# Two 1000 ms bars. flags: +1 buy aggressor, -1 sell aggressor.
#   bar 0 [0, 1000): buy 3, sell 1            -> ask 3, bid 1, delta +2, total 4
#   bar 1 [1000, 2000): buy 2, sell 5         -> ask 2, bid 5, delta -3, total 7
TS = np.array([100, 200, 300, 400, 1100, 1200, 1300], dtype=np.int64)
PRICE = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
VOLUME = np.array([3.0, 1.0, 2.0, 5.0, 2.0, 5.0, 0.0])
# Re-shape so the numbers are clean: assign explicit volumes/sides below.
TS = np.array([100, 200, 1100, 1200], dtype=np.int64)
PRICE = np.array([10.0, 10.0, 10.0, 10.0])
VOLUME = np.array([3.0, 1.0, 2.0, 5.0])
FLAGS = np.array([1.0, -1.0, 1.0, -1.0])   # buy3, sell1 | buy2, sell5
INTERVAL = 1000


class TestBarIndex:
    def test_grid(self):
        ids = bar_index(TS, INTERVAL)
        assert list(ids) == [0, 0, 1, 1]

    def test_anchor_shift(self):
        ids = bar_index(np.array([1000, 1999, 2000]), 1000, anchor=0)
        assert list(ids) == [1, 1, 2]


class TestBarDelta:
    def test_per_bar_figures(self):
        r = bar_delta(TS, PRICE, VOLUME, interval_ms=INTERVAL, flags=FLAGS)
        assert list(r["bar_ts"]) == [0, 1000]
        np.testing.assert_allclose(r["ask_vol"], [3.0, 2.0])
        np.testing.assert_allclose(r["bid_vol"], [1.0, 5.0])
        np.testing.assert_allclose(r["delta"], [2.0, -3.0])
        np.testing.assert_allclose(r["total"], [4.0, 7.0])
        # The anchor tick is the last trade of each bar.
        assert list(r["rep_ts"]) == [200, 1200]

    def test_empty(self):
        r = bar_delta(np.array([], dtype=np.int64), np.array([]), np.array([]),
                      interval_ms=INTERVAL)
        assert len(r["delta"]) == 0


class TestCumulativeDeltaOHLC:
    def test_ohlc_chains_and_bounds(self):
        r = cumulative_delta_ohlc(TS, PRICE, VOLUME, interval_ms=INTERVAL,
                                  flags=FLAGS)
        # bar 0: signed path 0 -> +3 -> +2 ; open 0, close +2, high +3, low 0
        # bar 1: open +2, signed +2 -> +4 -> -1 ; close -1, high +4, low -1
        np.testing.assert_allclose(r["open"], [0.0, 2.0])
        np.testing.assert_allclose(r["close"], [2.0, -1.0])
        np.testing.assert_allclose(r["high"], [3.0, 4.0])
        np.testing.assert_allclose(r["low"], [0.0, -1.0])
        # open chains to previous close
        assert r["open"][1] == r["close"][0]
        # bounds hold like a real candle
        assert np.all(r["high"] >= r["open"])
        assert np.all(r["high"] >= r["close"])
        assert np.all(r["low"] <= r["open"])
        assert np.all(r["low"] <= r["close"])

    def test_delta_matches_bar_delta(self):
        cvd = cumulative_delta_ohlc(TS, PRICE, VOLUME, interval_ms=INTERVAL,
                                    flags=FLAGS)
        bd = bar_delta(TS, PRICE, VOLUME, interval_ms=INTERVAL, flags=FLAGS)
        np.testing.assert_allclose(cvd["delta"], bd["delta"])

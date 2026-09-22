"""Tests for the pure L3/MBO detectors: detect_iceberg and detect_liquidity_grab."""

import numpy as np
import pytest

from tradetropy.core.data_types import MBO_ADD, MBO_CANCEL, MBO_MODIFY, MBO_TRADE
from tradetropy.ta.order_flow._core import detect_iceberg, detect_liquidity_grab


def _cols(rows):
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


# =====
# Iceberg (reloads at a price)
# =====
class TestIceberg:
    def test_reloads_detected(self):
        # Price 100 on bid side gets eaten then refilled 3 times -> iceberg.
        rows = [
            [1, 1, 1, 100.0, 5.0, MBO_ADD],
            [2, 1, 1, 100.0, 0.0, MBO_TRADE],   # eaten
            [3, 2, 1, 100.0, 5.0, MBO_ADD],     # reload 1
            [4, 2, 1, 100.0, 0.0, MBO_TRADE],
            [5, 3, 1, 100.0, 5.0, MBO_ADD],     # reload 2
            [6, 3, 1, 100.0, 0.0, MBO_TRADE],
            [7, 4, 1, 100.0, 5.0, MBO_ADD],     # reload 3 -> detect
        ]
        ts, oid, side, price, size, action = _cols(rows)
        res = detect_iceberg(ts, oid, side, price, size, action,
                             reload_threshold=3, window_ms=100)
        assert len(res["ts"]) == 1
        assert res["price"][0] == 100.0
        assert res["side"][0] == 1
        assert res["reloads"][0] == 3

    def test_no_iceberg_below_threshold(self):
        rows = [
            [1, 1, 1, 100.0, 5.0, MBO_ADD],
            [2, 1, 1, 100.0, 0.0, MBO_TRADE],
            [3, 2, 1, 100.0, 5.0, MBO_ADD],     # only 1 reload
        ]
        ts, oid, side, price, size, action = _cols(rows)
        res = detect_iceberg(ts, oid, side, price, size, action,
                             reload_threshold=3, window_ms=100)
        assert len(res["ts"]) == 0

    def test_reload_outside_window_ignored(self):
        rows = [
            [1, 1, 1, 100.0, 5.0, MBO_ADD],
            [2, 1, 1, 100.0, 0.0, MBO_TRADE],
            [9_999, 2, 1, 100.0, 5.0, MBO_ADD],  # too late
        ]
        ts, oid, side, price, size, action = _cols(rows)
        res = detect_iceberg(ts, oid, side, price, size, action,
                             reload_threshold=1, window_ms=100)
        assert len(res["ts"]) == 0


# =====
# Liquidity grab (sweep + reversal)
# =====
class TestLiquidityGrab:
    def test_upside_sweep_then_reversal(self):
        # Trades sweep price up 100->103 (3 ticks), then reverse back to 100.
        rows = [
            [1, 0, -1, 100.0, 1.0, MBO_TRADE],
            [2, 0, -1, 101.0, 1.0, MBO_TRADE],
            [3, 0, -1, 102.0, 1.0, MBO_TRADE],
            [4, 0, -1, 103.0, 1.0, MBO_TRADE],   # extreme
            [600, 0, -1, 100.0, 1.0, MBO_TRADE],  # reversal back through start
        ]
        ts, oid, side, price, size, action = _cols(rows)
        res = detect_liquidity_grab(ts, price, action, tick_size=1.0,
                                    sweep_levels=3, sweep_window_ms=100,
                                    reversal_window_ms=5000)
        assert len(res["ts"]) == 1
        assert res["price"][0] == 103.0   # the swept extreme
        assert res["side"][0] == -1       # up-sweep -> downside grab

    def test_no_grab_without_reversal(self):
        rows = [
            [1, 0, -1, 100.0, 1.0, MBO_TRADE],
            [2, 0, -1, 101.0, 1.0, MBO_TRADE],
            [3, 0, -1, 102.0, 1.0, MBO_TRADE],
            [4, 0, -1, 103.0, 1.0, MBO_TRADE],
            # price stays up, no reversal
            [600, 0, -1, 103.0, 1.0, MBO_TRADE],
        ]
        ts, oid, side, price, size, action = _cols(rows)
        res = detect_liquidity_grab(ts, price, action, tick_size=1.0,
                                    sweep_levels=3, sweep_window_ms=100,
                                    reversal_window_ms=5000)
        assert len(res["ts"]) == 0

    def test_no_grab_when_sweep_too_small(self):
        rows = [
            [1, 0, -1, 100.0, 1.0, MBO_TRADE],
            [2, 0, -1, 101.0, 1.0, MBO_TRADE],   # only 1 tick
            [600, 0, -1, 100.0, 1.0, MBO_TRADE],
        ]
        ts, oid, side, price, size, action = _cols(rows)
        res = detect_liquidity_grab(ts, price, action, tick_size=1.0,
                                    sweep_levels=3, sweep_window_ms=100,
                                    reversal_window_ms=5000)
        assert len(res["ts"]) == 0

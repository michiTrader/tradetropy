"""Tests for DeepTrades L3 mode: iceberg/grab upgrade of coincident events."""

import numpy as np
import pytest

from tradetropy.core.data_types import MBO_ADD, MBO_TRADE
from tradetropy.ta import DeepTrades
from tradetropy.ta.order_flow._core import (
    EVENT_ICEBERG, EVENT_LARGE_AGGRESSOR,
)


class _FakeMbo:
    """Minimal MboProxy stand-in exposing events()."""

    def __init__(self, rows):
        self._rows = np.array(rows, dtype=np.float64)

    def events(self):
        return self._rows


class _FakeBook:
    """Minimal orderbook proxy: book_as_of(ts) -> book dict."""

    def book_as_of(self, ts):
        return {
            "bid_px": np.array([99.0], dtype=np.float64),
            "bid_sz": np.array([5.0], dtype=np.float64),
            "ask_px": np.array([100.0], dtype=np.float64),
            "ask_sz": np.array([50.0], dtype=np.float64),
        }


def _tick_source():
    # [ts, price, volume, flags, bid, ask]; one large buy at ts=1007.
    rows = [
        [1000, 100.0, 1.0, 1, 100.0, 100.0],
        [1002, 100.0, 1.0, 1, 100.0, 100.0],
        [1004, 100.0, 1.0, 1, 100.0, 100.0],
        [1006, 100.0, 1.0, 1, 100.0, 100.0],
        [1007, 100.0, 20.0, 1, 100.0, 100.0],
    ]
    return np.array(rows, dtype=np.float64)


def test_iceberg_upgrade_at_coincident_trade():
    # MBO reloads at price 100 culminating in a 3rd reload at ts=1007.
    mbo_rows = [
        [1000, 1, 1, 100.0, 5.0, MBO_ADD],
        [1001, 1, 1, 100.0, 0.0, MBO_TRADE],
        [1002, 2, 1, 100.0, 5.0, MBO_ADD],   # reload 1
        [1003, 2, 1, 100.0, 0.0, MBO_TRADE],
        [1004, 3, 1, 100.0, 5.0, MBO_ADD],   # reload 2
        [1005, 3, 1, 100.0, 0.0, MBO_TRADE],
        [1007, 4, 1, 100.0, 5.0, MBO_ADD],   # reload 3 -> iceberg at ts=1007
    ]
    ind = DeepTrades(
        orderbook=_FakeBook(), mbo=_FakeMbo(mbo_rows),
        threshold=5.0, by="volume", window=5,
        reload_threshold=3, reload_window_ms=10_000,
    )
    out = ind.calculate(_tick_source())
    assert out.shape == (7, 5)
    # The detected large trade at ts=1007 is upgraded to Iceberg.
    et = ind._deep_events["event_type"]
    assert len(et) == 1
    assert int(et[0]) == EVENT_ICEBERG
    assert int(out[4, -1]) == EVENT_ICEBERG


def test_no_mbo_keeps_l2_classification():
    # Without an MBO proxy, the event keeps its L2 classification.
    ind = DeepTrades(orderbook=_FakeBook(), threshold=5.0, by="volume", window=5)
    out = ind.calculate(_tick_source())
    et = ind._deep_events["event_type"]
    assert len(et) == 1
    assert int(et[0]) == EVENT_LARGE_AGGRESSOR

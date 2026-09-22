"""
Self-recording of the live streaming event stream.

A non-simulated streaming session records the trade-driven tick rows through
the existing _process_tick recording path; the file must round-trip via
read_ticks so it can later be replayed (Task 7 parity).
"""

import numpy as np
import pytest

from tradetropy.core.broker import AccountInfo
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.io.io import read_ticks
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import SeshLiveBase
from tradetropy.streaming import FakeFeed, TradeEvent


class _RecordStreamSesh(SeshLiveBase):
    """Non-simulated streaming session (recording is active for live sessions)."""

    def __init__(self, events):
        super().__init__()
        self._events = events

    @property
    def supports_streaming(self) -> bool:
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
        return AccountInfo(
            balance=0.0, equity=0.0, margin=0.0, margin_free=0.0, profit=0.0
        )


def _hist_ticks(n=3, base_ts=1_000_000):
    out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
    for i in range(n):
        out[i, _TICK_COL["ts"]] = base_ts + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = 100.0
        out[i, _TICK_COL["volume"]] = 1.0
    return out


def test_streaming_records_replayable_ticks(tmp_path):
    rec = tmp_path / "rec_btc.h5"

    class _RecStrat(Strategy):
        warmup = 0

        def init(self):
            self.tp = self.subscribe_ticks(
                "BTCUSDT", window_size=100,
                record=str(rec), record_flush_every=2,
            )

        def on_data(self):
            pass

    events = [
        TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0 + 0.1 * i,
                   side=(1 if i % 2 == 0 else -1))
        for i in range(7)
    ]
    sesh = _RecordStreamSesh(events)
    engine = LiveEngine.by_ticks(_RecStrat(), sesh=sesh)
    engine.prepare({"BTCUSDT": _hist_ticks()})
    engine.run(blocking=True)
    engine.stop()  # flush remaining buffer

    td = read_ticks(rec, "BTCUSDT")
    data = td.data
    # All 7 streamed trades recorded (warmup=0, so none suppressed).
    assert len(data) == 7
    assert list(data[:, _TICK_COL["ts"]].astype(int)) == [
        2_000_000 + i for i in range(7)
    ]
    np.testing.assert_allclose(
        data[:, _TICK_COL["price"]], [100.0 + i for i in range(7)]
    )
    # Aggressor side preserved in flags (+1/-1) for order-flow replay.
    assert list(data[:, _TICK_COL["flags"]].astype(int)) == [
        1, -1, 1, -1, 1, -1, 1
    ]


def test_streaming_record_persists_symbol_metadata(tmp_path):
    """The record= path stores the symbol as HDF5 attrs, so read_ticks()
    recovers it without an explicit symbol argument."""
    rec = tmp_path / "rec_meta.h5"

    class _RecStrat(Strategy):
        warmup = 0

        def init(self):
            self.tp = self.subscribe_ticks(
                "ETHUSDT", window_size=100,
                record=str(rec), record_flush_every=2,
            )

        def on_data(self):
            pass

    events = [
        TradeEvent("ETHUSDT", 3_000_000 + i, 50.0 + i, 1.0, side=1)
        for i in range(5)
    ]
    sesh = _RecordStreamSesh(events)
    engine = LiveEngine.by_ticks(_RecStrat(), sesh=sesh)
    engine.prepare({"ETHUSDT": _hist_ticks()})
    engine.run(blocking=True)
    engine.stop()

    # No explicit symbol: it must be recovered from the persisted attrs.
    td = read_ticks(rec)
    assert td.symbol == "ETHUSDT"
    assert len(td.data) == 5

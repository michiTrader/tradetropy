"""Tests for the L3/MBO data layer: MboData, MboRing, MboProxy, IO, engine wiring."""

import numpy as np
import pytest

from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.core.data_types import (
    MboData, MBO_ADD, MBO_CANCEL, MBO_MODIFY, MBO_TRADE, N_MBO_COLS,
)
from tradetropy.data.data import MboRing
from tradetropy.io.io import _append_mbo_hdf5, read_mbo
from tradetropy.live.engine import LiveEngine
from tradetropy.models.strategy import Strategy
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.streaming import FakeFeed, MboEvent, TradeEvent


# =====
# MboData
# =====
class TestMboData:
    def test_layout_and_accessors(self):
        rows = [[1000, 5, 1, 100.0, 3.0, MBO_ADD]]
        md = MboData("BTC", np.array(rows, dtype=np.float64))
        assert md.ts[0] == 1000
        assert md.order_id[0] == 5
        assert md.side[0] == 1
        assert md.price[0] == 100.0
        assert md.size[0] == 3.0
        assert md.action[0] == MBO_ADD

    def test_bad_shape_raises(self):
        from tradetropy.exceptions import ConfigError
        with pytest.raises(ConfigError):
            MboData("BTC", np.zeros((1, 4)))


# =====
# MboRing reconstruction
# =====
class TestMboRing:
    def test_add_modify_cancel(self):
        r = MboRing(window_size=100)
        r.apply_event(1, order_id=1, side=1, price=100.0, size=3.0, action=MBO_ADD)
        r.apply_event(2, order_id=2, side=1, price=100.0, size=2.0, action=MBO_ADD)
        assert r.resting_size_at(1, 100.0) == pytest.approx(5.0)
        # modify order 1 down
        r.apply_event(3, order_id=1, side=1, price=100.0, size=1.0, action=MBO_MODIFY)
        assert r.resting_size_at(1, 100.0) == pytest.approx(3.0)
        # cancel order 2
        r.apply_event(4, order_id=2, side=1, price=100.0, size=0.0, action=MBO_CANCEL)
        assert r.resting_size_at(1, 100.0) == pytest.approx(1.0)

    def test_trade_reduces_and_removes(self):
        r = MboRing(window_size=100)
        r.apply_event(1, order_id=1, side=-1, price=101.0, size=5.0, action=MBO_ADD)
        r.apply_event(2, order_id=1, side=-1, price=101.0, size=2.0, action=MBO_TRADE)
        assert r.resting_size_at(-1, 101.0) == pytest.approx(2.0)
        r.apply_event(3, order_id=1, side=-1, price=101.0, size=0.0, action=MBO_TRADE)
        assert r.resting_size_at(-1, 101.0) == pytest.approx(0.0)

    def test_event_window(self):
        r = MboRing(window_size=100)
        for i in range(5):
            r.apply_event(1000 + i, order_id=i, side=1, price=100.0,
                          size=1.0, action=MBO_ADD)
        win = r.window()
        assert win.shape == (5, N_MBO_COLS)
        assert list(win[:, 0].astype(int)) == [1000, 1001, 1002, 1003, 1004]


# =====
# IO round-trip
# =====
class TestMboIO:
    def test_append_read_roundtrip(self, tmp_path):
        r = MboRing(window_size=100)
        r.apply_event(1, 1, 1, 100.0, 3.0, MBO_ADD)
        r.apply_event(2, 2, -1, 101.0, 2.0, MBO_ADD)
        r.apply_event(3, 1, 1, 100.0, 1.0, MBO_MODIFY)
        rows = r.to_rows()
        path = tmp_path / "mbo.h5"
        _append_mbo_hdf5(rows, path)
        md = read_mbo(path, "BTC")
        assert isinstance(md, MboData)
        assert len(md.data) == 3
        np.testing.assert_allclose(md.data, rows)


# =====
# Engine wiring: MBO events feed the MboProxy through the streaming path
# =====
class _MboStreamSesh(SeshSimulatorBase):
    def __init__(self, events, **kw):
        super().__init__(**kw)
        self._events = events

    @property
    def supports_streaming(self):
        return True

    def create_feed(self, **kwargs):
        return FakeFeed(self._events)


class _MboStrat(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("BTC/USDT", window_size=50)
        self.mbo = self.subscribe_mbo("BTC/USDT", window_size=1000)
        self.snapshots = []

    def on_data(self):
        self.snapshots.append(
            (int(self.tp.ts[-1]), len(self.mbo),
             self.mbo.resting_size_at(1, 100.0))
        )


def _hist():
    out = np.zeros((3, N_TICK_COLS), dtype=np.float64)
    for i in range(3):
        out[i, _TICK_COL["ts"]] = 1 + i
        for c in ("bid", "ask", "price"):
            out[i, _TICK_COL[c]] = 100.0
        out[i, _TICK_COL["volume"]] = 1.0
    return out


def test_mbo_events_feed_proxy_through_engine():
    events = [
        MboEvent("BTC/USDT", 1_000, order_id=1, side=1, price=100.0, size=3.0,
                 action=MBO_ADD),
        TradeEvent("BTC/USDT", 1_001, 100.0, 1.0, side=1),
        MboEvent("BTC/USDT", 1_002, order_id=2, side=1, price=100.0, size=2.0,
                 action=MBO_ADD),
        TradeEvent("BTC/USDT", 1_003, 100.0, 1.0, side=1),
    ]
    sesh = _MboStreamSesh(events, feed_type="tick")
    engine = LiveEngine.by_ticks(_MboStrat(), sesh=sesh)
    engine.prepare({"BTC/USDT": _hist()})
    engine.run(blocking=True)

    snaps = engine.strategy.snapshots
    # on_data fires on the 2 trades; by the 2nd trade both MBO adds applied.
    by_ts = {ts: (n, rest) for ts, n, rest in snaps}
    assert by_ts[1_001][0] == 1          # one MBO event seen so far
    assert by_ts[1_001][1] == pytest.approx(3.0)
    assert by_ts[1_003][0] == 2          # two MBO events seen
    assert by_ts[1_003][1] == pytest.approx(5.0)  # 3 + 2 resting at 100

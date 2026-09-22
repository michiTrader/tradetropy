"""
Engine parity for tick-mounted relative-threshold indicators.

A tick-mounted indicator whose threshold is relative ('pXX' / 'Nx') needs its
full trailing detection ``window`` to decide the current tick. The live/replay
loop feeds each indicator a causal window sized by ``max(min_periods, length)``;
LargeTrades / DeepTrades declare ``length = window`` for relative thresholds, so
their on_data() bands reproduce the backtest's full-series detection instead of
seeing a single tick (which would leave the 'pXX' threshold in perpetual warmup
-> NaN bands, so a strategy reading ``self.whales.price[-1]`` would fire in
backtest but never in live/replay).

Regression guard for that gap: the detection stream a strategy sees in on_data()
must be identical in BacktestEngine and ReplayEngine over the span both engines
evaluate (i.e. past the replay warmup).
"""

import numpy as np

from tradetropy.core import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.backtest.engine import BacktestEngine
from tradetropy.replay.engine import ReplayEngine
from tradetropy.session import SeshSimulatorBase
from tradetropy.ta.order_flow import LargeTrades

WARMUP = 30


def _raw(n=120):
    # TickData raw layout: [ts, bid, ask, volume, flags, volume_real, price]
    rng = np.random.default_rng(3)
    ts = np.arange(n) * 1000
    price = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    volume = rng.uniform(1.0, 3.0, n)
    for s in (55, 75, 95):
        volume[s] = 100.0
    flags = np.ones(n)
    vreal = volume.copy()
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, bid, ask, volume, flags, vreal, price])


def _strategy(sink):
    class S(Strategy):
        def init(self):
            self.tk = self.subscribe_ticks("SYM", window_size=50)
            self.whales = self.add_indicator(
                LargeTrades.refs(self.tk),
                LargeTrades(threshold="p90", by="volume", window=20),
            )

        def on_data(self):
            v = self.whales.price[-1]
            if v is not None and np.isfinite(v):
                sink.append((int(self.tk.ts[-1]),
                             float(self.whales.volume[-1]),
                             float(self.whales.side[-1])))
    return S()


def _run_backtest(raw):
    seen = []
    BacktestEngine.by_ticks(
        _strategy(seen),
        data=(TickData("SYM", raw, tick_size=0.01),),
        sesh=SeshSimulatorBase("tick"),
    ).run()
    return seen


def _run_replay(raw):
    seen = []
    eng = ReplayEngine.by_ticks(
        _strategy(seen),
        data=(TickData("SYM", raw, tick_size=0.01),),
        warmup_ticks=WARMUP,
        speed=float("inf"),
    )
    eng.prepare()
    eng._rebuild_to_cursor({"SYM": len(raw) - 1})
    return seen


def test_replay_detects_large_trades_via_bands():
    """Regression: replay must not leave the relative threshold in warmup."""
    raw = _raw()
    rp = _run_replay(raw)
    assert len(rp) > 0, "replay produced no detections (min_periods=1 bug)"


def test_backtest_replay_band_detection_parity():
    raw = _raw()
    warmup_ts = int(raw[WARMUP, 0])

    bt = _run_backtest(raw)
    rp = _run_replay(raw)

    bt_span = sorted(e for e in bt if e[0] >= warmup_ts)
    rp_span = sorted(rp)

    assert rp_span == bt_span
    assert len(rp_span) > 0
    # Magnitudes and sides carried on the bands match too (full tuple equality).
    assert [e[1] for e in rp_span] == [e[1] for e in bt_span]
    assert [e[2] for e in rp_span] == [e[2] for e in bt_span]

"""
Replay parity for same-millisecond ticks (candle volume and order flow).

Real trade tapes carry several trades in the same millisecond. The backtest and
the replay rewind path sum all of them into a candle; the forward playback loop
must do the same. A regression guard against the old ``ts > last`` de-dup in the
forward loop, which silently dropped same-millisecond trades and shrank replay
candle volume (and every order-flow metric) versus the backtest.
"""

import numpy as np

from tradetropy.core import TickData
from tradetropy.core.constants import _OHLC_COL
from tradetropy.models.strategy import Strategy
from tradetropy.backtest.engine import BacktestEngine
from tradetropy.replay.engine import ReplayEngine
from tradetropy.session import SeshSimulatorBase
from tradetropy.data._klines import build_candles_from_ticks

INTERVAL_MS = 60_000
DUPS = 6                      # trades sharing each millisecond
TICKS_PER_CANDLE = 10
VOL = 10.0
EXPECTED_PER_CANDLE = DUPS * TICKS_PER_CANDLE * VOL   # 600


def _raw(n_candles=6):
    # TickData raw layout: [ts, bid, ask, volume, flags, volume_real, price]
    rows = []
    for c in range(n_candles):
        base = c * INTERVAL_MS
        for k in range(TICKS_PER_CANDLE):
            ts = base + k * (INTERVAL_MS // TICKS_PER_CANDLE)
            price = 100.0 + 0.01 * k
            for _ in range(DUPS):
                rows.append([ts, price - 0.05, price + 0.05, VOL, 1.0, 7.0, price])
    return np.array(rows, dtype=np.float64)


class _S(Strategy):
    def init(self):
        self.c = self.subscribe_ohlc("SYM", "1m", window_size=200)
        self.tk = self.subscribe_ticks("SYM", window_size=1000)

    def on_data(self):
        pass


def _backtest_volumes(raw):
    r = build_candles_from_ticks(raw[:, 0], raw[:, 6], raw[:, 3], INTERVAL_MS)
    return list(r["closed_candles"][:, _OHLC_COL["volume"]])


def _replay_forward_volumes(raw):
    eng = ReplayEngine.by_ticks(
        _S(),
        data=(TickData("SYM", raw, tick_size=0.01),),
        warmup_ticks=0,
        speed=float("inf"),
    )
    eng.prepare()
    eng._replay_ultimo_ts = {"SYM": -1}
    eng._replay_ultimo_idx = {"SYM": -1}
    eng._sesh._cursor["SYM"] = len(raw) - 1
    eng._drain_pending_ticks("SYM")   # exact forward-loop drain

    ring = next(op._ohlc_ring for op in eng.strategy._ohlc_proxies
                if op.symbol == "SYM")
    n = min(ring._n_closed, ring._W)
    return list(ring.closed_window(_OHLC_COL["volume"], n))


def _replay_rewind_volumes(raw):
    eng = ReplayEngine.by_ticks(
        _S(),
        data=(TickData("SYM", raw, tick_size=0.01),),
        warmup_ticks=5,
        speed=float("inf"),
    )
    eng.prepare()
    eng._rebuild_to_cursor({"SYM": len(raw) - 1})
    ring = next(op._ohlc_ring for op in eng.strategy._ohlc_proxies
                if op.symbol == "SYM")
    n = min(ring._n_closed, ring._W)
    return list(ring.closed_window(_OHLC_COL["volume"], n))


def test_forward_replay_keeps_same_ms_ticks():
    raw = _raw()
    fwd = _replay_forward_volumes(raw)
    assert fwd, "forward replay produced no closed candles"
    # Every closed candle must carry the full same-ms volume, not 1/DUPS of it.
    assert all(v == EXPECTED_PER_CANDLE for v in fwd), fwd


def test_backtest_replay_candle_volume_parity():
    raw = _raw()
    bt = _backtest_volumes(raw)
    fwd = _replay_forward_volumes(raw)
    rewind = _replay_rewind_volumes(raw)
    assert fwd == bt
    assert rewind == bt
    assert bt and bt[0] == EXPECTED_PER_CANDLE

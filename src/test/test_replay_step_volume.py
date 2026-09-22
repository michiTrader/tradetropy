"""
Regression: replay volume must not shrink when play + step advance the cursor
concurrently.

Root cause guarded here
-----------------------
``BaseEngine._drain_pending_ticks`` used to read the ReplaySesh cursor TWICE per
drain, non-atomically:

    pendientes = sesh._fetch_pending_ticks(sym, last_idx)     # reads cursor C1
    ...apply pendientes...
    self._replay_ultimo_idx[sym] = sesh._cursor.get(sym)      # RE-reads cursor C2

In the UI two threads advance the cursor at once: the controller loop (while
playing) and the Bokeh IOLoop (the Step Forward button). If a ``step()`` lands
in the window between the two reads, the cursor moves C1 -> C2 (C2 > C1). The
drain applies ticks only up to C1 but marks ``_replay_ultimo_idx = C2``, so the
ticks in ``(C1, C2]`` are silently skipped forever (until a rewind). Their
volume never reaches the rings, so replayed candle volume falls below the
backtest -- both on the chart and in the data read from ``on_data()``.

The fix snapshots the cursor ONCE per drain and uses that single index both to
bound the fetch and to update ``_replay_ultimo_idx`` (``up_to=`` on
``_fetch_pending_ticks``), so a concurrent ``step()`` can never make the drain
skip a tick.

These run headless (no Bokeh server, no browser).
"""

import threading

import numpy as np
import pytest

from tradetropy.backtest.engine import BacktestEngine
from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.core.data_types import TickData
from tradetropy.models.strategy import Strategy
from tradetropy.replay.engine import ReplayEngine
from tradetropy.session import SeshSimulatorBase


def klines_to_ticks(klines: np.ndarray) -> np.ndarray:
    """Convert klines [N x 6/7] to a tick matrix [N x 7] using close as price."""
    klines = np.asarray(klines, dtype=np.float64)
    close = klines[:, 4]
    out = np.zeros((len(klines), N_TICK_COLS), dtype=np.float64)
    out[:, _TICK_COL["ts"]] = klines[:, 0]
    out[:, _TICK_COL["bid"]] = close
    out[:, _TICK_COL["ask"]] = close
    out[:, _TICK_COL["volume"]] = klines[:, 5]
    out[:, _TICK_COL["flags"]] = 0.0
    out[:, _TICK_COL["volume_real"]] = 0.0
    out[:, _TICK_COL["price"]] = close
    return out


class _VolStrategy(Strategy):
    """Accumulates the volume of every tick delivered to on_data()."""

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=5000)
        self.total_volume = 0.0
        self.n_ticks = 0

    def on_data(self):
        self.total_volume += float(self.tp.volume[-1])
        self.n_ticks += 1


@pytest.fixture
def ticks(klines_20k):
    # 300 candles: enough to accumulate meaningful volume, small enough to be
    # fast (the deterministic test drains the whole live region tick by tick).
    return klines_to_ticks(klines_20k[:300])


def _build_replay(ticks):
    engine = ReplayEngine.by_ticks(
        _VolStrategy(),
        data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        speed=float("inf"),
    )
    engine.prepare()
    return engine


def _backtest_volume(ticks) -> float:
    """Total on_data() tick volume seen by a plain backtest over the same data."""
    strat = _VolStrategy()
    BacktestEngine.by_ticks(
        strat,
        data=(TickData("BTCUSDT", ticks, tick_size=0.25),),
        sesh=SeshSimulatorBase("tick"),
    ).run()
    return strat.total_volume


@pytest.mark.unit
class TestReplayConcurrentStepVolume:
    def test_backtest_volume_matches_live_region(self, ticks):
        """Sanity: the backtest sums exactly the live-region tick volume."""
        engine = _build_replay(ticks)
        sym = "BTCUSDT"
        n_warmup = engine._sesh._warmup_ticks[sym]
        expected = float(ticks[n_warmup:, _TICK_COL["volume"]].sum())
        assert _backtest_volume(ticks) == pytest.approx(expected)

    def test_step_during_drain_does_not_skip_volume(self, ticks):
        """
        Deterministic reproduction of the play+step race.

        A ``step()`` is injected in the exact window between the fetch and the
        ``_replay_ultimo_idx`` update inside ``_drain_pending_ticks`` (by
        advancing the cursor from a wrapped ``_fetch_pending_ticks``). With the
        old double-read this skips ticks and shrinks the replayed volume; with
        the single-snapshot fix every tick is still applied exactly once.
        """
        engine = _build_replay(ticks)
        sym = "BTCUSDT"
        n_warmup = engine._sesh._warmup_ticks[sym]
        end = len(ticks) - 1

        real_fetch = engine._sesh._fetch_pending_ticks

        def racy_fetch(symbol, last_idx, *args, **kwargs):
            pend = real_fetch(symbol, last_idx, *args, **kwargs)
            # Simulate the OTHER stepper (controller loop / Step button) firing
            # in the race window, jumping the cursor ahead of what this drain
            # just read.
            cur = engine._sesh._cursor.get(symbol, -1)
            engine._sesh._cursor[symbol] = min(cur + 3, end)
            return pend

        engine._sesh._fetch_pending_ticks = racy_fetch

        # Start the drain bookkeeping at the live boundary, one tick pending.
        engine._replay_ultimo_ts = {sym: -1}
        engine._replay_ultimo_idx = {sym: n_warmup - 1}
        engine._sesh._cursor[sym] = n_warmup

        # Drive drains until the whole live region is consumed.
        for _ in range(len(ticks) * 4):
            engine._drain_pending_ticks(sym)
            if engine._replay_ultimo_idx[sym] >= end:
                break

        expected = float(ticks[n_warmup:, _TICK_COL["volume"]].sum())
        assert engine.strategy.total_volume == pytest.approx(expected)
        assert engine.strategy.n_ticks == (end - n_warmup + 1)

    def test_concurrent_threads_preserve_volume(self, ticks):
        """
        Stress guard: a real background stepper advances the cursor while the
        engine thread drains. With the fix no tick is ever skipped regardless of
        interleaving, so the replayed volume equals the backtest volume.
        """
        engine = _build_replay(ticks)
        sym = "BTCUSDT"
        n_warmup = engine._sesh._warmup_ticks[sym]
        end = len(ticks) - 1

        engine._replay_ultimo_ts = {sym: -1}
        engine._replay_ultimo_idx = {sym: n_warmup - 1}
        engine._sesh._cursor[sym] = n_warmup - 1

        stop = threading.Event()

        def stepper():
            # Mimic the controller loop + Step button hammering step().
            while not stop.is_set():
                engine._sesh.step(sym)

        t = threading.Thread(target=stepper, daemon=True)
        t.start()
        try:
            # Engine-thread drain until the cursor reaches the end.
            for _ in range(len(ticks) * 50):
                engine._drain_pending_ticks(sym)
                if engine._replay_ultimo_idx[sym] >= end:
                    break
        finally:
            stop.set()
            t.join(timeout=2.0)

        # Final catch-up drain in case the last step landed after the last drain.
        engine._drain_pending_ticks(sym)

        expected = float(ticks[n_warmup:, _TICK_COL["volume"]].sum())
        assert engine.strategy.total_volume == pytest.approx(expected)
        assert engine.strategy.n_ticks == (end - n_warmup + 1)

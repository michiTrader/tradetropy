"""
Isolated micro-benchmark for the klines run() loop (perf/run-loop).

Measures ONLY tradetropy's single-backtest run() over an SMA-crossover strategy,
for a matrix of (bars x indicator-count) scenarios, and prints:

  - the median wall-clock seconds per scenario (timing), and
  - a deterministic result FINGERPRINT per scenario (parity).

The fingerprint (final equity, #trades, sum of trade PnL, first/last equity)
lets each optimization in this branch prove byte-for-byte identical results
before/after the change - the hard parity constraint. Run it before and after
a change and diff the fingerprints; they must match exactly.

    uv run python benchmarks/bench_run_loop.py
    uv run python benchmarks/bench_run_loop.py --bars 60000 100000 --repeat 3

The strategy uses N SMAs on close (N configurable) so the 2- and 8-indicator
scenarios exercise the indicator read hot path at different widths. Only the
first two SMAs drive the signal, so the decision stream (and thus the trade
book / equity curve) is identical regardless of N - the extra indicators just
add read/advance cost, never change results. That keeps the fingerprint stable
across --indicators while still stressing the indicator machinery.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from tradetropy import BacktestEngine, Strategy
from tradetropy.core.data_types import KlineData
from tradetropy.ta import SMA

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
INTERVAL_MS = 60_000

# SMA lengths; the first two drive the signal, the rest are pure read/advance
# load. 8 distinct lengths so the 8-indicator scenario has no dedup collapse.
_SMA_LENGTHS = [10, 30, 15, 20, 25, 40, 50, 60]


def build_candle_matrix(bars: int) -> np.ndarray:
    """Tile the bundled BTCUSDT 1m sample to ``bars`` rows, monotonic ts."""
    from tradetropy.datasets import load_btcusd_1m

    base = load_btcusd_1m().data
    reps = (bars // len(base)) + 1
    mat = np.tile(base, (reps, 1))[:bars].copy()
    start = float(base[0, 0])
    mat[:, 0] = start + np.arange(bars, dtype=np.float64) * INTERVAL_MS
    return mat


def make_strategy_cls(n_indicators: int):
    """Build an SmaCross strategy class carrying ``n_indicators`` SMAs."""
    lengths = _SMA_LENGTHS[:n_indicators]

    class SmaCrossN(Strategy):
        def init(self):
            self.px = self.subscribe_ohlc(SYMBOL, TIMEFRAME, window_size=200)
            self._mas = [self.add_indicator(self.px.close, SMA(L)) for L in lengths]
            self.fast = self._mas[0]
            self.slow = self._mas[1]

        def on_data(self):
            if self.fast[-1] > self.slow[-1]:
                if not self.sesh.positions(SYMBOL):
                    self.sesh.buy(SYMBOL, volume=1)
            else:
                for pos in self.sesh.positions(SYMBOL):
                    self.sesh.position_close(pos.ticket)

    return SmaCrossN


def make_klines(matrix: np.ndarray) -> KlineData:
    return KlineData(
        symbol=SYMBOL, data=matrix, timeframe=TIMEFRAME, tick_size=0.1, digits=1,
    )


def run_once(matrix: np.ndarray, n_indicators: int):
    """Run one backtest; return (engine, elapsed_seconds)."""
    cls = make_strategy_cls(n_indicators)
    engine = BacktestEngine.by_klines(cls(), data=(make_klines(matrix),))
    # verbose=True routes the loop through range() instead of tqdm, so the
    # progress-bar refresh cost never skews the measured loop time.
    t0 = time.perf_counter()
    engine.run(verbose=True)
    dt = time.perf_counter() - t0
    return engine, dt


def fingerprint(engine) -> str:
    """Deterministic result signature (equity + trades) for parity checks."""
    b = engine.broker
    eq = b._eq_vals
    trades = b.get_trades()
    pnl = sum(t.pnl_net for t in trades)
    n_eq = len(eq)
    first_eq = eq[0] if eq else float("nan")
    last_eq = eq[-1] if eq else float("nan")
    return (
        f"n_eq={n_eq} first_eq={first_eq:.6f} last_eq={last_eq:.6f} "
        f"n_trades={len(trades)} pnl={pnl:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, nargs="+", default=[60_000, 100_000, 200_000])
    parser.add_argument("--indicators", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print(f"{'bars':>9}{'#ind':>6}{'median_s':>12}{'  fingerprint'}")
    print("-" * 90)
    for bars in args.bars:
        matrix = build_candle_matrix(bars)
        for n in args.indicators:
            samples = []
            fp = None
            for _ in range(args.repeat):
                engine, dt = run_once(matrix, n)
                samples.append(dt)
                fp = fingerprint(engine)
            med = statistics.median(samples)
            print(f"{bars:>9}{n:>6}{med:>12.4f}  {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

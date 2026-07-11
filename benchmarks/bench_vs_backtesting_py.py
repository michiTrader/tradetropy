"""
Reproducible benchmark: tradetropy vs backtesting.py.

Runs the SAME SMA-crossover strategy, over the SAME candle series, in two
frameworks and reports how many times faster tradetropy is - for a single
backtest and for a parameter optimization (grid search).

The point is a fair, out-of-the-box comparison anyone can reproduce:

    pip install tradetropy[bench]
    python benchmarks/bench_vs_backtesting_py.py

Notes on fairness (see docs/guide/performance.md for the full methodology):

- Identical data. A single candle matrix is built once (the bundled BTCUSDT
  1-minute sample tiled up to ``--bars`` rows with strictly increasing
  timestamps) and fed to BOTH engines, so neither sees different input.
- Identical signal. Both strategies go long while the fast SMA is above the
  slow SMA and flat otherwise, evaluated once per bar - the same decision on
  every candle.
- Out-of-the-box settings. Each framework runs ``run()`` / ``optimize()`` with
  its own defaults (no commission, grid search). tradetropy's optimizer uses a
  process pool by default; backtesting.py's grid does too. That is a real
  property of each tool, reported as measured, not tuned away.
- Median of repeats. Each measurement is repeated (``--repeat``) and the median
  wall-clock time is reported to damp noise.

Timing numbers depend on the machine, OS, Python and library versions - the
script prints all of them so a result is always reproducible in context.

The strategy classes are defined at module level (not inside functions) so they
pickle cleanly for the process-based optimizers on Windows (spawn).
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

from tradetropy import BacktestEngine, Strategy
from tradetropy.core.data_types import KlineData
from tradetropy.ta import SMA

try:
    from backtesting import Backtest
    from backtesting import Strategy as BtStrategy

    _HAS_BACKTESTING = True
except ImportError:  # pragma: no cover - only when the extra is absent
    _HAS_BACKTESTING = False
    BtStrategy = object  # placeholder so the class body below still parses

# ----------------------------------------------------------------------------
# Shared configuration
# ----------------------------------------------------------------------------

# Grid explored by both optimizers (identical size and constraint on both).
FAST_GRID = [5, 10, 15, 20, 25]
SLOW_GRID = [30, 40, 50, 60]
# 5 x 4 = 20 combinations; every fast < every slow, so no combo is filtered.

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
INTERVAL_MS = 60_000


def build_candle_matrix(bars: int) -> np.ndarray:
    """
    Build a candle matrix of ``bars`` rows from the bundled BTCUSDT sample.

    The 500-row sample is tiled until it reaches ``bars`` rows and the
    timestamp column is rewritten as a strictly increasing 1-minute grid, so
    both engines receive a clean, monotonic OHLCV series. Prices are the tiled
    sample (their absolute path is irrelevant to a timing benchmark; what
    matters is that both engines see the exact same numbers).

    Args:
        bars (int): Number of candles to produce.

    Returns:
        np.ndarray: [bars x 7] matrix (ts, open, high, low, close, volume,
            turnover).
    """
    from tradetropy.datasets import load_btcusd_1m

    base = load_btcusd_1m().data
    reps = (bars // len(base)) + 1
    mat = np.tile(base, (reps, 1))[:bars].copy()
    start = float(base[0, 0])
    mat[:, 0] = start + np.arange(bars, dtype=np.float64) * INTERVAL_MS
    return mat


# ----------------------------------------------------------------------------
# tradetropy strategy (module level -> picklable for the process pool)
# ----------------------------------------------------------------------------

class SmaCross(Strategy):
    """Long while the fast SMA is above the slow SMA, flat otherwise."""

    fast = 10
    slow = 30

    def init(self):
        self.px = self.subscribe_ohlc(
            SYMBOL, TIMEFRAME, window_size=max(self.slow * 3, 200)
        )
        self.ma_fast = self.add_indicator(self.px.close, SMA(self.fast))
        self.ma_slow = self.add_indicator(self.px.close, SMA(self.slow))

    def on_data(self):
        if self.ma_fast[-1] > self.ma_slow[-1]:
            if not self.sesh.positions(SYMBOL):
                self.sesh.buy(SYMBOL, volume=1)
        else:
            for pos in self.sesh.positions(SYMBOL):
                self.sesh.position_close(pos.ticket)


def make_tradetropy_klines(matrix: np.ndarray) -> KlineData:
    """Wrap the shared matrix in a KlineData for tradetropy."""
    return KlineData(
        symbol=SYMBOL, data=matrix, timeframe=TIMEFRAME,
        tick_size=0.1, digits=1,
    )


def tradetropy_single(matrix: np.ndarray) -> Callable[[], None]:
    """Build a zero-arg callable running one tradetropy backtest."""

    def _run() -> None:
        engine = BacktestEngine.by_klines(
            SmaCross(), data=(make_tradetropy_klines(matrix),)
        )
        engine.run()

    return _run


def tradetropy_optimize(matrix: np.ndarray) -> Callable[[], None]:
    """Build a zero-arg callable running one tradetropy grid optimization."""

    def _run() -> None:
        engine = BacktestEngine.by_klines(
            SmaCross(), data=(make_tradetropy_klines(matrix),)
        )
        engine.optimize(
            maximize="Return [%]",
            method="grid",
            fast=FAST_GRID,
            slow=SLOW_GRID,
        )

    return _run


# ----------------------------------------------------------------------------
# backtesting.py strategy (module level -> picklable for its optimizer)
# ----------------------------------------------------------------------------

def _sma(values, n):
    """Simple moving average helper for backtesting.py's self.I()."""
    return pd.Series(values).rolling(n).mean().to_numpy()


class BtSmaCross(BtStrategy):
    """Same signal as SmaCross, expressed for backtesting.py."""

    fast = 10
    slow = 30

    def init(self):
        close = self.data.Close
        self.ma_fast = self.I(_sma, close, self.fast)
        self.ma_slow = self.I(_sma, close, self.slow)

    def next(self):
        if self.ma_fast[-1] > self.ma_slow[-1]:
            if not self.position:
                self.buy()
        elif self.position:
            self.position.close()


def make_backtesting_df(matrix: np.ndarray) -> pd.DataFrame:
    """Build the OHLCV DataFrame backtesting.py expects (DatetimeIndex)."""
    idx = pd.to_datetime(matrix[:, 0], unit="ms", utc=True)
    return pd.DataFrame(
        {
            "Open": matrix[:, 1],
            "High": matrix[:, 2],
            "Low": matrix[:, 3],
            "Close": matrix[:, 4],
            "Volume": matrix[:, 5],
        },
        index=idx,
    )


def backtesting_single(matrix: np.ndarray) -> Callable[[], None]:
    """Build a zero-arg callable running one backtesting.py backtest."""
    df = make_backtesting_df(matrix)

    def _run() -> None:
        bt = Backtest(df, BtSmaCross, cash=1_000_000, commission=0.0)
        bt.run()

    return _run


def backtesting_optimize(matrix: np.ndarray) -> Callable[[], None]:
    """Build a zero-arg callable running one backtesting.py grid optimization."""
    df = make_backtesting_df(matrix)

    def _run() -> None:
        bt = Backtest(df, BtSmaCross, cash=1_000_000, commission=0.0)
        bt.optimize(
            fast=FAST_GRID,
            slow=SLOW_GRID,
            maximize="Return [%]",
            constraint=lambda p: p.fast < p.slow,
        )

    return _run


# ----------------------------------------------------------------------------
# Timing harness
# ----------------------------------------------------------------------------

def time_median(fn: Callable[[], None], repeat: int) -> float:
    """Run ``fn`` ``repeat`` times and return the median wall-clock seconds."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _version(mod_name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(mod_name)
    except Exception:
        return "unknown"


def print_environment(bars: int, repeat: int) -> None:
    """Print the machine / version context so results are reproducible."""
    import multiprocessing as mp

    print("Environment")
    print("-----------")
    print(f"  platform     : {platform.platform()}")
    print(f"  processor    : {platform.processor() or 'unknown'}")
    print(f"  cpu count    : {mp.cpu_count()}")
    print(f"  python       : {platform.python_version()}")
    print(f"  numpy        : {_version('numpy')}")
    print(f"  pandas       : {_version('pandas')}")
    print(f"  tradetropy      : {_version('tradetropy')}")
    print(f"  backtesting  : {_version('backtesting')}")
    print(f"  bars         : {bars}")
    print(f"  repeats      : {repeat}")
    print(f"  optimize grid: {len(FAST_GRID) * len(SLOW_GRID)} combinations")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bars", type=int, default=60_000,
        help="Number of candles in the shared series (default 60000).",
    )
    parser.add_argument(
        "--repeat", type=int, default=5,
        help="Repeats per measurement; the median is reported (default 5).",
    )
    args = parser.parse_args()

    if not _HAS_BACKTESTING:
        print(
            "backtesting.py is not installed. Install the benchmark extra:\n"
            "    pip install tradetropy[bench]\n"
            "or:  pip install backtesting",
            file=sys.stderr,
        )
        return 1

    print_environment(args.bars, args.repeat)

    matrix = build_candle_matrix(args.bars)

    print("Running (this can take a minute)...\n")

    tr_run = time_median(tradetropy_single(matrix), args.repeat)
    bt_run = time_median(backtesting_single(matrix), args.repeat)

    tr_opt = time_median(tradetropy_optimize(matrix), args.repeat)
    bt_opt = time_median(backtesting_optimize(matrix), args.repeat)

    run_speedup = bt_run / tr_run if tr_run else float("nan")
    opt_speedup = bt_opt / tr_opt if tr_opt else float("nan")

    print("Results (median wall-clock seconds)")
    print("-----------------------------------")
    print(f"{'Scenario':<22}{'tradetropy':>12}{'backtesting.py':>18}{'speedup':>12}")
    print(f"{'-' * 64}")
    print(
        f"{'single run()':<22}{tr_run:>12.4f}{bt_run:>18.4f}"
        f"{run_speedup:>11.1f}x"
    )
    print(
        f"{'optimize() grid':<22}{tr_opt:>12.4f}{bt_opt:>18.4f}"
        f"{opt_speedup:>11.1f}x"
    )
    print()
    print(
        f"tradetropy is {run_speedup:.1f}x faster on a single backtest and "
        f"{opt_speedup:.1f}x faster on the grid optimization "
        f"({args.bars} bars, median of {args.repeat})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

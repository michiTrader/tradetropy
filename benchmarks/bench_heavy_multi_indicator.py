"""
Heavy benchmark: tradetropy vs backtesting.py with a large dataset, multiple
indicators (including deliberate duplicates) and a wider optimize() grid.

This is a harder scenario than ``bench_vs_backtesting_py.py``: instead of one
SMA-crossover signal, both strategies carry SIX indicators per bar (SMA, EMA,
RSI, Bollinger Bands, MACD, ATR), and two of them are registered TWICE with
the same period (an explicit duplicate) to stress indicator-cache /
recomputation paths under repeated columns. The candle matrix also injects
duplicated timestamps (a burst of repeated ``ts`` rows at a fixed cadence) on
top of the large row count, so the engines see the same "messy" large dataset.

Usage:

    pip install tradetropy[bench]
    python benchmarks/bench_heavy_multi_indicator.py
    python benchmarks/bench_heavy_multi_indicator.py --bars 200000 --repeat 3

Notes on fairness (same methodology as ``bench_vs_backtesting_py.py``):

- Identical data. One candle matrix is built once (bundled BTCUSDT 1-minute
  sample tiled to ``--bars`` rows, with a configurable fraction of duplicated
  timestamps re-inserted) and fed to BOTH engines unchanged.
- Identical signal. Both strategies open long while fast SMA > slow SMA, flat
  otherwise - and ONLY the SMA cross drives the decision. The SMA is computed
  identically in both libraries, so the decision stream matches bar-for-bar and
  the returns are comparable. The other indicators (EMA, RSI, Bollinger Bands,
  MACD, ATR and the two duplicated bands) are computed every bar as pure LOAD
  but never gate the signal - a recursive indicator like RSI/EMA has a slightly
  different formula in each library, so gating on it would desync the decision
  stream and make returns incomparable.
- Out-of-the-box settings. ``run()`` / ``optimize()`` use each library's own
  defaults; tradetropy's optimizer uses its process pool by default, same as
  backtesting.py's grid.
- Identical trading conditions so the Return [%] is comparable, not just the
  timing: same initial capital, same FIXED position size (1 unit / 1 BTC per
  trade -> no compounding on either side), tradetropy ``tick_value == tick_size``
  and ``contract_size == 1`` so its PnL equals the pure price difference, no
  commission, no spread, market fill at the NEXT bar open on both. Without this
  alignment tradetropy's $10k default account + fixed size vs backtesting.py's $1M
  all-in (compounding) sizing make the returns look wildly different (e.g. 16%
  vs 100%) even though the per-bar decision stream is identical.
- Median of repeats. Each measurement is repeated (``--repeat``); the median
  wall-clock time is reported.

The optimize() grid explores the two SMA lengths (fast/slow) that drive the
decision, so it also reports the actual best parameters and top-N table found
by each optimizer, not just timing. Duplicate-timestamp rows are a known
divergence point: backtesting.py collapses duplicate DatetimeIndex rows while
tradetropy processes every bar, so with ``--dup-fraction > 0`` the returns will
differ (run with ``--dup-fraction 0`` for a byte-for-byte return comparison).
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
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.ta import SMA, EMA, RSI, BollingerBands, MACD, ATR

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

# Optimize grid over the two SMA lengths that actually drive the decision
# (kept identical to backtesting.py so returns are comparable). every fast <
# every slow, so the fast < slow constraint never empties the grid.
FAST_GRID = [5, 10, 15, 20, 25]
SLOW_GRID = [30, 40, 50, 60]
# 5 x 4 = 20 combinations.

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
INTERVAL_MS = 60_000

# Aligned trading conditions so BOTH frameworks price trades identically and
# their Return [%] is comparable (see docstring). Without this, tradetropy's
# default $10k account + fixed 1-unit size and backtesting.py's $1M all-in
# (compounding) sizing make the returns look wildly different even though the
# decision stream is the same.
#   - Same initial capital.
#   - Same FIXED position size (1 unit / 1 BTC per trade) -> no compounding on
#     either side.
#   - tradetropy's tick_value now defaults to tick_size (and contract_size to 1),
#     so its PnL per unit equals the pure price difference out of the box,
#     matching backtesting.py's size=1 unit PnL - no explicit tick_value.
#   - finalize_trades=True on both, so an open position at the end is closed
#     into the stats identically.
#   - No commission, no spread, market fill at NEXT bar open on both.
INITIAL_CASH = 1_000_000
FIXED_SIZE = 1
TICK_SIZE = 0.1

# Fraction of rows that are duplicated timestamps (same ts as the row before),
# injected at a fixed cadence to keep the series reproducible.
DUP_FRACTION = 0.05


def build_candle_matrix(bars: int, dup_fraction: float = DUP_FRACTION) -> np.ndarray:
    """
    Build a large candle matrix with a fraction of duplicated timestamps.

    Starts from the bundled BTCUSDT sample tiled to ``bars`` rows on a clean
    1-minute grid (like ``bench_vs_backtesting_py.py``), then overwrites every
    ``1/dup_fraction``-th row's timestamp with the PREVIOUS row's timestamp,
    so both engines must deal with the same duplicated-ts rows without either
    engine seeing a different length or set of duplicates.

    Args:
        bars (int): Number of candles to produce.
        dup_fraction (float): Fraction of rows (after the first) whose
            timestamp is duplicated from the previous row.

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

    if dup_fraction > 0:
        step = max(int(round(1.0 / dup_fraction)), 2)
        dup_idx = np.arange(step, bars, step)
        mat[dup_idx, 0] = mat[dup_idx - 1, 0]
        # keep ts non-decreasing after the duplication (ties allowed).
        for i in range(1, bars):
            if mat[i, 0] < mat[i - 1, 0]:
                mat[i, 0] = mat[i - 1, 0]

    return mat


# ----------------------------------------------------------------------------
# tradetropy strategy (module level -> picklable for the process pool)
# ----------------------------------------------------------------------------

class HeavySmaCross(Strategy):
    """
    Eight indicator bands per bar (SMA, EMA, RSI, Bollinger Bands x3, MACD x2,
    ATR), two of them registered twice with the same period (deliberate
    duplicates).

    IMPORTANT (comparability): only the fast/slow SMA cross drives the trading
    decision. SMA is computed IDENTICALLY in tradetropy and backtesting.py, so
    both frameworks take the same decision on every bar and their Return [%]
    is directly comparable. The other indicators (EMA, RSI, BB, MACD, ATR and
    the two duplicates) are computed every bar as pure read/advance LOAD but
    are never read in on_data() - a recursive indicator like RSI or EMA has a
    slightly different formula in each library, so gating the signal on it
    would desync the decision stream and make returns incomparable. Keeping
    them as load-only stresses the indicator hot path without breaking parity.
    """

    fast = 10
    slow = 30
    rsi_length = 14

    def init(self):
        self.px = self.subscribe_ohlc(
            SYMBOL, TIMEFRAME, window_size=max(self.slow * 3, 200)
        )
        self.ma_fast = self.add_indicator(self.px.close, SMA(self.fast))
        self.ma_slow = self.add_indicator(self.px.close, SMA(self.slow))
        self.rsi = self.add_indicator(self.px.close, RSI(self.rsi_length))

        # Extra load: EMA, Bollinger Bands, MACD, ATR - computed every bar,
        # never read by on_data().
        self.ema = self.add_indicator(self.px.close, EMA(self.fast))
        self.bb = self.add_indicator(self.px.close, BollingerBands(20, 2.0))
        self.macd = self.add_indicator(self.px.close, MACD())
        self.atr = self.add_indicator(
            [self.px.high, self.px.low, self.px.close], ATR(14)
        )

        # Deliberate duplicates: same indicator, same period, registered
        # again under a different name (stresses per-column recomputation
        # instead of relying on any cache dedup).
        self.ma_fast_dup = self.add_indicator(self.px.close, SMA(self.fast))
        self.rsi_dup = self.add_indicator(self.px.close, RSI(self.rsi_length))

    def on_data(self):
        long_signal = self.ma_fast[-1] > self.ma_slow[-1]
        if long_signal:
            if not self.sesh.positions(SYMBOL):
                self.sesh.buy(SYMBOL, volume=FIXED_SIZE)
        else:
            for pos in self.sesh.positions(SYMBOL):
                self.sesh.position_close(pos.ticket)


def make_aligned_sesh() -> SeshSimulatorBase:
    """Simulated session matching backtesting.py's conditions (see INITIAL_CASH)."""
    return SeshSimulatorBase(
        initial_balance=INITIAL_CASH,
        commission=0.0,
        use_spread=False,
        trade_on_close=False,
        finalize_trades=True,
    )


def make_tradetropy_klines(matrix: np.ndarray) -> KlineData:
    """
    Wrap the shared matrix in a KlineData for tradetropy.

    tick_value now defaults to tick_size (and contract_size to 1), so the PnL
    per unit equals the pure price difference out of the box - no need to set
    tick_value explicitly to match backtesting.py's unit model.
    """
    return KlineData(
        symbol=SYMBOL, data=matrix, timeframe=TIMEFRAME,
        tick_size=TICK_SIZE, digits=1,
    )


def tradetropy_single(matrix: np.ndarray) -> Callable[[], None]:
    """Build a zero-arg callable running one tradetropy backtest."""

    def _run() -> None:
        engine = BacktestEngine.by_klines(
            HeavySmaCross(), data=(make_tradetropy_klines(matrix),),
            sesh=make_aligned_sesh(),
        )
        engine.run()

    return _run


def tradetropy_optimize(matrix: np.ndarray):
    """Build a zero-arg callable running one tradetropy grid optimization."""

    def _run():
        engine = BacktestEngine.by_klines(
            HeavySmaCross(), data=(make_tradetropy_klines(matrix),),
            sesh=make_aligned_sesh(),
        )
        return engine.optimize(
            maximize="Return [%]",
            method="grid",
            fast=FAST_GRID,
            slow=SLOW_GRID,
            constraints=lambda p: p["fast"] < p["slow"],
        )

    return _run


# ----------------------------------------------------------------------------
# backtesting.py strategy (module level -> picklable for its optimizer)
# ----------------------------------------------------------------------------

def _sma(values, n):
    return pd.Series(values).rolling(n).mean().to_numpy()


def _ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy()


def _rsi(values, n):
    s = pd.Series(values)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).to_numpy()


def _bb_upper(values, n, k):
    s = pd.Series(values)
    mid = s.rolling(n).mean()
    std = s.rolling(n).std(ddof=0)
    return (mid + k * std).to_numpy()


def _bb_mid(values, n, k):
    return pd.Series(values).rolling(n).mean().to_numpy()


def _bb_lower(values, n, k):
    s = pd.Series(values)
    mid = s.rolling(n).mean()
    std = s.rolling(n).std(ddof=0)
    return (mid - k * std).to_numpy()


def _macd_line(values, fast, slow, signal):
    s = pd.Series(values)
    macd = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return macd.to_numpy()


def _macd_signal(values, fast, slow, signal):
    s = pd.Series(values)
    macd = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return macd.ewm(span=signal, adjust=False).mean().to_numpy()


def _atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    prev_close = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


class BtHeavySmaCross(BtStrategy):
    """Same signal + same extra indicator load as HeavySmaCross, for backtesting.py."""

    fast = 10
    slow = 30
    rsi_length = 14

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.ma_fast = self.I(_sma, close, self.fast)
        self.ma_slow = self.I(_sma, close, self.slow)
        self.rsi = self.I(_rsi, close, self.rsi_length)

        self.ema = self.I(_ema, close, self.fast)
        self.bb_upper = self.I(_bb_upper, close, 20, 2.0)
        self.bb_mid = self.I(_bb_mid, close, 20, 2.0)
        self.bb_lower = self.I(_bb_lower, close, 20, 2.0)
        self.macd = self.I(_macd_line, close, 12, 26, 9)
        self.macd_signal = self.I(_macd_signal, close, 12, 26, 9)
        self.atr = self.I(_atr, high, low, close, 14)

        # Deliberate duplicates, same as the tradetropy side.
        self.ma_fast_dup = self.I(_sma, close, self.fast)
        self.rsi_dup = self.I(_rsi, close, self.rsi_length)

    def next(self):
        long_signal = self.ma_fast[-1] > self.ma_slow[-1]
        if long_signal:
            if not self.position:
                self.buy(size=FIXED_SIZE)  # fixed 1 unit, NOT all-in
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
    df = make_backtesting_df(matrix)

    def _run() -> None:
        bt = Backtest(
            df, BtHeavySmaCross, cash=INITIAL_CASH, commission=0.0,
            trade_on_close=False, finalize_trades=True,
        )
        bt.run()

    return _run


def backtesting_optimize(matrix: np.ndarray):
    df = make_backtesting_df(matrix)

    def _run():
        bt = Backtest(
            df, BtHeavySmaCross, cash=INITIAL_CASH, commission=0.0,
            trade_on_close=False, finalize_trades=True,
        )
        return bt.optimize(
            fast=FAST_GRID,
            slow=SLOW_GRID,
            maximize="Return [%]",
            constraint=lambda p: p.fast < p.slow,
        )

    return _run


# ----------------------------------------------------------------------------
# Timing + result harness
# ----------------------------------------------------------------------------

def time_median(fn: Callable[[], None], repeat: int) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_median_with_last_result(fn, repeat: int):
    """Like time_median but also returns the LAST call's return value."""
    samples = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result


def _version(mod_name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(mod_name)
    except Exception:
        return "unknown"


def print_environment(bars: int, repeat: int, dup_fraction: float) -> None:
    import multiprocessing as mp

    n_grid = len(FAST_GRID) * len(SLOW_GRID)
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
    print(f"  dup ts frac  : {dup_fraction:.2%}")
    print(f"  indicators   : 8 per bar (SMA, EMA, RSI, BB x3, MACD x2, ATR;"
          f" +2 deliberate duplicates)")
    print(f"  repeats      : {repeat}")
    print(f"  optimize grid: {n_grid} combinations (fast x slow)")
    print()


def print_optimize_results_tradetropy(result, top_n: int) -> None:
    print("tradetropy optimize() results")
    print("---------------------------")
    print(f"  best params : {result.best_params}")
    print(f"  best fitness: {result.best_fitness:.6f}")
    df = result.to_dataframe().sort_values("fitness", ascending=False).head(top_n)
    cols = [c for c in df.columns if c.startswith("param_")] + ["fitness"]
    with pd.option_context("display.width", 120):
        print(df[cols].to_string(index=False))
    print()


def print_optimize_results_backtesting(stats, top_n: int) -> None:
    print("backtesting.py optimize() results")
    print("-----------------------------------")
    best_params = {
        "fast": stats._strategy.fast,
        "slow": stats._strategy.slow,
    }
    print(f"  best params : {best_params}")
    print(f"  best fitness (Return [%]): {stats['Return [%]']:.6f}")
    heatmap = getattr(stats, "_heatmap", None)
    if heatmap is not None:
        top = heatmap.sort_values(ascending=False).head(top_n)
        with pd.option_context("display.width", 120):
            print(top.to_string())
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bars", type=int, default=100_000,
        help="Number of candles in the shared series (default 100000).",
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="Repeats per timing measurement; the median is reported (default 3).",
    )
    parser.add_argument(
        "--dup-fraction", type=float, default=DUP_FRACTION,
        help="Fraction of rows with a duplicated timestamp (default 0.05).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Rows to print from each optimizer's result table (default 10).",
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

    print_environment(args.bars, args.repeat, args.dup_fraction)

    matrix = build_candle_matrix(args.bars, args.dup_fraction)
    n_dup = int(np.sum(np.diff(matrix[:, 0]) == 0))
    print(f"Duplicated-ts rows actually injected: {n_dup} / {args.bars}\n")

    print("Running single backtest() x2 frameworks...\n")
    tr_run = time_median(tradetropy_single(matrix), args.repeat)
    bt_run = time_median(backtesting_single(matrix), args.repeat)

    print("Running optimize() grid x2 frameworks (this can take a while)...\n")
    tr_opt_time, tr_opt_result = time_median_with_last_result(
        tradetropy_optimize(matrix), args.repeat
    )
    bt_opt_time, bt_opt_stats = time_median_with_last_result(
        backtesting_optimize(matrix), args.repeat
    )

    run_speedup = bt_run / tr_run if tr_run else float("nan")
    opt_speedup = bt_opt_time / tr_opt_time if tr_opt_time else float("nan")

    print("Timing results (median wall-clock seconds)")
    print("-------------------------------------------")
    print(f"{'Scenario':<22}{'tradetropy':>12}{'backtesting.py':>18}{'speedup':>12}")
    print(f"{'-' * 64}")
    print(
        f"{'single run()':<22}{tr_run:>12.4f}{bt_run:>18.4f}"
        f"{run_speedup:>11.1f}x"
    )
    print(
        f"{'optimize() grid':<22}{tr_opt_time:>12.4f}{bt_opt_time:>18.4f}"
        f"{opt_speedup:>11.1f}x"
    )
    print()

    print_optimize_results_tradetropy(tr_opt_result, args.top_n)
    print_optimize_results_backtesting(bt_opt_stats, args.top_n)

    print(
        f"tradetropy is {run_speedup:.1f}x faster on a single heavy backtest and "
        f"{opt_speedup:.1f}x faster on the {len(FAST_GRID) * len(SLOW_GRID)}-combination "
        f"optimization ({args.bars} bars, {args.dup_fraction:.0%} duplicated ts, "
        f"median of {args.repeat})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

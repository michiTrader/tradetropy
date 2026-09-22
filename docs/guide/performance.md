
## Performance

Tradetropy runs one event-driven engine for backtest, tick, live and replay so a
strategy behaves identically everywhere. That single loop is pure Python and
NumPy - no Cython, C extensions or compiler - so it installs as prebuilt wheels
on every desktop platform and builds on Termux. This page documents how fast the
engine is, how to reproduce the numbers, and what makes it fast.

## Reproducible benchmarks

Two benchmark scripts ship in `benchmarks/` (install the extra with
`pip install tradetropy[bench]`, which adds `backtesting.py` for the comparison):

- `benchmarks/bench_vs_backtesting_py.py` - runs the SAME SMA-crossover strategy
  over the SAME candle series in tradetropy and in `backtesting.py`, for a single
  `run()` and for a grid `optimize()`, and reports how many times faster tradetropy
  is. It prints the full machine / version context so a result is always
  reproducible in place.
- `benchmarks/bench_run_loop.py` - times ONLY tradetropy's single `run()` over a
  `(bars x indicator-count)` matrix and prints a deterministic result
  fingerprint (final equity, trade count, summed PnL). The fingerprint is the
  parity oracle: any optimization must leave it byte-for-byte identical.

```bash
uv run python benchmarks/bench_vs_backtesting_py.py --bars 60000 --repeat 5
uv run python benchmarks/bench_run_loop.py --bars 60000 100000 200000
```

### Methodology

The comparison is deliberately out-of-the-box and fair:

- Identical data. One candle matrix is built once (the bundled BTCUSDT 1-minute
  sample tiled to `--bars` rows with strictly increasing timestamps) and fed to
  both engines.
- Identical signal. Both strategies go long while the fast SMA is above the slow
  SMA and flat otherwise, evaluated once per bar.
- Default settings. Each framework runs with its own defaults (no commission,
  grid search, process-pool optimizer). That is reported as measured, not tuned.
- Median of repeats. Each measurement is repeated and the median wall-clock time
  is reported to damp noise.

Timing depends on the machine, OS, Python and library versions, so treat the
absolute seconds below as illustrative and the RATIOS as the takeaway.

## Results

Reference machine: Windows, CPython 3.12, NumPy 2.4, pandas 3.0, 60000 candles.
The run() figures below come from an interleaved A/B (tradetropy and
`backtesting.py` timed alternately) so they are robust to background load drift.

| Scenario          | tradetropy   | backtesting.py | tradetropy vs backtesting.py |
| ----------------- | --------- | -------------- | ------------------------- |
| single `run()`    | ~0.98 s   | ~0.80 s        | 1.23x slower              |
| `optimize()` grid | ~11.7 s   | ~19.8 s        | 1.7x faster               |

The optimizer wins and grows with scale (a process pool, pandas-free worker
stats and `shared_memory` input transfer). A single candle `run()` is the one
scenario where the event-driven engine is behind a vectorized one - that is
structural: the same loop that gives backtest <-> tick <-> live <-> replay
parity pays a per-bar interpreter cost a fully vectorized backtester avoids.

### Narrowing the run() gap

A round of run-loop optimizations (all pure Python, all parity-preserving)
narrowed that single-`run()` gap without changing a single result:

| Stage                              | tradetropy `run()` 60k | vs backtesting.py |
| ---------------------------------- | ------------------- | ----------------- |
| Before                             | ~1.12 s             | 1.43x slower      |
| After the run-loop optimizations   | ~0.98 s             | 1.23x slower      |

About 13% off tradetropy's own candle `run()` on the reference machine, moving the
gap from ~1.43x to ~1.23x. The changes:

- Reuse the per-symbol `_current_prices` dict in `update_kline` instead of
  allocating two dicts per bar.
- Cache the `SymbolConfig` once per bar and short-circuit `_simulate_spread`
  when there is no spread.
- Single-frame scalar indicator read: the common `self.ind[-1]` / `self.ind[-k]`
  of a windowed OHLC indicator in kline mode resolves straight from the
  pre-calculated column, skipping the `IndicatorProxy -> view.__getitem__ ->
  _scalar` chain. It falls back to the full windowed view for anything it does
  not cover (recursive developing bar, out-of-window index, slice), so values
  are identical.
- Shared per-proxy cursor: every view of a proxy references one mutable cursor
  holder, so advancing a bar is a single write instead of one per view (an OHLC
  proxy has 6 price views, a tick proxy 7, a multi-band indicator K bands).

Every step keeps the `bench_run_loop.py` fingerprint byte-for-byte identical and
the parity suite green (`test_parity_engines`, `test_indicator_engine_parity`,
`test_kline_indicator_fastpath`, `test_bbroker`, `test_stats`), so live, tick and
replay are unaffected.

## Running under PyPy (optional, free speedup)

The candle loop is the kind of workload PyPy's tracing JIT accelerates best: a
tight per-bar Python loop dominated by attribute access, method dispatch and
small arithmetic (proxy advance, indicator reads, the broker's per-bar
bookkeeping). An audit of the hot path (`backtest/_runner.py`,
`core/broker.py`, `data/_proxy.py`, `data/_views.py`) found NO CPython-only
constructs - no C-API tricks, no `ctypes`/`cffi`, no reference-count
assumptions, only plain Python plus NumPy - so it runs unchanged on PyPy.

Notes and caveats:

- NumPy runs on PyPy, but small-array operations carry a CPython-C-API interop
  cost there, so the net gain depends on the mix of interpreter work (which PyPy
  speeds up a lot) versus tiny NumPy calls (which it may not). The interpreter-
  bound per-bar orchestration is exactly where the JIT pays off, so an
  interpreter-heavy `run()` typically sees a large speedup on PyPy once the JIT
  has warmed up over enough bars.
- This is a "free" win only where a PyPy build with a compatible NumPy is
  available for your platform; CPython remains the supported default (and the
  only option on Termux). These PyPy figures are not measured in CI, so benchmark
  your own workload with `benchmarks/bench_run_loop.py` before relying on them.

## Where the time goes

For a candle `run()` the per-bar cost is spread across the loop with no single
bottleneck: loop orchestration and proxy advance, indicator reads
(`self.ind[-1]`), the broker's `update_kline` (spread simulation, current-price
bookkeeping) and equity recording. Because it is spread out, the wins above are
several small, independent cuts rather than one big change - and each was kept
only after measuring a real improvement with the fingerprint held constant. The
guiding rule: no change lands unless it is measurably faster AND leaves the
result identical.

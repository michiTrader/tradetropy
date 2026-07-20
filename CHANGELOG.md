# Changelog

All notable changes to Tradetropy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Release model - sponsor goals

Tradetropy is released progressively through [GitHub Sponsors](https://github.com/sponsors/michiTrader)
goals. Each goal, once reached, unlocks the next block of features publicly for
everyone. Sponsors who want the complete framework right away can get immediate
access to the full private repository as a sponsorship reward.

- Goal 10 sponsors - Tier 1:
  - LiveChart (live/interactive plotting)
  - ReplayEngine
  - TickVolumeProfile
  - FibRetracement
  - Monte Carlo robustness testing

- Goal 30 sponsors - Tier 2:
  - LivePool (multiple live strategies)
  - Delta Bars and Cumulative Delta (CVD)
  - Data recording (L1 / L2 / L3)
  - Deep Trades (L2)

- Goal 50 sponsors - Tier 3:
  - Pivot Pattern DSL
  - Order Book Imbalance
  - Deep Reload (L2 liquidity replenishment)
  - Deep Wall (L2 liquidity walls)
  - Stop Run (L2 stop sweeps)
  - Heatmap (L2 liquidity heatmap)

The authoritative feature-to-module mapping used to generate each public tier
lives in the tier manifest (see the release tooling).

## [Unreleased]

## [0.2.5]

### Added
- Zoom-aware level-of-detail (LOD) for the candlestick/volume panel: a chart
  with more than 6000 candles keeps the full series in a backing source and
  draws a bucket-aggregated view (open/close/high/low/volume) capped at 4000
  candles, refined automatically as you zoom in. Bucket aggregation preserves
  each bucket's true high/low, so the Y-axis autoscale still frames the panel
  correctly while scanning far fewer rows - large backtests plot noticeably
  lighter. Disabled automatically when the footprint panel is enabled (already
  a zoomed-in detail view).
- `CandlePatterns`: statistical candlestick pattern detector (annotation) with
  adaptive percentile thresholds, an optional price-context z-score filter and
  causal efficacy tracking (`last_pattern()`, `pattern_at()`, `patterns()`,
  `is_bullish()`/`is_bearish()`, `efficacy()`/`efficacy_all()`), free (tier 0).
- `ManualMarks`: strategy-driven manual marks/levels (segments with optional
  markers and labels) added, updated and closed from `on_data()`, free (tier 0).
- `TickData.to_klines()` / `ticks_to_klines()`: new `price_source='trade'` drops
  quote-only ticks (`volume == 0`) before building OHLC, so candles always land
  on a real traded price instead of a `(bid+ask)/2` fallback.
- `get_strategy_logger(display_tz=)`: renders the strategy logger's `{asctime}`
  column in a configurable zone (default UTC), decoupled from the machine's
  local zone and from what instant is actually logged.
- `BacktestEngine.run(stats_warn=)`: optionally silences the "Stats:
  insufficient sample" `UserWarning` for quick exploratory runs on small
  datasets. The reliability gating itself (metrics zeroed to NaN,
  `stats["_low_sample"]`) always applies regardless of this flag.
- Generic `source_cols`-driven `refs()` / `default_refs()` on `Indicator`, so
  any indicator that only declares `source_cols` (the common case) resolves
  its columns automatically - no per-indicator `refs()` override needed.

### Changed
- Internal `tradetropy.io` module split into focused submodules
  (`_backends`, `_common`, `_ticks`, `_klines`, `_book`, `_mbo`, `_record`,
  `_compat`); the public `tradetropy.io` API is unchanged.
- `bt.plot()` defaults to the WebGL canvas backend for smoother pan/zoom/
  crosshair interaction on dense charts; `show_price_tag` now defaults to
  `False` (its mousemove handler repaints on every tick boundary crossed).
  Both remain configurable via `PlotConfig`.
- Geometric indicators (annotation `draw()` primitives) are recalculated over
  the full OHLC history before a static plot draws them, so historical
  annotations are not clipped by the last partial-bar recalculation.

### Fixed
- `TickData.to_klines()` docstring now documents the `'trade'` price_source
  (was missing after it was added).

## [0.2.4]

### Changed
- Regenerated the badge logo (`docs/assets/logo-badge.png`) at higher
  resolution: a larger icon and more right-side padding on the pill so the
  "Tradetropy" wordmark no longer crowds the rounded edge. The wordmark is
  reproduced pixel-faithfully from the original typeface (alpha matte),
  reproducible via the new `tools/gen_logo.py`. Bumped the README badge
  display height accordingly.
- Plotting: large tick-backtest equity and trailing-drawdown curves are now
  min/max decimated for drawing, and overlay indicators participate in the
  price-panel autoscale (with zoom-aware point-cloud decimation) - noticeably
  smoother pan/zoom, while stats and markers are still computed from the full
  data.

### Fixed
- `bt.plot()` no longer fails with an ImportError on the equity panel: the
  equity-curve decimation helper it relies on is now present.

## [0.2.3]

### Added
- Interactive documentation demo (SMA crossover, daily GOOG) embedded on the
  docs landing page and the Examples page: a single canonical snippet drives
  the shown source, its performance stats and the interactive chart, so all
  three can never drift from each other. Generated by
  `tools/gen_docs_demo.py`, wired into `tools/generate_tier.py` so every
  public build regenerates only the demos allowed for its tier.
  - A degraded demo (insufficient sample) fails the build rather than
    publishing zeroed/NaN metrics.
- README static chart preview (`docs/assets/demos/sma_cross/chart.png`)
  linking to the live interactive chart on the docs site.
- Backtesting.py-style P&L panel: a circle marker at each trade's exit (sized
  by `|pnl|`), an entry-to-exit segment, and a hover tooltip with direction,
  entry/exit price, P&L and signed size (long/short) - now consistent between
  backtest and live.
- New badge-style logo (icon + "Tradetropy" wordmark) replacing the old
  text-only wordmark; regenerated square logo and favicon from the same
  source icon.
- README badges: Python versions, downloads, license, and CI status (tests,
  docs) alongside the existing PyPI badge; docs landing page download
  buttons (Get started / PyPI / GitHub).
- `docs-export` optional dev extra (`selenium` + `pillow`) for regenerating
  the demo's static PNG; falls back to a Selenium-Manager-resolved Chrome
  driver when no `chromedriver`/`geckodriver` binary is on `PATH`.

### Fixed
- `bt.plot()` Y-axis autoscale on pan/zoom in backtest, live and replay - the
  OHLC panel no longer freezes its Y range after the first manual scroll/pan
  (root causes: a `DataRange1d` on X that did not emit reliable range-change
  events, and a race in the Y-axis lock/autoscale interaction). Added a
  width-only mouse wheel zoom so scrolling never fights the Y autoscale.

## [0.2.2]

### Added
- Horizontal wordmark logo (`docs/assets/logo-wordmark.png`) used in the
  README, with a color legible on both light and dark backgrounds.

### Fixed
- README logo now uses an absolute GitHub URL so it renders correctly on the
  PyPI project page too (a relative path only worked on GitHub).

## [0.2.1]

### Added
- `tradetropy.__version__`, resolved via `importlib.metadata` (falls back to
  `"0.0.0"` when running from a source checkout with no installed package
  metadata, so it never breaks the import).

## [0.2.0]

### Added
- License (MIT)
- CONTRIBUTING.md
- GitHub Sponsors goal-based release model

### Changed
- Renamed the project from `tradear` to `tradetropy` (package, metadata,
  docs, CI, environment variables `TRADETROPY_SUPPRESS_DISCLAIMER` /
  `TRADETROPY_DEBUG_WARMUP`).

## [0.1.1]

### Fixed
- Removed the unused `ipython` core dependency (no code in tradetropy imports
  `IPython`; the Jupyter `_repr_html_` protocol needs no such import). A
  recent `ipython` release added a hard `psutil` dependency, and `psutil` has
  no prebuilt wheels for Android/Termux, breaking `pip install tradetropy` there.
  Also drops ~6 unused transitive dependencies for everyone.

## [0.1.0] - free tier

Initial public release. Everything below is available in the free tier.

### Added
- BacktestEngine (tick and kline modes) with realistic order simulation
- PoolBacktestEngine for parallel backtesting
- LiveEngine with unified warmup, over CCXT Pro streaming and MetaTrader 5
- TrainingEngine
- Strategy framework with the unified proxy model
- Optimization
- Technical analysis library: SMA, EMA, MACD, RSI, ATR, BollingerBands, ZigZag
- Volume Profile and Rolling Volume Profile
- Footprint analysis
- Confirmed Pivots and Swing detection
- Market-structure indicators (NBS, HHLL)
- Large Trades detection (L1)
- Multi-symbol strategies
- Statistics and performance metrics
- Static plotting with Bokeh
- Data I/O (CSV, Parquet, HDF5) and bundled sample datasets
- Structured logger

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

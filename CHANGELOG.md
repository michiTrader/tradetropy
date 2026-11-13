# Changelog

All notable changes to Tradetropy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.3] - 2026-11-13

### Fixed
- The MT5 live connector now passes trade request fields as keyword arguments.
  This fixes the Python SDK's `Unnamed arguments not allowed` error when
  placing market or pending orders, closing positions, updating SL/TP, or
  canceling pending orders.

## [0.3.2] - 2026-11-03

### Added
- `tradetropy.ta.order_blocks`: SMC order-block detectors over plain NumPy OHLC -
  `find_order_blocks()` plus the `SmcOrderBlock` indicator, `blocks_to_rects()`
  and `pack_active_bands()` for plotting. One detector parameterised on three
  orthogonal axes (anchor / zone / mitigation) with optional validation
  filters; every `Block` carries `confirmed_idx`, the bar at which it first
  becomes knowable from *closed* bars only, so no detector looks ahead and
  backtest/live stay in parity.
- `tradetropy.ta.ConfirmedPivotSeries`: `ConfirmedPivot` plus a causal query API
  (`count_upto()`, `pivots_upto()`, `pivots_from()`) returning
  `(conf_ts, kind, price, real_ts)` entries. The store is append-only and the
  caller keeps its own cursor, so folding pivots bar by bar costs
  O(#new pivots) - usually zero - instead of re-slicing the four band arrays
  (~75 us per read on a 6 300-bar window).
- `LiveOhlcRing.apply_kline()`: feed a full OHLC candle into the ring (kline
  feeds) instead of collapsing it through `process_tick()`.
  `aggregate=False` keeps the live snapshot semantics of a forming candle;
  `aggregate=True` folds finer rows into the subscribed bucket
  (high=max, low=min, close=last, volume=sum).
- `KlineBroker.sltp_ambiguous`: how many bars hit the SL and the TP of the same
  position, i.e. how much of the result depends on the tie-break guess (see
  Changed).
- Replay: `_klines_to_klines()` aggregates klines -> klines (open=first,
  high=max, low=min, close=last, volume=sum), and the playback kline loop is
  now event-driven: it waits on a controller step and drains only the native
  rows in `(last_idx, cursor]`, each with its full OHLC.

### Changed
- **Legend placement:** the backtest/live legend now sits on its own strip
  **above** the price frame (`legend_placement="outside_top"`, the new default),
  horizontal and with air between entries, instead of a vertical stack inside the
  top-left corner covering the first candles. Click still toggles each layer.
  `legend_placement="inside"` brings the old behaviour back and `"none"` hides it
  entirely. New `PlotConfig.legend_placement` and `bt.plot(legend_placement=)`.
- **`output="file"` no longer calls `show()`**: the HTML is serialised with
  `file_html`, so writing a chart never opens a browser tab. The page background
  is also painted in the theme colour via an `html,body{background:...}` rule in
  the `<head>`; without it the area the layout does not cover was rendered white
  by the browser on any dark-theme report.
- New `PlotConfig.volume_alpha` (and `bt.plot(volume_alpha=)`): opacity of the
  volume bars. `None` (default) keeps the theme value; `~0.12` reads as context
  instead of hiding the candles in wide screenshots.
- **Breaking (default flipped):** `sltp_tiebreak` now defaults to `"worst"` in
  `KlineBroker`, `SeshSimulatorBase` and `SeshMT5Sim` (kline feeds only). When a
  single bar touches both the SL and the TP of a position, the stop is assumed
  to have fired first, so no win is booked that the OHLC data cannot prove. The
  previous default, `"open"` (level closest to the bar open wins), is still
  available but is now opt-in; backtests that relied on it will change. Every
  ambiguous bar is counted in `broker.sltp_ambiguous`, so the exposure is
  measurable before and after.

### Fixed
- **MT5 live could not list orders**: `SeshMT5Live.orders()` and
  `order_history()` imported `_ORDER_TYPE_STR_MAP` from `tradetropy.sim_sesh`, a
  module that no longer exists, so every call raised `ModuleNotFoundError` (and
  a live strategy calls `orders()` on every bar). The string -> `OrderType` map
  now lives in the connector.
- **`SeshMT5Sim` profit and account info**: both read `cfg.tick_step`, an
  attribute `SymbolConfig` does not have (`tick_step` is only the dict key of
  `get_account_info()`), so they raised `AttributeError`. They now use
  `cfg.tick_size`.
- **Kline replay plotted a flat chart**: history and visible rows were
  re-bucketed through `_ticks_to_klines()`, which reads a tick price column that
  a kline row does not have, so every candle came out
  `open == high == low == close`. Kline feeds now go through
  `_klines_to_klines()`, which keeps native timestamps untouched when the
  requested interval already matches the data.
- **Replay chart opened empty**: `populate_history()` skipped the initial load
  whenever the dataset was older than 30 minutes by the wall clock - always
  true for a replay of historical data. A playback session now passes
  `max_age_minutes=0`.
- **Kline replay flickered and doubled the volume**: the old loop busy-polled
  `_fetch_last_kline` and re-emitted the same candle every poll, re-summing its
  volume. Each native row is now fed exactly once, driven by the replay cursor.
- **Live warmup never completed with several intervals**: the readiness counter
  took the *coarsest* subscribed interval, so a `1m` + `5m` strategy never
  reached its warmup and `on_data()` never ran. It now tracks the finest proxy.
- **Overlay autoscale flattened the candles**: a band holding timestamps
  (epoch ms, e.g. ZigZag's `[price, ts]` output) was fed to the price Y-range
  autoscale. Timestamp bands are now excluded from it.
- **Play/pause toggle out of sync**: the widget was always created in the
  "playing" state, so a replay that opens paused showed a play glyph whose first
  click only paused (no visible effect). Label, active state and styling are now
  seeded from `controller.paused`.
- **Sequence-valued `line_width`** (per-point widths) was passed straight to the
  glyph, which Bokeh rejects when a `source` is given; it is now moved into a
  `line_width` column of the `ColumnDataSource`. `Labels` are also handled
  before the width lookup, so a label primitive never picks up a line width.
- `read_klines()` accepted `avg_spread` but never wrote it into the
  `SymbolConfig`, so every kline backtest ran with zero spread unless the config
  was patched by hand.

## [0.3.1] - 2026-09-22

The 0.3.x line made it to PyPI as 0.3.1 (0.3.0 was skipped); its contents are
the 0.3.0 notes below, plus the removal of the unused `keyboard` dependency.

## [0.3.0] - 2026-07-28

### Added
- Pattern DSL: `{min,max}` repetition quantifier (immediate suffix to a node's
  type, e.g. `~any{1,3}`), expanding at parse time into an elastic gap of
  mandatory plus optional pivot copies. `{n}` is shorthand for exactly `n`
  copies. `$N` references are remapped automatically across the expansion
  (absolute references resolve to the first copy; references to a min-0 group
  and relative references inside a quantified node are rejected as
  ambiguous). Capped at `_MAX_OPTIONAL_NODES` expanded optional nodes to guard
  against the matcher's 2^K cost blow-up.
- `stats/stats.py` (module) split into `_freq.py`, `_prepare.py`, `metrics.py`,
  `_compute.py` and `_result.py` by responsibility; `stats.py` is now a thin
  backwards-compatible facade re-exporting every public and previously-public
  name (no API change).
- `# Positions` / `# Positions Long` / `# Positions Short` stats metrics:
  distinct positions (partial closes of the same position merged into one),
  replacing `# Trades Long` / `# Trades Short`. `# Trades` keeps counting
  every closed trade row (partial closes included). `bt.stats.positions`
  exposes the underlying per-position `DataFrame`.

- `bt.plot(header_text=, footer_text=)`: optional manual notes (raw HTML)
  rendered above/below the whole chart layout.
- `bt.plot(superimpose=)`: overlay translucent, wick-less candle bodies of a
  higher timeframe behind the main OHLC (backtesting.py-style). Accepts
  `True` (auto-selects the higher TF from the chart's own interval) or an
  explicit timeframe string (e.g. `"4h"`). Backtest-plot only.
- `bt.plot(derender_on_pan=)` (default `True`): temporarily hides heavy
  overlay glyphs (scatter markers, text labels) while the user drags or
  wheel-zooms the chart, restoring them once the interaction settles, so
  charts with many marker-based indicators (NBS, pivots, ...) stay fluid.
- `IndicatorPlotConfig.hide_on_pan` (per-indicator override of the above):
  defaults per renderer type (scatter/label/rect/arrow hide during pan,
  line/step/bar/span stay visible); dense bar panels (MACD histogram, etc.)
  instead get level-of-detail decimation so they never blank mid-drag.
- Sharpe/Sortino/Calmar/Profit Factor/SQN in the plot's stats bar now always
  show 2 decimals (previously rounded to a whole number for values >= 1).
- `bt.plot(title=)`: sets the browser tab / window title (default `'Backtest'`).

### Removed
- The floating price tag that followed the crosshair (TradingView-style) has
  been removed from the chart entirely; the native crosshair already reads
  the value off the axis. `show_price_tag` is still accepted by
  `PlotConfig`/`bt.plot()` for backwards compatibility and silently ignored,
  emitting a `DeprecationWarning`.

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
  `is_bullish()`/`is_bearish()`, `efficacy()`/`efficacy_all()`).
- `ManualMarks`: strategy-driven manual marks/levels (segments with optional
  markers and labels) added, updated and closed from `on_data()`.
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
  `tools/gen_docs_demo.py`.
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

## [0.1.0]

Initial public release.

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

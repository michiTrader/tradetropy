# Indicators

The `tradetropy.ta` package ships the classic technical studies plus market
structure and order-flow tools, all through a single declarative contract. You
add any of them with `add_indicator()` and read their developing values in
`on_data()`.

## Adding an indicator

`add_indicator(source, indicator)` attaches an indicator to a source and returns
a handle. Single-output indicators are read with `[-1]`; multi-output ones
expose named bands:

```python
from tradetropy.ta import SMA, BollingerBands

self.sma = self.add_indicator(self.btc.close, SMA(20))
self.bb  = self.add_indicator(self.btc.close, BollingerBands(20, 2.0))

# in on_data():
fast = self.sma[-1]
upper, mid, lower = self.bb.upper[-1], self.bb.mid[-1], self.bb.lower[-1]
```

Multi-source indicators declare the columns they expect through a `.refs()`
helper or by passing the `*_ref` accessors in order:

```python
from tradetropy.ta import Alligator, PivotPoints

self.alli = self.add_indicator(
    [self.btc.high_ref, self.btc.low_ref], Alligator(),
    name=['Jaw', 'Teeth', 'Lips'],
)
self.piv = self.add_indicator(PivotPoints.refs(self.btc), PivotPoints('classic', '1d'))
```

You can override any visual field at `add_indicator()` time (`name`, `color`,
`line_width`, ...); it merges over the indicator's own plot configuration.

### Session-based indicators

`MarketSessions`, `SessionLevels` and `KillZones` share the same UTC time-window
machinery but answer different questions:

- `MarketSessions` - only exposes a binary in/out-of-session series per named
  session, plus background zones. Use it to filter trades by time of day.
- `SessionLevels` - the running open/high/low of the CURRENT session occurrence
  plus the open/high/low/close of the last CLOSED occurrence, projected forward
  as reference levels (Session High/Low, Asian Range, Previous Session H/L).
- `KillZones` - the same idea scoped to narrow ICT kill zones (London Open, NY
  Open, London Close, Asian) instead of whole sessions; each zone's high/low
  freezes when the window closes and stays projected as a breakout reference
  until the SAME zone opens again.

```python
from tradetropy.ta import SessionLevels, KillZones

self.sl = self.add_indicator(
    SessionLevels.refs(self.btc), SessionLevels(sessions=['london', 'new_york']),
    plot=True,
)
self.kz = self.add_indicator(
    KillZones.refs(self.btc), KillZones(windows=['london_open', 'ny_open']),
    plot=True,
)

# in on_data():
prev_london_high = self.sl.london_prev_high[-1]     # yesterday's London high
if self.kz.london_open_active[-1] == 0.0:           # kill zone already closed
    if self.btc.close[-1] > self.kz.london_open_high[-1]:
        pass  # breakout above the London Open kill zone range
```

Both accept the same predefined-string-or-custom-dict window format as
`MarketSessions` (`{"name": "silver_bullet", "start": 10, "end": 11}`).

### Candlestick pattern detection

`CandlePatterns` is a statistical detector: it classifies a candle relative to
the recent distribution (e.g. a hammer needs a lower wick beyond an adaptive
percentile of the last `window` bars and a small body) rather than matching a
fixed shape, and can gate reversal patterns by price context (z-score of close
vs its rolling mean). Detection is pure and causal, so it is identical in
backtest, live and replay. It draws the pattern name on each candle and exposes
a query API - including each pattern's own causal hit-rate - through the handle
`add_indicator()` returns:

```python
from tradetropy.ta import CandlePatterns

self.candles = self.add_indicator(self.btc, CandlePatterns())

# in on_data():
if self.candles.last_pattern() == 'Bullish Engulfing':
    eff = self.candles.efficacy('Bullish Engulfing')
    if eff['sample_size'] >= 20 and eff['hit_rate'] > 0.55:
        self.sesh.buy('BTCUSDT', volume=1)
```

`last_pattern()` / `pattern_at(i)` / `patterns(n)` read the pattern at a bar
offset; `is_bullish()` / `is_bearish()` read the current bar's bias;
`efficacy(pattern)` / `efficacy_all()` return the causal hit-rate (only signals
whose `horizon` has elapsed are scored, so it never uses future bars).

### Manual marks

`ManualMarks` lets a strategy draw its own segments/levels from `on_data()` -
useful for annotating signals, support/resistance a strategy computes at
runtime, or debugging a detector visually. A mark is a line from
`(ts0, price0)` to `(ts1, price1)`; leaving the end open (`None`) keeps it
"live" until closed, extending to the latest bar/tick meanwhile:

```python
from tradetropy.ta import ManualMarks

self.marks = self.add_indicator(ManualMarks.refs(self.btc), ManualMarks())

# in on_data():
if some_signal:
    self.mark_id = self.marks.add_mark(
        price0=self.btc.close[-1], ts0=self.ts,
        color='#F6465D', label='Signal',
    )
if close_condition and self.mark_id is not None:
    self.marks.close_mark(self.mark_id, ts1=self.ts, price1=self.btc.close[-1])
```

`update_mark(mark_id, **fields)` edits any field of an open mark;
`remove_mark(mark_id)` / `clear_marks()` delete marks; `.marks` returns a
read-only snapshot of every current mark.


## Built-in catalog

- **Trend / moving averages**: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `HMA`,
  `KAMA`, `FRAMA`, `VIDYA`, `MACD`, `Ichimoku`, `ParabolicSAR`, `Supertrend`.
- **Oscillators / momentum**: `RSI`, `Stochastic`, `StochasticRSI`, `CCI`,
  `WilliamsR`, `Momentum`, `ROC`, `CMO`, `TSI`, `TRIX`, `UltimateOscillator`,
  `DeMarker`, `RVI`, `OsMA`, `BullsPower`, `BearsPower`, `Aroon`, `Vortex`,
  `SchaffTrendCycle`, `PO`, `PPO`, `BOP`, `DPO`, `MassIndex`, `ADX`.
- **Volatility**: `BollingerBands`, `ATR`, `StdDev`, `Envelopes`,
  `KeltnerChannels`, `DonchianChannels`.
- **Volume / order flow figures**: `OBV`, `VWAP`, `VWMA`, `MFI`, `ChaikinAD`,
  `ChaikinOsc`, `ForceIndex`, `EMV`, `MarketFacilitationIndex`.
- **Bill Williams**: `Alligator`, `GatorOscillator`, `AwesomeOscillator`,
  `AcceleratorOscillator`, `Fractals`.
- **Levels / structure**: `PivotPoints`, `PivotHighLow`, `ConfirmedPivot`,
  `ZigZag`, `SwingHL`, `EqualHL`, `HHLL`, `NBS`, `FairValueGap`, `OrderBlock`,
  `MarketSessions`, `SessionLevels`, `KillZones`.
- **Annotation**: `CandlePatterns` (statistical candlestick pattern detector
  with causal efficacy tracking), `ManualMarks` (strategy-driven marks/levels).
- **Volume profile**: `VolumeProfile`, `TickVolumeProfile`,
  `RollingVolumeProfile`.
- **Order flow**: `LargeTrades`, `DeepTrades`, `DeltaBars`, `CVD`, `VolumeInfo`,
  `COT`, `DeepWall`, `DeepReload`, `StopRun` - see
  [Order flow and L2](order-flow.md).

## Writing your own indicator

Indicators are external: subclass `Indicator`, implement `calculate(source)`,
set `name` / `category` / `output_names`, and assign a `plot_config`. A single
generic renderer turns any indicator's output into glyphs for both the static
and live charts - a normal indicator needs no special plotting code.

```python
import numpy as np
from tradetropy.ta import Indicator, IndicatorPlotConfig


class SMA(Indicator):
    name = 'sma'
    category = 'trend'      # trend|momentum|volatility|volume|structure|annotation|other

    def __init__(self, length: int):
        self.length = length
        self.plot_config = IndicatorPlotConfig()   # defaults are enough

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < self.length:
            return out
        cs = np.cumsum(source)
        L = self.length
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out
```

`calculate` returns `[N]` for a single band or `[K x N]` for a multi-band
indicator (one row per name in `output_names`), with `NaN` during warmup.

`category` decides default placement: `trend`/`volatility`/`structure`/
`annotation` overlay on price; `momentum`/`volume`/`other` get their own panel.
Override with `overlay=True|False` and set `panel_title`/`reference_lines` for
own-panel indicators. Per-band styling (`color`, `line_dash`, `line_width`, ...)
accepts a scalar or a list (one entry per band).

For geometry that is not "one value per bar" (histograms, zones, session bands),
implement an optional `draw()` method returning declarative primitives (`HBars`,
`HLines`, `Segments`, `Points`, `Rects`, `Labels` from `tradetropy.ta.draw`). The
same generic renderer draws them in both charts.

!!! tip "Recursive indicators and engine parity"
    If your indicator carries running state (EMA, Wilder smoothing), set
    `warmup_factor` on the class so the auto-warmup reserves enough bars to
    converge before the first `on_data()`, keeping backtest and live/replay in
    parity. `RSI` and `MACD` use `warmup_factor = 5`.

See the full API in the [Indicators reference](../reference/indicators.md).

# Plotting

Tradetropy renders interactive Bokeh charts for both finished backtests (static)
and live/replay sessions. Every indicator draws through one generic renderer, so
built-in and custom indicators plot the same way with no plotting-package code.

## Backtest charts

After a backtest, `bt.plot()` opens an interactive chart with the candles,
your indicators (overlays on price and own-panel studies), trade markers and the
equity curve:

```python
from tradetropy import BacktestEngine
from tradetropy.datasets import load_btcusd_1m

bt = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
bt.run()
bt.plot()                      # opens in the browser
bt.plot(theme='dark')          # dark theme
```

`PlotConfig` controls the chart chrome (theme, sizing, ...):

```python
from tradetropy import PlotConfig
bt.plot(config=PlotConfig(theme='dark'))
```

## Themes and indicator colors

The theme styles only the chrome (background, grid, candles, axes). Indicator
colors are fixed object attributes and are **independent of the theme**, so an
indicator looks the same in light and dark mode. Set an indicator's colors on
its object (constructor args or `IndicatorPlotConfig`), not via the theme.

## Candle alignment

By default an indicator's drawn geometry (bubbles, zones, labels) is snapped to
the candle grid so everything shares the candle time unit. Disable it for
sub-candle microstructure precision:

```python
bt.plot(align_indicators_to_candle=False)
```

## Live and replay charts

Live, replay and paper-trading sessions use `LiveChart`, which streams the same
glyphs in real time and adds trade overlays (closed-trade markers, average-price
line, take-profit/stop-loss lines):

```python
from tradetropy.plotting import PlotConfig
from tradetropy.plotting.live import LiveChart

chart = LiveChart(config=PlotConfig(theme='dark'), max_candles=200)
engine.run(live_chart=chart)
```

See [Replay and paper trading](replay.md) and [Live trading](live.md).

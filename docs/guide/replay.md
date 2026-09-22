# Replay and paper trading

Replay engines play a strategy back over historical data with an interactive
Bokeh chart (play / pause / step / speed), so you can visually audit behavior.
They are also the way to feed a recorded **order book** to L2 order-flow
indicators offline.

## ReplayEngine

`ReplayEngine` runs a programmatic strategy forward over historical ticks or
klines, with the same constructors as the other engines plus playback options:

```python
from tradetropy.replay import ReplayEngine
from tradetropy.plotting import PlotConfig
from tradetropy.plotting.live import LiveChart
from tradetropy.datasets import load_btcusd_1m

engine = ReplayEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),), speed=5.0)

chart = LiveChart(config=PlotConfig(theme='dark'), max_candles=200)
engine.run(live_chart=chart)     # play/pause/step/speed controls appear in the browser
```

Playback is forward-only; use **Restart** to rewind and replay from scratch.
`speed` is a ticks/sec multiplier (`float('inf')` = turbo).

## Replaying a recorded order book

Pass `book=BookData` (a single `BookData` or a tuple/list of them) to replay an
L2 book time-aligned with the ticks, which is what makes `DeepTrades` and the
order-book metrics work offline. The symbol is read from each `BookData.symbol`,
so you never pass a `{symbol: BookData}` mapping. The bundled `adausd_ticks` +
`adausd_book` datasets are aligned for this:

```python
from tradetropy.datasets import load_adausd_ticks, load_adausd_book

engine = ReplayEngine.by_ticks(
    DeepFlow(),
    data=(load_adausd_ticks(),),
    book=load_adausd_book(),
    speed=20.0,
)
engine.run(live_chart=chart)
```

See [Order flow and L2](order-flow.md) for the full `DeepTrades` strategy and
[`examples/orderflow_l2.py`](../examples.md).

## PaperEngine

`PaperEngine` is the discretionary counterpart: it plays data forward and
lets *you* place trades manually from the chart (order ticket), for practicing
setups without writing a programmatic strategy. It shares the same forward-only
playback controls as `ReplayEngine`.

## Live trade overlays

The live/replay/paper-trading chart draws closed-trade markers and connector
lines, the open position's average-price line, and dashed take-profit (green)
and stop-loss (red) lines with price labels. These update every tick and clear
on Restart.

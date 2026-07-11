# Order flow and L2

Tradetropy ships a full order-flow toolkit: large-trade detection, depth-aware
classification (absorption, sweeps), per-bar delta, cumulative delta, COT and L2
liquidity overlays. They are all tick-mounted indicators added with
`add_indicator()` and read in `on_data()`, and they plot through the same
declarative contract as every other indicator.

## Large Trades

`LargeTrades` highlights the largest aggressive prints ("whales") from the trade
stream. It needs only the tick feed - no order book. Detection is causal, so it
is safe for backtesting.

```python
import numpy as np
from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_mesu26_ticks
from tradetropy.ta import LargeTrades


class WhaleWatch(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks('MESU26', window_size=2000)
        self.whales = self.add_indicator(
            LargeTrades.refs(self.ticks),      # [ts, price, volume, flags, bid, ask]
            LargeTrades(threshold='p99', by='notional', window=1000),
        )

    def on_data(self):
        if not np.isnan(self.whales.price[-1]):   # this tick is a large trade
            side = self.whales.side[-1]           # +1 buy / -1 sell
            if side > 0:
                self.sesh.buy('MESU26', volume=1)


bt = BacktestEngine.by_ticks(WhaleWatch(), data=(load_mesu26_ticks(),))
bt.run()
```

Threshold modes:

- `'pXX'` - trailing quantile (e.g. `'p99'` keeps the top 1% within `window`).
- `'Nx'` - N times the trailing median (e.g. `'5x'`).
- `float` - a fixed absolute magnitude.

`by` selects the magnitude metric: `'volume'`, `'notional'` (price x volume) or
`'delta'` (net aggression, meaningful with `aggregate_ms > 0`). Set
`aggregate_ms` to merge a burst of child prints into one synthetic event before
thresholding.

## Deep Trades (L2 classification)

`DeepTrades` extends `LargeTrades` with the real-time L2 order book. It detects
the same outsized prints, then classifies each against the resting liquidity it
hit into **Large Aggressor**, **Absorption** (into a wall that holds) or
**Sweep** (clears several levels). With an L3/MBO stream it adds **Iceberg** and
**Liquidity Grab**.

Because classification needs depth, the order book is passed to the
**constructor** (it is not a numeric column source):

```python
self.deep = self.add_indicator(
    DeepTrades.refs(self.ticks),
    DeepTrades(self.book, threshold='p99', by='notional'),
)
```

`DeepTrades.class_name(event_type)` maps the numeric class code to a readable
name (`'aggressor'`, `'absorption'`, `'sweep'`, `'iceberg'`, `'liquidity_grab'`,
or `''` when the tick is not an event).

## How the order book reaches the engine

This is the key thing to understand for L2:

- **Live**: `subscribe_orderbook(symbol, depth)` returns an `OrderbookProxy` that
  the engine feeds automatically from the WebSocket book stream.
- **A plain `BacktestEngine` has no order book.** Its book stays `stale`, book
  metrics return `NaN`, and `DeepTrades` falls back to Large Aggressor for every
  event.
- **Offline with recorded/sample book data**, use `ReplayEngine` with
  `book=BookData` (a single `BookData` or a tuple/list of them; the symbol is
  read from each `BookData.symbol`). Internally it merges the ticks and book
  snapshots into one timestamp-ordered stream and replays them, so the proxy is
  fed the book as-of each trade.

The bundled `adausd_ticks` and `adausd_book` datasets are timestamp-aligned for
exactly this:

```python
import numpy as np
from tradetropy import Strategy
from tradetropy.datasets import load_adausd_ticks, load_adausd_book
from tradetropy.replay import ReplayEngine
from tradetropy.ta import DeepTrades


class DeepFlow(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks('ADAUSDT', window_size=2000)
        self.book = self.subscribe_orderbook('ADAUSDT', depth=5)
        self.deep = self.add_indicator(
            DeepTrades.refs(self.ticks),
            DeepTrades(self.book, threshold=2000.0, by='volume', window=500),
        )

    def on_data(self):
        et = self.deep.event_type[-1]
        if not np.isnan(et) and DeepTrades.class_name(int(et)) == 'sweep':
            self.sesh.buy('ADAUSDT', volume=1)


engine = ReplayEngine.by_ticks(
    DeepFlow(),
    data=(load_adausd_ticks(),),
    book=load_adausd_book(),               # <- the order book enters here
    speed=20.0,
)
engine.run()                               # opens the interactive replay chart
```

The runnable version is [`examples/orderflow_l2.py`](../examples.md).

!!! warning "Relative thresholds in replay"
    Relative thresholds (`'p99'`, `'5x'`) work in `BacktestEngine`, but through
    the `ReplayEngine` per-tick path they may not fire on short samples. For the
    replay L2 example above an **absolute** threshold is used so detections are
    reliable. When you have a live/recorded book you can revisit relative modes.

## Per-bar delta: DeltaBars, CVD, VolumeInfo, COT

Four tick-mounted panels turn the trade stream into per-bar order-flow figures.
Each classifies the aggressor side of every trade and aggregates into
fixed-interval bars (match `period` to your candle interval):

- `DeltaBars` - diverging histogram of per-bar delta (`ask_vol - bid_vol`).
- `CVD` - cumulative volume delta, drawn as candles or a diverging bar.
- `VolumeInfo` - a configurable per-bar numeric breakdown (delta max/min, buy,
  sell, total).
- `COT` - per-bar commitment figures (COT High / COT Low / Delta) as labels over
  each candle (GoCharting-style, not the weekly CFTC report).

```python
from tradetropy.ta import CVD, DeltaBars

self.ticks = self.subscribe_ticks('MESU26', window_size=5000)
self.delta = self.add_indicator(DeltaBars.refs(self.ticks), DeltaBars('1m'))
self.cvd   = self.add_indicator(CVD.refs(self.ticks), CVD('1m'))
```

## L2 liquidity overlays: DeepWall, DeepReload, StopRun

Three overlays read the order book's evolution over time (via the causal
`book_window()`): resting-liquidity walls, liquidity replenishment (the L2
analogue of an iceberg) and stop sweeps. Like `DeepTrades` they take the
order-book proxy in their constructor and are meaningful in live mode and in
replay of a recorded book.

```python
from tradetropy.ta import DeepWall, DeepReload, StopRun

self.walls  = self.add_indicator(DeepWall.refs(self.ticks), DeepWall(self.book))
self.reload = self.add_indicator(DeepReload.refs(self.ticks), DeepReload(self.book))
self.stops  = self.add_indicator(StopRun.refs(self.ticks), StopRun(self.book, tick_size=0.0001))
```

## Volume Profile

Two volume-by-price indicators, both added like any indicator and mounted on
data you already subscribed to:

- `VolumeProfile` - kline-based (distributes each candle's volume across its
  range).
- `TickVolumeProfile` - tick-based and precise (bins each trade at its real
  price and classifies the aggressor side).

Both expose developing `poc` / `vah` / `val` series and reset every `period`.
`RollingVolumeProfile` never resets - it aggregates the trailing `length`
candles (TradingView VPVR style).

```python
from tradetropy.ta import VolumeProfile

self.vp = self.add_indicator(
    VolumeProfile.refs(self.btc),
    VolumeProfile(period='1d', nodes='both'),
)
# in on_data(): self.vp.poc[-1], self.vp.vah[-1], self.vp.val[-1]
```

See the [Indicators reference](../reference/indicators.md) for every option.

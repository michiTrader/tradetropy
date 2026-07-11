# Core concepts

Tradetropy is built around one idea: **you write a strategy once and run it
unchanged** across backtest, live and replay. The engine differences are a
transport detail hidden behind the same interfaces. This page explains the
pieces you interact with.

## The Strategy

Every strategy subclasses `Strategy` and implements two methods:

- `init()` - runs once. Subscribe to data and declare indicators here.
- `on_data()` - runs on every new data point (a closed candle or a trade).
  Read your indicators and place orders here.

```python
from tradetropy import Strategy
from tradetropy.ta import RSI


class MeanReversion(Strategy):
    def init(self):
        self.btc = self.subscribe_ohlc('BTCUSD', '1m', window_size=300)
        self.rsi = self.add_indicator(self.btc.close, RSI(14))

    def on_data(self):
        if self.rsi[-1] < 30 and not self.sesh.positions('BTCUSD'):
            self.sesh.buy('BTCUSD', volume=1)
        elif self.rsi[-1] > 70:
            for pos in self.sesh.positions('BTCUSD'):
                self.sesh.position_close(pos.ticket)
```

## Subscriptions and proxies

Inside `init()` you subscribe to the data your strategy needs. Each
subscription returns a **proxy** - a live, causal view of that data that you
read in `on_data()`:

- `subscribe_ohlc(symbol, timeframe, window_size=...)` -> an OHLC proxy with
  `.open`, `.high`, `.low`, `.close`, `.volume`.
- `subscribe_ticks(symbol, window_size=...)` -> a tick proxy with `.price`,
  `.volume`, `.bid`, `.ask`, `.flags`, `.ts`.
- `subscribe_orderbook(symbol, depth=...)` -> an order-book proxy with
  `imbalance()`, `mid`, `spread`, `best_bid/ask` and a causal `book_as_of()`.

The column accessor is unified: in `init()` `self.btc.close` is a declarative
reference for `add_indicator()`, and in `on_data()` the same `self.btc.close[-1]`
reads the latest value. Index `[-1]` is the current data point, `[-2]` the
previous one, and so on.

`window_size` bounds how much trailing history the proxy keeps - size it to
cover the longest lookback your indicators need.

## Indicators

`add_indicator(source, indicator)` attaches an indicator to a data source and
returns a handle you read in `on_data()`. Indicators are external objects from
`tradetropy.ta` (or your own subclass of `Indicator`). See
[Indicators](indicators.md).

```python
self.fast = self.add_indicator(self.btc.close, SMA(10))
# ... in on_data():
value = self.fast[-1]
```

## The session (`self.sesh`)

`self.sesh` is the broker/account interface. It is the same API in backtest and
live:

- `self.sesh.buy(symbol, volume=...)` / `self.sesh.sell(symbol, volume=...)`
- `self.sesh.positions(symbol)` -> open positions
- `self.sesh.position_close(ticket)`

In a backtest the session is a simulator; in live it is a real broker
connector. Your `on_data()` code does not change.

## Engines

An engine drives a strategy over data. They share the `by_klines` /
`by_ticks` constructors and a `.run()` method:

| Engine                | Use for                                             |
|-----------------------|-----------------------------------------------------|
| `BacktestEngine`      | Fast historical simulation over candles or ticks    |
| `PoolBacktestEngine`  | Parallel backtests (optimization, parameter sweeps) |
| `ReplayEngine`        | Play back history with a chart and play/pause/step  |
| `PaperEngine`         | Discretionary manual playback for practice          |
| `LiveEngine`          | Real-time execution over a live session             |
| `LivePool`            | Run several live strategies under one supervisor    |

```python
from tradetropy import BacktestEngine
from tradetropy.datasets import load_btcusd_1m

bt = BacktestEngine.by_klines(MeanReversion(), data=(load_btcusd_1m(),))
bt.run()
print(bt.stats)
```

`by_klines` takes a tuple of `KlineData` (one per symbol); `by_ticks` takes a
tuple of `TickData`. See [Working with data](data.md) for how to build those.

!!! note "Order book and L2 indicators"
    A plain `BacktestEngine` has no order book, so L2 order-flow indicators
    (`DeepTrades`, `DeepWall`, ...) need a book supplied through
    `ReplayEngine(book=...)` or a live session. See
    [Order flow and L2](order-flow.md).

## Results and stats

After `run()`, `bt.stats` holds the performance metrics (return, drawdown,
Sharpe, profit factor, win rate, ...) and `bt.plot()` opens an interactive
chart. To test robustness, `bt.montecarlo(...)` runs a Monte Carlo analysis
- see [Monte Carlo robustness](robustness.md).

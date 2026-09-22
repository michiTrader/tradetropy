# Quickstart

This page takes you from install to your first backtest, chart and metrics in a
few minutes. Every snippet uses the [bundled datasets](../guide/data.md), so you
can copy-paste and run without any data files or API keys.

## 1. Install

```bash
pip install tradetropy
```

## 2. Write a strategy

A strategy subclasses `Strategy`, declares its data and indicators in `init()`,
and reacts to each new data point in `on_data()`. Orders go through the session,
`self.sesh`.

```python
from tradetropy import Strategy
from tradetropy.ta import SMA


class SmaCross(Strategy):
    """Long when the fast SMA is above the slow SMA."""

    def init(self):
        # Subscribe to 1-minute candles and attach two moving averages.
        self.btc = self.subscribe_ohlc('BTCUSD', '1m', window_size=200)
        self.fast = self.add_indicator(self.btc.close, SMA(10))
        self.slow = self.add_indicator(self.btc.close, SMA(30))

    def on_data(self):
        if self.fast[-1] > self.slow[-1]:
            if not self.sesh.positions('BTCUSD'):
                self.sesh.buy('BTCUSD', volume=1)
        else:
            for pos in self.sesh.positions('BTCUSD'):
                self.sesh.position_close(pos.ticket)
```

## 3. Run a backtest

`BacktestEngine.by_klines` takes one `KlineData` per symbol. The
`load_btcusd_1m()` loader returns exactly that.

```python
from tradetropy import BacktestEngine
from tradetropy.datasets import load_btcusd_1m

bt = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
bt.run()
print(bt.stats)          # performance metrics
```

## 4. Plot the result

```python
bt.plot()                # opens an interactive Bokeh chart in the browser
```

## 5. Where to go next

- [Core concepts](../guide/concepts.md) - the model behind strategies, sessions
  and engines.
- [Working with data](../guide/data.md) - datasets, multi-timeframe and
  multi-symbol, and reading/saving your own files.
- [Indicators](../guide/indicators.md) and [Order flow](../guide/order-flow.md).
- Runnable scripts in [Examples](../examples.md).

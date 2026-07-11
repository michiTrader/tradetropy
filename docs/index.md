# Tradetropy

A professional backtesting and live trading framework for algorithmic trading
strategies, with first-class support for footprint analysis and market
microstructure.

Write a strategy once and run it unchanged across **backtest**, **live** and
**replay** - the engine differences are a transport detail behind the same
`Strategy` API.

## Features

- **Backtesting engine** - high-performance backtesting over candles or ticks
  with realistic order simulation.
- **Live trading** - real-time execution over WebSockets (CCXT Pro) and MT5,
  with a single-writer, event-driven engine.
- **Order flow and footprint** - large trades, Deep Trades (L2/L3 absorption,
  sweeps), cumulative delta, COT, volume profile and liquidity overlays.
- **Indicator library** - the classic studies (SMA, EMA, MACD, RSI, Bollinger,
  ...) plus market-structure tools, all through one declarative contract.
- **Optimization and pools** - parameter optimization and parallel backtests.
- **Monte Carlo robustness** - confidence intervals, probability of loss, risk
  of ruin and a composite robustness score.
- **Data management** - efficient candle, tick and order-book handling with
  NumPy `.npz` / CSV IO built in (Parquet and HDF5 optional), plus bundled
  sample datasets.

## Install

```bash
pip install tradetropy
```

The base install already includes everything you need for backtesting,
plotting and data IO. Broker integrations and Parquet are optional extras - see
[Installation](getting-started/installation.md).

## A first backtest

Every loader in `tradetropy.datasets` returns ready-to-use data, so you can run a
strategy without downloading anything:

```python
from tradetropy import Strategy, BacktestEngine
from tradetropy.ta import SMA
from tradetropy.datasets import load_btcusd_1m


class SmaCross(Strategy):
    def init(self):
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


bt = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
bt.run()
print(bt.stats)
```

## Where to go next

- New here? Read the [Quickstart](getting-started/quickstart.md).
- Learn the model in [Core concepts](guide/concepts.md).
- Load, resample and save data in [Working with data](guide/data.md).
- Browse the full [API reference](reference/index.md).

# Backtesting and optimization

## Running a backtest

`BacktestEngine` simulates a strategy over historical data with realistic order
handling. Choose the constructor by data type:

- `BacktestEngine.by_klines(strategy, data=(KlineData, ...))` - candle-driven.
- `BacktestEngine.by_ticks(strategy, data=(TickData, ...))` - tick-driven.

```python
from tradetropy import BacktestEngine
from tradetropy.datasets import load_btcusd_1m

bt = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
bt.run()
print(bt.stats)
bt.plot()
```

By default the engine builds a simulator session with a starting balance and no
commission. To configure commission, balance or spread, pass a pre-built
`SeshSimulatorBase` as `sesh=`.

## Metrics

`bt.stats` is a `Stats` object with the standard performance figures:
return, annualized return and volatility, Sharpe/Sortino/Calmar, max and average
drawdown (depth and duration), profit factor, win rate, SQN and more.

```python
print(bt.stats['Return [%]'])
print(bt.stats['Max. Drawdown [%]'])
print(bt.stats['Sharpe Ratio'])
```

!!! note "Annualized metrics need enough data"
    Metrics like annualized return, Sharpe and Sortino are set to `NaN` when the
    backtest is shorter than a minimum span, or when there are too few closed
    trades. The bundled datasets are intentionally small, so you will see this
    warning in the examples - it is expected.

## Warmup

The engine automatically reserves warmup bars so indicators are converged before
the first `on_data()`. Recursive indicators (RSI, MACD) declare a
`warmup_factor` so enough history is reserved. Size your proxy `window_size` to
comfortably exceed the longest lookback you use.

## Parallel backtests and optimization

`PoolBacktestEngine` runs many backtests in parallel across processes - the
basis for parameter optimization and sweeps. Strategies and sessions are
supplied as factories (picklable, since they are rebuilt in worker processes).

```python
from tradetropy.backtest import PoolBacktestEngine
```

See the [Engines reference](../reference/engines.md) for the full API.

## Multi-symbol

Pass one data object per symbol. See
[Working with data](data.md#multiple-symbols) for a complete multi-symbol
example and how timestamp alignment works.

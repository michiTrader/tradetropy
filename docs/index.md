# Tradetropy

A professional backtesting and live trading framework for algorithmic trading
strategies, with first-class support for footprint analysis and market
microstructure.

Write a strategy once and run it unchanged across **backtest**, **live** and
**replay** - the engine differences are a transport detail behind the same
`Strategy` API.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[PyPI](https://pypi.org/project/tradetropy/){ .md-button }
[GitHub](https://github.com/michiTrader/tradetropy){ .md-button }

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
strategy without downloading anything. The **Strategy** tab below is a complete,
copy-paste-runnable backtest; the **Results** tab is its output; and the chart
underneath is generated from that very run - scroll to zoom, drag to pan.

=== "Strategy"

    ```python
    --8<-- "assets/demos/sma_cross/snippet.py"
    ```

=== "Results"

    ```text
    --8<-- "assets/demos/sma_cross/stats.txt"
    ```

<iframe src="assets/demos/sma_cross/chart.html"
        title="Tradetropy - SMA crossover interactive chart"
        loading="lazy"
        style="width: 100%; height: 640px; border: 0; margin: 1rem 0;">
</iframe>

See more in the [Examples](examples.md).

## Where to go next

- New here? Read the [Quickstart](getting-started/quickstart.md).
- Learn the model in [Core concepts](guide/concepts.md).
- Load, resample and save data in [Working with data](guide/data.md).
- Browse the full [API reference](reference/index.md).

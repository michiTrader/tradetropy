<p align="center">
  <img src="https://raw.githubusercontent.com/michiTrader/tradetropy/main/docs/assets/logo-wordmark.png" alt="Tradetropy" width="420">
</p>

# Tradetropy

[![PyPI](https://img.shields.io/pypi/v/tradetropy)](https://pypi.org/project/tradetropy/)

A professional backtesting and live trading framework for algorithmic trading
strategies, with first-class support for footprint analysis and market
microstructure.

Write a strategy once and run it unchanged across **backtest**, **live** and
**replay** - the engine differences are a transport detail behind the same
`Strategy` API.

## Features

- **Backtesting engine** over candles or ticks with realistic order simulation.
- **Live trading** over WebSockets (CCXT Pro) and MetaTrader 5.
- **Order flow and footprint**: large trades, Deep Trades (L2/L3 absorption and
  sweeps), cumulative delta, COT, volume profile and liquidity overlays.
- **Indicator library**: the classic studies plus market-structure tools, all
  through one declarative contract.
- **Optimization and pools**, **Monte Carlo robustness** and interactive
  **Bokeh plotting**.
- **Data management** with NumPy `.npz` / CSV IO built in (Parquet and HDF5
  optional) and bundled sample datasets.

## Installation

```bash
pip install tradetropy
```

The base install includes everything for backtesting, plotting and data IO.
Broker integrations and Parquet are optional extras:

```bash
pip install tradetropy[mt5]       # MetaTrader 5
pip install tradetropy[ccxt]      # CCXT crypto exchanges (live + streaming)
pip install tradetropy[bybit]     # Bybit (pybit)
pip install tradetropy[parquet]   # Parquet IO (pyarrow)
pip install tradetropy[hdf5]      # HDF5 IO (PyTables; not available on Termux)
pip install tradetropy[all]       # all runtime extras
```

On desktop (Windows/macOS/Linux) every dependency installs as a prebuilt wheel,
so no compiler is needed. **Termux (Android)** is the exception - the scientific
stack builds from source there; see the
[Termux install guide](https://michiTrader.github.io/tradetropy/getting-started/installation/#termux-android).

## Quickstart

Every loader in `tradetropy.datasets` returns ready-to-use data, so this runs with
no data files or API keys:

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


engine = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
engine.run()
print(engine.stats)
```

More runnable scripts are in [`examples/`](examples/).

## Documentation

Full documentation - user guide and API reference - is at
**https://michiTrader.github.io/tradetropy/**.

- [Installation](https://michiTrader.github.io/tradetropy/getting-started/installation/)
- [Quickstart](https://michiTrader.github.io/tradetropy/getting-started/quickstart/)
- [Core concepts](https://michiTrader.github.io/tradetropy/guide/concepts/)
- [Working with data](https://michiTrader.github.io/tradetropy/guide/data/)
- [Indicators](https://michiTrader.github.io/tradetropy/guide/indicators/) and
  [Order flow and L2](https://michiTrader.github.io/tradetropy/guide/order-flow/)
- [Live trading](https://michiTrader.github.io/tradetropy/guide/live/),
  [Replay](https://michiTrader.github.io/tradetropy/guide/replay/) and
  [Monte Carlo robustness](https://michiTrader.github.io/tradetropy/guide/robustness/)

To build the docs locally:

```bash
pip install tradetropy[docs]
mkdocs serve
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Development setup:

```bash
git clone https://github.com/michiTrader/tradetropy.git
cd tradetropy
uv sync
uv run pytest
```

## Risk disclaimer

Tradetropy is provided "AS IS", without warranty of any kind. Trading financial
instruments involves risk of loss of capital - you may lose part or all of your
funds. Always test first on a demo / testnet / paper account and verify that
orders, positions and balances behave as expected before trading with real
money. Broker and exchange APIs may change without notice and break a connector.
The authors and contributors are not responsible for any losses, damages or
execution failures arising from the use of this software. This is not financial,
investment or legal advice, and you are responsible for complying with the laws
of your jurisdiction and your broker/exchange terms of service.

The full disclaimer is available at runtime as `tradetropy.LIVE_DISCLAIMER`.

## License

MIT - see [LICENSE](LICENSE).

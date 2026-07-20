# Working with data

This page covers everything about getting market data into Tradetropy: the bundled
sample datasets, building higher timeframes from a single series, running
multiple symbols, and reading/saving your own files. The base binary format is
NumPy `.npz` (no extra needed); CSV is built in, and Parquet and HDF5 are
optional extras.

## The three data types

Tradetropy has three first-class containers, each wrapping a NumPy matrix plus the
symbol configuration the engines need:

- `KlineData` - OHLC candles: `[N x 7]` columns `ts, open, high, low, close,
  volume, turnover`.
- `TickData` - trades/quotes: `[N x 7]` columns `ts, bid, ask, volume, flags,
  volume_real, price`.
- `BookData` - L2 order book: a wide `[N x (2 + 4*levels)]` matrix with
  `.bid_px`, `.bid_sz`, `.ask_px`, `.ask_sz`.

## Bundled datasets

The `tradetropy.datasets` package ships a small, curated collection so you can run
everything with no downloads. Each loader returns a ready-to-use object:

```python
from tradetropy.datasets import (
    load_btcusd_1m, load_adausd_1m,      # 1-minute crypto candles (500 rows)
    load_aapl_1d, load_goog_1d,          # daily stock candles (300 rows)
    load_mesu26_ticks, load_mnqu26_ticks,  # futures ticks (2000 rows)
    load_adausd_ticks, load_adausd_book,   # paired ticks + L2 book
    list_datasets,
)

btc = load_btcusd_1m()          # KlineData
print(btc.symbol, btc.data.shape, btc.interval_ms)

for d in list_datasets():
    print(d['name'], d['kind'], d['symbol'], d['rows'])
```

| Loader                  | Type        | Symbol   | Rows | Format |
|-------------------------|-------------|----------|------|--------|
| `load_btcusd_1m()`      | `KlineData` | BTCUSD   | 500  | npz    |
| `load_adausd_1m()`      | `KlineData` | ADAUSD   | 500  | npz    |
| `load_aapl_1d()`        | `KlineData` | AAPL     | 300  | csv    |
| `load_goog_1d()`        | `KlineData` | GOOG     | 300  | csv    |
| `load_mesu26_ticks()`   | `TickData`  | MESU26   | 2000 | npz    |
| `load_mnqu26_ticks()`   | `TickData`  | MNQU26   | 2000 | npz    |
| `load_adausd_ticks()`   | `TickData`  | ADAUSD   | 2000 | npz    |
| `load_adausd_book()`    | `BookData`  | ADAUSD   | 950  | npz    |

The `adausd_ticks` and `adausd_book` datasets share the same timestamp range, so
they replay together for the L2 order-flow examples (see
[Order flow and L2](order-flow.md)).

To read a bundled file directly (e.g. with pandas), use `dataset_path`:

```python
import pandas as pd
from tradetropy.datasets import dataset_path

with dataset_path('aapl_1d') as path:
    df = pd.read_csv(path)
```

## Multi-timeframe from a single series

You do not need one file per timeframe. Load a single base series (e.g. 1m) and
subscribe to as many timeframes as you want from the same symbol - the engine
resamples each higher timeframe internally from the base candles. Subscribing to
`1m`, `5m` and `15m` over 1m data gives three synchronized OHLC proxies driven
off the one source:

```python
from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import SMA


class MultiTimeframe(Strategy):
    def init(self):
        # One symbol, three timeframes - all resampled internally from the
        # single 1m dataset passed to the engine.
        self.m1 = self.subscribe_ohlc('BTCUSDT', '1m', window_size=200)
        self.m5 = self.subscribe_ohlc('BTCUSDT', '5m', window_size=200)
        self.m15 = self.subscribe_ohlc('BTCUSDT', '15m', window_size=200)
        self.trend = self.add_indicator(self.m15.close, SMA(20))   # 15m bias
        self.fast = self.add_indicator(self.m5.close, SMA(5))      # 5m entry
        self.slow = self.add_indicator(self.m5.close, SMA(20))

    def on_data(self):
        ...


BacktestEngine.by_klines(MultiTimeframe(), data=(load_btcusd_1m(),)).run()
```

The target timeframes must be multiples of the base interval. The runnable
version is [`examples/multi_timeframe.py`](../examples.md).

### Offline resampling

To materialize a higher timeframe as its own `KlineData` outside the engine
(inspection, saving, feeding a different pipeline), use `KlineData.resample()`:

```python
from tradetropy.datasets import load_btcusd_1m

btc_1m = load_btcusd_1m()
btc_15m = btc_1m.resample('15m')
btc_1h  = btc_1m.resample('1h')
btc_1d  = btc_1m.resample('1d')

print(btc_1m.data.shape, '->', btc_1h.data.shape)
```

The target timeframe must be a multiple of the source interval. `resample()`
returns a new `KlineData` (it never mutates the original) with the symbol
configuration propagated.

To build candles from a tick stream instead, use `TickData.to_klines()`:

```python
from tradetropy.datasets import load_mesu26_ticks

ticks = load_mesu26_ticks()
candles_1m = ticks.to_klines('1m')       # KlineData from ticks
```

`to_klines()` accepts a `price_source` to control which price builds the OHLC:
`'price'` (default) uses the price column as-is, `'mid'` uses `(bid+ask)/2` for
every tick, and `'trade'` uses the price column but first drops quote-only
ticks (`volume == 0`), keeping only real trades. Use `'trade'` when the price
column may contain `(bid+ask)/2` fallbacks (e.g. from `normalize_ticks()` or MT5
quote ticks) and you need OHLC that always lands on a real traded price -
important when candles must respect `tick_size`:

```python
candles_from_trades = ticks.to_klines('1m', price_source='trade')
```

With `'trade'`, a bar interval with no real trades in it (only quotes) produces
no candle for that interval - it is not synthesized from quote midpoints.

### Timeframe strings

Timeframe strings are parsed everywhere the same way: `1m`, `15m`, `1h`, `4h`,
`1d`, `1w`, `1mo`. Other multiples (`5m`, `30m`, `2h`, ...) work too. `min` and
`wk` are aliases for `m` and `w`. Note that `mo` means month and a bare
uppercase `M` is rejected on purpose, so a minute (`1m`) can never be confused
with a month (`1mo`).

## Multiple symbols

Pass one data object per symbol to the engine. The strategy subscribes to each
symbol independently:

```python
from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_btcusd_1m, load_adausd_1m
from tradetropy.ta import SMA

SYMBOLS = ('BTCUSD', 'ADAUSD')


class MultiSma(Strategy):
    def init(self):
        self.feed, self.fast, self.slow = {}, {}, {}
        for sym in SYMBOLS:
            self.feed[sym] = self.subscribe_ohlc(sym, '1m', window_size=200)
            self.fast[sym] = self.add_indicator(self.feed[sym].close, SMA(10))
            self.slow[sym] = self.add_indicator(self.feed[sym].close, SMA(30))

    def on_data(self):
        for sym in SYMBOLS:
            if self.fast[sym][-1] > self.slow[sym][-1]:
                if not self.sesh.positions(sym):
                    self.sesh.buy(sym, volume=1)
            else:
                for pos in self.sesh.positions(sym):
                    self.sesh.position_close(pos.ticket)


bt = BacktestEngine.by_klines(
    MultiSma(), data=(load_btcusd_1m(), load_adausd_1m()),
)
bt.run()
```

By default symbols are aligned by row index (fast path, requires equal row
counts). To align by real timestamp instead - correct when symbols have
different frequencies or gaps - pass `align_by_ts=True` to
`BacktestEngine.by_ticks`.

## Reading and saving your own data

`tradetropy.io` reads and writes all three data types in NumPy `.npz` (the base
binary format), CSV, Parquet and HDF5. The readers auto-detect the format from
the extension and normalize the columns; the writers default to the format
implied by the extension (falling back to `.npz`).

```python
from tradetropy.io import (
    read_klines, read_ticks, read_book,
    save_klines, save_ticks, save_book,
)

# Klines - a broker CSV, an .npz export, or a Parquet/HDF5 file.
klines = read_klines('btc_1m.csv', 'BTCUSD', '1m')
klines = read_klines('session.npz', 'BTCUSD', '1m')       # symbol/tf from metadata

# Ticks.
ticks = read_ticks('ticks.npz', 'BTCUSD')

# Save in any format (format inferred from the extension).
save_klines(klines.data, 'out.npz')                        # npz (base format)
save_klines(klines.data, 'out.csv', format='csv')
save_ticks(ticks.data, 'ticks.parquet', format='parquet')  # needs the parquet extra
save_ticks(ticks.data, 'ticks.h5', format='hdf5')          # needs the hdf5 extra
```

Every domain object also has a `.save()` shortcut:

```python
klines.save('btc_1m.npz')
ticks.save('ticks.csv', format='csv')
```

!!! note "Parquet and HDF5 are optional"
    The base binary format is NumPy `.npz`, which needs no extra and works
    everywhere (including Termux). Reading/writing `.parquet` requires `pyarrow`
    (`pip install tradetropy[parquet]`) and `.h5`/`.hdf5` requires PyTables
    (`pip install tradetropy[hdf5]`); PyTables cannot build on Termux, so prefer
    `.npz` there. CSV and `.npz` always work out of the box.

## Order-book IO

`BookData` reads and saves like the others, and supports two on-disk layouts -
the tradetropy-native **wide** layout and the Binance `bookDepth`-style **long**
layout - translated at the file boundary (everything in memory is always wide):

```python
from tradetropy.io import read_book, save_book, convert_book

book = read_book('btc_book.npz', 'BTCUSD')                # wide, auto-detected
book = read_book('BTCUSD-bookDepth.csv', 'BTCUSD')        # long, auto-detected

save_book(book, 'btc_book.npz')                           # wide (default)
save_book(book, 'btc_bookdepth.csv', layout='long')       # long

convert_book('BTCUSD-bookDepth.csv', 'btc_book.npz', symbol='BTCUSD')  # long -> wide
```

See [Order flow and L2](order-flow.md) for how the order book is fed to the
engine and read by the L2 indicators.

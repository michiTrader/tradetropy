# Live trading

A `Strategy` runs unchanged in live mode - streaming is a transport detail behind
the same `Sesh` and proxy interfaces. This page covers crypto streaming (CCXT
Pro), recording data, and running several strategies at once.

!!! warning "Live trading places real orders"
    A session with API keys configured will send real orders to the venue. Test
    with public market data (no keys) and paper/sandbox accounts first. Never
    commit API keys - load them from environment variables or a secrets store.

## Streaming (CCXT Pro)

Crypto live sessions stream trades, klines and L2 depth over WebSockets via CCXT
Pro (one adapter covers Binance, Bybit, OKX and ~100 venues; part of
`pip install tradetropy[ccxt]`).

```python
from tradetropy import Strategy
from tradetropy.connectors.ccxt import SeshCCXTLive
from tradetropy.live import LiveEngine


class Flow(Strategy):
    def init(self):
        self.ticks = self.subscribe_ticks('BTC/USDT')
        self.book = self.subscribe_orderbook('BTC/USDT', depth=20)


sesh = SeshCCXTLive('binance')            # public market data, no keys needed
engine = LiveEngine.by_ticks(Flow(), sesh=sesh)
engine.run()                              # event-driven: every trade is processed
```

Key properties:

- **Event-driven, no missed ticks.** Trades and book deltas are an ordered log
  that is never dropped; quotes and partial klines coalesce to the latest.
- **Order-book metrics in `on_data()`.** The order-book proxy exposes
  `imbalance(depth)`, `mid`, `spread`, `best_bid/ask` and a causal
  `book_as_of(ts)`. While the book is unsynced (`stale`) the metrics return
  `NaN`, so a strategy never acts on a half-built book.
- **Private user-data stream.** On an authenticated session the engine also
  subscribes the private order/fill channels and keeps the session's orders,
  deals and positions current from the stream.

## On-demand data fetch (live only)

A live strategy can pull recent data directly from the venue, mirrored on the
strategy as `self.fetch_*`:

```python
def on_data(self):
    kl   = self.fetch_klines('ETH/USDT', '1h', limit=200)   # -> KlineData
    tk   = self.fetch_ticks('ETH/USDT', limit=500)          # -> TickData
    book = self.fetch_orderbook('BTC/USDT', depth=20)        # -> BookData
```

This is **live-only by design** and raises in backtest/optimize/replay - letting
a backtest pull current data would inject lookahead bias.

## Recording data

`Recorder` captures a symbol's live stream to disk without writing a strategy.
It streams over WebSocket when the session supports it and falls back to REST
polling otherwise.

```python
from tradetropy import Recorder
from tradetropy.connectors.ccxt import SeshCCXTLive

sesh = SeshCCXTLive('binance')

with Recorder(sesh) as rec:
    rec.add_stream('tick', 'BTC/USDT', 'rec/btc_ticks.npz')
    rec.run()                              # until Ctrl+C

# Record the L2 book for one hour, then auto-stop:
Recorder(sesh).add_stream(
    'orderbook', 'BTC/USDT', 'rec/btc_book.npz', depth=20,
).run(duration=3600)
```

The recorded files round-trip through `tradetropy.io` (`read_ticks` / `read_book`),
and replaying them through the same engine path reproduces the identical
`on_data()` decision stream - the record-once, replay-deterministically
guarantee. This is what makes `DeepTrades` backtestable. See
[Replay and paper trading](replay.md).

!!! note "Record format"
    The record path writes the base NumPy `.npz` format by default (no extra,
    works on Termux). Pass a `.h5`/`.hdf5` path instead to record HDF5 (needs
    `pip install tradetropy[hdf5]`). Because `.npz` is not appendable, an npz
    recording buffers to a small sidecar during the session and is finalized
    into the `.npz` file when the recorder stops.

## Running multiple strategies (LivePool)

`LivePool` supervises several live strategies under one process, organized into
**groups** that share a session (one WebSocket feed + broker), with per-group
isolation (`'thread'` or `'process'`).

```python
from tradetropy import LivePool
from tradetropy.connectors.ccxt import SeshCCXTLive


def on_fail(ev):
    print(f'[{ev.group}] {ev.strategy_name} crashed: {ev.exc!r}')


pool = LivePool(on_error=on_fail)
pool.add_group(
    strategies=[MakerStrat, TakerStrat],
    sesh=lambda: SeshCCXTLive('binance'),
    isolation='thread',
    name='binance-main',
)
pool.run()
```

If a strategy raises in `on_data()`, the pool quarantines it (the others keep
running), calls its `on_crash(exc)` hook and emits a `StrategyErrorEvent` to your
`on_error` callback. There is no implicit auto-restart; your callback can call
`pool.restart_group(name)`.

A complete, network-free example (fake streaming session + scripted feed) lives
in `examples/live_pool_demo.py`.

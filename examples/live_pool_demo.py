"""
LivePool example - run several live strategies under one supervisor.

This example is fully self-contained and needs NO network: it uses a
streaming-capable fake session (FakeLiveSesh + a scripted FakeFeed) so it runs
deterministically. Swap the session factory for a real one (e.g.
SeshCCXTLive("binance", cfg)) to go live - the strategies do not change.

Concepts shown
--------------
- A GROUP is a set of strategies that share one Sesh (and therefore one feed
  and one broker), co-located on a single engine thread (single-writer safe).
- Strategies that need their own Sesh go in their own group.
- Isolation is per group: "thread" (light) or "process" (hard crash isolation).
- A crashing strategy is quarantined; the others keep running. The pool
  on_error callback is your "manual protocol" hook.

Run:
    python examples/live_pool_demo.py
"""

from tradetropy import Strategy, LivePool
from tradetropy.session.fake_live_sesh import FakeLiveSesh
from tradetropy.streaming import FakeFeed, TradeEvent


# =============================================================================
# A streaming-capable fake session (stands in for e.g. SeshCCXTLive)
# =============================================================================


class DemoStreamSesh(FakeLiveSesh):
    """FakeLiveSesh that advertises streaming and returns a scripted feed."""

    def __init__(self, events, **kw):
        super().__init__(feed_type="tick", **kw)
        self._events = events

    @property
    def supports_streaming(self) -> bool:
        return True

    def create_feed(self, **kwargs):
        return FakeFeed(self._events)


def make_btc_sesh() -> DemoStreamSesh:
    """Module-level factory: one shared BTCUSDT session with 10 scripted trades."""
    events = [TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0, side=1)
              for i in range(10)]
    sesh = DemoStreamSesh(events)
    sesh.set_symbol("BTCUSDT", initial_price=100.0, tick_size=0.01)
    return sesh


def make_eth_sesh() -> DemoStreamSesh:
    """A separate session (own account/venue) streaming ETHUSDT."""
    events = [TradeEvent("ETHUSDT", 2_000_000 + i, 50.0 + i, 1.0, side=1)
              for i in range(10)]
    sesh = DemoStreamSesh(events)
    sesh.set_symbol("ETHUSDT", initial_price=50.0, tick_size=0.01)
    return sesh


# =============================================================================
# Two simple strategies
# =============================================================================


class PrintEveryTrade(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=200)
        self.n = 0

    def on_data(self):
        self.n += 1


class EthWatcher(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("ETHUSDT", window_size=200)
        self.n = 0

    def on_data(self):
        self.n += 1


def main() -> None:
    def on_fail(ev):
        # Your "manual protocol": alert, persist state, maybe restart, etc.
        print(f"[on_error] group={ev.group} strat={ev.strategy_name} "
              f"exc={ev.exc!r} phase={ev.phase}")

    pool = LivePool(on_error=on_fail)

    # Group A: two strategies SHARE one BTCUSDT session (one feed + broker).
    pool.add_group(
        strategies=[PrintEveryTrade, PrintEveryTrade],
        sesh=make_btc_sesh,
        isolation="thread",
        name="binance-btc",
    )

    # Group B: a strategy with its OWN session (separate account/venue).
    pool.add_group(
        strategies=[EthWatcher],
        sesh=make_eth_sesh,
        isolation="thread",
        name="okx-eth",
    )

    # Blocking run: the finite scripted feeds drain, then the pool stops.
    pool.run(blocking=True)

    for name, st in pool.status().items():
        print(f"group={name:14s} strategies={st.n_strategies} "
              f"active={st.n_active} faulted={st.n_faulted} "
              f"alive={st.alive} bus_max_depth={st.bus_max_depth}")

    a = pool.group("binance-btc").strategies
    b = pool.group("okx-eth").strategies
    print(f"binance-btc on_data calls: {[s.n for s in a]}")   # -> [10, 10]
    print(f"okx-eth     on_data calls: {[s.n for s in b]}")   # -> [10]


if __name__ == "__main__":
    main()

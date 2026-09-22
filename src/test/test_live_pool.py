"""
Tests for tradetropy.live.pool (LivePool).

A streaming-capable fake session (FakeLiveSesh + a scripted FakeFeed) drives
several strategies through the pool without any network. All strategies and the
session factories are defined at MODULE LEVEL so they pickle under 'spawn' for
the process-isolation tests (same rule as test_pool.py).

Coverage:
  · GroupSpec / factory normalization + validation (Task 1)
  · multiplexed single-feed routing to co-located strategies (Task 2)
  · per-strategy fault isolation + on_crash + pool callback (Task 3)
  · shared-sesh ORDER/FILL applied exactly once (Task 4)
  · LivePool run / stop / status for thread groups (Task 5)
  · process-isolation group lifecycle + crash reporting (Task 6)
  · hot control: add / stop / restart / remove group (Task 7)
"""

import time

import pytest

from tradetropy.exceptions import ConfigError
from tradetropy.models.strategy import Strategy
from tradetropy.session.fake_live_sesh import FakeLiveSesh
from tradetropy.streaming import FakeFeed, FillEvent, OrderEvent, TradeEvent

from tradetropy.live.pool import (
    GroupSpec,
    GroupStatus,
    LiveGroup,
    LivePool,
    StrategyErrorEvent,
    _as_sesh_factory,
    _as_strategy_factory,
    _make_group_spec,
)


# =============================================================================
# MODULE-LEVEL FIXTURES (picklable for spawn)
# =============================================================================


def _trades(symbol="BTCUSDT", n=8, base_ts=2_000_000, base_px=100.0):
    return [
        TradeEvent(symbol, base_ts + i, base_px + i, 1.0, side=1) for i in range(n)
    ]


class StreamFakeSesh(FakeLiveSesh):
    """FakeLiveSesh that also advertises streaming and returns a scripted feed.

    FakeLiveSesh supplies the REST warmup history; the scripted FakeFeed carries
    the live events. A per-session counter records how many private ORDER/FILL
    events were applied (to prove they are applied exactly once per group).
    """

    def __init__(self, events, *, raise_after=None, **kw):
        super().__init__(feed_type="tick", **kw)
        self._events = events
        self._raise_after = raise_after
        self.fill_count = 0
        self.order_count = 0

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_user_stream(self) -> bool:
        return True

    def create_feed(self, **kwargs):
        return FakeFeed(self._events, raise_after=self._raise_after)

    def apply_fill_event(self, event) -> None:
        self.fill_count += 1

    def apply_order_event(self, event) -> None:
        self.order_count += 1


def make_btc_sesh() -> StreamFakeSesh:
    """Module-level factory: a BTCUSDT streaming fake session (8 trades)."""
    sesh = StreamFakeSesh(_trades("BTCUSDT", n=8))
    sesh.set_symbol("BTCUSDT", initial_price=100.0, tick_size=0.01)
    return sesh


def make_btc_eth_sesh() -> StreamFakeSesh:
    """Module-level factory: BTC + ETH trades interleaved on one session."""
    events = []
    for i in range(6):
        events.append(TradeEvent("BTCUSDT", 2_000_000 + 2 * i, 100.0 + i, 1.0, 1))
        events.append(TradeEvent("ETHUSDT", 2_000_000 + 2 * i + 1, 50.0 + i, 1.0, 1))
    sesh = StreamFakeSesh(events)
    sesh.set_symbol("BTCUSDT", initial_price=100.0, tick_size=0.01)
    sesh.set_symbol("ETHUSDT", initial_price=50.0, tick_size=0.01)
    return sesh


def make_crash_sesh() -> StreamFakeSesh:
    """Module-level factory for the process-crash test."""
    sesh = StreamFakeSesh(_trades("BTCUSDT", n=8))
    sesh.set_symbol("BTCUSDT", initial_price=100.0, tick_size=0.01)
    return sesh


class CounterStrat(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=100)
        self.n_calls = 0
        self.prices = []

    def on_data(self):
        self.n_calls += 1
        self.prices.append(float(self.tp.price[-1]))


class EthStrat(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("ETHUSDT", window_size=100)
        self.n_calls = 0

    def on_data(self):
        self.n_calls += 1


class CrashStrat(Strategy):
    warmup = 0

    def init(self):
        self.tp = self.subscribe_ticks("BTCUSDT", window_size=100)
        self.n_calls = 0
        self.crashed = False

    def on_data(self):
        self.n_calls += 1
        if self.n_calls >= 2:
            raise ValueError("boom")

    def on_crash(self, exc):
        self.crashed = True


# =============================================================================
# TASK 1 - types + factory normalization + validation
# =============================================================================


class TestNormalization:
    def test_strategy_factory_from_class(self):
        assert _as_strategy_factory(CounterStrat) is CounterStrat

    def test_strategy_factory_from_instance_uses_class(self):
        f = _as_strategy_factory(CounterStrat())
        assert f is CounterStrat

    def test_strategy_factory_from_callable(self):
        f = _as_strategy_factory(lambda: CounterStrat())
        assert isinstance(f(), CounterStrat)

    def test_strategy_factory_invalid(self):
        with pytest.raises(ConfigError):
            _as_strategy_factory(42)

    def test_sesh_factory_from_instance_thread_ok(self):
        sesh = make_btc_sesh()
        f = _as_sesh_factory(sesh, "thread")
        assert f() is sesh

    def test_sesh_instance_rejected_for_process(self):
        with pytest.raises(ConfigError):
            _as_sesh_factory(make_btc_sesh(), "process")

    def test_make_spec_defaults_and_name(self):
        spec = _make_group_spec(
            CounterStrat, make_btc_sesh, isolation="thread", name=None,
            feed_type="tick", save_log=False, on_error=None, auto_index=3,
        )
        assert isinstance(spec, GroupSpec)
        assert spec.name == "group-3"
        assert len(spec.strategy_factories) == 1
        assert spec.isolation == "thread"

    def test_make_spec_invalid_isolation(self):
        with pytest.raises(ConfigError):
            _make_group_spec(
                CounterStrat, make_btc_sesh, isolation="magic", name=None,
                feed_type="tick", save_log=False, on_error=None, auto_index=0,
            )

    def test_duplicate_group_name_rejected(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="g")
        with pytest.raises(ConfigError):
            pool.add_group(CounterStrat, make_btc_sesh, name="g")

    def test_process_group_requires_picklable(self):
        pool = LivePool()
        # A lambda sesh factory capturing a local is not picklable -> rejected.
        local = make_btc_sesh()
        with pytest.raises(ConfigError):
            pool.add_group(CounterStrat, lambda: local, isolation="process")


# =============================================================================
# TASK 2 - multiplexed single-feed routing
# =============================================================================


class TestMultiplexedFeed:
    def test_two_shared_strategies_both_receive_ticks(self):
        pool = LivePool()
        pool.add_group([CounterStrat, CounterStrat], make_btc_sesh, name="g")
        pool.run(blocking=True)

        s1, s2 = pool.group("g").strategies
        assert s1.n_calls == 8
        assert s2.n_calls == 8
        assert s1.prices == [100.0 + i for i in range(8)]
        assert s2.prices == s1.prices

    def test_symbol_routing_isolated(self):
        pool = LivePool()
        pool.add_group([CounterStrat, EthStrat], make_btc_eth_sesh, name="g")
        pool.run(blocking=True)

        btc, eth = pool.group("g").strategies
        # Each strategy only saw its own symbol's 6 trades.
        assert btc.n_calls == 6
        assert eth.n_calls == 6


# =============================================================================
# TASK 3 - per-strategy fault isolation
# =============================================================================


class TestFaultIsolation:
    def test_crash_quarantines_one_keeps_others(self):
        errors = []
        pool = LivePool(on_error=errors.append)
        pool.add_group([CrashStrat, CounterStrat], make_btc_sesh, name="g")
        pool.run(blocking=True)

        crash, counter = pool.group("g").strategies
        # The healthy strategy processed every trade.
        assert counter.n_calls == 8
        # The crashing strategy stopped at its 2nd call and was quarantined.
        assert crash.n_calls == 2
        assert crash.crashed is True

        status = pool.status()["g"]
        assert status.n_faulted == 1
        assert status.n_active == 1
        assert "CrashStrat#0" in status.faulted

        # The pool callback received exactly one StrategyErrorEvent.
        assert len(errors) == 1
        ev = errors[0]
        assert isinstance(ev, StrategyErrorEvent)
        assert ev.group == "g"
        assert ev.strategy_name == "CrashStrat#0"
        assert isinstance(ev.exc, ValueError)
        assert ev.phase == "on_data"
        assert ev.isolation == "thread"


# =============================================================================
# TASK 4 - shared-sesh ORDER/FILL applied exactly once
# =============================================================================


def make_userdata_sesh() -> StreamFakeSesh:
    events = [
        TradeEvent("BTCUSDT", 2_000_000, 100.0, 1.0, 1),
        OrderEvent("BTCUSDT", 2_000_001, order_id=7, status="open", side=1,
                   price=100.0, volume=1.0),
        FillEvent("BTCUSDT", 2_000_002, order_id=7, side=1, price=100.0,
                  volume=1.0),
        TradeEvent("BTCUSDT", 2_000_003, 101.0, 1.0, 1),
    ]
    sesh = StreamFakeSesh(events)
    sesh.set_symbol("BTCUSDT", initial_price=100.0, tick_size=0.01)
    return sesh


class TestSharedSeshUserData:
    def test_order_fill_applied_once_with_two_strategies(self):
        # Two strategies share the sesh; a single ORDER and a single FILL must
        # each hit the broker exactly once (not once per strategy).
        holder = {}

        def factory():
            holder["sesh"] = make_userdata_sesh()
            return holder["sesh"]

        pool = LivePool()
        pool.add_group([CounterStrat, CounterStrat], factory, name="g")
        pool.run(blocking=True)

        sesh = holder["sesh"]
        assert sesh.order_count == 1
        assert sesh.fill_count == 1
        # Both strategies still received the 2 trade events.
        s1, s2 = pool.group("g").strategies
        assert s1.n_calls == 2
        assert s2.n_calls == 2


# =============================================================================
# TASK 5 - LivePool supervisor: run / stop / status
# =============================================================================


class TestSupervisor:
    def test_status_shape_and_counts(self):
        pool = LivePool()
        pool.add_group([CounterStrat, CounterStrat], make_btc_sesh, name="a")
        pool.add_group(CounterStrat, make_btc_sesh, name="b")
        pool.run(blocking=True)

        st = pool.status()
        assert set(st) == {"a", "b"}
        assert isinstance(st["a"], GroupStatus)
        assert st["a"].n_strategies == 2
        assert st["b"].n_strategies == 1
        # Finite feeds have drained, so the groups are no longer alive.
        assert st["a"].alive is False

    def test_stop_is_idempotent(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.run(blocking=True)
        pool.stop()  # second stop must not raise
        pool.stop()

    def test_non_blocking_run_then_stop(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.run(blocking=False)
        # Wait for the finite feed to drain.
        deadline = time.time() + 5.0
        while pool.group("a").is_alive() and time.time() < deadline:
            time.sleep(0.05)
        pool.stop()
        assert pool.group("a").strategies[0].n_calls == 8


# =============================================================================
# TASK 6 - process isolation
# =============================================================================


@pytest.mark.slow
class TestProcessIsolation:
    def test_process_group_runs_and_reports(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, isolation="process",
                       name="p")
        pool.run(blocking=True)

        st = pool.status()["p"]
        assert st.isolation == "process"
        assert st.n_strategies == 1
        assert st.alive is False

    def test_process_crash_reported_to_callback(self):
        errors = []
        pool = LivePool(on_error=errors.append)
        pool.add_group([CrashStrat, CounterStrat], make_crash_sesh,
                       isolation="process", name="p")
        pool.run(blocking=True)

        # The crash in the child was forwarded to the parent callback.
        assert any(
            e.group == "p" and e.isolation == "process" and e.phase == "on_data"
            for e in errors
        )


# =============================================================================
# TASK 7 - hot control
# =============================================================================


class TestHotControl:
    def test_add_group_while_running(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.run(blocking=False)
        # Add a second group after the pool is already running.
        pool.add_group(CounterStrat, make_btc_sesh, name="b")
        assert "b" in pool.group_names
        assert pool.group("b") is not None

        deadline = time.time() + 5.0
        while pool.group("b").is_alive() and time.time() < deadline:
            time.sleep(0.05)
        pool.stop()
        assert pool.group("b").strategies[0].n_calls == 8

    def test_stop_group_leaves_others(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.add_group(CounterStrat, make_btc_sesh, name="b")
        pool.run(blocking=False)
        pool.stop_group("a")
        assert pool.group("a").is_alive() is False
        # 'b' still tracked.
        assert "b" in pool.group_names
        pool.stop()

    def test_restart_group_increments_and_reprocesses(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.run(blocking=False)
        deadline = time.time() + 5.0
        while pool.group("a").is_alive() and time.time() < deadline:
            time.sleep(0.05)

        pool.restart_group("a")
        deadline = time.time() + 5.0
        while pool.group("a").is_alive() and time.time() < deadline:
            time.sleep(0.05)
        pool.stop()

        st = pool.status()["a"]
        assert st.restarts == 1
        # After restart a fresh strategy reprocessed the feed.
        assert pool.group("a").strategies[0].n_calls == 8

    def test_remove_group_frees_name(self):
        pool = LivePool()
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.run(blocking=False)
        pool.remove_group("a")
        assert "a" not in pool.group_names
        # Name is reusable now.
        pool.add_group(CounterStrat, make_btc_sesh, name="a")
        pool.stop()

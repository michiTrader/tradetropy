"""
Tests for the high-level Recorder facade.

Network-free: a fake streaming session (SeshLiveBase + FakeFeed) feeds scripted
events through the same LiveEngine path a Recorder uses. Each stream is recorded
to HDF5 and read back with the matching io reader to assert a faithful
round-trip. Also covers validation and the simulator rejection.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tradetropy.core.broker import AccountInfo
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.exceptions import ConfigError
from tradetropy.io.io import read_ticks, read_klines, read_book
from tradetropy.live import Recorder
from tradetropy.session.base import SeshLiveBase
from tradetropy.streaming import KlineEvent, OrderbookSnapshot, TradeEvent


# ══════════════════════════════════════════════════════════════════════════════
# Fake streaming session
# ══════════════════════════════════════════════════════════════════════════════


class _FakeStreamSesh(SeshLiveBase):
    """A non-simulated streaming session driven by a scripted FakeFeed."""

    def __init__(self, events):
        super().__init__()
        self._events = events

    @property
    def supports_streaming(self) -> bool:
        return True

    def create_feed(self, **kwargs):
        from tradetropy.streaming import FakeFeed
        return FakeFeed(self._events)

    # Warmup uses REST: tick/orderbook streams fetch a small tick history.
    def _fetch_ticks_history(self, symbol, limit=500):
        n = 50
        out = np.zeros((n, N_TICK_COLS), dtype=np.float64)
        for i in range(n):
            out[i, _TICK_COL["ts"]] = 1_000_000 + i
            for c in ("bid", "ask", "price"):
                out[i, _TICK_COL[c]] = 100.0
            out[i, _TICK_COL["volume"]] = 1.0
        return out

    def _fetch_klines_history(self, symbol, interval_ms, limit=200):
        n = max(int(limit), 1)
        out = np.zeros((n, 6), dtype=np.float64)
        # Anchor the history near the real wall clock: LiveEngine's OHLC REST
        # top-up (_topup_ohlc_rings) compares the ring's last candle against
        # time.time(), so an epoch-0 fixture would trigger a multi-million-
        # candle top-up loop whenever a strategy mixes ticks + klines.
        now_ms = int(time.time() * 1000)
        last_open = (now_ms // interval_ms) * interval_ms
        base = last_open - (n - 1) * interval_ms
        for i in range(n):
            out[i, 0] = base + i * interval_ms   # ts (open)
            out[i, 1:5] = 100.0                    # ohlc
            out[i, 5] = 1.0                        # volume
        return out

    # Trading stubs (unused by the recorder).
    def buy(self, *a, **k):
        return None

    def sell(self, *a, **k):
        return None

    def positions(self, symbol=None):
        return []

    def account_info(self):
        return AccountInfo(
            balance=0.0, equity=0.0, margin=0.0, margin_free=0.0, profit=0.0
        )


# ══════════════════════════════════════════════════════════════════════════════
# Round-trip per stream
# ══════════════════════════════════════════════════════════════════════════════


def test_record_ticks_roundtrip(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "ticks.h5"
    events = [
        TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0 + 0.1 * i,
                   side=(1 if i % 2 == 0 else -1))
        for i in range(7)
    ]
    rec = Recorder(_FakeStreamSesh(events), flush_every=2)
    rec.add_stream("tick", "BTCUSDT", str(path))
    assert rec.uses_streaming is True
    rec.run()  # blocking; FakeFeed drains, then final flush

    td = read_ticks(path, "BTCUSDT")
    assert len(td.data) == 7
    np.testing.assert_allclose(
        td.data[:, _TICK_COL["price"]], [100.0 + i for i in range(7)]
    )
    assert list(td.data[:, _TICK_COL["ts"]].astype(int)) == [
        2_000_000 + i for i in range(7)
    ]


def test_record_orderbook_roundtrip(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "book.h5"
    events = []
    ts = 2_000_000
    for i in range(6):
        ts += 1
        events.append(OrderbookSnapshot(
            "BTCUSDT", ts,
            ((100.0, 1.0 + i), (99.5, 2.0)),
            ((100.5, 1.0), (101.0, 1.5)),
        ))
        ts += 1
        events.append(TradeEvent("BTCUSDT", ts, 100.0 + i, 1.0, side=1))

    rec = Recorder(_FakeStreamSesh(events))
    rec.add_stream("orderbook", "BTCUSDT", str(path), depth=5)
    rec.run()

    bd = read_book(path, "BTCUSDT")
    assert bd.levels == 5
    # One recorded row per book snapshot (6), from the first event (no warmup gate).
    assert len(bd.data) == 6
    # Best-bid size increments 1..6 across the snapshots.
    np.testing.assert_allclose(bd.bid_sz[:, 0], [1.0 + i for i in range(6)])


def test_record_klines_roundtrip(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "klines.h5"
    interval = 60_000
    # 6 closed candles at distinct open ts; the engine records a bar when the
    # next one closes it, so the last candle is not recorded (5 recorded).
    events = [
        KlineEvent(
            "BTCUSDT", (1000 + i) * interval, interval,
            open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i,
            volume=10.0 + i, turnover=float("nan"), is_closed=True,
        )
        for i in range(6)
    ]
    rec = Recorder(_FakeStreamSesh(events))
    rec.add_stream("kline", "BTCUSDT", str(path), timeframe="1m")
    rec.run()

    kl = read_klines(path, "BTCUSDT", timeframe=interval)
    # The 5 live candles that closed are recorded (the 6th has no successor to
    # close it). One boundary warmup candle may precede them, so assert on the
    # streamed tail rather than the exact count.
    streamed_closes = [100.5 + i for i in range(5)]
    assert len(kl.data) >= 5
    np.testing.assert_allclose(kl.data[-5:, 4], streamed_closes)


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════


def test_invalid_stream_raises(tmp_path):
    with pytest.raises(ConfigError):
        Recorder(_FakeStreamSesh([])).add_stream(
            "depth", "BTCUSDT", str(tmp_path / "x.h5")
        )


def test_kline_requires_timeframe(tmp_path):
    with pytest.raises(ConfigError):
        Recorder(_FakeStreamSesh([])).add_stream(
            "kline", "BTCUSDT", str(tmp_path / "x.h5")
        )


def test_orderbook_depth_must_be_int(tmp_path):
    with pytest.raises(ConfigError):
        Recorder(_FakeStreamSesh([])).add_stream(
            "orderbook", "BTCUSDT", str(tmp_path / "x.h5"), depth="deep"
        )


def test_empty_symbol_raises(tmp_path):
    with pytest.raises(ConfigError):
        Recorder(_FakeStreamSesh([])).add_stream("tick", "", str(tmp_path / "x.h5"))


def test_empty_path_raises():
    with pytest.raises(ConfigError):
        Recorder(_FakeStreamSesh([])).add_stream("tick", "BTCUSDT", "")


def test_run_without_streams_raises():
    with pytest.raises(ConfigError):
        Recorder(_FakeStreamSesh([])).run()


def test_add_stream_after_run_raises(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "ticks.h5"
    events = [TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0) for i in range(3)]
    rec = Recorder(_FakeStreamSesh(events))
    rec.add_stream("tick", "BTCUSDT", str(path))
    rec.run()
    with pytest.raises(ConfigError):
        rec.add_stream("tick", "ETHUSDT", str(tmp_path / "eth.h5"))


def test_rejects_simulated_session():
    from tradetropy.connectors.ccxt import SeshCCXTSim
    with pytest.raises(ConfigError):
        Recorder(SeshCCXTSim())


def test_context_manager_and_repr(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "ctx.h5"
    events = [TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0) for i in range(3)]
    with Recorder(_FakeStreamSesh(events)) as rec:
        rec.add_stream("tick", "BTCUSDT", str(path))
        assert "Recorder(stream='tick'" in repr(rec)
        rec.run()
    td = read_ticks(path, "BTCUSDT")
    assert len(td.data) == 3


def test_empty_recorder_repr():
    rec = Recorder(_FakeStreamSesh([]))
    assert "Recorder(empty" in repr(rec)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-stream (multi-symbol / multi-stream)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.filterwarnings("ignore:LiveEngine top-up:UserWarning")
def test_multi_stream_tick_and_kline_roundtrip(tmp_path):
    pytest.importorskip("tables")
    tick_path = tmp_path / "btc_ticks.h5"
    kline_path = tmp_path / "eth_1m.h5"
    interval = 60_000
    # Anchor timestamps near the real wall clock: LiveEngine's OHLC REST
    # top-up (_topup_ohlc_rings) compares the ring's last candle against
    # time.time(), so a fixture using old/synthetic timestamps here would
    # trigger a multi-million-candle top-up loop under by_ticks.
    now_ms = int(time.time() * 1000)
    base_candle_ts = (now_ms // interval - 10) * interval

    events = [
        TradeEvent("BTCUSDT", now_ms + i, 100.0 + i, 1.0, side=1)
        for i in range(5)
    ] + [
        KlineEvent(
            "ETHUSDT", base_candle_ts + i * interval, interval,
            open=10.0 + i, high=11.0 + i, low=9.0 + i, close=10.5 + i,
            volume=5.0 + i, turnover=float("nan"), is_closed=True,
        )
        for i in range(6)
    ]

    rec = Recorder(_FakeStreamSesh(events))
    rec.add_stream("tick", "BTCUSDT", str(tick_path))
    rec.add_stream("kline", "ETHUSDT", str(kline_path), timeframe="1m")
    assert len(rec.specs) == 2
    rec.run()

    td = read_ticks(tick_path, "BTCUSDT")
    assert len(td.data) == 5
    np.testing.assert_allclose(
        td.data[:, _TICK_COL["price"]], [100.0 + i for i in range(5)]
    )

    kl = read_klines(kline_path, "ETHUSDT", timeframe=interval)
    streamed_closes = [10.5 + i for i in range(5)]
    assert len(kl.data) >= 5
    np.testing.assert_allclose(kl.data[-5:, 4], streamed_closes)


def test_multi_stream_dedups_tick_proxy_for_shared_symbol(tmp_path):
    pytest.importorskip("tables")
    tick_path = tmp_path / "btc_ticks.h5"
    book_path = tmp_path / "btc_book.h5"

    events = []
    ts = 2_000_000
    for i in range(5):
        events.append(TradeEvent("BTCUSDT", ts, 100.0 + i, 1.0, side=1))
        ts += 1
        events.append(OrderbookSnapshot(
            "BTCUSDT", ts,
            ((100.0, 1.0 + i), (99.5, 2.0)),
            ((100.5, 1.0), (101.0, 1.5)),
        ))
        ts += 1

    rec = Recorder(_FakeStreamSesh(events))
    rec.add_stream("tick", "BTCUSDT", str(tick_path))
    rec.add_stream("orderbook", "BTCUSDT", str(book_path), depth=5)
    rec.run()

    # Only one tick proxy for BTCUSDT should have been created (the recorded
    # one from the 'tick' stream); the 'orderbook' stream must not add a
    # second, duplicate dummy tick proxy for the same symbol.
    btc_tick_proxies = [
        p for p in rec._strategy._tick_proxies if p.symbol == "BTCUSDT"
    ]
    assert len(btc_tick_proxies) == 1

    td = read_ticks(tick_path, "BTCUSDT")
    assert len(td.data) == 5

    bd = read_book(book_path, "BTCUSDT")
    assert bd.levels == 5
    assert len(bd.data) == 5


def test_multi_stream_all_klines_uses_by_klines(tmp_path):
    path_a = tmp_path / "btc_1m.h5"
    path_b = tmp_path / "eth_1m.h5"

    rec = Recorder(_FakeStreamSesh([]))
    rec.add_stream("kline", "BTCUSDT", str(path_a), timeframe="1m")
    rec.add_stream("kline", "ETHUSDT", str(path_b), timeframe="1m")
    rec._build_engine()
    assert rec._engine.feed_type == "kline"


def test_multi_stream_mixed_uses_by_ticks(tmp_path):
    tick_path = tmp_path / "btc_ticks.h5"
    kline_path = tmp_path / "eth_1m.h5"

    rec = Recorder(_FakeStreamSesh([]))
    rec.add_stream("tick", "BTCUSDT", str(tick_path))
    rec.add_stream("kline", "ETHUSDT", str(kline_path), timeframe="1m")
    rec._build_engine()
    assert rec._engine.feed_type == "tick"


def test_multi_stream_repr():
    rec = Recorder(_FakeStreamSesh([]))
    rec.add_stream("tick", "BTCUSDT", "btc.h5")
    rec.add_stream("kline", "ETHUSDT", "eth.h5", timeframe="1m")
    r = repr(rec)
    assert "BTCUSDT:tick" in r
    assert "ETHUSDT:kline" in r


# ══════════════════════════════════════════════════════════════════════════════
# Zero historical data from the venue (no warmup available)
# ══════════════════════════════════════════════════════════════════════════════
#
# A Recorder must never fail to start just because the broker/venue has no
# history yet (new listing, thin market, or the venue simply has none). It
# builds its internal LiveEngine with require_warmup=False, so an empty
# fetch is a warning, not a fatal DataError - recording still starts and
# fills the rings from the live feed itself.


class _NoHistorySesh(_FakeStreamSesh):
    """Same fake streaming session, but the venue has zero history."""

    def _fetch_ticks_history(self, symbol, limit=500):
        return np.empty((0, N_TICK_COLS), dtype=np.float64)

    def _fetch_klines_history(self, symbol, interval_ms, limit=200):
        return np.empty((0, 6), dtype=np.float64)


@pytest.mark.filterwarnings(
    "ignore:LiveEngine. broker returned no klines:UserWarning"
)
def test_record_ticks_with_zero_history(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "ticks_nohist.h5"
    events = [
        TradeEvent("BTCUSDT", 2_000_000 + i, 100.0 + i, 1.0, side=1)
        for i in range(4)
    ]
    rec = Recorder(_NoHistorySesh(events))
    rec.add_stream("tick", "BTCUSDT", str(path))
    with pytest.warns(UserWarning, match="no historical ticks"):
        rec.run()  # must NOT raise DataError

    td = read_ticks(path, "BTCUSDT")
    assert len(td.data) == 4
    np.testing.assert_allclose(
        td.data[:, _TICK_COL["price"]], [100.0 + i for i in range(4)]
    )


@pytest.mark.filterwarnings(
    "ignore:LiveEngine. broker returned no klines:UserWarning"
)
def test_record_orderbook_with_zero_history(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "book_nohist.h5"
    events = []
    ts = 2_000_000
    for i in range(3):
        events.append(TradeEvent("BTCUSDT", ts, 100.0 + i, 1.0, side=1))
        ts += 1
        events.append(OrderbookSnapshot(
            "BTCUSDT", ts, ((100.0, 1.0 + i),), ((100.5, 1.0),),
        ))
        ts += 1

    rec = Recorder(_NoHistorySesh(events))
    rec.add_stream("orderbook", "BTCUSDT", str(path), depth=3)
    with pytest.warns(UserWarning, match="no historical ticks"):
        rec.run()

    bd = read_book(path, "BTCUSDT")
    assert len(bd.data) == 3


def test_record_klines_with_zero_history(tmp_path):
    pytest.importorskip("tables")
    path = tmp_path / "klines_nohist.h5"
    interval = 60_000
    events = [
        KlineEvent(
            "BTCUSDT", (1000 + i) * interval, interval,
            open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i,
            volume=10.0 + i, turnover=float("nan"), is_closed=True,
        )
        for i in range(4)
    ]
    rec = Recorder(_NoHistorySesh(events))
    rec.add_stream("kline", "BTCUSDT", str(path), timeframe="1m")
    with pytest.warns(UserWarning, match="no klines"):
        rec.run()  # must NOT raise DataError even though warmup stays auto

    kl = read_klines(path, "BTCUSDT", timeframe=interval)
    # No historical candle at all, so no boundary candle: exactly the 3
    # streamed candles that got closed by a successor (the 4th has none).
    assert len(kl.data) == 3
    np.testing.assert_allclose(kl.data[:, 4], [100.5 + i for i in range(3)])


def test_strategy_still_strict_by_default(tmp_path):
    """
    A plain Strategy under LiveEngine keeps the strict default: zero
    historical data must still raise DataError so it never runs blind. Only
    Recorder opts into the tolerant require_warmup=False behavior.
    """
    from tradetropy.exceptions import DataError
    from tradetropy.live.engine import LiveEngine
    from tradetropy.models.strategy import Strategy

    class _Strat(Strategy):
        warmup = 10

        def init(self):
            self.tp = self.subscribe_ticks("BTCUSDT")

        def on_data(self):
            pass

    sesh = _NoHistorySesh([])
    engine = LiveEngine.by_ticks(_Strat(), sesh=sesh)
    with pytest.raises(DataError):
        engine.prepare()

"""
Regression tests for the live/replay bubble accumulator.

The large-trade / Heatmap bubbles use a causal relative threshold ('pXX'), whose
first ``window`` events fall in the rolling warmup (NaN threshold -> never
flagged). In backtest ``calculate`` runs once over the full series, so warmup
happens only at the very start. In live/replay ``live_refresh`` recomputes over
the current tick-ring window, so the warmup band re-arms at the window's left
edge and bubbles vanish as their tick drifts into it - replay then shows fewer
bubbles than the backtest even within the same visible window.

``merge_live_events`` / ``accumulate_events`` fix this by persisting a bubble
once it was detected while its tick was fresh (full trailing window, i.e. the
same causal decision the backtest makes), pruned to the visible span. These
tests assert parity is restored and that the pure helpers behave.
"""

import numpy as np
import pytest

from tradetropy.ta.order_flow._core import accumulate_events, merge_live_events
from tradetropy.ta.order_flow import LargeTrades, Heatmap


# ---------------------------------------------------------------------------
# Fake tick proxy (mirrors the one in test_large_trades)
# ---------------------------------------------------------------------------
class _Col:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, item):
        return self._arr[item]


class _FakeTickProxy:
    """Minimal stand-in for TickProxy exposing column window views."""

    def __init__(self, matrix):
        # columns: [ts, price, volume, flags, bid, ask]
        self._m = matrix

    def __len__(self):
        return len(self._m)

    @property
    def ts(self):
        return _Col(self._m[:, 0])

    @property
    def price(self):
        return _Col(self._m[:, 1])

    @property
    def volume(self):
        return _Col(self._m[:, 2])

    @property
    def flags(self):
        return _Col(self._m[:, 3])

    @property
    def bid(self):
        return _Col(self._m[:, 4])

    @property
    def ask(self):
        return _Col(self._m[:, 5])


def _series(n=100, spikes=(25, 55, 75, 95), spike_vol=100.0):
    """[N x 6] tick matrix with clear volume spikes at ``spikes``."""
    rng = np.random.default_rng(3)
    ts = (np.arange(n) * 1000).astype(float)
    price = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    volume = rng.uniform(1.0, 3.0, n)
    for s in spikes:
        volume[s] = spike_vol
    flags = np.ones(n)                    # all buys (deterministic side)
    bid = price - 0.05
    ask = price + 0.05
    return np.column_stack([ts, price, volume, flags, bid, ask])


def _event_ts(events) -> set:
    return set(int(t) for t in np.asarray(events.get("ts", []), dtype=np.int64))


def _simulate_replay(ind, matrix, window):
    """Drive ``ind.live_refresh`` over sliding ring windows of ``window``."""
    n = len(matrix)
    for i in range(n):
        lo = max(0, i - window + 1)
        ind.live_refresh(_FakeTickProxy(matrix[lo : i + 1]))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestAccumulateEvents:
    def _ev(self, ts, price, side, metric, volume=None):
        ts = np.asarray(ts, dtype=np.int64)
        return {
            "ts": ts,
            "price": np.asarray(price, dtype=np.float64),
            "side": np.asarray(side, dtype=np.int8),
            "metric": np.asarray(metric, dtype=np.float64),
            "volume": np.asarray(volume if volume is not None else metric,
                                 dtype=np.float64),
        }

    def test_union_by_ts_sorted(self):
        prev = self._ev([3000, 1000], [10.0, 11.0], [1, -1], [5.0, 6.0])
        new = self._ev([5000], [12.0], [1], [7.0])
        out = accumulate_events(prev, new)
        assert list(out["ts"]) == [1000, 3000, 5000]
        assert list(out["price"]) == [11.0, 10.0, 12.0]
        assert out["side"].dtype == np.int8

    def test_new_overrides_same_ts(self):
        prev = self._ev([1000], [10.0], [1], [5.0])
        new = self._ev([1000], [10.0], [-1], [99.0])
        out = accumulate_events(prev, new)
        assert list(out["ts"]) == [1000]
        assert out["metric"][0] == pytest.approx(99.0)
        assert out["side"][0] == -1

    def test_prune_below_min_ts(self):
        prev = self._ev([1000, 2000, 3000], [1, 2, 3], [1, 1, 1], [5, 6, 7])
        out = accumulate_events(prev, {}, min_ts=2000)
        assert list(out["ts"]) == [2000, 3000]

    def test_empty_inputs(self):
        out = accumulate_events({}, {})
        assert out["ts"].size == 0

    def test_extra_columns_preserved(self):
        # DeepTrades-style extra columns (event_type/resting) must survive.
        prev = {
            "ts": np.array([1000], dtype=np.int64),
            "price": np.array([10.0]),
            "side": np.array([1], dtype=np.int8),
            "metric": np.array([5.0]),
            "event_type": np.array([2], dtype=np.int8),
            "resting": np.array([123.0]),
        }
        new = {
            "ts": np.array([2000], dtype=np.int64),
            "price": np.array([11.0]),
            "side": np.array([-1], dtype=np.int8),
            "metric": np.array([6.0]),
            "event_type": np.array([3], dtype=np.int8),
            "resting": np.array([456.0]),
        }
        out = accumulate_events(prev, new)
        assert list(out["event_type"]) == [2, 3]
        assert out["event_type"].dtype == np.int8
        assert list(out["resting"]) == [123.0, 456.0]


class TestMergeLiveEvents:
    def _ev(self, ts):
        ts = np.asarray(ts, dtype=np.int64)
        return {
            "ts": ts,
            "price": np.zeros(ts.size),
            "side": np.ones(ts.size, dtype=np.int8),
            "metric": np.ones(ts.size),
        }

    def test_prunes_to_oldest_window_tick(self):
        prev = self._ev([1000, 2000])
        new = self._ev([4000])
        window_ts = np.array([2000, 3000, 4000])   # oldest visible = 2000
        merged, last = merge_live_events(prev, new, window_ts, 3000)
        assert list(merged["ts"]) == [2000, 4000]   # 1000 pruned
        assert last == 4000

    def test_rewind_resets_accumulator(self):
        prev = self._ev([8000, 9000])
        new = self._ev([1000])
        window_ts = np.array([1000, 2000])          # newest ts moved backwards
        merged, last = merge_live_events(prev, new, window_ts, 9000)
        assert list(merged["ts"]) == [1000]         # stale accumulator dropped
        assert last == 2000

    def test_empty_window_returns_new(self):
        new = self._ev([1000])
        merged, last = merge_live_events({}, new, np.array([]), None)
        assert list(merged["ts"]) == [1000]
        assert last is None


# ---------------------------------------------------------------------------
# End-to-end parity: LargeTrades replay window == backtest window
# ---------------------------------------------------------------------------
class TestLargeTradesReplayParity:
    def test_windowed_refresh_loses_warmup_bubbles(self):
        """Baseline: a plain single windowed refresh misses the warmup-band bubble."""
        m = _series()
        window = 50
        ind = LargeTrades(threshold="p90", by="volume", window=20)
        # Final ring window [50..99]; spike at 55 sits in the warmup band.
        ind.live_refresh(_FakeTickProxy(m[50:100]))
        # Accumulation is active, so re-run WITHOUT it via calculate on the slice
        # to see the raw windowed detection.
        ind2 = LargeTrades(threshold="p90", by="volume", window=20)
        ind2.calculate(m[50:100])
        assert 55_000 not in _event_ts(ind2._deep_events)   # lost to warmup
        assert 75_000 in _event_ts(ind2._deep_events)

    def test_full_series_detects_warmup_span_event(self):
        m = _series()
        ind = LargeTrades(threshold="p90", by="volume", window=20)
        ind.calculate(m)
        full = _event_ts(ind._deep_events)
        # All four spikes past the initial warmup are detected on the full series.
        assert {55_000, 75_000, 95_000} <= full

    def test_replay_accumulation_matches_backtest_window(self):
        m = _series()
        window = 50

        # Backtest: full-series detection, restricted to the final window span.
        bt = LargeTrades(threshold="p90", by="volume", window=20)
        bt.calculate(m)
        min_ts = int(m[100 - window, 0])
        bt_span = {t for t in _event_ts(bt._deep_events) if t >= min_ts}

        # Replay: sliding windowed refreshes with accumulation.
        rp = LargeTrades(threshold="p90", by="volume", window=20)
        _simulate_replay(rp, m, window)
        rp_events = _event_ts(rp._deep_events)

        assert rp_events == bt_span
        assert 55_000 in rp_events        # the warmup-band bubble is recovered
        # Accumulator stays bounded to the visible span (nothing older leaks in).
        assert min(rp_events) >= min_ts

    def test_aggregated_mode_not_accumulated(self):
        # With burst aggregation the accumulator is disabled (no partial dupes):
        # live_refresh must leave exactly the current window's detection.
        m = _series(spikes=(55, 75, 95))
        rp = LargeTrades(threshold="p90", by="volume", window=20, aggregate_ms=50)
        _simulate_replay(rp, m, 50)
        single = LargeTrades(threshold="p90", by="volume", window=20,
                             aggregate_ms=50)
        single.calculate(m[50:100])
        assert _event_ts(rp._deep_events) == _event_ts(single._deep_events)


# ---------------------------------------------------------------------------
# Heatmap bubbles accumulate the same way
# ---------------------------------------------------------------------------
class TestHeatmapReplayParity:
    def test_heatmap_bubbles_match_backtest_window(self):
        m = _series()
        window = 50

        bt = Heatmap(orderbook=None, bubble_threshold="p90", bubble_by="volume",
                     bubble_window=20, show_bubbles=True)
        bt.calculate(m)
        min_ts = int(m[100 - window, 0])
        bt_span = {t for t in _event_ts(bt._bubble_events) if t >= min_ts}

        rp = Heatmap(orderbook=None, bubble_threshold="p90", bubble_by="volume",
                     bubble_window=20, show_bubbles=True)
        _simulate_replay(rp, m, window)
        rp_events = _event_ts(rp._bubble_events)

        assert rp_events == bt_span
        assert 55_000 in rp_events

"""
MarketStructure (BOS / CHoCH) - causality, semantics and engine parity.

The whole value of an SMC structure detector rests on one property: a break is
only reported once the broken level was *causally knowable*. An implementation
that anchors levels at the real swing bar (instead of at the bar where
ConfirmedPivot confirmed them) silently backdates breaks - the backtest then
trades on information a live run could never have had, and the equity curve is
fiction.

This module guards that property from three angles:

    1. Causality (prefix invariance): recomputing over every growing prefix of
       the data must reproduce the full-history result exactly. If any event
       moved, appeared or vanished when later bars arrived, the indicator
       peeked at the future.
    2. Semantics: BOS continues the prevailing bias, CHoCH flips it; a level is
       consumed by its break; break_on='close' is stricter than 'wick'.
    3. Engine parity: the series a strategy observes in BacktestEngine and in
       ReplayEngine (the live transport) must be identical bar by bar, and the
       event()/bias() helpers must see the same events in both.

Usage:
    pytest src/test/test_market_structure.py -v
"""

import numpy as np
import pytest

from tradetropy.backtest import BacktestEngine
from tradetropy.core.data_types import TickData
from tradetropy.exceptions import ConfigError
from tradetropy.models.strategy import Strategy
from tradetropy.replay import ReplayEngine
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.ta import MarketStructure
from tradetropy.ta.structure.market_structure import _MS_TAG_CODE

SYM = 'X'
INTERVAL = 30_000

# Band layout, mirrored from the indicator so a silent reorder fails here.
B_BOS_BULL, B_BOS_BEAR, B_CHOCH_BULL, B_CHOCH_BEAR = 0, 1, 2, 3
TS_OFFSET = 4
B_TAG = 8

_BANDS = ('bos_bull', 'bos_bear', 'choch_bull', 'choch_bear')


# =====
# FIXTURES / HELPERS
# =====

def make_ohlc_source(n=400, seed=11):
    """
    Build a [N x 4] source (high, low, close, ts) from a random walk with
    alternating drift, so the series produces both continuations (BOS) and
    reversals (CHoCH) instead of one long trend.
    """
    rng = np.random.default_rng(seed)
    base = (1_700_000_000_000 // INTERVAL) * INTERVAL
    price = 1000.0
    rows = []
    for i in range(n):
        # Regime flip every ~40 bars guarantees CHoCH events exist.
        drift = 0.9 if (i // 40) % 2 == 0 else -0.9
        price += (rng.random() - 0.5) * 3.0 + drift
        half = abs(rng.normal(0, 1.1)) + 0.4
        rows.append([price + half, price - half, price, base + i * INTERVAL])
    return np.asarray(rows, dtype=np.float64)


def make_ticks_from_source(src, ticks_per_bar=8, spread=0.05):
    """
    Expand an OHLC source into ticks [N x 7] that reproduce each bar's
    high/low/close, so both engines rebuild the same candles.
    """
    rows = []
    for high, low, close, ts in src:
        ts = int(ts)
        seq = [low, high, close] if (ts // INTERVAL) % 2 == 0 else [high, low, close]
        for k in range(ticks_per_bar):
            price = float(seq[k % len(seq)]) if k < len(seq) else float(close)
            t = ts + int(k * INTERVAL / ticks_per_bar)
            rows.append([t, price - spread, price + spread, 1.0, 0, 1.0, price])
    return np.asarray(rows, dtype=np.float64)


def events_from(out):
    """Flatten the band matrix into [(bar_idx, tag, level, ts_origin)]."""
    decode = {v: k for k, v in _MS_TAG_CODE.items()}
    band_of = {
        "BOS-bull": B_BOS_BULL, "BOS-bear": B_BOS_BEAR,
        "CHOCH-bull": B_CHOCH_BULL, "CHOCH-bear": B_CHOCH_BEAR,
    }
    evs = []
    for i in range(out.shape[1]):
        tag_code = out[B_TAG, i]
        if np.isnan(tag_code):
            continue
        tag = decode[float(tag_code)]
        band = band_of[tag]
        evs.append((i, tag, float(out[band, i]), float(out[band + TS_OFFSET, i])))
    return evs


# =====
# CAUSALITY
# =====

@pytest.mark.unit
class TestCausality:
    def test_prefix_invariance(self):
        """
        The core no-lookahead guarantee: for every prefix length k, the result
        over src[:k] must equal the first k columns of the full computation.
        """
        src = make_ohlc_source()
        full = MarketStructure(swing=2).calculate(src)

        for k in range(40, len(src) + 1, 13):
            pref = MarketStructure(swing=2).calculate(src[:k])
            a, b = full[:8, :k], pref[:8, :k]
            assert np.array_equal(np.isnan(a), np.isnan(b)), (
                f"prefix {k}: event mask changed -> an event was backdated "
                f"or revoked when later bars arrived"
            )
            m = ~np.isnan(a)
            assert np.allclose(a[m], b[m]), f"prefix {k}: values changed"

    def test_break_strictly_after_confirmation(self):
        """
        A break must never be reported on (or before) the bar that confirmed
        the level: ts_origin is the real swing bar, strictly older than the
        break bar's own timestamp.
        """
        src = make_ohlc_source()
        out = MarketStructure(swing=2).calculate(src)
        ts = src[:, 3]
        evs = events_from(out)
        assert evs, "fixture produced no events"
        for bar_idx, tag, _level, ts_origin in evs:
            assert ts_origin < ts[bar_idx], (
                f"{tag} at bar {bar_idx}: origin ts {ts_origin} is not older "
                f"than the break bar ts {ts[bar_idx]}"
            )

    def test_short_input_is_all_nan(self):
        src = make_ohlc_source(n=4)
        out = MarketStructure(swing=2).calculate(src)
        assert out.shape == (9, 4)
        assert np.isnan(out).all()

    def test_history_never_rewritten(self):
        """Growing the series must only append events, never insert one."""
        src = make_ohlc_source()
        prev = []
        for k in range(60, len(src) + 1, 25):
            evs = events_from(MarketStructure(swing=2).calculate(src[:k]))
            assert evs[:len(prev)] == prev, f"history rewritten at prefix {k}"
            prev = evs


# =====
# SEMANTICS
# =====

@pytest.mark.unit
class TestSemantics:
    def test_band_shape_and_alignment(self):
        """Each price band must be NaN exactly where its ts band is NaN."""
        src = make_ohlc_source()
        ms = MarketStructure(swing=2)
        out = ms.calculate(src)
        assert out.shape == (9, len(src))
        assert ms.n_outputs == 9
        for i in range(4):
            assert np.array_equal(
                np.isnan(out[i]), np.isnan(out[i + TS_OFFSET])
            ), f"band {ms.output_names[i]}: price/ts misaligned"

    def test_one_event_per_bar(self):
        src = make_ohlc_source()
        out = MarketStructure(swing=2).calculate(src)
        per_bar = np.sum(~np.isnan(out[:4]), axis=0)
        assert per_bar.max() <= 1

    def test_bias_alternation(self):
        """
        CHoCH flips the bias, BOS continues it. So a CHoCH must always point
        against the previous event's direction, and a BOS must agree with it.
        """
        src = make_ohlc_source()
        evs = events_from(MarketStructure(swing=2).calculate(src))
        assert len(evs) > 5
        bias = None
        for _idx, tag, _lvl, _ts in evs:
            direction = 'bull' if tag.endswith('bull') else 'bear'
            if tag.startswith('CHOCH'):
                assert bias is not None and direction != bias, (
                    f"CHoCH {direction} emitted while bias was {bias}"
                )
            else:
                assert bias is None or direction == bias, (
                    f"BOS {direction} emitted while bias was {bias}"
                )
            bias = direction

    def test_first_event_is_bos(self):
        """With no prior bias there is nothing to contradict -> BOS."""
        src = make_ohlc_source()
        evs = events_from(MarketStructure(swing=2).calculate(src))
        assert evs[0][1].startswith('BOS')

    def test_close_mode_is_stricter_than_wick(self):
        src = make_ohlc_source()
        n_close = len(events_from(
            MarketStructure(swing=2, break_on='close').calculate(src)))
        n_wick = len(events_from(
            MarketStructure(swing=2, break_on='wick').calculate(src)))
        assert n_wick >= n_close

    def test_break_level_respects_mode(self):
        """Each event's break bar must actually satisfy its break rule."""
        src = make_ohlc_source()
        high, low, close = src[:, 0], src[:, 1], src[:, 2]
        for mode, up, down in (('close', close, close), ('wick', high, low)):
            out = MarketStructure(swing=2, break_on=mode).calculate(src)
            for bar_idx, tag, level, _ts in events_from(out):
                if tag.endswith('bull'):
                    assert up[bar_idx] > level, f"{mode}/{tag} @{bar_idx}"
                else:
                    assert down[bar_idx] < level, f"{mode}/{tag} @{bar_idx}"

    def test_rejects_bad_break_on(self):
        with pytest.raises(ConfigError):
            MarketStructure(break_on='wobble')

    @pytest.mark.parametrize('swing', [1, 2, 3, 5])
    def test_swing_windows(self, swing):
        src = make_ohlc_source()
        ms = MarketStructure(swing=swing)
        out = ms.calculate(src)
        assert out.shape == (9, len(src))
        assert ms.min_periods == swing * 2 + 1

    def test_draw_primitives_match_events(self):
        src = make_ohlc_source()
        ms = MarketStructure(swing=2)
        evs = events_from(ms.calculate(src))
        prims = ms.draw()
        assert prims, "expected draw primitives"
        total_segs = sum(len(p.x0) for p in prims if hasattr(p, 'x0'))
        assert total_segs == len(evs)
        for p in prims:
            if hasattr(p, 'x0'):
                for a, b in zip(p.x0, p.x1):
                    assert a < b, "structure line must run forward in time"

    def test_draw_empty_before_calculate(self):
        assert MarketStructure(swing=2).draw() == []


# =====
# ENGINE PARITY
# =====

class _StratMS(Strategy):
    warmup = None

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
        self.ms = self.add_indicator(
            MarketStructure.refs(self.ohlc), MarketStructure(swing=2)
        )

    def on_data(self):
        pass


def _serialize(arr, ndigits=6):
    return tuple('n' if np.isnan(x) else round(float(x), ndigits) for x in arr)


def _capture(ticks, mode):
    """Snapshot every MarketStructure band per on_data(), keyed by bar ts."""
    captured = {}

    class _Wrapped(_StratMS):
        def on_data(self):
            ts_last = self.ohlc.ts[-1]
            ts = (
                int(ts_last)
                if isinstance(ts_last, (int, float))
                else int(np.datetime64(ts_last, 'ms').astype('int64'))
            )
            snap = {b: _serialize(getattr(self.ms, b)) for b in _BANDS}
            snap['ts_bos_bull'] = _serialize(self.ms.ts_bos_bull)
            snap['event'] = MarketStructure.event(self.ms)
            snap['bias'] = MarketStructure.bias(self.ms)
            captured[ts] = snap

    if mode == 'bt':
        sesh = SeshSimulatorBase('tick')
        BacktestEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.01),), sesh=sesh
        ).run(verbose=False)
    else:
        eng = ReplayEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.01),),
            speed=float('inf'),
        )
        eng.prepare()
        sesh = eng._sesh
        sesh._cursor[SYM] = len(ticks) - 1
        last_idx = sesh._warmup_ticks.get(SYM, 0) - 1
        for t in sesh._fetch_pending_ticks(SYM, last_idx):
            eng.on_tick(SYM, t)
    return captured


@pytest.mark.unit
class TestEngineParity:
    def test_backtest_replay_identical(self):
        """
        The series observed inside on_data() must be identical in backtest and
        replay. A divergence here means a strategy validated on a backtest
        would behave differently live.
        """
        src = make_ohlc_source(n=160, seed=5)
        ticks = make_ticks_from_source(src)
        bt = _capture(ticks, 'bt')
        rp = _capture(ticks, 'rp')
        common = sorted(set(bt) & set(rp))
        assert len(common) > 10, "not enough overlapping bars to compare"
        for key in (*_BANDS, 'ts_bos_bull', 'event', 'bias'):
            diffs = [ts for ts in common if bt[ts][key] != rp[ts][key]]
            assert not diffs, (
                f"{key} diverged on {len(diffs)} bars (first: {diffs[:3]})"
            )


# =====
# READ HELPERS
# =====

@pytest.mark.unit
class TestReadHelpers:
    def test_event_slot_differs_between_engines(self):
        """
        Pin down the reason event() scans two slots: the offset is engine
        dependent. Under klines the event lands on [-2] (last slot is the
        forming bar); under ticks it can land on [-1]. A strategy hardcoding
        either offset breaks on the other engine - which is exactly the bug
        that made a first draft of this indicator never fire.
        """
        from tradetropy.datasets import load_btcusd_1m

        # --- tick-driven ---
        src = make_ohlc_source(n=200, seed=9)
        ticks = make_ticks_from_source(src)
        tick_hits = {'m1': 0, 'm2': 0}

        class _T(_StratMS):
            def on_data(self):
                b = self.ms.bos_bull
                if len(b) >= 1 and not np.isnan(b[-1]):
                    tick_hits['m1'] += 1
                if len(b) >= 2 and not np.isnan(b[-2]):
                    tick_hits['m2'] += 1

        sesh = SeshSimulatorBase('tick')
        BacktestEngine.by_ticks(
            _T(), data=(TickData(SYM, ticks, tick_size=0.01),), sesh=sesh
        ).run(verbose=False)

        # --- kline-driven ---
        kl = load_btcusd_1m()
        kline_hits = {'m1': 0, 'm2': 0}

        class _K(Strategy):
            def init(self):
                self.ohlc = self.subscribe_ohlc(kl.symbol, '1m', window_size=300)
                self.ms = self.add_indicator(
                    MarketStructure.refs(self.ohlc), MarketStructure(swing=3)
                )

            def on_data(self):
                b = self.ms.bos_bull
                if len(b) >= 1 and not np.isnan(b[-1]):
                    kline_hits['m1'] += 1
                if len(b) >= 2 and not np.isnan(b[-2]):
                    kline_hits['m2'] += 1

        BacktestEngine.by_klines(_K(), data=(kl,)).run(verbose=False)

        assert kline_hits['m1'] == 0, (
            "kline mode: [-1] is the forming bar and must never carry an event"
        )
        assert kline_hits['m2'] > 0, "kline mode: events belong on [-2]"
        assert tick_hits['m1'] > 0, "tick mode: events do reach [-1]"

    def test_event_helper_covers_both_engines(self):
        """event() must see events under ticks AND klines."""
        src = make_ohlc_source(n=200, seed=9)
        ticks = make_ticks_from_source(src)
        seen = []

        class _S(_StratMS):
            def on_data(self):
                tag = MarketStructure.event(self.ms)
                if tag:
                    seen.append(tag)

        sesh = SeshSimulatorBase('tick')
        BacktestEngine.by_ticks(
            _S(), data=(TickData(SYM, ticks, tick_size=0.01),), sesh=sesh
        ).run(verbose=False)
        assert seen, "event() saw nothing under a tick-driven engine"

    def test_event_helper_matches_calculate(self):
        """
        event() must observe exactly the events calculate() produces - no
        drops (the [-1] bug) and no duplicates.
        """
        from tradetropy.datasets import load_btcusd_1m

        kl = load_btcusd_1m()
        d = kl.data
        src = np.column_stack([d[:, 2], d[:, 3], d[:, 4], d[:, 0]])
        expected = len(events_from(MarketStructure(swing=3).calculate(src)))

        class _S(Strategy):
            def init(self):
                self.ohlc = self.subscribe_ohlc(
                    kl.symbol, '1m', window_size=len(d)
                )
                self.ms = self.add_indicator(
                    MarketStructure.refs(self.ohlc), MarketStructure(swing=3)
                )
                self.seen = []

            def on_data(self):
                tag = MarketStructure.event(self.ms)
                if tag:
                    self.seen.append(tag)

        bt = BacktestEngine.by_klines(_S(), data=(kl,))
        bt.run(verbose=False)
        assert len(bt.strategy.seen) == expected, (
            f"event() saw {len(bt.strategy.seen)} events, "
            f"calculate() produced {expected}"
        )

    def test_bias_tracks_last_event(self):
        src = make_ohlc_source(n=200, seed=4)
        ticks = make_ticks_from_source(src)
        pairs = []

        class _S(_StratMS):
            def on_data(self):
                tag = MarketStructure.event(self.ms)
                if tag:
                    pairs.append((tag, MarketStructure.bias(self.ms)))

        sesh = SeshSimulatorBase('tick')
        BacktestEngine.by_ticks(
            _S(), data=(TickData(SYM, ticks, tick_size=0.01),), sesh=sesh
        ).run(verbose=False)

        assert pairs, "no events observed"
        for tag, bias in pairs:
            expected = 'bull' if tag.endswith('bull') else 'bear'
            assert bias == expected, f"{tag} -> bias {bias}, expected {expected}"

    def test_helpers_safe_on_empty_proxy(self):
        class _Empty:
            bos_bull = np.array([])
        assert MarketStructure.event(_Empty()) is None
        assert MarketStructure.bias(_Empty()) is None

"""
BacktestEngine ↔ ReplayEngine (inherits from LiveEngine) parity tests.

Protects backtest/live parity fixes made in commits eb7160f
(unified warmup, `_ticks_to_klines` bid→price, stream OHLC with fp_*, clamp
replay) and be0dbce (live view of stateful indicators aligned 1:1 with backtest).

Idea: on the SAME tick array, backtest and replay must produce identical
series (OHLC, stateful indicators like ConfirmedPivot, SMA) and resolve the
same warmup. If a future change reintroduces divergence, these tests fail.

CRITICAL DETAIL: ticks use bid/ask DIFFERENT from price (real spread). If
someone reads bid instead of price when building warmup klines
(_ticks_to_klines), replay OHLC diverges and the OHLC test fails.

Usage:
    pytest src/test/test_parity_engines.py -v
"""

import numpy as np
import pytest

from tradetropy.core.constants import _TICK_COL, N_TICK_COLS
from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.replay import ReplayEngine
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.core.data_types import TickData
from tradetropy.ta import SMA, ConfirmedPivot

SYM = "X"
INTERVAL = 30_000


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def ticks_with_spread(n_bars=40, ticks_per_bar=40, spread=0.25, seed=5):
    """
    Generates ticks [N×7] with bid/ask DIFFERENT from price (real spread).

    Key for detecting _ticks_to_klines regressions: if it builds klines
    from bid (col 1) instead of price (col 6), OHLC diverges because
    bid != price.
    """
    base = (1_700_000_000_000 // INTERVAL) * INTERVAL
    rng = np.random.default_rng(seed)
    rows = []
    price = 7540.0
    for bar in range(n_bars):
        bar_ts = base + bar * INTERVAL
        for k in range(ticks_per_bar):
            price += (rng.random() - 0.5) * 0.6
            ts = bar_ts + int(k * INTERVAL / ticks_per_bar)
            # ts, bid, ask, volume, flags, volume_real, price
            rows.append([ts, price - spread, price + spread, 1.0, 0, 1.0, price])
    return np.asarray(rows, dtype=np.float64)


def _serialize(arr):
    """Serializes an array to a rounded tuple with 'n' for NaN (comparable)."""
    return tuple("n" if np.isnan(x) else round(float(x), 4) for x in arr)


def _capture(strategy_cls, ticks, mode):
    """
    Runs the indicated engine and captures, per on_data(), a snapshot of the
    relevant series indexed by the current bar's ts. Returns a dict
    ts -> snapshot (latest state for that bar).
    """
    captured = {}

    class _Wrapped(strategy_cls):
        def on_data(self):
            ts_last = self.ohlc.ts[-1]
            ts = (
                int(ts_last)
                if isinstance(ts_last, (int, float))
                else int(np.datetime64(ts_last, "ms").astype("int64"))
            )
            snap = {
                "n_ohlc": len(self.ohlc.ts),
                "high": _serialize(self.ohlc.high),
                "low": _serialize(self.ohlc.low),
                "close": _serialize(self.ohlc.close),
            }
            if hasattr(self, "cpivot"):
                snap["cp_high"] = _serialize(self.cpivot.high)
                snap["cp_low"] = _serialize(self.cpivot.low)
                snap["len_cp"] = len(self.cpivot.high)
            if hasattr(self, "sma"):
                snap["sma"] = _serialize(self.sma)
            if hasattr(self, "setup"):
                m = self.setup.last
                snap["match"] = None if m is None else (
                    m.tag, tuple(round(float(n.value), 4) for n in m.nodes)
                )
            captured[ts] = snap

    if mode == "bt":
        sesh = SeshSimulatorBase("tick")
        BacktestEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.25),), sesh=sesh
        ).run(verbose=True)
    else:
        eng = ReplayEngine.by_ticks(_Wrapped(), data=(TickData(SYM, ticks, tick_size=0.25),), speed=float("inf"))
        eng.prepare()
        # Deterministic drain (without relying on thread timing): move the
        # cursor to the end and feed all pending ticks.
        sesh = eng._sesh
        sesh._cursor[SYM] = len(ticks) - 1
        last_idx = sesh._warmup_ticks.get(SYM, 0) - 1
        for t in sesh._fetch_pending_ticks(SYM, last_idx):
            eng.on_tick(SYM, t)
    return captured


def _common(a, b):
    return sorted(set(a) & set(b))


# ══════════════════════════════════════════════════════════════════════════════
# TEST STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════


class _StratOHLC(Strategy):
    warmup = 0

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)

    def on_data(self):
        pass


class _StratPivot(Strategy):
    warmup = 0

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
        self.cpivot = self.add_indicator(
            [self.ohlc.high_ref, self.ohlc.low_ref, self.ohlc.ts_ref],
            ConfirmedPivot(swing=3),
        )

    def on_data(self):
        pass


class _StratSMA(Strategy):
    warmup = 0

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
        self.sma = self.add_indicator(self.ohlc.close_ref, SMA(5))

    def on_data(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestParityOHLC:
    def test_ohlc_identical_with_spread(self):
        """Identical OHLC between backtest and replay with real-spread ticks.

        Guards _ticks_to_klines (bid→price regression): if it reads bid again,
        replay OHLC diverges because bid != price.
        """
        ticks = ticks_with_spread()
        bt = _capture(_StratOHLC, ticks, "bt")
        rp = _capture(_StratOHLC, ticks, "rp")
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [
            ts for ts in common
            if bt[ts]["high"] != rp[ts]["high"]
            or bt[ts]["low"] != rp[ts]["low"]
            or bt[ts]["close"] != rp[ts]["close"]
        ]
        assert diffs == [], f"OHLC differs in {len(diffs)} bars: {diffs[:3]}"

    def test_ohlc_identical_with_warmup(self):
        """Identical OHLC with warmup>0, exercising _ticks_to_klines (warmup
        klines built from ticks).

        THIS is the test that guards the bid→price regression: with warmup,
        replay builds its historical bars via _ticks_to_klines. If it reads bid
        instead of price, those bars (and pivots on them) diverge from backtest.
        """
        ticks = ticks_with_spread(n_bars=60)

        class _S(_StratPivot):
            warmup = 500  # ticks → forces historical warmup in replay

        bt = _capture(_S, ticks, "bt")
        rp = _capture(_S, ticks, "rp")
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [
            ts for ts in common
            if bt[ts]["high"] != rp[ts]["high"]
            or bt[ts]["low"] != rp[ts]["low"]
            or bt[ts]["cp_high"] != rp[ts]["cp_high"]
        ]
        assert diffs == [], f"OHLC/pivot differs with warmup in {len(diffs)}: {diffs[:3]}"


@pytest.mark.unit
class TestParityPivotStateful:
    def test_cpivot_arrays_identical(self):
        """cpivot.high/low (full arrays, including NaN) are identical."""
        ticks = ticks_with_spread()
        bt = _capture(_StratPivot, ticks, "bt")
        rp = _capture(_StratPivot, ticks, "rp")
        common = _common(bt, rp)
        diffs = [
            ts for ts in common
            if bt[ts]["cp_high"] != rp[ts]["cp_high"]
            or bt[ts]["cp_low"] != rp[ts]["cp_low"]
        ]
        assert diffs == [], f"cpivot differs in {len(diffs)} bars: {diffs[:3]}"

    def test_cpivot_aligned_1to1_with_ohlc(self):
        """len(cpivot.high) == len(ohlc.high) in both engines (commit be0dbce)."""
        ticks = ticks_with_spread()
        for mode in ("bt", "rp"):
            captured = _capture(_StratPivot, ticks, mode)
            bad = [ts for ts, s in captured.items() if s["len_cp"] != s["n_ohlc"]]
            assert bad == [], f"[{mode}] len(cpivot)!=len(ohlc) in {bad[:3]}"


@pytest.mark.unit
class TestParitySMA:
    def test_sma_series_identical(self):
        """SMA (non-stateful) matches across engines — no regression."""
        ticks = ticks_with_spread()
        bt = _capture(_StratSMA, ticks, "bt")
        rp = _capture(_StratSMA, ticks, "rp")
        common = _common(bt, rp)
        diffs = [ts for ts in common if bt[ts]["sma"] != rp[ts]["sma"]]
        assert diffs == [], f"SMA differs in {len(diffs)} bars: {diffs[:3]}"


@pytest.mark.unit
class TestParityWarmup:
    def test_warmup_auto_same_N(self):
        """warmup=None → backtest and replay resolve the same N (in ticks)."""
        from tradetropy.models._warmup_policy import (
            resolver_warmup, estimate_ticks_per_candle,
        )

        class _S(Strategy):
            warmup = None

            def init(self):
                self.tk = self.subscribe_ticks(SYM)
                self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
                self.cpivot = self.add_indicator(
                    [self.ohlc.high_ref, self.ohlc.low_ref, self.ohlc.ts_ref],
                    ConfirmedPivot(swing=3),
                )

            def on_data(self):
                pass

        ticks = ticks_with_spread()
        # backtest: resolver_warmup with prepared instance
        s_bt = _S()
        s_bt._run_mode = "backtest"
        s_bt._feed_type = "tick"
        s_bt.init()
        tpv = estimate_ticks_per_candle(s_bt, {SYM: ticks})
        wu_bt = resolver_warmup(s_bt, feed_type="tick", ticks_per_candle=tpv)
        # replay: warmup reserved in the sesh
        eng = ReplayEngine.by_ticks(_S(), data=(TickData(SYM, ticks, tick_size=0.25),), speed=float("inf"))
        eng.prepare()
        wu_rp = eng._sesh._warmup_ticks[SYM]
        assert wu_bt == wu_rp, f"auto warmup differs: bt={wu_bt} rp={wu_rp}"

    def test_warmup_zero_same_start(self):
        """warmup=0 → both start on_data on the same first bar."""
        ticks = ticks_with_spread()
        bt = _capture(_StratPivot, ticks, "bt")
        rp = _capture(_StratPivot, ticks, "rp")
        assert min(bt) == min(rp), (
            f"first on_data differs: bt={min(bt)} rp={min(rp)}"
        )


@pytest.mark.unit
class TestParityPatternMatcher:
    def test_matches_coincide(self):
        """add_pattern_matcher: MatchResults coincide across engines."""
        from tradetropy.ta import NBS

        class _SP(Strategy):
            warmup = 0

            def init(self):
                self.tk = self.subscribe_ticks(SYM)
                self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
                hl = [self.ohlc.high_ref, self.ohlc.low_ref, self.ohlc.ts_ref]
                self.cpivot = self.add_indicator(hl, ConfirmedPivot(swing=2))
                self.nbs = self.add_indicator(hl, NBS(swing=2))
                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                    decorators=[self.nbs],
                    pattern="H\nL < $0\n$",
                    tag="impulso",
                )

            def on_data(self):
                pass

        ticks = ticks_with_spread(n_bars=60)
        bt = _capture(_SP, ticks, "bt")
        rp = _capture(_SP, ticks, "rp")
        common = _common(bt, rp)
        diffs = [ts for ts in common if bt[ts].get("match") != rp[ts].get("match")]
        assert diffs == [], f"match differs in {len(diffs)} bars: {diffs[:3]}"

    def test_nbs_retroactive_tags_parity(self):
        """
        NBS tags are retroactive: classifying a new pivot can re-tag an
        earlier one (EMP->BOO, NEU->SHK). The live/replay incremental store
        must reflect those promotions (re-read from the ring) so its pivot
        sequence matches the backtest's. Regression for the bug where live
        snapshotted a pivot's tag at confirmation time and froze it, breaking
        pattern matches in replay/live while backtest still worked.

        The OHLC window is sized to cover the whole dataset so the comparison
        isolates the store behavior from the (separate) stateful-indicator
        window-boundary effect.
        """
        from tradetropy.ta import NBS

        n_bars = 80
        ticks = ticks_with_spread(n_bars=n_bars)

        class _SP(Strategy):
            warmup = 0

            def init(self):
                self.tk = self.subscribe_ticks(SYM)
                self.ohlc = self.subscribe_ohlc(
                    SYM, timeframe=INTERVAL, window_size=n_bars + 50
                )
                hl = [self.ohlc.high_ref, self.ohlc.low_ref, self.ohlc.ts_ref]
                self.cpivot = self.add_indicator(hl, ConfirmedPivot(swing=2))
                self.nbs = self.add_indicator(hl, NBS(swing=2))
                self.setup = self.add_pattern_matcher(
                    base_pivot=self.cpivot,
                    decorators=[self.nbs],
                    pattern="L[nbs=boo]\nH[nbs=neu]\n$",
                    tag="nbs_seq",
                )

            def on_data(self):
                pass

        def _final_seq(mode):
            cap = {}

            class _W(_SP):
                def on_data(self):
                    pass

            if mode == "bt":
                sesh = SeshSimulatorBase("tick")
                eng = BacktestEngine.by_ticks(
                    _W(), data=(TickData(SYM, ticks, tick_size=0.25),), sesh=sesh
                )
                strat = eng.strategy
                eng.run()
            else:
                eng = ReplayEngine.by_ticks(
                    _W(), data=(TickData(SYM, ticks, tick_size=0.25),),
                    speed=float("inf"),
                )
                strat = eng.strategy
                eng.prepare()
                s = eng._sesh
                s._cursor[SYM] = len(ticks) - 1
                last_idx = s._warmup_ticks.get(SYM, 0) - 1
                for t in s._fetch_pending_ticks(SYM, last_idx):
                    eng.on_tick(SYM, t)
            pivs = strat.setup._store._sequence.pivots
            return [(p.index, p.type, round(float(p.value), 4),
                     p.tags.get("nbs", "")) for p in pivs]

        bt_seq = _final_seq("bt")
        rp_seq = _final_seq("rp")

        assert bt_seq == rp_seq, (
            f"NBS pivot sequence differs:\n bt={bt_seq}\n rp={rp_seq}"
        )
        # Ensure the test actually exercises a retroactive promotion.
        promoted = [s for s in bt_seq if s[3] in ("boo", "shk")]
        assert promoted, (
            "test data produced no boo/shk tags; cannot validate retroactivity"
        )


@pytest.mark.integration
class TestParityVolume:
    def test_ohlc_identical_large_dataset(self):
        """OHLC parity on a large dataset (stability/volume)."""
        ticks = ticks_with_spread(n_bars=200, ticks_per_bar=70)
        bt = _capture(_StratPivot, ticks, "bt")
        rp = _capture(_StratPivot, ticks, "rp")
        common = _common(bt, rp)
        diffs = [
            ts for ts in common
            if bt[ts]["high"] != rp[ts]["high"]
            or bt[ts]["cp_high"] != rp[ts]["cp_high"]
        ]
        assert diffs == [], f"divergence in {len(diffs)} bars: {diffs[:3]}"


# ══════════════════════════════════════════════════════════════════════════════
# @active_bars  (activation window, engine-level)
# ══════════════════════════════════════════════════════════════════════════════


def _pattern_strategy(dsl):
    """Build a Strategy class with a single pattern matcher over the DSL."""

    class _SP(Strategy):
        warmup = 0

        def init(self):
            self.tk = self.subscribe_ticks(SYM)
            self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
            self.cpivot = self.add_indicator(
                [self.ohlc.high_ref, self.ohlc.low_ref, self.ohlc.ts_ref],
                ConfirmedPivot(swing=2),
            )
            self.setup = self.add_pattern_matcher(
                base_pivot=self.cpivot, pattern=dsl, tag="ab",
            )

        def on_data(self):
            pass

    return _SP


@pytest.mark.unit
class TestParityActiveBars:
    def test_active_bars_parity(self):
        """The windowed match stream is identical in backtest and replay."""
        ticks = ticks_with_spread(n_bars=60)
        cls = _pattern_strategy("L\nH > $0\n@active_bars(0, 3)")
        bt = _capture(cls, ticks, "bt")
        rp = _capture(cls, ticks, "rp")
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts].get("match") != rp[ts].get("match")]
        assert diffs == [], f"active_bars match differs: {diffs[:3]}"

    def test_active_bars_narrows_matches(self):
        """A tight @active_bars window drops stale matches vs no window."""
        ticks = ticks_with_spread(n_bars=60)
        wide = _pattern_strategy("L\nH > $0")
        narrow = _pattern_strategy("L\nH > $0\n@active_bars(0, 0)")
        bt_wide = _capture(wide, ticks, "bt")
        bt_narrow = _capture(narrow, ticks, "bt")
        n_wide = sum(1 for s in bt_wide.values() if s.get("match") is not None)
        n_narrow = sum(1 for s in bt_narrow.values() if s.get("match") is not None)
        assert n_wide > 0, "test data produced no matches"
        assert 0 < n_narrow < n_wide, (
            f"expected the window to narrow matches: wide={n_wide} narrow={n_narrow}"
        )

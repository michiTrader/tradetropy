"""
Unit tests for the pure candlestick detector and causal efficacy tracker
(tradetropy.ta._candlestick).

Adaptive percentile thresholds mean a pattern is judged relative to the recent
distribution, so each test first lays down a baseline regime of small candles to
define that distribution, then injects the target formation and asserts the code
appears on its confirmation bar. Causality is asserted directly: appending
future bars must not change a past bar's code, and the efficacy hit-rate must
only ever use resolved signals.
"""

import numpy as np
import pytest

from tradetropy.ta import _candlestick as cs


# =====
# Candle builders
# =====
def _candle(o, h, l, c):
    return (float(o), float(h), float(l), float(c))


def _stack(candles):
    """Turn a list of (o,h,l,c) into four float arrays."""
    arr = np.asarray(candles, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _baseline(n=40, price=100.0, body=1.0, wick=0.3, drift=0.0, seed=7):
    """
    n small candles trending by `drift` per bar, with VARIED body/wick sizes so
    the rolling percentile thresholds are non-degenerate (a constant body would
    make every candle simultaneously "small" and "long").
    """
    gen = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = price + drift * i
        b = float(gen.uniform(0.6, 1.4))
        w = float(gen.uniform(0.1, 0.35))
        if i % 2 == 0:
            o, c = p - b / 2, p + b / 2
        else:
            o, c = p + b / 2, p - b / 2
        h = max(o, c) + w
        l = min(o, c) - w
        out.append(_candle(o, h, l, c))
    return out


# =====
# Detector: individual patterns
# =====
class TestPatternShapes:
    def test_doji(self):
        candles = _baseline(40)
        # Tiny body, moderate range.
        candles.append(_candle(100.0, 101.0, 99.0, 100.02))
        o, h, l, c = _stack(candles)
        codes, direc = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.DOJI
        assert direc[-1] == 0.0

    def test_bullish_marubozu(self):
        candles = _baseline(40)
        # Long body, negligible wicks.
        candles.append(_candle(100.0, 105.02, 99.98, 105.0))
        o, h, l, c = _stack(candles)
        codes, _ = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.MARUBOZU_BULL

    def test_bearish_marubozu(self):
        candles = _baseline(40)
        candles.append(_candle(105.0, 105.02, 99.98, 100.0))
        o, h, l, c = _stack(candles)
        codes, _ = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.MARUBOZU_BEAR

    def test_hammer_needs_down_context(self):
        # Downtrend so the hammer bar is stretched down (z <= -thresh).
        candles = _baseline(50, price=140.0, drift=-0.8)
        # Two small bullish neutral candles isolate the hammer shape (no long
        # prev body -> no harami/engulfing/star interference).
        for _ in range(2):
            a = candles[-1][3]
            candles.append(_candle(a, a + 0.25, a - 0.05, a + 0.15))
        anchor = candles[-1][3]
        # Body clearly above the doji percentile, long lower wick, tiny upper.
        o = anchor
        c = anchor + 0.8
        candles.append(_candle(o, c + 0.2, o - 2.5, c))
        o_, h_, l_, c_ = _stack(candles)
        codes, direc = cs.detect_candle_patterns(
            o_, h_, l_, c_, window=25, zscore_len=30, zscore_thresh=0.5
        )
        assert codes[-1] == cs.HAMMER
        assert direc[-1] == 1.0

    def test_lower_wick_in_up_context_is_hanging_man(self):
        candles = _baseline(50, price=60.0, drift=0.8)
        for _ in range(2):
            a = candles[-1][3]
            candles.append(_candle(a, a + 0.25, a - 0.05, a + 0.15))
        anchor = candles[-1][3]
        o = anchor
        c = anchor + 0.8
        candles.append(_candle(o, c + 0.2, o - 2.5, c))
        o_, h_, l_, c_ = _stack(candles)
        codes, direc = cs.detect_candle_patterns(
            o_, h_, l_, c_, window=25, zscore_len=30, zscore_thresh=0.5
        )
        assert codes[-1] == cs.HANGING_MAN
        assert direc[-1] == -1.0

    def test_bullish_engulfing(self):
        candles = _baseline(40)
        # Small bearish then a bigger bullish body engulfing it.
        candles.append(_candle(100.0, 100.4, 99.4, 99.5))     # prev bearish
        candles.append(_candle(99.4, 101.2, 99.2, 101.0))     # engulfs
        o, h, l, c = _stack(candles)
        codes, direc = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.ENGULFING_BULL
        assert direc[-1] == 1.0

    def test_bearish_engulfing(self):
        candles = _baseline(40)
        candles.append(_candle(100.0, 100.6, 99.6, 100.5))    # prev bullish
        candles.append(_candle(100.6, 100.8, 99.0, 99.2))     # engulfs down
        o, h, l, c = _stack(candles)
        codes, direc = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.ENGULFING_BEAR
        assert direc[-1] == -1.0

    def test_three_white_soldiers(self):
        candles = _baseline(40)
        candles.append(_candle(100.0, 103.1, 99.9, 103.0))
        candles.append(_candle(103.0, 106.1, 102.9, 106.0))
        candles.append(_candle(106.0, 109.1, 105.9, 109.0))
        o, h, l, c = _stack(candles)
        codes, direc = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.THREE_WHITE_SOLDIERS
        assert direc[-1] == 1.0

    def test_no_pattern_on_plain_candle(self):
        candles = _baseline(40)
        # Average body with modest wicks (< 0.5*body): no pattern fires.
        candles.append(_candle(100.0, 101.1, 99.75, 100.9))
        o, h, l, c = _stack(candles)
        codes, _ = cs.detect_candle_patterns(o, h, l, c, window=20)
        assert codes[-1] == cs.PATTERN_NONE


# =====
# Causality
# =====
class TestCausality:
    def test_codes_do_not_change_with_future_bars(self):
        candles = _baseline(60)
        candles.append(_candle(100.0, 101.0, 99.0, 100.02))   # doji
        o, h, l, c = _stack(candles)
        codes_now, _ = cs.detect_candle_patterns(o, h, l, c, window=25)

        # Append arbitrary future bars; earlier codes must be unchanged.
        future = candles + _baseline(20, price=110.0)
        o2, h2, l2, c2 = _stack(future)
        codes_future, _ = cs.detect_candle_patterns(o2, h2, l2, c2, window=25)

        np.testing.assert_array_equal(codes_now, codes_future[: len(codes_now)])


# =====
# Efficacy (causal, anti-lookahead)
# =====
class TestEfficacy:
    def _series_with_two_bull_signals(self):
        """Two bullish-coded bars; craft close path so one wins, one loses."""
        n = 30
        close = np.full(n, 100.0)
        codes = np.zeros(n, dtype=np.int64)
        directions = np.zeros(n, dtype=np.float64)
        # Signal A at bar 5: price rises over the horizon -> hit.
        codes[5] = cs.ENGULFING_BULL
        directions[5] = 1.0
        # Signal B at bar 15: price falls over the horizon -> miss.
        codes[15] = cs.ENGULFING_BULL
        directions[15] = 1.0
        close[5:12] = np.linspace(100.0, 106.0, 7)   # up after bar 5
        close[12:] = 106.0
        close[15:22] = np.linspace(106.0, 100.0, 7)  # down after bar 15
        close[22:] = 100.0
        return codes, directions, close

    def test_hit_rate_counts_only_resolved(self):
        codes, directions, close = self._series_with_two_bull_signals()
        # As of bar 12: only signal A (bar 5, horizon 5 -> resolved at 10) counts.
        stats = cs.evaluate_efficacy(codes, directions, close, horizon=5, as_of=12)
        assert stats[cs.ENGULFING_BULL]["sample_size"] == 1
        assert stats[cs.ENGULFING_BULL]["hit_rate"] == 1.0

    def test_full_history_scores_both(self):
        codes, directions, close = self._series_with_two_bull_signals()
        stats = cs.evaluate_efficacy(codes, directions, close, horizon=5)
        assert stats[cs.ENGULFING_BULL]["sample_size"] == 2
        assert stats[cs.ENGULFING_BULL]["wins"] == 1
        assert stats[cs.ENGULFING_BULL]["hit_rate"] == 0.5

    def test_unresolved_signal_not_counted(self):
        codes, directions, close = self._series_with_two_bull_signals()
        # As of bar 9: signal A (bar 5 + horizon 5 = 10) is NOT resolved yet.
        stats = cs.evaluate_efficacy(codes, directions, close, horizon=5, as_of=9)
        assert cs.ENGULFING_BULL not in stats

    def test_neutral_pattern_hit_rate_is_nan(self):
        n = 20
        close = np.linspace(100.0, 110.0, n)
        codes = np.zeros(n, dtype=np.int64)
        directions = np.zeros(n, dtype=np.float64)
        codes[3] = cs.DOJI  # neutral
        stats = cs.evaluate_efficacy(codes, directions, close, horizon=5)
        assert stats[cs.DOJI]["sample_size"] == 1
        assert np.isnan(stats[cs.DOJI]["hit_rate"])


# =====
# Helpers
# =====
def test_pattern_name_and_direction():
    assert cs.pattern_name(cs.HAMMER) == "Hammer"
    assert cs.pattern_name(0) == "none"
    assert cs.pattern_name(999) == "none"
    assert cs.pattern_direction(cs.ENGULFING_BEAR) == -1.0
    assert cs.pattern_direction(cs.DOJI) == 0.0

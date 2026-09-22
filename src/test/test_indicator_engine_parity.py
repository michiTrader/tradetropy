"""
BacktestEngine <-> ReplayEngine parity for recursive OHLC indicators.

Recursive indicators (RSI = Wilder smoothing, MACD = EMA) are path-dependent:
their value depends on where the history window starts. Backtest and replay
must feed them the SAME causal window (= the OHLC proxy window_size) so the
series match, while windowed indicators (SMA) are unaffected.

This guards two specific regressions:
    1. Closed bars: backtest precomputed over full history while live/replay
       recomputed over only `min_periods` -> RSI/MACD diverged.
    2. Partial bar: backtest used `indicator.length` (< min_periods for RSI)
       -> calculate() returned all NaN -> partial RSI was NaN, while replay
       used `min_periods` and produced a value.

After the fix both engines use a causal window of `window_size` for closed AND
partial bars, so RSI/MACD coincide and the partial bar has a value with no jump
at close.

Usage:
    pytest src/test/test_indicator_engine_parity.py -v
"""

import numpy as np
import pytest

from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.replay import ReplayEngine
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.core.data_types import TickData
from tradetropy.ta import SMA, RSI, MACD, VIDYA, FRAMA

SYM = 'X'
INTERVAL = 30_000


def make_ticks(n_bars=60, ticks_per_bar=40, spread=0.25, seed=7):
    """
    Build ticks [N x 7] with a real spread (bid/ask != price) and a gently
    trending random walk so RSI/MACD take non-trivial values.
    """
    base = (1_700_000_000_000 // INTERVAL) * INTERVAL
    rng = np.random.default_rng(seed)
    rows = []
    price = 7540.0
    for bar in range(n_bars):
        bar_ts = base + bar * INTERVAL
        drift = 0.05 * np.sin(bar / 7.0)
        for k in range(ticks_per_bar):
            price += (rng.random() - 0.5) * 0.6 + drift
            ts = bar_ts + int(k * INTERVAL / ticks_per_bar)
            rows.append([ts, price - spread, price + spread, 1.0, 0, 1.0, price])
    return np.asarray(rows, dtype=np.float64)


def _serialize(arr, ndigits=6):
    """Serialize an array to a rounded tuple, 'n' for NaN (comparable)."""
    return tuple('n' if np.isnan(x) else round(float(x), ndigits) for x in arr)


def _capture(strategy_cls, ticks, mode):
    """
    Run the indicated engine, capturing a per-on_data snapshot of the SMA, RSI
    and MACD series indexed by the current bar ts. Returns {ts: snapshot}.
    """
    captured = {}

    class _Wrapped(strategy_cls):
        def on_data(self):
            ts_last = self.ohlc.ts[-1]
            ts = (
                int(ts_last)
                if isinstance(ts_last, (int, float))
                else int(np.datetime64(ts_last, 'ms').astype('int64'))
            )
            captured[ts] = {
                'n_ohlc': len(self.ohlc.ts),
                'sma': _serialize(self.sma),
                'rsi': _serialize(self.rsi),
                'macd': _serialize(self.macd.macd),
                'rsi_partial': _serialize(self.rsi)[-1],
                'vidya': _serialize(self.vidya) if hasattr(self, 'vidya') else (),
                'frama': _serialize(self.frama) if hasattr(self, 'frama') else (),
            }

    if mode == 'bt':
        sesh = SeshSimulatorBase('tick')
        BacktestEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.25),), sesh=sesh
        ).run(verbose=False)
    else:
        eng = ReplayEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.25),),
            speed=float('inf'),
        )
        eng.prepare()
        sesh = eng._sesh
        sesh._cursor[SYM] = len(ticks) - 1
        last_idx = sesh._warmup_ticks.get(SYM, 0) - 1
        for t in sesh._fetch_pending_ticks(SYM, last_idx):
            eng.on_tick(SYM, t)
    return captured


def _common(a, b):
    return sorted(set(a) & set(b))


class _StratRecursive(Strategy):
    warmup = 0

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
        self.sma = self.add_indicator(self.ohlc.close_ref, SMA(5))
        self.rsi = self.add_indicator(self.ohlc.close_ref, RSI(14))
        self.macd = self.add_indicator(self.ohlc.close_ref, MACD())

    def on_data(self):
        pass


class _StratRecursiveAutoWarmup(_StratRecursive):
    # Auto warmup (= max(min_periods, warmup_factor*length); 130 for MACD).
    # Exercises the warmup region: replay must reconstruct the same candle
    # history as the backtest so RSI/MACD match on the post-warmup bars.
    warmup = None


@pytest.mark.unit
class TestRecursiveIndicatorParity:
    def test_sma_series_identical(self):
        """SMA (windowed, not recursive) must match exactly across engines."""
        ticks = make_ticks()
        bt = _capture(_StratRecursive, ticks, 'bt')
        rp = _capture(_StratRecursive, ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts]['sma'] != rp[ts]['sma']]
        assert diffs == [], f'SMA differs in {len(diffs)} bars: {diffs[:3]}'

    def test_rsi_series_identical(self):
        """RSI (recursive) must match across engines once windows are unified."""
        ticks = make_ticks()
        bt = _capture(_StratRecursive, ticks, 'bt')
        rp = _capture(_StratRecursive, ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts]['rsi'] != rp[ts]['rsi']]
        assert diffs == [], f'RSI differs in {len(diffs)} bars: {diffs[:3]}'

    def test_macd_series_identical(self):
        """MACD (recursive) must match across engines once windows are unified."""
        ticks = make_ticks()
        bt = _capture(_StratRecursive, ticks, 'bt')
        rp = _capture(_StratRecursive, ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts]['macd'] != rp[ts]['macd']]
        assert diffs == [], f'MACD differs in {len(diffs)} bars: {diffs[:3]}'

    def test_backtest_partial_rsi_not_nan(self):
        """
        The developing (partial) bar must produce an RSI value in the backtest
        once there is enough closed history (was NaN because it used `length`
        instead of a window >= min_periods).
        """
        ticks = make_ticks()
        bt = _capture(_StratRecursive, ticks, 'bt')
        # Look at late bars where plenty of closed history exists.
        late = sorted(bt)[20:]
        nan_partials = [ts for ts in late if bt[ts]['rsi_partial'] == 'n']
        assert nan_partials == [], (
            f'backtest partial RSI is NaN in {len(nan_partials)} late bars'
        )

    def test_partial_rsi_matches_across_engines(self):
        """
        The developing (partial) bar RSI must be identical in backtest and
        replay (both use the same min_periods causal window over the same
        candles). The partial is intentionally seeded from a small window (like
        the original backtest), so it is not asserted to be continuous with the
        full-history closed value; what matters is that both engines agree.
        """
        ticks = make_ticks()
        bt = _capture(_StratRecursive, ticks, 'bt')
        rp = _capture(_StratRecursive, ticks, 'rp')
        common = _common(bt, rp)
        diffs = [
            ts for ts in common
            if bt[ts]['rsi_partial'] != rp[ts]['rsi_partial']
        ]
        assert diffs == [], (
            f'partial RSI differs across engines in {len(diffs)} bars: '
            f'{diffs[:3]}'
        )


@pytest.mark.unit
class TestRecursiveIndicatorParityWithWarmup:
    """
    With auto warmup (> 0) the replay reserves a warmup region and rebuilds it
    from historical klines. The recursive indicators must still match the
    backtest on the post-warmup bars, which requires the warmup to reconstruct
    the same trailing candle history (a full ring window). Regression guard for
    the ReplaySesh warmup sizing the klines by `n_ticks` instead of window_size.
    """

    @pytest.mark.filterwarnings("ignore:Tick warmup:UserWarning")
    def test_rsi_macd_match_with_auto_warmup(self):
        # ~220 bars so the 130-bar auto warmup leaves a real live region.
        ticks = make_ticks(n_bars=220, ticks_per_bar=40)
        bt = _capture(_StratRecursiveAutoWarmup, ticks, 'bt')
        rp = _capture(_StratRecursiveAutoWarmup, ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10, f'too few common bars: {len(common)}'
        rsi_diffs = [ts for ts in common if bt[ts]['rsi'] != rp[ts]['rsi']]
        macd_diffs = [ts for ts in common if bt[ts]['macd'] != rp[ts]['macd']]
        assert rsi_diffs == [], f'RSI differs (warmup) in {len(rsi_diffs)}: {rsi_diffs[:3]}'
        assert macd_diffs == [], f'MACD differs (warmup) in {len(macd_diffs)}: {macd_diffs[:3]}'


class _StratNewRecursive(Strategy):
    """
    Exercises the newly added recursive indicators (VIDYA = CMO-scaled EMA,
    FRAMA = fractal-dimension adaptive EMA). Both are path-dependent and set
    warmup_factor = 5, so backtest and replay must match on the closed bars
    once fed the same causal window.
    """
    warmup = 0

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL)
        self.sma = self.add_indicator(self.ohlc.close_ref, SMA(5))
        self.rsi = self.add_indicator(self.ohlc.close_ref, RSI(14))
        self.macd = self.add_indicator(self.ohlc.close_ref, MACD())
        self.vidya = self.add_indicator(self.ohlc.close_ref, VIDYA(9))
        self.frama = self.add_indicator(self.ohlc.close_ref, FRAMA(16))

    def on_data(self):
        pass


@pytest.mark.unit
class TestNewRecursiveIndicatorParity:
    def test_vidya_series_identical(self):
        """VIDYA (recursive) must match across engines with a unified window."""
        ticks = make_ticks()
        bt = _capture(_StratNewRecursive, ticks, 'bt')
        rp = _capture(_StratNewRecursive, ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts]['vidya'] != rp[ts]['vidya']]
        assert diffs == [], f'VIDYA differs in {len(diffs)} bars: {diffs[:3]}'

    def test_frama_series_identical(self):
        """FRAMA (recursive) must match across engines with a unified window."""
        ticks = make_ticks()
        bt = _capture(_StratNewRecursive, ticks, 'bt')
        rp = _capture(_StratNewRecursive, ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts]['frama'] != rp[ts]['frama']]
        assert diffs == [], f'FRAMA differs in {len(diffs)} bars: {diffs[:3]}'


# =====
# CandlePatterns parity (annotation + query API)
# =====
from tradetropy.ta import CandlePatterns


class _StratCandles(Strategy):
    """
    CandlePatterns is a windowed detector (trailing percentile window), so its
    code/direction stream and the causal query API must be identical in
    backtest and replay once both feed the same OHLC window.
    """
    warmup = 0

    def init(self):
        self.tk = self.subscribe_ticks(SYM)
        self.ohlc = self.subscribe_ohlc(SYM, timeframe=INTERVAL, window_size=300)
        self.candles = self.add_indicator(
            self.ohlc, CandlePatterns(window=40, zscore_len=25, horizon=5)
        )

    def on_data(self):
        pass


def _capture_candles(ticks, mode):
    """Per-bar snapshot of the candle code band and the query API by bar ts."""
    captured = {}

    class _Wrapped(_StratCandles):
        def on_data(self):
            ts_last = self.ohlc.ts[-1]
            ts = (
                int(ts_last)
                if isinstance(ts_last, (int, float))
                else int(np.datetime64(ts_last, 'ms').astype('int64'))
            )
            code = self.candles.code[-1]
            captured[ts] = {
                'code': 'n' if np.isnan(code) else int(code),
                'last': self.candles.last_pattern(),
            }

    if mode == 'bt':
        sesh = SeshSimulatorBase('tick')
        BacktestEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.25),), sesh=sesh
        ).run(verbose=False)
    else:
        eng = ReplayEngine.by_ticks(
            _Wrapped(), data=(TickData(SYM, ticks, tick_size=0.25),),
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
class TestCandlePatternsEngineParity:
    def test_code_stream_identical(self):
        """The pattern code band must match backtest <-> replay per bar."""
        ticks = make_ticks(n_bars=120, ticks_per_bar=30)
        bt = _capture_candles(ticks, 'bt')
        rp = _capture_candles(ticks, 'rp')
        common = _common(bt, rp)
        assert len(common) > 10
        diffs = [ts for ts in common if bt[ts]['code'] != rp[ts]['code']]
        assert diffs == [], f'candle code differs in {len(diffs)} bars: {diffs[:3]}'

    def test_query_last_pattern_identical(self):
        """The causal last_pattern() query must match backtest <-> replay."""
        ticks = make_ticks(n_bars=120, ticks_per_bar=30)
        bt = _capture_candles(ticks, 'bt')
        rp = _capture_candles(ticks, 'rp')
        common = _common(bt, rp)
        diffs = [ts for ts in common if bt[ts]['last'] != rp[ts]['last']]
        assert diffs == [], f'last_pattern differs in {len(diffs)} bars: {diffs[:3]}'

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig

# =====
# RSI (single-band + reference_lines)
# =====
class RSI(Indicator):
    """
    Relative Strength Index.
    source: close [N]
    """

    name = "rsi"
    category = "momentum"

    # Recursive (Wilder smoothing): reserve a few times `length` as warmup so
    # the value converges before on_data() starts (see auto_warmup_candles).
    warmup_factor = 5

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ["rsi"]
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_height=100,
            panel_title="RSI",
            reference_lines=[
                {"value": 70.0, "color": "#F6465D", "dash": "dashed",
                 "label": "70", "width": 0.8},
                {"value": 30.0, "color": "#0ECB81", "dash": "dashed",
                 "label": "30", "width": 0.8},
            ],
        )

    @property
    def min_periods(self) -> int:
        # Wilder's algorithm computes deltas: np.diff reduces N by 1.
        # With n=L+1 -> len(delta)=L -> range(L, L) empty -> loop does not iterate -> NaN.
        # Needs n=L+2 so that range(L, L+1) iterates at least once.
        return self.length + 2

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """Wilder RSI. O(N) with Python loop."""
        n = len(source)
        result = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L + 2:
            return result

        delta  = np.diff(source.astype(np.float64))
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_g = float(np.mean(gains[:L]))
        avg_l = float(np.mean(losses[:L]))

        for i in range(L, len(delta)):
            avg_g = (avg_g * (L - 1) + gains[i]) / L
            avg_l = (avg_l * (L - 1) + losses[i]) / L
            rs    = avg_g / avg_l if avg_l > 0 else np.inf
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return result


# =====
# WilliamsR  source: HLC [N×3]
# =====
class WilliamsR(Indicator):
    """
    Williams %R oscillator. Range: -100 (oversold) to 0 (overbought).
    source: HLC [N×3]
    """

    name = 'williams_r'
    category = 'momentum'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['wr']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Williams %R',
            reference_lines=[
                {'value': -20.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '-20'},
                {'value': -80.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '-80'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        L = self.length
        for i in range(L - 1, n):
            hh = np.max(high[i - L + 1:i + 1])
            ll = np.min(low[i - L + 1:i + 1])
            if hh != ll:
                out[i] = -100.0 * (hh - close[i]) / (hh - ll)
        return out


# =====
# CCI  source: HLC [N×3]
# =====
class CCI(Indicator):
    """
    Commodity Channel Index.
    source: HLC [N×3]
    """

    name = 'cci'
    category = 'momentum'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 20):
        self.length = length
        self.output_names = ['cci']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='CCI',
            reference_lines=[
                {'value': 100.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '100'},
                {'value': -100.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '-100'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        tp = (source[:, 0] + source[:, 1] + source[:, 2]) / 3.0
        for i in range(L - 1, n):
            window = tp[i - L + 1:i + 1]
            mean = np.mean(window)
            mad = np.mean(np.abs(window - mean))
            if mad > 0:
                out[i] = (tp[i] - mean) / (0.015 * mad)
        return out


# =====
# MFI  source: HLCV [N×4]
# =====
class MFI(Indicator):
    """
    Money Flow Index.
    source: HLCV [N×4]
    """

    name = 'mfi'
    category = 'momentum'
    source_cols = ('high', 'low', 'close', 'volume')

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['mfi']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='MFI',
            reference_lines=[
                {'value': 80.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '80'},
                {'value': 20.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '20'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        tp = (source[:, 0] + source[:, 1] + source[:, 2]) / 3.0
        mf = tp * source[:, 3]
        for i in range(L, n):
            pos = neg = 0.0
            for j in range(i - L + 1, i + 1):
                if tp[j] > tp[j - 1]:
                    pos += mf[j]
                elif tp[j] < tp[j - 1]:
                    neg += mf[j]
            if neg == 0:
                out[i] = 100.0
            else:
                out[i] = 100.0 - 100.0 / (1.0 + pos / neg)
        return out


# =====
# Stochastic  source: HLC [N×3]
# =====
class Stochastic(Indicator):
    """
    Stochastic Oscillator (%K and %D).
    source: HLC [N×3]
    """

    name = 'stochastic'
    category = 'momentum'
    source_cols = ('high', 'low', 'close')

    def __init__(self, k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3):
        self.k_period = k_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.length = k_period
        self.output_names = ['k', 'd']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Stochastic',
            color=['#2563EB', '#F59E0B'],
            reference_lines=[
                {'value': 80.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '80'},
                {'value': 20.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '20'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.k_period + self.k_smooth + self.d_smooth - 2

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        cs = np.cumsum(arr)
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((2, n), np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        L = self.k_period
        raw_k = np.full(n, np.nan, dtype=np.float64)
        for i in range(L - 1, n):
            hh = np.max(high[i - L + 1:i + 1])
            ll = np.min(low[i - L + 1:i + 1])
            if hh != ll:
                raw_k[i] = 100.0 * (close[i] - ll) / (hh - ll)
        out[0] = self._sma(raw_k, self.k_smooth)
        out[1] = self._sma(out[0], self.d_smooth)
        return out


# =====
# KeltnerChannels  source: HLC [N×3]
# =====
class KeltnerChannels(Indicator):
    """
    Keltner Channels (EMA +/- mult * ATR).
    source: HLC [N×3]
    """

    name = 'keltner'
    category = 'volatility'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 20, mult: float = 2.0):
        self.length = length
        self.mult = mult
        self.output_names = ['upper', 'mid', 'lower']
        self.plot_config = IndicatorPlotConfig(
            color=['#3B82F6', '#3B82F6', '#3B82F6'],
            line_dash=['solid', 'dashed', 'solid'],
            line_width=[1.5, 1.0, 1.5],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        L = self.length
        # EMA of close
        alpha = 2.0 / (L + 1)
        ema = np.full(n, np.nan, dtype=np.float64)
        if n >= L:
            ema[L - 1] = float(np.mean(close[:L]))
            for i in range(L, n):
                ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]
        # ATR (Wilder)
        prev_c = np.roll(close, 1)
        prev_c[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
        atr_alpha = 1.0 / L
        atr = np.full(n, np.nan, dtype=np.float64)
        if n >= L + 1:
            atr[L] = float(np.mean(tr[1:L + 1]))
            for i in range(L + 1, n):
                atr[i] = (1.0 - atr_alpha) * atr[i - 1] + atr_alpha * tr[i]
        out[0] = ema + self.mult * atr
        out[1] = ema
        out[2] = ema - self.mult * atr
        return out


# =====
# DonchianChannels  source: HLC [N×3]
# =====
class DonchianChannels(Indicator):
    """
    Donchian Channels (rolling high/mid/low).
    source: HLC [N×3]
    """

    name = 'donchian'
    category = 'volatility'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 20):
        self.length = length
        self.output_names = ['upper', 'mid', 'lower']
        self.plot_config = IndicatorPlotConfig(
            color=['#3B82F6', '#3B82F6', '#3B82F6'],
            line_dash=['solid', 'dashed', 'solid'],
            line_width=[1.5, 1.0, 1.5],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        high, low = source[:, 0], source[:, 1]
        L = self.length
        for i in range(L - 1, n):
            hh = np.max(high[i - L + 1:i + 1])
            ll = np.min(low[i - L + 1:i + 1])
            out[0, i] = hh
            out[1, i] = (hh + ll) / 2.0
            out[2, i] = ll
        return out



# =====
# OBV  source: [N×2] (close, volume)
# =====
class OBV(Indicator):
    """
    On-Balance Volume.
    source: close, volume [N×2]
    """

    name = 'obv'
    category = 'volume'
    source_cols = ('close', 'volume')

    def __init__(self):
        self.output_names = ['obv']
        self.plot_config = IndicatorPlotConfig(overlay=False, panel_title='OBV')

    @property
    def min_periods(self) -> int:
        return 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.zeros(n, dtype=np.float64)
        close, volume = source[:, 0], source[:, 1]
        out[0] = volume[0]
        for i in range(1, n):
            if close[i] > close[i - 1]:
                out[i] = out[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                out[i] = out[i - 1] - volume[i]
            else:
                out[i] = out[i - 1]
        return out


# =====
# VWAP  source: [N×6] (ts, open, high, low, close, volume)
# =====
class VWAP(Indicator):
    """
    Volume Weighted Average Price. Resets at the start of each day.
    source: ts, open, high, low, close, volume [N×6]
    """

    name = 'vwap'
    category = 'trend'
    source_cols = ('ts', 'open', 'high', 'low', 'close', 'volume')

    def __init__(self):
        self.output_names = ['vwap']
        self.plot_config = IndicatorPlotConfig(color='#F59E0B')

    @property
    def min_periods(self) -> int:
        return 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        ts = source[:, 0]
        high, low, close, volume = source[:, 2], source[:, 3], source[:, 4], source[:, 5]
        tp = (high + low + close) / 3.0
        DAY_MS = 86_400_000.0
        cum_tpv = cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            day = int(ts[i] // DAY_MS)
            if day != prev_day:
                cum_tpv = cum_vol = 0.0
                prev_day = day
            cum_tpv += tp[i] * volume[i]
            cum_vol += volume[i]
            if cum_vol > 0:
                out[i] = cum_tpv / cum_vol
        return out


# =====
# ADX  source: HLC [N×3]
# =====
class ADX(Indicator):
    """
    Average Directional Index (+DI, -DI, ADX).
    source: HLC [N×3]
    """

    name = 'adx'
    category = 'trend'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['adx', 'plus_di', 'minus_di']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='ADX',
            color=['#F59E0B', '#0ECB81', '#F6465D'],
            reference_lines=[
                {'value': 25.0, 'color': '#888888', 'dash': 'dashed', 'label': '25'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length * 2

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        L = self.length
        if n < L + 1:
            return out

        prev_high = np.roll(high, 1); prev_high[0] = high[0]
        prev_low = np.roll(low, 1); prev_low[0] = low[0]
        prev_close = np.roll(close, 1); prev_close[0] = close[0]

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        alpha = 1.0 / L
        atr = np.full(n, np.nan, dtype=np.float64)
        s_plus = np.full(n, np.nan, dtype=np.float64)
        s_minus = np.full(n, np.nan, dtype=np.float64)

        atr[L] = float(np.mean(tr[1:L + 1]))
        s_plus[L] = float(np.mean(plus_dm[1:L + 1]))
        s_minus[L] = float(np.mean(minus_dm[1:L + 1]))
        for i in range(L + 1, n):
            atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i]
            s_plus[i] = (1.0 - alpha) * s_plus[i - 1] + alpha * plus_dm[i]
            s_minus[i] = (1.0 - alpha) * s_minus[i - 1] + alpha * minus_dm[i]

        with np.errstate(invalid='ignore', divide='ignore'):
            di_plus = np.where(atr > 0, 100.0 * s_plus / atr, np.nan)
            di_minus = np.where(atr > 0, 100.0 * s_minus / atr, np.nan)
            dx = np.where(
                (di_plus + di_minus) > 0,
                100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus),
                np.nan,
            )

        adx = np.full(n, np.nan, dtype=np.float64)
        first = L * 2
        if n > first:
            valid_dx = dx[L + 1:first + 1]
            if not np.isnan(valid_dx).all():
                adx[first] = float(np.nanmean(valid_dx))
                for i in range(first + 1, n):
                    if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                        adx[i] = (1.0 - alpha) * adx[i - 1] + alpha * dx[i]

        out[0] = adx
        out[1] = di_plus
        out[2] = di_minus
        return out


# =====
# ParabolicSAR  source: HLC [N×3]
# =====
class ParabolicSAR(Indicator):
    """
    Parabolic SAR.
    source: HLC [N×3]
    """

    name = 'parabolic_sar'
    category = 'trend'
    source_cols = ('high', 'low', 'close')

    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.output_names = ['sar']
        self.plot_config = IndicatorPlotConfig(renderer='scatter', marker_size=3, color='#F59E0B')

    @property
    def min_periods(self) -> int:
        return 2

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < 2:
            return out
        high, low = source[:, 0], source[:, 1]
        af_start, af_step, af_max = self.af_start, self.af_step, self.af_max

        bull = high[1] >= high[0]
        af = af_start
        ep = high[1] if bull else low[1]
        sar = low[0] if bull else high[0]
        out[1] = sar

        for i in range(2, n):
            prev_sar = sar
            sar = prev_sar + af * (ep - prev_sar)
            if bull:
                sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
                if low[i] < sar:
                    bull = False
                    sar = ep
                    ep = low[i]
                    af = af_start
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, af_max)
            else:
                sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
                if high[i] > sar:
                    bull = True
                    sar = ep
                    ep = high[i]
                    af = af_start
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, af_max)
            out[i] = sar
        return out


# =====
# Supertrend  source: HLC [N×3]
# =====
class Supertrend(Indicator):
    """
    Supertrend indicator.
    source: HLC [N×3]
    """

    name = 'supertrend'
    category = 'trend'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 10, mult: float = 3.0):
        self.length = length
        self.mult = mult
        self.output_names = ['supertrend']
        self.plot_config = IndicatorPlotConfig(color='#F59E0B')

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        L = self.length
        if n < L + 1:
            return out

        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        alpha = 1.0 / L
        atr = np.full(n, np.nan, dtype=np.float64)
        atr[L] = float(np.mean(tr[1:L + 1]))
        for i in range(L + 1, n):
            atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i]

        hl2 = (high + low) / 2.0
        upper_band = hl2 + self.mult * atr
        lower_band = hl2 - self.mult * atr

        st = np.full(n, np.nan, dtype=np.float64)
        trend = np.ones(n, dtype=np.int8)  # 1=up, -1=down

        if np.isnan(atr[L]):
            return out

        st[L] = upper_band[L]
        trend[L] = -1 if close[L] <= upper_band[L] else 1

        for i in range(L + 1, n):
            if np.isnan(atr[i]):
                continue
            prev_upper = upper_band[i - 1] if not np.isnan(st[i - 1]) else upper_band[i]
            prev_lower = lower_band[i - 1] if not np.isnan(st[i - 1]) else lower_band[i]

            upper_band[i] = upper_band[i] if upper_band[i] < prev_upper or close[i - 1] > prev_upper else prev_upper
            lower_band[i] = lower_band[i] if lower_band[i] > prev_lower or close[i - 1] < prev_lower else prev_lower

            if trend[i - 1] == -1:
                if close[i] > upper_band[i]:
                    trend[i] = 1
                    st[i] = lower_band[i]
                else:
                    trend[i] = -1
                    st[i] = upper_band[i]
            else:
                if close[i] < lower_band[i]:
                    trend[i] = -1
                    st[i] = upper_band[i]
                else:
                    trend[i] = 1
                    st[i] = lower_band[i]

        out[L:] = st[L:]
        return out


# =====
# Ichimoku  source: HLC [N×3]
# Band indices: 0=tenkan, 1=kijun, 2=senkou_a, 3=senkou_b, 4=chikou
# Band 4 (chikou) is intentionally non-causal (shifted back kijun periods).
# =====
class Ichimoku(Indicator):
    """
    Ichimoku Cloud. Band 4 (chikou) is non-causal by design.
    source: HLC [N×3]
    """

    name = 'ichimoku'
    category = 'trend'
    source_cols = ('high', 'low', 'close')

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.length = senkou_b
        self.output_names = ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']
        self.plot_config = IndicatorPlotConfig(
            color=['#2563EB', '#F59E0B', '#0ECB81', '#F6465D', '#888888'],
            line_width=[1.2, 1.2, 1.0, 1.0, 1.0],
            line_dash=['solid', 'solid', 'solid', 'solid', 'dashed'],
        )

    @property
    def min_periods(self) -> int:
        return self.senkou_b

    def _mid(self, high: np.ndarray, low: np.ndarray, L: int) -> np.ndarray:
        n = len(high)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(L - 1, n):
            out[i] = (np.max(high[i - L + 1:i + 1]) + np.min(low[i - L + 1:i + 1])) / 2.0
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((5, n), np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]

        out[0] = self._mid(high, low, self.tenkan)
        out[1] = self._mid(high, low, self.kijun)
        # Senkou A: average of tenkan and kijun (not shifted for causality)
        out[2] = (out[0] + out[1]) / 2.0
        out[3] = self._mid(high, low, self.senkou_b)
        # Chikou: close shifted back kijun periods (non-causal by design)
        shift = self.kijun
        out[4, :n - shift] = close[shift:]
        return out


# =====
# ROC
# =====
class ROC(Indicator):
    """
    Rate of Change. Percentage change between n periods.
    source: close [N]
    """

    name = 'roc'
    category = 'momentum'

    def __init__(self, length: int = 12):
        self.length = length
        self.output_names = ['roc']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='ROC',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        for i in range(L, n):
            if source[i - L] != 0:
                out[i] = (source[i] - source[i - L]) / source[i - L] * 100.0
        return out


# =====
# PO (Price Oscillator)
# =====
class PO(Indicator):
    """
    Price Oscillator. Difference between two EMAs in points.
    source: close [N]
    """

    name = 'po'
    category = 'momentum'

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow
        self.length = slow
        self.output_names = ['po']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Price Oscillator',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.slow

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return out
        alpha = 2.0 / (length + 1)
        out[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        ema_fast = self._ema(source, self.fast)
        ema_slow = self._ema(source, self.slow)
        return ema_fast - ema_slow


# =====
# PPO (Percentage Price Oscillator)
# =====
class PPO(Indicator):
    """
    Percentage Price Oscillator (like MACD but in %). 3 bands: ppo, signal, histogram.
    source: close [N]
    """

    name = 'ppo'
    category = 'momentum'

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.length = slow
        self.output_names = ['ppo', 'signal', 'histogram']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='PPO',
            color=['#2563EB', '#F59E0B', '#888888'],
            line_width=[1.6, 1.2, 1.0],
            renderer=['line', 'line', 'bar'],
            bar_color_positive='#0ECB81',
            bar_color_negative='#F6465D',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal_period - 1

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return out
        alpha = 2.0 / (length + 1)
        out[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        ema_fast = self._ema(source, self.fast)
        ema_slow = self._ema(source, self.slow)
        with np.errstate(invalid='ignore', divide='ignore'):
            ppo_line = np.where(ema_slow != 0, (ema_fast - ema_slow) / ema_slow * 100.0, np.nan)
        first_valid = np.argmax(~np.isnan(ppo_line))
        if np.isnan(ppo_line[first_valid]):
            return out
        sig = self._ema(ppo_line[first_valid:], self.signal_period)
        out[0, first_valid:] = ppo_line[first_valid:]
        out[1, first_valid:] = sig
        out[2, first_valid:] = out[0, first_valid:] - out[1, first_valid:]
        return out


# =====
# StochasticRSI
# =====
class StochasticRSI(Indicator):
    """
    Stochastic RSI. Applies stochastic to the RSI. 2 bands: k, d.
    source: close [N]
    """

    name = 'stochrsi'
    category = 'momentum'

    def __init__(self, rsi_length: int = 14, stoch_length: int = 14, k_smooth: int = 3, d_smooth: int = 3):
        self.rsi_length = rsi_length
        self.stoch_length = stoch_length
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.length = rsi_length
        self.output_names = ['k', 'd']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Stochastic RSI',
            color=['#2563EB', '#F59E0B'],
            reference_lines=[
                {'value': 80.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '80'},
                {'value': 20.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '20'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.rsi_length + 2 + self.stoch_length + self.k_smooth + self.d_smooth - 2

    def _rsi(self, src: np.ndarray, L: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L + 2:
            return out
        delta = np.diff(src.astype(np.float64))
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_g = float(np.mean(gains[:L]))
        avg_l = float(np.mean(losses[:L]))
        for i in range(L, len(delta)):
            avg_g = (avg_g * (L - 1) + gains[i]) / L
            avg_l = (avg_l * (L - 1) + losses[i]) / L
            rs = avg_g / avg_l if avg_l > 0 else np.inf
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        for i in range(L - 1, n):
            window = arr[i - L + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) == L:
                out[i] = np.mean(valid)
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((2, n), np.nan, dtype=np.float64)
        rsi = self._rsi(source, self.rsi_length)
        L = self.stoch_length
        raw_k = np.full(n, np.nan, dtype=np.float64)
        for i in range(L - 1, n):
            window = rsi[i - L + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                hh = np.max(valid)
                ll = np.min(valid)
                if hh != ll and not np.isnan(rsi[i]):
                    raw_k[i] = (rsi[i] - ll) / (hh - ll) * 100.0
        out[0] = self._sma(raw_k, self.k_smooth)
        out[1] = self._sma(out[0], self.d_smooth)
        return out


# =====
# UltimateOscillator
# =====
class UltimateOscillator(Indicator):
    """
    Ultimate Oscillator (Larry Williams). Multi-timeframe: 7/14/28.
    source: HLC [N×3]
    """

    name = 'ultosc'
    category = 'momentum'
    source_cols = ('high', 'low', 'close')

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.length = period3
        self.output_names = ['ultosc']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Ultimate Oscillator',
            reference_lines=[
                {'value': 70.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '70'},
                {'value': 30.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '30'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.period3 + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        bp = close - np.minimum(low, np.roll(close, 1))
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        bp[0] = 0.0
        tr[0] = high[0] - low[0]
        for i in range(self.period3, n):
            s1 = np.sum(bp[i - self.period1 + 1:i + 1]) / np.sum(tr[i - self.period1 + 1:i + 1]) if np.sum(tr[i - self.period1 + 1:i + 1]) > 0 else 0
            s2 = np.sum(bp[i - self.period2 + 1:i + 1]) / np.sum(tr[i - self.period2 + 1:i + 1]) if np.sum(tr[i - self.period2 + 1:i + 1]) > 0 else 0
            s3 = np.sum(bp[i - self.period3 + 1:i + 1]) / np.sum(tr[i - self.period3 + 1:i + 1]) if np.sum(tr[i - self.period3 + 1:i + 1]) > 0 else 0
            out[i] = 100.0 * (4.0 * s1 + 2.0 * s2 + s3) / 7.0
        return out


# =====
# CMO (Chande Momentum Oscillator)
# =====
class CMO(Indicator):
    """
    Chande Momentum Oscillator. Range: -100 to +100.
    source: close [N]
    """

    name = 'cmo'
    category = 'momentum'

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['cmo']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='CMO',
            reference_lines=[
                {'value': 50.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '50'},
                {'value': -50.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '-50'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        delta = np.diff(source.astype(np.float64))
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        for i in range(L, len(delta)):
            su = float(np.sum(gains[i - L + 1:i + 1]))
            sd = float(np.sum(losses[i - L + 1:i + 1]))
            denom = su + sd
            if denom > 0:
                out[i + 1] = 100.0 * (su - sd) / denom
        return out


# =====
# TSI (True Strength Index)
# =====
class TSI(Indicator):
    """
    True Strength Index. Double smoothed, -100 to +100. 2 bands: tsi, signal.
    source: close [N]
    """

    name = 'tsi'
    category = 'momentum'

    def __init__(self, long: int = 25, short: int = 13, signal: int = 7):
        self.long = long
        self.short = short
        self.signal_period = signal
        self.length = long
        self.output_names = ['tsi', 'signal']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='True Strength Index',
            color=['#2563EB', '#F59E0B'],
            reference_lines=[
                {'value': 50.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '50'},
                {'value': -50.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '-50'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.long + self.short + self.signal_period

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        first_valid = np.argmax(~np.isnan(src))
        if np.isnan(src[first_valid]):
            return out
        alpha = 2.0 / (length + 1)
        out[first_valid] = src[first_valid]
        for i in range(first_valid + 1, n):
            if np.isnan(src[i]):
                out[i] = out[i - 1]
            else:
                out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((2, n), np.nan, dtype=np.float64)
        momentum = np.diff(source.astype(np.float64))
        abs_momentum = np.abs(momentum)
        pad = np.concatenate(([np.nan], momentum))
        pad_abs = np.concatenate(([np.nan], abs_momentum))
        pc1 = self._ema(pad, self.long)
        pd1 = self._ema(pad_abs, self.long)
        pc2 = self._ema(pc1, self.short)
        pd2 = self._ema(pd1, self.short)
        with np.errstate(invalid='ignore', divide='ignore'):
            tsi_line = np.where(pd2 != 0, pc2 / pd2 * 100.0, np.nan)
        out[0] = tsi_line
        out[1] = self._ema(tsi_line, self.signal_period)
        return out


# =====
# TRIX
# =====
class TRIX(Indicator):
    """
    Triple EMA Rate of Change. Long-term momentum.
    source: close [N]
    """

    name = 'trix'
    category = 'momentum'

    def __init__(self, length: int = 15):
        self.length = length
        self.output_names = ['trix']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='TRIX',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return 3 * self.length

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return out
        alpha = 2.0 / (length + 1)
        first_valid = np.argmax(~np.isnan(src))
        if np.isnan(src[first_valid]):
            return out
        out[first_valid] = src[first_valid]
        for i in range(first_valid + 1, n):
            if np.isnan(src[i]):
                out[i] = out[i - 1]
            else:
                out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        ema1 = self._ema(source, self.length)
        ema2 = self._ema(ema1, self.length)
        ema3 = self._ema(ema2, self.length)
        for i in range(1, n):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
                out[i] = (ema3[i] - ema3[i - 1]) / ema3[i - 1] * 10000.0
        return out


# =====
# AwesomeOscillator
# =====
class AwesomeOscillator(Indicator):
    """
    Awesome Oscillator. SMA(5) - SMA(34) of the midprice.
    source: HL [N×2]
    """

    name = 'ao'
    category = 'momentum'
    source_cols = ('high', 'low')

    def __init__(self):
        self.output_names = ['ao']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Awesome Oscillator',
            renderer='bar',
            bar_color_positive='#0ECB81',
            bar_color_negative='#F6465D',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return 34

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        cs = np.cumsum(arr)
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        mid = (source[:, 0] + source[:, 1]) / 2.0
        sma5 = self._sma(mid, 5)
        sma34 = self._sma(mid, 34)
        return sma5 - sma34


# =====
# BOP (Balance of Power)
# =====
class BOP(Indicator):
    """
    Balance of Power. Buying vs selling strength per bar.
    source: OHLC [N×4]
    """

    name = 'bop'
    category = 'momentum'
    source_cols = ('open', 'high', 'low', 'close')

    def __init__(self):
        self.output_names = ['bop']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Balance of Power',
            renderer='bar',
            bar_color_positive='#0ECB81',
            bar_color_negative='#F6465D',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        o, h, l, c = source[:, 0], source[:, 1], source[:, 2], source[:, 3]
        rng = h - l
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(rng > 0, (c - o) / rng, 0.0)
        return out


# =====
# DPO (Detrended Price Oscillator)
# =====
class DPO(Indicator):
    """
    Detrended Price Oscillator. Removes the trend to reveal cycles.
    source: close [N]
    """

    name = 'dpo'
    category = 'momentum'

    def __init__(self, length: int = 20):
        self.length = length
        self.output_names = ['dpo']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='DPO',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        cs = np.cumsum(arr)
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        sma = self._sma(source, self.length)
        shift = self.length // 2 + 1
        for i in range(shift, n):
            if not np.isnan(sma[i - shift]):
                out[i] = source[i] - sma[i - shift]
        return out


# =====
# MassIndex
# =====
class MassIndex(Indicator):
    """
    Mass Index. Detects reversal via range expansion/contraction.
    source: HL [N×2]
    """

    name = 'mi'
    category = 'momentum'
    source_cols = ('high', 'low')

    def __init__(self, length: int = 9, sum_length: int = 25):
        self.length = length
        self.sum_length = sum_length
        self.output_names = ['mi']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Mass Index',
            reference_lines=[
                {'value': 27.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '27'},
                {'value': 26.5, 'color': '#0ECB81', 'dash': 'dashed', 'label': '26.5'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + self.sum_length

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return out
        alpha = 2.0 / (length + 1)
        out[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        high, low = source[:, 0], source[:, 1]
        diff = high - low
        ema1 = self._ema(diff, self.length)
        ema2 = self._ema(ema1, self.length)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(ema2 != 0, ema1 / ema2, np.nan)
        for i in range(self.sum_length - 1, n):
            window = ratio[i - self.sum_length + 1:i + 1]
            if not np.isnan(window).any():
                out[i] = float(np.sum(window))
        return out


# =====
# ChaikinAD
# =====
class ChaikinAD(Indicator):
    """
    Chaikin Accumulation/Distribution Line.
    source: HLCV [N×4]
    """

    name = 'chaikin_ad'
    category = 'volume'
    source_cols = ('high', 'low', 'close', 'volume')

    def __init__(self):
        self.output_names = ['ad']
        self.plot_config = IndicatorPlotConfig(overlay=False, panel_title='Chaikin A/D')

    @property
    def min_periods(self) -> int:
        return 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.zeros(n, dtype=np.float64)
        high, low, close, volume = source[:, 0], source[:, 1], source[:, 2], source[:, 3]
        hl = high - low
        with np.errstate(invalid='ignore', divide='ignore'):
            mfm = np.where(hl > 0, ((close - low) - (high - close)) / hl, 0.0)
        mfv = mfm * volume
        out[0] = mfv[0]
        for i in range(1, n):
            out[i] = out[i - 1] + mfv[i]
        return out


# =====
# ChaikinOsc
# =====
class ChaikinOsc(Indicator):
    """
    Chaikin Oscillator. EMA of the Chaikin A/D Line.
    source: HLCV [N×4]
    """

    name = 'chaikin_osc'
    category = 'volume'
    source_cols = ('high', 'low', 'close', 'volume')

    def __init__(self, fast: int = 3, slow: int = 10):
        self.fast = fast
        self.slow = slow
        self.output_names = ['chaikin_osc']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Chaikin Oscillator',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.slow + 1

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return out
        alpha = 2.0 / (length + 1)
        out[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        ad = ChaikinAD().calculate(source)
        ema_fast = self._ema(ad, self.fast)
        ema_slow = self._ema(ad, self.slow)
        return ema_fast - ema_slow


# =====
# ForceIndex
# =====
class ForceIndex(Indicator):
    """
    Force Index. Price change * volume.
    source: close, volume [N×2]
    """

    name = 'fi'
    category = 'volume'
    source_cols = ('close', 'volume')

    def __init__(self, length: int = 13):
        self.length = length
        self.output_names = ['fi']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Force Index',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        valid = ~np.isnan(src)
        if np.sum(valid) < length:
            return out
        first_valid = np.argmax(valid)
        alpha = 2.0 / (length + 1)
        out[first_valid] = src[first_valid]
        for i in range(first_valid + 1, n):
            if np.isnan(src[i]):
                out[i] = out[i - 1]
            else:
                out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        close, volume = source[:, 0], source[:, 1]
        raw = np.empty(len(close), dtype=np.float64)
        raw[0] = 0.0
        raw[1:] = (close[1:] - close[:-1]) * volume[1:]
        return self._ema(raw, self.length)


# =====
# EMV (Ease of Movement)
# =====
class EMV(Indicator):
    """
    Ease of Movement. Ease of price movement.
    source: HLV [N×3]
    """

    name = 'emv'
    category = 'volume'
    source_cols = ('high', 'low', 'volume')

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['emv']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Ease of Movement',
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        cs = np.cumsum(arr)
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        high, low, volume = source[:, 0], source[:, 1], source[:, 2]
        mid = (high + low) / 2.0
        mid_move = np.diff(mid)
        rng = high - low
        avg_rng = (rng[:-1] + rng[1:]) / 2.0
        raw = np.empty(n, dtype=np.float64)
        raw[0] = 0.0
        with np.errstate(invalid='ignore', divide='ignore'):
            box_ratio = np.diff(volume) / avg_rng * 2.0
            raw[1:] = np.where(box_ratio != 0, mid_move / box_ratio, 0.0)
        return self._sma(raw, self.length)


# =====
# VWMA
# =====
class VWMA(Indicator):
    """
    Volume Weighted Moving Average. Volume-weighted SMA.
    source: close, volume [N×2]
    """

    name = 'vwma'
    category = 'trend'
    source_cols = ('close', 'volume')

    def __init__(self, length: int = 20):
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        close, volume = source[:, 0], source[:, 1]
        L = self.length
        for i in range(L - 1, n):
            v = volume[i - L + 1:i + 1]
            c = close[i - L + 1:i + 1]
            sv = np.sum(v)
            if sv > 0:
                out[i] = np.sum(c * v) / sv
        return out


# =====
# Aroon
# =====
class Aroon(Indicator):
    """
    Aroon Up/Down and Oscillator. Time since last HH/LL.
    source: HL [N×2]
    """

    name = 'aroon'
    category = 'trend'
    source_cols = ('high', 'low')

    def __init__(self, length: int = 25):
        self.length = length
        self.output_names = ['up', 'down', 'osc']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Aroon',
            color=['#0ECB81', '#F6465D', '#2563EB'],
            reference_lines=[
                {'value': 50.0, 'color': '#888888', 'dash': 'dashed', 'label': '50'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        high, low = source[:, 0], source[:, 1]
        L = self.length
        for i in range(L, n):
            hh_idx = L - np.argmax(high[i - L:i + 1])
            ll_idx = L - np.argmin(low[i - L:i + 1])
            out[0, i] = hh_idx / L * 100.0
            out[1, i] = ll_idx / L * 100.0
            out[2, i] = out[0, i] - out[1, i]
        return out


# =====
# Vortex
# =====
class Vortex(Indicator):
    """
    Vortex Indicator. Positive/Negative Directional Separation.
    source: HLC [N×3]
    """

    name = 'vortex'
    category = 'trend'
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['plus', 'minus', 'osc']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Vortex Indicator',
            color=['#0ECB81', '#F6465D', '#2563EB'],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        high, low, close = source[:, 0], source[:, 1], source[:, 2]
        prev_c = np.roll(close, 1); prev_c[0] = close[0]
        prev_h = np.roll(high, 1); prev_h[0] = high[0]
        prev_l = np.roll(low, 1); prev_l[0] = low[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
        vm_plus = np.abs(high - prev_l)
        vm_minus = np.abs(low - prev_h)
        for i in range(self.length, n):
            tr_sum = np.sum(tr[i - self.length + 1:i + 1])
            if tr_sum > 0:
                out[0, i] = np.sum(vm_plus[i - self.length + 1:i + 1]) / tr_sum
                out[1, i] = np.sum(vm_minus[i - self.length + 1:i + 1]) / tr_sum
                out[2, i] = out[0, i] - out[1, i]
        return out


# =====
# SchaffTrendCycle
# =====
class SchaffTrendCycle(Indicator):
    """
    Schaff Trend Cycle. MACD + stochastic.
    source: close [N]
    """

    name = 'stc'
    category = 'momentum'

    def __init__(self, fast: int = 23, slow: int = 50, cycle: int = 10):
        self.fast = fast
        self.slow = slow
        self.cycle = cycle
        self.length = slow
        self.output_names = ['stc']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Schaff Trend Cycle',
            reference_lines=[
                {'value': 80.0, 'color': '#F6465D', 'dash': 'dashed', 'label': '80'},
                {'value': 20.0, 'color': '#0ECB81', 'dash': 'dashed', 'label': '20'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.slow + self.cycle * 2

    def _ema(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return out
        alpha = 2.0 / (length + 1)
        out[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        macd_line = self._ema(source, self.fast) - self._ema(source, self.slow)
        first_macd = np.argmax(~np.isnan(macd_line))
        if np.isnan(macd_line[first_macd]):
            return out
        L = self.cycle
        prev = 50.0
        for i in range(first_macd, n):
            if np.isnan(macd_line[i]):
                out[i] = prev
                continue
            window_start = max(first_macd, i - L + 1)
            window = macd_line[window_start:i + 1]
            hh = float(np.max(window))
            ll = float(np.min(window))
            if hh == ll:
                stoch = 50.0
            else:
                stoch = (macd_line[i] - ll) / (hh - ll) * 100.0
            out[i] = prev + 0.5 * (stoch - prev)
            prev = out[i]
        return out


# =====
# Momentum (MT5 classic)
# =====
class Momentum(Indicator):
    """
    Momentum (classic MT5 version).
    MOM = close[i] / close[i - length] * 100

    Oscillates around 100: above indicates bullish momentum, below
    indicates bearish momentum. Distinct from ROC (which centers on 0).
    source: close [N]

    Usage:
        self.mom = self.add_indicator(self.btc.close_ref, Momentum(14))

    Access in on_data():
        self.mom.momentum[-1]   -> momentum value of the current candle
    """

    name = 'momentum'
    category = 'momentum'

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['momentum']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Momentum',
            reference_lines=[
                {'value': 100.0, 'color': '#888888', 'dash': 'dashed', 'label': '100'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        for i in range(L, n):
            if source[i - L] != 0:
                out[i] = source[i] / source[i - L] * 100.0
        return out


# =====
# OsMA (Moving Average of Oscillator)
# =====
class OsMA(Indicator):
    """
    Moving Average of Oscillator. It is the MACD histogram:
    OsMA = MACD_line - Signal_line

    where MACD_line = EMA(close, fast) - EMA(close, slow) and
    Signal_line = EMA(MACD_line, signal). Measures the acceleration of the
    MACD convergence/divergence.
    source: close [N]

    Usage:
        self.osma = self.add_indicator(self.btc.close_ref, OsMA(12, 26, 9))

    Access in on_data():
        self.osma.osma[-1]   -> OsMA histogram value
    """

    name = 'osma'
    category = 'momentum'

    # Recursive (EMA): reserve a few times `length` (= slow) as warmup.
    warmup_factor = 5

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.length = slow
        self.output_names = ['osma']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='OsMA',
            renderer='bar',
            bar_color_positive='#0ECB81',
            bar_color_negative='#F6465D',
            bar_alpha=0.75,
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal_period - 1

    def display_name(self) -> str:
        return f'OsMA({self.fast},{self.slow},{self.signal_period})'

    def col_name(self, symbol: str, col_source: str = '') -> str:
        return f'osma{self.fast}_{self.slow}_{self.signal_period}_{symbol}'

    def _ema_bulk(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        result = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return result
        alpha = 2.0 / (length + 1)
        result[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            result[i] = alpha * src[i] + (1.0 - alpha) * result[i - 1]
        return result

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)

        ema_fast = self._ema_bulk(source, self.fast)
        ema_slow = self._ema_bulk(source, self.slow)
        macd_line = ema_fast - ema_slow

        first_valid = np.argmax(~np.isnan(macd_line))
        if np.isnan(macd_line[first_valid]):
            return out

        macd_valid = macd_line[first_valid:]
        signal_seg = self._ema_bulk(macd_valid, self.signal_period)

        out[first_valid:] = macd_line[first_valid:] - signal_seg
        return out


# =====
# BullsPower (Elder)
# =====
class BullsPower(Indicator):
    """
    Bulls Power (Alexander Elder).
    BullsPower = high - EMA(close, length)

    Measures the buyers' ability to push the price above a consensus
    value (the EMA). Positive and increasing indicates buying strength.
    source: HLC [N x 3] - high(0), low(1), close(2)

    Usage:
        self.bulls = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.close_ref],
            BullsPower(13),
        )

    Access in on_data():
        self.bulls.bulls[-1]   -> Bulls Power value
    """

    name = 'bulls'
    category = 'momentum'
    warmup_factor = 5
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 13):
        self.length = length
        self.output_names = ['bulls']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Bulls Power',
            renderer='bar',
            bar_color_positive='#0ECB81',
            bar_color_negative='#F6465D',
            bar_alpha=0.75,
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def _ema_bulk(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        result = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return result
        alpha = 2.0 / (length + 1)
        result[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            result[i] = alpha * src[i] + (1.0 - alpha) * result[i - 1]
        return result

    def calculate(self, source: np.ndarray) -> np.ndarray:
        high = source[:, 0].astype(np.float64)
        close = source[:, 2].astype(np.float64)
        ema = self._ema_bulk(close, self.length)
        return high - ema


# =====
# BearsPower (Elder)
# =====
class BearsPower(Indicator):
    """
    Bears Power (Alexander Elder).
    BearsPower = low - EMA(close, length)

    Measures the sellers' ability to push the price below the consensus
    value (the EMA). Negative and decreasing indicates selling strength.
    source: HLC [N x 3] - high(0), low(1), close(2)

    Usage:
        self.bears = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.close_ref],
            BearsPower(13),
        )

    Access in on_data():
        self.bears.bears[-1]   -> Bears Power value
    """

    name = 'bears'
    category = 'momentum'
    warmup_factor = 5
    source_cols = ('high', 'low', 'close')

    def __init__(self, length: int = 13):
        self.length = length
        self.output_names = ['bears']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Bears Power',
            renderer='bar',
            bar_color_positive='#0ECB81',
            bar_color_negative='#F6465D',
            bar_alpha=0.75,
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def _ema_bulk(self, src: np.ndarray, length: int) -> np.ndarray:
        n = len(src)
        result = np.full(n, np.nan, dtype=np.float64)
        if n < length:
            return result
        alpha = 2.0 / (length + 1)
        result[length - 1] = float(np.mean(src[:length]))
        for i in range(length, n):
            result[i] = alpha * src[i] + (1.0 - alpha) * result[i - 1]
        return result

    def calculate(self, source: np.ndarray) -> np.ndarray:
        low = source[:, 1].astype(np.float64)
        close = source[:, 2].astype(np.float64)
        ema = self._ema_bulk(close, self.length)
        return low - ema



# =====
# DeMarker (DeM)
# =====
class DeMarker(Indicator):
    """
    DeMarker (Thomas DeMark). Exhaustion oscillator bounded in [0, 1]:

        DeMax[i] = max(high[i] - high[i-1], 0)
        DeMin[i] = max(low[i-1] - low[i], 0)
        DeM = SMA(DeMax, length) / (SMA(DeMax, length) + SMA(DeMin, length))

    Values above 0.7 suggest overbought; below 0.3 suggest oversold.
    source: HL [N x 2] - high(0), low(1)

    Usage:
        self.dem = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref], DeMarker(14),
        )

    Access in on_data():
        self.dem.demarker[-1]   -> DeMarker value (0..1)
    """

    name = 'demarker'
    category = 'momentum'
    source_cols = ('high', 'low')

    def __init__(self, length: int = 14):
        self.length = length
        self.output_names = ['demarker']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='DeMarker',
            reference_lines=[
                {'value': 0.7, 'color': '#F6465D', 'dash': 'dashed', 'label': '0.7'},
                {'value': 0.3, 'color': '#0ECB81', 'dash': 'dashed', 'label': '0.3'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        cs = np.cumsum(arr)
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L + 1:
            return out

        high = source[:, 0].astype(np.float64)
        low = source[:, 1].astype(np.float64)
        dh = np.diff(high, prepend=high[0])
        dl = -np.diff(low, prepend=low[0])
        de_max = np.where(dh > 0, dh, 0.0)
        de_min = np.where(dl > 0, dl, 0.0)
        # The first difference (i=0) is not real; it is zeroed out.
        de_max[0] = 0.0
        de_min[0] = 0.0

        sma_max = self._sma(de_max, L)
        sma_min = self._sma(de_min, L)
        denom = sma_max + sma_min
        valid = ~np.isnan(denom) & (denom > 0)
        out[valid] = sma_max[valid] / denom[valid]
        # Zero denominator (flat market) -> neutral value 0.5.
        flat = ~np.isnan(denom) & (denom == 0)
        out[flat] = 0.5
        return out


# =====
# RVI (Relative Vigor Index)
# =====
class RVI(Indicator):
    """
    Relative Vigor Index. Compares the close relative to the bar range,
    smoothed with a symmetric weighted average of 4 bars:

        co = close - open,  hl = high - low
        num = SMA(swma(co), length)
        den = SMA(swma(hl), length)
        RVI = num / den
        Signal = swma(RVI)

    where swma(x)[i] = (x[i] + 2*x[i-1] + 2*x[i-2] + x[i-3]) / 6. The crossover
    of RVI with its signal line generates the signals.
    source: OHLC [N x 4] - open(0), high(1), low(2), close(3)

    Usage:
        self.rvi = self.add_indicator(
            [self.btc.open_ref, self.btc.high_ref,
             self.btc.low_ref, self.btc.close_ref],
            RVI(10),
        )

    Access in on_data():
        self.rvi.rvi[-1]      -> RVI line
        self.rvi.signal[-1]   -> signal line
    """

    name = 'rvi'
    category = 'momentum'
    source_cols = ('open', 'high', 'low', 'close')

    def __init__(self, length: int = 10):
        self.length = length
        self.output_names = ['rvi', 'signal']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='RVI',
            color=['#2563EB', '#F59E0B'],
            line_width=[1.6, 1.2],
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.length + 3

    def _sma(self, arr: np.ndarray, L: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        cs = np.nancumsum(arr)
        out[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L
        return out

    def _swma(self, arr: np.ndarray) -> np.ndarray:
        """Symmetric weighted moving average of 4 bars: (x0 + 2x1 + 2x2 + x3) / 6."""
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(3, n):
            out[i] = (arr[i] + 2.0 * arr[i - 1] + 2.0 * arr[i - 2] + arr[i - 3]) / 6.0
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((2, n), np.nan, dtype=np.float64)
        L = self.length
        if n < L + 3:
            return out

        open_ = source[:, 0].astype(np.float64)
        high = source[:, 1].astype(np.float64)
        low = source[:, 2].astype(np.float64)
        close = source[:, 3].astype(np.float64)

        co = self._swma(close - open_)
        hl = self._swma(high - low)

        num = self._sma(co, L)
        den = self._sma(hl, L)

        rvi = np.full(n, np.nan, dtype=np.float64)
        valid = ~np.isnan(num) & ~np.isnan(den) & (den != 0)
        rvi[valid] = num[valid] / den[valid]

        out[0] = rvi
        out[1] = self._swma(rvi)
        return out

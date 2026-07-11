import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig

# =====
# SMA
# =====
class SMA(Indicator):
    """
    Simple Moving Average.
    source: close [N]

    Usage on ticks:
        self.sma_price = self.add_indicator(self.btc.price_ref, SMA(length=20))

    Usage on OHLC candles:
        self.btc_1m    = self.subscribe_ohlc("BTCUSDT", interval_min=1)
        self.sma_close = self.add_indicator(self.btc_1m.close_ref, SMA(length=10))
    """

    name = "sma"
    category = "trend"

    def __init__(self, length: int):
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return result
        cumsum = np.cumsum(source)
        result[L - 1:] = (cumsum[L - 1:] - np.concatenate(([0.0], cumsum[:-L]))) / L
        return result


# =====
# EMA
# =====
class EMA(Indicator):
    """
    Exponential Moving Average. alpha = 2 / (length + 1).
    Seed: SMA of the first `length` bars.
    source: close [N]

    Usage:
        self.ema_close = self.add_indicator(self.btc_1m.close_ref, EMA(length=20))
    """

    name = "ema"
    category = "trend"

    def __init__(self, length: int):
        self.length = length
        self._alpha = 2.0 / (length + 1)
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return result
        alpha = self._alpha
        result[L - 1] = float(np.mean(source[:L]))
        for i in range(L, n):
            result[i] = alpha * source[i] + (1.0 - alpha) * result[i - 1]
        return result


# =====
# MACD (multi-band: macd, signal, histogram)
# =====
class MACD(Indicator):
    """
    Moving Average Convergence Divergence.
    Produces 3 series: macd, signal, histogram.
    source: close [N]

    Args:
        fast   : fast EMA period (default 12)
        slow   : slow EMA period (default 26)
        signal : signal EMA period over MACD line (default 9)

    Usage:
        self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='1m')
        self.macd = self.add_indicator(
            self.btc.close_ref,
            MACD(12, 26, 9),
            plot=True, overlay=False,
        )

    Access in on_data():
        self.macd.macd[-1]      -> MACD value of current (partial) candle
        self.macd.signal[-1]    -> signal line
        self.macd.histogram[-1] -> histogram (MACD - Signal)

        # Bullish crossover:
        if self.macd.macd[-2] < self.macd.signal[-2] and \\
           self.macd.macd[-1] > self.macd.signal[-1]:
            ...  # golden cross

    Note:
        - The seed of each EMA is the SMA of the first `length` bars,
          same as the EMA in this module.
        - min_periods = slow + signal - 1 (minimum to have a valid histogram)
        - The histogram is rendered as colored bars (green/red).
    """

    name = "macd"
    category = "momentum"

    # Recursive (EMA): reserve a few times `length` (= slow) as warmup so the
    # value converges before on_data() starts (see auto_warmup_candles).
    warmup_factor = 5

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.length = slow

        self.output_names = ["macd", "signal", "histogram"]

        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_height=100,
            panel_title="MACD",
            autoscale=True,
            color=["#2563EB", "#F59E0B", "#888888"],
            line_width=[1.6, 1.2, 1.0],
            renderer=["line", "line", "bar"],
            bar_color_positive="#0ECB81",
            bar_color_negative="#F6465D",
            bar_alpha=0.75,
            reference_lines=[
                {"value": 0.0, "color": "#888888", "dash": "dashed",
                 "label": "", "width": 0.8},
            ],
        )

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal_period - 1

    def display_name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal_period})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"macd{self.fast}_{self.slow}_{self.signal_period}_{symbol}"

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
        out = np.full((3, n), np.nan, dtype=np.float64)

        ema_fast = self._ema_bulk(source, self.fast)
        ema_slow = self._ema_bulk(source, self.slow)

        macd_line = ema_fast - ema_slow

        first_valid = np.argmax(~np.isnan(macd_line))
        if np.isnan(macd_line[first_valid]):
            return out

        macd_valid_segment = macd_line[first_valid:]
        signal_segment = self._ema_bulk(macd_valid_segment, self.signal_period)

        out[0, first_valid:] = macd_line[first_valid:]
        out[1, first_valid:] = signal_segment
        out[2, first_valid:] = out[0, first_valid:] - out[1, first_valid:]

        return out


# =====
# WMA
# =====
class WMA(Indicator):
    """
    Weighted Moving Average (pesos lineales crecientes).
    source: close [N]

    Usage:
        self.wma = self.add_indicator(self.btc_1m.close_ref, WMA(length=20))
    """

    name = "wma"
    category = "trend"

    def __init__(self, length: int):
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return result
        weights = np.arange(1, L + 1, dtype=np.float64)
        weight_sum = weights.sum()
        for i in range(L - 1, n):
            result[i] = np.dot(source[i - L + 1:i + 1], weights) / weight_sum
        return result


# =====
# DEMA
# =====
class DEMA(Indicator):
    """
    Double Exponential Moving Average. Less lag than EMA.
    DEMA = 2 * EMA(src, L) - EMA(EMA(src, L), L)
    source: close [N]

    Usage:
        self.dema = self.add_indicator(self.btc_1m.close_ref, DEMA(length=20))
    """

    name = "dema"
    category = "trend"

    def __init__(self, length: int):
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return 2 * self.length - 1

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
        ema1 = self._ema(source, self.length)
        ema2 = self._ema(ema1, self.length)
        return 2.0 * ema1 - ema2


# =====
# TEMA
# =====
class TEMA(Indicator):
    """
    Triple Exponential Moving Average. Even less lag than DEMA.
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    source: close [N]

    Usage:
        self.tema = self.add_indicator(self.btc_1m.close_ref, TEMA(length=20))
    """

    name = "tema"
    category = "trend"

    def __init__(self, length: int):
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return 3 * self.length - 2

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
        ema1 = self._ema(source, self.length)
        ema2 = self._ema(ema1, self.length)
        ema3 = self._ema(ema2, self.length)
        return 3.0 * ema1 - 3.0 * ema2 + ema3


# =====
# KAMA
# =====
class KAMA(Indicator):
    """
    Kaufman Adaptive Moving Average. Adapts to volatility:
    fast in trends, slow in ranges.
    source: close [N]

    Usage:
        self.kama = self.add_indicator(self.btc_1m.close_ref, KAMA(length=10))
    """

    name = "kama"
    category = "trend"

    def __init__(self, length: int = 10, fast: int = 2, slow: int = 30):
        self.length = length
        self.fast = fast
        self.slow = slow
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L + 1:
            return result
        fast_sc = 2.0 / (self.fast + 1)
        slow_sc = 2.0 / (self.slow + 1)
        result[L] = float(np.mean(source[:L + 1]))
        for i in range(L + 1, n):
            direction = abs(source[i] - source[i - L])
            volatility = np.sum(np.abs(np.diff(source[i - L:i + 1])))
            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            result[i] = result[i - 1] + sc * (source[i] - result[i - 1])
        return result


# =====
# HMA
# =====
class HMA(Indicator):
    """
    Hull Moving Average. Minimal lag with optimal smoothing.
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    source: close [N]

    Usage:
        self.hma = self.add_indicator(self.btc_1m.close_ref, HMA(length=20))
    """

    name = "hma"
    category = "trend"

    def __init__(self, length: int):
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length

    def _wma(self, src: np.ndarray, L: int) -> np.ndarray:
        n = len(src)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < L:
            return out
        weights = np.arange(1, L + 1, dtype=np.float64)
        ws = weights.sum()
        for i in range(L - 1, n):
            out[i] = np.dot(src[i - L + 1:i + 1], weights) / ws
        return out

    def calculate(self, source: np.ndarray) -> np.ndarray:
        half = max(self.length // 2, 1)
        sqrt_L = max(int(np.sqrt(self.length)), 1)
        wma_half = self._wma(source, half)
        wma_full = self._wma(source, self.length)
        diff = 2.0 * wma_half - wma_full
        return self._wma(diff, sqrt_L)


# =====
# FRAMA (Fractal Adaptive Moving Average)
# =====
class FRAMA(Indicator):
    """
    Fractal Adaptive Moving Average (John Ehlers). Adjusts smoothing based on
    the fractal dimension of price: fast in trends, slow in ranges.

    For each window of `length` bars (length even), the fractal dimension D is
    estimated by comparing the volatility of the two halves to the total:

        N1 = (max-min of 1st half) / (length/2)
        N2 = (max-min of 2nd half) / (length/2)
        N3 = (max-min of total)    / length
        D  = (log(N1 + N2) - log(N3)) / log(2)

    and the smoothing factor alpha = exp(-4.6 * (D - 1)), clamped to [0.01, 1]:

        frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]

    source: close [N]

    Usage:
        self.frama = self.add_indicator(self.btc.close_ref, FRAMA(16))

    Access in on_data():
        self.frama[-1]   -> FRAMA value of the current candle
    """

    name = 'frama'
    category = 'trend'
    # Recursive: reserve warmup so it converges before on_data().
    warmup_factor = 5

    def __init__(self, length: int = 16):
        if length % 2 != 0:
            length += 1
        self.length = length
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return out

        half = L // 2
        log2 = np.log(2.0)
        # Seed with the SMA of the first complete window.
        out[L - 1] = float(np.mean(source[:L]))
        for i in range(L - 1, n):
            w = source[i - L + 1:i + 1]
            first = w[:half]
            second = w[half:]
            n1 = (np.max(first) - np.min(first)) / half
            n2 = (np.max(second) - np.min(second)) / half
            n3 = (np.max(w) - np.min(w)) / L
            if n1 > 0 and n2 > 0 and n3 > 0:
                d = (np.log(n1 + n2) - np.log(n3)) / log2
            else:
                d = 1.0
            alpha = np.exp(-4.6 * (d - 1.0))
            alpha = min(max(alpha, 0.01), 1.0)
            if i == L - 1:
                continue
            out[i] = alpha * source[i] + (1.0 - alpha) * out[i - 1]
        return out


# =====
# VIDYA (Variable Index Dynamic Average)
# =====
class VIDYA(Indicator):
    """
    Variable Index Dynamic Average (Tushar Chande). EMA whose smoothing factor
    is scaled by the strength of the trend measured with the Chande
    Momentum Oscillator (CMO):

        k = (2 / (length + 1)) * |CMO(cmo_period)|
        vidya[i] = close[i] * k + vidya[i-1] * (1 - k)

    In ranges (|CMO| low) it barely moves; in trends (|CMO| high) it follows
    price closely.
    source: close [N]

    Usage:
        self.vidya = self.add_indicator(self.btc.close_ref, VIDYA(9))

    Access in on_data():
        self.vidya[-1]   -> VIDYA value of the current candle
    """

    name = 'vidya'
    category = 'trend'
    warmup_factor = 5

    def __init__(self, length: int = 9, cmo_period: int = 9):
        self.length = length
        self.cmo_period = cmo_period
        self.plot_config = IndicatorPlotConfig()

    @property
    def min_periods(self) -> int:
        return self.cmo_period + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        P = self.cmo_period
        if n < P + 1:
            return out

        alpha = 2.0 / (self.length + 1)
        diff = np.diff(source, prepend=source[0])
        up = np.where(diff > 0, diff, 0.0)
        down = np.where(diff < 0, -diff, 0.0)

        start = P
        out[start] = source[start]
        for i in range(start + 1, n):
            su = np.sum(up[i - P + 1:i + 1])
            sd = np.sum(down[i - P + 1:i + 1])
            denom = su + sd
            cmo = (su - sd) / denom if denom > 0 else 0.0
            k = alpha * abs(cmo)
            out[i] = source[i] * k + out[i - 1] * (1.0 - k)
        return out

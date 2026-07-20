import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig

# =====
# BollingerBands (multi-band)
# =====
class BollingerBands(Indicator):
    """
    Bollinger Bands. Produces 3 series: upper, mid, lower.
    source: close [N]

    Usage:
        self.bb = self.add_indicator(
            self.btc.close_ref,
            BollingerBands(20, 2.0),
            name=["BB Upper", "BB Mid", "BB Lower"],
            plot=True, overlay=True,
            color=["#FF6B35", "#888888", "#3B82F6"],
        )

    Access:
        self.bb.upper[-1]   -> upper band of partial candle
        self.bb.mid[-1]     -> mid (SMA)
        self.bb.lower[-1]   -> lower band
        self.bb[0][-1]      -> equivalent to upper
    """

    name = "bb"
    category = "volatility"

    def __init__(self, length: int = 20, std_dev: float = 2.0):
        self.length = length
        self.std_dev = std_dev
        self.output_names = ["upper", "mid", "lower"]
        self.plot_config = IndicatorPlotConfig(
            color=["#3B82F6", "#3B82F6", "#3B82F6"],
            line_dash=["solid", "dashed", "solid"],
            line_width=[1.5, 1, 1.5],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full((3, n), np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return result

        cs = np.cumsum(source)
        sma = np.full(n, np.nan, dtype=np.float64)
        sma[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L

        cs2 = np.cumsum(source ** 2)
        mean_sq = np.full(n, np.nan, dtype=np.float64)
        mean_sq[L - 1:] = (cs2[L - 1:] - np.concatenate(([0.0], cs2[:-L]))) / L

        sigma = np.sqrt(np.maximum(mean_sq - sma ** 2, 0.0))

        result[0] = sma + self.std_dev * sigma
        result[1] = sma
        result[2] = sma - self.std_dev * sigma
        return result


# =====
# ATR (multi-source, single-band)
# =====
class ATR(Indicator):
    """
    Average True Range.
    source: HLC [N×3]

    Usage:
        self.bb = self.add_indicator(
            self.btc.close_ref,
            BollingerBands(20, 2.0),
            name=["BB Upper", "BB Mid", "BB Lower"],
            plot=True, overlay=True,
            color=["#FF6B35", "#888888", "#3B82F6"],
        )

    Access:
        self.bb.upper[-1]   -> upper band of partial candle
        self.bb.mid[-1]     -> mid (SMA)
        self.bb.lower[-1]   -> lower band
        self.bb[0][-1]      -> equivalent to upper
    """

    name = "atr"
    category = "volatility"
    source_cols = ("high", "low", "close")

    def __init__(self, length: int = 14):
        self.length = length
        self.plot_config = IndicatorPlotConfig(overlay=False)

    @property
    def min_periods(self) -> int:
        return self.length + 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L + 1:
            return result

        high = source[:, 0].astype(np.float64)
        low = source[:, 1].astype(np.float64)
        close = source[:, 2].astype(np.float64)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )

        alpha = 1.0 / L
        result[L] = float(np.mean(tr[1:L + 1]))
        for i in range(L + 1, n):
            result[i] = (1.0 - alpha) * result[i - 1] + alpha * tr[i]
        return result


# =====
# StdDev (Standard Deviation)
# =====
class StdDev(Indicator):
    """
    Standard Deviation. Rolling population standard deviation of price over
    a window of `length` bars (population, same criterion as this module's
    Bollinger Bands). Measures instantaneous volatility.
    source: close [N]

    Usage:
        self.sd = self.add_indicator(self.btc.close_ref, StdDev(20))

    Access in on_data():
        self.sd.stddev[-1]   -> standard deviation of the current candle
    """

    name = 'stddev'
    category = 'volatility'

    def __init__(self, length: int = 20):
        self.length = length
        self.output_names = ['stddev']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='StdDev',
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full(n, np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return out

        cs = np.cumsum(source)
        sma = np.full(n, np.nan, dtype=np.float64)
        sma[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L

        cs2 = np.cumsum(source ** 2)
        mean_sq = np.full(n, np.nan, dtype=np.float64)
        mean_sq[L - 1:] = (cs2[L - 1:] - np.concatenate(([0.0], cs2[:-L]))) / L

        out = np.sqrt(np.maximum(mean_sq - sma ** 2, 0.0))
        return out


# =====
# Envelopes (multi-band)
# =====
class Envelopes(Indicator):
    """
    Envelopes. Percentage bands around a simple moving average:
        mid   = SMA(close, length)
        upper = mid * (1 + deviation / 100)
        lower = mid * (1 - deviation / 100)

    Produce 3 series: upper, mid, lower. Drawn over the price.
    source: close [N]

    Usage:
        self.env = self.add_indicator(
            self.btc.close_ref, Envelopes(20, 0.5),
            name=['Env Upper', 'Env Mid', 'Env Lower'],
        )

    Access in on_data():
        self.env.upper[-1]   -> banda superior
        self.env.mid[-1]     -> media (SMA)
        self.env.lower[-1]   -> banda inferior
    """

    name = 'envelopes'
    category = 'volatility'

    def __init__(self, length: int = 20, deviation: float = 0.1):
        self.length = length
        self.deviation = deviation
        self.output_names = ['upper', 'mid', 'lower']
        self.plot_config = IndicatorPlotConfig(
            color=['#F59E0B', '#888888', '#F59E0B'],
            line_dash=['solid', 'dashed', 'solid'],
            line_width=[1.5, 1.0, 1.5],
        )

    @property
    def min_periods(self) -> int:
        return self.length

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        result = np.full((3, n), np.nan, dtype=np.float64)
        L = self.length
        if n < L:
            return result

        cs = np.cumsum(source)
        sma = np.full(n, np.nan, dtype=np.float64)
        sma[L - 1:] = (cs[L - 1:] - np.concatenate(([0.0], cs[:-L]))) / L

        k = self.deviation / 100.0
        result[0] = sma * (1.0 + k)
        result[1] = sma
        result[2] = sma * (1.0 - k)
        return result

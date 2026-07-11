"""
Bill Williams indicator set.

Implements the classic Bill Williams studies as offered by MT5:
    - Alligator            (three shifted smoothed moving averages)
    - GatorOscillator      (convergence/divergence of the Alligator lines)
    - AcceleratorOscillator (AC - acceleration of the Awesome Oscillator)
    - Fractals             (five-bar swing highs/lows)
    - MarketFacilitationIndex (BW MFI: price range per unit of volume)

All follow the same declarative contract as the rest of ``tradetropy.ta``:
subclass :class:`Indicator`, implement ``calculate`` (vectorized, NaN warmup)
and assign a :class:`IndicatorPlotConfig`. Detection is causal (values use only
past/current bars), so they are safe for backtesting and identical in live.
"""

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig


def _smma(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Smoothed Moving Average (Wilder smoothing), same as MT5 MODE_SMMA.

    The seed is the SMA of the first `length` values; afterwards
    smma[i] = (smma[i-1] * (length - 1) + arr[i]) / length.

    Args:
        arr (np.ndarray): Input series [N].
        length (int): Smoothing period.

    Returns:
        np.ndarray: SMMA series [N] with NaN during warmup.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    out[length - 1] = float(np.mean(arr[:length]))
    for i in range(length, n):
        out[i] = (out[i - 1] * (length - 1) + arr[i]) / length
    return out


def _shift_forward(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Displace a series `shift` bars into the future (MT5 forward shift).

    The value computed at bar i is placed at bar i + shift, so reading it at
    the current bar returns a value derived only from past data (causal). The
    first `shift` positions become NaN.

    Args:
        arr (np.ndarray): Input series [N].
        shift (int): Number of bars to shift forward (>= 0).

    Returns:
        np.ndarray: Shifted series [N].
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if shift <= 0:
        return arr.copy()
    if shift < n:
        out[shift:] = arr[:n - shift]
    return out


def _sma(arr: np.ndarray, length: int) -> np.ndarray:
    """Simple moving average [N] with NaN warmup."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    cs = np.cumsum(arr)
    out[length - 1:] = (cs[length - 1:] - np.concatenate(([0.0], cs[:-length]))) / length
    return out


# =====
# Alligator
# =====
class Alligator(Indicator):
    """
    Bill Williams Alligator. Three smoothed moving averages (SMMA) on the
    median price (high + low) / 2, shifted forward:

        Jaw   : SMMA(13), shift 8
        Teeth : SMMA(8),  shift 5
        Lips  : SMMA(5),  shift 3

    The shift is causal: the value shown at bar i comes from earlier data
    (bar i - shift), so reading it in on_data() never uses future information.
    source: HL [N x 2] - high(0), low(1)

    Usage:
        self.alli = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref],
            Alligator(),
            name=['Jaw', 'Teeth', 'Lips'],
        )

    Access in on_data():
        self.alli.jaw[-1]     -> jaw (blue line)
        self.alli.teeth[-1]   -> teeth (red line)
        self.alli.lips[-1]    -> lips (green line)
    """

    name = 'alligator'
    category = 'trend'
    # Recursive (SMMA): reserve warmup so it converges before on_data().
    warmup_factor = 5

    def __init__(
        self,
        jaw_period: int = 13, jaw_shift: int = 8,
        teeth_period: int = 8, teeth_shift: int = 5,
        lips_period: int = 5, lips_shift: int = 3,
    ):
        self.jaw_period = jaw_period
        self.jaw_shift = jaw_shift
        self.teeth_period = teeth_period
        self.teeth_shift = teeth_shift
        self.lips_period = lips_period
        self.lips_shift = lips_shift
        self.length = jaw_period + jaw_shift
        self.output_names = ['jaw', 'teeth', 'lips']
        self.plot_config = IndicatorPlotConfig(
            color=['#2563EB', '#F6465D', '#0ECB81'],
            line_width=[1.4, 1.4, 1.4],
        )

    @property
    def min_periods(self) -> int:
        return max(
            self.jaw_period + self.jaw_shift,
            self.teeth_period + self.teeth_shift,
            self.lips_period + self.lips_shift,
        )

    def display_name(self) -> str:
        return 'Alligator'

    def col_name(self, symbol: str, col_source: str = '') -> str:
        return f'alligator_{self.jaw_period}_{self.teeth_period}_{self.lips_period}_{symbol}'

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((3, n), np.nan, dtype=np.float64)
        mid = (source[:, 0].astype(np.float64) + source[:, 1].astype(np.float64)) / 2.0
        out[0] = _shift_forward(_smma(mid, self.jaw_period), self.jaw_shift)
        out[1] = _shift_forward(_smma(mid, self.teeth_period), self.teeth_shift)
        out[2] = _shift_forward(_smma(mid, self.lips_period), self.lips_shift)
        return out


# =====
# GatorOscillator
# =====
class GatorOscillator(Indicator):
    """
    Gator Oscillator. Shows the convergence/divergence of the Alligator
    lines as two histograms:

        upper =  |jaw - teeth|     (bars above zero)
        lower = -|teeth - lips|    (bars below zero)

    source: HL [N x 2] - high(0), low(1)

    Usage:
        self.gator = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref], GatorOscillator(),
        )

    Access in on_data():
        self.gator.upper[-1]   -> |jaw - teeth|
        self.gator.lower[-1]   -> -|teeth - lips|
    """

    name = 'gator'
    category = 'momentum'
    warmup_factor = 5

    def __init__(
        self,
        jaw_period: int = 13, jaw_shift: int = 8,
        teeth_period: int = 8, teeth_shift: int = 5,
        lips_period: int = 5, lips_shift: int = 3,
    ):
        self.jaw_period = jaw_period
        self.jaw_shift = jaw_shift
        self.teeth_period = teeth_period
        self.teeth_shift = teeth_shift
        self.lips_period = lips_period
        self.lips_shift = lips_shift
        self.length = jaw_period + jaw_shift
        self.output_names = ['upper', 'lower']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Gator Oscillator',
            renderer='bar',
            color=['#0ECB81', '#F6465D'],
            bar_alpha=0.75,
            reference_lines=[
                {'value': 0.0, 'color': '#888888', 'dash': 'dashed', 'label': '0'},
            ],
        )

    @property
    def min_periods(self) -> int:
        return max(
            self.jaw_period + self.jaw_shift,
            self.teeth_period + self.teeth_shift,
            self.lips_period + self.lips_shift,
        )

    def display_name(self) -> str:
        return 'Gator'

    def col_name(self, symbol: str, col_source: str = '') -> str:
        return f'gator_{self.jaw_period}_{self.teeth_period}_{self.lips_period}_{symbol}'

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((2, n), np.nan, dtype=np.float64)
        mid = (source[:, 0].astype(np.float64) + source[:, 1].astype(np.float64)) / 2.0
        jaw = _shift_forward(_smma(mid, self.jaw_period), self.jaw_shift)
        teeth = _shift_forward(_smma(mid, self.teeth_period), self.teeth_shift)
        lips = _shift_forward(_smma(mid, self.lips_period), self.lips_shift)
        out[0] = np.abs(jaw - teeth)
        out[1] = -np.abs(teeth - lips)
        return out


# =====
# AcceleratorOscillator
# =====
class AcceleratorOscillator(Indicator):
    """
    Accelerator Oscillator (AC) by Bill Williams. Measures the acceleration
    of the Awesome Oscillator:

        AO = SMA(median, 5) - SMA(median, 34)
        AC = AO - SMA(AO, 5)

    where median = (high + low) / 2. Anticipates changes in the AO.
    source: HL [N x 2] - high(0), low(1)

    Usage:
        self.ac = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref], AcceleratorOscillator(),
        )

    Access in on_data():
        self.ac.ac[-1]   -> Accelerator Oscillator value
    """

    name = 'ac'
    category = 'momentum'

    def __init__(self):
        self.length = 34
        self.output_names = ['ac']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='Accelerator Oscillator',
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
        return 34 + 5 - 1

    def calculate(self, source: np.ndarray) -> np.ndarray:
        mid = (source[:, 0].astype(np.float64) + source[:, 1].astype(np.float64)) / 2.0
        ao = _sma(mid, 5) - _sma(mid, 34)
        n = len(ao)
        out = np.full(n, np.nan, dtype=np.float64)
        # SMA(AO, 5) over the valid region (AO has NaN during warmup;
        # _sma with cumsum would propagate them, so the valid segment is
        # isolated).
        first = int(np.argmax(~np.isnan(ao)))
        if np.isnan(ao[first]):
            return out
        sma_ao = _sma(ao[first:], 5)
        out[first:] = ao[first:] - sma_ao
        return out


# =====
# Fractals
# =====
class Fractals(Indicator):
    """
    Bill Williams Fractals. Marks buy and sell fractals using the classic
    5-bar rule:

        - Bullish fractal (up)   : the center bar's high is greater than
          the highs of the two bars on each side.
        - Bearish fractal (down) : the center bar's low is less than
          the lows of the two bars on each side.

    The value appears on the confirmation bar (center + 2), never on the
    center bar, so it is strictly causal: the last 2 bars are
    always NaN (no right window large enough).
    source: HL [N x 3] - high(0), low(1), ts_ms(2)

    Returns [4 x N]:
        row 0 : up     - bullish fractal price at its confirmation bar
        row 1 : down   - bearish fractal price at its confirmation bar
        row 2 : _up_ts   - ts_ms of the real central bar (internal - plotting)
        row 3 : _down_ts - ts_ms of the real central bar (internal - plotting)

    Usage:
        self.frac = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
            Fractals(),
        )

    Access in on_data():
        self.frac.up[-1]     -> confirmed bullish fractal price (or NaN)
        self.frac.down[-1]   -> confirmed bearish fractal price (or NaN)
    """

    name = 'fractals'
    category = 'structure'
    output_names = ['up', 'down']
    ts_band_indices = [2, 3]
    ts_output_names = ['ts_up', 'ts_down']

    def __init__(self, n: int = 2):
        self.n = n
        self.length = n
        self.plot_config = IndicatorPlotConfig(
            renderer='scatter',
            marker=['inverted_triangle', 'triangle'],
            marker_size=10,
            color=['#F6465D', '#0ECB81'],
        )

    @property
    def min_periods(self) -> int:
        return self.n * 2 + 1

    def display_name(self) -> str:
        return 'Fractals'

    def col_name(self, symbol: str, col_source: str = '') -> str:
        return f'fractals{self.n}_{symbol}'

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        K = self.n
        up = np.full(n, np.nan, dtype=np.float64)
        down = np.full(n, np.nan, dtype=np.float64)
        up_ts = np.full(n, np.nan, dtype=np.float64)
        down_ts = np.full(n, np.nan, dtype=np.float64)

        if n < K * 2 + 1:
            return np.vstack([up, down, up_ts, down_ts])

        high = source[:, 0].astype(np.float64)
        low = source[:, 1].astype(np.float64)
        ts_ms = source[:, 2].astype(np.float64)

        for ci in range(K, n - K):
            confirm_idx = ci + K
            window_h = high[ci - K:ci + K + 1]
            if high[ci] == np.max(window_h) and np.argmax(window_h) == K:
                up[confirm_idx] = high[ci]
                up_ts[confirm_idx] = ts_ms[ci]
            window_l = low[ci - K:ci + K + 1]
            if low[ci] == np.min(window_l) and np.argmin(window_l) == K:
                down[confirm_idx] = low[ci]
                down_ts[confirm_idx] = ts_ms[ci]

        return np.vstack([up, down, up_ts, down_ts])


# =====
# MarketFacilitationIndex (BW MFI)
# =====
class MarketFacilitationIndex(Indicator):
    """
    Market Facilitation Index (BW MFI) by Bill Williams. Measures how far
    the price moves per unit of volume:

        BW MFI = (high - low) / volume

    (scaled by point in MT5; here left in price/volume units).
    Do not confuse with the volume-weighted Money Flow Index (MFI) in the
    momentum module.
    source: HLV [N x 3] - high(0), low(1), volume(2)

    Usage:
        self.bwmfi = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.volume_ref],
            MarketFacilitationIndex(),
        )

    Access in on_data():
        self.bwmfi.bwmfi[-1]   -> Market Facilitation Index value
    """

    name = 'bwmfi'
    category = 'volume'

    def __init__(self):
        self.length = 1
        self.output_names = ['bwmfi']
        self.plot_config = IndicatorPlotConfig(
            overlay=False,
            panel_title='BW MFI',
            renderer='bar',
            bar_alpha=0.75,
        )

    @property
    def min_periods(self) -> int:
        return 1

    def display_name(self) -> str:
        return 'BW MFI'

    def calculate(self, source: np.ndarray) -> np.ndarray:
        high = source[:, 0].astype(np.float64)
        low = source[:, 1].astype(np.float64)
        volume = source[:, 2].astype(np.float64)
        out = np.full(len(high), np.nan, dtype=np.float64)
        nz = volume > 0
        out[nz] = (high[nz] - low[nz]) / volume[nz]
        return out

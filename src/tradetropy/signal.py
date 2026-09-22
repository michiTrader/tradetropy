"""
tradetropy/signal.py
=================
Signal - condition detection functions over price series.

Independent class: does not require Strategy or any other framework component.
The user instantiates it wherever needed and combines it as desired.

    sig = Signal("closed")   # evaluates over closed candles (shift=1)
    sig = Signal("partial")  # evaluates including the partial candle/tick (shift=0)

"closed" -> ignores the current partial candle/tick, compares the two last
            CLOSED candles. Recommended in tick mode to avoid signals on
            incomplete data.

"partial" -> uses [-1] as the current value (partial or last tick).
             Correct in klines mode (each on_data is already a closed candle)
             or when the user wants to see real-time state.

Usage in strategy:
    class MyStrategy(Strategy):
        def init(self):
            self.ohlc  = self.subscribe_ohlc("BTCUSDT", 60_000)
            self.fast  = self.add_indicator(self.ohlc.close_ref, SMA(10))
            self.slow  = self.add_indicator(self.ohlc.close_ref, SMA(20))
            self.sig   = Signal("closed")

        def on_data(self):
            if self.sig.crossover(self.fast, self.slow):
                self.sesh.buy(...)
            if self.sig.crossunder(self.fast, self.slow):
                self.sesh.sell(...)

Standalone usage (outside strategy):
    sig = Signal()
    if sig.crossover(series_a, series_b):
        ...
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class Signal:
    """
    Condition detection functions over indexable series.

    Accepts any object that supports len() and negative indexing:
    WindowView, IndicatorProxy, MultiBandProxy, np.ndarray, list, etc.

    Args:
        mode: "closed" or "partial".
            "closed"  - evaluates over closed candles: compares [-2] vs [-3]
                        instead of [-1] vs [-2]. Recommended in tick mode to
                        avoid partial signals.
            "partial" - evaluates including the current value ([-1]), no shift.
                        Correct in klines mode or for point-in-time conditions.
        shift (int): Legacy kwarg. Use ``mode`` instead.
            1 → "closed", 0 → "partial".
    """

    def __init__(
        self,
        mode: Literal["closed", "partial"] = "partial",
        *,
        shift: int | None = None,
    ):
        if shift is not None:
            mode = "closed" if shift else "partial"
        if mode not in ("closed", "partial"):
            raise ValueError(f"mode must be 'closed' or 'partial', got {mode!r}")
        self._shift: int = 1 if mode == "closed" else 0

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    @staticmethod
    def _has_len(a, min_len: int) -> bool:
        try:
            return len(a) >= min_len
        except TypeError:
            return False

    @staticmethod
    def _f(a, idx: int) -> float:
        return float(a[idx])

    # ==========================================================================
    # Crossovers between two series
    # ==========================================================================

    def crossover(self, a, b) -> bool:
        """
        True when 'a' crosses above 'b'.

        Condition: a[-2] < b[-2] and a[-1] > b[-1].
        With shift=1, the indices are shifted one position backwards.

        Args:
            a: First series (supports indexing and len()).
            b: Second series (supports indexing and len()).

        Returns:
            bool: True if crossover detected, False otherwise.

        Example:
            if sig.crossover(sma_fast, sma_slow):
                sesh.buy(...)
        """
        o = self._shift
        if not self._has_len(a, 2 + o) or not self._has_len(b, 2 + o):
            return False
        return self._f(a, -2 - o) < self._f(b, -2 - o) and self._f(a, -1 - o) > self._f(b, -1 - o)

    def crossunder(self, a, b) -> bool:
        """
        True when 'a' crosses below 'b'.

        Condition: a[-2] > b[-2] and a[-1] < b[-1].
        With shift=1, the indices are shifted one position backwards.

        Args:
            a: First series (supports indexing and len()).
            b: Second series (supports indexing and len()).

        Returns:
            bool: True if crossunder detected, False otherwise.

        Example:
            if sig.crossunder(sma_fast, sma_slow):
                sesh.sell(...)
        """
        o = self._shift
        if not self._has_len(a, 2 + o) or not self._has_len(b, 2 + o):
            return False
        return self._f(a, -2 - o) > self._f(b, -2 - o) and self._f(a, -1 - o) < self._f(b, -1 - o)

    def cross(self, a, b) -> bool:
        """
        True if 'a' crossed 'b' in either direction.

        Equivalent to: crossover(a, b) or crossunder(a, b)

        Args:
            a: First series (supports indexing and len()).
            b: Second series (supports indexing and len()).

        Returns:
            bool: True if cross detected in either direction, False otherwise.

        Example:
            if sig.cross(price, sma):
                log.signal('cross detected')
        """
        o = self._shift
        if not self._has_len(a, 2 + o) or not self._has_len(b, 2 + o):
            return False
        a1, a2 = self._f(a, -1 - o), self._f(a, -2 - o)
        b1, b2 = self._f(b, -1 - o), self._f(b, -2 - o)
        return (a2 < b2 and a1 > b1) or (a2 > b2 and a1 < b1)

    # ==========================================================================
    # Crossovers against a fixed level
    # ==========================================================================

    def crossover_level(self, a, level: float) -> bool:
        """
        True when 'a' crosses above a fixed level.

        Condition: a[-2] < level and a[-1] > level

        Args:
            a: Series to check (supports indexing and len()).
            level (float): The reference level.

        Returns:
            bool: True if crossover detected, False otherwise.

        Example:
            if sig.crossover_level(rsi, 30):
                # RSI exits oversold
        """
        o = self._shift
        if not self._has_len(a, 2 + o):
            return False
        return self._f(a, -2 - o) < level < self._f(a, -1 - o)

    def crossunder_level(self, a, level: float) -> bool:
        """
        True when 'a' crosses below a fixed level.

        Condition: a[-2] > level and a[-1] < level

        Args:
            a: Series to check (supports indexing and len()).
            level (float): The reference level.

        Returns:
            bool: True if crossunder detected, False otherwise.

        Example:
            if sig.crossunder_level(rsi, 70):
                # RSI exits overbought
        """
        o = self._shift
        if not self._has_len(a, 2 + o):
            return False
        return self._f(a, -2 - o) > level > self._f(a, -1 - o)

    # ==========================================================================
    # Relative position
    # ==========================================================================

    def above(self, a, b) -> bool:
        """
        True if a[-1 - shift] > b[-1 - shift].

        With shift=1 evaluates the last closed candle; with shift=0
        evaluates the current value.

        Args:
            a: First series (supports indexing and len()).
            b: Second series (supports indexing and len()).

        Returns:
            bool: True if a is above b, False otherwise.
        """
        o = self._shift
        if not self._has_len(a, 1 + o) or not self._has_len(b, 1 + o):
            return False
        return self._f(a, -1 - o) > self._f(b, -1 - o)

    def below(self, a, b) -> bool:
        """
        True if a[-1 - shift] < b[-1 - shift].

        With shift=1 evaluates the last closed candle; with shift=0
        evaluates the current value.

        Args:
            a: First series (supports indexing and len()).
            b: Second series (supports indexing and len()).

        Returns:
            bool: True if a is below b, False otherwise.
        """
        o = self._shift
        if not self._has_len(a, 1 + o) or not self._has_len(b, 1 + o):
            return False
        return self._f(a, -1 - o) < self._f(b, -1 - o)

    def value_above(self, a, level: float) -> bool:
        """
        True if a[-1 - shift] > level.

        With shift=1 evaluates the last closed candle; with shift=0
        evaluates the current value.

        Args:
            a: Series to check (supports indexing and len()).
            level (float): The reference level.

        Returns:
            bool: True if value is above level, False otherwise.
        """
        o = self._shift
        if not self._has_len(a, 1 + o):
            return False
        return self._f(a, -1 - o) > level

    def value_below(self, a, level: float) -> bool:
        """
        True if a[-1 - shift] < level.

        With shift=1 evaluates the last closed candle; with shift=0
        evaluates the current value.

        Args:
            a: Series to check (supports indexing and len()).
            level (float): The reference level.

        Returns:
            bool: True if value is below level, False otherwise.
        """
        o = self._shift
        if not self._has_len(a, 1 + o):
            return False
        return self._f(a, -1 - o) < level

    def in_range(self, a, low: float, high: float) -> bool:
        """
        True if low <= a[-1 - shift] <= high.

        With shift=1 evaluates the last closed candle; with shift=0
        evaluates the current value.

        Args:
            a: Series to check (supports indexing and len()).
            low (float): Lower bound.
            high (float): Upper bound.

        Returns:
            bool: True if value is within range, False otherwise.
        """
        o = self._shift
        if not self._has_len(a, 1 + o):
            return False
        return low <= self._f(a, -1 - o) <= high

    # ==========================================================================
    # Direction
    # ==========================================================================

    def rising(self, a, n: int = 1) -> bool:
        """
        True if a[-1] > a[-1 - n].

        Args:
            a: Series to check (supports indexing and len()).
            n (int): Number of bars to look back. Default 1.
                n=1 - last value greater than previous.
                n=3 - last value greater than 3 bars ago.

        Returns:
            bool: True if series is rising, False otherwise.

        Example:
            if sig.rising(volume, n=3):
                # volume expanding
        """
        o = self._shift
        if not self._has_len(a, 1 + o + n):
            return False
        return self._f(a, -1 - o) > self._f(a, -1 - o - n)

    def falling(self, a, n: int = 1) -> bool:
        """
        True if a[-1] < a[-1 - n].

        Args:
            a: Series to check (supports indexing and len()).
            n (int): Number of bars to look back. Default 1.

        Returns:
            bool: True if series is falling, False otherwise.

        Example:
            if sig.falling(rsi, n=2):
                # RSI falling for 2 bars
        """
        o = self._shift
        if not self._has_len(a, 1 + o + n):
            return False
        return self._f(a, -1 - o) < self._f(a, -1 - o - n)

    def turning_up(self, a) -> bool:
        """
        True if 'a' changes direction upward (local valley).

        Condition: a[-3] > a[-2] and a[-1] > a[-2]

        Args:
            a: Series to check (supports indexing and len()).

        Returns:
            bool: True if turning up detected, False otherwise.

        Example:
            if sig.turning_up(rsi):
                # RSI turned upward
        """
        o = self._shift
        if not self._has_len(a, 3 + o):
            return False
        return self._f(a, -3 - o) > self._f(a, -2 - o) and self._f(a, -1 - o) > self._f(a, -2 - o)

    def turning_down(self, a) -> bool:
        """
        True if 'a' changes direction downward (local peak).

        Condition: a[-3] < a[-2] and a[-1] < a[-2]

        Args:
            a: Series to check (supports indexing and len()).

        Returns:
            bool: True if turning down detected, False otherwise.

        Example:
            if sig.turning_down(macd.histogram):
                # histogram turned downward
        """
        o = self._shift
        if not self._has_len(a, 3 + o):
            return False
        return self._f(a, -3 - o) < self._f(a, -2 - o) and self._f(a, -1 - o) < self._f(a, -2 - o)

    def streak(self, a) -> int:
        """
        Number of consecutive bars in the same direction.

        Returns:
            int: Positive for rising streak, negative for falling,
                 0 for no change or insufficient data.

        Example:
            s = sig.streak(btc.close)
            if s >= 3:
                # three consecutive bullish candles
            if s <= -3:
                # three consecutive bearish candles
        """
        o = self._shift
        n = len(a)
        if n < 2 + o:
            return 0

        last = self._f(a, -1 - o)
        prev = self._f(a, -2 - o)

        if last > prev:
            direction = 1
        elif last < prev:
            direction = -1
        else:
            return 0

        count = 1
        # max bars to look back (without going past array bounds)
        max_back = n - 1 - o
        for i in range(2, max_back + 1):
            curr = self._f(a, -i - o)
            nxt  = self._f(a, -i - 1 - o) if i + 1 + o <= n - 1 else None
            if nxt is None:
                break
            if direction == 1 and curr > nxt:
                count += 1
            elif direction == -1 and curr < nxt:
                count += 1
            else:
                break

        return count * direction

    # ==========================================================================
    # Extremes in window
    # ==========================================================================

    def highest(self, a, n: int) -> bool:
        """
        True if a[-1] is the maximum of the last n bars (inclusive).

        Args:
            a: Series to check (supports indexing and len()).
            n (int): Number of bars to check.

        Returns:
            bool: True if highest found, False otherwise.

        Example:
            if sig.highest(btc.high, 20):
                # new 20-bar high
        """
        o = self._shift
        if not self._has_len(a, n + o):
            return False
        window = [self._f(a, -i - o) for i in range(1, n + 1)]
        return window[0] == max(window)

    def lowest(self, a, n: int) -> bool:
        """
        True if a[-1] is the minimum of the last n bars (inclusive).

        Args:
            a: Series to check (supports indexing and len()).
            n (int): Number of bars to check.

        Returns:
            bool: True if lowest found, False otherwise.

        Example:
            if sig.lowest(btc.low, 20):
                # new 20-bar low
        """
        o = self._shift
        if not self._has_len(a, n + o):
            return False
        window = [self._f(a, -i - o) for i in range(1, n + 1)]
        return window[0] == min(window)

    # ==========================================================================
    # State change
    # ==========================================================================

    def changed(self, a) -> bool:
        """
        True if a[-1] != a[-2].

        Useful for discrete series (flags, regimes, states).

        Args:
            a: Series to check (supports indexing and len()).

        Returns:
            bool: True if value changed, False otherwise.

        Example:
            if sig.changed(regime):
                log.signal('regime change -> %s', regime[-1])
        """
        o = self._shift
        if not self._has_len(a, 2 + o):
            return False
        return self._f(a, -1 - o) != self._f(a, -2 - o)

    # ==========================================================================
    # Multi-series
    # ==========================================================================

    def all_rising(self, *series, n: int = 1) -> bool:
        """
        True if all series are rising (rising with n bars).

        Args:
            *series: Variable number of series to check.
            n (int): Number of bars to look back. Default 1.

        Returns:
            bool: True if all series are rising, False otherwise.

        Example:
            if sig.all_rising(ema_fast, ema_slow):
                # both EMAs rising
        """
        if not series:
            return False
        o = self._shift
        if not all(self._has_len(s, 1 + o + n) for s in series):
            return False
        return all(self._f(s, -1 - o) > self._f(s, -1 - o - n) for s in series)

    def all_falling(self, *series, n: int = 1) -> bool:
        """
        True if all series are falling (falling with n bars).

        Args:
            *series: Variable number of series to check.
            n (int): Number of bars to look back. Default 1.

        Returns:
            bool: True if all series are falling, False otherwise.

        Example:
            if sig.all_falling(ema_fast, ema_slow):
                # both EMAs falling
        """
        if not series:
            return False
        o = self._shift
        if not all(self._has_len(s, 1 + o + n) for s in series):
            return False
        return all(self._f(s, -1 - o) < self._f(s, -1 - o - n) for s in series)

from __future__ import annotations

import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.draw import Labels, Segments
from tradetropy.ta.structure._utils import extract_confirmed_pivots
from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL MAPS
# ══════════════════════════════════════════════════════════════════════════════

_MS_PRICE_IDX: dict[str, int] = {
    "BOS-bull": 0, "BOS-bear": 1, "CHOCH-bull": 2, "CHOCH-bear": 3,
}
_MS_TS_IDX: dict[str, int] = {k: v + 4 for k, v in _MS_PRICE_IDX.items()}

_MS_TAG_CODE: dict[str, float] = {
    "BOS-bull": 0.0, "BOS-bear": 1.0, "CHOCH-bull": 2.0, "CHOCH-bear": 3.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# MarketStructure — BOS / CHoCH  (Smart Money Concepts)
# ══════════════════════════════════════════════════════════════════════════════

class MarketStructure(Indicator, PivotIndicatorMixin):
    """
    Break of Structure (BOS) and Change of Character (CHoCH), strictly causal.

    Tracks the last *confirmed* swing high and swing low (via
    :class:`ConfirmedPivot`) and flags the bar on which price breaks one of
    them:

    - **BOS**   — the break continues the prevailing directional bias.
    - **CHoCH** — the break goes against the prevailing bias, i.e. the
      structure changes character. The bias flips on that bar.

    The first break of the series has no prior bias to contradict, so it is
    tagged BOS and simply seeds the bias.

    Causality
    ─────────
    A swing level only becomes breakable on bars *after* the bar where
    :class:`ConfirmedPivot` made it causally knowable (its confirmation bar),
    never from the real swing bar. Anchoring the level at the real swing bar
    would let the detector break a level the strategy could not yet know about
    — textbook lookahead, and the usual bug in SMC implementations. The value
    is therefore emitted on the break bar, while the *real* swing timestamp
    travels in the ``ts_*`` bands purely so plotting can draw the structure
    line back to its origin (same split as ``PivotHighLow`` / ``HHLL``).

    Output — [9 × N] float64:
        b0..b3  : broken level price (on the break bar)
                  b0=bos_bull  b1=bos_bear  b2=choch_bull  b3=choch_bear
        b4..b7  : ts_real of the BROKEN SWING bar (line origin — plotting)
                  same order: ts_bos_bull, ts_bos_bear,
                              ts_choch_bull, ts_choch_bear
        b8      : encoded tag (internal — PatternMatcher / not exposed)
                  0=BOS-bull 1=BOS-bear 2=CHOCH-bull 3=CHOCH-bear

    source : [N × 4] — high(0), low(1), close(2), ts_ms(3)

    Parameters
    ──────────
    swing    : int — ConfirmedPivot window (default 2)
    break_on : str — "close" (default) requires a candle CLOSE beyond the
               level; "wick" accepts a high/low piercing it. "close" is the
               stricter, more common SMC reading and produces fewer false
               breaks.

    Access in on_data()  —  READ THE CLOSED BAR AT ``[-2]``
    ───────────────────────────────────────────────────────
    This detector is ``use_partial = False``: it only evaluates *closed*
    candles, while the proxy window still carries the forming candle in its
    last slot. The freshest structure event therefore lands on ``[-2]``, and
    ``[-1]`` (the partial bar) is **always NaN**::

        def on_data(self):
            if not np.isnan(self.ms.choch_bull[-2]):
                ...   # bearish -> bullish shift on the bar that just closed

    Reading ``[-1]`` is the classic mistake here: it never fires, so the
    strategy silently never trades. Use the :meth:`event` helper to stay
    agnostic of the offset::

        tag = self.ms.event(self.ms)          # 'CHOCH-bull' | None

    Bands available (all NaN on bars without that event):
        bos_bull, bos_bear, choch_bull, choch_bear      → broken level price
        ts_bos_bull, ts_bos_bear, ts_choch_bull,
        ts_choch_bear                                   → ts of real swing bar

    Window sizing
    ─────────────
    This is a *stateful* structure detector: the bias and the active swing
    levels are rebuilt from the start of whatever window it is given. Give the
    OHLC proxy a ``window_size`` comfortably larger than the structure you care
    about (a few hundred bars) so the bias is stable; with a window barely
    above ``min_periods`` the detector keeps re-seeding its bias and every
    break degenerates into a BOS.

    PatternMatcher
    ──────────────
    Implements PivotIndicatorMixin, so it can decorate a base pivot.
    Available tags: 'BOS-bull' | 'BOS-bear' | 'CHOCH-bull' | 'CHOCH-bear'
    """

    name     = "mstruct"
    category = "structure"
    source_cols = ("high", "low", "close", "ts")

    use_partial = False

    tag_name      = "mstruct"
    is_base_pivot = False

    TAG_DECODE: dict[float, str] = {
        0.0: "BOS-bull",
        1.0: "BOS-bear",
        2.0: "CHOCH-bull",
        3.0: "CHOCH-bear",
    }

    def pivot_col_names(self, symbol: str) -> tuple[str, ...]:
        base = self.col_name(symbol)
        return (f"{base}_b8",)

    def __init__(
        self,
        swing: int = 2,
        break_on: str = "close",
        *,
        show_lines: bool = True,
        show_labels: bool = True,
        bull_color: str = "#0ECB81",
        bear_color: str = "#F6465D",
    ):
        if break_on not in ("close", "wick"):
            from tradetropy.exceptions import ConfigError
            raise ConfigError(
                f"MarketStructure(break_on={break_on!r}): expected 'close' or 'wick'."
            )

        self.swing    = swing
        self.break_on = break_on
        self.length   = swing

        self.show_lines  = show_lines
        self.show_labels = show_labels
        self._bull_color = bull_color
        self._bear_color = bear_color

        # Geometry captured by calculate() and replayed by draw(). One entry per
        # detected event: (ts_origin, ts_break, level, tag).
        self._events: list[tuple[float, float, float, str]] = []

        self.output_names    = ["bos_bull", "bos_bear", "choch_bull", "choch_bear"]
        self.ts_band_indices = [4, 5, 6, 7, 8]
        self.ts_output_names = [
            "ts_bos_bull", "ts_bos_bear", "ts_choch_bull", "ts_choch_bear", "_tag",
        ]

        self.plot_config = IndicatorPlotConfig(
            overlay  = True,
            renderer = "scatter",
            color = [bull_color, bear_color, bull_color, bear_color],
            marker = [
                "triangle",           # BOS bull
                "inverted_triangle",  # BOS bear
                "diamond",            # CHoCH bull
                "diamond",            # CHoCH bear
            ],
            marker_size  = 9,
            marker_alpha = 0.9,
            exclude_from_autoscale = True,
            name = "MarketStructure",
        )

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 1

    # ──────────────────────────────────────────────────────────────────────
    # READ HELPERS  (offset-safe access from on_data)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def event(proxy, offset: int = -2) -> str | None:
        """
        Return the structure tag on the last CLOSED bar, or None.

        Since this indicator ignores partial candles, the live window's last
        slot is the forming bar and the event sits one slot earlier. This
        helper hides that offset so a strategy never has to hardcode ``[-2]``
        (and never silently reads the always-NaN ``[-1]``).

        Args:
            proxy: The indicator proxy returned by ``add_indicator``.
            offset (int): Bar to inspect; -2 (default) is the last closed bar.

        Returns:
            str | None: 'BOS-bull' | 'BOS-bear' | 'CHOCH-bull' | 'CHOCH-bear',
                or None when the bar carries no structure event.

        Example:
            >>> tag = MarketStructure.event(self.ms)
            >>> if tag == 'CHOCH-bull':
            ...     ...
        """
        for band, tag in (
            ("bos_bull", "BOS-bull"),
            ("bos_bear", "BOS-bear"),
            ("choch_bull", "CHOCH-bull"),
            ("choch_bear", "CHOCH-bear"),
        ):
            series = getattr(proxy, band, None)
            if series is None or len(series) < abs(offset):
                continue
            if not np.isnan(series[offset]):
                return tag
        return None

    @staticmethod
    def bias(proxy, lookback: int = 500) -> str | None:
        """
        Resolve the prevailing directional bias from the most recent event.

        Walks back over closed bars until it finds a structure event; a bullish
        event (BOS-bull / CHOCH-bull) means the structure is bullish.

        Args:
            proxy: The indicator proxy returned by ``add_indicator``.
            lookback (int): How many closed bars to scan back at most.

        Returns:
            str | None: 'bull' | 'bear', or None when no event is in range
                (structure not established yet — do not trade).
        """
        bands = (
            ("bos_bull", "bull"), ("bos_bear", "bear"),
            ("choch_bull", "bull"), ("choch_bear", "bear"),
        )
        probe = getattr(proxy, "bos_bull", None)
        if probe is None:
            return None
        n = len(probe)
        for off in range(2, min(lookback, n) + 1):
            for band, side in bands:
                series = getattr(proxy, band, None)
                if series is None or len(series) < off:
                    continue
                if not np.isnan(series[-off]):
                    return side
        return None

    def display_name(self) -> str:
        return f"MarketStructure({self.swing})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"mstruct{self.swing}_{self.break_on}_{symbol}"

    # ──────────────────────────────────────────────────────────────────────
    # CALCULATE
    # ──────────────────────────────────────────────────────────────────────
    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        source : [N × 4] — high(0), low(1), close(2), ts_ms(3). Returns [9 × N].

        Values land on the BREAK bar. ``ts_*`` bands point at the real swing
        bar of the broken level so plotting can anchor the structure line.
        """
        n        = len(source)
        out      = np.full((8, n), np.nan, dtype=np.float64)
        tag_band = np.full(n, np.nan, dtype=np.float64)

        self._events = []

        if n < self.min_periods:
            return np.vstack([out, tag_band])

        high  = source[:, 0].astype(np.float64)
        low   = source[:, 1].astype(np.float64)
        close = source[:, 2].astype(np.float64)
        ts_ms = source[:, 3].astype(np.float64)

        # ConfirmedPivot consumes (high, low, ts): drop the close column.
        pivot_src = np.column_stack([high, low, ts_ms])
        pivots = extract_confirmed_pivots(pivot_src, self.swing, use_ts_fallback=False)

        if not pivots:
            return np.vstack([out, tag_band])

        # bar_idx (confirmation bar) -> pivots that become knowable there.
        by_bar: dict[int, list[tuple[str, float, float]]] = {}
        for bar_idx, ptype, price, ts_real in pivots:
            by_bar.setdefault(bar_idx, []).append((ptype, price, ts_real))

        # Active (unbroken) levels. known_from is the confirmation bar: the
        # level is only breakable on strictly later bars.
        act_h: tuple[float, float, int] | None = None   # (price, ts_real, known_from)
        act_l: tuple[float, float, int] | None = None
        bias: str | None = None

        # Upper break uses close or high; lower break uses close or low.
        up_src   = close if self.break_on == "close" else high
        down_src = close if self.break_on == "close" else low

        for i in range(n):
            # 1) Register pivots that became knowable on this bar. Done first so
            #    a level is never evaluated before it exists, and never on the
            #    same bar it was confirmed (guarded by known_from < i below).
            for ptype, price, ts_real in by_bar.get(i, ()):
                if ptype == 'H':
                    act_h = (price, ts_real, i)
                else:
                    act_l = (price, ts_real, i)

            # 2) Evaluate breaks of already-knowable levels.
            broke_up = (
                act_h is not None
                and act_h[2] < i
                and up_src[i] > act_h[0]
            )
            broke_down = (
                act_l is not None
                and act_l[2] < i
                and down_src[i] < act_l[0]
            )

            # A bar that pierces both sides is resolved in favour of the
            # prevailing bias (continuation wins over reversal) so a single
            # wide bar cannot manufacture a phantom CHoCH.
            if broke_up and broke_down:
                if bias == 'bear':
                    broke_up = False
                else:
                    broke_down = False

            if broke_up:
                level, ts_origin, _ = act_h
                tag = "BOS-bull" if bias in (None, 'bull') else "CHOCH-bull"
                out[_MS_PRICE_IDX[tag], i] = level
                out[_MS_TS_IDX[tag],    i] = ts_origin
                tag_band[i] = _MS_TAG_CODE[tag]
                self._events.append((ts_origin, float(ts_ms[i]), level, tag))
                bias  = 'bull'
                act_h = None

            elif broke_down:
                level, ts_origin, _ = act_l
                tag = "BOS-bear" if bias in (None, 'bear') else "CHOCH-bear"
                out[_MS_PRICE_IDX[tag], i] = level
                out[_MS_TS_IDX[tag],    i] = ts_origin
                tag_band[i] = _MS_TAG_CODE[tag]
                self._events.append((ts_origin, float(ts_ms[i]), level, tag))
                bias  = 'bear'
                act_l = None

        return np.vstack([out, tag_band])

    # ──────────────────────────────────────────────────────────────────────
    # DRAW
    # ──────────────────────────────────────────────────────────────────────
    def draw(self, cfg: IndicatorPlotConfig | None = None,
             *, interval_ms: int | None = None) -> list:
        """
        Structure lines from the broken swing bar to the break bar, plus a
        BOS / CHoCH text label.

        The scatter markers (from ``plot_config``) already sit on the break
        bar; these primitives add the horizontal run back to the level's
        origin, which is what makes the break readable on the chart.
        """
        if not self._events or not (self.show_lines or self.show_labels):
            return []

        prims: list = []
        bos_idx, choch_idx = [], []
        x0, x1, y, colors = [], [], [], []
        lab_x, lab_y, lab_text, lab_color = [], [], [], []

        for k, (ts_origin, ts_break, level, tag) in enumerate(self._events):
            if np.isnan(ts_origin) or np.isnan(ts_break):
                continue
            color = self._bull_color if tag.endswith("bull") else self._bear_color

            x0.append(int(ts_origin))
            x1.append(int(ts_break))
            y.append(level)
            colors.append(color)

            lab_x.append(int(ts_break))
            lab_y.append(level)
            lab_text.append("BOS" if tag.startswith("BOS") else "CHoCH")
            lab_color.append(color)

            (bos_idx if tag.startswith("BOS") else choch_idx).append(len(x0) - 1)

        if self.show_lines and x0:
            # CHoCH lines are dashed to separate a reversal from a continuation
            # at a glance; Segments carries one dash for the whole glyph, so
            # they are emitted as two primitives.
            for idxs, dash, alpha, width in (
                (bos_idx,   "solid",  0.75, 1.4),
                (choch_idx, "dashed", 0.85, 1.6),
            ):
                if not idxs:
                    continue
                prims.append(Segments(
                    x0=[x0[k] for k in idxs],
                    y0=[y[k] for k in idxs],
                    x1=[x1[k] for k in idxs],
                    y1=[y[k] for k in idxs],
                    color=[colors[k] for k in idxs],
                    alpha=alpha, width=width, dash=dash,
                ))

        if self.show_labels and lab_x:
            prims.append(Labels(
                x=lab_x, y=lab_y, text=lab_text,
                color=lab_color, font_size="9pt",
            ))

        return prims

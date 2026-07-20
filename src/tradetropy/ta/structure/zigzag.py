import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.structure.pivots import ConfirmedPivot, _collapse_pivots_to_zigzag, PivotDetector


# ══════════════════════════════════════════════════════════════════════════════
# ZIGZAG  (multi-source, multi-band, causal — trading)
# ══════════════════════════════════════════════════════════════════════════════
class ZigZag(Indicator):
    """
    Single-band ZigZag built on top of any pivot detector.

    Delegates pivot detection to the received `detector` (default:
    ConfirmedPivot) and collapses its [K × N] output into a single [N]
    series alternating H→L→H→L via _collapse_pivots_to_zigzag().

    This allows changing the detection algorithm without touching ZigZag:

        ZigZag(swing=3)                           # uses ConfirmedPivot by default
        ZigZag(swing=3, detector=MyCustomPivot)   # class with __init__(swing=n)
        ZigZag(detector=ConfirmedPivot(swing=5))  # pre-built instance

    The detector must return [K × N] with:
      row 0 → pivot highs (NaN where no pivot)
      row 1 → pivot lows  (NaN where no pivot)
      rows 2+ → internal auxiliary (ignored)

    Parameters
    ──────────
    swing    : int — passed to detector if `detector` is a class.
               Ignored if `detector` is already an instance.
    detector : type | Indicator | None
               · None  → uses ConfirmedPivot(swing=swing).
               · class → instantiates detector(swing=swing).
               · instance → uses directly; length is taken from the detector.

    default source expected (ConfirmedPivot):
        [N × 3] — high(0), low(1), ts_ms(2)

    Usage
    ───
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='1m')
                self.zz  = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    ZigZag(swing=3),
                    plot=True, overlay=True,
                    color="#F59E0B",
                    line_width=2.0,
                )

            def on_data(self):
                # Last confirmed pivot (NaN if current bar is not a pivot)
                p = self.zz[-1]
                # Search for the last pivot backwards:
                for k in range(1, len(self.zz) + 1):
                    if not np.isnan(self.zz[-k]):
                        last = float(self.zz[-k])
                        break

        # With alternative detector (PivotHighLow — lookahead, visual only):
        self.zz = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref],
            ZigZag(swing=3, detector=PivotHighLow),
        )

        # With pre-configured instance:
        self.zz = self.add_indicator(
            [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
            ZigZag(detector=ConfirmedPivot(swing=5)),
        )
    """

    name            = "zigzag"
    category        = "structure"
    source_cols     = ("high", "low", "ts")
    output_names    = ["zigzag"]
    ts_band_indices = [1]
    ts_output_names = ["ts_real"]
    use_partial = False   # needs full history: delegates to ConfirmedPivot

    def __init__(
        self,
        swing: int = 2,
        detector: "type | Indicator | None" = None,
    ):
        self.swing = swing

        if detector is None:
            self._detector: Indicator = ConfirmedPivot(swing=swing)
        elif isinstance(detector, type):
            import inspect
            sig = inspect.signature(detector.__init__)
            params = list(sig.parameters.keys())
            if "swing" in params:
                self._detector = detector(swing=swing)
            elif "n" in params:
                self._detector = detector(n=swing)
            else:
                self._detector = detector(swing)
        else:
            self._detector = detector

        self.length = getattr(self._detector, "length", swing)
        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            color="#E6A729",
            line_width=2.5,
            line_dash="solid",
        )

    @property
    def min_periods(self) -> int:
        return getattr(self._detector, "min_periods", self.swing * 2 + 1)

    def display_name(self) -> str:
        det_name = type(self._detector).__name__
        return f"ZigZag({det_name})"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        source : [N × K] — whatever the detector expects.
                 With ConfirmedPivot (default): high(0), low(1), ts_ms(2).
                 With PivotHighLow: high(0), low(1).

        returns: [2 × N]
          row 0 : pivot price on the confirmation bar, NaN elsewhere.
          row 1 : timestamp (ms) of the REAL pivot bar, NaN elsewhere.
        """
        raw = self._detector.calculate(source)
        ph = raw[0]
        pl = raw[1]

        det_ts_bands = getattr(self._detector, "ts_band_indices", [])
        ph_ts = raw[det_ts_bands[0]] if len(det_ts_bands) >= 1 else None
        pl_ts = raw[det_ts_bands[1]] if len(det_ts_bands) >= 2 else None

        prices, ts_real = _collapse_pivots_to_zigzag(ph, pl, ph_ts, pl_ts)

        if ts_real is None:
            ts_real = np.full(len(prices), np.nan, dtype=np.float64)

        return np.vstack([prices, ts_real])
        

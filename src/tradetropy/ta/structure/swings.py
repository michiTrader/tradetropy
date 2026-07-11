import numpy as np

from tradetropy.ta.base import Indicator, IndicatorPlotConfig
from tradetropy.ta.structure.pivots import ConfirmedPivot


# ══════════════════════════════════════════════════════════════════════════════
# SwingHL  (scatter + label + extension lines)
# ══════════════════════════════════════════════════════════════════════════════
class SwingHL(Indicator):
    """
    Swing High / Low confirmed with price labels and optional lines
    that extend until broken.

    Uses ConfirmedPivot internally for causal detection (no lookahead).

    Expected source: [N × 3] — high(0), low(1), ts_ms(2)

    Outputs — [4 × N]:
        row 0 : swing_high  — confirmed SH price (NaN where none)
        row 1 : swing_low   — confirmed SL price (NaN where none)
        row 2 : _sh_ts      — real SH timestamp (internal — plotting)
        row 3 : _sl_ts      — real SL timestamp (internal — plotting)

    Parameters
    ──────────
    swing       : int   — ConfirmedPivot window (default 2)
    show_labels : bool  — True → price labels on each swing (default True)
    show_lines  : bool  — True → horizontal lines from swing until broken
                          (default True)
    label_offset: float — price offset for the label as % of the price
                          (default 0.05 = 0.05%)
    bull_color  : str   — swing lows color (default green)
    bear_color  : str   — swing highs color (default red)

    Usage
    ───
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.swings = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    SwingHL(swing=3, show_labels=True, show_lines=True),
                    plot=True,
                )

            def on_data(self):
                # Last confirmed swing high
                for k in range(1, 50):
                    if not np.isnan(self.swings.swing_high[-k]):
                        last_sh = float(self.swings.swing_high[-k])
                        break
    """

    name     = "swinghl"
    category = "annotation"

    output_names    = ["swing_high", "swing_low"]
    ts_band_indices = [2, 3]
    ts_output_names = ["ts_high", "ts_low"]

    def __init__(
        self,
        swing: int = 2,
        show_labels: bool = True,
        show_lines: bool = True,
        label_offset: float = 0.05,
        bull_color: str = "#0ECB81",
        bear_color: str = "#F6465D",
    ):
        self.swing        = swing
        self.show_labels  = show_labels
        self.show_lines   = show_lines
        self.label_offset = label_offset / 100.0
        self._bull_color  = bull_color
        self._bear_color  = bear_color
        self.length       = swing

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="scatter",
            color=[bear_color, bull_color],
            marker=["triangle", "inverted_triangle"],
            marker_size=8,
            marker_alpha=0.85,
            name="SwingHL",
        )

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 1

    def display_name(self) -> str:
        return f"SwingHL({self.swing})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"swinghl{self.swing}_{symbol}"

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n  = len(source)
        out = np.full((4, n), np.nan, dtype=np.float64)

        detector = ConfirmedPivot(swing=self.swing)
        raw = detector.calculate(source)

        ph    = raw[0]
        pl    = raw[1]
        ph_ts = raw[2]
        pl_ts = raw[3]

        out[0] = ph
        out[1] = pl
        out[2] = ph_ts
        out[3] = pl_ts

        ts_ms = source[:, 2].astype(np.float64)
        if n >= 2:
            diffs = np.diff(ts_ms)
            diffs = diffs[diffs > 0]
            interval_ms = int(np.median(diffs)) if len(diffs) > 0 else 60_000
        else:
            interval_ms = 60_000

        lines: list[dict] = []
        labels: list[dict] = []

        high = source[:, 0].astype(np.float64)
        low  = source[:, 1].astype(np.float64)

        for i in range(n):
            if not np.isnan(ph[i]):
                price = float(ph[i])
                ts_start = float(ph_ts[i]) if not np.isnan(ph_ts[i]) else float(ts_ms[i])
                ts_end = float(ts_ms[-1]) + interval_ms
                if self.show_lines:
                    for j in range(i + 1, n):
                        if high[j] > price:
                            ts_end = float(ts_ms[j])
                            break
                    lines.append({
                        "ts_start": ts_start, "ts_end": ts_end,
                        "price": price, "color": self._bear_color,
                    })
                if self.show_labels:
                    labels.append({
                        "ts": ts_start, "price": price * (1 + self.label_offset),
                        "text": f"H: {price:,.2f}", "color": self._bear_color,
                    })

            if not np.isnan(pl[i]):
                price = float(pl[i])
                ts_start = float(pl_ts[i]) if not np.isnan(pl_ts[i]) else float(ts_ms[i])
                ts_end = float(ts_ms[-1]) + interval_ms
                if self.show_lines:
                    for j in range(i + 1, n):
                        if low[j] < price:
                            ts_end = float(ts_ms[j])
                            break
                    lines.append({
                        "ts_start": ts_start, "ts_end": ts_end,
                        "price": price, "color": self._bull_color,
                    })
                if self.show_labels:
                    labels.append({
                        "ts": ts_start, "price": price * (1 - self.label_offset),
                        "text": f"L: {price:,.2f}", "color": self._bull_color,
                    })

        self._swing_lines  = lines
        self._swing_labels = labels

        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """
        Emit the swing extension lines (HLines) and price labels (Labels).

        The swing high/low markers themselves keep drawing as a scatter series
        from plot_config; only the horizontal extension lines and their price
        labels are geometry emitted here.
        """
        from tradetropy.ta.draw import HLines, Labels

        lines = getattr(self, "_swing_lines", [])
        labels = getattr(self, "_swing_labels", [])
        prims: list = []
        if lines:
            prims.append(HLines(
                x0=[int(l["ts_start"]) for l in lines],
                x1=[int(l["ts_end"]) for l in lines],
                y=[float(l["price"]) for l in lines],
                color=[l["color"] for l in lines],
                alpha=0.5, width=1.0, dash="dashed",
            ))
        if labels:
            prims.append(Labels(
                x=[int(l["ts"]) for l in labels],
                y=[float(l["price"]) for l in labels],
                text=[l["text"] for l in labels],
                color=[l["color"] for l in labels],
                font_size="8pt", x_offset=4, y_offset=2,
                text_align="left", text_baseline="middle",
            ))
        return prims


# ══════════════════════════════════════════════════════════════════════════════
# EqualHL  (horizontal lines — Equal Highs/Lows)
# ══════════════════════════════════════════════════════════════════════════════
class EqualHL(Indicator):
    """
    Equal Highs / Equal Lows (EQH/EQL) — liquidity zones where two or more
    swings touch the same price level within a tolerance.

    Detects pairs of swing highs and swing lows whose price differs by less
    than `pct_tolerance` percentage. Draws horizontal lines between the two
    points with an "EQH" or "EQL" label.

    Expected source: [N × 3] — high(0), low(1), ts_ms(2)

    OUTPUTS — [2 × N]:
        row 0 : level  — price of the most recent active EQH/EQL level
                         (NaN if no active level on that bar)
        row 1 : type   — level type: +1.0 = EQH, -1.0 = EQL, NaN if none

    Access in on_data():
        level = self.eqhl.level[-1]
        kind  = self.eqhl.type[-1]

        if kind == 1.0:    # active EQH
            print(f"EQH at {level:.2f}")
        if kind == -1.0:   # active EQL
            print(f"EQL at {level:.2f}")

        # Price near a liquidity level (±0.1%):
        close = self.btc.close[-1]
        if not np.isnan(level) and abs(close - level) / level < 0.001:
            pass  # price touching level → possible liquidity sweep

    Parameters
    ──────────
    swing         : int   — ConfirmedPivot window (default 2)
    pct_tolerance : float — max % difference between two swings to consider
                            them equal (default 0.1 = 0.1%)
    eqh_color     : str   — Equal Highs color (default red)
    eql_color     : str   — Equal Lows  color (default green)
    show_labels   : bool  — True → "EQH"/"EQL" labels (default True)
    invalidate    : bool  — True → level is invalidated when price breaks
                            through it (default True). Produces NaN in
                            the series from that bar onward.

    Usage
    ───
        class MyStrategy(Strategy):
            def init(self):
                self.btc  = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.eqhl = self.add_indicator(
                    [self.btc.high_ref, self.btc.low_ref, self.btc.ts_ref],
                    EqualHL(swing=3, pct_tolerance=0.05),
                    plot=True,
                )

            def on_data(self):
                level = self.eqhl.level[-1]
                kind  = self.eqhl.type[-1]

                if np.isnan(level):
                    return

                close = self.btc.close[-1]
                # Price touching EQH → possible bullish liquidity sweep
                if kind == 1.0 and close >= level * 0.999:
                    self.log.signal("Price touching EQH at %.2f", level)
    """

    name     = "eqhl"
    category = "annotation"

    output_names    = ["level", "type"]
    ts_band_indices = []

    def __init__(
        self,
        swing: int = 2,
        pct_tolerance: float = 0.1,
        eqh_color: str = "#F6465D",
        eql_color: str = "#0ECB81",
        show_labels: bool = True,
        invalidate: bool = True,
    ):
        self.swing         = swing
        self.pct_tolerance = pct_tolerance / 100.0
        self._eqh_color    = eqh_color
        self._eql_color    = eql_color
        self.show_labels   = show_labels
        self.invalidate    = invalidate
        self.length        = swing

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="segment",
            name="EqHL",
        )

    @property
    def min_periods(self) -> int:
        return self.swing * 2 + 1

    def display_name(self) -> str:
        return f"EqHL({self.swing})"

    def col_name(self, symbol: str, col_source: str = "") -> str:
        return f"eqhl{self.swing}_{symbol}"

    def _find_equal_pairs(
        self, pivots: list[tuple[float, float]]
    ) -> list[tuple[float, float, float, float]]:
        pairs = []
        for a in range(len(pivots)):
            for b in range(a + 1, len(pivots)):
                ts1, p1 = pivots[a]
                ts2, p2 = pivots[b]
                if p1 == 0:
                    continue
                if abs(p1 - p2) / p1 <= self.pct_tolerance:
                    pairs.append((ts1, ts2, p1, p2))
        return pairs

    def calculate(self, source: np.ndarray) -> np.ndarray:
        n = len(source)
        out = np.full((2, n), np.nan, dtype=np.float64)

        if n < self.swing * 2 + 1:
            self._segments = []
            return out

        high  = source[:, 0].astype(np.float64)
        low   = source[:, 1].astype(np.float64)
        ts_ms = source[:, 2].astype(np.float64)

        detector = ConfirmedPivot(swing=self.swing)
        raw      = detector.calculate(source)

        ph    = raw[0]
        pl    = raw[1]
        ph_ts = raw[2]
        pl_ts = raw[3]

        highs_list: list[tuple[float, float, int]] = []
        lows_list:  list[tuple[float, float, int]] = []

        for i in range(n):
            if not np.isnan(ph[i]):
                t = float(ph_ts[i]) if not np.isnan(ph_ts[i]) else float(ts_ms[i])
                highs_list.append((t, float(ph[i]), i))
            if not np.isnan(pl[i]):
                t = float(pl_ts[i]) if not np.isnan(pl_ts[i]) else float(ts_ms[i])
                lows_list.append((t, float(pl[i]), i))

        def find_pairs_with_idx(pivots):
            pairs = []
            for a in range(len(pivots)):
                for b in range(a + 1, len(pivots)):
                    ts1, p1, idx1 = pivots[a]
                    ts2, p2, idx2 = pivots[b]
                    if p1 == 0:
                        continue
                    if abs(p1 - p2) / p1 <= self.pct_tolerance:
                        mid = (p1 + p2) / 2.0
                        pairs.append((ts1, ts2, p1, p2, mid, idx2))
            return pairs

        eqh_pairs = find_pairs_with_idx(highs_list)
        eql_pairs = find_pairs_with_idx(lows_list)

        levels: list[tuple[int, float, float]] = []
        for ts1, ts2, p1, p2, mid, start_bar in eqh_pairs:
            levels.append((start_bar, mid, +1.0))
        for ts1, ts2, p1, p2, mid, start_bar in eql_pairs:
            levels.append((start_bar, mid, -1.0))

        levels.sort(key=lambda x: x[0])

        active_level_price: float = np.nan
        active_level_type:   float = np.nan
        level_ptr: int = 0

        for i in range(n):
            while level_ptr < len(levels) and levels[level_ptr][0] <= i:
                _, price, kind = levels[level_ptr]
                active_level_price = price
                active_level_type   = kind
                level_ptr += 1

            if self.invalidate and not np.isnan(active_level_price):
                if active_level_type == 1.0 and high[i] > active_level_price:
                    active_level_price = np.nan
                    active_level_type   = np.nan
                elif active_level_type == -1.0 and low[i] < active_level_price:
                    active_level_price = np.nan
                    active_level_type   = np.nan

            out[0, i] = active_level_price
            out[1, i] = active_level_type

        segments: list[dict] = []

        for ts1, ts2, p1, p2, mid, _ in eqh_pairs:
            segments.append({
                "ts1": ts1, "ts2": ts2,
                "price": mid,
                "color": self._eqh_color,
                "label": "EQH" if self.show_labels else "",
            })

        for ts1, ts2, p1, p2, mid, _ in eql_pairs:
            segments.append({
                "ts1": ts1, "ts2": ts2,
                "price": mid,
                "color": self._eql_color,
                "label": "EQL" if self.show_labels else "",
            })

        self._segments = segments
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """Emit Equal High/Low levels as dotted HLines plus EQH/EQL labels."""
        from tradetropy.ta.draw import HLines, Labels

        segs = getattr(self, "_segments", [])
        if not segs:
            return []
        prims = [HLines(
            x0=[int(s["ts1"]) for s in segs],
            x1=[int(s["ts2"]) for s in segs],
            y=[float(s["price"]) for s in segs],
            color=[s["color"] for s in segs],
            alpha=0.8, width=1.5, dash="dotted",
        )]
        labeled = [s for s in segs if s.get("label")]
        if labeled:
            prims.append(Labels(
                x=[int(s["ts2"]) for s in labeled],
                y=[float(s["price"]) for s in labeled],
                text=[s["label"] for s in labeled],
                color=[s["color"] for s in labeled],
                font_size="8pt", x_offset=4, y_offset=2,
                text_align="left", text_baseline="middle",
            ))
        return prims
        

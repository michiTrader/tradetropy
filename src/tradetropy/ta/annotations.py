import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from tradetropy.ta.base import Indicator, IndicatorPlotConfig


def _mitigated_rects(bull_items, bear_items, show_mitigated, cfg, interval_ms,
                     extend_mult):
    """
    Build one Rects primitive for mitigation-based zone annotations (FVG / OB).

    Mirrors the geometry of the former render/_annotations.py::_render_mitigated_rects
    but as a declarative primitive: bull zones use color index 0, bear zones use
    index 2 of ``cfg.color``; an open zone extends ``extend_mult`` bars to the
    right, a mitigated one ends at its mitigation timestamp and is drawn fainter.

    Args:
        bull_items / bear_items (list): [bar_idx, top, bot, ts_open, ts_close]
            tuples; ts_close is NaN while the zone is open.
        show_mitigated (bool): include already-mitigated zones (fainter).
        cfg (IndicatorPlotConfig): effective plot config (colors / alphas).
        interval_ms (int | None): OHLC interval for the right extension.
        extend_mult (int): bars an open zone extends to the right.

    Returns:
        list[Rects]: a single-element list, or [] when nothing to draw.
    """
    from tradetropy.ta.draw import Rects

    colors = cfg.color if cfg is not None else None

    def _c(idx):
        if isinstance(colors, list):
            return colors[idx] if idx < len(colors) else colors[-1]
        return colors or "#888888"

    bull_color = _c(0)
    bear_color = _c(2) if isinstance(colors, list) and len(colors) > 2 else _c(0)
    fill_a = getattr(cfg, "rect_fill_alpha", 0.12) if cfg else 0.12
    line_a = getattr(cfg, "rect_line_alpha", 0.55) if cfg else 0.55
    line_w = getattr(cfg, "rect_line_width", 1.0) if cfg else 1.0
    mit_factor = 0.35
    extend_ms = int((interval_ms or 60_000) * extend_mult)

    x0, x1, y0, y1, fills, lines, fas, las = [], [], [], [], [], [], [], []
    for items, color in ((bull_items, bull_color), (bear_items, bear_color)):
        for item in items:
            _, top, bot, ts_open, ts_close = item
            mitigated = not np.isnan(ts_close)
            if mitigated and not show_mitigated:
                continue
            left = int(ts_open)
            right = int(ts_close) if mitigated else int(ts_open) + extend_ms
            x0.append(left)
            x1.append(right)
            y0.append(float(bot))
            y1.append(float(top))
            fills.append(color)
            lines.append(color)
            fas.append(fill_a * (mit_factor if mitigated else 1.0))
            las.append(line_a * (mit_factor if mitigated else 1.0))

    if not x0:
        return []
    return [Rects(
        x0=x0, x1=x1, y0=y0, y1=y1,
        fill_color=fills, line_color=lines,
        fill_alpha=fas, line_alpha=las, line_width=line_w,
    )]


# =====
# FairValueGap (rect annotation - multi-band)
# =====
_FVG_DEFAULT_BULL = "#0ECB81"
_FVG_DEFAULT_BEAR = "#F6465D"

class FairValueGap(Indicator):
    """
    Fair Value Gap (FVG) / Imbalance.

    Detects price gaps between the wick of candle i-1 and the wick of i+1:
        - Bullish FVG : low[i+1] > high[i-1]  -> price zone untouched to the upside
        - Bearish FVG : high[i+1] < low[i-1]  -> price zone untouched to the downside

    The FVG is marked at bar i (the "body" candle of the gap) and remains
    active until price fills it (mitigation). Once mitigated, the rectangle
    closes at that bar.

    Expected source: [N x 6] - ts(0), open(1), high(2), low(3), close(4), vol(5)

    Outputs - [8 x N] (4 price bands + 4 timestamp bands):

    Price bands (output_names, accessible by user):
        row 0 : bull_top    - top of bullish FVG  (NaN if no active FVG)
        row 1 : bull_bot    - bottom of bullish FVG
        row 2 : bear_top    - top of bearish FVG
        row 3 : bear_bot    - bottom of bearish FVG

    Timestamp bands (ts_band_indices, internal - plotting uses them to
    build rectangles with real coordinates on the X axis):
        row 4 : bull_ts_open  - open ts of bullish FVG  (ms)
        row 5 : bull_ts_close - close/mitigation ts of FVG (ms, NaN = still open)
        row 6 : bear_ts_open  - open ts of bearish FVG
        row 7 : bear_ts_close - close/mitigation ts of bearish FVG

    Access in on_data():
        self.fvg.bull_top[-1]  -> top of last active bullish FVG (or NaN)
        self.fvg.bull_bot[-1]  -> bottom of last active bullish FVG
        self.fvg.bear_top[-1]  -> top of last active bearish FVG
        self.fvg.bear_bot[-1]  -> bottom of last active bearish FVG

        # Detect if there is an active bullish FVG:
        if not np.isnan(self.fvg.bull_top[-1]):
            zone_top = self.fvg.bull_top[-1]
            zone_bot = self.fvg.bull_bot[-1]

    Args:
        mitigate : bool   - True (default) -> FVG disappears when price
                            enters the zone (closes the rectangle).
                            False -> FVGs persist indefinitely.
        min_gap  : float  - minimum gap size in points to be considered
                            (filters insignificant gaps). Default 0.0.
        show_mitigated : bool - True -> also draws already-mitigated FVGs
                                (more faint). Default False.

    Note on lookahead:
        Classic FVG detection requires seeing bar i+1 to confirm the gap.
        This introduces a 1-bar lookahead - standard in the ICT community
        and acceptable for backtesting, but should be considered for live
        trading strategies.

    Usage:
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.fvg = self.add_indicator(
                    [self.btc.ts_ref, self.btc.open_ref, self.btc.high_ref,
                     self.btc.low_ref, self.btc.close_ref, self.btc.volume_ref],
                    FairValueGap(mitigate=True),
                    plot=True,
                )

            def on_data(self):
                if not np.isnan(self.fvg.bull_bot[-1]):
                    # Active bullish FVG - price above bottom
                    if self.btc.close[-1] <= self.fvg.bull_top[-1]:
                        pass  # price inside FVG -> possible long entry
    """

    name     = "fvg"
    category = "annotation"

    # The 4 price bands are accessible to the user
    output_names    = ["bull_top", "bull_bot", "bear_top", "bear_bot"]
    # The 4 timestamp bands are internal (plotting interprets them for rects)
    ts_band_indices = [4, 5, 6, 7]
    ts_output_names = ["ts_bull_open", "ts_bull_close", "ts_bear_open", "ts_bear_close"]

    def __init__(
        self,
        mitigate: bool = True,
        min_gap: float = 0.0,
        show_mitigated: bool = False,
        bull_color: str = "#0ECB81",
        bear_color: str = "#F6465D",
    ):
        self.mitigate       = mitigate
        self.min_gap        = float(min_gap)
        self.show_mitigated = show_mitigated
        self.length         = 3  # minimum detection window

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="rect",
            color=[bull_color, bull_color, bear_color, bear_color],
            rect_fill_alpha=0.12,
            rect_line_alpha=0.55,
            rect_line_width=1.0,
            name="FVG",
        )

    @property
    def min_periods(self) -> int:
        return 3

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Args:
            source: [N x 6] - ts(0), open(1), high(2), low(3), close(4), vol(5)

        Returns:
            [8 x N]
            [0] bull_top, [1] bull_bot, [2] bear_top, [3] bear_bot,
            [4] bull_ts_open, [5] bull_ts_close, [6] bear_ts_open, [7] bear_ts_close
        """
        n = len(source)
        out = np.full((8, n), np.nan, dtype=np.float64)

        if n < 3:
            return out

        ts    = source[:, 0].astype(np.float64)
        high  = source[:, 2].astype(np.float64)
        low   = source[:, 3].astype(np.float64)
        close = source[:, 4].astype(np.float64)

        # Step 1: detect all FVGs with 1-bar lookahead
        # bull_fvgs / bear_fvgs : list of (bar_idx, top, bot, ts_open, ts_close | nan)
        bull_fvgs: list[list] = []
        bear_fvgs: list[list] = []

        for i in range(1, n - 1):
            bull_top = low[i + 1]
            bull_bot = high[i - 1]
            if bull_top > bull_bot and (bull_top - bull_bot) >= self.min_gap:
                bull_fvgs.append([i, bull_top, bull_bot, ts[i], np.nan])

            bear_top = low[i - 1]
            bear_bot = high[i + 1]
            if bear_top > bear_bot and (bear_top - bear_bot) >= self.min_gap:
                bear_fvgs.append([i, bear_top, bear_bot, ts[i], np.nan])

        # Step 2: mark mitigations
        if self.mitigate:
            for fvg in bull_fvgs:
                bar_i, top, bot, ts_open, _ = fvg
                for j in range(bar_i + 1, n):
                    if close[j] <= top:
                        fvg[4] = ts[j]
                        break

            for fvg in bear_fvgs:
                bar_i, top, bot, ts_open, _ = fvg
                for j in range(bar_i + 1, n):
                    if close[j] >= bot:
                        fvg[4] = ts[j]
                        break

        # Step 3: write to output array
        self._bull_fvgs = bull_fvgs
        self._bear_fvgs = bear_fvgs
        self._show_mitigated = self.show_mitigated

        bull_by_bar = {f[0]: f for f in bull_fvgs}
        bear_by_bar = {f[0]: f for f in bear_fvgs}

        active_bulls: list = []
        active_bears: list = []

        for i in range(n):
            if i in bull_by_bar:
                active_bulls.append(bull_by_bar[i])
            if i in bear_by_bar:
                active_bears.append(bear_by_bar[i])

            if self.mitigate:
                active_bulls = [
                    f for f in active_bulls
                    if np.isnan(f[4]) or ts[i] < f[4]
                ]
                active_bears = [
                    f for f in active_bears
                    if np.isnan(f[4]) or ts[i] < f[4]
                ]

            if active_bulls:
                last = active_bulls[-1]
                out[0, i] = last[1]  # bull_top
                out[1, i] = last[2]  # bull_bot
                out[4, i] = last[3]  # bull_ts_open
                out[5, i] = last[4]  # bull_ts_close (NaN if open)
            if active_bears:
                last = active_bears[-1]
                out[2, i] = last[1]  # bear_top
                out[3, i] = last[2]  # bear_bot
                out[6, i] = last[3]  # bear_ts_open
                out[7, i] = last[4]  # bear_ts_close (NaN if open)

        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """Emit the FVG zones as Rects primitives (open zones extend 3 bars)."""
        cfg = cfg or self.plot_config
        return _mitigated_rects(
            getattr(self, "_bull_fvgs", []),
            getattr(self, "_bear_fvgs", []),
            getattr(self, "_show_mitigated", False),
            cfg, interval_ms, extend_mult=3,
        )


# =====
# Shared session-window helpers
# =====
#
# Generic, module-level building blocks reused by every time-window annotation
# (MarketSessions, SessionLevels, KillZones). Kept pure NumPy / causal so any
# indicator built on a "named UTC hour window, possibly crossing midnight" can
# share the same parsing, masking and running-OHLC-per-occurrence logic instead
# of re-implementing it.
_SESSIONS_UTC = {
    "sydney":   (22, 0, 0, 0),
    "tokyo":    (0, 0, 9, 0),
    "london":   (8, 0, 17, 0),
    "new_york": (13, 0, 22, 0),
}
_SESSION_COLORS = {
    "sydney":   "#FFA500",
    "tokyo":    "#FF6B6B",
    "london":   "#4ECDC4",
    "new_york": "#45B7D1",
}


def _parse_session_windows(
    windows_input: list,
    predefined_utc: dict,
    predefined_colors: dict,
    color_overrides: "dict[str, str]",
    default_color: str = "#888888",
) -> list[dict]:
    """
    Normalize a user window spec list into dicts with a common shape.

    Shared by every UTC-hour-window annotation (MarketSessions, SessionLevels,
    KillZones): each accepts either a predefined string key (looked up in
    ``predefined_utc`` / ``predefined_colors``) or a custom dict with integer
    hours and optional minutes:
        {"name": "NY Open", "start": 14, "start_min": 30,
         "end": 16, "end_min": 0, "color": "#FF6B35"}

    Args:
        windows_input (list): Mix of predefined strings and/or custom dicts.
        predefined_utc (dict): name -> (h_start, m_start, h_end, m_end).
        predefined_colors (dict): name -> default color.
        color_overrides (dict): user overrides keyed by (predefined) name.
        default_color (str): fallback color for unrecognized/custom windows.

    Returns:
        list[dict]: normalized windows, each with keys
            name, start, start_min, end, end_min, color.
    """
    result = []
    for w in windows_input:
        if isinstance(w, str):
            key = w.lower()
            if key not in predefined_utc:
                import warnings
                warnings.warn(
                    f"Unknown window '{w}'. Options: {list(predefined_utc.keys())}"
                )
                continue
            h_start, m_start, h_end, m_end = predefined_utc[key]
            color = color_overrides.get(key, predefined_colors.get(key, default_color))
            result.append({
                "name": key, "start": h_start, "start_min": m_start,
                "end": h_end, "end_min": m_end, "color": color,
            })
        elif isinstance(w, dict):
            h_start = int(w.get("start", 0))
            m_start = int(w.get("start_min", 0))
            h_end   = int(w.get("end", 8))
            m_end   = int(w.get("end_min", 0))
            color   = w.get("color", default_color)
            name    = w.get("name",
                f"{h_start:02d}:{m_start:02d}_{h_end:02d}:{m_end:02d}utc")
            result.append({
                "name": name, "start": h_start, "start_min": m_start,
                "end": h_end, "end_min": m_end, "color": color,
            })
    return result


def _mask_in_window(
    ts_ms: np.ndarray,
    h_start: int,
    h_end: int,
    m_start: int = 0,
    m_end: int = 0,
) -> np.ndarray:
    """
    Boolean mask: True where the timestamp falls inside a daily UTC window.

    Supports windows that cross midnight (e.g. Sydney 22:00 -> 09:00 UTC): when
    ``end`` is not strictly after ``start`` the window is treated as wrapping
    through 00:00.

    Args:
        ts_ms (np.ndarray): Timestamps in ms (UTC).
        h_start / h_end (int): Window start/end hour (0-23).
        m_start / m_end (int): Window start/end minute (0-59).

    Returns:
        np.ndarray: Boolean mask, same shape as ts_ms.
    """
    minutes     = (ts_ms // 60_000) % 1440
    start_total = h_start * 60 + m_start
    end_total   = h_end   * 60 + m_end
    if start_total < end_total:
        return (minutes >= start_total) & (minutes < end_total)
    else:
        return (minutes >= start_total) | (minutes < end_total)


def _occurrence_ids(mask: np.ndarray) -> np.ndarray:
    """
    Assign an increasing occurrence id to each contiguous in-window block.

    Rows outside the window get id -1. Two ``True`` rows belong to the same
    occurrence only if every row between them is also ``True`` (i.e. no gap) -
    this is exactly what happens when the source has one row per contiguous
    bar and the window mask flips False for at least one bar between two daily
    passes. Purely causal: the id of a row never depends on future rows.

    Args:
        mask (np.ndarray): Boolean in-window mask (True = inside window).

    Returns:
        np.ndarray: int64 occurrence id per row (-1 outside any window).
    """
    n = len(mask)
    occ = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return occ
    current = -1
    prev = False
    for i in range(n):
        if mask[i]:
            if not prev:
                current += 1
            occ[i] = current
        prev = bool(mask[i])
    return occ


def _running_ohlc_by_occurrence(
    ts: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    occ: np.ndarray,
) -> dict:
    """
    Causal running OHLC per occurrence, plus the previous CLOSED occurrence's
    final OHLC projected forward until the next occurrence starts.

    For each row inside an occurrence (``occ[i] >= 0``):
        running_open  = open of the occurrence's first bar
        running_high  = cummax(high) within the occurrence, up to row i
        running_low   = cummin(low)  within the occurrence, up to row i

    For every row (inside or outside a window), ``prev_*`` holds the frozen
    OHLC of the last occurrence that has already CLOSED (a bar with a
    different, higher occurrence id, or the window ended) - NaN before the
    first occurrence has closed. ``prev_close`` is the close of that
    occurrence's last bar.

    Args:
        ts, open_, high, low, close (np.ndarray): OHLC arrays, float64.
        occ (np.ndarray): Occurrence id per row (-1 outside any window), as
            returned by :func:`_occurrence_ids`.

    Returns:
        dict[str, np.ndarray]: keys
            'open', 'high', 'low'                       (running, NaN outside)
            'prev_open', 'prev_high', 'prev_low', 'prev_close'  (projected)
    """
    n = len(occ)
    r_open = np.full(n, np.nan, dtype=np.float64)
    r_high = np.full(n, np.nan, dtype=np.float64)
    r_low  = np.full(n, np.nan, dtype=np.float64)
    p_open  = np.full(n, np.nan, dtype=np.float64)
    p_high  = np.full(n, np.nan, dtype=np.float64)
    p_low   = np.full(n, np.nan, dtype=np.float64)
    p_close = np.full(n, np.nan, dtype=np.float64)

    last_closed: "dict | None" = None
    cur_occ = -1
    cur_open = np.nan
    cur_high = np.nan
    cur_low = np.nan
    cur_close = np.nan

    for i in range(n):
        o = occ[i]
        if o >= 0:
            if o != cur_occ:
                # A new occurrence starts: the previous one (if any) is now
                # closed and becomes the projected reference.
                if cur_occ >= 0:
                    last_closed = {
                        "open": cur_open, "high": cur_high,
                        "low": cur_low, "close": cur_close,
                    }
                cur_occ = o
                cur_open = open_[i]
                cur_high = high[i]
                cur_low = low[i]
            else:
                cur_high = max(cur_high, high[i])
                cur_low = min(cur_low, low[i])
            cur_close = close[i]

            r_open[i] = cur_open
            r_high[i] = cur_high
            r_low[i] = cur_low
        else:
            if cur_occ >= 0:
                # The occurrence we were tracking has just ended (no row
                # since belongs to it) - close it out.
                last_closed = {
                    "open": cur_open, "high": cur_high,
                    "low": cur_low, "close": cur_close,
                }
                cur_occ = -1

        if last_closed is not None:
            p_open[i]  = last_closed["open"]
            p_high[i]  = last_closed["high"]
            p_low[i]   = last_closed["low"]
            p_close[i] = last_closed["close"]

    return {
        "open": r_open, "high": r_high, "low": r_low,
        "prev_open": p_open, "prev_high": p_high,
        "prev_low": p_low, "prev_close": p_close,
    }


def _merge_contiguous_blocks(
    ts_ms: np.ndarray, mask: np.ndarray, interval_ms: int
) -> list[tuple[int, int]]:
    """
    Merge contiguous in-window bars into (ts_start, ts_end) blocks.

    Args:
        ts_ms (np.ndarray): Timestamps in ms.
        mask (np.ndarray): Boolean in-window mask.
        interval_ms (int): Bar interval, used both to close the last block
            and to decide the max gap tolerated within one block.

    Returns:
        list[tuple[int, int]]: (ts_start, ts_end) per merged block.
    """
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []

    bloques   = []
    blk_start = int(ts_ms[idxs[0]])
    blk_prev  = int(ts_ms[idxs[0]])
    gap_max   = max(interval_ms * 2, 3_600_000)

    for i in idxs[1:]:
        t = int(ts_ms[i])
        if t - blk_prev > gap_max:
            bloques.append((blk_start, blk_prev + interval_ms))
            blk_start = t
        blk_prev = t

    bloques.append((blk_start, blk_prev + interval_ms))
    return bloques


def _estimate_interval_ms(ts_ms: np.ndarray) -> int:
    """Median of positive consecutive diffs, defaulting to 1 minute."""
    if len(ts_ms) >= 2:
        diffs = np.diff(ts_ms)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            return int(np.median(diffs))
    return 60_000


# =====
# MarketSessions (span annotation - market session zones)
# =====
class MarketSessions(Indicator):
    """
    Market session zones. Draws background rectangles and exposes a binary
    series per session (1.0 = inside, 0.0 = outside) accessible in
    on_data() to filter trades by time.

    Expected source: [N x 1] - ts_ms(0) (timestamps in milliseconds UTC)

    Outputs - [K x N] where K = number of configured sessions:
        row k : 1.0 if bar k falls inside session k, 0.0 otherwise.

    output_names is built dynamically in __init__ from session names,
    so attribute access works:

        self.sess.london[-1]    -> 1.0 if current bar is in London session
        self.sess.new_york[-1]  -> 1.0 if in New York session

    Predefined sessions (strings): "sydney", "tokyo", "london", "new_york".
    Also accepts custom dicts with name and UTC schedule.

    Args:
        sessions : list[str | dict]
            Sessions to show. Predefined strings: "sydney", "tokyo",
            "london", "new_york". Also accepts custom dicts:
                {"name": "Kill Zone", "start": 2, "end": 4, "color": "#FF0000"}
            where start/end are UTC hours (int, 0-23).
        colors : dict[str, str] | None
            Color overrides by session name.
            None -> uses predefined _SESSION_COLORS.
        alpha : float
            Fill opacity of rectangles (default 0.08).
        show_labels : bool
            True -> draws session name (default True).

    Usage:
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='1m')
                self.sess = self.add_indicator(
                    [self.btc.ts_ref],
                    MarketSessions(sessions=["london", "new_york"]),
                    plot=True,
                )

            def on_data(self):
                in_london   = self.sess.london[-1] == 1.0
                in_new_york = self.sess.new_york[-1] == 1.0

                # Session filter: only trade during London/NY overlap
                if not (in_london and in_new_york):
                    return

        # With custom session:
        self.sess = self.add_indicator(
            [self.btc.ts_ref],
            MarketSessions(sessions=[
                "london",
                {"name": "ny_open", "start": 13, "end": 15, "color": "#FF6B35"},
            ]),
        )
        # Access: self.sess.ny_open[-1]
    """

    name     = "sessions"
    category = "annotation"
    # output_names built in __init__ - empty list as class placeholder
    output_names: list = []
    ts_band_indices = []

    def __init__(
        self,
        sessions: "list[str | dict] | None" = None,
        colors: "dict[str, str] | None" = None,
        alpha: float = 0.08,
        show_labels: bool = True,
    ):
        if sessions is None:
            sessions = ["london", "new_york"]
        self._sessions_input  = sessions
        self._color_overrides = colors or {}
        self.alpha            = alpha
        self.show_labels      = show_labels
        self.length           = 1

        # Build output_names dynamically from session names.
        # Custom session names are normalized (spaces -> _) to be valid
        # Python attributes.
        parsed = self._parse_sessions()
        self.output_names = [
            s["name"].replace(" ", "_").replace("-", "_").lower()
            for s in parsed
        ]

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="span",
            name="Sessions",
        )

    @property
    def n_outputs(self) -> int:
        # K binary series, one per session
        return max(len(self.output_names), 1)

    @property
    def min_periods(self) -> int:
        return 1

    def _parse_sessions(self) -> list[dict]:
        """
        Converts the input list to normalized dicts.

        Predefined sessions (str): "sydney", "tokyo", "london", "new_york".
        Custom sessions (dict): accept integer hours and optionally minutes:
            {"name": "NY Open", "start": 14, "start_min": 30,
             "end": 16, "end_min": 0, "color": "#FF6B35"}
        Fields start_min / end_min are optional and default to 0.
        """
        return _parse_session_windows(
            self._sessions_input, _SESSIONS_UTC, _SESSION_COLORS,
            self._color_overrides,
        )

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Args:
            source: [N x 1] - ts_ms(0)

        Returns:
            [K x N] - K binary series (1.0 = inside, 0.0 = outside),
                      one per configured session.

        Additionally saves self._zonas so that plotting builds the
        background rectangles via renderer="span".
        """
        n     = len(source)
        ts_ms = source[:, 0].astype(np.float64) if source.ndim == 2 else source.astype(np.float64)

        sessions = self._parse_sessions()
        K        = len(sessions)

        if K == 0:
            return np.zeros((1, n), dtype=np.float64)

        out = np.zeros((K, n), dtype=np.float64)
        interval_ms = _estimate_interval_ms(ts_ms)

        zonas: list[dict] = []

        for k, sess in enumerate(sessions):
            mask = _mask_in_window(
                ts_ms,
                sess["start"], sess["end"],
                sess.get("start_min", 0), sess.get("end_min", 0),
            )
            # Binary series for on_data()
            out[k, :] = mask.astype(np.float64)

            # Blocks for plotting
            bloques = _merge_contiguous_blocks(ts_ms, mask, interval_ms)
            for ts_start, ts_end in bloques:
                zonas.append({
                    "ts_start": ts_start,
                    "ts_end":   ts_end,
                    "color":    sess["color"],
                    "label":    sess["name"] if self.show_labels else "",
                })

        self._zonas = zonas
        self._alpha = self.alpha
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """
        Emit session zones as full-height Rects plus screen-anchored Labels.

        Each session block becomes a translucent rectangle spanning the whole
        price axis; the session name is drawn once at the block's left edge,
        anchored to the bottom of the screen (y_units='screen').
        """
        from tradetropy.ta.draw import Rects, Labels

        zonas = getattr(self, "_zonas", [])
        if not zonas:
            return []
        alpha = getattr(self, "_alpha", self.alpha)
        huge = 1e15
        n = len(zonas)
        colors = [z["color"] for z in zonas]

        prims = [Rects(
            x0=[int(z["ts_start"]) for z in zonas],
            x1=[int(z["ts_end"]) for z in zonas],
            y0=[-huge] * n, y1=[huge] * n,
            fill_color=colors, line_color=colors,
            fill_alpha=[alpha] * n, line_alpha=[alpha * 2] * n,
            line_width=1.0,
        )]

        labeled = [z for z in zonas if z.get("label")]
        if labeled:
            prims.append(Labels(
                x=[int(z["ts_start"]) for z in labeled],
                y=[0.0] * len(labeled),
                text=[f" {z['label']}" for z in labeled],
                color=[z["color"] for z in labeled],
                font_size="8pt", x_offset=2, y_offset=2,
                text_align="left", text_baseline="bottom", y_units="screen",
            ))
        return prims


# =====
# SessionLevels (session OHLC - running + previous-session projected levels)
# =====
class SessionLevels(Indicator):
    """
    OHLC price levels per market session: the running open/high/low of the
    CURRENT occurrence of each session, plus the open/high/low/close of the
    last CLOSED occurrence projected forward as reference levels for the
    current one. Complements MarketSessions (which only exposes the binary
    in/out-of-session series) with actual price information - covers the
    classic "Session High/Low", "Asian Range" and "Previous Session High/Low"
    concepts used in ICT/SMC and institutional intraday analysis.

    Expected source: [N x 5] - ts(0), open(1), high(2), low(3), close(4)
    (use ``SessionLevels.refs(ohlc_proxy)`` to build the columns).

    Outputs - [7K x N] where K = number of configured sessions. For each
    session ``s`` (in the order given to ``sessions``), 7 bands:

        {s}_open        - open of the CURRENT (still open) occurrence.
                          NaN outside the session.
        {s}_high        - running high of the current occurrence (cummax
                          from the occurrence's first bar). NaN outside.
        {s}_low         - running low of the current occurrence (cummin).
                          NaN outside.
        {s}_prev_open   - open of the last CLOSED occurrence, projected
                          forward until the next occurrence starts.
        {s}_prev_high   - high of the last CLOSED occurrence, projected.
        {s}_prev_low    - low of the last CLOSED occurrence, projected.
        {s}_prev_close  - close of the last CLOSED occurrence, projected.

    All ``prev_*`` bands are NaN until the first occurrence of that session
    has closed (there is no "previous session" yet). Access in on_data():

        self.sl.london_high[-1]        -> running high of today's London session
        self.sl.london_prev_close[-1]  -> close of yesterday's London session

    Args:
        sessions : list[str | dict]
            Same format as MarketSessions: predefined strings ("sydney",
            "tokyo", "london", "new_york") or custom dicts
            {"name": "asia", "start": 0, "end": 8, "color": "#FF6B35"}.
        colors : dict[str, str] | None
            Color overrides by session name (used by draw()).

    Usage:
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='15m')
                self.sl = self.add_indicator(
                    SessionLevels.refs(self.btc),
                    SessionLevels(sessions=["tokyo", "london"]),
                    plot=True,
                )

            def on_data(self):
                # Trade only above yesterday's London high (bias filter)
                prev_high = self.sl.london_prev_high[-1]
                if not np.isnan(prev_high) and self.btc.close[-1] > prev_high:
                    pass  # bullish continuation setup
    """

    name     = "session_levels"
    category = "structure"
    output_names: list = []
    ts_band_indices: list = []

    _BAND_SUFFIXES = (
        "open", "high", "low", "prev_open", "prev_high", "prev_low", "prev_close",
    )

    def __init__(
        self,
        sessions: "list[str | dict] | None" = None,
        colors: "dict[str, str] | None" = None,
    ):
        if sessions is None:
            sessions = ["london", "new_york"]
        self._sessions_input  = sessions
        self._color_overrides = colors or {}
        self.length            = 1

        parsed = self._parse_sessions()
        self._session_names = [
            s["name"].replace(" ", "_").replace("-", "_").lower()
            for s in parsed
        ]
        self.output_names = [
            f"{name}_{suffix}"
            for name in self._session_names
            for suffix in self._BAND_SUFFIXES
        ]

        colors_per_band = [
            parsed[i]["color"]
            for i in range(len(parsed))
            for _ in self._BAND_SUFFIXES
        ]
        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="rect",
            color=colors_per_band or None,
            name="Session Levels",
        )

    @property
    def n_outputs(self) -> int:
        return max(len(self.output_names), 1)

    @property
    def min_periods(self) -> int:
        return 1

    def _parse_sessions(self) -> list[dict]:
        """Same predefined/custom window format as MarketSessions."""
        return _parse_session_windows(
            self._sessions_input, _SESSIONS_UTC, _SESSION_COLORS,
            self._color_overrides,
        )

    @staticmethod
    def refs(ohlc_proxy):
        """
        Build the ColumnRef list for this indicator in the expected order.

        Args:
            ohlc_proxy (OhlcProxy): Proxy returned by subscribe_ohlc().

        Returns:
            list[ColumnRef]: [ts, open, high, low, close] refs.
        """
        return [
            ohlc_proxy.ts_ref,
            ohlc_proxy.open_ref,
            ohlc_proxy.high_ref,
            ohlc_proxy.low_ref,
            ohlc_proxy.close_ref,
        ]

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Args:
            source: [N x 5] - ts(0), open(1), high(2), low(3), close(4)

        Returns:
            [7K x N] - 7 bands per configured session (see class docstring).

        Additionally saves self._occurrences (per session, list of closed/
        current occurrence boxes) so draw() can build the range rectangles
        and projected level lines.
        """
        n = len(source)
        if n == 0 or source.ndim != 2 or source.shape[1] < 5:
            return np.full((max(len(self.output_names), 1), max(n, 1)), np.nan)

        ts    = source[:, 0].astype(np.float64)
        open_ = source[:, 1].astype(np.float64)
        high  = source[:, 2].astype(np.float64)
        low   = source[:, 3].astype(np.float64)
        close = source[:, 4].astype(np.float64)

        sessions = self._parse_sessions()
        K = len(sessions)
        if K == 0:
            return np.full((1, n), np.nan, dtype=np.float64)

        out = np.full((7 * K, n), np.nan, dtype=np.float64)
        occurrences_by_session: list[list[dict]] = []

        for k, sess in enumerate(sessions):
            mask = _mask_in_window(
                ts, sess["start"], sess["end"],
                sess.get("start_min", 0), sess.get("end_min", 0),
            )
            occ = _occurrence_ids(mask)
            bands = _running_ohlc_by_occurrence(ts, open_, high, low, close, occ)

            base = 7 * k
            out[base + 0, :] = bands["open"]
            out[base + 1, :] = bands["high"]
            out[base + 2, :] = bands["low"]
            out[base + 3, :] = bands["prev_open"]
            out[base + 4, :] = bands["prev_high"]
            out[base + 5, :] = bands["prev_low"]
            out[base + 6, :] = bands["prev_close"]

            # Build per-occurrence boxes for draw(): one dict per closed or
            # in-progress occurrence with its bar range and final OHLC.
            occ_boxes: list[dict] = []
            uniq = np.unique(occ[occ >= 0])
            for oid in uniq:
                idxs = np.where(occ == oid)[0]
                i0, i1 = int(idxs[0]), int(idxs[-1])
                occ_boxes.append({
                    "ts_start": int(ts[i0]),
                    "ts_end":   int(ts[i1]),
                    "open":  float(open_[i0]),
                    "high":  float(np.max(high[idxs])),
                    "low":   float(np.min(low[idxs])),
                    "close": float(close[i1]),
                    "color": sess["color"],
                    "name":  sess["name"],
                })
            occurrences_by_session.append(occ_boxes)

        self._occurrences = occurrences_by_session
        self._sessions_parsed = sessions
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """
        Emit each occurrence's range as a Rects box (low..high) and the
        previous closed occurrence's open/high/low/close as HLines projected
        forward to the start of the next occurrence, plus a Labels marker at
        each box's left edge.

        A session that has not closed yet (the last occurrence in the data)
        gets no projected HLines - there is nothing to project until it
        closes and the next occurrence's calculate() run captures it via
        prev_*.
        """
        from tradetropy.ta.draw import Rects, HLines, Labels

        occurrences = getattr(self, "_occurrences", [])
        if not occurrences:
            return []

        rect_x0, rect_x1, rect_y0, rect_y1, rect_colors = [], [], [], [], []
        line_x0, line_x1, line_y, line_colors = [], [], [], []
        label_x, label_y, label_text, label_colors = [], [], [], []

        for occ_boxes in occurrences:
            for j, box in enumerate(occ_boxes):
                rect_x0.append(box["ts_start"])
                rect_x1.append(box["ts_end"])
                rect_y0.append(box["low"])
                rect_y1.append(box["high"])
                rect_colors.append(box["color"])

                label_x.append(box["ts_start"])
                label_y.append(box["high"])
                label_text.append(f" {box['name']}")
                label_colors.append(box["color"])

                # Project this (now closed) occurrence's OHLC as levels
                # spanning until the next occurrence starts (or, for the
                # last one, extend one box-width to the right).
                if j + 1 < len(occ_boxes):
                    proj_end = occ_boxes[j + 1]["ts_start"]
                else:
                    width = box["ts_end"] - box["ts_start"]
                    proj_end = box["ts_end"] + max(width, 1)

                for level_key in ("open", "high", "low", "close"):
                    line_x0.append(box["ts_end"])
                    line_x1.append(proj_end)
                    line_y.append(box[level_key])
                    line_colors.append(box["color"])

        prims = []
        if rect_x0:
            prims.append(Rects(
                x0=rect_x0, x1=rect_x1, y0=rect_y0, y1=rect_y1,
                fill_color=rect_colors, line_color=rect_colors,
                fill_alpha=0.10, line_alpha=0.5, line_width=1.0,
            ))
        if line_x0:
            prims.append(HLines(
                x0=line_x0, x1=line_x1, y=line_y,
                color=line_colors, alpha=0.6, width=1.0, dash="dashed",
            ))
        if label_x:
            prims.append(Labels(
                x=label_x, y=label_y, text=label_text, color=label_colors,
                font_size="8pt", x_offset=2, y_offset=-2,
                text_align="left", text_baseline="bottom",
            ))
        return prims


# =====
# KillZones (ICT-style narrow time windows - range + active flag)
# =====
#
# Default UTC windows for the classic ICT kill zones (approximate, DST-naive -
# pass custom dicts via `windows=` for a DST-aware or broker-specific variant).
# These times are widely published defaults, not a proprietary definition:
#   - London Open KZ : 07:00-10:00 UTC (London session opening drive)
#   - New York Open KZ : 12:00-15:00 UTC (NY session opening drive, overlaps
#     the tail of London)
#   - London Close KZ : 14:00-16:00 UTC (London/NY overlap into the London
#     close, a.k.a. "power hour")
#   - Asian KZ : 00:00-04:00 UTC (Tokyo session liquidity build-up, feeds the
#     London Open KZ)
_KILLZONES_UTC = {
    "asian":       (0, 0, 4, 0),
    "london_open": (7, 0, 10, 0),
    "ny_open":      (12, 0, 15, 0),
    "london_close": (14, 0, 16, 0),
}
_KILLZONE_COLORS = {
    "asian":        "#FFA500",
    "london_open":  "#4ECDC4",
    "ny_open":      "#45B7D1",
    "london_close": "#9B59B6",
}


class KillZones(Indicator):
    """
    ICT-style Kill Zones: narrow high-probability time windows within the
    trading day, exposing the developing range (high/low) of each window plus
    a binary "active" flag - the time-window analogue of an Opening Range /
    Initial Balance, scoped to the classic ICT sessions rather than a fixed
    N-minute window from an arbitrary session open.

    Unlike SessionLevels (whole-session OHLC), a Kill Zone's ``high``/``low``
    FREEZE the instant the window closes and stay projected forward as
    breakout reference levels until the SAME kill zone opens again (there is
    no "previous session" chaining across different kill zones).

    Expected source: [N x 5] - ts(0), open(1), high(2), low(3), close(4)
    (use ``KillZones.refs(ohlc_proxy)`` to build the columns).

    Outputs - [3K x N] where K = number of configured kill zones. For each
    zone ``z`` (in the order given to ``windows``), 3 bands:

        {z}_high    - while active: running high (cummax) of the current
                      window. Once the window closes: frozen at the window's
                      final high and projected forward until the SAME zone
                      opens again. NaN before the first occurrence.
        {z}_low     - symmetric running/frozen low.
        {z}_active  - 1.0 while inside the window, 0.0 otherwise.

    Access in on_data():
        self.kz.london_open_high[-1]   -> current/last London Open KZ high
        self.kz.london_open_active[-1] -> 1.0 if inside the window right now

        # Breakout above the last closed London Open KZ high:
        if self.kz.london_open_active[-1] == 0.0:
            if self.btc.close[-1] > self.kz.london_open_high[-1]:
                pass  # bullish breakout of the kill zone range

    Predefined windows (strings): "asian", "london_open", "ny_open",
    "london_close" (see module-level _KILLZONES_UTC for the exact UTC hours).
    Also accepts custom dicts, same format as MarketSessions:
        {"name": "silver_bullet", "start": 10, "end": 11, "color": "#FF0000"}

    Args:
        windows : list[str | dict]
            Kill zones to track. Defaults to all four predefined ones.
        colors : dict[str, str] | None
            Color overrides by zone name (used by draw()).

    Usage:
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.kz = self.add_indicator(
                    KillZones.refs(self.btc),
                    KillZones(windows=["london_open", "ny_open"]),
                    plot=True,
                )

            def on_data(self):
                if self.kz.ny_open_active[-1] == 1.0:
                    pass  # inside the NY Open kill zone right now
    """

    name     = "kill_zones"
    category = "structure"
    output_names: list = []
    ts_band_indices: list = []

    _BAND_SUFFIXES = ("high", "low", "active")

    def __init__(
        self,
        windows: "list[str | dict] | None" = None,
        colors: "dict[str, str] | None" = None,
    ):
        if windows is None:
            windows = list(_KILLZONES_UTC.keys())
        self._windows_input   = windows
        self._color_overrides = colors or {}
        self.length            = 1

        parsed = self._parse_windows()
        self._zone_names = [
            w["name"].replace(" ", "_").replace("-", "_").lower()
            for w in parsed
        ]
        self.output_names = [
            f"{name}_{suffix}"
            for name in self._zone_names
            for suffix in self._BAND_SUFFIXES
        ]

        colors_per_band = [
            parsed[i]["color"]
            for i in range(len(parsed))
            for _ in self._BAND_SUFFIXES
        ]
        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="rect",
            color=colors_per_band or None,
            name="Kill Zones",
        )

    @property
    def n_outputs(self) -> int:
        return max(len(self.output_names), 1)

    @property
    def min_periods(self) -> int:
        return 1

    def _parse_windows(self) -> list[dict]:
        """Same predefined/custom window format as MarketSessions."""
        return _parse_session_windows(
            self._windows_input, _KILLZONES_UTC, _KILLZONE_COLORS,
            self._color_overrides,
        )

    @staticmethod
    def refs(ohlc_proxy):
        """
        Build the ColumnRef list for this indicator in the expected order.

        Args:
            ohlc_proxy (OhlcProxy): Proxy returned by subscribe_ohlc().

        Returns:
            list[ColumnRef]: [ts, open, high, low, close] refs.
        """
        return [
            ohlc_proxy.ts_ref,
            ohlc_proxy.open_ref,
            ohlc_proxy.high_ref,
            ohlc_proxy.low_ref,
            ohlc_proxy.close_ref,
        ]

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Args:
            source: [N x 5] - ts(0), open(1), high(2), low(3), close(4)

        Returns:
            [3K x N] - 3 bands per configured kill zone (see class docstring).

        Additionally saves self._occurrences (per zone, list of closed/
        current window boxes) so draw() can build the range rectangles and
        projected breakout level lines.
        """
        n = len(source)
        if n == 0 or source.ndim != 2 or source.shape[1] < 5:
            return np.full((max(len(self.output_names), 1), max(n, 1)), np.nan)

        ts   = source[:, 0].astype(np.float64)
        high = source[:, 2].astype(np.float64)
        low  = source[:, 3].astype(np.float64)

        windows = self._parse_windows()
        K = len(windows)
        if K == 0:
            return np.full((1, n), np.nan, dtype=np.float64)

        out = np.full((3 * K, n), np.nan, dtype=np.float64)
        occurrences_by_zone: list[list[dict]] = []

        for k, win in enumerate(windows):
            mask = _mask_in_window(
                ts, win["start"], win["end"],
                win.get("start_min", 0), win.get("end_min", 0),
            )
            occ = _occurrence_ids(mask)

            running_high = np.full(n, np.nan, dtype=np.float64)
            running_low  = np.full(n, np.nan, dtype=np.float64)
            frozen_high  = np.full(n, np.nan, dtype=np.float64)
            frozen_low   = np.full(n, np.nan, dtype=np.float64)

            last_closed_high = np.nan
            last_closed_low  = np.nan
            cur_occ = -1
            cur_high = np.nan
            cur_low = np.nan

            for i in range(n):
                o = occ[i]
                if o >= 0:
                    if o != cur_occ:
                        if cur_occ >= 0:
                            last_closed_high = cur_high
                            last_closed_low = cur_low
                        cur_occ = o
                        cur_high = high[i]
                        cur_low = low[i]
                    else:
                        cur_high = max(cur_high, high[i])
                        cur_low = min(cur_low, low[i])
                    running_high[i] = cur_high
                    running_low[i] = cur_low
                else:
                    if cur_occ >= 0:
                        last_closed_high = cur_high
                        last_closed_low = cur_low
                        cur_occ = -1
                    frozen_high[i] = last_closed_high
                    frozen_low[i] = last_closed_low

            base = 3 * k
            # active band
            active = (occ >= 0).astype(np.float64)
            # high/low: running while active, frozen/projected while inactive
            band_high = np.where(occ >= 0, running_high, frozen_high)
            band_low  = np.where(occ >= 0, running_low, frozen_low)
            out[base + 0, :] = band_high
            out[base + 1, :] = band_low
            out[base + 2, :] = active

            occ_boxes: list[dict] = []
            uniq = np.unique(occ[occ >= 0])
            for oid in uniq:
                idxs = np.where(occ == oid)[0]
                i0, i1 = int(idxs[0]), int(idxs[-1])
                occ_boxes.append({
                    "ts_start": int(ts[i0]),
                    "ts_end":   int(ts[i1]),
                    "high":  float(np.max(high[idxs])),
                    "low":   float(np.min(low[idxs])),
                    "color": win["color"],
                    "name":  win["name"],
                })
            occurrences_by_zone.append(occ_boxes)

        self._occurrences = occurrences_by_zone
        self._windows_parsed = windows
        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """
        Emit each kill zone occurrence's range as a Rects box (low..high) and
        its high/low as HLines projected forward as breakout reference levels
        until the SAME kill zone opens again, plus a Labels marker.
        """
        from tradetropy.ta.draw import Rects, HLines, Labels

        occurrences = getattr(self, "_occurrences", [])
        if not occurrences:
            return []

        rect_x0, rect_x1, rect_y0, rect_y1, rect_colors = [], [], [], [], []
        line_x0, line_x1, line_y, line_colors = [], [], [], []
        label_x, label_y, label_text, label_colors = [], [], [], []

        for occ_boxes in occurrences:
            for j, box in enumerate(occ_boxes):
                rect_x0.append(box["ts_start"])
                rect_x1.append(box["ts_end"])
                rect_y0.append(box["low"])
                rect_y1.append(box["high"])
                rect_colors.append(box["color"])

                label_x.append(box["ts_start"])
                label_y.append(box["high"])
                label_text.append(f" {box['name']}")
                label_colors.append(box["color"])

                if j + 1 < len(occ_boxes):
                    proj_end = occ_boxes[j + 1]["ts_start"]
                else:
                    width = box["ts_end"] - box["ts_start"]
                    proj_end = box["ts_end"] + max(width, 1)

                for level_key in ("high", "low"):
                    line_x0.append(box["ts_end"])
                    line_x1.append(proj_end)
                    line_y.append(box[level_key])
                    line_colors.append(box["color"])

        prims = []
        if rect_x0:
            prims.append(Rects(
                x0=rect_x0, x1=rect_x1, y0=rect_y0, y1=rect_y1,
                fill_color=rect_colors, line_color=rect_colors,
                fill_alpha=0.12, line_alpha=0.55, line_width=1.0,
            ))
        if line_x0:
            prims.append(HLines(
                x0=line_x0, x1=line_x1, y=line_y,
                color=line_colors, alpha=0.65, width=1.2, dash="dashed",
            ))
        if label_x:
            prims.append(Labels(
                x=label_x, y=label_y, text=label_text, color=label_colors,
                font_size="8pt", x_offset=2, y_offset=-2,
                text_align="left", text_baseline="bottom",
            ))
        return prims


# =====
# OrderBlock (rect annotation - Order Block zones)
# =====
class OrderBlock(Indicator):
    """
    Order Block (OB) - price zone of the last opposite candle before a strong impulse.
    Core concept of ICT/SMC analysis.

    Detection:
        - Bullish impulse : candle range > ATR x atr_mult AND close > open
          The bullish OB is the last bearish candle (close < open) before the impulse.
        - Bearish impulse : candle range > ATR x atr_mult AND close < open
          The bearish OB is the last bullish candle (close > open) before the impulse.
        - The OB zone is [low, high] of that prior candle.

    Mitigation: OB closes when price (close) enters its zone.

    Expected source: [N x 6] - ts(0), open(1), high(2), low(3), close(4), vol(5)

    Outputs - [4 x N] (2 price bands + 2 timestamp bands):
        row 0 : bull_top    - top of active bullish OB (NaN if none)
        row 1 : bull_bot    - bottom of active bullish OB
        row 2 : bear_top    - top of active bearish OB
        row 3 : bear_bot    - bottom of active bearish OB

    Args:
        atr_period : int   - ATR period for measuring impulses (default 14)
        atr_mult   : float - ATR multiplier to consider a strong impulse
                             (default 1.5)
        mitigate   : bool  - True -> OB disappears when price enters
                             (default True)
        show_mitigated : bool - True -> shows already-mitigated OBs, more faint
                                (default False)

    Usage:
        class MyStrategy(Strategy):
            def init(self):
                self.btc = self.subscribe_ohlc("BTCUSDT", timeframe='5m')
                self.ob = self.add_indicator(
                    [self.btc.ts_ref, self.btc.open_ref, self.btc.high_ref,
                     self.btc.low_ref, self.btc.close_ref, self.btc.volume_ref],
                    OrderBlock(atr_mult=1.5),
                    plot=True,
                )

            def on_data(self):
                if not np.isnan(self.ob.bull_bot[-1]):
                    ob_top = self.ob.bull_top[-1]
                    ob_bot = self.ob.bull_bot[-1]
                    if self.btc.close[-1] <= ob_top:
                        pass
    """

    name     = "ob"
    category = "annotation"

    output_names    = ["bull_top", "bull_bot", "bear_top", "bear_bot"]
    ts_band_indices = [4, 5, 6, 7]
    ts_output_names = ["ts_bull_open", "ts_bull_close", "ts_bear_open", "ts_bear_close"]

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        mitigate: bool = True,
        show_mitigated: bool = False,
        bull_color: str = "#0ECB81",
        bear_color: str = "#F6465D",
    ):
        self.atr_period     = atr_period
        self.atr_mult       = atr_mult
        self.mitigate       = mitigate
        self.show_mitigated = show_mitigated
        self.length         = atr_period

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            exclude_from_autoscale=True,
            renderer="rect",
            color=[bull_color, bull_color, bear_color, bear_color],
            rect_fill_alpha=0.13,
            rect_line_alpha=0.6,
            rect_line_width=1.0,
            name="OB",
        )

    @property
    def min_periods(self) -> int:
        return self.atr_period + 2

    def display_name(self) -> str:
        return f"OB(ATR x{self.atr_mult})"

    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        n = len(high)
        atr = np.full(n, np.nan, dtype=np.float64)
        L = self.atr_period
        if n < L + 1:
            return atr
        prev_c = np.roll(close, 1)
        prev_c[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
        alpha = 1.0 / L
        atr[L] = float(np.mean(tr[1:L + 1]))
        for i in range(L + 1, n):
            atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i]
        return atr

    def calculate(self, source: np.ndarray) -> np.ndarray:
        """
        Args:
            source: [N x 6] - ts(0), open(1), high(2), low(3), close(4), vol(5)

        Returns:
            [8 x N]
        """
        n = len(source)
        out = np.full((8, n), np.nan, dtype=np.float64)
        if n < self.atr_period + 2:
            self._bull_obs = []
            self._bear_obs = []
            self._show_mitigated = self.show_mitigated
            return out

        ts    = source[:, 0].astype(np.float64)
        open_ = source[:, 1].astype(np.float64)
        high  = source[:, 2].astype(np.float64)
        low   = source[:, 3].astype(np.float64)
        close = source[:, 4].astype(np.float64)

        atr = self._compute_atr(high, low, close)
        rng = high - low

        bull_obs: list[list] = []
        bear_obs: list[list] = []

        for i in range(1, n):
            if np.isnan(atr[i]):
                continue
            threshold = atr[i] * self.atr_mult

            if close[i] > open_[i] and rng[i] > threshold:
                for j in range(i - 1, -1, -1):
                    if close[j] < open_[j]:
                        bull_obs.append([j, high[j], low[j], ts[j], np.nan])
                        break

            elif close[i] < open_[i] and rng[i] > threshold:
                for j in range(i - 1, -1, -1):
                    if close[j] > open_[j]:
                        bear_obs.append([j, high[j], low[j], ts[j], np.nan])
                        break

        seen_bull: set[int] = set()
        seen_bear: set[int] = set()
        bull_obs_dedup = []
        bear_obs_dedup = []
        for ob in bull_obs:
            if ob[0] not in seen_bull:
                seen_bull.add(ob[0])
                bull_obs_dedup.append(ob)
        for ob in bear_obs:
            if ob[0] not in seen_bear:
                seen_bear.add(ob[0])
                bear_obs_dedup.append(ob)
        bull_obs = bull_obs_dedup
        bear_obs = bear_obs_dedup

        if self.mitigate:
            for ob in bull_obs:
                bar_i, top, bot, ts_open, _ = ob
                for j in range(bar_i + 1, n):
                    if bot <= close[j] <= top:
                        ob[4] = ts[j]
                        break
                    if close[j] < bot:
                        ob[4] = ts[j]
                        break

            for ob in bear_obs:
                bar_i, top, bot, ts_open, _ = ob
                for j in range(bar_i + 1, n):
                    if bot <= close[j] <= top:
                        ob[4] = ts[j]
                        break
                    if close[j] > top:
                        ob[4] = ts[j]
                        break

        self._bull_obs       = bull_obs
        self._bear_obs       = bear_obs
        self._show_mitigated = self.show_mitigated

        bull_by_bar = {ob[0]: ob for ob in bull_obs}
        bear_by_bar = {ob[0]: ob for ob in bear_obs}
        active_bulls: list = []
        active_bears: list = []

        for i in range(n):
            if i in bull_by_bar:
                active_bulls.append(bull_by_bar[i])
            if i in bear_by_bar:
                active_bears.append(bear_by_bar[i])

            if self.mitigate:
                active_bulls = [
                    ob for ob in active_bulls
                    if np.isnan(ob[4]) or ts[i] < ob[4]
                ]
                active_bears = [
                    ob for ob in active_bears
                    if np.isnan(ob[4]) or ts[i] < ob[4]
                ]

            if active_bulls:
                last = active_bulls[-1]
                out[0, i] = last[1]
                out[1, i] = last[2]
                out[4, i] = last[3]
                out[5, i] = last[4]
            if active_bears:
                last = active_bears[-1]
                out[2, i] = last[1]
                out[3, i] = last[2]
                out[6, i] = last[3]
                out[7, i] = last[4]

        return out

    def draw(self, cfg=None, *, interval_ms=None) -> list:
        """Emit the Order Block zones as Rects primitives (open zones extend 20 bars)."""
        cfg = cfg or self.plot_config
        return _mitigated_rects(
            getattr(self, "_bull_obs", []),
            getattr(self, "_bear_obs", []),
            getattr(self, "_show_mitigated", False),
            cfg, interval_ms, extend_mult=20,
        )


# =====
# PivotPoints (floor-trader pivot levels - step series)
# =====
_PIVOT_METHODS = ('classic', 'camarilla', 'woodie', 'demark')

_PIVOT_LEVELS = {
    'classic':   ['pp', 'r1', 'r2', 'r3', 's1', 's2', 's3'],
    'woodie':    ['pp', 'r1', 'r2', 's1', 's2'],
    'camarilla': ['pp', 'r1', 'r2', 'r3', 'r4', 's1', 's2', 's3', 's4'],
    'demark':    ['pp', 'r1', 's1'],
}

_PIVOT_PP_COLOR = '#2563EB'
_PIVOT_R_COLOR = '#F6465D'
_PIVOT_S_COLOR = '#0ECB81'


class PivotPoints(Indicator):
    """
    Floor-trader Pivot Points. Horizontal support/resistance levels computed
    from the PREVIOUS period's OHLC (day, week, ...) and projected onto the
    current period. Strictly causal: within a period the levels are constant
    and depend only on the already-closed period.

    Available methods (`method`):
        - 'classic'   : PP, R1-R3, S1-S3 (standard floor-pivot formula).
        - 'woodie'    : close-weighted PP, R1-R2, S1-S2.
        - 'camarilla' : PP, R1-R4, S1-S4 (multipliers 1.1/12, 1.1/6, ...).
        - 'demark'    : PP, R1, S1 (based on the period's open/close relation).

    Drawn as 'step' series (flat within the period, with a jump at each
    boundary), TradingView-style.
    source: OHLCV [N x 5] - ts(0), open(1), high(2), low(3), close(4)
    (use ``PivotPoints.refs(ohlc_proxy)`` to build the columns).

    Args:
        method (str): 'classic' | 'camarilla' | 'woodie' | 'demark'.
        period (str): reset timeframe (e.g. '1d', '1w', '4h').
        anchor: period anchoring mode (see resolve_period_ids).

    Usage:
        self.piv = self.add_indicator(
            PivotPoints.refs(self.btc), PivotPoints('classic', '1d'),
        )

    Access in on_data():
        self.piv.pp[-1]   -> pivot point of the current period
        self.piv.r1[-1]   -> first resistance
        self.piv.s1[-1]   -> first support
    """

    name = 'pivots'
    category = 'structure'
    output_names: list = []
    ts_band_indices: list = []

    def __init__(self, method: str = 'classic', period: str = '1d', anchor='utc'):
        from tradetropy.core.constants import parse_timeframe

        method = method.lower()
        if method not in _PIVOT_METHODS:
            from tradetropy.exceptions import ConfigError
            raise ConfigError(
                f"PivotPoints: unknown method '{method}'. "
                f"Options: {list(_PIVOT_METHODS)}"
            )
        self.method = method
        self.period = period
        self.period_ms = parse_timeframe(period)
        self.anchor = anchor
        self.length = 1

        self.output_names = list(_PIVOT_LEVELS[method])

        # Color por tipo de nivel: PP azul, resistencias rojas, soportes verdes.
        colors, dashes, widths = [], [], []
        for lvl in self.output_names:
            if lvl == 'pp':
                colors.append(_PIVOT_PP_COLOR)
                dashes.append('solid')
                widths.append(1.6)
            elif lvl.startswith('r'):
                colors.append(_PIVOT_R_COLOR)
                dashes.append('dashed')
                widths.append(1.1)
            else:
                colors.append(_PIVOT_S_COLOR)
                dashes.append('dashed')
                widths.append(1.1)

        self.plot_config = IndicatorPlotConfig(
            overlay=True,
            renderer='step',
            exclude_from_autoscale=True,
            color=colors,
            line_dash=dashes,
            line_width=widths,
        )

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    @property
    def min_periods(self) -> int:
        return 1

    def display_name(self) -> str:
        return f'Pivots({self.method})'

    def col_name(self, symbol: str, col_source: str = '') -> str:
        return f'pivots_{self.method}_{self.period}_{symbol}'

    @staticmethod
    def refs(ohlc_proxy):
        """
        Build the ColumnRef list for this indicator in the expected order.

        Args:
            ohlc_proxy (OhlcProxy): Proxy returned by subscribe_ohlc().

        Returns:
            list[ColumnRef]: [ts, open, high, low, close] refs.
        """
        return [
            ohlc_proxy.ts_ref,
            ohlc_proxy.open_ref,
            ohlc_proxy.high_ref,
            ohlc_proxy.low_ref,
            ohlc_proxy.close_ref,
        ]

    def _levels(self, o: float, h: float, l: float, c: float) -> dict:
        """Compute the pivot levels for one previous-period OHLC."""
        rng = h - l
        if self.method == 'classic':
            pp = (h + l + c) / 3.0
            return {
                'pp': pp,
                'r1': 2.0 * pp - l, 's1': 2.0 * pp - h,
                'r2': pp + rng, 's2': pp - rng,
                'r3': h + 2.0 * (pp - l), 's3': l - 2.0 * (h - pp),
            }
        if self.method == 'woodie':
            pp = (h + l + 2.0 * c) / 4.0
            return {
                'pp': pp,
                'r1': 2.0 * pp - l, 's1': 2.0 * pp - h,
                'r2': pp + rng, 's2': pp - rng,
            }
        if self.method == 'camarilla':
            pp = (h + l + c) / 3.0
            return {
                'pp': pp,
                'r1': c + rng * 1.1 / 12.0, 's1': c - rng * 1.1 / 12.0,
                'r2': c + rng * 1.1 / 6.0, 's2': c - rng * 1.1 / 6.0,
                'r3': c + rng * 1.1 / 4.0, 's3': c - rng * 1.1 / 4.0,
                'r4': c + rng * 1.1 / 2.0, 's4': c - rng * 1.1 / 2.0,
            }
        # demark
        if c < o:
            x = h + 2.0 * l + c
        elif c > o:
            x = 2.0 * h + l + c
        else:
            x = h + l + 2.0 * c
        pp = x / 4.0
        return {'pp': pp, 'r1': x / 2.0 - l, 's1': x / 2.0 - h}

    def calculate(self, source: np.ndarray) -> np.ndarray:
        from tradetropy.ta._volume_profile import resolve_period_ids

        K = len(self.output_names)
        if source.ndim != 2 or source.shape[1] < 5 or len(source) == 0:
            return np.full((K, max(len(source), 1)), np.nan, dtype=np.float64)

        ts = source[:, 0].astype(np.int64)
        open_ = source[:, 1].astype(np.float64)
        high = source[:, 2].astype(np.float64)
        low = source[:, 3].astype(np.float64)
        close = source[:, 4].astype(np.float64)
        n = len(ts)

        out = np.full((K, n), np.nan, dtype=np.float64)

        period_id, _ = resolve_period_ids(ts, self.period_ms, self.anchor)

        # Aggregate OHLC per period in a single pass (ascending ts).
        # prev_levels holds the levels derived from the last CLOSED period.
        prev_levels = None
        cur_pid = period_id[0]
        cur_o = open_[0]
        cur_h = high[0]
        cur_l = low[0]
        cur_c = close[0]

        for i in range(n):
            pid = period_id[i]
            if pid != cur_pid:
                # A new period starts: the just-finished period becomes the
                # reference for this and the following bars of the new period.
                prev_levels = self._levels(cur_o, cur_h, cur_l, cur_c)
                cur_pid = pid
                cur_o = open_[i]
                cur_h = high[i]
                cur_l = low[i]
                cur_c = close[i]
            else:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
                cur_c = close[i]

            if prev_levels is not None:
                for k, lvl in enumerate(self.output_names):
                    out[k, i] = prev_levels[lvl]

        return out

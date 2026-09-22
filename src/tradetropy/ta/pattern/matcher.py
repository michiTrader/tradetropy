"""
matcher.py
==========
PatternMatcherProxy — handle that the strategy uses in on_data().
PatternMatcherDef   — declarative definition stored in strategy._pattern_matcher_defs.

The proxy is analogous to IndicatorProxy: the strategy declares it in init(),
the engine connects it to the PatternStore before the loop, and in on_data()
it only does an O(log n) or O(1) lookup.

Usage in Strategy
─────────────────
    def init(self):
        self.cpivot = self.add_indicator(..., ConfirmedPivot(swing=3))
        self.nbs    = self.add_indicator(..., NBS(swing=3))

        self.setup  = self.add_pattern_matcher(
            base_pivot=self.cpivot,
            decorators=[self.nbs],
            pattern = Pattern([...], tag="setup")
        )

    def on_data(self):
        match = self.setup.last
        if match:
            print(match.tag, match.values)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradetropy.ta.pattern.store   import PatternStore
    from tradetropy.ta.pattern.types   import MatchResult
    from tradetropy.ta.pattern.pattern import Pattern


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHER DEF
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatternMatcherDef:
    """
    Declarative definition of a PatternMatcher.

    ``base_pivot`` supplies the confirmed H/L sequence. ``decorators`` add
    tags such as ``nbs`` or ``hhll`` to that sequence.
    """

    base_pivot: object
    decorators: list
    pattern: 'Pattern'
    proxy: 'PatternMatcherProxy'


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHER PROXY
# ══════════════════════════════════════════════════════════════════════════════

class PatternMatcherProxy:
    """
    Handle that the strategy uses to access the last match.

    The engine connects it to the PatternStore during setup and advances
    it on each loop iteration.

    In backtest  → _advance(bar_index) does O(log n) lookup in PatternStore.
    In live      → _advance_live() returns store.last_match directly (O(1)).
    In both      → self.last returns MatchResult | None.
    """

    __slots__ = ('_store', '_cursor', '_last_match', '_mode',
                 '_tick_to_candle_map', '_ohlc_ring')

    def __init__(self):
        self._store:             'PatternStore | None' = None
        self._cursor:            int                   = -1
        self._last_match:        'MatchResult | None'  = None
        self._mode:              str                   = 'backtest'
        self._tick_to_candle_map: 'np.ndarray | None'  = None
        # Live only: the source OHLC ring, used to read the current closed-candle
        # index for the @active_bars activation window. None in backtest (the
        # index comes from the tick->candle mapping instead).
        self._ohlc_ring = None

    # ── Connection to store (called by engine) ─────────────────────────────────

    def _connect_backtest(self, store: 'PatternStore', ohlc_store=None) -> None:
        """Connect proxy to pre-calculated PatternStore (backtest/optimization)."""
        self._store = store
        self._mode  = 'backtest'
        if ohlc_store is not None:
            self._tick_to_candle_map = getattr(ohlc_store, 'tick_to_candle_mapping', None)

    def _connect_live(self, store: 'PatternStore', ohlc_ring=None) -> None:
        """
        Connect proxy to incremental PatternStore (live).

        Args:
            store: The incrementally-updated PatternStore.
            ohlc_ring: The source OHLC ring, so the @active_bars window can
                read the current closed-candle index. Optional; when absent
                the activation window is not enforced.
        """
        self._store = store
        self._mode  = 'live'
        self._ohlc_ring = ohlc_ring

    # ── Active window filtering ────────────────────────────────────────────────

    def _apply_active_window(self, match, current_candle_index):
        """
        Drop the match if it falls outside the pattern's @active_bars window.

        ``current_candle_index`` is the index of the last CLOSED candle, in the
        same frame as ``match.last_node.index`` (the confirmation bar of the
        last pivot). The age is their difference (0 == confirmation bar). With
        no @active_bars directive this is a no-op.
        """
        if match is None or self._store is None:
            return match
        pattern = self._store._pattern
        if not pattern.has_active_window or current_candle_index is None:
            return match
        age = current_candle_index - match.last_node.index
        return match if pattern.active_age_ok(age) else None

    # ── Cursor advance (called by engine on each iteration) ───────────────────

    def _advance(self, bar_index: int) -> None:
        """
        Backtest -- advance cursor and update the last match.
        O(log n) per iteration.
        Called by BacktestEngine after advancing indicators and BEFORE on_data().
        """
        self._cursor = bar_index

        if self._store is None:
            self._last_match = None
            return

        # tick_to_candle_map[bar_index] = n_closed = index of PARTIAL candle,
        # i.e. the number of completely closed candles up to this tick.
        # Closed candles are at positions 0..n_closed-1 in the PatternStore.
        # Candle n_closed is the partial -- not yet closed, store does not see it.
        #
        # Without this conversion last_match_at receives n_closed and may return
        # a match whose last pivot is in the partial candle (still open),
        # causing exactly 1-bar lookahead.
        if self._tick_to_candle_map is not None and bar_index < len(self._tick_to_candle_map):
            n_closed = int(self._tick_to_candle_map[bar_index])
            store_idx  = n_closed - 1   # last CLOSED candle
        else:
            store_idx = bar_index

        match = self._store.last_match_at(store_idx)
        self._last_match = self._apply_active_window(match, store_idx)

    def _advance_live(self) -> None:
        """
        Live -- update last match from store.
        O(1) — the store already has the most recent state.
        Called by LiveEngine after processing each tick/bar.
        """
        if self._store is None:
            self._last_match = None
            return

        match = self._store.last_match
        current_idx = (
            self._ohlc_ring._n_closed - 1
            if self._ohlc_ring is not None
            else None
        )
        self._last_match = self._apply_active_window(match, current_idx)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def last(self) -> 'MatchResult | None':
        """
        Last match found up to the current bar.

        Returns MatchResult if there is a match, None otherwise.

        Example
        ───────
            match = self.setup.last
            if match:
                n0 = match[0]
                n0.value      # 3500.0
                n0.index      # 45
                n0.timestamp  # 1234567890000
                n0.type       # "H"
                n0.tags       # {"type": "H", "nbs": "neu"}

                match.tag         # "strong_setup"
                match.indices     # [45, 48, 52]
                match.values      # [3500.0, 3200.0, 3600.0]
                match.timestamps  # [t1, t2, t3]
                match.first       # PivotPoint of first node
                match.last_node   # PivotPoint of last node
        """
        return self._last_match

    def __repr__(self) -> str:
        mode_str = f"mode={self._mode!r}"
        match_str = f"last={self._last_match!r}" if self._last_match else "last=None"
        return f"PatternMatcherProxy({mode_str}, {match_str})"

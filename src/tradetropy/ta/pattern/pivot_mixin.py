"""
pivot_mixin.py
==============
PivotIndicatorMixin — mixin that all indicators producing pivots with tags
must inherit (ConfirmedPivot, NBS, HHLL...).

Defines the contract that PatternMatcherDef needs to build the
FrozenPivotSequence from an OhlcDataStore.

Contract
────────
Each pivot indicator must implement:

    tag_name : str
        Name of the tag this indicator produces.
        ConfirmedPivot → "type"   (tags: "H" or "L")
        NBS            → "nbs"    (tags: "neu", "boo", "shk", "emp", "")
        HHLL           → "hhll"   (tags: "HH", "HL", "LH", "LL", "")

    pivot_col_names(symbol) → tuple[str, ...]
        Column names in OhlcDataStore.col_index produced by this indicator,
        in the order:
          ConfirmedPivot: (ph_col, pl_col, ph_ts_col, pl_ts_col)
          NBS/HHLL:       (tag_col,)   ← a single tag column

    is_base_pivot : bool
        True only in ConfirmedPivot — indicates this indicator produces
        the base columns (ph, pl, ph_ts, pl_ts).
        False in decorators (NBS, HHLL...) — they produce a single tag column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


class PivotIndicatorMixin:
    """
    Mixin for indicators that produce pivots with tags.

    Inherit in ConfirmedPivot, NBS, HHLL and any future indicator
    that wants to participate in PatternMatcher.

    Class attributes to override
    ────────────────────────────
    tag_name      : str  — name of the tag this indicator produces.
    is_base_pivot : bool — True only in ConfirmedPivot.

    Methods to override
    ───────────────────
    pivot_col_names(symbol) → tuple[str, ...]
        Returns the column names in OhlcDataStore that this indicator
        produces and that are needed to build the sequence.
    """

    # Override in subclasses
    tag_name:      str  = ""
    is_base_pivot: bool = False

    def pivot_col_names(self, symbol: str) -> tuple[str, ...]:
        """
        Returns the column names in OhlcDataStore.col_index
        that this indicator produces for the given symbol.

        ConfirmedPivot returns:
            (
                "cpivot3_BTCUSDT_b0",   # ph  (pivot high price)
                "cpivot3_BTCUSDT_b1",   # pl  (pivot low price)
                "cpivot3_BTCUSDT_b2",   # ph_ts (real PH timestamp)
                "cpivot3_BTCUSDT_b3",   # pl_ts (real PL timestamp)
            )

        NBS returns:
            (
                "nbs3_BTCUSDT_b0",      # tag column ("neu","boo","shk","emp","")
            )

        HHLL returns:
            (
                "hhll3_BTCUSDT_b0",     # tag column ("HH","HL","LH","LL","")
            )
        """
        raise NotImplementedError(
            f"{type(self).__name__}.pivot_col_names() not implemented. "
            f"Implement in the subclass that inherits PivotIndicatorMixin."
        )
        

"""
sequence.py
===========
FrozenPivotSequence — immutable sequence of PivotPoint built from
ConfirmedPivot + decorator indicator outputs (NBS, HHLL...).

Built once per backtest and shared between workers in
optimization (read-only, no locks).

Construction
────────────
In backtest — from OhlcDataStore (bulk-precomputed arrays):

    sequence = FrozenPivotSequence.from_ohlc_store(
        ohlc_store        = store,
        base_col_names    = ("cpivot3_BTCUSDT_b0", "cpivot3_BTCUSDT_b1",
                             "cpivot3_BTCUSDT_b2", "cpivot3_BTCUSDT_b3"),
        decorator_cols    = {
            "nbs":  "nbs3_BTCUSDT_b0",
            "hhll": "hhll3_BTCUSDT_b0",
        },
    )

In live — from direct arrays (already compressed historical rings):

    sequence = FrozenPivotSequence.from_arrays(
        ph_array         = ph,
        pl_array         = pl,
        ph_ts_array      = ph_ts,
        pl_ts_array      = pl_ts,
        decorator_arrays = {"nbs": nbs_tags, "hhll": hhll_tags},
    )

Usage
─────
    pivot_idx = sequence.pivot_idx_at_bar(bar_index)  # O(log n)
    pivot     = sequence.pivots[pivot_idx]
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tradetropy.ta.pattern.types import PivotPoint

if TYPE_CHECKING:
    from tradetropy.data.data import OhlcDataStore


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN PIVOT SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrozenPivotSequence:
    """
    Compressed sequence of pivots with their tags.

    Attributes
    ──────────
    pivots            : list of PivotPoint ordered by index (confirmation bar).
    _pivot_bar_indices: array of bar indices for binary search — O(log n).

    Immutable after construction — safe for sharing between processes.
    """

    pivots:             list[PivotPoint]
    _pivot_bar_indices: np.ndarray       # [n_pivots] — bar indices for bisect

    def __len__(self) -> int:
        return len(self.pivots)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_ohlc_store(
        cls,
        ohlc_store:     'OhlcDataStore',
        base_col_names: tuple[str, str, str, str],
        decorator_cols: dict[str, str],
        tag_decoders:   'dict[str, dict[float, str]] | None' = None,
    ) -> 'FrozenPivotSequence':
        """
        Builds the sequence from an OhlcDataStore (backtest).

        tag_decoders: maps tag_name → dict{float_code: str_tag}.
            Pass NBS.TAG_DECODE and HHLL.TAG_DECODE for correct decoding.
        """
        col = ohlc_store.col_index
        mat = ohlc_store.matrix

        ph_col, pl_col, ph_ts_col, pl_ts_col = base_col_names

        ph    = mat[:, col[ph_col]]
        pl    = mat[:, col[pl_col]]
        ph_ts = mat[:, col[ph_ts_col]]
        pl_ts = mat[:, col[pl_ts_col]]

        dec_arrays: dict[str, np.ndarray] = {}
        for tag_name, col_name in decorator_cols.items():
            if col_name in col:
                dec_arrays[tag_name] = mat[:, col[col_name]]

        return cls._build(
            ph=ph, pl=pl, ph_ts=ph_ts, pl_ts=pl_ts,
            decorator_arrays=dec_arrays,
            tag_decoders=tag_decoders or {},
        )

    @classmethod
    def from_arrays(
        cls,
        ph_array:         np.ndarray,
        pl_array:         np.ndarray,
        ph_ts_array:      np.ndarray,
        pl_ts_array:      np.ndarray,
        decorator_arrays: dict[str, np.ndarray],
        tag_decoders:     'dict[str, dict[float, str]] | None' = None,
    ) -> 'FrozenPivotSequence':
        """
        Builds the sequence from direct arrays (live or pool).

        tag_decoders: maps tag_name → dict{float_code: str_tag}.
        """
        return cls._build(
            ph=ph_array,
            pl=pl_array,
            ph_ts=ph_ts_array,
            pl_ts=pl_ts_array,
            decorator_arrays=decorator_arrays,
            tag_decoders=tag_decoders or {},
        )

    @classmethod
    def _build(
        cls,
        ph:               np.ndarray,
        pl:               np.ndarray,
        ph_ts:            np.ndarray,
        pl_ts:            np.ndarray,
        decorator_arrays: dict[str, np.ndarray],
        tag_decoders:     'dict[str, dict[float, str]] | None' = None,
    ) -> 'FrozenPivotSequence':
        """
        Common build logic: compresses NaN and creates PivotPoints.

        tag_decoders: maps tag_name → dict{float_code: str_tag}.
            NBS.TAG_DECODE returns "H-neu" etc. — the part after the
            hyphen is extracted so PatternNode can filter by {"nbs": "neu"}.
            HHLL.TAG_DECODE returns "HH", "HL" etc. — used directly.
        """
        if tag_decoders is None:
            tag_decoders = {}

        n      = len(ph)
        pivots = []

        for i in range(n):
            has_h = not np.isnan(ph[i])
            has_l = not np.isnan(pl[i])

            if not has_h and not has_l:
                continue

            if has_h:
                ptype = 'H'
                value = float(ph[i])
                ts    = float(ph_ts[i]) if not np.isnan(ph_ts[i]) else float(i)
            else:
                ptype = 'L'
                value = float(pl[i])
                ts    = float(pl_ts[i]) if not np.isnan(pl_ts[i]) else float(i)

            # build tags
            tags: dict[str, str] = {"type": ptype}
            for tag_name, arr in decorator_arrays.items():
                if i >= len(arr):
                    tags[tag_name] = ""
                    continue

                raw = arr[i]

                # NaN → no tag
                if isinstance(raw, (float, np.floating)) and np.isnan(raw):
                    tags[tag_name] = ""
                    continue

                # decode with the indicator's decoder if available
                decoder = tag_decoders.get(tag_name)
                if decoder is not None:
                    decoded = decoder.get(float(raw), "")
                    # NBS returns "H-neu", "L-boo" etc.
                    # extract only the part after the hyphen ("neu", "boo", "shk", "emp")
                    # so PatternNode filters with {"nbs": "neu"} instead of {"nbs": "H-neu"}
                    if "-" in decoded:
                        decoded = decoded.split("-", 1)[1]
                    tags[tag_name] = decoded
                else:
                    # no decoder — convert to string directly
                    if isinstance(raw, (float, np.floating)):
                        tags[tag_name] = str(int(raw))
                    else:
                        tags[tag_name] = str(raw) if raw else ""

            pivots.append(PivotPoint(
                index=i,
                timestamp=ts,
                value=value,
                type=ptype,
                tags=tags,
            ))

        bar_indices = np.array([p.index for p in pivots], dtype=np.int64)

        return cls(pivots=pivots, _pivot_bar_indices=bar_indices)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def pivot_idx_at_bar(self, bar_index: int) -> int:
        """
        Returns the index (in self.pivots) of the last pivot
        whose confirmation bar is <= bar_index.

        O(log n) via binary search.
        Returns -1 if there is no pivot up to bar_index.
        """
        if len(self._pivot_bar_indices) == 0:
            return -1

        # bisect_right returns the insertion position for bar_index + 1
        # subtracting 1 gives the last index <= bar_index
        pos = bisect.bisect_right(self._pivot_bar_indices, bar_index) - 1
        return pos  # -1 if bar_index < all indices

    def pivots_up_to_bar(self, bar_index: int) -> list[PivotPoint]:
        """
        Returns all confirmed pivots up to bar_index (inclusive).
        """
        idx = self.pivot_idx_at_bar(bar_index)
        if idx < 0:
            return []
        return self.pivots[:idx + 1]

    def __repr__(self) -> str:
        return f"FrozenPivotSequence(n_pivots={len(self.pivots)})"
        

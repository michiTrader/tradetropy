"""SMC block detectors — pure, causal zone marking over plain numpy OHLC.

One order-block detector parameterised on three orthogonal axes (anchor / zone /
mitigation) plus optional validation filters. Every detector returns
``list[Block]``; ``Block.family`` says which kind it is.

Causality contract
------------------
Each ``Block`` carries ``confirmed_idx``: the bar index at which it first becomes
knowable from *closed* bars only. No detector uses bars after ``confirmed_idx``
to decide a block's existence or geometry. Mitigation only *consumes* a block, so
it may be stamped later — the one field allowed to change as new bars close. This
is the same anti-lookahead rule ``MarketStructure`` and ``ConfirmedPivot`` follow,
which is what keeps backtest and live in parity.

Quick use::

    from tradetropy.ta.order_blocks import find_order_blocks
    obs = find_order_blocks(ohlc6, anchor="pivot_swing", zone="body", validate=("fvg",))

    # As a precomputed indicator the strategy queries per bar:
    self.obs = self.add_indicator(SmcOrderBlock.refs(self.ohlc),
                                  SmcOrderBlock(anchor="pivot_swing"))
    blk = self.obs.at_span(pivot_ts, closed_ts, kind="bearish")
"""

from __future__ import annotations

from ._core import (BEARISH, BREAKER, BULLISH, ORDER, PROPULSION, REJECTION,
                   VACUUM, Block, Pivot)
from .indicators import SmcOrderBlock, blocks_to_rects, pack_active_bands
from .order_block import find_order_blocks

__all__ = [
    "Block", "Pivot",
    "BULLISH", "BEARISH",
    "ORDER", "BREAKER", "REJECTION", "PROPULSION", "VACUUM",
    "find_order_blocks",
    "SmcOrderBlock", "blocks_to_rects", "pack_active_bands",
]

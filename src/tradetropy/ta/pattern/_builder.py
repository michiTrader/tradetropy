from __future__ import annotations

from typing import Callable, Iterator

from tradetropy.ta.pattern.pivot_mixin import PivotIndicatorMixin


def iter_pattern_matcher_defs(
    strategy,
    get_ind_def_fn: Callable,
) -> Iterator[tuple]:
    """
    Iterates over validated pattern matcher definitions.

    Yields (pm_def, base_indicator, ohlc_proxy, symbol, base_cols,
            decorator_info, tag_decoders)
    """
    for pm_def in strategy._pattern_matcher_defs:
        base_ind_def = get_ind_def_fn(pm_def.base_pivot)
        if base_ind_def is None:
            continue

        base_indicator = base_ind_def["indicator"]
        if not isinstance(base_indicator, PivotIndicatorMixin):
            continue

        ohlc_proxy = base_ind_def.get("ohlc_proxy")
        if ohlc_proxy is None:
            continue

        symbol    = ohlc_proxy.symbol
        base_cols = base_indicator.pivot_col_names(symbol)

        decorator_info: dict[str, str] = {}
        tag_decoders: dict[str, dict[float, str]] = {}
        for dec_proxy in pm_def.decorators:
            dec_def = get_ind_def_fn(dec_proxy)
            if dec_def is None:
                continue
            dec_indicator = dec_def["indicator"]
            if not isinstance(dec_indicator, PivotIndicatorMixin):
                continue
            dec_cols = dec_indicator.pivot_col_names(symbol)
            if dec_cols:
                tag_name = dec_indicator.tag_name
                decorator_info[tag_name] = dec_cols[0]
                decoder = getattr(dec_indicator, "TAG_DECODE", None)
                if decoder is not None:
                    tag_decoders[tag_name] = decoder

        yield pm_def, base_indicator, ohlc_proxy, symbol, base_cols, decorator_info, tag_decoders

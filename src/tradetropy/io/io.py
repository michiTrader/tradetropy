"""
Compatibility shim: ``tradetropy.io.io`` re-exports every symbol from the
split submodules (_backends, _common, _ticks, _klines, _book, _mbo, _record,
_compat) so existing ``from tradetropy.io.io import <symbol>`` imports (used
internally and by tests) keep working unchanged.

The real implementations live in the focused submodules below; this module
is intentionally just a re-export surface, not a place to add new code.
"""

from __future__ import annotations

from typing import Literal

# Generic format/backend helpers.
from tradetropy.io._backends import (
    _Format,
    _require_tables,
    _ensure_parent_dir,
    _write_hdf5_attrs,
    _read_hdf5_attrs,
    _write_npz,
    _read_npz,
    _read_npz_attrs,
    _read_attrs,
    _detect_format,
    _resolve_write_format,
    _read,
    _write,
)

# Shared source-normalization / timestamp helpers.
from tradetropy.io._common import (
    _TsFormat,
    _source_to_df_ticks,
    _source_to_df_klines,
    _ts_to_datetime,
    _datetime_to_ts_ms,
)

# Ticks (save/read) and raw trades.
from tradetropy.io._ticks import (
    save_ticks,
    read_ticks,
    read_trades,
    _MT5_TICK_HEADER_COLS,
    _MT5_FLAG_BUY,
    _MT5_FLAG_SELL,
    _sniff_mt5_tick_header,
    _mt5_aggressor_flags,
    _read_mt5_ticks_csv,
    _binance_is_buyer_maker_to_flags,
    _TRADE_SCHEMAS,
)

# Klines (save/read).
from tradetropy.io._klines import (
    save_klines,
    read_klines,
)

# Order book (save/read/convert) and on-disk layout helpers.
from tradetropy.io._book import (
    _Layout,
    save_book,
    read_book,
    convert_book,
    _source_to_df_book,
    _detect_book_layout,
    _normalize_long_book_df,
)

# L3 / market-by-order.
from tradetropy.io._mbo import read_mbo

# Live-recording append helpers (record= path).
from tradetropy.io._record import (
    _npz_record_sidecars,
    _append_record_npz,
    _consolidate_npz_record,
    _append_ticks_hdf5,
    _append_klines_hdf5,
    _append_book_hdf5,
    _append_mbo_hdf5,
    _append_ticks,
    _append_klines,
    _append_book,
    _append_mbo,
)

# Backward-compatible aliases.
from tradetropy.io._compat import (
    ticks_to_file,
    klines_to_file,
    ticks_from_file,
    klines_from_file,
    read_klines_csv,
    read_ticks_csv,
    book_to_file,
    book_from_file,
    save_proxy,
)

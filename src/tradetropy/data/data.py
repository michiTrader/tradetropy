# Facade - re-exports everything from the submodules.
# Existing imports (from tradetropy.data.data import X) keep working.
from tradetropy.data._klines import (
    build_candles_from_ticks,
    normalize_ticks,
    ticks_to_klines,
    resample_klines,
    validate_continuity,
)
from tradetropy.data._store import OhlcDataStore, TickDataStore
from tradetropy.data._ring import LiveRingBuffer, LiveOhlcRing, LiveBookRing, MboRing
from tradetropy.data._views import WindowView, OhlcIndicatorView, MultiOhlcIndicatorView
from tradetropy.data._proxy import ColumnRef, TickProxy, OhlcProxy, IndicatorProxy, MultiBandProxy, OrderbookProxy, MboProxy

__all__ = [
    "build_candles_from_ticks",
    "normalize_ticks",
    "ticks_to_klines",
    "resample_klines",
    "validate_continuity",
    "OhlcDataStore",
    "TickDataStore",
    "LiveRingBuffer",
    "LiveOhlcRing",
    "LiveBookRing",
    "MboRing",
    "WindowView",
    "OhlcIndicatorView",
    "MultiOhlcIndicatorView",
    "ColumnRef",
    "TickProxy",
    "OhlcProxy",
    "IndicatorProxy",
    "MultiBandProxy",
    "OrderbookProxy",
    "MboProxy",
]

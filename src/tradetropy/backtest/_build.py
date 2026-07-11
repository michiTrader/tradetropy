from __future__ import annotations

from collections import defaultdict

import numpy as np

from tradetropy.models.strategy import FeedType
from tradetropy.core.constants import (
    TICK_COLS,
    OHLC_COLS,
    N_TICK_COLS,
    N_OHLC_COLS,
    _TICK_COL,
    _OHLC_COL,
)
from tradetropy.data.data import (
    TickProxy,
    OhlcProxy,
    IndicatorProxy,
    ColumnRef,
    WindowView,
    OhlcDataStore,
    TickDataStore,
    OhlcIndicatorView,
    MultiOhlcIndicatorView,
    MultiBandProxy,
    build_candles_from_ticks,
)
from tradetropy.ta.base import Indicator
from tradetropy.models.strategy import Strategy
from tradetropy.exceptions import ConfigError


# HELPER: BUILD CANDLES FROM KLINES

def _build_candles_from_klines(klines: np.ndarray, interval_ms: int) -> dict:
    '''
    Build OHLC candles from klines data.

    Constructs a dict for OhlcDataStore from klines [N x >= 6].

    When interval_ms matches the data interval, each kline produces
    exactly one candle and its high/low are preserved.

    When aggregating (e.g. 1m data -> 5m candles), OHLC values are
    correctly propagated:
        open = open of first kline in group
        high = max(highs of group)
        low = min(lows of group)
        close = close of last kline in group
        volume = sum(volumes of group)

    Args:
        klines (np.ndarray): Input data [N x >= 6]
        interval_ms (int): Target interval in milliseconds

    Returns:
        dict: Contains closed_candles, tick_to_candle_map, accumulated_per_tick,
            candle_ts_per_tick, prices

    Note:
        Expected columns: ts(0), open(1), high(2), low(3), close(4), vol(5).
    '''
    from tradetropy.core.constants import N_OHLC_COLS, _OHLC_COL

    interval_ms = int(interval_ms)
    ts_col = klines[:, 0]
    open_col = klines[:, 1]
    high_col = klines[:, 2]
    low_col = klines[:, 3]
    close_col = klines[:, 4]
    volume_col = klines[:, 5]
    n = len(klines)

    # ── Step 1: assign each kline to its target candle ──────────────────────
    candle_ts_per_kline = (ts_col // interval_ms) * interval_ms
    is_new_candle = np.r_[True, candle_ts_per_kline[1:] != candle_ts_per_kline[:-1]]
    start_indices = np.where(is_new_candle)[0]
    end_indices = np.r_[start_indices[1:], n]
    count_per_candle = np.diff(np.r_[start_indices, n])

    n_total_candles = len(start_indices)
    kline_to_candle_map = np.repeat(np.arange(n_total_candles), count_per_candle)

    # ── Step 2: OHLC accumulators per kline (for partial candle in O(1)) ─────────
    # open per kline = open of the first kline in its group
    open_per_kline = np.repeat(open_col[start_indices], count_per_candle)

    # high accumulated up to each kline within its group
    high_acum = high_col.copy()
    low_acum = low_col.copy()
    for s, e in zip(start_indices, end_indices):
        np.maximum.accumulate(high_acum[s:e], out=high_acum[s:e])
        np.minimum.accumulate(low_acum[s:e], out=low_acum[s:e])

    # volume accumulated restarting at each group
    cumsum_vol = np.cumsum(volume_col)
    vol_at_start = cumsum_vol[start_indices] - volume_col[start_indices]
    vol_acum = cumsum_vol - np.repeat(vol_at_start, count_per_candle)

    # accumulated_per_kline[i] = [open, high, low, vol] up to kline i within its candle
    accumulated_per_kline = np.column_stack(
        [open_per_kline, high_acum, low_acum, vol_acum]
    )

    # ── Step 3: closed candles (all but the last) ────────────────────────
    close_indices = end_indices - 1
    n_closed_candles = n_total_candles - 1

    if n_closed_candles > 0:
        idx_c = close_indices[:n_closed_candles]
        closed_candles = np.column_stack(
            [
                candle_ts_per_kline[idx_c],  # ts open
                accumulated_per_kline[idx_c, 0],  # open
                accumulated_per_kline[idx_c, 1],  # high
                accumulated_per_kline[idx_c, 2],  # low
                close_col[idx_c],  # close (last price of group)
                accumulated_per_kline[idx_c, 3],  # volume
            ]
        ).astype(np.float64)
    else:
        closed_candles = np.empty((0, N_OHLC_COLS), dtype=np.float64)

    return {
        "closed_candles": closed_candles,
        "tick_to_candle_map": kline_to_candle_map,
        "accumulated_per_tick": accumulated_per_kline.astype(np.float64),
        "candle_ts_per_tick": candle_ts_per_kline,
        "prices": close_col,
    }


# MIXIN: BUILDING DATA STORES

class _StoreBuilderMixin:
    '''Shared methods for constructing data stores across engines.'''

    strategy: Strategy
    _tick_stores: dict
    _ohlc_stores: dict

    def _build_tick_store(
        self, symbol: str, tick_matrix: np.ndarray
    ) -> TickDataStore:
        '''
        Build TickDataStore for a specific symbol.

        Filters indicator definitions by exact symbol (multi-symbol fix).
        Supports multi-source indicators (list[ColumnRef]) and multi-band
        (n_outputs > 1) patterns.

        Args:
            symbol (str): Trading symbol
            tick_matrix (np.ndarray): Tick data matrix

        Returns:
            TickDataStore: Store with computed indicators
        '''
        col_index: dict[str, int] = dict(zip(TICK_COLS, range(N_TICK_COLS)))
        cols_ind: list[np.ndarray] = []

        for defn in self.strategy._indicator_defs:
            source = defn["source"]
            if not isinstance(source.proxy, TickProxy):
                continue
            if source.symbol != symbol:
                continue

            indicator   = defn["indicator"]
            sources     = defn.get("sources", [source])
            multi_source = defn.get("multi_source", False)
            multi_band   = defn.get("multi_band", False)

            # Build source array
            if multi_source:
                source_arr = np.column_stack([
                    tick_matrix[:, _TICK_COL[f._col_name]] for f in sources
                ])
            else:
                src_col = _TICK_COL[source._col_name]
                source_arr = tick_matrix[:, src_col]

            result = indicator.calculate(source_arr)

            if multi_band:
                K = indicator.n_outputs
                col_names = []
                for k in range(K):
                    band_col_name = f"{indicator.col_name(symbol, source._col_name)}_b{k}"
                    col_index[band_col_name] = N_TICK_COLS + len(cols_ind)
                    col_names.append(band_col_name)
                    cols_ind.append(result[k])
                defn["col_names"] = col_names
                defn["col_name"]  = col_names[0]
                defn["en_tick_store"] = True
                defn["tick_symbol"]   = symbol
            else:
                col_name = indicator.col_name(symbol, source._col_name)
                col_index[col_name] = N_TICK_COLS + len(cols_ind)
                defn["col_name"] = col_name
                defn["en_tick_store"] = True
                defn["tick_symbol"]   = symbol
                cols_ind.append(result.ravel() if result.ndim > 1 else result)

        if cols_ind:
            matrix = np.ascontiguousarray(
                np.hstack([tick_matrix, np.column_stack(cols_ind)])
            )
        else:
            matrix = np.ascontiguousarray(tick_matrix)

        return TickDataStore(matrix, col_index)

    def _build_ohlc_stores(self, data: dict, feed_type: FeedType = "tick") -> dict:
        '''
        Build OhlcDataStores for all OhlcProxy instances.

        Args:
            data (dict): Symbol -> data matrix mapping
            feed_type (FeedType): 'tick' or 'kline'

        Returns:
            dict: id(proxy) -> OhlcDataStore mapping

        Note:
            In tick mode: data[symbol] are ticks [N x 7], aggregated by interval.
            In kline mode: data[symbol] are klines [N x 7], interval from KlineData.
        '''
        ohlc_stores = {}

        # Group indicator_defs by source proxy -- O(n_indicators) once.
        # Avoids the O(proxies * indicators) loop that existed before.
        inds_by_proxy: dict[int, list] = defaultdict(list)
        for defn in self.strategy._indicator_defs:
            source = defn["source"]
            if isinstance(source.proxy, OhlcProxy):
                inds_by_proxy[id(source.proxy)].append(defn)

        for ohlc_proxy in self.strategy._ohlc_proxies:
            symbol = ohlc_proxy.symbol
            interval = ohlc_proxy.interval_ms
            arr = data[symbol].astype(np.float64)

            if feed_type == "tick":
                timestamps = arr[:, _TICK_COL["ts"]]
                prices = arr[:, _TICK_COL["price"]]
                volumes = arr[:, _TICK_COL["volume"]]
                candle_result = build_candles_from_ticks(
                    timestamps, prices, volumes, interval
                )
            else:
                # kline mode: treat each row as a "tick" with price=close
                candle_result = _build_candles_from_klines(arr, interval)

            closed_candles = candle_result["closed_candles"]
            tick_to_candle_map = candle_result["tick_to_candle_map"]
            accumulated_per_tick = candle_result["accumulated_per_tick"]
            candle_ts_per_tick = candle_result["candle_ts_per_tick"]
            prices_per_row = candle_result["prices"]

            col_index: dict[str, int] = dict(zip(OHLC_COLS, range(N_OHLC_COLS)))
            cols_ind: list[np.ndarray] = []

            # Only indicators from this proxy -- O(n_inds_of_this_proxy).
            # Dedup identical indicators (same class+params on the same source
            # column): compute once and let every duplicate share the column,
            # instead of recomputing and appending an identical column. Keyed by
            # (col_name, source cols, multi_band); col_name encodes
            # class+length+symbol and the source cols disambiguate same-name
            # indicators fed different inputs.
            _seen_ohlc: dict = {}
            for defn in inds_by_proxy[id(ohlc_proxy)]:
                source       = defn["source"]
                sources      = defn.get("sources", [source])
                indicator    = defn["indicator"]
                multi_source = defn.get("multi_source", False)
                multi_band   = defn.get("multi_band", False)

                _base_name = indicator.col_name(symbol, source._col_name)
                _src_key = (
                    tuple(_OHLC_COL[f._col_name] for f in sources)
                    if multi_source
                    else (_OHLC_COL[source._col_name],)
                )
                _dedup_key = (_base_name, _src_key, multi_band)
                _prev = _seen_ohlc.get(_dedup_key)
                if _prev is not None:
                    defn["col_name"]      = _prev["col_name"]
                    defn["en_tick_store"] = False
                    defn["ohlc_proxy"]    = ohlc_proxy
                    defn["src_col_idx"]   = _prev["src_col_idx"]
                    if multi_band:
                        defn["col_names"] = _prev["col_names"]
                    continue

                # Build source array
                if multi_source:
                    if len(closed_candles) > 0:
                        source_arr = np.column_stack([
                            closed_candles[:, _OHLC_COL[f._col_name]] for f in sources
                        ])
                    else:
                        source_arr = np.empty((0, len(sources)), dtype=np.float64)
                else:
                    src_col = _OHLC_COL[source._col_name]
                    source_arr = (
                        closed_candles[:, src_col]
                        if len(closed_candles) > 0
                        else np.array([], dtype=np.float64)
                    )

                # Single full-history pass (precomputed once for all closed
                # bars). Backtest/replay parity for recursive indicators (RSI,
                # MACD) is achieved by feeding the replay the same candle
                # history via warmup (it reconstructs a full window_size of
                # candles), not by windowing the backtest precompute. This keeps
                # the backtest at its original O(N) precompute speed.
                result = indicator.calculate(source_arr)

                if multi_band:
                    K = indicator.n_outputs
                    # region XXX debug
                    n_real_bands = result.shape[0] if result.ndim > 1 else 1
                    if K != n_real_bands:
                        raise ConfigError(
                            f"Indicator '{type(indicator).__name__}': n_outputs={K} but "
                            f"calculate returned {n_real_bands} rows. "
                            f"Check that n_outputs matches the calculate output shape."
                        )
                    #endregion 
                    col_names = []
                    src_idxs  = (
                        [_OHLC_COL[f._col_name] for f in sources]
                        if multi_source
                        else _OHLC_COL[source._col_name]
                    )
                    for k in range(K):
                        cname = f"{indicator.col_name(symbol, source._col_name)}_b{k}"
                        col_index[cname] = N_OHLC_COLS + len(cols_ind)
                        col_names.append(cname)
                        band_data = result[k] if len(result.shape) > 1 else result
                        cols_ind.append(band_data)
                    defn["col_names"]   = col_names
                    defn["col_name"]    = col_names[0]
                    defn["en_tick_store"] = False
                    defn["ohlc_proxy"]    = ohlc_proxy
                    defn["src_col_idx"]   = src_idxs
                    _seen_ohlc[_dedup_key] = {
                        "col_name": col_names[0],
                        "col_names": col_names,
                        "src_col_idx": src_idxs,
                    }
                else:
                    col_name = indicator.col_name(symbol, source._col_name)
                    col_index[col_name] = N_OHLC_COLS + len(cols_ind)
                    defn["col_name"]    = col_name
                    defn["en_tick_store"] = False
                    defn["ohlc_proxy"]    = ohlc_proxy
                    defn["src_col_idx"]   = (
                        [_OHLC_COL[f._col_name] for f in sources]
                        if multi_source
                        else _OHLC_COL[source._col_name]
                    )
                    cols_ind.append(result.ravel() if result.ndim > 1 else result)
                    _seen_ohlc[_dedup_key] = {
                        "col_name": col_name,
                        "src_col_idx": defn["src_col_idx"],
                    }

            if len(closed_candles) > 0 and cols_ind:
                ohlc_matrix = np.ascontiguousarray(
                    np.hstack([closed_candles, np.column_stack(cols_ind)])
                )
            elif len(closed_candles) > 0:
                ohlc_matrix = np.ascontiguousarray(closed_candles)
            else:
                n_cols = N_OHLC_COLS + len(cols_ind)
                ohlc_matrix = np.empty((0, n_cols), dtype=np.float64)

            ohlc_stores[id(ohlc_proxy)] = OhlcDataStore(
                matrix=ohlc_matrix,
                col_index=col_index,
                tick_to_candle_mapping=tick_to_candle_map,
                accumulated_by_tick=accumulated_per_tick,
                ts_candle_by_tick=candle_ts_per_tick,
                prices_per_tick=prices_per_row,
                interval_ms=interval,
                symbol=symbol,
                kline_mode=(feed_type == "kline"),
            )

        return ohlc_stores

    def _connect_proxies(self, tick_stores: dict, ohlc_stores: dict):
        '''Connect each proxy to its data store.'''
        for tp in self.strategy._tick_proxies:
            store = tick_stores.get(tp.symbol)
            if store is not None:
                tp._connect_backtest(store)

        for op in self.strategy._ohlc_proxies:
            op._connect_backtest(ohlc_stores[id(op)])

        for defn in self.strategy._indicator_defs:
            proxy: "IndicatorProxy | MultiBandProxy" = defn["proxy"]
            indicator: Indicator = defn["indicator"]
            source: ColumnRef = defn["source"]
            multi_band: bool = defn.get("multi_band", False)

            if defn.get("en_tick_store", True) and isinstance(source.proxy, TickProxy):
                # -- Tick store ------------------------------------------------
                tick_store = tick_stores[defn["tick_symbol"]]
                size = source.proxy._window_size

                if multi_band:
                    col_names = defn["col_names"]
                    for k, cname in enumerate(col_names):
                        col_idx = tick_store.col_index[cname]
                        view = WindowView(col_idx=col_idx, size=size, tick_store=tick_store)
                        proxy._connect_band(k, view)
                else:
                    col_idx = tick_store.col_index[defn["col_name"]]
                    view = WindowView(col_idx=col_idx, size=size, tick_store=tick_store)
                    proxy._connect(view)
            else:
                # -- OHLC store ------------------------------------------------
                ohlc_proxy: OhlcProxy = defn["ohlc_proxy"]
                ohlc_store: OhlcDataStore = ohlc_stores[id(ohlc_proxy)]
                src_col_idx = defn["src_col_idx"]
                size = ohlc_proxy._window_size

                if multi_band:
                    col_names    = defn["col_names"]
                    ind_col_idxs = [ohlc_store.col_index[n] for n in col_names]
                    shared_cache = {}
                    for k in range(len(col_names)):
                        view = MultiOhlcIndicatorView(
                            ohlc_store=ohlc_store,
                            indicator=indicator,
                            ind_col_idxs=ind_col_idxs,
                            src_col_idx=src_col_idx,
                            band_idx=k,
                            size=size,
                            shared_cache=shared_cache,
                        )
                        proxy._connect_band(k, view)
                else:
                    ind_col_idx = ohlc_store.col_index[defn["col_name"]]
                    view = OhlcIndicatorView(
                        ohlc_store=ohlc_store,
                        indicator=indicator,
                        ind_col_idx=ind_col_idx,
                        src_col_idx=src_col_idx,
                        size=size,
                    )
                    proxy._connect(view)
                    # Wire the single-frame scalar read fast path for the common
                    # kline-mode read (self.ind[-1] / [-k]); non-kline (tick)
                    # backtests keep the view path. See IndicatorProxy.
                    if ohlc_store.kline_mode and isinstance(src_col_idx, int):
                        proxy._enable_flat_kline(
                            ohlc_store, ind_col_idx, size,
                            getattr(indicator, "warmup_factor", 1) == 1,
                        )

    def _get_ind_def_for_proxy(self, proxy) -> "dict | None":
        '''Find the indicator definition corresponding to a proxy.'''
        for defn in self.strategy._indicator_defs:
            if defn["proxy"] is proxy:
                return defn
        return None

    def _build_pattern_stores(self, ohlc_stores: dict) -> None:
        '''Build pattern stores from OHLC data.'''
        # No patterns -> nothing to build. Guarding here also avoids importing
        # the pattern DSL, which is absent from lower-tier (gated) builds.
        if not getattr(self.strategy, "_pattern_matcher_defs", None):
            return
        from tradetropy.ta.pattern._builder import iter_pattern_matcher_defs
        from tradetropy.ta.pattern.sequence import FrozenPivotSequence
        from tradetropy.ta.pattern.store    import PatternStore

        for pm_def, _, ohlc_proxy, symbol, base_cols, decorator_cols, tag_decoders \
            in iter_pattern_matcher_defs(self.strategy, self._get_ind_def_for_proxy):

            ohlc_store = ohlc_stores[id(ohlc_proxy)]
            sequence = FrozenPivotSequence.from_ohlc_store(
                ohlc_store     = ohlc_store,
                base_col_names = base_cols,
                decorator_cols = decorator_cols,
                tag_decoders   = tag_decoders,
            )
            store = PatternStore(sequence, pm_def.pattern)
            pm_def.proxy._connect_backtest(store, ohlc_store=ohlc_store)

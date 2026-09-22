'''
Internal implementation of the backtesting loop (ticks and klines).

All functions receive self (BacktestEngine) as the first argument
and are assigned as methods at the end of engine.py.
'''

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import warnings

import numpy as np

from tradetropy.core.constants import (
    N_TICK_COLS,
    N_OHLC_COLS,
    _TICK_COL,
)
from tradetropy.data.data import TickProxy
from tradetropy.exceptions import StopEngine
import multiprocessing as _mp
from tradetropy.models._warmup_policy import (
    resolve_warmup, estimate_ticks_per_candle, log_warmup,
    auto_warmup_candles, debug_warmup_block,
)


# Setup

def _setup_strategy(self, verbose: bool, save_log: bool = False):
    '''Connect broker/session to strategy and call init().'''
    self.strategy._broker = self.broker
    self.strategy._sesh = self._sesh
    self.strategy._feed_type = self._feed_type
    self.strategy._verbose = verbose
    self.strategy._save_log = save_log
    self.strategy._set_run_mode("backtest")
    self.strategy.init()


# Store construction

def _build_tick_stores(self, data: dict) -> tuple[dict, dict, dict[str, np.ndarray]]:
    '''Build TickDataStores and OhlcDataStores in tick mode.'''
    tick_matrices: dict[str, np.ndarray] = {
        sym: arr[:, :N_TICK_COLS].astype(np.float64) for sym, arr in data.items()
    }
    symbols_with_tick_proxy = {tp.symbol for tp in self.strategy._tick_proxies}
    symbols_with_indicator_tick = {
        defn["source"].symbol
        for defn in self.strategy._indicator_defs
        if isinstance(defn["source"].proxy, TickProxy)
    }
    symbols_with_tick = symbols_with_tick_proxy | symbols_with_indicator_tick or set(tick_matrices.keys())

    tick_stores = {
        sym: self._build_tick_store(sym, tick_matrices[sym])
        for sym in symbols_with_tick if sym in tick_matrices
    }
    ohlc_stores = self._build_ohlc_stores(
        {sym: arr[:, :N_TICK_COLS] for sym, arr in data.items()},
        feed_type="tick",
    )
    self._connect_proxies(tick_stores, ohlc_stores)
    self._tick_stores = tick_stores
    self._ohlc_stores = ohlc_stores

    if self.strategy._fp_proxies:
        from tradetropy.models.footprint import build_fp_stores_for_strategy
        build_fp_stores_for_strategy(
            self.strategy,
            {sym: arr[:, :N_TICK_COLS] for sym, arr in data.items()},
            ohlc_stores,
            _TICK_COL,
        )

    self._build_pattern_stores(ohlc_stores)
    return tick_matrices, tick_stores, ohlc_stores


# Tick loop

def _build_book_replay(self, tick_matrices: dict) -> None:
    """
    Build the L2 book rings for the strategy and prepare the book replayer.

    Creates one LiveBookRing per subscribed OrderbookProxy (so book_as_of /
    DeepTrades work in the backtest), attaches it to the proxy, and - when a
    recorded book was passed via by_ticks(book=) - runs the sync preflight to
    warn on desync and (with sync_book=True) shift a recoverable clock offset.
    No book provided leaves the rings empty/stale, matching the classic
    no-book backtest (book_as_of returns None).

    Args:
        tick_matrices (dict): {symbol: tick matrix} being replayed.
    """
    from tradetropy.data._ring import LiveBookRing
    from tradetropy.data._book_replay import BookReplayer, resolve_book_sync

    books = getattr(self, "_book_inputs", {}) or {}
    self._book_rings = {}
    self._book_sync_reports = {}
    if not books:
        self._book_replayer = BookReplayer({})
        return

    for bp in getattr(self.strategy, "_book_proxies", []):
        ring = LiveBookRing(window_size=bp.window_size, levels=bp.depth)
        bp._book_ring = ring
        self._book_rings.setdefault(bp.symbol, []).append(ring)

    self._book_replayer = BookReplayer(books)
    trades = {
        sym: (mat[:, _TICK_COL["ts"]], mat[:, _TICK_COL["price"]])
        for sym, mat in tick_matrices.items()
    }
    reports = resolve_book_sync(
        books, trades, sync_book=getattr(self, "_sync_book", False),
        engine_label=type(self).__name__,
    )
    for sym, (_rep, off) in reports.items():
        if off:
            self._book_replayer.set_offset(sym, off)
    self._book_sync_reports = {s: r for s, (r, _o) in reports.items()}

    # Fully populate the rings now so the one-shot indicator precompute sees the
    # whole book (book_as_of stays causal by ts inside calculate). The loop
    # resets and re-drains them incrementally for direct book access.
    for sym, rings in self._book_rings.items():
        self._book_replayer.drain_to(sym, float("inf"), rings)


def _reset_books_for_loop(self) -> None:
    """
    Clear the book rings and rewind the replayer after the indicator precompute.

    The precompute fully populated the rings; the tick loop must replay the book
    incrementally so any DIRECT book read in on_data() (imbalance, mid, ...) is
    causal to the current tick. No-op when no book is attached.
    """
    if self._book_replayer is None or not self._book_rings:
        return
    for rings in self._book_rings.values():
        for r in rings:
            r.reset()
    self._book_replayer.reset()


def _drain_books_at(self, tick_matrices: dict, idx) -> None:
    """
    Drain each symbol's recorded book up to that symbol's current tick ts.

    Applies book rows causally (book_ts + offset <= tick_ts) so book_as_of
    sees the same images the live/replay path would at each trade. No-op when
    no book is attached.

    Args:
        tick_matrices (dict): {symbol: tick matrix}.
        idx (int | dict): Shared row index (fast path) or a {symbol: index}
            map (merge path, where each symbol advances on its own timeline).
    """
    if self._book_replayer is None or not self._book_rings:
        return
    for sym, rings in self._book_rings.items():
        mat = tick_matrices.get(sym)
        if mat is None:
            continue
        row_idx = idx[sym] if isinstance(idx, dict) else idx
        if row_idx is None or row_idx < 0 or row_idx >= len(mat):
            continue
        sym_ts = int(mat[row_idx, _TICK_COL["ts"]])
        self._book_replayer.drain_to(sym, sym_ts, rings)


def _run_ticks(self, data: dict, verbose: bool = True, save_log: bool = False):
    '''Entry point for backtest in tick mode.'''
    self._setup_strategy(verbose, save_log)
    self._validate_symbols(data)
    # Build + fully populate the book rings BEFORE the indicator precompute so
    # DeepTrades / L2 indicators (computed once over the whole series) read a
    # causal (as-of by ts) book. The rings are reset afterwards so the tick loop
    # re-drains them incrementally for causal DIRECT book access in on_data().
    self._build_book_replay(data)
    tick_matrices, _, _ = self._build_tick_stores(data)
    self._reset_books_for_loop()

    tick_proxies    = self.strategy._tick_proxies
    ohlc_proxies    = self.strategy._ohlc_proxies
    ind_proxies     = [d["proxy"] for d in self.strategy._indicator_defs]
    fp_proxies      = self.strategy._fp_proxies
    pattern_proxies = [d.proxy for d in self.strategy._pattern_matcher_defs]
    lengths = {sym: len(arr) for sym, arr in tick_matrices.items()}

    if self._align_by_ts and len(tick_matrices) > 1:
        self._run_ticks_merge_path(
            tick_matrices, tick_proxies, ohlc_proxies, ind_proxies,
            fp_proxies, pattern_proxies, lengths, verbose,
        )
    else:
        self._run_ticks_fast_path(
            tick_matrices, tick_proxies, ohlc_proxies, ind_proxies,
            fp_proxies, pattern_proxies, lengths, verbose,
        )


def _run_ticks_fast_path(
    self, tick_matrices, tick_proxies, ohlc_proxies, ind_proxies,
    fp_proxies, pattern_proxies, lengths, verbose,
):
    """Sequential loop synchronized by position (all symbols same N)."""
    N = min(lengths.values())
    if len(set(lengths.values())) > 1:
        warnings.warn(
            f"Symbols have different lengths {lengths}. "
            f"Using minimum N={N}.",
            stacklevel=2,
        )

    try:
        from tqdm import tqdm as _tqdm
        _iter = (
            range(N)
            if (_mp.current_process().name != 'MainProcess' or verbose)
            else _tqdm(range(N), desc="Backtest", unit="tick",
                       dynamic_ncols=True, leave=True)
        )
    except ImportError:
        _iter = range(N)

    _tpv = estimate_ticks_per_candle(self.strategy, tick_matrices)
    warmup = resolve_warmup(self.strategy, feed_type="tick", ticks_per_candle=_tpv)
    _auto = auto_warmup_candles(self.strategy)
    _n_ohlc_ind = sum(
        1 for d in self.strategy._indicator_defs
        if not isinstance(d["source"].proxy, TickProxy)
    )
    _sym0 = next(iter(tick_matrices))
    _ivl = self.strategy._ohlc_proxies[0].interval_ms if self.strategy._ohlc_proxies else 0
    _nvelas = len(np.unique((tick_matrices[_sym0][:, _TICK_COL["ts"]] // _ivl))) if _ivl else 0
    debug_warmup_block(
        "Backtest.by_ticks", "tick",
        total_data_points=N, warmup=warmup, n_indicators=_n_ohlc_ind,
        auto_bars=_auto, ticks_per_candle=_tpv, n_candles_in_dataset=_nvelas,
        extra={"on_data runs from tick": warmup},
    )
    for idx in _iter:
        for tp in tick_proxies:
            tp._advance(idx)
        for op in ohlc_proxies:
            op._advance(idx)
        for ip in ind_proxies:
            ip._advance(idx)
        for pm in pattern_proxies:
            pm._advance(idx)
        for fp in fp_proxies:
            fp._advance(idx)
            row = tick_matrices[fp.symbol][idx]
            fp.process_tick(
                timestamp_ms=int(row[_TICK_COL["ts"]]),
                price=float(row[_TICK_COL["price"]]),
                vol=float(row[_TICK_COL[fp.config.vol_col]]),
                bid=float(row[_TICK_COL["bid"]]),
                ask=float(row[_TICK_COL["ask"]]),
                flag=(
                    float(row[_TICK_COL[fp.config.aggressor_col]])
                    if fp.config.aggressor_col
                       and fp.config.aggressor_col in _TICK_COL
                    else 0.0
                ),
            )
        _primary_sym = next(iter(tick_matrices))
        _current_ts_ms = int(tick_matrices[_primary_sym][idx, _TICK_COL["ts"]])
        if self.broker is not None:
            for sym, mat in tick_matrices.items():
                row = mat[idx]
                ts_dt = datetime.fromtimestamp(
                    float(row[_TICK_COL["ts"]]) / 1000, tz=timezone.utc,
                )
                self.broker.update_tick(
                    symbol=sym, timestamp=ts_dt,
                    bid=float(row[_TICK_COL["bid"]]),
                    ask=float(row[_TICK_COL["ask"]]),
                    volume=float(row[_TICK_COL["volume"]]),
                    flags=int(row[_TICK_COL["flags"]]),
                    volume_real=float(row[_TICK_COL["volume_real"]]),
                    price=float(row[_TICK_COL["price"]]),
                )
        if self.broker is not None and self.broker._stopped:
            break
        self._sesh._ultimo_ts = _current_ts_ms
        self._drain_books_at(tick_matrices, idx)
        if idx >= warmup:
            try:
                self.strategy._in_on_data = True
                self.strategy.on_data()
            except StopEngine:
                pass
            finally:
                self.strategy._in_on_data = False


def _run_ticks_merge_path(
    self, tick_matrices, tick_proxies, ohlc_proxies, ind_proxies,
    fp_proxies, pattern_proxies, lengths, verbose,
):
    """
    Loop aligned by timestamp (merge of timelines from different symbols).

    Algorithm:
    1. Extract all unique timestamps across all symbols and sort chronologically.
    2. For each symbol, use np.searchsorted(side="right")-1 to map timeline
       timestamps to the symbol's actual row indices. This creates a forward-fill
       mapping: each timeline position maps to the most recent row with ts <= current_ts.
    3. Iterate over the unified timeline. For each timestamp, advance only the
       symbols that have a new tick (where searchsorted index differs from last).
       This eliminates look-ahead bias: no symbol is read before its natural time.
    4. Call on_data() only if at least one symbol produced a new tick (_alguno_nuevo).
    """
    warnings.warn(
        f"align_by_ts=True: aligning {list(tick_matrices.keys())} "
        f"by timestamp. Lengths: {lengths}.",
        stacklevel=2,
    )

    _all_ts = np.unique(np.concatenate([
        mat[:, _TICK_COL["ts"]] for mat in tick_matrices.values()
    ]))
    _all_ts.sort()
    M = len(_all_ts)

    _sym_idx_at: dict[str, np.ndarray] = {}
    for sym, mat in tick_matrices.items():
        _sym_idx_at[sym] = np.searchsorted(
            mat[:, _TICK_COL["ts"]], _all_ts, side="right",
        ) - 1

    _last_idx: dict[str, int] = {sym: -1 for sym in tick_matrices}

    _tp_por_sym: dict[str, list] = defaultdict(list)
    for tp in tick_proxies:
        _tp_por_sym[tp.symbol].append(tp)

    _op_por_sym: dict[str, list] = defaultdict(list)
    for op in ohlc_proxies:
        _op_por_sym[op.symbol].append(op)

    _ip_por_sym: dict[str, list] = defaultdict(list)
    for defn in self.strategy._indicator_defs:
        _ip_por_sym[defn["source"].symbol].append(defn["proxy"])

    _ip_global = [
        defn["proxy"]
        for defn in self.strategy._indicator_defs
        if len(defn.get("sources", [])) > 1
           and len({f.symbol for f in defn["sources"]}) > 1
    ]

    _fp_por_sym: dict[str, list] = defaultdict(list)
    for fp in fp_proxies:
        _fp_por_sym[fp.symbol].append(fp)

    warmup = resolve_warmup(
        self.strategy, feed_type="tick",
        ticks_per_candle=estimate_ticks_per_candle(self.strategy, tick_matrices),
    )
    log_warmup(self.strategy, "tick", warmup, "Backtest.by_ticks(merge)")

    try:
        from tqdm import tqdm as _tqdm
        _iter = (
            range(M)
            if (_mp.current_process().name != 'MainProcess' or verbose)
            else _tqdm(range(M), desc="Backtest", unit="tick",
                       dynamic_ncols=True, leave=True)
        )
    except ImportError:
        _iter = range(M)

    for t_pos in _iter:
        _current_ts_ms = int(_all_ts[t_pos])
        _alguno_nuevo = False
        for sym, mat in tick_matrices.items():
            real_idx = int(_sym_idx_at[sym][t_pos])
            if real_idx < 0 or real_idx == _last_idx[sym]:
                continue
            _last_idx[sym] = real_idx
            _alguno_nuevo = True
            for tp in _tp_por_sym[sym]:
                tp._advance(real_idx)
            for op in _op_por_sym[sym]:
                op._advance(real_idx)
            for ip in _ip_por_sym[sym]:
                ip._advance(real_idx)
            for fp in _fp_por_sym[sym]:
                fp._advance(real_idx)
                row = mat[real_idx]
                fp.process_tick(
                    timestamp_ms=int(row[_TICK_COL["ts"]]),
                    price=float(row[_TICK_COL["price"]]),
                    vol=float(row[_TICK_COL[fp.config.vol_col]]),
                    bid=float(row[_TICK_COL["bid"]]),
                    ask=float(row[_TICK_COL["ask"]]),
                    flag=(
                        float(row[_TICK_COL[fp.config.aggressor_col]])
                        if fp.config.aggressor_col
                           and fp.config.aggressor_col in _TICK_COL
                        else 0.0
                    ),
                )
            if self.broker is not None:
                row = mat[real_idx]
                ts_dt = datetime.fromtimestamp(
                    float(_current_ts_ms) / 1000, tz=timezone.utc,
                )
                self.broker.update_tick(
                    symbol=sym, timestamp=ts_dt,
                    bid=float(row[_TICK_COL["bid"]]),
                    ask=float(row[_TICK_COL["ask"]]),
                    volume=float(row[_TICK_COL["volume"]]),
                    flags=int(row[_TICK_COL["flags"]]),
                    volume_real=float(row[_TICK_COL["volume_real"]]),
                    price=float(row[_TICK_COL["price"]]),
                )

        for ip in _ip_global:
            ip._advance(t_pos)
        for pm in pattern_proxies:
            pm._advance(t_pos)
        if self.broker is not None and self.broker._stopped:
            break
        if not _alguno_nuevo:
            continue
        self._sesh._ultimo_ts = _current_ts_ms
        self._drain_books_at(tick_matrices, {
            s: int(_sym_idx_at[s][t_pos]) for s in tick_matrices
        })
        if t_pos >= warmup:
            try:
                self.strategy._in_on_data = True
                self.strategy.on_data()
            except StopEngine:
                pass
            finally:
                self.strategy._in_on_data = False


# Klines loop

def _run_klines(self, data: dict, verbose: bool = True, save_log: bool = False):
    """Entry point for the backtest in klines mode."""
    self._setup_strategy(verbose, save_log)
    self._validate_symbols_klines(data)
    self._execute_kline_loop(data, verbose=verbose)


def _execute_kline_loop(self, data: dict, verbose: bool = True):
    """Main loop for OHLC candle data."""
    kline_matrices: dict[str, np.ndarray] = {
        sym: arr[:, :N_OHLC_COLS].astype(np.float64)
        for sym, arr in data.items()
    }
    ohlc_stores = self._build_ohlc_stores(
        {sym: arr[:, :N_OHLC_COLS] for sym, arr in data.items()},
        feed_type="kline",
    )
    self._connect_proxies(tick_stores={}, ohlc_stores=ohlc_stores)
    self._ohlc_stores = ohlc_stores
    self._build_pattern_stores(ohlc_stores)

    ohlc_proxies    = self.strategy._ohlc_proxies
    ind_proxies     = [d["proxy"] for d in self.strategy._indicator_defs]
    pattern_proxies = [d.proxy for d in self.strategy._pattern_matcher_defs]
    lengths = {sym: len(arr) for sym, arr in kline_matrices.items()}

    _interval_ms_por_sym: dict[str, int] = {}
    if self._kline_inputs:
        _interval_ms_por_sym = {
            ki.symbol: ki.interval_ms for ki in self._kline_inputs
        }
    _all_same_interval = (
        len(set(_interval_ms_por_sym.values())) <= 1
        and len(set(lengths.values())) == 1
    )

    if _all_same_interval:
        M = min(lengths.values())
        _timeline = None
        _sym_idx_at = None
    else:
        warnings.warn(
            f"Symbols with different interval_ms or lengths {lengths}. "
            f"Using merge.",
            stacklevel=3,
        )
        _all_ts = np.unique(np.concatenate([
            mat[:, 0] for mat in kline_matrices.values()
        ]))
        _all_ts.sort()
        _timeline = _all_ts
        M = len(_timeline)
        _sym_idx_at = {
            sym: np.searchsorted(mat[:, 0], _timeline, side="right") - 1
            for sym, mat in kline_matrices.items()
        }

    warmup = resolve_warmup(self.strategy)
    log_warmup(self.strategy, "kline", warmup, "Backtest.by_klines")

    try:
        from tqdm import tqdm as _tqdm
        _iter = (
            range(M)
            if (_mp.current_process().name != 'MainProcess' or verbose)
            else _tqdm(range(M), desc="Backtest", unit="bar",
                       dynamic_ncols=True, leave=True)
        )
    except ImportError:
        _iter = range(M)

    # Constant across bars - resolve once instead of every iteration.
    _primary = next(iter(kline_matrices))
    _primary_mat = kline_matrices[_primary]

    # Pre-normalize OHLC to tick_size ONCE (vectorized) for the broker, instead
    # of 4 normalize_price calls per bar inside update_kline. Indicators/stores
    # keep the RAW matrices (unchanged): only the broker ever saw normalized
    # prices. normalize_price is round(x/tick)*tick; np.round matches Python's
    # round-half-to-even, so values are identical (guarded by the broker/parity
    # suites). A per-symbol copy leaves the raw data untouched.
    _broker_mats: dict = kline_matrices
    if self.broker is not None:
        _broker_mats = {}
        for sym, mat in kline_matrices.items():
            ts = self.broker._get_symbol_config(sym).tick_size
            nm = mat.copy()
            nm[:, 1:5] = np.round(nm[:, 1:5] / ts) * ts
            _broker_mats[sym] = nm

    for idx in _iter:
        for op in ohlc_proxies:
            op._advance(idx)
        for ip in ind_proxies:
            ip._advance(idx)
        for pm in pattern_proxies:
            pm._advance(idx)

        _current_ts_ms = int(
            _primary_mat[idx, 0]
            if _all_same_interval
            else _timeline[idx]
        )

        if self.broker is not None:
            for sym, mat in _broker_mats.items():
                bar_idx = (
                    idx
                    if _all_same_interval
                    else int(_sym_idx_at[sym][idx])
                )
                if not _all_same_interval and bar_idx < 0:
                    continue
                row = mat[bar_idx]
                # Pass epoch-ms only; the broker builds the datetime lazily
                # (from _ts_ms) if and when a trade/order actually needs it, so
                # the per-bar loop skips datetime.fromtimestamp entirely.
                self.broker.update_kline(
                    symbol=sym, timestamp=None,
                    open_price=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    _prenormalized=True,
                    _ts_ms=float(row[0]),
                )

        # Out-of-money / stop-out: the broker liquidated and flagged the run to
        # end. Stop cleanly before on_data (no decision on the wiped bar),
        # mirroring backtesting.py's _OutOfMoneyError break.
        if self.broker is not None and self.broker._stopped:
            break

        self._sesh._ultimo_ts = _current_ts_ms
        if idx >= warmup:
            try:
                self.strategy._in_on_data = True
                self.strategy.on_data()
            except StopEngine:
                pass
            finally:
                self.strategy._in_on_data = False


# Stats

# Map interval_ms -> pandas freq for compute_stats.
# In klines mode the interval is known precisely.
_MS_TO_STATS_FREQ: dict[int, str] = {
    1_000:       "1s",
    5_000:       "5s",
    10_000:      "10s",
    15_000:      "15s",
    30_000:      "30s",
    60_000:      "1min",
    300_000:     "5min",
    900_000:     "15min",
    1_800_000:   "30min",
    3_600_000:   "1h",
    7_200_000:   "2h",
    14_400_000:  "4h",
    86_400_000:  "1D",
    604_800_000: "1W",
}


def _build_stats(self) -> "Stats | None":
    """Builds post-backtest metrics. Returns None if there are no trades."""
    broker = self.broker
    if broker is None:
        return None

    freq: str | None = None
    if self._kline_inputs:
        interval_ms = self._kline_inputs[0].interval_ms
        freq = _MS_TO_STATS_FREQ.get(interval_ms)

    # Fast path: pandas-free metrics for the optimize/pool worker, so the child
    # process never imports pandas (~0.5-0.6 s saved per spawned worker). Reads
    # the broker's raw equity arrays directly (never broker.equity_curve, which
    # would build a pd.Series). Numeric parity with compute_stats is asserted by
    # test_stats_fast_parity.py, so the ranked fitness matches a real run().
    if getattr(self, "_fast_stats", False):
        from tradetropy.stats._fast import compute_stats_fast
        return compute_stats_fast(
            broker._eq_ts,
            broker._eq_vals,
            broker.get_trades(),
            initial_balance=broker.initial_balance,
            freq=freq,
        )

    from tradetropy.stats import compute_stats

    # Gate on the equity curve, not on closed trades. The broker records equity
    # (balance + floating PnL of open positions) on every bar/tick, so a
    # backtest whose position never closed still has a full equity curve. This
    # keeps backtest parity with replay/live, where that open-position equity is
    # always shown. compute_stats tolerates an empty trade list (trade metrics
    # report N/A).
    eq = broker.equity_curve
    if eq.empty:
        return None

    trades = broker.get_trades()

    try:
        return compute_stats(
            equity_curve=eq,
            trades=trades,
            initial_balance=broker.initial_balance,
            strategy=self.strategy,
            freq=freq,
            warn=getattr(self, "_stats_warn", True),
        )
    except Exception as e:
        import traceback
        warnings.warn(
            f"_build_stats: compute_stats raised {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}",
            stacklevel=3,
        )
        return None

"""
================
Unified backtesting engine with explicit constructors per data type.

API
===
  BacktestEngine.by_ticks(strategy, data=(TickData(...),), sesh=sesh)
    - data: tuple of TickData - one object per symbol
    - The engine is prepared but does NOT execute yet
    - engine.run() executes the loop
    - subscribe_ticks() ok   subscribe_ohlc() ok   subscribe_footprint() ok

  BacktestEngine.by_klines(strategy, data=(KlineData(...),), sesh=sesh)
    - data: tuple of KlineData - one object per symbol, each with
      interval_ms required (e.g. 300_000 for 5m)
    - subscribe_ticks() not supported (explicit error)  subscribe_ohlc() ok
    - engine.run() executes the loop

COMPLETE FLOW
=============
    from tradetropy.data_types import TickData, KlineData
    from tradetropy.backtest import BacktestEngine

    ticks = TickData("BTCUSDT", tick_matrix, tick_step=0.01)
    bt = BacktestEngine.by_ticks(MyStrategy(), data=(ticks,), sesh=sesh)
    bt.run()
    bt.stats
    bt.plot()

PREPARATION / EXECUTION SEPARATION
====================================
The by_ticks() and by_klines() constructors only:
  1. Store the typed data
  2. Register symbols in the broker

engine.run() does all the real work:
  1. strategy.init()
  2. Validations
  3. Build stores and connect proxies
  4. Tick-by-tick or bar-by-bar loop
  5. Metrics calculation (stats)
"""

from __future__ import annotations

from typing import Tuple

from tradetropy.models.strategy import FeedType, Strategy
from tradetropy.session.base import SeshSimulatorBase
from tradetropy.backtest._build import _StoreBuilderMixin
from tradetropy.core.data_types import (
    TickData,
    KlineData,
    _validate_inputs_ticks,
    _validate_inputs_klines,
    _inputs_to_dict,
    _normalize_data,
    _normalize_book,
)
from tradetropy.exceptions import TradingError, ConfigError


class BacktestEngine(_StoreBuilderMixin):
    """
    Unified backtesting engine.

        # Ticks
        bt = BacktestEngine.by_ticks(
            strategy = MyStrategy(),
            data     = (TickData("BTCUSDT", ...),),
            sesh     = sesh,
        )
        bt.run()

        # Multi-symbol klines
        bt = BacktestEngine.by_klines(
            strategy = MyStrategy(),
            data     = (
                KlineData("BTCUSDT", ..., timeframe='5m'),
                KlineData("ETHUSDT", ..., timeframe='1m'),
            ),
            sesh     = sesh,
        )
        bt.run()

    Result:
        engine.stats       - performance metrics
        engine.strategy    - the strategy with its final state
        engine.broker      - the internal broker
        engine.sesh        - the SeshSimulatorBase provided
    """

    #: Default log file write policy (opt-in).
    #: Backtest does NOT write a file unless save_log=True is passed to run().
    _DEFAULT_SAVE_LOG: bool = False

    def __init__(
        self,
        strategy: Strategy,
        sesh: SeshSimulatorBase,
        _feed_type: FeedType = "tick",
    ):
        if not isinstance(sesh, SeshSimulatorBase):
            raise ConfigError(
                f"`sesh` must be a simulated session (SeshSimulatorBase), "
                f"received {type(sesh).__name__}. "
                "A live session cannot be used in a backtest. "
                "Use SeshMT5Sim, SeshBybitSim, SeshCCXTSim or another simulated session."
            )
        self.strategy = strategy
        self._sesh = sesh
        # Inject the feed_type into the session: builds the internal broker if the
        # session was created in automatic mode (feed_type=None) or validates that
        # the explicit feed_type matches the engine's.
        if hasattr(sesh, "_bind_feed_type"):
            sesh._bind_feed_type(_feed_type)
        self._broker = sesh._broker
        self._feed_type = _feed_type

        self._tick_stores: dict = {}
        self._ohlc_stores: dict = {}
        self._stats: "Stats | None" = None
        self._plot_config = {}

        # When True, run() finalizes metrics via the pandas-free
        # stats._fast.compute_stats_fast (optimize/pool worker hot path, so the
        # child process never imports pandas). Numeric parity with compute_stats
        # is guarded by test_stats_fast_parity.py. Default False keeps the full
        # pandas Stats object for normal run()/plot().
        self._fast_stats: bool = False

        # Typed data - stored in by_ticks/by_klines, used in run()
        self._tick_inputs: tuple[TickData, ...] = ()
        self._kline_inputs: tuple[KlineData, ...] = ()

        # Timestamp alignment in multi-symbol tick mode
        self._align_by_ts: bool = False

        # Optional recorded L2 order book replayed alongside the ticks (tick
        # mode only). Built into LiveBookRings and drained in the tick loop so
        # DeepTrades / L2 metrics work in the backtest exactly as in replay.
        self._book_inputs: dict = {}
        self._sync_book: bool = False
        self._book_rings: dict = {}
        self._book_replayer = None
        self._book_sync_reports: dict = {}

    # ==========================================================================
    # Public properties
    # ==========================================================================

    @property
    def stats(self) -> "Stats | None":
        '''
        Performance metrics calculated automatically at the end of run().

        Returns None if there were no trades or no broker is configured.

        Returns:
            Stats: Performance metrics or None

        Example:
            bt = BacktestEngine.by_ticks(strategy, data=(ticks,), sesh=sesh).run()
            print(bt.stats)
            print(bt.stats['Sharpe Ratio'])
            trades = bt.stats.trades
            equity = bt.stats.equity_curve
        '''
        return self._stats

    @property
    def broker(self) -> "TickBroker | KlineBroker":
        return self._broker

    @property
    def out_of_money(self) -> bool:
        '''
        True if the backtest ended because the account was wiped (equity <= 0)
        or a margin stop-out fired, liquidating open positions and stopping the
        run cleanly (mirrors backtesting.py's out-of-money behavior).
        '''
        return bool(getattr(self._broker, "_out_of_money", False))

    @property
    def stopped_early(self) -> bool:
        '''True if the backtest terminated early (out-of-money or stop-out).'''
        return bool(getattr(self._broker, "_stopped", False))

    @property
    def sesh(self) -> SeshSimulatorBase:
        return self._sesh

    @property
    def feed_type(self) -> FeedType:
        '''Engine feed type: 'tick' or 'kline'.'''
        return self._feed_type

    @property
    def tick_store(self) -> "TickDataStore | None":
        '''First available TickDataStore (compatibility).'''
        return next(iter(self._tick_stores.values()), None)

    @property
    def tick_stores(self) -> dict:
        '''Full access: {symbol: TickDataStore}.'''
        return self._tick_stores

    # ==========================================================================
    # Constructors - prepare the engine, do NOT execute
    # ==========================================================================

    @classmethod
    def by_ticks(
        cls,
        strategy: Strategy,
        data: Tuple[TickData, ...],
        sesh: "SeshSimulatorBase | None" = None,
        align_by_ts: bool = False,
        book: "BookData | tuple | list | None" = None,
        sync_book: bool = False,
    ) -> "BacktestEngine":
        '''
        Prepare engine for tick data.

        Data must be a tuple of TickData, one per symbol.
        The engine does NOT run here - call .run() to start the loop.

        Args:
            strategy (Strategy): Trading strategy instance
            data (tuple): Tuple of TickData objects
            sesh (SeshSimulatorBase): Optional session for testing
            align_by_ts (bool): Align symbols by timestamp instead of position
            book (BookData | tuple | list | None): Optional recorded L2 order
                book replayed alongside the ticks. Pass a single BookData or a
                tuple/list of them; the symbol is read from each
                ``BookData.symbol`` (no ``{symbol: BookData}`` mapping). Enables
                DeepTrades and the L2 book metrics in a backtest (book_as_of is
                populated causally as the tick cursor advances, exactly as in
                replay). A sync preflight warns if the book is desynchronized
                from the trades.
            sync_book (bool): When True and the preflight finds a recoverable
                clock offset between the book and the ticks, the book timestamps
                are shifted onto the trade clock. Default False (only warn on
                desync). An unrecoverable desync is never shifted.

        Returns:
            BacktestEngine: Prepared engine instance

        Note:
            align_by_ts controls how multi-symbol tick data is synchronized:

            - False (default): Symbols aligned by matrix row index (fast path).
              All symbols must have identical number of rows. Each on_data() call
              processes the same row index across all symbols, regardless of their
              actual timestamps. Risk: introduces look-ahead bias if symbols have
              different tick frequencies or timestamps.

            - True: Symbols aligned by real timestamp (merge path). Builds a
              unified timeline from all unique timestamps across symbols, sorted
              chronologically. Each symbol forward-fills to its most recent tick
              with ts <= current_ts. on_data() fires only when at least one symbol
              produces a new tick. Eliminates look-ahead bias for realistic
              multi-feed scenarios (e.g., BTC + ADA from independent sources).
              Slower than fast path due to np.searchsorted() lookups.

            With a single symbol, this parameter has no effect (dispatcher always
            uses fast path).

        Example:
            ticks = TickData('BTCUSDT', tick_matrix, tick_step=0.01)
            bt = BacktestEngine.by_ticks(MyStrategy(), data=(ticks,), sesh=sesh)
            bt.run()
        '''
        inputs, feed_type = _normalize_data(data)
        if feed_type != "tick":
            raise ConfigError(
                "by_ticks() received KlineData. Use by_klines() for candles."
            )

        if sesh is None:
            sesh = SeshSimulatorBase()

        engine = cls(strategy, sesh=sesh, _feed_type="tick")
        engine._tick_inputs = inputs
        engine._align_by_ts = align_by_ts
        engine._book_inputs = _normalize_book(book)
        engine._sync_book = bool(sync_book)

        for inp in inputs:
            sesh.add_symbol(inp.config)

        return engine

    @classmethod
    def by_klines(
        cls,
        strategy: Strategy,
        data: Tuple[KlineData, ...],
        sesh: "SeshSimulatorBase | None" = None,
        book: "BookData | tuple | list | None" = None,
    ) -> "BacktestEngine":
        '''
        Prepare engine for OHLC candle data (klines).

        Data must be a tuple of KlineData, one per symbol.
        Each KlineData must have interval_ms configured.
        subscribe_ticks() is not supported in this mode.
        The engine does NOT run here - call .run() to start the loop.

        Args:
            strategy (Strategy): Trading strategy instance
            data (tuple): Tuple of KlineData objects
            sesh (SeshSimulatorBase): Optional session for testing
            book (BookData | tuple | list | None): Not supported in kline mode -
                the L2 order book needs per-trade timestamps for the causal
                book_as_of lookup. Passing a book raises ConfigError; use
                by_ticks(book=...).

        Returns:
            BacktestEngine: Prepared engine instance

        Note:
            If sesh is omitted, a SeshSimulatorBase() is built by default
            (balance 10,000, no commission). The internal broker's feed_type
            is inferred automatically from the engine mode.

        Example:
            klines_btc = KlineData('BTCUSDT', btc_matrix, timeframe='5m')
            klines_eth = KlineData('ETHUSDT', eth_matrix, timeframe='1m')
            bt = BacktestEngine.by_klines(
                MyStrategy(),
                data=(klines_btc, klines_eth),
                sesh=sesh,
            )
            bt.run()
        '''
        inputs, feed_type = _normalize_data(data)
        if feed_type != "kline":
            raise ConfigError(
                "by_klines() received TickData. Use by_ticks() for ticks."
            )
        if book is not None:
            raise ConfigError(
                "by_klines() does not support book=. The L2 order book needs "
                "per-trade timestamps for the causal book_as_of lookup; use "
                "BacktestEngine.by_ticks(..., book=...) instead."
            )

        if sesh is None:
            sesh = SeshSimulatorBase()

        engine = cls(strategy, sesh=sesh, _feed_type="kline")
        engine._kline_inputs = inputs

        for inp in inputs:
            sesh.add_symbol(inp.config)

        return engine

    # ==========================================================================
    # Execution
    # ==========================================================================

    def run(self, params: dict = None, verbose: bool = False,
            save_log: "bool | None" = None) -> "BacktestEngine":
        '''
        Execute the complete backtest.

        Call after by_ticks() or by_klines(). Returns self for optional chaining.

        Args:
            params (dict): Optional dict to override parameters before running
            verbose (bool): Enable strategy logger output to console
            save_log (bool): Write log to file. None uses engine default

        Returns:
            BacktestEngine: Self for method chaining

        Note:
            save_log=None uses _DEFAULT_SAVE_LOG (False in backtest).
            Requires verbose=True and log_file defined in strategy.

        Example:
            result = BacktestEngine.by_ticks(
                strategy, data=(ticks,), sesh=sesh,
            ).run()
        '''
        if save_log is None:
            save_log = self._DEFAULT_SAVE_LOG

        if params is not None:
            self.strategy.update_params(params)

        if self._tick_inputs:
            data_dict = _inputs_to_dict(self._tick_inputs)
            self._run_ticks(data_dict, verbose=verbose, save_log=save_log)
        elif self._kline_inputs:
            data_dict = _inputs_to_dict(self._kline_inputs)
            self._run_klines(data_dict, verbose=verbose, save_log=save_log)
        elif self._feed_type == "tick":
            raise TradingError(
                "No data. Use BacktestEngine.by_ticks("
                "strategy, data=(TickData(...),), sesh=sesh) "
                "before calling run()."
            )
        else:
            raise TradingError(
                "No data. Use BacktestEngine.by_klines("
                "strategy, data=(KlineData(...),), sesh=sesh) "
                "before calling run()."
            )

        self._finalize_open_trades()
        self._stats = self._build_stats()
        return self

    def _finalize_open_trades(self) -> None:
        '''
        Close still-open positions at the end of the backtest, per the session's
        finalize_trades policy (mirrors backtesting.py).

        With finalize_trades=True the open positions are closed at the last
        known price so they enter the trade-based stats. With finalize_trades
        =False (default) they are left open and, if any remain, a warning is
        emitted - same intent as backtesting.py's message.
        '''
        broker = self._broker
        if broker is None or getattr(broker, "_stopped", False):
            return
        if not broker.positions:
            return
        if getattr(broker, "finalize_trades", False):
            broker.finalize()
        elif not getattr(self, "_fast_stats", False):
            # Suppressed in the optimize/pool worker path (_fast_stats) so the
            # single aggregate progress bar is not drowned in per-candidate
            # warnings; a normal single run() still warns like backtesting.py.
            import warnings
            n_open = len(broker.positions)
            warnings.warn(
                f"tradetropy: backtest finished with {n_open} position(s) still "
                "open; they are excluded from the trade-based stats. Set "
                "finalize_trades=True on the session (e.g. "
                "SeshSimulatorBase(finalize_trades=True)) to liquidate them at "
                "the last price and fold them into the results.",
                stacklevel=2,
            )

    def rerun(self, params: dict = None) -> "BacktestEngine":
        '''
        Create new engine with same data, run with new parameters.

        Creates a new engine with the same data and a cloned session, runs
        the backtest with the provided parameters and returns the new instance.

        Args:
            params (dict): New parameters for the backtest

        Returns:
            BacktestEngine: New engine with results

        Example:
            bt_optimal = bt_original.rerun(params=result.best_params)
            bt_optimal.plot()
        '''
        new_engine = self._clone_engine()
        if params is not None:
            new_engine.strategy.update_params(params)
        new_engine.run()
        return new_engine

    # ==========================================================================
    # Visualization
    # ==========================================================================

    def plot(
        self,
        theme: str = "light",
        width: int = 1200,
        plot_trades: bool = True,
        plot_volume: bool = True,
        ohlc_style: str = "candle",
        plot_drawdown: bool = False,
        plot_footprint: bool = True,
        plot_stats: bool = False,
        plot_pl: bool = False,
        pl_height: int = 100,
        ohlc_height: int = 410,
        equity_height: int = 110,
        drawdown_height: int = 80,
        indicator_height: int = 100,
        footprint_zoom_range: int = 40,
        max_candles: int = 10_000,
        equity_mode: str = "return",
        equity_unit: str = "percent",
        output: str = "show",
        filename: str = "backtest.html",
        resample_timeframe: str | None = None,
        align_trades_to_candle: bool = True,
        max_trailing_dd: float | None = None,
    ) -> None:
        '''
        Generate interactive backtest chart.

        Requires: pip install bokeh

        Args:
            theme (str): 'light' or 'dark'
            width (int): Total width in pixels
            plot_trades (bool): Show trade entry/exit lines
            plot_volume (bool): Show volume bars
            ohlc_style (str): Price panel style - 'candle' (Japanese
                candlesticks, default) or 'bar' (OHLC bars: High-Low vertical
                with open/close ticks)
            plot_drawdown (bool): Show drawdown panel
            plot_footprint (bool): Show footprint (requires FpProxy)
            plot_stats (bool): Show statistics bar
            plot_pl (bool): Show P&L panel per trade
            pl_height (int): P&L panel height in pixels
            ohlc_height (int): Main OHLC panel height
            equity_height (int): Equity/return panel height
            drawdown_height (int): Drawdown panel height
            indicator_height (int): Default height for indicator panels
            footprint_zoom_range (int): Max candles to display footprint
            max_candles (int): Candle render limit (performance)
            equity_mode (str): 'none' | 'balance' | 'return'
            equity_unit (str): 'currency' | 'percent'
            output (str): 'notebook' | 'file' | 'show'
            filename (str): HTML filename (only if output='file')
            resample_timeframe (str): Resample candles to given timeframe
                (e.g. '1m', '5m', '1h', '1d').
            align_trades_to_candle (bool): Align trade timestamps to candle open
            max_trailing_dd (float): Maximum trailing drawdown (e.g. 0.2 = 20%)

        Example:
            bt.plot()
            bt.plot(theme='dark', plot_trades=False)
            bt.plot(theme='dark', width=1400, equity_mode='balance')
        '''
        kwargs = dict(
            theme=theme,
            width=width,
            plot_trades=plot_trades,
            plot_volume=plot_volume,
            ohlc_style=ohlc_style,
            plot_drawdown=plot_drawdown,
            plot_footprint=plot_footprint,
            plot_stats=plot_stats,
            plot_pl=plot_pl,
            pl_height=pl_height,
            ohlc_height=ohlc_height,
            equity_height=equity_height,
            drawdown_height=drawdown_height,
            indicator_height=indicator_height,
            footprint_zoom_range=footprint_zoom_range,
            max_candles=max_candles,
            equity_mode=equity_mode,
            equity_unit=equity_unit,
            output=output,
            filename=filename,
            resample_timeframe=resample_timeframe,
            align_trades_to_candle=align_trades_to_candle,
            max_trailing_dd=max_trailing_dd,
        )
        self._plot_config = kwargs
        from tradetropy.plotting import plot as _plot, PlotConfig
        _plot(self, PlotConfig(**kwargs))

    # ==========================================================================
    # Optimization
    # ==========================================================================

    def optimize(self, maximize=None, minimize=None, constraints=None,
                 method="grid", iterations=100, workers=None, progress=True,
                 **param_lists):
        '''
        Optimize strategy parameters.

        Each parameter is passed as an explicit list of values to explore:

            bt.optimize(
                maximize = 'Sharpe Ratio',
                fast_ma = [10, 20, 30, 50],
                threshold = [0.1, 0.5, 1.0],
                mode = ['trend', 'mean_revert'],
            )

        Args:
            maximize (str): Name of the metric to maximize
            minimize (str): Name of the metric to minimize
            constraints: Function (params: dict) -> bool for filtering
            method (str): 'grid' or 'random'
            iterations (int): Number of random samples (only with method='random')
            workers (int): Parallel processes (None -> cpu_count)
            progress (bool): Show a single aggregate progress bar counting
                completed backtests (default True). One bar for the whole
                optimization, not one per backtest.
            **param_lists: Parameter name -> list of values to explore

        Returns:
            OptimizationResult: With best_params, best_fitness, best_stats, top(n),
                and to_dataframe() method
        '''
        from tradetropy.optimize import (
            ParameterSpace,
            FitnessMetric,
            OptimizationResult,
            GridSearchOptimizer,
            RandomSearchOptimizer,
        )
        from tradetropy.optimize.task import _create_evaluation_function
        from tradetropy.backtest.pool_adapter import PoolEvaluator

        if maximize is None and minimize is None:
            raise ConfigError('Specify maximize or minimize.')
        if maximize is not None and minimize is not None:
            raise ConfigError('Specify only one: maximize or minimize.')
        metric = maximize or minimize
        is_max = maximize is not None

        space = ParameterSpace(**param_lists)
        if constraints is not None:
            space.add_constraint(constraints)

        fitness = FitnessMetric(metric=metric, maximize=is_max)
        evaluate_fn = _create_evaluation_function(_run_backtest_candidate, fitness)

        # Ship the input matrices to workers via shared_memory (allocated once
        # in the parent) instead of pickling a full copy into each worker. Only
        # tiny descriptors + metadata travel in the pickle stream. Mirrors
        # PoolBacktestEngine; workers only read the arrays.
        from tradetropy.backtest._shm_bundle import build_shm_bundle, release_shm
        data_bundle, shm_refs = build_shm_bundle(
            strategy_cls=type(self.strategy),
            sesh=self._sesh,
            tick_inputs=self._tick_inputs,
            kline_inputs=self._kline_inputs,
            align_by_ts=self._align_by_ts,
        )
        evaluator = PoolEvaluator(evaluate_fn, data_bundle, workers=workers,
                                  progress=progress, desc="Optimize")

        if method == 'grid':
            optimizer = GridSearchOptimizer(space, fitness)
        elif method == 'random':
            optimizer = RandomSearchOptimizer(space, fitness, iterations=iterations)
        else:
            raise ConfigError(f'Unknown optimization method: {method!r}. Use grid or random.')

        try:
            optimizer.run(evaluator)
        finally:
            release_shm(shm_refs)
        return OptimizationResult(optimizer.results, maximize=is_max)

    # ==========================================================================
    # Monte Carlo robustness
    # ==========================================================================

    def montecarlo(self, n_sims=1000, methods=None, metrics=None,
                   confidence=(0.95, 0.99), seed=None, workers=None):
        '''
        Run a Monte Carlo robustness test on the finished backtest.

        Generates many randomized variants of the result to estimate the
        distribution of performance metrics, confidence intervals, the
        probability of loss, the risk of ruin and a composite robustness score.

        Methods operate at one of three levels (they cannot be mixed across
        levels in a single call):

            - Trade level (fast, no re-run): 'shuffle_order', 'resample_trades',
              'skip_trades', 'randomize_slippage', 'random_start_index'.
            - Data level (re-runs the engine): 'randomize_prices',
              'random_start_bar'.
            - Parameter level (re-runs the engine): 'randomize_parameters'
              (pass an instance with a search space).

        Args:
            n_sims (int): Number of simulations to run.
            methods (Sequence): Method identifiers (str) or MCMethod instances.
                Defaults to ['resample_trades'].
            metrics (Sequence[str]): Stats keys to track. Defaults to a standard
                set (Return, Max Drawdown, Sharpe, Profit Factor, Win Rate).
            confidence (Sequence[float]): Confidence levels in (0, 1).
            seed (int): Base seed for reproducibility.
            workers (int): Parallel processes for the re-run path (None ->
                cpu_count). Ignored on the trade-level path.

        Returns:
            MonteCarloResult: With summary(), to_dataframe(), percentile(),
                confidence_interval(), probability_of_loss, risk_of_ruin() and
                robustness_score.

        Example:
            bt.run()
            mc = bt.montecarlo(n_sims=1000, methods=['shuffle_order'], seed=42)
            print(mc.summary())
            print(mc.robustness_score)
        '''
        from tradetropy.robustness import MonteCarlo, MonteCarloConfig

        if self._stats is None:
            raise TradingError(
                'Run the backtest before montecarlo(): no stats available.'
            )

        kwargs = dict(
            n_sims=n_sims,
            methods=methods if methods is not None else ['resample_trades'],
            confidence=confidence,
            seed=seed,
            workers=workers,
        )
        if metrics is not None:
            kwargs['metrics'] = metrics

        config = MonteCarloConfig(**kwargs)
        return MonteCarlo(self, config).run()

    # ==========================================================================
    # Internal helpers (visible, but only called internally)
    # ==========================================================================

    def _clone_engine(self):
        '''Create new BacktestEngine with same inputs and session, without running.'''
        if self._tick_inputs:
            engine = BacktestEngine.by_ticks(
                strategy=self.strategy.__class__(),
                data=self._tick_inputs,
                sesh=self._sesh.clone(),
                align_by_ts=self._align_by_ts,
            )
        elif self._kline_inputs:
            engine = BacktestEngine.by_klines(
                strategy=self.strategy.__class__(),
                data=self._kline_inputs,
                sesh=self._sesh.clone(),
            )
        else:
            raise TradingError("No inputs to clone the engine")

        engine._plot_config = self._plot_config.copy()
        return engine


# ==========================================================================
# Internal method assignment
# (imported as standalone functions, assigned as class methods)
# ==========================================================================

from tradetropy.backtest._runner import (               # noqa: E402
    _setup_strategy,
    _build_tick_stores,
    _build_book_replay,
    _drain_books_at,
    _reset_books_for_loop,
    _run_ticks,
    _run_ticks_fast_path,
    _run_ticks_merge_path,
    _run_klines,
    _execute_kline_loop,
    _build_stats,
)
from tradetropy.backtest._validation import (            # noqa: E402
    _validate_symbols,
    _validate_symbols_klines,
)

BacktestEngine._setup_strategy = _setup_strategy
BacktestEngine._build_tick_stores = _build_tick_stores
BacktestEngine._build_book_replay = _build_book_replay
BacktestEngine._drain_books_at = _drain_books_at
BacktestEngine._reset_books_for_loop = _reset_books_for_loop
BacktestEngine._run_ticks = _run_ticks
BacktestEngine._run_ticks_fast_path = _run_ticks_fast_path
BacktestEngine._run_ticks_merge_path = _run_ticks_merge_path
BacktestEngine._run_klines = _run_klines
BacktestEngine._execute_kline_loop = _execute_kline_loop
BacktestEngine._build_stats = _build_stats
BacktestEngine._validate_symbols = _validate_symbols
BacktestEngine._validate_symbols_klines = _validate_symbols_klines


# ---------------------------------------------------------------------------
# Optimization runner (module level, picklable for multiprocessing spawn)
# ---------------------------------------------------------------------------

def _run_backtest_candidate(data, params):
    # Runs ONE backtest with the given params and returns the metrics dict.
    # Runs in child processes of PoolEvaluator, hence lives at module level
    # (picklable). data is the bundle from BacktestEngine.optimize():
    #   {strategy_cls, sesh, tick_inputs, kline_inputs, align_by_ts}
    strategy = data['strategy_cls']()
    strategy.update_params(params)

    if data['tick_inputs']:
        eng = BacktestEngine.by_ticks(
            strategy,
            data=data['tick_inputs'],
            sesh=data['sesh'].clone(),
            align_by_ts=data['align_by_ts'],
        )
    elif data['kline_inputs']:
        eng = BacktestEngine.by_klines(
            strategy,
            data=data['kline_inputs'],
            sesh=data['sesh'].clone(),
        )
    else:
        return {}

    # Worker runs in a spawned child: use the pandas-free stats path so the
    # child never imports pandas. Only numeric metrics are needed to rank
    # candidates and to fill OptimizationResult.to_dataframe().
    eng._fast_stats = True
    eng.run()
    return dict(eng._stats) if eng._stats is not None else {}

"""
Tests for:
  · Auto-creation of parent directory in disk write paths
    (io.save_ticks / io.save_klines, ResultReporter.save_to_csv, append-HDF5).
  · Opt-in control of file logging via save_log in engines
    (BacktestEngine/ReplayEngine → False by default; LiveEngine → True).
"""

import itertools
import logging

import numpy as np
import pytest

from tradetropy.models.strategy import Strategy
from tradetropy.backtest import BacktestEngine
from tradetropy.core.data_types import KlineData
from tradetropy.session.base import SeshSimulatorBase


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_counter = itertools.count()


def _make_strategy_cls(log_file):
    """Creates a Strategy subclass with a unique name (avoids logger
    memoization by name across tests) and given log_file."""
    name = f"_SaveLogStrat_{next(_counter)}"

    def init(self):
        self._ohlc = self.subscribe_ohlc("MES", 1.0, window_size=10)

    def on_data(self):
        self.log.info("tick")

    return type(name, (Strategy,), {
        "log_file": log_file,
        "init": init,
        "on_data": on_data,
    })


def _has_file_handler(logger) -> bool:
    return any(isinstance(h, logging.FileHandler) for h in logger.handlers)


@pytest.fixture
def klines_simple():
    """5 1m candles. base multiple of 60_000."""
    base = 1_700_000_000_000
    return np.array([
        [base + i * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0, 1005.0]
        for i in range(5)
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Auto-creation of directories
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEnsureParentDir:
    def test_helper_creates_missing_dir(self, tmp_path):
        from tradetropy.io.io import _ensure_parent_dir
        target = tmp_path / "a" / "b" / "c.txt"
        assert not target.parent.exists()
        result = _ensure_parent_dir(target)
        assert target.parent.exists()
        assert result == target

    def test_save_ticks_creates_dir_nested(self, tmp_path):
        from tradetropy.io.io import save_ticks
        from tradetropy.core.constants import N_TICK_COLS
        arr = np.zeros((3, N_TICK_COLS), dtype=np.float64)
        arr[:, 0] = [1, 2, 3]
        # Use csv (no optional dependency); the mkdir in _write is format-agnostic
        # (it occurs before the csv/parquet/hdf5 branch).
        path = tmp_path / "nueva" / "carpeta" / "ticks.csv"
        assert not path.parent.exists()
        save_ticks(arr, path, format="csv")
        assert path.exists()

    def test_save_ticks_creates_dir_csv(self, tmp_path):
        from tradetropy.io.io import save_ticks
        from tradetropy.core.constants import N_TICK_COLS
        arr = np.zeros((3, N_TICK_COLS), dtype=np.float64)
        arr[:, 0] = [1, 2, 3]
        path = tmp_path / "csv_dir" / "ticks.csv"
        save_ticks(arr, path, format="csv")
        assert path.exists()

    def test_save_klines_creates_dir(self, tmp_path, klines_simple):
        from tradetropy.io.io import save_klines
        path = tmp_path / "k" / "deep" / "ohlc.csv"
        assert not path.parent.exists()
        save_klines(klines_simple, path, format="csv")
        assert path.exists()

    def test_reporter_save_to_csv_creates_dir(self, tmp_path):
        from tradetropy.optimize.reporter import ResultReporter
        from tradetropy.optimize.task import Result
        results = [Result(candidate_id=0, fitness=1.0, metrics={"sharpe": 2.0})]
        path = tmp_path / "rep" / "out" / "res.csv"
        assert not path.parent.exists()
        ResultReporter.save_to_csv(results, str(path))
        assert path.exists()

    def test_append_klines_hdf5_creates_dir(self, tmp_path, klines_simple):
        from tradetropy.io.io import _append_klines_hdf5
        pytest.importorskip("tables")
        path = tmp_path / "h5" / "data.h5"
        _append_klines_hdf5(klines_simple, path)
        assert path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# Opt-in file gating in Strategy
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStrategyGating:
    def test_save_log_false_no_file_handler(self, tmp_path):
        log_file = tmp_path / "logs" / "bt.log"
        strat = _make_strategy_cls(str(log_file))()
        strat._verbose = True
        strat._save_log = False
        strat._set_run_mode("backtest")
        logger = strat.log
        assert not _has_file_handler(logger)
        assert not log_file.exists()

    def test_save_log_true_adds_file_handler_and_dir(self, tmp_path):
        log_file = tmp_path / "logs" / "bt.log"
        strat = _make_strategy_cls(str(log_file))()
        strat._verbose = True
        strat._save_log = True
        strat._set_run_mode("backtest")
        logger = strat.log
        assert _has_file_handler(logger)
        # get_strategy_logger creates the parent directory automatically
        assert log_file.parent.exists()

    def test_save_log_true_but_no_log_file(self, tmp_path):
        strat = _make_strategy_cls(None)()
        strat._verbose = True
        strat._save_log = True
        strat._set_run_mode("backtest")
        logger = strat.log
        assert not _has_file_handler(logger)

    def test_set_save_log_invalidates_logger(self, tmp_path):
        log_file = tmp_path / "logs" / "bt.log"
        strat = _make_strategy_cls(str(log_file))()
        strat._verbose = True
        strat._set_run_mode("backtest")
        strat._set_save_log(False)
        assert not _has_file_handler(strat.log)
        strat._set_save_log(True)
        assert _has_file_handler(strat.log)


# ══════════════════════════════════════════════════════════════════════════════
# Engines: defaults and override
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEngineDefaults:
    def test_backtest_default_no_file(self, tmp_path, klines_simple):
        log_file = tmp_path / "logs" / "bt.log"
        Strat = _make_strategy_cls(str(log_file))
        engine = BacktestEngine.by_klines(
            Strat(),
            data=(KlineData("MES", klines_simple, timeframe=60_000, tick_size=0.01),),
            sesh=SeshSimulatorBase("kline"),
        ).run(verbose=True)  # save_log default = False
        assert engine.strategy._save_log is False
        assert not log_file.exists()

    def test_backtest_save_log_true_writes_file(self, tmp_path, klines_simple):
        log_file = tmp_path / "logs" / "bt.log"
        Strat = _make_strategy_cls(str(log_file))
        engine = BacktestEngine.by_klines(
            Strat(),
            data=(KlineData("MES", klines_simple, timeframe=60_000, tick_size=0.01),),
            sesh=SeshSimulatorBase("kline"),
        ).run(verbose=True, save_log=True)
        assert engine.strategy._save_log is True
        assert _has_file_handler(engine.strategy.log)
        # Force flush/close and verify that the file exists
        for h in engine.strategy.log.handlers:
            h.flush()
        assert log_file.exists()

    def test_backtest_default_save_log_constant(self):
        assert BacktestEngine._DEFAULT_SAVE_LOG is False

    def test_live_default_save_log_constant(self):
        from tradetropy.live import LiveEngine
        assert LiveEngine._DEFAULT_SAVE_LOG is True

    def test_replay_default_save_log_constant(self):
        from tradetropy.replay.engine import ReplayEngine
        assert ReplayEngine._DEFAULT_SAVE_LOG is False


# ══════════════════════════════════════════════════════════════════════════════
# pool.run accepts save_log (no-op)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPoolApiConsistency:
    def test_pool_run_accepts_save_log_param(self):
        import inspect
        from tradetropy.backtest.pool import PoolBacktestEngine
        sig = inspect.signature(PoolBacktestEngine.run)
        assert "save_log" in sig.parameters
        assert sig.parameters["save_log"].default is None

# Tests for the Monte Carlo robustness module (tradetropy.robustness).
import numpy as np
import pandas as pd
import pytest

from tradetropy.exceptions import ConfigError
from tradetropy.models import Strategy
from tradetropy.robustness import (
    MonteCarloConfig,
    ShuffleOrder,
    ResampleTrades,
    SkipTrades,
    RandomizeSlippage,
    RandomStartIndex,
    available_methods,
    resolve_method,
)

# Monte Carlo re-runs over short synthetic series intentionally trip the
# insufficient-sample stats warning (asserted on its own in test_stats.py).
# Colon matched with '.' since warning-filter fields split on ':'.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Stats. insufficient sample:UserWarning"
)


class KlineHoldStrat(Strategy):
    """Simple hold strategy for the re-run integration test."""

    hold = 5

    def init(self):
        self.k = self.subscribe_ohlc('SYM', 60_000, window_size=50)
        self._n = 0

    def on_data(self):
        self._n += 1
        if self._n == 10:
            self.sesh.buy('SYM', volume=1)
        elif self._n == 10 + self.hold:
            for p in self.sesh.positions('SYM'):
                self.sesh.position_close(p.ticket)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trades(n=50, seed=0):
    """Build a normalized trades DataFrame like Stats.trades."""
    rng = np.random.default_rng(seed)
    pnl = rng.normal(1.0, 5.0, size=n)
    t0 = pd.Timestamp('2024-01-01', tz='UTC')
    entry = t0 + pd.to_timedelta(np.arange(n) * 60, unit='s')
    exit_ = entry + pd.to_timedelta(30, unit='s')
    return pd.DataFrame({
        'entry_time': entry,
        'exit_time': exit_,
        'size': np.ones(n),
        'entry_price': 100.0 + np.arange(n) * 0.1,
        'exit_price': 100.0 + np.arange(n) * 0.1 + pnl,
        'direction': ['long'] * n,
        'commission': np.zeros(n),
        'pnl_net': pnl,
        'symbol': ['SYM'] * n,
    })


# ---------------------------------------------------------------------------
# Registry / resolution
# ---------------------------------------------------------------------------


def test_registry_contains_trade_methods():
    names = available_methods()
    for name in ('shuffle_order', 'resample_trades', 'skip_trades',
                 'randomize_slippage', 'random_start_index'):
        assert name in names


def test_resolve_string_returns_instance():
    m = resolve_method('shuffle_order')
    assert isinstance(m, ShuffleOrder)
    assert m.level == 'trades'


def test_resolve_instance_passthrough():
    inst = ResampleTrades(replace=False)
    assert resolve_method(inst) is inst


def test_resolve_unknown_raises():
    with pytest.raises(ConfigError):
        resolve_method('does_not_exist')


def test_resolve_invalid_type_raises():
    with pytest.raises(ConfigError):
        resolve_method(123)


# ---------------------------------------------------------------------------
# Config validation and mode derivation
# ---------------------------------------------------------------------------


def test_config_defaults_trade_mode():
    cfg = MonteCarloConfig(n_sims=100, methods=['shuffle_order'], seed=1)
    assert cfg.mode == 'trades'
    assert len(cfg.resolved) == 1


def test_config_rejects_bad_n_sims():
    with pytest.raises(ConfigError):
        MonteCarloConfig(n_sims=0, methods=['shuffle_order'])


def test_config_rejects_bad_confidence():
    with pytest.raises(ConfigError):
        MonteCarloConfig(n_sims=10, methods=['shuffle_order'], confidence=(1.5,))


def test_config_rejects_empty_methods():
    with pytest.raises(ConfigError):
        MonteCarloConfig(n_sims=10, methods=[])


def test_config_rejects_bad_workers():
    with pytest.raises(ConfigError):
        MonteCarloConfig(n_sims=10, methods=['shuffle_order'], workers=0)


# ---------------------------------------------------------------------------
# Trade-level methods
# ---------------------------------------------------------------------------


def test_shuffle_preserves_pnl_set():
    df = _make_trades()
    rng = np.random.default_rng(7)
    out = ShuffleOrder().apply(df, rng)
    assert len(out) == len(df)
    # Net profit preserved, order changed.
    assert np.isclose(out['pnl_net'].sum(), df['pnl_net'].sum())
    assert not np.array_equal(out['pnl_net'].to_numpy(), df['pnl_net'].to_numpy())


def test_shuffle_deterministic_with_seed():
    df = _make_trades()
    a = ShuffleOrder().apply(df, np.random.default_rng(42))
    b = ShuffleOrder().apply(df, np.random.default_rng(42))
    assert np.array_equal(a['pnl_net'].to_numpy(), b['pnl_net'].to_numpy())


def test_resample_with_replacement_shape():
    df = _make_trades()
    out = ResampleTrades(replace=True).apply(df, np.random.default_rng(1))
    assert len(out) == len(df)


def test_resample_without_replacement_is_permutation():
    df = _make_trades(n=30)
    out = ResampleTrades(replace=False).apply(df, np.random.default_rng(3))
    assert np.isclose(out['pnl_net'].sum(), df['pnl_net'].sum())


def test_skip_trades_reduces_count():
    df = _make_trades(n=200)
    out = SkipTrades(prob=0.5).apply(df, np.random.default_rng(2))
    assert 0 < len(out) < len(df)


def test_skip_trades_rejects_bad_prob():
    with pytest.raises(ConfigError):
        SkipTrades(prob=1.0)


def test_randomize_slippage_reduces_pnl():
    df = _make_trades()
    out = RandomizeSlippage(bps=5.0).apply(df, np.random.default_rng(9))
    # Slippage is a cost, so total pnl cannot increase.
    assert out['pnl_net'].sum() <= df['pnl_net'].sum() + 1e-9
    assert len(out) == len(df)


def test_randomize_slippage_rejects_negative():
    with pytest.raises(ConfigError):
        RandomizeSlippage(bps=-1.0)


def test_random_start_index_keeps_tail():
    df = _make_trades(n=100)
    out = RandomStartIndex(max_frac=0.5).apply(df, np.random.default_rng(5))
    assert 0 < len(out) <= len(df)
    # Kept rows are a contiguous tail of the original.
    assert out['pnl_net'].iloc[-1] == df['pnl_net'].iloc[-1]



# ---------------------------------------------------------------------------
# Trade-level run (core recomputation)
# ---------------------------------------------------------------------------


def _baseline_stats(n=60, seed=0, initial_balance=10_000.0):
    """Build a baseline Stats object from synthetic trades."""
    import warnings
    from tradetropy.robustness.core import _SyntheticTrade, _build_equity_curve
    from tradetropy.stats.stats import compute_stats

    df = _make_trades(n, seed)
    entry = df['entry_time'].to_numpy()
    exit_ = df['exit_time'].to_numpy()
    pnl = df['pnl_net'].to_numpy()
    trades = [
        _SyntheticTrade(
            entry_time=entry[i], exit_time=exit_[i], volume=df['size'].iloc[i],
            entry_price=df['entry_price'].iloc[i], exit_price=df['exit_price'].iloc[i],
            direction=df['direction'].iloc[i], commission=df['commission'].iloc[i],
            pnl_net=pnl[i], symbol=df['symbol'].iloc[i],
        )
        for i in range(n)
    ]
    eq = _build_equity_curve(entry, exit_, pnl, initial_balance)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_stats(
            eq, trades, initial_balance, freq='1min',
            min_trades=1, min_duration_ann=pd.Timedelta(0),
        )


def test_montecarlo_trade_run_shapes():
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig, MonteCarloResult

    base = _baseline_stats()
    cfg = MonteCarloConfig(n_sims=200, methods=['resample_trades'], seed=11)
    result = MonteCarlo(base, cfg).run()

    assert isinstance(result, MonteCarloResult)
    assert result.n_sims == 200
    df = result.to_dataframe()
    assert len(df) == 200
    assert 'Return [%]' in df.columns


def test_shuffle_preserves_return_varies_drawdown():
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig

    base = _baseline_stats(seed=3)
    cfg = MonteCarloConfig(
        n_sims=100,
        methods=['shuffle_order'],
        metrics=['Return [%]', 'Max. Drawdown [%]'],
        seed=7,
    )
    result = MonteCarlo(base, cfg).run()
    df = result.to_dataframe()

    # Shuffling preserves total PnL: every simulation has the same return.
    assert df['Return [%]'].std() < 1e-6
    # But the drawdown path depends on order, so it varies.
    assert df['Max. Drawdown [%]'].std() > 0.0


def test_montecarlo_reproducible_with_seed():
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig

    base = _baseline_stats(seed=5)
    a = MonteCarlo(base, MonteCarloConfig(
        n_sims=50, methods=['resample_trades'], seed=99)).run().to_dataframe()
    b = MonteCarlo(base, MonteCarloConfig(
        n_sims=50, methods=['resample_trades'], seed=99)).run().to_dataframe()
    pd.testing.assert_frame_equal(a, b)


def test_montecarlo_requires_trades():
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig
    from tradetropy.exceptions import DataError

    base = _baseline_stats()
    base['_trades'] = base.trades.iloc[0:0]
    with pytest.raises(DataError):
        MonteCarlo(base, MonteCarloConfig(n_sims=10, methods=['shuffle_order']))


# ---------------------------------------------------------------------------
# MonteCarloResult analytics
# ---------------------------------------------------------------------------


def _result(n_sims=300, methods=('resample_trades',), seed=21, **kw):
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig
    base = _baseline_stats(**kw)
    cfg = MonteCarloConfig(n_sims=n_sims, methods=list(methods), seed=seed,
                           confidence=(0.95, 0.99))
    return MonteCarlo(base, cfg).run(), base


def test_summary_table_structure():
    result, base = _result()
    summ = result.summary()
    assert set(['original', 'mean', 'std', 'p5', 'p50', 'p95',
                'ci_low', 'ci_high']).issubset(summ.columns)
    # Every tracked metric is a row.
    for m in result.metrics:
        assert m in summ.index


def test_percentile_and_confidence_interval():
    result, _ = _result()
    p50 = result.percentile('Return [%]', 50)
    lo, hi = result.confidence_interval('Return [%]', 0.95)
    assert lo <= p50 <= hi


def test_percentile_unknown_metric_raises():
    result, _ = _result()
    with pytest.raises(KeyError):
        result.percentile('Nonexistent Metric', 50)


def test_probability_of_loss_in_range():
    result, _ = _result()
    p = result.probability_of_loss
    assert 0.0 <= p <= 1.0


def test_risk_of_ruin_bounds_and_validation():
    result, _ = _result()
    assert 0.0 <= result.risk_of_ruin(0.5) <= 1.0
    # Tiny threshold -> more breaches than a large threshold.
    assert result.risk_of_ruin(0.01) >= result.risk_of_ruin(0.99)
    with pytest.raises(ValueError):
        result.risk_of_ruin(0.0)


def test_robustness_score_range():
    result, _ = _result()
    score = result.robustness_score
    assert 0.0 <= score <= 100.0



# ---------------------------------------------------------------------------
# Data-level methods + re-run path
# ---------------------------------------------------------------------------


def _kline_ramp(n=300, start=100.0, step=0.05):
    """Build a [N x 7] kline matrix: ts, open, high, low, close, vol, turnover."""
    arr = np.zeros((n, 7), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64) * 60_000.0
    close = start + np.arange(n, dtype=np.float64) * step
    open_ = close - step
    arr[:, 1] = open_
    arr[:, 2] = np.maximum(open_, close) + 0.02
    arr[:, 3] = np.minimum(open_, close) - 0.02
    arr[:, 4] = close
    arr[:, 5] = 10.0
    arr[:, 6] = close * 10.0
    return arr


def test_randomize_prices_keeps_ohlc_valid():
    from tradetropy.core.data_types import KlineData
    from tradetropy.robustness.methods import RandomizePrices

    kd = KlineData('SYM', _kline_ramp(), timeframe=60_000)
    out = RandomizePrices(noise_pct=0.01).apply((kd,), np.random.default_rng(1))
    data = out[0].data
    o, h, l, c = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    assert np.all(h >= np.maximum(o, c) - 1e-9)
    assert np.all(l <= np.minimum(o, c) + 1e-9)
    # Original input not mutated.
    assert not np.array_equal(data, kd.data)


def test_random_start_bar_drops_leading_rows():
    from tradetropy.core.data_types import KlineData
    from tradetropy.robustness.methods import RandomStartBar

    kd = KlineData('SYM', _kline_ramp(n=200), timeframe=60_000)
    out = RandomStartBar(max_frac=0.5).apply((kd,), np.random.default_rng(4))
    assert 0 < len(out[0].data) <= 200


def test_config_data_method_is_rerun_mode():
    from tradetropy.robustness import MonteCarloConfig
    cfg = MonteCarloConfig(n_sims=10, methods=['randomize_prices'])
    assert cfg.mode == 'rerun'


def test_config_rejects_mixed_levels():
    from tradetropy.robustness import MonteCarloConfig
    with pytest.raises(ConfigError):
        MonteCarloConfig(n_sims=10, methods=['shuffle_order', 'randomize_prices'])


@pytest.mark.integration
def test_rerun_randomize_prices_end_to_end():
    from tradetropy.core.data_types import KlineData
    from tradetropy.backtest import BacktestEngine
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig, MonteCarloResult

    kd = KlineData('SYM', _kline_ramp(n=400), timeframe=60_000)
    bt = BacktestEngine.by_klines(KlineHoldStrat(), data=(kd,)).run()
    assert bt.stats is not None and len(bt.stats.trades) > 0

    cfg = MonteCarloConfig(
        n_sims=8, methods=['randomize_prices'],
        metrics=['Return [%]', 'Max. Drawdown [%]'], seed=3, workers=1,
    )
    result = MonteCarlo(bt, cfg).run()
    assert isinstance(result, MonteCarloResult)
    assert result.n_sims == 8
    assert len(result.to_dataframe()) == 8


# ---------------------------------------------------------------------------
# Parameter permutation + robustness score
# ---------------------------------------------------------------------------


def test_randomize_parameters_samples_in_space():
    from tradetropy.robustness.methods import RandomizeParameters

    m = RandomizeParameters(fast=[10, 20, 30], slow=[50, 100])
    out = m.apply({}, np.random.default_rng(1))
    assert out['fast'] in (10, 20, 30)
    assert out['slow'] in (50, 100)


def test_randomize_parameters_deterministic():
    from tradetropy.robustness.methods import RandomizeParameters

    a = RandomizeParameters(fast=[10, 20, 30]).apply({}, np.random.default_rng(8))
    b = RandomizeParameters(fast=[10, 20, 30]).apply({}, np.random.default_rng(8))
    assert a == b


def test_randomize_parameters_rejects_space_and_kwargs():
    from tradetropy.optimize import ParameterSpace
    from tradetropy.robustness.methods import RandomizeParameters

    with pytest.raises(ConfigError):
        RandomizeParameters(space=ParameterSpace(a=[1, 2]), b=[3, 4])


def test_param_method_is_rerun_mode():
    from tradetropy.robustness import MonteCarloConfig
    from tradetropy.robustness.methods import RandomizeParameters

    cfg = MonteCarloConfig(n_sims=5, methods=[RandomizeParameters(fast=[1, 2])])
    assert cfg.mode == 'rerun'


def _result_from_arrays(returns, max_dd, initial_balance=10_000.0):
    from tradetropy.robustness import MonteCarloResult
    sims = pd.DataFrame({'Return [%]': returns})
    final_equity = initial_balance * (1.0 + np.asarray(returns) / 100.0)
    return MonteCarloResult(
        sims, None,
        final_equity=final_equity,
        max_drawdown=np.asarray(max_dd, dtype=float),
        initial_balance=initial_balance,
    )


def test_robustness_score_stable_beats_fragile():
    rng = np.random.default_rng(0)
    # Stable: consistently profitable, shallow drawdowns.
    stable = _result_from_arrays(
        returns=rng.normal(12.0, 1.5, 500),
        max_dd=rng.uniform(-6.0, -2.0, 500),
    )
    # Fragile: wide returns crossing zero, deep drawdowns.
    fragile = _result_from_arrays(
        returns=rng.normal(2.0, 25.0, 500),
        max_dd=rng.uniform(-70.0, -30.0, 500),
    )
    assert stable.robustness_score > fragile.robustness_score
    assert stable.probability_of_loss < fragile.probability_of_loss



# ---------------------------------------------------------------------------
# Engine integration: bt.montecarlo()
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_engine_montecarlo_trade_level():
    from tradetropy.core.data_types import KlineData
    from tradetropy.backtest import BacktestEngine
    from tradetropy.robustness import MonteCarloResult

    kd = KlineData('SYM', _kline_ramp(n=400), timeframe=60_000)
    bt = BacktestEngine.by_klines(KlineHoldStrat(), data=(kd,)).run()

    mc = bt.montecarlo(
        n_sims=100, methods=['shuffle_order', 'resample_trades'],
        metrics=['Return [%]', 'Max. Drawdown [%]'], seed=42,
    )
    assert isinstance(mc, MonteCarloResult)
    assert mc.n_sims == 100
    summ = mc.summary()
    assert 'Return [%]' in summ.index


def test_top_level_exports():
    import tradetropy
    assert hasattr(tradetropy, 'MonteCarlo')
    assert hasattr(tradetropy, 'MonteCarloConfig')
    assert hasattr(tradetropy, 'MonteCarloResult')


def test_montecarlo_before_run_raises():
    from tradetropy.core.data_types import KlineData
    from tradetropy.backtest import BacktestEngine
    from tradetropy.exceptions import TradingError

    kd = KlineData('SYM', _kline_ramp(n=100), timeframe=60_000)
    bt = BacktestEngine.by_klines(KlineHoldStrat(), data=(kd,))
    with pytest.raises(TradingError):
        bt.montecarlo(n_sims=10)



# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def test_plot_to_file(tmp_path):
    pytest.importorskip('bokeh')
    from tradetropy.robustness import MonteCarlo, MonteCarloConfig

    base = _baseline_stats(seed=4)
    cfg = MonteCarloConfig(
        n_sims=50, methods=['resample_trades'],
        metrics=['Return [%]', 'Max. Drawdown [%]'], seed=1,
    )
    result = MonteCarlo(base, cfg).run()
    out = tmp_path / 'mc.html'
    result.plot(output='file', filename=str(out))
    assert out.exists() and out.stat().st_size > 0

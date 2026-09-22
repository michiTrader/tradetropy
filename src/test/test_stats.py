import numpy as np
import pandas as pd
import pytest
import warnings


from tradetropy.core.broker import Trade, OrderType
from tradetropy.stats import Stats
from tradetropy.exceptions import DataError, ConfigError


# Small-sample fixtures intentionally trip the insufficient-sample stats warning
# (its own contract is asserted explicitly in TestLowSampleGating); the
# pandas/numpy subtract RuntimeWarning comes from std over NaN-bearing series.
# The colon in the stats message is matched with '.' (filter fields split on ':').
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Stats. insufficient sample:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in subtract:RuntimeWarning"
    ),
]


def _equity_curve(
    prices: list[float], *, start: str = "2024-01-01", freq: str = "1D"
) -> pd.Series:
    ts = pd.date_range(start=start, periods=len(prices), freq=freq, tz="UTC")
    return pd.Series(prices, index=ts, name="close")


def _make_trades(trades: list[dict]) -> list[Trade]:
    result = []
    for i, t in enumerate(trades):
        if t.get("exit_price") is not None:
            sign = 1.0 if t["direction"] == "long" else -1.0
            profit = sign * t["size"] * (t["exit_price"] - t["entry_price"])
        else:
            profit = 0.0
        result.append(Trade(
            ticket=i + 1,
            position_id=i + 1,
            symbol="TEST",
            type=OrderType.ORDER_TYPE_BUY if t["direction"] == "long" else OrderType.ORDER_TYPE_SELL,
            volume=t["size"],
            price=t["entry_price"],
            time=t["entry_time"],
            commission=t.get("commission", 0.0),
            profit=profit,
            price_close=t.get("exit_price") or 0.0,
            time_close=t.get("exit_time"),
        ))
    return result


def _make_equity_curve(prices: list[float], initial_balance: float = 1000) -> pd.Series:
    ts = pd.date_range(start="2024-01-01", periods=len(prices) + 1, freq="1D", tz="UTC")
    return pd.Series([initial_balance] + prices, index=ts, name="close")


@pytest.mark.unit
class TestStatsBasics:
    def test_to_dict_keys_and_order(self):
        equity = _equity_curve([100, 101, 102])
        trades = _make_trades([
            {
                "entry_time": equity.index[0],
                "exit_time": equity.index[-1],
                "size": 1.0,
                "entry_price": 100.0,
                "exit_price": 102.0,
                "direction": "long",
            },
        ])
        s = Stats(equity, trades, 1000)
        d = s.to_dict()
        assert list(d.keys()) == list(Stats._KEYS)

    def test_repr_shows_stats(self):
        equity = _make_equity_curve([1010], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[0],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=102,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)
        r = repr(s)
        assert "Return [%]" in r

    def test_buy_and_hold_long_trade_no_drawdown(self):
        equity = _make_equity_curve([1002, 1004, 1006, 1010], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[0],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=110,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)
        d = s.to_dict()

        assert d["Start"] == equity.index[0]
        assert d["End"] == equity.index[-1]
        assert d["Duration"] == pd.Timedelta(days=4)
        assert d["Exposure Time [%]"] == pytest.approx(100.0, abs=1e-4)
        assert d["Equity Final [$]"] == pytest.approx(1010.0, abs=1e-4)
        assert d["Equity Peak [$]"] == pytest.approx(1010.0, abs=1e-4)
        assert d["Return [%]"] == pytest.approx(1.0, abs=1e-4)
        assert np.isnan(d["Calmar Ratio"])  # low sample: duration 4d < 7d
        assert d["Max. Drawdown [%]"] == pytest.approx(0.0, abs=1e-4)
        assert d["Avg. Drawdown [%]"] == pytest.approx(0.0, abs=1e-4)
        assert d["Max. Drawdown Duration"] == pd.Timedelta(0)
        assert d["Avg. Drawdown Duration"] == pd.Timedelta(0)

        assert d["# Trades"] == 1
        assert d["Win Rate [%]"] == pytest.approx(100.0, abs=1e-4)
        assert d["Best Trade [%]"] == pytest.approx(10.0, abs=1e-4)
        assert d["Worst Trade [%]"] == pytest.approx(10.0, abs=1e-4)
        assert d["Avg. Trade [%]"] == pytest.approx(10.0, abs=1e-4)
        assert d["Max. Trade Duration"] == pd.Timedelta(days=4)
        assert d["Avg. Trade Duration"] == pd.Timedelta(days=4)
        assert np.isnan(d["Profit Factor"])  # low sample: 1 trade < 2
        assert d["Expectancy [%]"] == pytest.approx(10.0, abs=1e-4)
        assert np.isnan(d["SQN"])  # low sample: 1 trade < 2

        # Return (Ann.) and Volatility (Ann.) are nullified: duration 4d < 7d
        assert np.isnan(d["Return (Ann.) [%]"])
        assert np.isnan(d["Volatility (Ann.) [%]"])
        assert d["_low_sample"] is True

    def test_stats_desde_deals(self):
        equity = _make_equity_curve([1010], initial_balance=1000)
        trades = _make_trades([
            {
                "entry_time": equity.index[1],
                "exit_time": equity.index[-1],
                "size": 1.0,
                "entry_price": 100.0,
                "exit_price": 110.0,
                "direction": "long",
            },
        ])
        s = Stats(equity, trades, 1000.0)
        d = s.to_dict()
        assert d["# Trades"] == 1
        assert d["Equity Final [$]"] == pytest.approx(1010.0, abs=1e-4)
        assert d["Best Trade [%]"] == pytest.approx(10.0, abs=1e-4)

    def test_100_percent_losing_strategy(self):
        equity = _make_equity_curve([990, 980], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[1],
                    exit_time=equity.index[2],
                    size=1,
                    entry_price=100,
                    exit_price=90,
                    direction="long",
                ),
                dict(
                    entry_time=equity.index[2],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=90,
                    direction="long",
                ),
            ]
        )
        s = Stats(equity, deals, 1000)
        d = s.to_dict()

        assert d["# Trades"] == 2
        assert d["Win Rate [%]"] == pytest.approx(0.0, abs=1e-4)
        assert d["Return [%]"] == pytest.approx(-2.0, abs=1e-4)
        assert d["Best Trade [%]"] == pytest.approx(-10.0, abs=1e-4)
        assert d["Worst Trade [%]"] == pytest.approx(-10.0, abs=1e-4)
        assert d["Avg. Trade [%]"] == pytest.approx(-10.0, abs=1e-4)
        assert d["Profit Factor"] == pytest.approx(0.0, abs=1e-4)
        assert d["Expectancy [%]"] == pytest.approx(-10.0, abs=1e-4)
        assert np.isneginf(d["SQN"])

    def test_open_trade_includes_unrealized_and_not_counted(self):
        equity = _make_equity_curve([1005], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[1],
                    exit_time=None,
                    size=1,
                    entry_price=100,
                    exit_price=None,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)
        d = s.to_dict()

        assert d["# Trades"] == 0
        assert d["Equity Final [$]"] == pytest.approx(1005.0, abs=1e-4)
        assert d["Return [%]"] == pytest.approx(0.5, abs=1e-4)
        assert np.isnan(d["Profit Factor"])

    def test_initial_balance_zero_avoid_division_by_zero(self):
        equity = _equity_curve([100, 110])
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[0],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=110,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 0.0)
        d = s.to_dict()
        assert np.isinf(d["Return [%]"])

    def test_sharpe_sortino_std_cero_mu_cero_nan(self):
        equity = _equity_curve([100, 100, 100, 100])
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[0],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=100,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)
        d = s.to_dict()
        assert np.isnan(d["Sharpe Ratio"])
        assert np.isnan(d["Sortino Ratio"])

    def test_drawdown_and_duration(self):
        equity = _make_equity_curve([1005, 1010], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[0],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=110,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)
        d = s.to_dict()
        assert d["Max. Drawdown [%]"] == pytest.approx(0.0, abs=1e-4)
        assert d["Max. Drawdown Duration"] == pd.Timedelta(0)

    def test_short_trade(self):
        equity = _make_equity_curve([1020], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[1],
                    exit_time=equity.index[-1],
                    size=2,
                    entry_price=100,
                    exit_price=90,
                    direction="short",
                )
            ]
        )
        s = Stats(equity, deals, 1000)
        d = s.to_dict()
        assert d["Equity Final [$]"] == pytest.approx(1020.0, abs=1e-4)
        assert d["Best Trade [%]"] == pytest.approx(10.0, abs=1e-4)


@pytest.mark.unit
class TestPositionsGrouping:
    """# Positions / Long / Short and stats.positions (partial-close grouping)."""

    def test_partial_closes_count_as_one_position(self):
        # Same position_id closed in two partial closes must count once.
        equity = _make_equity_curve([1005, 1010], initial_balance=1000)
        trades = [
            Trade(
                ticket=1, position_id=100, symbol="TEST",
                type=OrderType.ORDER_TYPE_BUY,
                volume=1.0, price=100.0, time=equity.index[0],
                commission=0.0, profit=5.0,
                price_close=105.0, time_close=equity.index[1],
            ),
            Trade(
                ticket=2, position_id=100, symbol="TEST",
                type=OrderType.ORDER_TYPE_BUY,
                volume=1.0, price=100.0, time=equity.index[0],
                commission=0.0, profit=5.0,
                price_close=105.0, time_close=equity.index[-1],
            ),
        ]
        s = Stats(equity, trades, 1000)
        d = s.to_dict()

        assert d["# Trades"] == 2          # two Trade rows (two partial closes)
        assert d["# Positions"] == 1       # one distinct position_id
        assert d["# Positions Long"] == 1
        assert d["# Positions Short"] == 0

        pos = s.positions
        assert len(pos) == 1
        assert pos.iloc[0]["n_partials"] == 2
        assert pos.iloc[0]["size"] == pytest.approx(2.0)
        assert pos.iloc[0]["pnl_net"] == pytest.approx(10.0)

    def test_mixed_positions_long_and_short(self):
        equity = _make_equity_curve([1005, 1010, 1015], initial_balance=1000)
        trades = [
            # Position 100 (long), closed in two partials.
            Trade(
                ticket=1, position_id=100, symbol="TEST",
                type=OrderType.ORDER_TYPE_BUY,
                volume=1.0, price=100.0, time=equity.index[0],
                commission=0.0, profit=5.0,
                price_close=105.0, time_close=equity.index[1],
            ),
            Trade(
                ticket=2, position_id=100, symbol="TEST",
                type=OrderType.ORDER_TYPE_BUY,
                volume=1.0, price=100.0, time=equity.index[0],
                commission=0.0, profit=5.0,
                price_close=105.0, time_close=equity.index[2],
            ),
            # Position 200 (short), closed fully in one fill.
            Trade(
                ticket=3, position_id=200, symbol="TEST",
                type=OrderType.ORDER_TYPE_SELL,
                volume=1.0, price=100.0, time=equity.index[1],
                commission=0.0, profit=5.0,
                price_close=95.0, time_close=equity.index[-1],
            ),
        ]
        s = Stats(equity, trades, 1000)
        d = s.to_dict()

        assert d["# Trades"] == 3
        assert d["# Positions"] == 2
        assert d["# Positions Long"] == 1
        assert d["# Positions Short"] == 1
        assert len(s.positions) == 2

    def test_missing_position_id_falls_back_to_ticket(self):
        # Connectors without real position tracking (position_id == 0): each
        # trade must still count as its own position, never collapse together.
        equity = _make_equity_curve([1005, 1010], initial_balance=1000)
        trades = [
            Trade(
                ticket=1, position_id=0, symbol="TEST",
                type=OrderType.ORDER_TYPE_BUY,
                volume=1.0, price=100.0, time=equity.index[0],
                commission=0.0, profit=5.0,
                price_close=105.0, time_close=equity.index[1],
            ),
            Trade(
                ticket=2, position_id=0, symbol="TEST",
                type=OrderType.ORDER_TYPE_BUY,
                volume=1.0, price=100.0, time=equity.index[0],
                commission=0.0, profit=5.0,
                price_close=105.0, time_close=equity.index[-1],
            ),
        ]
        s = Stats(equity, trades, 1000)
        d = s.to_dict()

        assert d["# Trades"] == 2
        assert d["# Positions"] == 2  # no shared position_id -> fallback to ticket
        assert d["# Positions Long"] == 2


@pytest.mark.unit
class TestStatsValidation:
    def test_empty_trades_ok(self):
        # A backtest whose position never closed still has an equity curve
        # (broker records balance + floating PnL every bar). compute_stats must
        # tolerate an empty trade list: trade metrics report as N/A / 0, and the
        # equity curve is preserved. This keeps backtest parity with replay,
        # where an open position's equity is always visible.
        equity = _equity_curve([100, 101])
        s = Stats(equity, [], 1000)
        assert s["# Trades"] == 0
        assert s["# Positions"] == 0
        assert s["# Positions Long"] == 0
        assert s["# Positions Short"] == 0
        assert s.trades.empty
        assert len(s.equity_curve) == len(equity)
        assert float(s["Equity Final [$]"]) == pytest.approx(101.0)
        # Distribution metrics are gated to NaN with < min_trades closed trades.
        assert pd.isna(s["Profit Factor"])
        assert pd.isna(s["SQN"])

    def test_close_not_positive_error(self):
        equity = _make_equity_curve([100, 0, 101], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[1],
                    exit_time=equity.index[2],
                    size=1,
                    entry_price=100,
                    exit_price=100,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)

    def test_trade_temporal_inconsistent_error(self):
        equity = _make_equity_curve([101, 102], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[2],
                    exit_time=equity.index[1],
                    size=1,
                    entry_price=100,
                    exit_price=100,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)

    def test_trade_out_of_range_error(self):
        equity = _make_equity_curve([101, 102], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=pd.Timestamp("2023-12-31", tz="UTC"),
                    exit_time=equity.index[1],
                    size=1,
                    entry_price=100,
                    exit_price=100,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)

    def test_exit_out_of_range_error(self):
        equity = _make_equity_curve([101, 102], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[1],
                    exit_time=pd.Timestamp("2024-01-10", tz="UTC"),
                    size=1,
                    entry_price=100,
                    exit_price=100,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)

    def test_trade_closed_without_exit_price_error(self):
        equity = _make_equity_curve([101, 102], initial_balance=1000)
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[1],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=None,
                    direction="long",
                )
            ]
        )
        s = Stats(equity, deals, 1000)

    def test_inputs_como_numpy_y_trades_dataframe(self):
        base_ms = 1_700_000_000_000
        ts = pd.to_datetime([base_ms, base_ms + 86_400_000], unit="ms", utc=True)
        equity = pd.Series([1000, 1010], index=ts, name="close")
        trades = _make_trades([
            {
                "entry_time": ts[0],
                "exit_time": ts[1],
                "size": 1.0,
                "entry_price": 100.0,
                "exit_price": 110.0,
                "direction": "long",
            },
        ])
        s = Stats(equity, trades, 1000)
        d = s.to_dict()
        assert d["Equity Final [$]"] == pytest.approx(1010.0, abs=1e-4)

    def test_initial_balance_negative_error(self):
        equity = _equity_curve([100, 101])
        deals = _make_trades(
            [
                dict(
                    entry_time=equity.index[0],
                    exit_time=equity.index[-1],
                    size=1,
                    entry_price=100,
                    exit_price=101,
                    direction="long",
                )
            ]
        )
        with pytest.raises(ConfigError):
            Stats(equity, deals, -1)



@pytest.mark.unit
class TestLowSampleGating:
    """Gating of unreliable metrics with tiny samples (A+D)."""

    def _long_equity(self, n: int = 30) -> pd.Series:
        # Equity curve of n+1 daily points, with realistic variation.
        rng = np.random.default_rng(42)
        steps = 1.0 + rng.normal(0.001, 0.01, size=n)
        prices = 1000.0 * np.cumprod(steps)
        ts = pd.date_range("2024-01-01", periods=n + 1, freq="1D", tz="UTC")
        return pd.Series([1000.0, *prices.tolist()], index=ts, name="close")

    def test_short_sample_one_trade_nullifies_metrics(self):
        equity = _make_equity_curve([1010], initial_balance=1000)  # 1 day, 1 trade
        deals = _make_trades([
            dict(entry_time=equity.index[1], exit_time=equity.index[-1],
                 size=1, entry_price=100, exit_price=110, direction="long"),
        ])
        with pytest.warns(UserWarning, match="insufficient sample"):
            s = Stats(equity, deals, 1000)
        d = s.to_dict()
        assert d["_low_sample"] is True
        assert s.low_sample is True
        for k in ("Return (Ann.) [%]", "Volatility (Ann.) [%]",
                  "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                  "Profit Factor", "SQN"):
            assert np.isnan(d[k]), f"{k} should be NaN"
        # Always valid metrics:
        assert d["Return [%]"] == pytest.approx(1.0, abs=1e-4)
        assert d["# Trades"] == 1
        assert d["Win Rate [%]"] == pytest.approx(100.0, abs=1e-4)

    def test_sufficient_sample_not_nullified(self):
        equity = self._long_equity(40)  # 40 days > 7
        deals = _make_trades([
            dict(entry_time=equity.index[1], exit_time=equity.index[10],
                 size=1, entry_price=100, exit_price=105, direction="long"),
            dict(entry_time=equity.index[12], exit_time=equity.index[30],
                 size=1, entry_price=105, exit_price=110, direction="long"),
        ])
        s = Stats(equity, deals, 1000)
        d = s.to_dict()
        assert d["_low_sample"] is False
        assert np.isfinite(d["Return (Ann.) [%]"])
        assert np.isfinite(d["Volatility (Ann.) [%]"])

    def test_override_thresholds_disables_gating(self):
        equity = _make_equity_curve([1010], initial_balance=1000)  # 1 day, 1 trade
        deals = _make_trades([
            dict(entry_time=equity.index[1], exit_time=equity.index[-1],
                 size=1, entry_price=100, exit_price=110, direction="long"),
        ])
        s = Stats(equity, deals, 1000,
                  min_trades=0, min_duration_ann=pd.Timedelta(0))
        d = s.to_dict()
        assert d["_low_sample"] is False
        # Without gating, Return (Ann.) is back to a large (finite) number:
        assert np.isfinite(d["Return (Ann.) [%]"])
        assert np.isinf(d["Calmar Ratio"])  # max_dd == 0 → inf, no longer nullified

    def test_few_trades_but_long_duration(self):
        # Long duration (does not nullify annualized) but only 1 trade (nullifies SQN/PF).
        equity = self._long_equity(40)
        deals = _make_trades([
            dict(entry_time=equity.index[1], exit_time=equity.index[30],
                 size=1, entry_price=100, exit_price=110, direction="long"),
        ])
        with pytest.warns(UserWarning, match="trade"):
            s = Stats(equity, deals, 1000)
        d = s.to_dict()
        assert d["_low_sample"] is True
        assert np.isnan(d["Profit Factor"])
        assert np.isnan(d["SQN"])
        # Annualized metrics still valid: duration 40d >= 7d
        assert np.isfinite(d["Return (Ann.) [%]"])


class TestStatsWarnFlag:
    """BacktestEngine.run(stats_warn=) silences the low-sample UserWarning
    without changing the gating itself (metrics stay NaN / _low_sample=True)."""

    def _small_sample_engine(self):
        from tradetropy.models.strategy import Strategy
        from tradetropy.backtest import BacktestEngine
        from tradetropy.core.data_types import KlineData
        from tradetropy.session.base import SeshSimulatorBase

        base = 1_700_000_000_000
        klines = np.array([
            [base + i * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0, 1005.0]
            for i in range(5)
        ], dtype=np.float64)

        class _Once(Strategy):
            def init(self):
                self.k = self.subscribe_ohlc("MES", 60_000, window_size=5)
                self._bought = False

            def on_data(self):
                if not self._bought:
                    self.sesh.buy("MES", volume=1)
                    self._bought = True
                elif self.sesh.positions("MES"):
                    for pos in self.sesh.positions("MES"):
                        self.sesh.position_close(pos.ticket)

        return BacktestEngine.by_klines(
            _Once(),
            data=(KlineData("MES", klines, timeframe=60_000, tick_size=0.01),),
            sesh=SeshSimulatorBase("kline"),
        )

    def test_stats_warn_default_emits_warning(self):
        engine = self._small_sample_engine()
        with pytest.warns(UserWarning, match="insufficient sample"):
            engine.run()
        assert engine.stats["_low_sample"] is True

    def test_stats_warn_false_silences_warning(self):
        engine = self._small_sample_engine()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning now raises -> fails the test
            engine.run(stats_warn=False)
        # The gating itself is unaffected: still flagged and still NaN.
        assert engine.stats["_low_sample"] is True
        assert np.isnan(engine.stats["Sharpe Ratio"])

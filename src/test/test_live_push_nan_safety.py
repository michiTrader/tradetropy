"""
Live-chart push must never carry a non-finite float into Bokeh's JSON message
serializer (which raises "Out of range float values are not JSON compliant:
nan"). These guards keep the websocket push alive when a source value is NaN.
"""

import numpy as np
import pytest

from bokeh.models import ColumnDataSource
from bokeh.core.serialization import Serializer
from bokeh.core.json_encoder import serialize_json

from tradetropy.plotting.live.updater._trades_helpers import _build_trades_data_live


def _json_ok(data: dict) -> bool:
    """True when the data dict survives Bokeh's message JSON serialization."""
    serialize_json(Serializer().encode(data))
    return True


class TestTradesLineNaNSafety:
    def _theme(self):
        return {"trade_win": "#007F5F", "trade_loss": "#AD1D2B"}

    def test_nan_price_trade_dropped(self):
        trades = [
            {"entry_ts_ms": 1000, "exit_ts_ms": 2000,
             "entry_price": 100.0, "exit_price": 101.0, "pnl": 1.0, "direction": "buy"},
            {"entry_ts_ms": 3000, "exit_ts_ms": 4000,
             "entry_price": float("nan"), "exit_price": 102.0, "pnl": 0.0, "direction": "sell"},
        ]
        data = _build_trades_data_live(trades, self._theme(), 60_000)
        # Only the finite trade survives, and no NaN reaches the multiline ys.
        assert len(data["lines_ys"]) == 1
        flat = [v for pair in data["lines_ys"] for v in pair]
        assert all(np.isfinite(v) for v in flat)

    def test_result_is_json_serializable(self):
        trades = [
            {"entry_ts_ms": 3000, "exit_ts_ms": 4000,
             "entry_price": float("nan"), "exit_price": float("nan"),
             "pnl": 0.0, "direction": "sell"},
        ]
        data = _build_trades_data_live(trades, self._theme(), 60_000)
        cds = ColumnDataSource(data)
        assert _json_ok(cds.data)


class TestEquityNaNGuards:
    """The equity/drawdown updaters must skip a non-finite value instead of
    streaming it (which would break the push)."""

    def _fake_updater(self, equity_values):
        """Build a minimal object exposing the equity mixin against a fake broker."""
        import pandas as pd
        from tradetropy.plotting.live.updater._equity_mixin import EquityStatsMixin
        from tradetropy.plotting.config import PlotConfig

        class _Broker:
            initial_balance = 1000.0
            @property
            def equity_curve(self):
                idx = pd.to_datetime(np.arange(len(equity_values)), unit="ms")
                return pd.Series(equity_values, index=idx)

        class _Ref:
            broker = _Broker()
            source = ColumnDataSource({"ts": [], "equity": []})
            trailing_source = None
            config = PlotConfig()

        class _U(EquityStatsMixin):
            _equity_ref = _Ref()
            _max_candles = 100
            _drawdown_source = ColumnDataSource({"ts": [], "equity": [], "drawdown": []})

        return _U()

    def test_equity_skips_non_finite(self):
        u = self._fake_updater([1000.0, float("nan")])
        u._update_equity()   # must not raise, must not stream NaN
        assert _json_ok(u._equity_ref.source.data)
        # The NaN point was skipped (no row streamed for it).
        eq = list(u._equity_ref.source.data.get("equity", []))
        assert all(np.isfinite(v) for v in eq)

    def test_drawdown_skips_non_finite(self):
        # An all-zero equity curve makes drawdown 0/0 -> NaN.
        u = self._fake_updater([0.0, 0.0])
        u._update_drawdown()
        assert _json_ok(u._drawdown_source.data)
        dd = list(u._drawdown_source.data.get("drawdown", []))
        assert all(np.isfinite(v) for v in dd)

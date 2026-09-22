"""
tradetropy.stats._result - Stats (presentation-only result container).

``Stats`` wraps the OrderedDict produced by ``compute_stats()`` (in
``_compute.py``) as a ``pd.Series`` subclass for dict-like access and pandas
integration. It performs no calculation itself.

Note on the compute_stats <-> Stats dependency: ``compute_stats()`` builds a
``Stats`` instance as its return value, while ``Stats.__init__`` can also
compute metrics itself (legacy direct-construction API). That mutual need is
resolved with a lazy import inside ``__init__`` to avoid a circular import
between this module and ``_compute.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd

from tradetropy.stats._compute import MIN_DURATION_ANN, MIN_TRADES_FOR_STATS


class Stats(pd.Series):
    """
    Backtest results container. Presentation and data access only.

    Encapsulates all performance metrics calculated by compute_stats().
    Inherits from pd.Series for dict-like access and pandas integration.

    Preferred construction:
        stats = compute_stats(equity_curve, trades, initial_balance)

    Direct construction (wrapping pre-calculated metrics):
        stats = Stats(ordered_dict)

    Legacy API (deprecated but functional):
        stats = Stats(equity_curve, trades, initial_balance)

    Metric access:
        stats['Sharpe Ratio']        -> float
        stats['# Trades']            -> int (closed trade rows; a position
                                        partially closed several times counts
                                        once per partial close here)
        stats['# Positions']         -> int (distinct closed positions -
                                        partial closes of the same position
                                        merged into one)
        stats['Duration']            -> pd.Timedelta
        stats['Max. Trade Duration'] -> pd.Timedelta
        stats.trades                 -> pd.DataFrame (one row per Trade/close)
        stats.positions              -> pd.DataFrame (one row per position)
        stats.equity_curve           -> pd.Series
        stats.initial_balance        -> float
        stats.freq                   -> str (frequency used for ratios)
        stats.ann_factor             -> float (annualization factor used)

    Inspect the frequency and settings:
        stats.freq        # -> '1min', '5min', '1h' or '1D'
        stats.ann_factor  # -> 97.5, 252, etc.
    """

    _KEYS = (
        "Start",
        "End",
        "Duration",
        "Exposure Time [%]",
        "Equity Final [$]",
        "Equity Peak [$]",
        "Return [%]",
        "Return (Ann.) [%]",
        "Volatility (Ann.) [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max. Drawdown [%]",
        "Avg. Drawdown [%]",
        "Max. Drawdown Duration",
        "Avg. Drawdown Duration",
        "# Trades",
        "# Positions",
        "# Positions Long",
        "# Positions Short",
        "Win Rate [%]",
        "Best Trade [%]",
        "Worst Trade [%]",
        "Avg. Trade [%]",
        "Max. Trade Duration",
        "Avg. Trade Duration",
        "Profit Factor",
        "Expectancy [%]",
        "SQN",
        "Total Commissions [$]",
        "_low_sample",
        "_trades",
        "_positions",
        "_equity_curve",
        "_initial_balance",
        "_strategy",
        "_freq",
        "_ann_factor",
    )

    def __new__(cls, *args, **kwargs):
        return pd.Series.__new__(cls)

    def __init__(
        self,
        equity_curve_or_metrics: "pd.Series | OrderedDict",
        trades: Any = None,
        initial_balance: float = 0.0,
        strategy: Any = None,
        freq: str | None = None,
        *,
        min_trades: int = MIN_TRADES_FOR_STATS,
        min_duration_ann: pd.Timedelta = MIN_DURATION_ANN,
    ):
        if isinstance(equity_curve_or_metrics, OrderedDict):
            metrics = equity_curve_or_metrics
        else:
            # Lazy import: compute_stats() itself returns a Stats instance,
            # so importing it at module scope would create a circular import
            # between _result.py and _compute.py.
            from tradetropy.stats._compute import compute_stats
            metrics = compute_stats(
                equity_curve=equity_curve_or_metrics,
                trades=trades,
                initial_balance=initial_balance,
                strategy=strategy,
                freq=freq,
                min_trades=min_trades,
                min_duration_ann=min_duration_ann,
            )
        pd.Series.__init__(self, data=metrics, name="Stats")

    @property
    def _constructor(self):
        return pd.Series

    @property
    def trades(self) -> pd.DataFrame:
        return self["_trades"]

    @property
    def positions(self) -> pd.DataFrame:
        """
        Closed positions, one row per position_id (partial closes merged).

        A position closed across several partial closes appears as ONE row
        here (entry/exit from the first/last partial, size/pnl_net/commission
        summed, ``n_partials`` counting the partial closes) instead of one row
        per ``Trade``, so this is the frame to use for a per-position view
        that is not inflated by partial closes. See ``# Positions`` /
        ``# Positions Long`` / ``# Positions Short`` for the matching counts.

        Returns:
            pd.DataFrame: See ``build_positions_df`` for the exact schema.
        """
        return self["_positions"]

    @property
    def equity_curve(self) -> pd.Series:
        return self["_equity_curve"]

    @property
    def initial_balance(self) -> float:
        return self["_initial_balance"]

    @property
    def freq(self) -> str:
        """
        Frequency used for return, volatility, and ratio calculations.

        Returns:
            str: Frequency identifier (e.g. '1D', '1h', '5min', '1min')
        """
        return self.get('_freq', '1D')

    @property
    def ann_factor(self) -> float:
        """
        Annualization factor used in Sharpe, Sortino, and volatility metrics.

        Returns:
            float: Factor (e.g. 252 for daily, 252*6.5 for hourly)
        """
        return float(self.get('_ann_factor', 252.0))

    @property
    def low_sample(self) -> bool:
        """
        Indicates whether sample size was insufficient for reliable metrics.

        When True, annualized metrics (Return Ann., Volatility, Sharpe,
        Sortino, Calmar) and/or trade distribution metrics (Profit Factor,
        SQN) are reported as NaN.

        Returns:
            bool: True if metrics were gated due to low sample
        """
        return bool(self.get("_low_sample", False))

    def to_dict(self) -> OrderedDict:
        """
        Export stats to ordered dictionary.

        Returns:
            OrderedDict: All metrics with internal fields (prefixed with '_')
        """
        return OrderedDict(self)

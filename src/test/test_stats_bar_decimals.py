"""
Unit tests for _build_stats_div() decimal formatting of ratio metrics.

Sharpe/Sortino/Calmar/Profit Factor/SQN must always show 2 decimals in the
plot's stats bar, even when abs(value) >= 1 - the generic fallback formatter
used for other numeric stats rounds those to whole numbers otherwise.
"""
from collections import OrderedDict

import pandas as pd
import pytest

from tradetropy.plotting._layout import _build_stats_div
from tradetropy.stats import Stats

pytest.importorskip("bokeh")

_LIGHT_THEME = {"bg": "#FFFFFF", "text": "#000000", "grid_line": "#DDDDDD"}


def _make_stats(**overrides) -> Stats:
    metrics = OrderedDict([
        ("Duration",              pd.Timedelta(days=10)),
        ("Equity Final [$]",      12345.678),
        ("Return [%]",            23.456),
        ("Sharpe Ratio",          2.3456),
        ("Sortino Ratio",         3.0678),
        ("Calmar Ratio",          4.5),
        ("Avg. Drawdown [%]",     -1.2345),
        ("# Trades",              42),
        ("# Positions",           40),
        ("# Positions Long",      25),
        ("# Positions Short",     15),
        ("Win Rate [%]",          55.5),
        ("Avg. Trade [%]",        0.75),
        ("Profit Factor",         1.876),
        ("Expectancy [%]",        0.3),
        ("Total Commissions [$]", -12.5),
    ])
    metrics.update(overrides)
    return Stats(metrics)


def test_ratio_metrics_show_two_decimals():
    stats = _make_stats()
    div = _build_stats_div(stats, _LIGHT_THEME)
    html = div.text

    assert "2.35" in html  # Sharpe Ratio 2.3456 -> 2.35, not "2"
    assert "3.07" in html  # Sortino Ratio 3.0678 -> 3.07, not "3"
    assert "4.50" in html  # Calmar Ratio 4.5 -> 4.50, not "4" or "4.5"
    assert "1.88" in html  # Profit Factor 1.876 -> 1.88, not "2"


def test_non_ratio_integer_like_metrics_stay_unrounded_to_decimals():
    stats = _make_stats()
    div = _build_stats_div(stats, _LIGHT_THEME)
    html = div.text

    # # Trades / # Positions still use the generic ",.0f" fallback (unaffected).
    assert "42" in html
    assert "40" in html


def test_nan_ratio_renders_as_dash():
    import numpy as np
    stats = _make_stats(**{"Sharpe Ratio": np.nan})
    div = _build_stats_div(stats, _LIGHT_THEME)
    assert "—" in div.text


def test_none_stats_returns_empty_bar():
    div = _build_stats_div(None, _LIGHT_THEME)
    assert "stats-bar" in div.text

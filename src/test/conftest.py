import os
import webbrowser
import pytest
import numpy as np
from typing import TYPE_CHECKING


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    """Suprime apertura de navegador en tests de plotting."""
    monkeypatch.setattr(webbrowser, "open", lambda *a, **kw: None)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — small data (function, recreated per test)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def klines_1m():
    """13 candles of 1 minute. base_ts=1700000000000 = 2023-11-14T22:13:20 UTC."""
    return np.array([
        [1700000000000, 42500.5, 42510.2, 42495.1, 42505.8, 12.45, 529200.34],
        [1700000060000, 42505.8, 42520.0, 42501.3, 42515.6,  9.82, 417800.91],
        [1700000120000, 42515.6, 42518.9, 42490.4, 42498.2, 15.03, 638500.77],
        [1700000180000, 42498.2, 42505.7, 42485.9, 42490.6, 11.67, 496300.12],
        [1700000240000, 42490.6, 42508.1, 42488.0, 42502.4,  8.94, 380100.55],
        [1700000300000, 42502.4, 42530.0, 42498.6, 42525.9, 14.21, 604900.88],
        [1700000360000, 42525.9, 42540.2, 42510.5, 42518.3, 10.76, 457600.41],
        [1700000420000, 42518.3, 42522.8, 42495.7, 42501.1, 13.59, 577800.66],
        [1700000480000, 42501.1, 42515.0, 42492.3, 42508.7,  9.33, 396500.29],
        [1700000540000, 42508.7, 42535.6, 42505.4, 42530.2, 16.48, 700200.73],
        [1700000600000, 42530.2, 42545.9, 42520.1, 42540.8, 12.91, 549600.18],
        [1700000660000, 42540.8, 42548.3, 42525.6, 42533.4,  7.84, 333900.44],
        [1700000720000, 42533.4, 42560.0, 42530.2, 42555.9, 18.27, 777300.95],
    ], dtype=np.float64)


@pytest.fixture
def klines_5m():
    """4 candles of 5 minutes."""
    return np.array([
        [1699999800000, 42500.5, 42520.0, 42495.1, 42515.6, 22.27, 947001.25],
        [1700000120000, 42515.6, 42540.2, 42485.9, 42518.3, 60.61, 2577402.73],
        [1700000420000, 42518.3, 42548.3, 42492.3, 42533.4, 60.15, 2558002.30],
        [1700000720000, 42533.4, 42560.0, 42530.2, 42555.9, 18.27,  777300.95],
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — large data (session, generated once per session)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def klines_20k():
    """20,000 synthetic 1m candles. Fixed seed for reproducibility."""
    rng = np.random.default_rng(0)
    n   = 20_000
    p   = 50_000 + np.cumsum(rng.standard_normal(n) * 5)
    c   = p + rng.standard_normal(n) * 2
    h   = np.maximum(p, c) + rng.uniform(0, 5, n)
    lo  = np.minimum(p, c) - rng.uniform(0, 5, n)
    vol = rng.uniform(1, 50, n)
    ts  = 1_700_000_000_000 + np.arange(n) * 60_000
    return np.column_stack([ts, p, h, lo, c, vol, vol * c])


@pytest.fixture(scope="session")
def klines_100k():
    """100,000 synthetic 1m candles. Fixed seed for reproducibility."""
    rng = np.random.default_rng(1)
    n   = 100_000
    p   = 50_000 + np.cumsum(rng.standard_normal(n) * 5)
    c   = p + rng.standard_normal(n) * 2
    h   = np.maximum(p, c) + rng.uniform(0, 5, n)
    lo  = np.minimum(p, c) - rng.uniform(0, 5, n)
    vol = rng.uniform(1, 50, n)
    ts  = 1_700_000_000_000 + np.arange(n) * 60_000
    return np.column_stack([ts, p, h, lo, c, vol, vol * c])

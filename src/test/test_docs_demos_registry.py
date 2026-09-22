"""
Tests for the documentation-demo registry (tools/docs_demos).

These guard the contract the docs generator relies on: the registry imports,
every declared demo builds by executing its canonical snippet, and the
sma_cross demo runs and produces non-empty, non-degraded stats (no
"insufficient sample" warning, low_sample False).
"""

import sys
from pathlib import Path

import pytest

# The demo registry lives under the repo's tools/ dir, so it is not importable
# as tradetropy.*. Put tools/ on the path exactly like tools/gen_docs_demo.py
# does when run as a script.
_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from docs_demos import DEMOS, get_demo  # noqa: E402


def test_registry_imports_and_is_non_empty():
    assert DEMOS, "the demo registry must not be empty"
    ids = [d.demo_id for d in DEMOS]
    assert len(ids) == len(set(ids)), f"duplicate demo ids: {ids}"


def test_get_demo_roundtrip_and_unknown():
    for spec in DEMOS:
        assert get_demo(spec.demo_id) is spec
    with pytest.raises(KeyError):
        get_demo("does-not-exist")


def test_sma_cross_demo_is_registered():
    spec = get_demo("sma_cross")
    # The snippet is the canonical source shown in docs: it must use the daily
    # GOOG dataset and the Signal crossover/crossunder API (per the demo spec).
    assert "load_goog_1d" in spec.snippet
    assert "Signal" in spec.snippet
    assert "crossover" in spec.snippet
    assert "crossunder" in spec.snippet


def test_sma_cross_demo_builds_with_clean_stats():
    result = get_demo("sma_cross").build()
    assert result.engine is not None
    assert result.stats is not None
    # Not degraded: no insufficient-sample gating.
    assert not result.low_sample
    assert not result.degraded, result.degraded_reason
    # Produced real trades and a finite final equity.
    n_trades = int(result.stats.get("# Trades", 0))
    assert n_trades > 0, "demo produced no trades"
    equity_final = float(result.stats.get("Equity Final [$]"))
    assert equity_final > 0


def test_degraded_reason_reflects_low_sample():
    result = get_demo("sma_cross").build()
    # A clean demo has an empty degraded reason.
    assert result.degraded_reason == ""

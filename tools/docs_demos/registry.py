"""
Declarative registry of documentation demos (see package docstring).

Add a demo by appending a ``DemoSpec`` to ``DEMOS``: give it an ``id``, a human
``title``/``description`` and the canonical ``snippet``. The snippet MUST be a
complete, copy-paste-runnable
script that leaves a finished engine bound to ``engine_var`` (default
``"bt"``) in its own namespace. Everything else (chart, stats, snippet
file) is derived from that single source by ``tools/gen_docs_demo.py``.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class DemoResult:
    """
    Outcome of building (executing) a demo snippet.

    Attributes:
        engine: The finished engine object (already ``run()``), used to render
            the chart and read the stats.
        stats: The engine's ``stats`` object (or None if the engine exposes
            none).
        warnings (list[str]): Warning messages emitted while the snippet ran.
        low_sample (bool): The stats' ``low_sample`` flag - True when the
            sample was too small and annualized / trade-distribution metrics
            were zeroed to NaN.
    """

    engine: Any
    stats: Any
    warnings: List[str]
    low_sample: bool

    @property
    def degraded(self) -> bool:
        """
        True when the demo's stats are not fit to publish.

        A demo is degraded if the sample was insufficient - either the stats'
        ``low_sample`` flag is set, or an "insufficient sample" warning was
        emitted while running. The generator refuses to write artifacts for a
        degraded demo (a published ``stats.txt`` full of NaN/zero metrics would
        be worse than no demo at all).

        Returns:
            bool: True if the demo must not be published as-is.
        """
        if self.low_sample:
            return True
        return any("insufficient sample" in w.lower() for w in self.warnings)

    @property
    def degraded_reason(self) -> str:
        """Return a human-readable reason the demo is degraded, or ''."""
        reasons: List[str] = []
        if self.low_sample:
            reasons.append("stats.low_sample is True")
        insufficient = [w for w in self.warnings if "insufficient sample" in w.lower()]
        reasons.extend(insufficient)
        return "; ".join(reasons)


@dataclass(frozen=True)
class DemoSpec:
    """
    A single documentation demo.

    Attributes:
        demo_id (str): Short lowercase id, used as the artifact folder name
            (``docs/assets/demos/<demo_id>/``).
        title (str): Human title shown in the docs section.
        description (str): One-line description of what the demo shows.
        snippet (str): The canonical, copy-paste-runnable source. Must bind the
            finished engine to ``engine_var``.
        engine_var (str): Name the snippet binds the engine to (default
            ``"engine"``).
    """

    demo_id: str
    title: str
    description: str
    snippet: str
    engine_var: str = "bt"

    def build(self) -> DemoResult:
        """
        Execute the demo snippet and return its result.

        Runs the exact ``snippet`` in a fresh namespace (capturing any
        warnings), then reads the finished engine bound to ``engine_var``.

        Returns:
            DemoResult: The engine, its stats, captured warnings and the
            low-sample flag.

        Raises:
            RuntimeError: If the snippet did not bind ``engine_var`` to a
                non-None object.
        """
        namespace: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            code = compile(self.snippet, f"<demo:{self.demo_id}>", "exec")
            # The canonical snippet ends with a print(bt.stats) so it is a
            # complete copy-paste script for the user; swallow that stdout here
            # so building demos does not spam the generator/CI logs.
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace)  # noqa: S102 - trusted, in-repo demo source
            messages = [str(w.message) for w in caught]

        engine = namespace.get(self.engine_var)
        if engine is None:
            raise RuntimeError(
                f"demo {self.demo_id!r}: snippet did not bind "
                f"{self.engine_var!r} to an engine"
            )
        stats = getattr(engine, "stats", None)
        low_sample = bool(getattr(stats, "low_sample", False))
        return DemoResult(
            engine=engine,
            stats=stats,
            warnings=messages,
            low_sample=low_sample,
        )


# -----
# Demo snippets (canonical source of truth)
# -----

_SMA_CROSS_SNIPPET = '''\
from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_goog_1d
from tradetropy.signal import Signal
from tradetropy.ta import SMA


class SmaCross(Strategy):
    """Go long on a fast/slow SMA crossover, flatten on the crossunder."""

    def init(self):
        self.goog = self.subscribe_ohlc('GOOG', '1d', window_size=200)
        self.fast = self.add_indicator(self.goog.close, SMA(10))
        self.slow = self.add_indicator(self.goog.close, SMA(30))
        self.signal = Signal('partial')

    def on_data(self):
        if self.signal.crossover(self.fast, self.slow):
            if not self.sesh.positions('GOOG'):
                self.sesh.buy('GOOG', volume=10)
        elif self.signal.crossunder(self.fast, self.slow):
            for pos in self.sesh.positions('GOOG'):
                self.sesh.position_close(pos.ticket)


bt = BacktestEngine.by_klines(SmaCross(), data=(load_goog_1d(),))
bt.run()
print(bt.stats)
'''


# -----
# The registry
# -----

DEMOS: List[DemoSpec] = [
    DemoSpec(
        demo_id="sma_cross",
        title="SMA crossover",
        description=(
            "A fast/slow moving-average crossover on daily GOOG candles, using "
            "Signal.crossover / crossunder for entries and exits."
        ),
        snippet=_SMA_CROSS_SNIPPET,
    ),
]


def get_demo(demo_id: str) -> DemoSpec:
    """
    Return the demo spec with the given id.

    Args:
        demo_id (str): The demo's id.

    Returns:
        DemoSpec: The matching spec.

    Raises:
        KeyError: If no demo has that id.
    """
    for demo in DEMOS:
        if demo.demo_id == demo_id:
            return demo
    raise KeyError(
        f"Unknown demo {demo_id!r}. Available: {[d.demo_id for d in DEMOS]}"
    )

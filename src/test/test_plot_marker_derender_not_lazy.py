"""
Marker series (scatter) must blank only while panning, never be governed by the
zoom-threshold lazy-label hide.

Regression for the bug where an indicator's scatter glyphs (e.g. an HTF NBS's
triangles) were collected into the SAME list that ``_configure_lazy_labels``
uses for text labels. Two visible failures resulted:

  1. Zoomed out, the lazy-label callback (fires on every x_range change) hid the
     markers because the candle count exceeded ``labels_zoom_range`` and kept
     re-hiding them on each pan tick, overriding the derender's PanEnd restore -
     so the markers stayed invisible after the drag ended.
  2. A marker series owns a legend entry, so toggling its ``visible`` on every
     range tick made that legend entry flicker on/off continuously.

The fix keeps two separate lists: text labels feed ``_configure_lazy_labels``
(zoom hide) AND the derender (pan blank); marker series feed ONLY the derender.
This test asserts that separation on a fully built chart:

  - the lazy-labels CustomJS carries NO GlyphRenderer (only text labels), and
  - the indicator scatter renderer IS registered in the derender CustomJS.
"""
import numpy as np
import pytest

pytest.importorskip("bokeh")

from bokeh.models import CustomJS, GlyphRenderer, Scatter

from tradetropy import BacktestEngine, Strategy
from tradetropy.core.data_types import KlineData
from tradetropy.ta import NBS
import tradetropy.plotting.plotting as plotting_mod

_T0 = 1_700_000_000_000


def _klines(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    m = np.zeros((n, 7), dtype=np.float64)
    m[:, 0] = _T0 + np.arange(n) * 60_000.0
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    m[:, 1] = open_
    m[:, 2] = np.maximum(open_, close) + rng.uniform(0, 0.4, n)
    m[:, 3] = np.minimum(open_, close) - rng.uniform(0, 0.4, n)
    m[:, 4] = close
    m[:, 5] = rng.uniform(1, 10, n)
    return KlineData(symbol="SYM", data=m, timeframe="1m", tick_size=0.01)


class _HtfStrat(Strategy):
    """1m chart with a 5m NBS overlay (scatter markers with a legend entry)."""

    def init(self):
        self.o1 = self.subscribe_ohlc("SYM", "1m", window_size=300)
        self.o5 = self.subscribe_ohlc("SYM", "5m", window_size=300)
        self.nbs_5m = self.add_indicator(
            [self.o5.high_ref, self.o5.low_ref, self.o5.ts_ref],
            NBS(swing=2),
            name="nbs_5m",
        )

    def on_data(self):
        pass


def _build_layout(monkeypatch):
    strat = _HtfStrat()
    eng = BacktestEngine.by_klines(strat, data=(_klines(),))
    eng.run()

    captured = {}
    monkeypatch.setattr(
        plotting_mod, "_show_layout",
        lambda layout, config, theme: captured.setdefault("layout", layout),
    )
    plotting_mod.plot(eng, output="file", filename="_marker_derender_test.html")
    return captured["layout"]


def _all_customjs(layout):
    return [m for m in layout.references() if isinstance(m, CustomJS)]


def _lazy_labels_cb(layout):
    # The lazy-labels CustomJS is the one carrying both ``labels`` and
    # ``zoom_range`` in its args (see lazy_labels.js).
    return [cb for cb in _all_customjs(layout)
            if "labels" in cb.args and "zoom_range" in cb.args]


def _derender_cbs(layout):
    # derender.js callbacks carry ``renderers`` and a ``kind`` discriminator.
    return [cb for cb in _all_customjs(layout)
            if "renderers" in cb.args and "kind" in cb.args]


def _scatter_renderers(layout):
    return [r for r in layout.references()
            if isinstance(r, GlyphRenderer) and isinstance(r.glyph, Scatter)]


@pytest.mark.unit
class TestMarkerDerenderNotLazy:
    def test_scatter_marker_not_governed_by_lazy_labels(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        layout = _build_layout(monkeypatch)

        scatters = {id(s) for s in _scatter_renderers(layout)}
        assert scatters, "the 5m NBS should have produced scatter markers"

        # With the pre-fix shared list, the scatter markers were appended to the
        # lazy-label entries, so a lazy-labels CustomJS would exist AND list them
        # as ``labels`` (hidden past the zoom threshold). After the fix the
        # marker series never reaches the lazy-label list: either no lazy
        # callback is wired (no text labels here) or it carries only real labels.
        for cb in _lazy_labels_cb(layout):
            leaked = [lbl for lbl in cb.args["labels"] if id(lbl) in scatters]
            assert not leaked, (
                "scatter marker(s) leaked into the lazy-label list; they would "
                "be zoom-hidden and fight the pan-restore"
            )

    def test_scatter_marker_is_registered_in_derender(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        layout = _build_layout(monkeypatch)

        scatters = _scatter_renderers(layout)
        assert scatters, "the 5m NBS should have produced scatter markers"

        derender = _derender_cbs(layout)
        assert derender, "expected derender CustomJS to be wired"
        registered = set()
        for cb in derender:
            registered.update(id(r) for r in cb.args["renderers"])

        assert any(id(s) in registered for s in scatters), (
            "the NBS scatter markers must be registered in the derender so they "
            "blank while panning and restore when it stops"
        )

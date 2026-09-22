"""
Tests for the documentation-demo artifact generator (tools/gen_docs_demo.py).

Guards: generating the sma_cross demo writes the expected artifacts derived
from a single snippet execution (snippet.py verbatim, a clean public stats.txt,
an interactive chart.html), the PNG is a soft/optional artifact, and a DEGRADED
demo (insufficient sample) is refused fail-closed with no artifacts written.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import gen_docs_demo  # noqa: E402
from docs_demos import get_demo  # noqa: E402
from docs_demos.registry import DemoSpec  # noqa: E402


# A snippet that runs a real backtest but never trades, so the stats are
# degraded (0 closed trades < MIN_TRADES_FOR_STATS -> low_sample / insufficient
# sample warning). Used to exercise the fail-closed validation.
_DEGRADED_SNIPPET = '''\
from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_goog_1d


class DoNothing(Strategy):
    def init(self):
        self.goog = self.subscribe_ohlc('GOOG', '1d', window_size=50)

    def on_data(self):
        pass


bt = BacktestEngine.by_klines(DoNothing(), data=(load_goog_1d(),))
bt.run()
'''


def test_generate_sma_cross_writes_artifacts(tmp_path):
    spec = get_demo("sma_cross")
    out_dir = gen_docs_demo.generate_demo(spec, tmp_path)

    snippet = out_dir / "snippet.py"
    stats = out_dir / "stats.txt"
    chart = out_dir / "chart.html"

    assert snippet.is_file()
    assert stats.is_file()
    assert chart.is_file()

    # snippet.py is the canonical source, verbatim.
    assert snippet.read_text(encoding="utf-8") == spec.snippet

    # stats.txt carries the public metrics and NONE of the internal _rows.
    stats_text = stats.read_text(encoding="utf-8")
    assert "# Trades" in stats_text
    assert "Return [%]" in stats_text
    assert "_trades" not in stats_text
    assert "_equity_curve" not in stats_text

    # chart.html is a real Bokeh document.
    html = chart.read_text(encoding="utf-8")
    assert "BokehJS" in html or "bokeh" in html.lower()


def test_generate_all_demos(tmp_path):
    # The generator writes the sma_cross folder when generating all demos.
    written = gen_docs_demo.generate(tmp_path)
    assert (tmp_path / "sma_cross").is_dir()
    assert any(p.name == "sma_cross" for p in written)


def test_degraded_demo_is_refused_and_writes_nothing(tmp_path):
    spec = DemoSpec(
        demo_id="degraded_probe",
        title="Degraded probe",
        description="A no-trade backtest with insufficient sample.",
        snippet=_DEGRADED_SNIPPET,
    )
    # Sanity: the demo really is degraded.
    assert spec.build().degraded

    with pytest.raises(gen_docs_demo.DegradedDemoError):
        gen_docs_demo.generate_demo(spec, tmp_path)

    # Fail-closed: no artifact directory was created for the degraded demo.
    assert not (tmp_path / "degraded_probe").exists()


def test_write_png_falls_back_to_selenium_manager_driver(monkeypatch, tmp_path):
    """
    If Bokeh's own PATH-based driver lookup fails, _write_png retries with a
    driver resolved via _resolve_chrome_webdriver (Selenium Manager) before
    giving up. Both export_png calls and the fallback driver are mocked so
    this does not depend on a real browser being installed.
    """
    calls = []

    def fake_export_png(layout, filename=None, webdriver=None):
        calls.append(webdriver)
        if webdriver is None:
            raise RuntimeError("no driver on PATH")
        Path(filename).write_bytes(b"fake-png")
        return filename

    fake_driver = mock.Mock()
    monkeypatch.setattr(gen_docs_demo, "_resolve_chrome_webdriver", lambda: fake_driver)

    import bokeh.io as bokeh_io
    monkeypatch.setattr(bokeh_io, "export_png", fake_export_png)

    ok = gen_docs_demo._write_png(object(), tmp_path / "chart.png", "probe")

    assert ok is True
    assert (tmp_path / "chart.png").read_bytes() == b"fake-png"
    assert calls == [None, fake_driver]
    fake_driver.quit.assert_called_once()


def test_write_png_skips_with_warning_when_no_driver_available(monkeypatch, tmp_path):
    """When neither Bokeh's lookup nor the Selenium Manager fallback work, the
    PNG is skipped with a warning and no file is written."""
    import bokeh.io as bokeh_io

    def always_fail(layout, filename=None, webdriver=None):
        raise RuntimeError("no driver on PATH")

    monkeypatch.setattr(bokeh_io, "export_png", always_fail)
    monkeypatch.setattr(gen_docs_demo, "_resolve_chrome_webdriver", lambda: None)

    with pytest.warns(UserWarning, match="PNG export skipped"):
        ok = gen_docs_demo._write_png(object(), tmp_path / "chart.png", "probe")

    assert ok is False
    assert not (tmp_path / "chart.png").exists()

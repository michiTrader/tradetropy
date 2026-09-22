"""
Generate the documentation-demo artifacts from the demo registry.

For every demo in the registry (``tools/docs_demos/registry.py``), this
builds the demo by running its canonical snippet and writes, under
``docs/assets/demos/<demo_id>/``:

  * ``snippet.py`` - the exact, copy-paste-runnable source (single source of
    truth; the docs include this file verbatim);
  * ``stats.txt``  - the backtest's public performance metrics;
  * ``chart.html`` - the interactive Bokeh chart;
  * ``chart.png``  - a static fallback image (see the webdriver note below).

Because the chart, stats and snippet all derive from ONE execution of the same
snippet, they can never drift from one another or from what the user reads.

Validation (fail-closed): a demo whose stats are DEGRADED (insufficient sample:
``stats.low_sample`` set, or an "insufficient sample" warning) is NOT written -
the generator raises and fails the build. Publishing a ``stats.txt`` full of
NaN/zero metrics would be worse than shipping no demo at all.

PNG note: ``chart.png`` is exported with Bokeh's ``export_png``, which needs a
Selenium webdriver (Chrome/Firefox headless) and Pillow. When no webdriver is
available the PNG is skipped with a warning (the HTML / stats / snippet are
still written); it never fails the whole generator. Install the export stack
with the ``docs-export`` dev tooling (bokeh + selenium + pillow + a driver).

Usage:
    uv run python tools/gen_docs_demo.py                 # all demos
    uv run python tools/gen_docs_demo.py --demo sma_cross
    uv run python tools/gen_docs_demo.py --out docs/assets/demos
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List

# This tool lives under tools/; make the sibling demo registry importable when
# run as a script.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from docs_demos import DemoSpec, DemoResult, get_demo, DEMOS  # noqa: E402

# Default output root (relative to the repo root).
_DEFAULT_OUT = Path("docs") / "assets" / "demos"

# Plot options for the generated chart. plot_pl shows the per-trade P&L panel
# (the backtesting.py-style markers); a light theme reads well embedded in the
# docs. Kept here (not in the registry) because they are a rendering concern,
# not part of the demo's canonical source.
_PLOT_KWARGS = dict(
    theme="light",
    width=1000,
    plot_pl=True,
    plot_drawdown=True,
    # Pin the CPU canvas backend for the export: the interactive default is now
    # WebGL, but headless PNG export must stay deterministic (WebGL rendering
    # under a headless browser can vary or blank the canvas on some drivers).
    output_backend="canvas",
)


class DegradedDemoError(RuntimeError):
    """Raised when a demo's stats are not fit to publish (insufficient sample)."""


def _public_stats_text(result: DemoResult) -> str:
    """
    Render the demo's PUBLIC performance metrics as aligned text.

    Drops the internal ``_``-prefixed rows (``_trades``, ``_equity_curve``,
    ``_strategy`` ...) that the raw ``str(stats)`` would dump, leaving only the
    human-facing metrics for ``stats.txt``.

    Args:
        result (DemoResult): The built demo.

    Returns:
        str: The formatted public stats block.
    """
    stats = result.stats
    public_index = [k for k in stats.index if not str(k).startswith("_")]
    return stats[public_index].to_string()


def _render_layout(engine):
    """
    Build the Bokeh layout for an already-run engine without opening a browser.

    Monkeypatches ``bokeh.plotting.show`` (which the plotting code imports at
    call time inside ``_show_layout``) to CAPTURE the assembled layout instead
    of displaying it, then calls ``engine.plot(output="show", ...)``. The
    returned layout is the full themed column (CSS + figures), ready to hand to
    ``save`` / ``export_png``.

    Args:
        engine: A backtest engine after ``run()``.

    Returns:
        The assembled Bokeh layout object.

    Raises:
        RuntimeError: If plotting did not produce a layout.
    """
    import bokeh.plotting as bkp

    captured: dict = {}
    original_show = bkp.show

    def _capture(obj, *args, **kwargs):
        captured["layout"] = obj

    bkp.show = _capture
    try:
        engine.plot(output="show", **_PLOT_KWARGS)
    finally:
        bkp.show = original_show

    layout = captured.get("layout")
    if layout is None:
        raise RuntimeError("engine.plot() did not produce a layout to capture")
    return layout


def _write_html(layout, path: Path, title: str) -> None:
    """Save the interactive chart to a self-contained-ish HTML file (CDN JS)."""
    from bokeh.io import reset_output, save
    from bokeh.resources import CDN

    reset_output()
    save(layout, filename=str(path), title=title, resources=CDN)


def _resolve_chrome_webdriver():
    """
    Build a headless Chrome WebDriver via Selenium Manager as a fallback.

    Bokeh's own webdriver lookup only checks for a ``chromedriver``/
    ``geckodriver`` binary on the system PATH, so a machine with only Chrome
    installed (Selenium 4.6+ resolves the matching driver itself, on demand,
    via "Selenium Manager") still fails Bokeh's check. Building the driver
    directly and handing it to ``export_png(webdriver=...)`` sidesteps that.

    Returns:
        selenium.webdriver.Chrome | None: A headless driver, or None if
        Selenium/Chrome are not usable either.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless=new")
        return webdriver.Chrome(options=options)
    except Exception:  # noqa: BLE001 - any failure just means no fallback
        return None


def _write_png(layout, path: Path, demo_id: str) -> bool:
    """
    Export a static PNG of the chart. Skipped (with a warning) if it cannot.

    ``export_png`` needs a Selenium webdriver and Pillow. Bokeh's own lookup
    only finds a ``chromedriver``/``geckodriver`` binary on PATH; if that
    fails, a headless Chrome driver resolved via Selenium Manager is tried as
    a fallback before giving up. A missing driver/browser altogether is a
    soft failure: the PNG is skipped but the rest of the artifacts still ship.

    Args:
        layout: The Bokeh layout to render.
        path (Path): Destination PNG path.
        demo_id (str): Demo id (for the warning message).

    Returns:
        bool: True if the PNG was written, False if it was skipped.
    """
    from bokeh.io import export_png

    try:
        export_png(layout, filename=str(path))
        return True
    except Exception as first_exc:  # noqa: BLE001 - fall back before warning
        driver = _resolve_chrome_webdriver()
        if driver is None:
            warnings.warn(
                f"demo {demo_id!r}: PNG export skipped "
                f"({type(first_exc).__name__}: {first_exc}). Install a "
                f"Selenium webdriver + pillow to generate {path.name}.",
                stacklevel=2,
            )
            return False
        try:
            export_png(layout, filename=str(path), webdriver=driver)
            return True
        except Exception as exc:  # noqa: BLE001 - any driver/render error is soft
            warnings.warn(
                f"demo {demo_id!r}: PNG export skipped "
                f"({type(exc).__name__}: {exc}). Install a Selenium webdriver "
                f"+ pillow to generate {path.name}.",
                stacklevel=2,
            )
            return False
        finally:
            driver.quit()


def generate_demo(spec: DemoSpec, out_root: Path) -> Path:
    """
    Build one demo and write its artifacts under ``out_root/<demo_id>/``.

    Args:
        spec (DemoSpec): The demo to generate.
        out_root (Path): The demos root (e.g. ``docs/assets/demos``).

    Returns:
        Path: The demo's output directory.

    Raises:
        DegradedDemoError: If the demo's stats are degraded (insufficient
            sample); no artifacts are written in that case.
    """
    result = spec.build()
    if result.degraded:
        raise DegradedDemoError(
            f"demo {spec.demo_id!r} is degraded and will not be published: "
            f"{result.degraded_reason}"
        )

    out_dir = out_root / spec.demo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Canonical snippet (single source of truth).
    (out_dir / "snippet.py").write_text(spec.snippet, encoding="utf-8")

    # 2. Public stats.
    (out_dir / "stats.txt").write_text(
        _public_stats_text(result) + "\n", encoding="utf-8"
    )

    # 3. Interactive chart + static fallback.
    layout = _render_layout(result.engine)
    _write_html(layout, out_dir / "chart.html", title=f"Tradetropy - {spec.title}")
    png_ok = _write_png(layout, out_dir / "chart.png", spec.demo_id)

    wrote = "html+stats+snippet" + ("+png" if png_ok else " (png skipped)")
    print(f"  demo {spec.demo_id!r}: {wrote} -> {out_dir}")
    return out_dir


def generate(
    out_root: Path,
    *,
    demo_id: str | None = None,
) -> List[Path]:
    """
    Generate the artifacts for a set of demos.

    Args:
        out_root (Path): The demos root directory.
        demo_id (str | None): If given, generate only this demo. None generates
            every registered demo.

    Returns:
        list[Path]: The output directories written.

    Raises:
        DegradedDemoError: If any selected demo is degraded.
        KeyError: If ``demo_id`` is unknown.
    """
    if demo_id is not None:
        specs = [get_demo(demo_id)]
    else:
        specs = list(DEMOS)

    scope = f"demo {demo_id!r}" if demo_id is not None else "all demos"
    print(f"Generating docs demos ({scope}): {[s.demo_id for s in specs]}")

    written: List[Path] = []
    for spec in specs:
        written.append(generate_demo(spec, out_root))
    return written


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate documentation-demo artifacts.")
    ap.add_argument(
        "--demo", type=str, default=None,
        help="Generate only this demo id (default: all demos).",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help=f"Output demos root (default: {_DEFAULT_OUT}).",
    )
    ap.add_argument(
        "--repo-root", type=Path, default=Path.cwd(),
        help="Repo root the default --out is resolved against (default: cwd).",
    )
    args = ap.parse_args(argv)

    out_root = args.out if args.out is not None else (args.repo_root / _DEFAULT_OUT)

    try:
        generate(out_root, demo_id=args.demo)
    except DegradedDemoError as exc:
        print(f"VALIDATION FAILED - {exc}")
        return 1
    print("Docs demos generated OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

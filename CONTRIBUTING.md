# Contributing to Tradetropy

Thank you for your interest in contributing to Tradetropy!

## Development Setup

```bash
git clone https://github.com/michiTrader/tradetropy.git
cd tradetropy
uv sync
```

## Running Tests

```bash
uv run pytest                        # all tests
uv run pytest -m unit                # only unit tests
uv run pytest src/test/test_stats.py # single file
```

## Code Style

- Code, docstrings, and comments in **English**
- Follow existing patterns in the codebase
- Use type hints where practical
- Keep imports organized (stdlib, third-party, local)

## Maintaining the documentation demos

The interactive chart embedded on the docs landing page and the Examples page
(`docs/index.md`, `docs/examples.md`) is not hand-written - it is generated
from a single canonical snippet declared in `tools/docs_demos/registry.py`.
The snippet is executed once and its output produces four artifacts under
`docs/assets/demos/<demo_id>/`: `snippet.py` (the exact source shown in the
docs), `stats.txt` (the public performance metrics), `chart.html` (the
interactive Bokeh chart) and `chart.png` (a static fallback used by
`README.md`). Because all four come from one execution, they can never drift
from each other.

**Adding or changing a demo**

1. Edit (or add) a `DemoSpec` entry in `tools/docs_demos/registry.py`. The
   `snippet` field must be a complete, copy-paste-runnable script that ends
   with a finished `engine` bound in its own namespace (`engine.run()`
   already called).
2. Regenerate the artifacts:
   ```bash
   uv run python tools/gen_docs_demo.py --demo <demo_id>   # one demo
   uv run python tools/gen_docs_demo.py                    # all demos
   ```
3. The generator FAILS THE BUILD (raises, writes nothing) if the demo's stats
   are degraded - an insufficient sample (too few trades or too short a
   duration; see `Stats.low_sample`). This is intentional: a demo must never
   publish NaN/zero metrics. Pick a dataset/period that produces enough trades
   instead of silencing the check.
4. Commit the regenerated files under `docs/assets/demos/<demo_id>/` along
   with the registry change; `docs/index.md` / `docs/examples.md` include them
   via `pymdownx.snippets` (`--8<--`) and an `<iframe>`, so nothing else needs
   editing unless you add a brand-new demo id (then add its own tab/section).

**The PNG fallback (`chart.png`) needs a browser**

`chart.png` is exported with Bokeh's `export_png`, which needs a Selenium
webdriver and Pillow (the optional `docs-export` extra) plus an actual browser
installed:

```bash
uv sync --extra docs --extra docs-export
```

Install `docs` and `docs-export` TOGETHER in one `uv sync` call - `uv sync
--extra <name>` on its own is not additive with previously installed extras
and will silently drop `mkdocs`/`pymdown-extensions` if run alone. The
generator first tries Bokeh's own driver lookup (a `chromedriver`/
`geckodriver` binary on `PATH`); if that fails it falls back to a headless
Chrome driver resolved by Selenium's built-in "Selenium Manager"
(`selenium>=4.6`), so having Chrome installed is enough - no manual
chromedriver download needed. If neither is available, the PNG is skipped
with a warning and the rest of the artifacts (`chart.html`, `stats.txt`,
`snippet.py`) are still produced - only `README.md`'s static preview image
goes stale until the PNG is regenerated on a machine with a browser.

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request with a clear description

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

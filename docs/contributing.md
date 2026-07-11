# Contributing

Thanks for your interest in contributing to Tradetropy!

## Development setup

```bash
git clone https://github.com/michiTrader/tradetropy.git
cd tradetropy
uv sync            # project + dev tooling (pytest)
```

## Running tests

```bash
uv run pytest                          # all tests
uv run pytest src/test/test_stats.py   # a single file
```

## Building the docs

```bash
pip install tradetropy[docs]
mkdocs serve                           # live preview at http://127.0.0.1:8000
mkdocs build                           # static site into site/
```

## Code style

- Code, docstrings and comments in English.
- Single quotes for strings; a simple dash (`-`), never an em dash.
- Google-style docstrings with `Args:` / `Returns:` / `Raises:`.
- Follow the existing patterns; keep imports organized (stdlib, third-party,
  local).

## Pull requests

1. Create a feature branch.
2. Make your changes and add tests.
3. Run the test suite and ensure it passes.
4. Open a pull request with a clear description.

By contributing you agree that your contributions are licensed under the MIT
License.

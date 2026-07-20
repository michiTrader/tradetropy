# Installation

Tradetropy requires **Python 3.12+**.

## Base install

```bash
pip install tradetropy
```

The base package already includes the full backtesting engine, the indicator
library, the plotting stack (Bokeh) and data IO for CSV and the NumPy `.npz`
binary format. `.npz` is the base binary format: it needs no compiler and works
everywhere NumPy does (including Termux/Android), so you do **not** need any
extra to save, load or record data.

## Optional extras

Extras add broker/venue integrations and optional file formats. Install only
what you need:

```bash
pip install tradetropy[mt5]       # MetaTrader 5 connector
pip install tradetropy[ccxt]      # CCXT crypto exchanges (live + streaming)
pip install tradetropy[bybit]     # Bybit connector (pybit)
pip install tradetropy[parquet]   # Parquet read/save support (pyarrow)
pip install tradetropy[hdf5]      # HDF5 read/save support (PyTables)
pip install tradetropy[all]       # all runtime extras above at once
```

| Extra      | Adds                | Use it for                                  |
|------------|---------------------|---------------------------------------------|
| `mt5`      | `MetaTrader5`       | Trading and data through MetaTrader 5       |
| `ccxt`     | `ccxt`              | Crypto live trading and WebSocket streaming |
| `bybit`    | `pybit`             | The Bybit connector                         |
| `parquet`  | `pyarrow`           | Reading/saving `.parquet` files             |
| `hdf5`     | `tables` (PyTables) | Reading/saving `.h5` / `.hdf5` files        |
| `all`      | all of the above    | Everything at once                          |

!!! note
    The connectors import their third-party library lazily, so a missing extra
    only raises when you actually construct that connector - the rest of the
    library keeps working. Likewise, the HDF5 path raises a clear error only
    when you pass `format='hdf5'` (or a `.h5`/`.hdf5` path) without the `hdf5`
    extra; use the base `.npz` format instead.

## With uv

If you develop with [uv](https://docs.astral.sh/uv/):

```bash
uv add tradetropy
uv add "tradetropy[ccxt]"
```

## Termux (Android)

Most users install on Windows, macOS or Linux, where every dependency
(`numpy`, `pandas`, `bokeh`) ships as a prebuilt wheel and `pip install tradetropy`
just works with no compilation. **Termux is the exception**: it is Android on
`aarch64`, and PyPI publishes no wheels for it. The scientific stack, though,
already ships as prebuilt Termux packages (`pkg install`), so no compiler and
no `--no-build-isolation` build are needed - install `numpy` / `pandas` /
`contourpy` from `pkg` first, then `pip install tradetropy` picks them up.

Crucially, the base install does **not** require PyTables: it cannot build on
Termux (it links the HDF5 C library through Blosc2, which Termux does not
package), so tradetropy uses the NumPy `.npz` binary format instead. `.npz` is as
fast as HDF5 for whole-file save/load and needs nothing beyond NumPy, so you get
the full data-IO stack on Termux with no HDF5 at all. Do **not** try to install
the `hdf5` extra there.

```bash
# 1. Update Termux and install Python.
pkg update && pkg upgrade -y
pkg install -y python

# 2. If you ever ran `pip install numpy` (or pandas/contourpy) by mistake
#    BEFORE this step, pip's own wheel is now shadowing the pkg version and
#    the pkg install below will fail or silently not take effect. Undo it
#    first: uninstall the pip-installed copy, then purge both caches so a
#    stale wheel is never reused, and only then install via pkg.
#      pip uninstall -y numpy pandas contourpy
#      pkg clean
#      pip cache purge

# 3. Install the prebuilt scientific stack via pkg (NOT pip). contourpy is
#    pulled in by Bokeh (the plotting stack). One command is equivalent to
#    installing them separately.
pkg install -y python-numpy python-pandas python-contourpy

# Optional: only if you want to read/save .parquet files (pyarrow has a
# prebuilt Termux package). HDF5 does not work on Termux - do not install
# the hdf5 extra there (see note above).
pkg install -y python-pyarrow

# 4. Restart Termux (close and reopen the app) so the new libraries are
#    picked up by a fresh shell, then install tradetropy normally.
pip install tradetropy
```

Verify the versions tradetropy will use:

```bash
python -c "import numpy, pandas; print('numpy', numpy.__version__); print('pandas', pandas.__version__)"
```

!!! note "Verified on-device"
    Confirmed working on Termux with `numpy==2.4.4` and `pandas==3.0.4` (both
    satisfy tradetropy's `numpy>=2.0,<2.5` / `pandas>=3.0,<4` pins), installing
    `tradetropy==0.1.4` with no compilation step.

!!! warning
    Do **not** run `pip install --upgrade pip` on Termux. It ships a *patched*
    pip that fixes Termux-specific build issues; the upstream version breaks
    native builds. Also use the official Termux app (from
    [F-Droid](https://f-droid.org/en/packages/com.termux/) or the
    [GitHub releases](https://github.com/termux/termux-app/releases/latest)),
    **not** the Google Play build, which is unmaintained.

If a future Termux update ships an older `numpy`/`pandas` that falls outside
tradetropy's pins, fall back to building them from source against Termux's
toolchain (`pkg install build-essential cmake ninja libopenblas`, then
`pip install --no-build-isolation numpy pandas` before `pip install tradetropy`).

## From source

```bash
git clone https://github.com/michiTrader/tradetropy.git
cd tradetropy
uv sync            # installs the project + dev tooling (pytest)
uv run pytest      # run the test suite
```

## Documentation toolchain

To build this documentation site locally, install the `docs` extra:

```bash
pip install tradetropy[docs]
mkdocs serve
```

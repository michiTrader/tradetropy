"""
Tests for the long (Binance bookDepth-style) order-book layout.

Covers:
  - pure wide <-> long conversion round-trip (core.data_types)
  - read_book / save_book with layout='long' in csv/parquet/hdf5
  - convert_book helper (wide <-> long, cross-format)
  - reading a real Binance bookDepth CSV sample
  - no-regression: existing wide files still read as before
  - BookData is always the same wide in-memory form regardless of source layout
"""

import numpy as np
import pandas as pd
import pytest

from tradetropy.core.data_types import (
    BookData, book_flat_columns, book_long_columns,
    wide_rows_to_long, long_rows_to_wide,
)
from tradetropy.io import read_book, save_book, convert_book


# -----
# Fixtures / helpers
# -----

def _make_wide_book() -> BookData:
    """Two snapshots, 3 levels, distinct ts - a small wide BookData."""
    def row(ts, bpx, bsz, apx, asz):
        return [ts, 0.0] + bpx + bsz + apx + asz

    data = np.array([
        row(1_690_000_000_000, [100.0, 99.0, 98.0], [1.0, 2.0, 3.0],
            [101.0, 102.0, 103.0], [4.0, 5.0, 6.0]),
        row(1_690_000_030_000, [200.0, 199.0, 198.0], [1.5, 2.5, 3.5],
            [201.0, 202.0, 203.0], [0.5, 1.5, 2.5]),
    ], dtype=np.float64)
    return BookData("BTCUSDT", data, levels=3, tick_size=0.5)


# Real Binance bookDepth sample (the user's data): timestamp,percentage,depth,notional
_BINANCE_SAMPLE = """timestamp,percentage,depth,notional
2023-03-31 00:09:30,-5,50737.00000000,1359653.00000000
2023-03-31 00:09:30,-4,50493.00000000,1352884.00000000
2023-03-31 00:09:30,-3,50237.00000000,1345810.00000000
2023-03-31 00:09:30,-2,49976.00000000,1338687.00000000
2023-03-31 00:09:30,-1,26732.00000000,713123.00000000
2023-03-31 00:09:30,1,21460.00000000,566596.00000000
2023-03-31 00:09:30,2,50468.00000000,1324648.00000000
2023-03-31 00:09:30,3,52310.00000000,1372450.00000000
2023-03-31 00:09:30,4,52552.00000000,1378676.00000000
2023-03-31 00:09:30,5,52798.00000000,1384925.00000000
2023-03-31 00:10:00,-5,50949.00000000,1365451.00000000
2023-03-31 00:10:00,-4,50705.00000000,1358682.00000000
2023-03-31 00:10:00,-3,50449.00000000,1351608.00000000
2023-03-31 00:10:00,-2,50188.00000000,1344485.00000000
2023-03-31 00:10:00,-1,26180.00000000,698328.00000000
2023-03-31 00:10:00,1,21639.00000000,571364.00000000
2023-03-31 00:10:00,2,50151.00000000,1316546.00000000
2023-03-31 00:10:00,3,51997.00000000,1364458.00000000
2023-03-31 00:10:00,4,52239.00000000,1370684.00000000
2023-03-31 00:10:00,5,52485.00000000,1376933.00000000
"""


# -----
# Pure conversion (core.data_types)
# -----

def test_pure_wide_long_roundtrip_exact():
    book = _make_wide_book()
    long_df = wide_rows_to_long(book.data, book.levels)
    assert list(long_df.columns) == book_long_columns()
    back, levels = long_rows_to_wide(long_df)
    assert levels == book.levels
    assert np.allclose(back, book.data, equal_nan=True)


def test_long_columns_are_cumulative():
    book = _make_wide_book()
    long_df = wide_rows_to_long(book.data, book.levels)
    # First snapshot bid side: percentage -1,-2,-3 depth must be cumulative.
    ts0 = long_df["ts"].iloc[0]
    bids = long_df[(long_df["ts"] == ts0) & (long_df["percentage"] < 0)]
    bids = bids.sort_values("percentage", ascending=False)  # -1, -2, -3
    depths = bids["depth"].to_numpy()
    assert np.all(np.diff(depths) > 0)                       # strictly cumulative
    # cumulative depth at level -3 = 1 + 2 + 3 = 6
    assert depths[-1] == pytest.approx(6.0)


# -----
# read_book / save_book layout='long'
# -----

@pytest.mark.parametrize("fmt,ext", [
    ("csv", "csv"),
    pytest.param("parquet", "parquet",
                 marks=pytest.mark.skipif(
                     pytest.importorskip("pyarrow", reason="pyarrow missing") is None,
                     reason="pyarrow missing")),
    pytest.param("hdf5", "h5",
                 marks=pytest.mark.skipif(
                     pytest.importorskip("tables", reason="tables missing") is None,
                     reason="tables missing")),
])
def test_save_read_long_roundtrip(tmp_path, fmt, ext):
    book = _make_wide_book()
    path = tmp_path / f"book_long.{ext}"
    save_book(book, str(path), format=fmt, layout="long")
    back = read_book(str(path), "BTCUSDT")            # layout auto-detected as long
    assert isinstance(back, BookData)
    assert back.levels == book.levels
    assert np.allclose(back.ts, book.ts)
    assert np.allclose(back.bid_px, book.bid_px, equal_nan=True)
    assert np.allclose(back.bid_sz, book.bid_sz, equal_nan=True)
    assert np.allclose(back.ask_px, book.ask_px, equal_nan=True)
    assert np.allclose(back.ask_sz, book.ask_sz, equal_nan=True)


def test_forced_layout_overrides_autodetect(tmp_path):
    book = _make_wide_book()
    path = tmp_path / "book_long.csv"
    save_book(book, str(path), format="csv", layout="long")
    back = read_book(str(path), "BTCUSDT", layout="long")
    assert np.allclose(back.bid_px, book.bid_px, equal_nan=True)


# -----
# Real Binance bookDepth sample
# -----

def test_read_real_binance_bookdepth(tmp_path):
    path = tmp_path / "BTCUSDT-bookDepth.csv"
    path.write_text(_BINANCE_SAMPLE)
    book = read_book(str(path), "BTCUSDT")            # auto-detected long
    assert isinstance(book, BookData)
    assert book.levels == 5                            # bands 1..5 per side
    assert book.data.shape[0] == 2                     # two timestamps
    # De-cumulated best-level bid size for the first snapshot = depth at -1.
    assert book.bid_sz[0, 0] == pytest.approx(26732.0)
    # Second bid band size = depth(-2) - depth(-1) = 49976 - 26732.
    assert book.bid_sz[0, 1] == pytest.approx(49976.0 - 26732.0)
    # Reconstructed price = notional slice / depth slice > 0 and finite.
    assert np.all(np.isfinite(book.bid_px[0]))
    assert np.all(book.bid_px[0] > 0)
    # Timestamps parsed from the 'timestamp' column and strictly increasing.
    assert book.ts[1] > book.ts[0]


# -----
# convert_book
# -----

def test_convert_wide_to_long_and_back(tmp_path):
    book = _make_wide_book()
    wide_path = tmp_path / "wide.csv"
    long_path = tmp_path / "long.csv"
    save_book(book, str(wide_path), format="csv", layout="wide")

    # wide -> long
    convert_book(str(wide_path), str(long_path), symbol="BTCUSDT", to_layout="long")
    long_df = pd.read_csv(long_path)
    assert {"percentage", "depth", "notional"} <= set(long_df.columns)

    # long -> wide
    back_path = tmp_path / "back.csv"
    convert_book(str(long_path), str(back_path), symbol="BTCUSDT", to_layout="wide")
    back = read_book(str(back_path), "BTCUSDT")
    assert np.allclose(back.bid_px, book.bid_px, equal_nan=True)
    assert np.allclose(back.ask_sz, book.ask_sz, equal_nan=True)


# -----
# No-regression: existing wide behavior unchanged, BookData always wide
# -----

def test_wide_still_autodetected(tmp_path):
    book = _make_wide_book()
    path = tmp_path / "wide.csv"
    save_book(book, str(path), format="csv")           # default layout='wide'
    df = pd.read_csv(path)
    assert any(c.startswith("bid_px_") for c in df.columns)
    back = read_book(str(path), "BTCUSDT")             # auto -> wide
    assert back.levels == 3
    assert np.allclose(back.bid_px, book.bid_px)


def test_bookdata_is_always_wide_regardless_of_source(tmp_path):
    book = _make_wide_book()
    wide_path = tmp_path / "w.csv"
    long_path = tmp_path / "l.csv"
    save_book(book, str(wide_path), format="csv", layout="wide")
    save_book(book, str(long_path), format="csv", layout="long")
    from_wide = read_book(str(wide_path), "BTCUSDT")
    from_long = read_book(str(long_path), "BTCUSDT")
    # Same wide in-memory matrix shape and columns for both sources.
    assert from_wide.data.shape == from_long.data.shape
    assert book_flat_columns(from_wide.levels) == book_flat_columns(from_long.levels)
    assert np.allclose(from_wide.data, from_long.data, equal_nan=True)

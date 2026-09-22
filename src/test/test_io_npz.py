"""
Tests for the NumPy .npz base binary IO backend and the appendable
raw-binary recorder path that consolidates to .npz on close.

The .npz format is the dependency-free base binary format (HDF5/PyTables is an
optional extra that cannot build on some platforms such as Termux). These tests
assert:
  - save_*/read_* round-trip for ticks, klines and the L2 book through .npz;
  - metadata (symbol / timeframe / levels) embeds and recovers from the file;
  - the recorder append path (_append_*_npz -> _consolidate_npz_record) rebuilds
    the same array a live recording would, readable by read_ticks/read_book;
  - the bundled sample datasets (now .npz) still load.
"""

import numpy as np
import pytest

from tradetropy.core.data_types import TickData, KlineData, BookData
from tradetropy.core.constants import N_TICK_COLS, N_OHLC_COLS
from tradetropy.io import save_ticks, save_klines, save_book, read_ticks, read_klines, read_book


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def ticks():
    """8 ticks with distinct ms timestamps."""
    base = 1_700_000_000_000
    arr = np.zeros((8, N_TICK_COLS), dtype=np.float64)
    arr[:, 0] = base + np.arange(8) * 1000          # ts
    arr[:, 1] = 100.0 + np.arange(8) * 0.1          # bid
    arr[:, 2] = 100.1 + np.arange(8) * 0.1          # ask
    arr[:, 3] = 1.0 + np.arange(8)                  # volume
    arr[:, 6] = 100.05 + np.arange(8) * 0.1         # price
    return TickData("BTCUSDT", arr, tick_size=0.1, digits=2)


@pytest.fixture
def klines():
    """5 one-minute candles."""
    base = 1_700_000_000_000
    arr = np.array([
        [base + i * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0, 1005.0]
        for i in range(5)
    ], dtype=np.float64)
    return KlineData("ETHUSDT", arr, timeframe=60_000, tick_size=0.01)


@pytest.fixture
def book():
    """6 book snapshots, 3 levels per side (wide flat layout)."""
    from tradetropy.core.data_types import book_row_width
    k = 3
    n = 6
    base = 1_700_000_000_000
    arr = np.zeros((n, book_row_width(k)), dtype=np.float64)
    arr[:, 0] = base + np.arange(n) * 1000          # ts
    arr[:, 1] = 0.0                                 # kind (snapshot)
    # Fill bid/ask px/sz per level with arbitrary but distinct values.
    for r in range(n):
        for lvl in range(k):
            off = 2 + lvl * 4
            arr[r, off + 0] = 100.0 - lvl * 0.1 - r * 0.01   # bid_px
            arr[r, off + 1] = 5.0 + lvl + r                  # bid_sz
            arr[r, off + 2] = 100.1 + lvl * 0.1 + r * 0.01   # ask_px
            arr[r, off + 3] = 6.0 + lvl + r                  # ask_sz
    return BookData("ADAUSDT", arr, levels=k, tick_size=0.01)


# ------------------------------------------------------------------------------
# save/read round-trip
# ------------------------------------------------------------------------------

@pytest.mark.unit
class TestNpzRoundTrip:
    def test_ticks_roundtrip_symbol_from_attrs(self, tmp_path, ticks):
        path = tmp_path / "t.npz"
        ticks.save(str(path), format="npz")
        assert path.exists()
        # symbol recovered from embedded metadata (no explicit symbol arg).
        out = read_ticks(str(path))
        assert out.symbol == "BTCUSDT"
        assert len(out) == len(ticks)
        # ts preserved exactly (int ms), prices close.
        np.testing.assert_array_equal(
            out.data[:, 0].astype(np.int64), ticks.data[:, 0].astype(np.int64)
        )
        np.testing.assert_allclose(out.data[:, 6], ticks.data[:, 6])

    def test_ticks_autodetect_format_by_extension(self, tmp_path, ticks):
        path = tmp_path / "auto.npz"
        save_ticks(ticks.data, path, format="npz", metadata={"tradetropy_symbol": "BTCUSDT"})
        out = read_ticks(str(path))          # format=None -> detected from .npz
        assert out.symbol == "BTCUSDT"
        assert len(out) == 8

    def test_klines_roundtrip_timeframe_from_attrs(self, tmp_path, klines):
        path = tmp_path / "k.npz"
        klines.save(str(path), format="npz")
        out = read_klines(str(path))         # symbol + timeframe from metadata
        assert out.symbol == "ETHUSDT"
        assert int(out.interval_ms) == 60_000
        assert len(out) == 5
        np.testing.assert_allclose(out.data[:, 4], klines.data[:, 4])  # close

    def test_book_roundtrip_levels_from_attrs(self, tmp_path, book):
        path = tmp_path / "b.npz"
        book.save(str(path), format="npz")
        out = read_book(str(path))           # symbol + levels from metadata
        assert out.symbol == "ADAUSDT"
        assert int(out.levels) == 3
        assert len(out) == 6
        np.testing.assert_allclose(out.data, book.data)

    def test_npz_smaller_and_no_pickle(self, tmp_path, ticks):
        """The .npz loads without allow_pickle (portable, safe)."""
        path = tmp_path / "safe.npz"
        ticks.save(str(path), format="npz")
        with np.load(path) as z:              # default allow_pickle=False
            assert "_data" in z.files
            assert "_columns" in z.files
            assert "_meta" in z.files


# ------------------------------------------------------------------------------
# Recorder append -> consolidate path
# ------------------------------------------------------------------------------

@pytest.mark.unit
class TestNpzRecorderConsolidation:
    def test_tick_append_consolidate_roundtrip(self, tmp_path, ticks):
        from tradetropy.io.io import (
            _append_ticks, _consolidate_npz_record, _npz_record_sidecars,
        )
        path = tmp_path / "rec.npz"
        # Simulate three buffered flushes; metadata only on the first (as the
        # live flush path does via cfg._meta_written).
        _append_ticks(ticks.data[:3], path, {"tradetropy_symbol": "BTCUSDT"})
        _append_ticks(ticks.data[3:6], path, None)
        _append_ticks(ticks.data[6:], path, None)

        part, meta = _npz_record_sidecars(path)
        assert part.exists() and meta.exists()   # sidecars present pre-close
        assert not path.exists()                 # final .npz not written yet

        _consolidate_npz_record(path)            # engine.stop() does this
        assert path.exists()
        assert not part.exists() and not meta.exists()  # sidecars cleaned up

        out = read_ticks(str(path))
        assert out.symbol == "BTCUSDT"
        assert len(out) == 8
        np.testing.assert_array_equal(
            out.data[:, 0].astype(np.int64), ticks.data[:, 0].astype(np.int64)
        )

    def test_book_append_consolidate_roundtrip(self, tmp_path, book):
        from tradetropy.io.io import _append_book, _consolidate_npz_record
        path = tmp_path / "recbook.npz"
        _append_book(book.data[:4], path, book.levels, {"tradetropy_symbol": "ADAUSDT"})
        _append_book(book.data[4:], path, book.levels, None)
        _consolidate_npz_record(path)

        out = read_book(str(path))
        assert out.symbol == "ADAUSDT"
        assert int(out.levels) == 3
        assert len(out) == 6
        np.testing.assert_allclose(out.data, book.data)

    def test_consolidate_noop_without_sidecars(self, tmp_path):
        """Consolidation is a no-op for a path that was never recorded (e.g.
        an HDF5 recording), so it is safe to call for every proxy on stop()."""
        from tradetropy.io.io import _consolidate_npz_record
        path = tmp_path / "never.npz"
        _consolidate_npz_record(path)         # must not raise
        assert not path.exists()

    def test_empty_recording_consolidates_to_empty_npz(self, tmp_path):
        """A meta header with no rows consolidates to an empty, readable file."""
        from tradetropy.io.io import _append_ticks, _consolidate_npz_record
        path = tmp_path / "empty.npz"
        _append_ticks(np.zeros((0, N_TICK_COLS)), path, {"tradetropy_symbol": "BTCUSDT"})
        _consolidate_npz_record(path)
        assert path.exists()
        out = read_ticks(str(path))
        assert len(out) == 0
        assert out.symbol == "BTCUSDT"


# ------------------------------------------------------------------------------
# Bundled datasets (now .npz)
# ------------------------------------------------------------------------------

@pytest.mark.unit
class TestBundledNpzDatasets:
    def test_all_loaders(self):
        from tradetropy import datasets as d
        assert d.load_btcusd_1m().symbol == "BTCUSDT"
        assert d.load_adausd_1m().symbol == "ADAUSDT"
        assert d.load_mesu26_ticks().symbol == "MESU26"
        assert d.load_mnqu26_ticks().symbol == "MNQU26"
        assert len(d.load_adausd_ticks()) == 43629
        b = d.load_adausd_book()
        assert b.symbol == "ADAUSDT" and int(b.levels) == 6


# ------------------------------------------------------------------------------
# HDF5 is optional: a clear error when PyTables is missing
# ------------------------------------------------------------------------------

@pytest.mark.unit
class TestHdf5OptionalGuard:
    def test_require_tables_raises_actionable_error(self, monkeypatch):
        """When PyTables is absent, the HDF5 path raises a DataError pointing to
        the optional extra and to the .npz alternative (not a raw ImportError)."""
        import builtins
        import tradetropy.io.io as io
        from tradetropy.exceptions import DataError

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tables":
                raise ImportError("simulated: no PyTables")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DataError, match=r"tradetropy\[hdf5\]"):
            io._require_tables()


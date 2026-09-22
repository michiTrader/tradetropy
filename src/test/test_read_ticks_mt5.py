"""
Tests for read_ticks()'s source= parameter and MT5 tick export support.

MT5's "Export Ticks" tool writes a fixed tab-separated schema with <DATE> and
<TIME> as two separate columns and each row updating only a SUBSET of fields
(bid-only quote, full bid+ask quote, or last/volume trade) - not the generic
tradetropy schema (a single 'datetime' column, one full row per tick). These
tests assert:
  - the generic CSV path is unchanged (no regression from adding source=);
  - source='mt5' parses the MT5 schema correctly (forward-filled bid/ask,
    <FLAGS> bit 32/64 mapped to tradetropy's +1/-1 aggressor sign);
  - source='auto' (the default) raises an actionable DataError suggesting
    source='mt5' when it sniffs the MT5 header, instead of a raw KeyError;
  - an unrelated malformed CSV still raises a plain, non-misleading error.
"""

import numpy as np
import pytest

from tradetropy.io import read_ticks
from tradetropy.exceptions import DataError


MT5_SAMPLE = (
    "\ufeff<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n"
    "2026.07.13\t00:00:00.108\t7599.00\t\t\t\t2\n"
    "2026.07.13\t00:00:00.173\t7599.25\t7599.50\t\t\t6\n"
    "2026.07.13\t00:00:00.174\t7599.00\t\t\t\t2\n"
    "2026.07.13\t00:00:00.174\t\t\t7599.25\t1.00000000\t88\n"
    "2026.07.13\t00:00:00.216\t\t7599.00\t\t\t4\n"
    "2026.07.13\t00:00:00.217\t\t\t7599.25\t3.00000000\t56\n"
)


@pytest.fixture
def mt5_csv(tmp_path):
    path = tmp_path / "mt5_export.csv"
    path.write_text(MT5_SAMPLE, encoding="utf-8")
    return path


@pytest.fixture
def generic_csv(tmp_path):
    path = tmp_path / "generic.csv"
    path.write_text(
        "datetime,bid,ask,volume,flags,price\n"
        "2026-07-13 00:00:00.100,100.0,100.1,1.0,1,100.05\n"
        "2026-07-13 00:00:00.200,100.1,100.2,2.0,-1,100.15\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
class TestGenericCsvUnaffected:
    def test_generic_csv_still_works_with_default_source(self, generic_csv):
        ticks = read_ticks(str(generic_csv), "BTCUSDT")
        assert ticks.symbol == "BTCUSDT"
        assert len(ticks) == 2
        np.testing.assert_allclose(ticks.data[:, 1], [100.0, 100.1])   # bid
        np.testing.assert_allclose(ticks.data[:, 2], [100.1, 100.2])   # ask

    def test_generic_csv_with_explicit_source_auto(self, generic_csv):
        ticks = read_ticks(str(generic_csv), "BTCUSDT", source="auto")
        assert len(ticks) == 2


@pytest.mark.unit
class TestMt5ExplicitSource:
    def test_mt5_source_parses_forward_filled_bid_ask(self, mt5_csv):
        ticks = read_ticks(str(mt5_csv), "MESU26", source="mt5")
        assert ticks.symbol == "MESU26"
        assert len(ticks) == 6

        bid = ticks.data[:, 1]
        ask = ticks.data[:, 2]
        price = ticks.data[:, 6]

        # Row 0: bid-only tick with no PRIOR ask -> ask back-filled from the
        # first known ask (row 1), so the series never starts with a NaN.
        assert bid[0] == pytest.approx(7599.00)
        assert ask[0] == pytest.approx(7599.50)

        # Row 1: full bid+ask quote.
        assert bid[1] == pytest.approx(7599.25)
        assert ask[1] == pytest.approx(7599.50)

        # Row 2: bid-only update -> ask forward-filled from row 1.
        assert bid[2] == pytest.approx(7599.00)
        assert ask[2] == pytest.approx(7599.50)

        # Row 3: trade tick (last=7599.25) -> bid/ask forward-filled from row 2.
        assert bid[3] == pytest.approx(7599.00)
        assert ask[3] == pytest.approx(7599.50)
        assert price[3] == pytest.approx(7599.25)

        # Row 4: ask-only update -> bid forward-filled, ask updates.
        assert bid[4] == pytest.approx(7599.00)
        assert ask[4] == pytest.approx(7599.00)

    def test_mt5_source_maps_aggressor_flags(self, mt5_csv):
        ticks = read_ticks(str(mt5_csv), "MESU26", source="mt5")
        flags = ticks.data[:, 4]
        # bits: 2 -> 0 (no side), 6 -> 0, 2 -> 0, 88 = 64+16+8 -> sell (-1),
        # 4 -> 0, 56 = 32+16+8 -> buy (+1).
        np.testing.assert_array_equal(flags, [0.0, 0.0, 0.0, -1.0, 0.0, 1.0])

    def test_mt5_source_volume_and_row_count(self, mt5_csv):
        ticks = read_ticks(str(mt5_csv), "MESU26", source="mt5")
        volume = ticks.data[:, 3]
        np.testing.assert_allclose(volume, [0.0, 0.0, 0.0, 1.0, 0.0, 3.0])

    def test_mt5_source_fills_price_on_quote_only_rows(self, mt5_csv):
        """Rows with no <LAST> (pure quote ticks) get price = (bid+ask)/2 via
        normalize_ticks(), matching the generic CSV path's fallback rule. Row
        0 is bid-only with no PRIOR ask, but its ask is back-filled from the
        first known ask, so it too gets a valid midpoint price (no leading
        NaN)."""
        ticks = read_ticks(str(mt5_csv), "MESU26", source="mt5")
        price = ticks.data[:, 6]
        assert not np.isnan(price).any()
        # Row 0: bid=7599.00, ask back-filled to 7599.50 -> midpoint.
        assert price[0] == pytest.approx((7599.00 + 7599.50) / 2)
        # Row 1: full bid+ask quote, no <LAST> -> price = midpoint.
        assert price[1] == pytest.approx((7599.25 + 7599.50) / 2)

    def test_mt5_source_no_leading_nan_in_bid_ask(self, mt5_csv):
        """Regression: a leading bid-only (or ask-only) tick must not leave a
        NaN in bid/ask, which would later crash broker price normalization."""
        ticks = read_ticks(str(mt5_csv), "MESU26", source="mt5")
        assert not np.isnan(ticks.data[:, 1]).any()  # bid
        assert not np.isnan(ticks.data[:, 2]).any()  # ask

    def test_mt5_source_requires_symbol(self, mt5_csv):
        with pytest.raises(ValueError, match="symbol is required"):
            read_ticks(str(mt5_csv), source="mt5")

    def test_mt5_source_on_non_mt5_file_raises(self, generic_csv):
        with pytest.raises(DataError, match="does not look like an MT5"):
            read_ticks(str(generic_csv), "BTCUSDT", source="mt5")


@pytest.mark.unit
class TestAutoDetectSuggestsSource:
    def test_auto_detect_raises_actionable_error_on_mt5_csv(self, mt5_csv):
        with pytest.raises(DataError, match="source='mt5'"):
            read_ticks(str(mt5_csv), "MESU26")

    def test_unknown_malformed_csv_raises_plain_error(self, tmp_path):
        path = tmp_path / "malformed.csv"
        path.write_text("foo,bar\n1,2\n3,4\n", encoding="utf-8")
        with pytest.raises(DataError, match="missing a 'datetime'"):
            read_ticks(str(path), "XYZ")

    def test_unknown_source_value_raises(self, generic_csv):
        with pytest.raises(DataError, match="Unknown source"):
            read_ticks(str(generic_csv), "BTCUSDT", source="bogus")

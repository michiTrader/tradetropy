"""Tests for read_klines() and MetaTrader 5 bar-export support."""

import numpy as np
import pytest

from tradetropy.exceptions import DataError
from tradetropy.io import read_klines


MT5_BARS_SAMPLE = (
    "\ufeff<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t"
    "<TICKVOL>\t<VOL>\t<SPREAD>\n"
    "2026.05.04\t00:00:00\t5000.00\t5001.00\t4999.50\t5000.50\t"
    "120\t240\t2\n"
    "2026.05.04\t00:01:00\t5000.50\t5002.00\t5000.00\t5001.75\t"
    "135\t270\t2\n"
)


@pytest.fixture
def mt5_bars_csv(tmp_path):
    path = tmp_path / "MESU26_1m_2026_MAY.csv"
    path.write_text(MT5_BARS_SAMPLE, encoding="utf-8")
    return path


@pytest.mark.unit
class TestReadKlinesMt5:
    def test_mt5_source_reads_ohlcv(self, mt5_bars_csv):
        klines = read_klines(
            mt5_bars_csv,
            timeframe="1m",
            symbol="MESU26",
            source="mt5",
            tick_size=0.25,
            tick_value=1.25,
            contract_size=5.0,
            digits=2,
        )

        assert klines.symbol == "MESU26"
        assert klines.interval_ms == 60_000
        assert klines.tick_size == 0.25
        assert klines.tick_value == 1.25
        assert klines.contract_size == 5.0
        np.testing.assert_allclose(
            klines.data,
            [
                [
                    1_777_852_800_000, 5000.00, 5001.00, 4999.50,
                    5000.50, 120.0, np.nan,
                ],
                [
                    1_777_852_860_000, 5000.50, 5002.00, 5000.00,
                    5001.75, 135.0, np.nan,
                ],
            ],
            equal_nan=True,
        )

    def test_auto_source_suggests_mt5(self, mt5_bars_csv):
        with pytest.raises(DataError, match="source='mt5'"):
            read_klines(
                mt5_bars_csv,
                timeframe="1m",
                symbol="MESU26",
            )

    def test_explicit_source_rejects_non_mt5_file(self, tmp_path):
        path = tmp_path / "generic.csv"
        path.write_text(
            "datetime,open,high,low,close,volume\n"
            "2026-05-04 00:00:00,1,2,0,1.5,10\n",
            encoding="utf-8",
        )
        with pytest.raises(DataError, match="does not look like an MT5"):
            read_klines(path, timeframe="1m", symbol="MESU26", source="mt5")

    def test_mt5_source_requires_timeframe(self, mt5_bars_csv):
        with pytest.raises(ValueError, match="timeframe"):
            read_klines(mt5_bars_csv, symbol="MESU26", source="mt5")

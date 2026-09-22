"""
Tests for the canonical data normalizer shared by all engines.
"""

import numpy as np
import pytest

from tradetropy.core.data_types import TickData, KlineData, _normalize_data as normalize_data
from tradetropy.exceptions import ConfigError


def _ticks(symbol="BTCUSDT", n=10):
    arr = np.zeros((n, 7), dtype=np.float64)
    arr[:, 0] = np.arange(n)  # ts
    return TickData(symbol, arr, tick_size=0.01)


def _klines(symbol="BTCUSDT", n=10, interval_ms=60_000):
    arr = np.zeros((n, 7), dtype=np.float64)
    arr[:, 0] = np.arange(n) * interval_ms
    return KlineData(symbol, arr, timeframe=interval_ms, tick_size=0.01)


class TestNormalizeData:
    def test_single_tickdata(self):
        inputs, feed_type = normalize_data(_ticks())
        assert feed_type == "tick"
        assert len(inputs) == 1
        assert isinstance(inputs, tuple)

    def test_single_klinedata(self):
        inputs, feed_type = normalize_data(_klines())
        assert feed_type == "kline"
        assert len(inputs) == 1

    def test_tuple_ticks(self):
        inputs, feed_type = normalize_data((_ticks("BTC"), _ticks("ETH")))
        assert feed_type == "tick"
        assert len(inputs) == 2

    def test_list_klines_normalized_to_tuple(self):
        inputs, feed_type = normalize_data([_klines("BTC"), _klines("ETH")])
        assert feed_type == "kline"
        assert isinstance(inputs, tuple)
        assert len(inputs) == 2

    def test_dict_rejected(self):
        with pytest.raises(ConfigError, match="[Dd]ict"):
            normalize_data({"BTCUSDT": np.zeros((5, 7))})

    def test_mixed_types_rejected(self):
        with pytest.raises(ConfigError):
            normalize_data((_ticks("BTC"), _klines("ETH")))

    def test_duplicate_symbols_rejected(self):
        with pytest.raises(ConfigError, match="[Dd]uplic"):
            normalize_data((_ticks("BTC"), _ticks("BTC")))

    def test_empty_rejected(self):
        with pytest.raises(ConfigError, match="empty"):
            normalize_data(())

    def test_unsupported_type_rejected(self):
        with pytest.raises(ConfigError):
            normalize_data(np.zeros((5, 7)))

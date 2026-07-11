"""
Validation functions for BacktestEngine.
Validates that symbols subscribed by the strategy exist in the data.
"""

from tradetropy.exceptions import ConfigError


def _validate_symbols(self, data: dict):
    """Verifies that all subscribed symbols (tick + OHLC) have data."""
    all_proxies = self.strategy._tick_proxies + self.strategy._ohlc_proxies
    for p in all_proxies:
        if p.symbol not in data:
            raise ConfigError(
                f"Strategy subscribes '{p.symbol}' but no data exists for that symbol. "
                f"Available symbols: {list(data.keys())}"
            )


def _validate_symbols_klines(self, data: dict):
    """Validates klines mode: no subscribe_ticks, and all OHLC symbols have data."""
    if self.strategy._tick_proxies:
        symbols = [tp.symbol for tp in self.strategy._tick_proxies]
        raise ConfigError(
            f"by_klines() does not support subscribe_ticks(). "
            f"Problematic symbols: {symbols}. "
            f"With OHLC data you can only use subscribe_ohlc(). "
            f"If you have ticks, use by_ticks()."
        )
    for p in self.strategy._ohlc_proxies:
        if p.symbol not in data:
            raise ConfigError(
                f"Strategy subscribes '{p.symbol}' but no data exists for that symbol. "
                f"Available symbols: {list(data.keys())}"
            )

import pytest

from tradetropy.exceptions import ConfigError
from tradetropy.models.strategy import Strategy
from tradetropy.ta import ConfirmedPivot, NBS


class _PatternStrategy(Strategy):
    def init(self):
        self.ohlc = self.subscribe_ohlc('X', timeframe='1m')
        refs = [
            self.ohlc.high_ref,
            self.ohlc.low_ref,
            self.ohlc.ts_ref,
        ]
        self.cpivot = self.add_indicator(refs, ConfirmedPivot(swing=2))
        self.nbs = self.add_indicator(refs, NBS(swing=2))

    def on_data(self):
        pass


def _strategy_with_base_pivots():
    strategy = _PatternStrategy()
    strategy.init()
    return strategy


@pytest.mark.unit
class TestPatternMatcherApi:
    def test_explicit_base_pivot_and_decorators(self):
        strategy = _strategy_with_base_pivots()

        proxy = strategy.add_pattern_matcher(
            base_pivot=strategy.cpivot,
            decorators=[strategy.nbs],
            pattern='L\nH[nbs=neu]',
            tag='setup',
        )

        definition = strategy._pattern_matcher_defs[0]
        assert definition.proxy is proxy
        assert definition.base_pivot is strategy.cpivot
        assert definition.decorators == [strategy.nbs]

    def test_decorators_are_optional(self):
        strategy = _strategy_with_base_pivots()

        strategy.add_pattern_matcher(
            base_pivot=strategy.cpivot,
            pattern='L\nH',
            tag='setup',
        )

        definition = strategy._pattern_matcher_defs[0]
        assert definition.decorators == []

    def test_decorator_cannot_be_used_as_base_pivot(self):
        strategy = _strategy_with_base_pivots()

        with pytest.raises(ConfigError, match='first pattern matcher pivot'):
            strategy.add_pattern_matcher(
                base_pivot=strategy.nbs,
                pattern='H',
            )


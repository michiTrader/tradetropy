import numpy as np
import pytest

from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import (
    SmcBreakerBlock, SmcOrderBlock, SmcPropulsionBlock, SmcRejectionBlock, SmcVacuumBlock,
    find_breaker_blocks, find_order_blocks, find_propulsion_blocks, find_rejection_blocks,
    find_vacuum_blocks,
)


def _gap_candles():
    prices = np.array([
        [100, 101, 99, 100], [100, 101, 99, 100], [105, 106, 104, 105],
        [105, 106, 103, 104], [104, 105, 101, 103], [103, 105, 102, 104],
    ])
    ts = 1_700_000_040_000 + np.arange(len(prices)) * 60_000
    return np.column_stack([ts, prices, np.ones(len(prices))])


@pytest.mark.parametrize("indicator_type, detect", [
    (SmcOrderBlock, find_order_blocks),
    (SmcBreakerBlock, find_breaker_blocks),
    (SmcRejectionBlock, find_rejection_blocks),
    (SmcPropulsionBlock, find_propulsion_blocks),
    (SmcVacuumBlock, find_vacuum_blocks),
])
def test_smc_queries_and_bands_are_causal(indicator_type, detect):
    source = (_gap_candles() if indicator_type is SmcVacuumBlock
              else load_btcusd_1m().data[:200, :6])
    full = indicator_type()
    bands = full.calculate(source)
    blocks = detect(source)
    assert blocks
    assert full.blocks_upto(source[-1, 0]) == blocks
    assert bands.shape == (8, len(source))
    assert full.output_names == ["bull_top", "bull_bot", "bear_top", "bear_bot"]
    assert full.ts_band_indices == [4, 5, 6, 7]

    growing = indicator_type()
    for n, ts in enumerate(source[:, 0], start=1):
        np.testing.assert_array_equal(growing.calculate(source[:n]), bands[:, :n])
        known = full.blocks_upto(ts)
        assert known == growing.blocks_upto(ts)
        for side in ("bullish", "bearish"):
            active = [block for block in known if block.kind == side and not block.mitigated]
            assert full.newest_live(ts, kind=side) == (active[-1] if active else None)

    growing.calculate(source[:0])
    assert growing.blocks_upto(source[-1, 0]) == []
    assert growing.newest_live(source[-1, 0]) is None
    assert growing.draw() == []


def test_vacuum_bands_expose_mitigation_on_its_own_bar():
    source = _gap_candles()
    indicator = SmcVacuumBlock()
    bands = indicator.calculate(source)
    ts = source[:, 0]
    np.testing.assert_array_equal(bands[0], [np.nan, np.nan, 104, 104, np.nan, np.nan])
    np.testing.assert_array_equal(bands[5], [np.nan, np.nan, np.nan, np.nan, ts[4], np.nan])
    assert indicator.at_span(ts[2], ts[3], "bullish").mitigated_ts is None
    assert indicator.at_span(ts[2], ts[4], "bullish").mitigated_ts == ts[4]


def test_smc_vacuum_chart_rewinds_to_the_same_closed_candle_snapshot():
    from bokeh.document import Document
    from tradetropy import BacktestEngine, PlotConfig, Strategy
    from tradetropy.core import KlineData
    from tradetropy.plotting.live.document import build_live_document
    from tradetropy.replay import ReplayEngine

    class BlocksStrategy(Strategy):
        warmup = 2

        def init(self):
            candles = self.subscribe_ohlc("SYM", "1m", window_size=30)
            self.blocks = self.add_indicator(SmcVacuumBlock.refs(candles), SmcVacuumBlock())

    rows = _gap_candles()
    ts = rows[:, 0]
    data = KlineData(symbol="SYM", data=rows, timeframe="1m", tick_size=.01)
    backtest = BacktestEngine.by_klines(BlocksStrategy(), data=(data,))
    backtest.run()
    replay = ReplayEngine.by_klines(BlocksStrategy(), data=(data,))
    replay.prepare()
    _, updater = build_live_document(
        strategy=replay.strategy, config=PlotConfig(), max_candles=30,
        broker=replay._sesh._broker, doc=Document(), replay_controller=replay._ctrl,
        io_loop=None,
    )
    updater.populate_history(max_age_minutes=0)

    replay._rebuild_to_cursor({"SYM": 3})
    updater.repopulate_after_topup()
    expected, = backtest.strategy.blocks.blocks_upto(ts[2])
    actual, = replay.strategy.blocks.blocks_upto(ts[2])
    assert (actual.kind, actual.top, actual.bottom) == (expected.kind, expected.top, expected.bottom)
    assert (actual.origin_ts, actual.confirmed_ts) == (expected.origin_ts, expected.confirmed_ts)
    zones = updater._vp_refs[0].registry[("Vacuum Block", "Rects")]
    assert list(zones.data["top"]) == [104]

    replay._rebuild_to_cursor({"SYM": 2})
    updater.repopulate_after_topup()
    assert replay.strategy.blocks.blocks_upto(ts[2]) == []
    assert all(len(values) == 0 for values in zones.data.values())

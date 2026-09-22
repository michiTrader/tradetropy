"""
Tests for Tick→Kline conversion, resample to higher TF and timeframe utilities.
"""

import numpy as np
import pytest

from tradetropy.core.constants import (
    parse_timeframe,
    format_timeframe,
    TIMEFRAME_PRESETS,
    OHLCV_TURNOVER_COLS,
    N_OHLCV_TURNOVER_COLS,
)
from tradetropy.exceptions import ConfigError


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — parse_timeframe + OHLCV constants
# ══════════════════════════════════════════════════════════════════════════════


class TestParseTimeframe:
    @pytest.mark.parametrize(
        "tf, expected",
        [
            ("1m", 60_000),
            ("5m", 300_000),
            ("15m", 900_000),
            ("1h", 3_600_000),
            ("4h", 14_400_000),
            ("1d", 86_400_000),
            ("1w", 604_800_000),
            ("30s", 30_000),
            ("500ms", 500),
            ("h", 3_600_000),  # no number == 1h
        ],
    )
    def test_str_units(self, tf, expected):
        assert parse_timeframe(tf) == expected

    def test_case_insensitive(self):
        # 'H' is unambiguous (no other unit collides with it), so it stays
        # fully case-insensitive.
        assert parse_timeframe("1H") == 3_600_000
        # Bare uppercase 'M' is intentionally rejected (see
        # test_ambiguous_uppercase_m_rejected) instead of silently meaning
        # minute, to avoid the month/minute confusion some other venues
        # (Binance/ccxt/MT5 '1M' == month) would suggest.

    def test_int_passthrough(self):
        assert parse_timeframe(60_000) == 60_000
        assert parse_timeframe(300_000) == 300_000

    def test_float_truncates(self):
        assert parse_timeframe(60_000.0) == 60_000

    @pytest.mark.parametrize(
        "bad",
        ["", "5x", "abc", "m5", "-1m", "0", "0m", -100, 0],
    )
    def test_invalid_raises(self, bad):
        with pytest.raises(ConfigError):
            parse_timeframe(bad)

    def test_bool_rejected(self):
        with pytest.raises(ConfigError):
            parse_timeframe(True)

    def test_ms_vs_s_disambiguation(self):
        # "ms" must resolve before "s"
        assert parse_timeframe("100ms") == 100
        assert parse_timeframe("2s") == 2_000

    # -- New standardized units: mo (month), min (minute alias), wk (week alias) --

    @pytest.mark.parametrize(
        "tf, expected",
        [
            ("1mo", 2_592_000_000),
            ("1MO", 2_592_000_000),
            ("1Mo", 2_592_000_000),
            ("2mo", 5_184_000_000),
            ("15min", 900_000),
            ("1wk", 604_800_000),
            ("2wk", 1_209_600_000),
        ],
    )
    def test_new_units(self, tf, expected):
        assert parse_timeframe(tf) == expected

    def test_min_alias_matches_m(self):
        assert parse_timeframe("15min") == parse_timeframe("15m")
        assert parse_timeframe("1min") == parse_timeframe("1m")

    def test_wk_alias_matches_w(self):
        assert parse_timeframe("1wk") == parse_timeframe("1w")
        assert parse_timeframe("2wk") == parse_timeframe("2w")

    def test_month_is_fixed_30_days(self):
        assert parse_timeframe("1mo") == 30 * 86_400_000

    def test_regression_minute_vs_month_no_ambiguity(self):
        # 'm' (lowercase) always means minute; bare uppercase 'M' is
        # rejected outright (see test_ambiguous_uppercase_m_rejected)
        # instead of silently meaning minute, so month and minute can
        # never be confused either way.
        assert parse_timeframe("1m") == 60_000
        assert parse_timeframe("1mo") != 60_000

    def test_ambiguous_uppercase_m_rejected(self):
        # Bare uppercase 'M' as a unit is reserved: some venues (Binance,
        # ccxt, MT5) use '1M' to mean month, which would silently resolve
        # to 1 minute here if left case-insensitive. Reject it explicitly
        # and point the user to '1m' (minute) or '1mo' (month).
        for bad in ("1M", "15M", "M"):
            with pytest.raises(ConfigError):
                parse_timeframe(bad)

    @pytest.mark.parametrize(
        "tf, expected",
        [
            ("1mo", 2_592_000_000),
            ("1MO", 2_592_000_000),
            ("1Mo", 2_592_000_000),
            ("1mO", 2_592_000_000),
            ("15min", 900_000),
            ("15MIN", 900_000),
            ("1wk", 604_800_000),
            ("1WK", 604_800_000),
            ("1H", 3_600_000),
            ("1D", 86_400_000),
        ],
    )
    def test_uppercase_m_guard_does_not_affect_unambiguous_units(self, tf, expected):
        # The uppercase-'M' guard must only reject the bare, ambiguous
        # 'M'/'m' unit - not 'mo'/'MO'/'min'/'MIN' (already explicit) nor
        # any other unit letter.
        assert parse_timeframe(tf) == expected

    def test_regression_existing_values_unaffected(self):
        assert parse_timeframe("5m") == 300_000
        assert parse_timeframe("30m") == 1_800_000
        assert parse_timeframe("2h") == 7_200_000
        assert parse_timeframe("3d") == 259_200_000


class TestFormatTimeframe:
    @pytest.mark.parametrize(
        "ms, expected",
        [
            (60_000, "1m"),
            (900_000, "15m"),
            (3_600_000, "1h"),
            (14_400_000, "4h"),
            (86_400_000, "1d"),
            (604_800_000, "1w"),
            (2_592_000_000, "1mo"),
            (30_000, "30s"),
            (500, "500ms"),
        ],
    )
    def test_canonical_output(self, ms, expected):
        assert format_timeframe(ms) == expected

    def test_month_is_canonical_not_days(self):
        # 2_592_000_000 ms is also exactly 30d, but 'mo' is the canonical
        # (largest) unit and must win over 'd'.
        assert format_timeframe(2_592_000_000) == "1mo"

    def test_minute_alias_not_used_in_output(self):
        # 'min'/'wk' are input-only aliases; canonical output stays m/w.
        assert format_timeframe(60_000) == "1m"
        assert format_timeframe(604_800_000) == "1w"

    @pytest.mark.parametrize(
        "ms",
        [60_000, 900_000, 3_600_000, 14_400_000, 86_400_000, 604_800_000,
         2_592_000_000],
    )
    def test_roundtrip(self, ms):
        assert parse_timeframe(format_timeframe(ms)) == ms


def test_timeframe_presets_include_month():
    assert TIMEFRAME_PRESETS["1mo"] == 2_592_000_000


def test_month_parity_with_connectors():
    # parse_timeframe('1mo') must resolve to the same ms value the ccxt and
    # MT5 connector maps use as their internal monthly key ('1M' venue string).
    from tradetropy.connectors.ccxt import _MS_TO_CCXT_TF
    from tradetropy.connectors.mt5 import _MS_TO_TF_KEY

    month_ms = parse_timeframe("1mo")
    assert month_ms in _MS_TO_CCXT_TF
    assert _MS_TO_CCXT_TF[month_ms] == "1M"
    assert month_ms in _MS_TO_TF_KEY
    assert _MS_TO_TF_KEY[month_ms] == "1M"


def test_ohlcv_turnover_cols():
    assert OHLCV_TURNOVER_COLS == (
        "ts", "open", "high", "low", "close", "volume", "turnover",
    )
    assert N_OHLCV_TURNOVER_COLS == 7



# ══════════════════════════════════════════════════════════════════════════════
# Task 2 — ticks_to_klines
# ══════════════════════════════════════════════════════════════════════════════

from tradetropy.data._klines import ticks_to_klines, resample_klines, validate_continuity
from tradetropy.core.constants import _TICK_COL, N_TICK_COLS


def _mk_ticks(rows):
    """rows: list of (ts, price, volume). Builds N×7 array with bid=ask=price."""
    out = np.zeros((len(rows), N_TICK_COLS), dtype=np.float64)
    C = _TICK_COL
    for i, (ts, price, vol) in enumerate(rows):
        out[i, C["ts"]] = ts
        out[i, C["bid"]] = price
        out[i, C["ask"]] = price
        out[i, C["price"]] = price
        out[i, C["volume"]] = vol
        out[i, C["volume_real"]] = vol
    return out


@pytest.fixture
def ticks_3bars():
    # 3 candles of 1m (60_000 ms). base is an exact multiple of 60_000 (aligned to epoch).
    base = 1_700_000_040_000
    return _mk_ticks([
        # candle 0
        (base + 1_000, 100.0, 1.0),
        (base + 20_000, 105.0, 2.0),
        (base + 50_000, 102.0, 1.0),
        # candle 1
        (base + 61_000, 103.0, 3.0),
        (base + 90_000, 99.0, 1.0),
        # candle 2 (in progress / partial)
        (base + 121_000, 101.0, 2.0),
        (base + 140_000, 104.0, 1.0),
    ])


class TestTicksAKlines:
    def test_ohlcv_closed_bars(self, ticks_3bars):
        k = ticks_to_klines(ticks_3bars, 60_000)  # include_partial=False
        # Only 2 closed candles (the 3rd is partial and omitted)
        assert k.shape == (2, 7)
        base = 1_700_000_040_000
        # candle 0: open=100, high=105, low=100, close=102, vol=4
        np.testing.assert_allclose(k[0], [
            base, 100.0, 105.0, 100.0, 102.0, 4.0,
            100*1 + 105*2 + 102*1,  # turnover
        ])
        # candle 1: open=103, high=103, low=99, close=99, vol=4
        np.testing.assert_allclose(k[1], [
            base + 60_000, 103.0, 103.0, 99.0, 99.0, 4.0,
            103*3 + 99*1,
        ])

    def test_include_partial(self, ticks_3bars):
        k = ticks_to_klines(ticks_3bars, 60_000, include_partial=True)
        assert k.shape == (3, 7)
        base = 1_700_000_040_000
        # candle 2 partial: open=101, high=104, low=101, close=104, vol=3
        np.testing.assert_allclose(k[2], [
            base + 120_000, 101.0, 104.0, 101.0, 104.0, 3.0,
            101*2 + 104*1,
        ])

    def test_empty(self):
        assert ticks_to_klines(np.empty((0, N_TICK_COLS)), 60_000).shape == (0, 7)

    def test_str_timeframe(self, ticks_3bars):
        a = ticks_to_klines(ticks_3bars, "1m")
        b = ticks_to_klines(ticks_3bars, 60_000)
        np.testing.assert_array_equal(a, b)

    def test_volume_real_source(self, ticks_3bars):
        # volume and volume_real are equal in the fixture → same result
        a = ticks_to_klines(ticks_3bars, 60_000, volume_source="volume_real")
        b = ticks_to_klines(ticks_3bars, 60_000, volume_source="volume")
        np.testing.assert_array_equal(a, b)

    def test_mid_source_equals_price_when_bid_eq_ask(self, ticks_3bars):
        a = ticks_to_klines(ticks_3bars, 60_000, price_source="mid")
        b = ticks_to_klines(ticks_3bars, 60_000, price_source="price")
        np.testing.assert_allclose(a, b)

    def test_invalid_sources(self, ticks_3bars):
        from tradetropy.exceptions import DataError
        with pytest.raises(DataError):
            ticks_to_klines(ticks_3bars, 60_000, price_source="foo")
        with pytest.raises(DataError):
            ticks_to_klines(ticks_3bars, 60_000, volume_source="foo")

    def test_parity_with_replay_helper(self, ticks_3bars):
        """OHLCV (cols 0-5) must match _ticks_to_klines from replay."""
        from tradetropy.replay.replay_sesh import _ticks_to_klines
        ref = _ticks_to_klines(ticks_3bars, 60_000)  # includes ALL candles (incl. partial)
        mine = ticks_to_klines(ticks_3bars, 60_000, include_partial=True)
        np.testing.assert_allclose(mine[:, :6], ref[:, :6])


class TestTicksAKlinesTradeSource:
    """price_source='trade': drop quote-only ticks (volume == 0) whose price
    column holds a (bid+ask)/2 fallback (e.g. from normalize_ticks()/MT5
    quote ticks), so OHLC always lands on a real traded price."""

    TICK_SIZE = 0.25

    @staticmethod
    def _mk_mixed_ticks(rows):
        """rows: list of (ts, bid, ask, price, volume). Builds an N×7 array,
        unlike _mk_ticks() this allows price != bid/ask (quote midpoints)."""
        out = np.zeros((len(rows), N_TICK_COLS), dtype=np.float64)
        C = _TICK_COL
        for i, (ts, bid, ask, price, vol) in enumerate(rows):
            out[i, C["ts"]] = ts
            out[i, C["bid"]] = bid
            out[i, C["ask"]] = ask
            out[i, C["price"]] = price
            out[i, C["volume"]] = vol
            out[i, C["volume_real"]] = vol
        return out

    @pytest.fixture
    def ticks_with_quote_middles(self):
        # 2 closed candles of 1m + 1 partial candle, mixing real trades
        # (volume > 0, price on a 0.25 tick) with quote-only ticks
        # (volume == 0, price == (bid+ask)/2, off-tick).
        base = 1_700_000_040_000
        return self._mk_mixed_ticks([
            # candle 0: trades at 100.00 / 105.25 / 102.00, plus quote noise
            (base + 500, 99.75, 100.25, 100.00, 0.0),     # quote midpoint 100.00 (on-tick, harmless)
            (base + 1_000, 100.0, 100.0, 100.00, 1.0),    # trade
            (base + 10_000, 105.00, 105.50, 105.25, 0.0), # quote midpoint 105.25 (OFF-tick)
            (base + 20_000, 105.25, 105.25, 105.25, 2.0), # trade
            (base + 50_000, 102.0, 102.0, 102.00, 1.0),   # trade (close of candle 0)
            # candle 1: trades at 103.00 / 99.00, plus quote noise
            (base + 61_000, 103.0, 103.0, 103.00, 3.0),   # trade (open of candle 1)
            (base + 80_000, 98.75, 99.25, 99.00, 0.0),    # quote midpoint 99.00 (on-tick, harmless)
            (base + 90_000, 99.0, 99.0, 99.00, 1.0),      # trade (close of candle 1)
            # candle 2 (in progress / partial): only a quote tick, no trade
            (base + 121_000, 100.75, 101.25, 101.00, 0.0),  # quote midpoint, no trade
        ])

    def test_trade_source_excludes_quote_only_ticks(self, ticks_with_quote_middles):
        k = ticks_to_klines(
            ticks_with_quote_middles, 60_000, price_source="trade",
        )
        base = 1_700_000_040_000
        # Only 1 closed candle: candle 1 is closed but candle 0 too - both
        # have at least one trade. Candle 2 has no trade at all, so with
        # price_source='trade' it does not exist as a bar, and the "last
        # bar" that include_partial=False drops is candle 1 (the last one
        # that contains a trade), leaving only candle 0 closed.
        assert k.shape == (1, 7)
        np.testing.assert_allclose(k[0], [
            base, 100.0, 105.25, 100.0, 102.0, 4.0,
            100.0 * 1 + 105.25 * 2 + 102.0 * 1,
        ])

    def test_trade_source_respects_tick_size(self, ticks_with_quote_middles):
        k = ticks_to_klines(
            ticks_with_quote_middles, 60_000, price_source="trade",
            include_partial=True,
        )
        ohlc = k[:, 1:5]
        remainder = np.mod(ohlc, self.TICK_SIZE)
        np.testing.assert_allclose(remainder, 0.0, atol=1e-9)

    def test_price_source_default_breaks_tick_size(self, ticks_with_quote_middles):
        """Sanity check: the default 'price' source is the one that leaks the
        off-tick quote midpoint (105.25 candle 0's high stays fine here, but a
        genuinely off-tick midpoint like 100.125 would leak through)."""
        base = 1_700_000_040_000
        ticks = self._mk_mixed_ticks([
            (base + 1_000, 100.0, 100.0, 100.00, 1.0),      # trade, on-tick
            (base + 2_000, 100.0, 100.25, 100.125, 0.0),    # quote midpoint, OFF-tick
        ])
        k_price = ticks_to_klines(ticks, 60_000, include_partial=True)
        k_trade = ticks_to_klines(ticks, 60_000, price_source="trade", include_partial=True)
        assert np.mod(k_price[0, 2], self.TICK_SIZE) != 0.0  # high leaks the midpoint
        assert np.mod(k_trade[0, 2], self.TICK_SIZE) == 0.0  # trade-only stays on-tick

    def test_trade_source_all_quotes_returns_empty(self):
        base = 1_700_000_040_000
        ticks = self._mk_mixed_ticks([
            (base + 1_000, 100.0, 100.5, 100.25, 0.0),
            (base + 2_000, 100.0, 100.5, 100.25, 0.0),
        ])
        k = ticks_to_klines(ticks, 60_000, price_source="trade")
        assert k.shape == (0, 7)

    def test_trade_source_volume_source_still_configurable(self, ticks_with_quote_middles):
        a = ticks_to_klines(
            ticks_with_quote_middles, 60_000,
            price_source="trade", volume_source="volume_real",
        )
        b = ticks_to_klines(
            ticks_with_quote_middles, 60_000,
            price_source="trade", volume_source="volume",
        )
        np.testing.assert_array_equal(a, b)


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — resample_klines
# ══════════════════════════════════════════════════════════════════════════════


class TestResampleKlines:
    def test_1m_to_5m_matches_fixture(self, klines_1m, klines_5m):
        # klines_5m from the conftest is exactly the 1m→5m resample of klines_1m,
        # except the last 5m candle which is partial in the fixture (only one 1m candle).
        out, interval = resample_klines(klines_1m, 60_000, 300_000)
        assert interval == 300_000
        # The first 3 candles of 5m are complete and must match.
        np.testing.assert_allclose(out[:3], klines_5m[:3])

    def test_7col_preserves_turnover(self, klines_1m):
        out, _ = resample_klines(klines_1m, 60_000, 300_000)
        assert out.shape[1] == 7
        # turnover of the first group = sum of turnovers from source
        first_ts = (klines_1m[0, 0] // 300_000) * 300_000
        mask = (klines_1m[:, 0] // 300_000) * 300_000 == first_ts
        assert out[0, 6] == pytest.approx(klines_1m[mask, 6].sum())

    def test_6col_returns_6col(self, klines_1m):
        out, _ = resample_klines(klines_1m[:, :6], 60_000, 300_000)
        assert out.shape[1] == 6

    def test_parity_with_plotting_helper(self, klines_1m):
        from tradetropy.plotting._util import _resample_ohlc
        ref, ref_iv = _resample_ohlc(klines_1m[:, :6], 60_000, 300_000)
        mine, mine_iv = resample_klines(klines_1m[:, :6], 60_000, 300_000)
        assert mine_iv == ref_iv
        np.testing.assert_allclose(mine, ref)

    def test_target_le_source_passthrough(self, klines_1m):
        out, interval = resample_klines(klines_1m, 60_000, 60_000)
        assert interval == 60_000
        np.testing.assert_array_equal(out, klines_1m)

    def test_non_multiple_adjusts_with_warning(self, klines_1m):
        with pytest.warns(UserWarning):
            out, interval = resample_klines(klines_1m, 60_000, 70_000)
        assert interval == 120_000  # next multiple of 60_000

    def test_str_timeframe(self, klines_1m):
        a, ia = resample_klines(klines_1m, 60_000, "5m")
        b, ib = resample_klines(klines_1m, 60_000, 300_000)
        assert ia == ib
        np.testing.assert_array_equal(a, b)

    def test_empty(self):
        out, interval = resample_klines(np.empty((0, 7)), 60_000, 300_000)
        assert out.shape == (0, 7)

    def test_bad_shape(self):
        from tradetropy.exceptions import DataError
        with pytest.raises(DataError):
            resample_klines(np.zeros((5, 4)), 60_000, 300_000)



# ══════════════════════════════════════════════════════════════════════════════
# Task 4 — TickData.to_klines  /  Task 5 — KlineData.resample
# ══════════════════════════════════════════════════════════════════════════════

from tradetropy.core.data_types import TickData, KlineData


class TestTickDataToKlines:
    def test_returns_klinedata(self, ticks_3bars):
        td = TickData("BTCUSDT", ticks_3bars, tick_size=0.5, tick_value=1.25,
                      digits=1, avg_spread=2.0)
        kd = td.to_klines("1m")
        assert isinstance(kd, KlineData)
        assert kd.interval_ms == 60_000
        assert kd.symbol == "BTCUSDT"
        assert kd.data.shape[1] == 7

    def test_does_not_mutate_original(self, ticks_3bars):
        td = TickData("BTCUSDT", ticks_3bars)
        before = td.data.copy()
        data_id = id(td.data)
        _ = td.to_klines("1m")
        assert id(td.data) == data_id            # same object
        np.testing.assert_array_equal(td.data, before)  # same content

    def test_propagates_symbol_config(self, ticks_3bars):
        td = TickData("MES", ticks_3bars, tick_size=0.25, tick_value=1.25,
                      contract_size=5.0, digits=2, avg_spread=1.0,
                      volume_min=1.0, volume_max=50.0, volume_step=1.0)
        kd = td.to_klines(60_000)
        assert kd.tick_size == 0.25
        assert kd.tick_value == 1.25
        assert kd.contract_size == 5.0
        assert kd.avg_spread == 1.0
        assert kd.volume_min == 1.0
        assert kd.volume_max == 50.0
        assert kd.volume_step == 1.0

    def test_include_partial_flag(self, ticks_3bars):
        td = TickData("BTCUSDT", ticks_3bars)
        assert len(td.to_klines("1m").data) == 2
        assert len(td.to_klines("1m", include_partial=True).data) == 3

    def test_matches_pure_function(self, ticks_3bars):
        td = TickData("BTCUSDT", ticks_3bars)
        kd = td.to_klines("1m", include_partial=True)
        ref = ticks_to_klines(ticks_3bars, 60_000, include_partial=True)
        np.testing.assert_array_equal(kd.data, ref)

    def test_forwards_trade_price_source(self, ticks_3bars):
        ticks = ticks_3bars.copy()
        C = _TICK_COL
        # Add a quote-only midpoint that must be ignored by the trade source.
        ticks[2, C["bid"]] = 110.0
        ticks[2, C["ask"]] = 110.25
        ticks[2, C["price"]] = 110.125
        ticks[2, C["volume"]] = 0.0
        ticks[2, C["volume_real"]] = 0.0

        td = TickData("BTCUSDT", ticks)
        actual = td.to_klines("1m", include_partial=True, price_source="trade")
        expected = ticks_to_klines(
            ticks, "1m", include_partial=True, price_source="trade",
        )
        np.testing.assert_array_equal(actual.data, expected)
        assert actual.data[0, 2] == 105.0

    def test_forwards_volume_source(self, ticks_3bars):
        ticks = ticks_3bars.copy()
        C = _TICK_COL
        ticks[:, C["volume_real"]] = ticks[:, C["volume"]] * 10.0

        td = TickData("BTCUSDT", ticks)
        actual = td.to_klines("1m", include_partial=True, volume_source="volume_real")
        expected = ticks_to_klines(
            ticks, "1m", include_partial=True, volume_source="volume_real",
        )
        np.testing.assert_array_equal(actual.data, expected)


class TestKlineDataResample:
    def test_returns_new_klinedata(self, klines_1m):
        kd = KlineData("BTCUSDT", klines_1m, timeframe=60_000)
        out = kd.resample("5m")
        assert isinstance(out, KlineData)
        assert out.interval_ms == 300_000
        assert out is not kd

    def test_does_not_mutate_original(self, klines_1m):
        kd = KlineData("BTCUSDT", klines_1m, timeframe=60_000)
        before = kd.data.copy()
        data_id = id(kd.data)
        _ = kd.resample("5m")
        assert id(kd.data) == data_id
        assert kd.interval_ms == 60_000
        np.testing.assert_array_equal(kd.data, before)

    def test_propagates_config(self, klines_1m):
        kd = KlineData("MES", klines_1m, timeframe=60_000, tick_size=0.25,
                       tick_value=1.25, contract_size=5.0)
        out = kd.resample("5m")
        assert out.symbol == "MES"
        assert out.tick_size == 0.25
        assert out.tick_value == 1.25
        assert out.contract_size == 5.0

    def test_matches_fixture(self, klines_1m, klines_5m):
        kd = KlineData("BTCUSDT", klines_1m, timeframe=60_000)
        out = kd.resample("5m")
        np.testing.assert_allclose(out.data[:3], klines_5m[:3])

    def test_non_multiple_adjusts(self, klines_1m):
        kd = KlineData("BTCUSDT", klines_1m, timeframe=60_000)
        with pytest.warns(UserWarning):
            out = kd.resample(70_000)
        assert out.interval_ms == 120_000



# ══════════════════════════════════════════════════════════════════════════════
# Task 7 — ccxt connector (mocked, no network)
# ══════════════════════════════════════════════════════════════════════════════

from tradetropy.connectors.ccxt import fetch_klines, fetch_ticks
from tradetropy.exceptions import ConnectionError as TrConnectionError


class _MockExchange:
    """Exchange compatible with the ccxt interface used by the loader, no network."""

    def __init__(self):
        self.ohlcv_calls = []
        self.trades_calls = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.ohlcv_calls.append((symbol, timeframe, since, limit))
        # [ts, open, high, low, close, volume]
        return [
            [1_700_000_040_000, 100.0, 110.0, 95.0, 105.0, 10.0],
            [1_700_000_100_000, 105.0, 108.0, 102.0, 106.0, 8.0],
        ]

    def fetch_trades(self, symbol, since, limit):
        self.trades_calls.append((symbol, since, limit))
        return [
            {"timestamp": 1_700_000_040_001, "price": 100.0, "amount": 1.5, "side": "buy"},
            {"timestamp": 1_700_000_040_500, "price": 101.0, "amount": 2.0, "side": "sell"},
            {"timestamp": 1_700_000_041_000, "price": 100.5, "amount": 0.5, "side": None},
        ]


class TestFetchKlines:
    def test_returns_klinedata(self):
        ex = _MockExchange()
        kd = fetch_klines(ex, "BTC/USDT", "1m", limit=2, tick_size=0.5)
        assert isinstance(kd, KlineData)
        assert kd.symbol == "BTC/USDT"
        assert kd.interval_ms == 60_000
        assert kd.data.shape == (2, 7)
        assert kd.tick_size == 0.5
        # OHLCV mapped correctly
        np.testing.assert_allclose(kd.data[0, :6], [1_700_000_040_000, 100, 110, 95, 105, 10])

    def test_passes_args_to_ccxt(self):
        ex = _MockExchange()
        fetch_klines(ex, "ETH/USDT", "5m", since=123, limit=50)
        assert ex.ohlcv_calls == [("ETH/USDT", "5m", 123, 50)]

    def test_turnover_nan_default(self):
        kd = fetch_klines(_MockExchange(), "BTC/USDT", "1m")
        assert np.isnan(kd.data[:, 6]).all()

    def test_turnover_approx(self):
        kd = fetch_klines(_MockExchange(), "BTC/USDT", "1m", turnover_mode="approx")
        # close * volume
        np.testing.assert_allclose(kd.data[0, 6], 105.0 * 10.0)
        np.testing.assert_allclose(kd.data[1, 6], 106.0 * 8.0)

    def test_bad_turnover_mode(self):
        with pytest.raises(ConfigError):
            fetch_klines(_MockExchange(), "BTC/USDT", "1m", turnover_mode="xxx")

    def test_resample_after_fetch(self):
        kd = fetch_klines(_MockExchange(), "BTC/USDT", "1m")
        out = kd.resample("5m")
        assert out.interval_ms == 300_000

    def test_unknown_exchange_str_raises(self):
        with pytest.raises((TrConnectionError, ImportError)):
            fetch_klines("definitely_not_an_exchange_xyz", "BTC/USDT", "1m")


class TestFetchTicks:
    def test_returns_tickdata(self):
        td = fetch_ticks(_MockExchange(), "BTC/USDT", limit=3)
        assert isinstance(td, TickData)
        assert td.data.shape == (3, 7)
        C = _TICK_COL
        # price and volume mapped
        np.testing.assert_allclose(td.data[0, C["price"]], 100.0)
        np.testing.assert_allclose(td.data[1, C["volume"]], 2.0)
        np.testing.assert_allclose(td.data[1, C["volume_real"]], 2.0)
        # flags by side
        assert td.data[0, C["flags"]] == 1.0   # buy
        assert td.data[1, C["flags"]] == -1.0  # sell
        assert td.data[2, C["flags"]] == 0.0   # None

    def test_ticks_to_klines_roundtrip(self):
        td = fetch_ticks(_MockExchange(), "BTC/USDT")
        kd = td.to_klines("1m", include_partial=True)
        assert isinstance(kd, KlineData)
        assert kd.data.shape[1] == 7

    def test_passes_args_to_ccxt(self):
        ex = _MockExchange()
        fetch_ticks(ex, "ETH/USDT", since=99, limit=10)
        assert ex.trades_calls == [("ETH/USDT", 99, 10)]



# ══════════════════════════════════════════════════════════════════════════════
# Task 8 — public exports + validate_continuity
# ══════════════════════════════════════════════════════════════════════════════


def test_public_exports():
    import tradetropy.data as data_mod
    from tradetropy.data import (
        ticks_to_klines as f1,
        resample_klines as f2,
        validate_continuity as f3,
    )
    for name in ("ticks_to_klines", "resample_klines", "validate_continuity"):
        assert name in data_mod.__all__
    assert callable(f1) and callable(f2) and callable(f3)


class TestValidateContinuity:
    def test_continuous_ok(self, klines_1m):
        kd = KlineData("BTCUSDT", klines_1m, timeframe=60_000)
        rep = validate_continuity(kd)
        assert rep["ok"] is True
        assert rep["n_gaps"] == 0
        assert rep["missing_total"] == 0

    def test_detects_gap(self, klines_1m):
        # Remove 2 candles from the middle to create a gap of 2 missing.
        trimmed = np.delete(klines_1m, [5, 6], axis=0)
        kd = KlineData("BTCUSDT", trimmed, timeframe=60_000)
        rep = validate_continuity(kd)
        assert rep["ok"] is False
        assert rep["n_gaps"] == 1
        assert rep["missing_total"] == 2
        assert rep["gaps"][0]["missing"] == 2

    def test_detects_non_monotonic(self):
        arr = np.array([
            [0,     1, 1, 1, 1, 1, 1],
            [60000, 1, 1, 1, 1, 1, 1],
            [30000, 1, 1, 1, 1, 1, 1],  # goes backwards
        ], dtype=np.float64)
        rep = validate_continuity(arr, interval_ms=60_000)
        assert rep["ok"] is False
        assert 2 in rep["non_monotonic"]

    def test_ndarray_requires_interval(self):
        from tradetropy.exceptions import DataError
        with pytest.raises(DataError):
            validate_continuity(np.zeros((3, 7)))

    def test_short_series_ok(self):
        rep = validate_continuity(np.zeros((1, 7)), interval_ms=60_000)
        assert rep["ok"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Timeframe presets + to_binance_interval
# ══════════════════════════════════════════════════════════════════════════════

from tradetropy.core.constants import TIMEFRAME_PRESETS, to_binance_interval


class TestTimeframePresets:
    def test_presets_match_parse_timeframe(self):
        for label, ms in TIMEFRAME_PRESETS.items():
            assert parse_timeframe(label) == ms

    def test_common_presets_present(self):
        for tf in ("1m", "5m", "15m", "1h", "4h", "1d", "1w"):
            assert tf in TIMEFRAME_PRESETS

    def test_to_binance_interval(self):
        assert to_binance_interval("5m") == "5m"
        assert to_binance_interval(3_600_000) == "1h"
        assert to_binance_interval("1d") == "1d"

    def test_to_binance_interval_unsupported(self):
        with pytest.raises(ConfigError):
            to_binance_interval("5s")   # Binance does not have 5s candles
        with pytest.raises(ConfigError):
            to_binance_interval(7_000)  # arbitrary interval


# ══════════════════════════════════════════════════════════════════════════════
# binance (HTTP mocked, no network)
# ══════════════════════════════════════════════════════════════════════════════

from tradetropy.connectors import binance


# Raw response from /api/v3/klines: 12 fields per candle.
_BINANCE_KLINES_RAW = [
    [1_700_000_040_000, "42000.0", "42100.0", "41950.0", "42050.0",
     "10.5", 1_700_000_099_999, "441525.0", 120, "5.0", "210000.0", "0"],
    [1_700_000_100_000, "42050.0", "42080.0", "42010.0", "42060.0",
     "8.0", 1_700_000_159_999, "336360.0", 95, "4.0", "168000.0", "0"],
]

# Raw response from /api/v3/aggTrades.
_BINANCE_AGGTRADES_RAW = [
    {"a": 1, "p": "42000.0", "q": "1.5", "f": 10, "l": 12,
     "T": 1_700_000_040_100, "m": False, "M": True},  # buyer aggressor → +1
    {"a": 2, "p": "42010.0", "q": "0.8", "f": 13, "l": 13,
     "T": 1_700_000_040_500, "m": True, "M": True},   # seller aggressor → -1
]


class TestBinanceFetchKlines:
    def test_maps_columns_and_turnover(self, monkeypatch):
        captured = {}

        def fake_get(url, timeout=10.0):
            captured["url"] = url
            return _BINANCE_KLINES_RAW

        monkeypatch.setattr(binance, "_http_get_json", fake_get)

        kd = binance.fetch_klines("BTC/USDT", "1m", limit=2, tick_size=0.1)
        assert isinstance(kd, KlineData)
        assert kd.symbol == "BTCUSDT"               # normalized
        assert kd.interval_ms == 60_000
        assert kd.data.shape == (2, 7)
        # ts, open, high, low, close, volume
        np.testing.assert_allclose(
            kd.data[0, :6], [1_700_000_040_000, 42000, 42100, 41950, 42050, 10.5]
        )
        # turnover REAL = quoteAssetVolume (field 7)
        np.testing.assert_allclose(kd.data[0, 6], 441525.0)
        np.testing.assert_allclose(kd.data[1, 6], 336360.0)
        # URL well-formed
        assert "symbol=BTCUSDT" in captured["url"]
        assert "interval=1m" in captured["url"]
        assert "limit=2" in captured["url"]

    def test_resample_after_fetch(self, monkeypatch):
        monkeypatch.setattr(binance, "_http_get_json",
                            lambda url, timeout=10.0: _BINANCE_KLINES_RAW)
        kd = binance.fetch_klines("BTCUSDT", "1m")
        out = kd.resample("5m")
        assert out.interval_ms == 300_000

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(binance, "_http_get_json",
                            lambda url, timeout=10.0: [])
        kd = binance.fetch_klines("BTCUSDT", "1m")
        assert kd.data.shape == (0, 7)

    def test_unsupported_interval(self, monkeypatch):
        monkeypatch.setattr(binance, "_http_get_json",
                            lambda url, timeout=10.0: _BINANCE_KLINES_RAW)
        with pytest.raises(ConfigError):
            binance.fetch_klines("BTCUSDT", "5s")


class TestBinanceFetchTicks:
    def test_maps_trades(self, monkeypatch):
        monkeypatch.setattr(binance, "_http_get_json",
                            lambda url, timeout=10.0: _BINANCE_AGGTRADES_RAW)
        td = binance.fetch_ticks("BTC/USDT", limit=2)
        assert isinstance(td, TickData)
        assert td.data.shape == (2, 7)
        C = _TICK_COL
        np.testing.assert_allclose(td.data[0, C["price"]], 42000.0)
        np.testing.assert_allclose(td.data[0, C["volume"]], 1.5)
        # flags: m=False → +1 (buyer aggressor); m=True → -1
        assert td.data[0, C["flags"]] == 1.0
        assert td.data[1, C["flags"]] == -1.0

    def test_ticks_to_klines_roundtrip(self, monkeypatch):
        monkeypatch.setattr(binance, "_http_get_json",
                            lambda url, timeout=10.0: _BINANCE_AGGTRADES_RAW)
        td = binance.fetch_ticks("BTCUSDT")
        kd = td.to_klines("1m", include_partial=True)
        assert kd.data.shape[1] == 7

    def test_aggtrades_url_params(self, monkeypatch):
        captured = {}

        def fake_get(url, timeout=10.0):
            captured["url"] = url
            return []

        monkeypatch.setattr(binance, "_http_get_json", fake_get)
        binance.fetch_ticks("ETHUSDT", limit=10, from_id=99)
        assert "symbol=ETHUSDT" in captured["url"]
        assert "limit=10" in captured["url"]
        assert "fromId=99" in captured["url"]
        assert "/api/v3/aggTrades" in captured["url"]

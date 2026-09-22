"""Tests for the DeepTrades autofilter (book-aware class filtering)."""

import numpy as np
import pytest

from tradetropy.ta.order_flow._core import (
    EVENT_ABSORPTION,
    EVENT_ICEBERG,
    EVENT_LARGE_AGGRESSOR,
    EVENT_LIQUIDITY_GRAB,
    EVENT_SWEEP,
    apply_deep_autofilter,
    deep_trade_class_name,
    resolve_autofilter,
)


# =====
# resolve_autofilter
# =====
class TestResolveAutofilter:
    def test_none_returns_none(self):
        assert resolve_autofilter(None) is None

    def test_significant_drops_aggressor(self):
        keep = resolve_autofilter("significant")
        assert keep == frozenset(
            {EVENT_ABSORPTION, EVENT_SWEEP, EVENT_ICEBERG, EVENT_LIQUIDITY_GRAB}
        )
        assert EVENT_LARGE_AGGRESSOR not in keep

    def test_single_class_name_string(self):
        assert resolve_autofilter("sweep") == frozenset({EVENT_SWEEP})

    def test_iterable_of_names(self):
        keep = resolve_autofilter({"sweep", "absorption"})
        assert keep == frozenset({EVENT_SWEEP, EVENT_ABSORPTION})

    def test_iterable_is_case_insensitive(self):
        assert resolve_autofilter(["SWEEP"]) == frozenset({EVENT_SWEEP})

    def test_aggressor_is_a_valid_name(self):
        assert resolve_autofilter(["aggressor"]) == frozenset(
            {EVENT_LARGE_AGGRESSOR}
        )

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            resolve_autofilter("xxx")

    def test_invalid_class_in_iterable_raises(self):
        with pytest.raises(ValueError):
            resolve_autofilter(["sweep", "bogus"])


# =====
# apply_deep_autofilter
# =====
class TestApplyAutofilter:
    def test_none_keeps_everything(self):
        et = np.array([0, 1, 2, 0], dtype=np.int8)
        resting = np.array([5.0, 100.0, 5.0, np.nan])
        keep = apply_deep_autofilter(et, resting, None)
        assert keep.tolist() == [True, True, True, True]

    def test_drop_aggressor_with_book(self):
        # Aggressor (0) with finite resting -> classified, dropped under
        # 'significant'. Absorption (1) / sweep (2) kept.
        et = np.array([EVENT_LARGE_AGGRESSOR, EVENT_ABSORPTION, EVENT_SWEEP])
        resting = np.array([5.0, 100.0, 5.0])
        keep = apply_deep_autofilter(
            et, resting, resolve_autofilter("significant")
        )
        assert keep.tolist() == [False, True, True]

    def test_keep_unclassified_aggressor_no_book(self):
        # Aggressor with NaN resting (no book) is kept when keep_unclassified.
        et = np.array([EVENT_LARGE_AGGRESSOR, EVENT_LARGE_AGGRESSOR])
        resting = np.array([np.nan, 5.0])
        keep = apply_deep_autofilter(
            et, resting, resolve_autofilter("significant"),
            keep_unclassified=True,
        )
        # First (no book) kept, second (classified aggressor) dropped.
        assert keep.tolist() == [True, False]

    def test_strict_mode_drops_unclassified(self):
        et = np.array([EVENT_LARGE_AGGRESSOR, EVENT_SWEEP])
        resting = np.array([np.nan, 5.0])
        keep = apply_deep_autofilter(
            et, resting, resolve_autofilter("significant"),
            keep_unclassified=False,
        )
        # No-book aggressor dropped in strict mode; sweep kept.
        assert keep.tolist() == [False, True]

    def test_mbo_classes_kept(self):
        et = np.array([EVENT_ICEBERG, EVENT_LIQUIDITY_GRAB])
        resting = np.array([10.0, 10.0])
        keep = apply_deep_autofilter(
            et, resting, resolve_autofilter("significant")
        )
        assert keep.tolist() == [True, True]

    def test_empty_arrays(self):
        keep = apply_deep_autofilter(
            np.zeros(0, dtype=np.int8), np.zeros(0),
            resolve_autofilter("significant"),
        )
        assert keep.shape == (0,)
        assert keep.dtype == bool


# =====
# deep_trade_class_name
# =====
class TestClassName:
    def test_codes_map_to_names(self):
        assert deep_trade_class_name(EVENT_LARGE_AGGRESSOR) == "aggressor"
        assert deep_trade_class_name(EVENT_ABSORPTION) == "absorption"
        assert deep_trade_class_name(EVENT_SWEEP) == "sweep"
        assert deep_trade_class_name(EVENT_ICEBERG) == "iceberg"
        assert deep_trade_class_name(EVENT_LIQUIDITY_GRAB) == "liquidity_grab"

    def test_nan_and_none_are_empty(self):
        assert deep_trade_class_name(np.nan) == ""
        assert deep_trade_class_name(None) == ""

    def test_unknown_code_is_empty(self):
        assert deep_trade_class_name(99) == ""


# =====
# DeepTrades indicator integration
# =====
class _FakeBook:
    """Minimal orderbook proxy: book_as_of(ts) -> book dict or None."""

    def __init__(self, fn):
        self._fn = fn

    def book_as_of(self, ts):
        return self._fn(int(ts))


def _book(ask_sz, bid_sz=(5.0,)):
    asks = [(100.0 + i, s) for i, s in enumerate(ask_sz)]
    bids = [(99.0 - i, s) for i, s in enumerate(bid_sz)]
    return {
        "ask_px": np.array([p for p, _ in asks], dtype=np.float64),
        "ask_sz": np.array([s for _, s in asks], dtype=np.float64),
        "bid_px": np.array([p for p, _ in bids], dtype=np.float64),
        "bid_sz": np.array([s for _, s in bids], dtype=np.float64),
    }


def _book_fn(t):
    # t=4 sweep (clears 3 levels), t=5 plain aggressor (clears 2 of 3),
    # t=6 absorption (90 into a 100 wall, not cleared).
    if t == 4:
        return _book((5.0, 5.0, 5.0, 5.0))
    if t == 5:
        return _book((5.0, 5.0, 5.0))
    if t == 6:
        return _book((100.0, 5.0))
    return _book((5.0,))


def _src():
    # columns [ts, price, volume, flags(buy=1), bid, ask]
    rows = [
        [1, 100.0, 1.0, 1, 100.0, 100.0],
        [2, 100.0, 1.0, 1, 100.0, 100.0],
        [3, 100.0, 1.0, 1, 100.0, 100.0],
        [4, 100.0, 16.0, 1, 100.0, 100.0],   # sweep
        [5, 100.0, 10.0, 1, 100.0, 100.0],   # aggressor (book present)
        [6, 100.0, 90.0, 1, 100.0, 100.0],   # absorption
    ]
    return np.array(rows, dtype=np.float64)


def _mk(autofilter=None, keep_unclassified=True):
    from tradetropy.ta import DeepTrades
    return DeepTrades(
        _FakeBook(_book_fn), threshold=5.0, by="volume", window=3,
        min_resting_volume=50.0, absorption_ratio=0.8, stack_depth=3,
        autofilter=autofilter, keep_unclassified=keep_unclassified,
    )


class TestDeepTradesConstructor:
    def test_default_none(self):
        ind = _mk(autofilter=None)
        assert ind._autofilter_classes is None
        assert ind.keep_unclassified is True

    def test_significant(self):
        ind = _mk(autofilter="significant")
        assert ind._autofilter_classes == frozenset(
            {EVENT_ABSORPTION, EVENT_SWEEP, EVENT_ICEBERG, EVENT_LIQUIDITY_GRAB}
        )

    def test_class_set(self):
        ind = _mk(autofilter={"sweep"})
        assert ind._autofilter_classes == frozenset({EVENT_SWEEP})

    def test_keep_unclassified_flag(self):
        ind = _mk(keep_unclassified=False)
        assert ind.keep_unclassified is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _mk(autofilter="xxx")

    def test_requires_orderbook(self):
        from tradetropy.exceptions import ConfigError
        from tradetropy.ta import DeepTrades
        with pytest.raises(ConfigError):
            DeepTrades(orderbook=None, threshold=5.0, by="volume", window=3)


class TestDeepTradesCalculate:
    def _classes(self, ind, out):
        idx = np.where(~np.isnan(out[0]))[0]
        return {int(out[0, i]): int(out[4, i]) for i in idx}, idx

    def test_baseline_detects_three(self):
        ind = _mk(autofilter=None)
        out = ind.calculate(_src())
        idx = np.where(~np.isnan(out[0]))[0]
        # sweep(4), aggressor(5), absorption(6) all present.
        types = sorted(int(out[4, i]) for i in idx)
        assert types == [
            EVENT_LARGE_AGGRESSOR, EVENT_ABSORPTION, EVENT_SWEEP,
        ] or types == sorted(
            [EVENT_SWEEP, EVENT_LARGE_AGGRESSOR, EVENT_ABSORPTION]
        )
        assert len(ind._deep_events["ts"]) == idx.size == 3

    def test_significant_drops_aggressor_with_book(self):
        ind = _mk(autofilter="significant")
        out = ind.calculate(_src())
        idx = np.where(~np.isnan(out[0]))[0]
        kept = sorted(int(out[4, i]) for i in idx)
        assert EVENT_LARGE_AGGRESSOR not in kept
        assert set(kept) == {EVENT_SWEEP, EVENT_ABSORPTION}
        # coherence: _deep_events filtered identically.
        assert len(ind._deep_events["ts"]) == idx.size == 2
        assert EVENT_LARGE_AGGRESSOR not in ind._deep_events["event_type"]

    def test_none_identical_to_no_filter(self):
        base = _mk(autofilter=None).calculate(_src())
        # An indicator with autofilter that keeps every class must match.
        keep_all = _mk(
            autofilter={
                "aggressor", "absorption", "sweep", "iceberg", "liquidity_grab",
            }
        ).calculate(_src())
        np.testing.assert_array_equal(
            np.nan_to_num(base), np.nan_to_num(keep_all)
        )


class TestDeepTradesDrawAndParity:
    def test_draw_only_significant(self):
        ind = _mk(autofilter="significant")
        ind.calculate(_src())
        prims = ind.draw()
        assert len(prims) >= 1
        # Points x length matches the filtered events (2 significant).
        assert len(prims["Large Trades"][0].x) == 2

    def test_class_name_accessor(self):
        from tradetropy.ta import DeepTrades
        ind = _mk(autofilter=None)
        out = ind.calculate(_src())
        idx = np.where(~np.isnan(out[0]))[0]
        names = {DeepTrades.class_name(out[4, i]) for i in idx}
        assert names == {"aggressor", "absorption", "sweep"}
        # NaN tick -> empty string.
        assert DeepTrades.class_name(out[4, 0]) == ""

    def test_live_refresh_parity(self):
        # live_refresh rebuilds the source from the tick proxy and reuses
        # calculate, so the filtered events must match the direct path.
        src = _src()
        direct = _mk(autofilter="significant")
        direct.calculate(src)

        class _Col:
            def __init__(self, a):
                self._a = a

            def __getitem__(self, k):
                return self._a[k]

        class _Proxy:
            def __init__(self, s):
                self.ts = _Col(s[:, 0])
                self.price = _Col(s[:, 1])
                self.volume = _Col(s[:, 2])
                self.flags = _Col(s[:, 3])
                self.bid = _Col(s[:, 4])
                self.ask = _Col(s[:, 5])

            def __len__(self):
                return 6

        live = _mk(autofilter="significant")
        live.live_refresh(_Proxy(src))

        np.testing.assert_array_equal(
            direct._deep_events["ts"], live._deep_events["ts"]
        )
        np.testing.assert_array_equal(
            direct._deep_events["event_type"], live._deep_events["event_type"]
        )



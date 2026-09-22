"""
Inspect MarketStructure (BOS / CHoCH) in isolation - no strategy, no orders.

This script exists to answer one question: *is the indicator telling me the
truth?* It prints what every band carries, proves the detector is causal, and
renders a chart so the printed events can be checked against the candles by
eye.

Run:
    python examples/market_structure_inspect.py            # print + chart
    python examples/market_structure_inspect.py --no-chart # print only

What it checks
--------------
1. Band layout       - what each of the 9 output bands means and holds.
2. Event table       - every BOS/CHoCH with its broken level and origin bar.
3. Causality         - recompute over growing prefixes; the result must never
                       change retroactively (the no-lookahead proof).
4. Break rule        - each event's break bar really satisfies its own rule.
5. close vs wick     - how the break mode changes the event count.
6. Chart             - visual cross-check of the printed events.
"""

from __future__ import annotations

import argparse

import numpy as np

from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import MarketStructure
from tradetropy.ta.structure.market_structure import _MS_TAG_CODE

SWING = 3
BREAK_ON = "close"

_TAG_DECODE = {v: k for k, v in _MS_TAG_CODE.items()}
_BAND_OF = {
    "BOS-bull": 0, "BOS-bear": 1, "CHOCH-bull": 2, "CHOCH-bear": 3,
}


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def build_source(kl):
    """MarketStructure consumes [N x 4]: high, low, close, ts_ms."""
    d = kl.data          # columns: ts, open, high, low, close, volume, turnover
    return np.column_stack([d[:, 2], d[:, 3], d[:, 4], d[:, 0]])


def collect_events(out):
    """Flatten the band matrix into [(bar_idx, tag, level, ts_origin)]."""
    evs = []
    for i in range(out.shape[1]):
        code = out[8, i]
        if np.isnan(code):
            continue
        tag = _TAG_DECODE[float(code)]
        band = _BAND_OF[tag]
        evs.append((i, tag, float(out[band, i]), float(out[band + 4, i])))
    return evs


# =====
# 1. BAND LAYOUT
# =====
def show_band_layout(ms, out):
    rule("1. BAND LAYOUT - what the indicator returns")
    print(f"calculate() -> array {out.shape}  ({out.shape[0]} bands x N bars)\n")

    print("  PRICE bands (causal - safe to trade on):")
    for i, name in enumerate(ms.output_names):
        n = int(np.sum(~np.isnan(out[i])))
        print(f"    b{i}  {name:<12} {n:>4} events   price of the level that broke")

    print("\n  TIMESTAMP bands (plotting only - NOT signals):")
    for k, name in enumerate(ms.ts_output_names[:4]):
        print(f"    b{k + 4}  {name:<15}      ts of the REAL swing bar that was broken")
    print(f"    b8  _tag                    encoded tag (internal)")

    print("\n  Why the split: the VALUE lands on the bar where the break")
    print("  happened (that is when a strategy may act), while the TS points")
    print("  back at the swing bar that formed the level, so the chart can")
    print("  draw the structure line to its origin without any lookahead.")


# =====
# 2. EVENT TABLE
# =====
def show_events(evs, src, limit=18):
    rule("2. EVENT TABLE - what each event says")
    ts = src[:, 3]
    print(f"{'bar':>5}  {'tag':<11} {'broken level':>13}  {'bars since origin':>18}")
    print("-" * 74)
    for bar_idx, tag, level, ts_origin in evs[:limit]:
        origin_bar = int(np.argmin(np.abs(ts - ts_origin)))
        print(f"{bar_idx:>5}  {tag:<11} {level:>13,.1f}  {bar_idx - origin_bar:>18}")
    if len(evs) > limit:
        print(f"{'...':>5}  ({len(evs) - limit} more)")

    counts = {}
    for _i, tag, _l, _t in evs:
        counts[tag] = counts.get(tag, 0) + 1
    print(f"\n  totals: {counts}")
    print(f"  total events: {len(evs)}")
    print("\n  Reading it:")
    print("    BOS-bull    price broke a swing HIGH, continuing a bull bias")
    print("    BOS-bear    price broke a swing LOW,  continuing a bear bias")
    print("    CHOCH-bull  bearish structure just flipped bullish")
    print("    CHOCH-bear  bullish structure just flipped bearish")
    print("\n  'bars since origin' is the gap between the swing that formed the")
    print("  level and the bar that broke it - never 0, which is the whole")
    print("  point: the level had to be knowable BEFORE it could break.")


# =====
# 3. CAUSALITY
# =====
def check_causality(src, swing=SWING, step=17):
    rule("3. CAUSALITY - the no-lookahead proof")
    full = MarketStructure(swing=swing, break_on=BREAK_ON).calculate(src)
    n = len(src)
    tested = failed = 0

    for k in range(40, n + 1, step):
        pref = MarketStructure(swing=swing, break_on=BREAK_ON).calculate(src[:k])
        a, b = full[:8, :k], pref[:8, :k]
        tested += 1
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            failed += 1
            continue
        m = ~np.isnan(a)
        if not np.allclose(a[m], b[m]):
            failed += 1

    print(f"  recomputed over {tested} growing prefixes of the series")
    print(f"  mismatches: {failed}")
    if failed == 0:
        print("\n  PASS - every prefix reproduces the full-history result exactly.")
        print("  Nothing is ever rewritten when new bars arrive, so what the")
        print("  backtest saw at bar k is what a live run would have seen.")
    else:
        print("\n  FAIL - results changed retroactively: there is lookahead.")
    return failed == 0


# =====
# 4. BREAK RULE
# =====
def check_break_rule(src, swing=SWING):
    rule("4. BREAK RULE - does each event satisfy its own condition?")
    high, low, close = src[:, 0], src[:, 1], src[:, 2]
    all_ok = True

    for mode in ("close", "wick"):
        out = MarketStructure(swing=swing, break_on=mode).calculate(src)
        up, down = (close, close) if mode == "close" else (high, low)
        bad = 0
        evs = collect_events(out)
        for bar_idx, tag, level, _ts in evs:
            ok = up[bar_idx] > level if tag.endswith("bull") else down[bar_idx] < level
            bad += not ok
        all_ok &= bad == 0
        src_name = "close" if mode == "close" else "high/low"
        print(f"  break_on={mode:<6} {len(evs):>3} events, {bad} violations "
              f"(checked against {src_name})")

    print("\n  PASS - every break is real." if all_ok else "\n  FAIL")
    return all_ok


# =====
# 5. CLOSE VS WICK
# =====
def compare_modes(src, swing=SWING):
    rule("5. break_on MODE - how strict do you want the break?")
    for mode in ("close", "wick"):
        evs = collect_events(
            MarketStructure(swing=swing, break_on=mode).calculate(src)
        )
        counts = {}
        for _i, tag, _l, _t in evs:
            counts[tag] = counts.get(tag, 0) + 1
        print(f"  {mode:<6} {len(evs):>3} events   {counts}")
    print("\n  'close' needs a candle to CLOSE beyond the level; 'wick' accepts")
    print("  a pierce. 'wick' fires earlier and more often, 'close' filters out")
    print("  stop-hunt pierces. 'wick' can never report fewer events.")

    print("\n  swing window (structure granularity):")
    for sw in (2, 3, 5, 8):
        evs = collect_events(
            MarketStructure(swing=sw, break_on=BREAK_ON).calculate(src)
        )
        print(f"    swing={sw:<2} {len(evs):>3} events   "
              f"(needs {sw * 2 + 1} bars to confirm a pivot)")
    print("  Higher swing = coarser structure = fewer, more significant breaks.")


# =====
# 6. CHART
# =====
def render_chart(kl, swing=SWING, path="market_structure_inspect.html"):
    rule("6. CHART - visual cross-check")
    from tradetropy import BacktestEngine, Strategy
    from tradetropy.ta import SwingHL

    class _Inspect(Strategy):
        """Plots the structure only - places no orders."""

        def init(self):
            self.ohlc = self.subscribe_ohlc(
                kl.symbol, kl.timeframe, window_size=len(kl.data)
            )
            self.ms = self.add_indicator(
                MarketStructure.refs(self.ohlc),
                MarketStructure(swing=swing, break_on=BREAK_ON),
                plot=True, overlay=True,
            )
            self.sw = self.add_indicator(
                SwingHL.refs(self.ohlc),
                SwingHL(swing=swing, show_lines=False),
                plot=True, overlay=True,
            )
            self.seen: list[tuple[int, str]] = []

        def on_data(self):
            tag = MarketStructure.event(self.ms)
            if tag:
                self.seen.append((len(self.ohlc.ts), tag))

    bt = BacktestEngine.by_klines(_Inspect(), data=(kl,))
    bt.run(verbose=False)

    print(f"  events seen live in on_data(): {len(bt.strategy.seen)}")
    print("  (must match the event table above - if it does not, the strategy")
    print("   read the wrong slot; use MarketStructure.event())")

    bt.plot(
        output="file", filename=path, theme="dark",
        plot_drawdown=False, plot_pl=False, equity_mode="none",
    )
    print(f"\n  chart -> {path}")
    print("  On the chart:")
    print("    triangles         BOS   (structure continued)")
    print("    diamonds          CHoCH (structure flipped)")
    print("    solid line        BOS level, from swing bar to break bar")
    print("    dashed line       CHoCH level")
    print("    small triangles   SwingHL - the pivots the levels come from")
    return bt.strategy.seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-chart", action="store_true", help="skip chart rendering")
    ap.add_argument("--swing", type=int, default=SWING)
    args = ap.parse_args()

    swing = args.swing

    kl = load_btcusd_1m()
    src = build_source(kl)

    print(f"\nMarketStructure inspection")
    print(f"  data     {kl.symbol} {kl.timeframe}, {len(src)} bars")
    print(f"  params   swing={swing}, break_on={BREAK_ON!r}")

    ms = MarketStructure(swing=swing, break_on=BREAK_ON)
    out = ms.calculate(src)
    evs = collect_events(out)

    show_band_layout(ms, out)
    show_events(evs, src)
    causal_ok = check_causality(src, swing)
    rule_ok = check_break_rule(src, swing)
    compare_modes(src, swing)

    live_seen = None
    if not args.no_chart:
        live_seen = render_chart(kl, swing)

    rule("SUMMARY")
    print(f"  events detected      {len(evs)}")
    print(f"  causality            {'PASS' if causal_ok else 'FAIL'}")
    print(f"  break rule           {'PASS' if rule_ok else 'FAIL'}")
    if live_seen is not None:
        match = len(live_seen) == len(evs)
        print(f"  live == calculate    {'PASS' if match else 'FAIL'} "
              f"({len(live_seen)} vs {len(evs)})")
    print()


if __name__ == "__main__":
    main()

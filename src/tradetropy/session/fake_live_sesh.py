"""
fake_live_sesh.py
=================
Simulated session that generates random market data in real time (GBM),
mimicking SeshMT5Live for testing strategies with LiveEngine.
"""

from __future__ import annotations

import time
import threading
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from tradetropy.session.base import SeshSimulatorBase
from tradetropy.core.constants import N_TICK_COLS, _TICK_COL
from tradetropy.core.broker import AccountInfo, SymbolConfig
from tradetropy.exceptions import TradingError, ConfigError


# =====
# Heuristic defaults
# =====
_DEFAULT_VOLATILITY  = 0.0003
_DEFAULT_SPREAD_REL  = 0.0001
_DEFAULT_VOLUME_BASE = 10.0
_DEFAULT_VOLUME_NOISE = 5.0


# =====
# Interpolation helpers for kline->tick synthesis
# =====

def _interpolate_waypoints(waypoints: list[float], n: int) -> np.ndarray:
    if n <= 1:
        return np.array([waypoints[-1]], dtype=np.float64)
    dists = [abs(waypoints[i+1] - waypoints[i]) for i in range(len(waypoints)-1)]
    total = sum(dists) or 1.0
    n_segs = len(waypoints) - 1
    counts = [max(1, round(d / total * (n - 1))) for d in dists]
    diff = sum(counts) - (n - 1)
    longest = max(range(n_segs), key=lambda i: counts[i])
    counts[longest] -= diff
    result = []
    for seg_i in range(n_segs):
        start = waypoints[seg_i]
        end = waypoints[seg_i + 1]
        cnt = counts[seg_i]
        include_end = (seg_i == n_segs - 1)
        pts = np.linspace(start, end, cnt + 1)
        result.append(pts if include_end else pts[:-1])
    return np.concatenate(result)


# =====
# SYMBOL SPEC - rich per-symbol configuration
# =====

@dataclass
class _SymbolSpec:
    """
    Full symbol configuration for FakeLiveSesh.

    Price / market parameters
    =========================
    symbol        : symbol name (e.g. "MES", "EURUSD", "BTCUSDT")
    initial_price : opening price for the simulation
    volatility    : relative sigma per tick (default 0.0003 ~= 0.03% per tick).
                    Realistic examples:
                      Forex major (EURUSD) -> 0.00015
                      Index (NAS100)       -> 0.0008
                      Futures (MES)        -> 0.0006
                      Crypto (BTCUSDT)     -> 0.0015
    spread        : absolute spread in price units.
                    If None, uses initial_price * 0.0001.
    drift         : drift per tick in log-space (default 0.0 = no trend).
                    Positive -> upward trend, negative -> downward.

    Contract parameters (for the internal broker)
    ==============================================
    tick_size     : minimum price step (e.g. 0.25 for MES, 0.00001 for EURUSD)
    tick_value    : monetary value of one tick per contract (e.g. 1.25 for MES)
    contract_size : contract size (e.g. 5.0 for MES, 100_000 for EURUSD)
    digits        : price decimal places for normalization and display
    volume_min    : minimum order size
    volume_max    : maximum order size
    volume_step   : minimum volume increment

    Generator parameters
    ====================
    volume_base   : base volume per generated tick
    volume_noise  : uniform noise +/- applied to volume_base

    Examples
    ========
        # Micro E-mini S&P 500
        _SymbolSpec("MES", initial_price=5_200.0,
                   tick_size=0.25, tick_value=1.25, contract_size=5.0,
                   digits=2, volatility=0.0006, spread=0.25)

        # EUR/USD forex
        _SymbolSpec("EURUSD", initial_price=1.0850,
                   tick_size=0.00001, tick_value=1.0, contract_size=100_000.0,
                   digits=5, volatility=0.00015, spread=0.00003)

        # Bitcoin perpetual
        _SymbolSpec("BTCUSDT", initial_price=65_000.0,
                   tick_size=0.1, tick_value=0.1, contract_size=1.0,
                   digits=1, volatility=0.0015, spread=5.0,
                   volume_base=0.5, volume_noise=0.3)
    """

    symbol:        str
    initial_price: float

    volatility:    float         = _DEFAULT_VOLATILITY
    spread:        Optional[float] = None
    drift:         float         = 0.0

    tick_size:     float = 0.01
    tick_value:    float = 0.01
    contract_size: float = 1.0
    digits:        int   = 2
    volume_min:    float = 0.01
    volume_max:    float = 100.0
    volume_step:   float = 0.01

    volume_base:   float = _DEFAULT_VOLUME_BASE
    volume_noise:  float = _DEFAULT_VOLUME_NOISE

    def __post_init__(self):
        if self.spread is None:
            self.spread = self.initial_price * _DEFAULT_SPREAD_REL
        if self.tick_size > 0 and self.spread < self.tick_size:
            self.spread = self.tick_size

    @property
    def symbol_config(self) -> SymbolConfig:
        return SymbolConfig(
            name          = self.symbol,
            tick_size     = self.tick_size,
            tick_value    = self.tick_value,
            contract_size = self.contract_size,
            digits        = self.digits,
            volume_min    = self.volume_min,
            volume_max    = self.volume_max,
            volume_step   = self.volume_step,
        )

    def __repr__(self) -> str:
        return (
            f"_SymbolSpec({self.symbol!r}, price={self.initial_price}, "
            f"vol={self.volatility}, tick_size={self.tick_size}, "
            f"tick_value={self.tick_value})"
        )


# =====
# INTERNAL STATE PER SYMBOL (GBM generator)
# =====

class _SymbolState:
    """GBM generator for a symbol. Internal - use _SymbolSpec as the interface."""

    __slots__ = (
        "spec",
        "price",
        "_prev_price",
        "tick_history",
        "_lock",
    )

    def __init__(self, spec: _SymbolSpec):
        self.spec = spec
        self.price = float(spec.initial_price)
        self._prev_price = float(spec.initial_price)
        self.tick_history: list[np.ndarray] = []
        self._lock = threading.Lock()

    def _round(self, price: float) -> float:
        ts = self.spec.tick_size
        if ts <= 0:
            return price
        return round(round(price / ts) * ts, self.spec.digits)

    def generate_tick(self, ts_ms: Optional[int] = None) -> np.ndarray:
        with self._lock:
            prev_price = self._prev_price

            eps = np.random.standard_normal()
            self.price = self._round(
                self.price * np.exp(self.spec.drift + self.spec.volatility * eps)
            )
            self.price = max(self.price, self.spec.tick_size)

            half = self.spec.spread / 2.0
            bid  = self._round(self.price - half)
            ask  = self._round(self.price + half)
            vol  = max(0.0, self.spec.volume_base + np.random.uniform(
                -self.spec.volume_noise, self.spec.volume_noise
            ))
            ts = ts_ms if ts_ms is not None else int(time.time() * 1000)

            if self.price > prev_price:
                flag = float(8 | 32)
            elif self.price < prev_price:
                flag = float(8 | 64)
            else:
                flag = float(8 | (32 if np.random.random() < 0.5 else 64))

            self._prev_price = self.price

            tick = np.empty(N_TICK_COLS, dtype=np.float64)
            tick[_TICK_COL["ts"]]          = ts
            tick[_TICK_COL["bid"]]         = bid
            tick[_TICK_COL["ask"]]         = ask
            tick[_TICK_COL["price"]]       = self.price
            tick[_TICK_COL["volume"]]      = vol
            tick[_TICK_COL["flags"]]       = flag
            tick[_TICK_COL["volume_real"]] = vol

            self.tick_history.append(tick.copy())
            return tick

    def generate_history_ticks(self, limit: int) -> np.ndarray:
        if limit <= 0:
            return np.empty((0, N_TICK_COLS), dtype=np.float64)

        now_ms = int(time.time() * 1000)

        with self._lock:
            price = self.price

        prices = np.empty(limit, dtype=np.float64)
        for i in range(limit - 1, -1, -1):
            eps = np.random.standard_normal()
            price = price / np.exp(self.spec.drift + self.spec.volatility * eps)
            prices[i] = max(price, self.spec.tick_size)

        out = np.empty((limit, N_TICK_COLS), dtype=np.float64)
        half = self.spec.spread / 2.0
        for i in range(limit):
            p    = self._round(prices[i])
            prev = self._round(prices[i - 1]) if i > 0 else p
            vol  = max(0.0, self.spec.volume_base + np.random.uniform(
                -self.spec.volume_noise, self.spec.volume_noise
            ))

            if p > prev:
                flag = float(8 | 32)
            elif p < prev:
                flag = float(8 | 64)
            else:
                flag = float(8 | (32 if np.random.random() < 0.5 else 64))

            out[i, _TICK_COL["ts"]]          = now_ms - (limit - 1 - i) * 1000
            out[i, _TICK_COL["bid"]]         = self._round(p - half)
            out[i, _TICK_COL["ask"]]         = self._round(p + half)
            out[i, _TICK_COL["price"]]       = p
            out[i, _TICK_COL["volume"]]      = vol
            out[i, _TICK_COL["flags"]]       = flag
            out[i, _TICK_COL["volume_real"]] = vol
        return out

    def generate_history_klines(self, interval_ms: int, limit: int) -> np.ndarray:
        if limit <= 0:
            return np.empty((0, 6), dtype=np.float64)

        now_ms     = int(time.time() * 1000)
        ts_current = (now_ms // interval_ms) * interval_ms

        with self._lock:
            price = self.price

        ticks_per_kline = max(1, interval_ms // 1000)
        sigma_bar = self.spec.volatility * np.sqrt(ticks_per_kline)

        closes = np.empty(limit, dtype=np.float64)
        p = price
        for i in range(limit - 1, -1, -1):
            eps = np.random.standard_normal()
            p = p / np.exp(self.spec.drift * ticks_per_kline + sigma_bar * eps)
            closes[i] = max(p, self.spec.tick_size)

        out = np.empty((limit, 6), dtype=np.float64)
        for i in range(limit):
            ts = ts_current - (limit - i) * interval_ms
            c  = self._round(closes[i])
            o  = self._round(closes[i - 1]) if i > 0 else self._round(
                c * np.exp(-sigma_bar * np.random.standard_normal() * 0.5)
            )
            rng = abs(c - o) + sigma_bar * abs(np.random.standard_normal()) * c
            h = self._round(max(o, c) + rng * abs(np.random.standard_normal()) * 0.5)
            l = self._round(min(o, c) - rng * abs(np.random.standard_normal()) * 0.5)
            l = max(l, self.spec.tick_size)
            vol = max(0.0, self.spec.volume_base * ticks_per_kline + np.random.uniform(
                -self.spec.volume_noise, self.spec.volume_noise
            ) * ticks_per_kline)

            out[i] = [ts, o, h, l, c, vol]
        return out

    def _generate_ticks_from_klines(
        self,
        klines: np.ndarray,
        ticks_per_kline: int = 10,
        interval_ms: int = 60_000,
    ) -> np.ndarray:
        if len(klines) == 0:
            return np.empty((0, N_TICK_COLS), dtype=np.float64)

        rows = []
        half = self.spec.spread / 2.0

        for kline in klines:
            ts_bar, o, h, l, c, vol = (
                int(kline[0]), float(kline[1]), float(kline[2]),
                float(kline[3]), float(kline[4]), float(kline[5])
            )

            if c >= o:
                waypoints = [o, l, h, c]
            else:
                waypoints = [o, h, l, c]

            n = max(ticks_per_kline, 4)
            prices = _interpolate_waypoints(waypoints, n)

            ts_end = ts_bar + interval_ms - 1
            timestamps = np.linspace(ts_bar, ts_end, n, dtype=np.int64)

            vol_per_tick = vol / n if n > 0 else 0.0

            for i, (ts_tick, price) in enumerate(zip(timestamps, prices)):
                price = self._round(price)
                bid = self._round(price - half)
                ask = self._round(price + half)

                if i == 0:
                    flag = float(8)
                elif price > prices[i - 1]:
                    flag = float(8 | 32)
                elif price < prices[i - 1]:
                    flag = float(8 | 64)
                else:
                    flag = float(8)

                tick = np.empty(N_TICK_COLS, dtype=np.float64)
                tick[_TICK_COL["ts"]]          = float(ts_tick)
                tick[_TICK_COL["bid"]]         = bid
                tick[_TICK_COL["ask"]]         = ask
                tick[_TICK_COL["price"]]       = price
                tick[_TICK_COL["volume"]]      = vol_per_tick
                tick[_TICK_COL["flags"]]       = flag
                tick[_TICK_COL["volume_real"]] = vol_per_tick
                rows.append(tick)

        return np.array(rows, dtype=np.float64)

    def reset(self) -> None:
        with self._lock:
            self.price = float(self.spec.initial_price)
            self.tick_history.clear()


# =====
# FAKE LIVE SESH - generates synthetic data in real time
# =====

class FakeLiveSesh(SeshSimulatorBase):
    """
    Simulated session that mimics SeshMT5Live by generating synthetic prices
    with GBM.

    Compatible with LiveEngine.by_ticks() and LiveEngine.by_klines() without
    changing a single line of the strategy.

    Symbols are added after creating the session via set_symbol().

    Session parameters
    ------------------
    feed_type       : "tick" or "kline" -- internal broker type
    initial_balance : initial balance of the simulated account
    commission      : commission per trade
    seed            : seed for reproducibility (None = random)

    Example -- MES + EURUSD
    -----------------------
        sesh = FakeLiveSesh(initial_balance=25_000.0)
        sesh.set_symbol("MES",    initial_price=5_200.0,
                        tick_size=0.25,    tick_value=1.25,
                        contract_size=5.0, digits=2,
                        volatility=0.0006, spread=0.25)
        sesh.set_symbol("EURUSD", initial_price=1.0850,
                        tick_size=0.00001, tick_value=1.0,
                        contract_size=100_000.0, digits=5,
                        volatility=0.00015, spread=0.00003)
        engine = LiveEngine.by_ticks(MyStrategy(), sesh=sesh)
        engine.run()

    Example -- construction with from_dicts
    ----------------------------------------
        sesh = FakeLiveSesh.from_dicts(
            symbols       = ["BTCUSDT"],
            initial_price = {"BTCUSDT": 65_000.0},
            volatility    = {"BTCUSDT": 0.0015},
            spread        = {"BTCUSDT": 10.0},
        )
    """

    def __init__(
        self,
        feed_type: "str | None" = None,
        initial_balance: float = 10_000.0,
        commission: float = 0.0,
        seed: Optional[int] = None,
        **broker_kwargs,
    ):
        super().__init__(
            feed_type=feed_type,
            initial_balance=initial_balance,
            commission=commission,
            **broker_kwargs,
        )

        if seed is not None:
            np.random.seed(seed)

        self._states: dict[str, _SymbolState] = {}
        self._last_ts_per_symbol: dict[str, int] = {}

    def set_symbol(
        self,
        symbol: str,
        initial_price: float,
        volatility: float = _DEFAULT_VOLATILITY,
        spread: Optional[float] = None,
        drift: float = 0.0,
        tick_size: float = 0.01,
        tick_value: float = 0.01,
        contract_size: float = 1.0,
        digits: int = 2,
        volume_min: float = 0.01,
        volume_max: float = 100.0,
        volume_step: float = 0.01,
        volume_base: float = _DEFAULT_VOLUME_BASE,
        volume_noise: float = _DEFAULT_VOLUME_NOISE,
    ) -> None:
        """
        Add or overwrite a symbol in the simulation.

        All contract and generator parameters are configured here,
        without needing to import _SymbolSpec.

        Example:
            sesh.set_symbol("EURUSD", initial_price=1.0850,
                            volatility=0.00015, spread=0.00003,
                            tick_size=0.00001, tick_value=1.0,
                            contract_size=100_000.0, digits=5)
        """
        spec = _SymbolSpec(
            symbol=symbol,
            initial_price=initial_price,
            volatility=volatility,
            spread=spread,
            drift=drift,
            tick_size=tick_size,
            tick_value=tick_value,
            contract_size=contract_size,
            digits=digits,
            volume_min=volume_min,
            volume_max=volume_max,
            volume_step=volume_step,
            volume_base=volume_base,
            volume_noise=volume_noise,
        )
        self._states[symbol] = _SymbolState(spec)
        self._last_ts_per_symbol[symbol] = 0
        try:
            self.configure_symbol(spec)
        except Exception:
            pass

    @classmethod
    def from_dicts(
        cls,
        symbols:        list[str],
        initial_price:  Optional[dict[str, float]] = None,
        volatility:     Optional[dict[str, float]] = None,
        spread:         Optional[dict[str, float]] = None,
        drift:          Optional[dict[str, float]] = None,
        volume_base:    Optional[dict[str, float]] = None,
        tick_size:      Optional[dict[str, float]] = None,
        tick_value:     Optional[dict[str, float]] = None,
        contract_size:  Optional[dict[str, float]] = None,
        digits:         Optional[dict[str, int]]   = None,
        **kwargs,
    ) -> "FakeLiveSesh":
        """
        Alternative constructor with dicts separated by parameter.
        Useful for quick configurations of one or two symbols.

        Example:
            sesh = FakeLiveSesh.from_dicts(
                symbols       = ["EURUSD"],
                initial_price = {"EURUSD": 1.0850},
                volatility    = {"EURUSD": 0.00015},
                spread        = {"EURUSD": 0.00003},
            )
        """
        ip  = initial_price or {}
        vol = volatility    or {}
        spd = spread        or {}
        dr  = drift         or {}
        vb  = volume_base   or {}
        ts  = tick_size     or {}
        tv  = tick_value    or {}
        cs  = contract_size or {}
        dg  = digits        or {}

        sesh = cls(**kwargs)
        for sym in symbols:
            sesh.set_symbol(
                symbol        = sym,
                initial_price = float(ip.get(sym, 1000.0)),
                volatility    = float(vol.get(sym, _DEFAULT_VOLATILITY)),
                spread        = float(spd[sym]) if sym in spd else None,
                drift         = float(dr.get(sym, 0.0)),
                volume_base   = float(vb.get(sym, _DEFAULT_VOLUME_BASE)),
                tick_size     = float(ts.get(sym, 0.01)),
                tick_value    = float(tv.get(sym, 0.01)),
                contract_size = float(cs.get(sym, 1.0)),
                digits        = int(dg.get(sym, 2)),
            )
        return sesh

    # ── Public utility API ──────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        return self._get_state(symbol).price

    def set_price(self, symbol: str, price: float) -> None:
        state = self._get_state(symbol)
        with state._lock:
            state.price = float(price)

    def get_tick_history(self, symbol: str) -> np.ndarray:
        state = self._get_state(symbol)
        with state._lock:
            if not state.tick_history:
                return np.empty((0, N_TICK_COLS), dtype=np.float64)
            return np.array(state.tick_history, dtype=np.float64)

    def reset_symbol(self, symbol: str) -> None:
        self._get_state(symbol).reset()
        self._last_ts_per_symbol[symbol] = 0

    def add_symbol_sim(self, spec: _SymbolSpec) -> None:
        self.set_symbol(
            symbol=spec.symbol,
            initial_price=spec.initial_price,
            volatility=spec.volatility,
            spread=spec.spread,
            drift=spec.drift,
            tick_size=spec.tick_size,
            tick_value=spec.tick_value,
            contract_size=spec.contract_size,
            digits=spec.digits,
            volume_min=spec.volume_min,
            volume_max=spec.volume_max,
            volume_step=spec.volume_step,
            volume_base=spec.volume_base,
            volume_noise=spec.volume_noise,
        )

    # ── Data primitives (LiveEngine API) ────────────────────────────────

    def _fetch_ticks_history(self, symbol: str, limit: int = 500) -> np.ndarray:
        return self._get_state(symbol).generate_history_ticks(limit)

    def _fetch_last_tick(self, symbol: str) -> np.ndarray:
        state  = self._get_state(symbol)
        now_ms = int(time.time() * 1000)
        last   = self._last_ts_per_symbol.get(symbol, 0)
        if now_ms <= last:
            now_ms = last + 1
        tick = state.generate_tick(ts_ms=now_ms)
        self._last_ts_per_symbol[symbol] = now_ms
        self._last_ts = now_ms
        self._update_broker_tick(symbol, tick, now_ms)
        return tick

    def _fetch_klines_history(
        self, symbol: str, interval_ms: int, limit: int = 200
    ) -> np.ndarray:
        return self._get_state(symbol).generate_history_klines(interval_ms, limit)

    def _fetch_coherent_history(
        self,
        symbol: str,
        n_ticks: int,
        ohlc_proxies: list,
    ) -> dict:
        data = {}

        if not ohlc_proxies:
            data[symbol] = self._fetch_ticks_history(symbol, n_ticks)
            return data

        intervals = sorted(set(op.interval_ms for op in ohlc_proxies))

        klines_per_interval = {}
        for interval in intervals:
            # Warm-fill a full ring window of klines so recursive indicators
            # (RSI = Wilder, MACD = EMA) seed from the same trailing causal
            # window as the backtest, keeping backtest/live parity.
            n_klines_needed = max(
                50,
                n_ticks * 1000 // interval + 1,
                int(max(
                    (op._window_size for op in ohlc_proxies
                     if op.interval_ms == interval),
                    default=50,
                )),
            )
            klines = self._get_state(symbol).generate_history_klines(interval, n_klines_needed)
            klines_per_interval[interval] = klines
            data[(symbol, interval)] = klines

        finest_interval = intervals[0]
        finest_klines = klines_per_interval[finest_interval]

        ticks_per_kline = max(4, n_ticks // max(len(finest_klines), 1))
        ticks = self._get_state(symbol)._generate_ticks_from_klines(
            finest_klines,
            ticks_per_kline=ticks_per_kline,
            interval_ms=finest_interval,
        )
        data[symbol] = ticks

        return data

    def _fetch_last_kline(self, symbol: str, interval_ms: int) -> np.ndarray:
        state  = self._get_state(symbol)
        now_ms = int(time.time() * 1000)
        ts_bar = (now_ms // interval_ms) * interval_ms
        tick   = state.generate_tick(ts_ms=now_ms)
        self._last_ts = now_ms
        self._update_broker_tick(symbol, tick, now_ms)
        c = float(tick[_TICK_COL["price"]])
        v = float(tick[_TICK_COL["volume"]])
        return np.array([float(ts_bar), c, c, c, c, v], dtype=np.float64)

    # ── Custom methods compatible with SeshMT5Sim ───────────────────────

    def close_all(self, symbol: str) -> list:
        return [{"ticket": p.ticket, "ok": self.position_close(p.ticket)}
                for p in self.positions(symbol)]

    def get_last_price(self, symbol: str) -> float:
        return self.get_price(symbol)

    def calculate_margin(self, symbol: str, volume: float, side: str = "Buy") -> float:
        cfg = self._broker._get_symbol_config(symbol)
        if not self._broker.use_margin:
            return 0.0
        return volume * cfg.contract_size * self.get_price(symbol) * self._broker.margin_rate

    def calculate_profit(
        self, symbol: str, volume: float,
        price_open: float, price_close: float, side: str = "Buy",
    ) -> float:
        cfg = self._broker._get_symbol_config(symbol)
        diff = (price_close - price_open) if side.lower() == "buy" else (price_open - price_close)
        return (diff / cfg.tick_size) * cfg.tick_value * volume

    def get_account_info(self) -> dict:
        a = self._broker.account_info()
        return {**vars(a), "login": 0, "leverage": 1,
                "currency": "USD", "company": "FakeLive", "server": "Simulation"}

    def get_instrument_info(self, symbol: str) -> dict:
        spec = self._get_state(symbol).spec
        return {
            "symbol":              spec.symbol,
            "digits":              spec.digits,
            "volume_min":          spec.volume_min,
            "volume_max":          spec.volume_max,
            "volume_step":         spec.volume_step,
            "tick_step":           spec.tick_size,
            "tick_value":          spec.tick_value,
            "trade_contract_size": spec.contract_size,
        }

    def is_market_open(self, symbol: str) -> bool:
        return True

    def disconnect(self) -> None: pass
    def reconnect(self) -> None:  pass
    def is_connected(self) -> bool: return True
    def is_market_open(self, symbol: str) -> bool: return True

    # ── feed internal broker ────────────────────────────────────────────

    def _update_broker_tick(self, symbol: str, tick: np.ndarray, ts_ms: int) -> None:
        if not hasattr(self._broker, 'update_tick'):
            return
        from datetime import datetime, timezone
        ts  = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        bid = float(tick[_TICK_COL["bid"]])
        ask = float(tick[_TICK_COL["ask"]])
        if bid <= 0 or ask <= 0:
            mid = float(tick[_TICK_COL["price"]])
            bid = bid or mid
            ask = ask or mid
        self._broker.update_tick(
            symbol=symbol, timestamp=ts,
            bid=bid, ask=ask,
            volume=float(tick[_TICK_COL["volume"]]),
            flags=int(tick[_TICK_COL["flags"]]),
            volume_real=float(tick[_TICK_COL["volume_real"]]),
            price=float(tick[_TICK_COL["price"]]),
        )

    # ── internal ────────────────────────────────────────────────────────

    def _get_state(self, symbol: str) -> _SymbolState:
        state = self._states.get(symbol)
        if state is None:
            raise TradingError(
                f"FakeLiveSesh: symbol '{symbol}' not registered. "
                f"Available: {list(self._states)}. "
                f"Add it with sesh.set_symbol('{symbol}', initial_price=...)."
            )
        return state

    def __repr__(self) -> str:
        a = self._broker.account_info()
        prices = {s: f"{self._states[s].price:.5g}" for s in self._states}
        return (
            f"FakeLiveSesh(symbols={list(self._states)}, prices={prices}, "
            f"balance={a.balance:.2f})"
        )

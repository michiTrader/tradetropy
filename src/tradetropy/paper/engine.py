"""
tradetropy.paper.engine
======================
PaperEngine -- manual, forward-only market simulation for practice and
discretionary backtesting.

Unlike ReplayEngine, it runs NO programmatic strategy: it replays a historical
dataset under the same pause/play/step/speed controls, but trading decisions
come entirely from the user through the UI Order Ticket. To enforce trading
discipline (and prevent look-ahead cheating) it is strictly forward-only:

    - Backward navigation is disabled (``_ALLOW_BACKWARD = False``): the « Step
      Backward control and the <<< backward-direction toggle are hidden, and
      ``_rebuild_step_back`` raises if ever reached.
    - The only way to undo is Restart: it wipes the manual order ledger, resets
      the account balance/equity to the baseline and moves the clock to index 0
      (handled by the inherited rewind via ReplaySesh.reset()).

The engine synthesizes its strategy from a declarative ``PaperConfig`` (which
data to show + which indicators), so there is no user-written strategy class.

Example
-------
    from tradetropy.paper import PaperConfig, PaperEngine

    config = PaperConfig(symbol="BTCUSDT", interval_ms=60_000, ticks=True)
    engine = PaperEngine.by_ticks(config, data=(ticks,))
    engine.run(live_chart=chart)     # play forward, trade from the Order Ticket
"""

from __future__ import annotations

from tradetropy.exceptions import ConfigError
from tradetropy.models.strategy import Strategy
from tradetropy.playback.base import BaseEngine
from tradetropy.replay.replay_sesh import ReplaySesh
from tradetropy.paper.config import PaperConfig, PaperIndicator


# =============================================================================
# SYNTHESIZED MANUAL STRATEGY
# =============================================================================

def _normalize_spec(spec):
    """
    Normalize an indicator spec into (indicator, source, overrides).

    Accepts a bare Indicator (mounted on 'close'), a PaperIndicator, or a
    (indicator, source[, overrides]) tuple.
    """
    if isinstance(spec, PaperIndicator):
        return spec.indicator, spec.source, dict(spec.overrides)
    if isinstance(spec, tuple):
        if len(spec) == 2:
            return spec[0], spec[1], {}
        if len(spec) == 3:
            return spec[0], spec[1], dict(spec[2])
        raise ConfigError(
            "Indicator tuple must be (indicator, source[, overrides])."
        )
    # Bare Indicator instance.
    return spec, "close", {}


def build_manual_strategy_class(config: PaperConfig):
    """
    Build a no-op Strategy subclass that subscribes/declares from the config.

    The returned class takes no constructor arguments (the config is embedded as
    a class attribute), so BaseEngine._calc_warmup can re-instantiate it with
    ``strategy.__class__()`` to recompute warmup. on_data() is intentionally
    empty: all trading happens through the UI Order Ticket.

    Args:
        config (PaperConfig): Declarative session configuration.

    Returns:
        type: A Strategy subclass ready to instantiate with no arguments.
    """

    class _ManualStrategy(Strategy):
        """Synthesized strategy for manual paper trading (no programmatic logic)."""

        _paper_config = config
        warmup = config.warmup

        def init(self):
            cfg = self._paper_config
            self.ticks = None
            self.fp = None
            self.book = None

            self.ohlc = self.subscribe_ohlc(
                cfg.symbol, cfg.interval_ms, window_size=cfg.ohlc_window,
            )
            if cfg.ticks:
                self.ticks = self.subscribe_ticks(
                    cfg.symbol, window_size=cfg.tick_window,
                )
            if cfg.footprint:
                fp_kwargs = cfg.footprint if isinstance(cfg.footprint, dict) else {}
                self.fp = self.subscribe_footprint(
                    cfg.symbol, cfg.interval_ms, **fp_kwargs,
                )
            if cfg.orderbook_depth:
                self.book = self.subscribe_orderbook(
                    cfg.symbol, depth=cfg.orderbook_depth,
                )

            for spec in cfg.indicators:
                indicator, source, overrides = _normalize_spec(spec)
                ref = self._resolve_source(source)
                self.add_indicator(ref, indicator, **overrides)

        def on_data(self):
            """No-op: manual paper trading routes all orders through the UI."""
            pass

        def _resolve_source(self, source):
            """Resolve an indicator source token to a concrete ref/proxy."""
            if callable(source):
                return source(self)
            if isinstance(source, str):
                token = source.lower()
                if token == "ohlc":
                    return self.ohlc
                if token == "ticks":
                    if self.ticks is None:
                        raise ConfigError(
                            "Indicator source 'ticks' requires ticks=True in "
                            "the PaperConfig."
                        )
                    return self.ticks
                if token == "price":
                    if self.ticks is None:
                        raise ConfigError(
                            "Indicator source 'price' requires ticks=True in "
                            "the PaperConfig."
                        )
                    return self.ticks.price_ref
                if token in ("close", "open", "high", "low", "volume"):
                    return getattr(self.ohlc, f"{token}_ref")
                raise ConfigError(
                    f"Unknown indicator source token {source!r}. Use one of "
                    "'close'/'open'/'high'/'low'/'volume'/'price'/'ohlc'/'ticks', "
                    "a ColumnRef, or a callable."
                )
            # ColumnRef / list[ColumnRef] / proxy passed straight through.
            return source

    return _ManualStrategy


# =============================================================================
# PAPER ENGINE
# =============================================================================

class PaperEngine(BaseEngine):
    """
    Manual, forward-only paper-trading engine driven by the UI Order Ticket.

    Specialization of BaseEngine that forbids backward navigation
    (``_ALLOW_BACKWARD = False``) so the user cannot peek ahead or rewind single
    trades; the only reset is Restart (wipes the ledger, restores the baseline
    balance and seeks to index 0). It synthesizes its strategy from a
    PaperConfig, so no programmatic strategy is supplied.
    """

    _DEFAULT_SAVE_LOG: bool = False
    _ALLOW_BACKWARD: bool = False
    _WANTS_ORDER_TICKET: bool = True

    @classmethod
    def _build_sesh(cls, config: PaperConfig, sesh):
        """Resolve the session: use the provided one or build a ReplaySesh."""
        if sesh is not None:
            return sesh
        return ReplaySesh(
            initial_balance=config.initial_balance,
            commission=config.commission,
            **config.sesh_kwargs,
        )

    @classmethod
    def by_ticks(
        cls,
        config: PaperConfig,
        data,
        sesh: "ReplaySesh | None" = None,
        history: "dict | None" = None,
        warmup_ticks: "int | None" = None,
        warmup_pct: float = 0.20,
        speed: float = 1.0,
        base_rate: float = 1.0,
        poll_interval: float = 0.0,
        book: "BookData | tuple | list | None" = None,
        chart_ohlc_interval_ms: int = 60_000,
    ) -> "PaperEngine":
        """
        Build a forward-only paper-trading engine over historical ticks.

        Args:
            config (PaperConfig): Declarative session configuration.
            data: Tuple of TickData (one per symbol).
            sesh (ReplaySesh | None): Optional pre-configured session; if omitted
                a ReplaySesh is built from config.initial_balance / commission.
            (others): Same as ReplayEngine.by_ticks.

        Returns:
            PaperEngine: Configured forward-only paper-trading engine.
        """
        strategy = build_manual_strategy_class(config)()
        sesh = cls._build_sesh(config, sesh)
        return super().by_ticks(
            strategy,
            data,
            sesh=sesh,
            history=history,
            warmup_ticks=warmup_ticks,
            warmup_pct=warmup_pct,
            speed=speed,
            base_rate=base_rate,
            poll_interval=poll_interval,
            book=book,
            chart_ohlc_interval_ms=chart_ohlc_interval_ms,
        )

    @classmethod
    def by_klines(
        cls,
        config: PaperConfig,
        data,
        sesh: "ReplaySesh | None" = None,
        history: "dict | None" = None,
        warmup_ticks: "int | None" = None,
        warmup_pct: float = 0.20,
        speed: float = 1.0,
        base_rate: float = 1.0,
        poll_interval: float = 0.0,
    ) -> "PaperEngine":
        """
        Build a forward-only paper-trading engine over historical klines.

        Args:
            config (PaperConfig): Declarative session configuration.
            data: Tuple of KlineData (one per symbol-interval pair).
            sesh (ReplaySesh | None): Optional pre-configured session.
            (others): Same as ReplayEngine.by_klines.

        Returns:
            PaperEngine: Configured forward-only paper-trading engine.
        """
        strategy = build_manual_strategy_class(config)()
        sesh = cls._build_sesh(config, sesh)
        return super().by_klines(
            strategy,
            data,
            sesh=sesh,
            history=history,
            warmup_ticks=warmup_ticks,
            warmup_pct=warmup_pct,
            speed=speed,
            base_rate=base_rate,
            poll_interval=poll_interval,
        )

    # =========================================================================
    # FORWARD-ONLY ENFORCEMENT
    # =========================================================================

    def _rebuild_step_back(self, n: int = 1) -> None:
        """
        Backward navigation is forbidden in manual paper-trading mode.

        The controller already gates this (allow_backward=False) so the UI never
        triggers it; this override is the hard backstop. Use Restart to reset.

        Raises:
            ConfigError: Always.
        """
        raise ConfigError(
            "PaperEngine is forward-only: stepping backward in time is not "
            "allowed. Click Restart to wipe the ledger and reset to the start."
        )

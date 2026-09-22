"""
tradetropy.replay.engine
=====================
ReplayEngine -- reproduces historical data simulating a live broker feed,
with pause/play/step/speed controls accessible from the Bokeh browser.

It is the automated playback engine: it executes a programmatic strategy over a
historical dataset so you can visually audit its behavior. Playback is
forward-only (Step Backward is disabled, like PaperEngine); use Restart to
rewind to the beginning and replay from scratch.

All the cursor/rewind/controller machinery lives in the shared
``tradetropy.playback.BaseEngine``; ReplayEngine only specializes it for the
automated case (a user-supplied Strategy, forward-only navigation).

PUBLIC API
==========
    from tradetropy.replay import ReplayEngine
    from tradetropy.plotting.live import LiveChart
    from tradetropy.plotting import PlotConfig
    from tradetropy.io import read_ticks

    ticks = read_ticks("records/ticks_MESM26.h5", "MESM26")

    engine = ReplayEngine.by_ticks(
        SmaCrossover(),
        data        = {"MESM26": ticks},
        speed       = 1.0,
        base_rate   = 1.0,
        warmup_pct  = 0.20,
    )

    chart = LiveChart(config=PlotConfig(theme="dark"), max_candles=200)
    engine.run(live_chart=chart)

    # Equivalent manual flow (fine-grained control):
    #   engine.attach_chart(chart)
    #   chart.start()
    #   engine.run()

Pause/play/step/speed buttons appear automatically in the browser when the
chart is started -- no need to instantiate ReplaySesh or PlaybackController
manually.

CONSTRUCTOR PARAMETERS
======================
strategy    : Strategy instance with init() declared.
data        : dict {symbol: tick_array} (by_ticks)
              or {symbol: kline_array} (by_klines).
              Also accepts {(symbol, interval_ms): klines} directly.
              Value can be a numpy ndarray or any object with a `.data`
              attribute returning an ndarray (TickData, KlineData...).
sesh        : ReplaySesh | None -- pre-configured session. If omitted, the
              engine creates a ReplaySesh internally using the broker
              parameters provided (commission, initial_balance, ...).
speed       : float -- ticks/sec multiplier. ticks/sec = base_rate * speed.
              float("inf") = turbo (no pause). Default: 1.0.
base_rate   : float -- ticks/sec base when speed=1.0 (default 1.0).
warmup_pct  : float in (0, 1) -- fraction of the dataset used as warmup
              (default 0.20).

ESCAPE HATCH
============
    engine._sesh   -> ReplaySesh
    engine._ctrl   -> PlaybackController
"""

from __future__ import annotations

from tradetropy.playback.base import BaseEngine, _to_array  # noqa: F401  (re-export)


class ReplayEngine(BaseEngine):
    """
    Automated, forward-only replay engine over historical data.

    Specialization of BaseEngine for auditing a programmatic strategy: it keeps
    backward navigation disabled (``_ALLOW_BACKWARD = False``) so the Step
    Backward (« ) control is hidden, matching PaperEngine. Use Restart to
    rewind to the beginning and replay from scratch. Replay does not persist the
    log by default (``_DEFAULT_SAVE_LOG = False``), same as backtest.

    Quick mode (data + broker kwargs -> automatic ReplaySesh):
        engine = ReplayEngine.by_ticks(strategy, data=data, commission=1.5)

    Full control mode (pre-configured ReplaySesh):
        sesh = ReplaySesh(feed_type='tick', commission=1.5)
        engine = ReplayEngine.by_ticks(strategy, data=data, sesh=sesh)
    """

    _DEFAULT_SAVE_LOG: bool = False
    _ALLOW_BACKWARD: bool = False

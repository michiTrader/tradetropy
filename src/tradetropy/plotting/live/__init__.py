"""
tradetropy.plotting.live
=====================
Real-time charting system built on Bokeh Server.

Usage
-----
    from tradetropy.plotting.live import LiveChart
    from tradetropy.plotting import PlotConfig

    chart = LiveChart(
        config      = PlotConfig(theme="dark", width=1400),
        max_candles = 300,
        port        = 5006,
    )

    engine.attach_chart(chart)
    chart.start()
    engine.run()
"""

from tradetropy.plotting.live.chart import LiveChart
from tradetropy.plotting.live.navigation import LiveNavigationController

__all__ = ["LiveChart", "LiveNavigationController"]

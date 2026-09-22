# Examples

## Interactive example

The chart below is produced by the strategy on the **Strategy** tab, run against
the bundled daily GOOG dataset - no data files and no API keys. The **Results**
tab shows the exact performance metrics from that run. Scroll to zoom, drag to
pan, and toggle series from the legend.

=== "Strategy"

    ```python
    --8<-- "assets/demos/sma_cross/snippet.py"
    ```

=== "Results"

    ```text
    --8<-- "assets/demos/sma_cross/stats.txt"
    ```

<iframe src="../assets/demos/sma_cross/chart.html"
        title="Tradetropy - SMA crossover interactive chart"
        loading="lazy"
        style="width: 100%; height: 640px; border: 0; margin: 1rem 0;">
</iframe>

## More runnable scripts

Runnable scripts live in the [`examples/`](https://github.com/michiTrader/tradetropy/tree/main/examples)
directory. They all use the [bundled datasets](guide/data.md), so they run with
no data files or API keys:

```bash
python examples/sma_cross.py
```

| Script                  | What it shows                                                    |
|-------------------------|------------------------------------------------------------------|
| `sma_cross.py`          | A basic SMA-crossover backtest on BTCUSD 1-minute candles.       |
| `multi_timeframe.py`    | Subscribing to 1m/5m/15m of one symbol; the engine resamples internally. |
| `multi_symbol.py`       | A multi-symbol backtest (BTCUSD + ADAUSD) side by side.          |
| `volume_profile.py`     | Trading the developing Volume Profile value area.                |
| `large_trades.py`       | Order flow: following large trades on MESU26 futures ticks.      |
| `orderflow_l2.py`       | L2 Deep Trades over a recorded order book via `ReplayEngine`.    |
| `live_pool_demo.py`     | Running several live strategies under `LivePool` (network-free). |

The order-flow L2 script opens an interactive replay chart in the browser; the
others print performance metrics to the console.

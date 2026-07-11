# Monte Carlo robustness

A single equity curve does not tell you whether a result is robust or just a
lucky ordering of trades. The `tradetropy.robustness` module runs a Monte Carlo
test: it generates many randomized variants of the result and reports the
**distribution** of the metrics - confidence intervals, probability of loss,
risk of ruin and a composite robustness score.

## From a finished backtest

```python
from tradetropy import BacktestEngine
from tradetropy.datasets import load_btcusd_1m

bt = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
bt.run()

mc = bt.montecarlo(
    n_sims     = 1000,
    methods    = ['shuffle_order', 'resample_trades'],
    confidence = (0.95, 0.99),
    seed       = 42,
)

print(mc.summary())              # per-metric table: original, mean, std, p5, p50, p95, CI
print(mc.probability_of_loss)    # P(final equity < initial balance)
print(mc.risk_of_ruin(0.5))      # P(drawdown exceeds 50% of equity)
print(mc.robustness_score)       # composite score in [0, 100]
mc.plot(theme='dark')            # equity cone + metric histograms
```

## Methods and levels

Methods operate at one of three levels and cannot be mixed across levels in a
single call:

- **Trade level** (fast, no re-run; perturb the closed-trade list):
  `shuffle_order`, `resample_trades`, `skip_trades`, `randomize_slippage`,
  `random_start_index`.
- **Data level** (re-runs the engine in parallel): `randomize_prices`,
  `random_start_bar`.
- **Parameter level** (re-runs the engine): `randomize_parameters`.

The explicit API gives full control and lets you compose methods of the same
level:

```python
from tradetropy.robustness import MonteCarlo, MonteCarloConfig
from tradetropy.robustness.methods import ShuffleOrder, ResampleTrades, SkipTrades

result = MonteCarlo(bt, MonteCarloConfig(
    n_sims     = 2000,
    methods    = [ShuffleOrder(), ResampleTrades(replace=True), SkipTrades(prob=0.1)],
    metrics    = ['Return [%]', 'Max. Drawdown [%]', 'Sharpe Ratio', 'Profit Factor'],
    confidence = (0.95, 0.99),
    seed       = 42,
)).run()

result.percentile('Max. Drawdown [%]', 5)
result.confidence_interval('Return [%]', 0.95)
result.to_dataframe()
```

The composite `robustness_score` (0-100) is a transparent weighted blend of
profitability, drawdown safety and consistency.

"""
Monte Carlo robustness test on a finished backtest.

A single good-looking equity curve is not proof that a strategy is robust: it is
only one realization out of many possible orderings of the same edge. This
example runs an SMA-crossover backtest on the bundled BTCUSDT 1-minute candles
and then stress-tests the result with Monte Carlo, estimating the distribution
of performance metrics, confidence intervals, the probability of loss, the risk
of ruin and a composite robustness score.

Two robustness levels are shown:

    - Trade level (fast, no re-run): perturb the closed-trade list in-process.
      Here we shuffle the trade order and resample the trades with replacement.
    - Data level (re-runs the engine): jitter the input prices and re-run the
      whole backtest for every simulation, so indicators and fills are
      recomputed on perturbed data.

Run it directly:

    python examples/montecarlo_robustness.py
"""

from tradetropy import BacktestEngine, Strategy
from tradetropy.datasets import load_btcusd_1m
from tradetropy.ta import SMA


class SmaCross(Strategy):
    """Go long when the fast SMA is above the slow SMA."""

    def init(self):
        self.btc = self.subscribe_ohlc('BTCUSDT', '1m', window_size=200)
        self.fast = self.add_indicator(self.btc.close, SMA(4))
        self.slow = self.add_indicator(self.btc.close, SMA(9))

    def on_data(self):
        if self.fast[-1] > self.slow[-1]:
            if not self.sesh.positions('BTCUSDT'):
                self.sesh.buy('BTCUSDT', volume=1)
        else:
            for pos in self.sesh.positions('BTCUSDT'):
                self.sesh.position_close(pos.ticket)


def report(title, result):
    """
    Print the robustness analytics of a Monte Carlo result.

    Args:
        title (str): Section heading.
        result (MonteCarloResult): The finished robustness run.
    """
    print(f'\n{title}')
    print('=' * len(title))
    print(result.summary().round(2))
    print(f'\nProbability of loss : {result.probability_of_loss:.1%}')
    print(f'Risk of ruin (>50%) : {result.risk_of_ruin(0.5):.1%}')
    print(f'Return 95% CI       : {result.confidence_interval("Return [%]", 0.95)}')
    print(f'Robustness score    : {result.robustness_score:.1f} / 100')


if __name__ == '__main__':
    # Baseline backtest - the single realization we are about to stress-test.
    bt = BacktestEngine.by_klines(SmaCross(), data=(load_btcusd_1m(),))
    bt.run()
    print('Baseline stats')
    print('==============')
    print(bt.stats)

    # Trade-level robustness: reorder and resample the closed trades. Fast,
    # because it post-processes the finished trade list without re-running.
    trade_level = bt.montecarlo(
        n_sims=2000,
        methods=['shuffle_order', 'resample_trades'],
        seed=42,
    )
    report('Trade-level Monte Carlo (2000 sims)', trade_level)

    # Data-level robustness: jitter the input prices and re-run the engine for
    # every simulation, so indicators and fills react to the perturbed data.
    # This path is heavier (it re-runs the backtest per simulation).
    data_level = bt.montecarlo(
        n_sims=200,
        methods=['randomize_prices'],
        seed=42,
    )
    report('Data-level Monte Carlo (200 sims, price jitter)', data_level)

    # Equity cone + metric histograms (trade-level retains the equity curves).
    trade_level.plot(output='file', filename='montecarlo.html')
    print('\nSaved equity cone and metric histograms to montecarlo.html')

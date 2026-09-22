"""
Monte Carlo result container.

``MonteCarloResult`` is the immutable object returned by ``MonteCarlo.run()``
and by ``BacktestEngine.montecarlo()``. It stores the per-simulation metrics and
exposes the robustness analytics: summary table, percentiles, confidence
intervals, probability of loss, risk of ruin and a composite robustness score.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tradetropy.stats import Stats

# Composite robustness score weights (must sum to 1.0). See robustness_score.
_W_PROFITABILITY = 0.4
_W_DD_SAFETY = 0.3
_W_CONSISTENCY = 0.3


class MonteCarloResult:
    """
    Immutable container for Monte Carlo robustness results.

    Attributes:
        original_stats (Stats): Baseline (unperturbed) statistics.

    Example:
        result = bt.montecarlo(n_sims=1000, methods=['shuffle_order'], seed=42)
        print(result.summary())
        result.percentile('Max. Drawdown [%]', 5)
        result.confidence_interval('Return [%]', 0.95)
        result.probability_of_loss
        result.risk_of_ruin(0.5)
        result.robustness_score
    """

    def __init__(
        self,
        sims: pd.DataFrame,
        original: 'Stats',
        *,
        final_equity: np.ndarray,
        max_drawdown: np.ndarray,
        initial_balance: float,
        confidence: Sequence[float] = (0.95, 0.99),
        equity_paths: Optional[List[pd.Series]] = None,
        method_names: Optional[List[str]] = None,
    ):
        """
        Initialize a Monte Carlo result.

        Args:
            sims (pd.DataFrame): Per-simulation tracked metrics (one row each).
            original (Stats): Baseline statistics.
            final_equity (np.ndarray): Final equity per simulation.
            max_drawdown (np.ndarray): Max. Drawdown [%] per simulation (<= 0).
            initial_balance (float): Starting capital.
            confidence (Sequence[float]): Confidence levels for intervals.
            equity_paths (list[pd.Series] | None): Equity curves for plotting.
            method_names (list[str] | None): Names of the methods applied.
        """
        self._sims = sims.reset_index(drop=True)
        self._original = original
        self._final_equity = np.asarray(final_equity, dtype=float)
        self._max_dd = np.asarray(max_drawdown, dtype=float)
        self._initial_balance = float(initial_balance)
        self._confidence = tuple(confidence)
        self._equity_paths = equity_paths or []
        self._method_names = method_names or []

    # ---- Primary accessors --------------------------------------------------

    @property
    def original_stats(self) -> 'Stats':
        """Baseline (unperturbed) Stats object."""
        return self._original

    @property
    def n_sims(self) -> int:
        """Number of simulations run."""
        return len(self._sims)

    @property
    def metrics(self) -> List[str]:
        """Tracked metric names."""
        return list(self._sims.columns)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the raw per-simulation metrics.

        Returns:
            pd.DataFrame: One row per simulation, one column per tracked metric.
        """
        return self._sims.copy()

    # ---- Distribution queries -----------------------------------------------

    def _column(self, metric: str) -> np.ndarray:
        """
        Return a finite-valued metric column as an array.

        Args:
            metric (str): Tracked metric name.

        Returns:
            np.ndarray: Finite values of the metric across simulations.

        Raises:
            KeyError: If the metric was not tracked.
        """
        if metric not in self._sims.columns:
            raise KeyError(
                f"Metric '{metric}' was not tracked. "
                f"Tracked metrics: {self.metrics}."
            )
        col = self._sims[metric].to_numpy(dtype=float)
        return col[np.isfinite(col)]

    def percentile(self, metric: str, q: float) -> float:
        """
        Compute a percentile of a tracked metric.

        Args:
            metric (str): Tracked metric name.
            q (float): Percentile in [0, 100].

        Returns:
            float: The requested percentile (NaN if no finite values).
        """
        col = self._column(metric)
        return float(np.percentile(col, q)) if col.size else float('nan')

    def confidence_interval(
        self, metric: str, level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Two-sided confidence interval of a tracked metric.

        Args:
            metric (str): Tracked metric name.
            level (float | None): Confidence level in (0, 1). Defaults to the
                first configured level.

        Returns:
            tuple[float, float]: (lower, upper) percentile bounds.
        """
        if level is None:
            level = self._confidence[0]
        alpha = 1.0 - level
        lo = self.percentile(metric, 100.0 * alpha / 2.0)
        hi = self.percentile(metric, 100.0 * (1.0 - alpha / 2.0))
        return lo, hi

    @property
    def probability_of_loss(self) -> float:
        """
        Probability that final equity ends below the initial balance.

        Returns:
            float: Fraction of simulations with a net loss, in [0, 1].
        """
        if self._final_equity.size == 0:
            return float('nan')
        return float(np.mean(self._final_equity < self._initial_balance))

    def risk_of_ruin(self, threshold: float = 0.5) -> float:
        """
        Probability that the max drawdown exceeds a capital threshold.

        Args:
            threshold (float): Drawdown magnitude as a fraction of equity in
                (0, 1]. For example 0.5 means a 50% peak-to-trough drop.

        Returns:
            float: Fraction of simulations breaching the threshold, in [0, 1].

        Raises:
            ValueError: If threshold is outside (0, 1].
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError('risk_of_ruin threshold must be in (0, 1].')
        if self._max_dd.size == 0:
            return float('nan')
        dd_frac = np.abs(self._max_dd) / 100.0
        return float(np.mean(dd_frac >= threshold))

    # ---- Composite score ----------------------------------------------------

    @property
    def robustness_score(self) -> float:
        """
        Composite robustness score in [0, 100].

        Transparent weighted blend of three sub-scores, each in [0, 1]:

            - profitability (weight 0.4): fraction of simulations that end
              profitable (1 - probability_of_loss).
            - drawdown safety (weight 0.3): 1 - |p95 drawdown| / 100, clipped
              to [0, 1]; rewards small worst-case drawdowns.
            - consistency (weight 0.3): ratio of the 5th-percentile return to
              the median return, clipped to [0, 1]; near 1 when the downside
              tail stays close to the median, 0 when the median is non-positive.

        A high score means most simulations are profitable, drawdowns stay
        contained and the unlucky tail is not far from the typical outcome.

        Returns:
            float: Score in [0, 100] (NaN if there are no simulations).
        """
        if self.n_sims == 0:
            return float('nan')

        profitability = 1.0 - self.probability_of_loss

        dd_p95 = np.percentile(np.abs(self._max_dd), 95) if self._max_dd.size else 0.0
        dd_safety = float(np.clip(1.0 - dd_p95 / 100.0, 0.0, 1.0))

        ret = self._return_pct_array()
        if ret.size:
            median_ret = float(np.median(ret))
            tail_ret = float(np.percentile(ret, 5))
            consistency = (
                float(np.clip(tail_ret / median_ret, 0.0, 1.0))
                if median_ret > 0.0 else 0.0
            )
        else:
            consistency = 0.0

        score = (
            _W_PROFITABILITY * profitability
            + _W_DD_SAFETY * dd_safety
            + _W_CONSISTENCY * consistency
        )
        return float(np.clip(score, 0.0, 1.0) * 100.0)

    def _return_pct_array(self) -> np.ndarray:
        """
        Return per-simulation return percentage, finite values only.

        Uses the tracked 'Return [%]' column when present, otherwise derives it
        from final equity and the initial balance.

        Returns:
            np.ndarray: Finite return percentages across simulations.
        """
        if 'Return [%]' in self._sims.columns:
            return self._column('Return [%]')
        if self._initial_balance > 0 and self._final_equity.size:
            ret = (self._final_equity / self._initial_balance - 1.0) * 100.0
            return ret[np.isfinite(ret)]
        return np.array([], dtype=float)

    # ---- Summary ------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Per-metric summary table across all simulations.

        Returns:
            pd.DataFrame: Indexed by metric, with columns: original, mean, std,
            min, p5, p50, p95, max, ci_low, ci_high (CI at the first
            configured confidence level).
        """
        level = self._confidence[0]
        rows = {}
        for metric in self._sims.columns:
            col = self._column(metric)
            original = self._original.get(metric, np.nan) \
                if self._original is not None else np.nan
            lo, hi = self.confidence_interval(metric, level)
            rows[metric] = {
                'original': float(original) if original is not None else np.nan,
                'mean': float(np.mean(col)) if col.size else np.nan,
                'std': float(np.std(col, ddof=1)) if col.size > 1 else np.nan,
                'min': float(np.min(col)) if col.size else np.nan,
                'p5': self.percentile(metric, 5),
                'p50': self.percentile(metric, 50),
                'p95': self.percentile(metric, 95),
                'max': float(np.max(col)) if col.size else np.nan,
                'ci_low': lo,
                'ci_high': hi,
            }
        return pd.DataFrame.from_dict(rows, orient='index')

    # ---- Visualization ------------------------------------------------------

    def plot(self, theme='light', width=1100, height=320,
             output='show', filename='montecarlo.html'):
        """
        Plot the equity cone and metric histograms with Bokeh.

        Args:
            theme (str): 'light' or 'dark'.
            width (int): Figure width in pixels.
            height (int): Per-panel height in pixels.
            output (str): 'show', 'file' or 'notebook'.
            filename (str): Output HTML path when output == 'file'.

        Returns:
            The Bokeh layout object.

        Note:
            The equity cone is only drawn for the trade-level path, which
            retains the per-simulation equity curves.
        """
        from tradetropy.robustness.plot import plot_result
        return plot_result(self, theme=theme, width=width, height=height,
                           output=output, filename=filename)

    # ---- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f'MonteCarloResult('
            f'n_sims={self.n_sims}, '
            f'methods={self._method_names}, '
            f'P(loss)={self.probability_of_loss:.3f}, '
            f'score={self.robustness_score:.1f})'
        )

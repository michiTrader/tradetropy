"""
Result storage and retrieval utilities.
"""

from typing import List
from tradetropy.optimize.task import Result


class ResultReporter:
    """
    Utilities for saving and loading optimization results.

    Provides methods to convert results to DataFrames and persist to CSV files.
    """

    @staticmethod
    def to_dataframe(results: List[Result]):
        """
        Convert results list to pandas DataFrame.

        Args:
            results: List of Result objects

        Returns:
            DataFrame with fitness, metrics, and parameter columns
        """
        import pandas as pd

        rows = []
        for r in results:
            row = {'fitness': r.fitness, 'error': r.error}
            if r.metrics:
                row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def save_to_csv(results: List[Result], filepath: str) -> None:
        """
        Save results to CSV file.

        Creates parent directory if it does not exist.

        Args:
            results: List of Result objects
            filepath: Path where CSV will be saved
        """
        import pandas as pd
        from tradetropy.io.io import _ensure_parent_dir

        filepath = _ensure_parent_dir(filepath)
        df = ResultReporter.to_dataframe(results)
        df.to_csv(filepath, index=False)

    @staticmethod
    def load_from_csv(filepath: str):
        """
        Load results from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            pandas DataFrame with optimization results
        """
        import pandas as pd

        return pd.read_csv(filepath)
import numpy as np


class OhlcDataStore:
    """
    Data store for OHLC candles during backtesting.

    Matrix layout:
        - Columns 0-5: ts, open, high, low, close, volume (closed candles)
        - Columns 6+: Pre-calculated indicators

    Provides efficient access to candle data and OHLC state for each tick.
    """

    __slots__ = (
        "matrix",
        "col_index",
        "tick_to_candle_mapping",
        "accumulated_by_tick",
        "ts_candle_by_tick",
        "prices_per_tick",
        "interval_ms",
        "n_closed_candles",
        "symbol",
        "kline_mode",
    )

    def __init__(
        self,
        matrix: np.ndarray,
        col_index: dict,
        tick_to_candle_mapping: np.ndarray,
        accumulated_by_tick: np.ndarray,
        ts_candle_by_tick: np.ndarray,
        prices_per_tick: np.ndarray,
        interval_ms: int,
        symbol: str = '',
        kline_mode: bool = False,
    ):
        """
        Initialize OHLC data store.

        Args:
            matrix (ndarray): Closed candle matrix
            col_index (dict): Column name to index mapping
            tick_to_candle_mapping (ndarray): Candle index for each tick
            accumulated_by_tick (ndarray): Accumulated OHLC per tick
            ts_candle_by_tick (ndarray): Candle timestamp per tick
            prices_per_tick (ndarray): Tick prices
            interval_ms (int): Candle interval in milliseconds
            symbol (str): Trading symbol
            kline_mode (bool): If True, includes partial candle in window
        """
        self.matrix = matrix
        self.col_index = col_index
        self.tick_to_candle_mapping = tick_to_candle_mapping
        self.accumulated_by_tick = accumulated_by_tick
        self.ts_candle_by_tick = ts_candle_by_tick
        self.prices_per_tick = prices_per_tick
        self.interval_ms = int(interval_ms)
        self.n_closed_candles = len(matrix)
        self.symbol = symbol
        self.kline_mode = kline_mode

    def partial_tick_candle(self, tick_idx: int) -> np.ndarray:
        """
        Compute partial candle state at a given tick.

        Args:
            tick_idx (int): Tick index

        Returns:
            ndarray: [6] array with ts, open, high, low, close, volume
        """
        result = np.empty(6, dtype=np.float64)
        acum = self.accumulated_by_tick[tick_idx]
        result[0] = self.ts_candle_by_tick[tick_idx]
        result[1] = acum[0]  # open
        result[2] = acum[1]  # high
        result[3] = acum[2]  # low
        result[4] = self.prices_per_tick[tick_idx]  # close
        result[5] = acum[3]  # volume
        return result


class TickDataStore:
    """
    Data store for raw tick data during backtesting.

    Matrix layout [N_ticks x (7 + n_indicator_cols)]:
        - Columns 0-6: Raw tick data (ts, bid, ask, volume, flags,
          volume_real, price)
        - Columns 7+: Pre-calculated indicator values on ticks
    """

    __slots__ = ("matrix", "n_ticks", "col_index")

    def __init__(self, matrix: np.ndarray, col_index: dict):
        """
        Initialize tick data store.

        Args:
            matrix (ndarray): Tick data matrix
            col_index (dict): Column name to index mapping
        """
        self.matrix = matrix
        self.n_ticks = len(matrix)
        self.col_index = col_index

    def col(self, name: str) -> np.ndarray:
        """
        Get column by name.

        Args:
            name (str): Column name

        Returns:
            ndarray: Column data
        """
        return self.matrix[:, self.col_index[name]]

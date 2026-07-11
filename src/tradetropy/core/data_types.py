"""
data_types.py v2.0
==================
Typed data structures for the backtesting system.

Principle: 'make illegal states unrepresentable'.
The object type determines how the engine processes data -
no mode parameter or shape detection.

  TickData   - array [N x 7] of real ticks (ts/bid/ask/volume/flags/volume_real/price)
  KlineData  - array [N x 7] of OHLC candles + turnover
               (ts/open/high/low/close/volume/turnover)
               Requires mandatory `timeframe`: defines each candle duration.

KEY DIFFERENCE vs v1.0
======================
- KlineData.timeframe is MANDATORY (previously optional). Without it, the engine
  cannot group candles into different intervals. It accepts a string ('1m',
  '5m', '1h', ...) or an int in ms; the normalized ms value is read-only on
  `KlineData.interval_ms`.
- TickData and KlineData have the same number of columns (7), so distinction
  by shape is impossible - the object type is the only source of truth.
- _normalizar_inputs removed - engines use TickData/KlineData directly.

USAGE
-----
    from tradetropy.data_types import TickData, KlineData

    ticks  = TickData('BTCUSDT', tick_matrix, tick_size=0.01)

    klines_btc = KlineData('BTCUSDT', btc_matrix, timeframe='5m', tick_size=0.25)
    klines_eth = KlineData('ETHUSDT', eth_matrix, timeframe='1m', tick_size=0.01)

    bt = BacktestEngine.by_klines(
        strategy = MyStrategy(),
        data     = (klines_btc, klines_eth),
    )
    bt.run()

INVARIANTS
==========
- All inputs in a session must be the same type (TickData or KlineData).
  Mixing them raises ValueError in the engine.
- data.data is never validated here - the engine validates shape on connect.
- KlineData.interval_ms > 0 always. Engine raises ValueError if 0 or None.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Union

import numpy as np

try:
    from tradetropy.core.broker import SymbolConfig
except ImportError:
    from tradetropy.core.broker import SymbolConfig

from tradetropy.exceptions import ConfigError, DataError


def infer_price_digits(tick_size, default: int = 2) -> int:
    """
    Deduce the number of price decimal places implied by a tick size.

    Counts the significant decimals of ``tick_size`` (e.g. 0.0001 -> 4,
    0.25 -> 2, 0.5 -> 1, 1 -> 0). Used to auto-fill ``digits`` when a data
    object is built without it, so small-priced symbols (e.g. 0.1844) are
    normalized and displayed with enough precision.

    Args:
        tick_size (float): Minimum price step.
        default (int): Fallback when ``tick_size`` is missing / non-positive.

    Returns:
        int: Decimal places implied by ``tick_size`` (>= 0), or ``default``.

    Example:
        infer_price_digits(0.0001)  # -> 4
    """
    try:
        ts = float(tick_size)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(ts) or ts <= 0:
        return default
    s = f"{ts:.12f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def _format_rows(
    data: np.ndarray,
    columns: tuple[str, ...],
    n: int,
    offset: int = 0,
) -> str:
    """
    Format data as an aligned text table with alternating row colors.

    Shows a head/tail preview (like DataFrame.head()/tail()): the first and
    last n rows, with an ellipsis line in between when the data has more than
    2*n rows. When it has 2*n rows or fewer, every row is shown. This mirrors
    _format_rows_html so the text and HTML reprs are consistent.

    Args:
        data (ndarray): The [N x C] data matrix.
        columns (tuple[str]): Column names, must contain 'ts'.
        n (int): Number of rows to show from the head and from the tail.
        offset (int): Unused, kept for backward compatibility.

    Returns:
        str: The formatted table (or "(empty)" when there are no rows).
    """
    import pandas as pd

    if len(data) == 0:
        return "(empty)"

    ts_col = list(columns).index("ts")
    cols = [c for c in columns if c != "ts"]
    col_idx = [list(columns).index(c) for c in cols]

    total = len(data)
    if total <= 2 * n:
        row_ids = list(range(total))
        ellipsis_after = -1
    else:
        row_ids = list(range(n)) + list(range(total - n, total))
        ellipsis_after = n - 1

    _GRAY = "\033[48;5;237m"
    _RESET = "\033[0m"

    header = f"{'datetime':>20}  " + "  ".join(f"{c:>12}" for c in cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for display_i, i in enumerate(row_ids):
        ts_str = pd.to_datetime(
            int(data[i, ts_col]), unit="ms", utc=True
        ).strftime("%Y-%m-%d %H:%M:%S")
        vals = "  ".join(f"{data[i, j]:>12.4f}" for j in col_idx)
        line = f"{ts_str:>20}  {vals}"
        if display_i % 2 == 1:
            line = f"{_GRAY}{line}{_RESET}"
        lines.append(line)
        if i == ellipsis_after:
            lines.append(f"... ({total - 2 * n} more rows) ...")
    return "\n".join(lines)


def _format_rows_html(
    data: np.ndarray,
    columns: tuple[str, ...],
    title: str,
    n: int = 5,
) -> str:
    """
    Render the first and last n rows of data as an HTML table for Jupyter.

    Shows a head/tail preview (like DataFrame.head()/tail()) with an ellipsis
    row in between when the data has more than 2*n rows. The 'ts' column is
    rendered as a UTC datetime string; numeric columns are shown with 4
    decimals. Styling is inline so it renders identically in light and dark
    notebook themes.

    Args:
        data (ndarray): The [N x C] data matrix.
        columns (tuple[str]): Column names, must contain 'ts'.
        title (str): Header text shown above the table (the repr line).
        n (int): Number of rows to show from the head and from the tail.

    Returns:
        str: An HTML fragment (table) suitable for _repr_html_().
    """
    import pandas as pd

    header_html = (
        f'<div style="font-family:monospace;font-size:12px;margin-bottom:4px">'
        f'<b>{title}</b></div>'
    )
    if len(data) == 0:
        return header_html + '<div style="font-family:monospace">(empty)</div>'

    ts_col = list(columns).index("ts")
    cols = [c for c in columns if c != "ts"]
    col_idx = [list(columns).index(c) for c in cols]

    total = len(data)
    if total <= 2 * n:
        head_rows = list(range(total))
        tail_rows: list[int] = []
    else:
        head_rows = list(range(n))
        tail_rows = list(range(total - n, total))

    def _fmt_ts(ts_ms: float) -> str:
        return pd.to_datetime(int(ts_ms), unit="ms", utc=True).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    th_style = (
        "text-align:right;padding:2px 8px;border-bottom:1px solid #888;"
        "font-family:monospace"
    )
    td_style = "text-align:right;padding:2px 8px;font-family:monospace"

    head_cells = (
        f'<th style="{th_style}">#</th>'
        f'<th style="{th_style}">datetime</th>'
        + "".join(f'<th style="{th_style}">{c}</th>' for c in cols)
    )
    body_lines = [f"<tr>{head_cells}</tr>"]

    def _row_html(i: int) -> str:
        cells = (
            f'<td style="{td_style}">{i}</td>'
            f'<td style="{td_style}">{_fmt_ts(data[i, ts_col])}</td>'
            + "".join(f'<td style="{td_style}">{data[i, j]:.4f}</td>' for j in col_idx)
        )
        return f"<tr>{cells}</tr>"

    for i in head_rows:
        body_lines.append(_row_html(i))
    if tail_rows:
        span = len(cols) + 2
        body_lines.append(
            f'<tr><td colspan="{span}" style="{td_style};text-align:center">'
            f"... ({total - 2 * n} more rows) ...</td></tr>"
        )
        for i in tail_rows:
            body_lines.append(_row_html(i))

    table = (
        '<table style="border-collapse:collapse">'
        + "".join(body_lines)
        + "</table>"
    )
    return header_html + table


def _parse_date_to_ms(value) -> int:
    """Convert date string or numeric timestamp to milliseconds since epoch."""
    if isinstance(value, (int, float)):
        v = int(value)
        return v * 1000 if v < 1e11 else v
    import pandas as pd
    return int(pd.Timestamp(value).timestamp() * 1000)


def _apply_filter(
    data: np.ndarray,
    mask: "np.ndarray | Callable | None" = None,
    start=None,
    end=None,
    idx_start=None,
    idx_end=None,
) -> np.ndarray:
    """
    Apply filter combining time range, positional index range, boolean mask
    and callable.

    Args:
        data (ndarray): The [N x C] data matrix.
        mask: Boolean array [N] or callable ``d -> bool array``.
        start: Time-range start (date string or ms timestamp), inclusive.
        end: Time-range end (date string or ms timestamp), inclusive.
        idx_start (int | None): Positional index range start, inclusive.
        idx_end (int | None): Positional index range end, exclusive
            (half-open, matching Python slicing).

    Returns:
        ndarray: The filtered rows (all conditions combined with AND).
    """
    if data.size == 0:
        return data

    result_mask = np.ones(len(data), dtype=bool)

    if idx_start is not None or idx_end is not None:
        idx_mask = np.zeros(len(data), dtype=bool)
        idx_mask[idx_start:idx_end] = True
        result_mask &= idx_mask

    if start is not None or end is not None:
        ts = data[:, 0]
        if start is not None:
            result_mask &= ts >= _parse_date_to_ms(start)
        if end is not None:
            result_mask &= ts <= _parse_date_to_ms(end)

    if mask is not None:
        if callable(mask):
            mask = mask(data)
        result_mask &= mask

    return data[result_mask]


def _getitem_data(original, key):
    """
    Index a TickData/KlineData by position, returning the same type.

    Args:
        original: The TickData/KlineData instance.
        key (int | slice): Integer position (returns a 1-row object) or a
            slice (returns the ranged object). Negative indices supported.

    Returns:
        A new object of the same type with the selected rows and the same
        metadata/config propagated.

    Raises:
        IndexError: If an integer key is out of range.
        TypeError: If key is neither an int nor a slice.
    """
    if isinstance(key, slice):
        return _slice_data(original, original.data[key])
    if isinstance(key, (int, np.integer)):
        n = len(original.data)
        i = int(key)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(
                f'index {int(key)} out of range for {type(original).__name__} '
                f'with {n} rows'
            )
        return _slice_data(original, original.data[i:i + 1])
    raise TypeError(
        f'{type(original).__name__} indices must be int or slice, '
        f'not {type(key).__name__}'
    )


def _slice_data(original, new_data: np.ndarray):
    """Create a copy of a TickData/KlineData with new data, propagating all config."""
    return replace(original, data=new_data)


# ==============================================================================
# UNION TYPE - used in type annotations for engines
# ==============================================================================

#: Data type accepted by BacktestEngine.by_ticks() / by_klines().
#: A homogeneous tuple - never mix TickData with KlineData.
SymbolInput = Union['TickData', 'KlineData']


# ==============================================================================
# TICK DATA
# ==============================================================================


@dataclass
class TickData:
    """
    Tick data for a symbol.

    Attributes:
        symbol (str): Symbol name (e.g. 'BTCUSDT', 'MES')
        data (ndarray): [N x 7] array with columns: ts, bid, ask, volume, flags,
                        volume_real, price
        tick_size (float): Minimum price step (e.g. 0.25 for MES, 0.01 for forex)
        tick_value (float): Monetary value per tick per contract (e.g. 1.25 for
                            MES). Defaults to tick_size (so PnL per unit equals
                            the price difference, like backtesting.py).
        contract_size (float): Contract size (1 for most)
        digits (int): Decimal places for price normalization
        avg_spread (float): Average spread in ticks

    Example:
        ticks = TickData(
            symbol='BTCUSDT',
            data=tick_matrix,
            tick_size=0.01,
        )
        bt = BacktestEngine.by_ticks(MyStrategy(), data=(ticks,))
        bt.run()
    """

    symbol: str
    data: np.ndarray
    tick_size: float = 0.01
    tick_value: "float | None" = None
    contract_size: float = 1.0
    digits: int | None = None
    avg_spread: float = 0.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01

    @property
    def config(self) -> SymbolConfig:
        """
        Build SymbolConfig for passing to broker.add_symbol().

        Returns:
            SymbolConfig: Configuration object with this symbol's parameters
        """
        return SymbolConfig(
            name=self.symbol,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
            digits=self.digits,
            avg_spread=self.avg_spread,
            volume_min=self.volume_min,
            volume_max=self.volume_max,
            volume_step=self.volume_step,
        )

    def __post_init__(self):
        self.symbol = str(self.symbol)
        self.data = np.asarray(self.data, dtype=np.float64)
        # tick_value defaults to tick_size so PnL equals the pure price
        # difference per unit (contract_size=1), matching backtesting.py's
        # unit model out of the box. Set an explicit tick_value only for
        # instruments where a tick is worth a different amount than its size
        # (e.g. futures like MES: tick_size 0.25, tick_value 1.25).
        if self.tick_value is None:
            self.tick_value = self.tick_size
        if self.digits is None:
            self.digits = infer_price_digits(self.tick_size)

    def to_klines(
        self,
        interval_ms,
        *,
        include_partial: bool = False,
        price_source: str = 'price',
        volume_source: str = 'volume',
    ) -> 'KlineData':
        """
        Convert ticks to KlineData without mutating self.

        Args:
            interval_ms (int or str): Target duration in milliseconds or string
                                      format. Accepts ms (int) or str ('5m', '1h')
            include_partial (bool): Include the current incomplete candle
                                    (default False)
            price_source (str): Price column - 'price' (default) or 'mid' for
                                (bid+ask)/2
            volume_source (str): Volume column - 'volume' (default) or
                                 'volume_real'

        Returns:
            KlineData: New KlineData with candles, symbol config propagated

        Example:
            klines = ticks.to_klines('5m')
        """
        # Import local: avoids core -> data cycle
        from tradetropy.core.constants import parse_timeframe
        from tradetropy.data._klines import ticks_to_klines

        resolved_ms = parse_timeframe(interval_ms)
        matrix = ticks_to_klines(
            self.data,
            resolved_ms,
            include_partial=include_partial,
            price_source=price_source,
            volume_source=volume_source,
        )
        return KlineData(
            symbol=self.symbol,
            data=matrix,
            timeframe=resolved_ms,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
            digits=self.digits,
            avg_spread=self.avg_spread,
            volume_min=self.volume_min,
            volume_max=self.volume_max,
            volume_step=self.volume_step,
        )

    def save(self, path, format=None, **kwargs):
        """
        Save ticks to file.

        Args:
            path: Destination file path.
            format: Output format ('csv', 'parquet', 'hdf5', 'npz'). None
                (default) infers from the extension, falling back to 'npz'.
            **kwargs: Extra arguments passed to save_ticks() (ts_format,
                hdf5_key, compression).

        Returns:
            pd.DataFrame: The saved tick data.

        Example:
            ticks.save('btc_ticks.h5')
            ticks.save('btc_ticks.parquet', format='parquet')
        """
        from tradetropy.io.io import save_ticks
        metadata = {
            "tradetropy_symbol": self.symbol,
            "tradetropy_tick_size": self.tick_size,
            "tradetropy_tick_value": self.tick_value,
            "tradetropy_contract_size": self.contract_size,
            "tradetropy_digits": self.digits,
            "tradetropy_avg_spread": self.avg_spread,
            "tradetropy_volume_min": self.volume_min,
            "tradetropy_volume_max": self.volume_max,
            "tradetropy_volume_step": self.volume_step,
        }
        return save_ticks(self.data, path, format=format, metadata=metadata, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        header = (
            f"TickData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"tick_size={self.tick_size}, tick_value={self.tick_value})"
        )
        from tradetropy.core.constants import TICK_COLS
        table = _format_rows(self.data, TICK_COLS, 5)
        return f"{header}\n{table}"

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter/IPython (head + tail preview)."""
        from tradetropy.core.constants import TICK_COLS
        title = (
            f"TickData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"tick_size={self.tick_size}, tick_value={self.tick_value})"
        )
        return _format_rows_html(self.data, TICK_COLS, title)

    def head(self, n: int = 5) -> 'TickData':
        """Return a new TickData with the first n rows."""
        return _slice_data(self, self.data[:n])

    def tail(self, n: int = 5) -> 'TickData':
        """Return a new TickData with the last n rows."""
        start = max(0, len(self.data) - n)
        return _slice_data(self, self.data[start:])

    def __getitem__(self, key) -> 'TickData':
        """
        Index by position, returning a new TickData of the same type.

        Args:
            key (int | slice): Integer position (a 1-row TickData) or a slice
                (the ranged TickData). Negative indices supported.

        Returns:
            TickData: New TickData with the selected rows and same metadata.

        Example:
            ticks[100]        # single row (as a 1-row TickData)
            ticks[100:200]    # intermediate range
            ticks[-50:]       # last 50 rows (like tail(50))
        """
        return _getitem_data(self, key)

    def filter(
        self,
        mask: "np.ndarray | Callable | None" = None,
        *,
        start: "str | int | None" = None,
        end: "str | int | None" = None,
        idx_start: "int | None" = None,
        idx_end: "int | None" = None,
    ) -> 'TickData':
        """
        Filter ticks without mutating the original.

        Args:
            mask: Boolean array [N] or callable ``lambda d: d[:, col] > val``.
            start: Start date ('2024-01-01', '2024-01-01 10:30') or ms timestamp.
            end: End date (same format), inclusive.
            idx_start (int | None): Positional index range start, inclusive.
            idx_end (int | None): Positional index range end, exclusive.

        Returns:
            TickData: New TickData with the same metadata.

        Example:
            ticks.filter(start='2024-01-01', end='2024-01-31')
            ticks.filter(start='2024-01-01 10:00:00')
            ticks.filter(idx_start=1000, idx_end=2000)
            ticks.filter(mask=ticks.data[:, 3] > 0)
            ticks.filter(lambda d: d[:, 1] > 50000)
        """
        return _slice_data(
            self, _apply_filter(self.data, mask, start, end, idx_start, idx_end)
        )

    def to_df(self):
        """
        Build a pandas DataFrame of the ticks (a fresh copy each call).

        Returns:
            pd.DataFrame: Columns TICK_COLS (ts, bid, ask, volume, flags,
                volume_real, price) plus a 'datetime' column derived from 'ts'.

        Example:
            df = ticks.to_df()
        """
        import pandas as pd
        from tradetropy.core.constants import TICK_COLS
        df = pd.DataFrame(self.data, columns=list(TICK_COLS))
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df


# ==============================================================================
# KLINE DATA
# ==============================================================================


@dataclass
class KlineData:
    """
    OHLC candle + turnover data for a symbol.

    Attributes:
        symbol (str): Symbol name
        data (ndarray): [N x 7] array with columns: ts, open, high, low, close,
                        volume, turnover (turnover may be NaN if unavailable)
        timeframe (int | str): Candle duration. Accepts a timeframe string
                           ('1m', '5m', '1h', '1d', etc.) or an integer number
                           of milliseconds, parsed via parse_timeframe(). The
                           standard recommended set is '1m', '15m', '1h',
                           '4h', '1d', '1w', '1mo' ('min'/'wk' are accepted
                           aliases for 'm'/'w'; 'mo' is a fixed 30-day month).
                           The normalized duration in ms is exposed read-only
                           as `interval_ms`.
        tick_size (float): Minimum price step
        tick_value (float): Monetary value per tick. Defaults to tick_size (so
                            PnL per unit equals the price difference, like
                            backtesting.py).
        contract_size (float): Contract size
        digits (int): Decimal places for price
        avg_spread (float): Average spread in ticks

    Example:
        klines_btc = KlineData(
            symbol='BTCUSDT',
            data=btc_matrix,
            timeframe='5m',
            tick_size=0.01,
        )
        klines_eth = KlineData(
            symbol='ETHUSDT',
            data=eth_matrix,
            timeframe=60_000,
            tick_size=0.01,
        )
        bt = BacktestEngine.by_klines(
            MyStrategy(),
            data = (klines_btc, klines_eth),
        )
        bt.run()
    """

    symbol: str
    data: np.ndarray
    timeframe: "int | str"
    tick_size: float = 0.01
    tick_value: "float | None" = None
    contract_size: float = 1.0
    digits: int | None = None
    avg_spread: float = 0.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01

    @property
    def config(self) -> SymbolConfig:
        """
        Build SymbolConfig for passing to broker.add_symbol().

        Returns:
            SymbolConfig: Configuration object with this symbol's parameters
        """
        return SymbolConfig(
            name=self.symbol,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
            digits=self.digits,
            avg_spread=self.avg_spread,
            volume_min=self.volume_min,
            volume_max=self.volume_max,
            volume_step=self.volume_step,
        )

    @property
    def interval_ms(self) -> int:
        """Candle duration in milliseconds (normalized from `timeframe`)."""
        return self._interval_ms

    def __post_init__(self):
        from tradetropy.core.constants import parse_timeframe

        self.symbol = str(self.symbol)
        self.data = np.asarray(self.data, dtype=np.float64)
        self._interval_ms = parse_timeframe(self.timeframe)
        # tick_value defaults to tick_size so PnL equals the pure price
        # difference per unit (contract_size=1), matching backtesting.py's
        # unit model out of the box. Set an explicit tick_value only for
        # instruments where a tick is worth a different amount than its size
        # (e.g. futures like MES: tick_size 0.25, tick_value 1.25).
        if self.tick_value is None:
            self.tick_value = self.tick_size
        if self.digits is None:
            self.digits = infer_price_digits(self.tick_size)
        if not self._interval_ms or self._interval_ms <= 0:
            raise ConfigError(
                f'KlineData.timeframe must resolve to a positive duration. '
                f'Received: {self.timeframe!r}. '
                f'Examples: "1m", "5m", "1h", 60_000 (ms).'
            )

    def resample(self, timeframe) -> 'KlineData':
        """
        Resample candles to higher timeframe without mutating self.

        Args:
            timeframe (int or str): Target timeframe as a string ('1h', '4h',
                                    etc., standard set: '1m', '15m', '1h',
                                    '4h', '1d', '1w', '1mo') or an integer
                                    number of milliseconds. Must be a multiple
                                    of self.interval_ms; if not, adjusted to
                                    next multiple (with warning)

        Returns:
            KlineData: New KlineData at higher interval, config propagated

        Example:
            klines_1h = klines_1m.resample('1h')
        """
        from tradetropy.data._klines import resample_klines

        matrix, new_interval = resample_klines(
            self.data, self.interval_ms, timeframe
        )
        return KlineData(
            symbol=self.symbol,
            data=matrix,
            timeframe=new_interval,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
            digits=self.digits,
            avg_spread=self.avg_spread,
            volume_min=self.volume_min,
            volume_max=self.volume_max,
            volume_step=self.volume_step,
        )

    def save(self, path, format=None, **kwargs):
        """
        Save klines to file.

        Args:
            path: Destination file path.
            format: Output format ('csv', 'parquet', 'hdf5', 'npz'). None
                (default) infers from the extension, falling back to 'npz'.
            **kwargs: Extra arguments passed to save_klines() (ts_format,
                hdf5_key, compression, include_partial).

        Returns:
            pd.DataFrame: The saved kline data.

        Example:
            klines.save('btc_1h.h5')
            klines.resample('1h').save('btc_1h.parquet', format='parquet')
        """
        from tradetropy.io.io import save_klines
        metadata = {
            "tradetropy_symbol": self.symbol,
            "tradetropy_interval_ms": self.interval_ms,
            "tradetropy_tick_size": self.tick_size,
            "tradetropy_tick_value": self.tick_value,
            "tradetropy_contract_size": self.contract_size,
            "tradetropy_digits": self.digits,
            "tradetropy_avg_spread": self.avg_spread,
            "tradetropy_volume_min": self.volume_min,
            "tradetropy_volume_max": self.volume_max,
            "tradetropy_volume_step": self.volume_step,
        }
        return save_klines(self.data, path, format=format, metadata=metadata, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        from tradetropy.core.constants import format_timeframe
        label = format_timeframe(self.interval_ms)
        header = (
            f"KlineData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"timeframe={label}, tick_size={self.tick_size})"
        )
        from tradetropy.core.constants import OHLCV_TURNOVER_COLS
        table = _format_rows(self.data, OHLCV_TURNOVER_COLS, 5)
        return f"{header}\n{table}"

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter/IPython (head + tail preview)."""
        from tradetropy.core.constants import OHLCV_TURNOVER_COLS, format_timeframe
        label = format_timeframe(self.interval_ms)
        title = (
            f"KlineData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"timeframe={label}, tick_size={self.tick_size})"
        )
        return _format_rows_html(self.data, OHLCV_TURNOVER_COLS, title)

    def head(self, n: int = 5) -> 'KlineData':
        """Return a new KlineData with the first n rows."""
        return _slice_data(self, self.data[:n])

    def tail(self, n: int = 5) -> 'KlineData':
        """Return a new KlineData with the last n rows."""
        start = max(0, len(self.data) - n)
        return _slice_data(self, self.data[start:])

    def __getitem__(self, key) -> 'KlineData':
        """
        Index by position, returning a new KlineData of the same type.

        Args:
            key (int | slice): Integer position (a 1-row KlineData) or a slice
                (the ranged KlineData). Negative indices supported.

        Returns:
            KlineData: New KlineData with the selected rows and same metadata.

        Example:
            klines[500]        # single row (as a 1-row KlineData)
            klines[1000:1010]  # intermediate range
            klines[-50:]       # last 50 rows (like tail(50))
        """
        return _getitem_data(self, key)

    def filter(
        self,
        mask: "np.ndarray | Callable | None" = None,
        *,
        start: "str | int | None" = None,
        end: "str | int | None" = None,
        idx_start: "int | None" = None,
        idx_end: "int | None" = None,
    ) -> 'KlineData':
        """
        Filter klines without mutating the original.

        Args:
            mask: Boolean array [N] or callable ``lambda d: d[:, col] > val``.
            start: Start date ('2024-01-01', '2024-01-01 10:30') or ms timestamp.
            end: End date (same format), inclusive.
            idx_start (int | None): Positional index range start, inclusive.
            idx_end (int | None): Positional index range end, exclusive.

        Returns:
            KlineData: New KlineData with the same metadata.

        Example:
            klines.filter(start='2024-01-01', end='2024-01-31')
            klines.filter(idx_start=1000, idx_end=2000)
            klines.filter(mask=klines.data[:, 5] > 100)
            klines.filter(lambda d: d[:, 4] > 50000)
        """
        return _slice_data(
            self, _apply_filter(self.data, mask, start, end, idx_start, idx_end)
        )

    def to_df(self):
        """
        Build a pandas DataFrame of the candles (a fresh copy each call).

        Returns:
            pd.DataFrame: Columns OHLCV_TURNOVER_COLS (ts, open, high, low,
                close, volume, turnover) plus a 'datetime' column from 'ts'.

        Example:
            df = klines.to_df()
        """
        import pandas as pd
        from tradetropy.core.constants import OHLCV_TURNOVER_COLS
        df = pd.DataFrame(self.data, columns=list(OHLCV_TURNOVER_COLS))
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

# ==============================================================================
# INTERNAL HELPERS - used by engines
# ==============================================================================


def book_flat_columns(levels: int) -> list:
    """
    Column names for the flat L2 order-book row layout.

    A book event is stored as a single row:
        [ts, kind, bid_px_0..K-1, bid_sz_0..K-1, ask_px_0..K-1, ask_sz_0..K-1]

    where ``kind`` is 0 for a snapshot and 1 for a delta, level 0 is the best
    (top of book), and width = 2 + 4 * levels. Shared by BookData, LiveBookRing
    and the book IO so the on-disk and in-memory layouts match exactly.

    Args:
        levels (int): Number of book levels K retained per side.

    Returns:
        list[str]: Column names in row order.
    """
    cols = ["ts", "kind"]
    cols += [f"bid_px_{i}" for i in range(levels)]
    cols += [f"bid_sz_{i}" for i in range(levels)]
    cols += [f"ask_px_{i}" for i in range(levels)]
    cols += [f"ask_sz_{i}" for i in range(levels)]
    return cols


def book_row_width(levels: int) -> int:
    """Width of a flat book row for K levels (2 + 4 * levels)."""
    return 2 + 4 * int(levels)


# Compact top-of-book summary columns used by BookData's text/HTML repr. The
# full L2 depth is available via BookData.to_df().
BOOK_SUMMARY_COLS = (
    "ts",
    "kind",
    "best_bid",
    "best_ask",
    "spread",
    "mid",
    "bid_sz_0",
    "ask_sz_0",
)


# -----
# Long (Binance bookDepth-style) order-book layout and wide <-> long conversion
# -----

# Canonical long book columns. A book event is stored as several rows, one per
# price level per side.
BOOK_LONG_COLS = ("ts", "percentage", "depth", "notional")


def book_long_columns() -> list:
    """
    Column names for the long (Binance bookDepth-style) order-book layout.

    A single book event is stored as several rows, one per price level per side:
        ts, percentage, depth, notional

    where ``percentage`` is a signed 1-based level index (negative = bid side,
    positive = ask side; |percentage| == 1 is the top of book), ``depth`` is the
    CUMULATIVE size up to and including that level, and ``notional`` is the
    cumulative notional (running sum of price * size). This mirrors the Binance
    Futures ``bookDepth`` historical dataset (aggregated bands at 1..5 from the
    mid), so such files read directly, and it round-trips losslessly with the
    wide layout (:func:`book_flat_columns`) because de-cumulation on read exactly
    inverts cumulation on write.

    Returns:
        list[str]: Column names ('ts', 'percentage', 'depth', 'notional').
    """
    return list(BOOK_LONG_COLS)


def _emit_long_side(ts, px, sz, sign, out) -> None:
    """
    Append cumulative long rows for one side of one book event.

    Iterates levels from best (index 0) outward, skipping empty levels (NaN or
    zero size), accumulating depth and notional so the emitted rows follow the
    Binance bookDepth cumulative convention.

    Args:
        ts: Event timestamp (ms).
        px: Level prices [K] for this side.
        sz: Level sizes [K] for this side.
        sign: -1 for the bid side, +1 for the ask side.
        out: Dict of lists with keys 'ts', 'percentage', 'depth', 'notional'
            appended in place.
    """
    cum_depth = 0.0
    cum_notional = 0.0
    for i in range(len(px)):
        s = sz[i]
        p = px[i]
        if not np.isfinite(s) or not np.isfinite(p) or s == 0.0:
            continue
        cum_depth += float(s)
        cum_notional += float(p) * float(s)
        out["ts"].append(float(ts))
        out["percentage"].append(float(sign * (i + 1)))
        out["depth"].append(cum_depth)
        out["notional"].append(cum_notional)


def wide_rows_to_long(arr: np.ndarray, levels: int):
    """
    Convert wide flat book rows to the long (cumulative bookDepth) layout.

    Each wide row (one book event) expands to up to ``2 * levels`` long rows -
    one per non-empty level per side - carrying the CUMULATIVE depth and notional
    so the output matches the Binance bookDepth convention and de-cumulates back
    to the exact wide row via :func:`long_rows_to_wide`.

    Args:
        arr (np.ndarray): Wide book rows [N x (2 + 4*levels)] (see
            :func:`book_flat_columns`).
        levels (int): Number of levels K per side.

    Returns:
        pd.DataFrame: Long rows with columns from :func:`book_long_columns`.

    Example:
        long_df = wide_rows_to_long(book.data, book.levels)
    """
    import pandas as pd
    a = np.asarray(arr, dtype=np.float64)
    k = int(levels)
    out = {"ts": [], "percentage": [], "depth": [], "notional": []}
    if a.size:
        ts_all = a[:, 0]
        bid_px = a[:, 2 : 2 + k]
        bid_sz = a[:, 2 + k : 2 + 2 * k]
        ask_px = a[:, 2 + 2 * k : 2 + 3 * k]
        ask_sz = a[:, 2 + 3 * k : 2 + 4 * k]
        for r in range(a.shape[0]):
            _emit_long_side(ts_all[r], bid_px[r], bid_sz[r], -1, out)
            _emit_long_side(ts_all[r], ask_px[r], ask_sz[r], +1, out)
    return pd.DataFrame(
        {c: np.asarray(out[c], dtype=np.float64) for c in BOOK_LONG_COLS}
    )


def long_rows_to_wide(df):
    """
    Convert long (cumulative bookDepth) rows back to wide flat book rows.

    The inverse of :func:`wide_rows_to_long`. Rows are grouped by timestamp;
    within each side the cumulative ``depth`` / ``notional`` are de-cumulated by
    ascending level so ``size = depth_j - depth_{j-1}`` and
    ``price = (notional_j - notional_{j-1}) / size``. This exactly inverts the
    cumulation for files written by tradetropy, and reconstructs per-band VWAP
    prices for a real Binance ``bookDepth`` file (an approximation inherent to
    that percentage-band source, which carries no explicit per-level price).

    Args:
        df (pd.DataFrame): Long rows with the columns of
            :func:`book_long_columns` ('ts', 'percentage', 'depth', 'notional').

    Returns:
        tuple[np.ndarray, int]: Wide book rows [N x (2 + 4*levels)] (kind = 0,
        snapshot) and the resolved level count K.

    Raises:
        DataError: If required long columns are missing.
    """
    from collections import defaultdict

    missing = [c for c in BOOK_LONG_COLS if c not in df.columns]
    if missing:
        raise DataError(
            f"long_rows_to_wide(): missing column(s) {missing}. "
            f"Expected {list(BOOK_LONG_COLS)}."
        )

    ts_vals = df["ts"].to_numpy(dtype=np.float64)
    pct = df["percentage"].to_numpy(dtype=np.float64)
    depth = df["depth"].to_numpy(dtype=np.float64)
    notional = df["notional"].to_numpy(dtype=np.float64)

    level_idx = (np.abs(np.rint(pct)).astype(np.int64) - 1)
    side_sign = np.sign(pct).astype(np.int64)

    if len(level_idx) == 0:
        return np.empty((0, book_row_width(0)), dtype=np.float64), 0
    levels = int(level_idx.max()) + 1

    # Unique timestamps in first-seen order (preserve input ordering).
    order = []
    row_of = {}
    for t in ts_vals:
        if t not in row_of:
            row_of[t] = len(order)
            order.append(t)

    width = book_row_width(levels)
    out = np.full((len(order), width), np.nan, dtype=np.float64)
    out[:, 0] = np.asarray(order, dtype=np.float64)
    out[:, 1] = 0.0  # kind = snapshot

    bpx0 = 2
    bsz0 = 2 + levels
    apx0 = 2 + 2 * levels
    asz0 = 2 + 3 * levels

    # Collect (level, depth, notional) per (ts, side) then de-cumulate by level.
    groups = defaultdict(lambda: {"bid": [], "ask": []})
    for i in range(len(ts_vals)):
        side = "bid" if side_sign[i] < 0 else "ask"
        groups[ts_vals[i]][side].append((int(level_idx[i]), depth[i], notional[i]))

    for t, sides in groups.items():
        ridx = row_of[t]
        for side, items in sides.items():
            items.sort(key=lambda x: x[0])
            prev_depth = 0.0
            prev_notional = 0.0
            for lvl, cum_d, cum_n in items:
                if lvl < 0 or lvl >= levels:
                    continue
                s = cum_d - prev_depth
                n = cum_n - prev_notional
                prev_depth = cum_d
                prev_notional = cum_n
                p = n / s if (np.isfinite(s) and s != 0.0) else np.nan
                if side == "bid":
                    out[ridx, bpx0 + lvl] = p
                    out[ridx, bsz0 + lvl] = s
                else:
                    out[ridx, apx0 + lvl] = p
                    out[ridx, asz0 + lvl] = s

    return out, levels


@dataclass
class BookData:
    """
    L2 order-book (depth) data for a symbol.

    Stored as a flat [N x (2 + 4*levels)] matrix (mirroring TickData/KlineData's
    flat layout) so it persists and replays the same way. Each row is one book
    event in the layout described by :func:`book_flat_columns`.

    Attributes:
        symbol (str): Symbol name.
        data (np.ndarray): [N x (2 + 4*levels)] flat book rows.
        levels (int): Number of levels K retained per side.
        tick_size (float): Minimum price step.

    Example:
        book = BookData('BTCUSDT', rows, levels=10, tick_size=0.01)
        book.bid_px[-1]   # best-N bid prices of the last event
    """

    symbol: str
    data: np.ndarray
    levels: int
    tick_size: float = 0.01

    def __post_init__(self):
        self.symbol = str(self.symbol)
        self.data = np.asarray(self.data, dtype=np.float64)
        width = book_row_width(self.levels)
        if self.data.ndim != 2 or (self.data.size and self.data.shape[1] != width):
            raise ConfigError(
                f"BookData.data must be [N x {width}] for levels={self.levels}, "
                f"got shape {self.data.shape}."
            )

    @property
    def ts(self) -> np.ndarray:
        """Event timestamps [N] (ms)."""
        return self.data[:, 0]

    @property
    def kind(self) -> np.ndarray:
        """Event kind [N]: 0 = snapshot, 1 = delta."""
        return self.data[:, 1]

    @property
    def bid_px(self) -> np.ndarray:
        """Bid prices [N x K], level 0 = best."""
        k = self.levels
        return self.data[:, 2 : 2 + k]

    @property
    def bid_sz(self) -> np.ndarray:
        """Bid sizes [N x K], level 0 = best."""
        k = self.levels
        return self.data[:, 2 + k : 2 + 2 * k]

    @property
    def ask_px(self) -> np.ndarray:
        """Ask prices [N x K], level 0 = best."""
        k = self.levels
        return self.data[:, 2 + 2 * k : 2 + 3 * k]

    @property
    def ask_sz(self) -> np.ndarray:
        """Ask sizes [N x K], level 0 = best."""
        k = self.levels
        return self.data[:, 2 + 3 * k : 2 + 4 * k]

    @property
    def best_bid(self) -> np.ndarray:
        """
        Best (top-of-book) bid price per event [N].

        Level 0 is the best level in the flat layout, so this is the first bid
        column. NaN where the bid side is empty. Mirrors OrderbookProxy.best_bid
        as a full series.
        """
        if self.levels == 0 or self.data.size == 0:
            return np.empty(len(self.data), dtype=np.float64)
        return self.bid_px[:, 0]

    @property
    def best_ask(self) -> np.ndarray:
        """
        Best (top-of-book) ask price per event [N].

        Level 0 is the best level in the flat layout, so this is the first ask
        column. NaN where the ask side is empty. Mirrors OrderbookProxy.best_ask
        as a full series.
        """
        if self.levels == 0 or self.data.size == 0:
            return np.empty(len(self.data), dtype=np.float64)
        return self.ask_px[:, 0]

    @property
    def mid(self) -> np.ndarray:
        """
        Mid price per event [N]: (best_bid + best_ask) / 2.

        NaN where either side is missing (NumPy NaN propagation). Mirrors
        OrderbookProxy.mid as a full series.
        """
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> np.ndarray:
        """
        Spread per event [N]: best_ask - best_bid.

        NaN where either side is missing (NumPy NaN propagation). Mirrors
        OrderbookProxy.spread as a full series.
        """
        return self.best_ask - self.best_bid

    def save(self, path, format=None, **kwargs):
        """
        Save order-book rows to file.

        Args:
            path: Destination file path.
            format: Output format ('csv', 'parquet', 'hdf5', 'npz'). None
                (default) infers from the extension, falling back to 'npz'.
            **kwargs: Extra arguments passed to save_book() (layout, ts_format,
                hdf5_key, compression, metadata).

        Returns:
            pd.DataFrame: The saved book data.

        Example:
            book.save('btc_book.h5')
            book.save('btc_book.parquet', format='parquet')
            book.save('btc_bookdepth.csv', format='csv', layout='long')
        """
        from tradetropy.io.io import save_book
        metadata = {
            "tradetropy_symbol": self.symbol,
            "tradetropy_levels": self.levels,
            "tradetropy_tick_size": self.tick_size,
        }
        return save_book(
            self, path, format=format, metadata=metadata, **kwargs
        )

    def to_df(self):
        """
        Build a pandas DataFrame of the book events (a fresh copy each call).

        Returns:
            pd.DataFrame: Columns from book_flat_columns(self.levels) (ts, kind,
                bid_px_*/bid_sz_*/ask_px_*/ask_sz_*) plus a 'datetime' column
                derived from 'ts'.

        Example:
            df = book.to_df()
        """
        import pandas as pd
        df = pd.DataFrame(self.data, columns=book_flat_columns(self.levels))
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def _summary_matrix(self) -> np.ndarray:
        """
        Build the top-of-book summary matrix [N x 8] used by the text/HTML repr.

        Columns match :data:`BOOK_SUMMARY_COLS`
        (ts, kind, best_bid, best_ask, spread, mid, bid_sz_0, ask_sz_0). The
        full L2 depth remains available via :meth:`to_df`.

        Returns:
            np.ndarray: [N x 8] summary matrix (empty [0 x 8] when there are no
            rows).
        """
        n = len(self.data)
        if n == 0 or self.levels == 0:
            return np.empty((0, len(BOOK_SUMMARY_COLS)), dtype=np.float64)
        return np.column_stack(
            [
                self.ts,
                self.kind,
                self.best_bid,
                self.best_ask,
                self.spread,
                self.mid,
                self.bid_sz[:, 0],
                self.ask_sz[:, 0],
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        header = (
            f"BookData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"levels={self.levels}, tick_size={self.tick_size})"
        )
        table = _format_rows(self._summary_matrix(), BOOK_SUMMARY_COLS, 5)
        return f"{header}\n{table}"

    def _repr_html_(self) -> str:
        """Rich HTML top-of-book table for Jupyter/IPython (head + tail preview)."""
        title = (
            f"BookData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"levels={self.levels}, tick_size={self.tick_size})"
        )
        return _format_rows_html(self._summary_matrix(), BOOK_SUMMARY_COLS, title)

    def head(self, n: int = 5) -> 'BookData':
        """Return a new BookData with the first n events."""
        return _slice_data(self, self.data[:n])

    def tail(self, n: int = 5) -> 'BookData':
        """Return a new BookData with the last n events."""
        start = max(0, len(self.data) - n)
        return _slice_data(self, self.data[start:])

    def __getitem__(self, key) -> 'BookData':
        """
        Index by position, returning a new BookData of the same type.

        Args:
            key (int | slice): Integer position (a 1-row BookData) or a slice
                (the ranged BookData). Negative indices supported.

        Returns:
            BookData: New BookData with the selected events and same metadata.

        Example:
            book[100]        # single event (as a 1-row BookData)
            book[100:200]    # intermediate range
            book[-50:]       # last 50 events (like tail(50))
        """
        return _getitem_data(self, key)

    def filter(
        self,
        mask: "np.ndarray | Callable | None" = None,
        *,
        start: "str | int | None" = None,
        end: "str | int | None" = None,
        idx_start: "int | None" = None,
        idx_end: "int | None" = None,
    ) -> 'BookData':
        """
        Filter book events without mutating the original.

        Args:
            mask: Boolean array [N] or callable ``lambda d: d[:, col] > val``.
            start: Start date ('2024-01-01', '2024-01-01 10:30') or ms timestamp.
            end: End date (same format), inclusive.
            idx_start (int | None): Positional index range start, inclusive.
            idx_end (int | None): Positional index range end, exclusive.

        Returns:
            BookData: New BookData with the same metadata (levels, tick_size).

        Example:
            book.filter(start='2024-01-01', end='2024-01-31')
            book.filter(idx_start=1000, idx_end=2000)
            book.filter(lambda d: d[:, 1] == 0)   # snapshots only
        """
        return _slice_data(
            self, _apply_filter(self.data, mask, start, end, idx_start, idx_end)
        )


# L3 / MBO (market-by-order) columns and action codes.
MBO_COLS = ("ts", "order_id", "side", "price", "size", "action")
N_MBO_COLS = len(MBO_COLS)
_MBO_COL = {name: i for i, name in enumerate(MBO_COLS)}

# MBO action codes.
MBO_ADD = 0
MBO_MODIFY = 1
MBO_CANCEL = 2
MBO_TRADE = 3


@dataclass
class MboData:
    """
    L3 / market-by-order data for a symbol.

    Per-order events stored as a flat [M x 6] matrix:
        ts, order_id, side (+1 bid / -1 ask), price, size, action
    where action is MBO_ADD / MBO_MODIFY / MBO_CANCEL / MBO_TRADE.

    Attributes:
        symbol (str): Symbol name.
        data (np.ndarray): [M x 6] flat MBO event rows.
        tick_size (float): Minimum price step.
    """

    symbol: str
    data: np.ndarray
    tick_size: float = 0.01

    def __post_init__(self):
        self.symbol = str(self.symbol)
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2 or (self.data.size and self.data.shape[1] != N_MBO_COLS):
            raise ConfigError(
                f"MboData.data must be [M x {N_MBO_COLS}] "
                f"({', '.join(MBO_COLS)}), got shape {self.data.shape}."
            )

    @property
    def ts(self) -> np.ndarray:
        return self.data[:, _MBO_COL["ts"]]

    @property
    def order_id(self) -> np.ndarray:
        return self.data[:, _MBO_COL["order_id"]]

    @property
    def side(self) -> np.ndarray:
        return self.data[:, _MBO_COL["side"]]

    @property
    def price(self) -> np.ndarray:
        return self.data[:, _MBO_COL["price"]]

    @property
    def size(self) -> np.ndarray:
        return self.data[:, _MBO_COL["size"]]

    @property
    def action(self) -> np.ndarray:
        return self.data[:, _MBO_COL["action"]]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        header = (
            f"MboData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"tick_size={self.tick_size})"
        )
        table = _format_rows(self.data, MBO_COLS, 5)
        return f"{header}\n{table}"

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter/IPython (head + tail preview)."""
        title = (
            f"MboData(symbol={self.symbol!r}, rows={len(self.data)}, "
            f"tick_size={self.tick_size})"
        )
        return _format_rows_html(self.data, MBO_COLS, title)

    def head(self, n: int = 5) -> 'MboData':
        """Return a new MboData with the first n events."""
        return _slice_data(self, self.data[:n])

    def tail(self, n: int = 5) -> 'MboData':
        """Return a new MboData with the last n events."""
        start = max(0, len(self.data) - n)
        return _slice_data(self, self.data[start:])

    def __getitem__(self, key) -> 'MboData':
        """
        Index by position, returning a new MboData of the same type.

        Args:
            key (int | slice): Integer position (a 1-row MboData) or a slice
                (the ranged MboData). Negative indices supported.

        Returns:
            MboData: New MboData with the selected events and same metadata.

        Example:
            mbo[100]        # single event (as a 1-row MboData)
            mbo[100:200]    # intermediate range
            mbo[-50:]       # last 50 events (like tail(50))
        """
        return _getitem_data(self, key)

    def filter(
        self,
        mask: "np.ndarray | Callable | None" = None,
        *,
        start: "str | int | None" = None,
        end: "str | int | None" = None,
        idx_start: "int | None" = None,
        idx_end: "int | None" = None,
    ) -> 'MboData':
        """
        Filter MBO events without mutating the original.

        Args:
            mask: Boolean array [N] or callable ``lambda d: d[:, col] > val``.
            start: Start date ('2024-01-01', '2024-01-01 10:30') or ms timestamp.
            end: End date (same format), inclusive.
            idx_start (int | None): Positional index range start, inclusive.
            idx_end (int | None): Positional index range end, exclusive.

        Returns:
            MboData: New MboData with the same metadata (tick_size).

        Example:
            mbo.filter(start='2024-01-01', end='2024-01-31')
            mbo.filter(idx_start=1000, idx_end=2000)
            mbo.filter(lambda d: d[:, 5] == MBO_TRADE)   # trades only
        """
        return _slice_data(
            self, _apply_filter(self.data, mask, start, end, idx_start, idx_end)
        )

    def to_df(self):
        """
        Build a pandas DataFrame of the MBO events (a fresh copy each call).

        Returns:
            pd.DataFrame: Columns MBO_COLS (ts, order_id, side, price, size,
                action) plus a 'datetime' column derived from 'ts'.

        Example:
            df = mbo.to_df()
        """
        import pandas as pd
        df = pd.DataFrame(self.data, columns=list(MBO_COLS))
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df


def _validate_inputs_ticks(inputs: tuple) -> None:
    """Validate all elements are TickData."""
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TickData):
            raise ConfigError(
                f'by_ticks() expected TickData at position {i}, '
                f'received {type(inp).__name__}. '
                f'For candles use by_klines() with KlineData.'
            )


def _validate_inputs_klines(inputs: tuple) -> None:
    """Validate all elements are KlineData."""
    for i, inp in enumerate(inputs):
        if not isinstance(inp, KlineData):
            raise ConfigError(
                f'by_klines() expected KlineData at position {i}, '
                f'received {type(inp).__name__}. '
                f'For ticks use by_ticks() with TickData.'
            )


def _inputs_to_dict(inputs: tuple) -> dict:
    """Convert tuple of TickData/KlineData to dict mapping symbol to data."""
    result = {}
    for inp in inputs:
        if inp.symbol in result:
            raise ConfigError(
                f'Duplicate symbol {inp.symbol!r} in data tuple. '
                f'Each symbol must appear once.'
            )
        result[inp.symbol] = inp.data
    return result


def _inputs_to_configs(inputs: tuple) -> dict:
    """Extract dict mapping symbol to SymbolConfig from inputs tuple."""
    return {inp.symbol: inp.config for inp in inputs}


# ==============================================================================
# CANONICAL NORMALIZER - single source of truth for `data` in ALL engines
# ==============================================================================


def _normalize_data(data) -> tuple:
    """
    Canonical data normalizer, shared by BacktestEngine, LiveEngine (via replay),
    ReplayEngine, and PoolBacktestEngine.

    Object type is the only source of truth for tick vs kline - no mode parameter
    or shape detection.

    Accepts:
        - Single TickData / KlineData
        - Homogeneous tuple or list of them

    Returns:
        (inputs_tuple, feed_type) with feed_type in {'tick', 'kline'}.

    Raises ConfigError if:
        - Dict is passed (legacy dict-based API removed)
        - TickData mixed with KlineData
        - Duplicate symbols present
        - Collection is empty or unsupported type
    """
    if isinstance(data, dict):
        raise ConfigError(
            'Dict-based `data` API was removed. Use a tuple of TickData/KlineData, '
            'e.g. data=(TickData(\'BTCUSDT\', arr, tick_size=0.01),). '
            'Object type determines tick vs kline.'
        )

    if isinstance(data, (TickData, KlineData)):
        inputs = (data,)
    elif isinstance(data, (tuple, list)):
        inputs = tuple(data)
    else:
        raise ConfigError(
            f'`data` must be TickData/KlineData or tuple of them, '
            f'received {type(data).__name__}.'
        )

    if not inputs:
        raise ConfigError(
            '`data` is empty: pass at least one TickData/KlineData.'
        )

    first = inputs[0]
    if isinstance(first, TickData):
        feed_type = 'tick'
        _validate_inputs_ticks(inputs)
    elif isinstance(first, KlineData):
        feed_type = 'kline'
        _validate_inputs_klines(inputs)
    else:
        raise ConfigError(
            f'Unsupported `data` element: {type(first).__name__}. '
            f'Use TickData or KlineData.'
        )

    symbols = [inp.symbol for inp in inputs]
    dups = sorted({s for s in symbols if symbols.count(s) > 1})
    if dups:
        raise ConfigError(
            f'Duplicate symbols in `data`: {dups}. '
            f'Each symbol must appear once.'
        )

    return inputs, feed_type


def _normalize_book(book) -> dict:
    """
    Canonical order-book normalizer, shared by every engine that accepts a
    recorded L2 book (BacktestEngine, PoolBacktestEngine, ReplayEngine,
    PaperEngine).

    Symmetric with :func:`_normalize_data`: the symbol is taken from each
    ``BookData.symbol`` (the book already knows its symbol), so the caller
    passes the book(s) directly instead of a ``{symbol: BookData}`` mapping.

    Accepts:
        - None -> ``{}`` (no book)
        - Single BookData
        - Homogeneous tuple or list of BookData

    Returns:
        dict: ``{symbol: BookData}`` keyed by each book's own symbol.

    Raises:
        ConfigError: If a dict is passed (mapping API removed), a non-BookData
            element is present, or two books share the same symbol.
    """
    if book is None:
        return {}

    if isinstance(book, dict):
        raise ConfigError(
            'Dict-based `book` API was removed. Pass BookData directly, e.g. '
            'book=recorded_book or book=(book_btc, book_eth); the symbol is '
            'read from each BookData.symbol.'
        )

    if isinstance(book, BookData):
        inputs = (book,)
    elif isinstance(book, (tuple, list)):
        inputs = tuple(book)
    else:
        raise ConfigError(
            f'`book` must be a BookData or a tuple/list of them, '
            f'received {type(book).__name__}.'
        )

    for b in inputs:
        if not isinstance(b, BookData):
            raise ConfigError(
                f'`book` elements must be BookData, got {type(b).__name__}.'
            )

    symbols = [b.symbol for b in inputs]
    dups = sorted({s for s in symbols if symbols.count(s) > 1})
    if dups:
        raise ConfigError(
            f'Duplicate symbols in `book`: {dups}. '
            f'Each symbol must appear once.'
        )

    return {b.symbol: b for b in inputs}


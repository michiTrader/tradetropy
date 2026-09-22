from __future__ import annotations

import numpy as np

from tradetropy.ta.structure.pivots import ConfirmedPivot


def extract_confirmed_pivots(
    source: np.ndarray,
    swing: int,
    *,
    use_ts_fallback: bool = True,
) -> list[tuple[int, str, float, float]]:
    """
    Compute ConfirmedPivot and return list of (bar_idx, type, price, ts_real).
    """
    cp  = ConfirmedPivot(swing=swing)
    raw = cp.calculate(source)

    ph, pl, ph_ts, pl_ts = raw[0], raw[1], raw[2], raw[3]
    n = len(ph)

    ts_ms = source[:, 2].astype(np.float64) if use_ts_fallback else None

    pivots: list[tuple[int, str, float, float]] = []
    for i in range(n):
        if not np.isnan(ph[i]):
            ts = float(ph_ts[i]) if (not use_ts_fallback or not np.isnan(ph_ts[i])) else float(ts_ms[i])
            pivots.append((i, 'H', float(ph[i]), ts))
        elif not np.isnan(pl[i]):
            ts = float(pl_ts[i]) if (not use_ts_fallback or not np.isnan(pl_ts[i])) else float(ts_ms[i])
            pivots.append((i, 'L', float(pl[i]), ts))
    return pivots

"""Small statistical helpers that must match the legacy LDEO_IX semantics exactly."""

from __future__ import annotations

import numpy as np


def medianan(x: np.ndarray, na: int = 0) -> float:
    """Legacy ``medianan(x, na)``: mean of the central ``2*na+1`` order statistics.

    Sorts the finite values of ``x`` and averages those at 1-based positions
    ``round((-na:na) + L/2)`` (clipped to ``[1, L]``). ``na=0`` selects a single
    central element (NOT the two-element average a plain even-length median would give),
    which is why this is not interchangeable with :func:`numpy.nanmedian`.
    """
    xs = np.sort(np.asarray(x, float)[np.isfinite(x)])
    L = xs.size
    if L == 0:
        return np.nan
    idx = np.round(np.arange(-na, na + 1) + L / 2.0).astype(int)
    idx = idx[(idx >= 1) & (idx <= L)]
    return float(np.mean(xs[idx - 1]))

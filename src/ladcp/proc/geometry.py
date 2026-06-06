"""Bin geometry helpers."""

from __future__ import annotations

import numpy as np

from ..models import RawADCP


def bin_distances(adcp: RawADCP) -> np.ndarray:
    """Vertical distance from transducer to each bin centre [m], positive away.

    RDI depth cells are referenced vertically; centre of bin k (1-based) is
    ``blank + cell/2 + (k-1)*cell``.
    """
    k = np.arange(1, adcp.n_cells + 1)
    return adcp.blank_m + adcp.cell_m / 2.0 + (k - 1) * adcp.cell_m

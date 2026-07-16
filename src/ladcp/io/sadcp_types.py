"""The shared ship-ADCP data contract, neutral home for all three readers.

:mod:`.sadcp_vmdas` (raw VmDAS), :mod:`.sadcp_codas` (CODAS-processed NetCDF) and
:mod:`.sadcp_ek80` (EK80 ADCP-mode) are peers that all return a :class:`SadcpDataset`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SadcpDataset:
    """Absolute ocean-velocity time series from one shipboard-ADCP instrument.

    ``u``/``v`` are east/north ocean velocity in the **true** (geographic) frame, the
    frame :func:`ladcp.qa.inverse.compute_velocity_full` expects for its ``sadcp=``
    constraint. ``depth`` is positive-down below the sea surface.
    """

    time: np.ndarray            # [t] datetime64[ns] UTC (sorted)
    lat: np.ndarray             # [t] deg N
    lon: np.ndarray             # [t] deg E
    depth: np.ndarray           # [z] m, positive down
    u: np.ndarray               # [z, t] absolute ocean east, m/s
    v: np.ndarray               # [z, t] absolute ocean north, m/s
    freq_khz: int
    transducer_depth: float
    file_type: str              # "STA"/"LTA" (raw VmDAS) or "CODAS" (sadcp_codas)
    source: str                 # ingested folder
    n_files: int

    @property
    def n_ens(self) -> int:
        return int(self.time.size)

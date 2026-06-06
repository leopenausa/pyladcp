"""Phase 1 — dual-head ingest.

Decodes the down (Master) and up (Slave) PD0 files and assembles them into a single
:class:`DualHead` container, keeping the two heads separate (no rotation, no editing).
Also extracts the instrument-config scalars that the golden records in ``p`` so they can
be asserted in tests.

The vertical bin-depth axis follows the legacy convention: bin ``i`` (0-based) sits at
``dist_first + i * cell`` metres from the transducer (positive away from the head:
downward for the down-looker, upward for the up-looker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import CastParams
from ..io.pd0 import read_pd0
from ..models import RawADCP


@dataclass
class DualHead:
    """Both LADCP heads for one cast, raw and unedited."""

    station: str
    down: RawADCP
    up: RawADCP | None = None
    params: CastParams | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_up(self) -> bool:
        return self.up is not None

    def bin_depth(self, head: RawADCP) -> np.ndarray:
        """Vertical distance [m] from the transducer to each bin centre (always +)."""
        dist0 = head.meta.get("dist_first_m")
        if dist0 is None:                       # fallback: blank + half cell
            dist0 = head.blank_m + head.cell_m / 2.0
        return dist0 + np.arange(head.n_cells) * head.cell_m


def load_dualhead(
    down_path: str,
    up_path: str | None = None,
    *,
    station: str = "",
    params: CastParams | None = None,
) -> DualHead:
    """Read down (+ optional up) PD0 files into a :class:`DualHead`."""
    down = read_pd0(down_path, head="down", facing_hint="down")
    up = read_pd0(up_path, head="up", facing_hint="up") if up_path else None
    return DualHead(station=station, down=down, up=up, params=params,
                    meta={"down_path": down_path, "up_path": up_path})


def extract_config(dh: DualHead) -> dict[str, Any]:
    """Pull the instrument-config scalars that the golden stores in ``p``.

    Keys mirror the legacy field names (``nbin_d``, ``blen_d``, ``dist_d`` …) so the
    validation harness can diff against the MORIA-80 ``p``-struct directly.
    """
    d = dh.down
    cfg: dict[str, Any] = {
        "nping_total": [d.n_ens] + ([dh.up.n_ens] if dh.has_up else []),
        "nbin_d": d.n_cells,
        "blen_d": d.cell_m,
        "blnk_d": d.blank_m,
        "dist_d": d.meta.get("dist_first_m"),
        "beamangle": d.meta.get("beam_angle_deg"),
        "wm_d": d.wm_mode,
        "ppe_d": d.pings_per_ens,
        "coord_d": d.coord_frame.value,
        "serial_d": d.serial,
    }
    if dh.has_up:
        u = dh.up
        cfg.update(
            nbin_u=u.n_cells, blen_u=u.cell_m, blnk_u=u.blank_m,
            dist_u=u.meta.get("dist_first_m"), wm_u=u.wm_mode,
            ppe_u=u.pings_per_ens, coord_u=u.coord_frame.value, serial_u=u.serial,
        )
    return cfg

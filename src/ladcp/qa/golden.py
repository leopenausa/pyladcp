"""Load golden QA targets from the MORIA-80 ``p``-struct.

``New_golden/MORIA-80_LEO/MORIA-80.mat`` contains the legacy ``p`` parameter struct,
which records the exact ingest counts, thresholds, beam ranges, attitude offsets and
bottom depth produced by the reference LDEO_IX run. We read those scalars/small arrays
straight from the file so the validation harness never hardcodes (and drifts from) the
golden numbers.

The companion ``MORIA-80.ladcp.mat`` ``da`` (raw bin x ensemble archive) is truncated
and cannot be read, so only derived/scalar targets are available — by design we
reproduce the raw fields ourselves from the PD0 files and validate the derived numbers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# Repo-relative default; override in tests if the tree moves.
DEFAULT_MAT = (
    Path(__file__).resolve().parents[3]
    / "New_golden" / "MORIA-80_LEO" / "MORIA-80.mat"
)

# p-struct fields we expose as QA targets (legacy name -> kept name).
_SCALAR_FIELDS = (
    "nping_total", "nbin_d", "nbin_u", "blen_d", "blen_u", "blnk_d", "blnk_u",
    "dist_d", "dist_u", "beamangle", "pglim", "elim", "vlim", "wlim", "tiltmax",
    "weighbin1", "xmv", "xmc", "battery", "up_dn_comp_off", "up_dn_pit_off",
    "up_dn_rol_off", "dn_range", "up_range", "ts_att_dn", "ts_att_up",
    "zbottom", "maxdepth", "zbottomerror", "btrk_mode", "btrk_used", "btrk_range",
    "outlier", "outlier_n", "fix_compass", "rotup2down", "avdz", "lat", "lon",
)


@lru_cache(maxsize=4)
def load_p_struct(mat_path: str | None = None) -> dict[str, Any]:
    """Return the golden ``p`` fields in ``_SCALAR_FIELDS`` as a plain dict."""
    import scipy.io as sio

    path = Path(mat_path) if mat_path else DEFAULT_MAT
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False,
                    variable_names=["p"])
    p = m["p"]
    out: dict[str, Any] = {}
    for f in _SCALAR_FIELDS:
        if f in p._fieldnames:
            v = getattr(p, f)
            out[f] = v.tolist() if isinstance(v, np.ndarray) else v
    # warnings live in p.warn (cell array of strings)
    if "warn" in p._fieldnames:
        w = getattr(p, "warn")
        out["warn"] = [str(x) for x in (w if isinstance(w, np.ndarray) else [w])]
    return out


# Counts that only appear in the processing .log (not in p), kept here as the
# single source of truth for the ingest/screen gates.
LOG_COUNTS = {
    "pg_removed_down": 59727,
    "pg_removed_up": 65340,
    "tilt_removed_gt22": 6,
    "tilt_removed_deriv4": 116,
    "edit_bin_mask": 5071,
    "edit_sidelobe": 2022,
}

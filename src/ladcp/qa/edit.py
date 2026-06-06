"""edit_data — bin masking + side-lobe contamination (Fig 14 stage).

Builds the combined up+down target-strength field and applies the legacy ``edit_data.m``
edits so the contaminated regions can be visualized (before/after) for QA:

* bin masking      : remove bin 1 of each head (nearest-transducer, ringing).
* side-lobe        : near the surface (up-looker) and near the seabed (down-looker),
                     cells within ``(1 - cos(beam_angle)) * range`` of the boundary are
                     contaminated by the side-lobe reflection.
* below-bottom     : down-looker cells more than ``dzbelow`` below the seabed.

Geometry (depth ``izm = package_depth + bin_offset``; up offset ``+zu``, down ``-zd``;
all depths negative-down) is validated against the golden. NOTE: the legacy *counts*
(bin-mask 5071, side-lobe 2022) operate on the correlation-derived weight matrix whose
exact finite pattern depends on loadrdi internals not reproducible from the available
files (the golden raw ``da`` archive is truncated). The counts here are therefore
indicative; the figure and contaminated-region geometry are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bottom import BottomResult
from .depth import SyncResult
from .ingest import DualHead
from .screen import _ERR_COMP, _PG_FIELD

_DZBELOW = 16.0          # legacy p.dzbelow(1) = 2 * bin size; margin below seabed


@dataclass
class EditResult:
    bin_no: np.ndarray           # [nu+1+nd] signed bin number (0 = gap row)
    ts_before: np.ndarray        # [nbins, nens] target strength
    ts_after: np.ndarray         # [nbins, nens] target strength after edits
    counts: dict[str, int]       # indicative counts (see module note)


def _ts_field(head, nens: int | None = None) -> np.ndarray:
    """Median-over-beams target strength [nbin, nens] in dB (counts*0.45)."""
    ts = np.nanmedian(head.echo, axis=0) * 0.45
    return ts[:, :nens] if nens is not None else ts


def edit_data(dh: DualHead, sync: SyncResult, bottom: BottomResult) -> EditResult:
    d, u = dh.down, dh.up
    nd, nu = d.n_cells, (u.n_cells if u is not None else 0)
    nens = d.n_ens                                  # joint grid = down ping times (z is here)
    z = sync.z_on_ping                              # package depth +down
    zneg = -z                                       # negative-down convention
    zbottom = bottom.zbottom

    zd = d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2) + np.arange(nd) * d.cell_m
    beam = d.meta.get("beam_angle_deg", 20)

    # combined TS: up bins flipped, gap row, down bins  (matches edit_data.m layout)
    ts_d = _ts_field(d, nens)
    rows = []
    bin_no = []
    if nu:
        zu = u.meta.get("dist_first_m", u.blank_m + u.cell_m / 2) + np.arange(nu) * u.cell_m
        ts_u = _ts_field(u, nens)
        rows.append(np.flipud(ts_u)); bin_no += list(range(-nu, 0))
    rows.append(np.full((1, nens), np.nan)); bin_no.append(0)
    rows.append(ts_d); bin_no += list(range(1, nd + 1))
    ts_before = np.vstack(rows)

    # combined bin offset vector [fliplr(zu), 0, -zd] and depth matrix izm
    offset = []
    if nu:
        offset += list(np.flipud(zu))
    offset += [np.nan]
    offset += list(-zd)
    offset = np.array(offset)[:, None]
    izm = zneg[None, :] + offset                    # [nbins, nens] negative-down

    # good / weight-finite mask (pg + errvel; the loadrdi velocity edits)
    good = np.zeros_like(ts_before, dtype=bool)
    r0 = 0
    if nu:
        gu = ~((u.pct_good[_PG_FIELD][:, :nens] < 50)
               | (np.abs(u.vel[_ERR_COMP][:, :nens]) > 0.2))
        good[:nu] = np.flipud(gu); r0 = nu
    good[r0] = False                                # gap row
    gd = ~((d.pct_good[_PG_FIELD] < 50) | (np.abs(d.vel[_ERR_COMP]) > 0.2))
    good[r0 + 1:] = gd

    ts_after = ts_before.copy()
    counts: dict[str, int] = {}

    # --- below-bottom (down-looker) ---
    below = np.isfinite(izm) & (izm < -zbottom - _DZBELOW)
    counts["below_bottom"] = int(np.sum(below & good))
    ts_after[below] = np.nan; good = good & ~below

    # --- bin masking: bin 1 of each head ---
    nbad = 0
    if nu:
        r_up1 = nu - 1                              # up bin 1 = last up row (nearest TX)
        nbad += int(np.sum(good[r_up1])); ts_after[r_up1] = np.nan; good[r_up1] = False
    r_dn1 = r0 + 1                                  # down bin 1 = first down row
    nbad += int(np.sum(good[r_dn1])); ts_after[r_dn1] = np.nan; good[r_dn1] = False
    counts["bin_mask"] = nbad

    # --- side-lobe contamination ---
    nbad = 0
    cell = d.cell_m
    if nu:
        zlim_surf = (1 - np.cos(np.radians(beam))) * zneg[None, :] - 0.015 * cell
        sl = np.isfinite(izm) & (izm > zlim_surf)
        nbad += int(np.sum(sl & good)); ts_after[sl] = np.nan; good = good & ~sl
    zlim_bot = (-zbottom + (1 - np.cos(np.radians(beam))) * (zneg[None, :] + zbottom)
                + 0.015 * cell)
    sl = np.isfinite(izm) & (izm < zlim_bot)
    nbad += int(np.sum(sl & good)); ts_after[sl] = np.nan
    counts["side_lobe"] = nbad

    return EditResult(np.array(bin_no), ts_before, ts_after, counts)

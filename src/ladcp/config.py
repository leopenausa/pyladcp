"""Per-cast processing parameters (mirrors legacy ``p``/``ps`` + ``default.m``).

Only the knobs that matter for the current scope are modelled explicitly; the rest can
ride in ``extra``. The MORIA-05 preset encodes the effective values resolved from the
legacy ``set_cast_params.m`` (see docs/VALIDATION_MORIA05.md §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CastParams:
    station: str
    cruise_id: str = ""

    # head selection: 1 = up+down, 2 = down only, 3 = up only
    up_dn_looker: int = 1

    # inversion
    dz: float = 8.0                 # vertical resolution [m]

    # editing
    edit_mask_dn_bins: tuple[int, ...] = (1,)
    edit_mask_up_bins: tuple[int, ...] = (1,)
    cut: float = 0.0                # surface cut / draft trim [ensembles or m, per legacy]
    pglim: float = 0.0              # percent-good minimum
    elim: float = 0.2               # error-velocity limit [m/s]
    vlim: float = 1.0               # horizontal velocity limit [m/s]
    wlim: float = 0.08              # vertical velocity limit [m/s]
    tiltmax: tuple[float, float] = (22.0, 4.0)
    weighbin1: float = 0.1

    # bottom track: 0 off, 1/2/3 modes per legacy getbtrack
    btrk_mode: int = 3

    # super-ensemble outlier removal (legacy ps.outlier; process_cast STEP 11 lanarrow):
    # number of 1%-worst-residual rejection passes. 0 disables (single solve).
    outlier: float = 1.0

    # timing / geomagnetism
    timoff: float = 0.0             # clock offset [decimal days]
    drot: float | None = None       # magnetic declination [deg]; None -> compute IGRF

    # depth: 1 = integrate W; 2 = inverse + bottom reflection (default)
    getdepth: int = 2

    # sADCP (designed-for; see DATA_CONTRACT §2.3 / VALIDATION §7)
    sadcp: int = 0                  # selector id (0 = none requested)
    sadcpfac: float = 0.0           # constraint weight (0 = compare-only)
    sadcp_mismatch_policy: str = "reject"   # reject | downweight | operator_override
    sadcp_max_offset_deg: float = 0.1       # legacy position-check threshold

    # CTD / nav field map for the cleaned .cnv (1-based field numbers, like legacy)
    ctd_fields_per_line: int = 6
    ctd_pressure_field: int = 3
    ctd_temperature_field: int = 5
    ctd_salinity_field: int = 6
    ctd_time_field: int = 4
    ctd_time_base: int = 0          # 0 = elapsed seconds
    nav_lat_field: int = 1
    nav_lon_field: int = 2

    extra: dict[str, Any] = field(default_factory=dict)


def _moria_params(station: str) -> CastParams:
    """Effective LDEO_IX parameters shared by the clean MORIA dual-head casts.

    These are the values resolved from ``set_cast_params.m`` merged over ``default.m``;
    none of the per-station override branches apply to the clean casts 05–08, so they
    share one parameter set (see docs/VALIDATION_MORIA05.md §3).
    """
    return CastParams(
        station=station,
        cruise_id="MORIA",
        up_dn_looker=1,
        dz=8.0,
        edit_mask_dn_bins=(1,),
        edit_mask_up_bins=(1,),
        cut=7.0,
        pglim=50.0,
        elim=0.2,
        vlim=1.0,
        wlim=0.08,
        tiltmax=(22.0, 4.0),
        btrk_mode=3,
        timoff=0.0,
        drot=None,            # IGRF-computed; golden 05 = -2.5456, 06 = -2.6418 deg
        getdepth=2,
        sadcp=75,             # declared in legacy, but no file supplied -> inert
        sadcpfac=3.0,
    )


def moria05_params() -> CastParams:
    """Effective parameters for the MORIA-05 validation cast."""
    return _moria_params("MORIA-05")


def moria06_params() -> CastParams:
    """Effective parameters for the MORIA-06 confirmation cast (2040 m, different regime)."""
    return _moria_params("MORIA-06")


def moria07_params() -> CastParams:
    """Effective parameters for the MORIA-07 confirmation cast."""
    return _moria_params("MORIA-07")

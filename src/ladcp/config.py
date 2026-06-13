"""Per-cast processing parameters (mirrors legacy ``p``/``ps`` + ``default.m``).

Only the knobs that matter for the current scope are modelled explicitly; the rest can
ride in ``extra``. The MORIA-05 preset encodes the effective values resolved from the
legacy ``set_cast_params.m`` (see docs/VALIDATION_MORIA05.md §3).
"""

from __future__ import annotations

from collections.abc import Callable
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
    # extra down-looker near-field bins to NaN before super-ensembles, for a known
    # fixed-range target hung below the package (e.g. a corer on a short cable). 1-based
    # like edit_mask_dn_bins. ALWAYS EMPTY unless the user opts in explicitly
    # (--nearfield-dn-bins): no cruise preset sets it (user decision 2026-06-12 -- no
    # silent station-specific edits). The nearfield_errvel_ratio QA WARN detects the
    # situation and its note suggests the bins to mask (on the MORIA monocorer block
    # 03-28 excl 07-10 that was bins 3,4 = the 26 m + 34 m cells).
    edit_nearfield_dn_bins: tuple[int, ...] = ()
    # manual brush edits replayed from a journal (ladcp.edits): rectangles
    # (head, bin_first, bin_last, ens_first, ens_last); bins 1-based inclusive per
    # head, ensembles 0-based inclusive in joint-trimmed space. NEVER set by a
    # preset -- only --edits / the Studio brush populate it.
    edit_manual_flags: tuple[tuple[str, int, int, int, int], ...] = ()
    edit_sidelobes: bool = True     # legacy edit_data: drop side-lobe-contaminated cells
                                    # (surface up-reflection + range-dependent seabed wedge)
                                    # from the velocity weights (golden default on)
    dzbelow: float = 16.0           # below-bottom margin [m] (legacy p.dzbelow(1)=2*bin)
    # operator seabed override (legacy p.zbottom / p.guessbottom). zbottom: hard override --
    # take this seabed depth verbatim, skip detect_bottom (fixes a false-lock when the true
    # depth is known). guessbottom: soft seed -- bias the echo-stack search to a window around
    # this depth (steers off a far multiple). NaN = auto-detect (the default).
    zbottom: float = float("nan")
    guessbottom: float = float("nan")
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

    # sound-speed correction of water velocities (legacy p.soundcorr, ON by default in
    # prepinv.m:35). RDI computes Doppler velocity with the single fixed sound speed
    # configured at deployment; when that value is wrong every velocity carries a
    # proportional scale error. We rescale ru/rv/rw per ensemble by
    # c_in-situ(package depth) / c_firmware before super-ensemble formation, mirroring
    # prepinv.m:419-444. c_in-situ comes from the CTD (TEOS-10); with no CTD it falls
    # back to the legacy estimate sounds(P, ADCP_temp, S=34.5). On MORIA-80 the firmware
    # ran a fixed 1450 m/s vs an in-situ median 1492 -> a ~2.9 % scale (legacy audit
    # 2026-06-13: this was the one material un-ported default-ON mechanism).
    soundcorr: bool = True

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
    share one parameter set (see docs/VALIDATION_MORIA05.md §3). The monocorer
    near-field mask is NOT applied here -- masking is always the user's explicit call
    (``--nearfield-dn-bins``); the QA hung-device WARN points at the affected casts
    (MORIA block 03-28 excl 07-10: bins 3,4).
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
    """Effective parameters for the MORIA-05 validation cast.

    Golden-validation preset: legacy never masked the monocorer band, so reproducing
    the golden products requires the near-field mask OFF. The production path
    (:func:`resolve_params`) applies it per station.
    """
    p = _moria_params("MORIA-05")
    p.edit_nearfield_dn_bins = ()
    return p


# --- Cruise registry + layered resolution (#4a) -------------------------------------
#
# A cruise's *processing* knobs (the operator's ``set_cast_params.m`` defaults) are shared
# across its clean dual-head casts; only position/time/declination vary per cast, and those
# are resolved downstream (IGRF declination, CTD position). So a cruise maps to one base
# ``CastParams`` factory, and :func:`resolve_params` layers per-cast facts on top of it:
#
#     cruise preset  ->  station label  ->  (header-derived config)  ->  explicit overrides
#
# Header-derived geometry is applied separately by ``ingest.apply_header_config`` once the
# PD0 files are open; CLI/`overrides` (e.g. an operator-supplied ``drot``) win last.

def _generic_params(station: str, cruise_id: str) -> CastParams:
    """Operator-community defaults for a cruise without a dedicated preset.

    These are the ``set_cast_params.m`` values shared verbatim by every cruise
    seen so far: dual-head, bin-1 masked both heads, dz=8,
    cut=7, pglim=50, elim=0.2, vlim=1, wlim=0.08, tilt 22/4, btrk_mode=3.
    Cruise-specific layers (e.g. MORIA's monocorer near-field mask) belong in
    a dedicated preset; nothing cruise-specific is applied here.
    """
    return CastParams(
        station=station,
        cruise_id=cruise_id,
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
        drot=None,            # IGRF-computed from the cast position
        getdepth=2,
    )


CRUISES: dict[str, Callable[[str], CastParams]] = {  # cruise_id -> base-params factory
    "MORIA": _moria_params,
}


def resolve_params(
    cruise: str,
    station: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> CastParams:
    """Build the per-cast :class:`CastParams` for ``station`` of ``cruise`` (#4a).

    Replaces the previous hard-wired ``moria05_params()`` call: the base processing knobs
    come from the cruise preset, the station label is stamped in, and any ``overrides``
    (CLI flags such as an explicit ``drot``) are applied last. Instrument geometry is *not*
    set here -- it is reconciled from the PD0 headers by ``ingest.apply_header_config`` once
    the files are open, so geometry always reflects the actual data.

    A cruise without a registered preset resolves to :func:`_generic_params` with its
    name stamped as ``cruise_id`` -- so a new cruise processes (and labels its exports)
    correctly out of the box, and a preset is only needed for cruise-specific layers.
    """
    key = cruise.upper()
    if key in CRUISES:
        params = CRUISES[key](station)
    else:
        params = _generic_params(station, key)
    for name, value in (overrides or {}).items():
        if not hasattr(params, name):
            raise AttributeError(f"CastParams has no field {name!r}")
        setattr(params, name, value)
    return params

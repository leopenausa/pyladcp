"""Reader for the GO-SHIP golden LADCP NetCDF (LDEO_IX IX_13beta output).

This is the validation target for the clean-slate rebuild (see ``docs/PLAN_REBUILD.md``).
Unlike the MORIA ``.lad``/``.bot``/``.log`` ASCII triple (see :mod:`ladcp.io.golden`),
each RB1606/P18 station ships a single ``NNN.nc`` whose attributes embed BOTH the full
LDEO_IX parameter struct AND a step-by-step processing log. That lets us validate every
intermediate stage of our pipeline, not just final ``u``/``v``.

Two things are exposed:

* :func:`read_golden_nc` -> :class:`GoldenNC`  — gridded profile + kinematics arrays,
  plus ``.params`` (every global attr) and ``.log`` (the parsed step log).
* :func:`expected_values` -> dict — the compact per-station "expected" table the plan
  asks for (lag, bottom depth, n super-ensembles, velerr, per-step removed counts,
  CHECKINV residuals), drawn from the robust scalar attrs where available and from the
  log text otherwise.

Provenance of the numbers (so future readers know which source wins):
- Scalars that exist as attrs (``velerr``, ``zbottom``, ``nt``, declination, ...) are
  read from attrs — they are the authoritative floats, not regex-rounded log strings.
- Counts that only appear in prose (best-lag, removed-below-bottom, outlier bins per
  step, inversion matrix lengths, CHECKINV rms) are scraped from the log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr


# --------------------------------------------------------------------------- #
# Profile container
# --------------------------------------------------------------------------- #
@dataclass
class GoldenNC:
    """Parsed golden NetCDF profile for one station.

    Arrays keep the file's native sign/unit conventions: ``u`` east, ``v`` north
    (m/s); ``z``/``p`` positive-down (m / dbar). ``params`` is the verbatim global-attr
    dict (the LDEO_IX ``p``/``ps`` struct, flattened). ``log`` is the scraped step log.
    """

    station: str
    # primary solution [z]
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    uerr: np.ndarray
    p: np.ndarray
    nvel: np.ndarray
    # depth means (scalars)
    ubar: float
    vbar: float
    # identity
    lat: float
    lon: float
    date: np.datetime64 | None
    # every other gridded var, keyed by name (zbot/sadcp/tim/shear/acoustic/ctd_*)
    vars: dict[str, np.ndarray] = field(default_factory=dict)
    # full global-attr struct + parsed log
    params: dict[str, Any] = field(default_factory=dict)
    log: GoldenLog | None = None

    @property
    def n_superensembles(self) -> int:
        return int(self.vars["tim_hour"].size) if "tim_hour" in self.vars else 0


def read_golden_nc(path: str) -> GoldenNC:
    """Load a golden ``NNN.nc`` into a :class:`GoldenNC`."""
    ds = xr.open_dataset(path)
    try:
        params = {k: _attr_value(v) for k, v in ds.attrs.items()}
        station = str(params.get("ladcp_station", "")).strip() or _station_from_path(path)

        date = None
        if "time_start" in params:
            ts = np.asarray(params["time_start"], dtype=int)  # [Y M D h m s]
            if ts.size >= 6:
                date = np.datetime64(
                    f"{ts[0]:04d}-{ts[1]:02d}-{ts[2]:02d}T"
                    f"{ts[3]:02d}:{ts[4]:02d}:{ts[5]:02d}"
                )

        # every data var AND non-primary coordinate (zbot, z_sadcp, tim are coords),
        # pulled as plain ndarrays
        all_vars = {name: ds[name].values for name in ds.data_vars}
        coord_vars = {name: ds[name].values for name in ds.coords}
        primary = {"u", "v", "uerr", "p", "nvel", "z"}
        extra = {k: v for k, v in {**coord_vars, **all_vars}.items() if k not in primary}

        g = GoldenNC(
            station=str(station),
            z=np.asarray(ds["z"].values, dtype=float),
            u=np.asarray(all_vars["u"], dtype=float),
            v=np.asarray(all_vars["v"], dtype=float),
            uerr=np.asarray(all_vars["uerr"], dtype=float),
            p=np.asarray(all_vars["p"], dtype=float),
            nvel=np.asarray(all_vars["nvel"]),
            ubar=float(all_vars["ubar"]),
            vbar=float(all_vars["vbar"]),
            lat=float(params.get("lat", np.nan)),
            lon=float(params.get("lon", np.nan)),
            date=date,
            vars=extra,
            params=params,
            log=scrape_log(str(params.get("LOG_Inverse_log", ""))),
        )
        return g
    finally:
        ds.close()


def _attr_value(v: Any) -> Any:
    """Normalise an xarray attr into a plain python / numpy value."""
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size > 1 else (v.item() if v.size == 1 else [])
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _station_from_path(path: str) -> str:
    m = re.search(r"(\d{3})\.nc$", path)
    return m.group(1) if m else path


# --------------------------------------------------------------------------- #
# Step-by-step log scraper
# --------------------------------------------------------------------------- #
@dataclass
class GoldenLog:
    """Structured view of the LDEO_IX ``LOG_Inverse_log`` attribute.

    ``steps`` maps step-number -> raw section text. ``values`` is the flat dict of
    scalars/counts pulled out of the prose. ``raw`` keeps the whole string.
    """

    raw: str
    steps: dict[int, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)


_STEP_HDR = re.compile(r"#+\s*\[[^\]]*\]\s*step\s+(\d+):", re.IGNORECASE)


def _split_steps(text: str) -> dict[int, str]:
    """Split the log into ``{step_number: section_text}`` using the banner headers."""
    matches = list(_STEP_HDR.finditer(text))
    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[int(m.group(1))] = text[m.start():end]
    return out


def _f(pat: str, s: str, group: int = 1):
    m = re.search(pat, s)
    return float(m.group(group)) if m else None


def _i(pat: str, s: str, group: int = 1):
    m = re.search(pat, s)
    return int(m.group(group)) if m else None


def scrape_log(text: str) -> GoldenLog:
    """Extract per-step checkpoints from the LDEO_IX processing log.

    Each section is matched in isolation so identical phrases in step 10 (form
    super-ensembles) and step 12 (re-form) do not collide. Missing keys mean the line
    was not present (e.g. SADCP rms is absent when SADCP was unused) — left out rather
    than set to NaN, so the absence is detectable.
    """
    log = GoldenLog(raw=text, steps=_split_steps(text))
    s = log.steps
    v = log.values

    # --- step 3: nav + declination --------------------------------------- #
    if 3 in s:
        sec = s[3]
        v["nav_scans"] = _i(r"number of NAV scans:\s*(\d+)", sec)
        v["nav_delta_t"] = _f(r"delta t :\s*([\d.]+)", sec)
        v["declination"] = _f(r"magnetic declination of\s*([-\d.]+)", sec)

    # --- step 4: get bottom-track ---------------------------------------- #
    if 4 in s:
        sec = s[4]
        v["btrk_localmax_valid"] = _i(r"localmax2: found\s*(\d+)\s*valid", sec)
        v["btrk_below_range"] = _i(r"found\s*(\d+)\s*bottom depth below btrk_range", sec)
        v["btrk_distances_created"] = _i(r"created\s*(\d+)\s*bottom distances", sec)
        v["btrk_profiles_removed_wdiff"] = _i(
            r"removed\s*(\d+)\s*bottom track profiles", sec)
        v["btrk_boutlier_removed"] = _i(r"boutlier removed\s*(\d+)", sec)
        v["btrk_data_created"] = _i(r"created\s*(\d+)\s*bottom track data", sec)

    # --- step 6: load CTD time series + best lag ------------------------- #
    if 6 in s:
        sec = s[6]
        v["ctd_scans_read"] = _i(r"read\s*(\d+)\s*CTD scans", sec)
        v["ctd_scans_interp"] = _i(r"interpolated to\s*(\d+)\s*CTD scans", sec)
        v["ctd_pressure_spikes_removed"] = _i(r"removed\s*(\d+)\s*pressure spikes", sec)
        v["ctd_max_depth"] = _f(r"CTD max depth\s*:\s*([\d.]+)", sec)
        v["best_lag_scans"] = _i(r"best lag W:\s*(-?\d+)\s*CTD scans", sec)
        v["besttlag"] = _i(r"BESTTLAG:\s*lag is:\s*(-?\d+)", sec)

    # --- step 7: surface/seabed/depth ------------------------------------ #
    if 7 in s:
        sec = s[7]
        v["maxdepth_downtrace"] = _f(r"maxdepth form down-trace is\s*([\d.]+)", sec)
        v["maxdepth_uptrace"] = _f(r"maxdepth from up-trace\s+is\s*([\d.]+)", sec)
        m = re.search(r"bottom found at\s*(\d+)\s*\+/-\s*(\d+)", sec)
        if m:
            v["bottom_found_depth"] = int(m.group(1))
            v["bottom_found_error"] = int(m.group(2))
        v["values_removed_below_bottom"] = _i(
            r"removing\s*(\d+)\s*values\s+below recognized bottom", sec)
        m = re.search(r"LADCP minus CTD depth mean:\s*([-\d.]+)\s*std:\s*([\d.]+)", sec)
        if m:
            v["ladcp_ctd_depth_mean"] = float(m.group(1))
            v["ladcp_ctd_depth_std"] = float(m.group(2))

    # --- step 9: edit data ----------------------------------------------- #
    if 9 in s:
        sec = s[9]
        v["edit_bin_mask_nan"] = _i(r"set\s*(\d+)\s*weights to NaN", sec)
        v["edit_sidelobe_nan"] = _i(
            r"side-lobe contamination\s*:\s*set\s*(\d+)\s*weights", sec)

    # --- steps 10 & 12: (re-)form super ensembles ------------------------ #
    for step, prefix in ((10, "se10"), (12, "se12")):
        if step not in s:
            continue
        sec = s[step]
        v[f"{prefix}_bt_discarded_hab"] = _i(
            r"discarded\s*(\d+)\s*bottom tracks velocities because of height", sec)
        v[f"{prefix}_outlier_down"] = _i(
            r"Outlier discarded\s*(\d+)\s*bins down", sec)
        v[f"{prefix}_outlier_up"] = _i(r"Outlier discarded\s*(\d+)\s*bins up", sec)
        v[f"{prefix}_bt_finite_ens"] = _i(
            r"found\s*(\d+)\s*finite bottom track ensembles", sec)
        v[f"{prefix}_reduced_len"] = _i(
            r"reduced profile length =\s*(\d+)\s*super-ensemble", sec)

    # --- step 13: SADCP --------------------------------------------------- #
    if 13 in s:
        v["sadcp_profiles_found"] = _i(r"found\s*(\d+)\s*SADCP profiles", s[13])

    # --- step 14: inverse solution + CHECKINV ---------------------------- #
    if 14 in s:
        sec = s[14]
        v["inv_barotropic_vel_err"] = _f(r"Barotropic velocity error\s*([\d.]+)", sec)
        v["inv_superens_vel_err"] = _f(r"super ensemble velocity error\s*([\d.]+)", sec)
        v["inv_velocity_error_set"] = _f(r"set velocity error to:\s*([\d.]+)", sec)
        v["inv_constraints_removed_lowweight"] = _i(
            r"remove\s*(\d+)\s*constaints below minimum weight", sec)
        v["inv_len_d"] = _i(r"length of\s+d:\s*(\d+)", sec)
        v["inv_len_A1"] = _i(r"length of A1:\s*(\d+)", sec)
        v["inv_len_A2"] = _i(r"length of A2:\s*(\d+)", sec)
        # CHECKINV residuals (value, threshold)
        m = re.search(r"Velocity profile error:\s*([\d.]+)\s*should be about noise:\s*([\d.]+)", sec)
        if m:
            v["checkinv_vel_err"] = float(m.group(1))
            v["checkinv_vel_noise"] = float(m.group(2))
        v["checkinv_btrk_rms"] = _f(r"Check bottom track rms:\s*([\d.]+)", sec)
        v["checkinv_sadcp_rms"] = _f(r"Check SADCP\s+rms:\s*([\d.]+)", sec)
        v["checkinv_gps_diff"] = _f(r"GPS-LADCP ship spd diff:\s*([\d.]+)", sec)

    return log


# --------------------------------------------------------------------------- #
# Compact per-station expected-values table
# --------------------------------------------------------------------------- #
def expected_values(g: GoldenNC) -> dict[str, Any]:
    """Compact per-station "expected" table for stage-by-stage validation.

    Combines authoritative scalar attrs with log-only counts. Keys are stable so a
    validation harness can diff a freshly computed pipeline against them.
    """
    p = g.params
    lv = (g.log.values if g.log else {})
    exp: dict[str, Any] = {
        "station": g.station,
        "lat": g.lat,
        "lon": g.lon,
        "date": str(g.date) if g.date is not None else None,
        # config
        "software": p.get("software"),
        "dz": p.get("dz"),
        "btrk_mode": p.get("btrk_mode"),
        "up_dn_looker": p.get("up_dn_looker"),
        "reference": p.get("BAR_ref_descr"),
        "sadcp_avail": p.get("INPUT_SADCP_profile_avail"),
        # magnetics
        "declination_deg": p.get("GEN_Magnetic_deviation_deg", p.get("drot")),
        # depth / geometry
        "bottom_depth_m": p.get("zbottom"),
        "bottom_depth_err_m": p.get("zbottomerror"),
        "max_depth_m": p.get("maxdepth"),
        "n_raw_ensembles": p.get("nt"),
        "n_superensembles": g.n_superensembles,
        "n_z_bins": int(g.z.size),
        # CTD lag
        "ctd_time_offset_s": p.get("ctdtimoff"),
        "best_lag_scans": lv.get("best_lag_scans"),
        # inversion error scales
        "velerr": p.get("velerr"),
        "barvelerr": p.get("barvelerr"),
        "superens_std_min": p.get("superens_std_min"),
        # bottom-track residuals (attrs are the authoritative floats)
        "btrk_u_bias": p.get("btrk_u_bias"),
        "btrk_u_std": p.get("btrk_u_std"),
        "btrk_v_bias": p.get("btrk_v_bias"),
        "btrk_v_std": p.get("btrk_v_std"),
        # depth-mean solution
        "ubar": g.ubar,
        "vbar": g.vbar,
    }
    # fold in the full set of log-scraped checkpoints under a namespace
    exp["log"] = dict(lv)
    return exp

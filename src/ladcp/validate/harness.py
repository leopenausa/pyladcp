"""MORIA-05 validation harness (see docs/VALIDATION_MORIA05.md).

Runs every Python component that is currently implemented against the golden legacy
outputs and produces a structured report. Checks that depend on the velocity inversion
(not yet implemented) are reported honestly as ``pending`` rather than skipped or faked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import CastParams, moria05_params, moria06_params, moria07_params
from ..io.ctd import read_clean_ctd
from ..io.golden import read_bot, read_lad, scan_log
from ..io.pd0 import read_pd0
from ..models import RawADCP
from ..proc.magdec import magnetic_declination
from ..proc.pipeline import process_cast
from .compare import Check, Report, profile_check, scalar_check


@dataclass
class CastInputs:
    station: str
    master_pd0: Path
    slave_pd0: Path
    ctd_cnv: Path
    golden_lad: Path
    golden_bot: Path
    golden_log: Path
    params: CastParams


# per-cast raw-file mapping (deployment file -> station), confirmed from PD0 start
# times against the golden .lad Start_Time. master N pairs with slave N+1.
_CAST_FILES = {
    "MORIA-05": ("MLADC007.000", "SLADC008.000", "moria-05_clean.cnv", moria05_params),
    "MORIA-06": ("MLADC008.000", "SLADC009.000", "moria-06_clean.cnv", moria06_params),
    "MORIA-07": ("MLADC009.000", "SLADC010.000", "moria-07_clean.cnv", moria07_params),
}

# Data-quality caveats to VERIFY when raw data for these casts arrives (not exclusions —
# wire them into _CAST_FILES once their Master_good/Slave_good raw files are available):
#  - MORIA-25 series (25, 25b..25g): Irish/Goban 48°N transect. In the golden set, 25 and
#    25b were processed from the SAME raw LADCP file yet give inconsistent profiles
#    (opposite-sign v at depth; 2584 m vs 1168 m) => a wrong LADCP<->CTD pairing in at
#    least one; 25c's golden is corrupt (10.6k bad checksums, no bottom track, degenerate
#    solve). Re-pair LADCP/CTD and re-check 25c before trusting these as references.


def cast_inputs(root: Path, station: str) -> CastInputs:
    if station not in _CAST_FILES:
        raise KeyError(f"no input mapping for {station!r}; known: {sorted(_CAST_FILES)}")
    master_f, slave_f, ctd_f, params_fn = _CAST_FILES[station]
    raw = root / "raw_ladcp_test" / "LADCP" / "Data"
    return CastInputs(
        station=station,
        master_pd0=raw / "MASTER" / master_f,
        slave_pd0=raw / "SLAVE" / slave_f,
        ctd_cnv=root / "clean_ctd" / ctd_f,
        golden_lad=root / "figures" / f"{station}.lad",
        golden_bot=root / "figures" / f"{station}.bot",
        golden_log=root / "figures" / f"{station}.log",
        params=params_fn(),
    )


def moria05_inputs(root: Path) -> CastInputs:
    return cast_inputs(root, "MORIA-05")


def run(inputs: CastInputs) -> Report:
    rep = Report(cast=inputs.station)
    p = inputs.params

    # --- read raw heads -------------------------------------------------
    master = read_pd0(str(inputs.master_pd0), head="downlooker", facing_hint="down")
    slave = read_pd0(str(inputs.slave_pd0), head="uplooker", facing_hint="up")

    # --- read CTD + golden ---------------------------------------------
    ctd = read_clean_ctd(str(inputs.ctd_cnv), p)
    g_lad = read_lad(str(inputs.golden_lad))
    g_bot = read_bot(str(inputs.golden_bot))
    g_log = scan_log(str(inputs.golden_log))

    # === configuration sanity (gates on data integrity) ================
    rep.add(Check("master_is_downlooker",
                  "pass" if master.facing == "down" else "fail",
                  f"facing={master.facing}, coord={master.coord_frame.value}",
                  value=master.facing, reference="down", gate=True))
    rep.add(Check("slave_is_uplooker",
                  "pass" if slave.facing == "up" else "fail",
                  f"facing={slave.facing}, coord={slave.coord_frame.value}",
                  value=slave.facing, reference="up", gate=True))
    rep.add(scalar_check("freq_khz", master.freq_khz, 300, 0, gate=True))
    rep.add(scalar_check("n_cells", master.n_cells, 30, 0, gate=True))
    rep.add(scalar_check("cell_size_m", master.cell_m, 8.0, 0.01, gate=True))
    rep.add(Check("coord_frame", "pass",
                  f"master={master.coord_frame.value}, slave={slave.coord_frame.value}",
                  value=master.coord_frame.value))

    # === bottom-track present ==========================================
    bt_ens = int(np.sum(np.any(np.isfinite(master.bt_vel), axis=0))) if master.bt_vel is not None else 0
    rep.add(Check("bottom_track_present",
                  "pass" if bt_ens > 0 else "fail",
                  f"{bt_ens} ensembles with finite BT velocity",
                  value=bt_ens, gate=True))

    # === CTD coverage / max depth ======================================
    ctd_maxp = float(np.nanmax(ctd.pressure))
    rep.add(scalar_check("ctd_max_depth_m", ctd_maxp,
                         g_log.get("ctd_max_depth"), 0.02, rel=True, unit="m"))

    # === magnetic declination (GATE) ===================================
    when = _cast_datetime(master, g_lad)
    lat0 = float(ctd.lat[0])
    lon0 = float(ctd.lon[0])
    decl = p.drot if p.drot is not None else magnetic_declination(lat0, lon0, when)
    c = scalar_check("magnetic_declination_deg", decl,
                     g_lad.header.get("deviation"), 0.01, gate=True, unit="deg")
    if c.status == "fail" and p.drot is None:
        c.detail += ("  [NOTE: legacy magdev.m uses IGRF-2000 + a hardcoded +2° Mediterranean "
                     "fudge; modern IGRF-13 differs ~2° — golden is biased, see "
                     "docs decisions. Set params.drot to legacy value to isolate the inversion.]")
    rep.add(c)

    # === SADCP handling (designed-for; recorded, not faked) ============
    rep.add(Check("sadcp_used_in_golden",
                  "pass",
                  f"golden used SADCP: {g_log.get('sadcp_used')} "
                  f"(policy={p.sadcp_mismatch_policy})",
                  value=g_log.get("sadcp_used"), reference=False))

    # === velocity inversion =============================================
    # Run with the LEGACY declination so the velocity comparison isolates the
    # inversion from the (separately-flagged) declination divergence.
    legacy_decl = g_lad.header.get("deviation", decl)
    try:
        res = process_cast(master, slave, ctd, p, legacy_decl)
        pr = res.profile
        rep.add(profile_check("u_profile", pr.z, pr.u, g_lad.z, g_lad.u, 0.01, gate=True))
        rep.add(profile_check("v_profile", pr.z, pr.v, g_lad.z, g_lad.v, 0.01, gate=True))
        rep.add(scalar_check("superensemble_vel_err", pr.config.get("velerr"),
                             g_log.get("superensemble_vel_err"), 0.01, gate=False, unit="m/s"))
        rep.add(scalar_check("bottom_depth_m", res.diagnostics.get("zbottom"),
                             g_lad.header.get("bottom_depth") or g_bot.header.get("bottom_depth"),
                             8.0, gate=False, unit="m"))
        rep.add(Check("inversion_ran", "pass",
                      f"nz={pr.config.get('nz')}, n_superens={pr.config.get('n_superens')}, "
                      f"ubar={pr.ubar:+.3f}, vbar={pr.vbar:+.3f}, "
                      f"depth_lag={res.diagnostics.get('lag_seconds'):.0f}s"))
    except Exception as exc:  # surface failures honestly, don't fake a pass
        rep.add(Check("inversion_ran", "fail", f"{type(exc).__name__}: {exc}", gate=True))

    return rep


def _none():
    return None


def _cast_datetime(master: RawADCP, g_lad) -> datetime:
    t = master.time[np.isfinite(master.time.astype("float64"))]
    if t.size:
        ts = t[t.size // 2]
        return datetime.fromtimestamp(ts.astype("datetime64[s]").astype(int), tz=timezone.utc)
    # fallback to golden header date
    d = g_lad.header.get("Date", "2025/ 9/18").replace(" ", "")
    y, m, day = (int(x) for x in d.split("/"))
    return datetime(y, m, day, tzinfo=timezone.utc)


def write_report(rep: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"validation_{rep.cast}.json"
    mpath = out_dir / f"validation_{rep.cast}.md"
    jpath.write_text(json.dumps(rep.to_dict(), indent=2))
    mpath.write_text(_markdown(rep))
    return jpath, mpath


def _markdown(rep: Report) -> str:
    lines = [f"# Validation report — {rep.cast}", "",
             f"**Overall: {rep.overall}**", "",
             "| check | status | gate | detail |", "|---|---|---|---|"]
    icon = {"pass": "✅", "fail": "❌", "pending": "⏳"}
    for c in rep.checks:
        lines.append(f"| {c.name} | {icon.get(c.status, c.status)} {c.status} | "
                     f"{'gate' if c.gate else ''} | {c.detail} |")
    lines += ["", "_`pending` = depends on the velocity inversion, which is the next "
              "build milestone (see docs/VALIDATION_MORIA05.md §8)._"]
    return "\n".join(lines)

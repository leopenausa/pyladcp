"""Cross-station validation harness for the velocity solvers (#2/#5).

The four reprocessed LDEO_IX goldens form a regime ladder we validate the inverse
against. Each was run with the same param set (``ps.shear==1``, ``dz=8``,
``weightmin=0.05``, ``smoofac=0``, ``botfac=barofac=1``, ``outlier=1``,
``offsetup2down=1``); they differ in depth, region and which reference constraints were
active:

==============  =======  ====================  ===============  =====
station         depth m   region                reference        SADCP
==============  =======  ====================  ===============  =====
MORIA-80         1073     N Atlantic            Nav + Bottom      no
MORIA-10          549     Gulf of Biscay        Nav + Bottom      no
FDCCC1_001        125     W Med (Cap de Creus)  Nav + Bottom      no [1]
FDCCC1_002        225     W Med                 Nav + Bottom + S  YES
==============  =======  ====================  ===============  =====

[1] FDCCC1_001 *requested* SADCP (``p.sadcp=75``, ``ps.sadcpfac=3``) but the SADCP file
    was missing at runtime -- its log reports "no SADCP data", so only Nav + Bottom were
    used. FDCCC1_002 is the only golden that truly exercised the SADCP constraint
    (9 profiles survived time/space matching; ``CHECKSADCP rms 0.020``).

This module reads the golden ``dr`` velocity targets straight from each ``.mat`` (the
single source of truth, never hardcoded) and scores a computed profile against them.
The SADCP constraint rows the golden actually used are embedded in ``dr.z_sadcp`` /
``u_sadcp`` / ``v_sadcp`` / ``uerr_sadcp`` -- so the ``lainsadcp`` block can be validated
against a real golden even before the raw FDCCC1 inputs are available. The companion
:data:`STATIONS` registry records, per station, whether the *raw* inputs are on disk
(only then can the full pipeline run end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# Repo root (.../LADCP_project), three parents up from src/ladcp/qa/.
_REPO = Path(__file__).resolve().parents[3]
_GOLDEN = _REPO / "New_golden"                          # local-only full goldens
_FIXGOLDEN = _REPO / "tests" / "fixtures" / "New_golden"  # committed (CI-visible)


@dataclass(frozen=True)
class Station:
    """One golden cast in the validation ladder."""

    name: str
    mat: Path                       # golden <name>.mat (dr/p/ps/f)
    depth_m: float
    region: str
    uses_sadcp: bool                # SADCP constraint active in the golden run
    raw_down: Path | None = None    # raw down-looker PD0, if local
    raw_up: Path | None = None      # raw up-looker PD0, if local
    ctd: Path | None = None         # cleaned CTD .cnv, if local
    drot: float | None = None       # golden p.drot, when known (NaN in FDCCC1/MORIA-10)

    @property
    def has_raw(self) -> bool:
        """True only when the full pipeline can run from raw on this machine."""
        return bool(self.raw_down and self.raw_up and self.ctd
                    and self.raw_down.exists() and self.raw_up.exists()
                    and self.ctd.exists())


# MORIA-80 raw lives in the committed test fixture; MORIA-10 raw is the VmDAS archive under
# raw_ladcp_test/ (master MLADC012 + slave SLADC013, paired by time via discovery -- the
# deployment indices are offset by one); FDCCC1 raw is not on disk (pending). The MORIA-10
# golden's p.drot=-3.5687 is the legacy IGRF00 value (true IGRF-13 here is ~-1.25); we score
# against the golden frame, so we use its drot to isolate solver quality -- see
# ladcp-magdec-finding.
_RAW = _REPO / "raw_ladcp_test" / "LADCP" / "Data"
_FIX = _FIXGOLDEN / "Good"
STATIONS: dict[str, Station] = {
    "MORIA-80": Station(
        name="MORIA-80",
        mat=_FIXGOLDEN / "MORIA-80_LEO" / "MORIA-80.mat",   # committed fixture
        depth_m=1073.0, region="N Atlantic", uses_sadcp=False,
        raw_down=_FIX / "LADCP" / "MORIA-80-LADCP-M.000",
        raw_up=_FIX / "LADCP" / "MORIA-80-LADCP-S.000",
        ctd=_FIX / "CTD" / "moria-80_clean.cnv",
        drot=-9.878379,
    ),
    "MORIA-10": Station(
        name="MORIA-10",
        mat=_GOLDEN / "MORIA-10_LEO" / "MORIA-10.mat",
        depth_m=549.0, region="Gulf of Biscay", uses_sadcp=False,
        raw_down=_RAW / "MASTER" / "MLADC012.000",
        raw_up=_RAW / "SLAVE" / "SLADC013.000",
        ctd=_REPO / "clean_ctd" / "moria-10_clean.cnv",
        drot=-3.568734,
    ),
    "FDCCC1_001": Station(
        name="FDCCC1_001",
        mat=_GOLDEN / "FDCCC1_001_LEO" / "FDCCC1_001.mat",
        depth_m=125.0, region="W Med (Cap de Creus)", uses_sadcp=False,
        raw_down=_REPO / "FDCCC1" / "MA001000.000",     # beam coords -> b2earth at ingest
        raw_up=_REPO / "FDCCC1" / "SL001000.000",
        ctd=_REPO / "FDCCC1" / "FDCCC1_001.cnv",
    ),
    "FDCCC1_002": Station(
        name="FDCCC1_002",
        mat=_GOLDEN / "FDCCC1_002_LEO" / "FDCCC1_002.mat",
        depth_m=225.0, region="W Med (Cap de Creus)", uses_sadcp=True,
        raw_down=_REPO / "FDCCC1" / "MA002000.000",
        raw_up=_REPO / "FDCCC1" / "SL002000.000",
        ctd=_REPO / "FDCCC1" / "FDCCC1_002.cnv",
    ),
}


@dataclass(frozen=True)
class DrTargets:
    """Golden velocity targets read from a station ``dr`` struct.

    All depths positive-down [m]; velocities true east/north [m/s]. SADCP fields are
    empty (size <= 1, all-NaN) for casts that ran without the SADCP constraint.
    """

    name: str
    z: np.ndarray               # [nz] output depth grid
    u: np.ndarray               # [nz] absolute east velocity (dr.u)
    v: np.ndarray               # [nz] absolute north velocity (dr.v)
    uerr: np.ndarray            # [nz] velocity uncertainty (dr.uerr)
    nvel: np.ndarray            # [nz] samples per bin (dr.nvel)
    ubar: float                 # barotropic east reference (dr.ubar)
    vbar: float                 # barotropic north reference (dr.vbar)
    u_shear: np.ndarray         # [nz] shear-method baroclinic east (dr.u_shear_method)
    v_shear: np.ndarray         # [nz] shear-method baroclinic north (dr.v_shear_method)
    zbot: np.ndarray            # [nb] bottom-track depth grid (dr.zbot)
    ubot: np.ndarray            # [nb] bottom-track east (dr.ubot)
    vbot: np.ndarray            # [nb] bottom-track north (dr.vbot)
    z_sadcp: np.ndarray         # [ns] SADCP constraint depths actually used
    u_sadcp: np.ndarray         # [ns] SADCP east used by the inverse
    v_sadcp: np.ndarray         # [ns] SADCP north used by the inverse
    uerr_sadcp: np.ndarray      # [ns] SADCP weight/uncertainty

    @property
    def has_sadcp(self) -> bool:
        """True when the golden carries a real (>1 finite) SADCP constraint."""
        z = self.z_sadcp
        return z.size > 1 and np.isfinite(z).sum() > 1


def _arr(obj, field: str) -> np.ndarray:
    """Return ``obj.field`` as a 1-D float array (empty if absent)."""
    if field not in getattr(obj, "_fieldnames", []):
        return np.asarray([], dtype=float)
    return np.atleast_1d(np.asarray(getattr(obj, field), dtype=float)).ravel()


@lru_cache(maxsize=8)
def load_dr(station: str) -> DrTargets:
    """Load the golden ``dr`` velocity targets for ``station`` from its ``.mat``."""
    return load_dr_mat(STATIONS[station].mat, station)


def load_dr_mat(path: str | Path, name: str = "") -> DrTargets:
    """Load ``dr`` velocity targets from any legacy LDEO result ``.mat``.

    The registry-free entry point: :func:`load_dr` serves the golden-ladder
    stations; this serves arbitrary legacy cruise products (e.g. a whole
    ``LADCP_processed/`` directory being compared by ``ladcp-compare``).
    """
    import scipy.io as sio

    dr = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False,
                     variable_names=["dr"])["dr"]
    station = name or Path(path).stem

    def scalar(f: str) -> float:
        return float(getattr(dr, f)) if f in dr._fieldnames else float("nan")

    return DrTargets(
        name=station,
        z=_arr(dr, "z"), u=_arr(dr, "u"), v=_arr(dr, "v"),
        uerr=_arr(dr, "uerr"), nvel=_arr(dr, "nvel"),
        ubar=scalar("ubar"), vbar=scalar("vbar"),
        u_shear=_arr(dr, "u_shear_method"), v_shear=_arr(dr, "v_shear_method"),
        zbot=_arr(dr, "zbot"), ubot=_arr(dr, "ubot"), vbot=_arr(dr, "vbot"),
        z_sadcp=_arr(dr, "z_sadcp"), u_sadcp=_arr(dr, "u_sadcp"),
        v_sadcp=_arr(dr, "v_sadcp"), uerr_sadcp=_arr(dr, "uerr_sadcp"),
    )


@dataclass(frozen=True)
class Score:
    """Agreement of a computed component against its golden target."""

    corr: float                 # Pearson correlation over jointly-finite samples
    rms: float                  # RMS difference [m/s]
    bias: float                 # mean(computed - golden) [m/s]
    n: int                      # number of compared samples

    def __str__(self) -> str:
        return f"corr={self.corr:.3f} rms={self.rms:.4f} bias={self.bias:+.4f} n={self.n}"


def score(computed: np.ndarray, golden: np.ndarray) -> Score:
    """Correlation / RMS / bias of ``computed`` vs ``golden`` on shared finite samples."""
    a = np.asarray(computed, dtype=float)
    b = np.asarray(golden, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 2:
        return Score(corr=float("nan"), rms=float("nan"), bias=float("nan"), n=n)
    da, db = a[ok], b[ok]
    corr = float(np.corrcoef(da, db)[0, 1])
    rms = float(np.sqrt(np.mean((da - db) ** 2)))
    bias = float(np.mean(da - db))
    return Score(corr=corr, rms=rms, bias=bias, n=n)


def score_profile(z: np.ndarray, u: np.ndarray, v: np.ndarray,
                  dr: DrTargets) -> dict[str, Score]:
    """Score a computed (z, u, v) profile against the golden ``dr.u`` / ``dr.v``.

    The computed profile is interpolated onto the golden depth grid first, so solvers
    that build a slightly different grid still compare fairly.
    """
    ug = np.interp(dr.z, z, u, left=np.nan, right=np.nan)
    vg = np.interp(dr.z, z, v, left=np.nan, right=np.nan)
    return {"u": score(ug, dr.u), "v": score(vg, dr.v)}

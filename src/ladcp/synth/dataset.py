"""Orchestrator: assemble + write a synthetic dual-head station in the discovery layout.

``generate_station`` writes ``<out>/LADCP/<station>-LADCP-{M,S}.000`` and
``<out>/CTD/<station>_clean.cnv`` so that ``ladcp-qa <station> --root <out>`` discovers
and processes it. It returns the file paths and the :class:`OceanTruth` so the recovery
test can score the solved profile against the known input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.pd0_write import write_head_pd0
from .forward import synth_head
from .profile import OceanTruth, cast_trajectory, ctd_series, ocean_truth


@dataclass
class SynthConfig:
    """Parameters for one synthetic station (sane MORIA-like defaults)."""

    out: str | Path = "synthetic_station"
    station: str = "SYNTH-01"
    seed: int = 0
    noise: float = 0.0                 # m/s velocity noise (0 = clean recovery target)
    # ocean truth
    depth: float = 1080.0              # package max depth [m] (deepest the cast reaches)
    seabed: float = 1100.0            # seabed / water depth [m]; cast stops ~20 m above it
    u0: float = -0.06                 # barotropic east [m/s]
    v0: float = 0.03                  # barotropic north [m/s]
    shear_amp: float = 0.18           # thermocline shear amplitude [m/s]
    shear_scale_m: float = 140.0      # thermocline e-folding width [m]
    # geometry -- 30 bins x 8 m reaches ~250 m range, enough for the bottom-track support
    # guard to confirm a seabed lock over the geometrically visible package-depth span
    freq_khz: int = 150
    n_cells: int = 30
    cell_m: float = 8.0
    blank_m: float = 4.0
    dist_first_m: float = 12.0
    beam_angle_deg: int = 20
    # position + timing
    lat: float = 62.16
    lon: float = -11.53
    ping_dt: float = 1.2              # ~1 Hz; near the CTD rate keeps the clock sync tight
    descent_mps: float = 0.8
    ascent_mps: float = 0.8
    surface_soak_s: float = 60.0
    t0_iso: str = "2025-06-01T08:00:00"


@dataclass
class SynthPaths:
    root: Path
    down: Path
    up: Path
    ctd: Path
    extra: dict = field(default_factory=dict)


def generate_station(cfg: SynthConfig) -> tuple[SynthPaths, OceanTruth]:
    """Generate and write one synthetic dual-head station; return paths + the truth."""
    # truth is defined over the full water column (to the seabed); the cast profiles to cfg.depth
    truth = ocean_truth(depth=cfg.seabed, u0=cfg.u0, v0=cfg.v0,
                        shear_amp=cfg.shear_amp, shear_scale_m=cfg.shear_scale_m)
    traj = cast_trajectory(depth=cfg.depth, descent_mps=cfg.descent_mps,
                           ascent_mps=cfg.ascent_mps, ping_dt=cfg.ping_dt,
                           surface_soak_s=cfg.surface_soak_s)
    t0 = np.datetime64(cfg.t0_iso)

    geom = dict(freq_khz=cfg.freq_khz, n_cells=cfg.n_cells, cell_m=cfg.cell_m,
                blank_m=cfg.blank_m, dist_first_m=cfg.dist_first_m,
                beam_angle_deg=cfg.beam_angle_deg, seed=cfg.seed, noise=cfg.noise)
    down = synth_head(truth, traj, facing="down", t0=t0, seabed=cfg.seabed, **geom)
    up = synth_head(truth, traj, facing="up", t0=t0, seabed=cfg.seabed,
                    compass_offset=0.5, **geom)
    ctd = ctd_series(traj, lat0=cfg.lat, lon0=cfg.lon, seed=cfg.seed, noise=cfg.noise)

    root = Path(cfg.out)
    ladcp_dir = root / "LADCP"
    ctd_dir = root / "CTD"
    ladcp_dir.mkdir(parents=True, exist_ok=True)
    ctd_dir.mkdir(parents=True, exist_ok=True)

    down_path = ladcp_dir / f"{cfg.station}-LADCP-M.000"
    up_path = ladcp_dir / f"{cfg.station}-LADCP-S.000"
    ctd_path = ctd_dir / f"{cfg.station}_clean.cnv"

    write_head_pd0(str(down_path), down)
    write_head_pd0(str(up_path), up)
    _write_cnv(ctd_path, ctd)

    paths = SynthPaths(root=root, down=down_path, up=up_path, ctd=ctd_path)
    return paths, truth


def _write_cnv(path: Path, ctd: dict) -> None:
    """Write the headerless 6-column cleaned ``.cnv`` (lat lon pres time temp sal)."""
    cols = np.column_stack([ctd["lat"], ctd["lon"], ctd["pressure"],
                            ctd["elapsed_s"], ctd["temperature"], ctd["salinity"]])
    np.savetxt(path, cols, fmt=["%+.5f", "%+.5f", "%+.4f", "%+.4f", "%+.4f", "%+.4f"])

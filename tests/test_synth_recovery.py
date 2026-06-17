"""End-to-end recovery test: the synthetic generator's known profile is recovered.

Generates a clean (noise=0) dual-head station in-process, runs the full pipeline
(load -> synchronize -> merge -> super-ensemble -> inverse solve), and asserts the solved
absolute profile matches the known ocean truth. This is the known-answer accuracy test the
roadmap called for; it also guards the synthetic generator against regressions.

The committed example fixture under ``tests/fixtures/synthetic/`` is regenerated with::

    ladcp-synth --out tests/fixtures/synthetic --station SYNTH-01 --seed 0 --noise 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ladcp.config import resolve_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.bottom import detect_bottom
from ladcp.qa.depth import synchronize
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import compute_velocity_full
from ladcp.qa.validate import score
from ladcp.synth import SynthConfig, generate_station


def _solve(tmp_path, **cfg_kw):
    cfg = SynthConfig(out=tmp_path, station="SYNTH-01", **cfg_kw)
    paths, truth = generate_station(cfg)
    params = resolve_params("LADCP", cfg.station)
    dh = load_dualhead(str(paths.down), str(paths.up), station=cfg.station, params=params)
    ctd = read_ctd_cnv(str(paths.ctd), params=params)
    # drot=0: the synthetic velocities are already true-north earth frame (no declination)
    res = compute_velocity_full(dh, ctd, drot=0.0, params=params, solver="inverse")
    ut = np.interp(res.vp.z, truth.z, truth.u, left=np.nan, right=np.nan)
    vt = np.interp(res.vp.z, truth.z, truth.v, left=np.nan, right=np.nan)
    return res, truth, cfg, score(res.vp.u, ut), score(res.vp.v, vt)


def test_clean_recovery(tmp_path):
    res, truth, cfg, su, sv = _solve(tmp_path, seed=0, noise=0.0)

    # the linear-shear u is the exactly-resolvable mode: recovered near-perfectly
    assert su.corr > 0.99, f"u corr {su.corr}"
    assert su.rms < 0.02, f"u rms {su.rms}"
    # the curved v mode is mildly damped by the inverse smoothing -- still a strong recovery
    assert sv.corr > 0.95, f"v corr {sv.corr}"
    assert sv.rms < 0.05, f"v rms {sv.rms}"
    # barotropic reference recovered (no systematic bias)
    assert abs(su.bias) < 0.02, f"u bias {su.bias}"
    assert abs(sv.bias) < 0.02, f"v bias {sv.bias}"
    assert abs(res.vp.ubar - truth.ubar) < 0.02
    assert abs(res.vp.vbar - truth.vbar) < 0.02


def test_seabed_recovered(tmp_path):
    res, truth, cfg, *_ = _solve(tmp_path, seed=0, noise=0.0)
    # echo-stack lock; the bottom-track range-gain biases it ~1 bin deep (a real effect)
    assert abs(res.zbottom - cfg.seabed) < 1.5 * cfg.cell_m


def test_profile_monotone_and_bounded(tmp_path):
    res, truth, cfg, *_ = _solve(tmp_path, seed=0, noise=0.0)
    assert np.all(np.diff(res.vp.z) > 0)
    assert res.vp.z[-1] <= cfg.seabed + cfg.cell_m


def test_detect_bottom_standalone(tmp_path):
    """The synthetic echo field alone drives the depth-stack seabed lock."""
    cfg = SynthConfig(out=tmp_path, station="SYNTH-01", seed=0, noise=0.0)
    paths, _ = generate_station(cfg)
    params = resolve_params("LADCP", cfg.station)
    dh = load_dualhead(str(paths.down), str(paths.up), station=cfg.station, params=params)
    ctd = read_ctd_cnv(str(paths.ctd), params=params)
    sync = synchronize(dh, ctd)
    bottom = detect_bottom(dh, sync, ctd)
    assert not bottom.is_fallback              # a real echo-stack lock, not the zmax bound
    assert abs(bottom.zbottom - cfg.seabed) < 1.5 * cfg.cell_m


def test_noisy_still_recovers(tmp_path):
    """With realistic noise the pipeline still recovers the profile (looser bound)."""
    res, truth, cfg, su, sv = _solve(tmp_path, seed=0, noise=0.01)
    assert su.corr > 0.95, f"u corr {su.corr}"
    assert sv.corr > 0.95, f"v corr {sv.corr}"


_FIXTURE = Path(__file__).parent / "fixtures" / "synthetic"


def test_committed_fixture_processes():
    """The committed example dataset still parses and solves (guards the shipped bytes).

    Regenerate with: ``ladcp-synth --out tests/fixtures/synthetic --station SYNTH-01 --seed 0``
    """
    down = _FIXTURE / "LADCP" / "SYNTH-01-LADCP-M.000"
    if not down.exists():
        pytest.skip("synthetic example fixture not present")
    params = resolve_params("LADCP", "SYNTH-01")
    dh = load_dualhead(str(down), str(_FIXTURE / "LADCP" / "SYNTH-01-LADCP-S.000"),
                       station="SYNTH-01", params=params)
    ctd = read_ctd_cnv(str(_FIXTURE / "CTD" / "SYNTH-01_clean.cnv"), params=params)
    res = compute_velocity_full(dh, ctd, drot=0.0, params=params, solver="inverse")
    assert np.isfinite(res.vp.u).any() and np.isfinite(res.vp.v).any()
    assert np.isfinite(res.zbottom)

"""StationSession parity + cache contract (Studio PR 2).

The session cache must be invisible in the results: every ``StationSession.solve`` is
bit-identical to a fresh ``compute_velocity_full`` following the exact ``ladcp-qa``
per-station recipe — including after other configurations have run on the same cached
context (mutation guard). The cache tiers are pinned too: weight/solver changes never
rebuild, edit changes do.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace as dc_replace

import numpy as np
import pytest

from ladcp.config import resolve_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.ingest import apply_header_config, load_dualhead
from ladcp.qa.inverse import compute_velocity_full
from ladcp.session import EditConfig, SadcpConfig, SessionConfig, SolveConfig, StationSession

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")
DROT = -9.878379


def fresh_solve(cfg: SessionConfig):
    """The exact ladcp-qa recipe (_run_one + _velocity_outputs), no session involved."""
    overrides = {}
    if cfg.edit.nearfield_dn_bins is not None:
        overrides["edit_nearfield_dn_bins"] = cfg.edit.nearfield_dn_bins
    if cfg.edit.dzbelow is not None:
        overrides["dzbelow"] = cfg.edit.dzbelow
    if not cfg.edit.soundcorr:
        overrides["soundcorr"] = False
    params = resolve_params("MORIA", "MORIA-80", overrides=overrides or None)
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=params)
    apply_header_config(params, dh)
    ctd = read_ctd_cnv(str(CTD), params=params)
    if cfg.edit.down_only and dh.has_up:
        dh = dc_replace(dh, up=None)
    return compute_velocity_full(dh, ctd, drot=cfg.solve.drot, params=params,
                                 solver=cfg.solve.solver, botfac=cfg.solve.botfac,
                                 barofac=cfg.solve.barofac, smoofac=cfg.solve.smoofac,
                                 sadcpfac=cfg.solve.sadcpfac)


def assert_results_identical(a, b):
    """Bit-identical solve products (NaN == NaN)."""
    eq = np.testing.assert_array_equal
    for f in ("z", "u", "v", "uerr", "nvel"):
        eq(getattr(a.vp, f), getattr(b.vp, f), err_msg=f"vp.{f}")
    eq([a.vp.ubar, a.vp.vbar], [b.vp.ubar, b.vp.vbar], err_msg="vp.ubar/vbar")
    for f in ("z", "u", "v", "u_shear", "v_shear"):
        eq(getattr(a.shear, f), getattr(b.shear, f), err_msg=f"shear.{f}")
    eq([a.zbottom], [b.zbottom], err_msg="zbottom")
    for f in ("resid_z", "resid_u", "resid_v"):
        eq(getattr(a, f), getattr(b, f), err_msg=f)
    assert (a.bp is None) == (b.bp is None)
    if a.bp is not None:
        for f in ("z", "u", "v", "uerr", "n"):
            eq(getattr(a.bp, f), getattr(b.bp, f), err_msg=f"bp.{f}")
    assert (a.err is None) == (b.err is None)
    if a.err is not None:
        for f in ("depth", "resid_u", "resid_v", "u_oce", "v_oce"):
            eq(getattr(a.err, f), getattr(b.err, f), err_msg=f"err.{f}")
    assert (a.weights is None) == (b.weights is None)
    if a.weights is not None:
        assert set(a.weights.ocean) == set(b.weights.ocean)
        for k in a.weights.ocean:
            eq(a.weights.ocean[k], b.weights.ocean[k], err_msg=f"weights.ocean[{k}]")


@pytest.fixture(scope="module")
def session():
    return StationSession(DOWN, UP, CTD, station="MORIA-80", cruise="MORIA")


CFGS = {
    "default": SessionConfig(solve=SolveConfig(drot=DROT)),
    "weights": SessionConfig(solve=SolveConfig(drot=DROT, botfac=0.5, smoofac=0.1)),
    "shear": SessionConfig(solve=SolveConfig(drot=DROT, solver="shear")),
    "down-only": SessionConfig(edit=EditConfig(down_only=True),
                               solve=SolveConfig(drot=DROT)),
    "edit": SessionConfig(edit=EditConfig(nearfield_dn_bins=(3, 4), dzbelow=24.0),
                          solve=SolveConfig(drot=DROT)),
    "no-soundcorr": SessionConfig(edit=EditConfig(soundcorr=False),
                                  solve=SolveConfig(drot=DROT)),
}


@pytest.mark.parametrize("name", CFGS, ids=CFGS.keys())
def test_session_matches_fresh_solve(session, name):
    assert_results_identical(session.solve(CFGS[name]), fresh_solve(CFGS[name]))


def test_soundcorr_toggle_reaches_params(session):
    """Regression: StationSession used to drop EditConfig(soundcorr=False) on the floor,
    silently solving with sound-speed correction on (unlike ladcp-qa --no-soundcorr)."""
    prep = session.prepare(EditConfig(soundcorr=False))
    assert prep.params.soundcorr is False


def test_resolve_repeatable_after_other_configs(session):
    """Mutation guard: a cached context re-solved later still gives the first answer."""
    first = session.solve(CFGS["default"])
    session.solve(CFGS["weights"])
    session.solve(CFGS["shear"])
    assert_results_identical(session.solve(CFGS["default"]), first)


# ---------------------------------------------------------------- cache tiers

def test_weight_and_solver_changes_never_rebuild(monkeypatch):
    import ladcp.qa.inverse as inverse
    calls = []
    real = inverse.build_solve_context
    monkeypatch.setattr(inverse, "build_solve_context",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    ses = StationSession(DOWN, UP, CTD, station="MORIA-80", cruise="MORIA")
    for cfg in (CFGS["default"], CFGS["weights"], CFGS["shear"]):
        ses.solve(cfg)
    assert len(calls) == 1
    ses.solve(CFGS["edit"])                       # edit change -> one more build
    assert len(calls) == 2
    ses.solve(CFGS["edit"])                       # cached again
    assert len(calls) == 2


def test_sadcp_profile_cached_per_config(session, monkeypatch):
    import ladcp.qa.cli as qacli
    fake = np.column_stack([np.arange(10.0, 60.0, 10.0), np.full(5, 0.1),
                            np.full(5, -0.1), np.full(5, 0.02)])
    calls = []
    monkeypatch.setattr(qacli, "_sadcp_profile",
                        lambda *a, **k: calls.append(a[0]) or fake)
    sa = SadcpConfig(folder="some/sadcp")
    r1 = session.solve(SessionConfig(sadcp=sa, solve=SolveConfig(drot=DROT)))
    r2 = session.solve(SessionConfig(sadcp=sa, solve=SolveConfig(drot=DROT, sadcpfac=6.0)))
    assert len(calls) == 1                        # same SadcpConfig -> one ingest
    assert calls[0]["fac"] == 3.0                 # opts dict shape (fac rides along)
    assert r1.sadcp is fake and r2.sadcp is fake
    session.solve(SessionConfig(sadcp=SadcpConfig(folder="other/sadcp"),
                                solve=SolveConfig(drot=DROT)))
    assert len(calls) == 2                        # different SadcpConfig -> new ingest


def test_session_requires_ctd():
    ses = StationSession(DOWN, UP, None, station="MORIA-80", cruise="MORIA")
    with pytest.raises(ValueError, match="needs a CTD"):
        ses.solve(SessionConfig())


def test_declination_fallback_on_nan_position():
    import datetime as dt

    from ladcp.session import resolve_declination
    drot, source = resolve_declination(float("nan"), float("nan"),
                                       dt.datetime(2026, 6, 11))
    assert (drot, source) == (0.0, "fallback-zero")

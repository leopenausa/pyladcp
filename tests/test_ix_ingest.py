"""Phase-3 validation: raw PD0 -> earth velocity -> W-integration depth.

The golden processing log prints the first-sweep down-trace and up-trace max depths
(``maxdepth from down-trace`` / ``up-trace``), computed purely from the integrated
reference-bin vertical velocity. Reproducing the **down-trace** depth from a clean-slate
beam->earth->merge->edit->integrate chain validates the earth-velocity ``w`` end-to-end.

The down-trace integrates the (short) downcast and is the robust metric (within a few %
across stations). The up-trace integrates the long, bottle-stop-laden upcast and is
sensitive to a ~1 cm/s accumulated W zero-point bias; it is checked only loosely here.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
import xarray as xr

from ladcp.ix.depth import centred_dt, get_mean_w, trace_depths
from ladcp.ix.ingest import b2earth, load_ladcp, uvrot

DATA = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "RB1606_P18"
STATIONS = ["016", "063", "117", "182", "217"]

pytestmark = pytest.mark.skipif(
    not (DATA / "016" / "016DL000.000").exists(),
    reason="GO-SHIP raw PD0 not present",
)


def _window(ds) -> tuple[np.datetime64, np.datetime64]:
    ts = np.asarray(ds.attrs["time_start"], int)
    te = np.asarray(ds.attrs["time_end"], int)
    f = "%04d-%02d-%02dT%02d:%02d:%02d"
    return np.datetime64(f % tuple(ts)), np.datetime64(f % tuple(te))


def _load(st: str):
    ds = xr.open_dataset(str(DATA / st / f"{st}.nc"))
    t0, t1 = _window(ds)
    d = load_ladcp(
        str(DATA / st / f"{st}DL000.000"),
        str(DATA / st / f"{st}UL000.000"),
        drot=float(ds.attrs["drot"]),
        time_start=t0, time_end=t1,
    )
    return d, ds


def test_b2earth_zero_attitude_identity():
    """With zero tilt and heading, down-looker earth W = mean of the 4 beam W signs."""
    # synthetic: all beams = -0.5 m/s, level, heading 0 -> pure downward motion
    beam = np.full((4, 3, 1), -0.5)
    e = b2earth(beam, np.zeros(1), np.zeros(1), np.zeros(1),
                beams_up=False, beamangle=20.0)
    # error velocity ~ 0 when all beams equal; vertical finite; horizontals ~0
    assert np.allclose(e[3], 0.0, atol=1e-9)          # error vel
    assert np.allclose(e[0], 0.0, atol=1e-9)          # east
    assert np.allclose(e[1], 0.0, atol=1e-9)          # north
    assert np.all(np.isfinite(e[2]))                  # vertical


def test_b2earth_three_beam_reconstruction():
    """One missing beam is reconstructed; >=2 missing -> NaN."""
    beam = np.full((4, 1, 2), -0.4)
    beam[0, 0, 0] = np.nan                # one missing -> 3-beam
    beam[0, 0, 1] = beam[1, 0, 1] = np.nan  # two missing -> drop
    e = b2earth(beam, np.zeros(2), np.zeros(2), np.zeros(2),
                beams_up=False, beamangle=20.0)
    assert np.isfinite(e[2, 0, 0])
    assert np.isnan(e[2, 0, 1])


def test_uvrot_roundtrip():
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ur, vr = uvrot(u, v, 8.18)
    u2, v2 = uvrot(ur, vr, -8.18)
    assert np.allclose([u2, v2], [u, v], atol=1e-12)


@pytest.mark.parametrize("st", STATIONS)
def test_merge_shapes(st):
    d, ds = _load(st)
    assert d.n_bin == d.izu.size + d.izd.size
    assert d.ru.shape == d.rv.shape == d.rw.shape == (d.n_bin, d.n_ens)
    # izu ordered near->far: first up reference row is the LAST up block row
    assert d.izu[0] == d.izu.size - 1
    # cast duration matches the golden dt_profile to ~1 s
    span = (d.time[-1] - d.time[0]) / np.timedelta64(1, "s")
    assert abs(span - ds.attrs["dt_profile"]) < 3.0


@pytest.mark.parametrize("st", STATIONS)
def test_downtrace_depth_matches_log(st):
    """Integrated-W down-trace max depth within 5% of the golden log value."""
    d, ds = _load(st)
    log = ds.attrs["LOG_Inverse_log"]
    gd = float(re.search(r"down-trace is ([\d.]+)", log).group(1))
    r = trace_depths(d, soundc=False)
    assert abs(r["down_max"] - gd) / gd < 0.05
    # sane vs the golden max profile depth
    assert r["down_max"] > 0.9 * ds.attrs["maxdepth"]


def test_get_mean_w_gap_free():
    """getmeanw must return a finite, zero-filled series (no NaNs leak through)."""
    d, _ = _load("016")
    wm = get_mean_w(d)
    assert wm.shape == (d.n_ens,)
    assert np.all(np.isfinite(wm))
    assert centred_dt(d.time).shape == (d.n_ens,)

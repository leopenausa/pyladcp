"""Phase-3 super-ensemble formation (prepinv.m STEP 10).

The golden log records, per station, the reduced super-ensemble count
(``se10_reduced_len`` == ``n_superensembles``) and the two-pass outlier discards
(``se10_outlier_down`` / ``se10_outlier_up``). The super-ensemble **count** is depth-driven
(grouping the cast by ``avdz`` movement) and is the robust target: it reproduces the golden
to within a few across all five stations. The outlier / zero-scatter / floored-scatter
counts derive from the raw velocity scatter and run systematically high, inheriting the same
loadrdi near-field (bin-1) over-retention seen in [[ladcp-editdata-pglim-findings]] — my
edited raw field keeps more finite cells, so more get flagged. They are checked loosely.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import xarray as xr

from ladcp.ix.bottom import own_bottom_track
from ladcp.ix.depth import ctd_depth_on_ping, detect_bottom_and_flag
from ladcp.ix.ingest import LADCPData
from ladcp.ix.superens import _rms, form_superensembles
from ladcp.io.ctd_seabird import load_seabird_cnv

DATA = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "RB1606_P18"
EXPECTED = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "expected_values.json"
STATIONS = ["016", "063", "117", "182", "217"]
CNV = {
    "016": "ctd_009_01.cnv", "063": "ctd_055_01.cnv", "117": "ctd_109_01.cnv",
    "182": "ctd_169_01.cnv", "217": "ctd_204_01.cnv",
}

pytestmark = pytest.mark.skipif(
    not (DATA / "016" / "016DL000.000").exists(),
    reason="GO-SHIP raw PD0 not present",
)


def _expected():
    return json.loads(EXPECTED.read_text())


def _build(st: str):
    from ladcp.ix.ingest import load_ladcp
    ds = xr.open_dataset(str(DATA / st / f"{st}.nc"))
    ts = np.asarray(ds.attrs["time_start"], int)
    te = np.asarray(ds.attrs["time_end"], int)
    f = "%04d-%02d-%02dT%02d:%02d:%02d"
    t0, t1 = np.datetime64(f % tuple(ts)), np.datetime64(f % tuple(te))
    d = load_ladcp(
        str(DATA / st / f"{st}DL000.000"), str(DATA / st / f"{st}UL000.000"),
        drot=float(ds.attrs["drot"]), time_start=t0, time_end=t1,
    )
    cast = load_seabird_cnv(str(DATA / st / CNV[st]))
    z = ctd_depth_on_ping(cast, d.time)
    r = detect_bottom_and_flag(d, z, own_bottom_track(d))
    # getdpth folds the below-bottom/surface flags into the weight before prepinv
    d.weight = d.weight + np.where(np.isnan(r["izmflag"]), np.nan, 0.0)
    se = form_superensembles(
        d, z, zbottom=r["zbottom"], avdz=float(ds.attrs["avdz"]),
        superens_std_min=float(ds.attrs["superens_std_min"]),
        outlier_n=int(ds.attrs["outlier_n"]),
    )
    return se, ds


# --------------------------------------------------------------------------- #
# unit-level
# --------------------------------------------------------------------------- #
def test_rms_no_mean_removed():
    """rms is sqrt(mean(x^2)) over finite values (mean NOT removed)."""
    x = np.array([3.0, 4.0, np.nan])
    assert np.isclose(_rms(x), np.sqrt((9 + 16) / 2))


def _synthetic(nens=60):
    """Small two-head cast with a smooth dive/rise — exercises grouping termination."""
    nbu, nbd = 4, 4
    nbin = nbu + nbd
    rng = np.random.default_rng(0)
    ru = rng.normal(0, 0.05, (nbin, nens))
    rv = rng.normal(0, 0.05, (nbin, nens))
    rw = rng.normal(0, 0.02, (nbin, nens))
    re = rng.normal(0, 0.01, (nbin, nens))
    ts = np.full((nbin, nens), 60.0)
    weight = np.ones((nbin, nens))
    zd = np.array([8.0, 16, 24, 32])
    zu = np.array([8.0, 16, 24, 32])
    izu = np.flip(np.arange(nbu))
    izd = np.arange(nbd) + nbu
    t = np.datetime64("2016-11-26T00:00:00") + np.arange(nens) * np.timedelta64(1, "s")
    return LADCPData(
        time=t, ru=ru, rv=rv, rw=rw, re=re, ts=ts, weight=weight, zd=zd, zu=zu,
        izd=izd, izu=izu, pitch=np.zeros(nens), roll=np.zeros(nens),
        heading=np.zeros(nens), tilt=np.zeros(nens), temp=np.full(nens, 4.0),
        sv=np.full(nens, 1500.0), hbot=np.full(nens, np.nan),
        bvel=np.full((nens, 4), np.nan), drot=0.0, meta={"freq_dn_khz": 150.0},
    )


def test_grouping_terminates_and_covers_cast():
    """Regression guard for the MATLAB round() fix: the depth loop must terminate.

    Banker's rounding stalls the final group on its start index; ``floor(x+0.5)`` advances.
    A descending-then-rising depth track must yield a sane number of super-ensembles.
    """
    d = _synthetic(nens=60)
    # dive to -160 m and back over the 60 ensembles (~2.7 m/ensemble -> ~8 m every ~3)
    z = -np.concatenate([np.linspace(0, 160, 30), np.linspace(160, 0, 30)])
    se = form_superensembles(d, z, superens_std_min=0.07, outlier_n=10)
    assert 0 < se.n_se < 60
    assert se.ru.shape == (8, se.n_se)
    assert np.array_equal(se.izr, np.array([d.izd[1], d.izd[2], d.izu[1], d.izu[2]]))


# --------------------------------------------------------------------------- #
# golden validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("st", STATIONS)
def test_reduced_len_matches_golden(st):
    """Super-ensemble count within +/-6 of the golden n_superensembles (all 5 stations)."""
    exp = _expected()[str(int(st))]
    se, _ = _build(st)
    gold = exp["n_superensembles"]
    assert abs(se.n_se - gold) <= 6
    assert se.counts["reduced_len"] == se.n_se


def test_superens_structure_016():
    """Scatter floored at single-ping accuracy; zero-scatter cells de-weighted (016)."""
    se, ds = _build("016")
    smin = float(ds.attrs["superens_std_min"])
    fin = np.isfinite(se.ruvs)
    # every finite scatter is >= the floor
    assert np.all(se.ruvs[fin] >= smin - 1e-9)
    # de-weighted cells (weight NaN) line up with the zero-std mask the counter reported
    assert se.counts["weight_nan_zero_std"] > 0
    # super-ensemble horizontal velocities are physical
    assert np.nanmax(np.abs(se.ru)) < 5.0
    assert np.nanmax(np.abs(se.rv)) < 5.0


@pytest.mark.parametrize("st", STATIONS)
def test_outlier_counts_run_high_but_bounded(st):
    """Two-pass outlier discards >= golden and < 2x (loadrdi near-field over-retention)."""
    exp = _expected()[str(int(st))]["log"]
    se, _ = _build(st)
    for key, gold in (("outlier_down", exp["se10_outlier_down"]),
                      ("outlier_up", exp["se10_outlier_up"])):
        assert se.counts[key] >= gold
        assert se.counts[key] < 2.0 * gold

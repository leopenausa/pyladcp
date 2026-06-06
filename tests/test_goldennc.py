"""Phase-1 validation: the golden NetCDF reader + log scraper.

Target = GO-SHIP RB1606/P18 (LDEO_IX IX_13beta). These tests pin the parsed values
against the data files so a regression in the reader/scraper is caught immediately, and
they assert the internal consistency between the log prose and the authoritative attrs.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.io.goldennc import expected_values, read_golden_nc, scrape_log

DATA = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "RB1606_P18"
STATIONS = ["016", "063", "117", "182", "217"]


def _path(st: str) -> pathlib.Path:
    return DATA / st / f"{st}.nc"


pytestmark = pytest.mark.skipif(
    not _path("016").exists(), reason="GO-SHIP golden data not present"
)


@pytest.mark.parametrize("st", STATIONS)
def test_reader_shapes_consistent(st):
    g = read_golden_nc(str(_path(st)))
    nz = g.z.size
    assert g.u.shape == g.v.shape == g.uerr.shape == g.p.shape == (nz,)
    assert g.nvel.size == nz
    # zbot / sadcp blocks have their own lengths but match their partners
    assert g.vars["ubot"].shape == g.vars["vbot"].shape == g.vars["zbot"].shape
    assert g.vars["u_sadcp"].shape == g.vars["v_sadcp"].shape
    # super-ensemble kinematics share the tim axis
    nt = g.vars["tim_hour"].size
    for k in ("shiplat", "shiplon", "uship", "vship", "zctd", "wctd"):
        assert g.vars[k].size == nt


@pytest.mark.parametrize("st", STATIONS)
def test_log_matches_authoritative_attrs(st):
    """Numbers scraped from the prose log must agree with the struct attrs."""
    g = read_golden_nc(str(_path(st)))
    lv = g.log.values
    # final super-ensemble count: log's last "reduced profile length" == tim axis
    assert lv["se12_reduced_len"] == g.n_superensembles
    # inversion matrix dims: A1 == n super-ens, A2 == n depth bins
    assert lv["inv_len_A1"] == g.n_superensembles
    assert lv["inv_len_A2"] == g.z.size
    # velocity error set in getinv == the velerr attr (log rounds to 3 dp)
    assert lv["inv_velocity_error_set"] == pytest.approx(g.params["velerr"], abs=1e-3)
    # declination in log == GEN attr (log rounds to 1 dp)
    assert lv["declination"] == pytest.approx(
        g.params["GEN_Magnetic_deviation_deg"], abs=0.05
    )


@pytest.mark.parametrize("st", STATIONS)
def test_expected_values_complete(st):
    e = expected_values(read_golden_nc(str(_path(st))))
    for key in (
        "declination_deg", "bottom_depth_m", "n_raw_ensembles",
        "n_superensembles", "n_z_bins", "velerr", "best_lag_scans", "ubar", "vbar",
    ):
        assert e[key] is not None, f"missing {key}"
    assert e["dz"] == 8
    assert e["software"].endswith("IX_13beta")
    assert np.isfinite(e["bottom_depth_m"])


def test_split_steps_isolates_repeated_phrases():
    """Steps 10 and 12 share wording; the splitter must not let them collide."""
    g = read_golden_nc(str(_path("016")))
    lv = g.log.values
    # both reductions present and independently captured
    assert "se10_reduced_len" in lv and "se12_reduced_len" in lv
    assert set(g.log.steps).issuperset({3, 4, 6, 7, 9, 10, 12, 13, 14})


def test_scrape_log_handles_empty():
    log = scrape_log("")
    assert log.values == {}
    assert log.steps == {}

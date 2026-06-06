"""Phase-2 validation: the Sea-Bird 24 Hz CTD/nav loader vs the golden NetCDF.

The decimation/despike recipe is validated by reproducing the golden ``ctd_t``/``ctd_s``
profiles (interpolated onto the golden ``z`` grid) to within a few milli-units median,
and the per-scan nav by reproducing ``nav_start``/``nav_end`` to sub-arcminute.
"""

from __future__ import annotations

import glob
import pathlib

import numpy as np
import pytest
import xarray as xr

from ladcp.io.ctd_seabird import (
    load_seabird_cnv,
    nav_position,
    p2z,
    parse_cnv_header,
    profile_on_z,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "RB1606_P18"
STATIONS = ["016", "063", "117", "182", "217"]

pytestmark = pytest.mark.skipif(
    not (DATA / "016").exists(), reason="GO-SHIP golden data not present"
)


def _cnv(st: str) -> str:
    return glob.glob(str(DATA / st / "ctd_*.cnv"))[0]


def test_p2z_checkvalue():
    """Saunders & Fofonoff checkvalue: p=1000 bar (=10000 dbar), lat=30 -> 9712.654 m."""
    assert p2z(10000.0, 30.0) == pytest.approx(9712.654, abs=0.01)


def test_header_column_map():
    hdr = parse_cnv_header(_cnv("016"))
    cols = hdr["columns"]
    for key in ("timeS", "prDM", "t090C", "sal00", "latitude", "longitude"):
        assert key in cols
    assert hdr["start_time"] is not None
    assert hdr["nvalues"] and hdr["nvalues"] > 100000


def test_badflag_not_a_threshold():
    """The SBE bad sentinel must not wipe ordinary negatives (longitude ~ -110)."""
    cast = load_seabird_cnv(_cnv("016"))
    assert np.isfinite(cast.ctd.lon).any()
    assert np.nanmedian(cast.ctd.lon) < -100  # eastern Pacific, ~ -110


@pytest.mark.parametrize("st", STATIONS)
def test_ctd_profile_matches_golden(st):
    ds = xr.open_dataset(str(DATA / st / f"{st}.nc"))
    lat = float(ds.attrs["lat"])
    cast = load_seabird_cnv(_cnv(st), lat_ref=lat)
    ti, si = profile_on_z(cast, ds.z.values, lat=lat)
    ct, cs = ds.ctd_t.values, ds.ctd_s.values
    mt = np.isfinite(ct) & np.isfinite(ti)
    ms = np.isfinite(cs) & np.isfinite(si)
    assert mt.sum() > 50 and ms.sum() > 50
    # median agreement is essentially exact; surface bins can differ (mixed layer)
    assert np.nanmedian(np.abs(ti - ct)[mt]) < 0.005     # deg C
    assert np.nanmedian(np.abs(si - cs)[ms]) < 0.002     # PSU
    ds.close()


@pytest.mark.parametrize("st", STATIONS)
def test_nav_positions_match_golden(st):
    ds = xr.open_dataset(str(DATA / st / f"{st}.nc"))
    cast = load_seabird_cnv(_cnv(st), lat_ref=float(ds.attrs["lat"]))
    for when, attr in (("start", "nav_start"), ("end", "nav_end")):
        lat, lon = nav_position(cast, when)
        g = ds.attrs[attr]
        glat, glon = g[0] + g[1] / 60, g[2] + g[3] / 60
        # sub-arcminute (start can drift slightly: cnv includes pre-cast soak)
        assert abs(lat - glat) * 60 < 0.5
        assert abs(lon - glon) * 60 < 0.5
    ds.close()

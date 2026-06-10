"""Nav-track reader + SADCP clock-desync estimation (FDCCC1 curveball).

The acquisition PC's clock can be arbitrarily wrong (FDCCC1: ~12 years) while the
embedded GPS positions stay correct; the offset is recovered by sliding the
position series against an independently timestamped nav track.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.io.nav import NavTrack, estimate_time_offset, read_nav, shift_time
from ladcp.io.sadcp_vmdas import SadcpDataset

_SADO = (
    "fecha,longitud,latitud,rumbo,velocidad,profundidad,cog,sog,fecha_telegrama\n"
    "01-03-2022 00:00:00,2.6237833,41.4849,0,8.8,97.5,57,8.8,28-02-2022 23:59:59\n"
    "01-03-2022 00:00:01,2.6236667,41.4848333,0,8.8,97.44,58,8.8,01-03-2022 00:00:00\n"
    "01-03-2022 00:00:02,2.6238833,41.4849333,0,8.8,97.44,58,8.8,01-03-2022 00:00:01\n"
)


def test_read_nav_sado_posicion(tmp_path):
    f = tmp_path / "01032022.posicion.txt"
    f.write_text(_SADO)
    nav = read_nav(f)
    assert nav.n == 3
    assert nav.time[0] == np.datetime64("2022-03-01T00:00:00")
    assert nav.lat[0] == pytest.approx(41.4849)
    assert nav.lon[0] == pytest.approx(2.6237833)


def test_read_nav_sado_csv_variant(tmp_path):
    # the *_navsado.csv re-export: ';' separator and DD/MM/YYYY dates
    f = tmp_path / "01032022_navsado.csv"
    f.write_text(_SADO.replace(",", ";").replace("01-03-2022", "01/03/2022")
                 .replace("28-02-2022", "28/02/2022"))
    nav = read_nav(f)
    assert nav.n == 3 and nav.lat[0] == pytest.approx(41.4849)


def test_read_nav_dir_concatenates_and_skips_unparseable(tmp_path):
    (tmp_path / "01032022.posicion.txt").write_text(_SADO)
    later = _SADO.replace("01-03-2022 00:00:0", "02-03-2022 00:00:0")
    (tmp_path / "02032022.posicion.txt").write_text(later)
    (tmp_path / "notes.txt").write_text("not navigation at all\n")
    nav = read_nav(tmp_path)
    assert nav.n == 6
    assert np.all(np.diff(nav.time.astype("int64")) >= 0)     # time-sorted


def test_read_nav_generic_csv(tmp_path):
    f = tmp_path / "track.csv"
    f.write_text("time,lat,lon\n2022-03-01T00:00:00,41.0,2.0\n"
                 "2022-03-01T00:00:10,41.001,2.001\n")
    nav = read_nav(f)
    assert nav.n == 2 and nav.lon[1] == pytest.approx(2.001)


def _synthetic_track(n=2000, dt_s=30.0, t0="2022-03-01T00:00:00"):
    """A wiggly ship track (unique as a function of time) at dt_s sampling."""
    t = np.datetime64(t0) + (np.arange(n) * dt_s * 1e9).astype("timedelta64[ns]")
    s = np.arange(n) * dt_s
    lat = 41.5 + 0.3 * np.sin(2 * np.pi * s / 9000.0) + s * 2e-6
    lon = 2.5 + 0.3 * np.cos(2 * np.pi * s / 13000.0) + s * 3e-6
    return t, lat, lon


def test_estimate_time_offset_recovers_known_shift():
    t, lat, lon = _synthetic_track()
    nav = NavTrack(time=t, lat=lat, lon=lon)
    true_offset = 383_809_944.0                  # ~12.2 y, the FDCCC1 scale
    wrong = t - np.timedelta64(int(true_offset * 1e9), "ns")
    # the mis-clocked instrument saw a sub-window of the track
    sl = slice(200, 1500)
    r = estimate_time_offset(wrong[sl], lat[sl], lon[sl], nav)
    assert r["offset_s"] == pytest.approx(true_offset, abs=1.0)
    assert r["median_m"] < 5.0
    assert r["overlap"] == pytest.approx(1.0)


def test_estimate_time_offset_rejects_disjoint_track():
    t, lat, lon = _synthetic_track(n=500)
    nav = NavTrack(time=t, lat=lat, lon=lon)
    # positions nowhere near the nav track -> no offset can align them, but the
    # scan must still complete and report the (poor) best fit rather than crash;
    # a *time* window that can never overlap is the hard error
    t2, la2, lo2 = _synthetic_track(n=5, t0="2022-06-01T00:00:00")
    with pytest.raises(ValueError, match="too few"):
        estimate_time_offset(t2, np.full(5, np.nan), lo2, nav)


def test_shift_time_returns_shifted_copy():
    t, lat, lon = _synthetic_track(n=4)
    ds = SadcpDataset(time=t, lat=lat, lon=lon, depth=np.array([10.0]),
                      u=np.zeros((1, 4)), v=np.zeros((1, 4)), freq_khz=75,
                      transducer_depth=5.0, file_type="STA", source="syn",
                      n_files=1)
    out = shift_time(ds, 90.5)
    assert out.time[0] - ds.time[0] == np.timedelta64(90500, "ms")
    assert "clock +90.50 s" in out.source
    assert ds.time[0] == t[0]                                  # original untouched

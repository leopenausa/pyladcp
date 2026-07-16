"""EK80 ADCP-mode reader + file tools (ladcp.io.sadcp_ek80 / ek80_files, ladcp-ek80).

Built on a miniature ICES SONAR-netCDF4 file: a ``/Sonar/Beam_group1/ADCP`` group with a
``Mean_current`` subgroup of ragged (vlen) earth-frame current profiles, mirroring the
real Simrad layout (fine vertical grid, NaN-padded, ns-since-1601 time).
"""
from __future__ import annotations

import numpy as np
import pytest

from ladcp.io import ek80_files as ek
from ladcp.io.sadcp_ek80 import read_ek80

netCDF4 = pytest.importorskip("netCDF4")

N_MEAN = 4
VSI = 0.05                      # vertical sample interval [m]
D0 = 14.0                       # first-sample depth [m]
N_SAMP = 600                    # native samples per profile (~14..44 m valid below)
N_VALID = 400                   # first N_SAMP cells finite, rest NaN
U0, V0 = 0.10, -0.20
# 2026-06-25T10:52:45Z as ns since 1601-01-01 (FILETIME-style epoch the reader expects)
_S_1601_TO_1970 = 11644473600
_T0_NS1601 = int((np.datetime64("2026-06-25T10:52:45") - np.datetime64("1970-01-01"))
                 / np.timedelta64(1, "ns")) + _S_1601_TO_1970 * 1_000_000_000


def _write_ek80(path):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.Conventions = "SONAR-netCDF4-2.0"
        g = nc.createGroup("/Sonar/Beam_group1/ADCP")
        g.createDimension("ping_time", N_MEAN)
        g.createVariable("vertical_sample_interval", "f4", ("ping_time",))[:] = VSI
        g.createVariable("depth_first_sample_center", "f4", ("ping_time",))[:] = D0
        g.createVariable("error_velocity", "f4", ("ping_time",))[:] = 0.01

        mc = g.createGroup("Mean_current")
        mc.createDimension("mean_time", N_MEAN)
        mc.createVariable("mean_time", "u8", ("mean_time",))[:] = (
            np.uint64(_T0_NS1601)
            + np.arange(N_MEAN, dtype=np.uint64) * np.uint64(3_000_000_000))   # 3 s apart
        mc.createVariable("mean_platform_latitude", "f8", ("mean_time",))[:] = 51.9
        mc.createVariable("mean_platform_longitude", "f8", ("mean_time",))[:] = -16.4
        mc.createVariable("averaging", "i4", ("mean_time",))[:] = 50

        vlt = nc.createVLType("f4", "sample_t")
        prof = np.full(N_SAMP, np.nan, "f4")
        prof[:N_VALID] = 1.0
        for name, val in (("current_velocity_geographical_east", U0),
                          ("current_velocity_geographical_north", V0),
                          ("current_velocity_geographical_down", 0.0)):
            var = mc.createVariable(name, vlt, ("mean_time",))
            for i in range(N_MEAN):
                var[i] = (prof * val).astype("f4")
        q = mc.createVariable("quality", nc.createVLType("i4", "qual_t"), ("mean_time",))
        for i in range(N_MEAN):
            row = np.zeros(N_SAMP, "i4")
            row[:N_VALID] = 50
            q[i] = row


@pytest.fixture
def ek80_file(tmp_path):
    p = tmp_path / "MORIA2ADCP150Khz-D20260625-T105245.nc"
    _write_ek80(str(p))
    return p


def test_read_ek80_shapes_depth_and_velocity(ek80_file):
    ds = read_ek80(str(ek80_file), bin_m=8.0, z_max=120.0)
    assert ds.file_type == "EK80" and ds.freq_khz == 150
    assert ds.n_ens == N_MEAN
    assert ds.u.shape == (ds.depth.size, N_MEAN)          # [z, t], absolute earth-frame
    # valid layer ~ D0 .. D0 + N_VALID*VSI = 14 .. 34 m
    valid = np.isfinite(ds.u).any(axis=1)
    zlo, zhi = ds.depth[valid].min(), ds.depth[valid].max()
    assert 8 <= zlo <= 20 and 28 <= zhi <= 40
    # constant input -> the rebinned values recover U0/V0
    assert np.nanmedian(ds.u) == pytest.approx(U0, abs=1e-3)
    assert np.nanmedian(ds.v) == pytest.approx(V0, abs=1e-3)
    # time decoded from ns-since-1601
    assert str(ds.time[0]).startswith("2026-06-25T10:52:45")


def test_slim_extract_roundtrips(ek80_file, tmp_path):
    dst = tmp_path / "slim" / ek80_file.name
    ek.slim_extract(str(ek80_file), str(dst))
    assert dst.exists()
    full = read_ek80(str(ek80_file), bin_m=8.0, z_max=120.0)
    slim = read_ek80(str(dst), bin_m=8.0, z_max=120.0)
    assert slim.n_ens == full.n_ens
    np.testing.assert_allclose(slim.depth, full.depth)
    np.testing.assert_allclose(np.nan_to_num(slim.u), np.nan_to_num(full.u), atol=1e-6)


def test_scan_reads_filename_and_peeks_time(ek80_file):
    rows = ek.scan([str(ek80_file)], peek=True)
    assert len(rows) == 1
    r = rows[0]
    assert r["start"].strftime("%Y-%m-%d %H:%M:%S") == "2026-06-25 10:52:45"
    assert r["time_source"] == "peek" and r["n"] == N_MEAN
    assert r["end"] >= r["start"]


def test_correlate_maps_cast_to_file(ek80_file):
    rows = ek.scan([str(ek80_file)], peek=False)          # filename start is enough
    casts = [("MORIA2_02", _dt("2026-06-25T11:00:00"), 51.9, -16.4),
             ("MORIA2_99", _dt("2026-06-26T00:00:00"), 0.0, 0.0)]
    mapping = ek.correlate(rows, casts, pre_min=20, post_min=170)
    assert mapping["MORIA2_02"] == [str(ek80_file)]       # cast 8 min after file start
    assert mapping["MORIA2_99"] == []                     # next day -> no coverage


def test_correlate_is_gap_aware():
    """A file's coverage ends at min(next_start, start + file_dur): a pre-gap file must
    not span a transit gap, while a file starting just before the window still counts."""
    from datetime import timedelta

    t0 = _dt("2026-06-25T10:00:00")
    rows = [{"file": "a.nc", "start": t0},
            {"file": "b.nc", "start": t0 + timedelta(minutes=6)},
            {"file": "c.nc", "start": t0 + timedelta(minutes=60)}]
    # cast window opens at t0+63 (pre=20): a ends at next_start (t0+6), b at
    # start+file_dur (t0+14, not the 54 min gap to c) -- only c (ends t0+68) covers it
    casts = [("ST_1", t0 + timedelta(minutes=83), None, None)]
    mapping = ek.correlate(rows, casts, pre_min=20, post_min=170, file_dur_min=8)
    assert mapping["ST_1"] == ["c.nc"]


def test_cli_timetable_runs(ek80_file, capsys):
    from ladcp.ek80_cli import main
    rc = main(["timetable", str(ek80_file)])
    assert rc == 0
    assert "MORIA2ADCP150Khz" in capsys.readouterr().out


def _dt(iso):
    from datetime import datetime, timezone
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)

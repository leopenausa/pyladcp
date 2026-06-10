"""ladcp-compare: time-based station pairing + cruise comparison report."""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.qa import compare as C

xr = pytest.importorskip("xarray")
sio = pytest.importorskip("scipy.io")

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _write_legacy_mat(path, *, date, z, u, v, ubar=0.1):
    dr = {"name": "syn", "date": np.array(date, dtype=float),
          "lat": 42.0, "lon": 3.0,
          "z": z, "u": u, "v": v, "uerr": np.full(z.size, 0.01),
          "nvel": np.full(z.size, 50.0), "ubar": ubar, "vbar": -0.05,
          "zbot": z[-3:], "ubot": u[-3:], "vbot": v[-3:]}
    sio.savemat(str(path), {"dr": dr})


def _write_ours_nc(root, station, *, time_utc, z, u, v, ubar=0.1):
    d = root / "stations" / station
    d.mkdir(parents=True)
    ds = xr.Dataset(
        {"u": ("depth", u), "v": ("depth", v),
         "u_bt": ("depth_bt", u[-3:]), "v_bt": ("depth_bt", v[-3:])},
        coords={"depth": z, "depth_bt": z[-3:]},
        attrs={"station": station, "time_utc": time_utc, "latitude_deg": 42.0,
               "longitude_deg": 3.0, "ubar_ms": ubar, "vbar_ms": -0.05})
    ds.to_netcdf(d / f"{station}.nc")


def _setup(tmp_path, shift_u=0.0):
    z = np.arange(8.0, 200.0, 8.0)
    u = 0.1 * np.sin(z / 40.0) + 0.2
    v = -0.1 * np.cos(z / 60.0)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_legacy_mat(legacy / "SYN_001.mat", date=[2022, 3, 2, 6, 0, 0],
                      z=z, u=u, v=v)
    _write_legacy_mat(legacy / "SYN_002.mat", date=[2022, 3, 2, 12, 0, 0],
                      z=z, u=u + 0.05, v=v)
    ours = tmp_path / "qa_out"
    _write_ours_nc(ours, "t1-01", time_utc="2022-03-02T06:10:00",
                   z=z, u=u + shift_u, v=v)
    return ours, legacy, z, u, v


def test_pair_by_time_and_perfect_score(tmp_path):
    ours, legacy, z, u, v = _setup(tmp_path)
    o = C.scan_ours(ours)
    m = C.scan_legacy(legacy)
    assert len(o) == 1 and len(m) == 2
    pairs, ours_only, legacy_only = C.pair_by_time(o, m)
    assert len(pairs) == 1
    oc, lc, dt = pairs[0]
    assert lc.name == "SYN_001"                  # nearest in time (10 min vs 6 h)
    assert [x.name for x in legacy_only] == ["SYN_002"]
    r = C.compare_pair(oc, lc, dt)
    assert r.u.corr == pytest.approx(1.0)
    assert r.u.rms == pytest.approx(0.0, abs=1e-9)
    assert r.dubar == pytest.approx(0.0)


def test_known_bias_is_measured(tmp_path):
    ours, legacy, *_ = _setup(tmp_path, shift_u=0.03)
    pairs, _, _ = C.pair_by_time(C.scan_ours(ours), C.scan_legacy(legacy))
    r = C.compare_pair(*pairs[0][:2], pairs[0][2])
    assert r.u.bias == pytest.approx(0.03, abs=1e-6)
    assert r.u.corr == pytest.approx(1.0)


def test_write_report_and_cli(tmp_path):
    ours, legacy, *_ = _setup(tmp_path)
    out = tmp_path / "rep"
    rc = C.main(["--ours", str(ours), "--legacy", str(legacy),
                 "-o", str(out), "--title", "syn"])
    assert rc == 0
    assert (out / "comparison.csv").exists()
    assert (out / "comparison_report.pdf").stat().st_size > 10_000
    assert "SYN_002" in (out / "unpaired.txt").read_text()


def test_alternate_substitution_is_labelled(tmp_path):
    ours, legacy, z, u, v = _setup(tmp_path, shift_u=0.05)
    alt = tmp_path / "qa_alt"
    _write_ours_nc(alt, "t1-01", time_utc="2022-03-02T06:10:00", z=z, u=u, v=v)
    out = tmp_path / "rep_alt"
    rc = C.main(["--ours", str(ours), "--legacy", str(legacy),
                 "--alt-dir", str(alt), "--alt-stations", "t1-01",
                 "--alt-label", "botfac=0", "-o", str(out)])
    assert rc == 0
    txt = (out / "comparison.csv").read_text()
    assert "botfac=0" in txt
    # the substituted (bias-free) run is what got scored
    import pandas as pd
    df = pd.read_csv(out / "comparison.csv")
    assert df.loc[0, "u_bias_ms"] == pytest.approx(0.0, abs=1e-6)


def test_alternate_missing_station_raises(tmp_path):
    ours, legacy, *_ = _setup(tmp_path)
    alt = tmp_path / "qa_alt_empty"
    (alt / "stations").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="t1-01"):
        C.substitute_alternates(C.scan_ours(ours), alt, ["t1-01"], "x")

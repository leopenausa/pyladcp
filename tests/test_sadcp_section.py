"""SADCP section plots: distance axis, station marks, figure + CLI smoke."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from ladcp.io.sadcp_vmdas import SadcpDataset, save_cache
from ladcp.plots.sadcp_section import (
    _station_marks,
    along_track_km,
    main,
    sadcp_section_figure,
)

matplotlib.use("Agg")


def _synthetic_ds(nens: int = 120, nz: int = 30) -> SadcpDataset:
    t0 = np.datetime64("2025-10-03T00:00:00", "ns")
    time = t0 + np.arange(nens) * np.timedelta64(120, "s")
    lat = 61.0 + np.linspace(0, 0.5, nens)              # steady northward transect
    lon = -11.0 + np.linspace(0, 0.2, nens)
    depth = 15.0 + 8.0 * np.arange(nz)
    rng = np.random.default_rng(3)
    u = 0.2 * np.ones((nz, nens)) + rng.normal(0, 0.02, (nz, nens))
    v = -0.1 * np.ones((nz, nens)) + rng.normal(0, 0.02, (nz, nens))
    u[20:, ::7] = np.nan                                 # some missing deep cells
    return SadcpDataset(time=time, lat=lat, lon=lon, depth=depth, u=u, v=v,
                        freq_khz=150, transducer_depth=5.0, file_type="STA",
                        source="synthetic", n_files=1)


def test_along_track_km_monotonic_and_scaled():
    ds = _synthetic_ds()
    x = along_track_km(ds.lat, ds.lon)
    assert x[0] == 0.0
    assert np.all(np.diff(x) >= 0)
    # 0.5 deg lat ~ 55 km plus the lon component
    assert 50 < x[-1] < 75


def test_along_track_km_nan_nav_keeps_monotonic():
    ds = _synthetic_ds()
    lat = ds.lat.copy()
    lat[10:14] = np.nan
    x = along_track_km(lat, ds.lon)
    assert np.all(np.isfinite(x))
    assert np.all(np.diff(x) >= 0)


def test_station_marks_inside_window_only():
    ds = _synthetic_ds()
    x = np.arange(ds.n_ens, dtype=float)
    stations = [("ST-01", "2025-10-03T01:00:00"),       # inside
                ("ST-02", "2025-10-05T00:00:00")]       # after the section -> dropped
    marks = _station_marks(stations, ds, x)
    assert len(marks) == 1
    xm, label = marks[0]
    assert label == "ST-01"
    assert abs(xm - 30) <= 1                            # 1 h / 120 s = ensemble 30


@pytest.mark.parametrize("by", ["time", "distance"])
def test_section_figure_builds(tmp_path, by):
    import matplotlib.pyplot as plt
    ds = _synthetic_ds()
    out = tmp_path / f"sec_{by}.png"
    fig = sadcp_section_figure(ds, by=by, stations=[("ST-01", "2025-10-03T01:00:00")],
                               title="synthetic", savepath=str(out))
    assert out.exists() and out.stat().st_size > 10_000
    plt.close(fig)


def test_section_figure_rejects_empty_window():
    ds = _synthetic_ds()
    with pytest.raises(ValueError, match="fewer than 2 ensembles"):
        sadcp_section_figure(ds, time_start="2026-01-01", time_end="2026-01-02")


def test_cli_smoke_from_cache(tmp_path):
    import json
    folder = tmp_path / "DATA"
    folder.mkdir()
    save_cache(_synthetic_ds(), folder / "sadcp_cache.npz")
    index = {"casts": {"ST-01": {"utc": "2025-10-03T01:00:00"}}}
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps(index))
    out = tmp_path / "sec.png"
    rc = main(["--sadcp", str(folder), "--by", "distance", "--index", str(idx),
               "-o", str(out)])
    assert rc == 0 and out.exists()

"""Shared fixtures — synthetic solved-station objects for the export tests (#3).

Pure in-memory :class:`~ladcp.export.StationExport` builders so the export-format tests
need no MORIA fixtures and run anywhere.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ladcp.export import StationExport
from ladcp.models import Metric, QCMetrics, Status
from ladcp.qa.inverse import (
    BottomProfile,
    ShearProfile,
    VelocityProfile,
    VelocityResult,
)


def _make_export(station="MORIA-80", *, lat=62.16, lon=-11.53, nz=20,
                 with_sadcp=True, with_bottom=True) -> StationExport:
    z = np.arange(10.0, 10.0 * (nz + 1), 10.0)[:nz]
    u = np.sin(z / 100.0) * 0.2
    v = np.cos(z / 100.0) * 0.2
    u[2] = np.nan                                  # a NaN bin to exercise row filtering
    v[2] = np.nan
    uerr = np.full(nz, 0.01)
    nvel = np.full(nz, 12.0)
    vp = VelocityProfile(z=z, u=u, v=v, uerr=uerr, nvel=nvel, ubar=-0.06, vbar=0.03,
                         n=nvel.copy())
    shear = ShearProfile(z=z, u=u * 0.5, v=v * 0.5, w=np.zeros(nz),
                         u_shear=np.gradient(u), v_shear=np.gradient(v), n=nvel.copy())

    bp = None
    if with_bottom:
        zb = z[: nz - 4]
        bp = BottomProfile(z=zb, u=u[: nz - 4] + 0.005, v=v[: nz - 4],
                           uerr=np.full(zb.size, 0.012), n=np.full(zb.size, 8.0))

    sadcp = None
    if with_sadcp:
        zs = np.arange(10.0, 110.0, 10.0)
        sadcp = np.column_stack([zs, np.sin(zs / 100.0) * 0.18,
                                 np.cos(zs / 100.0) * 0.18, np.full(zs.size, 0.02)])

    result = VelocityResult(vp=vp, bp=bp, shear=shear, zbottom=1057.0,
                            resid_z=z.copy(), resid_u=uerr.copy(), resid_v=uerr.copy(),
                            sadcp=sadcp)

    qc = QCMetrics(station=station)
    qc.add(Metric(name="battery", value=37.0, unit="V", status=Status.WARN,
                  threshold=40.0, note="below nominal"))
    qc.add(Metric(name="tilt_max", value=8.0, unit="deg", status=Status.OK,
                  threshold=20.0))

    return StationExport(station=station, cruise="MORIA", lat=lat, lon=lon,
                         time=datetime(2026, 6, 7, 14, 30, 0), drot=-9.878379,
                         solver="inverse", result=result, qc=qc,
                         sadcp_source="sADCP/sadcp_150/DATA" if with_sadcp else None)


@pytest.fixture
def synth_export():
    """One synthetic solved station."""
    return _make_export()


@pytest.fixture
def synth_exports():
    """Two synthetic stations with different depth grids (ragged), for cruise tests."""
    return [
        _make_export("MORIA-79", lat=61.87, lon=-10.91, nz=18, with_sadcp=False),
        _make_export("MORIA-80", lat=62.16, lon=-11.53, nz=22),
    ]

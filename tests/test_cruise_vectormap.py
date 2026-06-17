"""Plan-view depth-integrated velocity vector map (``ladcp-vectormap``)."""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.plots.cruise_section import CruiseSection
from ladcp.plots.cruise_vectormap import (
    _parse_layers,
    cruise_vectormap_figure,
    layer_reduce,
)


def _section(ns=4, nz=41):
    z = np.linspace(0, 800, nz)        # 20 m bins
    u = np.tile(np.full(nz, 0.1), (ns, 1))     # constant 0.1 m/s east everywhere
    v = np.tile(np.full(nz, 0.2), (ns, 1))     # constant 0.2 m/s north
    bottom = np.array([300.0, 500.0, 700.0, 800.0])[:ns]
    for j in range(ns):
        u[j, z > bottom[j]] = np.nan
        v[j, z > bottom[j]] = np.nan
    return CruiseSection(
        stations=np.array([f"s{j}" for j in range(ns)]),
        depth=z, u=u, v=v, uerr=np.full((ns, nz), 0.02),
        lat=np.linspace(42.2, 42.4, ns), lon=np.linspace(3.3, 3.6, ns),
        bottom_depth=bottom)


def test_parse_layers():
    assert _parse_layers(None) == [(0.0, float("inf"))]
    assert _parse_layers(["0:200", "200:800"]) == [(0.0, 200.0), (200.0, 800.0)]
    assert _parse_layers(["100:"])[0][1] == float("inf")
    with pytest.raises(ValueError):
        _parse_layers(["bad"])


def test_reduce_mean_recovers_constant():
    sec = _section()
    u, v = layer_reduce(sec, 0, 200, "mean")
    assert np.allclose(u, 0.1, atol=1e-9)
    assert np.allclose(v, 0.2, atol=1e-9)


def test_reduce_integral_is_velocity_times_thickness():
    sec = _section()
    # station 3 spans 0..800; integral of constant 0.1 over 0..200 ~ 0.1*200 = 20 m2/s
    u, _ = layer_reduce(sec, 0, 200, "integral")
    assert u[3] == pytest.approx(0.1 * 200.0, abs=0.5)


def test_layer_below_seabed_is_nan():
    sec = _section()
    # station 0 seabed at 300 m -> the 400:800 layer is entirely below it
    u, v = layer_reduce(sec, 400, 800, "mean")
    assert np.isnan(u[0]) and np.isnan(v[0])
    # station 3 (seabed 800) has data there
    assert np.isfinite(u[3])


def test_figure_builds_two_layers():
    import matplotlib
    matplotlib.use("Agg")
    sec = _section()
    fig = cruise_vectormap_figure(sec, [(0, 200), (200, 800)], reduce="mean")
    assert fig is not None
    fig2 = cruise_vectormap_figure(sec, [(0, 800)], reduce="integral")
    assert fig2 is not None

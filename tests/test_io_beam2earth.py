"""Synthetic tests for the beam->earth transform (port of legacy ``b2earth``).

Beams are constructed from a known instrument-frame velocity via the exact inverse of the
downward-convex relations, so a zero-tilt/zero-heading transform must recover that velocity;
heading then rotates it; a single nulled beam is reconstructed by the error-velocity=0 closure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ladcp.io.beam2earth import beam_to_earth

_THETA = 20.0  # Workhorse beam angle [deg]


def _down_beams(vx, vy, vz, ve, theta_deg=_THETA):
    """Exact inverse of the downward-convex instrument-frame relations -> 4 beam velocities."""
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    b1 = c * (vz + ve) + s * vx
    b2 = c * (vz + ve) - s * vx
    b3 = c * (vz - ve) - s * vy
    b4 = c * (vz - ve) + s * vy
    return np.array([b1, b2, b3, b4], dtype=float)


def _field(beams, ncell=3, nt=5):
    """Broadcast a 4-vector of beam velocities to a [4, ncell, nt] field."""
    return np.repeat(np.repeat(beams[:, None, None], ncell, 1), nt, 2).astype(float)


def _zeros(nt=5):
    return np.zeros(nt)


def test_identity_recovers_instrument_velocity():
    """Zero tilt + zero heading -> earth equals the instrument-frame velocity it was built from."""
    vx, vy, vz, ve = 0.30, -0.20, 0.05, 0.0
    vel = _field(_down_beams(vx, vy, vz, ve))
    earth, n3 = beam_to_earth(vel, _zeros(), _zeros(), _zeros(),
                              beam_angle_deg=_THETA, beams_up=False)
    assert np.allclose(earth[0], vx) and np.allclose(earth[1], vy)
    assert np.allclose(earth[2], vz) and np.allclose(earth[3], ve, atol=1e-12)
    assert n3 == 0


def test_hardcoded_unit_east():
    """A pure +x instrument flow with identity attitude -> earth (1, 0, 0, 0)."""
    vel = _field(_down_beams(1.0, 0.0, 0.0, 0.0))
    earth, _ = beam_to_earth(vel, _zeros(), _zeros(), _zeros(),
                             beam_angle_deg=_THETA, beams_up=False)
    assert np.allclose([earth[0].mean(), earth[1].mean(), earth[2].mean()], [1.0, 0.0, 0.0])


def test_heading_90_rotates_horizontal():
    """Heading 90 deg maps instrument (VX,VY)=(1,0) to earth (0,-1) (VXE=VY, VYE=-VX)."""
    vel = _field(_down_beams(1.0, 0.0, 0.0, 0.0))
    head = np.full(5, 90.0)
    earth, _ = beam_to_earth(vel, head, _zeros(), _zeros(),
                             beam_angle_deg=_THETA, beams_up=False)
    assert np.allclose(earth[0], 0.0, atol=1e-12)
    assert np.allclose(earth[1], -1.0)
    # horizontal speed is preserved under rotation
    assert np.allclose(np.hypot(earth[0], earth[1]), 1.0)


def test_3beam_closure_recovers_full_solution():
    """With VE=0 the 4-beam set is consistent; nulling any one beam is reconstructed exactly."""
    vel4 = _field(_down_beams(0.25, 0.10, -0.03, 0.0), ncell=2, nt=4)
    earth4, _ = beam_to_earth(vel4, _zeros(4), _zeros(4), _zeros(4),
                              beam_angle_deg=_THETA, beams_up=False)
    for k in range(4):
        vel3 = vel4.copy()
        vel3[k] = np.nan                                   # drop one beam everywhere
        earth3, n3 = beam_to_earth(vel3, _zeros(4), _zeros(4), _zeros(4),
                                   beam_angle_deg=_THETA, beams_up=False)
        assert np.allclose(earth3[:3], earth4[:3], atol=1e-9)
        assert n3 == vel4.shape[1] * vel4.shape[2]         # every cell reconstructed


def test_two_missing_beams_yield_nan():
    vel = _field(_down_beams(0.2, 0.0, 0.0, 0.0), ncell=2, nt=3)
    vel[0] = np.nan
    vel[1] = np.nan
    earth, n3 = beam_to_earth(vel, _zeros(3), _zeros(3), _zeros(3),
                              beam_angle_deg=_THETA, beams_up=False)
    assert np.isnan(earth[:3]).all()
    assert n3 == 0


def test_up_and_down_differ():
    """The convex up/down branches give different earth velocities for identical beams."""
    vel = _field(_down_beams(0.2, 0.1, 0.05, 0.0))
    e_down, _ = beam_to_earth(vel, _zeros(), _zeros(), _zeros(),
                              beam_angle_deg=_THETA, beams_up=False)
    e_up, _ = beam_to_earth(vel, _zeros(), _zeros(), _zeros(),
                            beam_angle_deg=_THETA, beams_up=True)
    assert not np.allclose(e_down[:3], e_up[:3])


_CRUISE2 = Path("CRUISE2_raw") / "MA001000.000"


@pytest.mark.skipif(not _CRUISE2.exists(), reason="CRUISE2 raw not on disk")
def test_read_pd0_flips_beam_to_earth():
    """A real beam-coordinate file ingests as EARTH with finite, physical velocities."""
    from ladcp.io.pd0 import read_pd0
    from ladcp.models import CoordFrame

    d = read_pd0(str(_CRUISE2))
    assert d.coord_frame == CoordFrame.EARTH
    assert d.meta["beam_angle_deg"] == 20
    assert d.meta["beam2earth"]["n_3beam"] > 0
    # tiny error velocity + small median w are the signature of a correct 4-beam transform
    assert np.nanmedian(np.abs(d.vel[3])) < 0.1
    assert abs(np.nanmedian(d.vel[2])) < 0.1

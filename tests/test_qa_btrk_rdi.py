"""RDI-firmware bottom-track preference (legacy btrk_mode=3 semantics).

The own water-track BT samples a near-seabed water cell and inherits
boundary-layer flow (~10-15% of the current leaks into the package velocity on
strong-drift casts -- the FDCCC1 t1-01/t1-02/t99-03 whole-profile offsets).
When the PD0 carries firmware BT pings they are the velocity source.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ladcp.qa.bottom import _RDI_BT_ELIM, _RDI_BT_VLIM, _rdi_bottom_track


def _dh(bt_vel, bt_range=None):
    return SimpleNamespace(down=SimpleNamespace(bt_vel=bt_vel, bt_range=bt_range))


def test_no_firmware_bt_returns_none():
    assert _rdi_bottom_track(_dh(None), 5) is None


def test_clean_track_is_used_with_ranges():
    n = 8
    btv = np.full((4, n), np.nan)
    btv[0] = 0.10            # u
    btv[1] = -0.05           # v
    btv[2] = 0.01            # w
    btv[3] = 0.02            # error velocity, fine
    btr = np.full((4, n), 100.0)
    out = _rdi_bottom_track(_dh(btv, btr), n)
    assert out is not None
    rvel, rw, rhbot = out
    assert np.isfinite(rvel).sum() == n
    np.testing.assert_allclose(np.real(rvel), 0.10)
    np.testing.assert_allclose(rhbot, 100.0)


def test_screening_drops_bad_samples():
    n = 10
    btv = np.zeros((4, n))
    btv[0] = 0.1
    btv[3, 1] = _RDI_BT_ELIM * 2          # error velocity too big
    btv[0, 2] = _RDI_BT_VLIM * 2          # implausible speed
    btv[0, 3] = -32.768                   # firmware sentinel
    btv[0, 4] = np.nan                    # no lock
    out = _rdi_bottom_track(_dh(btv), n)
    assert out is not None
    rvel, _, _ = out
    assert np.isfinite(rvel[0]) and np.isfinite(rvel[5:]).all()
    assert not np.isfinite(rvel[1:5]).any()


def test_too_few_finite_falls_back_to_none():
    n = 6
    btv = np.full((4, n), np.nan)
    btv[:, 0] = [0.1, 0.0, 0.0, 0.0]      # only 1 usable sample (< 4 minimum)
    assert _rdi_bottom_track(_dh(btv), n) is None

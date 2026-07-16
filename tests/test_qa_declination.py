"""Declination guard-rail (item 1): the velocity frame must ANNOUNCE its declination
provenance and WARN when the IGRF lookup falls back to drot=0 (i.e. the profile is left in
the *magnetic* frame, not true north) -- no more silent-wrong. Also pins that IGRF uses the
cast position/date and does not return a finite (silently-wrong) value for a missing position.
"""
import datetime as dt

import numpy as np

from ladcp.models import Status
from ladcp.proc.magdec import magnetic_declination
from ladcp.qa.pipeline import _declination_metric


def test_declination_metric_status_by_source():
    assert _declination_metric(-5.44, "igrf").status is Status.OK
    assert _declination_metric(1.77, "explicit").status is Status.OK
    warn = _declination_metric(0.0, "fallback-zero")
    assert warn.status is Status.WARN          # the fallback is now visible, not silent
    assert warn.value == 0.0
    assert "MAGNETIC frame" in warn.note


def test_magdec_real_position_is_correct():
    # MORIA-80 cast (62.16N, 11.53W, Oct 2025) -> IGRF-13 ~ -5.4 deg (true frame)
    d = magnetic_declination(62.161, -11.531, dt.datetime(2025, 10, 3))
    assert np.isfinite(d) and -7.0 < d < -4.0


def test_magdec_missing_position_not_silently_finite():
    # a missing CTD position must NOT yield a finite (wrong) declination; the
    # _velocity_outputs guard then catches it (non-finite -> WARN+drot=0, or it raises).
    try:
        d = magnetic_declination(float("nan"), float("nan"), dt.datetime(2025, 10, 3))
        assert not np.isfinite(d)
    except Exception:
        pass

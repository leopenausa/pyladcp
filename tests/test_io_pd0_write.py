"""Round-trip tests for the PD0 writer: write_head_pd0 -> read_pd0 recovers fields.

This is the byte-accuracy gate for the synthetic-dataset generator: every field the
reader decodes must survive a write/read cycle (within the format's quantisation).
"""

from __future__ import annotations

import numpy as np

from ladcp.io.pd0 import read_pd0
from ladcp.io.pd0_write import write_head_pd0
from ladcp.models import CoordFrame, RawADCP


def _make_head(*, facing="down", n_cells=4, n_ens=3, with_bt=True) -> RawADCP:
    rng = np.random.default_rng(0)
    t0 = np.datetime64("2025-10-03T06:30:00.000")
    time = t0 + np.arange(n_ens) * np.timedelta64(1500, "ms")  # 1.5 s, exercises hundredths
    vel = rng.normal(0, 0.2, (4, n_cells, n_ens))
    vel[3] = rng.normal(0, 0.01, (n_cells, n_ens))             # error velocity small
    vel[:, -1, 0] = np.nan                                      # a bad cell -> sentinel
    echo = rng.integers(40, 120, (4, n_cells, n_ens)).astype(float)
    corr = rng.integers(60, 130, (4, n_cells, n_ens)).astype(float)
    pctg = np.full((4, n_cells, n_ens), 100.0)
    bt_range = bt_vel = None
    if with_bt:
        bt_range = np.full((4, n_ens), np.nan)
        bt_vel = np.full((4, n_ens), np.nan)
        bt_range[:, 1] = 950.0 + np.arange(4)        # one ping sees the bottom (per-beam)
        bt_vel[:, 1] = [0.03, -0.02, 0.0, 0.0]
    return RawADCP(
        head=facing, coord_frame=CoordFrame.EARTH, facing=facing, freq_khz=150,
        n_beams=4, n_cells=n_cells, cell_m=8.0, blank_m=4.0, wm_mode=15, pings_per_ens=1,
        serial=12345, time=time, vel=vel, echo=echo, corr=corr, pct_good=pctg,
        heading=np.array([200.0, 200.5, 201.0]), pitch=np.array([1.0, -1.5, 0.5]),
        roll=np.array([-0.5, 0.5, 1.0]), temperature=np.array([10.1, 10.2, 10.3]),
        sound_speed=np.array([1500.0, 1501.0, 1499.0]), salinity=np.full(n_ens, 35.0),
        xmit_voltage=np.full(n_ens, 120.0),
        bt_range=bt_range, bt_vel=bt_vel,
        meta={"dist_first_m": 12.0, "beam_angle_deg": 20},
    )


def test_roundtrip_geometry_and_frame(tmp_path):
    h = _make_head(facing="down")
    p = tmp_path / "down.000"
    write_head_pd0(str(p), h)
    r = read_pd0(str(p), facing_hint="down")

    assert r.coord_frame == CoordFrame.EARTH
    assert r.n_cells == h.n_cells
    assert r.n_beams == h.n_beams
    assert r.freq_khz == 150
    assert r.facing == "down"
    assert r.cell_m == h.cell_m
    assert r.blank_m == h.blank_m
    assert r.serial == 12345
    assert r.pings_per_ens == 1
    assert r.wm_mode == 15
    assert r.meta["dist_first_m"] == 12.0
    assert r.meta["beam_angle_deg"] == 20


def test_roundtrip_velocity_and_counts(tmp_path):
    h = _make_head()
    p = tmp_path / "down.000"
    write_head_pd0(str(p), h)
    r = read_pd0(str(p), facing_hint="down")

    # velocity within mm/s quantisation; NaN preserved
    finite = np.isfinite(h.vel)
    assert np.allclose(r.vel[finite], h.vel[finite], atol=1e-3)
    assert np.array_equal(np.isnan(r.vel), np.isnan(h.vel))
    # echo / correlation / percent-good are integer counts -> exact
    assert np.array_equal(r.echo, h.echo)
    assert np.array_equal(r.corr, h.corr)
    assert np.array_equal(r.pct_good, h.pct_good)


def test_roundtrip_attitude_and_time(tmp_path):
    h = _make_head()
    p = tmp_path / "down.000"
    write_head_pd0(str(p), h)
    r = read_pd0(str(p), facing_hint="down")

    assert np.allclose(r.heading, h.heading, atol=0.01)
    assert np.allclose(r.pitch, h.pitch, atol=0.01)
    assert np.allclose(r.roll, h.roll, atol=0.01)
    assert np.allclose(r.temperature, h.temperature, atol=0.01)
    assert np.allclose(r.sound_speed, h.sound_speed, atol=1.0)
    # time to the hundredth of a second
    dt = (r.time - h.time) / np.timedelta64(1, "ms")
    assert np.all(np.abs(dt) <= 10)


def test_roundtrip_bottom_track(tmp_path):
    h = _make_head(with_bt=True)
    p = tmp_path / "down.000"
    write_head_pd0(str(p), h)
    r = read_pd0(str(p), facing_hint="down")

    assert r.bt_range is not None and r.bt_vel is not None
    finite = np.isfinite(h.bt_range)
    assert np.allclose(r.bt_range[finite], h.bt_range[finite], atol=0.01)
    assert np.array_equal(np.isnan(r.bt_range), np.isnan(h.bt_range))
    bvf = np.isfinite(h.bt_vel)
    assert np.allclose(r.bt_vel[bvf], h.bt_vel[bvf], atol=1e-3)


def test_roundtrip_up_facing_sysconfig(tmp_path):
    h = _make_head(facing="up")
    p = tmp_path / "up.000"
    write_head_pd0(str(p), h)
    # no hint: facing must come from the sysconfig bit-7 we wrote
    r = read_pd0(str(p))
    assert r.meta["facing_from_bit"] == "up"
    assert r.meta["beam_angle_deg"] == 20  # beam angle independent of facing bit

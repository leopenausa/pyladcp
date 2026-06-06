"""VmDAS shipboard-ADCP ingest (ladcp.io.sadcp_vmdas).

The reader recovers absolute ocean velocity (water-track + ship velocity from the GPS
track) from VmDAS STA/LTA, caches the folder once, and windows it to a cast's ``svel``.
The deterministic tests run on a hand-built two-/three-ensemble PD0+nav buffer; an
optional real-data test exercises the MORIA tree when it is present.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import pytest

from ladcp.io.sadcp_vmdas import (
    extract_profile,
    ingest_dir,
    load_cache,
    load_or_ingest,
    read_vmdas_file,
    save_cache,
)

_BAM_INV = 2**31 / 180.0
N_CELLS = 5
CELL_M = 10.0
DIST1_M = 10.0
XDUCER = 5.0


def _datatype(type_id: int, length: int, fields: dict[int, bytes]) -> bytearray:
    """A PD0 data type: id in the first two bytes, ``fields`` written at given offsets."""
    b = bytearray(length)
    struct.pack_into("<H", b, 0, type_id)
    for off, raw in fields.items():
        b[off:off + len(raw)] = raw
    return b


def _fixed_leader() -> bytearray:
    f = {
        4: struct.pack("<H", 1),               # sysconfig: 150 kHz, downlooker
        8: bytes([4]),                          # beams
        9: bytes([N_CELLS]),                    # cells
        10: struct.pack("<H", 1),               # pings/ens
        12: struct.pack("<H", int(CELL_M * 100)),
        14: struct.pack("<H", 400),             # blank
        25: bytes([3 << 3]),                    # EX byte -> earth coordinates
        32: struct.pack("<H", int(DIST1_M * 100)),
        54: struct.pack("<I", 12345),           # serial
    }
    return _datatype(0x0000, 75, f)


def _variable_leader(when: np.datetime64) -> bytearray:
    dt = when.astype("datetime64[s]").item()
    f = {
        4: bytes([dt.year - 2000, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0]),
        14: struct.pack("<H", 1500),            # sound speed
        18: struct.pack("<H", 0),               # heading
    }
    return _datatype(0x0080, 40, f)


def _velocity(u_mm: np.ndarray, v_mm: np.ndarray) -> bytearray:
    """Earth velocity block: per cell [east, north, up, err] int16 mm/s."""
    arr = np.zeros((N_CELLS, 4), dtype="<i2")
    arr[:, 0] = u_mm
    arr[:, 1] = v_mm
    return _datatype(0x0100, 2 + N_CELLS * 4 * 2, {2: arr.tobytes()})


def _pct_good(good_mask: np.ndarray) -> bytearray:
    """Percent-good block: field index 3 (4-beam) carries the screen value per cell."""
    arr = np.zeros((N_CELLS, 4), dtype="<u1")
    arr[:, 3] = np.where(good_mask, 100, 0)
    return _datatype(0x0400, 2 + N_CELLS * 4, {2: arr.tobytes()})


def _nav(lat: float, lon: float) -> bytearray:
    raw = {
        14: struct.pack("<i", round(lat * _BAM_INV)),
        18: struct.pack("<i", round(lon * _BAM_INV)),
        26: struct.pack("<i", round(lat * _BAM_INV)),
        30: struct.pack("<i", round(lon * _BAM_INV)),
    }
    return _datatype(0x2000, 40, raw)


def _ensemble(when, lat, lon, u_mm, v_mm, good_mask) -> bytes:
    parts = [_fixed_leader(), _variable_leader(when), _velocity(u_mm, v_mm),
             _pct_good(good_mask), _nav(lat, lon)]
    ndt = len(parts)
    header_len = 6 + 2 * ndt
    offsets, pos = [], header_len
    for p in parts:
        offsets.append(pos)
        pos += len(p)
    ens_size = pos
    buf = bytearray(ens_size + 2)
    buf[0] = buf[1] = 0x7F
    struct.pack_into("<H", buf, 2, ens_size)
    buf[5] = ndt
    for k, off in enumerate(offsets):
        struct.pack_into("<H", buf, 6 + 2 * k, off)
    pos = header_len
    for p in parts:
        buf[pos:pos + len(p)] = p
        pos += len(p)
    struct.pack_into("<H", buf, ens_size, sum(buf[:ens_size]) & 0xFFFF)
    return bytes(buf)


def _write_sta(path, *, lat0=62.0, lon0=-11.0, ship_east=1.0, n=3, dt_s=120,
               u_water=0.10, v_water=-0.20, bad_cell=None):
    """Write an n-ensemble STA file: constant water velocity, ship moving east at
    ``ship_east`` m/s (so absolute = water + ship). ``bad_cell`` gets zero percent-good."""
    t0 = np.datetime64("2025-10-03T06:30:00")
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    good = np.ones(N_CELLS, bool)
    if bad_cell is not None:
        good[bad_cell] = False
    blob = b""
    for i in range(n):
        when = t0 + np.timedelta64(i * dt_s, "s")
        lon = lon0 + (ship_east * i * dt_s) / m_per_deg_lon
        u_mm = np.full(N_CELLS, int(u_water * 1000))
        v_mm = np.full(N_CELLS, int(v_water * 1000))
        blob += _ensemble(when, lat0, lon, u_mm, v_mm, good)
    path.write_bytes(blob)
    return t0, t0 + np.timedelta64((n - 1) * dt_s, "s")


# --------------------------------------------------------------------------- #
# synthetic-buffer tests
# --------------------------------------------------------------------------- #
def test_read_vmdas_file_decodes_nav_and_geometry(tmp_path):
    f = tmp_path / "syn.STA"
    _write_sta(f)
    time, lat, lon, depth, uw, vw, freq = read_vmdas_file(str(f), transducer_depth=XDUCER)
    assert freq == 150
    assert time.size == 3
    np.testing.assert_allclose(lat, 62.0, atol=1e-4)
    # depth = xducer + range-to-bin1 + k*cell
    np.testing.assert_allclose(depth, XDUCER + DIST1_M + np.arange(N_CELLS) * CELL_M)
    # water-track velocity is the (ship-relative) value we wrote
    np.testing.assert_allclose(np.nanmean(uw), 0.10, atol=1e-3)


def test_ingest_adds_ship_velocity_to_get_absolute(tmp_path):
    f = tmp_path / "syn.STA"
    _write_sta(f, ship_east=1.0, u_water=0.10, v_water=-0.20)
    ds = ingest_dir(tmp_path, file_type="STA", transducer_depth=XDUCER)
    # absolute east = water (0.10) + ship east (1.0); north = water (-0.20) + 0
    np.testing.assert_allclose(np.nanmean(ds.u), 1.10, atol=2e-3)
    np.testing.assert_allclose(np.nanmean(ds.v), -0.20, atol=2e-3)


def test_extract_profile_window_and_average(tmp_path):
    f = tmp_path / "syn.STA"
    t0, t1 = _write_sta(f, ship_east=1.0)
    ds = ingest_dir(tmp_path, transducer_depth=XDUCER)
    sv = extract_profile(ds, time_start=t0, time_end=t1, lat=62.0, lon=-11.0)
    assert sv is not None and sv.shape == (N_CELLS, 4)
    np.testing.assert_allclose(sv[:, 1], 1.10, atol=2e-3)     # u
    np.testing.assert_allclose(sv[:, 2], -0.20, atol=2e-3)    # v
    assert np.all(sv[:, 3] > 0)                               # verr floored, positive


def test_position_gate_rejects_far_station(tmp_path):
    f = tmp_path / "syn.STA"
    t0, t1 = _write_sta(f, lat0=62.0, lon0=-11.0)
    ds = ingest_dir(tmp_path, transducer_depth=XDUCER)
    # > 0.1 deg away -> rejected
    assert extract_profile(ds, time_start=t0, time_end=t1, lat=55.0, lon=-3.0) is None


def test_empty_time_window_returns_none(tmp_path):
    f = tmp_path / "syn.STA"
    _write_sta(f)
    ds = ingest_dir(tmp_path, transducer_depth=XDUCER)
    assert extract_profile(ds, time_start="2024-01-01T00:00",
                           time_end="2024-01-01T01:00", lat=62.0, lon=-11.0) is None


def test_pctgood_screen_blanks_bad_cell(tmp_path):
    f = tmp_path / "syn.STA"
    _write_sta(f, bad_cell=2)
    _, _, _, _, uw, _, _ = read_vmdas_file(str(f), transducer_depth=XDUCER,
                                           pctgood_min=25.0)
    assert np.all(np.isnan(uw[2]))           # screened cell
    assert np.all(np.isfinite(uw[0]))        # neighbours kept


def test_cache_roundtrip_and_reuse(tmp_path):
    f = tmp_path / "syn.STA"
    _write_sta(f)
    ds = ingest_dir(tmp_path, transducer_depth=XDUCER)
    cache = tmp_path / "c.npz"
    save_cache(ds, cache)
    back = load_cache(cache)
    np.testing.assert_array_equal(ds.time, back.time)
    np.testing.assert_allclose(np.nan_to_num(ds.u), np.nan_to_num(back.u))
    assert back.freq_khz == ds.freq_khz and back.file_type == ds.file_type
    # load_or_ingest reuses the cache rather than re-parsing
    reused = load_or_ingest(tmp_path, cache=cache)
    np.testing.assert_array_equal(reused.time, ds.time)


# --------------------------------------------------------------------------- #
# real-data test (skipped unless the MORIA SADCP tree is present)
# --------------------------------------------------------------------------- #
_REAL = pathlib.Path(__file__).resolve().parents[1] / "sADCP" / "sadcp_150" / "DATA"


@pytest.mark.skipif(not _REAL.exists(), reason="MORIA sADCP raw tree not present")
def test_real_moria_sadcp_matches_ladcp_surface():
    """On MORIA-80 the absolute SADCP surface velocity must agree with the LADCP golden."""
    import scipy.io as sio

    ds = load_or_ingest(_REAL, file_type="STA", transducer_depth=5.0)
    sv = extract_profile(ds, time_start="2025-10-03T06:27:12",
                         time_end="2025-10-03T07:21:39", lat=62.16135, lon=-11.53085)
    assert sv is not None and sv.shape[0] > 20
    fig = pathlib.Path(__file__).resolve().parents[1] / "figures" / "MORIA-80.mat"
    if not fig.exists():
        return
    dr = sio.loadmat(str(fig), squeeze_me=True, struct_as_record=False)["dr"]
    z, u, v = np.atleast_1d(dr.z), np.atleast_1d(dr.u), np.atleast_1d(dr.v)
    upper = sv[sv[:, 0] <= 300]
    lu = np.interp(upper[:, 0], z, u)
    lv = np.interp(upper[:, 0], z, v)
    assert np.nanmean(np.abs(upper[:, 1] - lu)) < 0.08      # east agrees to <0.08 m/s
    assert np.nanmean(np.abs(upper[:, 2] - lv)) < 0.08

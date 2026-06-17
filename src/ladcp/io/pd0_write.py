"""Writer for RDI Workhorse PD0 binary files (one head per file).

The inverse of :mod:`ladcp.io.pd0`: serialises a :class:`~ladcp.models.RawADCP`
(or a hand-built ensemble stream) back to a real PD0 byte stream that
``read_pd0`` round-trips. Emits fixed leader, variable leader, velocity,
correlation, echo intensity, percent-good, and bottom-track data types.

Byte offsets here are the exact mirror of the reader (``ladcp.io.pd0`` is the
spec). Velocity is written in EARTH frame (components ``[east, north, up, err]``),
which is what the velocity solve requires; the reader applies no rotation when the
fixed-leader EX byte reports earth coordinates.

This is the foundation for the synthetic-dataset generator (:mod:`ladcp.synth`),
which forward-models a known ocean profile into these blocks.
"""

from __future__ import annotations

import struct

import numpy as np

from ..models import CoordFrame, RawADCP

# PD0 data-type IDs (mirror ladcp.io.pd0)
ID_FIXED = 0x0000
ID_VARIABLE = 0x0080
ID_VELOCITY = 0x0100
ID_CORRELATION = 0x0200
ID_ECHO = 0x0300
ID_PCT_GOOD = 0x0400
ID_BOTTOM_TRACK = 0x0600

_BADVEL = -32768  # RDI bad-velocity sentinel (mm/s)

# inverse of the reader's decode tables
_FREQ_CODE = {75: 0, 150: 1, 300: 2, 600: 3, 1200: 4, 2400: 5}
_BEAM_CODE = {15: 0, 20: 1, 30: 2, 0: 3}
_COORD_CODE = {CoordFrame.BEAM: 0, CoordFrame.INSTRUMENT: 1,
               CoordFrame.SHIP: 2, CoordFrame.EARTH: 3}

# fixed block lengths (>= the highest offset each block touches)
_FL_LEN = 59       # serial at +54..+57
_VL_LEN = 65       # standard variable-leader length; ADC ch1 at +35
_BT_LEN = 85       # BT range MSB at +77..+80


def _datatype(type_id: int, length: int, fields: dict[int, bytes]) -> bytearray:
    """A PD0 data type: id in the first two bytes, ``fields`` written at offsets."""
    b = bytearray(length)
    struct.pack_into("<H", b, 0, type_id)
    for off, raw in fields.items():
        b[off:off + len(raw)] = raw
    return b


def fixed_leader(*, freq_khz: int, n_beams: int, n_cells: int, cell_m: float,
                 blank_m: float, pings_per_ens: int, wm_mode: int,
                 dist_first_m: float, beam_angle_deg: int, facing: str,
                 serial: int | None, coord: CoordFrame = CoordFrame.EARTH) -> bytearray:
    """Build the fixed-leader block (system geometry + coordinate frame)."""
    freq_code = _FREQ_CODE.get(int(freq_khz), 1)
    beam_code = _BEAM_CODE.get(int(beam_angle_deg), 1)
    sysconf = freq_code | (0x80 if facing == "up" else 0) | (beam_code << 8)
    fields = {
        4: struct.pack("<H", sysconf),
        8: bytes([n_beams]),
        9: bytes([n_cells]),
        10: struct.pack("<H", int(pings_per_ens)),
        12: struct.pack("<H", round(cell_m * 100)),
        14: struct.pack("<H", round(blank_m * 100)),
        16: bytes([wm_mode & 0xFF]),
        25: bytes([_COORD_CODE[coord] << 3]),
        32: struct.pack("<H", round(dist_first_m * 100)),
        54: struct.pack("<I", int(serial) if serial else 0),
    }
    return _datatype(ID_FIXED, _FL_LEN, fields)


def variable_leader(when: np.datetime64, *, sound_speed: float, heading: float,
                    pitch: float, roll: float, salinity: float, temperature: float,
                    xmit_voltage: float | None) -> bytearray:
    """Build the variable-leader block (time, attitude, environment)."""
    dt = when.astype("datetime64[s]").item()
    hs = int(round((when - when.astype("datetime64[s]")) / np.timedelta64(10, "ms")))
    fields = {
        4: bytes([dt.year - 2000, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                  max(0, min(99, hs))]),
        14: struct.pack("<H", round(sound_speed)),
        18: struct.pack("<H", round(heading * 100) % 36000),
        20: struct.pack("<h", round(pitch * 100)),
        22: struct.pack("<h", round(roll * 100)),
        24: struct.pack("<H", round(salinity)),
        26: struct.pack("<h", round(temperature * 100)),
        35: bytes([int(np.clip(xmit_voltage if xmit_voltage is not None else 0, 0, 255))]),
    }
    return _datatype(ID_VARIABLE, _VL_LEN, fields)


def _to_i2_mm(comp: np.ndarray) -> np.ndarray:
    """Cell-major (n_cells, 4) earth velocity [m/s] -> int16 mm/s with bad sentinel."""
    mm = np.where(np.isfinite(comp), np.round(comp * 1000.0), _BADVEL)
    mm = np.clip(mm, _BADVEL, 32767)
    return mm.astype("<i2")


def velocity(uvwe: np.ndarray) -> bytearray:
    """Velocity block from a (n_cells, 4) earth array [east, north, up, err] m/s."""
    n_cells = uvwe.shape[0]
    arr = _to_i2_mm(uvwe)
    return _datatype(ID_VELOCITY, 2 + n_cells * 4 * 2, {2: arr.tobytes()})


def _u1_block(type_id: int, counts: np.ndarray) -> bytearray:
    """Echo/correlation/percent-good block from a (n_cells, 4) count array."""
    n_cells = counts.shape[0]
    arr = np.clip(np.nan_to_num(counts, nan=0.0), 0, 255).astype("<u1")
    return _datatype(type_id, 2 + n_cells * 4, {2: arr.tobytes()})


def bottom_track(rng_m: np.ndarray, bvel: np.ndarray) -> bytearray:
    """Bottom-track block: per-beam range [m] (4,) and velocity [m/s] (4,).

    A non-finite or zero range writes range 0, which the reader decodes as "no
    bottom" (NaN). Velocity is earth-frame mm/s with the bad sentinel.
    """
    rng_cm = np.where(np.isfinite(rng_m), np.round(np.asarray(rng_m) * 100.0), 0.0)
    rng_cm = np.clip(rng_cm, 0, None).astype(np.int64)
    lsb = (rng_cm % 65536).astype("<u2")
    msb = (rng_cm // 65536).astype("<u1")
    bv = _to_i2_mm(np.asarray(bvel, float)[:, None])[:, 0]
    return _datatype(ID_BOTTOM_TRACK, _BT_LEN,
                     {16: lsb.tobytes(), 24: bv.tobytes(), 77: msb.tobytes()})


def build_ensemble(blocks: list[bytearray]) -> bytes:
    """Assemble data-type blocks into one PD0 ensemble (header + offsets + checksum)."""
    ndt = len(blocks)
    header_len = 6 + 2 * ndt
    offsets, pos = [], header_len
    for b in blocks:
        offsets.append(pos)
        pos += len(b)
    ens_size = pos
    buf = bytearray(ens_size + 2)
    buf[0] = buf[1] = 0x7F
    struct.pack_into("<H", buf, 2, ens_size)
    buf[5] = ndt
    for k, off in enumerate(offsets):
        struct.pack_into("<H", buf, 6 + 2 * k, off)
    pos = header_len
    for b in blocks:
        buf[pos:pos + len(b)] = b
        pos += len(b)
    struct.pack_into("<H", buf, ens_size, sum(buf[:ens_size]) & 0xFFFF)
    return bytes(buf)


def write_pd0(path: str, ensembles: list[bytes]) -> None:
    """Write a list of assembled ensembles to a PD0 file."""
    with open(path, "wb") as fh:
        for ens in ensembles:
            fh.write(ens)


def write_head_pd0(path: str, head: RawADCP) -> None:
    """Serialise a single-head :class:`RawADCP` (EARTH frame) to a PD0 file.

    Every ensemble carries fixed/variable leaders, velocity, correlation, echo,
    percent-good, and (when ``head.bt_range`` is present) a bottom-track block.
    """
    if head.coord_frame != CoordFrame.EARTH:
        raise ValueError(f"write_head_pd0 expects EARTH frame, got {head.coord_frame.value!r}")
    beam_angle = int(head.meta.get("beam_angle_deg", 20))
    has_bt = head.bt_range is not None and head.bt_vel is not None
    ensembles = []
    for it in range(head.n_ens):
        fl = fixed_leader(
            freq_khz=head.freq_khz, n_beams=head.n_beams, n_cells=head.n_cells,
            cell_m=head.cell_m, blank_m=head.blank_m, pings_per_ens=head.pings_per_ens,
            wm_mode=head.wm_mode,
            dist_first_m=float(head.meta.get("dist_first_m", head.blank_m + head.cell_m)),
            beam_angle_deg=beam_angle, facing=head.facing, serial=head.serial,
            coord=CoordFrame.EARTH)
        xv = None if head.xmit_voltage is None else head.xmit_voltage[it]
        vl = variable_leader(
            head.time[it], sound_speed=_safe(head.sound_speed[it], 1500.0),
            heading=_safe(head.heading[it], 0.0), pitch=_safe(head.pitch[it], 0.0),
            roll=_safe(head.roll[it], 0.0), salinity=_safe(head.salinity[it], 35.0),
            temperature=_safe(head.temperature[it], 10.0), xmit_voltage=xv)
        blocks = [
            fl, vl,
            velocity(head.vel[:, :, it].T),
            _u1_block(ID_CORRELATION, head.corr[:, :, it].T),
            _u1_block(ID_ECHO, head.echo[:, :, it].T),
            _u1_block(ID_PCT_GOOD, head.pct_good[:, :, it].T),
        ]
        if has_bt:
            blocks.append(bottom_track(head.bt_range[:, it], head.bt_vel[:, it]))
        ensembles.append(build_ensemble(blocks))
    write_pd0(path, ensembles)


def _safe(x: float, default: float) -> float:
    return float(x) if np.isfinite(x) else default

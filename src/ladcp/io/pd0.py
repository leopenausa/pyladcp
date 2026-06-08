"""Reader for RDI Workhorse PD0 binary files (LADCP, one head per file).

Parses fixed leader, variable leader, velocity, correlation, echo intensity, percent
good, and bottom-track data types. Auto-detects the coordinate frame (beam vs earth)
from the fixed-leader EX byte, matching what the LDEO_IX ``loadrdi.m`` relies on.

Reference: Teledyne RDI "WorkHorse Commands and Output Data Format" PD0 structure.
"""

from __future__ import annotations

import re
import struct

import numpy as np

from ..models import CoordFrame, RawADCP

# PD0 data-type IDs
ID_FIXED = 0x0000
ID_VARIABLE = 0x0080
ID_VELOCITY = 0x0100
ID_CORRELATION = 0x0200
ID_ECHO = 0x0300
ID_PCT_GOOD = 0x0400
ID_BOTTOM_TRACK = 0x0600

_BADVEL = -32768            # RDI bad-velocity sentinel (mm/s)
_FREQ = {0: 75, 1: 150, 2: 300, 3: 600, 4: 1200, 5: 2400}
_COORD = {0: CoordFrame.BEAM, 1: CoordFrame.INSTRUMENT, 2: CoordFrame.SHIP, 3: CoordFrame.EARTH}


def _ensemble_offsets(buf: bytes):
    """Yield (start, n_datatypes, [type_offsets]) for each ensemble header in buf."""
    n = len(buf)
    pos = 0
    while pos < n - 4:
        if buf[pos] == 0x7F and buf[pos + 1] == 0x7F:
            ens_size = struct.unpack_from("<H", buf, pos + 2)[0]
            # need ensemble + 2-byte checksum
            if pos + ens_size + 2 > n:
                break
            # validate checksum
            chk = struct.unpack_from("<H", buf, pos + ens_size)[0]
            if (sum(buf[pos:pos + ens_size]) & 0xFFFF) != chk:
                pos += 1
                continue
            ndt = buf[pos + 5]
            offs = [struct.unpack_from("<H", buf, pos + 6 + 2 * k)[0] for k in range(ndt)]
            yield pos, ndt, offs
            pos += ens_size + 2
        else:
            pos += 1


def read_pd0(path: str, head: str = "", facing_hint: str | None = None) -> RawADCP:
    """Read a single-head PD0 file into a :class:`RawADCP`."""
    with open(path, "rb") as fh:
        buf = fh.read()

    ensembles = list(_ensemble_offsets(buf))
    if not ensembles:
        raise ValueError(f"no valid PD0 ensembles found in {path}")

    # --- fixed leader from first ensemble (assumed constant config) ---
    p0, _, offs0 = ensembles[0]
    fl = _find_type(buf, p0, offs0, ID_FIXED)
    if fl is None:
        raise ValueError("no fixed leader in first ensemble")
    sysconf = struct.unpack_from("<H", buf, fl + 4)[0]
    freq = _FREQ.get(sysconf & 0x07, 0)
    # Orientation precedence (mirrors legacy loadrdi.m, which trusts the file's
    # role and uses the sysconfig bit only to warn): explicit hint > filename
    # heuristic > sysconfig bit-7. The bit alone is NOT reliable across instrument
    # families — RDI BroadBand 150 (e.g. GO-SHIP RB1606) reports bit-7=1 on both
    # the down- and up-looker, so trusting it misreads the downlooker as up.
    facing_from_bit = "up" if (sysconf & 0x80) else "down"
    facing_from_name = _facing_from_name(path)
    facing = facing_hint or facing_from_name or facing_from_bit
    facing_warning = None
    if facing != facing_from_bit:
        facing_warning = (
            f"sysconfig bit-7 indicates '{facing_from_bit}' but orientation "
            f"resolved to '{facing}' (from {'hint' if facing_hint else 'filename'})"
        )
    n_beams = buf[fl + 8]
    n_cells = buf[fl + 9]
    pings_per_ens = struct.unpack_from("<H", buf, fl + 10)[0]
    cell_m = struct.unpack_from("<H", buf, fl + 12)[0] / 100.0
    blank_m = struct.unpack_from("<H", buf, fl + 14)[0] / 100.0
    # vertical distance to centre of bin 1 (FL offset 32, cm) -> legacy p.dist_d/dist_u
    dist_first_m = struct.unpack_from("<H", buf, fl + 32)[0] / 100.0
    wm_mode = buf[fl + 16]
    ex = buf[fl + 25]
    coord_frame = _COORD.get((ex >> 3) & 0x03, CoordFrame.BEAM)

    nt = len(ensembles)
    nc = n_cells
    nb = 4  # PD0 always stores 4 beam slots

    time = np.empty(nt, dtype="datetime64[ns]")
    vel = np.full((nb, nc, nt), np.nan)
    echo = np.full((nb, nc, nt), np.nan)
    corr = np.full((nb, nc, nt), np.nan)
    pctg = np.full((nb, nc, nt), np.nan)
    heading = np.full(nt, np.nan)
    pitch = np.full(nt, np.nan)
    roll = np.full(nt, np.nan)
    temperature = np.full(nt, np.nan)
    sound_speed = np.full(nt, np.nan)
    salinity = np.full(nt, np.nan)
    xmit_v = np.full(nt, np.nan)
    bt_range = np.full((nb, nt), np.nan)
    bt_vel = np.full((nb, nt), np.nan)
    have_bt = False

    for it, (pos, _ndt, offs) in enumerate(ensembles):
        vl = _find_type(buf, pos, offs, ID_VARIABLE)
        if vl is not None:
            yr = buf[vl + 4]
            year = 2000 + yr if yr < 80 else 1900 + yr
            mo, da, hh, mm, ss, hs = (buf[vl + 5], buf[vl + 6], buf[vl + 7],
                                      buf[vl + 8], buf[vl + 9], buf[vl + 10])
            time[it] = _to_dt64(year, mo, da, hh, mm, ss, hs)
            sound_speed[it] = struct.unpack_from("<H", buf, vl + 14)[0]
            heading[it] = struct.unpack_from("<H", buf, vl + 18)[0] * 0.01
            pitch[it] = struct.unpack_from("<h", buf, vl + 20)[0] * 0.01
            roll[it] = struct.unpack_from("<h", buf, vl + 22)[0] * 0.01
            salinity[it] = struct.unpack_from("<H", buf, vl + 24)[0]
            temperature[it] = struct.unpack_from("<h", buf, vl + 26)[0] * 0.01
            # ADC channel 1 (offset 35) is transmit voltage proxy on WH
            xmit_v[it] = buf[vl + 35]

        ve = _find_type(buf, pos, offs, ID_VELOCITY)
        if ve is not None:
            raw = np.frombuffer(buf, dtype="<i2", count=nc * nb, offset=ve + 2).reshape(nc, nb).T
            v = raw.astype(float)
            v[raw == _BADVEL] = np.nan
            vel[:, :, it] = v / 1000.0  # mm/s -> m/s

        for did, arr in ((ID_CORRELATION, corr), (ID_ECHO, echo), (ID_PCT_GOOD, pctg)):
            t = _find_type(buf, pos, offs, did)
            if t is not None:
                raw = np.frombuffer(buf, dtype="<u1", count=nc * nb,
                                    offset=t + 2).reshape(nc, nb).T
                arr[:, :, it] = raw.astype(float)

        bt = _find_type(buf, pos, offs, ID_BOTTOM_TRACK)
        if bt is not None:
            have_bt = True
            rng_lsb = np.frombuffer(buf, dtype="<u2", count=4, offset=bt + 16).astype(float)
            # high byte of BT range (extends to deep water) lives at offset 77..80
            rng_msb = np.frombuffer(buf, dtype="<u1", count=4, offset=bt + 77).astype(float)
            rng_cm = rng_lsb + rng_msb * 65536.0
            rng = rng_cm / 100.0
            rng[rng_cm == 0] = np.nan
            bt_range[:, it] = rng
            bv = np.frombuffer(buf, dtype="<i2", count=4, offset=bt + 24).astype(float)
            bv[bv == _BADVEL] = np.nan
            bt_vel[:, it] = bv / 1000.0

    serial = _read_serial(buf, p0, offs0, fl)

    # beam -> earth: rotate per-head at load (legacy loadrdi b2earth) so the rest of the pipeline,
    # which assumes earth-frame vel[0..3] = (u,v,w,e), runs unchanged. Earth-frame files skip this.
    beam2earth_info = None
    if coord_frame == CoordFrame.BEAM:
        from .beam2earth import beam_to_earth
        beam_angle = _beam_angle(sysconf)
        beams_up = facing == "up"
        vel, n3 = beam_to_earth(vel, heading, pitch, roll,
                                beam_angle_deg=beam_angle, beams_up=beams_up)
        n3_bt = 0
        if have_bt:
            bt_e, n3_bt = beam_to_earth(bt_vel[:, None, :], heading, pitch, roll,
                                        beam_angle_deg=beam_angle, beams_up=beams_up)
            bt_vel = bt_e[:, 0, :]
        beam2earth_info = {"n_3beam": n3, "n_3beam_bt": n3_bt, "beam_angle_deg": beam_angle}
        coord_frame = CoordFrame.EARTH

    return RawADCP(
        head=head or facing,
        coord_frame=coord_frame,
        facing=facing,
        freq_khz=freq,
        n_beams=n_beams,
        n_cells=n_cells,
        cell_m=cell_m,
        blank_m=blank_m,
        wm_mode=wm_mode,
        pings_per_ens=pings_per_ens,
        serial=serial,
        time=time,
        vel=vel,
        echo=echo,
        corr=corr,
        pct_good=pctg,
        heading=heading,
        pitch=pitch,
        roll=roll,
        temperature=temperature,
        sound_speed=sound_speed,
        salinity=salinity,
        xmit_voltage=xmit_v,
        bt_range=bt_range if have_bt else None,
        bt_vel=bt_vel if have_bt else None,
        meta={"path": path, "ex_byte": ex, "n_ensembles": nt,
              "sysconfig": sysconf, "facing_from_bit": facing_from_bit,
              "facing_warning": facing_warning, "dist_first_m": dist_first_m,
              "beam_angle_deg": _beam_angle(sysconf), "beam2earth": beam2earth_info},
    )


def _facing_from_name(path: str) -> str | None:
    """Infer instrument orientation from the path when unambiguous.

    Recognises common LADCP naming conventions for the down-looker
    (``MASTER`` dir, ``…downlook…``, RB1606-style ``…<digits>DL<digits>…``)
    and up-looker (``SLAVE`` dir, ``…uplook…``, ``…<digits>UL<digits>…``).
    Short head tokens (``DL``/``UL``/``DN``/``UP``) are matched only when
    embedded between digits, to avoid false hits on incidental substrings
    (e.g. ``result.000`` contains "ul"). Returns ``None`` when the path carries
    no clear signal, so the caller can fall back to the sysconfig bit.
    """
    p = path.replace("\\", "/").lower()
    name = p.rsplit("/", 1)[-1]
    down = ("master" in p) or ("downlook" in p) or ("dnlook" in p) \
        or bool(re.search(r"\d(?:dl|dn)\d", name))
    up = ("slave" in p) or ("uplook" in p) \
        or bool(re.search(r"\d(?:ul|up)\d", name))
    if down and not up:
        return "down"
    if up and not down:
        return "up"
    return None


def _beam_angle(sysconf: int) -> int:
    """Beam angle [deg] from the fixed-leader system-config word.

    The angle is the low two bits of the *high* byte (legacy ``rditype.m`` ``angles[15 20 30 0]``
    indexed by the MSB's bits 0-1). The previous decode read LSB bits 6-7 -- which are XDCR-HD +
    facing -- and so returned 20 deg only by luck for down-lookers (facing bit 0) and a bogus 0 deg
    for up-lookers (facing bit 1). That latent error was harmless for earth-frame data (the up
    head's angle was unused) but breaks the beam->earth transform (1/sin(0)); read the right bits.
    """
    return {0: 15, 1: 20, 2: 30, 3: 0}.get((sysconf >> 8) & 0x03, 20)


def _find_type(buf: bytes, pos: int, offs, type_id: int):
    """Return absolute byte offset of the data type with ``type_id`` in this ensemble."""
    for o in offs:
        if struct.unpack_from("<H", buf, pos + o)[0] == type_id:
            return pos + o
    return None


def _to_dt64(year, mo, da, hh, mm, ss, hs) -> np.datetime64:
    try:
        base = np.datetime64(f"{year:04d}-{mo:02d}-{da:02d}T{hh:02d}:{mm:02d}:{ss:02d}")
        return base + np.timedelta64(int(hs) * 10, "ms")
    except Exception:
        return np.datetime64("NaT")


def _read_serial(buf, pos, offs, fl):
    """Instrument serial is in the FL at offset 54 (uint32) on newer firmware; best-effort."""
    try:
        sn = struct.unpack_from("<I", buf, fl + 54)[0]
        return int(sn) if sn not in (0, 0xFFFFFFFF) else None
    except Exception:
        return None

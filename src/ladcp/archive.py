"""Auto-built, incremental LADCP archive index (#4a).

Turns a raw acquisition tree into a station -> file map so the pipeline never has to be
told which deployment file is which cast. The chain:

1. **scan** the LADCP PD0 deployment files once for their ``(t_start, t_end)`` and head
   orientation -- cached by ``name + size + mtime`` so later runs only touch newly
   arrived files. Files come from ``MASTER/`` + ``SLAVE/`` subdirs when present (MORIA
   layout), else from a recursive ``*.000`` sweep of the whole tree (flat layouts such
   as FDCCC1's ``MA*``/``SL*`` in one dir). Each file is classified down/up-looker by
   the path convention when it carries one, else by the PD0 sysconfig facing bit -- so
   no naming convention is required;
2. **anchor** each cast with a Seabird CTD ``.hex`` header (:mod:`ladcp.io.ctd_hex`), which
   supplies the station label + absolute NMEA UTC + GPS position;
3. **match** the master cast for that UTC -- the window containing it, or (when the deck-unit
   logging was restarted and the real cast begins *after* the on-station time, as on MORIA
   stations 01-04) the largest cast starting shortly after it, never a sub-minute stub -- and
   **pair** the slave by time overlap (:func:`ladcp.discovery.best_overlap`).

The result is written to a small JSON sidecar (``.ladcp_archive.json``) that
:func:`ladcp.discovery.discover` reads in preference to the hard-coded manifest.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .discovery import best_overlap
from .io.ctd_hex import scan_ctd_dir
from .io.pd0 import _facing_from_name, read_pd0

INDEX_NAME = ".ladcp_archive.json"
VERSION = 2

# A deployment file with fewer ensembles than this is a logging stub / false start (the
# largest such stub seen on MORIA 01-04 is 261 ens; the smallest real cast is ~5400), so it
# is never chosen as a cast when a genuine cast is also in range.
MIN_CAST_ENS = 300
# A cast may begin this long after the logged on-station UTC (deck-unit troubleshooting on
# MORIA 01-04 delayed the real logging by up to ~55 min); kept well under the >5 h cast spacing.
_START_TOL = np.timedelta64(2, "h")


@dataclass
class CastEntry:
    """One resolved cast: which raw files belong to it, and the anchor it came from."""

    station: str
    master: str                 # path to down-looker PD0
    slave: str | None           # path to up-looker PD0 (time-paired)
    ctd_hex: str                # the Seabird header that anchored it
    utc: str                    # ISO cast time (from the CTD header)
    lat: float
    lon: float
    depth: float | None
    provenance: str             # how the master was chosen


def _fingerprint(p: Path) -> tuple[int, int]:
    st = p.stat()
    return st.st_size, int(st.st_mtime)


def _scan_files(paths: list[Path], cache: dict, root: Path) -> dict:
    """Return ``relpath -> {size, mtime, t0, t1, n_ens, facing}``, decoding only changed files.

    ``facing`` resolves the file's head: a path naming convention when present
    (``MASTER``/``SLAVE`` dirs, ``DL``/``UL`` tokens), else the PD0 sysconfig
    orientation bit -- so flat trees with opaque names still classify.
    """
    out: dict[str, dict] = {}
    for p in paths:
        rel = os.path.relpath(p.resolve(), root)
        size, mtime = _fingerprint(p)
        prev = cache.get(rel)
        if prev and prev["size"] == size and prev["mtime"] == mtime \
                and "facing" in prev:
            out[rel] = prev                                 # unchanged -> reuse cached span
            continue
        r = read_pd0(str(p), head="down", facing_hint="down")
        out[rel] = {
            "size": size, "mtime": mtime,
            "t0": str(r.time.min().astype("datetime64[s]")),
            "t1": str(r.time.max().astype("datetime64[s]")),
            "n_ens": int(r.n_ens),
            "facing": _facing_from_name(str(p)) or r.meta["facing_from_bit"],
        }
    return out


def _span(entry: dict) -> tuple[np.datetime64, np.datetime64]:
    return np.datetime64(entry["t0"]), np.datetime64(entry["t1"])


def build_index(
    ladcp_dir: str | Path,
    ctd_dir: str | Path,
    *,
    root: str | Path = ".",
    master_subdir: str = "MASTER",
    slave_subdir: str = "SLAVE",
    out: str | Path | None = None,
    rescan: bool = False,
) -> dict:
    """Build (or incrementally update) the archive index and write it to ``out``.

    ``root`` is the base the stored paths are made relative to. With ``rescan=False`` the
    previous scan cache is reused for unchanged files, so only new arrivals are decoded.
    Returns the index dict.
    """
    root = Path(root).resolve()
    ladcp = Path(ladcp_dir)
    mdir, sdir = ladcp / master_subdir, ladcp / slave_subdir
    out_path = Path(out) if out else (root / INDEX_NAME)

    prev = {} if rescan else _load_raw(out_path)
    scan_cache = prev.get("scan_cache", {})

    if mdir.is_dir() and sdir.is_dir():
        files = sorted(mdir.glob("*.000")) + sorted(sdir.glob("*.000"))
    else:
        # flat / unconventional layout: sweep the whole tree; per-file facing
        # (path convention, else the PD0 orientation bit) does the classifying
        files = sorted(ladcp.rglob("*.000"))
    scan = _scan_files(files, scan_cache, root)

    m_info = {rel: (*_span(e), int(e["n_ens"])) for rel, e in scan.items()
              if e["facing"] == "down"}
    s_spans = {rel: _span(e) for rel, e in scan.items()
               if e["facing"] == "up"}

    casts: dict[str, dict] = {}
    master_claims: dict[str, list[str]] = {}
    for h in scan_ctd_dir(ctd_dir):
        master_rel, provenance = _match_master(h.utc, m_info)
        if master_rel is None:
            continue
        master_claims.setdefault(master_rel, []).append(h.station)
        slave_rel = best_overlap(m_info[master_rel][:2], s_spans)
        casts[h.station] = asdict(CastEntry(
            station=h.station,
            master=str(root / master_rel),
            slave=str(root / slave_rel) if slave_rel else None,
            ctd_hex=str(h.path),
            utc=str(h.utc), lat=h.lat, lon=h.lon, depth=h.depth,
            provenance=provenance,
        ))

    # two anchors resolving to one deployment file is almost always a matching error
    # (a missing CTD cast, or casts closer together than _START_TOL) -- make it visible
    for stations in master_claims.values():
        if len(stations) > 1:
            for st in stations:
                casts[st]["provenance"] += f" SHARED-MASTER({','.join(stations)})"

    index = {
        "version": VERSION,
        "root": str(root),
        "ladcp_dir": str(ladcp),
        "master_subdir": master_subdir,
        "slave_subdir": slave_subdir,
        "scan_cache": scan,
        "casts": casts,
    }
    out_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def _match_master(utc: np.datetime64, m_info: dict) -> tuple[str | None, str]:
    """Pick the master cast for ``utc`` from ``rel -> (t0, t1, n_ens)``.

    Order of preference: the cast whose window contains ``utc``; else the *first* cast that
    starts within :data:`_START_TOL` after it (the deck-unit logging on MORIA 01-04 was
    restarted mid-station, so the real cast can begin after the on-station time and is buried
    among sub-minute stubs -- those are excluded by :data:`MIN_CAST_ENS`, and on back-to-back
    shelf stations (FDCCC1 t3-01..03, ~1 h apart) the *next* station's cast can also start
    inside the tolerance, so earliest-start, not largest, is the correct pick); else, only if
    nothing looks like a cast, fall back to the nearest window/start so genuinely small
    (shallow-shelf) casts still resolve. Never a stub when a genuine cast qualifies.
    """
    def is_cast(n: int) -> bool:
        return n >= MIN_CAST_ENS

    contains = [(rel, n) for rel, (t0, t1, n) in m_info.items() if t0 <= utc <= t1]
    cast_contains = [(rel, n) for rel, n in contains if is_cast(n)]
    if cast_contains:
        return max(cast_contains, key=lambda kv: kv[1])[0], "ctd-utc-in-master-window"

    forward = [(rel, t0, n) for rel, (t0, t1, n) in m_info.items()
               if utc < t0 <= utc + _START_TOL]
    cast_forward = [(rel, t0, n) for rel, t0, n in forward if is_cast(n)]
    if cast_forward:
        return min(cast_forward, key=lambda kv: kv[1])[0], "ctd-utc-nearest-cast-start"

    # nothing is clearly a cast -- keep the older, size-blind behaviour as a safety net
    if contains:
        return max(contains, key=lambda kv: kv[1])[0], "ctd-utc-in-master-window"
    near = [(rel, abs(t0 - utc)) for rel, (t0, _, _) in m_info.items()
            if abs(t0 - utc) <= _START_TOL]
    if near:
        return min(near, key=lambda kv: kv[1])[0], "ctd-utc-nearest-master-start"
    return None, "unmatched"


def _load_raw(path: Path) -> dict:
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_index(path: str | Path) -> dict:
    """Load an archive index JSON (``{}`` if missing/corrupt)."""
    return _load_raw(Path(path))


def index_station(index: dict, station: str) -> CastEntry | None:
    """Return the :class:`CastEntry` for ``station`` from a loaded index, or ``None``."""
    rec = (index.get("casts") or {}).get(station)
    return CastEntry(**rec) if rec else None

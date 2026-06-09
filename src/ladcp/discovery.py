"""Station -> input-file discovery (#4a).

Two layouts are supported:

* **Curated** -- the renamed, per-station files in ``New_golden/Good`` (``MORIA-80-LADCP-M.000``
  / ``-S.000`` + ``moria-80_clean.cnv``). A glob on the station id finds them directly.
* **Raw VmDAS archive** -- the on-disk acquisition tree (``raw_ladcp_test/LADCP/Data/{MASTER,
  SLAVE}``) where files carry deployment names (``MLADC012.000``) that do *not* encode the
  station. The station->master mapping comes from the cruise cast log (a per-cruise
  :data:`MANIFESTS` entry); the **slave is then auto-paired by time overlap**, because the
  master/slave deployment indices are offset (MORIA-10 = master ``MLADC012`` + slave
  ``SLADC013``) and only the recorded time spans align them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StationFiles:
    """Resolved inputs for one cast."""

    down: Path
    up: Path | None
    ctd: Path | None
    label: str
    ctd_utc: str | None = None   # CTD cast-start UTC from the archive index (sync absolute prior)


@dataclass(frozen=True)
class RawArchiveEntry:
    """One raw-archive station, as recorded in the cruise cast log.

    ``master`` names the down-looker deployment file (the only station->file link the bytes
    cannot supply); ``slave_dir`` holds the up-looker files, the overlapping one of which is
    selected by :func:`pair_slave_by_time`. Paths are relative to the discovery ``root``.
    """

    master: str
    slave_dir: str
    ctd: str | None = None


# Per-cruise station->file map for the raw VmDAS archive. One line per cast; the slave is
# resolved by time, so only the master deployment file (from the cast sheet) is named here.
MANIFESTS: dict[str, dict[str, RawArchiveEntry]] = {
    "MORIA": {
        "MORIA-10": RawArchiveEntry(
            master="raw_ladcp_test/LADCP/Data/MASTER/MLADC012.000",
            slave_dir="raw_ladcp_test/LADCP/Data/SLAVE",
            ctd="clean_ctd/moria-10_clean.cnv",
        ),
    },
}


def _time_span(path: Path) -> tuple[np.datetime64, np.datetime64]:
    """First and last ensemble time of a PD0 file."""
    from .io.pd0 import read_pd0

    r = read_pd0(str(path), head="down", facing_hint="down")
    t = r.time
    return t.min(), t.max()


def best_overlap(
    master_span: tuple[np.datetime64, np.datetime64],
    candidates: dict[str, tuple[np.datetime64, np.datetime64]],
) -> str | None:
    """Return the candidate key whose time span overlaps ``master_span`` the most.

    Pure (no IO) so the pairing rule can be unit-tested directly. ``None`` if no candidate
    overlaps at all.
    """
    m0, m1 = master_span
    best_key, best = None, np.timedelta64(0, "s")
    for key, (c0, c1) in candidates.items():
        overlap = min(m1, c1) - max(m0, c0)
        if overlap > best:
            best, best_key = overlap, key
    return best_key


def pair_slave_by_time(master: Path, slave_dir: Path, *, pattern: str = "S*.000") -> Path:
    """Pick the up-looker file in ``slave_dir`` whose time span overlaps ``master`` most.

    Master/slave deployment indices are offset in the VmDAS archive, so they must be paired
    by recorded time rather than filename. Raises if nothing overlaps.
    """
    span = _time_span(master)
    cands = {str(p): _time_span(p) for p in sorted(slave_dir.glob(pattern))}
    key = best_overlap(span, cands)
    if key is None:
        raise SystemExit(f"no slave file in {slave_dir} overlaps {master.name} in time")
    return Path(key)


def _normalize(cruise: str, station: str) -> str:
    """Map a bare id (``10``) to a manifest label (``MORIA-10``); pass labels through."""
    return station if "-" in station or "_" in station else f"{cruise.upper()}-{station}"


def _discover_curated(root: Path, st: str) -> StationFiles:
    """Glob the curated ``New_golden/Good`` layout (``LADCP/`` + ``CTD/``)."""
    def pick(*patterns: str) -> Path | None:
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise SystemExit(f"ambiguous match for {pat!r}: {[h.name for h in hits]}")
        return None

    down = pick(f"LADCP/*{st}*-M.000", f"LADCP/*{st}*M*.000")
    if down is None:
        raise SystemExit(f"no down-looker found for station {st!r} under {root}/LADCP")
    up = pick(f"LADCP/*{st}*-S.000", f"LADCP/*{st}*S*.000")
    ctd = pick(f"CTD/*{st}*.cnv")
    label = down.name.split("-LADCP")[0] if "-LADCP" in down.name else st
    return StationFiles(down=down, up=up, ctd=ctd, label=label)


def _load_index(path: Path) -> dict:
    """Read an archive-index JSON (``{}`` if missing/corrupt). Kept here to avoid a cycle."""
    import json
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _find_clean_ctd(ctd_dir: Path, station: str) -> Path | None:
    """Glob a cleaned ``.cnv`` for ``station`` under ``ctd_dir`` (prefer a *clean* name)."""
    num = station.split("-")[-1].split("_")[-1]
    for pat in (f"*{num}*clean*.cnv", f"*{station}*.cnv", f"*{num}*.cnv"):
        hits = sorted(ctd_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _ctd_from_hex(rec: dict, label: str, cache_dir: str | Path | None) -> Path | None:
    """Build (or reuse) a cleaned ``.cnv`` from the index's anchor ``.hex`` (Path A).

    Used only when ``--from-hex`` is set and no pre-processed ``.cnv`` was found.
    Returns ``None`` (with a warning) if the record has no ``.hex`` or the optional
    CTD_project converter is unavailable -- the run then proceeds without CTD.
    """
    hexp = rec.get("ctd_hex")
    if not hexp:
        return None
    from .io.ctd_raw import DEFAULT_CACHE_DIR, cnv_from_hex
    try:
        return cnv_from_hex(hexp, label, cache_dir=cache_dir or DEFAULT_CACHE_DIR)
    except (RuntimeError, FileNotFoundError) as e:
        import warnings
        warnings.warn(f"--from-hex: could not convert {label} from {hexp}: {e}",
                      stacklevel=2)
        return None


def discover(station: str, *, root: Path, cruise: str = "MORIA",
             index: str | Path | dict | None = None,
             ctd_dir: Path | None = None,
             from_hex: bool = False,
             ctd_cache: str | Path | None = None) -> StationFiles:
    """Resolve a station's (down, up, ctd) inputs.

    Resolution order: an **archive index** (auto-built station->file map) first, then the
    raw-archive manifest, then the curated glob layout. ``root`` is the base for the manifest
    /curated layouts; the index stores its own (root-relative) paths. When the index resolves
    the LADCP heads, the cleaned CTD ``.cnv`` is globbed from ``ctd_dir`` (default ``root/CTD``)
    -- the index's raw ``.hex`` is the anchor, not the processing input.

    CTD ingest coexists in two forms: a pre-processed ``.cnv`` (preferred when present),
    or -- when ``from_hex`` is set and no ``.cnv`` is found -- the raw Seabird ``.hex``
    anchored by the index, converted on the fly and cached under ``ctd_cache`` (Path A).
    """
    label = _normalize(cruise, station)

    # 1. archive index (explicit dict/path, else <root>/.ladcp_archive.json if present)
    idx = index if isinstance(index, dict) else _load_index(
        Path(index) if index else root / ".ladcp_archive.json")
    rec = (idx.get("casts") or {}).get(label)
    if rec is not None:
        ctd = _find_clean_ctd(ctd_dir or (root / "CTD"), label)
        if ctd is None and from_hex:                  # Path A: convert the anchor .hex
            ctd = _ctd_from_hex(rec, label, ctd_cache)
        return StationFiles(down=Path(rec["master"]),
                            up=Path(rec["slave"]) if rec["slave"] else None,
                            ctd=ctd, label=label, ctd_utc=rec.get("utc"))

    # 2. raw-archive manifest (slave auto-paired by time)
    entry = MANIFESTS.get(cruise.upper(), {}).get(label)
    if entry is not None:
        master = root / entry.master
        if not master.exists():
            raise SystemExit(f"manifest master not found: {master}")
        slave = pair_slave_by_time(master, root / entry.slave_dir)
        ctd = (root / entry.ctd) if entry.ctd else None
        return StationFiles(down=master, up=slave, ctd=ctd, label=label)

    # 3. curated glob layout
    return _discover_curated(root, station)

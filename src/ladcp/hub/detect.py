"""Cruise-data detection for ``ladcp init``: pure functions, typed proposals, no writes.

Every function here only *looks* — filename globs and directory listings, never file
contents — so a scan of a whole cruise share is cheap and safe (wizard spec §7).
Anything that would read or copy data (PD0 header decoding for the archive index,
CTD ``.hex`` conversion, EK80 extraction) happens later, in the init flow's *apply*
step, after the user confirmed the proposal built from these results.

The detections deliberately reuse the conventions the rest of the package already
honors: the curated ``LADCP/``+``CTD/`` layout of :func:`ladcp.discovery.discover`,
the ``MASTER/``+``SLAVE/`` archive layout of :func:`ladcp.archive.build_index`, the
``<root>/codas`` product convention of Studio's
:func:`~ladcp.studio.state.discover_codas_products`, and the SADO/VmDAS/EK80 file
extensions of the io readers. Beam-vs-earth coordinates need no detection: ingest
auto-rotates beam data to earth (:func:`ladcp.qa.ingest.load_dualhead`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import CRUISES

__all__ = ["Detection", "LadcpDetection", "CtdDetection", "SadcpCandidate",
           "NavCandidate", "detect", "curated_station_labels"]

# directories a scan must not descend into: outputs, caches, and journals the
# pipeline itself creates (a rescan of qa_out would "detect" our own products)
_SKIP_DIRS = {"qa_out", "ctd_from_hex", ".ladcp_edits", "exports", "stations",
              "figures", ".git", "__pycache__"}
_MAX_DEPTH = 3


def _dirs(root: Path, max_depth: int = _MAX_DEPTH):
    """Bounded, hidden/skip-aware directory walk (root first, sorted, no file I/O)."""
    stack = [(Path(root), 0)]
    while stack:
        d, depth = stack.pop(0)
        yield d
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            continue
        stack.extend((c, depth + 1) for c in children
                     if not c.name.startswith(".") and c.name not in _SKIP_DIRS)


# ---------------------------------------------------------------------------
# LADCP casts

@dataclass(frozen=True)
class StationGuess:
    """One curated-layout cast paired by filename (``*-LADCP-M.000`` + ``-S.000``)."""

    label: str
    master: str                       # paths relative to the scanned root
    slave: str | None
    ctd: str | None = None


@dataclass(frozen=True)
class LadcpDetection:
    dir: str | None                   # LADCP directory, relative to root ('' = root)
    layout: str                       # "curated" | "master-slave" | "flat" | "none"
    stations: tuple[StationGuess, ...]   # curated layout only (name-paired)
    n_down: int = 0                   # file counts (master-slave / flat layouts)
    n_up: int = 0
    evidence: str = ""


def _curated_pairs(ladcp_dir: Path, root: Path) -> list[StationGuess]:
    """Name-paired casts in one directory (the ``*-LADCP-M.000`` convention)."""
    out = []
    for m in sorted(ladcp_dir.glob("*-M.000")):
        label = m.name.split("-LADCP")[0] if "-LADCP" in m.name else m.name[:-len("-M.000")]
        cands = sorted(ladcp_dir.glob(f"{m.name[:-len('-M.000')]}-S.000"))
        out.append(StationGuess(label=label,
                                master=str(m.relative_to(root)),
                                slave=str(cands[0].relative_to(root)) if cands else None))
    return out


def _detect_ladcp(root: Path) -> LadcpDetection:
    """The most convention-conformant directory holding PD0 deployment files."""
    candidates: list[tuple[int, LadcpDetection]] = []      # (score, detection)
    for d in _dirs(root):
        rel = str(d.relative_to(root)) if d != root else ""
        named = "ladcp" in d.name.lower()
        mdir, sdir = d / "MASTER", d / "SLAVE"
        if mdir.is_dir() and sdir.is_dir():
            n_dn = len(list(mdir.glob("*.000")))
            n_up = len(list(sdir.glob("*.000")))
            if n_dn:
                det = LadcpDetection(rel, "master-slave", (), n_dn, n_up,
                                     f"{n_dn} MASTER/ + {n_up} SLAVE/ *.000 files")
                candidates.append((30 + 5 * named, det))
                continue
        pd0s = list(d.glob("*.000"))
        if not pd0s:
            continue
        pairs = _curated_pairs(d, root)
        if pairs:
            n_up = sum(1 for p in pairs if p.slave)
            det = LadcpDetection(rel, "curated", tuple(pairs), len(pairs), n_up,
                                 f"{len(pairs)} name-paired cast(s) "
                                 f"({n_up} with an up-looker)")
            candidates.append((40 + 5 * named + len(pairs), det))
        else:
            det = LadcpDetection(rel, "flat", (), len(pd0s), 0,
                                 f"{len(pd0s)} *.000 file(s), names not pairable — "
                                 "the archive index (time pairing) is needed")
            candidates.append((10 + 5 * named, det))
    if not candidates:
        return LadcpDetection(None, "none", (), evidence="no *.000 PD0 files found")
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]


def curated_station_labels(root) -> list[str]:
    """Station labels enumerable from filenames alone (no index, no ``.hex`` anchors).

    The fallback universe for ``ladcp process --new/--all`` on cruises whose archive
    index is empty — ``ladcp-index`` anchors casts on raw CTD ``.hex`` files, so a
    directory holding only cleaned ``.cnv`` indexes zero casts even though every
    cast is discoverable by name. Only the standard ``LADCP/`` directory counts:
    that is the one place :func:`ladcp.discovery.discover`'s curated globs resolve,
    so a label from anywhere else could not actually be processed.
    """
    det = _detect_ladcp(Path(root))
    if det.layout != "curated" or det.dir != "LADCP":
        return []
    return [s.label for s in det.stations]


# ---------------------------------------------------------------------------
# CTD

@dataclass(frozen=True)
class CtdDetection:
    dir: str | None                   # relative to root
    n_cnv: int = 0
    n_hex: int = 0
    missing_cnv: tuple[str, ...] = () # station labels with no matching .cnv
    converter: str | None = None      # CTD_project location when --from-hex can work
    evidence: str = ""


def _detect_ctd(root: Path, stations: tuple[StationGuess, ...]) -> CtdDetection:
    best: tuple[int, Path, int, int] | None = None         # (score, dir, n_cnv, n_hex)
    for d in _dirs(root):
        n_cnv = len(list(d.glob("*.cnv")))
        n_hex = len(list(d.glob("*.hex")))
        if not (n_cnv or n_hex):
            continue
        score = n_cnv + n_hex + 20 * ("ctd" in d.name.lower())
        if best is None or score > best[0]:
            best = (score, d, n_cnv, n_hex)
    if best is None:
        return CtdDetection(None, evidence="no .cnv or .hex CTD files found")
    _, d, n_cnv, n_hex = best

    missing = []
    for s in stations:                 # the same globs discovery uses per station
        num = s.label.split("-")[-1].split("_")[-1]
        if not any(d.glob(f"*{num}*.cnv")):
            missing.append(s.label)

    converter = None
    if n_hex:
        try:
            from ..io.ctd_raw import _find_ctd_project
            converter = str(_find_ctd_project())
        except Exception:
            converter = None
    rel = str(d.relative_to(root)) if d != root else ""
    ev = f"{n_cnv} .cnv + {n_hex} .hex"
    if missing:
        ev += f"; {len(missing)} cast(s) without a cleaned .cnv"
    return CtdDetection(rel, n_cnv, n_hex, tuple(missing), converter, ev)


# ---------------------------------------------------------------------------
# ship-ADCP + nav

@dataclass(frozen=True)
class SadcpCandidate:
    source: str                       # "vmdas" | "codas" | "ek80"
    path: str                         # relative to root when inside it
    evidence: str
    extracted: bool = False           # already a per-station EK80 extract tree:
                                      # nothing left to copy, point [sadcp] at it


@dataclass(frozen=True)
class NavCandidate:
    path: str
    evidence: str


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)) or "."
    except ValueError:
        return str(p)


def _detect_sadcp(root: Path) -> tuple[SadcpCandidate, ...]:
    out: list[SadcpCandidate] = []
    ek_dirs: dict[str, tuple[int, bool]] = {}   # rel -> (n_nc, ek80-named)
    for d in _dirs(root):
        for ft in ("STA", "LTA"):
            n = len(list(d.glob(f"*.{ft}")))
            if n:
                out.append(SadcpCandidate("vmdas", _rel(d, root),
                                          f"{n} VmDAS .{ft} file(s)"))
        n_nc = len(list(d.glob("*.nc")))
        named = "ek80" in d.name.lower()
        if n_nc and (named or (n_nc >= 2
                               and "codas" not in Path(_rel(d, root)).parts)):
            ek_dirs[_rel(d, root)] = (n_nc, named)

    # per-station extract trees (the `ladcp-ek80 extract --out` layout): >=2 .nc-rich
    # sibling folders collapse into ONE candidate for their parent — each cast picks
    # its files by the time window, so the parent is the right [sadcp] folder
    # (guide ch. 8) and a 30-station tree must not become 30 radio buttons.
    by_parent: dict[str, list[str]] = {}
    for rel in ek_dirs:
        by_parent.setdefault(str(Path(rel).parent), []).append(rel)
    grouped: set[str] = set()
    for parent, kids in by_parent.items():
        # never group direct children of the cruise root: top-level dirs are
        # heterogeneous by nature (extract trees live under EK80/, adcp_local/, …)
        if parent != "." and len(kids) >= 2:
            total = sum(ek_dirs[k][0] for k in kids)
            out.append(SadcpCandidate("ek80", parent, extracted=True,
                                      evidence=f"{len(kids)} station folder(s), "
                                               f"{total} .nc file(s) — per-station "
                                               "EK80 extracts; each cast picks its "
                                               "files by time window"))
            grouped.update(kids)
    for rel, (n_nc, named) in ek_dirs.items():
        if rel in grouped or rel in by_parent and len(by_parent[rel]) >= 2:
            continue                             # covered by (or is) a grouped parent
        out.append(SadcpCandidate("ek80", rel,
                                  f"{n_nc} .nc file(s)"
                                  + ("" if named else " — possibly EK80")
                                  + " (verify with ladcp-ek80 timetable)"))
    from ..studio.state import codas_label, discover_codas_products
    for nc in discover_codas_products(root):
        folder = nc.parent.parent if nc.parent.name == "contour" else nc
        out.append(SadcpCandidate("codas", _rel(folder, root),
                                  f"CODAS product {codas_label(nc)}"))
    # stable, human-sensible choice order: the common sources first, EK80 last
    order = {"vmdas": 0, "codas": 1, "ek80": 2}
    out.sort(key=lambda c: (order[c.source], c.path))
    return tuple(out)


def _detect_nav(root: Path) -> tuple[NavCandidate, ...]:
    """SADO exports / nav directories the ``--sadcp-timeoff auto`` machinery can read."""
    out: list[NavCandidate] = []
    seen: set[Path] = set()
    for d in _dirs(root):
        hits = sorted(list(d.glob("*posicion*")) + list(d.glob("*navsado*")))
        if hits:
            if d not in seen:
                seen.add(d)
                out.append(NavCandidate(_rel(d, root),
                                        f"{len(hits)} SADO track file(s), "
                                        f"e.g. {hits[0].name}"))
        elif "nav" in d.name.lower() and (list(d.glob("*.csv")) or list(d.glob("*.txt"))):
            if d not in seen:
                seen.add(d)
                out.append(NavCandidate(_rel(d, root), "nav-named dir with csv/txt tracks"))
    return tuple(out)


# ---------------------------------------------------------------------------
# the whole picture

@dataclass(frozen=True)
class Detection:
    root: str
    name: str                         # proposed cruise name (the directory's)
    preset: bool                      # a registered params preset matches the name
    ladcp: LadcpDetection = field(default_factory=lambda: LadcpDetection(None, "none", ()))
    ctd: CtdDetection = field(default_factory=lambda: CtdDetection(None))
    sadcp: tuple[SadcpCandidate, ...] = ()
    nav: tuple[NavCandidate, ...] = ()


def detect(root) -> Detection:
    """Scan ``root`` (filenames only) into one confirmable :class:`Detection`."""
    root = Path(root).resolve()
    name = root.name or "CRUISE"
    ladcp = _detect_ladcp(root)
    return Detection(root=str(root), name=name, preset=name.upper() in CRUISES,
                     ladcp=ladcp, ctd=_detect_ctd(root, ladcp.stations),
                     sadcp=_detect_sadcp(root), nav=_detect_nav(root))

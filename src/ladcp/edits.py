"""Manual edit journals: recorded, replayable brush edits over raw cells.

A journal is one station's set of user-drawn rectangle flags over the raw
ensemble matrix, persisted as ``<root>/.ladcp_edits/<station>.json`` so the
Studio brush and ``ladcp-qa --edits`` replay the *same* edits bit-identically.
Edits are never applied implicitly: the CLI needs the explicit ``--edits``
flag and Studio surfaces the journal it is using.

Coordinate convention (the single place it is defined; converted to merged
rows exactly once, in :func:`ladcp.qa.superens._edit_velocity_mask`):

* ``head`` is ``"down"`` or ``"up"`` -- the physical instrument head.
* ``bin_first``/``bin_last`` are **1-based inclusive** instrument bins of that
  head, matching ``--nearfield-dn-bins`` and ``edit_mask_dn_bins``.
* ``ens_first``/``ens_last`` are **0-based inclusive** ensemble indices in the
  joint-trimmed space the merge uses (``n = min(n_down, n_up)``, shift 0).

Rectangles are clamped at apply time; an out-of-range rectangle (or an up-head
rectangle on a single-head solve) is a silent no-op, so a journal survives the
``--down-only`` toggle.

``entries[*].kind`` is ``"rect"`` in version 1. Loaders refuse unknown kinds
and newer versions outright -- silently skipping edits would change science.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

JOURNAL_VERSION = 1
JOURNAL_DIRNAME = ".ladcp_edits"

#: journal entry fields that define solve-relevant geometry (the cache key);
#: everything else (id, note, view, created) is display metadata.
_GEOM_FIELDS = ("head", "bin_first", "bin_last", "ens_first", "ens_last")
_HEADS = ("down", "up")


@dataclass
class Journal:
    """In-memory form of one station's edit journal (see the module docstring)."""

    station: str
    version: int = JOURNAL_VERSION
    created: str = ""
    updated: str = ""
    raw: dict = field(default_factory=dict)       # {"down"/"up": {file, size, n_ens}}
    joint_n_ens: int | None = None
    next_id: int = 1
    entries: list[dict] = field(default_factory=list)

    @property
    def n_entries(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        return {"version": self.version, "station": self.station,
                "created": self.created, "updated": self.updated,
                "raw": self.raw, "joint_n_ens": self.joint_n_ens,
                "next_id": self.next_id, "entries": self.entries}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def journal_path(root, station: str) -> Path:
    """The canonical journal location for ``station`` under cruise ``root``."""
    return Path(root) / JOURNAL_DIRNAME / f"{station}.json"


def new_journal(station: str) -> Journal:
    now = _utcnow()
    return Journal(station=station, created=now, updated=now)


def _check_entry(e: dict, where: str) -> None:
    kind = e.get("kind", "rect")
    if kind != "rect":
        raise ValueError(f"{where}: journal entry kind {kind!r} requires a newer "
                         "pyladcp (this version replays 'rect' only); refusing to "
                         "apply a partial edit set")
    head = e.get("head")
    if head not in _HEADS:
        raise ValueError(f"{where}: entry head must be 'down' or 'up', got {head!r}")
    for k in ("bin_first", "bin_last", "ens_first", "ens_last"):
        v = e.get(k)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"{where}: entry field {k!r} must be an integer, "
                             f"got {v!r}")
    if e["bin_first"] > e["bin_last"] or e["ens_first"] > e["ens_last"]:
        raise ValueError(f"{where}: empty rectangle (bin {e['bin_first']}-"
                         f"{e['bin_last']}, ens {e['ens_first']}-{e['ens_last']})")


def load_journal(path) -> Journal:
    """Parse + validate a journal file. Raises ``ValueError`` on anything suspect."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"--edits: cannot read {path}: {e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"--edits: {path} is not valid JSON: {e}") from None
    version = data.get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"--edits: {path}: missing/invalid journal version")
    if version > JOURNAL_VERSION:
        raise ValueError(f"--edits: {path} is journal version {version}; this "
                         f"pyladcp reads up to {JOURNAL_VERSION} -- upgrade pyladcp")
    station = data.get("station")
    if not station or not isinstance(station, str):
        raise ValueError(f"--edits: {path}: missing station")
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"--edits: {path}: entries must be a list")
    for i, e in enumerate(entries):
        _check_entry(e, f"{path} entry #{i + 1}")
    return Journal(station=station, version=version,
                   created=data.get("created", ""), updated=data.get("updated", ""),
                   raw=data.get("raw", {}), joint_n_ens=data.get("joint_n_ens"),
                   next_id=int(data.get("next_id", len(entries) + 1)),
                   entries=entries)


def save_journal(journal: Journal, path) -> None:
    """Atomic write: ``<path>.tmp`` + ``os.replace`` (same-directory rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    journal.updated = _utcnow()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(journal.to_dict(), indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_journal(journal: Journal, path, down_path, up_path=None) -> None:
    """Staleness guard: the raw files must match the journal's fingerprints.

    Compares basename + size (PD0 edits always change the size; no hashing
    needed). Raises ``ValueError`` naming what changed and the remedy. The
    per-head ``n_ens`` fingerprint is checked separately post-ingest (the
    caller has a ``DualHead`` there); absent fingerprints (hand-written
    journal) verify as OK.
    """
    for head, fpath in (("down", down_path), ("up", up_path)):
        fp = journal.raw.get(head)
        if not fp or fpath is None:
            continue
        p = Path(fpath)
        problems = []
        if fp.get("file") and fp["file"] != p.name:
            problems.append(f"file {fp['file']!r} -> {p.name!r}")
        if fp.get("size") is not None and p.exists() and fp["size"] != p.stat().st_size:
            problems.append(f"size {fp['size']} -> {p.stat().st_size}")
        if problems:
            raise ValueError(
                f"--edits: {path}: the {head} raw file changed since these edits "
                f"were drawn ({'; '.join(problems)}); the rectangles would land on "
                f"the wrong cells -- delete or re-create the journal")


def manual_flags(journal: Journal) -> tuple[tuple[str, int, int, int, int], ...]:
    """Canonical hashable geometry: sorted, de-duplicated, metadata-free.

    This is what enters :class:`ladcp.session.EditConfig` (and so the prepare
    cache key): editing a note or re-ordering entries never re-prepares.
    """
    return tuple(sorted({tuple(e[k] for k in _GEOM_FIELDS) for e in journal.entries}))


def resolve_edits_arg(edits, station: str):
    """Resolve a ``--edits`` value for one station -> ``Path | None``.

    ``edits`` is a journal *file* (used as-is) or a directory holding
    ``<station>.json`` files (missing file = no edits for that station, not an
    error). Raises ``ValueError`` when the path does not exist at all.
    """
    if not edits:
        return None
    p = Path(edits)
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / f"{station}.json"
        return cand if cand.is_file() else None
    raise ValueError(f"--edits: {p} does not exist (give a journal file or the "
                     f"{JOURNAL_DIRNAME} directory)")

"""Studio server state: the station list, discovery context, and session LRU."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from pathlib import Path

from ..config import DEFAULT_CRUISE as _QA_CRUISE_DEFAULT
from ..edits import Journal, journal_path, load_journal, manual_flags, verify_journal
from ..session import SadcpConfig, SessionConfig, StationSession

log = logging.getLogger("ladcp.studio")

MAX_SESSIONS = 3                      # LRU cap: a prepared session is order-100 MB
_QA_ROOT_DEFAULT = "New_golden/Good"  # ladcp-qa --root default (for minimal cli strings)


@dataclass
class StationEntry:
    """Explicit per-station files (tests / non-discovery launches)."""

    label: str
    down: str
    up: str | None = None
    ctd: str | None = None
    ctd_utc: object = None


def codas_label(path: str | Path) -> str:
    """Short dropdown label for a CODAS product: the processing-tree directory name
    (``…/codas/os150nb_enr/contour/os150nb.nc`` and ``…/codas/os150nb_enr`` both
    label as ``os150nb_enr``)."""
    p = Path(path)
    if p.suffix.lower() == ".nc":
        p = p.parent
    if p.name == "contour":
        p = p.parent
    return p.name or str(p)


def raw_label(path: str | Path) -> str:
    """Short dropdown label for a raw VmDAS folder: the last informative path part
    (``sADCP/sadcp_75/DATA`` -> ``sadcp_75`` — the trailing ``DATA`` says nothing)."""
    for part in reversed(Path(path).parts):
        if part.lower() not in ("data", "", "/", "."):
            return part
    return "raw"


def discover_codas_products(root: str | Path) -> list[Path]:
    """CODAS contour NetCDFs under the ``<root>/codas`` convention."""
    base = Path(root) / "codas"
    return sorted(base.glob("*/contour/*.nc")) + sorted(base.glob("*.nc"))


def merge_discovered_codas(root: str | Path,
                           flagged: list[SadcpConfig]) -> list[SadcpConfig]:
    """``flagged`` + products auto-discovered under ``<root>/codas`` (deduped).

    Discovered products only *populate the dropdown* — picking one is an explicit
    user action, and the default selection stays with the explicitly flagged
    sources — so with several products on disk (instruments x STA/ENR routes) the
    scientific choice remains the user's.
    """
    from ..io.sadcp_codas import resolve_codas_nc
    seen = set()
    for c in flagged:
        try:
            seen.add(resolve_codas_nc(c.folder).resolve())
        except FileNotFoundError:
            pass                                 # flagged paths are validated at launch
    out = list(flagged)
    for nc in discover_codas_products(root):
        r = nc.resolve()
        if r in seen:
            continue
        seen.add(r)
        folder = nc.parent.parent if nc.parent.name == "contour" else nc
        out.append(SadcpConfig(folder=str(folder), source="codas"))
    return out


class StudioState:
    """The server's station list, discovery context, and LRU of prepared sessions."""

    def __init__(self, labels: list[str], *, root: str = _QA_ROOT_DEFAULT,
                 cruise: str = _QA_CRUISE_DEFAULT, index: str | None = None,
                 from_hex: bool = False, ctd_cache: str | None = None,
                 sadcp: SadcpConfig | list[SadcpConfig] | None = None,
                 sadcp_codas: list[SadcpConfig] | None = None,
                 sadcp_found: list[SadcpConfig] | None = None,
                 explicit: dict[str, StationEntry] | None = None):
        self.labels = list(labels)
        self.root = root
        self.cruise = cruise
        self.index = index
        self.from_hex = from_hex
        self.ctd_cache = ctd_cache
        raws = [sadcp] if isinstance(sadcp, SadcpConfig) else list(sadcp or [])
        self.sadcp = raws[0] if raws else None   # primary launch identity (or None)
        # the GUI's source dropdown: key -> identity, primary first. Each solve still
        # carries exactly ONE SadcpConfig, so the ladcp-qa CLI contract is unchanged.
        # A single raw source keeps the plain key "raw"; several get folder-derived
        # names (--sadcp sADCP/sadcp_75/DATA --sadcp sADCP/sadcp_150/DATA ->
        # "sadcp_75", "sadcp_150").
        sources: OrderedDict[str, SadcpConfig] = OrderedDict()
        self.sadcp_origin: dict[str, str] = {}   # key -> "flag" | "found"

        def add(key: str, cfg: SadcpConfig, origin: str = "flag") -> None:
            base, n = key, 2
            while key in sources:                # same name twice: suffix, keep both
                key, n = f"{base}-{n}", n + 1
            sources[key] = cfg
            self.sadcp_origin[key] = origin

        for cfg in raws:
            if cfg.source != "vmdas":
                add(codas_label(cfg.folder), cfg)
            else:
                add("raw" if len(raws) == 1 else raw_label(cfg.folder), cfg)
        for cfg in sadcp_codas or []:
            add(codas_label(cfg.folder), cfg)
        for cfg in sadcp_found or []:            # discovered: in the dropdown, never the
            add(codas_label(cfg.folder), cfg, "found")   # default constraint
        self.sadcp_sources = sources
        self._explicit = dict(explicit or {})
        self._sessions: OrderedDict[str, StationSession] = OrderedDict()
        self._map_lock = threading.Lock()
        self._station_locks: dict[str, threading.Lock] = {}

    def has_station(self, label: str) -> bool:
        """Whether ``label`` was part of the launch work-list (explicit or discovered)."""
        return label in self.labels or label in self._explicit

    def lock_for(self, label: str) -> threading.Lock:
        with self._map_lock:
            return self._station_locks.setdefault(label, threading.Lock())

    def session(self, label: str) -> StationSession:
        """The (LRU-cached) session for ``label``; discovers files on first use."""
        with self._map_lock:
            ses = self._sessions.get(label)
            if ses is not None:
                self._sessions.move_to_end(label)
                return ses
        entry = self._explicit.get(label)
        if entry is not None:
            ses = StationSession(entry.down, entry.up, entry.ctd, station=entry.label,
                                 cruise=self.cruise, ctd_utc=entry.ctd_utc)
        else:
            from ..discovery import discover
            sf = discover(label, root=Path(self.root), cruise=self.cruise,
                          index=self.index, from_hex=self.from_hex,
                          ctd_cache=self.ctd_cache)
            ses = StationSession(sf.down, sf.up, sf.ctd, station=sf.label,
                                 cruise=self.cruise, ctd_utc=sf.ctd_utc)
        with self._map_lock:
            self._sessions[label] = ses
            while len(self._sessions) > MAX_SESSIONS:    # evict least-recently used
                old, _ = self._sessions.popitem(last=False)
                log.info("studio: evicted session %s (LRU cap %d)", old, MAX_SESSIONS)
        return ses

    # -- manual-edit journals ---------------------------------------------------------
    #
    # Journals are keyed by the CANONICAL station label (ses.station, the label
    # discovery resolves -- "MORIA-80"), never by the launch token ("80"): the
    # emitted `ladcp-qa --edits` command replays through discovery, and its
    # station-match guard would reject a token-named journal.

    def edits_path(self, ses: StationSession) -> Path:
        return journal_path(self.root, ses.station)

    def load_edits(self, ses: StationSession) -> Journal | None:
        """The station's journal, or ``None`` when no file exists (ValueError when
        the file is unreadable/foreign -- mapped to HTTP 400 by the endpoints)."""
        p = self.edits_path(ses)
        return load_journal(p) if p.is_file() else None

    def attach_edits(self, cfg: SessionConfig, ses: StationSession) -> SessionConfig:
        """Return ``cfg`` with the journal's rectangles attached (verified fresh).

        The journal is the single source of truth for manual edits -- the request
        body never carries rectangles -- so every config-consuming endpoint MUST
        route through here or its solve/PNG cache keys silently diverge.
        """
        j = self.load_edits(ses)
        if j is None or not j.entries:
            return cfg
        verify_journal(j, self.edits_path(ses), ses.down, ses.up)
        return _dc_replace(cfg, edit=_dc_replace(cfg.edit, manual_flags=manual_flags(j)))

    def cli_context(self) -> dict:
        """Non-default discovery args for :meth:`SessionConfig.to_cli` (minimal commands)."""
        ctx: dict = {}
        if self._explicit:                       # explicit files: no discovery context
            return ctx
        if self.root != _QA_ROOT_DEFAULT:
            ctx["root"] = self.root
        if self.cruise != _QA_CRUISE_DEFAULT:
            ctx["cruise"] = self.cruise
        if self.index is not None:
            ctx["index"] = self.index
        return ctx

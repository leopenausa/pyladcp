"""The freshness rule: which stations does ``ladcp process --new`` still owe?

No state file (wizard spec §9.3): a station is *done* when its QA report
(``<outdir>/stations/<label>/<label>_qa.json``, the last artifact
:func:`~ladcp.qa.pipeline.process_station` rewrites) exists and is newer than every
input — the raw PD0 head(s), the CTD file, the station's edit journal, and
``cruise.toml`` itself. Anything else is ``missing`` (never processed) or ``stale``
(an input or the config changed), and rerunning ``--new`` after an interruption
resumes exactly where the batch died. The rule deliberately errs toward
reprocessing: a touched-but-unchanged input costs one redundant run, never a stale
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_CRUISE
from ..discovery import discover
from ..edits import journal_path

__all__ = ["StationState", "station_state", "select_new"]

FRESH, STALE, MISSING = "fresh", "stale", "missing"


@dataclass(frozen=True)
class StationState:
    """One station's freshness verdict + the input that decided it."""

    label: str
    state: str          # "fresh" | "stale" | "missing"
    reason: str         # e.g. "MORIA-07-LADCP-M.000 is newer than the QA report"


def station_state(station: str, *, root, outdir, cruise: str = DEFAULT_CRUISE,
                  index=None, from_hex: bool = False, ctd_cache=None,
                  config_path=None) -> StationState:
    """Classify ``station`` as fresh / stale / missing against its current inputs.

    A station whose files no longer resolve through discovery counts as ``missing``
    (a run would record the error; hiding it from ``--new`` would silently shrink
    the batch). Only mtimes are compared — no file contents are read, so a sweep
    over a whole cruise stays share-friendly (spec §7).
    """
    root = Path(root)
    try:
        sf = discover(station, root=root, cruise=cruise, index=index,
                      from_hex=from_hex, ctd_cache=ctd_cache)
    except Exception as e:
        return StationState(station, MISSING, f"inputs unresolved: {e}")

    done = Path(outdir) / "stations" / sf.label / f"{sf.label}_qa.json"
    if not done.is_file():
        return StationState(sf.label, MISSING, "no QA report yet")
    done_mtime = done.stat().st_mtime

    inputs: list[Path] = [p for p in (sf.down, sf.up, sf.ctd) if p is not None]
    journal = journal_path(root, sf.label)
    if journal.is_file():
        inputs.append(journal)
    if config_path is not None and Path(config_path).is_file():
        inputs.append(Path(config_path))
    for p in inputs:
        try:
            if p.stat().st_mtime > done_mtime:
                return StationState(sf.label, STALE,
                                    f"{p.name} is newer than the QA report")
        except OSError:
            return StationState(sf.label, STALE, f"{p.name} is unreadable")
    return StationState(sf.label, FRESH, "outputs newer than every input")


def select_new(stations: list[str], **kwargs) -> tuple[list[str], list[StationState]]:
    """Split ``stations`` into the ``--new`` work list (missing + stale) and all states.

    Returns ``(todo, states)`` with ``states`` in input order, so callers can both
    run the pending stations and report why each one was (not) selected.
    """
    states = [station_state(s, **kwargs) for s in stations]
    todo = [st.label for st in states if st.state != FRESH]
    return todo, states

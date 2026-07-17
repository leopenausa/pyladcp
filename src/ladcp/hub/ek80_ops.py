"""EK80 share→slim-local-copy operations shared by the wizard front ends (EK-B).

Thin composition of :mod:`ladcp.io.ek80_files` (scan / correlate / slim_extract)
into the two things both the GUI panel and the terminal wizard need: the
cast-coverage timetable, and the per-station extraction job list with a progress
callback. ``ladcp-ek80`` (the expert CLI) keeps its own driver; the equivalent
commands are rendered by :func:`commands` so every front end can display the
terminal path it is automating (decision 2026-07-17).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..io import ek80_files as ek

__all__ = ["DEFAULT_OUT", "timetable", "build_jobs", "extract_jobs", "commands"]

DEFAULT_OUT = "ek80"                    # <cruise-root>/ek80/<station>/<file>.nc


def timetable(paths: list[str], index_path: str | Path, *, pre: float = 20.0,
              post: float = 170.0) -> dict:
    """The cast↔file coverage table (header peeks only, nothing written).

    Returns ``{n_files, covered, stations: [{station, n, files}]}`` — the table the
    wizard shows before anything is copied.
    """
    rows = ek.scan(paths, peek=True)
    mapping = ek.correlate(rows, ek.read_casts(str(index_path)),
                           pre_min=pre, post_min=post)
    stations = [{"station": s, "n": len(files),
                 "files": [os.path.basename(f) for f in files]}
                for s, files in mapping.items()]
    return {"n_files": len(rows),
            "covered": sum(1 for s in stations if s["n"]),
            "stations": stations}


def build_jobs(paths: list[str], index_path: str | Path, *, pre: float = 20.0,
               post: float = 170.0,
               stations: list[str] | None = None) -> list[tuple[str, str]]:
    """The ``(station, src_path)`` extraction list (filename windows are enough)."""
    rows = ek.scan(paths, peek=False)
    mapping = ek.correlate(rows, ek.read_casts(str(index_path)),
                           pre_min=pre, post_min=post)
    if stations is not None:
        mapping = {s: f for s, f in mapping.items() if s in set(stations)}
    return [(station, f) for station, files in sorted(mapping.items()) for f in files]


def extract_jobs(jobs: list[tuple[str, str]], out_root: str | Path,
                 progress=None) -> tuple[int, int, list[str]]:
    """Slim-extract every job into ``<out_root>/<station>/`` -> ``(ok, bytes, errors)``.

    ``progress(i, n, station, filename)`` fires before each file. A failing file is
    recorded and skipped — a share hiccup must not abandon the rest of the copy.
    """
    out_root = Path(out_root)
    ok = total = 0
    errors: list[str] = []
    for i, (station, src) in enumerate(jobs, 1):
        name = os.path.basename(src)
        if progress:
            progress(i, len(jobs), station, name)
        try:
            total += ek.slim_extract(src, str(out_root / station / name))
            ok += 1
        except Exception as e:
            errors.append(f"{station}/{name}: {type(e).__name__}: {e}")
    return ok, total, errors


def commands(paths: list[str], index_path, out_root, *, pre: float = 20.0,
             post: float = 170.0) -> dict[str, str]:
    """The equivalent ``ladcp-ek80`` invocations, for display next to every button."""
    p = " ".join(str(x) for x in paths)
    win = f"--index {index_path} --pre {pre:g} --post {post:g}"
    return {"timetable": f"ladcp-ek80 timetable {p} {win}",
            "extract": f"ladcp-ek80 extract {p} {win} --out {out_root}"}

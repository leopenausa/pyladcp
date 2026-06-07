"""Optional raw-CTD ingest: build the cleaned 6-col ``.cnv`` from a Seabird ``.hex``.

This is the consumer side of LADCP roadmap #7. Most runs read an already-cleaned
CTD ``.cnv`` (see :mod:`ladcp.io.ctd_cnv`); some operators, though, only have the
raw Seabird cast (``.hex`` + ``.XMLCON``) and no pre-processed profile. For those,
:func:`cnv_from_hex` runs the operator's locked datcnv → wild-edit → 1 s bin
recipe and writes the same headerless 6-column file the rest of the pipeline
consumes.

The recipe itself lives in the **CTD_project** package (``ctd_pipeline``), which
is an *optional* dependency — pyladcp does not require it. It is located, in
order, via the ``LADCP_CTD_PROJECT`` environment variable, then a sibling
``CTD_project`` directory next to this repository. If neither is found, a clear
:class:`RuntimeError` explains how to point at it; the cleaned-``.cnv`` path
(:func:`ladcp.io.ctd_cnv.read_ctd_cnv`) keeps working regardless.

Converted files are written to a **dedicated cache directory** and reused on
later runs (skip reconversion when the cache is newer than the ``.hex``), so a
station's raw cast is processed once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default folder converted .cnv files are cached in (relative to cwd), reused
# across runs. Overridable per call; the CLI exposes it as --ctd-cache.
DEFAULT_CACHE_DIR = "ctd_from_hex"


def _find_ctd_project() -> Path:
    """Locate the CTD_project repo (holding the ``ctd_pipeline`` package).

    ``LADCP_CTD_PROJECT`` wins; otherwise a sibling ``CTD_project`` directory next
    to this repository root. Raises with guidance if neither has ``ctd_pipeline``.
    """
    candidates = []
    env = os.environ.get("LADCP_CTD_PROJECT")
    if env:
        candidates.append(Path(env))
    # repo root is .../src/ladcp/io/ctd_raw.py -> parents[3]
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root.parent / "CTD_project")

    for c in candidates:
        if (c / "ctd_pipeline" / "__init__.py").exists():
            return c
    raise RuntimeError(
        "raw-CTD ingest needs the CTD_project package (ctd_pipeline), which was "
        "not found. Set LADCP_CTD_PROJECT to its path, or place CTD_project beside "
        f"this repo. Looked in: {', '.join(str(c) for c in candidates)}"
    )


def _load_converter():
    """Import ``convert_for_ladcp`` from CTD_project, adding it to ``sys.path`` lazily."""
    proj = _find_ctd_project()
    if str(proj) not in sys.path:
        sys.path.insert(0, str(proj))
    try:
        from ctd_pipeline.convert_for_ladcp import convert_for_ladcp
    except ImportError as e:                                   # missing gsw/scipy/etc.
        raise RuntimeError(
            f"found CTD_project at {proj} but could not import convert_for_ladcp "
            f"({e}). Ensure its deps (numpy, scipy, gsw) are installed in this env."
        ) from e
    return convert_for_ladcp


def xmlcon_for(hex_path: str | Path) -> Path:
    """Return the ``.XMLCON`` sibling of a Seabird ``.hex``.

    Matched case-insensitively on the *whole* filename, not just the extension: some
    archives name the config with a different-case station letter than the ``.hex`` (e.g.
    ``MORIA-25B-CTD.XMLCON`` beside ``MORIA-25b-CTD.hex``), which a stem-preserving
    ``with_suffix`` lookup misses on a case-sensitive filesystem.
    """
    hex_path = Path(hex_path)
    for ext in (".XMLCON", ".xmlcon"):                       # exact-case fast path
        cand = hex_path.with_suffix(ext)
        if cand.exists():
            return cand
    want = hex_path.stem.lower()                             # e.g. "moria-25b-ctd"
    for p in hex_path.parent.iterdir():
        if p.is_file() and p.suffix.lower() == ".xmlcon" and p.stem.lower() == want:
            return p
    raise FileNotFoundError(f"no .XMLCON beside {hex_path}")


def cnv_from_hex(hex_path: str | Path, station: str, *,
                 cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 xmlcon_path: str | Path | None = None,
                 force: bool = False) -> Path:
    """Convert a Seabird ``.hex`` to the cached 6-col LADCP ``.cnv``; return its path.

    The converted file is written as ``<cache_dir>/<station>_clean.cnv`` and reused
    on later runs when it is newer than the ``.hex`` (unless ``force``). ``station``
    is the label used for the cache filename. Raises :class:`RuntimeError` if the
    CTD_project converter is unavailable.
    """
    hex_path = Path(hex_path)
    if not hex_path.exists():
        raise FileNotFoundError(f"CTD .hex not found: {hex_path}")
    xml = Path(xmlcon_path) if xmlcon_path else xmlcon_for(hex_path)

    cache = Path(cache_dir)
    out = cache / f"{station}_clean.cnv"
    if out.exists() and not force and out.stat().st_mtime >= hex_path.stat().st_mtime:
        return out                                            # fresh cache -> reuse

    convert_for_ladcp = _load_converter()
    cache.mkdir(parents=True, exist_ok=True)
    convert_for_ladcp(str(hex_path), str(xml), str(out), station=station)
    return out

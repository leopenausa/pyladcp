"""The ``ladcp-studio`` server: session state, JSON bridge, FastAPI app, entry point.

Endpoints (all JSON; profile arrays are NaN-sanitised to ``null`` for the browser):

* ``GET  /api/stations`` -- the launch work-list + whether a SADCP source is loaded.
* ``POST /api/station/{label}/prepare`` -- warm the edit-tier cache (~1.5 s cold).
* ``POST /api/station/{label}/solve`` -- solve a :class:`~ladcp.session.SessionConfig`
  on the cached context (~30 ms warm); the response carries the solved profile, the
  resolved declination, timings, the available QA panels, and the exact ``ladcp-qa``
  command line that reproduces it.
* ``POST /api/station/{label}/qa/{panel}`` -- one QA figure (the existing matplotlib
  panels: raw/depth/edit/velocity/weights/...) rendered as PNG for the posted config.

The static single-page UI (no build step) is served from ``static/`` at ``/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:                                  # optional extra: pip install 'pyladcp[gui]'
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.staticfiles import StaticFiles
except ImportError:                   # core install: module imports, create_app refuses
    FastAPI = HTTPException = Request = Response = StaticFiles = None

from ..session import (
    EditConfig,
    SadcpConfig,
    SessionConfig,
    SolveConfig,
    StationSession,
    parse_timeoff,
)

log = logging.getLogger("ladcp.studio")

MAX_SESSIONS = 3                      # LRU cap: a prepared session is order-100 MB
_QA_ROOT_DEFAULT = "New_golden/Good"  # ladcp-qa --root default (for minimal cli strings)
_QA_CRUISE_DEFAULT = "MORIA"


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


# ---------------------------------------------------------------------------
# JSON bridge


def config_from_body(body: dict, state: StudioState) -> SessionConfig:
    """Request JSON -> :class:`SessionConfig` (ValueError on malformed values).

    The SADCP *identities* are fixed at launch (``--sadcp`` / ``--sadcp-codas`` on
    ``ladcp-studio``); the request picks one by key via ``sadcp_key`` (``"off"``
    disables) and weights it via ``solve.sadcpfac``. The pre-dropdown boolean
    ``use_sadcp`` still works and means the first source.
    """
    e = dict(body.get("edit") or {})
    s = dict(body.get("solve") or {})
    nearfield = e.get("nearfield_dn_bins")
    edit = EditConfig(
        down_only=bool(e.get("down_only", False)),
        nearfield_dn_bins=None if nearfield is None else tuple(int(b) for b in nearfield),
        dzbelow=None if e.get("dzbelow") is None else float(e["dzbelow"]))
    solver = s.get("solver", "inverse")
    if solver not in ("shear", "inverse"):
        raise ValueError(f"solver must be 'shear' or 'inverse', got {solver!r}")
    solve = SolveConfig(
        solver=solver,
        drot=None if s.get("drot") is None else float(s["drot"]),
        botfac=float(s.get("botfac", 1.0)), barofac=float(s.get("barofac", 1.0)),
        smoofac=float(s.get("smoofac", 0.0)), sadcpfac=float(s.get("sadcpfac", 3.0)))
    sources = state.sadcp_sources
    key = body.get("sadcp_key")
    if key is None:                              # legacy boolean protocol
        # default ON only when a source was explicitly flagged: discovered ("found")
        # products are offered, never silently activated
        flagged = [c for k, c in sources.items()
                   if state.sadcp_origin.get(k, "flag") != "found"]
        use = bool(body.get("use_sadcp", bool(flagged)))
        pick = flagged[0] if flagged else (next(iter(sources.values())) if sources else None)
        sadcp = pick if use else None
    elif key == "off":
        sadcp = None
    elif key in sources:
        sadcp = sources[key]
    else:
        raise ValueError(f"unknown SADCP source {key!r} "
                         f"(launched with: {', '.join(sources) or 'none'})")
    return SessionConfig(edit=edit, sadcp=sadcp, solve=solve)


def _arr(x) -> list:
    """1-D array -> JSON-safe list (NaN/inf -> null; browsers reject NaN literals)."""
    a = np.asarray(x, float)
    return [float(v) if np.isfinite(v) else None for v in a]


def _num(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def solve_payload(state: StudioState, label: str, cfg: SessionConfig) -> dict:
    """Run one solve on the station's (locked) session and shape the JSON response."""
    with state.lock_for(label):
        ses = state.session(label)
        prepared = cfg.edit in ses._prepared
        t0 = time.perf_counter()
        result = ses.solve(cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        prep = ses._prepared[cfg.edit]
        stages = dict(prep.timings)
        dn = prep.dh.down                        # bins <-> metres for the near-field UI
        dn_geom = {"first_m": round(float(prep.dh.bin_depth(dn)[0]), 2),
                   "cell_m": float(dn.cell_m), "n_bins": int(dn.n_cells)}
        if cfg.solve.drot is not None:
            drot, drot_source = cfg.solve.drot, "explicit"
        else:
            drot, drot_source = ses.declination(cfg.edit)
    vp = result.vp
    bt = None
    if result.bp is not None and result.bp.n_bins > 0:
        bt = {"z": _arr(result.bp.z), "u": _arr(result.bp.u), "v": _arr(result.bp.v),
              "uerr": _arr(result.bp.uerr)}
    sadcp = None
    if result.sadcp is not None and len(result.sadcp):
        sa = np.asarray(result.sadcp, float)     # [m,4] rows: z, u, v, verr (true frame)
        sadcp = {"z": _arr(sa[:, 0]), "u": _arr(sa[:, 1]), "v": _arr(sa[:, 2]),
                 "verr": _arr(sa[:, 3])}
    return {
        "station": label,
        "solver": cfg.solve.solver,
        "drot": _num(drot), "drot_source": drot_source,
        "solve_ms": round(ms, 1), "prepared": prepared, "stages": stages,
        "zbottom": _num(result.zbottom),
        "profile": {"z": _arr(vp.z), "u": _arr(vp.u), "v": _arr(vp.v),
                    "uerr": _arr(vp.uerr), "nvel": _arr(vp.nvel),
                    "ubar": _num(vp.ubar), "vbar": _num(vp.vbar)},
        "bt": bt,
        "sadcp": sadcp,
        "sadcp_bins": 0 if result.sadcp is None else int(result.sadcp.shape[0]),
        "dn_geom": dn_geom,
        "panels": _available_panels(result),
        "cli": cfg.to_cli(label, **state.cli_context()),
    }


# ---------------------------------------------------------------------------
# QA panels: the existing matplotlib figures, rendered on demand for one config


_MPL_LOCK = threading.Lock()          # matplotlib is not thread-safe
_PNG_CACHE: OrderedDict[tuple, bytes] = OrderedDict()
_PNG_CACHE_MAX = 48


def _available_panels(result) -> list[str]:
    """Panel names renderable for this solve (acquisition panels are always on)."""
    names = ["raw", "alignment", "depth", "edit",
             "velocity", "shear", "inverse", "error", "drift"]
    if result.weights is not None:
        names.append("weights")
    if result.btrk is not None and result.btrk.n_own:
        names.append("btrack")
    if result.sadcp is not None and len(result.sadcp):
        names.append("sadcp")
    return names


def render_panel(state: StudioState, label: str, panel: str,
                 cfg: SessionConfig) -> bytes:
    """One QA panel as PNG bytes for ``cfg`` (solve is ~30 ms warm; render dominates)."""
    key = (label, cfg, panel)
    cached = _PNG_CACHE.get(key)
    if cached is not None:
        _PNG_CACHE.move_to_end(key)
        return cached

    with state.lock_for(label):
        ses = state.session(label)
        prep = ses.prepare(cfg.edit)
        result = ses.solve(cfg)

    import matplotlib
    matplotlib.use("Agg", force=False)
    import io as _io

    from ..plots.alignment import alignment_figure
    from ..plots.btrack_figure import btrack_figure
    from ..plots.depth_figure import depth_figure
    from ..plots.drift_figure import drift_figure
    from ..plots.edit_figure import edit_figure
    from ..plots.error_figure import error_figure
    from ..plots.inverse_figure import constraint_weights_figure, inverse_diagnostics_figure
    from ..plots.raw_dashboard import raw_dashboard
    from ..plots.sadcp_figure import sadcp_figure
    from ..plots.shear_figure import shear_figure
    from ..plots.velocity_figure import velocity_figure

    if panel not in _available_panels(result):
        raise KeyError(f"panel {panel!r} not available for this configuration "
                       f"(have: {', '.join(_available_panels(result))})")
    makers = {
        "raw": lambda: raw_dashboard(prep.dh),
        "alignment": lambda: alignment_figure(prep.dh),
        "depth": lambda: depth_figure(prep.dh, prep.ctd),
        "edit": lambda: edit_figure(prep.dh, prep.ctd),
        "velocity": lambda: velocity_figure(result.vp, bottom=result.bp, station=label),
        "shear": lambda: shear_figure(result.shear, station=label),
        "inverse": lambda: inverse_diagnostics_figure(result, station=label),
        "weights": lambda: constraint_weights_figure(result.weights, station=label),
        "error": lambda: error_figure(result, station=label),
        "drift": lambda: drift_figure(result, station=label),
        "btrack": lambda: btrack_figure(result, station=label),
        "sadcp": lambda: sadcp_figure(result, station=label),
    }
    with _MPL_LOCK:
        import matplotlib.pyplot as plt
        fig = makers[panel]()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
    png = buf.getvalue()
    _PNG_CACHE[key] = png
    while len(_PNG_CACHE) > _PNG_CACHE_MAX:
        _PNG_CACHE.popitem(last=False)
    return png


# ---------------------------------------------------------------------------
# app factory + entry point


def create_app(state: StudioState):
    """Build the FastAPI app over ``state`` (missing extra -> install hint)."""
    if FastAPI is None:                          # pragma: no cover - exercised manually
        raise SystemExit("ladcp-studio needs the GUI extra: "
                         "pip install 'pyladcp[gui]'")

    app = FastAPI(title="pyladcp studio", docs_url=None, redoc_url=None)

    def _check(label: str) -> None:
        if label not in state.labels and label not in state._explicit:
            raise HTTPException(404, f"unknown station {label!r} "
                                     f"(launched with: {', '.join(state.labels)})")

    from contextlib import contextmanager

    @contextmanager
    def _data_errors():
        """Processing-layer failures (bad paths, unreadable data) -> clean HTTP 400.

        Without this a bad --sadcp folder (e.g. 'no .STA files under ...') surfaces
        as a raw 500 traceback instead of a message the UI status line can show.
        """
        try:
            yield
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from None

    @app.get("/api/stations")
    def stations() -> dict:
        return {"stations": state.labels, "cruise": state.cruise,
                "sadcp": state.sadcp is not None,
                "sadcp_folder": state.sadcp.folder if state.sadcp else None,
                "sadcp_sources": [{"key": k, "source": c.source, "folder": c.folder,
                                   "origin": state.sadcp_origin.get(k, "flag")}
                                  for k, c in state.sadcp_sources.items()]}

    @app.post("/api/station/{label}/prepare")
    async def prepare(label: str, request: Request) -> dict:
        _check(label)
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        try:
            cfg = config_from_body(body, state)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from None
        with _data_errors(), state.lock_for(label):
            ses = state.session(label)
            cached = cfg.edit in ses._prepared
            t0 = time.perf_counter()
            ses.prepare(cfg.edit)
            ms = (time.perf_counter() - t0) * 1000.0
        return {"station": label, "cached": cached, "prepare_ms": round(ms, 1)}

    @app.post("/api/station/{label}/solve")
    async def solve(label: str, request: Request) -> dict:
        _check(label)
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        try:
            cfg = config_from_body(body, state)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from None
        with _data_errors():
            return solve_payload(state, label, cfg)

    @app.post("/api/station/{label}/lad")
    async def lad(label: str, request: Request) -> Response:
        """The current solution as an LDEO ``.lad`` text file (download)."""
        _check(label)
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        try:
            cfg = config_from_body(body, state)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from None
        import tempfile

        from ..qa.export import write_lad
        with _data_errors(), state.lock_for(label):
            ses = state.session(label)
            prep = ses.prepare(cfg.edit)
            result = ses.solve(cfg)
            drot = cfg.solve.drot
            if drot is None:
                drot, _src = ses.declination(cfg.edit)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{label}.lad"
            write_lad(result.vp, str(path), station=label, lat=prep.lat, lon=prep.lon,
                      drot=drot, time=prep.when)
            text = path.read_text(encoding="utf-8")
        return Response(content=text, media_type="text/plain",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{label}.lad"'})

    @app.post("/api/station/{label}/qa/{panel}")
    async def qa_panel(label: str, panel: str, request: Request) -> Response:
        _check(label)
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        try:
            cfg = config_from_body(body, state)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from None
        try:
            with _data_errors():
                png = render_panel(state, label, panel, cfg)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from None
        return Response(content=png, media_type="image/png")

    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static), html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-studio",
        description="pyladcp Studio: interactive single-station LADCP processing "
                    "in the browser (local server)")
    ap.add_argument("stations", nargs="*", help="station id(s) to work on, e.g. 80 79")
    ap.add_argument("--root", default=_QA_ROOT_DEFAULT,
                    help=f"base dir for file discovery (default: {_QA_ROOT_DEFAULT})")
    ap.add_argument("--cruise", default=_QA_CRUISE_DEFAULT,
                    help=f"cruise preset (default: {_QA_CRUISE_DEFAULT})")
    ap.add_argument("--index", default=None,
                    help="archive index JSON (ladcp-index build); with no station "
                         "ids, serves every cast in the index")
    ap.add_argument("--from-hex", action="store_true",
                    help="build missing cleaned CTD from the index's raw .hex anchor")
    ap.add_argument("--ctd-cache", default=None, help="cache dir for --from-hex .cnv")
    ap.add_argument("--sadcp", metavar="PATH", action="append", default=[],
                    help="ship-ADCP source for the inverse constraint (as in ladcp-qa); "
                         "repeatable — each folder becomes an entry in the GUI's "
                         "source dropdown (e.g. the 75 and 150 kHz instruments)")
    ap.add_argument("--sadcp-codas", metavar="PATH", action="append", default=[],
                    help="CODAS-processed product (contour NetCDF or its processing "
                         "dir) offered in the GUI's SADCP source dropdown alongside "
                         "the raw --sadcp; repeatable, one entry per product")
    ap.add_argument("--sadcp-source", choices=("vmdas", "codas"), default="vmdas")
    ap.add_argument("--sadcp-filetype", choices=("STA", "LTA"), default="STA")
    ap.add_argument("--sadcp-xducer", type=float, default=5.0)
    ap.add_argument("--sadcp-timeoff", default=None, metavar="SECONDS|auto")
    ap.add_argument("--sadcp-nav", metavar="PATH", default=None)
    ap.add_argument("--sadcp-reingest", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    ap.add_argument("--port", type=int, default=8642, help="port (default: 8642)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")   # QA panels render headless

    labels = list(args.stations)
    if not labels and args.index:
        from ..qa.cli import _all_station_labels
        labels = _all_station_labels(args.index, Path(args.root), args.cruise)
    if not labels:
        ap.error("give one or more station ids, or --index to serve the whole archive")

    raw_cfgs = []
    for p in args.sadcp:                     # the --sadcp-* knobs apply to every entry
        try:
            cfg = SadcpConfig(folder=p, source=args.sadcp_source,
                              filetype=args.sadcp_filetype, xducer=args.sadcp_xducer,
                              timeoff=parse_timeoff(args.sadcp_timeoff),
                              nav=args.sadcp_nav, reingest=args.sadcp_reingest)
            cfg.validate_folder()            # fail at launch, not at the first solve
        except ValueError as e:
            ap.error(str(e))
        raw_cfgs.append(cfg)

    codas_cfgs = []
    for p in args.sadcp_codas:
        from ..io.sadcp_codas import resolve_codas_nc
        try:
            resolve_codas_nc(p)              # fail at launch: exists + one unambiguous .nc
        except FileNotFoundError as e:
            ap.error(f"--sadcp-codas: {e}")
        codas_cfgs.append(SadcpConfig(folder=p, source="codas"))
    # products discovered under <root>/codas join the dropdown (selecting one is an
    # explicit user action; the default constraint stays with the explicit flags)
    found_cfgs = merge_discovered_codas(args.root, codas_cfgs)[len(codas_cfgs):]
    if found_cfgs:
        print(f"studio: offering CODAS products found under {Path(args.root) / 'codas'} "
              f"in the SADCP source dropdown: "
              + ", ".join(codas_label(c.folder) for c in found_cfgs), flush=True)

    # fail at launch, not as a 500 at the first solve: every station id must resolve
    # to files (catches a missing --root, a typo'd id, or a stray token parsed as a
    # station). Path checks only -- --from-hex CTD conversion still happens lazily.
    from ..discovery import _load_index, discover
    root_p = Path(args.root)
    idx = _load_index(Path(args.index)) if args.index else None
    for lab in labels:
        try:
            discover(lab, root=root_p, cruise=args.cruise, index=idx, from_hex=False)
        except (FileNotFoundError, ValueError) as e:
            msg = f"station {lab!r}: {e}"
            if not root_p.is_dir():
                msg += (f"  [--root {args.root!r} does not exist under {Path.cwd()} "
                        f"— pass --root <cruise folder>]")
            ap.error(msg)

    state = StudioState(labels, root=args.root, cruise=args.cruise, index=args.index,
                        from_hex=args.from_hex, ctd_cache=args.ctd_cache, sadcp=raw_cfgs,
                        sadcp_codas=codas_cfgs, sadcp_found=found_cfgs)
    app = create_app(state)

    import uvicorn
    url = f"http://{args.host}:{args.port}"
    print(f"pyladcp studio: {len(labels)} station(s) at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        import webbrowser
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0

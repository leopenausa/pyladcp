"""The Studio FastAPI app: routes only — state, payloads and rendering live next door."""
from __future__ import annotations

import time
from pathlib import Path

try:                                  # optional extra: pip install 'pyladcp[gui]'
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.staticfiles import StaticFiles
except ImportError:                   # core install: module imports, create_app refuses
    FastAPI = HTTPException = Request = Response = StaticFiles = None

from ..edits import new_journal, save_journal, verify_journal
from ..session import SessionConfig
from .payloads import _head_geom, _joint_n, config_from_body, solve_payload
from .render import render_heatmap, render_panel
from .state import StudioState


def create_app(state: StudioState, hub_dir=None):
    """Build the FastAPI app over ``state`` (missing extra -> install hint).

    ``hub_dir`` (a cruise directory) additionally mounts the ``/api/hub/*`` surface
    — the setup wizard + cruise dashboard served at ``/hub`` (wizard phase E).
    """
    if FastAPI is None:                          # pragma: no cover - exercised manually
        raise SystemExit("ladcp-studio needs the GUI extra: "
                         "pip install 'pyladcp[gui]'")

    app = FastAPI(title="pyladcp studio", docs_url=None, redoc_url=None)
    if hub_dir is not None:
        from .hub_api import add_hub_routes
        add_hub_routes(app, hub_dir)

    def _check(label: str) -> None:
        if not state.has_station(label):
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

    async def _body_config(label: str, request: Request) -> SessionConfig:
        """Request JSON -> config WITH the station's journal attached.

        Every config-consuming endpoint routes through here: the journal is the
        single source of truth for manual edits, so a request body never carries
        rectangles and no endpoint can solve/render a different edit set. The one
        escape hatch is ``ignore_edits: true`` -- an explicit request for the
        no-edits solution (the edit view's baseline inset / A-B comparison);
        the response then reports ``manual_edits: 0``.
        """
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        try:
            cfg = config_from_body(body, state)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from None
        if body.get("ignore_edits"):
            return cfg
        with _data_errors():
            return state.attach_edits(cfg, state.session(label))

    @app.post("/api/station/{label}/prepare")
    async def prepare(label: str, request: Request) -> dict:
        _check(label)
        cfg = await _body_config(label, request)
        with _data_errors(), state.lock_for(label):
            ses = state.session(label)
            cached = ses.is_prepared(cfg.edit)
            t0 = time.perf_counter()
            ses.prepare(cfg.edit)
            ms = (time.perf_counter() - t0) * 1000.0
        return {"station": label, "cached": cached, "prepare_ms": round(ms, 1)}

    @app.post("/api/station/{label}/solve")
    async def solve(label: str, request: Request) -> dict:
        _check(label)
        cfg = await _body_config(label, request)
        with _data_errors():
            return solve_payload(state, label, cfg)

    @app.post("/api/station/{label}/lad")
    async def lad(label: str, request: Request) -> Response:
        """The current solution as an LDEO ``.lad`` text file (download)."""
        _check(label)
        cfg = await _body_config(label, request)
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
        cfg = await _body_config(label, request)
        try:
            with _data_errors():
                png = render_panel(state, label, panel, cfg)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from None
        return Response(content=png, media_type="image/png")

    # -- brush Edit view ---------------------------------------------------------

    def _edits_payload(label: str) -> dict:
        """Journal + freshness for the UI (skeleton when no file exists yet)."""
        ses = state.session(label)
        p = state.edits_path(ses)
        j = state.load_edits(ses) or new_journal(ses.station)
        stale = None
        if j.entries:
            try:
                verify_journal(j, p, ses.down, ses.up)
            except ValueError as e:
                stale = str(e)
        return {"station": ses.station, "path": str(p), "stale": stale,
                "journal": j.to_dict()}

    @app.post("/api/station/{label}/edit/meta")
    async def edit_meta(label: str, request: Request) -> dict:
        """Grid geometry for the Edit view: the client maps pixels fractionally."""
        _check(label)
        cfg = await _body_config(label, request)
        with _data_errors(), state.lock_for(label):
            ses = state.session(label)
            prep = ses.prepare(cfg.edit)
            dh = prep.dh
            meta = {"station": label, "joint_n_ens": _joint_n(dh),
                    "heads": {"down": _head_geom(dh, "down"),
                              "up": _head_geom(dh, "up")}}
        meta.update(_edits_payload(label))
        return meta

    @app.post("/api/station/{label}/edit/heatmap/{head}/{view}")
    async def edit_heatmap(label: str, head: str, view: str,
                           request: Request) -> Response:
        _check(label)
        if head not in ("down", "up"):
            raise HTTPException(404, f"head must be 'down' or 'up', got {head!r}")
        cfg = await _body_config(label, request)
        try:
            with _data_errors():
                png = render_heatmap(state, label, head, view, cfg)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from None
        return Response(content=png, media_type="image/png")

    @app.get("/api/station/{label}/edits")
    def edits_get(label: str) -> dict:
        _check(label)
        with _data_errors(), state.lock_for(label):
            return _edits_payload(label)

    @app.post("/api/station/{label}/edits")
    async def edits_add(label: str, request: Request) -> dict:
        """Append one rectangle (creating the journal on first use) and persist.

        The body carries the entry plus the usual config keys (for ``down_only``
        geometry); rectangles are clamped to the station's real grid here so a
        persisted journal is in-range by construction.
        """
        _check(label)
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        entry = dict(body.get("entry") or {})
        try:
            cfg = config_from_body(body, state)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from None
        with _data_errors(), state.lock_for(label):
            ses = state.session(label)
            prep = ses.prepare(cfg.edit)
            dh = prep.dh
            head = entry.get("head")
            geom = _head_geom(dh, head) if head in ("down", "up") else None
            if geom is None:
                raise HTTPException(
                    400, f"entry head must be a present head, got {entry.get('head')!r}")
            n = _joint_n(dh)
            try:
                b0 = max(int(entry["bin_first"]), 1)
                b1 = min(int(entry["bin_last"]), geom["n_bins"])
                e0 = max(int(entry["ens_first"]), 0)
                e1 = min(int(entry["ens_last"]), n - 1)
            except (KeyError, TypeError, ValueError):
                raise HTTPException(
                    400, "entry needs integer bin_first/bin_last/ens_first/ens_last") \
                    from None
            if b0 > b1 or e0 > e1:
                raise HTTPException(400, "rectangle is empty after clamping to the grid")
            p = state.edits_path(ses)
            j = state.load_edits(ses) or new_journal(ses.station)
            if j.entries:                         # never grow a stale journal
                verify_journal(j, p, ses.down, ses.up)
            from datetime import datetime, timezone
            j.entries.append({
                "id": j.next_id, "kind": "rect", "head": head,
                "bin_first": b0, "bin_last": b1, "ens_first": e0, "ens_last": e1,
                "view": str(entry.get("view", "")), "note": str(entry.get("note", "")),
                "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            j.next_id += 1
            for name, hh, path in (("down", dh.down, ses.down), ("up", dh.up, ses.up)):
                if path is None:
                    continue
                fp = dict(j.raw.get(name) or {})
                fp["file"] = Path(path).name
                fp["size"] = Path(path).stat().st_size
                if hh is not None:                # n_ens only when this head is loaded
                    fp["n_ens"] = int(hh.n_ens)
                j.raw[name] = fp
            if dh.up is not None or ses.up is None:
                j.joint_n_ens = n                 # down_only never records a false joint
            save_journal(j, p)
            return _edits_payload(label)

    @app.delete("/api/station/{label}/edits/{entry_id}")
    def edits_delete(label: str, entry_id: int) -> dict:
        _check(label)
        with _data_errors(), state.lock_for(label):
            ses = state.session(label)
            j = state.load_edits(ses)
            if j is None or not any(e["id"] == entry_id for e in j.entries):
                raise HTTPException(404, f"no edit #{entry_id} in {label}'s journal")
            j.entries = [e for e in j.entries if e["id"] != entry_id]
            save_journal(j, state.edits_path(ses))
            return _edits_payload(label)

    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static), html=True), name="static")
    return app

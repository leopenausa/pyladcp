"""The Studio hub API: the setup wizard + cruise dashboard endpoints (wizard phase E).

The GUI front end of the ``ladcp`` cruise hub — every endpoint is a thin JSON shim
over the same engine the terminal uses (:mod:`ladcp.hub.detect`,
:mod:`~ladcp.hub.cruise_config`, :mod:`~ladcp.hub.status`,
:func:`ladcp.qa.batch.run_batch`), so the window and the CLI can never disagree.

Processing runs in one background thread at a time (a cruise laptop, not a job
farm): stations go through ``run_batch`` one by one so the job endpoint reports
exact per-station progress, and a second process request while one runs gets 409.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path

from ..hub import cruise_config as cc

try:                                  # optional extra: pip install 'pyladcp[gui]'
    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse
except ImportError:                   # pragma: no cover - core install
    HTTPException = Request = FileResponse = None


class _Job:
    """One background processing run: plan, per-station results, liveness."""

    def __init__(self, plan: list[str]):
        self.plan = list(plan)
        self.results: list[dict] = []          # {label, status} as they finish
        self.current: str | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def payload(self) -> dict:
        return {"running": self.running, "total": len(self.plan),
                "done": self.results, "current": self.current, "error": self.error}


def _run_job(job: _Job, ccfg: cc.CruiseConfig) -> None:
    """Process the plan station-by-station (exact progress; same batch machinery)."""
    from ..qa.batch import run_batch
    from ..session import SessionConfig

    args = cc.merged_qa_args(ccfg)
    try:
        cfg = SessionConfig.from_args(args)
        if cfg.sadcp is not None:
            cfg.sadcp.validate_folder()
    except ValueError as e:
        job.error = str(e)
        return
    for st in job.plan:
        job.current = st
        try:
            res = run_batch([st], cfg, root=args.root, cruise=args.cruise,
                            index=args.index, from_hex=args.from_hex,
                            ctd_cache=args.ctd_cache, outdir=args.outdir,
                            make_plots=True, formats={"xlsx", "odv", "nc", "csv"},
                            edits=args.edits, jobs=1,
                            params_global=ccfg.params_global,
                            params_station=ccfg.params_station)
            job.results.extend({"label": lb, "status": s} for lb, s in res)
        except Exception as e:                 # a bad cast must not kill the job thread
            job.results.append({"label": st, "status": "error"})
            job.error = f"{st}: {type(e).__name__}: {e}"
    job.current = None


class _Ek80Job:
    """One EK80 slim-extraction run: (station, src) plan, per-file progress."""

    def __init__(self, plan: list[tuple[str, str]], out_root: Path):
        self.plan = plan
        self.out_root = out_root
        self.done = 0
        self.ok = 0
        self.bytes = 0
        self.current: str | None = None
        self.errors: list[str] = []
        self.thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def payload(self) -> dict:
        return {"running": self.running, "total": len(self.plan), "done": self.done,
                "ok": self.ok, "bytes": self.bytes, "current": self.current,
                "errors": self.errors, "out": str(self.out_root)}


def _run_ek80_job(job: _Ek80Job) -> None:
    from ..hub import ek80_ops

    def progress(i, n, station, name):
        job.done = i - 1
        job.current = f"{station}/{name}"

    job.ok, job.bytes, job.errors = ek80_ops.extract_jobs(job.plan, job.out_root,
                                                          progress=progress)
    job.done = len(job.plan)
    job.current = None


def add_hub_routes(app, hub_dir: Path) -> None:
    """Mount the ``/api/hub/*`` surface for the cruise directory ``hub_dir``."""
    hub_dir = Path(hub_dir).resolve()
    jobs: dict[str, _Job] = {}                 # {"current": the one job}

    def _config_path() -> Path | None:
        p = hub_dir / cc.CONFIG_NAME
        return p if p.is_file() else None

    def _load() -> cc.CruiseConfig:
        p = _config_path()
        if p is None:
            raise HTTPException(409, f"no {cc.CONFIG_NAME} in {hub_dir} — "
                                     "finish the setup wizard first")
        try:
            return cc.load_config(p)
        except cc.ConfigError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/hub/state")
    def hub_state() -> dict:
        p = _config_path()
        return {"dir": str(hub_dir), "configured": p is not None,
                "config": str(p) if p else None}

    @app.get("/api/hub/detect")
    def hub_detect() -> dict:
        from ..hub.detect import detect
        return asdict(detect(hub_dir))

    @app.post("/api/hub/preview")
    async def hub_preview(request: Request) -> dict:
        import tomli_w
        raw = await request.json()
        try:
            cc._build(raw, hub_dir / cc.CONFIG_NAME)       # validate without writing
        except cc.ConfigError as e:
            raise HTTPException(400, str(e)) from None
        return {"toml": tomli_w.dumps(raw)}

    @app.post("/api/hub/config")
    async def hub_config(request: Request) -> dict:
        body = await request.json()
        raw = body.get("config", body)
        try:
            cc.save_config(raw, hub_dir / cc.CONFIG_NAME)
        except cc.ConfigError as e:
            raise HTTPException(400, str(e)) from None
        out = {"written": str(hub_dir / cc.CONFIG_NAME), "indexed": None}
        if body.get("build_index"):
            from ..archive import build_index
            det_ladcp = body.get("ladcp_dir", "LADCP")
            det_ctd = body.get("ctd_dir", "CTD")
            try:
                idx = build_index(hub_dir / det_ladcp, hub_dir / det_ctd, root=hub_dir)
                out["indexed"] = len(idx["casts"])
            except Exception as e:             # setup must survive a corrupt PD0
                out["index_error"] = f"{type(e).__name__}: {e}"
        return out

    @app.get("/api/hub/status")
    def hub_status() -> dict:
        from ..hub.status import gather
        return gather(_load())

    @app.post("/api/hub/process")
    async def hub_process(request: Request) -> dict:
        body = await request.json()
        job = jobs.get("current")
        if job is not None and job.running:
            raise HTTPException(409, "a processing run is already active")
        ccfg = _load()
        stations = body.get("stations") or []
        if not stations:
            from ..hub.status import gather
            data = gather(ccfg)
            mode = body.get("mode", "new")
            stations = [e["label"] for e in data["stations"]
                        if mode == "all" or e["freshness"] != "fresh"]
        if not stations:
            return {"started": False, "reason": "nothing to do — all casts current"}
        job = _Job(stations)
        job.thread = threading.Thread(target=_run_job, args=(job, ccfg), daemon=True)
        jobs["current"] = job
        job.thread.start()
        return {"started": True, "total": len(stations)}

    @app.get("/api/hub/job")
    def hub_job() -> dict:
        job = jobs.get("current")
        return job.payload() if job else {"running": False, "total": 0,
                                          "done": [], "current": None, "error": None}

    # -- EK80 share -> slim local copy (EK-B) --------------------------------------
    def _index_path() -> Path:
        args = cc.merged_qa_args(_load())
        return Path(args.index) if args.index else Path(args.root) / ".ladcp_archive.json"

    def _ek80_window(body) -> tuple[list[str], Path, float, float]:
        paths = body.get("paths") or []
        if not paths:
            raise HTTPException(400, "give at least one EK80 path (dir, glob or mount)")
        idx = _index_path()
        if not idx.is_file():
            raise HTTPException(409, "no archive index yet — EK80 windows need the "
                                     "cast times (finish setup / build the index first)")
        return paths, idx, float(body.get("pre", 20.0)), float(body.get("post", 170.0))

    @app.post("/api/hub/ek80/timetable")
    async def ek80_timetable(request: Request) -> dict:
        from ..hub import ek80_ops
        paths, idx, pre, post = _ek80_window(await request.json())
        try:
            table = ek80_ops.timetable(paths, idx, pre=pre, post=post)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from None
        table["cmd"] = ek80_ops.commands(paths, idx, hub_dir / ek80_ops.DEFAULT_OUT,
                                         pre=pre, post=post)["timetable"]
        return table

    @app.post("/api/hub/ek80/extract")
    async def ek80_extract(request: Request) -> dict:
        from ..hub import ek80_ops
        body = await request.json()
        ej = jobs.get("ek80")
        if ej is not None and ej.running:
            raise HTTPException(409, "an EK80 extraction is already running")
        paths, idx, pre, post = _ek80_window(body)
        try:
            plan = ek80_ops.build_jobs(paths, idx, pre=pre, post=post,
                                       stations=body.get("stations"))
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from None
        if not plan:
            return {"started": False,
                    "reason": "no EK80 files fall in any cast window (logging gap?)"}
        out_root = hub_dir / ek80_ops.DEFAULT_OUT
        ej = _Ek80Job(plan, out_root)
        ej.thread = threading.Thread(target=_run_ek80_job, args=(ej,), daemon=True)
        jobs["ek80"] = ej
        ej.thread.start()
        return {"started": True, "total": len(plan), "out": str(out_root),
                "cmd": ek80_ops.commands(paths, idx, out_root,
                                         pre=pre, post=post)["extract"]}

    @app.get("/api/hub/ek80/job")
    def ek80_job() -> dict:
        ej = jobs.get("ek80")
        return ej.payload() if ej else {"running": False, "total": 0, "done": 0,
                                        "ok": 0, "bytes": 0, "current": None,
                                        "errors": []}

    @app.get("/api/hub/config")
    def hub_config_get() -> dict:
        ccfg = _load()
        return {"path": str(ccfg.path), "raw": ccfg.raw}

    def _station_dir(label: str) -> Path:
        args = cc.merged_qa_args(_load())
        return Path(args.outdir) / "stations" / label

    @app.get("/api/hub/scorecard/{label}")
    def hub_scorecard(label: str) -> dict:
        d = _station_dir(label)
        txt = d / f"{label}_qa.txt"
        pdf = d / f"{label}_report.pdf"
        if not txt.is_file():
            raise HTTPException(404, f"no QA report for {label} yet")
        return {"text": txt.read_text(encoding="utf-8"), "pdf": pdf.is_file()}

    @app.get("/api/hub/report/{label}")
    def hub_report(label: str):
        pdf = _station_dir(label) / f"{label}_report.pdf"
        if not pdf.is_file():
            raise HTTPException(404, f"no PDF report for {label} (processed with "
                                     "--no-plots?)")
        return FileResponse(pdf, media_type="application/pdf", filename=pdf.name)

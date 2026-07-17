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

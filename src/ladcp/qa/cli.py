"""Command-line driver for the acquisition-QA stage.

Compact form — give one or more station ids and let it find the files under a root
directory (expects ``<root>/LADCP`` and ``<root>/CTD`` with the usual MORIA names)::

    ladcp-qa 80                       # one station
    ladcp-qa 79 80 82                 # batch
    ladcp-qa 80 --root New_golden/Good --out qa_out

Explicit form — name the files directly (for non-standard layouts)::

    ladcp-qa --down …-M.000 --up …-S.000 --ctd …_clean.cnv --station MORIA-80

Each station yields ``<station>_qa.txt`` + ``_qa.json`` (report), the QA PNGs, and a
combined ``<station>_report.pdf`` (scorecard + figures). ``--no-plots`` skips the figures.

Batches show a progress bar and stay quiet on the console; a failing station is logged and
skipped rather than aborting the run. ``--verbose`` streams per-station detail instead of the
bar, and a timestamped run log is written to ``<outdir>/ladcp-qa.log`` (``--log`` / ``--no-log``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from ..config import resolve_params
from ..discovery import discover
from ..io.ctd_cnv import read_ctd_cnv
from ..session import SessionConfig, resolve_declination
from .ingest import apply_header_config, load_dualhead
from .report import assess, text_report
from .runlog import ProgressBar, setup_logging, teardown_logging

log = logging.getLogger("ladcp.qa")


def _run_one(down, up, ctd_path, station, outdir, make_plots, drot=None,
             solver="inverse", sadcp_opts=None, cruise="MORIA", formats=None, ctd_utc=None,
             inv_opts=None):
    """Process one station into ``<outdir>/stations/<station>/``.

    Returns ``(status, export)``: ``status`` is the QA verdict string (``"ok"``/``"warn"``/
    ``"fail"``) and ``export`` is a :class:`~ladcp.export.StationExport` when a velocity
    solution was produced (``None`` for acquisition-only stations).
    """
    overrides = {}
    if inv_opts and inv_opts.get("nearfield_dn_bins") is not None:
        overrides["edit_nearfield_dn_bins"] = inv_opts["nearfield_dn_bins"]
    if inv_opts and inv_opts.get("dzbelow") is not None:
        overrides["dzbelow"] = inv_opts["dzbelow"]
    params = resolve_params(cruise, station, overrides=overrides or None)
    dh = load_dualhead(down, up, station=station, params=params)
    apply_header_config(params, dh)             # geometry/head-count from the PD0 headers
    ctd = read_ctd_cnv(ctd_path, params=params) if ctd_path else None
    if ctd is not None and ctd_utc and "utc_start" not in ctd.meta:
        ctd.meta["utc_start"] = ctd_utc         # index cast-start UTC -> sync prior
    qc = assess(dh, ctd=ctd)

    st_dir = Path(outdir) / "stations" / station
    fig_dir = st_dir / "figures"
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n", encoding="utf-8")
    (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2),
                                               encoding="utf-8")
    log.info("[%-5s] %s  ->  %s/", qc.overall_status.value.upper(), station, st_dir)

    # velocity solve (requires CTD + earth-frame data): .lad + .bot text, figures
    from ..models import CoordFrame
    earth = all(h.coord_frame == CoordFrame.EARTH for h in (dh.down, dh.up) if h is not None)
    result = None
    export = None
    down_only = bool(inv_opts and inv_opts.get("down_only"))
    if ctd is not None and not earth:
        log.warning("        velocity skipped: %s-coordinate data is unsupported (beam frames are "
                    "auto-rotated to earth at ingest; only earth/beam are handled); QA metrics "
                    "still written", dh.down.coord_frame.value)
    if ctd is not None and earth and not dh.has_up and not down_only:
        log.warning("        velocity skipped: no up-looker (pass --down-only to solve from "
                    "the down-looker alone); QA metrics still written")
    if ctd is not None and earth and (dh.has_up or down_only):
        dh_solve = dh
        if down_only and dh.has_up:
            from dataclasses import replace
            dh_solve = replace(dh, up=None)
            log.warning("        velocity: --down-only -- up-looker EXCLUDED from the solve "
                        "(acquisition QA above still covers both heads)")
        if not dh_solve.has_up:
            from ..models import Metric, Status
            qc.add(Metric("single_head_solve", "down-only", "", Status.WARN,
                          source_stage="qa.cli",
                          note="velocity solved from the down-looker alone: reduced "
                               "near-surface coverage, reference layer from down bins only"))
        result, meta = _velocity_outputs(dh_solve, ctd, station, st_dir, drot, solver,
                                         sadcp_opts, inv_opts)
        from ..qa.checks import consistency_checks
        for m in consistency_checks(result):       # checkinv -> scorecard
            qc.add(m)
        qc.add(_declination_metric(meta["drot"], meta["drot_source"]))
        # refresh the persisted QA text/json now that consistency checks are in
        (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n", encoding="utf-8")
        (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2),
                                                   encoding="utf-8")
        from ..export import StationExport
        export = StationExport(station=station, cruise=cruise, lat=meta["lat"],
                               lon=meta["lon"], time=meta["when"], drot=meta["drot"],
                               solver=solver, result=result, qc=qc,
                               sadcp_source=meta["sadcp_source"])

    if make_plots:
        from ..plots.pdf_report import build_report
        paths = build_report(dh, qc, str(st_dir), station, ctd=ctd, velocity=result,
                             figdir=str(fig_dir))
        log.info("        report: %s", paths["report.pdf"])

    if export is not None and formats:
        _write_station_exports(export, st_dir, formats)

    return qc.overall_status.value, export


def _velocity_outputs(dh, ctd, station, out, drot, solver="inverse", sadcp_opts=None,
                      inv_opts=None):
    import numpy as np

    from ..qa.export import write_bot, write_lad
    from ..qa.inverse import compute_velocity_full

    lat = float(np.nanmedian(ctd.lat))
    lon = float(np.nanmedian(ctd.lon))
    when = dh.down.time[0].astype("datetime64[s]").item()
    if drot is not None:
        drot_source = "explicit"                # user-supplied --drot
    else:                                       # IGRF-13 from cast position + date
        drot, drot_source = resolve_declination(lat, lon, when, logger=log)

    t_lad = dh.down.time
    sadcp = (_sadcp_profile(sadcp_opts, t_lad.min(), t_lad.max(), lat, lon, solver)
             if sadcp_opts else None)
    io = inv_opts or {}
    result = compute_velocity_full(dh, ctd, drot=drot, params=dh.params, solver=solver,
                                   sadcp=sadcp,
                                   sadcpfac=(sadcp_opts or {}).get("fac", 3.0),
                                   botfac=io.get("botfac", 1.0),
                                   barofac=io.get("barofac", 1.0),
                                   smoofac=io.get("smoofac", 0.0))
    vp, bp = result.vp, result.bp
    lad = out / f"{station}.lad"
    write_lad(vp, str(lad), station=station, lat=lat, lon=lon, drot=drot, time=when)
    log.info("        velocity: %s  (solver %s, drot %+.2f deg, ubar %+.3f)",
             lad, solver, drot, vp.ubar)

    if bp is not None and bp.n_bins > 0:
        bot = out / f"{station}.bot"
        write_bot(bp, str(bot), station=station, lat=lat, lon=lon, drot=drot,
                  zbottom=result.zbottom, time=when)
        log.info("        bottom-track: %s  (%d bins)", bot, bp.n_bins)

    if sadcp_opts and sadcp is not None:
        sadcp_src = sadcp_opts["folder"]
        if sadcp_opts.get("source") == "codas":
            sadcp_src = f"codas:{sadcp_src}"        # provenance: CODAS-calibrated product
    else:
        sadcp_src = None
    meta = {"lat": lat, "lon": lon, "when": when, "drot": drot, "drot_source": drot_source,
            "sadcp_source": sadcp_src}
    return result, meta


def _declination_metric(drot, source):
    """QA metric making the velocity frame's declination provenance visible (WARN if the
    IGRF lookup fell back to 0, i.e. the profile is in the *magnetic* frame, not true north)."""
    from ..models import Metric, Status
    note = {
        "igrf": "IGRF-13 from cast position + date",
        "explicit": "user-supplied --drot",
        "fallback-zero": "IGRF unavailable -- profile is in the MAGNETIC frame, NOT true north",
    }.get(source, source)
    return Metric(name="declination", value=round(float(drot), 3), unit="deg",
                  status=Status.OK if source in ("igrf", "explicit") else Status.WARN,
                  source_stage="qa.magdec", note=note)


def _write_station_exports(export, st_dir, formats) -> None:
    """Per-station shareable files: ``<station>.xlsx`` and ``<station>.nc``."""
    from ..export import ExportDependencyError

    station = export.station
    if "nc" in formats:
        from ..export.netcdf import write_station_nc
        write_station_nc(export, str(st_dir / f"{station}.nc"))
        log.info("        netcdf: %s", st_dir / f"{station}.nc")
    if "xlsx" in formats:
        from ..export.xlsx import write_station_xlsx
        try:
            write_station_xlsx(export, str(st_dir / f"{station}.xlsx"))
            log.info("        excel: %s", st_dir / f"{station}.xlsx")
        except ExportDependencyError as e:
            log.warning("        excel skipped: %s", e)


def _write_cruise_exports(exports, outdir, cruise, formats) -> None:
    """Cruise-level aggregates under ``<outdir>/exports/`` (batch / --all-stations only)."""
    from ..export import ExportDependencyError

    exp_dir = Path(outdir) / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if "csv" in formats:
        import pandas as pd

        from ..export.tables import summary_row
        csv = exp_dir / f"{cruise}_summary.csv"
        pd.DataFrame([summary_row(e) for e in exports]).to_csv(csv, index=False)
        log.info("  exports: %s", csv)
    if "odv" in formats:
        from ..export.odv import write_odv
        odv = exp_dir / f"{cruise}_ladcp_odv.txt"
        write_odv(exports, str(odv), cruise=cruise)
        log.info("  exports: %s", odv)
    if "nc" in formats:
        from ..export.netcdf import write_cruise_nc
        nc = exp_dir / f"{cruise}_ladcp.nc"
        write_cruise_nc(exports, str(nc), cruise=cruise)
        log.info("  exports: %s", nc)
    if "xlsx" in formats:
        from ..export.xlsx import write_cruise_xlsx
        xlsx = exp_dir / f"{cruise}_ladcp.xlsx"
        try:
            write_cruise_xlsx(exports, str(xlsx), cruise=cruise)
            log.info("  exports: %s", xlsx)
        except ExportDependencyError as e:
            log.warning("  exports: cruise Excel skipped: %s", e)


def _sadcp_profile(opts, time_start, time_end, lat, lon, solver):
    """Build the cast's ship-ADCP constraint profile from a VmDAS folder or CODAS file.

    Loads the dataset once (raw VmDAS folder ingested+cached, or a CODAS-processed
    NetCDF read directly per ``--sadcp-source``), then windows it to this cast's LADCP
    time span and position. Only the ``inverse`` solver consumes the constraint; with
    ``shear`` the folder is ignored with a notice. Returns the ``svel`` array or ``None``.
    """
    if solver != "inverse":
        log.info("        sadcp: ignored (constraint applies only to --solver inverse)")
        return None
    from ..io.sadcp_vmdas import extract_profile

    if opts.get("source") == "codas":
        from ..io.sadcp_codas import read_codas_nc
        ds = read_codas_nc(opts["folder"])
    else:
        from ..io.sadcp_vmdas import load_or_ingest
        ds = load_or_ingest(opts["folder"], cache=opts.get("cache"),
                            force=opts.get("reingest", False),
                            file_type=opts.get("file_type", "STA"),
                            transducer_depth=opts.get("xducer", 5.0))
    toff = opts.get("timeoff")
    if toff == "auto":
        from ..io.nav import estimate_time_offset, read_nav
        nav = read_nav(opts["nav"])
        est = estimate_time_offset(ds.time, ds.lat, ds.lon, nav)
        toff = est["offset_s"]
        opts["timeoff"] = toff          # estimate once, reuse for every station
        log.info("        sadcp: clock offset %+.2f s estimated from nav track "
                 "(track residual %.0f m median / %.0f m p90, overlap %.0f%%)",
                 toff, est["median_m"], est["p90_m"], 100 * est["overlap"])
    if toff:
        from ..io.nav import shift_time
        ds = shift_time(ds, float(toff))
    sv = extract_profile(ds, time_start=time_start, time_end=time_end, lat=lat, lon=lon,
                         dtok_min=opts.get("dtok_min", 0.0))
    if sv is None:
        log.info("        sadcp: no usable %s kHz data at this station "
                 "(time/position window empty) -- constraint skipped", ds.freq_khz)
    else:
        log.info("        sadcp: %d bins from %s kHz %s (%s)",
                 sv.shape[0], ds.freq_khz, ds.file_type, ds.source)
    return sv


# ---------------------------------------------------------------------------
# --jobs N: parallel batch over stations (one worker process per station).
# Stations are fully independent; ~80% of a station's wall time is figure
# rendering, so process-level parallelism scales near-linearly. The worker
# functions are top-level so they pickle under the spawn start method
# (Windows/macOS).

def _pool_init() -> None:
    """Worker initializer: force the headless matplotlib backend."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg", force=True)


def _pool_task(task: dict) -> tuple[int, str, str, object, str]:
    """Process one station in a worker: returns (index, label, status, export, log_text).

    Per-station log records are captured into a buffer (file-log format) and written
    sequentially by the parent, so ``ladcp-qa.log`` never interleaves stations.
    """
    import io as _io

    buf = _io.StringIO()
    lg = logging.getLogger("ladcp.qa")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    bh = logging.StreamHandler(buf)
    bh.setLevel(logging.DEBUG)
    bh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    lg.addHandler(bh)

    label, status, export = task["item"], "error", None
    try:
        sf = discover(task["item"], root=Path(task["root"]), cruise=task["cruise"],
                      index=task["index"], from_hex=task["from_hex"],
                      ctd_cache=task["ctd_cache"])
        label = sf.label
        status, export = _run_one(str(sf.down), str(sf.up) if sf.up else None,
                                  str(sf.ctd) if sf.ctd else None, label, task["outdir"],
                                  task["make_plots"], drot=task["drot"],
                                  solver=task["solver"], sadcp_opts=task["sadcp_opts"],
                                  cruise=task["cruise"], formats=task["formats"],
                                  ctd_utc=sf.ctd_utc, inv_opts=task["inv_opts"])
    except (Exception, SystemExit) as e:           # one bad cast must not abort the batch
        lg.error("[ERROR] %s: %s: %s", label, type(e).__name__, e, exc_info=True)
    bh.flush()
    return task["_i"], label, status, export, buf.getvalue()


def _append_worker_log(text: str) -> None:
    """Write a worker's captured log block into the parent's run-log file verbatim."""
    if not text:
        return
    for h in logging.getLogger("ladcp.qa").handlers:
        if isinstance(h, logging.FileHandler):
            h.stream.write(text)
            h.flush()


def _warm_sadcp(opts: dict) -> None:
    """Pre-fork SADCP setup: build the ingest cache and resolve ``timeoff='auto'`` once.

    Without this every worker would race to parse the raw VmDAS tree and re-estimate
    the clock offset against the nav track.
    """
    if opts.get("source") == "codas":
        ds = None                                   # NetCDF read is cheap per worker
    else:
        from ..io.sadcp_vmdas import load_or_ingest
        ds = load_or_ingest(opts["folder"], cache=opts.get("cache"),
                            force=opts.get("reingest", False),
                            file_type=opts.get("file_type", "STA"),
                            transducer_depth=opts.get("xducer", 5.0))
        opts["reingest"] = False                    # workers reuse the cache just built
    if opts.get("timeoff") == "auto":
        if ds is None:
            from ..io.sadcp_codas import read_codas_nc
            ds = read_codas_nc(opts["folder"])
        from ..io.nav import estimate_time_offset, read_nav
        nav = read_nav(opts["nav"])
        est = estimate_time_offset(ds.time, ds.lat, ds.lon, nav)
        opts["timeoff"] = est["offset_s"]
        log.info("sadcp: clock offset %+.2f s estimated from nav track (pre-fork; "
                 "track residual %.0f m median)", est["offset_s"], est["median_m"])


def build_parser() -> argparse.ArgumentParser:
    """The ``ladcp-qa`` argument parser (separate so tests can parse command strings)."""
    ap = argparse.ArgumentParser(prog="ladcp-qa",
                                 description="LADCP acquisition quality assessment")
    ap.add_argument("stations", nargs="*", help="station id(s), e.g. 80 or 79 80 82")
    ap.add_argument("--root", default="New_golden/Good",
                    help="base dir for file discovery (default: New_golden/Good)")
    ap.add_argument("--cruise", default="MORIA",
                    help="cruise preset for params + raw-archive manifest (default: MORIA)")
    ap.add_argument("--index", default=None,
                    help="archive index JSON (ladcp-index build); resolves raw files by station")
    ap.add_argument("-o", "--out", "--outdir", dest="outdir", default="qa_out",
                    help="output directory (default: qa_out)")
    ap.add_argument("--no-plots", action="store_true", help="skip figures/PDF")
    ap.add_argument("--drot", type=float, default=None,
                    help="magnetic declination [deg] for velocity (default: IGRF from position)")
    ap.add_argument("--solver", choices=("shear", "inverse"), default="inverse",
                    help="velocity solver: full constrained inverse (default) or shear "
                         "shape+reference")
    ap.add_argument("--botfac", type=float, default=1.0,
                    help="bottom-track constraint weight, legacy ps.botfac "
                         "(inverse only; default: 1)")
    ap.add_argument("--barofac", type=float, default=1.0,
                    help="GPS barotropic constraint weight, legacy ps.barofac "
                         "(inverse only; default: 1)")
    ap.add_argument("--smoofac", type=float, default=0.0,
                    help="curvature-smoothing weight, legacy ps.smoofac "
                         "(inverse only; default: 0, golden value)")
    ap.add_argument("--down-only", action="store_true",
                    help="solve velocity from the down-looker alone, ignoring any up-looker "
                         "(cross-check / single-instrument casts); acquisition QA still "
                         "covers both heads")
    ap.add_argument("--nearfield-dn-bins", metavar="LIST", default=None,
                    help="override the down-looker near-field device mask: comma 1-based "
                         "bins (e.g. 3,4) or 'none' to disable; default: the cruise preset "
                         "(MORIA sets 3,4 on the monocorer block 03-28)")
    # ship-ADCP (SADCP) constraint (inverse solver only)
    ap.add_argument("--dzbelow", type=float, default=None, metavar="METERS",
                    help="below-/near-seabed cell rejection margin [m] (default: cruise "
                         "preset, 16 = 2 legacy bins). Raise on shallow shelf casts when "
                         "a bottom-depth underestimate lets a contaminated near-bottom "
                         "cell through (e.g. 24-32)")
    ap.add_argument("--sadcp", metavar="PATH",
                    help="shipboard-ADCP data for the inverse constraint: a VmDAS folder "
                         "(STA/LTA; ingested once and cached as sadcp_cache.npz) or, with "
                         "--sadcp-source codas, a CODAS contour NetCDF (file or its "
                         "processing dir)")
    ap.add_argument("--sadcp-source", choices=("vmdas", "codas"), default="vmdas",
                    help="what --sadcp points at: raw VmDAS averages (default) or a "
                         "CODAS-processed (edited+calibrated) NetCDF product")
    ap.add_argument("--sadcpfac", type=float, default=3.0,
                    help="ship-ADCP constraint weight (default: 3, the golden value)")
    ap.add_argument("--sadcp-filetype", choices=("STA", "LTA"), default="STA",
                    help="VmDAS average to read (default: STA, short-term)")
    ap.add_argument("--sadcp-xducer", type=float, default=5.0,
                    help="SADCP transducer depth below waterline [m] (default: 5)")
    ap.add_argument("--sadcp-reingest", action="store_true",
                    help="re-parse the raw SADCP tree, ignoring any existing cache")
    ap.add_argument("--sadcp-timeoff", default=None, metavar="SECONDS|auto",
                    help="clock correction added to the SADCP timestamps [s], or 'auto' "
                         "to estimate it from --sadcp-nav by track matching (for "
                         "acquisition PCs that were not synchronised to GPS time)")
    ap.add_argument("--sadcp-nav", metavar="PATH", default=None,
                    help="independently timestamped navigation track (file or dir; SADO "
                         "'posicion' exports or time/lat/lon CSV) for --sadcp-timeoff auto")
    # explicit single-station override
    ap.add_argument("--down", help="down-looker (Master) PD0 file")
    ap.add_argument("--up", help="up-looker (Slave) PD0 file")
    ap.add_argument("--ctd", help="cleaned CTD .cnv (enables depth/bottom)")
    ap.add_argument("--from-hex", action="store_true",
                    help="if no cleaned CTD .cnv is found, build one from the index's raw "
                         "Seabird .hex anchor (needs CTD_project; off by default)")
    ap.add_argument("--ctd-cache", default=None,
                    help="dir to cache --from-hex converted .cnv for reuse "
                         "(default: ctd_from_hex)")
    ap.add_argument("--station", default="", help="station label (explicit mode)")
    # shareable exports (roadmap #3)
    ap.add_argument("--no-export", action="store_true",
                    help="skip the xlsx/odv/nc/csv exports (keep lad/bot/report/qa)")
    ap.add_argument("--formats", default="xlsx,odv,nc,csv",
                    help="comma-list of export formats to emit (default: xlsx,odv,nc,csv)")
    ap.add_argument("--all-stations", action="store_true",
                    help="process every station in the --index and build the cruise exports/")
    ap.add_argument("--cruise-export", action="store_true",
                    help="also build the cruise-level exports/ aggregate over the named stations")
    # run logging / progress (long batches)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream per-station detail to the console (default: progress bar only)")
    ap.add_argument("--log", metavar="FILE", default=None,
                    help="run-log path (default: <outdir>/ladcp-qa.log)")
    ap.add_argument("--no-log", action="store_true", help="do not write a run-log file")
    ap.add_argument("--no-progress", action="store_true", help="disable the batch progress bar")
    ap.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                    help="process N stations in parallel (default: 1; 0 = one per CPU). "
                         "Each worker holds one cast in memory -- reduce N if you swap")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    valid_fmts = {"xlsx", "odv", "nc", "csv"}
    if args.no_export:
        formats: set[str] = set()
    else:
        formats = {f.strip() for f in args.formats.split(",") if f.strip()}
        bad = formats - valid_fmts
        if bad:
            ap.error(f"unknown --formats value(s): {', '.join(sorted(bad))} "
                     f"(choose from {', '.join(sorted(valid_fmts))})")

    try:
        cfg = SessionConfig.from_args(args)
    except ValueError as e:           # same messages the inline checks used to emit
        ap.error(str(e))
    sadcp_opts = cfg.sadcp_opts()
    inv_opts = cfg.inv_opts()

    # resolve the work list: explicit single file set, or a batch of station ids
    explicit = bool(args.down)
    if explicit:
        plan = [args.station or Path(args.down).stem]
    else:
        plan = list(args.stations)
        if args.all_stations:
            plan = _all_station_labels(args.index, Path(args.root), args.cruise)
            if not plan:
                ap.error("--all-stations: no casts found in the archive index "
                         "(give --index path/to/.ladcp_archive.json)")
        if not plan:
            ap.error("give one or more station ids, use --all-stations, or --down/--up/--ctd")

    n = len(plan)
    console_detail = args.verbose or n <= 1            # stream detail for -v or a single cast
    logfile = None if args.no_log else (args.log or str(Path(args.outdir) / "ladcp-qa.log"))
    setup_logging(logfile, console_level=logging.INFO if console_detail else logging.WARNING)
    log.info("ladcp-qa: %d station(s), solver=%s, out=%s", n, args.solver, args.outdir)
    bar = ProgressBar(n, enabled=(not explicit and n > 1 and not console_detail
                                  and not args.no_progress))

    root = Path(args.root)
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, n))
    results: list[tuple[str, str]] = []                # (label, status)
    exports = []
    try:
        if jobs > 1 and not explicit:
            # one station per worker process: stations are fully independent, and the
            # bulk of a station's wall time is matplotlib rendering (CPU-bound)
            if sadcp_opts:
                _warm_sadcp(sadcp_opts)            # build the cache / resolve 'auto' ONCE
            log.info("parallel: %d worker processes", jobs)
            base = dict(root=str(root), cruise=args.cruise, index=args.index,
                        from_hex=args.from_hex, ctd_cache=args.ctd_cache,
                        outdir=args.outdir, make_plots=not args.no_plots, drot=args.drot,
                        solver=args.solver, sadcp_opts=sadcp_opts, formats=formats,
                        inv_opts=inv_opts)
            # each worker must NOT spin up a full BLAS thread pool: 6 workers x 16
            # OpenBLAS threads thrash the cores (measured 3x slower on the 40-cast
            # MORIA soak: 465s vs 148s). The pool itself is the parallelism, so
            # workers run single-threaded BLAS; an explicit user env setting wins.
            # The spawn context (all platforms) makes the limit apply at numpy load.
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx = multiprocessing.get_context("spawn")
            slots: list[tuple[str, str, object] | None] = [None] * n
            with ProcessPoolExecutor(max_workers=jobs, initializer=_pool_init,
                                     mp_context=ctx) as ex:
                futs = {ex.submit(_pool_task, dict(base, item=item, _i=i)): i
                        for i, item in enumerate(plan)}
                for fut in as_completed(futs):
                    try:
                        i, label, status, export, text = fut.result()
                    except Exception as e:         # un-picklable result / dead worker
                        i = futs[fut]
                        label, status, export = plan[i], "error", None
                        text = f"[ERROR] {label}: {type(e).__name__}: {e}\n"
                    slots[i] = (label, status, export)
                    _append_worker_log(text)
                    bar.advance(f"{label} [{status}]")
            for slot in slots:                     # plan order: deterministic summary/exports
                if slot is None:
                    continue
                label, status, export = slot
                results.append((label, status))
                if export is not None:
                    exports.append(export)
        else:
            for item in plan:
                label = item
                bar.start(label)
                ctd_utc = None
                try:
                    if explicit:
                        down, up, ctd_path = args.down, args.up, args.ctd
                    else:
                        sf = discover(item, root=root, cruise=args.cruise, index=args.index,
                                      from_hex=args.from_hex, ctd_cache=args.ctd_cache)
                        label = sf.label
                        down = str(sf.down)
                        up = str(sf.up) if sf.up else None
                        ctd_path = str(sf.ctd) if sf.ctd else None
                        ctd_utc = sf.ctd_utc
                    bar.start(label)
                    status, export = _run_one(down, up, ctd_path, label, args.outdir,
                                              not args.no_plots, drot=args.drot,
                                              solver=args.solver, sadcp_opts=sadcp_opts,
                                              cruise=args.cruise, formats=formats,
                                              ctd_utc=ctd_utc, inv_opts=inv_opts)
                    if export is not None:
                        exports.append(export)
                except (Exception, SystemExit) as e:   # one bad cast must not abort the batch
                    status = "error"
                    bar.clear()
                    log.error("[ERROR] %s: %s: %s", label, type(e).__name__, e, exc_info=True)
                results.append((label, status))
                bar.advance(f"{label} [{status}]")
        bar.close()

        # cruise-level aggregate only when explicitly requested (batch / whole index)
        if formats and exports and (args.all_stations or args.cruise_export):
            _write_cruise_exports(exports, args.outdir, args.cruise, formats)

        _log_summary(results, logfile, console_detail)
    finally:
        teardown_logging()

    return 1 if any(s in ("fail", "error") for _, s in results) else 0


def _log_summary(results: list[tuple[str, str]], logfile, console_detail: bool) -> None:
    """Log a one-line tally plus any problem stations; echo to console when it's quiet."""
    from collections import Counter

    counts = Counter(status for _, status in results)
    tally = ", ".join(f"{counts[k]} {k}" for k in ("ok", "warn", "fail", "error") if counts[k])
    summary = f"done: {len(results)} station(s) — {tally}"
    problems = [(label, status) for label, status in results if status in ("fail", "error")]
    log.info(summary)
    for label, status in problems:
        log.info("  %-14s %s", label, status)
    if logfile:
        log.info("run log: %s", logfile)
    if not console_detail:                             # console handler is quiet -> echo it
        print(summary)
        for label, status in problems:
            print(f"  {label}: {status}")
        if logfile:
            print(f"run log: {logfile}")


def _all_station_labels(index, root: Path, cruise: str) -> list[str]:
    """Every cast label in the archive index (for --all-stations)."""
    import json

    idx_path = Path(index) if index else root / ".ladcp_archive.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted((idx.get("casts") or {}).keys())


if __name__ == "__main__":
    raise SystemExit(main())

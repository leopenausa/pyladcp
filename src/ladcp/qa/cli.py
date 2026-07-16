"""``ladcp-qa`` — argument parsing, work-list resolution, logging and the worker pool.

The science lives in :mod:`ladcp.qa.pipeline` (its ``process_station`` is the
pipeline's table of contents; start reading there). This module only turns a command
line into ``process_station`` calls: serially, or one worker process per station
(``--jobs``).

Compact form — give one or more station ids and let it find the files under a root
directory (expects ``<root>/LADCP`` and ``<root>/CTD`` with the usual MORIA names)::

    ladcp-qa 80                       # one station
    ladcp-qa 79 80 82                 # batch
    ladcp-qa 80 --root New_golden/Good --out qa_out

Explicit form — name the files directly (for non-standard layouts)::

    ladcp-qa --down …-M.000 --up …-S.000 --ctd …_clean.cnv --station MORIA-80

A ``cruise.toml`` in the working directory (or a parent, or ``--config``) supplies
per-cruise defaults for all of these options — root, index, SADCP source, solver
knobs, and per-station ``[params]`` overrides (:mod:`ladcp.hub.cruise_config`).
Typed flags always win over the file.

Each station yields ``<station>_qa.txt`` + ``_qa.json`` (report), the QA PNGs, and a
combined ``<station>_report.pdf`` (scorecard + figures). ``--no-plots`` skips the figures.

Batches show a progress bar and stay quiet on the console; a failing station is logged and
skipped rather than aborting the run. ``--verbose`` streams per-station detail instead of the
bar, and a timestamped run log is written to ``<outdir>/ladcp-qa.log`` (``--log`` / ``--no-log``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..config import DEFAULT_CRUISE
from ..discovery import all_station_labels, discover
from ..hub.cruise_config import (
    ConfigError,
    apply_to_args,
    explicit_dests,
    find_config,
    load_config,
    merge_params,
    station_params,
)
from ..session import SessionConfig
from .pipeline import process_station, warm_sadcp, write_cruise_exports
from .runlog import ProgressBar, setup_logging, teardown_logging

log = logging.getLogger("ladcp.qa")


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
        pov = merge_params(task["params_global"], task["params_station"], label)
        status, export = process_station(str(sf.down), str(sf.up) if sf.up else None,
                                         str(sf.ctd) if sf.ctd else None, label,
                                         task["outdir"], task["make_plots"], task["cfg"],
                                         cruise=task["cruise"], formats=task["formats"],
                                         ctd_utc=sf.ctd_utc, edits=task.get("edits"),
                                         hint_root=task["root"],
                                         param_overrides=pov or None)
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


def build_parser() -> argparse.ArgumentParser:
    """The ``ladcp-qa`` argument parser (separate so tests can parse command strings)."""
    ap = argparse.ArgumentParser(prog="ladcp-qa",
                                 description="LADCP acquisition quality assessment")
    ap.add_argument("stations", nargs="*", help="station id(s), e.g. 80 or 79 80 82")
    ap.add_argument("--config", metavar="PATH", default=None,
                    help="cruise.toml supplying defaults for the options below (default: "
                         "auto-discovered in the current directory or its parents; typed "
                         "flags always override the file)")
    ap.add_argument("--no-config", action="store_true",
                    help="ignore any discovered cruise.toml (built-in defaults only)")
    ap.add_argument("--root", default="New_golden/Good",
                    help="base dir for file discovery (default: New_golden/Good)")
    ap.add_argument("--cruise", default=DEFAULT_CRUISE,
                    help="cruise preset for params + raw-archive manifest (default: "
                         f"{DEFAULT_CRUISE}, the generic operator-community defaults; a "
                         "registered name like MORIA adds cruise-specific layers)")
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
                    help="mask down-looker near-field bins (a device hung below the "
                         "package): comma 1-based bins, e.g. 3,4. Default: NO mask -- "
                         "always your explicit call; the nearfield_errvel_ratio WARN "
                         "names the bins to use when a hung device is detected")
    ap.add_argument("--edits", metavar="PATH", default=None,
                    help="replay manual brush edits from a Studio journal: a "
                         "<station>.json file (single station) or the .ladcp_edits "
                         "directory (per-station files looked up by label; a station "
                         "without a journal runs unedited). Edits are NEVER applied "
                         "without this flag; the applied set is listed in the QA "
                         "report")
    # ship-ADCP (SADCP) constraint (inverse solver only)
    ap.add_argument("--no-soundcorr", action="store_true",
                    help="disable the sound-speed correction of water velocities "
                         "(legacy p.soundcorr, ON by default): rescales ru/rv/rw per "
                         "ensemble by c_in-situ/c_firmware before super-ensembles")
    ap.add_argument("--dzbelow", type=float, default=None, metavar="METERS",
                    help="below-/near-seabed cell rejection margin [m] (default: cruise "
                         "preset, 16 = 2 legacy bins). Raise on shallow shelf casts when "
                         "a bottom-depth underestimate lets a contaminated near-bottom "
                         "cell through (e.g. 24-32)")
    ap.add_argument("--zbottom", type=float, default=None, metavar="METERS",
                    help="operator seabed-depth override [m] (legacy p.zbottom): use this "
                         "depth verbatim and skip auto detection. For fixing a detect_bottom "
                         "false-lock when the true depth is known (echo-sounder/logsheet). "
                         "Single-station runs only.")
    ap.add_argument("--guessbottom", type=float, default=None, metavar="METERS",
                    help="operator seabed seed [m] (legacy p.guessbottom): keep auto detection "
                         "but restrict the echo-stack search to within 50 m of this depth, "
                         "steering it off a far multiple. Single-station runs only.")
    ap.add_argument("--sadcp", metavar="PATH",
                    help="shipboard-ADCP data for the inverse constraint: a VmDAS folder "
                         "(STA/LTA; ingested once and cached as sadcp_cache.npz) or, with "
                         "--sadcp-source codas, a CODAS contour NetCDF (file or its "
                         "processing dir)")
    ap.add_argument("--sadcp-source", choices=("vmdas", "codas", "ek80"), default="vmdas",
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

    # cruise.toml layer: merged into the namespace BEFORE any option is consumed, so
    # the file flows through the exact validation the flags it stands in for would.
    # Precedence per knob: explicit flags > cruise.toml > preset > generic defaults.
    ccfg = None
    if args.config and args.no_config:
        ap.error("--config and --no-config are mutually exclusive")
    if not args.no_config:
        cfg_path = Path(args.config) if args.config else find_config()
        if args.config and not cfg_path.is_file():
            ap.error(f"--config: {cfg_path} does not exist")
        if cfg_path is not None:
            try:
                ccfg = load_config(cfg_path)
                apply_to_args(ccfg, args,
                              explicit_dests(build_parser,
                                             sys.argv[1:] if argv is None else argv))
            except ConfigError as e:
                ap.error(str(e))

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
        if cfg.sadcp is not None:     # fail at launch, not minutes in at the first solve
            cfg.sadcp.validate_folder()
    except ValueError as e:           # same messages the inline checks used to emit
        ap.error(str(e))

    # resolve the work list: explicit single file set, or a batch of station ids
    explicit = bool(args.down)
    if explicit:
        plan = [args.station or Path(args.down).stem]
    else:
        plan = list(args.stations)
        if args.all_stations:
            plan = all_station_labels(args.index, Path(args.root))
            if not plan:
                ap.error("--all-stations: no casts found in the archive index "
                         "(give --index path/to/.ladcp_archive.json)")
        if not plan:
            ap.error("give one or more station ids, use --all-stations, or --down/--up/--ctd")

    if args.edits:
        edits_p = Path(args.edits)
        if not edits_p.exists():
            ap.error(f"--edits: {edits_p} does not exist (give a journal file or the "
                     ".ladcp_edits directory)")
        if edits_p.is_file() and len(plan) > 1:
            ap.error("--edits FILE applies to a single station; give the .ladcp_edits "
                     "directory to batch-replay per-station journals")

    if (args.zbottom is not None or args.guessbottom is not None) and len(plan) > 1:
        ap.error("--zbottom/--guessbottom are per-cast seabed overrides; run one station "
                 f"at a time (got {len(plan)} stations)")

    n = len(plan)
    console_detail = args.verbose or n <= 1            # stream detail for -v or a single cast
    logfile = None if args.no_log else (args.log or str(Path(args.outdir) / "ladcp-qa.log"))
    setup_logging(logfile, console_level=logging.INFO if console_detail else logging.WARNING)
    if ccfg is not None:
        log.info("config: %s", ccfg.path)
    log.info("ladcp-qa: %d station(s), solver=%s, out=%s", n, args.solver, args.outdir)
    bar = ProgressBar(n, enabled=(not explicit and n > 1 and not console_detail
                                  and not args.no_progress))

    root = Path(args.root)
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, n))
    results: list[tuple[str, str]] = []                # (label, status)
    exports = []
    try:
        if cfg.sadcp is not None and cfg.solve.solver == "inverse":
            cfg = warm_sadcp(cfg)          # build the cache / resolve 'auto' ONCE per batch
        if jobs > 1 and not explicit:
            # one station per worker process: stations are fully independent, and the
            # bulk of a station's wall time is matplotlib rendering (CPU-bound)
            log.info("parallel: %d worker processes", jobs)
            base = dict(root=str(root), cruise=args.cruise, index=args.index,
                        from_hex=args.from_hex, ctd_cache=args.ctd_cache,
                        outdir=args.outdir, make_plots=not args.no_plots,
                        cfg=cfg, formats=formats, edits=args.edits,
                        params_global=ccfg.params_global if ccfg else {},
                        params_station=ccfg.params_station if ccfg else {})
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
                    pov = station_params(ccfg, label) if ccfg else None
                    status, export = process_station(down, up, ctd_path, label, args.outdir,
                                                     not args.no_plots, cfg,
                                                     cruise=args.cruise, formats=formats,
                                                     ctd_utc=ctd_utc, edits=args.edits,
                                                     hint_root=None if explicit else str(root),
                                                     param_overrides=pov or None)
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
            write_cruise_exports(exports, args.outdir, args.cruise, formats)

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


if __name__ == "__main__":
    raise SystemExit(main())

"""``ladcp-qa`` — argument parsing, work-list resolution and run logging.

The science lives in :mod:`ladcp.qa.pipeline` (its ``process_station`` is the
pipeline's table of contents; start reading there). This module only turns a command
line into a :func:`ladcp.qa.batch.run_batch` call — the shared batch loop (serial,
or one worker process per station with ``--jobs``) that the ``ladcp`` hub drives too.

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
import sys
from pathlib import Path

from ..config import DEFAULT_CRUISE
from ..discovery import all_station_labels
from ..hub.cruise_config import (
    ConfigError,
    apply_to_args,
    explicit_dests,
    find_config,
    load_config,
)
from ..session import SessionConfig
from .batch import log_summary, run_batch
from .runlog import setup_logging, teardown_logging

log = logging.getLogger("ladcp.qa")


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
                    help="solve velocity from the down-looker alone, EXCLUDING an up-looker "
                         "that exists (cross-check); casts with no up-looker file solve "
                         "down-only automatically; acquisition QA still covers both heads")
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

    try:
        results = run_batch(
            plan, cfg, root=args.root, cruise=args.cruise, index=args.index,
            from_hex=args.from_hex, ctd_cache=args.ctd_cache, outdir=args.outdir,
            make_plots=not args.no_plots, formats=formats, edits=args.edits,
            jobs=args.jobs,
            explicit_files=(args.down, args.up, args.ctd) if explicit else None,
            params_global=ccfg.params_global if ccfg else None,
            params_station=ccfg.params_station if ccfg else None,
            cruise_export=bool(args.all_stations or args.cruise_export),
            progress_enabled=(not explicit and n > 1 and not console_detail
                              and not args.no_progress))
        log_summary(results, logfile, console_detail)
    finally:
        teardown_logging()

    return 1 if any(s in ("fail", "error") for _, s in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

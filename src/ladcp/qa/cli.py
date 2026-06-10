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
from pathlib import Path

from ..config import resolve_params
from ..discovery import discover
from ..io.ctd_cnv import read_ctd_cnv
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
    (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n")
    (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2))
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
        (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n")
        (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2))
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
    else:
        drot_source = "igrf"                    # IGRF-13 from cast position + date
        try:
            from ..proc.magdec import magnetic_declination
            drot = magnetic_declination(lat, lon, when)
        except Exception as e:                  # ppigrf missing / bad coeffs / bad input
            log.warning("        declination: IGRF failed (%s: %s) -- velocity LEFT IN "
                        "MAGNETIC FRAME (drot=0, NOT true north)", type(e).__name__, e)
            drot, drot_source = 0.0, "fallback-zero"
        if not np.isfinite(drot):               # NaN position -> NaN declination
            log.warning("        declination: IGRF non-finite at lat=%.4f lon=%.4f (missing CTD "
                        "position?) -- velocity LEFT IN MAGNETIC FRAME (drot=0, NOT true north)",
                        lat, lon)
            drot, drot_source = 0.0, "fallback-zero"

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
    sv = extract_profile(ds, time_start=time_start, time_end=time_end, lat=lat, lon=lon,
                         dtok_min=opts.get("dtok_min", 0.0))
    if sv is None:
        log.info("        sadcp: no usable %s kHz data at this station "
                 "(time/position window empty) -- constraint skipped", ds.freq_khz)
    else:
        log.info("        sadcp: %d bins from %s kHz %s (%s)",
                 sv.shape[0], ds.freq_khz, ds.file_type, ds.source)
    return sv


def main(argv: list[str] | None = None) -> int:
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

    sadcp_opts = None
    if args.sadcp:
        sadcp_opts = {"folder": args.sadcp, "source": args.sadcp_source,
                      "fac": args.sadcpfac,
                      "file_type": args.sadcp_filetype, "xducer": args.sadcp_xducer,
                      "reingest": args.sadcp_reingest}
    nearfield = None
    if args.nearfield_dn_bins is not None:
        s = args.nearfield_dn_bins.strip().lower()
        try:
            nearfield = () if s in ("none", "") else tuple(
                int(b) for b in s.split(",") if b.strip())
        except ValueError:
            ap.error(f"--nearfield-dn-bins: expected comma-separated bin numbers or "
                     f"'none', got {args.nearfield_dn_bins!r}")
    inv_opts = {"botfac": args.botfac, "barofac": args.barofac, "smoofac": args.smoofac,
                "down_only": args.down_only, "nearfield_dn_bins": nearfield}

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
    results: list[tuple[str, str]] = []                # (label, status)
    exports = []
    try:
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
                                          cruise=args.cruise, formats=formats, ctd_utc=ctd_utc,
                                          inv_opts=inv_opts)
                if export is not None:
                    exports.append(export)
            except (Exception, SystemExit) as e:       # one bad cast must not abort the batch
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
        idx = json.loads(idx_path.read_text())
    except (OSError, ValueError):
        return []
    return sorted((idx.get("casts") or {}).keys())


if __name__ == "__main__":
    raise SystemExit(main())

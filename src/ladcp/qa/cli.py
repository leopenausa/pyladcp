"""Command-line driver for the acquisition-QA stage.

Compact form — give one or more station ids and let it find the files under a root
directory (expects ``<root>/LADCP`` and ``<root>/CTD`` with the usual MORIA names)::

    ladcp-qa 80                       # one station
    ladcp-qa 79 80 82                 # batch
    ladcp-qa 80 --root New_golden/Good --out qa_out

Explicit form — name the files directly (for non-standard layouts)::

    ladcp-qa --down …-M.000 --up …-S.000 --ctd …_clean.cnv --station MORIA-80

Each station yields ``<station>_qa.txt`` + ``_qa.json`` (report), the four QA PNGs, and a
combined ``<station>_report.pdf`` (scorecard + figures). ``--no-plots`` skips the figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import resolve_params
from ..discovery import discover
from ..io.ctd_cnv import read_ctd_cnv
from .ingest import apply_header_config, load_dualhead
from .report import assess, text_report


def _run_one(down, up, ctd_path, station, outdir, make_plots, drot=None,
             solver="shear", sadcp_opts=None, cruise="MORIA", formats=None):
    """Process one station into ``<outdir>/stations/<station>/``.

    Returns ``(rc, export)``: ``rc`` is the per-station exit code (1 only on FAIL) and
    ``export`` is a :class:`~ladcp.export.StationExport` when a velocity solution was
    produced (``None`` for acquisition-only stations).
    """
    params = resolve_params(cruise, station)
    dh = load_dualhead(down, up, station=station, params=params)
    apply_header_config(params, dh)             # geometry/head-count from the PD0 headers
    ctd = read_ctd_cnv(ctd_path, params=params) if ctd_path else None
    qc = assess(dh, ctd=ctd)

    st_dir = Path(outdir) / "stations" / station
    fig_dir = st_dir / "figures"
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n")
    (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2))
    print(f"[{qc.overall_status.value.upper():5}] {station}  ->  {st_dir}/")

    # velocity solve (requires both heads + CTD): .lad + .bot text, figures via the report
    result = None
    export = None
    if dh.has_up and ctd is not None:
        result, meta = _velocity_outputs(dh, ctd, station, st_dir, drot, solver, sadcp_opts)
        from ..qa.checks import consistency_checks
        for m in consistency_checks(result):       # checkinv -> scorecard
            qc.add(m)
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
        print(f"        report: {paths['report.pdf']}")

    if export is not None and formats:
        _write_station_exports(export, st_dir, formats)

    return (0 if qc.overall_status.value != "fail" else 1), export


def _velocity_outputs(dh, ctd, station, out, drot, solver="shear", sadcp_opts=None):
    import numpy as np

    from ..qa.export import write_bot, write_lad
    from ..qa.inverse import compute_velocity_full

    lat = float(np.nanmedian(ctd.lat))
    lon = float(np.nanmedian(ctd.lon))
    when = dh.down.time[0].astype("datetime64[s]").item()
    if drot is None:
        try:
            from ..proc.magdec import magnetic_declination
            drot = magnetic_declination(lat, lon, when)
        except Exception:
            drot = 0.0

    t_lad = dh.down.time
    sadcp = (_sadcp_profile(sadcp_opts, t_lad.min(), t_lad.max(), lat, lon, solver)
             if sadcp_opts else None)
    result = compute_velocity_full(dh, ctd, drot=drot, params=dh.params, solver=solver,
                                   sadcp=sadcp,
                                   sadcpfac=(sadcp_opts or {}).get("fac", 3.0))
    vp, bp = result.vp, result.bp
    lad = out / f"{station}.lad"
    write_lad(vp, str(lad), station=station, lat=lat, lon=lon, drot=drot, time=when)
    print(f"        velocity: {lad}  (solver {solver}, drot {drot:+.2f} deg, "
          f"ubar {vp.ubar:+.3f})")

    if bp is not None and bp.n_bins > 0:
        bot = out / f"{station}.bot"
        write_bot(bp, str(bot), station=station, lat=lat, lon=lon, drot=drot,
                  zbottom=result.zbottom, time=when)
        print(f"        bottom-track: {bot}  ({bp.n_bins} bins)")

    meta = {"lat": lat, "lon": lon, "when": when, "drot": drot,
            "sadcp_source": (sadcp_opts.get("folder") if sadcp_opts and sadcp is not None
                             else None)}
    return result, meta


def _write_station_exports(export, st_dir, formats) -> None:
    """Per-station shareable files: ``<station>.xlsx`` and ``<station>.nc``."""
    from ..export import ExportDependencyError

    station = export.station
    if "nc" in formats:
        from ..export.netcdf import write_station_nc
        write_station_nc(export, str(st_dir / f"{station}.nc"))
        print(f"        netcdf: {st_dir / f'{station}.nc'}")
    if "xlsx" in formats:
        from ..export.xlsx import write_station_xlsx
        try:
            write_station_xlsx(export, str(st_dir / f"{station}.xlsx"))
            print(f"        excel: {st_dir / f'{station}.xlsx'}")
        except ExportDependencyError as e:
            print(f"        excel skipped: {e}")


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
        print(f"  exports: {csv}")
    if "odv" in formats:
        from ..export.odv import write_odv
        odv = exp_dir / f"{cruise}_ladcp_odv.txt"
        write_odv(exports, str(odv), cruise=cruise)
        print(f"  exports: {odv}")
    if "nc" in formats:
        from ..export.netcdf import write_cruise_nc
        nc = exp_dir / f"{cruise}_ladcp.nc"
        write_cruise_nc(exports, str(nc), cruise=cruise)
        print(f"  exports: {nc}")
    if "xlsx" in formats:
        from ..export.xlsx import write_cruise_xlsx
        xlsx = exp_dir / f"{cruise}_ladcp.xlsx"
        try:
            write_cruise_xlsx(exports, str(xlsx), cruise=cruise)
            print(f"  exports: {xlsx}")
        except ExportDependencyError as e:
            print(f"  exports: cruise Excel skipped: {e}")


def _sadcp_profile(opts, time_start, time_end, lat, lon, solver):
    """Build the cast's ship-ADCP constraint profile from a VmDAS folder.

    Ingests (and caches) the folder once, then windows it to this cast's LADCP time span
    and position. Only the ``inverse`` solver consumes the constraint; with ``shear`` the
    folder is ignored with a notice. Returns the ``svel`` array or ``None``.
    """
    if solver != "inverse":
        print("        sadcp: ignored (constraint applies only to --solver inverse)")
        return None
    from ..io.sadcp_vmdas import extract_profile, load_or_ingest

    ds = load_or_ingest(opts["folder"], cache=opts.get("cache"),
                        force=opts.get("reingest", False),
                        file_type=opts.get("file_type", "STA"),
                        transducer_depth=opts.get("xducer", 5.0))
    sv = extract_profile(ds, time_start=time_start, time_end=time_end, lat=lat, lon=lon,
                         dtok_min=opts.get("dtok_min", 0.0))
    if sv is None:
        print(f"        sadcp: no usable {ds.freq_khz} kHz data at this station "
              "(time/position window empty) -- constraint skipped")
    else:
        print(f"        sadcp: {sv.shape[0]} bins from {ds.freq_khz} kHz "
              f"{ds.file_type} ({ds.source})")
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
    ap.add_argument("--solver", choices=("shear", "inverse"), default="shear",
                    help="velocity solver: shear shape+reference (default) or full inverse")
    # ship-ADCP (SADCP) constraint (inverse solver only)
    ap.add_argument("--sadcp", metavar="DIR",
                    help="VmDAS shipboard-ADCP folder (STA/LTA) for the inverse constraint; "
                         "ingested once and cached as sadcp_cache.npz")
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
        sadcp_opts = {"folder": args.sadcp, "fac": args.sadcpfac,
                      "file_type": args.sadcp_filetype, "xducer": args.sadcp_xducer,
                      "reingest": args.sadcp_reingest}

    if args.down:                                   # explicit mode (per-station only)
        station = args.station or Path(args.down).stem
        rc, _ = _run_one(args.down, args.up, args.ctd, station, args.outdir,
                         not args.no_plots, drot=args.drot, solver=args.solver,
                         sadcp_opts=sadcp_opts, cruise=args.cruise, formats=formats)
        return rc

    stations = list(args.stations)
    if args.all_stations:
        stations = _all_station_labels(args.index, Path(args.root), args.cruise)
        if not stations:
            ap.error("--all-stations: no casts found in the archive index "
                     "(give --index path/to/.ladcp_archive.json)")
    if not stations:
        ap.error("give one or more station ids, use --all-stations, or --down/--up/--ctd")

    root = Path(args.root)
    rc = 0
    exports = []
    for st in stations:
        sf = discover(st, root=root, cruise=args.cruise, index=args.index,
                      from_hex=args.from_hex, ctd_cache=args.ctd_cache)
        st_rc, export = _run_one(str(sf.down), (str(sf.up) if sf.up else None),
                                 (str(sf.ctd) if sf.ctd else None), sf.label, args.outdir,
                                 not args.no_plots, drot=args.drot, solver=args.solver,
                                 sadcp_opts=sadcp_opts, cruise=args.cruise, formats=formats)
        rc |= st_rc
        if export is not None:
            exports.append(export)

    # cruise-level aggregate only when explicitly requested (batch / whole index)
    if formats and exports and (args.all_stations or args.cruise_export):
        _write_cruise_exports(exports, args.outdir, args.cruise, formats)

    return rc


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

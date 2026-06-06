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

from ..config import moria05_params
from ..io.ctd_cnv import read_ctd_cnv
from .ingest import load_dualhead
from .report import assess, text_report


def _resolve(root: Path, st: str) -> tuple[str, str | None, str | None, str]:
    """Locate (down, up, ctd, label) for station id ``st`` under ``root``."""
    def pick(*patterns):
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise SystemExit(f"ambiguous match for {pat!r}: {[h.name for h in hits]}")
        return None

    down = pick(f"LADCP/*{st}*-M.000", f"LADCP/*{st}*M*.000")
    if down is None:
        raise SystemExit(f"no down-looker found for station {st!r} under {root}/LADCP")
    up = pick(f"LADCP/*{st}*-S.000", f"LADCP/*{st}*S*.000")
    ctd = pick(f"CTD/*{st}*.cnv")
    label = down.name.split("-LADCP")[0] if "-LADCP" in down.name else st
    return str(down), (str(up) if up else None), (str(ctd) if ctd else None), label


def _run_one(down, up, ctd_path, station, outdir, make_plots, drot=None,
             solver="shear", sadcp_opts=None) -> int:
    params = moria05_params()
    dh = load_dualhead(down, up, station=station, params=params)
    ctd = read_ctd_cnv(ctd_path, params=params) if ctd_path else None
    qc = assess(dh, ctd=ctd)

    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{station}_qa.txt").write_text(text_report(qc) + "\n")
    (out / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2))
    print(f"[{qc.overall_status.value.upper():5}] {station}  ->  {out}/")

    # velocity solve (requires both heads + CTD): .lad + .bot text, figures via the report
    result = None
    if dh.has_up and ctd is not None:
        result = _velocity_outputs(dh, ctd, station, out, drot, solver, sadcp_opts)
        from ..qa.checks import consistency_checks
        for m in consistency_checks(result):       # checkinv -> scorecard
            qc.add(m)
        # refresh the persisted QA text/json now that consistency checks are in
        (out / f"{station}_qa.txt").write_text(text_report(qc) + "\n")
        (out / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2))

    if make_plots:
        from ..plots.pdf_report import build_report
        paths = build_report(dh, qc, str(out), station, ctd=ctd, velocity=result)
        print(f"        report: {paths['report.pdf']}")

    return 0 if qc.overall_status.value != "fail" else 1


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

    return result


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
                    help="dir holding LADCP/ and CTD/ (default: New_golden/Good)")
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
    ap.add_argument("--station", default="", help="station label (explicit mode)")
    args = ap.parse_args(argv)

    sadcp_opts = None
    if args.sadcp:
        sadcp_opts = {"folder": args.sadcp, "fac": args.sadcpfac,
                      "file_type": args.sadcp_filetype, "xducer": args.sadcp_xducer,
                      "reingest": args.sadcp_reingest}

    if args.down:                                   # explicit mode
        station = args.station or Path(args.down).stem
        return _run_one(args.down, args.up, args.ctd, station, args.outdir,
                        not args.no_plots, drot=args.drot, solver=args.solver,
                        sadcp_opts=sadcp_opts)

    if not args.stations:
        ap.error("give one or more station ids, or use --down/--up/--ctd")

    root = Path(args.root)
    rc = 0
    for st in args.stations:
        down, up, ctd, label = _resolve(root, st)
        rc |= _run_one(down, up, ctd, label, args.outdir, not args.no_plots,
                       drot=args.drot, solver=args.solver, sadcp_opts=sadcp_opts)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

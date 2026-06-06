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
             solver="shear") -> int:
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
        result = _velocity_outputs(dh, ctd, station, out, drot, solver)

    if make_plots:
        from ..plots.pdf_report import build_report
        paths = build_report(dh, qc, str(out), station, ctd=ctd, velocity=result)
        print(f"        report: {paths['report.pdf']}")

    return 0 if qc.overall_status.value != "fail" else 1


def _velocity_outputs(dh, ctd, station, out, drot, solver="shear"):
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

    result = compute_velocity_full(dh, ctd, drot=drot, params=dh.params, solver=solver)
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
    # explicit single-station override
    ap.add_argument("--down", help="down-looker (Master) PD0 file")
    ap.add_argument("--up", help="up-looker (Slave) PD0 file")
    ap.add_argument("--ctd", help="cleaned CTD .cnv (enables depth/bottom)")
    ap.add_argument("--station", default="", help="station label (explicit mode)")
    args = ap.parse_args(argv)

    if args.down:                                   # explicit mode
        station = args.station or Path(args.down).stem
        return _run_one(args.down, args.up, args.ctd, station, args.outdir,
                        not args.no_plots, drot=args.drot, solver=args.solver)

    if not args.stations:
        ap.error("give one or more station ids, or use --down/--up/--ctd")

    root = Path(args.root)
    rc = 0
    for st in args.stations:
        down, up, ctd, label = _resolve(root, st)
        rc |= _run_one(down, up, ctd, label, args.outdir, not args.no_plots,
                       drot=args.drot, solver=args.solver)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

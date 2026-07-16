"""``ladcp-ek80`` — prep Simrad EK80 ADCP-mode currents for the inverse constraint.

The EK80 echosounder, run in ADCP/current mode, can stand in for a ship-ADCP where none
is available. Its processed SONAR-netCDF4 files are large (~0.5 GB each, hundreds per
cruise), so this command exists to work with them **without copying everything local**:

  ladcp-ek80 timetable <dir|glob ...> --index .ladcp_archive.json   # what covers each cast
  ladcp-ek80 extract   <dir|glob ...> --index .ladcp_archive.json --out adcp_local

``timetable`` reads only filenames + tiny time headers; ``extract`` slims each file to its
ADCP current group (~30–100 MB) into ``<out>/<station>/``. Point ``ladcp-qa --sadcp`` at a
station folder with ``--sadcp-source ek80`` (see guide chapter 8). The slim copies are
read identically to the originals by :mod:`ladcp.io.sadcp_ek80`.
"""
from __future__ import annotations

import argparse
import csv
import os

from .io import ek80_files as ek


def _fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "?"


def _cmd_timetable(a) -> int:
    rows = ek.scan(a.paths, peek=not a.no_peek)
    if not rows:
        print("no EK80 files found")
        return 1
    print(f"{'file':52} {'start (UTC)':19} {'end (UTC)':19} {'dur':>7} {'n':>6} src")
    print("-" * 114)
    for r in rows:
        dur = (f"{(r['end'] - r['start']).total_seconds() / 60:.1f}m"
               if (r["start"] and r["end"]) else "?")
        print(f"{os.path.basename(r['file']):52} {_fmt(r['start']):19} {_fmt(r['end']):19} "
              f"{dur:>7} {str(r['n'] or ''):>6} {r['time_source']}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "start_utc", "end_utc", "duration_min", "n_ens",
                        "lat", "lon", "time_source"])
            for r in rows:
                dur = ((r["end"] - r["start"]).total_seconds() / 60
                       if (r["start"] and r["end"]) else "")
                w.writerow([r["file"], r["start"].isoformat() if r["start"] else "",
                            r["end"].isoformat() if r["end"] else "",
                            f"{dur:.2f}" if dur != "" else "", r["n"] or "",
                            r["lat"] if r["lat"] is not None else "",
                            r["lon"] if r["lon"] is not None else "", r["time_source"]])
        print(f"\nwrote {a.csv}")

    if a.index:
        casts = ek.read_casts(a.index)
        want = {s.strip() for s in a.station.split(",")} if a.station else None
        if want:
            casts = [c for c in casts if c[0] in want]
        mapping = ek.correlate(rows, casts, pre_min=a.pre, post_min=a.post)
        print(f"\n{'station':14} {'cast UTC':19}  EK80 files in window")
        print("-" * 60)
        for label, utc, *_ in casts:
            files = mapping.get(label, [])
            print(f"{label:14} {_fmt(utc):19}  {len(files):2d} file(s)"
                  + (f"  {os.path.basename(files[0])} .. {os.path.basename(files[-1])}"
                     if files else "  — none (no EK80 coverage)"))
    return 0


def _cmd_extract(a) -> int:
    # Build the (station, src) job list: from an index window, or a flat list.
    jobs: list[tuple[str, str]] = []
    if a.index:
        rows = ek.scan(a.paths, peek=False)        # filenames are enough to window
        mapping = ek.correlate(rows, ek.read_casts(a.index), pre_min=a.pre, post_min=a.post)
        want = {s.strip() for s in a.station.split(",")} if a.station else None
        if want:
            missing = want - set(mapping)
            if missing:
                print(f"station(s) not in index: {', '.join(sorted(missing))}")
                return 1
            mapping = {s: f for s, f in mapping.items() if s in want}
        for station, files in mapping.items():
            jobs += [(station, f) for f in files]
        if not jobs:
            scope = f"station {a.station}" if a.station else "any cast window"
            print(f"no EK80 files fall in {scope} (likely an EK80 logging gap over that cast)")
            return 1
    else:
        jobs = [("", f) for f in ek.collect(a.paths) if f.lower().endswith(".nc")]
        if not jobs:
            print("no EK80 .nc files found")
            return 1

    ok = tot = 0
    for i, (station, src) in enumerate(jobs, 1):
        dst = os.path.join(a.out, station, os.path.basename(src))
        try:
            sz = ek.slim_extract(src, dst)
            ok += 1
            tot += sz
            print(f"[{i}/{len(jobs)}] {station or '-':12} {os.path.basename(src):46} "
                  f"-> {sz / 1e6:.1f} MB")
        except Exception as e:
            print(f"[{i}/{len(jobs)}] {station or '-':12} {os.path.basename(src):46} "
                  f"FAIL {type(e).__name__}: {e}")
    print(f"\ndone: {ok}/{len(jobs)} files, {tot / 1e6:.0f} MB into {a.out}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-ek80",
        description="prep EK80 ADCP-mode currents for the --sadcp-source ek80 constraint")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("timetable", help="low-IO start/end table; correlate casts with --index")
    t.add_argument("paths", nargs="+", help="EK80 dirs or globs (.nc and/or .raw)")
    t.add_argument("--no-peek", action="store_true",
                   help="filenames only (skip the nc/raw time-header peek)")
    t.add_argument("--index", help=".ladcp_archive.json: map each cast to its EK80 files")
    t.add_argument("--station", help="with --index: restrict the cast map to this station "
                                     "(comma-separated for several)")
    t.add_argument("--csv", help="also write the table to this CSV")
    t.add_argument("--pre", type=float, default=20.0,
                   help="cast-window start, minutes before cast UTC (default 20)")
    t.add_argument("--post", type=float, default=170.0,
                   help="cast-window end, minutes after cast UTC (default 170)")
    t.set_defaults(func=_cmd_timetable)

    e = sub.add_parser("extract",
                       help="slim each file to its ADCP current group (~30-100 MB) under --out")
    e.add_argument("paths", nargs="+", help="EK80 dirs or globs of .nc files (e.g. an SMB mount)")
    e.add_argument("--out", required=True, help="output root; files land in <out>/<station>/")
    e.add_argument("--index", help=".ladcp_archive.json: keep only on-station files, by station")
    e.add_argument("--station", help="with --index: extract just this station "
                                     "(comma-separated for several), into <out>/<station>/")
    e.add_argument("--pre", type=float, default=20.0, help="cast-window pre-minutes (default 20)")
    e.add_argument("--post", type=float, default=170.0,
                   help="cast-window post-minutes (default 170)")
    e.set_defaults(func=_cmd_extract)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

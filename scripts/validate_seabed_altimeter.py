#!/usr/bin/env python
"""Validate seabed detection against the CTD altimeter (diagnostic, not a CI test).

The CTD altimeter reads height above the seabed and is reliable within ~28 m of the bed, so
``seabed = depth + altimeter`` (over those samples) is an independent ground truth for casts
that approached close enough. This scores :func:`ladcp.qa.bottom.detect_bottom` against it
across a cruise's stations. It is a *diagnostic* tool -- not all casts carry an altimeter and
it is not part of routine LADCP processing -- so it lives here, outside the test suite, and
depends on local cruise data.

Usage::

    python scripts/validate_seabed_altimeter.py \
        --root /path/to/cruise/pyladcp_data/MORIA \
        --altimeter-cnv /path/to/CTD/raw/sensor_comparison \
        --stations 80 10 03 20 ...        # default: every cast in the index

The ``--altimeter-cnv`` directory holds Sea-Bird ``.cnv`` files with an ``altM`` column.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np


def altimeter_seabed(cnv_dir: Path, station: str, *, alt_max: float = 28.0,
                     min_samples: int = 3) -> float:
    """Median ``depth + altimeter`` over samples within ``alt_max`` of the bed, or NaN."""
    matches = list(cnv_dir.glob(f"*{station}*.cnv"))
    if not matches:
        return float("nan")
    names: dict[str, int] = {}
    rows: list[list[float]] = []
    started = False
    for ln in matches[0].read_text(errors="ignore").splitlines():
        if ln.startswith("# name"):
            m = re.match(r"# name (\d+) = (\S+):", ln)
            if m:
                names[m.group(2)] = int(m.group(1))
        elif ln.strip() == "*END*":
            started = True
        elif started:
            try:
                rows.append([float(x) for x in ln.split()])
            except ValueError:
                pass
    if "altM" not in names or not rows:
        return float("nan")
    d = np.array(rows)
    alt, dep = d[:, names["altM"]], d[:, names["depSM"]]
    ok = (alt > 0.5) & (alt < alt_max) & np.isfinite(dep)
    return float(np.median(dep[ok] + alt[ok])) if ok.sum() >= min_samples else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="cruise pyladcp_data root")
    ap.add_argument("--altimeter-cnv", required=True, type=Path,
                    help="dir of operator .cnv files carrying the altM column")
    ap.add_argument("--cruise", default="MORIA")
    ap.add_argument("--ctd-cache", default="ctd_from_hex")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="station ids (default: every cast in the archive index)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from ladcp.config import resolve_params
    from ladcp.discovery import discover
    from ladcp.io.ctd_cnv import read_ctd_cnv
    from ladcp.qa.bottom import detect_bottom
    from ladcp.qa.depth import synchronize
    from ladcp.qa.ingest import apply_header_config, load_dualhead

    index = str(args.root / ".ladcp_archive.json")
    casts = json.load(open(index))["casts"]
    stations = args.stations or [k.replace(f"{args.cruise}-", "") for k in sorted(casts)]

    print(f"{'station':12s}{'altimeter':>10s}{'detect':>9s}{'diff':>8s}{'err':>7s}{'mode':>7s}")
    errs = []
    for s in stations:
        alt = altimeter_seabed(args.altimeter_cnv, s)
        try:
            sf = discover(s, root=args.root, cruise=args.cruise, index=index,
                          from_hex=True, ctd_cache=args.ctd_cache)
            p = resolve_params(args.cruise, sf.label)
            dh = load_dualhead(str(sf.down), str(sf.up) if sf.up else None,
                               station=sf.label, params=p)
            apply_header_config(p, dh)
            ctd = read_ctd_cnv(str(sf.ctd), params=p)
            b = detect_bottom(dh, synchronize(dh, ctd), ctd=ctd)
        except Exception as e:  # noqa: BLE001 - diagnostic, report and continue
            print(f"{s:12s}  ERR {type(e).__name__}: {e}")
            continue
        diff = b.zbottom - alt if np.isfinite(b.zbottom) and np.isfinite(alt) else float("nan")
        if np.isfinite(diff):
            errs.append(abs(diff))
        mode = "echo" if np.isfinite(b.error) else "zmax"
        flag = " <<" if np.isfinite(diff) and abs(diff) > 20 else ""
        print(f"{sf.label:12s}{alt:10.1f}{b.zbottom:9.1f}{diff:8.1f}{b.error:7.1f}"
              f"{mode:>7s}{flag}")
    if errs:
        e = np.array(errs)
        print(f"\nN={e.size}  median|diff|={np.median(e):.1f}  mean={np.mean(e):.1f}  "
              f"max={np.max(e):.1f} m   >20 m: {(e > 20).sum()}")


if __name__ == "__main__":
    main()

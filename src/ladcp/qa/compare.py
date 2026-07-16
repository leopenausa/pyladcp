"""``ladcp-compare`` -- cruise-level pyladcp vs legacy-LDEO comparison report.

Pairs every station of a ``ladcp-qa`` run with the corresponding legacy LDEO_IX
result (``*.mat`` with a ``dr`` struct) **by cast time**, so no filename
convention is assumed on either side: legacy products carry ``dr.date``, ours
carry the ``time_utc`` NetCDF attribute. Produces:

* ``comparison.csv`` -- one row per paired station (corr / rms / bias for u, v,
  bottom track; barotropic offsets; depth coverage),
* ``comparison_report.pdf`` -- per-station u/v profile overlays + a summary page
  (rms/corr per station, ubar/vbar scatter, depth coverage),
* unpaired stations on both sides listed explicitly (nothing dropped silently).

CLI::

    ladcp-compare --ours <qa_out> --legacy <LADCP_processed> -o <report_dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .validate import DrTargets, Score, load_dr_mat, score

_PAIR_TOL_H = 3.0           # max |cast-time difference| for a station pair


@dataclass
class LegacyCast:
    path: Path
    name: str
    time: np.datetime64 | None
    lat: float
    lon: float
    dr: DrTargets
    tim_span: tuple[np.datetime64, np.datetime64] | None = None   # SE grid span (dr.tim)


@dataclass
class OurCast:
    path: Path                  # the station .nc
    station: str
    time: np.datetime64 | None
    lat: float
    lon: float
    variant: str = ""           # non-empty when substituted from an alternate run


def _legacy_time(dr_raw) -> np.datetime64 | None:
    """Cast time from a legacy ``dr.date`` ([y m d H M S]) if present."""
    date = getattr(dr_raw, "date", None)
    if date is None:
        return None
    d = np.asarray(date).ravel().astype(int)
    if d.size < 6:
        return None
    return np.datetime64(f"{d[0]:04d}-{d[1]:02d}-{d[2]:02d}"
                         f"T{d[3]:02d}:{d[4]:02d}:{d[5]:02d}")


def _legacy_tim_span(dr_raw):
    """Super-ensemble time span from ``dr.tim`` (LDEO julian days, midnight-based)."""
    tim = getattr(dr_raw, "tim", None)
    if tim is None:
        return None
    t = np.asarray(tim, float).ravel()
    t = t[np.isfinite(t)]
    if t.size < 2:
        return None
    # LDEO julian.m counts from noon (astronomical JD); decoding against midnight
    # removes the 12 h: verified on CRUISE2_001 (cast 01:36 UTC, tim decodes 13:36
    # under the astronomical epoch)
    epoch = np.datetime64("2000-01-01T00:00:00")
    def conv(jd):
        return epoch + np.timedelta64(round((jd - 2451545.0) * 86400 * 1000), "ms")
    return conv(float(t.min())), conv(float(t.max()))


def scan_legacy(folder: str | Path) -> list[LegacyCast]:
    """Load every legacy result ``*.mat`` (skipping ``*.ladcp.mat`` companions)."""
    import scipy.io as sio

    out = []
    for p in sorted(Path(folder).glob("*.mat")):
        if p.name.endswith(".ladcp.mat"):
            continue
        try:
            raw = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False,
                              variable_names=["dr"])["dr"]
        except (KeyError, ValueError, OSError, NotImplementedError):
            continue                            # not a dr-result mat
        out.append(LegacyCast(
            path=p, name=p.stem, time=_legacy_time(raw),
            lat=float(getattr(raw, "lat", np.nan)),
            lon=float(getattr(raw, "lon", np.nan)),
            dr=load_dr_mat(p),
            tim_span=_legacy_tim_span(raw),
        ))
    return out


def scan_ours(folder: str | Path) -> list[OurCast]:
    """Find every station NetCDF under ``<folder>/stations/*/<station>.nc``."""
    import xarray as xr

    out = []
    for p in sorted(Path(folder).glob("stations/*/*.nc")):
        if p.parent.name != p.stem:
            continue                            # only the station's own product
        with xr.open_dataset(p) as ds:
            t = ds.attrs.get("time_utc")
            out.append(OurCast(
                path=p, station=str(ds.attrs.get("station", p.stem)),
                time=np.datetime64(t) if t else None,
                lat=float(ds.attrs.get("latitude_deg", np.nan)),
                lon=float(ds.attrs.get("longitude_deg", np.nan)),
            ))
    return out


def substitute_alternates(ours: list[OurCast], alt_dir: str | Path,
                          stations: list[str], label: str) -> list[OurCast]:
    """Swap selected stations for their version from an alternate run dir.

    Used when specific casts need a different processing choice (e.g. ``--botfac 0``
    on shallow casts with bad near-bottom BT samples): the report then carries the
    alternate result, explicitly labelled, instead of silently mixing settings.
    Raises if a requested station is missing from the alternate run.
    """
    alt = {o.station: o for o in scan_ours(alt_dir)}
    missing = [s for s in stations if s not in alt]
    if missing:
        raise FileNotFoundError(
            f"alternate run {alt_dir} has no station(s): {', '.join(missing)}")
    out = []
    for o in ours:
        if o.station in stations:
            a = alt[o.station]
            a.variant = label
            out.append(a)
        else:
            out.append(o)
    return out


def pair_by_time(ours: list[OurCast], legacy: list[LegacyCast],
                 tol_h: float = _PAIR_TOL_H):
    """Greedy nearest-time 1:1 pairing. Returns (pairs, ours_only, legacy_only)."""
    cands = []
    for i, oc in enumerate(ours):
        if oc.time is None:
            continue
        for j, lc in enumerate(legacy):
            if lc.time is None:
                continue
            dt = abs((oc.time - lc.time) / np.timedelta64(1, "h"))
            if dt <= tol_h:
                cands.append((dt, i, j))
    used_o: set[int] = set()
    used_l: set[int] = set()
    pairs = []
    for dt, i, j in sorted(cands):
        if i in used_o or j in used_l:
            continue
        used_o.add(i)
        used_l.add(j)
        pairs.append((ours[i], legacy[j], dt))
    ours_only = [o for k, o in enumerate(ours) if k not in used_o]
    legacy_only = [m for k, m in enumerate(legacy) if k not in used_l]
    pairs.sort(key=lambda t: str(t[1].name))
    return pairs, ours_only, legacy_only


@dataclass
class PairResult:
    station: str
    legacy_name: str
    dt_min: float               # pairing time difference [min]
    u: Score
    v: Score
    ubot: Score
    vbot: Score
    dubar: float
    dvbar: float
    zmax_ours: float
    zmax_legacy: float
    profile: dict               # arrays for the overlay figure
    variant: str = ""           # alternate-run label ("" = the main run)


def _own_sadcp_profile(sadcp_ds, lc: LegacyCast, oc: OurCast):
    """Ship-ADCP profile from our (clock-corrected) dataset over this cast's window.

    The window comes from the legacy super-ensemble time span (``dr.tim``, exact)
    when available, else +-30 min around our cast start. Returns an ``[nz, 4]``
    ``(z, u, v, verr)`` array or ``None``.
    """
    from ..io.sadcp_vmdas import extract_profile

    if lc.tim_span is not None:
        t0, t1 = lc.tim_span
    elif oc.time is not None:
        t0, t1 = oc.time - np.timedelta64(30, "m"), oc.time + np.timedelta64(30, "m")
    else:
        return None
    return extract_profile(sadcp_ds, time_start=t0, time_end=t1,
                           lat=oc.lat, lon=oc.lon)


def compare_pair(oc: OurCast, lc: LegacyCast, dt_h: float,
                 sadcp_ds=None) -> PairResult:
    """Score one station: ours interpolated onto the legacy depth grid."""
    import xarray as xr

    with xr.open_dataset(oc.path) as ds:
        z = ds["depth"].values
        u = ds["u"].values
        v = ds["v"].values
        uerr = ds["uerr"].values if "uerr" in ds else np.full(z.shape, np.nan)
        zb = ds["depth_bt"].values if "depth_bt" in ds else np.array([])
        ub = ds["u_bt"].values if "u_bt" in ds else np.array([])
        vb = ds["v_bt"].values if "v_bt" in ds else np.array([])
        ubar = float(ds.attrs.get("ubar_ms", np.nan))
        vbar = float(ds.attrs.get("vbar_ms", np.nan))

    dr = lc.dr
    ui = np.interp(dr.z, z, u, left=np.nan, right=np.nan)
    vi = np.interp(dr.z, z, v, left=np.nan, right=np.nan)
    if zb.size > 1 and dr.zbot.size > 1:
        ubi = np.interp(dr.zbot, zb, ub, left=np.nan, right=np.nan)
        vbi = np.interp(dr.zbot, zb, vb, left=np.nan, right=np.nan)
    else:
        ubi = np.full(dr.zbot.shape, np.nan)
        vbi = np.full(dr.zbot.shape, np.nan)

    own_sadcp = _own_sadcp_profile(sadcp_ds, lc, oc) if sadcp_ds is not None else None

    lbar_u = dr.ubar if np.isfinite(dr.ubar) else np.nan
    lbar_v = dr.vbar if np.isfinite(dr.vbar) else np.nan
    return PairResult(
        station=oc.station, legacy_name=lc.name, dt_min=dt_h * 60.0,
        u=score(ui, dr.u), v=score(vi, dr.v),
        ubot=score(ubi, dr.ubot), vbot=score(vbi, dr.vbot),
        dubar=float(ubar - lbar_u), dvbar=float(vbar - lbar_v),
        zmax_ours=float(np.nanmax(z)) if z.size else np.nan,
        zmax_legacy=float(np.nanmax(dr.z)) if dr.z.size else np.nan,
        profile={"z": z, "u": u, "v": v, "uerr": uerr,
                 "lz": dr.z, "lu": dr.u, "lv": dr.v,
                 "luerr": dr.uerr if dr.uerr.size == dr.z.size
                 else np.full(dr.z.shape, np.nan),
                 # ship-ADCP: the constraint profile legacy actually used (dr) and,
                 # when a dataset is supplied, our own extraction over the cast window
                 "lsz": dr.z_sadcp if dr.has_sadcp else np.array([]),
                 "lsu": dr.u_sadcp if dr.has_sadcp else np.array([]),
                 "lsv": dr.v_sadcp if dr.has_sadcp else np.array([]),
                 "sadcp": own_sadcp},
        variant=oc.variant,
    )


def write_report(results: list[PairResult], ours_only, legacy_only,
                 out_dir: str | Path, title: str) -> tuple[Path, Path]:
    """Write ``comparison.csv`` + ``comparison_report.pdf``; returns their paths."""
    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "station": r.station, "legacy": r.legacy_name,
            "variant": r.variant or "main",
            "pair_dt_min": round(r.dt_min, 1),
            "u_corr": r.u.corr, "u_rms_ms": r.u.rms, "u_bias_ms": r.u.bias,
            "v_corr": r.v.corr, "v_rms_ms": r.v.rms, "v_bias_ms": r.v.bias,
            "n_bins": r.u.n,
            "ubot_rms_ms": r.ubot.rms, "vbot_rms_ms": r.vbot.rms,
            "dubar_ms": r.dubar, "dvbar_ms": r.dvbar,
            "zmax_ours_m": r.zmax_ours, "zmax_legacy_m": r.zmax_legacy,
        })
    csv_path = out / "comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")

    pdf_path = out / "comparison_report.pdf"
    with PdfPages(pdf_path) as pdf:
        from ..plots.compare_figures import profile_pages, summary_page
        summary_page(pdf, results, title)
        profile_pages(pdf, results)

    if ours_only or legacy_only:
        lines = ["unpaired stations:"]
        lines += [f"  ours-only:   {o.station} ({o.time})" for o in ours_only]
        lines += [f"  legacy-only: {m.name} ({m.time})" for m in legacy_only]
        (out / "unpaired.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, pdf_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-compare",
        description="pyladcp vs legacy-LDEO cruise comparison report")
    ap.add_argument("--ours", required=True,
                    help="ladcp-qa output dir (holds stations/*/<st>.nc)")
    ap.add_argument("--legacy", required=True,
                    help="legacy processed dir (LDEO *.mat with dr structs)")
    ap.add_argument("--tol-hours", type=float, default=_PAIR_TOL_H,
                    help="max cast-time difference for a station pair "
                         f"(default {_PAIR_TOL_H} h)")
    ap.add_argument("--alt-dir", metavar="DIR", default=None,
                    help="alternate ladcp-qa run dir to substitute --alt-stations from "
                         "(e.g. a --botfac 0 rerun of shallow casts)")
    ap.add_argument("--alt-stations", metavar="LIST", default=None,
                    help="comma-separated stations to take from --alt-dir")
    ap.add_argument("--alt-label", default="alternate",
                    help="label stamped on substituted stations in the report "
                         "(default: 'alternate')")
    ap.add_argument("--sadcp", metavar="PATH", default=None,
                    help="ship-ADCP VmDAS folder (STA/LTA; cached): overlay the "
                         "ship-ADCP profile over each cast window on the report")
    ap.add_argument("--sadcp-timeoff", type=float, default=None, metavar="SECONDS",
                    help="clock correction added to the ship-ADCP timestamps "
                         "(the value ladcp-qa --sadcp-timeoff auto reported)")
    ap.add_argument("--sadcp-filetype", choices=("STA", "LTA"), default="STA")
    ap.add_argument("--sadcp-xducer", type=float, default=5.0,
                    help="ship-ADCP transducer depth below waterline [m] (default 5)")
    ap.add_argument("--title", default="pyladcp vs legacy LDEO_IX")
    ap.add_argument("-o", "--out", default=None,
                    help="report dir (default <ours>/legacy_compare)")
    args = ap.parse_args(argv)
    if bool(args.alt_dir) != bool(args.alt_stations):
        ap.error("--alt-dir and --alt-stations go together")

    ours = scan_ours(args.ours)
    legacy = scan_legacy(args.legacy)
    if args.alt_dir:
        stations = [s.strip() for s in args.alt_stations.split(",") if s.strip()]
        ours = substitute_alternates(ours, args.alt_dir, stations, args.alt_label)
    if not ours:
        ap.error(f"no station NetCDFs under {args.ours}/stations/")
    if not legacy:
        ap.error(f"no legacy dr .mat files under {args.legacy}")

    sadcp_ds = None
    if args.sadcp:
        from ..io.sadcp_vmdas import load_or_ingest
        sadcp_ds = load_or_ingest(args.sadcp, file_type=args.sadcp_filetype,
                                  transducer_depth=args.sadcp_xducer)
        if args.sadcp_timeoff:
            from ..io.nav import shift_time
            sadcp_ds = shift_time(sadcp_ds, args.sadcp_timeoff)

    pairs, ours_only, legacy_only = pair_by_time(ours, legacy,
                                                 tol_h=args.tol_hours)
    results = [compare_pair(oc, lc, dt, sadcp_ds=sadcp_ds) for oc, lc, dt in pairs]
    out_dir = args.out or str(Path(args.ours) / "legacy_compare")
    csv_path, pdf_path = write_report(results, ours_only, legacy_only,
                                      out_dir, args.title)

    print(f"paired {len(results)}/{len(ours)} stations "
          f"({len(ours_only)} ours-only, {len(legacy_only)} legacy-only)")
    for r in results:
        print(f"  {r.station:8} vs {r.legacy_name:12} "
              f"u: {r.u}   v: corr={r.v.corr:.3f} rms={r.v.rms:.4f}")
    print(f"wrote {csv_path}\nwrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    # removes the 12 h: verified on FDCCC1_001 (cast 01:36 UTC, tim decodes 13:36
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


def _profile_pages(pdf, results: list[PairResult], per_page: int = 10):
    import matplotlib.pyplot as plt

    for start in range(0, len(results), per_page):
        chunk = results[start:start + per_page]
        ncol = 5
        nrow = int(np.ceil(len(chunk) / ncol)) * 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for k, r in enumerate(chunk):
            col = k % ncol
            row0 = (k // ncol) * 2
            for off, comp in ((0, "u"), (1, "v")):
                ax = axes[row0 + off, col]
                p = r.profile
                # +-1 sigma solution-uncertainty bands (legacy dr.uerr / our nc uerr;
                # LDEO carries one error profile for both components)
                lfin = np.isfinite(p["lu"]) & np.isfinite(p["luerr"])
                if lfin.any():
                    ax.fill_betweenx(p["lz"][lfin],
                                     p[f"l{comp}"][lfin] - p["luerr"][lfin],
                                     p[f"l{comp}"][lfin] + p["luerr"][lfin],
                                     color="0.3", alpha=0.18, lw=0)
                ofin = np.isfinite(p[comp]) & np.isfinite(p["uerr"])
                if ofin.any():
                    ax.fill_betweenx(p["z"][ofin],
                                     p[comp][ofin] - p["uerr"][ofin],
                                     p[comp][ofin] + p["uerr"][ofin],
                                     color="tab:red", alpha=0.15, lw=0)
                ax.plot(p[f"l{comp}"], p["lz"], "-", color="0.3", lw=1.4,
                        label="legacy ±1σ")
                ax.plot(p[comp], p["z"], "-", color="tab:red", lw=1.0,
                        label="pyladcp ±1σ")
                ci = 1 if comp == "u" else 2
                if p["sadcp"] is not None and p["sadcp"].shape[0]:
                    s = p["sadcp"]
                    ax.plot(s[:, ci], s[:, 0], "o-", color="tab:green", ms=2.5,
                            lw=0.8, alpha=0.85, label="ship ADCP")
                if p["lsz"].size > 1:
                    ax.plot(p["lsu"] if comp == "u" else p["lsv"], p["lsz"], "s",
                            color="tab:olive", ms=2.5, alpha=0.8, mfc="none",
                            label="ship ADCP (legacy constraint)")
                ax.invert_yaxis()
                ax.grid(alpha=0.3)
                s = getattr(r, comp)
                tag = f" [{r.variant}]" if r.variant else ""
                ax.set_title(f"{r.station}{tag} {comp}  r={s.corr:.2f} "
                             f"rms={100 * s.rms:.1f}cm/s", fontsize=8,
                             color="tab:blue" if r.variant else "black")
                if col == 0:
                    ax.set_ylabel("depth [m]")
        for k in range(len(chunk), ncol * (nrow // 2)):
            axes[(k // ncol) * 2, k % ncol].axis("off")
            axes[(k // ncol) * 2 + 1, k % ncol].axis("off")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color="0.3", lw=1.4, label="legacy ±1σ"),
                   Line2D([], [], color="tab:red", lw=1.0, label="pyladcp ±1σ")]
        if any(r.profile["sadcp"] is not None and r.profile["sadcp"].shape[0]
               for r in chunk):
            handles.append(Line2D([], [], color="tab:green", marker="o", ms=3,
                                  lw=0.8, label="ship ADCP"))
        if any(r.profile["lsz"].size > 1 for r in chunk):
            handles.append(Line2D([], [], color="tab:olive", marker="s", ms=3,
                                  lw=0, mfc="none",
                                  label="ship ADCP (legacy constraint)"))
        fig.legend(handles=handles, loc="upper left", fontsize=7, ncol=len(handles),
                   frameon=False)
        pdf.savefig(fig)
        plt.close(fig)


def _summary_page(pdf, results: list[PairResult], title: str):
    import matplotlib.pyplot as plt

    names = [r.station + ("*" if r.variant else "") for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - 0.2, [100 * r.u.rms for r in results], 0.4, label="u")
    ax.bar(x + 0.2, [100 * r.v.rms for r in results], 0.4, label="v")
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_ylabel("rms vs legacy [cm/s]")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(x - 0.2, [r.u.corr for r in results], 0.4, label="u")
    ax.bar(x + 0.2, [r.v.corr for r in results], 0.4, label="v")
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_ylabel("profile correlation")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot([100 * r.dubar for r in results], [100 * r.dvbar for r in results],
            "o", ms=5)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("ubar (pyladcp - legacy) [cm/s]")
    ax.set_ylabel("vbar (pyladcp - legacy) [cm/s]")
    ax.set_title("barotropic offset per station", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot([r.zmax_legacy for r in results], [r.zmax_ours for r in results],
            "o", ms=5)
    lim = max([*[r.zmax_legacy for r in results],
               *[r.zmax_ours for r in results], 1.0])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("legacy max depth [m]")
    ax.set_ylabel("pyladcp max depth [m]")
    ax.set_title("depth coverage", fontsize=9)
    ax.grid(alpha=0.3)

    med_u = float(np.nanmedian([r.u.rms for r in results]))
    med_v = float(np.nanmedian([r.v.rms for r in results]))
    med_cu = float(np.nanmedian([r.u.corr for r in results]))
    fig.suptitle(f"{title}\n{len(results)} stations -- median rms "
                 f"u {100 * med_u:.1f} / v {100 * med_v:.1f} cm/s, "
                 f"median corr(u) {med_cu:.2f}", fontsize=12)
    subs = {r.station: r.variant for r in results if r.variant}
    if subs:
        note = "* substituted runs: " + "; ".join(
            f"{st} = {lab}" for st, lab in subs.items())
        fig.text(0.01, 0.005, note, fontsize=8, color="tab:blue")
    pdf.savefig(fig)
    plt.close(fig)


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
        _summary_page(pdf, results, title)
        _profile_pages(pdf, results)

    if ours_only or legacy_only:
        lines = ["unpaired stations:"]
        lines += [f"  ours-only:   {o.station} ({o.time})" for o in ours_only]
        lines += [f"  legacy-only: {m.name} ({m.time})" for m in legacy_only]
        (out / "unpaired.txt").write_text("\n".join(lines) + "\n")
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

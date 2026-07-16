"""The per-station processing pipeline: configuration in, reports + products out.

New to the code? This module is the pipeline's table of contents:
:func:`process_station` wires every stage together in order (ingest → screen →
depth/time-sync → bottom track → super-ensembles → solver → report/export). Read it
alongside ``src/ladcp/README.md``, which gives the recommended reading order and a
module-by-module map. Follow :func:`process_station`'s calls to jump into whichever
stage you care about.

Both drivers sit on top of this module: ``ladcp-qa`` (:mod:`ladcp.qa.cli`, which owns
argument parsing, work-list resolution, logging and the worker pool) and the Studio
server (via :class:`ladcp.session.StationSession`, which shares
:func:`sadcp_profile` and the same solve calls with a cache in between).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import DEFAULT_CRUISE, resolve_params
from ..io.ctd_cnv import read_ctd_cnv
from ..session import SessionConfig, edit_overrides, resolve_declination
from .ingest import apply_header_config, load_dualhead
from .report import assess, text_report

log = logging.getLogger("ladcp.qa")


def process_station(down, up, ctd_path, station, outdir, make_plots, cfg: SessionConfig,
                    cruise=DEFAULT_CRUISE, formats=None, ctd_utc=None,
                    edits=None, hint_root=None, param_overrides=None):
    """Process one station into ``<outdir>/stations/<station>/``.

    ``cfg`` carries the full solve configuration (edit knobs, SADCP identity, solve
    weights); by this point any ``timeoff='auto'`` has been resolved to seconds
    (:func:`warm_sadcp`). Returns ``(status, export)``: ``status`` is the QA verdict
    string (``"ok"``/``"warn"``/``"fail"``) and ``export`` is a
    :class:`~ladcp.export.StationExport` when a velocity solution was produced
    (``None`` for acquisition-only stations).

    ``edits`` is the ``--edits`` value (journal file or ``.ladcp_edits`` dir) and is the
    single application point for manual edits -- resolved, staleness-verified and turned
    into params here. ``hint_root`` (discovery mode only) lets the no-``--edits`` path
    WARN about an existing unapplied journal.

    ``param_overrides`` is the station's merged ``cruise.toml`` ``[params]`` layer
    (:func:`ladcp.hub.cruise_config.station_params`): applied to the resolved
    :class:`~ladcp.config.CastParams` *under* the edit knobs, so explicit flags and
    journals keep winning over the config file.
    """
    from ..edits import journal_path, load_journal, manual_flags, resolve_edits_arg, verify_journal
    journal = jpath = None
    if edits:
        jpath = resolve_edits_arg(edits, station)
        if jpath is not None:
            journal = load_journal(jpath)
            if journal.station != station:
                raise ValueError(f"--edits: {jpath} is the journal for station "
                                 f"{journal.station!r}, not {station!r}")
            verify_journal(journal, jpath, down, up)

    jflags = manual_flags(journal) if journal is not None and journal.entries else None
    overrides = dict(param_overrides or {})
    overrides.update(edit_overrides(cfg.edit, manual_flags=jflags))
    params = resolve_params(cruise, station, overrides=overrides or None)
    dh = load_dualhead(down, up, station=station, params=params)
    apply_header_config(params, dh)             # geometry/head-count from the PD0 headers
    if journal is not None:                     # second staleness tripwire, post-ingest
        for name, head in (("down", dh.down), ("up", dh.up)):
            fp = journal.raw.get(name) or {}
            if head is not None and fp.get("n_ens") is not None \
                    and fp["n_ens"] != head.n_ens:
                raise ValueError(f"--edits: {jpath}: the {name} file now holds "
                                 f"{head.n_ens} ensembles (journal recorded "
                                 f"{fp['n_ens']}); delete or re-create the journal")
    ctd = read_ctd_cnv(ctd_path, params=params) if ctd_path else None
    if ctd is not None and ctd_utc and "utc_start" not in ctd.meta:
        ctd.meta["utc_start"] = ctd_utc         # index cast-start UTC -> sync prior
    qc = assess(dh, ctd=ctd)

    from ..models import Metric, Status
    if journal is not None and journal.entries:
        n_dn = sum(1 for e in journal.entries if e["head"] == "down")
        qc.add(Metric("manual_edits", journal.n_entries, "", Status.OK,
                      source_stage="qa.edits",
                      note=f"{journal.n_entries} manual rectangle(s) replayed from "
                           f"{jpath} (down: {n_dn}, up: {journal.n_entries - n_dn})"))
    elif not edits and hint_root:
        cand = journal_path(hint_root, station)
        if cand.is_file():
            try:
                unapplied = load_journal(cand)
            except ValueError:
                unapplied = None
            if unapplied is not None and unapplied.entries:
                qc.add(Metric("manual_edits_unapplied", unapplied.n_entries, "",
                              Status.WARN, source_stage="qa.edits",
                              note=f"a manual edit journal with {unapplied.n_entries} "
                                   f"rectangle(s) exists but was NOT applied; re-run "
                                   f"with --edits {cand}"))

    st_dir = Path(outdir) / "stations" / station
    fig_dir = st_dir / "figures"
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n", encoding="utf-8")
    (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2),
                                               encoding="utf-8")
    log.info("[%-5s] %s  ->  %s/", qc.overall_status.value.upper(), station, st_dir)

    # velocity solve (requires CTD + earth-frame data): .lad + .bot text, figures
    from ..models import CoordFrame
    earth = all(h.coord_frame == CoordFrame.EARTH for h in (dh.down, dh.up) if h is not None)
    result = None
    export = None
    down_only = cfg.edit.down_only
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
        result, meta = _velocity_outputs(dh_solve, ctd, station, st_dir, cfg)
        from ..qa.checks import consistency_checks
        for m in consistency_checks(result):       # checkinv -> scorecard
            qc.add(m)
        qc.add(_declination_metric(meta["drot"], meta["drot_source"]))
        # refresh the persisted QA text/json now that consistency checks are in
        (st_dir / f"{station}_qa.txt").write_text(text_report(qc) + "\n", encoding="utf-8")
        (st_dir / f"{station}_qa.json").write_text(json.dumps(qc.to_dict(), indent=2),
                                                   encoding="utf-8")
        from ..export import StationExport
        export = StationExport(station=station, cruise=cruise, lat=meta["lat"],
                               lon=meta["lon"], time=meta["when"], drot=meta["drot"],
                               solver=cfg.solve.solver, result=result, qc=qc,
                               sadcp_source=meta["sadcp_source"])

    if make_plots:
        from ..plots.pdf_report import build_report
        paths = build_report(dh, qc, str(st_dir), station, ctd=ctd, velocity=result,
                             figdir=str(fig_dir))
        log.info("        report: %s", paths["report.pdf"])

    if export is not None and formats:
        write_station_exports(export, st_dir, formats)

    return qc.overall_status.value, export


def _velocity_outputs(dh, ctd, station, out, cfg: SessionConfig):
    import numpy as np

    from ..plots.sadcp_figure import sadcp_rms_discrepancy
    from ..qa.export import write_bot, write_lad
    from ..qa.inverse import build_solve_context, compute_velocity_full

    solver = cfg.solve.solver
    lat = float(np.nanmedian(ctd.lat))
    lon = float(np.nanmedian(ctd.lon))
    when = dh.down.time[0].astype("datetime64[s]").item()
    drot = cfg.solve.drot
    if drot is not None:
        drot_source = "explicit"                # user-supplied --drot
    else:                                       # IGRF-13 from cast position + date
        drot, drot_source = resolve_declination(lat, lon, when, logger=log)

    t_lad = dh.down.time
    sadcp = (sadcp_profile(cfg.sadcp, t_lad.min(), t_lad.max(), lat, lon, solver)
             if cfg.sadcp is not None else None)
    sadcpfac = cfg.solve.sadcpfac
    # Build the expensive front end once so the SADCP-withheld validation solve below reuses
    # it (~30 ms) instead of rebuilding (~1.2 s); the main solve stays bit-identical.
    context = build_solve_context(dh, ctd, dz=8.0, params=dh.params)
    result = compute_velocity_full(dh, ctd, drot=drot, params=dh.params, solver=solver,
                                   sadcp=sadcp, sadcpfac=sadcpfac,
                                   botfac=cfg.solve.botfac,
                                   barofac=cfg.solve.barofac,
                                   smoofac=cfg.solve.smoofac,
                                   context=context)
    # Independent empirical uncertainty: re-solve with the ship-ADCP withheld (sadcpfac=0) so
    # the LADCP-vs-SADCP comparison is not circular, then RMS over the shared depth range. Only
    # meaningful when the SADCP was actually pulling the main solve (inverse, sadcp, fac>0).
    if solver == "inverse" and sadcp is not None and sadcpfac > 0:
        withheld = compute_velocity_full(dh, ctd, drot=drot, params=dh.params, solver=solver,
                                         sadcp=sadcp, sadcpfac=0.0,
                                         botfac=cfg.solve.botfac,
                                         barofac=cfg.solve.barofac,
                                         smoofac=cfg.solve.smoofac,
                                         context=context)
        result.sadcp_independent_rms = sadcp_rms_discrepancy(withheld)
    vp, bp = result.vp, result.bp
    lad = out / f"{station}.lad"
    write_lad(vp, str(lad), station=station, lat=lat, lon=lon, drot=drot, time=when)
    log.info("        velocity: %s  (solver %s, drot %+.2f deg, ubar %+.3f)",
             lad, solver, drot, vp.ubar)
    if result.sadcp_independent_rms is not None and np.isfinite(result.sadcp_independent_rms):
        log.info("        empirical uncertainty: %.3f m/s  (RMS LADCP-SADCP, SADCP withheld)",
                 result.sadcp_independent_rms)

    if bp is not None and bp.n_bins > 0:
        bot = out / f"{station}.bot"
        write_bot(bp, str(bot), station=station, lat=lat, lon=lon, drot=drot,
                  zbottom=result.zbottom, time=when)
        log.info("        bottom-track: %s  (%d bins)", bot, bp.n_bins)

    if cfg.sadcp is not None and sadcp is not None:
        sadcp_src = cfg.sadcp.folder
        if cfg.sadcp.source == "codas":
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


def write_station_exports(export, st_dir, formats) -> None:
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


def write_cruise_exports(exports, outdir, cruise, formats) -> None:
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


def _load_sadcp_dataset(sadcp):
    """Load the ship-ADCP dataset for a :class:`~ladcp.session.SadcpConfig` (per source)."""
    if sadcp.source == "codas":
        from ..io.sadcp_codas import read_codas_nc
        return read_codas_nc(sadcp.folder)
    if sadcp.source == "ek80":
        from ..io.sadcp_ek80 import read_ek80
        return read_ek80(sadcp.folder, transducer_depth=sadcp.xducer)
    from ..io.sadcp_vmdas import load_or_ingest
    return load_or_ingest(sadcp.folder, force=sadcp.reingest,
                          file_type=sadcp.filetype, transducer_depth=sadcp.xducer)


def sadcp_profile(sadcp, time_start, time_end, lat, lon, solver):
    """Build the cast's ship-ADCP constraint profile for a :class:`SadcpConfig`.

    Loads the dataset once (raw VmDAS folder ingested+cached, or a CODAS/EK80 NetCDF
    read directly per ``--sadcp-source``), then windows it to this cast's LADCP time
    span and position. Only the ``inverse`` solver consumes the constraint; with
    ``shear`` the folder is ignored with a notice. Returns the ``svel`` array or ``None``.

    ``timeoff='auto'`` is resolved here when still present (library callers); the CLI
    batch path resolves it once up front instead (:func:`warm_sadcp`).
    """
    if solver != "inverse":
        log.info("        sadcp: ignored (constraint applies only to --solver inverse)")
        return None
    from ..io.sadcp_vmdas import extract_profile

    ds = _load_sadcp_dataset(sadcp)
    toff = sadcp.timeoff
    if toff == "auto":
        from ..io.nav import estimate_time_offset, read_nav
        nav = read_nav(sadcp.nav)
        est = estimate_time_offset(ds.time, ds.lat, ds.lon, nav)
        toff = est["offset_s"]
        log.info("        sadcp: clock offset %+.2f s estimated from nav track "
                 "(track residual %.0f m median / %.0f m p90, overlap %.0f%%)",
                 toff, est["median_m"], est["p90_m"], 100 * est["overlap"])
    if toff:
        from ..io.nav import shift_time
        ds = shift_time(ds, float(toff))
    sv = extract_profile(ds, time_start=time_start, time_end=time_end, lat=lat, lon=lon)
    if sv is None:
        log.info("        sadcp: no usable %s kHz data at this station "
                 "(time/position window empty) -- constraint skipped", ds.freq_khz)
    else:
        log.info("        sadcp: %d bins from %s kHz %s (%s)",
                 sv.shape[0], ds.freq_khz, ds.file_type, ds.source)
    return sv


def warm_sadcp(cfg: SessionConfig) -> SessionConfig:
    """One-time SADCP setup for a batch: build the ingest cache and resolve
    ``timeoff='auto'`` to seconds, returning the updated (still frozen) config.

    Without this every station (or worker process) would race to parse the raw VmDAS
    tree and re-estimate the clock offset against the nav track. ``reingest`` is
    cleared afterwards so the stations reuse the cache just built.
    """
    from dataclasses import replace

    sadcp = cfg.sadcp
    ds = None
    if sadcp.source not in ("codas", "ek80"):       # NetCDF reads are cheap per station
        ds = _load_sadcp_dataset(sadcp)
        sadcp = replace(sadcp, reingest=False)
    if sadcp.timeoff == "auto":
        if ds is None:
            ds = _load_sadcp_dataset(sadcp)
        from ..io.nav import estimate_time_offset, read_nav
        nav = read_nav(sadcp.nav)
        est = estimate_time_offset(ds.time, ds.lat, ds.lon, nav)
        sadcp = replace(sadcp, timeoff=est["offset_s"])
        log.info("sadcp: clock offset %+.2f s estimated from nav track (once for the "
                 "batch; track residual %.0f m median)", est["offset_s"], est["median_m"])
    return replace(cfg, sadcp=sadcp)

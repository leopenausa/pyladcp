"""The shared batch runner: a station plan in, ``process_station`` calls out.

Extracted verbatim from ``ladcp.qa.cli`` so the ``ladcp`` hub (``ladcp process``) and
``ladcp-qa`` drive the exact same loop — serial or one worker process per station
(``jobs``), per-cast exception isolation, interleaving-safe run logs, and the
cruise-level export aggregate. There is deliberately no second orchestration path
(wizard spec §6): anything that processes stations calls :func:`run_batch`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..config import DEFAULT_CRUISE
from ..discovery import discover
from ..hub.cruise_config import merge_params
from .pipeline import process_station, warm_sadcp, write_cruise_exports
from .runlog import ProgressBar

log = logging.getLogger("ladcp.qa")


# ---------------------------------------------------------------------------
# jobs > 1: parallel batch over stations (one worker process per station).
# Stations are fully independent; ~80% of a station's wall time is figure
# rendering, so process-level parallelism scales near-linearly. The worker
# functions are top-level so they pickle under the spawn start method
# (Windows/macOS).

def _pool_init() -> None:
    """Worker initializer: force the headless matplotlib backend."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg", force=True)


def _pool_task(task: dict) -> tuple[int, str, str, object, str]:
    """Process one station in a worker: returns (index, label, status, export, log_text).

    Per-station log records are captured into a buffer (file-log format) and written
    sequentially by the parent, so ``ladcp-qa.log`` never interleaves stations.
    """
    import io as _io

    buf = _io.StringIO()
    lg = logging.getLogger("ladcp.qa")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    bh = logging.StreamHandler(buf)
    bh.setLevel(logging.DEBUG)
    bh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    lg.addHandler(bh)

    label, status, export = task["item"], "error", None
    try:
        sf = discover(task["item"], root=Path(task["root"]), cruise=task["cruise"],
                      index=task["index"], from_hex=task["from_hex"],
                      ctd_cache=task["ctd_cache"])
        label = sf.label
        pov = merge_params(task["params_global"], task["params_station"], label)
        status, export = process_station(str(sf.down), str(sf.up) if sf.up else None,
                                         str(sf.ctd) if sf.ctd else None, label,
                                         task["outdir"], task["make_plots"], task["cfg"],
                                         cruise=task["cruise"], formats=task["formats"],
                                         ctd_utc=sf.ctd_utc, edits=task.get("edits"),
                                         hint_root=task["root"],
                                         param_overrides=pov or None)
    except (Exception, SystemExit) as e:           # one bad cast must not abort the batch
        lg.error("[ERROR] %s: %s: %s", label, type(e).__name__, e, exc_info=True)
    bh.flush()
    return task["_i"], label, status, export, buf.getvalue()


def _append_worker_log(text: str) -> None:
    """Write a worker's captured log block into the parent's run-log file verbatim."""
    if not text:
        return
    for h in logging.getLogger("ladcp.qa").handlers:
        if isinstance(h, logging.FileHandler):
            h.stream.write(text)
            h.flush()


def run_batch(plan: list[str], cfg, *, root, cruise: str = DEFAULT_CRUISE, index=None,
              from_hex: bool = False, ctd_cache=None, outdir: str = "qa_out",
              make_plots: bool = True, formats=frozenset(), edits=None, jobs: int = 1,
              explicit_files: tuple | None = None, params_global: dict | None = None,
              params_station: dict | None = None, cruise_export: bool = False,
              progress_enabled: bool = False) -> list[tuple[str, str]]:
    """Run :func:`~ladcp.qa.pipeline.process_station` over ``plan`` -> ``(label, status)`` list.

    ``plan`` holds station ids resolved through discovery, unless ``explicit_files``
    names one cast's files directly (``(down, up, ctd)``, the CLI's ``--down`` mode —
    then ``plan`` is its single label). ``params_global``/``params_station`` are the
    cruise.toml ``[params]`` layers; ``cruise_export`` builds the cruise-level
    ``exports/`` aggregate from the successful stations. Logging must already be set
    up (:func:`~ladcp.qa.runlog.setup_logging`); a failing cast is logged and
    recorded as ``"error"``, never aborting the batch.
    """
    root = Path(root)
    n = len(plan)
    jobs = jobs if jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, n))
    params_global = params_global or {}
    params_station = params_station or {}
    bar = ProgressBar(n, enabled=progress_enabled)
    results: list[tuple[str, str]] = []                # (label, status)
    exports = []

    if cfg.sadcp is not None and cfg.solve.solver == "inverse":
        cfg = warm_sadcp(cfg)              # build the cache / resolve 'auto' ONCE per batch
    if jobs > 1 and explicit_files is None:
        # one station per worker process: stations are fully independent, and the
        # bulk of a station's wall time is matplotlib rendering (CPU-bound)
        log.info("parallel: %d worker processes", jobs)
        base = dict(root=str(root), cruise=cruise, index=index,
                    from_hex=from_hex, ctd_cache=ctd_cache,
                    outdir=outdir, make_plots=make_plots,
                    cfg=cfg, formats=formats, edits=edits,
                    params_global=params_global, params_station=params_station)
        # each worker must NOT spin up a full BLAS thread pool: 6 workers x 16
        # OpenBLAS threads thrash the cores (measured 3x slower on the 40-cast
        # MORIA soak: 465s vs 148s). The pool itself is the parallelism, so
        # workers run single-threaded BLAS; an explicit user env setting wins.
        # The spawn context (all platforms) makes the limit apply at numpy load.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = multiprocessing.get_context("spawn")
        slots: list[tuple[str, str, object] | None] = [None] * n
        with ProcessPoolExecutor(max_workers=jobs, initializer=_pool_init,
                                 mp_context=ctx) as ex:
            futs = {ex.submit(_pool_task, dict(base, item=item, _i=i)): i
                    for i, item in enumerate(plan)}
            for fut in as_completed(futs):
                try:
                    i, label, status, export, text = fut.result()
                except Exception as e:         # un-picklable result / dead worker
                    i = futs[fut]
                    label, status, export = plan[i], "error", None
                    text = f"[ERROR] {label}: {type(e).__name__}: {e}\n"
                slots[i] = (label, status, export)
                _append_worker_log(text)
                bar.advance(f"{label} [{status}]")
        for slot in slots:                     # plan order: deterministic summary/exports
            if slot is None:
                continue
            label, status, export = slot
            results.append((label, status))
            if export is not None:
                exports.append(export)
    else:
        for item in plan:
            label = item
            bar.start(label)
            ctd_utc = None
            try:
                if explicit_files is not None:
                    down, up, ctd_path = explicit_files
                else:
                    sf = discover(item, root=root, cruise=cruise, index=index,
                                  from_hex=from_hex, ctd_cache=ctd_cache)
                    label = sf.label
                    down = str(sf.down)
                    up = str(sf.up) if sf.up else None
                    ctd_path = str(sf.ctd) if sf.ctd else None
                    ctd_utc = sf.ctd_utc
                bar.start(label)
                pov = merge_params(params_global, params_station, label)
                status, export = process_station(down, up, ctd_path, label, outdir,
                                                 make_plots, cfg,
                                                 cruise=cruise, formats=formats,
                                                 ctd_utc=ctd_utc, edits=edits,
                                                 hint_root=(None if explicit_files is not None
                                                            else str(root)),
                                                 param_overrides=pov or None)
                if export is not None:
                    exports.append(export)
            except (Exception, SystemExit) as e:   # one bad cast must not abort the batch
                status = "error"
                bar.clear()
                log.error("[ERROR] %s: %s: %s", label, type(e).__name__, e, exc_info=True)
            results.append((label, status))
            bar.advance(f"{label} [{status}]")
    bar.close()

    # cruise-level aggregate only when explicitly requested (batch / whole index)
    if formats and exports and cruise_export:
        write_cruise_exports(exports, outdir, cruise, formats)
    return results


def log_summary(results: list[tuple[str, str]], logfile, console_detail: bool) -> None:
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

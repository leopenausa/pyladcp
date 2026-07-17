"""``ladcp`` — the cruise hub (docs/WIZARD_SPEC.md): one command per cruise directory.

Current surface (docs/WIZARD_PLAN.md phases B–D): ``ladcp init`` (the setup wizard,
:mod:`~ladcp.hub.init_flow`), ``ladcp status`` (the mid-cruise dashboard,
:mod:`~ladcp.hub.status` — also what a bare ``ladcp`` shows in a cruise directory),
``ladcp config`` (show / validate / edit the ``cruise.toml``) and ``ladcp process``
(run stations through the shared :func:`~ladcp.qa.batch.run_batch` loop, selecting
work by the freshness rule). ``studio`` arrives in phase E.

The hub never grows a second orchestration or configuration path: it fills the same
``ladcp-qa`` argparse namespace from ``cruise.toml``
(:func:`~ladcp.hub.cruise_config.apply_to_args` with nothing marked explicit) and
drives the same batch loop ``ladcp-qa`` uses, so a hub run and the equivalent
``ladcp-qa`` invocation are the same computation by construction.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ..config import CRUISES
from ..discovery import all_station_labels
from ..session import SessionConfig
from . import cruise_config as cc
from .freshness import select_new

log = logging.getLogger("ladcp.hub")


# ---------------------------------------------------------------------------
# shared plumbing

def _resolve_config(config_arg: str | None) -> Path:
    """The cruise.toml this invocation works on (``--config`` or auto-discovered)."""
    if config_arg:
        p = Path(config_arg)
        if not p.is_file():
            raise SystemExit(f"ladcp: --config: {p} does not exist")
        return p
    p = cc.find_config()
    if p is None:
        raise SystemExit("ladcp: no cruise.toml found in this directory or its parents; "
                         "run from the cruise directory, pass --config, or set one up "
                         "with `ladcp init`")
    return p


# ---------------------------------------------------------------------------
# ladcp status

def _cmd_status(ns) -> int:
    path = _resolve_config(ns.config)
    try:
        ccfg = cc.load_config(path)
    except cc.ConfigError as e:
        print(f"ladcp status: {e}", file=sys.stderr)
        return 1
    from .status import gather, render
    data = gather(ccfg)
    if ns.json:
        import json
        print(json.dumps(data, indent=2))
    else:
        for line in render(data):
            print(line)
    return 0


# ---------------------------------------------------------------------------
# ladcp config

def _base_params_source(cruise: str) -> str:
    return (f"preset:{cruise.upper()}" if cruise.upper() in CRUISES
            else "generic defaults")


def _cmd_config_show(ccfg: cc.CruiseConfig) -> int:
    """Every resolved ladcp-qa option with its provenance (cruise.toml | default)."""
    from ..qa.cli import build_parser
    defaults = {a.dest: a.default for a in build_parser()._actions}
    print(f"config: {ccfg.path}")
    print(f"{'option':<20} {'value':<44} source")
    shown: set[str] = set()
    rows = [dest for dest, _ in cc._SCHEMA.values()] + \
           [dest for dest, _ in cc._SADCP_SCHEMA.values()]
    for dest in rows:
        if dest in shown:
            continue
        shown.add(dest)
        if dest in ccfg.args_map:
            value, source = ccfg.args_map[dest], "cruise.toml"
        elif dest in defaults:
            value, source = defaults[dest], "default"
        else:                                   # pragma: no cover - schema/CLI drift guard
            continue
        print(f"{dest:<20} {str(value):<44} {source}")
    cruise = str(ccfg.args_map.get("cruise", defaults.get("cruise", "")))
    print(f"\nbase cast params: {_base_params_source(cruise)} "
          f"(cruise {cruise!r})")
    if ccfg.params_global:
        print("[params] overrides (cruise.toml):")
        for k, v in ccfg.params_global.items():
            print(f"  {k} = {v!r}")
    for station, ov in ccfg.params_station.items():
        print(f"[params.{station}] overrides (cruise.toml):")
        for k, v in ov.items():
            print(f"  {k} = {v!r}")
    if ccfg.n_sadcp > 1:
        print(f"\nnote: {ccfg.n_sadcp} [[sadcp]] sources listed; processing uses the first")
    return 0


def _cmd_config_validate(ccfg: cc.CruiseConfig) -> int:
    """Schema already passed (load_config); check the referenced paths + cross-fields."""
    problems: list[str] = []
    checks = [("data.root", ccfg.args_map.get("root"), "dir"),
              ("data.index", ccfg.args_map.get("index"), "file"),
              ("sadcp.folder", ccfg.args_map.get("sadcp"), "any"),
              ("sadcp.nav", ccfg.args_map.get("sadcp_nav"), "any"),
              ("edit.edits", ccfg.args_map.get("edits"), "any")]
    for name, value, kind in checks:
        if value is None:
            continue
        p = Path(value)
        ok = p.is_dir() if kind == "dir" else p.is_file() if kind == "file" else p.exists()
        if not ok:
            problems.append(f"{name}: {p} does not exist")
    try:                # cross-field validation (e.g. timeoff='auto' needs nav)
        cfg = SessionConfig.from_args(cc.merged_qa_args(ccfg))
        if cfg.sadcp is not None:
            cfg.sadcp.validate_folder()
    except ValueError as e:
        problems.append(str(e))
    if problems:
        print(f"config: {ccfg.path}")
        for p in problems:
            print(f"  PROBLEM: {p}")
        return 1
    print(f"config: {ccfg.path} — OK")
    return 0


def _cmd_config_edit(path: Path) -> int:
    """$EDITOR on a scratch copy; only a copy that validates replaces the real file."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        print("ladcp config edit: set $EDITOR (or edit the file directly; "
              "`ladcp config validate` checks it afterwards)", file=sys.stderr)
        return 1
    tmp = path.with_name(path.name + ".edit")
    tmp.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    rc = subprocess.run([*shlex.split(editor), str(tmp)]).returncode
    if rc != 0:
        tmp.unlink(missing_ok=True)
        print(f"ladcp config edit: editor exited with {rc}; {path.name} unchanged",
              file=sys.stderr)
        return 1
    try:
        cc.load_config(tmp)
    except cc.ConfigError as e:
        print(f"NOT saved — the edited copy does not validate:\n  "
              f"{str(e).replace(str(tmp), str(path))}\n"
              f"your edits are kept at {tmp}; fix and retry, or delete it",
              file=sys.stderr)
        return 1
    os.replace(tmp, path)
    print(f"saved: {path}")
    return 0


def _cmd_config(ns) -> int:
    path = _resolve_config(ns.config)
    if ns.action == "edit":
        return _cmd_config_edit(path)
    try:
        ccfg = cc.load_config(path)
    except cc.ConfigError as e:
        print(f"ladcp config: {e}", file=sys.stderr)
        return 1
    return _cmd_config_show(ccfg) if ns.action == "show" else _cmd_config_validate(ccfg)


# ---------------------------------------------------------------------------
# ladcp process

def _cmd_process(ns) -> int:
    from ..qa.batch import log_summary, run_batch
    from ..qa.runlog import setup_logging, teardown_logging

    path = _resolve_config(ns.config)
    try:
        ccfg = cc.load_config(path)
        args = cc.merged_qa_args(ccfg)
        cfg = SessionConfig.from_args(args)
        if cfg.sadcp is not None:         # fail at launch, not minutes in at a solve
            cfg.sadcp.validate_folder()
    except (cc.ConfigError, ValueError) as e:
        print(f"ladcp process: {e}", file=sys.stderr)
        return 1

    universe = all_station_labels(args.index, Path(args.root))
    if not universe:                  # .cnv-only cruises index 0 casts (.hex anchors);
        from .detect import curated_station_labels  # fall back to name enumeration
        universe = curated_station_labels(args.root)
    if ns.stations:                                   # named stations: unconditional
        plan = list(ns.stations)
        selection = f"{len(plan)} named station(s)"
    else:
        if not universe:
            print(f"ladcp process: no casts in the archive index under {args.root} "
                  "and none enumerable by filename (run `ladcp init`, or build an "
                  "index with ladcp-index)", file=sys.stderr)
            return 1
        if ns.all or ns.force:                        # everything, freshness ignored
            plan = universe
            selection = (f"all {len(plan)} station(s)"
                         + (" (--force: freshness ignored)" if ns.force else ""))
        else:                                         # the default: missing + stale only
            plan, states = select_new(universe, root=args.root, outdir=args.outdir,
                                      cruise=args.cruise, index=args.index,
                                      from_hex=args.from_hex, ctd_cache=args.ctd_cache,
                                      config_path=ccfg.path)
            fresh = len(states) - len(plan)
            missing = sum(1 for s in states if s.state == "missing")
            stale = sum(1 for s in states if s.state == "stale")
            selection = (f"{len(states)} station(s): {fresh} fresh, "
                         f"{stale} stale, {missing} missing")
            if not plan:
                print(f"{selection} — nothing to do (use --all or --force to reprocess)")
                return 0

    n = len(plan)
    console_detail = ns.verbose or n <= 1
    logfile = None if ns.no_log else str(Path(args.outdir) / "ladcp-qa.log")
    setup_logging(logfile, console_level=logging.INFO if console_detail else logging.WARNING)
    qa_log = logging.getLogger("ladcp.qa")
    qa_log.info("config: %s", ccfg.path)
    qa_log.info("ladcp process: %s -> processing %d station(s), out=%s",
                selection, n, args.outdir)
    if not console_detail:
        print(f"{selection} -> processing {n}")
    try:
        results = run_batch(
            plan, cfg, root=args.root, cruise=args.cruise, index=args.index,
            from_hex=args.from_hex, ctd_cache=args.ctd_cache, outdir=args.outdir,
            make_plots=not ns.no_plots, formats={"xlsx", "odv", "nc", "csv"},
            edits=args.edits, jobs=ns.jobs,
            params_global=ccfg.params_global, params_station=ccfg.params_station,
            cruise_export=bool(ns.all or ns.force),   # partial runs must not
            progress_enabled=(n > 1 and not console_detail   # rebuild cruise exports
                              and not ns.no_progress))
        log_summary(results, logfile, console_detail)
    finally:
        teardown_logging()
    return 1 if any(s in ("fail", "error") for _, s in results) else 0


# ---------------------------------------------------------------------------
# entry point

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ladcp",
        description="The LADCP cruise hub: drives the ladcp-* commands from one "
                    "cruise.toml (run inside a cruise directory).")
    sub = ap.add_subparsers(dest="cmd")

    from .init_flow import add_init_parser
    add_init_parser(sub)

    p = sub.add_parser("status", help="the mid-cruise dashboard: pending casts, "
                                      "QA rollup, loose ends (default when run bare)")
    p.add_argument("--config", metavar="PATH", default=None,
                   help="cruise.toml to use (default: auto-discovered upward from cwd)")
    p.add_argument("--json", action="store_true",
                   help="emit the dashboard as JSON (for scripts)")

    p = sub.add_parser("config", help="show / validate / edit the cruise.toml")
    p.add_argument("action", choices=("show", "validate", "edit"),
                   help="show: resolved options with provenance; validate: schema + "
                        "referenced paths; edit: $EDITOR on a copy, saved only if valid")
    p.add_argument("--config", metavar="PATH", default=None,
                   help="cruise.toml to use (default: auto-discovered upward from cwd)")

    p = sub.add_parser("process", help="process stations per the cruise.toml")
    p.add_argument("stations", nargs="*",
                   help="station label(s) to (re)process unconditionally; default: "
                        "every indexed station that is missing or stale (the "
                        "freshness rule)")
    p.add_argument("--new", action="store_true",
                   help="process missing/stale stations only (the default when no "
                        "stations are named; spelled out for scripts)")
    p.add_argument("--all", action="store_true",
                   help="process every indexed station and rebuild the cruise exports")
    p.add_argument("--force", action="store_true",
                   help="ignore freshness: reprocess everything (same as --all)")
    p.add_argument("--config", metavar="PATH", default=None,
                   help="cruise.toml to use (default: auto-discovered upward from cwd)")
    p.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                   help="process N stations in parallel (default: 1; 0 = one per CPU)")
    p.add_argument("--no-plots", action="store_true", help="skip figures/PDF")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="stream per-station detail instead of the progress bar")
    p.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    p.add_argument("--no-log", action="store_true", help="do not write a run-log file")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    ns = ap.parse_args(argv)
    if ns.cmd == "init":
        from .init_flow import run_init
        try:
            return run_init(ns)
        except (KeyboardInterrupt, EOFError):     # Ctrl-C / Ctrl-D: leave nothing half-done
            print("\nladcp init: aborted")        # (every write is atomic + post-confirm)
            return 130
    if ns.cmd == "status":
        return _cmd_status(ns)
    if ns.cmd == "config":
        return _cmd_config(ns)
    if ns.cmd == "process":
        if ns.stations and (ns.all or ns.force):
            ap.error("give station labels OR --all/--force, not both")
        return _cmd_process(ns)
    # bare `ladcp`: the dashboard when a cruise.toml is found, the init hint otherwise
    if cc.find_config() is not None:
        return _cmd_status(argparse.Namespace(config=None, json=False))
    print("no cruise.toml found in this directory or its parents — run "
          "`ladcp init` from the cruise directory to set one up.\n")
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The ``ladcp-studio`` command: argument parsing, launch-time validation, uvicorn."""
from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from ..session import SadcpConfig, parse_timeoff
from .app import create_app
from .state import (
    _QA_CRUISE_DEFAULT,
    _QA_ROOT_DEFAULT,
    StudioState,
    codas_label,
    merge_discovered_codas,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-studio",
        description="pyladcp Studio: interactive single-station LADCP processing "
                    "in the browser (local server)")
    ap.add_argument("stations", nargs="*", help="station id(s) to work on, e.g. 80 79")
    ap.add_argument("--root", default=_QA_ROOT_DEFAULT,
                    help=f"base dir for file discovery (default: {_QA_ROOT_DEFAULT})")
    ap.add_argument("--cruise", default=_QA_CRUISE_DEFAULT,
                    help=f"cruise preset (default: {_QA_CRUISE_DEFAULT})")
    ap.add_argument("--index", default=None,
                    help="archive index JSON (ladcp-index build); with no station "
                         "ids, serves every cast in the index")
    ap.add_argument("--from-hex", action="store_true",
                    help="build missing cleaned CTD from the index's raw .hex anchor")
    ap.add_argument("--ctd-cache", default=None, help="cache dir for --from-hex .cnv")
    ap.add_argument("--sadcp", metavar="PATH", action="append", default=[],
                    help="ship-ADCP source for the inverse constraint (as in ladcp-qa); "
                         "repeatable — each folder becomes an entry in the GUI's "
                         "source dropdown (e.g. the 75 and 150 kHz instruments)")
    ap.add_argument("--sadcp-codas", metavar="PATH", action="append", default=[],
                    help="CODAS-processed product (contour NetCDF or its processing "
                         "dir) offered in the GUI's SADCP source dropdown alongside "
                         "the raw --sadcp; repeatable, one entry per product")
    ap.add_argument("--sadcp-source", choices=("vmdas", "codas"), default="vmdas")
    ap.add_argument("--sadcp-filetype", choices=("STA", "LTA"), default="STA")
    ap.add_argument("--sadcp-xducer", type=float, default=5.0)
    ap.add_argument("--sadcp-timeoff", default=None, metavar="SECONDS|auto")
    ap.add_argument("--sadcp-nav", metavar="PATH", default=None)
    ap.add_argument("--sadcp-reingest", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    ap.add_argument("--port", type=int, default=8642, help="port (default: 8642)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")   # QA panels render headless

    labels = list(args.stations)
    if not labels and args.index:
        from ..discovery import all_station_labels
        labels = all_station_labels(args.index, Path(args.root))
    if not labels:
        ap.error("give one or more station ids, or --index to serve the whole archive")

    raw_cfgs = []
    for p in args.sadcp:                     # the --sadcp-* knobs apply to every entry
        try:
            cfg = SadcpConfig(folder=p, source=args.sadcp_source,
                              filetype=args.sadcp_filetype, xducer=args.sadcp_xducer,
                              timeoff=parse_timeoff(args.sadcp_timeoff),
                              nav=args.sadcp_nav, reingest=args.sadcp_reingest)
            cfg.validate_folder()            # fail at launch, not at the first solve
        except ValueError as e:
            ap.error(str(e))
        raw_cfgs.append(cfg)

    codas_cfgs = []
    for p in args.sadcp_codas:
        from ..io.sadcp_codas import resolve_codas_nc
        try:
            resolve_codas_nc(p)              # fail at launch: exists + one unambiguous .nc
        except FileNotFoundError as e:
            ap.error(f"--sadcp-codas: {e}")
        codas_cfgs.append(SadcpConfig(folder=p, source="codas"))
    # products discovered under <root>/codas join the dropdown (selecting one is an
    # explicit user action; the default constraint stays with the explicit flags)
    found_cfgs = merge_discovered_codas(args.root, codas_cfgs)[len(codas_cfgs):]
    if found_cfgs:
        print(f"studio: offering CODAS products found under {Path(args.root) / 'codas'} "
              f"in the SADCP source dropdown: "
              + ", ".join(codas_label(c.folder) for c in found_cfgs), flush=True)

    # fail at launch, not as a 500 at the first solve: every station id must resolve
    # to files (catches a missing --root, a typo'd id, or a stray token parsed as a
    # station). Path checks only -- --from-hex CTD conversion still happens lazily.
    from ..discovery import _load_index, discover
    root_p = Path(args.root)
    idx = _load_index(Path(args.index)) if args.index else None
    for lab in labels:
        try:
            discover(lab, root=root_p, cruise=args.cruise, index=idx, from_hex=False)
        except (FileNotFoundError, ValueError) as e:
            msg = f"station {lab!r}: {e}"
            if not root_p.is_dir():
                msg += (f"  [--root {args.root!r} does not exist under {Path.cwd()} "
                        f"— pass --root <cruise folder>]")
            ap.error(msg)

    state = StudioState(labels, root=args.root, cruise=args.cruise, index=args.index,
                        from_hex=args.from_hex, ctd_cache=args.ctd_cache, sadcp=raw_cfgs,
                        sadcp_codas=codas_cfgs, sadcp_found=found_cfgs)
    app = create_app(state)

    import uvicorn
    url = f"http://{args.host}:{args.port}"
    print(f"pyladcp studio: {len(labels)} station(s) at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        import webbrowser
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0

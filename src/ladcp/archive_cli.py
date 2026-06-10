"""``ladcp-index`` -- build / inspect the LADCP archive index (#4a).

Scan a raw acquisition tree once and cache a station -> file map so the processing CLI never
has to be told which deployment file is which cast::

    ladcp-index build --ladcp raw_ladcp_test/LADCP/Data --ctd raw_CTD
    ladcp-index show                      # print the resolved casts

Re-running ``build`` only decodes newly arrived files (the scan cache is keyed by
name+size+mtime); pass ``--rescan`` to force a full re-decode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .archive import INDEX_NAME, build_index, load_index


def _cmd_build(args: argparse.Namespace) -> int:
    idx = build_index(args.ladcp, args.ctd, root=args.root,
                      master_subdir=args.master_subdir, slave_subdir=args.slave_subdir,
                      out=args.out, rescan=args.rescan)
    out = args.out or str(Path(args.root) / INDEX_NAME)
    n_new = "(incremental)" if not args.rescan else "(full rescan)"
    print(f"indexed {len(idx['casts'])} casts from {len(idx['scan_cache'])} files {n_new}")
    print(f"  -> {out}")
    _print_casts(idx)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    idx = load_index(args.index or str(Path(args.root) / INDEX_NAME))
    if not idx:
        print("no index found (run `ladcp-index build` first)")
        return 1
    _print_casts(idx)
    return 0


def _print_casts(idx: dict) -> None:
    for st, r in sorted((idx.get("casts") or {}).items()):
        slave = os.path.basename(r["slave"]) if r["slave"] else "-"
        print(f"  {st:12} {os.path.basename(r['master']):16} {slave:16} "
              f"{r['utc']}  ({r['provenance']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ladcp-index",
                                 description="build / inspect the LADCP archive index")
    ap.add_argument("--root", default=".", help="base dir stored paths are relative to")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="scan the archive and (re)write the index")
    b.add_argument("--ladcp", required=True,
                   help="LADCP archive dir: MASTER/ + SLAVE/ subdirs when present, else "
                        "all *.000 in the tree (heads classified by name or PD0 facing bit)")
    b.add_argument("--ctd", required=True, help="dir of Seabird CTD .hex/.hdr files (anchor)")
    b.add_argument("--master-subdir", default="MASTER")
    b.add_argument("--slave-subdir", default="SLAVE")
    b.add_argument("-o", "--out", default=None, help=f"index path (default <root>/{INDEX_NAME})")
    b.add_argument("--rescan", action="store_true", help="force full re-decode (ignore cache)")
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("show", help="print the resolved casts")
    s.add_argument("--index", default=None, help=f"index path (default <root>/{INDEX_NAME})")
    s.set_defaults(func=_cmd_show)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

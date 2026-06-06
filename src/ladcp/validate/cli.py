"""CLI entry point: ``ladcp-validate``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .harness import cast_inputs, run, write_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LADCP validation harness")
    ap.add_argument("cast", nargs="?", default="MORIA-05",
                    help="cast to validate (MORIA-05, MORIA-06)")
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="project root containing raw_ladcp_test/, clean_ctd/, figures/")
    ap.add_argument("--out", type=Path, default=Path("validation_out"))
    args = ap.parse_args(argv)

    try:
        inputs = cast_inputs(args.root, args.cast)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2
    rep = run(inputs)
    jpath, mpath = write_report(rep, args.out)

    icon = {"pass": "PASS", "fail": "FAIL", "pending": "....."}
    print(f"\nValidation: {rep.cast}\n")
    for c in rep.checks:
        gate = " [gate]" if c.gate else ""
        print(f"  {icon.get(c.status, c.status):5}  {c.name}{gate}: {c.detail}")
    print(f"\n  OVERALL: {rep.overall}")
    print(f"  report: {mpath}  /  {jpath}\n")
    return 0 if rep.overall in ("PASS", "INCOMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())

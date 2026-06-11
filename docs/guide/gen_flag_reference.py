"""Generate appendix-a-flags.md from the live CLI --help output.

Run from the repository root:  python docs/guide/gen_flag_reference.py

The generated file is committed; ``tests/test_guide_commands.py`` regenerates it and
fails when the committed copy is stale, so the appendix can never drift from the code.
``COLUMNS`` is pinned for deterministic argparse wrapping.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "appendix-a-flags.md"

COMMANDS: list[tuple[str, list[str]]] = [
    ("ladcp-qa", ["ladcp-qa", "--help"]),
    ("ladcp-index", ["ladcp-index", "--help"]),
    ("ladcp-index build", ["ladcp-index", "build", "--help"]),
    ("ladcp-index show", ["ladcp-index", "show", "--help"]),
    ("ladcp-compare", ["ladcp-compare", "--help"]),
    ("ladcp-sadcp-section", ["ladcp-sadcp-section", "--help"]),
]

HEADER = """\
# Appendix A · CLI flag reference

This appendix is **generated from the live `--help` output** of each pyladcp command
(`docs/guide/gen_flag_reference.py`); a CI test fails whenever it goes stale, so what
you read here is exactly what the installed code accepts.

"""


def generate() -> str:
    # 80 columns: the widest monospace line that fits the PDF page (code blocks don't wrap)
    env = dict(os.environ, COLUMNS="80", MPLBACKEND="Agg")
    parts = [HEADER]
    for title, cmd in COMMANDS:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        parts.append(f"## `{title}`\n\n```text\n{res.stdout.rstrip()}\n```\n")
    return "\n".join(parts)


def main() -> int:
    text = generate()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"STALE: {OUT} does not match the live --help output. "
                  f"Regenerate with: python {Path(__file__).relative_to(Path.cwd())}")
            return 1
        print(f"OK: {OUT} is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

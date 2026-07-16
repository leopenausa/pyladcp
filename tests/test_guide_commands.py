"""The guide cannot rot: execute every CI-marked command in docs/guide/ and keep the
generated flag appendix in sync with the live --help output.

A fenced ``bash`` block in a guide chapter is executed here when the line immediately
above the fence is the marker ``<!-- guide-test -->``. Blocks run with the repository
root as cwd inside a sandbox: the fixture tree is symlinked into a tmp dir so guide
outputs (``qa_out/`` etc.) never touch the working tree.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "guide"

_BLOCK = re.compile(r"<!-- guide-test -->\s*\n```bash\n(.*?)```", re.DOTALL)


def _marked_blocks() -> list[tuple[str, str]]:
    out = []
    for md in sorted(GUIDE.glob("*.md")):
        for i, block in enumerate(_BLOCK.findall(md.read_text(encoding="utf-8"))):
            out.append((f"{md.name}#{i}", block.strip()))
    return out


_BLOCKS = _marked_blocks()


def test_guide_has_marked_blocks():
    """Chapters 4+ promise CI-tested commands; make sure the marker survives edits."""
    assert len(_BLOCKS) >= 2


@pytest.mark.skipif(sys.platform == "win32",
                    reason="guide commands are bash-flavored; inside Git Bash the console "
                           "scripts resolve to the MS-Store python stub. The Windows CLI "
                           "itself is covered by the rest of the suite.")
@pytest.mark.parametrize("name,script", _BLOCKS, ids=[n for n, _ in _BLOCKS])
def test_guide_command_runs(name, script, tmp_path):
    # sandbox: the guide's relative fixture paths resolve, outputs land in tmp
    fixture_root = tmp_path / "tests" / "fixtures"
    fixture_root.parent.mkdir(parents=True)
    fixture_root.symlink_to(REPO / "tests" / "fixtures")
    env = dict(os.environ, MPLBACKEND="Agg")
    res = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                         cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, f"{name} failed:\n{script}\n--- stderr ---\n{res.stderr[-2000:]}"


def test_flag_appendix_is_current():
    """appendix-a-flags.md is generated from --help; fail when stale."""
    res = subprocess.run(
        ["python", str(GUIDE / "gen_flag_reference.py"), "--check"],
        cwd=REPO, capture_output=True, text=True,
        env=dict(os.environ, MPLBACKEND="Agg"),
    )
    assert res.returncode == 0, res.stdout + res.stderr

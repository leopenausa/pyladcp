"""--jobs N parallel batch runner: identical outputs to serial + error isolation.

A second station is synthesized by copying the committed MORIA-80 fixture under the
name MORIA-81, so the pool genuinely runs two independent stations. Outputs must be
byte-identical between --jobs 1 and --jobs 2 (the acceptance criterion: parallelism
must not change a single result bit).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ladcp.qa import cli

pytestmark = pytest.mark.slow

FIX = Path(__file__).resolve().parent / "fixtures" / "New_golden" / "Good"


@pytest.fixture(scope="module")
def two_station_root(tmp_path_factory):
    """Curated-layout root holding the fixture cast as both MORIA-80 and MORIA-81."""
    root = tmp_path_factory.mktemp("jobs_root")
    (root / "LADCP").mkdir()
    (root / "CTD").mkdir()
    for st in ("80", "81"):
        shutil.copy(FIX / "LADCP" / "MORIA-80-LADCP-M.000",
                    root / "LADCP" / f"MORIA-{st}-LADCP-M.000")
        shutil.copy(FIX / "LADCP" / "MORIA-80-LADCP-S.000",
                    root / "LADCP" / f"MORIA-{st}-LADCP-S.000")
        shutil.copy(FIX / "CTD" / "moria-80_clean.cnv",
                    root / "CTD" / f"moria-{st}_clean.cnv")
    return root


def _run(root: Path, out: Path, jobs: int, stations=("80", "81")) -> int:
    return cli.main([*stations, "--root", str(root), "--out", str(out),
                     "--jobs", str(jobs), "--no-plots", "--no-export", "--no-progress"])


def test_parallel_outputs_identical_to_serial(two_station_root, tmp_path):
    out1, out2 = tmp_path / "serial", tmp_path / "jobs2"
    assert _run(two_station_root, out1, jobs=1) == 0
    assert _run(two_station_root, out2, jobs=2) == 0
    compared = 0
    for st in ("MORIA-80", "MORIA-81"):
        for suffix in (".lad", ".bot", "_qa.txt"):
            a = out1 / "stations" / st / f"{st}{suffix}"
            b = out2 / "stations" / st / f"{st}{suffix}"
            assert a.exists() and b.exists(), f"missing {st}{suffix}"
            assert a.read_bytes() == b.read_bytes(), f"{st}{suffix} differs serial vs --jobs 2"
            compared += 1
    assert compared == 6


def test_parallel_log_has_both_stations(two_station_root, tmp_path):
    out = tmp_path / "logrun"
    assert _run(two_station_root, out, jobs=2) == 0
    text = (out / "ladcp-qa.log").read_text(encoding="utf-8")
    assert "MORIA-80" in text and "MORIA-81" in text
    assert "parallel: 2 worker processes" in text


def test_parallel_error_isolation(two_station_root, tmp_path):
    """A nonexistent station errors; the good one still completes; exit code is 1."""
    out = tmp_path / "errrun"
    rc = _run(two_station_root, out, jobs=2, stations=("80", "nope"))
    assert rc == 1
    assert (out / "stations" / "MORIA-80" / "MORIA-80.lad").exists()
    text = (out / "ladcp-qa.log").read_text(encoding="utf-8")
    assert "[ERROR]" in text and "nope" in text

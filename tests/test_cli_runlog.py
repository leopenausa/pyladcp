"""Progress bar + run-log file + batch failure isolation for the QA CLI."""

from __future__ import annotations

import io
import logging
import pathlib

import matplotlib
import pytest

matplotlib.use("Agg")

from ladcp.qa.cli import main as cli_main
from ladcp.qa.runlog import ProgressBar, setup_logging, teardown_logging

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
CTD80 = GOOD / "CTD" / "moria-80_clean.cnv"
_HAS_80 = CTD80.exists()


# --- ProgressBar (pure) -------------------------------------------------------------
def test_progress_bar_nontty_lines_and_count():
    buf = io.StringIO()                              # not a tty -> one line per advance
    bar = ProgressBar(2, stream=buf, enabled=True)
    bar.start("MORIA-01")
    bar.advance("MORIA-01 [warn]")
    bar.advance("MORIA-02 [warn]")
    bar.close()
    out = buf.getvalue()
    assert "1/2" in out and "2/2" in out
    assert bar.done == 2


def test_progress_bar_disabled_is_silent():
    buf = io.StringIO()
    bar = ProgressBar(0, stream=buf)                 # total 0 -> disabled
    bar.start("x"); bar.advance("x"); bar.close()
    assert buf.getvalue() == ""


# --- run-log file -------------------------------------------------------------------
def test_setup_logging_writes_file(tmp_path):
    logfile = tmp_path / "run.log"
    setup_logging(logfile, console_level=logging.CRITICAL)   # silence console during test
    try:
        logging.getLogger("ladcp.qa").info("hello %d", 42)
    finally:
        teardown_logging()
    assert logfile.exists()
    assert "hello 42" in logfile.read_text()
    # teardown detaches handlers
    assert logging.getLogger("ladcp.qa").handlers == []


# --- CLI integration ----------------------------------------------------------------
@pytest.mark.skipif(not _HAS_80, reason="MORIA New_golden fixture not present")
def test_cli_batch_logs_and_isolates_failure(tmp_path, capsys):
    # one good station + one bogus id: the batch must finish, log both, and exit non-zero
    rc = cli_main(["80", "99", "--root", str(GOOD), "--out", str(tmp_path), "--no-plots"])
    assert rc == 1                                   # 99 errors -> non-zero
    log = (tmp_path / "ladcp-qa.log").read_text()
    assert "MORIA-80" in log and "[ERROR] 99" in log
    assert "done: 2 station(s)" in log
    # the good station still produced its outputs
    assert (tmp_path / "stations" / "MORIA-80" / "MORIA-80.lad").exists()
    # quiet (non-verbose) console still surfaces the summary + the failure
    out = capsys.readouterr().out
    assert "done: 2 station(s)" in out


@pytest.mark.skipif(not _HAS_80, reason="MORIA New_golden fixture not present")
def test_cli_no_log_flag(tmp_path):
    rc = cli_main(["--down", str(GOOD / "LADCP" / "MORIA-80-LADCP-M.000"),
                   "--up", str(GOOD / "LADCP" / "MORIA-80-LADCP-S.000"),
                   "--station", "MORIA-80", "--no-plots", "--no-export", "--no-log",
                   "-o", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / "ladcp-qa.log").exists()

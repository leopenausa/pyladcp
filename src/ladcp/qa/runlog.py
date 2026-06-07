"""Progress bar + run logging for the QA CLI batches.

Two small, dependency-free helpers used by :mod:`ladcp.qa.cli`:

* :class:`ProgressBar` -- an in-place stderr bar for long station batches (one line per step
  when the stream is not a TTY, so redirected/CI runs still show progress).
* :func:`setup_logging` -- routes the CLI's per-station detail to a timestamped log file
  (always) and echoes it to the console at a chosen verbosity, so failures and what was
  produced are recorded even on a quiet (non-``--verbose``) run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "ladcp.qa"


class ProgressBar:
    """Minimal progress bar on a stream (``sys.stderr`` by default).

    Renders in place on a TTY (``\\r`` + clear-to-end); falls back to one line per advance
    otherwise. A disabled instance (``enabled=False`` or ``total<=0``) is a silent no-op, so
    callers need no branching.
    """

    def __init__(self, total: int, *, enabled: bool = True, stream=None, width: int = 26):
        self.total = int(total)
        self.done = 0
        self.stream = stream or sys.stderr
        self.width = width
        self.enabled = bool(enabled) and self.total > 0
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def _render(self, label: str) -> None:
        if not self.enabled:
            return
        frac = self.done / self.total if self.total else 1.0
        fill = int(round(self.width * frac))
        bar = "█" * fill + "░" * (self.width - fill)
        line = f"[{bar}] {self.done}/{self.total}  {label}"
        if self.tty:
            self.stream.write("\r" + line + "\033[K")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def start(self, label: str) -> None:
        """Show the item about to be processed (count not advanced)."""
        self._render(f"→ {label}")

    def advance(self, label: str = "") -> None:
        """Mark one item finished and redraw."""
        self.done += 1
        self._render(label)

    def clear(self) -> None:
        """Erase the in-place line so the caller can print something else cleanly."""
        if self.enabled and self.tty:
            self.stream.write("\r\033[K")
            self.stream.flush()

    def close(self) -> None:
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()


def setup_logging(logfile: str | Path | None, *, console_level: int) -> logging.Logger:
    """Configure the ``ladcp.qa`` logger: a detailed file log plus a console echo.

    ``logfile`` is the run-log path (``None`` to skip the file). ``console_level`` sets the
    stdout echo level -- ``logging.INFO`` to stream per-station detail (``--verbose`` / a
    single station), ``logging.WARNING`` to show only problems while a progress bar carries
    the rest. Idempotent: re-running clears prior handlers (safe for repeated in-process
    calls, e.g. tests). Pair with :func:`teardown_logging` to close the file.
    """
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                          "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)
    return log


def teardown_logging() -> None:
    """Detach and close the ``ladcp.qa`` handlers (flush + release the log file)."""
    log = logging.getLogger(LOGGER_NAME)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()

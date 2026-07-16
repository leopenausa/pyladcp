"""The ``ladcp`` umbrella (phase B): freshness rule, process selection, config actions.

Everything runs in the quick lane: ``discover``/``process_station`` are monkeypatched
(in :mod:`ladcp.qa.batch` and :mod:`ladcp.hub.freshness`) so selection and precedence
are asserted without solving; the real solve parity is covered by
``tests/test_cruise_config.py`` (slow) and the untouched golden CLI tests.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from ladcp.discovery import StationFiles
from ladcp.hub import cli as hub
from ladcp.hub import freshness

# ---------------------------------------------------------------------------
# a synthetic cruise directory: real files (for mtimes), fake index, fake pipeline


def _make_cruise(tmp_path: Path, stations=("MORIA-07", "MORIA-08")) -> Path:
    for st in stations:
        (tmp_path / f"{st}-M.000").write_bytes(b"pd0")
        (tmp_path / f"{st}-S.000").write_bytes(b"pd0")
        (tmp_path / f"{st}.cnv").write_text("ctd", encoding="utf-8")
    idx = {"version": 2, "casts": {st: {} for st in stations}}
    (tmp_path / ".ladcp_archive.json").write_text(json.dumps(idx), encoding="utf-8")
    (tmp_path / "cruise.toml").write_text(
        '[cruise]\nname = "MORIA"\n[data]\nroot = "."\nout = "qa_out"\n'
        'index = ".ladcp_archive.json"\n', encoding="utf-8")
    return tmp_path


def _fake_discover_for(root: Path):
    def fake_discover(station, *, root=root, cruise="LADCP", index=None,
                      from_hex=False, ctd_cache=None, ctd_dir=None):
        r = Path(root)
        label = station if station.startswith("MORIA-") else f"MORIA-{station}"
        if not (r / f"{label}-M.000").exists():
            raise FileNotFoundError(f"no master for {label}")
        return StationFiles(down=r / f"{label}-M.000", up=r / f"{label}-S.000",
                            ctd=r / f"{label}.cnv", label=label, ctd_utc=None)
    return fake_discover


def _write_qa_json(outdir: Path, label: str) -> Path:
    st = Path(outdir) / "stations" / label
    st.mkdir(parents=True, exist_ok=True)
    p = st / f"{label}_qa.json"
    p.write_text('{"overall_status": "ok"}', encoding="utf-8")
    return p


def _age(path: Path, seconds: float) -> None:
    """Backdate ``path`` so a later write is unambiguously newer (no sleep needed)."""
    t = time.time() - seconds
    os.utime(path, (t, t))


def _bump(path: Path, seconds: float = 60.0) -> None:
    """Push ``path``'s mtime into the future — unambiguously newer than any report."""
    t = time.time() + seconds
    os.utime(path, (t, t))


@pytest.fixture()
def cruise(tmp_path, monkeypatch):
    root = _make_cruise(tmp_path)
    fake = _fake_discover_for(root)
    calls: list[str] = []

    def fake_process(down, up, ctd_path, station, outdir, make_plots, cfg, cruise="LADCP",
                     formats=None, ctd_utc=None, edits=None, hint_root=None,
                     param_overrides=None):
        calls.append(station)
        _write_qa_json(outdir, station)         # the real pipeline's done-marker
        return "ok", None

    from ladcp.qa import batch
    monkeypatch.setattr(batch, "discover", fake)
    monkeypatch.setattr(batch, "process_station", fake_process)
    monkeypatch.setattr(freshness, "discover", fake)
    monkeypatch.chdir(root)
    return root, calls


# ---------------------------------------------------------------------------
# freshness

def test_station_state_matrix(cruise):
    root, _ = cruise
    out = root / "qa_out"
    kw = dict(root=root, outdir=out, cruise="MORIA", config_path=root / "cruise.toml")

    assert freshness.station_state("MORIA-07", **kw).state == "missing"     # no qa.json

    for f in root.iterdir():                        # all inputs older than the report
        _age(f, 60)
    qa = _write_qa_json(out, "MORIA-07")
    assert freshness.station_state("MORIA-07", **kw).state == "fresh"

    _bump(root / "MORIA-07-M.000")                                          # input newer
    st = freshness.station_state("MORIA-07", **kw)
    assert st.state == "stale" and "MORIA-07-M.000" in st.reason

    _age(root / "MORIA-07-M.000", 60)
    _bump(root / "cruise.toml")                                             # config newer
    st = freshness.station_state("MORIA-07", **kw)
    assert st.state == "stale" and "cruise.toml" in st.reason

    _age(root / "cruise.toml", 60)
    assert freshness.station_state("MORIA-07", **kw).state == "fresh"
    qa.unlink()
    assert freshness.station_state("MORIA-07", **kw).state == "missing"


def test_station_state_unresolvable_is_missing(cruise):
    root, _ = cruise
    st = freshness.station_state("MORIA-99", root=root, outdir=root / "qa_out")
    assert st.state == "missing" and "unresolved" in st.reason


def test_select_new_splits(cruise):
    root, _ = cruise
    for f in root.iterdir():
        _age(f, 60)
    _write_qa_json(root / "qa_out", "MORIA-07")
    todo, states = freshness.select_new(["MORIA-07", "MORIA-08"],
                                        root=root, outdir=root / "qa_out")
    assert todo == ["MORIA-08"]
    assert [s.state for s in states] == ["fresh", "missing"]


# ---------------------------------------------------------------------------
# ladcp process

def test_process_default_runs_only_missing_then_nothing(cruise, capsys):
    root, calls = cruise
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    assert calls == ["MORIA-07", "MORIA-08"]
    calls.clear()
    for f in root.iterdir():
        _age(f, 60)                                 # inputs older than the fresh reports
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    assert calls == []
    assert "nothing to do" in capsys.readouterr().out


def test_process_reruns_stale_only(cruise):
    root, calls = cruise
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    calls.clear()
    for f in root.iterdir():
        _age(f, 60)
    _bump(root / "MORIA-08.cnv")
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    assert calls == ["MORIA-08"]


def test_process_named_station_is_unconditional(cruise):
    root, calls = cruise
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    calls.clear()
    for f in root.iterdir():
        _age(f, 60)
    assert hub.main(["process", "MORIA-07", "--no-log", "--no-progress"]) == 0
    assert calls == ["MORIA-07"]


def test_process_all_and_force_ignore_freshness(cruise):
    root, calls = cruise
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    for f in root.iterdir():
        _age(f, 60)
    for flag in ("--all", "--force"):
        calls.clear()
        assert hub.main(["process", flag, "--no-log", "--no-progress"]) == 0
        assert calls == ["MORIA-07", "MORIA-08"], flag


def test_process_labels_with_all_is_an_error(cruise, capsys):
    with pytest.raises(SystemExit):
        hub.main(["process", "MORIA-07", "--all"])
    assert "not both" in capsys.readouterr().err


def test_process_without_config_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as ei:
        hub.main(["process"])
    assert "no cruise.toml" in str(ei.value)


def test_process_config_error_names_offender(cruise, capsys):
    root, _ = cruise
    (root / "cruise.toml").write_text("[solve]\nbogus = 1\n", encoding="utf-8")
    assert hub.main(["process", "--no-log"]) == 1
    assert "unknown key" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ladcp config

def test_config_show_annotates_provenance(cruise, capsys):
    root, _ = cruise
    (root / "cruise.toml").write_text(
        '[cruise]\nname = "MORIA"\n[solve]\nbotfac = 0.5\n'
        '[params]\npglim = 30.0\n[params.MORIA-07]\nzbottom = 120.0\n', encoding="utf-8")
    assert hub.main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "cruise               MORIA" in out and "cruise.toml" in out
    assert "botfac               0.5" in out
    assert "solver               inverse" in out and "default" in out
    assert "preset:MORIA" in out
    assert "pglim = 30.0" in out and "[params.MORIA-07]" in out


def test_config_validate_flags_missing_paths(cruise, capsys):
    root, _ = cruise
    assert hub.main(["config", "validate"]) == 0
    (root / "cruise.toml").write_text(
        '[data]\nroot = "nowhere"\n[sadcp]\nfolder = "gone"\n', encoding="utf-8")
    assert hub.main(["config", "validate"]) == 1
    out = capsys.readouterr().out
    assert "data.root" in out and "sadcp.folder" in out


def _set_editor(monkeypatch, tmp_path: Path, body: str) -> None:
    """EDITOR = a tiny python script(file) so the test never opens a real editor."""
    helper = tmp_path / "fake_editor.py"
    helper.write_text("import sys, pathlib\n"
                      "p = pathlib.Path(sys.argv[1])\n" + body, encoding="utf-8")
    editor = f"{Path(sys.executable).as_posix()} {helper.as_posix()}"
    monkeypatch.setenv("EDITOR", editor)


def test_config_edit_saves_valid(cruise, monkeypatch, tmp_path):
    root, _ = cruise
    _set_editor(monkeypatch, tmp_path,
                "p.write_text(p.read_text() + '[solve]\\nbotfac = 0.5\\n')\n")
    assert hub.main(["config", "edit"]) == 0
    assert "botfac = 0.5" in (root / "cruise.toml").read_text(encoding="utf-8")
    assert not (root / "cruise.toml.edit").exists()


def test_config_edit_refuses_invalid_and_keeps_edits(cruise, monkeypatch, tmp_path, capsys):
    root, _ = cruise
    before = (root / "cruise.toml").read_text(encoding="utf-8")
    _set_editor(monkeypatch, tmp_path,
                "p.write_text(p.read_text() + '[solve]\\nbogus = 1\\n')\n")
    assert hub.main(["config", "edit"]) == 1
    assert (root / "cruise.toml").read_text(encoding="utf-8") == before
    kept = root / "cruise.toml.edit"
    assert kept.exists() and "bogus" in kept.read_text(encoding="utf-8")
    assert "NOT saved" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# bare ladcp

def test_bare_ladcp_points_at_config(cruise, capsys):
    assert hub.main([]) == 0
    out = capsys.readouterr().out
    assert "cruise.toml:" in out and "process" in out


def test_bare_ladcp_without_config_hints_init(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert hub.main([]) == 0
    assert "no cruise.toml" in capsys.readouterr().out

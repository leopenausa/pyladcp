"""``ladcp status`` (phase D): freshness block, QA rollup, loose ends, --json.

Everything quick-lane: a synthetic curated tree (filenames + crafted qa.json files),
with ``freshness``'s discovery monkeypatched only where a station must resolve to
specific fake heads. Rendering is asserted on substance (counts, offenders,
actions), not exact layout.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ladcp.hub import cli as hub
from ladcp.hub import cruise_config as cc
from ladcp.hub import status


def _age(path: Path, seconds: float = 60.0) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def _qa_json(outdir: Path, label: str, overall: str = "ok",
             metrics: dict[str, str] | None = None) -> Path:
    d = outdir / "stations" / label
    d.mkdir(parents=True, exist_ok=True)
    payload = {"station": label, "overall_status": overall, "warnings": [],
               "metrics": {name: {"status": s} for name, s in (metrics or {}).items()}}
    p = d / f"{label}_qa.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture()
def cruise(tmp_path):
    """Curated 3-station tree: 01 dual-head+ctd, 02 single-head, 03 dual no ctd."""
    (tmp_path / "LADCP").mkdir()
    for name in ("A-01-LADCP-M.000", "A-01-LADCP-S.000",
                 "A-02-LADCP-M.000",
                 "A-03-LADCP-M.000", "A-03-LADCP-S.000"):
        (tmp_path / "LADCP" / name).write_bytes(b"")
    (tmp_path / "CTD").mkdir()
    (tmp_path / "CTD" / "a-01_clean.cnv").write_text("", encoding="utf-8")
    (tmp_path / "CTD" / "a-02_clean.cnv").write_text("", encoding="utf-8")
    (tmp_path / "cruise.toml").write_text(
        '[cruise]\nname = "A"\n[data]\nroot = "."\nout = "qa_out"\n', encoding="utf-8")
    for f in tmp_path.rglob("*"):
        _age(f)
    _age(tmp_path)
    return tmp_path


def test_gather_full_picture(cruise):
    _qa_json(cruise / "qa_out", "A-01", "ok", {"tilt_max": "ok"})
    _qa_json(cruise / "qa_out", "A-02", "warn", {"nearfield_errvel_ratio": "warn"})
    ccfg = cc.load_config(cruise / "cruise.toml")
    data = status.gather(ccfg)
    assert data["n_stations"] == 3
    assert data["freshness"] == {"fresh": 2, "stale": 0, "missing": 1}
    assert data["qa"] == {"ok": 1, "warn": 1, "fail": 0}
    by = {e["label"]: e for e in data["stations"]}
    assert by["A-03"]["freshness"] == "missing" and by["A-03"]["qa"] is None
    assert by["A-02"]["problems"] == ["nearfield_errvel_ratio"]
    assert "single-head (no up-looker)" in by["A-02"]["loose_ends"]
    assert "no CTD (velocity impossible)" in by["A-03"]["loose_ends"]
    assert by["A-01"]["loose_ends"] == []
    assert data["index_stale"] is False


def test_gather_flags_unconstrained_and_journal(cruise):
    _qa_json(cruise / "qa_out", "A-01", "ok", {"tilt_max": "ok"})   # no sadcp_* metric
    jdir = cruise / ".ladcp_edits"
    jdir.mkdir()
    (jdir / "A-01.json").write_text("{}", encoding="utf-8")
    (cruise / "cruise.toml").write_text(
        '[cruise]\nname = "A"\n[data]\nroot = "."\nout = "qa_out"\n'
        '[sadcp]\nfolder = "sadcp"\n', encoding="utf-8")
    _age(cruise / "cruise.toml")
    data = status.gather(cc.load_config(cruise / "cruise.toml"))
    le = {e["label"]: e["loose_ends"] for e in data["stations"]}
    assert "last solve had no SADCP constraint" in le["A-01"]
    assert "edit journal not applied" in le["A-01"]


def test_gather_index_staleness(cruise):
    (cruise / ".ladcp_archive.json").write_text(
        json.dumps({"version": 2, "casts": {"A-01": {
            "station": "A-01",
            "master": str(cruise / "LADCP" / "A-01-LADCP-M.000"),
            "slave": str(cruise / "LADCP" / "A-01-LADCP-S.000"),
            "ctd_hex": None, "utc": None, "lat": None, "lon": None,
            "depth": None, "provenance": "test"}}}), encoding="utf-8")
    _age(cruise / ".ladcp_archive.json", 30)             # older than the new arrival,
    data = status.gather(cc.load_config(cruise / "cruise.toml"))     # newer than the rest
    assert data["n_stations"] == 1                       # index wins over name pairing
    assert data["index_stale"] is False
    new = cruise / "LADCP" / "A-09-LADCP-M.000"          # a cast arrives on watch
    new.write_bytes(b"")
    data = status.gather(cc.load_config(cruise / "cruise.toml"))
    assert data["index_stale"] is True


def test_render_blocks_and_actions(cruise):
    _qa_json(cruise / "qa_out", "A-01", "fail", {"ctd_sync_corr": "fail"})
    _qa_json(cruise / "qa_out", "A-02", "warn", {"tilt_max": "warn"})
    text = "\n".join(status.render(status.gather(cc.load_config(cruise / "cruise.toml"))))
    assert "casts: 3" in text and "1 unprocessed" in text
    assert "-> ladcp process" in text
    assert "1 ok" not in text and "1 warn, 1 fail" in text
    fail_line = next(line for line in text.splitlines() if "[FAIL]" in line)
    assert "A-01" in fail_line and "ctd_sync_corr" in fail_line \
        and "A-01_report.pdf" in fail_line
    assert text.splitlines().index(fail_line) < text.splitlines().index(
        next(line for line in text.splitlines() if "[WARN]" in line))
    assert "single-head" in text and "--down-only" in text


def test_render_all_current(cruise):
    for st in ("A-01", "A-02", "A-03"):
        _qa_json(cruise / "qa_out", st, "ok")
    (cruise / "CTD" / "a-03_clean.cnv").write_text("", encoding="utf-8")
    _age(cruise / "CTD" / "a-03_clean.cnv")
    # A-02 stays single-head -> still a loose end; drop it from the tree instead
    (cruise / "LADCP" / "A-02-LADCP-M.000").unlink()
    (cruise / "qa_out" / "stations" / "A-02" / "A-02_qa.json").unlink()
    text = "\n".join(status.render(status.gather(cc.load_config(cruise / "cruise.toml"))))
    assert "nothing to process" in text


def test_cli_status_and_bare_dispatch(cruise, monkeypatch, capsys):
    _qa_json(cruise / "qa_out", "A-01", "ok")
    monkeypatch.chdir(cruise)
    assert hub.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "cruise A —" in out and "casts: 3" in out
    assert hub.main([]) == 0                             # bare ladcp = the dashboard
    assert "casts: 3" in capsys.readouterr().out


def test_cli_status_json_schema(cruise, monkeypatch, capsys):
    _qa_json(cruise / "qa_out", "A-01", "warn", {"tilt_max": "warn"})
    monkeypatch.chdir(cruise)
    assert hub.main(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"config", "cruise", "root", "outdir", "n_stations",
                         "freshness", "qa", "stations", "index_stale",
                         "sadcp_source", "sadcp_folder"}
    assert set(data["stations"][0]) == {"label", "freshness", "reason", "qa",
                                        "problems", "loose_ends"}


def test_bare_ladcp_without_config_still_hints(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert hub.main([]) == 0
    assert "ladcp init" in capsys.readouterr().out

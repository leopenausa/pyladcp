"""``ladcp-qa --edits``: journal replay, provenance, the unapplied-journal hint.

Edits are never applied without the explicit flag (no-silent-edits policy); when
they are, the QA report says so, and when a journal exists unapplied the report
WARNs with the exact re-run command -- the same actionable-note contract as the
near-field detector.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import matplotlib
import pytest

matplotlib.use("Agg")

import numpy as np

from ladcp.edits import journal_path, new_journal, save_journal
from ladcp.qa.cli import main as cli_main

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"

pytestmark = pytest.mark.skipif(
    not (GOOD / "CTD" / "moria-80_clean.cnv").exists(),
    reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def edits_root(tmp_path_factory):
    """A writable copy of the fixture root, so journals never touch fixtures/."""
    root = tmp_path_factory.mktemp("edits_root")
    for sub in ("LADCP", "CTD"):
        shutil.copytree(GOOD / sub, root / sub)
    return root


def _write_journal(root, entries, station="MORIA-80"):
    j = new_journal(station)
    j.entries = entries
    j.next_id = len(entries) + 1
    p = journal_path(root, station)
    save_journal(j, p)
    return p


def _rect(eid=1, b0=3, b1=4, e0=0, e1=10**9, head="down", note="test band"):
    return {"id": eid, "kind": "rect", "head": head, "bin_first": b0, "bin_last": b1,
            "ens_first": e0, "ens_last": e1, "view": "errvel", "note": note,
            "created": "2026-06-12T00:00:00Z"}


def _qa_json(out, station="MORIA-80") -> dict:
    p = out / "stations" / station / f"{station}_qa.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _lad_uv(out, station="MORIA-80") -> np.ndarray:
    rows = []
    for line in (out / "stations" / station / f"{station}.lad").read_text().splitlines():
        parts = line.split()
        try:
            rows.append([float(x) for x in parts[:3]])
        except (ValueError, IndexError):
            continue
    return np.asarray(rows)


def _metric(qa: dict, name: str) -> dict | None:
    m = qa.get("metrics", qa)
    if isinstance(m, dict) and name in m:
        return m[name]
    for item in (m if isinstance(m, list) else []):
        if item.get("name") == name:
            return item
    return None


BASE = ["--no-plots", "--no-export"]


def test_edits_replay_matches_nearfield_flag(edits_root, tmp_path):
    """--edits with a full-cast bins-3-4 rect == --nearfield-dn-bins 3,4, and the
    QA report carries the manual_edits provenance metric."""
    jp = _write_journal(edits_root, [_rect()])
    out_e = tmp_path / "edited"
    out_n = tmp_path / "nearfield"
    assert cli_main(["80", "--root", str(edits_root), "--out", str(out_e),
                     "--edits", str(jp)] + BASE) in (0, 1)
    assert cli_main(["80", "--root", str(edits_root), "--out", str(out_n),
                     "--nearfield-dn-bins", "3,4"] + BASE) in (0, 1)
    np.testing.assert_array_equal(_lad_uv(out_e), _lad_uv(out_n))

    m = _metric(_qa_json(out_e), "manual_edits")
    assert m is not None and "1 manual rectangle(s) replayed" in m["note"]
    assert str(jp) in m["note"]
    assert _metric(_qa_json(out_e), "manual_edits_unapplied") is None


def test_edits_dir_resolves_per_station(edits_root, tmp_path):
    _write_journal(edits_root, [_rect()])
    out = tmp_path / "out"
    assert cli_main(["80", "--root", str(edits_root), "--out", str(out),
                     "--edits", str(edits_root / ".ladcp_edits")] + BASE) in (0, 1)
    assert _metric(_qa_json(out), "manual_edits") is not None


def test_hint_warns_on_unapplied_journal(edits_root, tmp_path):
    _write_journal(edits_root, [_rect(), _rect(eid=2, b0=6, b1=6, note="x")])
    out = tmp_path / "out"
    assert cli_main(["80", "--root", str(edits_root), "--out", str(out)] + BASE) in (0, 1)
    qa = _qa_json(out)
    m = _metric(qa, "manual_edits_unapplied")
    assert m is not None and m["status"] == "warn"
    assert "NOT applied" in m["note"] and "--edits" in m["note"]
    assert qa["overall_status"] == "warn"


def test_no_hint_for_empty_journal(edits_root, tmp_path):
    _write_journal(edits_root, [])
    out = tmp_path / "out"
    assert cli_main(["80", "--root", str(edits_root), "--out", str(out)] + BASE) in (0, 1)
    assert _metric(_qa_json(out), "manual_edits_unapplied") is None


def test_stale_journal_is_an_error(edits_root, tmp_path):
    jp = _write_journal(edits_root, [_rect()])
    j = json.loads(jp.read_text())
    j["raw"] = {"down": {"file": "MORIA-80-LADCP-M.000", "size": 1, "n_ens": 2}}
    jp.write_text(json.dumps(j), encoding="utf-8")
    out = tmp_path / "out"
    rc = cli_main(["80", "--root", str(edits_root), "--out", str(out),
                   "--edits", str(jp)] + BASE)
    assert rc == 1                                       # station errored, not silently run
    assert not (out / "stations" / "MORIA-80" / "MORIA-80.lad").exists()


def test_station_mismatch_is_an_error(edits_root, tmp_path):
    jp = _write_journal(edits_root, [_rect()], station="MORIA-79")
    out = tmp_path / "out"
    rc = cli_main(["80", "--root", str(edits_root), "--out", str(out),
                   "--edits", str(jp)] + BASE)
    assert rc == 1


def test_edits_file_rejected_for_batches(edits_root, tmp_path):
    jp = _write_journal(edits_root, [_rect()])
    with pytest.raises(SystemExit):
        cli_main(["79", "80", "--root", str(edits_root), "--out", str(tmp_path / "o"),
                  "--edits", str(jp)] + BASE)


def test_missing_edits_path_rejected(edits_root, tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["80", "--root", str(edits_root), "--out", str(tmp_path / "o"),
                  "--edits", str(tmp_path / "nope")] + BASE)

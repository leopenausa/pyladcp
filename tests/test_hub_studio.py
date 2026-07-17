"""The Studio cruise-hub surface (phase E): /api/hub/* + the `ladcp studio` launcher.

Quick lane throughout: the app is built over an empty StudioState with a synthetic
filename tree, and processing is monkeypatched at the shared batch seam
(`ladcp.qa.batch.run_batch`) — the endpoint contract is what's under test, not the
solver (that parity is proven elsewhere).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ladcp.hub import cli as hub  # noqa: E402
from ladcp.studio.app import create_app  # noqa: E402
from ladcp.studio.state import StudioState  # noqa: E402


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "LADCP").mkdir()
    for name in ("B-01-LADCP-M.000", "B-01-LADCP-S.000",
                 "B-02-LADCP-M.000", "B-02-LADCP-S.000"):
        (tmp_path / "LADCP" / name).write_bytes(b"")
    (tmp_path / "CTD").mkdir()
    for st in ("01", "02"):
        (tmp_path / "CTD" / f"b-{st}_clean.cnv").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(tmp_path):
    root = _tree(tmp_path)
    state = StudioState([], root=str(root))
    app = create_app(state, hub_dir=root)
    return TestClient(app), root


def test_wizard_flow_over_http(client):
    c, root = client
    assert c.get("/api/hub/state").json()["configured"] is False

    det = c.get("/api/hub/detect").json()
    assert det["ladcp"]["layout"] == "curated"
    assert [s["label"] for s in det["ladcp"]["stations"]] == ["B-01", "B-02"]

    raw = {"cruise": {"name": "B"}, "data": {"root": ".", "out": "qa_out"}}
    prev = c.post("/api/hub/preview", json=raw)
    assert prev.status_code == 200 and 'name = "B"' in prev.json()["toml"]
    bad = c.post("/api/hub/preview", json={"solve": {"bogus": 1}})
    assert bad.status_code == 400 and "unknown key" in bad.json()["detail"]

    res = c.post("/api/hub/config", json={"config": raw, "build_index": False})
    assert res.status_code == 200
    assert (root / "cruise.toml").is_file()
    assert c.get("/api/hub/state").json()["configured"] is True

    status = c.get("/api/hub/status").json()
    assert status["n_stations"] == 2
    assert status["freshness"] == {"fresh": 0, "stale": 0, "missing": 2}


def test_status_before_setup_is_409(client):
    c, _ = client
    assert c.get("/api/hub/status").status_code == 409
    assert c.post("/api/hub/process", json={}).status_code == 409


def test_process_job_and_scorecard(client, monkeypatch):
    c, root = client
    raw = {"cruise": {"name": "B"}, "data": {"root": ".", "out": "qa_out"}}
    assert c.post("/api/hub/config", json={"config": raw}).status_code == 200

    def fake_run_batch(plan, cfg, *, outdir, **kw):
        for st in plan:
            d = Path(outdir) / "stations" / st
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{st}_qa.json").write_text("{}", encoding="utf-8")
            (d / f"{st}_qa.txt").write_text(f"scorecard {st}\n", encoding="utf-8")
        return [(st, "ok") for st in plan]

    from ladcp.qa import batch
    monkeypatch.setattr(batch, "run_batch", fake_run_batch)

    assert c.get("/api/hub/scorecard/B-01").status_code == 404
    res = c.post("/api/hub/process", json={"mode": "new"}).json()
    assert res == {"started": True, "total": 2}
    for _ in range(100):                       # the job thread is real; poll it out
        j = c.get("/api/hub/job").json()
        if not j["running"]:
            break
        time.sleep(0.05)
    assert [d["status"] for d in j["done"]] == ["ok", "ok"]
    assert j["error"] is None

    sc = c.get("/api/hub/scorecard/B-01").json()
    assert sc["text"].startswith("scorecard") and sc["pdf"] is False
    assert c.get("/api/hub/report/B-01").status_code == 404   # --no-plots equivalent

    res = c.post("/api/hub/process", json={"mode": "new"}).json()
    assert res["started"] is False and "nothing to do" in res["reason"]


def test_ladcp_studio_translates_config(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    (root / "sadcp").mkdir()
    (root / "cruise.toml").write_text(
        '[cruise]\nname = "B"\n[data]\nroot = "."\n'
        '[sadcp]\nfolder = "sadcp"\nsource = "codas"\n', encoding="utf-8")
    monkeypatch.chdir(root)
    captured: list[list[str]] = []
    from ladcp.studio import cli as studio_cli
    monkeypatch.setattr(studio_cli, "main", lambda argv: captured.append(argv) or 0)
    assert hub.main(["studio", "--no-browser"]) == 0
    argv = captured[0]
    assert "B-01" in argv and "B-02" in argv                 # the whole cruise served
    assert "--sadcp-codas" in argv and str(root / "sadcp") in argv
    assert "--hub-dir" in argv and "--start-page" in argv and "hub" in argv
    assert "--no-browser" in argv

    captured.clear()                                          # named station -> editor
    assert hub.main(["studio", "B-01", "--no-browser"]) == 0
    argv = captured[0]
    assert argv[0] == "B-01" and "B-02" not in argv
    assert "--start-page" not in argv                         # editor landing


def test_ladcp_studio_without_config_opens_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured: list[list[str]] = []
    from ladcp.studio import cli as studio_cli
    monkeypatch.setattr(studio_cli, "main", lambda argv: captured.append(argv) or 0)
    assert hub.main(["studio", "--no-browser"]) == 0
    argv = captured[0]
    assert "--hub-dir" in argv and str(tmp_path) in " ".join(argv)
    assert "--root" not in argv                               # nothing to translate yet

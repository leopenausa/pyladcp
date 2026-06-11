"""ladcp-studio server contract (Studio PR 3).

The HTTP layer must be a transparent window onto StationSession: solve responses match
a direct session solve number-for-number (after NaN -> null), and every response's
``cli`` string parses back to the configuration that produced it.
"""

from __future__ import annotations

import pathlib
import shlex

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ladcp.qa.cli import build_parser  # noqa: E402
from ladcp.session import SessionConfig, SolveConfig, StationSession  # noqa: E402
from ladcp.studio.server import (  # noqa: E402
    StationEntry,
    StudioState,
    config_from_body,
    create_app,
)

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def state():
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    return StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry})


@pytest.fixture(scope="module")
def client(state):
    return TestClient(create_app(state))


def test_stations_endpoint(client):
    r = client.get("/api/stations")
    assert r.status_code == 200
    assert r.json() == {"stations": ["MORIA-80"], "cruise": "MORIA",
                        "sadcp": False, "sadcp_folder": None}


def test_prepare_then_cached(client):
    r1 = client.post("/api/station/MORIA-80/prepare", json={})
    assert r1.status_code == 200 and r1.json()["cached"] is False
    r2 = client.post("/api/station/MORIA-80/prepare", json={})
    assert r2.status_code == 200 and r2.json()["cached"] is True


def test_solve_matches_direct_session(client):
    body = {"solve": {"drot": -9.878379, "botfac": 0.5}}
    r = client.post("/api/station/MORIA-80/solve", json=body)
    assert r.status_code == 200
    p = r.json()

    direct = StationSession(DOWN, UP, CTD, station="MORIA-80", cruise="MORIA").solve(
        SessionConfig(solve=SolveConfig(drot=-9.878379, botfac=0.5)))
    for key, ref in (("z", direct.vp.z), ("u", direct.vp.u), ("v", direct.vp.v),
                     ("uerr", direct.vp.uerr)):
        got = np.array([np.nan if x is None else x for x in p["profile"][key]])
        np.testing.assert_array_equal(got, np.asarray(ref, float), err_msg=key)
    assert p["zbottom"] == pytest.approx(direct.zbottom)
    assert p["drot_source"] == "explicit"
    assert p["solver"] == "inverse"
    assert p["bt"] is not None and len(p["bt"]["z"]) == direct.bp.z.size
    assert sum(x is not None for x in p["bt"]["u"]) == direct.bp.n_bins


def test_solve_cli_string_roundtrips(client):
    body = {"edit": {"down_only": True},
            "solve": {"solver": "inverse", "botfac": 0.0, "smoofac": 0.2}}
    r = client.post("/api/station/MORIA-80/solve", json=body)
    assert r.status_code == 200
    cli = r.json()["cli"]
    args = build_parser().parse_args(shlex.split(cli)[1:])
    assert args.stations == ["MORIA-80"]
    assert SessionConfig.from_args(args) == config_from_body(body, StudioState([]))


def test_solve_weight_change_uses_cache(client):
    r = client.post("/api/station/MORIA-80/solve",
                    json={"solve": {"drot": -9.878379, "barofac": 2.0}})
    assert r.status_code == 200
    p = r.json()
    assert p["prepared"] is True                 # context cached by earlier tests
    assert p["solve_ms"] < 1000                  # warm solve, not a rebuild


def test_bad_config_is_400(client):
    r = client.post("/api/station/MORIA-80/solve", json={"solve": {"botfac": "x"}})
    assert r.status_code == 400
    r = client.post("/api/station/MORIA-80/solve", json={"solve": {"solver": "magic"}})
    assert r.status_code == 400
    assert "solver" in r.json()["detail"]


def test_unknown_station_is_404(client):
    assert client.post("/api/station/NOPE/solve", json={}).status_code == 404


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "studio.js" in r.text and "ladcp" in r.text.lower()


def test_config_from_body_defaults_match_sessionconfig():
    assert config_from_body({}, StudioState([])) == SessionConfig()


# ---------------------------------------------------------------- QA panels (PR 4)

def test_qa_panel_returns_png(client):
    r = client.post("/api/station/MORIA-80/qa/velocity",
                    json={"solve": {"drot": -9.878379}})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(r.content) > 10_000                # a real figure, not a stub


def test_qa_panel_cached_second_time(client):
    body = {"solve": {"drot": -9.878379}}
    first = client.post("/api/station/MORIA-80/qa/shear", json=body)
    again = client.post("/api/station/MORIA-80/qa/shear", json=body)
    assert first.status_code == again.status_code == 200
    assert first.content == again.content         # byte-identical from the PNG cache


def test_qa_unknown_panel_is_404(client):
    r = client.post("/api/station/MORIA-80/qa/nonsense", json={})
    assert r.status_code == 404
    assert "not available" in r.json()["detail"]


def test_qa_unavailable_panel_is_404(client):
    # no --sadcp at launch -> the sadcp panel cannot render
    r = client.post("/api/station/MORIA-80/qa/sadcp", json={})
    assert r.status_code == 404


def test_solve_lists_available_panels(client):
    r = client.post("/api/station/MORIA-80/solve", json={"solve": {"drot": -9.878379}})
    panels = r.json()["panels"]
    assert "velocity" in panels and "raw" in panels and "weights" in panels
    assert "sadcp" not in panels                  # no SADCP source at launch


# ---------------------------------------------------------------- polish (PR 5)

def test_solve_reports_stage_timings(client):
    r = client.post("/api/station/MORIA-80/solve", json={"solve": {"drot": -9.878379}})
    stages = r.json()["stages"]
    assert set(stages) >= {"load_ms", "ctd_ms", "build_ms"}
    assert all(isinstance(v, (int, float)) and v >= 0 for v in stages.values())


def test_lad_download(client):
    r = client.post("/api/station/MORIA-80/lad", json={"solve": {"drot": -9.878379}})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert 'filename="MORIA-80.lad"' in r.headers["content-disposition"]
    text = r.text
    assert text.startswith("Filename")            # LDEO .lad header
    assert "Columns     =" in text
    # data rows: z, u, v (+ extras); spot-check the first numeric row parses
    row = next(line for line in text.splitlines() if line and line[0].isdigit())
    assert len(row.split()) >= 3


def test_lad_matches_direct_export(client, tmp_path):
    from ladcp.qa.export import write_lad
    body = {"solve": {"drot": -9.878379, "botfac": 0.5}}
    r = client.post("/api/station/MORIA-80/lad", json=body)

    ses = StationSession(DOWN, UP, CTD, station="MORIA-80", cruise="MORIA")
    cfg = SessionConfig(solve=SolveConfig(drot=-9.878379, botfac=0.5))
    result = ses.solve(cfg)
    prep = ses.prepare(cfg.edit)
    path = tmp_path / "direct.lad"
    write_lad(result.vp, str(path), station="MORIA-80", lat=prep.lat, lon=prep.lon,
              drot=-9.878379, time=prep.when)
    assert r.text == path.read_text(encoding="utf-8")


# ------------------------------------------------- data errors are clean 400s (PR 5)

def test_bad_sadcp_folder_is_400_not_500(tmp_path):
    from ladcp.session import SadcpConfig
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    st = StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry},
                     sadcp=SadcpConfig(folder=str(tmp_path / "empty_sadcp")))
    c = TestClient(create_app(st))
    r = c.post("/api/station/MORIA-80/solve",
               json={"solve": {"drot": -9.878379}, "use_sadcp": True})
    assert r.status_code == 400
    assert "FileNotFoundError" in r.json()["detail"]
    # constraint off -> same station still solves fine
    ok = c.post("/api/station/MORIA-80/solve",
                json={"solve": {"drot": -9.878379}, "use_sadcp": False})
    assert ok.status_code == 200


def test_launch_rejects_sadcp_dir_without_sta(tmp_path, capsys):
    from ladcp.studio.server import main
    sub = tmp_path / "DATA"
    sub.mkdir()
    (sub / "x.STA").write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        main(["80", "--sadcp", str(tmp_path), "--no-browser"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "no .STA files directly under" in err
    assert "DATA" in err                          # the helpful subfolder hint

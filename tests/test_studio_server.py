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

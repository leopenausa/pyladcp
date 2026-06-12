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
                        "sadcp": False, "sadcp_folder": None, "sadcp_sources": []}


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


def test_solve_payload_carries_sadcp_trace(monkeypatch):
    import ladcp.qa.cli as qacli
    from ladcp.session import SadcpConfig
    fake = np.column_stack([np.arange(20.0, 120.0, 20.0), np.full(5, 0.10),
                            np.full(5, -0.05), np.full(5, 0.02)])
    monkeypatch.setattr(qacli, "_sadcp_profile", lambda *a, **k: fake)
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    st = StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry},
                     sadcp=SadcpConfig(folder="fake/sadcp"))
    c = TestClient(create_app(st))
    p = c.post("/api/station/MORIA-80/solve",
               json={"solve": {"drot": -9.878379}, "use_sadcp": True}).json()
    assert p["sadcp_bins"] == 5
    assert p["sadcp"]["z"] == [20.0, 40.0, 60.0, 80.0, 100.0]
    assert p["sadcp"]["u"] == [0.10] * 5 and p["sadcp"]["verr"] == [0.02] * 5
    # constraint off -> no trace in the payload
    off = c.post("/api/station/MORIA-80/solve",
                 json={"solve": {"drot": -9.878379}, "use_sadcp": False}).json()
    assert off["sadcp"] is None and off["sadcp_bins"] == 0


# ------------------------------------------------- SADCP source dropdown (raw | codas)

def _two_source_state(tmp_path):
    from ladcp.session import SadcpConfig
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    codas = tmp_path / "codas" / "os150nb_enr"
    (codas / "contour").mkdir(parents=True)
    (codas / "contour" / "os150nb.nc").write_bytes(b"")
    return StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry},
                       sadcp=SadcpConfig(folder="sADCP/DATA"),
                       sadcp_codas=[SadcpConfig(folder=str(codas), source="codas")])


def test_stations_payload_lists_sources(tmp_path):
    st = _two_source_state(tmp_path)
    info = TestClient(create_app(st)).get("/api/stations").json()
    assert [s["key"] for s in info["sadcp_sources"]] == ["raw", "os150nb_enr"]
    assert [s["source"] for s in info["sadcp_sources"]] == ["vmdas", "codas"]
    assert info["sadcp"] is True                  # legacy fields still describe primary
    assert info["sadcp_folder"] == "sADCP/DATA"


def test_sadcp_key_selects_the_source(tmp_path, monkeypatch):
    import ladcp.qa.cli as qacli
    seen = []
    fake = np.column_stack([np.arange(20.0, 120.0, 20.0), np.full(5, 0.10),
                            np.full(5, -0.05), np.full(5, 0.02)])
    monkeypatch.setattr(qacli, "_sadcp_profile",
                        lambda opts, *a, **k: seen.append(opts) or fake)
    st = _two_source_state(tmp_path)
    c = TestClient(create_app(st))
    drot = {"solve": {"drot": -9.878379}}
    assert c.post("/api/station/MORIA-80/solve",
                  json=dict(drot, sadcp_key="os150nb_enr")).json()["sadcp_bins"] == 5
    assert seen[-1]["source"] == "codas" and seen[-1]["folder"].endswith("os150nb_enr")
    assert c.post("/api/station/MORIA-80/solve",
                  json=dict(drot, sadcp_key="raw")).json()["sadcp_bins"] == 5
    assert seen[-1]["source"] == "vmdas" and seen[-1]["folder"] == "sADCP/DATA"
    off = c.post("/api/station/MORIA-80/solve", json=dict(drot, sadcp_key="off")).json()
    assert off["sadcp_bins"] == 0


def test_sadcp_key_round_trips_through_cli(tmp_path, monkeypatch):
    """The hard contract holds per source: each choice is one ladcp-qa invocation."""
    import ladcp.qa.cli as qacli
    fake = np.column_stack([[50.0], [0.1], [0.0], [0.02]])
    monkeypatch.setattr(qacli, "_sadcp_profile", lambda *a, **k: fake)
    st = _two_source_state(tmp_path)
    c = TestClient(create_app(st))
    p = c.post("/api/station/MORIA-80/solve",
               json={"solve": {"drot": -9.878379}, "sadcp_key": "os150nb_enr"}).json()
    args = build_parser().parse_args(shlex.split(p["cli"])[1:])
    assert args.sadcp.endswith("os150nb_enr") and args.sadcp_source == "codas"


def test_unknown_sadcp_key_is_400(tmp_path):
    st = _two_source_state(tmp_path)
    r = TestClient(create_app(st)).post("/api/station/MORIA-80/solve",
                                        json={"sadcp_key": "os75nb_enr"})
    assert r.status_code == 400
    assert "unknown SADCP source" in r.json()["detail"]


def test_legacy_use_sadcp_means_first_source(tmp_path, monkeypatch):
    import ladcp.qa.cli as qacli
    seen = []
    fake = np.column_stack([[50.0], [0.1], [0.0], [0.02]])
    monkeypatch.setattr(qacli, "_sadcp_profile",
                        lambda opts, *a, **k: seen.append(opts) or fake)
    st = _two_source_state(tmp_path)
    c = TestClient(create_app(st))
    c.post("/api/station/MORIA-80/solve",
           json={"solve": {"drot": -9.878379}, "use_sadcp": True})
    assert seen[-1]["source"] == "vmdas"


def test_codas_label_and_discovery(tmp_path):
    from ladcp.studio.server import codas_label, discover_codas_products
    assert codas_label("codas/os150nb_enr/contour/os150nb.nc") == "os150nb_enr"
    assert codas_label("codas/os150nb_enr") == "os150nb_enr"
    assert codas_label("a/flat.nc") == "a"        # bare .nc: its directory names it
    base = tmp_path / "codas"
    (base / "os75nb_sta" / "contour").mkdir(parents=True)
    (base / "os75nb_sta" / "contour" / "os75nb.nc").write_bytes(b"")
    (base / "loose.nc").write_bytes(b"")
    found = discover_codas_products(tmp_path)
    assert [codas_label(p) for p in found] == ["os75nb_sta", "codas"]
    assert discover_codas_products(tmp_path / "nowhere") == []


def test_launch_rejects_missing_codas_product(tmp_path, capsys):
    from ladcp.studio.server import main
    with pytest.raises(SystemExit) as exc:
        main(["80", "--sadcp-codas", str(tmp_path / "nope"), "--no-browser"])
    assert exc.value.code == 2
    assert "no CODAS NetCDF" in capsys.readouterr().err


# ------------------------------------------------- launch validates station ids

def test_launch_rejects_unresolvable_station(tmp_path, capsys):
    """A stray token parsed as a station id (or a typo) errors at launch with the
    discovery message -- previously a raw 500 at the first solve."""
    from ladcp.studio.server import main
    (tmp_path / "LADCP").mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["-", "--root", str(tmp_path), "--no-browser"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "station '-'" in err and "no down-looker" in err


def test_launch_missing_root_gets_a_hint(tmp_path, capsys):
    from ladcp.studio.server import main
    with pytest.raises(SystemExit) as exc:
        main(["80", "--root", str(tmp_path / "not_a_cruise"), "--no-browser"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "station '80'" in err and "pass --root <cruise folder>" in err


# ------------------------------------------------- toggle + multi-source dropdown

def test_two_raw_sources_get_folder_labels(tmp_path):
    from ladcp.session import SadcpConfig
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    st = StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry},
                     sadcp=[SadcpConfig(folder="sADCP/sadcp_75/DATA"),
                            SadcpConfig(folder="sADCP/sadcp_150/DATA")])
    info = TestClient(create_app(st)).get("/api/stations").json()
    assert [s["key"] for s in info["sadcp_sources"]] == ["sadcp_75", "sadcp_150"]
    assert all(s["origin"] == "flag" for s in info["sadcp_sources"])


def test_found_products_are_marked(tmp_path):
    from ladcp.session import SadcpConfig
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    st = StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry},
                     sadcp=SadcpConfig(folder="sADCP/DATA"),
                     sadcp_found=[SadcpConfig(folder=str(tmp_path / "os75nb_enr"),
                                              source="codas")])
    info = TestClient(create_app(st)).get("/api/stations").json()
    assert [(s["key"], s["origin"]) for s in info["sadcp_sources"]] == \
        [("raw", "flag"), ("os75nb_enr", "found")]


def test_merge_discovered_codas_dedupes_flagged(tmp_path):
    from ladcp.session import SadcpConfig
    from ladcp.studio.server import merge_discovered_codas
    base = tmp_path / "codas"
    for tree in ("os150nb_enr", "os75nb_enr"):
        (base / tree / "contour").mkdir(parents=True)
        (base / tree / "contour" / f"{tree.split('_')[0]}.nc").write_bytes(b"")
    flagged = [SadcpConfig(folder=str(base / "os150nb_enr"), source="codas")]
    merged = merge_discovered_codas(tmp_path, flagged)
    assert len(merged) == 2                       # flagged + the one genuinely new
    assert merged[0] is flagged[0]
    assert merged[1].folder.endswith("os75nb_enr")
    # no codas/ at all: flagged pass through untouched
    assert merge_discovered_codas(tmp_path / "elsewhere", flagged) == flagged


def test_raw_label_skips_generic_data_dir():
    from ladcp.studio.server import raw_label
    assert raw_label("sADCP/sadcp_75/DATA") == "sadcp_75"
    assert raw_label("sADCP/sadcp_150") == "sadcp_150"
    assert raw_label("DATA") == "raw"


def test_solve_payload_carries_dn_geometry(client):
    p = client.post("/api/station/MORIA-80/solve",
                    json={"solve": {"drot": -9.878379}}).json()
    g = p["dn_geom"]
    assert g["cell_m"] == 8.0                     # MORIA WH300 geometry
    assert 0 < g["first_m"] < 20 and g["n_bins"] > 4


def test_found_sources_are_never_the_default_constraint(tmp_path):
    """A launch with only discovered CODAS products must not silently constrain."""
    from ladcp.session import SadcpConfig
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    st = StudioState(["MORIA-80"], cruise="MORIA", explicit={"MORIA-80": entry},
                     sadcp_found=[SadcpConfig(folder=str(tmp_path), source="codas")])
    cfg = config_from_body({}, st)               # default request
    assert cfg.sadcp is None                     # offered in the dropdown, not active
    on = config_from_body({"use_sadcp": True}, st)   # explicit boolean = explicit intent
    assert on.sadcp is not None


# ---------------------------------------------------------------------------
# brush Edit view: journal CRUD, heatmaps, attach-everywhere (PR: studio-brush-edit)


@pytest.fixture(scope="module")
def estate(tmp_path_factory):
    """A state with a writable root, so journals live under <root>/.ladcp_edits."""
    root = tmp_path_factory.mktemp("edit_root")
    entry = StationEntry(label="MORIA-80", down=str(DOWN), up=str(UP), ctd=str(CTD))
    return StudioState(["MORIA-80"], root=str(root), cruise="MORIA",
                       explicit={"MORIA-80": entry})


@pytest.fixture(scope="module")
def eclient(estate):
    return TestClient(create_app(estate))


def _entries(payload):
    return payload["journal"]["entries"]


def _profile(client, body=None):
    r = client.post("/api/station/MORIA-80/solve",
                    json=dict({"solve": {"drot": -9.878379}}, **(body or {})))
    assert r.status_code == 200
    p = r.json()
    u = np.array([np.nan if x is None else x for x in p["profile"]["u"]])
    return p, u


def _clear_journal(client):
    for e in list(_entries(client.get("/api/station/MORIA-80/edits").json())):
        assert client.delete(f"/api/station/MORIA-80/edits/{e['id']}").status_code == 200


def test_edits_skeleton_then_crud(eclient, estate):
    r = eclient.get("/api/station/MORIA-80/edits")
    assert r.status_code == 200
    p = r.json()
    assert p["stale"] is None and _entries(p) == []
    assert p["path"].endswith(".ladcp_edits/MORIA-80.json")

    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 3, "bin_last": 4,
                                     "ens_first": 0, "ens_last": 10 ** 9,
                                     "view": "errvel", "note": "band"}})
    assert r.status_code == 200
    p = r.json()
    [e] = _entries(p)
    assert e["id"] == 1 and e["head"] == "down"
    assert e["ens_last"] < 10 ** 9                       # clamped to the real cast
    assert p["journal"]["joint_n_ens"] == e["ens_last"] + 1
    assert p["journal"]["raw"]["down"]["n_ens"] >= p["journal"]["joint_n_ens"]
    assert pathlib.Path(p["path"]).is_file()             # persisted (atomically)

    assert eclient.delete("/api/station/MORIA-80/edits/99").status_code == 404
    r = eclient.delete("/api/station/MORIA-80/edits/1")
    assert r.status_code == 200 and _entries(r.json()) == []


def test_solve_attaches_journal_and_delete_restores(eclient):
    _clear_journal(eclient)
    _, base_u = _profile(eclient)

    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 3, "bin_last": 4,
                                     "ens_first": 0, "ens_last": 10 ** 9}})
    assert r.status_code == 200
    eid = _entries(r.json())[0]["id"]

    p, edited_u = _profile(eclient)
    assert p["manual_edits"] == 1
    assert "--edits" in p["cli"] and ".ladcp_edits/MORIA-80.json" in p["cli"]
    assert not np.array_equal(edited_u, base_u, equal_nan=True)

    assert eclient.delete(f"/api/station/MORIA-80/edits/{eid}").status_code == 200
    p, restored_u = _profile(eclient)
    assert p["manual_edits"] == 0 and "--edits" not in p["cli"]
    np.testing.assert_array_equal(restored_u, base_u)


def test_edited_cli_roundtrips_through_from_args(eclient, estate):
    _clear_journal(eclient)
    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 3, "bin_last": 4,
                                     "ens_first": 5, "ens_last": 400}})
    assert r.status_code == 200
    p, _u = _profile(eclient)
    args = build_parser().parse_args(shlex.split(p["cli"])[1:])
    cfg = SessionConfig.from_args(args)
    assert cfg.edit.manual_flags == (("down", 3, 4, 5, 400),)
    _clear_journal(eclient)


def test_heatmap_views_cached_and_brush_does_not_invalidate(eclient):
    _clear_journal(eclient)
    pngs = {}
    for view in ("errvel", "echo"):
        r = eclient.post(f"/api/station/MORIA-80/edit/heatmap/down/{view}", json={})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:4] == b"\x89PNG"
        pngs[view] = r.content
    assert pngs["errvel"] != pngs["echo"]

    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 2, "bin_last": 2,
                                     "ens_first": 0, "ens_last": 50}})
    assert r.status_code == 200
    r = eclient.post("/api/station/MORIA-80/edit/heatmap/down/errvel", json={})
    assert r.status_code == 200 and r.content == pngs["errvel"]   # base PNG untouched
    _clear_journal(eclient)

    assert eclient.post("/api/station/MORIA-80/edit/heatmap/down/nope",
                        json={}).status_code == 404
    assert eclient.post("/api/station/MORIA-80/edit/heatmap/sideways/errvel",
                        json={}).status_code == 404
    r = eclient.post("/api/station/MORIA-80/edit/heatmap/up/errvel",
                     json={"edit": {"down_only": True}})
    assert r.status_code == 404                          # no up-looker in this config


def test_edit_meta_geometry(eclient):
    r = eclient.post("/api/station/MORIA-80/edit/meta", json={})
    assert r.status_code == 200
    m = r.json()
    assert m["joint_n_ens"] > 0
    assert m["heads"]["down"]["n_bins"] > 0 and m["heads"]["down"]["cell_m"] > 0
    assert m["heads"]["up"]["n_bins"] > 0
    assert m["journal"]["station"] == "MORIA-80"
    r = eclient.post("/api/station/MORIA-80/edit/meta",
                     json={"edit": {"down_only": True}})
    assert r.json()["heads"]["up"] is None


def test_post_clamps_to_grid_and_rejects_bad_entries(eclient):
    _clear_journal(eclient)
    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 1, "bin_last": 999,
                                     "ens_first": -5, "ens_last": 7}})
    assert r.status_code == 200
    [e] = _entries(r.json())
    meta = eclient.post("/api/station/MORIA-80/edit/meta", json={}).json()
    assert e["bin_last"] == meta["heads"]["down"]["n_bins"]
    assert e["ens_first"] == 0
    _clear_journal(eclient)

    assert eclient.post("/api/station/MORIA-80/edits",
                        json={"entry": {"head": "sideways", "bin_first": 1,
                                        "bin_last": 1, "ens_first": 0,
                                        "ens_last": 1}}).status_code == 400
    assert eclient.post("/api/station/MORIA-80/edits",
                        json={"entry": {"head": "down"}}).status_code == 400
    assert eclient.post(
        "/api/station/MORIA-80/edits",
        json={"entry": {"head": "down", "bin_first": 5, "bin_last": 3,
                        "ens_first": 0, "ens_last": 1}}).status_code == 400


def test_stale_journal_blocks_solve_and_post(eclient, estate):
    import json as _json
    _clear_journal(eclient)
    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 3, "bin_last": 3,
                                     "ens_first": 0, "ens_last": 9}})
    assert r.status_code == 200
    jp = pathlib.Path(r.json()["path"])
    doc = _json.loads(jp.read_text())
    doc["raw"]["down"]["size"] = 1                       # the raw file "changed"
    jp.write_text(_json.dumps(doc), encoding="utf-8")

    r = eclient.post("/api/station/MORIA-80/solve", json={})
    assert r.status_code == 400 and "delete or re-create" in r.json()["detail"]
    r = eclient.post("/api/station/MORIA-80/edits",
                     json={"entry": {"head": "down", "bin_first": 4, "bin_last": 4,
                                     "ens_first": 0, "ens_last": 9}})
    assert r.status_code == 400                          # never grow a stale journal
    assert eclient.get("/api/station/MORIA-80/edits").json()["stale"] is not None

    jp.unlink()                                          # the documented remedy
    assert eclient.post("/api/station/MORIA-80/solve", json={}).status_code == 200


def test_journal_uses_canonical_label_not_launch_token(tmp_path):
    """Launched as `ladcp-studio 80`, the journal must be MORIA-80.json: the emitted
    `ladcp-qa --edits` replay resolves the canonical label through discovery, and its
    station-match guard rejects a token-named journal (caught live, 2026-06-12)."""
    import shutil
    root = tmp_path / "root"
    for sub in ("LADCP", "CTD"):
        shutil.copytree(GOOD / sub, root / sub)
    st = StudioState(["80"], root=str(root), cruise="MORIA")
    c = TestClient(create_app(st))
    r = c.post("/api/station/80/edits",
               json={"entry": {"head": "down", "bin_first": 3, "bin_last": 4,
                               "ens_first": 0, "ens_last": 10 ** 9}})
    assert r.status_code == 200
    p = r.json()
    assert p["station"] == "MORIA-80"
    assert p["path"].endswith(".ladcp_edits/MORIA-80.json")
    assert (root / ".ladcp_edits" / "MORIA-80.json").is_file()
    assert p["journal"]["station"] == "MORIA-80"

    solved = c.post("/api/station/80/solve", json={}).json()
    assert solved["manual_edits"] == 1
    assert "MORIA-80.json" in solved["cli"]
    # the emitted command parses and carries the journal's geometry
    args = build_parser().parse_args(shlex.split(solved["cli"])[1:])
    assert SessionConfig.from_args(args).edit.manual_flags[0][:3] == ("down", 3, 4)

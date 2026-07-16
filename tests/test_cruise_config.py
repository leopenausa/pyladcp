"""cruise.toml: discovery, validation, precedence and the ladcp-qa merge (phase A).

The quick-lane tests drive ``ladcp-qa`` with ``discover``/``process_station``
monkeypatched away, asserting exactly what reaches the pipeline; the slow test proves
parity end-to-end on the committed MORIA-80 fixture (a config-driven run must be
byte-identical to the flag-driven run it stands in for).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ladcp.hub import cruise_config as cc
from ladcp.qa import cli

FIX = Path(__file__).resolve().parent / "fixtures" / "New_golden" / "Good"


# ---------------------------------------------------------------------------
# find_config

def test_find_config_walks_parents(tmp_path):
    (tmp_path / "cruise.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert cc.find_config(sub) == tmp_path / "cruise.toml"
    assert cc.find_config(tmp_path) == tmp_path / "cruise.toml"


def test_find_config_stops_at_git_boundary(tmp_path):
    (tmp_path / "cruise.toml").write_text("", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "src"
    sub.mkdir()
    assert cc.find_config(sub) is None          # the repo's .git fences the search
    assert cc.find_config(tmp_path) is not None


def test_find_config_none(tmp_path):
    assert cc.find_config(tmp_path) is None


# ---------------------------------------------------------------------------
# load_config: projection + validation

def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cruise.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_projects_args_and_resolves_paths(tmp_path):
    p = _write(tmp_path, """
[cruise]
name = "FDCCC"
[data]
root = "raw"
out = "qa_out"
index = "/abs/idx.json"
[ctd]
from_hex = true
[edit]
nearfield_dn_bins = [3, 4]
soundcorr = false
dzbelow = 24.0
[solve]
botfac = 0.5
[sadcp]
folder = "vmdas/DATA"
source = "vmdas"
timeoff = "auto"
nav = "nav/track.csv"
""")
    cfg = cc.load_config(p)
    m = cfg.args_map
    assert m["cruise"] == "FDCCC"
    assert m["root"] == str(tmp_path / "raw")                  # relative to the config dir
    assert m["outdir"] == str(tmp_path / "qa_out")
    assert m["index"] == "/abs/idx.json"                       # absolute stays absolute
    assert m["from_hex"] is True
    assert m["nearfield_dn_bins"] == "3,4"                     # TOML list -> CLI string
    assert m["no_soundcorr"] is True                           # soundcorr=false inverts
    assert m["dzbelow"] == 24.0
    assert m["botfac"] == 0.5
    assert m["sadcp"] == str(tmp_path / "vmdas" / "DATA")
    assert m["sadcp_timeoff"] == "auto"
    assert m["sadcp_nav"] == str(tmp_path / "nav" / "track.csv")


def test_load_params_global_and_station(tmp_path):
    p = _write(tmp_path, """
[params]
pglim = 30.0
edit_mask_dn_bins = [1, 2]
[params.MORIA-07]
zbottom = 3850.0
pglim = 40.0
""")
    cfg = cc.load_config(p)
    assert cfg.params_global == {"pglim": 30.0, "edit_mask_dn_bins": (1, 2)}
    assert cfg.params_station == {"MORIA-07": {"zbottom": 3850.0, "pglim": 40.0}}
    merged = cc.station_params(cfg, "MORIA-07")
    assert merged == {"pglim": 40.0, "edit_mask_dn_bins": (1, 2), "zbottom": 3850.0}
    assert cc.station_params(cfg, "MORIA-08") == {"pglim": 30.0,
                                                  "edit_mask_dn_bins": (1, 2)}


@pytest.mark.parametrize("text,needle", [
    ("[bogus]\nx = 1\n", "unknown table"),
    ("[solve]\nbogus = 1\n", "unknown key"),
    ("[solve]\nsolver = 'magic'\n", "expected one of"),
    ("[sadcp]\nsource = 'vmdas'\n", "'folder' is required"),
    ("[sadcp]\nfolder = 'x'\nsource = 'sonar'\n", "expected one of"),
    ("[params]\nbogus_knob = 1\n", "not a CastParams field"),
    ("[params.MORIA-07]\nbogus_knob = 1\n", "not a CastParams field"),
    ("[params]\nstation = 'X'\n", "not a CastParams field"),
    ("[edit]\ndzbelow = 'deep'\n", "expected a number"),
    ("[ctd]\nfrom_hex = 'yes'\n", "expected true/false"),
    ("[edit]\nnearfield_dn_bins = 'both'\n", "expected a list"),
    ("[sadcp]\nfolder = 'x'\ntimeoff = 'later'\n", "seconds or 'auto'"),
    ("solve = 3\n", "expected a table"),
    ("x = \n", "invalid TOML"),
])
def test_load_rejects_with_named_offender(tmp_path, text, needle):
    p = _write(tmp_path, text)
    with pytest.raises(cc.ConfigError) as ei:
        cc.load_config(p)
    assert needle in str(ei.value)
    assert "cruise.toml" in str(ei.value)


def test_sadcp_array_uses_first(tmp_path):
    p = _write(tmp_path, """
[[sadcp]]
folder = "codas_run"
source = "codas"
[[sadcp]]
folder = "vmdas/DATA"
""")
    cfg = cc.load_config(p)
    assert cfg.n_sadcp == 2
    assert cfg.args_map["sadcp"] == str(tmp_path / "codas_run")
    assert cfg.args_map["sadcp_source"] == "codas"


# ---------------------------------------------------------------------------
# save_config round-trip

def test_save_round_trips_losslessly(tmp_path):
    data = {
        "cruise": {"name": "MORIA2"},
        "data": {"root": ".", "out": "qa_out"},
        "solve": {"botfac": 0.5, "solver": "inverse"},
        "sadcp": [{"folder": "codas", "source": "codas"}],
        "params": {"pglim": 30.0, "MORIA-07": {"zbottom": 3850.0}},
    }
    p = cc.save_config(data, tmp_path / "cruise.toml")
    assert cc.load_config(p).raw == data
    assert not list(tmp_path.glob("*.tmp"))     # atomic write left no temp file behind


def test_save_refuses_invalid(tmp_path):
    with pytest.raises(cc.ConfigError):
        cc.save_config({"solve": {"bogus": 1}}, tmp_path / "cruise.toml")
    assert not (tmp_path / "cruise.toml").exists()


# ---------------------------------------------------------------------------
# precedence: explicit flags > cruise.toml > parser defaults

def test_explicit_dests_distinguishes_typed_from_default():
    ex = cc.explicit_dests(cli.build_parser, ["80", "--botfac", "1.0", "--down-only"])
    assert {"stations", "botfac", "down_only"} <= ex
    assert "barofac" not in ex and "root" not in ex


def _merged_args(tmp_path, toml_text, argv):
    p = _write(tmp_path, toml_text)
    ap = cli.build_parser()
    args = ap.parse_args(argv)
    cfg = cc.load_config(p)
    cc.apply_to_args(cfg, args, cc.explicit_dests(cli.build_parser, argv))
    return args


def test_toml_overrides_parser_default(tmp_path):
    args = _merged_args(tmp_path, "[solve]\nbotfac = 0.5\n", ["80"])
    assert args.botfac == 0.5


def test_typed_flag_beats_toml_even_at_default_value(tmp_path):
    args = _merged_args(tmp_path, "[solve]\nbotfac = 0.5\n", ["80", "--botfac", "1.0"])
    assert args.botfac == 1.0


# ---------------------------------------------------------------------------
# ladcp-qa merge end-to-end (pipeline monkeypatched away; quick lane)

@pytest.fixture()
def captured(monkeypatch):
    """Run cli.main with discover/process_station faked; capture what each station gets."""
    from ladcp.discovery import StationFiles
    calls: dict[str, dict] = {}

    def fake_discover(item, *, root, cruise, index=None, from_hex=False, ctd_cache=None):
        return StationFiles(down=Path(f"{item}-M.000"), up=None, ctd=None,
                            label=f"X-{item}", ctd_utc=None)

    def fake_process(down, up, ctd_path, station, outdir, make_plots, cfg, cruise="LADCP",
                     formats=None, ctd_utc=None, edits=None, hint_root=None,
                     param_overrides=None):
        calls[station] = dict(cfg=cfg, cruise=cruise, outdir=outdir,
                              param_overrides=param_overrides)
        return "ok", None

    monkeypatch.setattr(cli, "discover", fake_discover)
    monkeypatch.setattr(cli, "process_station", fake_process)
    return calls


def test_cli_runs_from_config(tmp_path, captured):
    p = _write(tmp_path, """
[cruise]
name = "FDCCC"
[data]
out = "qa_out"
[edit]
soundcorr = false
nearfield_dn_bins = [3, 4]
[solve]
botfac = 0.5
[params]
pglim = 30.0
[params.X-07]
zbottom = 120.0
""")
    rc = cli.main(["07", "08", "--config", str(p), "--no-log", "--no-progress"])
    assert rc == 0
    got = captured["X-07"]
    assert got["cruise"] == "FDCCC"
    assert got["outdir"] == str(tmp_path / "qa_out")           # resolved vs the config dir
    assert got["cfg"].solve.botfac == 0.5
    assert got["cfg"].edit.soundcorr is False
    assert got["cfg"].edit.nearfield_dn_bins == (3, 4)
    assert got["param_overrides"] == {"pglim": 30.0, "zbottom": 120.0}
    assert captured["X-08"]["param_overrides"] == {"pglim": 30.0}


def test_cli_flag_overrides_config(tmp_path, captured):
    p = _write(tmp_path, "[solve]\nbotfac = 0.5\n[cruise]\nname = 'FDCCC'\n")
    rc = cli.main(["07", "--config", str(p), "--botfac", "2.0", "--cruise", "MORIA",
                   "--out", str(tmp_path / "flagged"), "--no-log", "--no-progress"])
    assert rc == 0
    got = captured["X-07"]
    assert got["cfg"].solve.botfac == 2.0
    assert got["cruise"] == "MORIA"
    assert got["outdir"] == str(tmp_path / "flagged")


def test_cli_autodiscovers_and_no_config_opts_out(tmp_path, captured, monkeypatch):
    _write(tmp_path, "[solve]\nbotfac = 0.5\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["07", "--no-log", "--no-progress"]) == 0
    assert captured["X-07"]["cfg"].solve.botfac == 0.5
    captured.clear()
    assert cli.main(["07", "--no-config", "--no-log", "--no-progress"]) == 0
    assert captured["X-07"]["cfg"].solve.botfac == 1.0
    assert captured["X-07"]["param_overrides"] is None


def test_cli_rejects_bad_config(tmp_path, capsys):
    p = _write(tmp_path, "[solve]\nbogus = 1\n")
    with pytest.raises(SystemExit) as ei:
        cli.main(["07", "--config", str(p)])
    assert ei.value.code == 2
    assert "unknown key" in capsys.readouterr().err


def test_cli_config_missing_file(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["07", "--config", str(tmp_path / "nope.toml")])
    assert "does not exist" in capsys.readouterr().err


def test_cli_config_and_no_config_conflict(tmp_path, capsys):
    p = _write(tmp_path, "")
    with pytest.raises(SystemExit):
        cli.main(["07", "--config", str(p), "--no-config"])
    assert "mutually exclusive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# full parity on the committed fixture: config-driven == flag-driven, byte for byte

@pytest.mark.slow
@pytest.mark.skipif(not FIX.exists(), reason="MORIA New_golden fixture not present")
def test_config_run_matches_flag_run(tmp_path):
    toml = tmp_path / "cruise.toml"
    toml.write_text(f"""
[cruise]
name = "MORIA"
[data]
root = {str(FIX)!r}
out = "from_config"
[params."MORIA-80"]
dzbelow = 24.0
""", encoding="utf-8")
    rc = cli.main(["80", "--config", str(toml), "--no-plots", "--no-export",
                   "--no-log", "--no-progress"])
    assert rc == 0
    rc = cli.main(["80", "--no-config", "--root", str(FIX), "--cruise", "MORIA",
                   "--dzbelow", "24", "--out", str(tmp_path / "from_flags"),
                   "--no-plots", "--no-export", "--no-log", "--no-progress"])
    assert rc == 0
    for suffix in (".lad", ".bot", "_qa.txt"):
        a = tmp_path / "from_config" / "stations" / "MORIA-80" / f"MORIA-80{suffix}"
        b = tmp_path / "from_flags" / "stations" / "MORIA-80" / f"MORIA-80{suffix}"
        assert a.exists() and b.exists(), f"missing MORIA-80{suffix}"
        assert a.read_bytes() == b.read_bytes(), f"MORIA-80{suffix}: config vs flags differ"

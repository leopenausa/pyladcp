"""``ladcp init`` + the detection engine (phase C).

Quick lane: detection proposals on a synthetic filename tree (detect never reads
file contents, so empty files are enough), ``init --yes`` parity, the scripted
interactive driver, and the curated station-universe fallback. Slow lane: the plan's
exit criterion verbatim — ``ladcp init --yes`` on an ``ladcp-synth`` cruise, then
``ladcp process`` solves it for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ladcp.hub import cli as hub
from ladcp.hub import cruise_config as cc
from ladcp.hub import detect

# ---------------------------------------------------------------------------
# a filename-only cruise tree (detection is pure globbing)

def _tree(tmp_path: Path, *, hex_files: bool = False, sadcp: bool = True) -> Path:
    (tmp_path / "LADCP").mkdir()
    for name in ("A-01-LADCP-M.000", "A-01-LADCP-S.000", "A-02-LADCP-M.000"):
        (tmp_path / "LADCP" / name).write_bytes(b"")
    (tmp_path / "CTD").mkdir()
    (tmp_path / "CTD" / "a-01_clean.cnv").write_text("", encoding="utf-8")
    if hex_files:
        (tmp_path / "CTD" / "a-02.hex").write_text("", encoding="utf-8")
    if sadcp:
        d = tmp_path / "sADCP" / "DATA"
        d.mkdir(parents=True)
        (d / "x1.STA").write_bytes(b"")
        (d / "x2.STA").write_bytes(b"")
        nc = tmp_path / "codas" / "os150_enr" / "contour"
        nc.mkdir(parents=True)
        (nc / "os150.nc").write_bytes(b"")
        ek = tmp_path / "EK80_run"
        ek.mkdir()
        (ek / "d.nc").write_bytes(b"")
        nav = tmp_path / "nav"
        nav.mkdir()
        (nav / "ship.posicion.txt").write_text("", encoding="utf-8")
    out = tmp_path / "qa_out" / "stations"          # must be ignored by the scan
    out.mkdir(parents=True)
    (out / "stale.cnv").write_text("", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# detection

def test_detect_full_tree(tmp_path):
    det = detect.detect(_tree(tmp_path, hex_files=True))
    assert det.ladcp.layout == "curated" and det.ladcp.dir == "LADCP"
    labels = [s.label for s in det.ladcp.stations]
    assert labels == ["A-01", "A-02"]
    assert det.ladcp.stations[0].slave is not None
    assert det.ladcp.stations[1].slave is None           # single-head cast
    assert det.ctd.dir == "CTD" and det.ctd.n_cnv == 1 and det.ctd.n_hex == 1
    assert det.ctd.missing_cnv == ("A-02",)
    kinds = {(c.source, c.path) for c in det.sadcp}
    assert ("vmdas", str(Path("sADCP") / "DATA")) in kinds
    assert ("codas", str(Path("codas") / "os150_enr")) in kinds
    assert ("ek80", "EK80_run") in kinds
    assert det.nav and det.nav[0].path == "nav"
    assert not det.preset                                # tmp dir name is no preset


def test_detect_empty_tree(tmp_path):
    det = detect.detect(tmp_path)
    assert det.ladcp.layout == "none" and det.ladcp.dir is None
    assert det.ctd.dir is None and det.sadcp == () and det.nav == ()


def test_detect_master_slave_layout(tmp_path):
    for sub in ("MASTER", "SLAVE"):
        d = tmp_path / "LADCP" / sub
        d.mkdir(parents=True)
        (d / "m0001.000").write_bytes(b"")
    det = detect.detect(tmp_path)
    assert det.ladcp.layout == "master-slave"
    assert det.ladcp.n_down == 1 and det.ladcp.n_up == 1
    assert det.ladcp.stations == ()                      # needs the index (time pairing)


def test_curated_labels_only_from_standard_dir(tmp_path):
    _tree(tmp_path)
    assert detect.curated_station_labels(tmp_path) == ["A-01", "A-02"]
    other = tmp_path / "elsewhere"
    (tmp_path / "LADCP").rename(other)                   # discover() could not glob this
    assert detect.curated_station_labels(tmp_path) == []


# ---------------------------------------------------------------------------
# init --yes (non-interactive parity)

def test_init_yes_writes_valid_config(tmp_path, monkeypatch, capsys):
    _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert hub.main(["init", "--yes", "--name", "SYNC", "--no-trial"]) == 0
    cfg = cc.load_config(tmp_path / "cruise.toml")
    assert cfg.args_map["cruise"] == "SYNC"
    assert "sadcp" not in cfg.args_map                   # --yes never auto-picks a source
    out = capsys.readouterr().out
    assert "never auto-picks" in out and "sADCP" in out  # ...but it lists what it found
    assert "single-head cast(s)" in out


def test_init_yes_with_sadcp_and_nav_flags(tmp_path, monkeypatch):
    _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert hub.main(["init", "--yes", "--no-trial", "--sadcp", "sADCP/DATA",
                     "--nav", "nav"]) == 0
    cfg = cc.load_config(tmp_path / "cruise.toml")
    assert cfg.args_map["sadcp"].endswith(str(Path("sADCP") / "DATA"))
    assert cfg.args_map["sadcp_source"] == "vmdas"
    assert cfg.args_map["sadcp_timeoff"] == "auto"
    assert cfg.args_map["sadcp_nav"].endswith("nav")


def test_init_yes_from_hex_flags_are_tristate(tmp_path, monkeypatch):
    _tree(tmp_path, hex_files=True)
    monkeypatch.chdir(tmp_path)
    assert hub.main(["init", "--yes", "--no-trial", "--no-from-hex"]) == 0
    assert "from_hex" not in cc.load_config(tmp_path / "cruise.toml").args_map
    assert hub.main(["init", "--yes", "--force", "--no-trial", "--from-hex"]) == 0
    assert cc.load_config(tmp_path / "cruise.toml").args_map["from_hex"] is True


def test_init_offers_from_hex_on_master_slave_hex_only_cruise(tmp_path, monkeypatch):
    """The MORIA_2 shape: MASTER/SLAVE archive + raw .hex only (no .cnv anywhere).

    No name-paired stations exist, so missing_cnv is empty — the from-hex offer
    must key off n_cnv == 0 instead (regression: the offer never appeared).
    """
    for sub in ("MASTER", "SLAVE"):
        d = tmp_path / "LADCP" / sub
        d.mkdir(parents=True)
        (d / "m0001.000").write_bytes(b"")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "cast01.hex").write_text("", encoding="utf-8")
    from ladcp.io import ctd_raw
    monkeypatch.setattr(ctd_raw, "_find_ctd_project", lambda: tmp_path / "CTD_project")
    monkeypatch.chdir(tmp_path)
    # --yes auto-enables from-hex when it is needed and the converter is available
    assert hub.main(["init", "--yes", "--no-trial"]) == 0
    assert cc.load_config(tmp_path / "cruise.toml").args_map["from_hex"] is True


def test_init_refuses_existing_config_without_force(tmp_path, monkeypatch, capsys):
    _tree(tmp_path)
    (tmp_path / "cruise.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert hub.main(["init", "--yes", "--no-trial"]) == 1
    assert "already exists" in capsys.readouterr().out
    assert (tmp_path / "cruise.toml").read_text(encoding="utf-8") == ""
    assert hub.main(["init", "--yes", "--force", "--no-trial"]) == 0
    assert (tmp_path / "cruise.toml").read_text(encoding="utf-8") != ""


def test_init_no_pd0s_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert hub.main(["init", "--yes"]) == 1
    assert "no *.000 PD0 files" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the interactive driver (scripted answers through the injectable ask/say)

def _run_scripted(root: Path, answers: list[str], argv: list[str] | None = None):
    from ladcp.hub.init_flow import run_init
    ns = hub.build_parser().parse_args(["init", "--root", str(root), *(argv or [])])
    it = iter(answers)
    said: list[str] = []
    rc = run_init(ns, ask=lambda prompt="": next(it), say=said.append)
    return rc, said


def test_interactive_picks_sadcp_source(tmp_path):
    root = _tree(tmp_path)
    #        ladcp  ctd  sadcp  nav   name     write  trial
    answers = ["y", "y", "1",   "0",  "CRUZ",  "y",   "n"]
    rc, said = _run_scripted(root, answers)
    assert rc == 0
    cfg = cc.load_config(root / "cruise.toml")
    assert cfg.args_map["cruise"] == "CRUZ"
    assert cfg.args_map["sadcp_source"] == "vmdas"
    assert "sadcp_timeoff" not in cfg.args_map           # nav declined with 0
    assert any("cruise.toml to be written" in s for s in said)


def test_interactive_decline_write_leaves_nothing(tmp_path):
    root = _tree(tmp_path, sadcp=False)
    answers = ["y", "y", "CRUZ", "n"]                    # decline at the write gate
    rc, said = _run_scripted(root, answers)
    assert rc == 1
    assert not (root / "cruise.toml").exists()
    assert any("nothing written" in s for s in said)


def test_interactive_trial_offer_runs_hub_process(tmp_path, monkeypatch):
    root = _tree(tmp_path, sadcp=False)
    ran: list[list[str]] = []
    monkeypatch.setattr(hub, "_cmd_process",
                        lambda ns: ran.append(list(ns.stations)) or 0)
    answers = ["y", "y", "CRUZ", "y", "y", "y"]          # trial yes, batch yes
    rc, _ = _run_scripted(root, answers)
    assert rc == 0
    assert ran == [["A-02"], []]                         # mid-cruise trial, then --new


# ---------------------------------------------------------------------------
# curated universe fallback: process --new on an index-less cruise

def test_process_universe_falls_back_to_names(tmp_path, monkeypatch):
    root = _tree(tmp_path, sadcp=False)
    (root / "cruise.toml").write_text('[cruise]\nname = "A"\n[data]\nroot = "."\n',
                                      encoding="utf-8")
    monkeypatch.chdir(root)
    from ladcp.discovery import StationFiles
    from ladcp.hub import freshness
    from ladcp.qa import batch
    calls: list[str] = []

    def fake_discover(st, *, root, cruise="LADCP", index=None, from_hex=False,
                      ctd_cache=None, ctd_dir=None):
        return StationFiles(down=Path(root) / "LADCP" / f"{st}-LADCP-M.000", up=None,
                            ctd=None, label=st, ctd_utc=None)

    def fake_process(down, up, ctd_path, station, outdir, make_plots, cfg, **kw):
        calls.append(station)
        d = Path(outdir) / "stations" / station
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{station}_qa.json").write_text("{}", encoding="utf-8")
        return "ok", None

    monkeypatch.setattr(batch, "discover", fake_discover)
    monkeypatch.setattr(batch, "process_station", fake_process)
    monkeypatch.setattr(freshness, "discover", fake_discover)
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    assert calls == ["A-01", "A-02"]                     # no index anywhere, names only


# ---------------------------------------------------------------------------
# the plan's exit criterion, verbatim (real synth data, real solve)

@pytest.mark.slow
def test_init_then_process_synth_cruise(tmp_path, monkeypatch, capsys):
    from ladcp.synth.cli import main as synth_main
    assert synth_main(["--out", str(tmp_path), "--station", "SYN-01", "--seed", "7"]) in (0, None)
    monkeypatch.chdir(tmp_path)
    assert hub.main(["init", "--yes", "--no-trial"]) == 0
    assert hub.main(["process", "--no-plots", "--no-log", "--no-progress"]) == 0
    assert (tmp_path / "qa_out" / "stations" / "SYN-01" / "SYN-01_qa.json").is_file()
    capsys.readouterr()
    assert hub.main(["process", "--no-log", "--no-progress"]) == 0
    assert "nothing to do" in capsys.readouterr().out    # freshness: solved once, done

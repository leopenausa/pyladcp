"""SessionConfig <-> ladcp-qa round-trip contract (Studio PR 1).

The hard contract behind the Studio GUI: every configuration is expressible as a
``ladcp-qa`` command line, and parsing that command line back recovers the identical
configuration. These tests also pin the ``inv_opts``/``sadcp_opts`` dicts to the exact
shapes ``_run_one`` consumed before the config layer existed (no behavior change).
"""

from __future__ import annotations

import shlex

import pytest

from ladcp.qa.cli import build_parser
from ladcp.session import (
    EditConfig,
    SadcpConfig,
    SessionConfig,
    SolveConfig,
    parse_nearfield,
    parse_timeoff,
)


def roundtrip(cfg: SessionConfig, station: str = "80", **ctx) -> SessionConfig:
    """cfg -> command line -> argparse -> cfg again."""
    cli = cfg.to_cli(station, **ctx)
    argv = shlex.split(cli)[1:]                  # drop the 'ladcp-qa' prog name
    args = build_parser().parse_args(argv)
    assert args.stations == [station]
    return SessionConfig.from_args(args)


# ---------------------------------------------------------------- defaults

def test_default_config_matches_argparse_defaults():
    args = build_parser().parse_args(["80"])
    assert SessionConfig.from_args(args) == SessionConfig()


def test_default_to_cli_is_bare():
    assert SessionConfig().to_cli("80") == "ladcp-qa 80"


def test_discovery_context_is_emitted():
    cli = SessionConfig().to_cli("t1-02", root="raw", cruise="FDCCC1",
                                 index="idx.json", outdir="qa")
    assert cli == ("ladcp-qa t1-02 --root raw --cruise FDCCC1 "
                   "--index idx.json --out qa")


# ---------------------------------------------------------------- round-trip battery

CONFIGS = [
    SessionConfig(),
    SessionConfig(solve=SolveConfig(solver="shear")),
    SessionConfig(solve=SolveConfig(drot=-5.44, botfac=0.5, barofac=2.0, smoofac=0.1)),
    SessionConfig(edit=EditConfig(down_only=True)),
    SessionConfig(edit=EditConfig(nearfield_dn_bins=(3, 4), dzbelow=24.0)),
    SessionConfig(edit=EditConfig(nearfield_dn_bins=())),          # explicit 'none'
    SessionConfig(edit=EditConfig(soundcorr=False)),               # --no-soundcorr
    SessionConfig(sadcp=SadcpConfig(folder="sADCP/OS150")),
    SessionConfig(sadcp=SadcpConfig(folder="sADCP/OS150", source="codas",
                                    filetype="LTA", xducer=7.0, timeoff=42.5,
                                    nav="nav/posicion", reingest=True),
                  solve=SolveConfig(sadcpfac=4.5)),
    SessionConfig(sadcp=SadcpConfig(folder="sADCP dir/OS75", timeoff="auto",
                                    nav="nav track.csv")),         # paths with spaces
    SessionConfig(edit=EditConfig(down_only=True, nearfield_dn_bins=(2,), dzbelow=32.0),
                  sadcp=SadcpConfig(folder="s", source="codas", timeoff=-3600.0),
                  solve=SolveConfig(solver="inverse", drot=1.8, botfac=0.0,
                                    barofac=0.0, smoofac=0.2, sadcpfac=0.5)),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=range(len(CONFIGS)))
def test_roundtrip(cfg):
    assert roundtrip(cfg) == cfg


def test_roundtrip_with_context():
    cfg = CONFIGS[7]
    assert roundtrip(cfg, station="MORIA-80", root="New_golden/Good",
                     cruise="MORIA", index="i.json", outdir="out") == cfg


# ---------------------------------------------------------------- opts-dict parity
# Pinned to the exact dicts main() built inline before the config layer (cli.py history).

def test_inv_opts_shape_default():
    assert SessionConfig().inv_opts() == {
        "botfac": 1.0, "barofac": 1.0, "smoofac": 0.0,
        "down_only": False, "nearfield_dn_bins": None, "dzbelow": None,
        "soundcorr": True}


def test_inv_opts_from_args():
    args = build_parser().parse_args(
        ["80", "--botfac", "0.5", "--smoofac", "0.1", "--down-only",
         "--nearfield-dn-bins", "3,4", "--dzbelow", "24"])
    assert SessionConfig.from_args(args).inv_opts() == {
        "botfac": 0.5, "barofac": 1.0, "smoofac": 0.1,
        "down_only": True, "nearfield_dn_bins": (3, 4), "dzbelow": 24.0,
        "soundcorr": True}


def test_sadcp_opts_none_without_sadcp():
    assert SessionConfig().sadcp_opts() is None


def test_sadcp_opts_from_args():
    args = build_parser().parse_args(
        ["80", "--sadcp", "sADCP", "--sadcp-source", "codas", "--sadcpfac", "4",
         "--sadcp-filetype", "LTA", "--sadcp-xducer", "7", "--sadcp-reingest",
         "--sadcp-timeoff", "12.5", "--sadcp-nav", "nav.csv"])
    assert SessionConfig.from_args(args).sadcp_opts() == {
        "folder": "sADCP", "source": "codas", "fac": 4.0, "file_type": "LTA",
        "xducer": 7.0, "reingest": True, "timeoff": 12.5, "nav": "nav.csv"}


# ---------------------------------------------------------------- validation parity
# Same error text the CLI emitted when the checks were inline in main().

def test_nearfield_parse():
    assert parse_nearfield("3,4") == (3, 4)
    assert parse_nearfield(" 3 , 4 ") == (3, 4)
    assert parse_nearfield("none") == ()
    assert parse_nearfield("NONE") == ()
    assert parse_nearfield("") == ()


def test_nearfield_parse_error_text():
    with pytest.raises(ValueError, match=r"--nearfield-dn-bins: expected comma-separated "
                                         r"bin numbers or 'none', got 'x,y'"):
        parse_nearfield("x,y")


def test_timeoff_parse():
    assert parse_timeoff(None) is None
    assert parse_timeoff("auto") == "auto"
    assert parse_timeoff("12.5") == 12.5
    assert parse_timeoff("-3600") == -3600.0


def test_timeoff_parse_error_text():
    with pytest.raises(ValueError, match=r"--sadcp-timeoff: expected seconds or 'auto', "
                                         r"got 'soon'"):
        parse_timeoff("soon")


def test_timeoff_auto_requires_nav():
    with pytest.raises(ValueError, match="--sadcp-timeoff auto needs --sadcp-nav"):
        SadcpConfig(folder="s", timeoff="auto")


def test_cli_maps_config_errors_to_argparse_exit(capsys):
    with pytest.raises(SystemExit) as exc:
        from ladcp.qa.cli import main
        main(["80", "--nearfield-dn-bins", "x,y", "--no-log"])
    assert exc.value.code == 2
    assert "--nearfield-dn-bins: expected" in capsys.readouterr().err


# ---------------------------------------------------------------- hashability (cache keys)

def test_configs_are_hashable_cache_keys():
    a = SessionConfig(edit=EditConfig(nearfield_dn_bins=(3, 4)))
    b = SessionConfig(edit=EditConfig(nearfield_dn_bins=(3, 4)))
    assert hash(a) == hash(b) and a == b
    assert len({a.edit, b.edit, EditConfig()}) == 2


# ---------------------------------------------------------------- launch-time folder check

def test_validate_folder_missing_dir(tmp_path):
    cfg = SadcpConfig(folder=str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="is not a directory"):
        cfg.validate_folder()


def test_validate_folder_no_sta_files_hints_subfolder(tmp_path):
    (tmp_path / "DATA").mkdir()
    (tmp_path / "DATA" / "x.STA").write_bytes(b"")
    with pytest.raises(ValueError, match=r"no \.STA files directly under .* "
                                         r"\(not searched recursively\).*DATA"):
        SadcpConfig(folder=str(tmp_path)).validate_folder()


def test_validate_folder_accepts_sta_files(tmp_path):
    (tmp_path / "a.STA").write_bytes(b"")
    SadcpConfig(folder=str(tmp_path)).validate_folder()       # no raise


def test_validate_folder_cache_counts_unless_reingest(tmp_path):
    from ladcp.io.sadcp_vmdas import CACHE_NAME
    (tmp_path / CACHE_NAME).write_bytes(b"")
    SadcpConfig(folder=str(tmp_path)).validate_folder()       # cache satisfies ingest
    with pytest.raises(ValueError, match=r"no \.STA files directly under"):
        SadcpConfig(folder=str(tmp_path), reingest=True).validate_folder()


def test_validate_folder_codas_existence_only(tmp_path):
    nc = tmp_path / "contour.nc"
    with pytest.raises(ValueError, match="does not exist"):
        SadcpConfig(folder=str(nc), source="codas").validate_folder()
    nc.write_bytes(b"")
    SadcpConfig(folder=str(nc), source="codas").validate_folder()


def test_qa_cli_rejects_bad_sadcp_at_launch(tmp_path, capsys):
    sub = tmp_path / "DATA"
    sub.mkdir()
    (sub / "x.STA").write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        from ladcp.qa.cli import main
        main(["80", "--sadcp", str(tmp_path), "--no-log"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "no .STA files directly under" in err
    assert "DATA" in err                          # the helpful subfolder hint

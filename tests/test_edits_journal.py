"""Manual edit journals (ladcp.edits) + the rectangle mask + session caching.

The journal is the single source of truth for brush edits: geometry-only tuples
key the solve cache (notes never invalidate), loaders refuse anything they do
not fully understand, and a full-cast rectangle over the near-field bins must be
bit-identical to ``--nearfield-dn-bins`` (both are members of the same
``_edit_velocity_mask`` family).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.edits import (
    journal_path,
    load_journal,
    manual_flags,
    new_journal,
    resolve_edits_arg,
    save_journal,
    verify_journal,
)
from ladcp.session import EditConfig, SessionConfig

pytestmark = pytest.mark.slow

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

needs_fixtures = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


def _entry(eid=1, head="down", b0=3, b1=4, e0=0, e1=999, note="band"):
    return {"id": eid, "kind": "rect", "head": head, "bin_first": b0, "bin_last": b1,
            "ens_first": e0, "ens_last": e1, "view": "errvel", "note": note,
            "created": "2026-06-12T00:00:00Z"}


def _journal(tmp_path, entries, station="MORIA-80", **extra) -> pathlib.Path:
    j = new_journal(station)
    j.entries = entries
    j.next_id = len(entries) + 1
    for k, v in extra.items():
        setattr(j, k, v)
    p = tmp_path / f"{station}.json"
    save_journal(j, p)
    return p


# --------------------------------------------------------------------------- #
# journal round-trip + atomicity + canonical geometry
# --------------------------------------------------------------------------- #
def test_round_trip(tmp_path):
    p = _journal(tmp_path, [_entry(), _entry(eid=2, head="up", b0=1, b1=1, note="x")],
                 raw={"down": {"file": "M.000", "size": 10, "n_ens": 100}},
                 joint_n_ens=100)
    j = load_journal(p)
    assert j.station == "MORIA-80" and j.n_entries == 2 and j.next_id == 3
    assert j.raw["down"]["size"] == 10 and j.joint_n_ens == 100
    assert j.entries[1]["head"] == "up"


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = _journal(tmp_path, [_entry()])
    assert p.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_manual_flags_geometry_only():
    j = new_journal("X")
    j.entries = [_entry(note="a"), _entry(eid=2, note="b"),          # duplicate geometry
                 _entry(eid=3, head="up", b0=1, b1=2, e0=5, e1=9)]
    flags = manual_flags(j)
    assert flags == (("down", 3, 4, 0, 999), ("up", 1, 2, 5, 9))     # sorted + deduped
    j.entries[0]["note"] = "edited note"                             # metadata only
    assert manual_flags(j) == flags


def test_journal_path_layout():
    assert journal_path("/r", "MORIA-80") == pathlib.Path("/r/.ladcp_edits/MORIA-80.json")


# --------------------------------------------------------------------------- #
# loader refusals: silently skipping edits would change science
# --------------------------------------------------------------------------- #
def test_rejects_newer_version(tmp_path):
    p = _journal(tmp_path, [_entry()], version=2)
    with pytest.raises(ValueError, match="version 2"):
        load_journal(p)


def test_rejects_unknown_kind(tmp_path):
    e = _entry()
    e["kind"] = "cond"
    p = _journal(tmp_path, [e])
    with pytest.raises(ValueError, match="kind 'cond'"):
        load_journal(p)


@pytest.mark.parametrize("field,value", [("head", "sideways"), ("bin_first", "3"),
                                         ("ens_last", None), ("bin_last", True)])
def test_rejects_bad_entry_fields(tmp_path, field, value):
    e = _entry()
    e[field] = value
    p = _journal(tmp_path, [e])
    with pytest.raises(ValueError):
        load_journal(p)


def test_rejects_empty_rectangle(tmp_path):
    p = _journal(tmp_path, [_entry(b0=4, b1=3)])
    with pytest.raises(ValueError, match="empty rectangle"):
        load_journal(p)


def test_rejects_garbage_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_journal(p)


# --------------------------------------------------------------------------- #
# staleness guard
# --------------------------------------------------------------------------- #
def test_verify_size_mismatch(tmp_path):
    raw = tmp_path / "M.000"
    raw.write_bytes(b"x" * 64)
    j = new_journal("X")
    j.raw = {"down": {"file": "M.000", "size": 63, "n_ens": 1}}
    with pytest.raises(ValueError, match=r"size 63 -> 64"):
        verify_journal(j, "j.json", raw)


def test_verify_basename_mismatch(tmp_path):
    raw = tmp_path / "OTHER.000"
    raw.write_bytes(b"x")
    j = new_journal("X")
    j.raw = {"down": {"file": "M.000", "size": 1}}
    with pytest.raises(ValueError, match="'M.000' -> 'OTHER.000'"):
        verify_journal(j, "j.json", raw)


def test_verify_ok_and_absent_fingerprints(tmp_path):
    raw = tmp_path / "M.000"
    raw.write_bytes(b"x" * 5)
    j = new_journal("X")
    j.raw = {"down": {"file": "M.000", "size": 5}}
    verify_journal(j, "j.json", raw)            # matching -> no raise
    j.raw = {}
    verify_journal(j, "j.json", raw)            # hand-written journal -> OK


# --------------------------------------------------------------------------- #
# --edits argument resolution
# --------------------------------------------------------------------------- #
def test_resolve_file_and_dir(tmp_path):
    p = _journal(tmp_path, [_entry()])
    assert resolve_edits_arg(str(p), "ANY") == p
    assert resolve_edits_arg(str(tmp_path), "MORIA-80") == p
    assert resolve_edits_arg(str(tmp_path), "MORIA-79") is None     # no journal = no edits
    assert resolve_edits_arg(None, "MORIA-80") is None
    with pytest.raises(ValueError, match="does not exist"):
        resolve_edits_arg(str(tmp_path / "nope"), "MORIA-80")


# --------------------------------------------------------------------------- #
# rectangle mask: exactness, clamping, honest counts (synthetic MergedHeads)
# --------------------------------------------------------------------------- #
def _merged(nbin_d=6, nbin_u=4, nens=20):
    from ladcp.qa.superens import MergedHeads
    nbin = nbin_d + nbin_u
    ru = np.zeros((nbin, nens))
    return MergedHeads(ru=ru, rv=ru.copy(), rw=ru.copy(), re=ru.copy(),
                       weight=np.ones_like(ru),
                       offset=np.concatenate([-np.arange(nbin_u, 0, -1.0),
                                              np.arange(1.0, nbin_d + 1)]) * 8.0,
                       izd=np.arange(nbin_d) + nbin_u,
                       izu=np.flip(np.arange(nbin_u)),
                       hrot=np.zeros(nens))


def _mask(merged, flags, counts=None):
    from ladcp.qa.superens import _edit_velocity_mask
    nens = merged.ru.shape[1]
    z = np.full(nens, 500.0)
    izm = z[None, :] + merged.offset[:, None]
    return _edit_velocity_mask(izm, z, merged, zbottom=None, edit_sidelobes=False,
                               dzbelow=16.0, mask_dn_bins=(), mask_up_bins=(),
                               manual_flags=tuple(flags), manual_counts=counts)


def test_rect_masks_exact_cells():
    m = _merged()
    counts: dict = {}
    mask = _mask(m, [("down", 2, 3, 5, 9), ("up", 1, 1, 0, 19)], counts)
    expect = np.zeros_like(mask)
    expect[m.izd[1:3], 5:10] = True               # down bins 2-3 -> rows izd[1..2]
    expect[m.izu[0], :] = True                    # up bin 1 -> row izu[0]
    np.testing.assert_array_equal(mask, expect)
    assert counts == {"manual_removed_down": 10, "manual_removed_up": 20}


def test_rect_clamps_and_noops():
    m = _merged()
    counts: dict = {}
    mask = _mask(m, [("down", 5, 99, -7, 10_000),     # clamps to bins 5-6, all ens
                     ("down", 7, 99, 0, 19),          # fully past the head: no-op
                     ("up", 1, 2, 30, 40)], counts)   # past the cast: no-op
    expect = np.zeros_like(mask)
    expect[m.izd[4:6], :] = True
    np.testing.assert_array_equal(mask, expect)
    assert counts == {"manual_removed_down": 40}


def test_up_rect_noop_on_single_head():
    m = _merged(nbin_u=0)
    counts: dict = {}
    mask = _mask(m, [("up", 1, 4, 0, 19)], counts)
    assert not mask.any() and counts == {}


def test_counts_never_resurrect_and_stay_honest():
    m = _merged()
    m.ru[m.izd[1], :] = np.nan                    # screen already NaN'd down bin 2
    counts: dict = {}
    mask = _mask(m, [("down", 2, 2, 0, 19)], counts)
    assert mask[m.izd[1]].all()                   # cells masked (no-op on NaN data)...
    assert counts == {"manual_removed_down": 0}   # ...but counted as newly removed: none


# --------------------------------------------------------------------------- #
# EditConfig / SessionConfig integration
# --------------------------------------------------------------------------- #
def test_editconfig_hashable_and_distinct():
    a = EditConfig(manual_flags=(("down", 3, 4, 0, 99),))
    b = EditConfig(manual_flags=(("down", 3, 4, 0, 98),))
    assert a != b and len({a, b, EditConfig()}) == 3


def test_to_cli_requires_journal_path():
    cfg = SessionConfig(edit=EditConfig(manual_flags=(("down", 3, 4, 0, 99),)))
    with pytest.raises(ValueError, match="manual edits"):
        cfg.to_cli("MORIA-80")
    cmd = cfg.to_cli("MORIA-80", edits="/r/.ladcp_edits/MORIA-80.json")
    assert "--edits /r/.ladcp_edits/MORIA-80.json" in cmd


def test_from_args_attaches_file_journal(tmp_path):
    from ladcp.qa.cli import build_parser
    p = _journal(tmp_path, [_entry(e1=99)])
    args = build_parser().parse_args(["80", "--edits", str(p)])
    cfg = SessionConfig.from_args(args)
    assert cfg.edit.manual_flags == (("down", 3, 4, 0, 99),)
    # a directory cannot map to one config: resolved per station later, empty here
    args = build_parser().parse_args(["80", "--edits", str(tmp_path)])
    assert SessionConfig.from_args(args).edit.manual_flags == ()


# --------------------------------------------------------------------------- #
# bit-parity with --nearfield-dn-bins + session cache behaviour (real fixture)
# --------------------------------------------------------------------------- #
@needs_fixtures
def test_manual_rect_bit_identical_to_nearfield_mask():
    """A full-cast rectangle over bins 3-4 IS the near-field mask, bit for bit."""
    from ladcp.config import resolve_params
    from ladcp.io.ctd_cnv import read_ctd_cnv
    from ladcp.qa.ingest import load_dualhead
    from ladcp.qa.inverse import compute_velocity_full

    p = resolve_params("MORIA", "MORIA-80")
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    n = min(dh.down.n_ens, dh.up.n_ens)

    p_nf = resolve_params("MORIA", "MORIA-80",
                          overrides={"edit_nearfield_dn_bins": (3, 4)})
    p_man = resolve_params("MORIA", "MORIA-80",
                           overrides={"edit_manual_flags": (("down", 3, 4, 0, n - 1),)})
    nf = compute_velocity_full(dh, ctd, drot=-9.878379, params=p_nf).vp
    man = compute_velocity_full(dh, ctd, drot=-9.878379, params=p_man).vp
    np.testing.assert_array_equal(man.u, nf.u)
    np.testing.assert_array_equal(man.v, nf.v)
    assert man.ubar == nf.ubar and man.vbar == nf.vbar


@needs_fixtures
def test_session_cache_keys_and_lru(monkeypatch):
    import ladcp.session as sess
    monkeypatch.setattr(sess, "_PREPARED_MAX", 2)
    ses = sess.StationSession(str(DOWN), str(UP), str(CTD), station="MORIA-80", cruise="MORIA")
    e0 = EditConfig()
    e1 = EditConfig(manual_flags=(("down", 3, 4, 0, 10),))
    ses.prepare(e0)
    ses.prepare(e1)
    assert set(ses._prepared) == {e0, e1}
    ses.prepare(e0)                                    # touch e0 -> e1 becomes LRU
    e2 = EditConfig(manual_flags=(("down", 5, 5, 0, 10),))
    ses.prepare(e2)                                    # evicts e1, keeps e0
    assert set(ses._prepared) == {e0, e2}

"""``ladcp init`` — first-run setup: discover, confirm, write ``cruise.toml``, trial.

The wizard proposes, the user disposes (spec §3): every step shows what
:mod:`~ladcp.hub.detect` found and what it inferred, and nothing is written before
its confirmation — the full ``cruise.toml`` is printed before saving, and the
archive-index build (the one step that reads PD0 headers) asks first too.

Non-interactive parity (spec §7): every answer is expressible as a flag, so
``ladcp init --yes --root …`` runs over SSH and in scripts. In ``--yes`` mode the
proposals are accepted as-is, with one deliberate exception: a ship-ADCP source is
never *auto-chosen* (it changes the science of every solve) — detected candidates
are listed and picked interactively or via ``--sadcp``/``--sadcp-source``.

The prompt driver is two injectable callables (``ask``/``say``), so the Studio
setup page (v2) and the tests drive the same flow without a terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import CRUISES
from . import cruise_config as cc
from .detect import SadcpCandidate, detect

__all__ = ["run_init"]


# ---------------------------------------------------------------------------
# prompt primitives (injectable for tests / the v2 Studio page)

def _confirm(ask, say, q: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        r = ask(f"{q} [{hint}] ").strip().lower()
        if not r:
            return default
        if r in ("y", "yes"):
            return True
        if r in ("n", "no"):
            return False
        say("please answer y or n")


def _text(ask, q: str, default: str) -> str:
    r = ask(f"{q} [{default}] ").strip()
    return r or default


def _choose(ask, say, title: str, options: list[str], *,
            none_label: str | None = None) -> int | None:
    say(title)
    for i, o in enumerate(options, 1):
        say(f"  {i}. {o}")
    if none_label:
        say(f"  0. {none_label}")
    while True:
        r = ask("> ").strip()
        if none_label and r in ("", "0"):
            return None
        if r.isdigit() and 1 <= int(r) <= len(options):
            return int(r) - 1
        say(f"enter a number 1-{len(options)}" + (", or 0" if none_label else ""))


# ---------------------------------------------------------------------------
# the flow

def run_init(ns, *, ask=input, say=print) -> int:
    root = Path(ns.root).resolve()
    if not root.is_dir():
        say(f"ladcp init: {root} is not a directory")
        return 1
    cfg_path = root / cc.CONFIG_NAME
    if cfg_path.exists() and not ns.force:
        if ns.yes or not _confirm(ask, say, f"{cfg_path} already exists — overwrite?",
                                  default=False):
            say(f"ladcp init: {cfg_path} already exists (use --force to overwrite)")
            return 1

    say(f"scanning {root} (filenames only, nothing is read or written yet) ...")
    det = detect(root)

    # -- step 1: LADCP casts ---------------------------------------------------------
    ladcp_dir = ns.ladcp
    if ladcp_dir is None:
        if det.ladcp.dir is None:
            say(f"ladcp init: {det.ladcp.evidence} under {root} — point --ladcp at the "
                "directory holding the PD0 deployment files, or re-run from the "
                "cruise directory")
            return 1
        ladcp_dir = det.ladcp.dir or "."
        say(f"\nLADCP casts: {ladcp_dir}/ — {det.ladcp.evidence}")
        if not ns.yes and not _confirm(ask, say, "use this directory?"):
            ladcp_dir = _text(ask, "LADCP directory (relative to the cruise root)",
                              ladcp_dir)
    if not (root / ladcp_dir).is_dir():
        say(f"ladcp init: LADCP directory {root / ladcp_dir} does not exist")
        return 1
    down_only_labels = [s.label for s in det.ladcp.stations if s.slave is None]
    if down_only_labels:
        say(f"  note: {len(down_only_labels)} single-head cast(s) "
            f"({', '.join(down_only_labels[:4])}{'…' if len(down_only_labels) > 4 else ''})"
            " — process those with `ladcp process <station>` plus --down-only via "
            "[params], or a per-station [edit] later")

    # -- step 2: CTD -----------------------------------------------------------------
    ctd_dir = ns.ctd or det.ctd.dir
    from_hex = False
    if ctd_dir is None:
        say("\nCTD: nothing found — velocity solves need a CTD time series; "
            "QA-only processing still works")
    else:
        say(f"\nCTD: {ctd_dir or '.'}/ — {det.ctd.evidence}")
        if not ns.yes and not _confirm(ask, say, "use this directory?"):
            ctd_dir = _text(ask, "CTD directory (relative to the cruise root)", ctd_dir)
        # any cast without a cleaned .cnv — incl. MASTER/SLAVE archives, where no
        # name-paired stations exist (missing_cnv empty) but n_cnv tells the story
        need_hex = det.ctd.n_hex > 0 and (bool(det.ctd.missing_cnv) or det.ctd.n_cnv == 0)
        if ns.from_hex is not None:
            from_hex = ns.from_hex
        elif need_hex and det.ctd.converter:
            say(f"  raw .hex casts can be converted on the fly (CTD_project found at "
                f"{det.ctd.converter})")
            from_hex = True if ns.yes else _confirm(ask, say, "enable --from-hex?")
        elif need_hex:
            say("  some casts have only raw .hex and the CTD_project converter was "
                "NOT found — install it or set LADCP_CTD_PROJECT to enable --from-hex")

    # -- step 3: ship-ADCP constraint (never auto-chosen) -----------------------------
    sadcp: SadcpCandidate | None = None
    nav = None
    if ns.no_sadcp:
        say("\nship-ADCP: skipped (--no-sadcp)")
    elif ns.sadcp:
        src = ns.sadcp_source or "vmdas"
        sadcp = SadcpCandidate(src, ns.sadcp, "named on the command line")
        say(f"\nship-ADCP: {ns.sadcp} ({src}, from --sadcp)")
    elif det.sadcp:
        say("\nship-ADCP constraint sources found:")
        if ns.yes:
            for c in det.sadcp:
                say(f"  - {c.path} ({c.source}: {c.evidence})")
            say("  none chosen (--yes never auto-picks a constraint source; "
                "re-run with --sadcp PATH --sadcp-source …, or add a [sadcp] "
                "table to cruise.toml later)")
        else:
            i = _choose(ask, say, "pick the source to constrain the inverse with:",
                        [f"{c.path} ({c.source}: {c.evidence})" for c in det.sadcp],
                        none_label="no ship-ADCP constraint")
            sadcp = det.sadcp[i] if i is not None else None
    else:
        say("\nship-ADCP: no candidate sources found (add a [sadcp] table later)")
    if sadcp is not None and sadcp.source == "ek80":
        say("  EK80 note: a SHALLOW constraint (~15-140 m) — most valuable on "
            "single-head casts and for upper-ocean referencing; on dual-head casts "
            "it is a consistency check more than a constraint (guide ch. 8). "
            "The slim-extraction step is offered after the index build; nothing "
            "is copied without that explicit step")

    if sadcp is not None and sadcp.source == "vmdas":
        if ns.nav:
            nav = ns.nav
        elif det.nav and not ns.yes:
            say("  an independent nav track enables the SADCP clock check "
                "(--sadcp-timeoff auto):")
            i = _choose(ask, say, "  use one?",
                        [f"{c.path} ({c.evidence})" for c in det.nav],
                        none_label="no clock correction")
            nav = det.nav[i].path if i is not None else None
        elif det.nav:
            say(f"  nav tracks found ({det.nav[0].path}) — pass --nav to enable "
                "the 'auto' clock correction")

    # -- step 4: cruise identity -----------------------------------------------------
    name = ns.name or det.name
    if not ns.yes:
        name = _text(ask, "\ncruise name (labels the params preset and the exports)",
                     name)
    preset = name.upper() in CRUISES
    say(f"cruise {name!r}: " + ("registered preset found — its parameter layers apply"
                                if preset else
                                "no registered preset — generic operator defaults "
                                "apply ([params] in cruise.toml overrides any field)"))

    # -- step 5: compose + confirm + write cruise.toml --------------------------------
    raw: dict = {"cruise": {"name": name},
                 "data": {"root": ".", "out": ns.out or "qa_out"}}
    if from_hex:
        raw["ctd"] = {"from_hex": True}
    if sadcp is not None:
        entry: dict = {"folder": sadcp.path, "source": sadcp.source}
        if nav:
            entry["nav"] = nav
            entry["timeoff"] = "auto"
        raw["sadcp"] = entry

    import tomli_w
    say("\ncruise.toml to be written:\n")
    for line in tomli_w.dumps(raw).splitlines():
        say(f"    {line}")
    if not ns.yes and not _confirm(ask, say, f"write {cfg_path}?"):
        say("nothing written")
        return 1
    cc.save_config(raw, cfg_path)
    say(f"wrote {cfg_path}")

    # -- step 6: archive index (the one detection that reads file headers) ------------
    labels = [s.label for s in det.ladcp.stations]
    if det.ctd.n_hex and ctd_dir is not None:
        if ns.yes or _confirm(ask, say, "build the archive index now (scans PD0 "
                                        "headers, writes .ladcp_archive.json)?"):
            from ..archive import build_index
            try:
                idx = build_index(root / ladcp_dir, root / (ctd_dir or "."), root=root)
            except Exception as e:      # one corrupt PD0 must not sink the whole setup
                say(f"index build failed ({type(e).__name__}: {e}) — continuing; "
                    "build it later with ladcp-index")
            else:
                say(f"indexed {len(idx['casts'])} cast(s) -> .ladcp_archive.json")
                labels = sorted(idx["casts"]) or labels
    elif labels:
        say(f"index skipped (no raw CTD .hex anchors); {len(labels)} cast(s) "
            "enumerable by filename")
    else:
        say("index skipped (no .hex anchors) and no name-pairable casts — "
            "process stations explicitly: ladcp process <station>")

    # -- step 6b: EK80 share -> slim local copy (needs the index for cast windows) -----
    if sadcp is not None and sadcp.source == "ek80":
        if getattr(sadcp, "extracted", False):
            say(f"EK80: {sadcp.path} is already a per-station extract tree — "
                "nothing to copy; each cast picks its files by time window")
        else:
            _ek80_offer(ns, root, cfg_path, raw, sadcp.path, ask, say)

    # -- step 7: trial station (suggested, skippable) ----------------------------------
    if labels and not ns.no_trial:
        trial = ns.trial if ns.trial not in (None, "auto") else labels[len(labels) // 2]
        wanted = (ns.trial is not None) if ns.yes else \
            _confirm(ask, say, f"\nprocess a trial station now ({trial}, "
                               "prints its QA scorecard)?")
        if wanted:
            rc = _process(cfg_path, [trial], say)
            _show_scorecard(cfg_path, trial, say)
            if rc == 0 and not ns.yes and len(labels) > 1 and \
                    _confirm(ask, say, "run the whole batch now (ladcp process)?"):
                _process(cfg_path, [], say)

    say("\ndone. next steps: `ladcp process` (new/stale casts), `ladcp config show`"
        + (", `ladcp process --all` for everything" if labels else ""))
    return 0


def _ek80_offer(ns, root: Path, cfg_path: Path, raw: dict, src: str, ask, say) -> None:
    """Timetable + slim extraction for an ek80 source (spec §3.4: copy only on consent).

    ``--yes`` runs it only with ``--ek80-extract`` (the explicit consent flag);
    interactively the timetable is shown first and the copy confirmed after.
    On success ``cruise.toml`` is re-pointed at the local slim copy (``ek80/``).
    """
    idx = root / ".ladcp_archive.json"
    if not idx.is_file():
        say("EK80 extraction skipped: it needs the archive index for the cast "
            "windows (build it, then use ladcp-ek80 or the Studio hub)")
        return
    if ns.yes and not ns.ek80_extract:
        say("EK80 extraction skipped (--yes never copies without --ek80-extract)")
        return
    if not ns.yes and not _confirm(ask, say, "\ncompute the EK80 timetable now "
                                             "(header peeks only, nothing copied)?"):
        return
    from . import ek80_ops
    src_abs = str(root / src)                    # absolute paths pass through the join
    try:
        table = ek80_ops.timetable([src_abs], idx, pre=ns.ek80_pre, post=ns.ek80_post)
        say(f"  {table['n_files']} EK80 file(s); {table['covered']}/"
            f"{len(table['stations'])} cast(s) covered")
        for row in table["stations"][:12]:
            say(f"    {row['station']:<14} {row['n']} file(s)")
        cmds = ek80_ops.commands([src_abs], idx, root / ek80_ops.DEFAULT_OUT,
                                 pre=ns.ek80_pre, post=ns.ek80_post)
        say(f"  terminal equivalent: {cmds['extract']}")
        if not table["covered"]:
            say("  nothing to extract (logging gap over every cast window)")
            return
        if not ns.yes and not _confirm(ask, say,
                                       f"extract slim copies into {root / 'ek80'}/"
                                       "<station>/ now?"):
            return
        jobs = ek80_ops.build_jobs([src_abs], idx, pre=ns.ek80_pre, post=ns.ek80_post)
        ok, nbytes, errors = ek80_ops.extract_jobs(
            jobs, root / ek80_ops.DEFAULT_OUT,
            progress=lambda i, n, st, f: say(f"  [{i}/{n}] {st:<14} {f}"))
        say(f"  extracted {ok}/{len(jobs)} file(s), {nbytes / 1e6:.0f} MB -> ek80/")
        for e in errors[:5]:
            say(f"  {e}")
        if ok:
            raw["sadcp"] = {**raw.get("sadcp", {}), "folder": ek80_ops.DEFAULT_OUT,
                            "source": "ek80"}
            cc.save_config(raw, cfg_path)
            say('  cruise.toml re-pointed: [sadcp] folder = "ek80"')
    except Exception as e:                       # a share hiccup must not sink init
        say(f"  EK80 flow failed ({type(e).__name__}: {e}) — finish with ladcp-ek80")


def _process(cfg_path: Path, stations: list[str], say) -> int:
    """Run stations through the hub's own process path (one orchestration path)."""
    from .cli import _cmd_process
    ns = argparse.Namespace(stations=stations, new=False, all=False, force=False,
                            config=str(cfg_path), jobs=1, no_plots=False,
                            verbose=bool(stations), no_progress=True, no_log=False)
    try:
        return _cmd_process(ns)
    except SystemExit as e:                     # keep init interactive on process errors
        say(f"trial run failed: {e}")
        return 1


def _show_scorecard(cfg_path: Path, station: str, say) -> None:
    try:
        ccfg = cc.load_config(cfg_path)
    except cc.ConfigError:
        return
    outdir = Path(ccfg.args_map.get("outdir", cfg_path.parent / "qa_out"))
    st_dir = outdir / "stations" / station
    txt = st_dir / f"{station}_qa.txt"
    if txt.is_file():
        say("")
        for line in txt.read_text(encoding="utf-8").splitlines():
            say(f"    {line}")
    pdf = st_dir / "report.pdf"
    if pdf.is_file():
        say(f"\nfull report: {pdf}")
    elif (st_dir / f"{station}_report.pdf").is_file():
        say(f"\nfull report: {st_dir / f'{station}_report.pdf'}")


# `--from-hex/--no-from-hex` need a tri-state default (None = decide from detection);
# argparse's BooleanOptionalAction spells the flags differently, so wire it by hand.
def add_init_parser(sub) -> None:
    """Register the ``init`` subcommand (kept here so the flow owns its flags)."""
    p = sub.add_parser("init", help="first-run setup: discover data, confirm, write "
                                    "cruise.toml, optional trial station")
    p.add_argument("--root", default=".", help="cruise directory to scan (default: .)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="accept every proposal; never prompts (a ship-ADCP source is "
                        "still only chosen via --sadcp)")
    p.add_argument("--name", default=None, help="cruise name (default: the directory's)")
    p.add_argument("--ladcp", metavar="DIR", default=None,
                   help="LADCP PD0 directory, relative to --root (default: detected)")
    p.add_argument("--ctd", metavar="DIR", default=None,
                   help="CTD directory, relative to --root (default: detected)")
    p.add_argument("--from-hex", dest="from_hex", action="store_true", default=None,
                   help="convert raw .hex CTD casts on the fly (needs CTD_project)")
    p.add_argument("--no-from-hex", dest="from_hex", action="store_false",
                   help="never convert .hex, even when .cnv coverage is incomplete")
    p.add_argument("--sadcp", metavar="PATH", default=None,
                   help="ship-ADCP source to constrain with (skips the choice step)")
    p.add_argument("--sadcp-source", choices=("vmdas", "codas", "ek80"), default=None,
                   help="what --sadcp points at (default: vmdas)")
    p.add_argument("--no-sadcp", action="store_true", help="no ship-ADCP constraint")
    p.add_argument("--nav", metavar="PATH", default=None,
                   help="independent nav track: enables --sadcp-timeoff auto (vmdas)")
    p.add_argument("--ek80-extract", action="store_true",
                   help="with an ek80 --sadcp source: slim-extract the on-station "
                        "files into <root>/ek80/<station>/ after the index build "
                        "(the explicit consent to copy; interactive runs ask instead)")
    p.add_argument("--ek80-pre", type=float, default=20.0, metavar="MIN",
                   help="EK80 cast-window start, minutes before cast UTC (default 20)")
    p.add_argument("--ek80-post", type=float, default=170.0, metavar="MIN",
                   help="EK80 cast-window end, minutes after cast UTC (default 170)")
    p.add_argument("--out", default=None, help="output directory (default: qa_out)")
    p.add_argument("--trial", nargs="?", const="auto", default=None, metavar="STATION",
                   help="process a trial station after setup (default pick: mid-cruise)")
    p.add_argument("--no-trial", action="store_true", help="skip the trial-station offer")
    p.add_argument("--force", action="store_true", help="overwrite an existing cruise.toml")

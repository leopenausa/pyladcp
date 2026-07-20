"""``ladcp status`` — the mid-cruise dashboard (wizard spec §4).

Three blocks, in the order a watchstander needs them: what still has to be
processed (the freshness rule over the station universe), how the processed casts
scored (the ``<station>_qa.json`` rollup, worst offenders named), and the loose
ends (missing CTD, single-head casts, unconstrained solves, unapplied edit
journals, a stale index). Every line ends with its action.

Strictly read-only and cheap: mtimes, filename globs and the per-station QA JSONs —
never a PD0 payload — so ``ladcp`` answers in a heartbeat over SSH on a full
cruise. (The plan sketched an inline index *refresh*; that decodes new raw files
and writes, so it stays an explicit action — status just tells you when it is
due.) ``--json`` emits the same picture for scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import cruise_config as cc
from .detect import curated_station_labels
from .freshness import station_state

__all__ = ["gather", "render"]

_QA_ORDER = ("fail", "warn", "ok")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _index_stale(idx_path: Path, root: Path) -> bool:
    """Any PD0 newer than the index file — new arrivals the index has not seen."""
    if not idx_path.is_file():
        return False
    cutoff = idx_path.stat().st_mtime
    ladcp = root / "LADCP"
    if not ladcp.is_dir():
        return False
    try:
        return any(p.stat().st_mtime > cutoff for p in ladcp.rglob("*.000"))
    except OSError:
        return False


def gather(ccfg: cc.CruiseConfig) -> dict:
    """The whole dashboard as one JSON-ready dict (the ``--json`` payload)."""
    args = cc.merged_qa_args(ccfg)
    root = Path(args.root)
    outdir = Path(args.outdir)
    idx_path = Path(args.index) if args.index else root / ".ladcp_archive.json"
    idx = _load_json(idx_path) or {}
    universe = sorted((idx.get("casts") or {}).keys())
    if not universe:
        universe = curated_station_labels(root)

    from ..discovery import discover
    from ..edits import journal_path
    has_sadcp = "sadcp" in ccfg.args_map

    stations: list[dict] = []
    for st in universe:
        fs = station_state(st, root=root, outdir=outdir, cruise=args.cruise,
                           index=idx or None, from_hex=args.from_hex,
                           ctd_cache=args.ctd_cache, config_path=ccfg.path)
        label = fs.label
        entry: dict = {"label": label, "freshness": fs.state, "reason": fs.reason,
                       "qa": None, "problems": [], "loose_ends": []}
        try:
            sf = discover(st, root=root, cruise=args.cruise, index=idx or None,
                          from_hex=args.from_hex, ctd_cache=args.ctd_cache)
        except Exception:
            sf = None
            entry["loose_ends"].append("inputs unresolved")
        qa = _load_json(outdir / "stations" / label / f"{label}_qa.json")
        if qa is not None:
            entry["qa"] = qa.get("overall_status")
            entry["problems"] = sorted(
                name for name, m in (qa.get("metrics") or {}).items()
                if m.get("status") in ("warn", "fail"))
        if sf is not None:
            if sf.ctd is None:
                entry["loose_ends"].append("no CTD (velocity impossible)")
            if sf.up is None:
                entry["loose_ends"].append("single-head (no up-looker)")
        # processed but no velocity solution (.lad is written on every solve) —
        # only when a CTD exists, else the no-CTD loose end already explains it
        if qa is not None and (sf is None or sf.ctd is not None) \
                and not (outdir / "stations" / label / f"{label}.lad").is_file():
            entry["loose_ends"].append("processed, no velocity solution")
        if has_sadcp and qa is not None and \
                not any(k.startswith("sadcp_") for k in qa.get("metrics") or {}):
            entry["loose_ends"].append("last solve had no SADCP constraint")
        if journal_path(root, label).is_file() and (
                qa is None or "manual_edits" not in (qa.get("metrics") or {})):
            entry["loose_ends"].append("edit journal not applied")
        stations.append(entry)

    fresh = {s: 0 for s in ("fresh", "stale", "missing")}
    for e in stations:
        fresh[e["freshness"]] += 1
    qa_counts = {s: sum(1 for e in stations if e["qa"] == s) for s in _QA_ORDER}
    return {"config": str(ccfg.path), "cruise": args.cruise, "root": str(root),
            "outdir": str(outdir), "n_stations": len(stations),
            "freshness": fresh, "qa": qa_counts, "stations": stations,
            "index_stale": _index_stale(idx_path, root),
            # the configured constraint identity (the GUI's EK80 panel keys off this)
            "sadcp_source": args.sadcp_source if has_sadcp else None,
            "sadcp_folder": ccfg.args_map.get("sadcp")}


def render(data: dict) -> list[str]:
    """The dashboard as plain, SSH-friendly text (priority order, action per line)."""
    lines: list[str] = []
    st = data["stations"]
    lines.append(f"cruise {data['cruise']} — {data['config']}")
    if not st:
        lines.append("no stations found (no archive index, no name-pairable casts) — "
                     "run `ladcp init`, or ladcp process <station> explicitly")
        return lines

    # 1. what still needs processing
    f = data["freshness"]
    pending = [e for e in st if e["freshness"] != "fresh"]
    lines.append(f"casts: {data['n_stations']} — {f['fresh']} fresh, "
                 f"{f['stale']} stale, {f['missing']} unprocessed")
    for e in pending[:8]:
        lines.append(f"  {e['freshness']:<11} {e['label']:<14} ({e['reason']})")
    if len(pending) > 8:
        lines.append(f"  … {len(pending) - 8} more")
    if pending:
        lines.append("  -> ladcp process")

    # 2. QA rollup, worst first
    q = data["qa"]
    n_proc = sum(q.values())
    if n_proc:
        lines.append(f"QA: {q['ok']} ok, {q['warn']} warn, {q['fail']} fail "
                     f"({n_proc} processed)")
        offenders = [e for e in st if e["qa"] in ("fail", "warn")]
        offenders.sort(key=lambda e: (e["qa"] != "fail", e["label"]))
        for e in offenders[:6]:
            probs = ", ".join(e["problems"][:3]) or "see report"
            pdf = Path(data["outdir"]) / "stations" / e["label"] / f"{e['label']}_report.pdf"
            lines.append(f"  [{e['qa'].upper():<4}] {e['label']:<14} {probs}  -> {pdf}")
        if len(offenders) > 6:
            lines.append(f"  … {len(offenders) - 6} more")

    # 3. loose ends
    loose = [(e["label"], le) for e in st for le in e["loose_ends"]]
    if loose or data["index_stale"]:
        lines.append("loose ends:")
        actions = {"no CTD (velocity impossible)": "check the CTD dir / --from-hex",
                   "single-head (no up-looker)":
                       "solves down-only automatically (reduced near-surface coverage)",
                   "processed, no velocity solution":
                       "ladcp process {label} (cause in {label}_qa.txt)",
                   "last solve had no SADCP constraint": "ladcp process {label}",
                   "edit journal not applied":
                       'set edits = ".ladcp_edits" under [edit] in cruise.toml',
                   "inputs unresolved": "ladcp-index build / check the data root"}
        for label, le in loose[:10]:
            lines.append(f"  {label:<14} {le}  -> {actions[le].format(label=label)}")
        if len(loose) > 10:
            lines.append(f"  … {len(loose) - 10} more")
        if data["index_stale"]:
            lines.append("  PD0 files newer than the archive index  "
                         "-> ladcp-index build (or re-run ladcp init)")
    if not pending and not loose and not data["index_stale"]:
        lines.append("nothing to process — all casts current.")
    return lines

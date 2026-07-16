"""Studio JSON bridge: request bodies -> :class:`SessionConfig`, solves -> payloads."""
from __future__ import annotations

import time

import numpy as np

from ..session import EditConfig, SessionConfig, SolveConfig
from .state import StudioState


def config_from_body(body: dict, state: StudioState) -> SessionConfig:
    """Request JSON -> :class:`SessionConfig` (ValueError on malformed values).

    The SADCP *identities* are fixed at launch (``--sadcp`` / ``--sadcp-codas`` on
    ``ladcp-studio``); the request picks one by key via ``sadcp_key`` (``"off"``
    disables) and weights it via ``solve.sadcpfac``. The pre-dropdown boolean
    ``use_sadcp`` still works and means the first source.
    """
    e = dict(body.get("edit") or {})
    s = dict(body.get("solve") or {})
    nearfield = e.get("nearfield_dn_bins")
    edit = EditConfig(
        down_only=bool(e.get("down_only", False)),
        nearfield_dn_bins=None if nearfield is None else tuple(int(b) for b in nearfield),
        dzbelow=None if e.get("dzbelow") is None else float(e["dzbelow"]),
        zbottom=None if e.get("zbottom") is None else float(e["zbottom"]),
        guessbottom=None if e.get("guessbottom") is None else float(e["guessbottom"]))
    solver = s.get("solver", "inverse")
    if solver not in ("shear", "inverse"):
        raise ValueError(f"solver must be 'shear' or 'inverse', got {solver!r}")
    solve = SolveConfig(
        solver=solver,
        drot=None if s.get("drot") is None else float(s["drot"]),
        botfac=float(s.get("botfac", 1.0)), barofac=float(s.get("barofac", 1.0)),
        smoofac=float(s.get("smoofac", 0.0)), sadcpfac=float(s.get("sadcpfac", 3.0)))
    sources = state.sadcp_sources
    key = body.get("sadcp_key")
    if key is None:                              # legacy boolean protocol
        # default ON only when a source was explicitly flagged: discovered ("found")
        # products are offered, never silently activated
        flagged = [c for k, c in sources.items()
                   if state.sadcp_origin.get(k, "flag") != "found"]
        use = bool(body.get("use_sadcp", bool(flagged)))
        pick = flagged[0] if flagged else (next(iter(sources.values())) if sources else None)
        sadcp = pick if use else None
    elif key == "off":
        sadcp = None
    elif key in sources:
        sadcp = sources[key]
    else:
        raise ValueError(f"unknown SADCP source {key!r} "
                         f"(launched with: {', '.join(sources) or 'none'})")
    return SessionConfig(edit=edit, sadcp=sadcp, solve=solve)


def _arr(x) -> list:
    """1-D array -> JSON-safe list (NaN/inf -> null; browsers reject NaN literals)."""
    a = np.asarray(x, float)
    return [float(v) if np.isfinite(v) else None for v in a]


def _num(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def _joint_n(dh) -> int:
    """Ensembles the merge will use: joint-trimmed when both heads are present."""
    return dh.down.n_ens if dh.up is None else min(dh.down.n_ens, dh.up.n_ens)


def _head_geom(dh, head) -> dict | None:
    h = dh.down if head == "down" else dh.up
    if h is None:
        return None
    return {"n_bins": int(h.n_cells), "cell_m": float(h.cell_m),
            "first_m": round(float(dh.bin_depth(h)[0]), 2)}


def _available_panels(result) -> list[str]:
    """Panel names renderable for this solve (acquisition panels are always on)."""
    names = ["raw", "alignment", "depth", "edit",
             "velocity", "shear", "inverse", "error", "drift"]
    if result.weights is not None:
        names.append("weights")
    if result.btrk is not None and result.btrk.n_own:
        names.append("btrack")
    if result.sadcp is not None and len(result.sadcp):
        names.append("sadcp")
    return names


def solve_payload(state: StudioState, label: str, cfg: SessionConfig) -> dict:
    """Run one solve on the station's (locked) session and shape the JSON response."""
    with state.lock_for(label):
        ses = state.session(label)
        prepared = ses.is_prepared(cfg.edit)
        t0 = time.perf_counter()
        result = ses.solve(cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        prep = ses.prepare(cfg.edit)             # cache hit: the solve just built it
        stages = dict(prep.timings)
        dn = prep.dh.down                        # bins <-> metres for the near-field UI
        dn_geom = {"first_m": round(float(prep.dh.bin_depth(dn)[0]), 2),
                   "cell_m": float(dn.cell_m), "n_bins": int(dn.n_cells)}
        if cfg.solve.drot is not None:
            drot, drot_source = cfg.solve.drot, "explicit"
        else:
            drot, drot_source = ses.declination(cfg.edit)
    vp = result.vp
    bt = None
    if result.bp is not None and result.bp.n_bins > 0:
        bt = {"z": _arr(result.bp.z), "u": _arr(result.bp.u), "v": _arr(result.bp.v),
              "uerr": _arr(result.bp.uerr)}
    sadcp = None
    if result.sadcp is not None and len(result.sadcp):
        sa = np.asarray(result.sadcp, float)     # [m,4] rows: z, u, v, verr (true frame)
        sadcp = {"z": _arr(sa[:, 0]), "u": _arr(sa[:, 1]), "v": _arr(sa[:, 2]),
                 "verr": _arr(sa[:, 3])}
    cli_kwargs = state.cli_context()
    if cfg.edit.manual_flags:                    # journal-backed solve: --edits replays it
        cli_kwargs["edits"] = str(state.edits_path(ses))
    return {
        "station": label,
        "solver": cfg.solve.solver,
        "drot": _num(drot), "drot_source": drot_source,
        "solve_ms": round(ms, 1), "prepared": prepared, "stages": stages,
        "zbottom": _num(result.zbottom),
        "profile": {"z": _arr(vp.z), "u": _arr(vp.u), "v": _arr(vp.v),
                    "uerr": _arr(vp.uerr), "nvel": _arr(vp.nvel),
                    "ubar": _num(vp.ubar), "vbar": _num(vp.vbar)},
        "bt": bt,
        "sadcp": sadcp,
        "sadcp_bins": 0 if result.sadcp is None else int(result.sadcp.shape[0]),
        "dn_geom": dn_geom,
        "manual_edits": len(cfg.edit.manual_flags),
        "panels": _available_panels(result),
        "cli": cfg.to_cli(label, **cli_kwargs),
    }

"""``cruise.toml`` — the cruise-level configuration file (wizard spec §5).

One file at the cruise root holds everything a cruise's processing needs: data
locations, CTD wiring, the chosen ship-ADCP source, solver knobs, and per-cast
parameter overrides. ``ladcp-qa`` (and later the ``ladcp`` hub) auto-discovers it in
the working directory or its parents; command-line flags are *overrides* on top, so
the precedence is always

    explicit flags  >  cruise.toml  >  built-in cruise preset  >  generic defaults.

Schema (every table optional; unknown tables/keys are errors, so a typo never
silently falls back to a default):

    [cruise]  name
    [data]    root, index, out
    [ctd]     from_hex, cache
    [edit]    down_only, nearfield_dn_bins, dzbelow, soundcorr, edits
    [solve]   solver, drot, botfac, barofac, smoofac, sadcpfac
    [sadcp]   folder, source, filetype, xducer, timeoff, nav, reingest
    [params]  any :class:`~ladcp.config.CastParams` field (cruise-wide)
    [params.<station-label>]  per-station overrides (win over [params])

``[sadcp]`` may also be an array of tables (``[[sadcp]]``) listing several candidate
sources; ``ladcp-qa`` constrains with the first and warns about the rest (Studio
grows multi-source support in a later phase). Relative paths are resolved against
the config file's directory, so a run started anywhere in the cruise tree lands its
outputs in the same place. ``[params]`` keys are validated against ``CastParams`` at
load time — a new cruise never needs a code edit (spec §5).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path

try:                                    # stdlib on 3.11+; the tomli backport on 3.10
    import tomllib
except ModuleNotFoundError:             # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

from ..config import CastParams

__all__ = ["CONFIG_NAME", "ConfigError", "CruiseConfig", "apply_to_args",
           "explicit_dests", "find_config", "load_config", "merge_params",
           "save_config", "station_params"]

CONFIG_NAME = "cruise.toml"

log = logging.getLogger("ladcp.hub")


class ConfigError(ValueError):
    """A cruise.toml problem; the message names the file and the offending table/key."""


# ---------------------------------------------------------------------------
# Schema: one row per knob. Load-time validation, the namespace merge and the
# (phase B) provenance annotation all read this table, so a new knob is wired once.
#
# (table, key) -> (ladcp-qa argparse dest, kind)
# kinds: str | float | bool | not-bool (TOML true -> dest False) | int-list
#        (TOML [3,4] -> the CLI's "3,4" string) | timeoff (seconds or 'auto') |
#        path (string, resolved against the config directory)

_SCHEMA: dict[tuple[str, str], tuple[str, str]] = {
    ("cruise", "name"): ("cruise", "str"),
    ("data", "root"): ("root", "path"),
    ("data", "index"): ("index", "path"),
    ("data", "out"): ("outdir", "path"),
    ("ctd", "from_hex"): ("from_hex", "bool"),
    ("ctd", "cache"): ("ctd_cache", "path"),
    ("edit", "down_only"): ("down_only", "bool"),
    ("edit", "nearfield_dn_bins"): ("nearfield_dn_bins", "int-list"),
    ("edit", "dzbelow"): ("dzbelow", "float"),
    ("edit", "soundcorr"): ("no_soundcorr", "not-bool"),
    ("edit", "edits"): ("edits", "path"),
    ("solve", "solver"): ("solver", "str"),
    ("solve", "drot"): ("drot", "float"),
    ("solve", "botfac"): ("botfac", "float"),
    ("solve", "barofac"): ("barofac", "float"),
    ("solve", "smoofac"): ("smoofac", "float"),
    ("solve", "sadcpfac"): ("sadcpfac", "float"),
}

_SADCP_SCHEMA: dict[str, tuple[str, str]] = {
    "folder": ("sadcp", "path"),
    "source": ("sadcp_source", "str"),
    "filetype": ("sadcp_filetype", "str"),
    "xducer": ("sadcp_xducer", "float"),
    "timeoff": ("sadcp_timeoff", "timeoff"),
    "nav": ("sadcp_nav", "path"),
    "reingest": ("sadcp_reingest", "bool"),
}

_CHOICES: dict[tuple[str, str], tuple[str, ...]] = {   # mirror the CLI's choices=
    ("solve", "solver"): ("shear", "inverse"),
    ("sadcp", "source"): ("vmdas", "codas", "ek80"),
    ("sadcp", "filetype"): ("STA", "LTA"),
}

_TABLES = ("cruise", "data", "ctd", "edit", "solve", "sadcp", "params")

# station/cruise_id are stamped by resolve_params from the run itself; letting the
# config override them would silently mislabel every output file.
_PARAM_FIELDS = {f.name for f in fields(CastParams)} - {"station", "cruise_id"}


@dataclass(frozen=True)
class CruiseConfig:
    """A loaded cruise.toml: the raw document plus its pre-validated projections."""

    path: Path
    raw: dict                               # parsed TOML, verbatim (round-trips to disk)
    args_map: dict[str, object]             # argparse dest -> value, paths resolved
    n_sadcp: int                            # sources listed ([[sadcp]] array size)
    params_global: dict[str, object]        # [params] cruise-wide CastParams overrides
    params_station: dict[str, dict[str, object]]   # [params.<label>] per-station layers


# ---------------------------------------------------------------------------
# locate / load / save

def find_config(start: Path | None = None) -> Path | None:
    """Locate ``cruise.toml`` in ``start`` (default: cwd) or its parents (git-style).

    Stops at the filesystem root or at the first directory that contains a ``.git``
    entry: a cruise directory inside a code checkout must never silently pick up an
    unrelated config from above the repository.
    """
    d = (Path.cwd() if start is None else Path(start)).resolve()
    for cand in (d, *d.parents):
        p = cand / CONFIG_NAME
        if p.is_file():
            return p
        if (cand / ".git").exists():
            break
    return None


def load_config(path: str | Path) -> CruiseConfig:
    """Parse + validate ``path`` into a :class:`CruiseConfig` (raises :class:`ConfigError`)."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: {e}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from None
    return _build(raw, path)


def save_config(data: dict, path: str | Path) -> Path:
    """Validate + atomically write ``data`` (a raw TOML document dict) as cruise.toml.

    Refuses to write anything :func:`load_config` would reject, and writes via a
    temp file + ``os.replace`` so an interrupted save never leaves a half-written
    config (spec §7).
    """
    import tomli_w

    path = Path(path)
    _build(data, path)                      # validation only; raw dict stays untouched
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# validation + projection

def _bad(path: Path, where: str, msg: str) -> ConfigError:
    return ConfigError(f"{path}: {where}: {msg}")


def _coerce(kind: str, value, path: Path, where: str):
    """TOML value -> the value the argparse namespace carries (raises on a type mismatch)."""
    if kind in ("str", "path"):
        if not isinstance(value, str):
            raise _bad(path, where, f"expected a string, got {value!r}")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _bad(path, where, f"expected a number, got {value!r}")
        return float(value)
    if kind in ("bool", "not-bool"):
        if not isinstance(value, bool):
            raise _bad(path, where, f"expected true/false, got {value!r}")
        return (not value) if kind == "not-bool" else value
    if kind == "int-list":
        if not isinstance(value, list) or any(isinstance(b, bool) or not isinstance(b, int)
                                              for b in value):
            raise _bad(path, where, f"expected a list of bin numbers, got {value!r}")
        return ",".join(str(b) for b in value) or "none"
    if kind == "timeoff":
        if isinstance(value, str):
            if value != "auto":
                raise _bad(path, where, f"expected seconds or 'auto', got {value!r}")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _bad(path, where, f"expected seconds or 'auto', got {value!r}")
        return float(value)
    raise AssertionError(kind)              # pragma: no cover - schema self-consistency


def _resolve(base: Path, value: str) -> str:
    """Path values are relative to the config's directory, not the caller's cwd."""
    return str(base / value)                # pathlib: joining an absolute path keeps it


def _check_table(raw: dict, tname: str, schema: dict, path: Path) -> dict[str, object]:
    """One table's ``dest -> value`` projection (unknown keys are errors)."""
    out: dict[str, object] = {}
    table = raw.get(tname)
    if table is None:
        return out
    if not isinstance(table, dict):
        raise _bad(path, f"[{tname}]", "expected a table")
    for key, value in table.items():
        row = schema.get((tname, key) if schema is _SCHEMA else key)
        if row is None:
            known = sorted(k[1] if isinstance(k, tuple) else k
                           for k in schema if not isinstance(k, tuple) or k[0] == tname)
            raise _bad(path, f"[{tname}] {key}", f"unknown key (known: {', '.join(known)})")
        dest, kind = row
        where = f"[{tname}] {key}"
        coerced = _coerce(kind, value, path, where)
        choices = _CHOICES.get((tname, key))
        if choices and coerced not in choices:
            raise _bad(path, where, f"expected one of {', '.join(choices)}, got {value!r}")
        out[dest] = coerced
    return out


def _check_params(raw: dict, path: Path) -> tuple[dict, dict[str, dict]]:
    """Validate ``[params]`` (+ per-station subtables) against ``CastParams`` fields."""
    table = raw.get("params")
    if table is None:
        return {}, {}
    if not isinstance(table, dict):
        raise _bad(path, "[params]", "expected a table")

    def one(entries: dict, where: str) -> dict[str, object]:
        ov: dict[str, object] = {}
        for key, value in entries.items():
            if key not in _PARAM_FIELDS:
                raise _bad(path, f"{where} {key}",
                           "not a CastParams field (see ladcp.config.CastParams)")
            if isinstance(value, dict) and key != "extra":
                raise _bad(path, f"{where} {key}", "a table is only valid for 'extra'")
            # TOML has arrays, CastParams has tuples (bin masks, tiltmax, manual flags)
            if isinstance(value, list):
                value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
            ov[key] = value
        return ov

    params_global: dict[str, object] = {}
    params_station: dict[str, dict[str, object]] = {}
    for key, value in table.items():
        if isinstance(value, dict) and key != "extra":  # station subtable ('extra' is a field)
            params_station[key] = one(value, f"[params.{key}]")
        else:
            params_global.update(one({key: value}, "[params]"))
    return params_global, params_station


def _build(raw: dict, path: Path) -> CruiseConfig:
    if not isinstance(raw, dict):
        raise _bad(path, "top level", "expected TOML tables")
    unknown = set(raw) - set(_TABLES)
    if unknown:
        raise _bad(path, f"[{sorted(unknown)[0]}]",
                   f"unknown table (known: {', '.join(_TABLES)})")
    base = path.parent

    args_map: dict[str, object] = {}
    for tname in ("cruise", "data", "ctd", "edit", "solve"):
        args_map.update(_check_table(raw, tname, _SCHEMA, path))

    # [sadcp]: a single table or an [[sadcp]] array; ladcp-qa uses the first source
    sadcp_raw = raw.get("sadcp")
    entries = ([sadcp_raw] if isinstance(sadcp_raw, dict)
               else list(sadcp_raw) if isinstance(sadcp_raw, list)
               else [] if sadcp_raw is None
               else None)
    if entries is None:
        raise _bad(path, "[sadcp]", "expected a table or an array of tables")
    projected = [_check_table({"sadcp": e}, "sadcp", _SADCP_SCHEMA, path) for e in entries]
    for i, (e, proj) in enumerate(zip(entries, projected, strict=True)):
        if "folder" not in e:
            where = f"[[sadcp]] #{i + 1}" if len(entries) > 1 else "[sadcp]"
            raise _bad(path, where, "'folder' is required")
        proj["sadcp"] = _resolve(base, proj["sadcp"])
        if proj.get("sadcp_nav") is not None:
            proj["sadcp_nav"] = _resolve(base, proj["sadcp_nav"])
    if projected:
        args_map.update(projected[0])

    for dest in ("root", "index", "outdir", "ctd_cache", "edits"):
        if dest in args_map:
            args_map[dest] = _resolve(base, args_map[dest])

    params_global, params_station = _check_params(raw, path)
    return CruiseConfig(path=path, raw=raw, args_map=args_map, n_sadcp=len(entries),
                        params_global=params_global, params_station=params_station)


# ---------------------------------------------------------------------------
# the CLI merge (precedence: explicit flags > cruise.toml > preset > generic)

def explicit_dests(build_parser, argv: list[str]) -> set[str]:
    """The argparse dests the user actually typed on this command line.

    Re-parses ``argv`` against a twin parser whose defaults are all suppressed, so
    the namespace holds exactly the typed options — the set that must keep winning
    over cruise.toml values. This is how "flag left at its default" and "flag typed
    with the default's value" are told apart.
    """
    twin = build_parser()
    for a in twin._actions:
        a.default = argparse.SUPPRESS
    ns, _ = twin.parse_known_args(list(argv))
    return set(vars(ns))


def apply_to_args(cfg: CruiseConfig, args: argparse.Namespace,
                  explicit: set[str]) -> list[str]:
    """Merge ``cfg`` into a parsed ``ladcp-qa`` namespace; typed flags win per knob.

    Returns the dests taken from the config (phase B's ``ladcp config show``
    provenance). The merge happens *before* :meth:`SessionConfig.from_args`, so the
    config flows through the exact same validation as the flags it stands in for.
    """
    applied = []
    for dest, value in cfg.args_map.items():
        if dest in explicit:
            continue
        setattr(args, dest, value)
        applied.append(dest)
    if cfg.n_sadcp > 1:
        log.warning("cruise.toml lists %d [[sadcp]] sources; ladcp-qa constrains with "
                    "the first (%s)", cfg.n_sadcp, cfg.args_map.get("sadcp"))
    return applied


def merged_qa_args(cfg: CruiseConfig):
    """A ``ladcp-qa`` namespace holding this config over the built-in defaults.

    Nothing is marked explicit, so the config wins over every parser default — the
    same merge ``ladcp-qa --config`` performs, minus any typed flags. This is how
    the hub's subcommands resolve root/outdir/cruise/… without a second config path.
    """
    from ..qa.cli import build_parser  # call-time: qa.cli imports this module
    args = build_parser().parse_args([])
    apply_to_args(cfg, args, explicit=set())
    return args


def merge_params(params_global: dict, params_station: dict[str, dict],
                 label: str) -> dict[str, object]:
    """The ``[params]`` + ``[params.<label>]`` overrides for one station (station wins)."""
    ov = dict(params_global)
    ov.update(params_station.get(label, {}))
    return ov


def station_params(cfg: CruiseConfig, label: str) -> dict[str, object]:
    """:func:`merge_params` over a loaded config, keyed by the resolved station label."""
    return merge_params(cfg.params_global, cfg.params_station, label)

"""Reproducible per-solve configuration + the cached station solve engine (Studio).

Frozen dataclasses that mirror the ``ladcp-qa`` option surface one-to-one, so a
configuration can travel between the CLI, the library, and the upcoming interactive
Studio GUI without drift. The hard contract: **every configuration is expressible as a
``ladcp-qa`` invocation** (:meth:`SessionConfig.to_cli`) and parsing that invocation
back recovers the identical configuration (:meth:`SessionConfig.from_args`) -- the
round-trip is enforced by ``tests/test_session_config.py``.

:class:`StationSession` (PR 2) is the staged solve cache the grouping anticipates:
:class:`EditConfig` fields invalidate the expensive build (sync / editing / bottom
detect / super-ensembles, ~1.2 s on MORIA-80), :class:`SadcpConfig` identifies the
ship-ADCP product (profile cached per session, ingest cached on disk), and
:class:`SolveConfig` only re-runs the constrained inverse (~30 ms). Results are
bit-identical to a fresh :func:`~ladcp.qa.inverse.compute_velocity_full` -- enforced
by ``tests/test_session_solve.py`` -- so anything Studio produces, ``ladcp-qa``
reproduces.

``ladcp-qa`` builds its ``inv_opts``/``sadcp_opts`` dicts through this module
(:meth:`SessionConfig.inv_opts` / :meth:`SessionConfig.sadcp_opts`), keeping a single
source of truth for option names, defaults, and validation.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                # heavy imports stay call-time
    from .io.ctd_cnv import CTDTimeSeries
    from .models import DualHead
    from .qa.inverse import VelocityResult

__all__ = ["EditConfig", "SadcpConfig", "SolveConfig", "SessionConfig", "StationSession",
           "parse_nearfield", "parse_timeoff", "resolve_declination"]

log = logging.getLogger("ladcp.session")


def parse_nearfield(text: str) -> tuple[int, ...]:
    """``--nearfield-dn-bins`` value -> 1-based bin tuple (``'none'``/empty -> ``()``).

    Raises ``ValueError`` with the exact ``ladcp-qa`` error text on a malformed list.
    """
    s = text.strip().lower()
    try:
        return () if s in ("none", "") else tuple(int(b) for b in s.split(",") if b.strip())
    except ValueError:
        raise ValueError(f"--nearfield-dn-bins: expected comma-separated bin numbers or "
                         f"'none', got {text!r}") from None


def parse_timeoff(value: str | None) -> str | float | None:
    """``--sadcp-timeoff`` value -> seconds (float), ``'auto'`` or ``None``.

    Raises ``ValueError`` with the exact ``ladcp-qa`` error text on a malformed value.
    """
    if value is None or value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"--sadcp-timeoff: expected seconds or 'auto', "
                         f"got {value!r}") from None


@dataclass(frozen=True)
class EditConfig:
    """Editing/preparation options: changing any of these re-runs the expensive build.

    ``None`` means "use the cruise preset" (see :class:`ladcp.config.CastParams`), so
    the resolved value can differ per cruise/station while the *configuration* stays
    portable. No preset sets ``nearfield_dn_bins`` -- the mask is always the user's
    explicit call -- so for it ``None`` and ``()`` both mean "no mask".
    """

    down_only: bool = False                              # --down-only
    nearfield_dn_bins: tuple[int, ...] | None = None     # --nearfield-dn-bins; () = disable
    dzbelow: float | None = None                         # --dzbelow [m]
    soundcorr: bool = True                                # --no-soundcorr disables (legacy ON)
    zbottom: float | None = None                         # --zbottom [m]: hard seabed override
    guessbottom: float | None = None                     # --guessbottom [m]: soft seabed seed
    # manual journal rectangles (ladcp.edits.manual_flags): canonical geometry-only
    # tuples, so a note edit in the journal never invalidates the prepare cache.
    # Reproducing this on the CLI needs the journal file: to_cli(edits=<path>).
    manual_flags: tuple[tuple[str, int, int, int, int], ...] = ()  # --edits


@dataclass(frozen=True)
class SadcpConfig:
    """Identity of the ship-ADCP product feeding the inverse constraint.

    The constraint *weight* (``--sadcpfac``) is a solve-stage knob and lives on
    :class:`SolveConfig`; this class only describes which data to ingest, so it can key
    the on-disk SADCP cache.
    """

    folder: str                                          # --sadcp PATH
    source: str = "vmdas"                                # --sadcp-source vmdas|codas
    filetype: str = "STA"                                # --sadcp-filetype STA|LTA
    xducer: float = 5.0                                  # --sadcp-xducer [m]
    timeoff: str | float | None = None                   # --sadcp-timeoff SECONDS|'auto'
    nav: str | None = None                               # --sadcp-nav PATH
    reingest: bool = False                               # --sadcp-reingest

    def __post_init__(self) -> None:
        if self.timeoff == "auto" and not self.nav:
            raise ValueError("--sadcp-timeoff auto needs --sadcp-nav")

    def validate_folder(self) -> None:
        """Launch-time existence check on ``folder`` (raises ``ValueError``).

        Fails fast at the CLI instead of minutes later at the first solve: VmDAS
        ingest does NOT recurse, so pointing ``--sadcp`` at the parent of the
        ``DATA/`` subfolder that actually holds the ``.STA`` files is a silent
        trap. A present ingest cache counts as valid (the raw averages may have
        been archived away) unless ``--sadcp-reingest`` forces a re-parse.
        """
        p = Path(self.folder)
        if self.source in ("codas", "ek80"):        # NetCDF file or its directory
            if not p.exists():
                raise ValueError(f"--sadcp: {p} does not exist")
            return
        if not p.is_dir():
            raise ValueError(f"--sadcp: {p} is not a directory")
        from .io.sadcp_vmdas import CACHE_NAME
        if not self.reingest and (p / CACHE_NAME).exists():
            return                                  # ingest loads the cache directly
        ft = self.filetype
        if not list(p.glob(f"*.{ft}")):
            subs = sorted(str(c.relative_to(p)) for c in p.iterdir()
                          if c.is_dir() and list(c.glob(f"*.{ft}")))
            msg = f"--sadcp: no .{ft} files directly under {p} (not searched recursively)"
            if subs:
                msg += (f"; found .{ft} files in: "
                        + ", ".join(f"{p}/{s}" for s in subs[:4])
                        + " — point --sadcp there")
            raise ValueError(msg)


@dataclass(frozen=True)
class SolveConfig:
    """Solve-stage knobs: re-running the inverse with these is ~30 ms on a cached build."""

    solver: str = "inverse"                              # --solver shear|inverse
    drot: float | None = None                            # --drot [deg]; None -> IGRF
    botfac: float = 1.0                                  # --botfac
    barofac: float = 1.0                                 # --barofac
    smoofac: float = 0.0                                 # --smoofac
    sadcpfac: float = 3.0                                # --sadcpfac


@dataclass(frozen=True)
class SessionConfig:
    """One station-solve's reproducible state = edit + SADCP identity + solve knobs."""

    edit: EditConfig = field(default_factory=EditConfig)
    sadcp: SadcpConfig | None = None
    solve: SolveConfig = field(default_factory=SolveConfig)

    # -- ladcp-qa bridge ------------------------------------------------------------
    @classmethod
    def from_args(cls, args) -> SessionConfig:
        """Build from a parsed ``ladcp-qa`` namespace (:func:`ladcp.qa.cli.build_parser`).

        Performs the same conversions/validation ``ladcp-qa`` applies, raising
        ``ValueError`` with the CLI's error text (the CLI maps it onto ``ap.error``).
        """
        nearfield = (None if args.nearfield_dn_bins is None
                     else parse_nearfield(args.nearfield_dn_bins))
        # --edits FILE attaches the journal's geometry here (launch-time validation);
        # a DIRECTORY is resolved per station later (qa.cli._run_one), so it cannot
        # be represented in a single SessionConfig and stays empty here.
        manual: tuple = ()
        edits_arg = getattr(args, "edits", None)
        if edits_arg and Path(edits_arg).is_file():
            from .edits import load_journal
            from .edits import manual_flags as _manual_flags
            manual = _manual_flags(load_journal(edits_arg))
        edit = EditConfig(down_only=args.down_only, nearfield_dn_bins=nearfield,
                          dzbelow=args.dzbelow, manual_flags=manual,
                          soundcorr=not getattr(args, "no_soundcorr", False),
                          zbottom=getattr(args, "zbottom", None),
                          guessbottom=getattr(args, "guessbottom", None))
        sadcp = None
        if args.sadcp:
            sadcp = SadcpConfig(folder=args.sadcp, source=args.sadcp_source,
                                filetype=args.sadcp_filetype, xducer=args.sadcp_xducer,
                                timeoff=parse_timeoff(args.sadcp_timeoff),
                                nav=args.sadcp_nav, reingest=args.sadcp_reingest)
        solve = SolveConfig(solver=args.solver, drot=args.drot, botfac=args.botfac,
                            barofac=args.barofac, smoofac=args.smoofac,
                            sadcpfac=args.sadcpfac)
        return cls(edit=edit, sadcp=sadcp, solve=solve)

    def to_cli(self, station: str, *, root: str | None = None, cruise: str | None = None,
               index: str | None = None, outdir: str | None = None,
               edits: str | None = None) -> str:
        """The minimal ``ladcp-qa`` command line reproducing this configuration.

        Only non-default options are emitted, so the command reads like a recipe.
        ``station`` is the id ``ladcp-qa`` resolves through discovery (index-driven
        mode; the explicit ``--down/--up/--ctd`` mode is out of scope here), and the
        discovery context (``root``/``cruise``/``index``/``outdir``) is included only
        when given.

        Manual edits live in a journal file, not in flags, so a config carrying
        ``edit.manual_flags`` needs its journal path (``edits=``) to be expressible
        -- and the emitted command reproduces this solve only while that journal is
        unchanged (its staleness fingerprints make any divergence detectable).
        """
        if self.edit.manual_flags and edits is None:
            raise ValueError("this configuration carries manual edits; pass "
                             "edits=<journal path> so the command can reproduce them")
        parts = ["ladcp-qa", shlex.quote(station)]

        def opt(flag: str, value) -> None:
            parts.append(flag)
            parts.append(shlex.quote(_fmt(value)))

        if root is not None:
            opt("--root", root)
        if cruise is not None:
            opt("--cruise", cruise)
        if index is not None:
            opt("--index", index)
        if outdir is not None:
            opt("--out", outdir)

        s = self.solve
        if s.solver != "inverse":
            opt("--solver", s.solver)
        if s.drot is not None:
            opt("--drot", s.drot)
        if s.botfac != 1.0:
            opt("--botfac", s.botfac)
        if s.barofac != 1.0:
            opt("--barofac", s.barofac)
        if s.smoofac != 0.0:
            opt("--smoofac", s.smoofac)

        e = self.edit
        if e.down_only:
            parts.append("--down-only")
        if e.nearfield_dn_bins is not None:
            opt("--nearfield-dn-bins",
                ",".join(str(b) for b in e.nearfield_dn_bins) or "none")
        if e.dzbelow is not None:
            opt("--dzbelow", e.dzbelow)
        if e.zbottom is not None:
            opt("--zbottom", e.zbottom)
        if e.guessbottom is not None:
            opt("--guessbottom", e.guessbottom)
        if not e.soundcorr:
            parts.append("--no-soundcorr")
        if edits is not None:
            opt("--edits", edits)

        sa = self.sadcp
        if sa is not None:
            opt("--sadcp", sa.folder)
            if sa.source != "vmdas":
                opt("--sadcp-source", sa.source)
            if s.sadcpfac != 3.0:
                opt("--sadcpfac", s.sadcpfac)
            if sa.filetype != "STA":
                opt("--sadcp-filetype", sa.filetype)
            if sa.xducer != 5.0:
                opt("--sadcp-xducer", sa.xducer)
            if sa.timeoff is not None:
                opt("--sadcp-timeoff", sa.timeoff)
            if sa.nav is not None:
                opt("--sadcp-nav", sa.nav)
            if sa.reingest:
                parts.append("--sadcp-reingest")
        return " ".join(parts)

    # -- option dicts consumed by the qa pipeline (single source of truth) -----------
    def inv_opts(self) -> dict:
        """The ``inv_opts`` dict ``ladcp-qa`` passes to ``_run_one``."""
        return {"botfac": self.solve.botfac, "barofac": self.solve.barofac,
                "smoofac": self.solve.smoofac, "down_only": self.edit.down_only,
                "nearfield_dn_bins": self.edit.nearfield_dn_bins,
                "dzbelow": self.edit.dzbelow, "soundcorr": self.edit.soundcorr,
                "zbottom": self.edit.zbottom, "guessbottom": self.edit.guessbottom}

    def sadcp_opts(self) -> dict | None:
        """The ``sadcp_opts`` dict ``ladcp-qa`` passes to ``_run_one`` (``None`` = no SADCP)."""
        if self.sadcp is None:
            return None
        return {"folder": self.sadcp.folder, "source": self.sadcp.source,
                "fac": self.solve.sadcpfac, "file_type": self.sadcp.filetype,
                "xducer": self.sadcp.xducer, "reingest": self.sadcp.reingest,
                "timeoff": self.sadcp.timeoff, "nav": self.sadcp.nav}


def _fmt(value) -> str:
    """Compact CLI rendering: floats lose a trailing ``.0`` (``2.0`` -> ``'2'``)."""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def resolve_declination(lat: float, lon: float, when, *, logger=None) -> tuple[float, str]:
    """IGRF-13 declination for a cast, with the ``ladcp-qa`` fallbacks -> ``(drot, source)``.

    ``source`` is ``"igrf"``, or ``"fallback-zero"`` when the lookup failed or the
    position is non-finite: the velocity then stays in the magnetic frame (drot=0) and
    the caller's declination QA metric WARNs. ``logger`` lets ``ladcp-qa`` keep the
    warnings in its own run log.
    """
    lg = logger if logger is not None else log
    try:
        from .proc.magdec import magnetic_declination
        drot = magnetic_declination(lat, lon, when)
    except Exception as e:                      # ppigrf missing / bad coeffs / bad input
        lg.warning("        declination: IGRF failed (%s: %s) -- velocity LEFT IN "
                   "MAGNETIC FRAME (drot=0, NOT true north)", type(e).__name__, e)
        return 0.0, "fallback-zero"
    if not np.isfinite(drot):                   # NaN position -> NaN declination
        lg.warning("        declination: IGRF non-finite at lat=%.4f lon=%.4f (missing CTD "
                   "position?) -- velocity LEFT IN MAGNETIC FRAME (drot=0, NOT true north)",
                   lat, lon)
        return 0.0, "fallback-zero"
    return drot, "igrf"


# ---------------------------------------------------------------------------
# StationSession: the staged solve cache (Studio PR 2)

_PREPARED_MAX = 4               # _Prepared entries kept per session (each ~100 MB)


@dataclass
class _Prepared:
    """One :class:`EditConfig`'s cached solve inputs (the expensive, weight-free part)."""

    params: object                  # resolved CastParams (with the edit overrides)
    dh: DualHead                    # ingested heads (up dropped when down_only)
    ctd: CTDTimeSeries
    context: tuple                  # build_solve_context output (se, z, merged, bt, sync, bottom)
    lat: float                      # cast position (CTD medians) + start time, for
    lon: float                      # declination and the SADCP window
    when: object
    timings: dict = field(default_factory=dict)  # stage wall times [ms]: load/ctd/build


class StationSession:
    """Cached single-station solve engine: re-solve in ~30 ms, rebuild only on edits.

    Holds one station's ingested data and the expensive solve context per
    :class:`EditConfig`, so sweeping :class:`SolveConfig` knobs (weights, solver,
    declination) costs only the inverse. This is the backend of the Studio GUI and a
    convenient library front door::

        ses = StationSession(down, up, ctd, station="MORIA-80")
        base = ses.solve(SessionConfig())                                  # ~1.5 s
        nobt = ses.solve(SessionConfig(solve=SolveConfig(botfac=0.0)))     # ~30 ms

    Results are bit-identical to a fresh
    :func:`~ladcp.qa.inverse.compute_velocity_full` with the same options (enforced by
    ``tests/test_session_solve.py``), and every configuration is reproducible on the
    command line via :meth:`SessionConfig.to_cli`.

    A session is single-station and not thread-safe; guard concurrent ``solve`` calls
    with a lock (the Studio server does).
    """

    def __init__(self, down, up=None, ctd=None, *, station: str = "",
                 cruise: str = "MORIA", ctd_utc=None, dz: float = 8.0):
        from pathlib import Path
        self.down = str(down)
        self.up = str(up) if up else None
        self.ctd_path = str(ctd) if ctd else None
        self.station = station or Path(self.down).stem
        self.cruise = cruise
        self.ctd_utc = ctd_utc                  # index cast-start UTC -> sync prior
        self.dz = float(dz)
        # LRU-capped: every distinct EditConfig (e.g. each brush stroke in the Studio
        # Edit view) holds its own ~100 MB _Prepared; unbounded, a long manual-editing
        # session would exhaust RAM. 4 keeps the common toggles (baseline/down-only/
        # current-edits) warm.
        from collections import OrderedDict
        self._prepared: OrderedDict[EditConfig, _Prepared] = OrderedDict()
        self._sadcp_profiles: dict[SadcpConfig, np.ndarray | None] = {}
        self._declination: tuple[float, str] | None = None

    # -- warm tier -------------------------------------------------------------------
    def prepare(self, edit: EditConfig | None = None) -> _Prepared:
        """Ingest + build the solve context for ``edit`` (cached; ~1.5 s first time).

        Mirrors the ``ladcp-qa`` per-station recipe exactly: resolve the cruise params
        with the edit overrides, load both heads, apply the header geometry, read the
        CTD, then drop the up-looker if ``down_only``.
        """
        edit = edit if edit is not None else EditConfig()
        prep = self._prepared.get(edit)
        if prep is not None:
            self._prepared.move_to_end(edit)
            return prep
        if self.ctd_path is None:
            raise ValueError("StationSession needs a CTD time series to solve "
                             "(pass ctd= when constructing the session)")
        import time as _time

        from .config import resolve_params
        from .io.ctd_cnv import read_ctd_cnv
        from .qa.ingest import apply_header_config, load_dualhead
        timings: dict = {}

        def tick(name, fn, *a, **kw):
            t0 = _time.perf_counter()
            out = fn(*a, **kw)
            timings[name] = round((_time.perf_counter() - t0) * 1000.0, 1)
            return out

        overrides = {}
        if edit.nearfield_dn_bins is not None:
            overrides["edit_nearfield_dn_bins"] = edit.nearfield_dn_bins
        if edit.dzbelow is not None:
            overrides["dzbelow"] = edit.dzbelow
        if edit.zbottom is not None:
            overrides["zbottom"] = edit.zbottom
        if edit.guessbottom is not None:
            overrides["guessbottom"] = edit.guessbottom
        if edit.manual_flags:
            overrides["edit_manual_flags"] = edit.manual_flags
        params = resolve_params(self.cruise, self.station, overrides=overrides or None)
        dh = tick("load_ms", load_dualhead, self.down, self.up,
                  station=self.station, params=params)
        apply_header_config(params, dh)
        ctd = tick("ctd_ms", read_ctd_cnv, self.ctd_path, params=params)
        if self.ctd_utc and "utc_start" not in ctd.meta:
            ctd.meta["utc_start"] = self.ctd_utc
        if edit.down_only and dh.has_up:
            dh = replace(dh, up=None)

        from .qa.inverse import build_solve_context
        context = tick("build_ms", build_solve_context, dh, ctd, dz=self.dz, params=params)
        prep = _Prepared(params=params, dh=dh, ctd=ctd, context=context,
                         lat=float(np.nanmedian(ctd.lat)), lon=float(np.nanmedian(ctd.lon)),
                         when=dh.down.time[0].astype("datetime64[s]").item(),
                         timings=timings)
        self._prepared[edit] = prep
        while len(self._prepared) > _PREPARED_MAX:
            self._prepared.popitem(last=False)
        return prep

    # -- hot tier --------------------------------------------------------------------
    def solve(self, cfg: SessionConfig | None = None) -> VelocityResult:
        """Solve ``cfg`` on the cached context (~30 ms once prepared)."""
        cfg = cfg if cfg is not None else SessionConfig()
        prep = self.prepare(cfg.edit)
        drot = cfg.solve.drot
        if drot is None:
            drot, _ = self.declination(cfg.edit)
        sadcp = self._sadcp(cfg, prep)
        from .qa.inverse import compute_velocity_full
        return compute_velocity_full(prep.dh, prep.ctd, drot=drot, dz=self.dz,
                                     params=prep.params, solver=cfg.solve.solver,
                                     sadcp=sadcp, sadcpfac=cfg.solve.sadcpfac,
                                     botfac=cfg.solve.botfac, barofac=cfg.solve.barofac,
                                     smoofac=cfg.solve.smoofac, context=prep.context)

    def declination(self, edit: EditConfig | None = None) -> tuple[float, str]:
        """The station's IGRF declination ``(drot, source)`` (cached; edit-independent)."""
        if self._declination is None:
            prep = self.prepare(edit)
            self._declination = resolve_declination(prep.lat, prep.lon, prep.when)
        return self._declination

    def _sadcp(self, cfg: SessionConfig, prep: _Prepared) -> np.ndarray | None:
        """The cast-windowed ship-ADCP constraint profile, cached per :class:`SadcpConfig`."""
        if cfg.sadcp is None:
            return None
        from .qa.cli import _sadcp_profile  # call-time: qa.cli imports this module
        t = prep.dh.down.time
        if cfg.solve.solver != "inverse":       # constraint ignored (with the CLI's notice)
            return _sadcp_profile(cfg.sadcp_opts(), t.min(), t.max(),
                                  prep.lat, prep.lon, cfg.solve.solver)
        if cfg.sadcp not in self._sadcp_profiles:
            self._sadcp_profiles[cfg.sadcp] = _sadcp_profile(
                cfg.sadcp_opts(), t.min(), t.max(), prep.lat, prep.lon, "inverse")
        return self._sadcp_profiles[cfg.sadcp]


def _check_field_coverage() -> None:   # pragma: no cover - import-time self-check
    """Guard against silently adding config fields ``to_cli`` does not emit."""
    known = {
        EditConfig: {"down_only", "nearfield_dn_bins", "dzbelow", "manual_flags",
                     "soundcorr", "zbottom", "guessbottom"},
        SadcpConfig: {"folder", "source", "filetype", "xducer", "timeoff", "nav",
                      "reingest"},
        SolveConfig: {"solver", "drot", "botfac", "barofac", "smoofac", "sadcpfac"},
    }
    for klass, names in known.items():
        have = {f.name for f in fields(klass)}
        if have != names:
            raise RuntimeError(f"{klass.__name__} fields changed ({have ^ names}); "
                               "update SessionConfig.to_cli/from_args and this check")


_check_field_coverage()

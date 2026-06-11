"""Reproducible per-solve configuration (Studio roadmap, PR 1).

Frozen dataclasses that mirror the ``ladcp-qa`` option surface one-to-one, so a
configuration can travel between the CLI, the library, and the upcoming interactive
Studio GUI without drift. The hard contract: **every configuration is expressible as a
``ladcp-qa`` invocation** (:meth:`SessionConfig.to_cli`) and parsing that invocation
back recovers the identical configuration (:meth:`SessionConfig.from_args`) -- the
round-trip is enforced by ``tests/test_session_config.py``.

The grouping anticipates the staged solve cache (PR 2): :class:`EditConfig` fields
invalidate the expensive build (sync / editing / bottom detect / super-ensembles,
~1.2 s), :class:`SadcpConfig` identifies the ship-ADCP product (ingest cached on
disk), and :class:`SolveConfig` only re-runs the constrained inverse (~30 ms). All
three are frozen and hashable so they can key those cache tiers directly.

``ladcp-qa`` builds its ``inv_opts``/``sadcp_opts`` dicts through this module
(:meth:`SessionConfig.inv_opts` / :meth:`SessionConfig.sadcp_opts`), keeping a single
source of truth for option names, defaults, and validation.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, fields

__all__ = ["EditConfig", "SadcpConfig", "SolveConfig", "SessionConfig",
           "parse_nearfield", "parse_timeoff"]


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
    portable.
    """

    down_only: bool = False                              # --down-only
    nearfield_dn_bins: tuple[int, ...] | None = None     # --nearfield-dn-bins; () = disable
    dzbelow: float | None = None                         # --dzbelow [m]


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
        edit = EditConfig(down_only=args.down_only, nearfield_dn_bins=nearfield,
                          dzbelow=args.dzbelow)
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
               index: str | None = None, outdir: str | None = None) -> str:
        """The minimal ``ladcp-qa`` command line reproducing this configuration.

        Only non-default options are emitted, so the command reads like a recipe.
        ``station`` is the id ``ladcp-qa`` resolves through discovery (index-driven
        mode; the explicit ``--down/--up/--ctd`` mode is out of scope here), and the
        discovery context (``root``/``cruise``/``index``/``outdir``) is included only
        when given.
        """
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
                "dzbelow": self.edit.dzbelow}

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


def _check_field_coverage() -> None:   # pragma: no cover - import-time self-check
    """Guard against silently adding config fields ``to_cli`` does not emit."""
    known = {
        EditConfig: {"down_only", "nearfield_dn_bins", "dzbelow"},
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

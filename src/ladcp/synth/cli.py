"""``ladcp-synth`` — generate a shippable synthetic dual-head LADCP station.

Writes a self-contained example (no private dependencies) that runs end-to-end through
``ladcp-qa``. The ocean profile and seabed are known, so the output can be sanity-checked
against the printed truth; the same generator backs the recovery test.
"""

from __future__ import annotations

import argparse

from .dataset import SynthConfig, generate_station


def build_parser() -> argparse.ArgumentParser:
    d = SynthConfig()                           # defaults sourced from the dataclass (no drift)
    p = argparse.ArgumentParser(
        prog="ladcp-synth",
        description="Generate a synthetic dual-head LADCP station (PD0 + CTD .cnv).")
    p.add_argument("--out", default=str(d.out), help="output root directory")
    p.add_argument("--station", default=d.station, help="station label / file prefix")
    p.add_argument("--seed", type=int, default=d.seed, help="RNG seed (determinism)")
    p.add_argument("--noise", type=float, default=d.noise,
                   help="velocity noise std [m/s] (0 = clean recovery target)")
    p.add_argument("--depth", type=float, default=d.depth, help="package max depth [m]")
    p.add_argument("--seabed", type=float, default=d.seabed,
                   help="seabed / water depth [m] (cast stops ~20 m above it)")
    p.add_argument("--u0", type=float, default=d.u0, help="barotropic east velocity [m/s]")
    p.add_argument("--v0", type=float, default=d.v0, help="barotropic north velocity [m/s]")
    p.add_argument("--shear-amp", type=float, default=d.shear_amp,
                   help="baroclinic shear amplitude [m/s]")
    p.add_argument("--shear-scale", type=float, default=d.shear_scale_m,
                   help="shear e-folding scale [m]")
    p.add_argument("--n-cells", type=int, default=d.n_cells, help="bins per head")
    p.add_argument("--cell-m", type=float, default=d.cell_m, help="bin length [m]")
    p.add_argument("--lat", type=float, default=d.lat, help="station latitude [deg]")
    p.add_argument("--lon", type=float, default=d.lon, help="station longitude [deg]")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SynthConfig(
        out=args.out, station=args.station, seed=args.seed, noise=args.noise,
        depth=args.depth, seabed=args.seabed, u0=args.u0, v0=args.v0,
        shear_amp=args.shear_amp, shear_scale_m=args.shear_scale,
        n_cells=args.n_cells, cell_m=args.cell_m, lat=args.lat, lon=args.lon)
    paths, truth = generate_station(cfg)

    print(f"Synthetic station '{cfg.station}' written under {paths.root}/")
    print(f"  down : {paths.down}")
    print(f"  up   : {paths.up}")
    print(f"  ctd  : {paths.ctd}")
    print(f"  truth: ubar={truth.ubar:+.3f} m/s  vbar={truth.vbar:+.3f} m/s  "
          f"seabed={cfg.seabed:.0f} m")
    print(f"\nProcess it with:\n  ladcp-qa {cfg.station} --root {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

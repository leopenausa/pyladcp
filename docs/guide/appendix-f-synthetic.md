# Appendix F · Synthetic test station (`ladcp-synth`)

`ladcp-synth` generates a **complete, self-contained LADCP station from a known ocean
velocity profile** — real dual-head PD0 files plus a cleaned CTD `.cnv`, in the layout
`ladcp-qa` auto-discovers. Because *you* set the answer, you can run the whole pipeline
and check that it gives the answer back.

It needs no data, no network, and no private dependencies. Everything is forward-modelled
in NumPy from a handful of parameters and a seed, so the output is deterministic and
license-clean.

```bash
ladcp-synth --out synthetic_station --seed 0
ladcp-qa SYNTH-01 --root synthetic_station --out qa_out
```

The first command prints the truth it built in (so you have something to check against):

```text
Synthetic station 'SYNTH-01' written under synthetic_station/
  down : synthetic_station/LADCP/SYNTH-01-LADCP-M.000
  up   : synthetic_station/LADCP/SYNTH-01-LADCP-S.000
  ctd  : synthetic_station/CTD/SYNTH-01_clean.cnv
  truth: ubar=-0.060 m/s  vbar=+0.030 m/s  seabed=1100 m
```

## What it generates

A station in the curated discovery layout (the same one [chapter 4](04-first-station.md)
describes), so `ladcp-qa <station> --root <dir>` just works:

```text
synthetic_station/
├── LADCP/
│   ├── SYNTH-01-LADCP-M.000     # down-looker  — real RDI PD0 (earth frame)
│   └── SYNTH-01-LADCP-S.000     # up-looker    — real RDI PD0
└── CTD/
    └── SYNTH-01_clean.cnv       # 6-column cleaned CTD with merged GPS
```

These are *genuine* PD0 byte streams written by pyladcp's own writer (`ladcp.io.pd0_write`)
and read back by the same decoder a real instrument file goes through — not a shortcut that
injects pre-decoded arrays. The cast is forward-modelled the way a real one is acquired:

- a **down-then-up cast** that descends to ~20 m above the seabed and back, with the
  package pinging on deck before and after (so the clock-sync step has a real cast window
  to lock onto);
- a **known ocean current**: a thermocline shear in *u* (an eastward surface layer over a
  counter-flowing deep layer) with *v* veering through it, on top of a barotropic mean;
- a **flat seabed** that shows up as a constant-depth echo return, so the bottom detector
  and bottom-track constraint have something to find;
- realistic attitude, sound speed, and a near-constant (station-keeping) GPS track.

A seed makes it reproducible; a `--noise` knob adds measurement scatter when you want a
less idealised cast (see [Appendix A](appendix-a-flags.md) for the full flag list).

## Why it exists — use cases and value

### 1 · A first run with no data of your own
The only real station shipped in the repository (MORIA-80) is one specific cast. If you've
just installed pyladcp and want to *see it work* end-to-end before pointing it at your own
cruise, `ladcp-synth` gives you a complete station to process in two commands — no download,
no credentials, nothing to clean first.

### 2 · Learning what each stage does
Because the truth is known and printed, the synthetic station is a teaching aid: change a
parameter (`--u0`, `--shear-amp`, `--seabed`, `--noise`) and watch how the QA report, the
velocity profile, and the bottom detection respond. You can see exactly what a clean cast
*should* look like in every figure of the report before you have to judge a messy real one.

### 3 · A known-answer accuracy test
This is the scientific value. With a real cast you can compare pyladcp to another processor
(see [`ladcp-compare`](appendix-a-flags.md)), but you can never compare it to *truth* —
nobody knows the true ocean velocity. The synthetic station closes that gap: the recovery
test (`tests/test_synth_recovery.py`) generates a clean station, runs the full inverse, and
asserts the solved profile matches the injected current (correlation > 0.99 in *u*, the
barotropic reference within ~1 cm/s, the seabed within a bin). It is the only test in the
suite that checks the solver against a *known* answer rather than against another estimate.

### 4 · A probe that finds silent failures
A known-answer harness exercises the pipeline on inputs real data rarely produces — and that
is where quiet bugs hide. The synthetic station found a real one: on an exceptionally clean
fit the inverse's two-pass refinement could drive the velocity error toward zero, collapse
the data weights below the keep-threshold, and return a **NaN barotropic reference with no
warning**. Real, noisy casts never reached that edge, so it had been invisible. The fix (a
floor on the refined velocity error, plus a guard that warns instead of failing silently)
shipped with a regression test built on exactly this synthetic case.

### 5 · CI and reproducible bug reports
The generated station is small and deterministic, so it doubles as a fast smoke test in
continuous integration and as a way to file a reproducible bug report: a seed and a flag
set fully specify the input, so anyone can regenerate the identical station and see the
same behaviour.

## What it is *not*

The synthetic station is idealised on purpose. The package holds station over a flat
seabed, the seabed return is a clean Gaussian, the water is smooth between bins, and the
noise (when enabled) is white. It is therefore a test of the **processing chain on
well-behaved input**, not a stand-in for the messy realities — beam-coordinate data,
fragmented files, sidelobe contamination, sloping bathymetry, sound-speed structure — that
[chapters 5](05-cruise-workflow.md) and [9](09-troubleshooting.md) deal with. Validation
against real casts and against a reference processor (MORIA-80; `ladcp-compare`) remains the
authority; the synthetic station complements it by adding the one thing real data can't —
a known answer.

## Regenerating the bundled example

A small pre-built copy lives in the repository at `tests/fixtures/synthetic/` so the
recovery test and this guide have a stable artifact. Rebuild it with:

```bash
ladcp-synth --out tests/fixtures/synthetic --station SYNTH-01 --seed 0
```

The generator ships inside the package, so the bundled binary is just a convenience — the
canonical source is the one command above.

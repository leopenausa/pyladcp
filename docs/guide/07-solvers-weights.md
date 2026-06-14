# 7 · Solvers & weight tuning

The defaults reproduce the validated legacy behaviour and are right for most casts —
**don't tune until a scorecard row or a figure tells you to**
([chapter 6](06-qa-report.md)). This chapter explains what the knobs physically do, so
that when you *do* turn one, you know what you're trading.

## The two solvers

**`--solver inverse` (the default)** is the full constrained least-squares inverse
(legacy `getinv`). Every measured cell contributes one equation — *ocean velocity at
that depth + package velocity at that time = what the instrument saw* — and the system
is augmented with the constraints described below. It is solved in two passes: the
first pass's residuals reject the worst 1% of the data (outlier removal) and set the
velocity error; the second pass re-solves on the cleaned data.

**`--solver shear`** is the classic shear method (legacy `ps.shear==1`): vertically
differentiate, bin, integrate back to a baroclinic *shape*, and pin that shape with a
barotropic reference. It cannot blend constraints with depth the way the inverse does.

**When to use which:** the inverse, always — except as a **cross-check**. The two
solvers make different assumptions, so their agreement is evidence the solution is
data-driven, not constraint-driven (on the validation cruises they agree to 1–2 cm/s).
A cast where they *diverge* is telling you the constraints are doing heavy lifting —
look at the weights figure to see which one.

**Before either solver runs**, pyladcp applies the legacy **sound-speed correction**
(`p.soundcorr`, on by default). A Doppler velocity scales with the sound speed the
instrument assumed, so the water velocities are rescaled per ensemble by the in-situ
sound speed (from the CTD profile at the package depth) over the value the firmware was
configured with. When the firmware ran a fixed sound speed far from the real profile —
e.g. 1450 m/s against an in-situ ~1492 — this is a few-percent correction to the whole
column (on the bundled example station it moved the depth-mean by ~0.16 cm/s, onto the
validated reference). It is a faithful default; `--no-soundcorr` only exists to reproduce
a legacy run that omitted it.

## The four weights

The inverse blends four information sources. Each weight scales how strongly that
source pulls relative to the LADCP water-track data:
**1 ≈ legacy-standard balance · 0 = off · >1 = trust it more.**

| flag | default | what it physically is |
|---|---|---|
| `--botfac` | 1 | **Bottom track** — velocity over ground from the seabed echo. Anchors the absolute (barotropic) level *near the bottom*, where it is by far the strongest reference. |
| `--barofac` | 1 | **GPS navigation** — the ship+package displacement over the cast. One row pinning the *depth-mean* (barotropic) velocity of the whole profile. |
| `--sadcpfac` | 3 | **Ship-ADCP** — the hull instrument's absolute currents. Pins the *upper-ocean* bins over its range. (3 is the validated golden value; only active with `--sadcp`.) |
| `--smoofac` | 0 | **Curvature smoothing** — a penalty on profile wiggliness. At the default 0, only *ill-constrained* bins (weight < 0.3× the median) are smoothed — the legacy behaviour. Raising it smooths everything: noise goes down, real shear goes with it. |

Two mental models that prevent most tuning mistakes:

- The constraints fix the **reference level**, the water-track data fixes the
  **shape**. If the shape is wrong, no weight will fix it — that's an editing problem
  ([chapter 6](06-qa-report.md), edit figure). If the *level* is wrong, it's exactly
  one of these constraints misbehaving.
- The three reference constraints act at **different depths** (bottom track: deep;
  ship-ADCP: shallow; GPS: the mean). On a deep cast with all three active, no single
  one is critical. On a shallow cast with no usable bottom track, the GPS row may be
  carrying everything — which is why shallow casts deserve a look at the weights
  figure.

## Reading — and acting on — the weights figure

`figures/<station>_weights.png` (legacy Fig. 12; shown in
[chapter 6](06-qa-report.md)) stacks each constraint's accumulated weight over the
ocean bins and the package time-series. Use it to answer two questions:

1. **"Which constraint owns the reference here?"** Before blaming the bottom track for
   an offset, check it actually carries weight in the suspect depth range.
2. **"Why did my flag change nothing?"** If `--sadcpfac 6` produced an identical
   profile, the figure will show the ship-ADCP rows never had weight — usually no
   SADCP ensembles matched the cast window ([chapter 9](09-troubleshooting.md)).

## `--down-only`

Solves from the down-looker alone, ignoring any up-looker (the acquisition QA still
covers both heads). Two uses:

- **Single-instrument casts** — no up-looker existed; this is simply how you process.
- **Cross-check** — when you suspect one head (compass trouble, contamination),
  compare the dual solve against `--down-only`. On clean stations they reproduce to a
  few cm/s even at depth.

The result carries a `single_head_solve` WARN: reduced near-surface coverage, and the
reference layer built from down-looker bins only. **Don't deliver `--down-only`
products from casts whose down-looker is the contaminated head** — e.g. a device hung
below the package (see the near-field recipe below): in the dual solve the up-looker
dilutes the contamination; down-only concentrates it.

## Recipes

**Bad near-bottom bottom-track samples (shallow casts).**
Symptom: `bottom_track_consistency` WARN, or a whole-profile offset on a cast where
the bottom-track figure shows the two tracks disagreeing — on very shallow casts the
firmware samples can be few and bad.
Recipe: rerun with `--botfac 0` — the GPS-barotropic row carries the reference
instead. Compare both runs; if they differ materially, the bottom track was the
problem. Report the botfac-0 version, labelled
(`ladcp-compare --alt-dir ... --alt-label "botfac=0"` keeps it honest in comparisons).

**A contaminated cell right above the seabed (shelf casts).**
Symptom: one hot bin at the profile bottom on a shallow cast; happens when a small
bottom-depth underestimate lets a side-lobe-contaminated cell through the edit.
Recipe: raise the near-seabed rejection margin — `--dzbelow 24` (default 16 m = two
legacy bins; go 24–32). Costs you the deepest bin or two; that's the trade.

**A rigid target hung below the package.**
Symptom: `nearfield_errvel_ratio` WARN across consecutive stations; a band of bad
velocity at fixed *range* (not fixed depth) in the down-looker.
Recipe: mask the affected bins — `--nearfield-dn-bins 3,4` (1-based; geometry: each
bin is `dz` metres, so a target at ~26 m with 8-m bins sits in bins 3–4; the WARN's
note names the hot bin for you). The mask is always your explicit call — no cruise
preset applies it silently — so for a whole-leg device, pass the flag for the whole
batch run. In Studio, the near-field toggle takes bins or a depth range (`22-38m`).

**Weakly constrained shallow cast (`shallow_cast` WARN).**
Don't fight it with weights — a 90-m cast yields a handful of super-ensembles and no
amount of tuning makes it a deep profile. Verify the reference (weights figure: who
carries it?), consider `--botfac 0` *or* default per the bottom-track figure, and set
expectations in the cruise report.

**"The two solvers disagree."**
Run `--solver shear` next to the default. Small differences (1–2 cm/s) are normal.
Large ones localise the problem: disagreement *everywhere* → reference trouble
(constraints); disagreement *in a depth band* → data/editing trouble there (error
figure, edit figure).

!!! warning "What not to do"
    Don't raise `--smoofac` to make a profile "look nicer" for a deliverable — it
    removes real shear along with the noise, and the figures will no longer show what
    the instrument measured. Smooth downstream, in analysis, where it's reversible.

# 1 · LADCP primer

If you already process LADCP data, skip to [chapter 4](04-first-station.md). This is
the four-page version of the theory — just enough to make the processing decisions in
chapters 6–7 with understanding rather than superstition.

## What the instrument measures

A lowered ADCP is one or two self-contained Acoustic Doppler Current Profilers
strapped to the CTD rosette — typically a **down-looker** and an **up-looker**
(in pyladcp: *master* and *slave*). As the package descends and ascends, each head
measures the Doppler shift of sound scattered by plankton and particles in ~8-m bins
out to ~100–170 m from the package.

The catch: each ping measures the velocity of the water **relative to the package**,
and the package itself is moving — swinging below a drifting ship while being lowered
at ~1 m/s. The package motion is large compared to the ocean currents you're after,
and no sensor measures it directly at depth. *Removing the unknown package motion is
the entire LADCP processing problem.*

## The shear method

The classic trick (Fischer & Visbeck, 1993): differentiate each ping's velocity
profile in depth. The package velocity is the same for every bin of one ping, so it
**cancels in the vertical shear**. Average the shear of thousands of pings into depth
bins, then integrate from bottom to top — and you have the *shape* of the velocity
profile (the **baroclinic** part), with one unknown left: the integration constant
(the depth-mean, or **barotropic** part).

The shear method's weakness follows from its trick: errors integrate. A small bias in
the shear becomes a slowly growing velocity error over the water column.

## The inverse method

Visbeck (2002) recast the problem as one big least-squares system — the **LDEO_IX
inverse** that pyladcp re-implements, and its default solver. Each measured cell
contributes one equation:

$$u_{ocean}(z_{cell}) + u_{package}(t_{ping}) = u_{measured}$$

with *both* the ocean profile and the package-velocity time series as unknowns.
Because each ping links one package velocity to several depths, and the package
visits each depth twice (down- and upcast), the system is heavily overdetermined —
**except** for one degeneracy: you can add a constant to every ocean velocity and
subtract it from every package velocity. That's the barotropic unknown again, now
explicit, and it is fixed by adding **constraint equations**:

| constraint | what it is | where it acts |
|---|---|---|
| **bottom track** | velocity over ground from the seabed echo, when the down-looker is within range | near the bottom — the strongest single reference |
| **GPS-barotropic** | the ship+package displacement over the whole cast | the depth-mean of the profile |
| **ship-ADCP** | the hull ADCP's absolute currents during the cast window | the upper ocean, over the hull instrument's range |

The three references overlap in what they pin but act at different depths — which is
why a deep cast with all three is robust, and a shallow cast with a bad bottom track
needs attention ([chapter 7](07-solvers-weights.md)).

pyladcp ships both solvers. They share all the data preparation and differ only in
the final solve, so their agreement is a powerful sanity check (on the validation
cruises: 1–2 cm/s).

## From pings to a profile — the pipeline

What actually happens between a raw PD0 file and a `.lad` profile:

1. **Ingest** — decode the PD0s, pair and time-merge the two heads, rotate
   beam-coordinate data to earth coordinates, apply the magnetic declination
   (IGRF-13 from the cast position).
2. **Screen/edit** — remove cells failing percent-good, error-velocity, tilt and
   speed limits; mask side-lobe and below-bottom contamination
   ([chapter 6](06-qa-report.md), edit figure).
3. **Synchronize** — align the LADCP and CTD clocks via the vertical velocity both
   instruments saw; the CTD provides pressure (true depth) and position.
4. **Super-ensembles** — average pings into packets at a common vertical spacing;
   detect the seabed from the bottom echoes.
5. **Solve** — the inverse (or shear) with the constraints above.
6. **QA** — score every stage; render the report ([chapter 6](06-qa-report.md)).

## What good data looks like

A healthy cast has: all four beams per head performing within ~20% of each other;
package tilt mostly under ~10°; a clean descent/ascent depth "V"; bottom echoes
clustering tightly at the seabed; and the two independent bottom tracks agreeing.
Each of those statements is a scorecard row, and [chapter 6](06-qa-report.md) covers
the ways they fail.

Typical error magnitude, done right: a few cm/s over the full water column — set by
ping noise, the constraint quality, and ocean variability during the ~2 h cast
(the water *does* change while you sample it; Thurnherr (2010) treats the error
budget rigorously).

## Where to read the real derivations

- Fischer, J., & Visbeck, M. (1993). Deep velocity profiling with self-contained
  ADCPs. *J. Atmos. Oceanic Technol.*, 10, 764–773 — the shear method.
- Visbeck, M. (2002). Deep velocity profiling using Lowered Acoustic Doppler Current
  Profilers: Bottom track and inverse solutions. *J. Atmos. Oceanic Technol.*, 19,
  794–807 — the inverse; the paper pyladcp implements.
- Thurnherr, A. M. (2010). A practical assessment of the errors associated with
  full-depth LADCP profiles. *J. Atmos. Oceanic Technol.*, 27, 1215–1227 — what the
  error bars mean.

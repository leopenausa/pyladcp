# `src/ladcp/` — source-tree onramp

A reading guide for anyone opening the code: experienced LDEO_IX users coming to
review the port, and oceanographers / grad students meeting the pipeline for the
first time. It does **not** repeat the science — the inline docstrings do that,
and they are the authority. This page only tells you *where to start*, *what each
module is for*, and *which legacy step it corresponds to*.

If a term here is unfamiliar (super-ensemble, baroclinic, bottom track, shear vs
inverse, declination…), the **[guide glossary](../../docs/guide/appendix-e-glossary.md)**
defines every one in a sentence. For the design contract (units, frames, data
objects) see **[docs/design/ARCHITECTURE.md](../../docs/design/ARCHITECTURE.md)**; for the knob-
and-output mapping from the MATLAB toolbox see
**[appendix B · LDEO_IX ↔ pyladcp](../../docs/guide/appendix-b-legacy-map.md)**.

## Conventions used everywhere

- **SI units.** Velocity in m/s, depth in metres, time as UTC `datetime64[ns]`.
- **Horizontal velocity is split** into `u` (east) and `v` (north). The MATLAB
  code packs them as complex `u + iv`; we keep them as two real arrays.
- **Depth `z` is positive-down.** `NaN` always means "missing / edited out".
- **Frames are named at the point they change.** Velocities are *magnetic*
  east/north until the magnetic declination (`drot`) rotates them to *true* in
  the solver — docstrings say which frame an array is in.
- Most docstrings cite the legacy file they port (`prepinv.m`, `getshear2.m`,
  `b2earth`, …). That citation is provenance for reviewers; you do **not** need
  the MATLAB source to follow the Python — the surrounding comment explains the
  intent in plain terms.

## The shortest path through the pipeline

A single station flows raw files → quality-screened ensembles → super-ensembles →
a velocity solve → outputs. Read in this order to follow one cast end to end:

1. **`models.py`** — the data objects every stage passes around (`RawADCP`,
   `DualHead`, profiles…). Start here so the later signatures read clearly.
2. **`io/pd0.py`** — decode one RDI Workhorse `.000` (PD0) binary file per head.
3. **`qa/ingest.py`** — pair the down-looker (Master) and up-looker (Slave) into
   one `DualHead`, the object the rest of the pipeline consumes.
4. **`qa/screen.py`** — throw out bad cells/pings (percent-good, error velocity,
   tilt…), reproducing the legacy `loadrdi.m` edits.
5. **`qa/depth.py`** — work out where the package was through the cast by
   matching the LADCP clock to the CTD pressure record (the LADCP clock is not
   GPS-set, so this time-sync is load-bearing).
6. **`qa/bottom.py`** — find the seabed from the echo and form the bottom-track
   reference (the strongest absolute velocity anchor, near the bottom).
7. **`qa/superens.py`** — average consecutive pings into **super-ensembles**, the
   unit the solvers actually see, and derive the dual-head compass offset.
8. **`qa/inverse.py`** (shear method) and **`qa/inverse_full.py`** (full
   least-squares inverse) — turn super-ensembles into the ocean velocity profile.
   The inverse is the default solver; the shear method is the robust cross-check.
9. **`qa/report.py`** → **`qa/export.py`** / `export/` — score the cast (the QA
   traffic-light report) and write the products (`.lad`/`.bot`, NetCDF, figures).

`qa/pipeline.py` wires all of the above together (`process_station` runs the stages
in order) — read it last, as the table of contents. `qa/cli.py` is only the
`ladcp-qa` front end: argument parsing, work-list resolution, and the worker pool.

## Module map by role

Grouped by job, with the legacy step each one ports. "Phase N" labels in the
docstrings are this project's internal stage names; the numbers below are the
legacy LDEO_IX 18-step order (see ARCHITECTURE.md §1).

### Input / ingest — `io/`
| module | role | legacy |
|---|---|---|
| `io/pd0.py` | read one RDI PD0 (`.000`) file; auto-detect beam vs earth frame | `loadrdi` (1) |
| `io/beam2earth.py` | rotate per-beam radial velocities to earth coords | `b2earth` (2) |
| `io/nav.py` | ship GPS track + SADCP clock-desync correction | (3) |
| `io/ctd_cnv.py` · `io/ctd_hex.py` · `io/ctd_raw.py` | read the cleaned CTD profile / time-series (and raw `.hex` path) | (5,6) |
| `io/sadcp_vmdas.py` · `io/sadcp_codas.py` | read shipboard-ADCP data for the optional upper-ocean constraint | (13) |

### Per-cast processing — `qa/`
| module | role | legacy |
|---|---|---|
| `qa/ingest.py` | pair Master/Slave heads → `DualHead` | (1) |
| `qa/screen.py` | per-cell / per-ensemble threshold rejection | part of `loadrdi` (9) |
| `qa/attitude.py` | tilt / heading / transmit-voltage engineering health | (8) |
| `qa/range.py` · `qa/beams.py` | profiling range and per-beam signal health (from `plotraw`) | (8) |
| `qa/depth.py` | CTD↔LADCP time sync + package depth trajectory | (7) |
| `qa/bestlag.py` | robust integer lag between two ~1 Hz series (used by the sync) | `bestlag` |
| `qa/bottom.py` | seabed detection + bottom-track reference | `getbtrack`/`getdpthi` (4,7) |
| `qa/superens.py` | super-ensemble averaging + dual-head compass offset | `prepinv` (10) |
| `qa/edit.py` | bin masking + side-lobe wedge editing | `edit_data` (9,11) |
| `qa/inverse.py` | shear-method baroclinic profile + barotropic reference | `getshear2` (15) |
| `qa/inverse_full.py` | full sparse least-squares inverse (default solver) | `getinv` (14) |
| `qa/stats.py` | small statistical helpers matched bit-for-bit to legacy | — |
| `qa/checks.py` | post-solve consistency checks | `checkinv` (17) |

### Output, QA & validation
| module | role |
|---|---|
| `qa/report.py` | assemble the per-cast QA scorecard + text report |
| `qa/export.py` | write the LDEO `.lad` / `.bot` text profiles |
| `export/` | NetCDF (`netcdf.py`), Excel (`xlsx.py`), ODV (`odv.py`), shared column metadata (`tables.py`) |
| `plots/` | the QA figure set and the combined report PDF (`pdf_report.py`) |
| `qa/validate.py` · `qa/golden.py` · `qa/compare.py` | cross-validate solver output against the legacy "golden" results |

### Glue & configuration
| module | role |
|---|---|
| `config.py` | `CastParams` — the per-cast knobs (mirrors legacy `p`/`ps`) |
| `session.py` | reproducible solve configuration + cached engine used by Studio |
| `discovery.py` | find a station's input files under a cruise directory |
| `edits.py` | record/replay manual brush edits as a per-station journal |
| `archive.py` · `archive_cli.py` | the station↔raw-file index (`ladcp-index`) |
| `proc/magdec.py` | magnetic declination from the IGRF model (replaces `magdev.m`) |
| `qa/pipeline.py` | the per-station pipeline (`process_station`: ingest → … → export) |
| `qa/cli.py` · `qa/runlog.py` | the `ladcp-qa` front end (args, batch, pool) + its logging |
| `studio/` | the optional interactive editing GUI (`ladcp-studio`) |

## Command-line entry points

| command | module | what it does |
|---|---|---|
| `ladcp-qa` | `qa/cli.py` | process one or more stations → report + products |
| `ladcp-index` | `archive_cli.py` | build/query the station↔raw-file index |
| `ladcp-compare` | `qa/compare.py` | diff pyladcp output against legacy `.mat`/`.lad` |
| `ladcp-sadcp-section` | `plots/sadcp_section.py` | plot a shipboard-ADCP section |
| `ladcp-studio` | `studio/server.py` | launch the interactive Studio GUI |

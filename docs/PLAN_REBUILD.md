# LADCP pipeline — rebuild plan (clean slate, GO-SHIP golden)

**Date:** 2026-06-04. **Decision:** restart the processing build from first principles.

## Why restart
- The **MORIA 2025** dataset we were validating against has serious quality problems (e.g. the
  station-25 series: same raw LADCP file producing inconsistent profiles, corrupt casts; only
  MORIA-05 ever reproduced cleanly). Chasing it produced no durable progress.
- New **golden standard = GO-SHIP RB1606 / P18** LADCP profiles — community-produced, archived at
  NCEI, processed by A. Thurnherr with **LDEO_IX IX_13beta**. See `test_data/goship/README.md`.

## Principles
1. **The MATLAB is the authority.** Re-derive each stage from `legacy_code/vbIX/LDEO_IX-636b06141b2e/`
   (`process_cast.m` = the 18-step driver; `loadrdi/getdpth/getbtrack/loadctd/loadnav/edit_data/
   prepinv/getinv`). Do **not** port from our previous MORIA-tuned modules — they may encode wrong,
   MORIA-specific assumptions.
2. **The golden `.nc` is unusually rich** — its attributes embed the *complete parameter struct* AND a
   *step-by-step processing log* (counts removed per step, time lags, bottom depth, super-ensemble
   count, velerr, constraint weights). So we can validate **every intermediate stage**, not just final
   u/v. This is the biggest lever: build one stage, diff its numbers against the log, move on.
3. **Validate per-stage, per-station.** Start with the quiet deep stations; use equatorial **063**
   (|u|≈1.3 m/s, strong shear) as the stress test.
4. **Reuse only format/physics-level code** (see below). Everything dynamical gets rebuilt + revalidated.

## Reuse vs rebuild
**Safe to reuse (format/physics, low bias risk — but re-verify against `.nc`):**
- `io/pd0.py` — PD0 reader. Proven on RB1606 150 kHz DL + 300 kHz UL; has the orientation fix
  (filename/hint > sysconfig bit). Keep.
- `proc/magdec.py` — IGRF declination (note: golden used `magdec`/`magdev`; check which, and the sign).
- Dataclass/`validate` *framework* concept (Check/Report) — reusable, but retarget at the `.nc`.

**Rebuild fresh from MATLAB (do NOT copy `proc/inversion.py`, `prepare.py`, `depth.py`, `pipeline.py`,
`io/ctd.py`, `io/golden.py` — these are MORIA-shaped):**
- **CTD/nav loader** — the GO-SHIP input is a full Sea-Bird **24 Hz** `.cnv` with **per-scan lat/lon**
  (cols timeJ, timeS, prDM, t090C, sal00, latitude, longitude). Build `loadctd`+`loadnav`: decimate to
  ~1 Hz, despike, derive depth from pressure, best-lag to ADCP (`besttlag`). MORIA gave us a pre-cleaned
  6-col file — that loader does not apply here.
- **Golden `.nc` reader** — parse u/v/z/ev + the param struct + the log. This is the validation target.
- **Ingestion → earth velocity → depth/bottom → edit → super-ensembles** — from `loadrdi`, `getdpth`,
  `edit_data`, `prepinv`. Validate intermediate counts vs the `.nc` log.
- **Inverse solve** — from `getinv` (weighted L2; smoothing, bottom-track, barotropic, zero-mean).
- **Bottom track** — `getbtrack` builds its **own** bottom track from target-strength echo maxima in
  addition to RDI (the RB1606-008 log: "created 205 bottom track data"). We never built this; it matters.
- **SADCP** — `loadsadcp` + the SADCP constraint. **Required here** (all 5 stations used it; it
  constrains the upper ~500 m). Input = CODAS os75nb `.nc` in `sadcp/`.
- **Refinements** — `lanarrow` 1% outlier loop (STEP 11) + two-pass velerr (we have faithful versions
  to re-derive, but rebuild against the log, don't copy).

## Config (from the golden `.nc` param struct — same across the 5; e.g. station 008)
`software=IX_13beta, dz=8, btrk_mode=2, btrk_range=[300 50], btrk_wlim/wstd≈0.05/0.067,
beamangle=20, weightmin=0.05, nav_error=30, outlier=1, smallfac=[1 0] (small-shear OFF),
smoofac=0, barofac=1, botfac=1, sadcpfac=1, getdepth=1 (integrate W), up_dn_looker=1,
pglim=0, elim=0.5, vlim=2.5, wlim=0.2, tiltmax=[22 4]`.
Each station's exact struct is in its own `.nc` attributes — **read it, don't hardcode**.

## Build phases (suggested order for next session)
1. **Golden `.nc` reader** + a tiny per-station "expected values" table scraped from the log
   (lag, bottom depth, n super-ens, velerr, removed-counts). Cheap, unlocks all later validation.
2. **CTD/nav loader** (Sea-Bird 24 Hz + per-scan nav). Validate decimated P/T/S + track vs `.nc`
   `ctd_t/ctd_s/zctd/xship/yship`.
3. **Raw → earth velocity + depth/bottom + edit + super-ensembles.** Validate counts vs log.
4. **Inverse solve** (no SADCP yet; Nav+BT). Validate u/v on quiet stations.
5. **Own bottom track** (`getbtrack` target-strength). 6. **SADCP** load+constraint (closes upper ocean).
7. **lanarrow + two-pass velerr.** Then sweep all 5 + add cross-instrument cruises.

## Validation criteria
Per station: stage counts within the log's values; final u/v vs golden within a few cm/s (use the
golden `ev`/`velerr` as the natural error scale, ~0.02–0.16 m/s). Equatorial 063 is the shear stress test.

## Status of old code
`src/ladcp/` (MORIA build) stays in the tree as **reference only**. Build new modules validated against
the `.nc`; pull a line over from the old tree only after confirming it matches the MATLAB + `.nc`, never
wholesale. The MORIA harness/fixtures are not the target anymore.

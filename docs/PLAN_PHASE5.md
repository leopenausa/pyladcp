# Phase 5 — Velocity Stage (LADCP inverse)

> Status: planning → **5a in progress**. Builds on the validated QA-first acquisition
> stage (`src/ladcp/qa/`). Velocity was deferred during the 2026-06-05 QA-first pivot;
> re-opened 2026-06-05 after the acquisition stage + PDF report shipped.

**Goal:** turn the validated QA foundation into LDEO-IX velocity profiles, validated
against the MORIA-80 golden `.lad` / `.bot` / `dr`-struct.

**Authority:**
- Legacy MATLAB: `legacy_code/vbIX/LDEO_IX-636b06141b2e/{prepinv,getinv,getshear2,checkinv,getbtrack}.m`
- Golden trace: `New_golden/MORIA-80_LEO/MORIA-80.log` (steps 10 = prepinv, 14 = getinv/checkinv)
- Golden targets: `New_golden/MORIA-80_LEO/MORIA-80.{lad,bot,mat}` (`.lad` u/v/ev, `.bot`
  bottom-track, `dr`-struct in `.mat`)

**Reuse policy:** clean-slate under `src/ladcp/qa/` (velocity-adjacent). Old `src/ladcp/ix/`
is set aside but readable for reference — do NOT port blindly (it was MORIA-mistuned and
carried bugs, e.g. the numpy banker's-rounding grouping hang).

---

## 5a — Super-ensembles (`prepinv.m`)  ← START HERE
- New module `src/ladcp/qa/superens.py`.
- Group raw ensembles into super-ensembles.
- **Validate reduced count = 218** (golden log step 10: "reduced ensemble size is 218").
- Extract the **exact dual-head compass offset**: golden velocity offset **−60.23°**
  (tilt −59.57°). Resolves the `rotup2down` question; QA-stage estimate −58.26° is close
  but not exact (needs super-ensembles).
- Watch-out: MATLAB round-half-away vs numpy banker's rounding — use `floor(x+0.5)`
  (the old ix port hit an infinite grouping loop here).

## 5b — Velocity merge + deferred corrections
- Merge up/down into earth-frame `l.u`/`l.v` (enables exact hspeed count + middle-hour=358).
- **Magnetic declination decision:** product uses correct IGRF-13 (−0.45°); validation
  feeds golden −2.5456° via `params.drot` so the inversion diffs cleanly without a 2°
  rotation masking other differences. (see `ladcp-magdec-finding`)
- **Velocity bin-length sound-speed correction** (`getdpthi` 428–438): rescale the `d.izm`
  bin-depth grid by `ss/sv`. (The scalar `zbottom` correction shipped in QA stage; this
  grid correction is the velocity-side counterpart.)

## 5c — Inverse solver (`getinv.m` + `getshear2.m`)
- Least-squares inversion: shear solution + barotropic reference.
- Known divergences to honor (`ladcp-legacy-vs-python-audit`): `lanarrow`, two-pass `velerr`.
- Validate `u`/`v`/`uerr` vs golden `.lad` and the `dr`-struct.

## 5d — Bottom-track velocities (`getbtrack` velocity branch)  ← DONE
- Geometry already done in QA stage; add bottom-track velocities → golden `.bot`
  (`checkbtrk`, log step 14: rms 0.016 < 0.084).
- **Shipped:** `bottom_track_velocity` (`qa/bottom.py`, per-ping `bvel = -u_package` from
  the seabed cell, W-ref + boutlier reject), `bottom_referenced_profile` + `_btrk_reference`
  (`qa/inverse.py`, the `lainbott` profile + reference branch), `write_bot` (`qa/export.py`).
- **Key gotcha:** the reference/`.bot` MUST use the *raw* `merged.ru` per-ping, NOT the
  super-ensemble `se.ru` — `se.ru` has the per-ping reference velocity removed, which cancels
  the package term and breaks `ru - bvel` (gave a ~0.06 m/s offset; per-ping gives 0.026).
- **Geom filter (prepinv STEP 10):** keep bottom track only where the package is 50–300 m
  above the *known* seabed and |height − hbot| < 100 m — drops mid-water false echoes.
- **Reference = equal-weight-per-depth-bin** of (bottom-profile − baroclinic), not per-cell
  (per-cell over-weights the cell-dense up-looker band).
- **Validated vs MORIA-80:** `.bot` U corr 0.991 / rms 0.026, V corr 0.92 / rms 0.013;
  `.lad` U corr 0.998 / rms 0.0165 (was 0.025); ubar −0.051 vs golden −0.065 (was −0.041).
- The old handoff note "bottom-track pulled ubar the wrong way" is SUPERSEDED — that was a
  flawed earlier attempt; this faithful one clearly improves ubar.
- Residual ~0.014 ubar gap = ~0.02 m/s DC offset in our `bvel` vs golden = the editing-weight
  fidelity wall (golden raw `da` truncated). Tests: `tests/test_qa_inverse.py` (4 new, 13 total).

## 5e — Figures 1/3/12  ← DONE
- Fig 1 (velocity profile) `plots/velocity_figure.py` (+ bottom-track dot overlay).
- Fig 3 (shear) `plots/shear_figure.py`: ∂u/∂z, ∂v/∂z + integrated baroclinic + sample count.
- Fig 12 (inversion diagnostics) `plots/inverse_figure.py`: baroclinic→absolute decomposition,
  per-cell fit residual vs depth (rms ~0.086 m/s), residual distribution. NB legacy Fig 12 is
  the full-inverse *constraint weights* (`checkinv.m`) — N/A to our reduced shear+ref solve, so
  this is a modern-equivalent solution-quality figure, not a port.
- All `fig=`-injectable; folded into the PDF report (now 8 pages: scorecard + raw/align/depth/
  edit + velocity/shear/inverse). `compute_velocity_full` returns a `VelocityResult`
  (vp + bp + shear + residual diagnostics) consumed by CLI + `build_report(..., velocity=)`.

## 5f — Export + end-to-end validation  ← DONE (profile-level)
- `write_lad` / `write_bot` shipped; `.lad`/`.bot` validated vs golden (see 5c/5d). Full
  `dr`-struct assembly not built (text products + figures are the deliverables); profile-level
  agreement only (editing-weight fidelity wall). 160 tests pass; qa_out/ regenerated for 79/80/82.

---

## Known risks / fidelity walls
- Golden raw weight archive `da` is **truncated** → no array-level golden for editing
  weights; exact edit/hspeed counts stay *indicative*. Velocity profiles (`.lad`) and
  `dr` ARE validatable.
- Inversion is sensitive to the editing weight pattern we can't fully reproduce → expect
  profile-level (not bit-level) agreement.

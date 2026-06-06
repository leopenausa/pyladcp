# Validation Harness Spec — MORIA-05 (DRAFT v0.1)

Status: **draft for review.** No code yet. Companion to `DATA_CONTRACT.md`.
Purpose: define the single-cast harness that proves the Python inverse core reproduces
the legacy LDEO_IX (IX_10) result for one clean cast, before generalising.

---

## 1. Objective & scope

Reproduce the golden **MORIA-05** absolute-velocity profile with the Python re-implementation
and show it matches legacy within tolerance (§6). MORIA-05 chosen because: deep/full-depth
(~4190 m), clean 1:1 file mapping, fresh battery, stable earth-coord config, **dual-head**.

**Active constraints for this cast (from golden `.log`):** LADCP (down+up) + CTD time-series +
GPS (barotropic) + **bottom-track**. **SADCP: not used** — the combined `.mat` failed the legacy
position check cruise-wide, so it never entered any golden inversion. The harness therefore does
not require SADCP, but treats its absence as an explicit recorded state, not a silent skip (§7).

Out of scope here: deployment→cast cutting of multi-dump casts (01–04), SADCP combine, batch/UI.

---

## 2. Fixed inputs (all present on disk)

| Role | Path | Notes |
|------|------|-------|
| LADCP downlooker (master) | `raw_ladcp_test/LADCP/Data/MASTER/MLADC007.000` | RDI WH300, earth coords, 30×8 m |
| LADCP uplooker (slave) | `raw_ladcp_test/LADCP/Data/SLAVE/SLADC008.000` | uplooker |
| CTD+GPS time series | `clean_ctd/moria-05_clean.cnv` | 6 col, 1 Hz, elapsed-time base |
| Bottom-track | (decoded from master PD0) | RDI BT mode |
| Magnetic declination | computed (IGRF, cast lat/lon/date) | target −2.5456° (§5) |
| SADCP | none supplied | recorded as "not used" |
| **Golden reference** | `figures/MORIA-05.{lad,bot,mat,log}` | comparison target |

Pre-step needed: **trim** the deployment file to the in-water cast window (legacy `_good` cut).
For 05 the mapping is 1 dump→1 cast, so trimming = drop pre/post-deck ensembles (surface-detect).

---

## 3. Effective parameters for station 05

Resolved from `set_cast_params.m` (none of the per-leg/per-stn override `if` branches match stn 05)
merged over `default.m`. The harness must apply these exact values:

```
ps.up_dn_looker = 1          # both heads
ps.dz           = 8          # m, inversion grid
p.edit_mask_dn_bins = [1]    # drop downlooker bin 1
p.edit_mask_up_bins = [1]    # drop uplooker bin 1
p.cut    = 7                 # rosette/draft trim
p.pglim  = 50                # %good minimum
p.elim   = 0.2               # error-velocity limit
p.vlim   = 1.0               # horizontal velocity limit
p.wlim   = 0.08              # vertical velocity limit
p.tiltmax = [22 4]
p.btrk_mode = 3              # bottom track on
p.timoff = 0                 # no clock offset for stn 05
p.drot   = NaN -> IGRF       # declination computed from position/date
p.weighbin1 = 0.1
ps.sadcpfac = 3 (declared)   # but no SADCP supplied -> inert this cast
p.sadcp     = 75 (declared)
getdepth = 2                 # depth from W-integration + bottom reflection (default)
```

CTD/nav field map (from `set_cast_params.m`): `ctd_fields_per_line=6`,
`pressure=3, temperature=5, salinity=6, time=4 (elapsed base 0)`; `nav_lat=1, nav_lon=2`.

---

## 4. Processing path exercised (legacy 18-step subset)

Active for MORIA-05: 1 load LADCP (+magdev) · 2 fix beams/compass · 3 load GPS · 4 bottom-track
· (5 CTD profile: none) · 6 CTD time-series · 7 surface/seabed/depth · 8 pitch/roll · 9 edit
· 10 super-ensembles · 11 outlier removal · 12 re-form · (13 SADCP: inert) · **14 inverse**
· 15 shear (cross-check) · 16 CTD merge · 17 plot/warn · 18 save `.lad/.bot/.mat/.log`.

Numeric-risk notes to watch during bring-up:
- **Coordinate frame:** stn 05 is earth-coords (EX11111) → reader applies declination rotation only.
- **Solver:** legacy uses L2 Moore-Penrose normal equations (`lesqfit`), not L1 — replicate L2.
- **Random-outlier removal** (`lanarrow`, top-1%): order/threshold must match or diffs appear as
  small scatter; pin the rule deterministically (no RNG).
- **Dual-head compass merge:** "rot up2down use mean up/down compass" — must merge before inversion.

---

## 5. Golden reference anchors (from MORIA-05.{lad,bot,log})

Scalars (checkpoints):

| Quantity | Golden value | Source |
|---|---|---|
| Date / start time | 2025-09-18 05:35:29 UTC | .lad |
| Start lat / lon | 46°29.1216′N / 5°51.5136′W | .lad |
| Magnetic deviation | **−2.545590°** (log: −2.5) | .lad/.log |
| Inversion grid dz | 8 m | .log |
| Depth grid extent | 8 … 4184 m (523 bins) | .lad |
| Bottom depth | 4196 m | .bot |
| CTD max depth | 4189 m | .log |
| Finite RDI bottom velocities | 598 | .log |
| Bottom-track ensembles used | 33 | .log |
| Barotropic velocity error | 0.005176 m/s | .log |
| Super-ensemble velocity error | 0.022528 m/s | .log |

Profile arrays: `MORIA-05.lad` columns `z u v ev` (523 rows); `MORIA-05.bot` columns `z u v err`
(bottom-referenced, 31 rows from 3952 m down). Full `dr` struct in `MORIA-05.mat`.

---

## 6. Comparison method & tolerances

Run Python → emit `ProfileResult` + legacy-format `.lad`/`.bot` (parity kept). Then:

1. **Declination** must match to ≤ 0.01° (else u/v rotate). *Gate.*
2. **Depth grid** identical (8 m, same z vector).
3. **Velocity profile** `u`,`v` on common z: report `max|Δ|`, `median|Δ|`, RMS.
   **Pass:** `median|Δu|, median|Δv| < 0.01 m/s` (accepted tolerance), `max|Δ| < ~3·uerr`.
4. **Depth-mean** `ubar`,`vbar`: |Δ| < 0.01 m/s.
5. **Bottom-track profile** `zbot/ubot/vbot`: median |Δ| < 0.01 m/s; bottom depth within 1 bin.
6. **Scalars** in §5 table: relative diff < 1% (counts exact where integer).
7. `ev`/`uerr`: compared as a **diagnostic** (not a gate) — error scaling is semi-empirical.

Output of harness: `validation_MORIA05_report.{json,md}` — per-check value, Δ, status; plus an
overlay plot (Python vs golden u,v vs z). Single top-level **PASS/FAIL**.

---

## 7. SADCP design note (future feature — keep, don't reject plainly)

SADCP is a **first-class future input**, not used by MORIA-05 but to be properly handled, not
silently dropped as legacy does. Design now (build later):
- Ingestion contract per `DATA_CONTRACT.md §2.3` (owned VmDAS→combined builder, deferred).
- Position/time-overlap check produces an **explicit QC flag with the measured offset distance**
  (e.g. `sadcp_position_offset_deg`), surfaced in `QCMetrics`, never a silent zero.
- On mismatch, configurable policy: `reject | downweight | operator_override` (default `reject`
  **with loud flag**), replacing legacy's hard >0.1° silent drop.
- For this harness: SADCP simply recorded as `status=not_supplied`; zero effect on the numeric diff.
This keeps the inverse matrix's SADCP constraint slot (`lainsadcp` equivalent) wired and tested-by-
absence, so enabling it on future cruises is a data/weight change, not a re-architecture.

---

## 8. Exit criteria

Harness PASSES when §6 checks 1–6 pass on MORIA-05. On pass, repeat unchanged on **MORIA-06**
(2040 m, different regime) as a confirmation cast. Passing both gates generalisation to
multi-dump casts (01–04), the SADCP feature, and batch/UI work.

## 9. Open confirmations
- IGRF model/epoch to match legacy `magdev.m` declination (−2.5456°): confirm coefficient set
  (IGRF-13/14) and that 0.01° agreement is achievable; declination is a gate.
- Whether to diff against `.lad` (ASCII, rounded to 0.001 m/s) or `.mat` (full precision) as the
  primary numeric reference — recommend `.mat` for precision, `.lad` for byte-parity of output.
- Confirm MORIA-06 file mapping (MLADC008/SLADC009) and that its `.log` also shows no SADCP.

# LADCP Python Pipeline — Data Contract (DRAFT v0.1)

Status: **draft for review**. No code yet. This defines the interfaces that let the
back-end (processing) and front-end (QA review) be built against a stable boundary.
Derived from the MORIA 2025 legacy LDEO_IX (IX_10) pipeline + decoded raw data.

Conventions: SI units. Horizontal velocity carried as **separate `u` (east), `v` (north)**
floats (the MATLAB code packs them as complex `u+iv`; we keep them split). Depth `z`
positive-down in metres. Time as timezone-aware UTC (`datetime64[ns]`, UTC).
NaN = missing. Arrays described as `[dim]`.

---

## 1. Pipeline stages & ownership

```
   raw PD0 (M/S deployment dumps)          clean CTD .cnv (+merged GPS)     raw VmDAS sADCP (75/150kHz)
            |  OWNED                                | INGESTED                        |  OWNED
   [S0] cut deployment -> per-cast              (produced by CTD pipeline)    [S0'] combine -> sadcp.nc
            |                                        |                                |
            +--------------------+-------------------+----------------+---------------+
                                 v
            [S1..S17] PER-CAST PROCESSING (owned)  ── inverse method core ──> ProfileResult + QCMetrics
                                 v
            [S18] OUTPUT: profile.nc / .lad / qc.json / figures
```

- **Owned by this tool:** S0 LADCP deployment→cast cutting; S0' sADCP VmDAS→combined; all per-cast
  processing S1–S18; QA metrics & report.
- **Ingested (consumed as-is):** cleaned CTD `.cnv` with GPS merged (CTD pipeline produces these).
- **Deferred / pluggable:** shear-method solver (architecture must allow it; inverse is the core).

Legacy 18-step order (kept as internal stage names): 1 load LADCP · 2 fix beams/compass · 3 load GPS
· 4 bottom-track · 5 CTD profile · 6 CTD time-series · 7 surface/seabed/depth · 8 pitch/roll · 9 edit
· 10 super-ensembles · 11 outlier removal · 12 re-form · 13 sADCP · 14 inverse solution · 15 shear
· 16 CTD merge for output · 17 plot/warn · 18 save.

---

## 2. Inputs

### 2.1 Raw LADCP (per head, RDI PD0 / `.000`) — OWNED reader
Decoded facts (MORIA): 300 kHz, 4-beam, WM15, 30 cells × 8 m, blank 176 cm.
Reader MUST expose per ensemble: time(UTC), beam/earth velocities (auto-detect EX coord frame),
echo amplitude, correlation, percent-good, pitch, roll, heading, temperature, bottom-track vel +
range, built-in sound speed, system config (freq, up/down), WN/WS/WF/WM, pings/ens, ping interval.
MUST handle: MASTER=downlooker / SLAVE=uplooker; single-head casts (`up_dn_looker` 1/2/3);
**two coordinate frames** (early beam-coord vs main earth-coord) and **constant-ping-rate PPI**.

### 2.2 Cleaned CTD time-series (`.cnv`) — INGESTED
Whitespace-delimited, **no header** (header_lines=0), 6 columns, 1 Hz:

| col | field | unit | notes |
|-----|-------|------|-------|
| 1 | latitude | deg | GPS, merged in |
| 2 | longitude | deg | GPS, merged in |
| 3 | pressure | dbar | |
| 4 | elapsed time | s | base=elapsed (0); monotonic from cast start |
| 5 | temperature | °C | in-situ |
| 6 | salinity | PSU | practical |

Contract requirements: cols 1–2 provide the **navigation** stream (no separate nav file); time is
**elapsed seconds** matched to LADCP by cross-correlating CTD pressure vs integrated-w depth.
(Parameterised: `ctd_*_field`, `ctd_time_base` — kept configurable for non-MORIA cruises.)

### 2.3 Combined sADCP (`sadcp.nc` / `.mat`) — OWNED builder (from VmDAS)
Contract object `SADCP`: `time[t]` UTC, `lat[t]`, `lon[t]`, `depth[z]`, `u[z,t]`, `v[z,t]`,
`uerr[z,t]`, plus provenance (which freqs combined, averaging window). Used as upper-ocean
constraint; weight `sadcpfac` (0 = comparison-only, no constraint).
**Future-feature, handle explicitly (NOT legacy's silent drop):** position/time-overlap check emits a
QC flag with the measured offset (`sadcp_position_offset_deg`); mismatch policy is configurable
`reject | downweight | operator_override` (default `reject` **+ loud flag**). The inversion's SADCP
constraint slot stays wired even when unused, so enabling it later is a data/weight change only.

### 2.4 Per-cast parameters (`CastParams`)
Mirrors legacy `p`/`ps`. Key knobs the operator sets at sea (defaults in parens):
`up_dn_looker`(1) · `edit_mask_dn_bins`/`edit_mask_up_bins`([1]) · `dz`(8) · `cut`(0) ·
`pglim`(50) · `elim`(0.2) · `vlim`(1.0) · `wlim`(0.08) · `tiltmax`([22,4]) · `btrk_mode`(3) ·
`sadcp`(id)/`sadcpfac` · `drot`(NaN→IGRF) · `timoff`(0) · `getdepth`(2).

---

## 3. `ProfileResult` — core output contract (≈ legacy `dr`)

The single object that the inversion produces and everything downstream consumes.
Grouped; shapes use `nz` (depth grid), `nt` (super-ensembles).

**Identity / metadata**
`station` (str) · `name` (str) · `date` (UTC, mid-cast) · `lat`,`lon` (mean deg) ·
`magnetic_deviation` (deg) · `cruise_id` · `processed_by` · `software_version` ·
`config` (freq, n_heads, coord_frame, dz, constraints_used[]).

**Primary solution** (`[nz]`)
`z` (m) · `p` (dbar) · `u`,`v` (m/s, absolute ocean) · `uerr` (m/s, rescaled) · `nvel` (samples/bin).

**Depth-mean** `ubar`,`vbar` (m/s).

**Per-constraint sub-solutions** (for QC overlay & cross-check)
- bottom-track referenced: `zbot`,`ubot`,`vbot`,`uerrbot`
- sADCP profile used: `z_sadcp`,`u_sadcp`,`v_sadcp`,`uerr_sadcp`
- down/up-cast separate: `u_do`,`v_do`,`u_up`,`v_up` (baroclinic)
- shear-method (when enabled): `u_shear`,`v_shear`,`w_shear`
- `ensemble_vel_err` (single-ping vel error from shear scatter)

**Package & ship kinematics** (`[nt]`)
`tim`(UTC),`tim_hour` · `zctd`,`wctd` · `uctd`,`vctd`,`uctderr` · `xctd`,`yctd` ·
`shiplat`,`shiplon`,`xship`,`yship`,`uship`,`vship`.

**Acoustic diagnostics** (`[nz]`)
`range`,`range_do`,`range_up` (m) · `ts`,`ts_out` (dB target strength, 2nd & last down bin).

**CTD merged for output** (`[nz]`) `ctd_t`,`ctd_s` (interp to z).

---

## 4. `QCMetrics` — acquisition-quality contract (PRIORITY deliverable)

Read-only object computed every cast; drives the QA report and the at-sea go/no-go view.
Each metric: `value`, `unit`, `status` ∈ {ok, warn, fail}, `threshold`, `source_stage`.

**A. Engineering / acquisition** (from raw + ProcFig2)
- `tilt_mean`,`tilt_max` (deg) — fail if > `tiltmax`
- `heading_ok` (bool/jumps) · `pitch_roll_range`
- `echo_amplitude_profile[bin]` (dB, per beam) · `correlation_profile[bin]`
- `pct_good_profile[bin]`
- `beam_solution` (4-beam vs 3-beam fraction) — 3-beam ⇒ no error velocity ⇒ warn
- `battery_voltage` trend (xmv) · `pings_per_ensemble` · `coord_frame` · `bins_masked`

**B. Geometry / coverage**
- `max_depth` (m) · `bottom_detected` (bool) · `height_above_bottom_min` (m)
- `n_superensembles` · `dz` · `profile_gap_pct` (Σdt where dt>3·mean) — large ⇒ barotropic weight down
- `acoustic_range_down`,`acoustic_range_up` (m) — fail if < ~60 m (errors blow up)

**C. Inversion quality**
- `velerr` (super-ensemble velocity error, m/s)
- `uerr_median` (m/s)
- `residual_u_std`,`residual_v_std` (m/s, from `geterr`) — headline scatter metric
- `residual_bias_near_bins` (should be ~0)
- `shear_inverse_diff` — large ⇒ clock/tilt problem (→ check `timoff`)
- `updown_bias_u`,`updown_bias_v` (m/s) — warn if |·| > 0.02 (GPS problems)
- `ocean_velocity_banding` (horizontal=good / slanted=bad; from ProcFig3 structure)

**D. Constraint diagnostics**
- `constraints_active` {gps, bottom_track, sadcp} + weights (`barofac`,`botfac`,`sadcpfac`)
- `bottom_track_bias` (checkbtrk) — large ⇒ sloping-seabed sidelobe; suggest `btrk_mode=0`
- `sadcp_rms_discrepancy` (m/s, LADCP−SADCP upper ocean) — **the cross-instrument accuracy metric**
- `sadcp_profiles_removed` (low weight count)

**E. Magnetic**
- `declination_deg` · `declination_source` {IGRF-computed, manual} —
  **NaN ⇒ critical:** legacy auto-zeroes `sadcpfac`,`barofac` (shear-only), must surface as fail.

**F. Warnings** `warnings[]` (free text, ≈ legacy `p.warn`/`p.warnp`) · `overall_status`.

---

## 5. Output artifacts (S18)

- `MORIA-XX.nc` — `ProfileResult` as CF/NetCDF (xarray). Primary product.
- `MORIA-XX.lad` — legacy ASCII (`z u v ev` + header lat/lon/deviation) for byte-level diffing vs golden.
- `MORIA-XX_qc.json` — `QCMetrics` (front-end reads this).
- `MORIA-XX.bot` — bottom-track profile (legacy parity).
- `MORIA-XX.log` — processing log + warnings.
- figures — QA figure set (parity targets: ProcFig 1,2,3,4; ours may restyle).

---

## 6. Validation contract (proves the re-implementation)

For each golden cast, compare Python vs legacy on the **`ProfileResult`** interface:
- `u`,`v` on common `z` grid: report max & RMS abs diff (target: ≪ `uerr`, e.g. < 1 cm/s median).
- `ubar`,`vbar`, `magnetic_deviation`, `zbot`/`ubot` near-bottom.
- Golden sources: `figures/MORIA-XX.lad` (+ `.mat`, `.bot`). Definitive refs to come.
Golden-cast diff is the gate before generalising beyond one cast.

---

## 7. Open items to confirm
1. ~~Deployment→station cut mapping~~ — RESOLVED from `logsheets/`. Stn 01-04 = multi-dump (problem
   casts); **05-08 clean 1:1** (05=MLADC007/SLADC008 … 08=MLADC010/SLADC011). 09/10 ambiguous in scan.
   Script/coord switch at stn 03; battery change between casts 4 & 5.
2. sADCP combine recipe (WinADCP averaging window; 75+150 merge). **Validation workaround:** ingest the
   existing combined `.mat` (e.g. `MORIA_s75_new.mat`) as-is; own the combine step later.
3. Tolerance: ACCEPTED — median |Δu|,|Δv| < 1 cm/s vs golden `.lad`.
4. Output: keep `.lad`/`.bot` parity permanently, or validation-only then NetCDF-only? (pending)
5. NetCDF convention target (CF-1.x? GO-SHIP/OceanSITES?). (pending)
6. Provide existing combined sADCP `.mat` used for golden casts 05/06; confirm 09/10 file mapping.
```

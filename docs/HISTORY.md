# pyladcp — History

A condensed record of how the package evolved. Full detail lives in git history.

- **GO-SHIP era (abandoned).** First attempt validated against GO-SHIP/RB1606 goldens; the
  velocity inversion never validated cleanly and the effort hit repeated dead ends.
- **MORIA pivot (2026-06-05).** Switched the reference to MORIA 2025 data — stations
  79/80/82, with **station 80 fully instrumented** (`.log` 18-step trace, `.lad`/`.bot`,
  `.mat` with the `p`/`dr` structs). The early MORIA-tuned build (`ix/`) was set aside;
  only the PD0 reader and IGRF-13 magdec were carried forward.
- **QA-first rebuild.** Built the acquisition quality-assessment stage clean under
  `ladcp/qa/`: ingest, screen, beams/range/attitude, depth-sync, bottom detection, editing,
  and the modern figures (raw overview, alignment, depth, edit) + a PDF report. Validated
  bit-exact where possible against MORIA-80 `p`-struct scalars.
- **Velocity (Phase 5).** Super-ensembles (`prepinv`), dual-head merge, **shear-method**
  baroclinic profile + **reference** (the LDEO_IX `ps.shear==1` path), bottom detection with
  sound-speed correction, bottom-track velocities + `.bot`, and the velocity/shear/inverse
  figures folded into the PDF. Validated vs MORIA-80: `u` corr **0.998**, `.bot` corr
  **0.991**, `zbottom` within ~1 m of golden.
- **Magnetic declination finding.** Our IGRF-13 (`ppigrf`) declination is correct; the
  golden `p.drot = -9.878°` is a legacy IGRF-2000 + hardcoded-fudge bug (~4.4° off). The
  product default is per-station IGRF-13; `--drot` reproduces the golden for validation.
- **Foundation / packaging (2026-06).** Removed the superseded GO-SHIP/MORIA build, made
  the tree lean and version-controlled, and stood up the `pyladcp` package (MIT, CI,
  in-repo MORIA-80 fixtures).

## Known fidelity walls (not bugs)
- The golden raw weight archive (`da`) is **truncated** → exact per-cell edit / hspeed /
  weight counts are not bit-reproducible; profiles (`.lad`, `dr`) are validated at profile
  level (corr/RMS), not bit level.
- Super-ensemble `reduced_len` 223 vs golden 218 — CTD-synced depth trajectory vs golden
  integrated `zz`, a trajectory fidelity limit, not dropped data.
- Dual-head tilt-offset objective is flat near its minimum → an ill-conditioned cross-check,
  not a precise number (the compass offset is the one used).

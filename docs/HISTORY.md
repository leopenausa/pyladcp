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
- **Full inverse + ship-ADCP (2026-06).** Added the standard LDEO_IX sparse least-squares
  inverse (`getinv`/`lain*`) as `--solver inverse` alongside the shear solve, with bottom-track
  and barotropic-nav constraints and an optional VmDAS ship-ADCP (`lainsadcp`) constraint.
  Validated on MORIA 79/80/82: SADCP vs LADCP-inverse agree 0.015–0.05 m/s over the upper 300 m.
- **Inverse becomes the default solver (2026-06-10).** After the trust audit showed
  shear ≈ inverse to 1–2 cm/s cruise-wide and the inverse matched golden MORIA-80 at
  0.999, `--solver inverse` became the default. The constraint weights were exposed
  (`--botfac`/`--barofac`/`--smoofac`, joining `--sadcpfac`) and the legacy Figure 12
  constraint-weights plot added to the report. The battery metric was capped at WARN
  (the 0.33×xmv conversion is board-dependent; it had FAILed 17/40 healthy MORIA casts),
  and a `--down-only` solve path added (single-head `merge_heads`, down-bin reference).
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

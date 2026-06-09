# pyladcp — Roadmap

Status snapshot (2026-06): the acquisition-QA stage and a velocity solver
(**shear method + reference**, LDEO_IX `ps.shear==1`) are complete and validated against
the MORIA-80 golden (`u` corr 0.998; bottom-track `.bot` corr 0.991). The package is now
`pyladcp` (MIT, CI). See [HISTORY.md](HISTORY.md) for how we got here and
[ARCHITECTURE.md](ARCHITECTURE.md) for stages/interfaces.

The work below is sequenced; each step gets its own detailed plan before execution.

### Locked direction (decided 2026-06)
- **Audience:** public pip package (`pyladcp`, import `ladcp`). API will churn through
  #2/#4 — the public-API freeze + docs site + PyPI publish are deferred to **#7**.
- **Validation:** MORIA-80 is the in-repo golden (`tests/fixtures/`); additional
  instruments/cruises (held locally) drive #4/#2 cross-validation.
- **Honesty:** the **editing-weight fidelity wall** (golden raw `da` archive truncated →
  exact per-cell edit weights unreproducible) limits bit-level agreement; profiles are
  validated at profile level.

### Steps
1. **Foundation** ✅ — git, lean tree, `pyladcp` skeleton, MIT, CI, fixtures. (this work)
2. **Full inverse + ship-ADCP constraint (#2 + #5)** ✅ — the standard LDEO_IX sparse
   inverse (`getinv`/`lain*`) shipped as `solver="inverse"`, alongside the reduced
   shear+reference solve; SADCP (`lainsadcp`) and bottom-track as weighted constraints with
   error covariance. The VmDAS SADCP ingester (`io/sadcp_vmdas.py`) recovers absolute ocean
   velocity from raw STA/LTA; validated end-to-end on MORIA 79/80/82 — SADCP vs LADCP-inverse
   agree to 0.015–0.05 m/s (corr 0.86–0.97) over the upper 300 m.
3. **Data-driven ingest/config (#4a)** — *in progress.* Built: header-derived instrument
   config (`ingest.apply_header_config`), a cruise-keyed param resolver (`config.resolve_params`
   + `CRUISES`), flexible file discovery (`discovery.py`) that pairs the up-looker to the
   down-looker by **time overlap** (the VmDAS master/slave deployment indices are offset —
   MORIA-10 = master `MLADC012` + slave `SLADC013`), and Seabird `.cnv` column auto-mapping.
   First cross-val station unlocked: **MORIA-10** (Gulf of Biscay, 549 m) runs end-to-end from
   the raw archive — u corr 0.91–0.94, v corr 0.89–0.97, rms 0.023–0.041, vbar within 0.007,
   seabed exact (558 m). Then an **auto-built, incremental archive index** (`archive.py`,
   `ladcp-index` CLI): the Seabird CTD `.hex`/`.hdr` header (`io/ctd_hex.py`) supplies station
   + absolute NMEA UTC + GPS position, which matches the LADCP master by time-window and pairs
   the slave by overlap; results cache to `.ladcp_archive.json`, re-scanning only new files.
   Replaces the hard-coded manifest — `ladcp-qa <st> --index …` resolves raw files by station
   with no naming convention. Validated on 79/80/82 (auto-resolved to MLADC036/037/039, matching
   the curated files). Remaining: FDCCC1 (raw not on disk + publication-gated).
4. **Robustness (#4b)** — single-head casts, beam-vs-earth frames, acquisition-script
   variance, edge cases.
5. **Outputs (#3)** ✅ — Excel + ODV + NetCDF + CSV export (per-station files and cruise-level
   aggregates under `exports/`); shipped as `ladcp/export/`.
6. **Release hardening (#7)** — public-API freeze, `ruff format` pass, docs site,
   versioning/CHANGELOG, PyPI publish.
7. **CTD-pipeline integration (#6)** ✅ — raw Seabird `.hex` → the cleaned 6-col `.cnv`
   the solver consumes, so casts without a pre-processed profile still run. The recipe
   lives in CTD_project (`ctd_pipeline.convert_for_ladcp`: datcnv → ITS-90/PSS-78 derive →
   SBE Wild Edit → 1 s time-bin → 6-col extract); pyladcp calls it through `io/ctd_raw.py`
   as an **optional** dependency (located via `LADCP_CTD_PROJECT` or a sibling dir; pyladcp
   does not require it). Wired into `discover(from_hex=, ctd_cache=)` and
   `ladcp-qa --from-hex` (default off — a pre-processed `.cnv` always wins; conversion is
   the fallback), caching converted files to a reuse folder. Validated **byte-for-byte vs
   the operator on all 23 MORIA stations** (prDM ≤0.002 dbar, T/S ≤0.001); end-to-end on
   MORIA-79/80/82, which had no operator CTD and now get it from their `.hex` anchor.
   (The work also surfaced + fixed a real SBE-conformance bug in CTD_project's own Wild Edit
   — see memory [[ctd-wildedit-sbe-fix]].) Future option: an *agentic* variant with
   per-cast safety checks on top of this deterministic recipe.

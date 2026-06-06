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
3. **Data-driven ingest/config (#4a)** — derive instrument config from PD0 headers + a
   typed config schema; flexible file discovery; load + process the other datasets. (Gates
   broad validation of #2; may run before/with #2 to unlock the test data.)
4. **Robustness (#4b)** — single-head casts, beam-vs-earth frames, acquisition-script
   variance, edge cases.
5. **Outputs (#3)** — Excel + ODV-ready export, once the solution object is final.
6. **Release hardening (#7)** — public-API freeze, `ruff format` pass, docs site,
   versioning/CHANGELOG, PyPI publish.
7. **CTD-pipeline integration (#6)** — agentic raw-CTD → cleaned `.cnv` per station with
   safety checks (depends on the CTD_project; see memory `ladcp-ctd-pipeline-integration`).

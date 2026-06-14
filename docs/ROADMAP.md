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
3. **Data-driven ingest/config (#4a)** — *in progress.* Header-derived instrument config
   (`ingest.apply_header_config`), cruise-keyed param resolver (`config.resolve_params`),
   time-overlap file discovery (`discovery.py`; VmDAS master/slave indices are offset), and
   `.cnv` column auto-mapping. An auto-built incremental archive index (`archive.py`,
   `ladcp-index`) anchors station/UTC/GPS off the Seabird `.hex`/`.hdr` header and resolves raw
   files by station with no naming convention (cache `.ladcp_archive.json`). Cross-val: MORIA-10
   runs end-to-end from the raw archive (u corr 0.91–0.94, seabed exact). Remaining: cruise-2
   goldens (local-only + publication-gated).
4. **Robustness (#4b)** — single-head casts, beam-vs-earth frames, acquisition-script
   variance, edge cases.
5. **Outputs (#3)** ✅ — Excel + ODV + NetCDF + CSV export (per-station files and cruise-level
   aggregates under `exports/`); shipped as `ladcp/export/`.
6. **Release hardening (#7)** — public-API freeze, `ruff format` pass, docs site,
   versioning/CHANGELOG, **PyPI publish** (name `pyladcp` free as of 2026-06; plan:
   GitHub-Actions Trusted Publishing, TestPyPI dry-run first, Zenodo DOI on the
   same release; audit the sdist contents before the first upload).
7. **CTD-pipeline integration (#6)** ✅ — raw Seabird `.hex` → cleaned 6-col `.cnv`, so casts
   without a pre-processed profile still run. Recipe lives in CTD_project
   (`ctd_pipeline.convert_for_ladcp`); pyladcp calls it via `io/ctd_raw.py` as an **optional**
   dep, wired into `ladcp-qa --from-hex` (default off — a real `.cnv` always wins). Validated
   byte-for-byte vs the operator on all 23 MORIA stations; MORIA-79/80/82 now get CTD from their
   `.hex` anchor. (Surfaced + fixed an SBE Wild-Edit bug in CTD_project — [[ctd-wildedit-sbe-fix]].)
   Future: an *agentic* variant with per-cast safety checks.

### Legacy-port fidelity backlog — CLOSED 2026-06-14

The 17-stage deep audit (legacy LDEO_IX `.m` ↔ pyladcp) catalogued every divergence and
ranked the un-ported items. The high-value accuracy items were ported and merged:

- **`p.soundcorr`** — in-situ sound-speed velocity rescale (PR #54; MORIA-80 ūbar −6.31→−6.47 vs golden −6.50).
- **`loadrdi` wlim/vlim** — water-cell velocity edits (PR #53; golden u-rms 1.32→0.64).
- **uship true→magnetic frame** — barotropic constraint frame fix (PR #56).
- **operator seabed override** — `--zbottom` / `--guessbottom`, CLI + Studio (PR #57/#58).
- **shear-vs-inverse consistency WARN** (PR #54).

The remaining items were each **scoped and quantified on FDCCC (30) + MORIA (36)**, then
declined — each for a distinct, *measured* reason (writeups in `legacy_audit_work/findings/`):

| item | verdict | why (measured) |
|---|---|---|
| per-ping `outlier` edit | defer | accuracy-neutral; ~224k cells dropped for no net gain |
| STEP-12 `offsetup2down` | defer | net-negative trade vs golden + fidelity gap (degrades the golden cast) |
| `tilt_weight` | skip | structural no-op — the inverse weights by velocity *scatter* (`ruvs`), not the correlation weight legacy scales; faithful A/B bit-identical on all 66 casts |
| `detect_bottom` shallow false-lock retune | no action | no real bug — sub-metre median vs the faithful poly; apparent false-locks are unreliable golden `p.zbottom`; genuine uncertain casts already self-flag |

**Conclusion:** the legacy-port backlog is exhausted of high-value accuracy items. pyladcp's
velocity + bottom path is in a solid, validated state. Remaining un-ported legacy items
(no-CTD vertical-velocity-integration depth path, `.mat dr` writer) are feature-completeness,
not accuracy, and belong to **#7 / #4b** rather than fidelity.

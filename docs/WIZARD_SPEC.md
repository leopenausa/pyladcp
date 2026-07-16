# `ladcp` — the cruise hub (spec v0.2, 2026-07-16)

> Status: **approved** — distilled from the design interview of 2026-07-16;
> §9 open questions settled 2026-07-16. Implementation plan: `docs/WIZARD_PLAN.md`.

## 1. Purpose

One command that takes a person from "a directory of cruise data" to "processed,
QA-scored velocity profiles" — and then stays useful as the cruise grows. It serves
three people:

1. **A colleague on first contact** — cloned pyladcp, has data, never ran it. The hub
   finds their files, explains its choices, and gets them to a first profile without
   reading the guide.
2. **The maintainer at cruise start** — knows everything, wants the setup ritual
   (index, CTD wiring, SADCP selection, cruise params) automated, consistent, and
   captured in a reviewable file.
3. **A ship operator mid-cruise** — casts arrive on watch; they need "what's new,
   process it, is it OK" in three keystrokes, over SSH.

It is explicitly **not** a teaching tool first (though explanations link guide
chapters), and it does not replace the expert commands — it drives them.

## 2. Product shape

A new **umbrella command `ladcp`** with subcommands. Existing `ladcp-*` commands are
untouched and remain the expert layer; the hub composes them through their library
entry points (never by shelling out).

```
ladcp init      # first-run setup wizard: discover → confirm → cruise.toml → trial
ladcp status    # the mid-cruise dashboard (default when run bare in a cruise dir)
ladcp process   # process new/named/all stations per the config
ladcp studio    # launch Studio preloaded with this cruise's config
ladcp config    # show / edit / validate cruise.toml
```

Running bare `ladcp` in a directory with a `cruise.toml` shows **status**; without
one it offers **init**. Setup is therefore just the hub's first-run path.

Two front ends, one core: v1 ships the **terminal** front end; a **Studio setup
page** reuses the same wizard engine (question graph, detection, validation) in v2.

## 3. First-run flow (`ladcp init`)

Auto-detect, then confirm — the wizard proposes, the user disposes. Every step shows
what was found, what it inferred, and asks before writing anything.

1. **Point at the data.** One root (or several); the wizard scans with the same
   low-IO discipline as `ladcp-ek80` (headers only, never full reads on shares).
2. **LADCP casts.** Find PD0 pairs (master/slave), pair by time overlap, detect
   beam-vs-earth coords, single-head casts (propose `--down-only`), fragmented
   deployments. Build/refresh the archive index (`ladcp-index` engine).
3. **CTD.** Find cleaned `.cnv` per station; where only raw `.hex` exists, offer
   `--from-hex` conversion (flag the CTD_project dependency clearly).
4. **Ship-ADCP.** Detect all candidate constraint sources — raw VmDAS trees
   (`.STA`/`.LTA`), CODAS contour NetCDFs, EK80 SONAR-netCDF4 runs — present them as
   a choice (like Studio's source dropdown). For EK80 on a remote share, offer the
   timetable + slim-extract flow, never copying without asking.
5. **Nav.** Find GPS track files; offer `--sadcp-timeoff auto` when a raw VmDAS
   source was chosen.
6. **Cruise identity & params.** Name the cruise; start from a built-in preset or
   generic defaults; any confirmed deviations are written as `[params]` overrides.
7. **Write `cruise.toml`** (atomic write; shown in full before saving).
8. **Trial station (suggested, skippable).** Offer to process one representative
   station now, print its QA scorecard inline, offer to open the PDF report, then
   ask whether to run the batch.

## 4. The hub (`ladcp status` + actions)

At-a-glance, in priority order:

- **New casts since last run** — stations on disk not yet processed (index refresh).
- **QA rollup** — ok/warn/fail/error counts with the worst offenders named.
- **Loose ends** — stations missing CTD, missing up-looker, no SADCP coverage,
  unapplied edit journals, stale journals.

Each status line maps to an action (`ladcp process <station>`, `ladcp studio
<station>`, …). Advanced users get the same power non-interactively:
`ladcp process --new`, `ladcp process 07 12`, `ladcp process --all`.

## 5. `cruise.toml` — first-class configuration

The config is **the cruise interface**, not just the wizard's memory:

- **`ladcp-qa` (and Studio) learn to read it** — auto-discovered in the working/cruise
  directory or passed via `--config`. Command-line flags become *overrides* on top.
- **Precedence:** explicit flags > `cruise.toml` > built-in cruise preset > generic
  defaults. Every resolved value must be traceable to its source (`ladcp config show`
  annotates provenance).
- **Contents:** cruise identity; data roots and layout hints; CTD wiring (incl.
  from-hex); the chosen SADCP source(s) with their knobs; solver flags
  (`SessionConfig` surface); **and `[params]` / `[params.<station>]` sections that can
  override any `CastParams` field** — a new cruise never needs a code edit. The
  in-code registry remains only as built-in presets.
- Hand-editable, diffable, and versionable; the wizard round-trips it losslessly.

## 6. Architecture notes (constraints on the plan, not the plan)

- Built on the phase-2/3 seams: the wizard assembles a **`SessionConfig`** + station
  list and drives **`qa.pipeline.process_station`** / the batch machinery — no
  parallel orchestration path. Config→params flows through **`edit_overrides`** and
  `resolve_params` (extended for TOML-sourced overrides).
- Detection reuses what exists: `discovery`, `archive.build_index`, `ek80_files.scan`,
  `sadcp_codas.resolve_codas_nc`, Studio's `merge_discovered_codas`; SADCP warming via
  `warm_sadcp`.
- The wizard engine (question graph + detection + validation) is front-end-agnostic so
  the Studio page can reuse it unchanged.
- Standard library TOML (`tomllib`) for reading; writing needs a chosen writer (open
  question below).

## 7. Hard requirements (ship realities)

- **Fully offline** — no network for any feature; help text and guide references
  resolve locally.
- **Interruption-safe** — a killed batch resumes where it left off; `cruise.toml` and
  the index are written atomically, never left half-updated.
- **Share-friendly** — discovery is low-IO (filename/header peeks); anything that
  copies data (EK80 extracts, CTD conversion) is explicit and confirmed.
- **Non-interactive parity** — every wizard decision is expressible via flags/config,
  so all flows run in scripts and over flaky SSH (`ladcp init --yes --root … --sadcp …`).

## 8. Phasing

- **v1:** `ladcp` umbrella with `init` (full discovery scope: PD0+CTD incl. from-hex,
  VmDAS, CODAS, EK80, nav), `status`, `process`, `config`; first-class `cruise.toml`
  in `ladcp-qa`; trial-station suggestion; terminal front end.
- **v2:** Studio setup page on the shared engine; QA-diagnosis suggestions after runs
  ("bottom track weak → consider `--guessbottom`"); outputs-freshness in status;
  `ladcp studio` deep integration.

## 9. Decisions (settled 2026-07-16)

1. **Config location/name** — `<cruise-root>/cruise.toml`, next to
   `.ladcp_archive.json`. `ladcp` searches cwd then parent directories (git-style);
   `--config` overrides. One directory per cruise.
2. **TOML writing** — add the `tomli-w` dependency (tiny, pure-Python, no
   transitive deps); stdlib `tomllib` for reading.
3. **Resume granularity** — freshness rule, no state file: a station is *done* if
   its outputs exist and are newer than its inputs and `cruise.toml`.
   `process --new` runs missing/stale only; `--force` reprocesses.
4. **`ladcp studio` v1 scope** — thin launcher: `ladcp studio [station]` starts
   `ladcp-studio` with the cruise config (roots, SADCP source, params) preloaded.
   Deep integration stays v2.
5. **Name collision** — verified 2026-07-16: no `ladcp` executable on the base
   PATH or in the `ladcp_pipeline` env; the bare name is free.

## 10. Out of scope (this feature)

- Detiding, uncertainty research features (separate roadmap items).
- Multi-cruise aggregation; the hub manages one cruise per config.
- Editing raw data or journals (Studio's job); the hub only points there.

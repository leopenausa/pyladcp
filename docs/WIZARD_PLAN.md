# `ladcp` cruise hub — development plan (v1.0, 2026-07-16)

> Implements `docs/WIZARD_SPEC.md` (spec v0.2, approved). Five phases, one PR each,
> shippable in order — every phase leaves `main` releasable and each later phase
> builds only on merged seams. Estimated sizes are relative (S < M < L).

## Ground truth (seams this plan builds on)

- `SessionConfig` (`session.py:185`, frozen: `edit/sadcp/solve`), built by
  `SessionConfig.from_args` (`session.py:194`); `edit_overrides` (`session.py:95`)
  is the config→params bridge; `resolve_params(cruise, station, overrides=)`
  (`config.py:199`) applies override dicts onto `CastParams` via `setattr` and
  already raises on unknown fields — the `[params]` TOML sections plug in here.
- `qa.pipeline.process_station(...) -> (status, export)` (`pipeline.py:30`) and the
  batch loop in `qa/cli.py:main` (`cli.py:227`, serial + `ProcessPoolExecutor`).
- Discovery: `archive.build_index` (`archive.py:100`, writes `.ladcp_archive.json`
  with an mtime-keyed `scan_cache`), `discovery.discover` (`discovery.py:174`),
  `ek80_files.scan` (`ek80_files.py:125`), `sadcp_codas.resolve_codas_nc`
  (`sadcp_codas.py:36`), Studio's `merge_discovered_codas` (`studio/state.py:59`),
  `warm_sadcp` (`pipeline.py:342`).
- CTD from-hex: `ctd_raw.cnv_from_hex` (`ctd_raw.py:92`), already mtime-fresh.
- QA rollup source: per-station `<outdir>/stations/<st>/<st>_qa.json` with
  `overall_status` + per-metric statuses (`models.py:105`).
- Studio launcher: `studio/cli.py:main` (`cli.py:20`).
- No `tomllib`/`tomli-w` usage exists yet; entry points live in
  `pyproject.toml [project.scripts]` (9 commands, no bare `ladcp` — name verified free).

New code lives in **`src/ladcp/hub/`** (`cruise_config.py`, `freshness.py`,
`detect.py`, `init_flow.py`, `status.py`, `cli.py`) plus one entry point
`ladcp = ladcp.hub.cli:main`.

---

## Phase A — `cruise.toml` core + first-class in `ladcp-qa` (M)

The config layer everything else reads. No `ladcp` command yet.

1. **`hub/cruise_config.py`**
   - `CruiseConfig` dataclass mirroring the TOML schema: `[cruise]` (name, preset),
     `[data]` (root, ladcp_dir, master/slave subdirs, index path), `[ctd]`
     (dir, from_hex, cache), `[[sadcp]]` (source/folder/filetype/xducer/timeoff/nav —
     maps 1:1 onto `SadcpConfig`), `[solve]` (maps onto `SolveConfig`), `[edit]`
     (maps onto `EditConfig`), `[params]` + `[params.<station>]` (free-form
     `CastParams` overrides, validated against `CastParams` field names at load).
   - `find_config(start: Path) -> Path | None` — cwd-then-parents search
     (stop at filesystem root; also stop at `.git` boundary).
   - `load_config(path) -> CruiseConfig` (stdlib `tomllib`), with clear errors
     naming the offending table/key.
   - `save_config(cfg, path)` — `tomli_w` serialize, atomic write
     (temp file + `os.replace`, same pattern for every hub write).
   - `to_session_config(cfg, cli_args) -> SessionConfig` + provenance map:
     precedence **explicit flags > cruise.toml > preset > generic**. Explicitness
     detection: re-parse argv against a defaults-suppressed twin parser
     (`argparse.SUPPRESS`) so only user-typed flags override the TOML.
   - `params_overrides(cfg, station) -> dict` — merged `[params]` +
     `[params.<station>]`, fed to `resolve_params(overrides=)` after
     `edit_overrides` (station-specific wins).
2. **`ladcp-qa` learns the config**: `--config PATH` flag + auto-discovery of
   `cruise.toml` in cwd/parents (skippable with `--no-config`). `qa/cli.py`
   resolves TOML → args before `SessionConfig.from_args`; `[params]` overrides
   thread through `process_station` (extend its `edits`/overrides path — one new
   optional `param_overrides` argument, default `None`, so existing callers are
   untouched).
3. **Packaging**: add `tomli-w` to `dependencies` in `pyproject.toml`.
4. **Tests** (`tests/test_cruise_config.py`): round-trip load/save losslessness;
   upward search + `.git` boundary; precedence matrix (flag vs toml vs preset vs
   generic, incl. "flag left at default does not shadow toml"); unknown
   `[params]` key → clear error; `ladcp-qa --config` end-to-end on the synth
   fixtures (quick lane, `tmp_path`).

**Exit criteria:** a hand-written `cruise.toml` fully drives an `ladcp-qa` run with
zero flags; `ruff` + quick lane green.

## Phase B — umbrella `ladcp` skeleton: `config` + `process` (M)

1. **`hub/cli.py`**: `ladcp` entry point with subparsers
   (`init/status/process/studio/config`); bare `ladcp` → `status` when a
   `cruise.toml` is found, else prints the init hint. Register
   `ladcp = "ladcp.hub.cli:main"` in `[project.scripts]`.
2. **`ladcp config`**: `show` (resolved values annotated with provenance:
   `flag | cruise.toml | preset:<name> | default`), `validate` (load + schema +
   referenced-paths existence), `edit` (open `$EDITOR`, validate on close,
   refuse to save invalid).
3. **`hub/freshness.py`**: `station_state(station, cfg) -> fresh | stale | missing`.
   Done-marker = `<st>_qa.json` mtime; inputs = master/slave PD0 (+ CTD file,
   edits journal if any) and `cruise.toml` itself. Pure function over paths —
   no state file (spec §9.3).
4. **`ladcp process`**: selection modes `--new` (missing+stale), explicit station
   labels, `--all`, `--force`; builds the work list, then drives the *existing*
   batch machinery. Refactor `qa/cli.py:main`'s plan-then-run body into a
   reusable `qa.pipeline.run_batch(plan, cfg, ...)` (serial + `-j` parallel,
   per-cast exception isolation, cruise exports) so hub and `ladcp-qa` share one
   loop — no parallel orchestration path (spec §6). Interruption safety falls out
   of the freshness rule: rerunning `--new` resumes where it died.
5. **Tests**: freshness matrix (missing / stale-by-input / stale-by-config /
   fresh); `ladcp process --new` skips fresh stations (synth data, quick lane);
   `run_batch` refactor covered by existing `test_cli_layout.py` /
   `test_cli_jobs.py` staying green unchanged.

**Exit criteria:** in a directory with `cruise.toml`, `ladcp process --new`
processes exactly the missing/stale stations; `ladcp config show` traces every
value to its source.

## Phase C — detection engine + `ladcp init` (L, the wizard itself)

1. **`hub/detect.py`** — pure detection functions, each returning a typed
   *proposal* (found items + inferred choice + evidence string), no I/O writes:
   - LADCP: wrap `archive.build_index` + scan; propose master/slave layout,
     flag single-head casts (`--down-only` proposal), beam-vs-earth from PD0
     headers, fragmented deployments.
   - CTD: `.cnv` per station via `_find_clean_ctd` conventions; hex-only →
     from-hex proposal (states the CTD_project dependency,
     `ctd_raw._find_ctd_project` check up front).
   - SADCP: candidates from VmDAS trees (`.STA`/`.LTA` globs), CODAS
     (`resolve_codas_nc` + `merge_discovered_codas` logic lifted out of Studio),
     EK80 (`ek80_files.scan`, header-peek only). Present as a source choice;
     EK80-on-share → offer timetable + `slim_extract`, never silent copies.
   - Nav: GPS track candidates; propose `timeoff='auto'` when raw VmDAS chosen.
   - Low-IO discipline throughout: filename/header peeks only (reuse
     `scan_cache` pattern); anything that copies or converts is a *proposal*
     executed only after confirmation.
2. **`hub/init_flow.py`** — the question sequence (spec §3 steps 1–8) as data:
   ordered steps, each `(detect → propose → confirm → apply)`. Front-end-agnostic:
   terminal prompt driver in v1, Studio page reuses the step list in v2.
   Non-interactive parity: `ladcp init --yes` accepts every answer via flags
   (`--root`, `--cruise`, `--preset`, `--sadcp-source`, `--ctd-dir`,
   `--from-hex`, `--no-trial`, …); interactive mode is just "fill unanswered
   steps by prompting". Ends with: full `cruise.toml` printed, confirm, atomic
   write, then the **trial station** offer (pick a representative mid-cruise
   station with CTD present; run `process_station`; print the text scorecard;
   offer the PDF path; offer the batch).
3. **Tests**: detection proposals on synthetic trees (`tmp_path` fixtures:
   fake PD0 pairs via `ladcp-synth` writer, dummy `.cnv`/`.hex`, fake codas
   `contour/*.nc` names); `init --yes` end-to-end produces a valid loadable
   `cruise.toml` on synth data; prompt driver unit-tested with scripted answers
   (inject an `input_fn`).

**Exit criteria:** `ladcp init --yes --root <synth tree>` writes a `cruise.toml`
that Phase B's `ladcp process --all` runs green, no prompts, no network, no full
file reads during discovery.

## Phase D — `ladcp status` (S/M)

1. **`hub/status.py`**: three blocks in priority order (spec §4):
   - *New casts*: refresh index (incremental via `scan_cache`), diff against
     `stations/` outputs using Phase B freshness → "N new / M stale".
   - *QA rollup*: read every `<st>_qa.json` → ok/warn/fail/error counts, worst
     offenders named with their failing metric.
   - *Loose ends*: stations missing CTD (index has cast, no `.cnv`/hex),
     single-head (no slave), no SADCP coverage (from qa.json sadcp metrics),
     unapplied/stale edit journals (`manual_edits_unapplied` seam,
     `pipeline.py:83-95`).
   - Every line ends with its action (`→ ladcp process 07`, `→ ladcp studio 12`).
2. Bare `ladcp` dispatches here. Plain-text output, SSH-friendly, no color
   dependencies beyond what `qa/runlog.py` already uses; `--json` for scripts.
3. **Tests**: rollup counts from crafted qa.json trees; new-cast detection after
   adding a file to a synth tree; `--json` schema stability.

**Exit criteria:** on the MORIA2 working directory, `ladcp` prints an accurate
dashboard in <2 s without touching raw PD0 payloads.

## Phase E — the Studio hub GUI + docs (M/L; revised 2026-07-17)

> Revised per the user's course correction: the goal is a *guided window*, not just
> a launcher. Studio grows the hub pages; `ladcp studio` is the one command that
> pops it up (spec v0.3).

1. **`studio/hub_api.py`**: `/api/hub/*` — state, detect, preview/save config,
   status, process (background job, per-station progress), scorecard, report PDF.
   Every endpoint is a JSON shim over the phase A–D engine.
2. **`static/hub.html` + `hub.js`**: the wizard (scan cards → choices → toml
   preview → save+index → trial with inline scorecard) and the dashboard (pending /
   QA / loose ends with process buttons and a progress bar).
3. **`ladcp studio [station]`**: translates `cruise.toml` → the `studio/cli.py`
   argv (vmdas→`--sadcp`, codas→`--sadcp-codas`; ek80 warned + skipped) and lands
   on the wizard (no config), the dashboard (config, no station), or the editor
   (station named). `--hub-dir`/`--start-page` on `ladcp-studio` carry the mode.
4. **Docs**: guide chapter 11 "The cruise hub" (both front ends), README "guided
   way" quickstart, spec v0.3.
5. **Tests**: TestClient wizard flow + job endpoint (monkeypatched at the
   `run_batch` seam); `ladcp studio` argv-translation tests.

**Exit criteria:** a colleague's first-contact path is `pip install`,
`ladcp studio`, follow the window; the terminal parity path stays
`ladcp init --yes …`.

---

## Cross-cutting rules

- **One orchestration path**: hub always drives `qa.pipeline` / `run_batch`;
  never a second solve loop (spec §6).
- **Atomic writes everywhere**: `cruise.toml`, index, qa.json rewrites — temp +
  `os.replace` (spec §7).
- **Offline**: no network calls; guide references are local paths (spec §7).
- **Quick-lane discipline**: all hub tests run on synth fixtures in the
  `not slow` lane; golden-data tests only where a real solve is exercised.
- **Verification per phase**: `conda run -n ladcp_pipeline python -m pytest
  tests/ -m "not slow" -q` + `ruff check .` + one manual end-to-end on the
  MORIA2 tree before merge.

## Risks & mitigations

- **Precedence correctness** (flag-at-default vs explicit flag): the
  suppressed-defaults twin-parser technique; locked by the Phase A precedence
  matrix test.
- **`run_batch` extraction** touches the proven `qa/cli.py` loop: do it as a
  pure move (behavior-preserving), rely on existing CLI tests as the guard.
- **Freshness false-positives** (e.g. touched-but-unchanged inputs cause
  reprocess): acceptable by design — the rule errs toward reprocessing;
  `--force`/explicit labels always available.
- **Windows**: `os.replace` atomicity and path handling are cross-platform; CI
  already runs the matrix on linux/mac/windows.

## Deferred to v2 (per spec §8)

Studio setup page on the shared step engine; QA-diagnosis suggestions after
runs; outputs-freshness in status; `ladcp studio` deep integration.

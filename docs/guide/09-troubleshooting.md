# 9 · Troubleshooting

A symptom → cause → fix catalogue, ordered by where in the workflow you hit it.
Symptoms quote the actual message where one exists. When a symptom matches a scorecard
row, [chapter 6](06-qa-report.md) has the full story for that row.

## Install & environment

**Symptom:** `ladcp-qa: command not found` after installing.
**Cause:** the environment where pip installed pyladcp is not active.
**Fix:** activate the env (`conda activate ...` / `source .venv/bin/activate`) or
reinstall with the right interpreter: `python -m pip install -e .`

**Symptom:** Excel files missing; log says `excel skipped: ...`.
**Cause:** `openpyxl` is not installed — Excel is an optional extra.
**Fix:** `pip install "pyladcp[export]"`. NetCDF/ODV/CSV write regardless.

**Symptom:** `raw-CTD ingest needs the CTD_project package (ctd_pipeline), which was
not found`.
**Cause:** `--from-hex` needs the companion CTD_pipeline package.
**Fix:** clone CTD_pipeline next to the pyladcp checkout, or point at it with
`LADCP_CTD_PROJECT=/path/to/CTD_project`. Or skip `--from-hex` entirely by providing
processed `.cnv` files.

**Symptom:** figures crash or hang on a headless server.
**Fix:** prefix commands with `MPLBACKEND=Agg`.

## Indexing (`ladcp-index`)

**Symptom:** `indexed 0 casts from 0 files`.
**Cause #1 (by far the most common):** the path variables are empty — a new terminal
forgot `$B`, so `--ladcp "$B/raw_ladcp"` became `/raw_ladcp`.
**Fix:** `echo "$B"` — if it prints nothing, set it again ([chapter 5, step 0](05-cruise-workflow.md)).
**Cause #2:** `--root` was placed after `build` (it belongs *before*:
`ladcp-index --root "$B" build ...`).

**Symptom:** `no index found (run 'ladcp-index build' first)`.
**Cause:** `ladcp-qa --index` points at a path where no `.ladcp_archive.json` exists
(or `show` was run before `build`).
**Fix:** run the build step; the default index lands at `<root>/.ladcp_archive.json`.

**Symptom:** a station you *know* was occupied is absent from `ladcp-index show`.
**Cause:** its CTD cast is missing or unreadable — the CTD `.hex` header is the anchor
that gives a raw file its station identity; with no anchor, the cast cannot appear.
**Fix:** locate the missing/misnamed CTD files, then rebuild. Always compare the cast
count against the cruise logsheets.

**Symptom:** `SHARED-MASTER(...)` in a station's provenance.
**Cause:** two CTD anchors claim the same LADCP deployment file — typical when
back-to-back shallow stations were recorded into one file without stopping the
instrument.
**Fix:** usually correct as resolved; review those stations' depth figures with extra
care (does each profile reach its station's expected depth?).

**Symptom:** stale pairings after files were added or renamed.
**Fix:** `ladcp-index --root "$B" build ... --rescan` forces a full re-decode.

## Running `ladcp-qa`

**Symptom:** `velocity skipped: no up-looker`.
**Cause:** the cast has only a down-looker file (or the slave never paired).
**Fix:** if the up-looker truly is missing, `--down-only` solves from the down-looker
alone — the scorecard then carries a `single_head_solve` WARN (reduced near-surface
coverage). If the slave file exists but didn't pair, check `ladcp-index show`.

**Symptom:** `velocity skipped: ...-coordinate data is unsupported`.
**Cause:** a head recorded in a frame pyladcp cannot rotate (beam coordinates *are*
rotated automatically; this message means something else — e.g. ship or instrument
frame without attitude data).
**Fix:** acquisition metrics are still valid; the velocity needs the data re-exported
in beam or earth coordinates.

**Symptom:** `declination: IGRF failed ... velocity LEFT IN MAGNETIC FRAME` (and a
`declination` WARN on the scorecard).
**Cause:** no usable position (missing CTD GPS) or the IGRF lookup failed; velocities
are then relative to *magnetic* north.
**Fix:** supply `--drot <deg>` explicitly, or fix the CTD position source. Do not
deliver magnetic-frame profiles.

**Symptom:** one cast crashes during a batch run.
**Behaviour:** the batch continues — the cast is logged as `[ERROR]` in
`qa_out/ladcp-qa.log` and skipped, never fatal.
**Fix:** rerun that station alone with `-v` to stream the full traceback.

**Symptom:** the run is slow and you can't see why.
**Fix:** `-v` streams per-stage detail instead of the progress bar; the log file
always has the full detail afterwards.

## The profile looks wrong

**Symptom:** the whole profile is offset by several cm/s against a reference
(ship-ADCP, a repeat cast, a legacy result) — shape fine, level wrong.
**Cause:** the *reference* (barotropic) part of the solution, not the shear. The usual
suspects, in order: bottom-track quality (`bottom_track_consistency` row, bottom-track
figure), a pre-cast surface soak or post-cast drift contaminating the in-water window,
ship drift on a long wire (`ship_drift`).
**Fix:** read the bottom-track figure first. If the two tracks disagree, try
`--botfac 0` (drop the bottom-track constraint; GPS-barotropic carries the reference)
and compare. pyladcp already prefers the RDI firmware track and restricts the window
to the deep segment containing the deepest ping — but a cast with bad firmware samples
*and* heavy drift can still need the manual fallback.

**Symptom:** velocity *directions* look rotated (e.g. a known along-slope current
comes out cross-slope).
**Cause:** declination (check the `declination` row: value and provenance) or a
corrupted compass-offset estimate between the heads.
**Fix:** pyladcp cross-checks the head alignment against the in-water window
automatically (deck pings can corrupt the full-record estimate); if you still suspect
rotation, compare against the ship-ADCP overlay (`--sadcp`) — a rotation shows up as a
depth-independent angle error in the upper ocean.

**Symptom:** noisy, blocky profile on a shallow shelf cast.
**Cause:** `shallow_cast` (<100 m) — few super-ensembles, weak constraints.
**Fix:** expectations first (it will never look like a 2000 m cast). Then
[chapter 7](07-solvers-weights.md): check the weights figure; on bad near-bottom BT
samples use `--botfac 0`; if a contaminated near-bottom cell slips through a
bottom-depth underestimate, raise `--dzbelow` (e.g. 24–32).

**Symptom:** the seabed line sits at the wrong depth — the profile stops a cell or two
short of the bottom, or `bottom_depth` WARNs that two estimates disagree by >25 m.
**Cause:** automatic detection locked onto the wrong echo — a mid-water scattering
layer, a bottom *multiple* (a re-reflection at ~2× or 3× the true depth), or, on a cast
that barely reached the bed, a return too weak to pin. The depth figure shows it: the
seabed line doesn't sit under the cluster of bottom-track points.
**Fix:** give it the depth you trust from the echo-sounder or the logsheet.
`--zbottom <m>` takes that depth verbatim and skips detection; `--guessbottom <m>` keeps
auto-detection but confines it to within 50 m of your value (use this when you only know
the depth roughly). Both are per-cast — run that one station on its own. A correct seabed
fixes both the near-bottom side-lobe editing *and* the bottom-track reference. In
[Studio](10-studio.md) the same two controls live in the left rail (*set seabed by
hand → exact / hint*); set one and watch the seabed line and the near-bottom cells snap
into place. (Detection is accurate to ~1 m on the great majority of casts — reach for
this only on the handful the depth figure flags.)

**Symptom:** a band of bad velocities at a fixed *distance below the package* on
every cast of a leg (and `nearfield_errvel_ratio` WARNs).
**Cause:** a device hung under the rosette (corer, extra bottle, transponder) — a
rigid target in the down-looker's near field.
**Fix:** mask those bins: `--nearfield-dn-bins 3,4` (or whatever the geometry says),
or per-cruise via the preset. Don't deliver `--down-only` products from the affected
stations.

**Symptom:** the upper ~100 m is missing (`profile_surface_coverage` WARN).
**Cause:** the ADCP started recording mid-cast (clipped/fragmented file).
**Fix:** nothing to compute — the upper column is *unsampled, not zero*. Make sure
downstream users see the coverage column, not just `u`/`v`.

**Symptom:** profile is empty / a handful of bins, on a cast that clearly worked.
**Cause:** CTD↔LADCP mis-sync — the depth window landed on the wrong part of the
recording. The scorecard makes this loud: `ctd_sync_locate` and/or
`ctd_sync_coverage` WARN.
**Fix:** confirm the file pairing (`ladcp-index show`), check the depth figure's
descent/ascent "V", and verify the CTD file covers the full cast (a CTD record that
stops mid-ascent is handled — package depth is integrated past the end — but a wrong
*file* is not recoverable).

## Ship-ADCP

**Symptom:** the SADCP constraint "does nothing" / `no SADCP data` in the log.
**Cause:** no ensembles fell inside the cast's time/space window — wrong directory,
wrong file type, or a clock problem.
**Fix:** check `--sadcp` points at the dir holding the `.STA`/`.LTA` files (or pass
`--sadcp-filetype LTA`); for clock trouble see the next entry. `--sadcp-reingest`
rebuilds the cache after you change the source.

**Symptom:** SADCP timestamps are absurd (wrong year/decade) or the overlay is offset
in time.
**Cause:** the acquisition PC's clock was not synced to GPS — offsets from seconds to
*years* happen at sea.
**Fix:** `--sadcp-timeoff auto --sadcp-nav <track>` recovers the constant offset by
sliding the SADCP's embedded GPS against an independently timestamped navigation track
(any time/lat/lon CSV or a SADO export; file or directory). A known offset can be
given directly in seconds.

**Symptom:** `sadcp_consistency` WARNs on many casts of a leg.
**Cause:** if the *LADCP* were wrong it would vary cast-by-cast; a systematic
discrepancy usually means the SADCP product needs work (misalignment/amplitude
calibration, per-ping editing).
**Fix:** process the ship-ADCP properly with CODAS and feed the edited product back
via `--sadcp-source codas` ([chapter 8](08-ship-adcp.md)).

## Comparison against a legacy processing

**Symptom:** `ladcp-compare` pairs fewer stations than expected; the rest are in
`unpaired.txt`.
**Cause:** pairing is by cast time with a tolerance (`--tol-hours`, default 3 h); a
station outside the tolerance (clock issues, reprocessed subsets) stays unpaired —
never silently force-matched.
**Fix:** widen `--tol-hours` deliberately, or accept the unpaired listing.

**Symptom:** one or two stations dominate the comparison statistics.
**Fix:** rerun just those with adjusted options into a separate dir and substitute
them explicitly: `--alt-dir DIR --alt-stations a,b --alt-label "botfac=0"` — the
substitution is labelled in the CSV, the profile titles and the summary page.

---

Still stuck? Open an [issue](https://github.com/leopenausa/pyladcp/issues) with the
station's `_qa.txt`, the relevant lines of `ladcp-qa.log`, and (if you can share it)
the report PDF — those three artifacts answer most questions.

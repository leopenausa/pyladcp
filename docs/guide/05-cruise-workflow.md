# 5 · Whole-cruise workflow

This chapter is the recipe for the situation you actually face after a cruise: a
directory of raw LADCP deployment files **that carry no station numbers**, a directory
of CTD casts, maybe a ship-ADCP archive — and the wish to process all of it without
hand-editing anything.

The whole workflow is four commands. They were designed so that a previously unseen
cruise — different naming conventions, different instruments, even desynced acquisition
clocks — processes end-to-end with **zero manual steps**.

## Step 0 · set the cruise root

Every command below uses `$B` for the cruise folder. Set it once per terminal:

```bash
cd /path/to/cruise            # the folder holding your raw data
B="$PWD"
echo "$B"                     # MUST print the path — if empty, set it again
```

!!! warning "An empty `$B` is the #1 support question"
    A new terminal forgets `$B`. If it is unset, `--ladcp "$B/raw_ladcp"` silently
    becomes `--ladcp /raw_ladcp` and the index reports
    **"indexed 0 casts from 0 files"**. Check `echo "$B"` first.

Your data can be laid out either way:

```text
$B/raw_ladcp/                      $B/raw_ladcp/
├── MASTER/  *.000   (down)   or   ├── MA*.000  SL*.000  ...   (flat —
└── SLAVE/   *.000   (up)          │   any names, both heads mixed)
```

With `MASTER/`/`SLAVE/` subdirs, the directory tells pyladcp which head is which.
In a **flat tree**, each PD0's own header does (the sysconfig *facing* bit) — no file
naming convention is required. Beam-coordinate instruments are rotated to earth
automatically at ingest.

## Step 1 · build the index

Raw deployment files have no station identity, so pyladcp builds one from the **CTD**:
each Seabird `.hex` header carries the station label, the GPS position, and absolute
NMEA UTC time. That anchor is matched to the LADCP deployment file covering the same
time window; the second head pairs by time overlap.

```bash
ladcp-index --root "$B" build --ladcp "$B/raw_ladcp" --ctd "$B/raw_CTD"
```

Expect a line like `indexed 38 casts from 96 files`, and the result cached in
`$B/.ladcp_archive.json`. Re-running only scans new files (`--rescan` forces a full
re-decode).

!!! note "`--root` comes *before* `build`"
    `ladcp-index --root "$B" build ...`, not `ladcp-index build --root ...`.

Two situations the index reports rather than guesses about:

- **A station you expected is missing** → its CTD cast is missing or unreadable; with
  no anchor, the cast cannot exist in the index. Compare the cast count against your
  logsheets.
- **`SHARED-MASTER` flag** → two CTD anchors claim the same deployment file (typical
  for back-to-back shelf stations recorded into one file). The pairing is usually
  right, but review those stations' reports with extra care.

## Step 2 · check the pairing

```bash
ladcp-index --root "$B" show
```

One line per station with its resolved master + slave file. Thirty seconds spent here —
confirming every cast got both heads and the names look sane — saves an hour of
confusion after the batch run.

## Step 3 · process everything

```bash
ladcp-qa --all-stations \
    --index "$B/.ladcp_archive.json" --root "$B" \
    --cruise MYCRUISE \
    -o "$B/qa_out"
```

A progress bar tracks the stations; full detail and any per-cast failures go to
`$B/qa_out/ladcp-qa.log`. **One bad cast is logged and skipped, never fatal** to the
batch.

What the options do:

- `--cruise MYCRUISE` — any name works. An unregistered cruise gets the shared
  operator defaults (8-m bins, bin-1 masks, tilt limits, RDI-firmware bottom track)
  and the name is stamped on the exports. Register a preset in `ladcp.config.CRUISES`
  only when a cruise needs special handling.
- `--all-stations` — every cast in the index, plus the **cruise-level exports**
  (see below). For a subset: `ladcp-qa 07 12 31 ... --cruise-export`.

Two common additions:

```bash
    --from-hex --ctd-cache "$B/ctd_from_hex" \      # build CTD from raw .hex
    --sadcp "$B/sADCP/DATA" \                       # ship-ADCP constraint
```

- `--from-hex` converts raw Seabird `.hex`/`.XMLCON` into the cleaned 1-s `.cnv` the
  solver needs (requires the companion CTD_pipeline package — see the
  [README](https://github.com/leopenausa/pyladcp#requirements)). A pre-processed
  `.cnv`, when present, always wins.
- `--sadcp` adds the ship's ADCP as an absolute constraint on the upper ocean
  ([chapter 8](08-ship-adcp.md)). If the ADCP PC's clock was wrong, add
  `--sadcp-timeoff auto --sadcp-nav <track>` and pyladcp recovers the offset by
  sliding the embedded GPS against an independently timestamped navigation track.

### What a cruise run produces

```text
$B/qa_out/
├── stations/<station>/        # per-station: report PDF, .lad, .bot, .nc, .xlsx, QA
├── exports/                   # cruise-level aggregates
│   ├── MYCRUISE_ladcp.xlsx    #   all stations in one workbook
│   ├── MYCRUISE_ladcp.nc      #   ... one NetCDF
│   ├── MYCRUISE_ladcp_odv.txt #   ... Ocean Data View import
│   └── MYCRUISE_summary.csv   #   one row per station (verdict, depths, metrics)
└── ladcp-qa.log               # the full run log
```

Triage from `MYCRUISE_summary.csv`: sort by verdict, read the WARN/FAIL stations'
reports first ([chapter 6](06-qa-report.md) tells you what each flag means).

## Step 4 · compare against a legacy processing

If the cruise was previously processed with LDEO_IX, close the loop — compare every
station, statistically and visually:

```bash
ladcp-compare --ours "$B/qa_out" --legacy "$B/legacy_processed" \
              -o "$B/qa_out/legacy_compare"
```

Stations are paired **by cast time** (legacy `dr.date` vs our `time_utc`), so no
filename conventions are assumed. You get:

- `comparison.csv` — per-station correlation, rms, bias for `u`, `v` and the bottom
  track, plus barotropic offsets and depth coverage;
- `comparison_report.pdf` — a summary page and per-station profile overlays with
  ±1σ solution-uncertainty bands;
- `unpaired.txt` — stations found on only one side. **Never silently dropped.**

Useful extras: `--alt-dir DIR --alt-stations a,b --alt-label "botfac=0"` substitutes
named stations from an alternate run (clearly labelled everywhere); `--sadcp <dir>`
overlays the ship-ADCP profile on each cast window as an independent third opinion.

## The whole thing on one screen

```bash
B="$PWD"
ladcp-index --root "$B" build --ladcp "$B/raw_ladcp" --ctd "$B/raw_CTD"
ladcp-index --root "$B" show
ladcp-qa --all-stations --index "$B/.ladcp_archive.json" --root "$B" \
         --cruise MYCRUISE -o "$B/qa_out"
ladcp-compare --ours "$B/qa_out" --legacy "$B/legacy_processed" \
              -o "$B/qa_out/legacy_compare"   # if a legacy processing exists
```

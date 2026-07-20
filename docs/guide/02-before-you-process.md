# 2 · Before you process

What pyladcp expects to find on disk, and what to take care of *at sea* so that
processing later is the four-command routine of [chapter 5](05-cruise-workflow.md)
rather than archaeology.

## LADCP files

- **Format:** RDI PD0 binary (`*.000` deployment files) from Workhorse-class
  instruments. Dual-head (down + up looker) is the design case; a cast with only a
  down-looker file solves down-only automatically (with a `single_head_solve` WARN);
  `--down-only` explicitly excludes an up-looker that exists.
- **Coordinates:** earth or beam — beam-coordinate data is rotated to earth
  automatically at ingest using the PD0's own geometry. (Recording in beam
  coordinates is actually the more conservative at-sea choice: it preserves the
  per-beam radial velocities.)
- **Naming:** none required. Heads are told apart by the `MASTER/`/`SLAVE/`
  directory convention when present, else by each file's own header (the sysconfig
  facing bit). Station identity comes from the CTD anchor, not from filenames.
- **One file may span several casts** (back-to-back shallow stations without
  stopping the recording) — the index handles it and flags it (`SHARED-MASTER`).

## CTD files

The CTD cast is **required** — it provides pressure (true depth), position, and the
absolute time anchor that gives raw LADCP files their station identity.

Two ways to provide it:

1. **Processed `.cnv`** (preferred when you have it): 1-s time series with merged
   GPS (lat/lon columns), pressure, temperature, salinity. This is the standard
   SBE-processing output most cruises produce anyway.
2. **Raw `.hex` + `.XMLCON`** with `--from-hex`: pyladcp builds the cleaned 1-s
   `.cnv` itself (needs the companion CTD_pipeline package — see
   [chapter 3](03-installation.md)). Stale XMLCON appended-data flags are reconciled
   against the `.hex` header automatically.

Either way, **the `.hex`/`.hdr` headers must be present for the indexing step** —
they carry the station label, NMEA UTC time and GPS position that anchor everything.

!!! warning "The one thing you cannot fix ashore"
    If the CTD deck unit was not receiving NMEA position/time, the `.hex` header has
    no anchor — and that cast cannot be automatically indexed. Verify NMEA feeds
    *before the first station*, not after the cruise.

## Navigation & ship-ADCP (optional but valuable)

- **An independent navigation track** (any time/lat/lon series: SADO exports, a
  bridge logger, …) enables the ship-ADCP clock-desync fix
  (`--sadcp-timeoff auto`, [chapter 8](08-ship-adcp.md)) and is generally the
  cheapest insurance against clock trouble.
- **Ship-ADCP (VmDAS) archive**: keep the *whole* `DATA` tree — `.STA`/`.LTA`
  averages for the direct route, and the raw `.ENR` + `.N1R`/`.N2R` single-ping
  files if you ever want the full CODAS product. Disk is cheap; re-occupying the
  cruise track is not.

## Recommended cruise layout

Any layout works (the index doesn't care), but this one makes chapter 5 copy-paste:

```text
CRUISE/
├── raw_ladcp/          # PD0s — MASTER/ + SLAVE/, or flat
├── raw_CTD/            # .hex .hdr .bl .XMLCON (+ processed .cnv if you have them)
├── sADCP/              # VmDAS tree as recorded (optional)
└── Navigation/         # independent nav export (optional)
```

## At sea: the logsheet

Processing problems are diagnosed months later with exactly two sources: the data
and the logsheet. Per cast, record at minimum: station label (**matching the CTD
console entry** — it becomes the station identity), deployment file names per head,
in/out-of-water times (UTC), echo-sounder depth, instrument battery voltage, and
*anything unusual* — device hung under the rosette, winch stops, restarted
recordings. Three of the validated pathologies in
[chapter 9](09-troubleshooting.md) are confirmed from logsheet notes, not from the
data. A printable checklist is in [appendix D](appendix-d-logsheets.md).

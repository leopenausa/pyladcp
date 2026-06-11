# Ingesting a new cruise (no code changes required)

The pipeline is designed so a previously unseen cruise — different naming
conventions, different instruments, desynced acquisition clocks — processes
end-to-end with four commands. Validated on MORIA (N Atlantic, earth-coord
WH300s, MASTER/SLAVE dirs) and a second full cruise (beam-coord WH300s,
flat `MA*`/`SL*` dir, OS75 SADCP with a ~12-year clock error).

## 1. Build the archive index

```bash
ladcp-index --root <work> build --ladcp <raw>/LADCP --ctd <raw>/CTD
```

* LADCP dir may hold `MASTER/` + `SLAVE/` subdirs **or** a flat tree — heads
  are classified by path convention when present, else by the PD0 sysconfig
  facing bit. No filename convention required.
* Each cast is anchored by its Seabird `.hex` header (station label, NMEA UTC,
  position) and matched to the deployment file covering that time; slave pairs
  by overlap. Two anchors claiming one file are flagged `SHARED-MASTER` in the
  provenance — review those.
* Stations whose CTD cast is missing simply don't appear (they cannot be
  anchored); compare the cast count against expectations.

## 2. Process every station

```bash
ladcp-qa --all-stations \
    --index <work>/.ladcp_archive.json --root <work> \
    --from-hex --ctd-cache <work>/ctd_from_hex \
    --cruise <NAME> --out <out>
```

* `--cruise <NAME>`: any name works — an unregistered cruise gets the shared
  operator defaults (dz=8, bin-1 masks, cut=7, pglim=50, tilt 22/4, btrk 3)
  with `<NAME>` stamped on exports. Register a preset in `ladcp.config.CRUISES`
  only for cruise-specific layers (e.g. MORIA's monocorer mask).
* `--from-hex` builds cleaned 1-s `.cnv` from the raw `.hex`+`.XMLCON`
  (needs the CTD_project checkout). Stale XMLCON appended-data flags are
  reconciled against the `.hex` header automatically.
* Beam-coordinate PD0s are rotated to earth at ingest; magnetic declination is
  IGRF from the cast position unless `--drot` is given.

## 3. Shipboard ADCP (optional constraint / validation)

```bash
  ... --sadcp <raw>/SADCP_STA_dir --sadcpfac 3 \
      --sadcp-timeoff auto --sadcp-nav <raw>/Navigation
```

* If the SADCP acquisition PC was not synced to GPS time (wrong dates/times in
  the STA files), `--sadcp-timeoff auto` recovers the constant clock offset by
  sliding the STA's embedded GPS positions against an independently timestamped
  navigation track (`--sadcp-nav`: SADO `posicion` exports or any time/lat/lon
  CSV; file or directory). On the second validation cruise this matched the operators' manual
  correction to 5 s with a 10-m median track residual. A known offset can be
  given directly in seconds.
* A folder holding both a compiled STA and its per-deployment parts is fine —
  duplicate ensembles are dropped.

## 4. Compare against a legacy-processed reference (when one exists)

```bash
ladcp-compare --ours <out> --legacy <legacy_processed_dir> -o <out>/legacy_compare
```

* Pairs stations **by cast time** (legacy `dr.date` vs our `time_utc`) — no
  filename assumptions; unpaired stations on either side are listed in
  `unpaired.txt`, never silently dropped.
* Emits `comparison.csv` (corr/rms/bias for u, v, bottom track; barotropic
  offsets; depth coverage) and `comparison_report.pdf` (summary page + per-
  station u/v profile overlays with ±1σ solution-uncertainty bands).
* `--alt-dir DIR --alt-stations a,b --alt-label "botfac=0"` substitutes named
  stations from an alternate run — explicitly labelled in the CSV, profile
  titles and summary page, never silently mixed.
* `--sadcp <STA dir> --sadcp-timeoff <seconds>` overlays the ship-ADCP profile
  over each cast window (green) next to the constraint profile the legacy run
  actually used (olive squares, from `dr.z_sadcp`) — an independent third
  opinion on the upper ocean.

## Reference results

| cruise | stations | median u rms vs legacy | notes |
|--------|----------|------------------------|-------|
| cruise 2 | 30/31    | ~1.4 cm/s (corr 0.97)  | 31st has no CTD raw; 2 shallow casts reported with `--botfac 0` (bad near-bottom firmware BT samples, labelled in the report) |

Three fixes discovered on that cruise are now defaults: the bottom track prefers the
RDI firmware track when the PD0 carries one (legacy `btrk_mode=3`; the own
water-track BT inherits boundary-layer flow and biased strong-drift casts by
~10–15 % of the current), the in-water window is the deep segment containing
the deepest ping (pre-cast surface soaks no longer leak drift into the
barotropic reference), and package depth is W-integrated past the end of a
prematurely-stopped CTD record.

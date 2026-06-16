# Appendix C · File formats

## `<station>.lad` — the velocity profile (LDEO text format)

```text
Filename    = MORIA-80
Date        = 2025/ 4/29
Start_Time  = 12:34:56
Start_Lat   = ...
Start_Lon   = ...
Deviation   = -5.441000        ← magnetic declination applied [deg]
Columns     = z:u:v:ev
  10.0  0.012 -0.034  0.039
  18.0  0.015 -0.031  0.037
  ...
```

Columns: depth [m, positive down], east velocity, north velocity (true frame,
declination applied), velocity uncertainty [m/s]. Bins with no solution are omitted,
not written as zeros. The format follows the legacy `.lad` layout, so downstream
readers keep working.

## `<station>.bot` — bottom-track-referenced profile

Same header plus a `Bottom depth=` line; `Columns = z:u:v:err`. Near-seabed ocean
velocity referenced by the bottom track alone — an independent check on the
near-bottom part of `.lad`.

## `<station>.nc` — NetCDF (CF conventions)

The profile with `z` as coordinate; `u`, `v`, `uerr`, `nvel` (samples per bin) as
variables; scalar `latitude`, `longitude`, `time` coordinates; global attributes
carrying station, cruise, solver, constraint weights, declination (+ provenance) and
the SADCP source when one was used. `u`/`v` carry CF `ancillary_variables` pointing at
`uerr`, which is a CF `standard_error` of the velocity — a *formal* uncertainty (the
inverse covariance), **not** a full error budget (for the empirical error, see
`sadcp_independent_rms` in [chapter 6](06-qa-report.md) / the summary CSV). Open with
xarray: `xr.open_dataset("MORIA-80.nc")`.

## `<station>.xlsx` — Excel (needs `pyladcp[export]`)

Sheets: profile (`z/u/v/uerr/nvel`), bottom track, QA metrics, and a metadata sheet
(position, time, settings) — self-locating, so a sheet copied out of the workbook
still identifies its cast.

## `<station>_qa.json` — the scorecard, machine-readable

```json
{
  "station": "MORIA-80",
  "overall_status": "warn",
  "warnings": ["..."],
  "metrics": {
    "ctd_sync_corr": {"value": 0.975, "unit": "", "status": "ok",
                       "note": "offset 431 s; ...", "source_stage": "qa.depth"}
  }
}
```

One entry per scorecard row ([chapter 6](06-qa-report.md)); `status` ∈
`ok|warn|fail`. `_qa.txt` is the same content rendered for humans.

## Cruise-level exports (`exports/`, from `--all-stations` / `--cruise-export`)

| file | content |
|---|---|
| `<CRUISE>_ladcp.nc` | all stations, station dimension + per-station coords |
| `<CRUISE>_ladcp.xlsx` | one workbook, one sheet per product |
| `<CRUISE>_ladcp_odv.txt` | Ocean Data View generic spreadsheet import (station, position, time, depth, u, v, err) |
| `<CRUISE>_summary.csv` | one row per station: verdict, depths, key metrics (incl. `sadcp_independent_rms_ms`, the withheld empirical uncertainty) — the triage entry point |

## Inputs (for reference)

- **PD0** (`*.000`): RDI binary deployment files; ensembles with velocity,
  correlation, echo amplitude, percent-good, attitude, and (when the firmware
  produced it) bottom-track data. pyladcp reads them natively (`ladcp.io.pd0`).
- **CTD `.cnv`**: SeaBird processed text; pyladcp auto-maps the column names it
  needs (pressure, T, S, lat/lon, time).
- **CTD `.hex`/`.hdr`**: raw SeaBird; only the *header* is read at indexing time
  (station, NMEA UTC, position); the full file is converted only under
  `--from-hex`.
- **VmDAS `.STA`/`.LTA`**: ship-ADCP averages (PD0 dialect); `.ENR`+`.N1R/.N2R`
  single-ping raw is CODAS territory ([chapter 8](08-ship-adcp.md)).

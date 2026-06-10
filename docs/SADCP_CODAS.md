# SADCP processing with CODAS — one-page guide

How to turn raw shipboard-ADCP (VmDAS) data into a calibrated, edited product with
CODAS, and how that product plugs into pyladcp. CODAS (UH/SOEST) adds three things the
raw `.STA`/`.LTA` averages lack: an **editing pass** (bad bins flagged), a **watertrack
calibration** (transducer misalignment angle + amplitude scale), and **smoothed GPS
navigation**. None of that is needed for the on-station LADCP constraint — a 40-cast
MORIA A/B showed raw and CODAS constraints interchangeable to ~1 cm/s — but it matters
**underway** (a misalignment of φ degrees biases velocities by ≈ sin φ × ship speed;
an amplitude error of a% biases by ≈ a% × ship speed ≈ 10 cm/s per % at 5 m/s), so use
CODAS for transect/section work and as an independent check of the raw chain.

## 1. One-time setup (~15 min)

```bash
bash scripts/codas_install.sh     # conda env "pycodas" + codas3/pycurrents/onship + demos
```

## 2. Process a cruise (MORIA OS150 example, ~5 min)

```bash
bash scripts/codas_moria_os150.sh          # for another cruise/instrument, copy + edit
```

The script does four things: (1) `adcptree.py` makes the processing tree; (2) stages the
`.STA` files as `NNN_`-prefixed **chronological copies** — do not skip this: parts of
CODAS glob inputs by *filename*, and on MORIA a name-ordered load misaligned every
position by 682 ensembles and silently poisoned all absolute velocities (the watertrack
cal *still looked fine*; only checking positions against a known cast caught it);
(3) writes the control file; (4) runs `quick_adcp.py --auto`.

Control-file parameters that matter (`q_py.cnt`):

| parameter | meaning / how to choose |
|---|---|
| `--yearbase` | year of the first data day; the NetCDF time axis is "days since `yearbase`-01-01" |
| `--datatype sta` | input flavour. Release CODAS accepts `sta`/`lta`/`uhdas`/`pingdata` only — VmDAS single-ping (`enx`) is rejected despite the help text |
| `--sonar os150nb` | instrument + mode; picks bin geometry defaults (e.g. `os75bb` for the OS75) |
| `--ens_len 120` | the onboard STA averaging interval [s] — read it from the VmDAS `.VMO` config |
| `--max_search_depth` | how deep to look for the seabed in bottom-tracking; ≥ shelf depth of the cruise |

## 3. Calibrate (the step `--auto` does NOT do)

`--auto` *computes* the watertrack calibration but does not *apply* it. Read
`cal/watertrk/adcpcal.out` and use the **median** amplitude and phase:

```bash
conda activate pycodas && cd <workdir>/os150nb_sta
tail -20 cal/watertrk/adcpcal.out      # e.g. amplitude median 1.021, phase median 0.003
quick_adcp.py --cntfile q_py.cnt --auto \
  --steps2rerun rotate:navsteps:calib:matfiles:netcdf \
  --rotate_angle 0.003 --rotate_amplitude 1.021
```

Significance: **phase** = transducer misalignment (deg, rotates u/v); **amplitude** =
beam-geometry scale factor (multiplies speeds). Both produce ship-speed-proportional
errors, so they are invisible on station and dominant at full steam. Only trust a cal
with a healthy point count (MORIA: 72/96) and re-check after applying that the product
is sane against a known cast position — *a good cal alone does not validate the product*.

The deliverable is **`contour/<sonar>.nc`**: absolute ocean u/v per ensemble (`pflag`
editing applied, end-of-average time stamps), positions, `uship`/`vship`.

## 4. Use it in pyladcp

Raw VmDAS stays the default; CODAS is opt-in wherever a SADCP source is accepted:

```bash
# inverse constraint (per-cast window is cut automatically)
ladcp-qa 80 --index .ladcp_archive.json \
  --sadcp <workdir>/os150nb_sta/contour/os150nb.nc --sadcp-source codas
```

`--sadcp` takes the `.nc` file, its `contour/` dir, or the processing dir. Exports
record the provenance as `sadcp_source: codas:<path>`. Keep the default (`vmdas`) for
routine station processing; pick `codas` when the cruise has long steaming legs inside
cast windows or when you want the calibrated/edited series.

## 5. Section plots

```bash
# raw VmDAS (needs the GPS-leak screen --speed-max, on by default)
ladcp-sadcp-section --sadcp <cruise>/sADCP/sadcp_150/DATA --by time -o sec_time.png
# CODAS product (already edited; cleaner underway)
ladcp-sadcp-section --sadcp <workdir>/os150nb_sta/contour/os150nb.nc --source codas \
  --by distance --index .ladcp_archive.json -o sec_dist.png
```

`--by time|distance` picks the x-axis (ship clock vs along-track distance); `--index`
draws LADCP station ticks; `--max-depth`/`--clim`/`--start`/`--end` crop and scale.
`--speed-max` only affects raw VmDAS input.

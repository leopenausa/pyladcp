# Appendix A · CLI flag reference

This appendix is **generated from the live `--help` output** of each pyladcp command
(`docs/guide/gen_flag_reference.py`); a CI test fails whenever it goes stale, so what
you read here is exactly what the installed code accepts.


## `ladcp-qa`

```text
usage: ladcp-qa [-h] [--root ROOT] [--cruise CRUISE] [--index INDEX]
                [-o OUTDIR] [--no-plots] [--drot DROT]
                [--solver {shear,inverse}] [--botfac BOTFAC]
                [--barofac BAROFAC] [--smoofac SMOOFAC] [--down-only]
                [--nearfield-dn-bins LIST] [--dzbelow METERS] [--sadcp PATH]
                [--sadcp-source {vmdas,codas}] [--sadcpfac SADCPFAC]
                [--sadcp-filetype {STA,LTA}] [--sadcp-xducer SADCP_XDUCER]
                [--sadcp-reingest] [--sadcp-timeoff SECONDS|auto]
                [--sadcp-nav PATH] [--down DOWN] [--up UP] [--ctd CTD]
                [--from-hex] [--ctd-cache CTD_CACHE] [--station STATION]
                [--no-export] [--formats FORMATS] [--all-stations]
                [--cruise-export] [-v] [--log FILE] [--no-log] [--no-progress]
                [-j N]
                [stations ...]

LADCP acquisition quality assessment

positional arguments:
  stations              station id(s), e.g. 80 or 79 80 82

options:
  -h, --help            show this help message and exit
  --root ROOT           base dir for file discovery (default: New_golden/Good)
  --cruise CRUISE       cruise preset for params + raw-archive manifest
                        (default: MORIA)
  --index INDEX         archive index JSON (ladcp-index build); resolves raw
                        files by station
  -o OUTDIR, --out OUTDIR, --outdir OUTDIR
                        output directory (default: qa_out)
  --no-plots            skip figures/PDF
  --drot DROT           magnetic declination [deg] for velocity (default: IGRF
                        from position)
  --solver {shear,inverse}
                        velocity solver: full constrained inverse (default) or
                        shear shape+reference
  --botfac BOTFAC       bottom-track constraint weight, legacy ps.botfac
                        (inverse only; default: 1)
  --barofac BAROFAC     GPS barotropic constraint weight, legacy ps.barofac
                        (inverse only; default: 1)
  --smoofac SMOOFAC     curvature-smoothing weight, legacy ps.smoofac (inverse
                        only; default: 0, golden value)
  --down-only           solve velocity from the down-looker alone, ignoring
                        any up-looker (cross-check / single-instrument casts);
                        acquisition QA still covers both heads
  --nearfield-dn-bins LIST
                        override the down-looker near-field device mask: comma
                        1-based bins (e.g. 3,4) or 'none' to disable; default:
                        the cruise preset (MORIA sets 3,4 on the monocorer
                        block 03-28)
  --dzbelow METERS      below-/near-seabed cell rejection margin [m] (default:
                        cruise preset, 16 = 2 legacy bins). Raise on shallow
                        shelf casts when a bottom-depth underestimate lets a
                        contaminated near-bottom cell through (e.g. 24-32)
  --sadcp PATH          shipboard-ADCP data for the inverse constraint: a
                        VmDAS folder (STA/LTA; ingested once and cached as
                        sadcp_cache.npz) or, with --sadcp-source codas, a
                        CODAS contour NetCDF (file or its processing dir)
  --sadcp-source {vmdas,codas}
                        what --sadcp points at: raw VmDAS averages (default)
                        or a CODAS-processed (edited+calibrated) NetCDF
                        product
  --sadcpfac SADCPFAC   ship-ADCP constraint weight (default: 3, the golden
                        value)
  --sadcp-filetype {STA,LTA}
                        VmDAS average to read (default: STA, short-term)
  --sadcp-xducer SADCP_XDUCER
                        SADCP transducer depth below waterline [m] (default:
                        5)
  --sadcp-reingest      re-parse the raw SADCP tree, ignoring any existing
                        cache
  --sadcp-timeoff SECONDS|auto
                        clock correction added to the SADCP timestamps [s], or
                        'auto' to estimate it from --sadcp-nav by track
                        matching (for acquisition PCs that were not
                        synchronised to GPS time)
  --sadcp-nav PATH      independently timestamped navigation track (file or
                        dir; SADO 'posicion' exports or time/lat/lon CSV) for
                        --sadcp-timeoff auto
  --down DOWN           down-looker (Master) PD0 file
  --up UP               up-looker (Slave) PD0 file
  --ctd CTD             cleaned CTD .cnv (enables depth/bottom)
  --from-hex            if no cleaned CTD .cnv is found, build one from the
                        index's raw Seabird .hex anchor (needs CTD_project;
                        off by default)
  --ctd-cache CTD_CACHE
                        dir to cache --from-hex converted .cnv for reuse
                        (default: ctd_from_hex)
  --station STATION     station label (explicit mode)
  --no-export           skip the xlsx/odv/nc/csv exports (keep
                        lad/bot/report/qa)
  --formats FORMATS     comma-list of export formats to emit (default:
                        xlsx,odv,nc,csv)
  --all-stations        process every station in the --index and build the
                        cruise exports/
  --cruise-export       also build the cruise-level exports/ aggregate over
                        the named stations
  -v, --verbose         stream per-station detail to the console (default:
                        progress bar only)
  --log FILE            run-log path (default: <outdir>/ladcp-qa.log)
  --no-log              do not write a run-log file
  --no-progress         disable the batch progress bar
  -j N, --jobs N        process N stations in parallel (default: 1; 0 = one
                        per CPU). Each worker holds one cast in memory --
                        reduce N if you swap
```

## `ladcp-index`

```text
usage: ladcp-index [-h] [--root ROOT] {build,show} ...

build / inspect the LADCP archive index

positional arguments:
  {build,show}
    build       scan the archive and (re)write the index
    show        print the resolved casts

options:
  -h, --help    show this help message and exit
  --root ROOT   base dir stored paths are relative to
```

## `ladcp-index build`

```text
usage: ladcp-index build [-h] --ladcp LADCP --ctd CTD
                         [--master-subdir MASTER_SUBDIR]
                         [--slave-subdir SLAVE_SUBDIR] [-o OUT] [--rescan]

options:
  -h, --help            show this help message and exit
  --ladcp LADCP         LADCP archive dir: MASTER/ + SLAVE/ subdirs when
                        present, else all *.000 in the tree (heads classified
                        by name or PD0 facing bit)
  --ctd CTD             dir of Seabird CTD .hex/.hdr files (anchor)
  --master-subdir MASTER_SUBDIR
  --slave-subdir SLAVE_SUBDIR
  -o OUT, --out OUT     index path (default <root>/.ladcp_archive.json)
  --rescan              force full re-decode (ignore cache)
```

## `ladcp-index show`

```text
usage: ladcp-index show [-h] [--index INDEX]

options:
  -h, --help     show this help message and exit
  --index INDEX  index path (default <root>/.ladcp_archive.json)
```

## `ladcp-compare`

```text
usage: ladcp-compare [-h] --ours OURS --legacy LEGACY [--tol-hours TOL_HOURS]
                     [--alt-dir DIR] [--alt-stations LIST]
                     [--alt-label ALT_LABEL] [--sadcp PATH]
                     [--sadcp-timeoff SECONDS] [--sadcp-filetype {STA,LTA}]
                     [--sadcp-xducer SADCP_XDUCER] [--title TITLE] [-o OUT]

pyladcp vs legacy-LDEO cruise comparison report

options:
  -h, --help            show this help message and exit
  --ours OURS           ladcp-qa output dir (holds stations/*/<st>.nc)
  --legacy LEGACY       legacy processed dir (LDEO *.mat with dr structs)
  --tol-hours TOL_HOURS
                        max cast-time difference for a station pair (default
                        3.0 h)
  --alt-dir DIR         alternate ladcp-qa run dir to substitute --alt-
                        stations from (e.g. a --botfac 0 rerun of shallow
                        casts)
  --alt-stations LIST   comma-separated stations to take from --alt-dir
  --alt-label ALT_LABEL
                        label stamped on substituted stations in the report
                        (default: 'alternate')
  --sadcp PATH          ship-ADCP VmDAS folder (STA/LTA; cached): overlay the
                        ship-ADCP profile over each cast window on the report
  --sadcp-timeoff SECONDS
                        clock correction added to the ship-ADCP timestamps
                        (the value ladcp-qa --sadcp-timeoff auto reported)
  --sadcp-filetype {STA,LTA}
  --sadcp-xducer SADCP_XDUCER
                        ship-ADCP transducer depth below waterline [m]
                        (default 5)
  --title TITLE
  -o OUT, --out OUT     report dir (default <ours>/legacy_compare)
```

## `ladcp-sadcp-section`

```text
usage: ladcp-sadcp-section [-h] --sadcp PATH [--source {vmdas,codas}]
                           [--by {time,distance}] [--filetype {STA,LTA}]
                           [--xducer XDUCER] [--start START] [--end END]
                           [--max-depth MAX_DEPTH] [--clim CLIM]
                           [--speed-max SPEED_MAX] [--anomaly] [--index INDEX]
                           [--title TITLE] [-o OUT]

Shipboard-ADCP u/v sections (depth vs time or distance)

options:
  -h, --help            show this help message and exit
  --sadcp PATH          VmDAS folder (STA/LTA; ingested once and cached) or,
                        with --source codas, a CODAS contour NetCDF (file or
                        processing dir)
  --source {vmdas,codas}
                        what --sadcp points at: raw VmDAS averages (default)
                        or a CODAS-processed NetCDF product
  --by {time,distance}  x-axis: ship clock (default) or along-track distance
  --filetype {STA,LTA}
  --xducer XDUCER       transducer depth below waterline [m] (default 5)
  --start START         window start (ISO, e.g. 2025-10-03)
  --end END             window end (ISO)
  --max-depth MAX_DEPTH
                        crop depth axis [m]
  --clim CLIM           symmetric colour range [m/s] (default: robust auto)
  --speed-max SPEED_MAX
                        blank ensembles whose median speed exceeds this [m/s]
                        (GPS-leak screen; 0 disables; default 1.5)
  --anomaly             subtract each ensemble's depth-mean velocity
                        (baroclinic shear section; reveals vertical structure
                        under a barotropic flow)
  --index INDEX         archive index JSON: draw LADCP station ticks
  --title TITLE         figure title
  -o OUT, --out OUT     output PNG (default sadcp_section.png)
```

## `ladcp-studio`

```text
usage: ladcp-studio [-h] [--root ROOT] [--cruise CRUISE] [--index INDEX]
                    [--from-hex] [--ctd-cache CTD_CACHE] [--sadcp PATH]
                    [--sadcp-source {vmdas,codas}]
                    [--sadcp-filetype {STA,LTA}] [--sadcp-xducer SADCP_XDUCER]
                    [--sadcp-timeoff SECONDS|auto] [--sadcp-nav PATH]
                    [--sadcp-reingest] [--host HOST] [--port PORT]
                    [--no-browser]
                    [stations ...]

pyladcp Studio: interactive single-station LADCP processing in the browser
(local server)

positional arguments:
  stations              station id(s) to work on, e.g. 80 79

options:
  -h, --help            show this help message and exit
  --root ROOT           base dir for file discovery (default: New_golden/Good)
  --cruise CRUISE       cruise preset (default: MORIA)
  --index INDEX         archive index JSON (ladcp-index build); with no
                        station ids, serves every cast in the index
  --from-hex            build missing cleaned CTD from the index's raw .hex
                        anchor
  --ctd-cache CTD_CACHE
                        cache dir for --from-hex .cnv
  --sadcp PATH          ship-ADCP source for the inverse constraint (as in
                        ladcp-qa); the GUI can then toggle/weight it per solve
  --sadcp-source {vmdas,codas}
  --sadcp-filetype {STA,LTA}
  --sadcp-xducer SADCP_XDUCER
  --sadcp-timeoff SECONDS|auto
  --sadcp-nav PATH
  --sadcp-reingest
  --host HOST           bind address (default: localhost)
  --port PORT           port (default: 8642)
  --no-browser          do not open the browser
```

# LADCP Rebuild — QA-First Plan (MORIA New_golden)

> **Pivot (2026-06-05):** GO-SHIP golden abandoned. New reference = MORIA `New_golden/`
> (stations 79/80/82 "Good"; station 80 fully instrumented with `.log`/`.lad`/`.bot`/`.mat`/figures).
> **New priority: data-acquisition quality assessment BEFORE any velocity computation.**
> Velocity inversion is explicitly out of scope for this build.

## Guiding principles
- **Acquisition QA is independent of the inverse.** Figures 2/14/4/6 and all health
  metrics derive from raw instrument fields (echo amplitude, correlation, attitude,
  percent-good) — never from u/v. Build and trust these first.
- **Validate against unambiguous physics**, not suspect velocities. Targets come from
  the MORIA-80 `p`-struct (`MORIA-80.mat`) and `.log` step counts.
- **Modular, scalable**: each QA diagnostic is its own small module returning numbers
  + a render hook. Figures are a thin layer ("same info, modern layout").
- **Reuse only** `io/pd0.py` (PD0 decode) and `proc/magdec.py`. Everything in `ix/`
  and the inversion path is set aside (not deleted, not extended).

## Validation targets (MORIA-80) — frozen reference
| Quantity | Golden | Source |
|---|---|---|
| ensembles down/up | 3933 / 3935, 30 bins | `p.nping_total`, log |
| bin len / blank / 1st cell / beam | 8 m / 1.76 m / 10.11 m / 20° | `p.blen_*,blnk_*,dist_*,beamangle` |
| thresholds | pg≥50, elim 0.2, vlim 1, wlim 0.08, tilt [22,4] | `p.pglim,elim,vlim,wlim,tiltmax` |
| pg<50 removed | 59727 down / 65340 up | log step 1 |
| beam range down/up | [170,170,170,170] / [170,154,162,170] | `p.dn_range,up_range` |
| xmit V / battery | [125,121] / 41.26 V | `p.xmv,battery` |
| dual-head offsets | comp −60.23°, pitch −1.35°, roll +1.11° | `p.up_dn_*` |
| tilt rejections | 6 (>22°), 116 (deriv>4°) | log step 1 |
| edit counts | bin-mask 5071, side-lobe 2022 | log step 9 |
| bottom / maxdepth | 1079.0 / 1072.7 m (±0.56) | `p.zbottom,maxdepth` |
| **NOT available** | raw bin×ensemble arrays (`da` truncated) | — |

## Instrument config (from launch scripts — ground truth, not guessed)
Master(down)+Slave(up), hardware-synchronized (SA001/SI0/SW75). WM15, 20 bins ×
8 m, blank 1.76 m, 1 ping/ensemble, 1 ensemble/s, narrowband (WB1), WV170,
fixed sound speed EC1450, EX11111, EA0/EB0.

## CTD `.cnv` format (clean files)
Whitespace columns, **no header** (header_lines=0):
`1:lat  2:lon  3:pressure[dbar]  4:time[s]  5:temperature  6:salinity`
(per `f.ctd_*_field` in MORIA-80.mat). Surface-soak shows pressure bobbing 4–9 dbar.

---

## Scope of THIS build: **Ingest + Core Health**
(Editing visualization Fig14, surface/bottom Fig4 detail, and velocity follow later.)

### Phase 0 — Skeleton & harness
- New package subtree `src/ladcp/qa/` + `src/ladcp/plots/`.
- `io/ctd_cnv.py`: clean-CTD reader → (lat, lon, p, t, s, time).
- `validation/moria80_targets.py`: the frozen table above as assertable constants,
  loaded from `MORIA-80.mat` `p`-struct at test time (no hardcoding drift).
- Test fixtures point at `New_golden/Good/LADCP/MORIA-80-LADCP-{M,S}.000`.

### Phase 1 — Dual-head ingest (`qa/ingest.py`)
- Decode both PD0 files via existing `io/pd0.py`.
- Extract per-instrument config (bins, blen, blank, dist, beamangle, xmv/xmc,
  serial/instid, ping counts) → assert against `p`.
- Assemble raw fields: velocity[bin,ens,beam or uvw], echo amplitude, correlation,
  percent-good, heading/pitch/roll, time. Up + down kept separate (no rotation yet).
- **Gate:** ensemble/bin counts and all config scalars match `p`/log exactly.

### Phase 2 — Threshold screening (`qa/screen.py`) — the `loadrdi` edits
- Percent-good < 50 → NaN; count removed (target 59727 down / 65340 up).
- Error-velocity > 0.2, horizontal speed > vlim, bin-1 weight ×0.1.
- Tilt > 22° and |d(tilt)| > 4° ensemble rejection (targets 6 / 116).
- **Gate:** removal counts match log step 1 within tight tolerance.

### Phase 3 — Core health metrics (`qa/beams.py`, `range.py`, `attitude.py`, `wfield.py`)
- **beams.py** — per-beam S/N from echo amplitude; broken/bad/weak flags
  (ports `checkbeam` logic in `plotraw.m`). Validate beam-range % pattern.
- **range.py** — profiling range from correlation drop; `dn_range`/`up_range`
  per beam (targets [170,170,170,170] / [170,154,162,170]).
- **attitude.py** — tilt/heading/depth/xmv time series + stats; dual-head
  alignment offset via tilt-sensor fit (`checktilt`/`fixcompass`) → comp −60.23°,
  pitch −1.35°, roll +1.11°. Battery from xmv.
- **wfield.py** — W as f(bin-depth, ensemble) array (the "bowtie") + 3-beam fraction.
- **report.py** — aggregate all metrics into a JSON + human-readable QA report,
  including reproduced warnings (no fixed ping rate, N horiz vel >1 m/s, etc.).

### Phase 4 — Figures (modern layout, `plots/`)
- `raw_dashboard.py` ≈ Fig 2 (W field, beam performance, range, depth/tilt/heading/xmv).
- `alignment.py` ≈ Fig 6 (heading/pitch/roll difference vs down).
- Driven entirely by Phase-3 outputs; styling is Python-idiomatic, not pixel-faithful.

### Cross-station check
Run the full chain on 79 / 80 / 82. Station 80 is the numeric anchor (has `p`);
79 & 82 confirm robustness (config consistency, beam health, no crashes).

## MATLAB authority files for this build
`loadrdi.m` (ingest+screen counts), `plotraw.m`/`checkbeam` (beams, range, W field),
`checktilt.m` + `fixcompass.m` (attitude offsets), `getbtrack.m`/`checkbtrk2.m`
(beam/bottom echo, later). Figs 4/14 deferred with their scripts `getdpth*`/`edit_data`.

## Out of scope (later phases)
Editing visualization (Fig 14), surface/bottom detail (Fig 4), super-ensembles,
inverse solution (Figs 1/3/12), SADCP, magnetic-declination decision, `.lad`/`.bot` export.

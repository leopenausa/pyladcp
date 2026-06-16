# 8 · Ship-ADCP integration

Most research vessels run a hull-mounted ADCP (VmDAS/UHDAS) continuously. For LADCP
work it is two things at once:

- an **absolute velocity constraint** on the upper ocean — the only reference that
  acts there (the bottom track acts at depth, GPS only on the depth-mean), and
- an **independent instrument measuring the same water** — the headline external
  accuracy check on your finished profiles (`sadcp_consistency` and the withheld
  `sadcp_independent_rms`, [chapter 6](06-qa-report.md)).

> **In-sample vs independent.** When the ship-ADCP is used as a constraint (the default),
> the LADCP is pulled toward it, so `sadcp_consistency` understates the true error.
> pyladcp therefore also reports `sadcp_independent_rms` — the same RMS from an automatic
> second solve with the ship-ADCP **withheld** (`sadcpfac=0`) — as the honest empirical
> velocity-uncertainty estimate (~0.02–0.06 m/s for high-quality data; Thurnherr 2010).

You can use it three ways, in increasing order of effort: feed the raw VmDAS averages
straight in (sufficient for on-station work), fix a broken acquisition clock first,
or process it properly with CODAS (needed for underway/section work).

## The direct route: raw VmDAS averages

```bash
ladcp-qa ... --sadcp "$B/sADCP/sadcp_150/DATA"
```

`--sadcp` points at the folder holding the `.STA` (short-term average) files; pyladcp
cuts each cast's time/space window automatically and adds the profile as the
`lainsadcp` constraint with weight `--sadcpfac` (default 3, the validated value —
[chapter 7](07-solvers-weights.md)).

Practical knobs:

- `--sadcp-filetype LTA` — read long-term averages instead of `.STA`.
- `--sadcp-xducer 5` — transducer depth below the waterline [m]; offsets the bin
  depths.
- The tree is parsed once and cached (`sadcp_cache.npz` next to the data);
  `--sadcp-reingest` rebuilds the cache after you change the source. A folder
  holding both a compiled file and its per-deployment parts is fine — duplicate
  ensembles are dropped.

**How good is the raw constraint?** On station, very: a 40-cast A/B against the fully
processed CODAS product showed the two constraints interchangeable at the ~1 cm/s
level. The raw averages' weaknesses appear *underway* (see the CODAS section below).

## When the acquisition clock is wrong

The VmDAS PC's clock is not always synced to GPS — offsets from seconds to *years*
occur in real archives, and a wrong clock silently breaks the cast-window matching.

```bash
ladcp-qa ... --sadcp "$B/sADCP/DATA" \
    --sadcp-timeoff auto --sadcp-nav "$B/Navigation"
```

`--sadcp-timeoff auto` recovers the constant offset by sliding the SADCP's *embedded*
GPS positions against an independently timestamped navigation track (`--sadcp-nav`:
any time/lat/lon CSV or a SADO `posicion` export; file or directory). On the
validation cruise that needed it, the automatic estimate matched the operators'
manual correction to ~5 s with a 10 m median track residual. A known offset can be
passed directly in seconds. The applied correction is logged and stamped into the
exports' provenance.

## Constraint *and* referee

With `--sadcp` active you get both roles at once:

- the **weights figure** shows where in the column the SADCP rows pull
  ([chapter 7](07-solvers-weights.md));
- the **ship-ADCP comparison page** and the `sadcp_consistency` row score the
  *finished* profile against the SADCP over their shared depths
  ([chapter 6](06-qa-report.md)).

Don't skip the referee role even if you don't want the constraint: a run with
`--sadcpfac 0 --sadcp <dir>` keeps the comparison while taking the SADCP out of the
solve — useful when you need the two instruments to stay strictly independent.
(When you *do* keep the constraint, `sadcp_independent_rms` already gives you this
withheld comparison automatically, so you get both numbers from a single run.)

## Section plots — `ladcp-sadcp-section`

The SADCP archive is also a data set in its own right. `ladcp-sadcp-section` renders
the cruise's u/v fields as depth–time or depth–distance sections:

```bash
# raw VmDAS
ladcp-sadcp-section --sadcp "$B/sADCP/sadcp_150/DATA" --by time -o sec_time.png

# CODAS product (already edited; the right source for underway sections)
ladcp-sadcp-section --sadcp <workdir>/os150nb_sta/contour/os150nb.nc --source codas \
    --by distance --index "$B/.ladcp_archive.json" -o sec_dist.png
```

- `--by time|distance` — x-axis: ship clock or along-track distance.
- `--index` — draws the LADCP station positions as ticks, so you can see which
  features your casts sampled.
- `--max-depth` / `--clim` — crop and colour-scale.
- `--speed-max` (default 1.5 m/s) — blanks ensembles whose median water speed is
  implausible; built for raw-VmDAS GPS leaks, it also catches residual bad CODAS
  ensembles. `0` disables.
- `--anomaly` — subtracts each ensemble's depth-mean before rendering: the
  **baroclinic (shear) section**. Use it when the section looks vertically
  homogeneous — wherever the depth-mean flow is as large as the vertical structure,
  the absolute view renders each column as one colour, and the anomaly view is where
  the structure lives.

## When you need CODAS

The raw `.STA` averages are formed onboard from **unedited** pings. On station that's
fine; **underway** the bins beyond the bubble/noise-limited range are garbage that
survives average-level screening — on the validation data, deep underway cells had an
rms u of 0.68 m/s in the STA product vs 0.13 m/s after per-ping editing. CODAS
(UH/SOEST) adds what the raw averages lack: a per-bin editing pass, a watertrack
**calibration** (misalignment angle + amplitude scale — both produce
ship-speed-proportional errors, invisible on station, dominant at full steam), and
smoothed navigation.

So: **raw VmDAS for routine station processing; CODAS for transects, sections, or any
cruise with long steaming legs inside cast windows.** The full recipe — install
script, the chronological-staging trap, the single-ping (ENR) route, the calibration
step `--auto` does *not* apply — lives in
[`docs/SADCP_CODAS.md`](https://github.com/leopenausa/pyladcp/blob/main/docs/SADCP_CODAS.md)
and the printable one-pager
[`guide/sadcp_codas_guide.pdf`](https://github.com/leopenausa/pyladcp/blob/main/guide/sadcp_codas_guide.pdf).

Once processed, the CODAS product plugs into everything above:

```bash
ladcp-qa 80 ... --sadcp <workdir>/os150nb_sta/contour/os150nb.nc --sadcp-source codas
```

`--sadcp` accepts the `contour/<sonar>.nc` file, its `contour/` dir, or the processing
dir; exports record the provenance as `sadcp_source: codas:<path>`.

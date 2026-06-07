# CTD ingest for LADCP — design + build result (#7)

**Status:** BUILT + validated. 2026-06-07. Recipe locked + byte-verified.

## Build result (2026-06-07)

- **CTD_project** (`ctd_pipeline`, branch `ladcp-export`): `convert_for_ladcp(hex,
  xmlcon, out)` = datcnv (`step01`) → derive ITS-90 `t090C` + PSS-78 `sal00`
  (inside the converter) → **SBE Wild Edit** → single-segment 1 s time bin →
  headerless 6-col extract. Plus `ladcp_compare` harness and unit tests.
- **Validation:** byte-for-byte vs the operator's clean `.cnv` on **all 23** MORIA
  stations with raw + reference — prDM ≤0.002 dbar, T/S ≤0.001, lat/lon/timeS ≈0.
- **Wild-edit:** the SBE two-pass algorithm, implemented to spec — pass 1 only
  *temporarily* flags to clean the std estimate; the final bad-flags come from
  pass 2 (σ=20) against the recomputed std. This is gentle (matches the operator's
  1–4 dropped scans). `wild_edit=True` is the default and is correct; no off/on
  tradeoff. **Finding:** CTD_project's `steps/step03_wildedit.py` does *not* follow
  the spec (marks pass-1 σ=2 flags permanently + quadratic detrend) → over-removes
  valid data in the main pipeline. Flagged for a separate CTD_project decision.
- **LADCP** (`pyladcp`, branch `ctd-from-hex`): `io/ctd_raw.cnv_from_hex()` (optional
  CTD_project dep, located via `LADCP_CTD_PROJECT` or sibling dir; caches to a
  reuse folder), wired into `discover(from_hex=, ctd_cache=)` and `ladcp-qa
  --from-hex/--ctd-cache` (default **off**). End-to-end confirmed on MORIA-79/80/82
  — the golden stations that had no operator CTD now get it from their `.hex` anchor.

## Goal

Let an LADCP run get its CTD nav + 1 s time-series (`lat lon pressure time[s] T S`)
from **either** of two sources, with no change to the downstream consumer:

- **Path A — raw `.hex`** (this plan): operator has the Seabird raw cast files
  (`.hex` + `.XMLCON`). We run the locked recipe to produce the 6-col `.cnv`.
- **Path B — pre-processed `.cnv`** (already works): operator inherited a cleaned
  `.cnv` and has no raw files. We read it directly.

Both must coexist; the choice is per-station, not global (a cruise can have raw
for some casts and only `.cnv` for others).

## What already exists (reused, not rebuilt)

- `io/ctd_cnv.read_ctd_cnv()` already reads **both** cnv shapes: a headerless
  6-col clean file (via `CastParams` field numbers) **and** a fully-headered
  Seabird `.cnv` (auto-mapped through `_SBE_ROLES`). → **Path B is done at the IO
  layer.** Nothing to add there.
- `io/ctd_hex.read_hex_header()` / `scan_ctd_dir()` — header anchor (station, UTC,
  GPS). Used by the archive index. Path A reuses these to locate `.hex` + station.
- `discovery.discover()` already resolves the cleaned `.cnv` per station
  (`_find_clean_ctd` glob from `ctd_dir`). Path B plugs into this unchanged.
- CTD_project `steps/`: `step01_convert.convert(hex, xmlcon, out)`,
  `step03_wildedit.run(cnv_in, out, std_pass1=2.0, std_pass2=20.0, …)`,
  `step09_binavg.run(in, out, bin_by="seconds"/"time", bin_size=1.0)`.

## Path A — the converter (lives in CTD_project)

New public function in CTD_project, imported by LADCP as a library
(env `ladcp_pipeline` already has `gsw`; CTD_project is on `sys.path` or pip-installed).

```python
# ctd_pipeline/convert_for_ladcp.py   (CTD_project, branch separately)
def convert_for_ladcp(
    hex_path: str,
    xmlcon_path: str,
    out_path: str,
    *,
    wild_edit: bool = True,          # locked recipe = on; near-no-op at 1 Hz
    std_pass1: float = 2.0,
    std_pass2: float = 20.0,
    npoint: int = 100,
    bin_seconds: float = 1.0,
) -> str:
    """Raw Seabird .hex (+ XMLCON) -> the 6-col LADCP CTD .cnv.

    Composes datcnv -> wildedit -> binavg(time,1s) -> 6-col extract:
        latitude longitude prDM timeS t090C sal00   (ITS-90 T, PSS-78 S)
    Headerless, formats %+.5f %+.5f %+.4f %+.4f %+.3f %+.3f.
    Returns out_path.
    """
```

Composition + the three deltas vs the existing 9-step pipeline:

1. **datcnv** — `step01.convert`. **Delta:** must emit `t090C` (ITS-90) and
   `sal00` (PSS-78). The pipeline carries ITS-68 internally and derives salinity
   at step08. **DECIDED (2026-06-07):** compute both inside `convert_for_ladcp`
   (`t090 = t68/1.00024`, `gsw`/PSS-78 salinity) — `step01.convert` stays
   untouched, no change to the shared 9-step pipeline.
2. **wildedit** — `step03.run` with `std_pass1=2.0, std_pass2=20.0, npoint=100`.
   Reused verbatim (CTD_project CLAUDE.md already documents 2-then-20, 100/block).
3. **binavg** — `step09.run` with `bin_by="seconds"` (or `"time"`), `bin_size=1.0`.
   Frequency-adaptive; required (identity at 1 Hz MORIA, real at 24 Hz).
4. **extract/format** — new: select the 6 columns in order, headerless, the fixed
   format string. This is the only genuinely new I/O.

Skips: cell-thermal-mass (04), filter (05), align (06), loop-edit (07) — the
recipe is exactly datcnv + wildedit + binavg + extract.

## Path A — the LADCP side

```python
# src/ladcp/io/ctd_raw.py   (LADCP)
def cnv_from_hex(hex_path, xmlcon_path, out_dir) -> Path:
    """Thin wrapper: import convert_for_ladcp, write <station>_clean.cnv to out_dir,
    return the path. Then read_ctd_cnv() consumes it (Path B reader, unchanged)."""
```

Discovery integration (`discovery.discover`) — resolve CTD per station in this
order, so both paths coexist and B wins when a clean file already exists:

1. **Existing clean `.cnv`** in `ctd_dir` (current `_find_clean_ctd`) → Path B.
2. Else, if a `.hex` + `.XMLCON` exist for the station (located via the archive
   index `.hex` anchor) and `--from-hex` is allowed → convert (Path A), cache the
   `.cnv` next to the index, then read it.
3. Else → CTD `None` (run proceeds without CTD nav, as today).

CLI: `ladcp-qa --from-hex` opts into Path A. **DECIDED (2026-06-07): default off**
= current behaviour preserved, use whatever clean `.cnv` is present; conversion
only runs when `--from-hex` is explicitly set. `--ctd-dir` already names the
pre-processed folder for Path B.

## Validation harness (build BEFORE wiring)

`ctd-compare` — proves the converter reproduces the operator's Seabird output
before any of the discovery wiring is trusted:

- For stations 79 / 80 / 82: run `convert_for_ladcp(hex, xmlcon)` → 6-col `.cnv`,
  compare column-by-column against the Seabird clean `.cnv`
  (`raw_ctd/moria-*-ctd_cnv_we_1s.cnv`, 6-col extract).
- Pass target: lat/lon/P/time Δ = 0; T/S within ~0.001 (rounding only), matching
  the MORIA-10 byte-level proof already in hand.
- Lives in CTD_project (it tests CTD_project code); LADCP gets a smaller
  integration test that `cnv_from_hex` → `read_ctd_cnv` yields a sane
  `CTDTimeSeries`.

## Two-repo hygiene

- CTD_project: own git (`main`, commit+push there), env `ctd_pipeline`, its own
  CLAUDE.md/SOP rules — branch separately for the converter + `ctd-compare`.
- LADCP: `io/ctd_raw.py` + discovery/CLI wiring + integration test on its own branch.
- Build order: (1) CTD_project `convert_for_ladcp` + `ctd-compare`, validate 79/80/82;
  (2) only then LADCP `ctd_raw.py` + `--from-hex`.

## Decisions (2026-06-07)

1. **ITS-90 / sal00** — computed inside `convert_for_ladcp`; `step01.convert`
   untouched. ✅
2. **`--from-hex` default** — off (explicit opt-in); Path B is the default. ✅

## Open questions (still to settle, non-blocking — sensible defaults assumed)

3. **Where the converted `.cnv` is cached** — beside `.ladcp_archive.json`, or a
   dedicated `clean_ctd/` out dir? Draft assumes a cache dir keyed by station.
4. **CTD_project import mechanism** — pip-install (editable) into `ladcp_pipeline`,
   or `sys.path` shim? Affects how `convert_for_ladcp` is imported.

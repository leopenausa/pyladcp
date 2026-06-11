# Appendix B · LDEO_IX ↔ pyladcp mapping

For readers migrating from the legacy MATLAB software. This is the **core mapping**
(the knobs and outputs you actually touch), not an exhaustive parameter audit.

## Solver & constraint parameters

| LDEO_IX | pyladcp | notes |
|---|---|---|
| `ps.shear = 0` (inverse) | `--solver inverse` | the default in both |
| `ps.shear = 1` (shear) | `--solver shear` | |
| `ps.botfac` | `--botfac` | default 1 |
| `ps.barofac` | `--barofac` | default 1 |
| `ps.sadcpfac` | `--sadcpfac` | default 3 |
| `ps.smoofac` | `--smoofac` | default 0; at 0 only ill-constrained bins are smoothed (same branch as legacy) |
| `ps.outlier` | (fixed at the legacy two-pass behaviour) | worst 1% rejected between passes |
| `ps.dz` | cruise preset (`dz`, default 8 m) | output grid spacing |
| `p.btrk_mode = 3` (RDI firmware BT) | the default | own water-track BT used only when no firmware track exists |
| `p.drot` | `--drot` | **differs by default**: legacy needed a hand-entered value; pyladcp computes IGRF-13 from the cast position and records the provenance on the scorecard |
| `p.tiltmax` | cruise preset (22°/4° smoothed) | same editing semantics |
| `p.pglim` | cruise preset (50) | percent-good threshold |
| `p.vlim` | cruise preset (2.5 m/s) | speed screen |
| `p.sadcp` | `--sadcp <path>` | plus `--sadcp-source codas` for CODAS products |

## Outputs

| LDEO_IX | pyladcp | notes |
|---|---|---|
| `dr.z`, `dr.u`, `dr.v`, `dr.uerr` | `<st>.lad` / `<st>.nc` (`z u v uerr`) | same quantities |
| `dr.ubar`, `dr.vbar` | `.lad` header + NetCDF attrs | barotropic reference |
| `dr.zbot`, `dr.ubot`, `dr.vbot` | `<st>.bot` | bottom-track-referenced profile |
| `dr.u_shear_method` etc. | `--solver shear` run | |
| `.lad` text file | `.lad` — **same format** (`Columns = z:u:v:ev`) | compatible with downstream readers |
| `.bot` text file | `.bot` — same format (`z:u:v:err` + `Bottom depth`) | |
| processing log / diary | `qa_out/ladcp-qa.log` + `<st>_qa.txt/.json` | the scorecard has no legacy equivalent |
| Figures 1–14 | report PDF pages | cross-reference list in [chapter 6](06-qa-report.md) |

## Behavioural differences to know about

These are deliberate, validated departures — each was found to be a legacy
limitation during cross-validation:

1. **Magnetic declination**: pyladcp uses IGRF-13 from the cast position/date by
   default; legacy results processed years apart can carry stale hand-entered
   declinations. Expect small direction differences against old products for this
   reason alone.
2. **Bottom track**: the RDI firmware track is preferred when present (legacy
   `btrk_mode=3`); the own water-track BT inherits boundary-layer flow and can bias
   strong-drift casts.
3. **In-water window**: the barotropic reference uses the deep segment containing
   the deepest ping — pre-cast surface soaks no longer leak ship drift into the
   reference.
4. **Compass offset**: the dual-head alignment is cross-checked against the
   in-water-only estimate (deck pings can corrupt the full-record value).
5. **Package depth**: integrated past the end of a prematurely-stopped CTD record
   using the ADCP's vertical velocity.

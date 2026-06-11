# Appendix E · Glossary

**ADCP** — Acoustic Doppler Current Profiler: measures water velocity in range bins
along its acoustic beams via the Doppler shift of backscattered sound.

**LADCP** — *Lowered* ADCP: one or two ADCPs on the CTD rosette, profiling the full
water column during the cast.

**Down-looker / up-looker (master / slave)** — the two heads of a dual-head LADCP.
"Master/slave" is the synchronization wiring convention; pyladcp pairs them by time
regardless.

**PD0** — RDI's binary file format for ADCP data (`*.000` deployment files).

**Beam vs earth coordinates** — raw radial velocities along the four beams vs
rotated east/north/up components. pyladcp rotates beam-coordinate data to earth at
ingest.

**Percent-good (pg)** — RDI's per-cell quality fraction; cells under the threshold
(default 50) are removed.

**Error velocity** — the redundant fourth-beam combination that should be ≈0 in
homogeneous flow; large values flag bad cells, and its near/far-field ratio flags
hung devices ([chapter 6](06-qa-report.md)).

**Side-lobe contamination** — near the seabed, energy from the beams' side lobes
hits the bottom before the main lobe finishes the bin: a contaminated wedge that is
edited away (the white wedge in the edit figure).

**Bottom track (BT)** — the instrument's velocity over ground measured from the
seabed echo: the strongest absolute reference, available within ~range of the
bottom. *RDI firmware BT* = measured by the instrument itself; *own BT* = recomputed
from near-seabed water cells.

**Water track** — the ordinary water-velocity pings (everything that is not bottom
track).

**Super-ensemble** — a packet of consecutive pings averaged together at the
package's current depth; the unit of data entering the solvers.

**Baroclinic / barotropic** — the depth-varying *shape* of the velocity profile vs
its depth-mean. The shear carries the baroclinic part; the constraints (BT, GPS,
SADCP) pin the barotropic part.

**Shear method** — solve by integrating the bin-averaged vertical shear
(package motion cancels in the shear); needs a separate barotropic reference.

**Inverse method** — solve ocean profile and package-velocity time series together
in one constrained least-squares system (Visbeck 2002); pyladcp's default.

**Constraint weights** (`botfac`, `barofac`, `sadcpfac`, `smoofac`) — how strongly
each information source pulls the inverse relative to the water-track data
([chapter 7](07-solvers-weights.md)).

**Magnetic declination (`drot`)** — the angle between magnetic and true north at
the cast position; the instrument's compass is magnetic, so profiles are rotated by
this angle (IGRF-13 model, automatic).

**IGRF** — International Geomagnetic Reference Field: the standard model from which
the declination is computed.

**SADCP** — *Shipboard* ADCP: the vessel's hull-mounted ADCP, running continuously
(VmDAS or UHDAS acquisition).

**STA / LTA** — VmDAS short/long-term averages of the shipboard ADCP, formed
onboard from unedited pings.

**CODAS** — the UH/SOEST processing system for shipboard-ADCP data (editing,
watertrack calibration, navigation smoothing); see [chapter 8](08-ship-adcp.md).

**`.lad` / `.bot`** — the LDEO text formats for the velocity profile and the
bottom-track-referenced profile ([appendix C](appendix-c-formats.md)).

**Scorecard** — pyladcp's per-cast QA table; every metric is a traffic light
(OK/WARN/FAIL) and the strictest one is the cast's verdict
([chapter 6](06-qa-report.md)).

**Golden** — a legacy-LDEO_IX result used as a validation target during pyladcp's
development.

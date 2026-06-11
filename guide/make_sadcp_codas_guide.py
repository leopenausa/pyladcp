"""Generate the one-page SADCP/CODAS guide (PDF) from docs/SADCP_CODAS.md content.

Same pure-matplotlib machinery as make_cli_guide.py (Sheet), different header.

Run:  MPLBACKEND=Agg python guide/make_sadcp_codas_guide.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cli_guide import ACCENT, INK, MUTE, OK_BG, OK_INK, RULE, Sheet  # noqa: E402


class CodasSheet(Sheet):
    def _header(self, subtitle):
        title = self.ax.text(0.060, 0.960, "pyladcp", transform=self.fig.transFigure,
                             fontsize=20, weight="bold", color=ACCENT, va="top",
                             ha="left", parse_math=False)
        self.fig.canvas.draw()
        x_right = self.fig.transFigure.inverted().transform(
            title.get_window_extent(self.fig.canvas.get_renderer()))[1][0]
        self._t(x_right + 0.018, 0.953, "·  SADCP processing with CODAS", size=12.5,
                color=INK)
        self._t(0.060, 0.934, subtitle, size=8.6, color=MUTE)
        self.ax.plot([0.060, 0.940], [0.917, 0.917], transform=self.fig.transFigure,
                     color=RULE, lw=1.0)

    def footer(self, page_no):
        self.ax.plot([0.060, 0.940], [0.038, 0.038], transform=self.fig.transFigure,
                     color=RULE, lw=0.8)
        self._t(0.060, 0.032,
                "pyladcp · docs/SADCP_CODAS.md · currents.soest.hawaii.edu (CODAS docs)",
                size=7.0, color=MUTE)
        self._t(0.905, 0.032, f"page {page_no} / 1", size=7.0, color=MUTE)


def _page(pdf):
    s = CodasSheet("raw VmDAS → edited + calibrated shipboard-ADCP product → "
                   "pyladcp (--sadcp-source codas)  —  one-page recipe")

    # ----------------------------- LEFT COLUMN -----------------------------------
    s.head("L", "Why CODAS")
    s.body("L", "CODAS adds what raw .STA/.LTA averages lack: an editing pass (pflag),")
    s.body("L", "a watertrack calibration (misalignment angle + amplitude scale), and")
    s.body("L", "smoothed GPS navigation. On-station the raw constraint is already fine")
    s.body("L", "(40-cast MORIA A/B: interchangeable to ~1 cm/s) — CODAS matters")
    s.body("L", "UNDERWAY: phase φ biases ≈ sinφ × ship speed; an amplitude error of")
    s.body("L", "a% biases ≈ 10 cm/s per % at 5 m/s. Use it for sections/transects and")
    s.body("L", "as an independent check of the raw chain.")
    s.gap("L")

    s.head("L", "One-time setup  (~15 min)")
    s.code("L", ["bash scripts/codas_install.sh",
                 "#  conda env pycodas + codas3/pycurrents/onship",
                 "#  + demos + etopo (linked into the env)"])
    s.gap("L", 0.006)

    s.head("L", "Process a cruise  (MORIA OS150, ~5 min)")
    s.code("L", ["bash scripts/codas_moria_os150.sh",
                 "# other instruments: env overrides, e.g. OS75:",
                 "SADCP_DATA=.../sadcp_75/DATA SADCP_SONAR=os75nb \\",
                 "SADCP_PROC=os75nb_sta SADCP_DB=aMORIA75 \\",
                 "  bash scripts/codas_moria_os150.sh"])
    s.body("L", "The script: (1) adcptree.py makes the tree; (2) stages NNN_-prefixed")
    s.body("L", "CHRONOLOGICAL copies of the .STA files; (3) writes the control file;")
    s.body("L", "(4) runs quick_adcp.py --auto.")
    s.gap("L", 0.006)
    s.callout("L", ["Never feed name-ordered files: parts of CODAS glob by FILENAME.",
                    "On MORIA that misaligned every position by 682 ensembles and",
                    "silently poisoned all absolute velocities — while the watertrack",
                    "cal still looked fine. Always re-check the product's positions",
                    "against a known cast before using it."])
    s.gap("L", 0.004)

    s.head("L", "Single-ping route  (use for sections / underway)")
    s.code("L", ["bash scripts/codas_moria_os150_enr.sh",
                 "# OS75 (serial feeds swapped): ENR_GYRO_NR=2 + ENR_* overrides"])
    s.body("L", "STA averages are formed onboard from raw UNEDITED pings: underway,")
    s.body("L", "bins beyond the bubble/noise-limited range are garbage that keeps")
    s.body("L", "pg≥50 and survives average-level editing — MORIA deep (>300 m)")
    s.body("L", "underway rms u was 0.68 m/s (STA) vs 0.13 m/s after per-ping editing")
    s.body("L", "(on station both routes are clean). This route runs the full UHDAS")
    s.body("L", "chain on .ENR + N1R/N2R: per-ping editing BEFORE averaging,")
    s.body("L", "beam→earth with gyro + attitude correction, working bottom-track")
    s.body("L", "cal; it also lowered the apparent amplitude error +2.1% → +0.9%.")
    s.body("L", "Its workarounds (deployment-grouped staging, ping-number nav")
    s.body("L", "matching, topog link) are already encoded in the script.")

    # ----------------------------- RIGHT COLUMN ----------------------------------
    s.head("R", "Control-file parameters that matter")
    s.kv("R", [
        ("--yearbase", "year of first data; nc time = days since Jan 1"),
        ("--datatype sta", "release CODAS: sta/lta/uhdas/pingdata only"),
        ("--sonar os150nb", "instrument + ping type (nb = narrowband)"),
        ("--ens_len 120", "onboard STA interval [s] — read the .VMO"),
        ("--max_search_depth", "seabed search for editing; ≥ shelf depth"),
    ], dx=0.165)
    s.gap("R", 0.008)

    s.head("R", "Calibrate — the step --auto does NOT do")
    s.code("R", ["cd <workdir>/os150nb_sta",
                 "tail -20 cal/watertrk/adcpcal.out   # medians",
                 "quick_adcp.py --cntfile q_py.cnt --auto \\",
                 "  --steps2rerun rotate:navsteps:calib:matfiles:netcdf \\",
                 "  --rotate_angle 0.003 --rotate_amplitude 1.021"])
    s.body("R", "phase = transducer misalignment [deg]; amplitude = scale factor.")
    s.body("R", "Both errors grow with ship speed (invisible on station). Trust a cal")
    s.body("R", "only with a healthy point count, and verify the product afterwards.")
    s.gap("R", 0.006)
    s.callout("R", ["Deliverable: contour/<sonar>.nc — absolute ocean u/v per",
                    "ensemble (pflag editing applied), positions, uship/vship.",
                    "MORIA cals: OS150 STA 1.021/0.003° · ENR 1.009/−0.09° ·",
                    "OS75 1.010/−0.22°."], bg=OK_BG, ink=OK_INK)
    s.gap("R", 0.004)

    s.head("R", "Use it in pyladcp  (vmdas stays the default)")
    s.code("R", ["ladcp-qa 80 --index .ladcp_archive.json \\",
                 "  --sadcp <workdir>/os150nb_sta/contour/os150nb.nc \\",
                 "  --sadcp-source codas"])
    s.body("R", "--sadcp takes the .nc, its contour/ dir, or the processing dir;")
    s.body("R", "exports record provenance as  sadcp_source: codas:<path>.")
    s.body("R", "Keep vmdas for routine station work; pick codas for cruises with")
    s.body("R", "long steaming legs in cast windows or to use the edited series.")
    s.gap("R", 0.008)

    s.head("R", "Section plots")
    s.code("R", ["# raw VmDAS (GPS-leak screen --speed-max on by default)",
                 "ladcp-sadcp-section --sadcp .../sadcp_150/DATA \\",
                 "  --by time -o sec_time.png",
                 "# CODAS product (already edited; cleaner underway)",
                 "ladcp-sadcp-section \\",
                 "  --sadcp <workdir>/os150nb_sta/contour/os150nb.nc \\",
                 "  --source codas --by distance \\",
                 "  --index .ladcp_archive.json -o sec_dist.png"])
    s.body("R", "--by time|distance picks the x-axis; --index draws LADCP station")
    s.body("R", "ticks; --max-depth/--clim crop and scale. --speed-max (median-speed")
    s.body("R", "blanking, 1.5 m/s) runs on either source: built for raw GPS leaks,")
    s.body("R", "also drops residual bad CODAS ensembles; 0 disables. --anomaly")
    s.body("R", "subtracts each ensemble's depth-mean → the baroclinic/shear view")
    s.body("R", "(barotropic signal removed, so weak structure shows at tight --clim).")

    s.footer(1)
    s.save(pdf)


def main(out: str = "guide/sadcp_codas_guide.pdf") -> str:
    with PdfPages(out) as pdf:
        _page(pdf)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()

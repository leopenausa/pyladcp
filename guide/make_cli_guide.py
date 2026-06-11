"""Generate the two-page `ladcp-qa` CLI guide (PDF).

Pure matplotlib so it needs nothing beyond the project env. Letter portrait, two columns
per page. Page 1 is the exact run recipe (copy-paste, with the cruise-root step up front);
page 2 is the flag/output reference + gotchas.

Run:  MPLBACKEND=Agg python guide/make_cli_guide.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# palette
INK = "#1a2730"
MUTE = "#5b6b75"
ACCENT = "#1f6f8b"
CODE_BG = "#eef2f4"
CODE_INK = "#0f3a47"
RULE = "#cdd6db"
WARN_BG = "#fbf0d8"
WARN_INK = "#8a5a00"
OK_BG = "#e6f2ea"
OK_INK = "#1d6b3f"

PAGE = (8.5, 11.0)
LH = 0.0150          # body line height (figure fraction)
COL = {"L": 0.060, "R": 0.535}     # left x of each column
COLW = 0.405         # column width
TOP = 0.895
BOTTOM = 0.045


class Sheet:
    """One letter page with two flowing columns and code/callout boxes."""

    def __init__(self, subtitle: str):
        self.fig = plt.figure(figsize=PAGE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.cur = {"L": TOP, "R": TOP}
        self._header(subtitle)

    # -- low-level ---------------------------------------------------------------
    def _t(self, x, y, s, *, size=8.0, color=INK, weight="normal",
           family="sans-serif", style="normal", mono=False):
        self.ax.text(x, y, s, transform=self.fig.transFigure, fontsize=size, color=color,
                     weight=weight, family=("monospace" if mono else family), style=style,
                     va="top", ha="left", parse_math=False)

    def _header(self, subtitle):
        title = self.ax.text(0.060, 0.960, "pyladcp", transform=self.fig.transFigure,
                             fontsize=20, weight="bold", color=ACCENT, va="top", ha="left",
                             parse_math=False)
        self.fig.canvas.draw()
        x_right = self.fig.transFigure.inverted().transform(
            title.get_window_extent(self.fig.canvas.get_renderer()))[1][0]
        self._t(x_right + 0.018, 0.953, "·  ladcp-qa  CLI guide", size=12.5, color=INK)
        self._t(0.060, 0.934, subtitle, size=8.6, color=MUTE)
        self.ax.plot([0.060, 0.940], [0.917, 0.917], transform=self.fig.transFigure,
                     color=RULE, lw=1.0)

    # -- flow primitives ---------------------------------------------------------
    def head(self, col, s):
        y = self.cur[col]
        self.ax.text(COL[col], y, s.upper(), transform=self.fig.transFigure, fontsize=9.4,
                     weight="bold", color=ACCENT, va="top", ha="left", parse_math=False)
        self.ax.plot([COL[col], COL[col] + COLW], [y - 0.011, y - 0.011],
                     transform=self.fig.transFigure, color=RULE, lw=0.8)
        self.cur[col] = y - 0.026

    def body(self, col, s, *, color=INK, size=7.9, dx=0.0):
        self._t(COL[col] + dx, self.cur[col], s, size=size, color=color)
        self.cur[col] -= LH

    def code(self, col, lines):
        pad = 0.006
        h = len(lines) * LH + 2 * pad
        y0 = self.cur[col]
        box = FancyBboxPatch((COL[col], y0 - h + pad), COLW, h - pad,
                             boxstyle="round,pad=0.004,rounding_size=0.006",
                             transform=self.fig.transFigure, facecolor=CODE_BG,
                             edgecolor="none", zorder=0)
        self.ax.add_patch(box)
        yy = y0 - pad - 0.002
        for ln in lines:
            self._t(COL[col] + 0.012, yy, ln, size=7.2, color=CODE_INK, mono=True)
            yy -= LH
        self.cur[col] = y0 - h - 0.006

    def callout(self, col, lines, *, bg=WARN_BG, ink=WARN_INK):
        pad = 0.006
        h = len(lines) * LH + 2 * pad
        y0 = self.cur[col]
        box = FancyBboxPatch((COL[col], y0 - h + pad), COLW, h - pad,
                             boxstyle="round,pad=0.004,rounding_size=0.006",
                             transform=self.fig.transFigure, facecolor=bg,
                             edgecolor="none", zorder=0)
        self.ax.add_patch(box)
        yy = y0 - pad - 0.002
        for ln in lines:
            self._t(COL[col] + 0.012, yy, ln, size=7.4, color=ink)
            yy -= LH
        self.cur[col] = y0 - h - 0.006

    def kv(self, col, rows, *, dx=0.150, key_size=7.5):
        for k, v in rows:
            self._t(COL[col], self.cur[col], k, size=key_size, color=CODE_INK, mono=True)
            self._t(COL[col] + dx, self.cur[col], v, size=7.5, color=INK)
            self.cur[col] -= LH

    def gap(self, col, g=0.012):
        self.cur[col] -= g

    def footer(self, page_no):
        self.ax.plot([0.060, 0.940], [0.038, 0.038], transform=self.fig.transFigure,
                     color=RULE, lw=0.8)
        self._t(0.060, 0.032,
                "pyladcp · github.com/leopenausa/pyladcp · ladcp-qa --help · ladcp-index --help",
                size=7.0, color=MUTE)
        self._t(0.905, 0.032, f"page {page_no} / 2", size=7.0, color=MUTE)

    def save(self, pdf):
        pdf.savefig(self.fig)
        plt.close(self.fig)


# ================================ PAGE 1 — RECIPE ================================
def _page1(pdf):
    s = Sheet("Process a whole cruise, step by step. Run the steps in order, in one terminal.")

    # ----- LEFT: set the root + folder layout -----
    s.head("L", "Step 0 · set the cruise root  (first!)")
    s.body("L", "Every command uses  $B  = the cruise folder that", color=MUTE, size=7.6)
    s.body("L", "holds raw_ladcp/, raw_CTD/ and sADCP/. Set it once:", color=MUTE, size=7.6)
    s.gap("L", 0.004)
    s.code("L", [
        "cd /path/to/cruise/MORIA   # folder with raw_*",
        'B="$PWD"                    # = where you are now',
        'echo "$B"                   # MUST print that path',
    ])
    s.callout("L", [
        "If  echo \"$B\"  prints nothing, STOP and set it",
        "again. An empty $B turns --ladcp \"$B/raw_ladcp\"",
        "into \"/raw_ladcp\"  →  \"indexed 0 casts from 0",
        "files\". A new terminal forgets $B — re-set it.",
    ])
    s.gap("L", 0.006)

    s.head("L", "Your folder must look like this")
    s.code("L", [
        "$B/",
        "├─ raw_ladcp/",
        "│   ├─ MASTER/  *.000   down-looker",
        "│   └─ SLAVE/   *.000   up-looker",
        "├─ raw_CTD/   *.hex *.hdr *.bl *.XMLCON",
        "│             (one set per station)",
        "└─ sADCP/     ship-ADCP (optional)",
        "    └─ sadcp_150/DATA/  *.STA",
    ])
    s.body("L", "Raw LADCP files carry NO station number — the", color=MUTE, size=7.5)
    s.body("L", "CTD .hex (station + GPS UTC) is the anchor that", color=MUTE, size=7.5)
    s.body("L", "tells the index which raw file is which cast.", color=MUTE, size=7.5)
    s.gap("L", 0.004)
    s.body("L", "raw_ladcp/ may instead be a FLAT tree (no", color=MUTE, size=7.5)
    s.body("L", "MASTER/SLAVE) — heads split by the PD0 facing", color=MUTE, size=7.5)
    s.body("L", "bit. Beam-coord PD0s are rotated to earth on read.", color=MUTE, size=7.5)
    s.gap("L", 0.012)

    s.head("L", "New / non-MORIA cruise")
    s.body("L", "--cruise <NAME>: any name works. An unknown", color=MUTE, size=7.5)
    s.body("L", "cruise gets shared operator defaults (dz 8, bin-1", color=MUTE, size=7.5)
    s.body("L", "mask, tilt 22/4, RDI-firmware bottom track),", color=MUTE, size=7.5)
    s.body("L", "stamped <NAME> on the exports.", color=MUTE, size=7.5)
    s.gap("L", 0.006)
    s.body("L", "SADCP clock not synced to GPS (wrong dates in", color=MUTE, size=7.5)
    s.body("L", "the STA files)? Recover the offset automatically:", color=MUTE, size=7.5)
    s.gap("L", 0.003)
    s.code("L", [
        "ladcp-qa ... --sadcp-timeoff auto \\",
        '    --sadcp-nav "$B/Navigation"',
    ])
    s.body("L", "slides the STA GPS against an independent nav", color=MUTE, size=7.5)
    s.body("L", "track (SADO export or any time/lat/lon CSV).", color=MUTE, size=7.5)
    s.body("L", "Shallow shelf cast leaking a bad near-bottom", color=MUTE, size=7.5)
    s.body("L", "cell? raise the reject margin:  --dzbelow 24.", color=MUTE, size=7.5)

    # ----- RIGHT: the three run steps -----
    s.head("R", "Step 1 · build the index  (once)")
    s.body("R", "Note:  --root  comes BEFORE the word  build.", color=WARN_INK, size=7.6)
    s.gap("R", 0.004)
    s.code("R", [
        'ladcp-index --root "$B" build \\',
        '    --ladcp "$B/raw_ladcp" \\',
        '    --ctd   "$B/raw_CTD"',
    ])
    s.body("R", "Expect  “indexed ~40 casts from ~100 files”.", color=MUTE, size=7.5)
    s.body("R", "If it says 0 casts / 0 files → $B is unset (Step 0).", color=MUTE, size=7.5)
    s.gap("R", 0.008)

    s.head("R", "Step 2 · check the pairing")
    s.code("R", ['ladcp-index --root "$B" show'])
    s.body("R", "Lists each station with its master + slave file.", color=MUTE, size=7.5)
    s.body("R", "Confirm every cast got both before the long run.", color=MUTE, size=7.5)
    s.gap("R", 0.008)

    s.head("R", "Step 3 · process the whole cruise")
    s.body("R", "The full constrained inverse is the default solver.", color=MUTE, size=7.5)
    s.gap("R", 0.004)
    s.code("R", [
        "ladcp-qa --all-stations \\",
        '    --index "$B/.ladcp_archive.json" \\',
        '    --root "$B" --cruise <NAME> \\',
        "    --from-hex --ctd-cache \"$B/ctd_from_hex\" \\",
        '    --sadcp "$B/sADCP/sadcp_150/DATA" \\',
        '    -o "$B/qa_out"',
    ])
    s.callout("R", [
        "A live progress bar shows X/N stations. Full",
        "detail + any failures → $B/qa_out/ladcp-qa.log.",
        "One bad cast is logged and skipped, not fatal.",
    ], bg=OK_BG, ink=OK_INK)
    s.gap("R", 0.006)
    s.body("R", "Then open  qa_out/stations/<st>/<st>_report.pdf", size=7.7)
    s.body("R", "→ scorecard (OK / WARN / FAIL) + the figures.", size=7.7)
    s.gap("R", 0.014)

    s.head("R", "Step 4 · compare vs a legacy reference")
    s.body("R", "Have LDEO *.mat from a prior processing? Pair", color=MUTE, size=7.5)
    s.body("R", "by cast time (no filename assumptions) and score:", color=MUTE, size=7.5)
    s.gap("R", 0.003)
    s.code("R", [
        'ladcp-compare --ours "$B/qa_out" \\',
        '    --legacy "$B/LADCP_processed" \\',
        '    -o "$B/qa_out/legacy_compare"',
    ])
    s.body("R", "→ comparison.csv + comparison_report.pdf (u/v", color=MUTE, size=7.5)
    s.body("R", "overlays with ±1σ bands). unpaired.txt lists any", color=MUTE, size=7.5)
    s.body("R", "station matched on only one side — never dropped.", color=MUTE, size=7.5)

    s.footer(1)
    s.save(pdf)


# ============================== PAGE 2 — REFERENCE ==============================
def _page2(pdf):
    s = Sheet("Reference — what you get, every flag, and the things that trip people up.")

    # ----- LEFT: outputs + recipes -----
    s.head("L", "What you get  (under  $B/qa_out/)")
    s.code("L", [
        "qa_out/",
        "├─ stations/<MORIA-xx>/",
        "│   ├─ <st>_report.pdf    ★ scorecard+figs",
        "│   ├─ <st>.lad  <st>.bot velocity / b-track",
        "│   ├─ <st>.nc   <st>.xlsx NetCDF / Excel",
        "│   ├─ <st>_qa.txt/.json  QA metrics",
        "│   └─ figures/*.png",
        "├─ exports/               cruise-level",
        "│   ├─ MORIA_ladcp.xlsx / .nc",
        "│   ├─ MORIA_ladcp_odv.txt  (Ocean Data View)",
        "│   └─ MORIA_summary.csv",
        "└─ ladcp-qa.log           run log",
    ])
    s.body("L", "exports/ is written by --all-stations (or add", color=MUTE, size=7.5)
    s.body("L", "--cruise-export to a named list of stations).", color=MUTE, size=7.5)
    s.gap("L", 0.010)

    s.head("L", "Common variations")
    s.body("L", "One station, full solve + ship-ADCP:", color=MUTE, size=7.5)
    s.gap("L", 0.003)
    s.code("L", [
        'ladcp-qa 80 --index "$B/.ladcp_archive.json" \\',
        '   --root "$B" --from-hex \\',
        '   --sadcp "$B/sADCP/sadcp_150/DATA" \\',
        '   -o "$B/qa_out"',
    ])
    s.body("L", "A few stations + a cruise workbook:", color=MUTE, size=7.5)
    s.gap("L", 0.003)
    s.code("L", ["ladcp-qa 79 80 82 ...  --cruise-export"])
    s.body("L", "No Excel? skip it (or install the extra):", color=MUTE, size=7.5)
    s.gap("L", 0.003)
    s.code("L", [
        "ladcp-qa ... --formats odv,nc,csv",
        "pip install pyladcp[export]   # enables .xlsx",
    ])
    s.body("L", "Stream detail instead of the bar:  add  -v", color=MUTE, size=7.5)
    s.gap("L", 0.010)

    s.head("L", "Constraint weights — physical meaning")
    s.body("L", "The inverse blends four information sources. Each", color=MUTE, size=7.5)
    s.body("L", "weight scales how strongly that source pulls the", color=MUTE, size=7.5)
    s.body("L", "solution relative to the LADCP water-track data:", color=MUTE, size=7.5)
    s.body("L", "1 = legacy-standard balance · 0 = off · >1 = trust more.", color=MUTE,
           size=7.5)
    s.gap("L", 0.004)
    s.kv("L", [
        ("--botfac W", "bottom track (default 1). Velocity over"),
        ("", "ground from the seabed echo: anchors the"),
        ("", "absolute (barotropic) level near the bottom."),
        ("--sadcpfac W", "ship ADCP (default 3). The hull ADCP's"),
        ("", "absolute currents: pins the upper-ocean bins."),
        ("--barofac W", "GPS navigation (default 1). Ship drift"),
        ("", "over the cast: one row pinning the depth-"),
        ("", "mean (barotropic) velocity."),
        ("--smoofac W", "vertical smoothing (default 0). Curvature"),
        ("", "penalty on the profile; at 0 only ill-"),
        ("", "constrained bins are smoothed (golden)."),
    ], dx=0.118, key_size=7.2)
    s.gap("L", 0.004)
    s.body("L", "Where each constraint acts (and how hard) is drawn", color=MUTE, size=7.5)
    s.body("L", "per cast: figures/<st>_weights.png (legacy Fig 12).", color=MUTE, size=7.5)

    # ----- RIGHT: flags + gotchas -----
    s.head("R", "ladcp-index   (--root BEFORE build/show)")
    s.kv("R", [
        ("--root D", "cruise root paths are relative to"),
        ("build", "scan + cache station→file map"),
        ("  --ladcp D", "archive dir (has MASTER/ + SLAVE/)"),
        ("  --ctd D", "CTD .hex/.hdr dir (the anchor)"),
        ("  --rescan", "force full re-decode (ignore cache)"),
        ("show", "print the resolved casts"),
    ], dx=0.150)
    s.gap("R", 0.010)

    s.head("R", "ladcp-qa   (order-free flags)")
    s.kv("R", [
        ("--all-stations", "every cast in the index + exports/"),
        ("--cruise NAME", "preset; unknown → operator defaults"),
        ("--cruise-export", "exports/ for a named station list"),
        ("--index F", "archive index → files per station"),
        ("--solver S", "inverse (default, full) | shear"),
        ("--sadcp D", "ship-ADCP source (inverse constraint)"),
        ("--sadcp-source X", "vmdas (default) | codas"),
        ("--sadcp-timeoff T", "STA clock fix [s] or 'auto'"),
        ("--sadcp-nav P", "nav track for --sadcp-timeoff auto"),
        ("--botfac W", "bottom-track weight (see page-left)"),
        ("--sadcpfac W", "ship-ADCP weight (default 3)"),
        ("--barofac W", "GPS-barotropic weight (default 1)"),
        ("--smoofac W", "smoothing weight (default 0)"),
        ("--dzbelow M", "near-seabed reject margin [m]"),
        ("--down-only", "solve from the down-looker alone"),
        ("--nearfield-dn-bins", "device-band mask override / 'none'"),
        ("--from-hex", "build CTD from raw .hex if no .cnv"),
        ("--ctd-cache D", "reuse dir for --from-hex"),
        ("--formats L", "xlsx,odv,nc,csv (default: all)"),
        ("-o D", "output folder (default qa_out)"),
        ("-v / --verbose", "stream detail instead of the bar"),
        ("--root D", "base for cleaned CTD/ + fallback find"),
    ], dx=0.170, key_size=7.3)
    s.gap("R", 0.010)

    s.head("R", "Gotchas")
    for ln in [
        "• Set $B first and check echo \"$B\" (Step 0).",
        "• ladcp-index: --root goes BEFORE build / show.",
        "• Excel needs openpyxl (pip install pyladcp[export]);",
        "  else .xlsx is skipped, the rest still write.",
        "• --from-hex needs CTD_project; a clean .cnv wins.",
        "• Heads pair by time, so fragmented / offset",
        "  master-slave deployment files still match.",
        "• battery is an uncalibrated estimate (0.33·xmv):",
        "  it can WARN but never FAILs a cast.",
        "• nearfield_errvel_ratio WARN >1.7 = a rigid target",
        "  hung below the package (e.g. a corer); mask its",
        "  bins per cruise (MORIA does 03–28 automatically).",
        "• --down-only is a cross-check — don't deliver it",
        "  where the down-looker is contaminated.",
        "• Headless server? prefix  MPLBACKEND=Agg.",
    ]:
        s.body("R", ln, size=7.4)
    s.gap("R", 0.012)

    s.head("R", "ladcp-compare   (vs legacy LDEO .mat)")
    s.kv("R", [
        ("--ours D", "ladcp-qa out dir (stations/*/.nc)"),
        ("--legacy D", "legacy processed dir (dr *.mat)"),
        ("--alt-dir D", "sub a rerun in for --alt-stations"),
        ("--sadcp D", "overlay ship-ADCP on each cast"),
        ("-o D", "report dir → .csv + report .pdf"),
    ], dx=0.150, key_size=7.2)

    s.footer(2)
    s.save(pdf)


def main(out: str = "guide/ladcp_cli_guide.pdf") -> str:
    with PdfPages(out) as pdf:
        _page1(pdf)
        _page2(pdf)
    return out


if __name__ == "__main__":
    print("wrote", main())

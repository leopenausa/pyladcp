#!/bin/bash
# CODAS *single-ping* processing of the MORIA OS150 (VmDAS ENR + N1R/N2R).
# Phase 1b of the CODAS-integration plan (memory: ladcp-codas-sadcp-plan).
#
# Release quick_adcp.py rejects --datatype enx, so single-ping goes through the
# documented "reform" route instead: FakeUHDAS recasts ENR (raw beam pings) +
# N1R/N2R (NMEA: $INHDT gyro, $PASHR attitude, $GPGGA fixes) as a UHDAS-style
# tree, then quick_adcp.py --datatype uhdas runs FULL CODAS processing —
# single-ping editing, beam->earth with gyro heading + PASHR heading correction,
# our own 120 s averages. Reference: codas_demos/adcp_pyproc/ps0918_vmdas.
#
# Prereqs: conda env `pycodas` (scripts/codas_install.sh).
# Usage:   bash scripts/codas_moria_os150_enr.sh [WORKDIR]
#   WORKDIR default: /home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas
set -euo pipefail
source /home/leo/anaconda3/etc/profile.d/conda.sh
conda activate pycodas

DATA=/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/sADCP/sadcp_150/DATA
WORK=${1:-/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas}
CRUISE=moria150            # uhdas cruisename; config file must be ${CRUISE}_proc.py
PROC=os150nb_enr           # processing tree (single-ping route)
EA=45.89                   # onboard beam-3 alignment, from the .VMO (AlignmentOffsetEA)

mkdir -p "$WORK"
cd "$WORK"

echo "=== [1/5] stage chronological ENR+N1R+N2R copies ==="
# Same trap as the STA route (see codas_moria_os150.sh): parts of the chain glob
# by FILENAME, and MORIA's name order is not chronological. Stage NNN_-prefixed
# copies per basename group, ordered by the first ENR ensemble's RTC time.
STAGE="$WORK/enr_chrono"
if [ ! -d "$STAGE" ]; then
    mkdir -p "$STAGE"
    python - "$STAGE" "$DATA"/*.ENR << 'PYEOF'
import shutil, struct, sys
from pathlib import Path

def first_time(path):
    b = Path(path).read_bytes()[:200000]
    i = b.find(b"\x7f\x7f")
    while i != -1:
        try:
            ndt = b[i + 5]
            offs = struct.unpack_from("<%dH" % ndt, b, i + 6)
            for o in offs:
                if struct.unpack_from("<H", b, i + o)[0] == 0x0080:  # variable leader
                    y, mo, d, h, mi, s = struct.unpack_from("<6B", b, i + o + 4)
                    return (2000 + y, mo, d, h, mi, s)
        except (IndexError, struct.error):
            pass
        i = b.find(b"\x7f\x7f", i + 2)
    return None        # empty / aborted recording

# Group parts (_000000, _000001, ...) by DEPLOYMENT: pycurrents' convert_rbins
# globs nav rbins with the name minus the part token, because VmDAS rolls ENR
# and N?R files at DIFFERENT moments — same-numbered parts do NOT cover the
# same pings (e.g. ENR _000016 pings 30785-32087 vs N2R _000016 33028-35196).
# A per-part prefix would break that deployment-wide glob and silently drop
# whole blocks; prefix per deployment instead (parts already sort correctly).
deps = {}
for enr in sys.argv[2:]:
    p = Path(enr)
    dep = p.stem.rsplit("_", 1)[0]          # MORIA-SADCP150-5002_000016 -> ...5002
    t = first_time(enr)
    if t is None:
        print(f"SKIP (no ensembles): {enr}", file=sys.stderr)
        continue
    deps.setdefault(dep, []).append((t, p))

stage = Path(sys.argv[1])
order = sorted(deps.items(), key=lambda kv: min(t for t, _ in kv[1]))
for k, (dep, parts) in enumerate(order, 1):
    for _, enr in sorted(parts):
        for sib in (enr, enr.with_suffix(".N1R"), enr.with_suffix(".N2R")):
            if sib.exists():
                shutil.copy2(sib, stage / f"{k:03d}_{sib.name}")
print(f"staged {len(order)} deployments, "
      f"{sum(len(p) for p in deps.values())} parts")
PYEOF
fi
ls "$STAGE"/*.ENR | wc -l

echo "=== [2/5] reform: FakeUHDAS (ENR -> UHDAS-style tree) ==="
UHDAS_DIR="$WORK/uhdas_style/${CRUISE}_os150"
mkdir -p "$WORK/uhdas_style"
if [ ! -d "$UHDAS_DIR" ]; then
    python - << PYEOF
import numpy as np
from pycurrents.adcp.vmdas import FakeUHDAS

class FakeUHDAS_pingmatch(FakeUHDAS):
    """Match nav records to ENR blocks by PING NUMBER, not nearest-in-time.

    Stock _find_time_pingnum takes the nav record nearest in TIME and demands
    it carry the exact ping number — but MORIA's VmDAS PC clock drifts 0-13 s
    against the GPS-stamped NMEA stream (cal "nav - pc" 7 +- 6 s), so whole
    2-h blocks failed ("no ping number match") whenever the offset exceeded a
    ping interval: 90/274 gps rbins empty, every other block missing around
    Oct-02/03. The pings always existed in the nav set; key on them directly
    and use time only to disambiguate ping-number resets.
    """
    @staticmethod
    def _find_time_pingnum(bfsa, dday, pingnum):
        hits = np.nonzero(bfsa[:, -1] == pingnum)[0]
        if hits.size == 0:
            return None
        return int(hits[np.argmin(np.abs(bfsa[hits, 0] - dday))])

# (instrument, message, N_R number): N1R = \$INHDT gyro (reliable heading);
# N2R = \$PASHR POS/MV-style attitude (heading correction), \$GPGGA (positions),
# \$GPHDT (secondary heading).
navinfo = [
    ("N1R", "hdg", 1),
    ("N2R", "pmv", 2),
    ("N2R", "gps", 2),
    ("N2R", "hdg", 2),
]
F = FakeUHDAS_pingmatch(yearbase=2025,
              sourcedir="$STAGE",
              destdir="$UHDAS_DIR",
              sonar="os150",
              navinfo=navinfo,
              ship="mo",
              dt_factor=3)   # tolerant block-splitting (variable ping rate)
F()
PYEOF
fi
find "$UHDAS_DIR/raw/os150" -name "*.raw" | wc -l   # (no ls|head: SIGPIPE + pipefail)

echo "=== [3/5] uhdas proc config ==="
CFG="$WORK/uhdas_config"
mkdir -p "$CFG"
cat > "$CFG/${CRUISE}_proc.py" << PYEOF
cruiseid = "$CRUISE"
yearbase = 2025
uhdas_dir = "$UHDAS_DIR"
shipname = "MORIA"

# serial inputs (rbin directories made by the reform step)
pos_inst = "N2R"
pos_msg = "gps"
pitch_inst = ""
pitch_msg = ""
roll_inst = ""
roll_msg = ""
hdg_inst = "N1R"          # \$INHDT gyro: reliable heading for beam->earth
hdg_msg = "hdg"

hdg_inst_msgs = [
    ("N1R", "hdg"),
    ("N2R", "pmv"),
    ("N2R", "hdg"),
]
hcorr_inst = "N2R"        # \$PASHR attitude: accurate heading correction
hcorr_msg = "pmv"
hcorr_gap_fill = 0.0
acc_heading_cutoff = 0.02

# ADCP transformations
h_align = dict(os150=$EA)         # onboard EA from the .VMO
ducer_depth = dict(os150=5)
scalefactor = dict(os150bb=1.0, os150nb=1.0)
soundspeed = dict(os150bb=None, os150nb=None)
salinity = dict(os150bb=None, os150nb=None)

# quick_adcp.py values
max_search_depth = dict(os150bb=2000, os150nb=2000)
weakprof_numbins = dict(os150bb=None, os150nb=None)
enslength = dict(os150bb=120, os150nb=120)    # match the onboard STA interval
xducer_dx = dict()
xducer_dy = dict()
PYEOF

echo "=== [4/5] processing tree + control file ==="
if [ ! -d "$PROC" ]; then
    adcptree.py "$PROC" --datatype uhdas --cruisename "$CRUISE" --configpath "$CFG"
fi
cd "$PROC"
cat > q_py.cnt << EOF
 --yearbase 2025              ## year of first data (cruise Sep-Oct 2025)
 --cruisename $CRUISE         ## must match config/${CRUISE}_proc.py
 --dbname aMORIA150e          ## single-ping database (e suffix = ENR route)
 --datatype uhdas             ## reformed VmDAS single-ping
 --sonar os150nb              ## OS150 narrowband
 --ens_len 120                ## our averaging interval [s] (= onboard STA)
 --update_gbin                ## build gbins from the reformed rbins (required)
 --configtype python          ## config/ files are python (*_proc.py)
 --ping_headcorr              ## apply PASHR-gyro heading correction per ping
 --max_search_depth 2000      ## bottom identification on the shelf legs
EOF
cat q_py.cnt

echo "=== [5/5] quick_adcp.py --auto ==="
rm -rf "$UHDAS_DIR/gbin"      # --update_gbin refuses to run over stale gbins
quick_adcp.py --cntfile q_py.cnt --auto

echo
echo "=== CALIBRATIONS ==="
for f in cal/watertrk/adcpcal.out cal/botmtrk/btcaluv.out; do
    if [ -f "$f" ]; then echo "--- $f ---"; tail -20 "$f"; fi
done
echo "=== CODAS OS150 SINGLE-PING RUN COMPLETE ==="

#!/bin/bash
# CODAS processing of the MORIA OS150 shipboard ADCP (VmDAS STA averages).
# Phase 1 of the CODAS-integration plan (memory: ladcp-codas-sadcp-plan).
#
# NOTE on single-ping: the release quick_adcp.py only accepts datatype
# uhdas/lta/sta/pingdata -- VmDAS ENX/ENS/ENR single-ping requires the
# documented reform route (reform_vmdas.py -> vmdas2uhdas.py -> adcptree
# --datatype uhdas), kept as phase-1b. STA still yields the watertrack/
# bottomtrack calibration (misalignment + amplitude) and CODAS editing.
#
# Prereqs: conda env `pycodas` (see the plan memory / currents.soest.hawaii.edu
# conda route: codas3 + pycurrents + onship installed from the stable branches).
#
# Usage:  bash scripts/codas_moria_os150.sh [WORKDIR]
#   WORKDIR default: /home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas
# Other instruments: override via env, e.g. for the OS75 (also narrowband):
#   SADCP_DATA=.../sADCP/sadcp_75/DATA SADCP_SONAR=os75nb \
#   SADCP_PROC=os75nb_sta SADCP_DB=aMORIA75 bash scripts/codas_moria_os150.sh
set -euo pipefail
source /home/leo/anaconda3/etc/profile.d/conda.sh
conda activate pycodas

DATA=${SADCP_DATA:-/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/sADCP/sadcp_150/DATA}
WORK=${1:-/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas}
PROC=${SADCP_PROC:-os150nb_sta}
SONAR=${SADCP_SONAR:-os150nb}
DBNAME=${SADCP_DB:-aMORIA150}

mkdir -p "$WORK"
cd "$WORK"

echo "=== [1/4] processing tree ==="
if [ ! -d "$PROC" ]; then
    adcptree.py "$PROC" --datatype sta --cruisename MORIA
fi
cd "$PROC"

echo "=== [2/4] data file list (time-sorted STA) ==="
# CODAS requires CHRONOLOGICAL input, and parts of the load chain (the ldcodas
# .cmd block list in load/) glob by FILENAME regardless of --data_filelist. A
# plain name sort is wrong for MORIA: the MoriaTest150*/MORIA-SADCP150001 files
# (cruise start, 682 ensembles) sort AFTER the dashed MORIA-SADCP150-* files,
# which loaded the db blocks out of order and misaligned the time-keyed nav
# merge by those 682 ensembles — poisoning every absolute velocity in the first
# product (found 2026-06-10 via the pyladcp reader: on-station "ocean" u read
# ~-4 m/s ship speed). Fix: stage COPIES whose NAME order == time order
# (NNN_ prefix from each file's first-ensemble RTC in the PD0 variable leader),
# so every internal glob and the filelist agree on chronology. Copies, not
# symlinks: pycurrents resolves symlinks and falls back to the original names.
STAGE="$WORK/sta_chrono_$PROC"
rm -rf "$STAGE" && mkdir -p "$STAGE"
python - "$STAGE" "$DATA"/*.STA << 'PYEOF'
import shutil, struct, sys
from pathlib import Path

def scan(path):
    """(first ensemble RTC, has any valid GPS fix) from the PD0 stream."""
    b = Path(path).read_bytes()
    t0, has_nav = None, False
    i = b.find(b"\x7f\x7f")
    while i != -1:
        try:
            ndt = b[i + 5]
            offs = struct.unpack_from("<%dH" % ndt, b, i + 6)
            for o in offs:
                tid = struct.unpack_from("<H", b, i + o)[0]
                if tid == 0x0080 and t0 is None:        # variable leader
                    y, mo, d, h, mi, s = struct.unpack_from("<6B", b, i + o + 4)
                    t0 = (2000 + y, mo, d, h, mi, s)
                elif tid == 0x2000 and not has_nav:     # VmDAS nav block
                    la = struct.unpack_from("<i", b, i + o + 14)[0]
                    lo = struct.unpack_from("<i", b, i + o + 18)[0]
                    has_nav = (la, lo) != (0, 0)
        except (IndexError, struct.error):
            pass
        if t0 is not None and has_nav:
            break
        i = b.find(b"\x7f\x7f", i + 2)
    return t0, has_nav

usable = []
for f in sys.argv[2:]:
    t0, has_nav = scan(f)
    if t0 is None:
        print(f"SKIP (no ensembles): {f}", file=sys.stderr)
    elif not has_nav:
        # an all-bad-nav file translates to a profile-less .cmd block that
        # crashes the loader (get_cmd_timerange IndexError) — e.g. the OS75
        # bench test MoriaTest75001 (11 ensembles, no GPS)
        print(f"SKIP (no valid GPS fix): {f}", file=sys.stderr)
    else:
        usable.append((t0, f))

stage = Path(sys.argv[1])
for k, (_, f) in enumerate(sorted(usable), 1):
    shutil.copy2(f, stage / f"{k:03d}_{Path(f).name}")
PYEOF
mkdir -p ping
ls "$STAGE"/*.STA > ping/data_filelist.txt
wc -l ping/data_filelist.txt
head -3 ping/data_filelist.txt

echo "=== [3/4] control file ==="
cat > q_py.cnt << EOF
 --yearbase 2025              ## year of first data (cruise Sep-Oct 2025)
 --cruisename MORIA           ## titles
 --dbname $DBNAME             ## database name; in adcpdb
 --datatype sta               ## VmDAS short-term averages (~2 min)
 --data_filelist ping/data_filelist.txt  ## sorted in time order
 --sonar $SONAR               ## instrument + ping type (nb = narrowband)
 --ens_len 120                ## the onboard STA averaging interval [s]
 --max_search_depth 2000     ## allow bottom identification on the shelf legs
EOF
cat q_py.cnt

echo "=== [4/4] quick_adcp.py --auto ==="
quick_adcp.py --cntfile q_py.cnt --auto

echo
echo "=== CALIBRATIONS ==="
[ -f cals.txt ] && cat cals.txt || true
for f in cal/watertrk/adcpcal.out cal/botmtrk/btcaluv.out; do
    if [ -f "$f" ]; then echo "--- $f ---"; tail -20 "$f"; fi
done
echo "=== CODAS OS150 STA RUN COMPLETE ==="

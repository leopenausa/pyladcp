#!/bin/bash
# CODAS processing of the MORIA OS75 shipboard ADCP (VmDAS STA averages).
# Sibling of codas_moria_os150.sh, pre-set for the 75 kHz narrowband instrument.
# Phase 1 of the CODAS-integration plan (memory: ladcp-codas-sadcp-plan).
#
# Why a dedicated OS75 script: the OS150 one takes the same env overrides, but the
# OS75 has its own defaults (data dir, sonar, db name) and an ens_len/search-depth
# you must confirm per cruise -- baking them in makes the rerun a one-liner and
# keeps the OS75 quirks (bench-test file, swapped serial feeds) documented in place.
#
# This is the STA route: good enough for the LADCP on-station inverse constraint
# (raw vs CODAS interchangeable to ~1 cm/s on station -- the cal errors are
# ship-speed-proportional and vanish at rest). For SECTIONS / underway work use the
# single-ping ENR route instead (scripts/codas_moria_os150_enr.sh) -- STA bins
# beyond the noise-limited range survive CODAS editing underway. MORIA's OS75 fed
# the gyro on the second NMEA serial line, so the ENR route needs ENR_GYRO_NR=2.
#
# NOTE on single-ping: the release quick_adcp.py only accepts datatype
# uhdas/lta/sta/pingdata -- VmDAS ENX/ENS/ENR single-ping requires the reform route
# (reform_vmdas.py -> vmdas2uhdas.py -> adcptree --datatype uhdas). STA still yields
# the watertrack/bottomtrack calibration (misalignment + amplitude) and CODAS editing.
#
# Prereqs: conda env `pycodas` (bash scripts/codas_install.sh; codas3 + pycurrents
# + onship from the stable branches).
#
# Usage:  bash scripts/codas_moria_os75.sh [WORKDIR]
#   WORKDIR default: /home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas
# Tunables (override via env if your cruise differs):
#   SADCP_DATA      raw OS75 VmDAS dir holding the *.STA files
#   SADCP_ENS_LEN   onboard STA averaging interval [s] -- READ IT FROM THE .VMO config
#   SADCP_MAXDEPTH  bottom-search depth [m]; >= the cruise's deepest shelf
set -euo pipefail
source /home/leo/anaconda3/etc/profile.d/conda.sh
conda activate pycodas

DATA=${SADCP_DATA:-/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/sADCP/sadcp_75/DATA}
WORK=${1:-/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas}
PROC=${SADCP_PROC:-os75nb_sta}
SONAR=${SADCP_SONAR:-os75nb}
DBNAME=${SADCP_DB:-aMORIA75}
ENS_LEN=${SADCP_ENS_LEN:-120}        # OS150 used 120 s; confirm the OS75 .VMO value
MAXDEPTH=${SADCP_MAXDEPTH:-2000}     # OS75 reaches deeper than the OS150; raise if needed

mkdir -p "$WORK"
cd "$WORK"

echo "=== [1/4] processing tree ==="
# quick_adcp.py --auto refuses to load into a tree that already holds a database
# ("A database is already in place; ... only by using the 'incremental'"). A bare
# "skip adcptree if the dir exists" guard is therefore NOT rerun-safe: the stale db
# from a previous run blocks the reload. Archive any existing tree aside (a dated
# .prev_* copy, so the old product is preserved) and always build a clean tree.
if [ -d "$PROC" ]; then
    BAK="${PROC}.prev_$(date +%Y%m%d_%H%M%S)"
    echo "existing $PROC -> $BAK (archived; delete it once the new run checks out)"
    mv "$PROC" "$BAK"
fi
adcptree.py "$PROC" --datatype sta --cruisename MORIA
cd "$PROC"

echo "=== [2/4] data file list (time-sorted STA) ==="
# CODAS requires CHRONOLOGICAL input, and parts of the load chain (the ldcodas
# .cmd block list in load/) glob by FILENAME regardless of --data_filelist. A
# plain name sort is wrong for MORIA: the MoriaTest*/MORIA-SADCP75001 files
# (cruise start) sort AFTER the dashed MORIA-SADCP75-* files, which loads the db
# blocks out of order and misaligns the time-keyed nav merge — poisoning every
# absolute velocity (on-station "ocean" u reads ~ship speed). Fix: stage COPIES
# whose NAME order == time order (NNN_ prefix from each file's first-ensemble RTC
# in the PD0 variable leader), so every internal glob and the filelist agree on
# chronology. Copies, not symlinks: pycurrents resolves symlinks and falls back to
# the original names.
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
 --ens_len $ENS_LEN           ## the onboard STA averaging interval [s]
 --max_search_depth $MAXDEPTH ## allow bottom identification on the shelf legs
EOF
cat q_py.cnt

echo "=== [4/4] quick_adcp.py --auto ==="
quick_adcp.py --cntfile q_py.cnt --auto

echo
echo "=== CALIBRATIONS ==="
# --auto COMPUTES the watertrack cal but does NOT apply it. Read the MEDIAN amplitude
# + phase below, then apply with the rotate step (see docs/SADCP_CODAS.md §3):
#   quick_adcp.py --cntfile q_py.cnt --auto \
#     --steps2rerun rotate:navsteps:calib:matfiles:netcdf \
#     --rotate_angle <phase> --rotate_amplitude <amplitude>
[ -f cals.txt ] && cat cals.txt || true
for f in cal/watertrk/adcpcal.out cal/botmtrk/btcaluv.out; do
    if [ -f "$f" ]; then echo "--- $f ---"; tail -20 "$f"; fi
done
echo
echo "Deliverable for pyladcp: $WORK/$PROC/contour/$SONAR.nc"
echo "Rerun: ladcp-qa --all-stations --index .ladcp_archive.json --root . \\"
echo "         --sadcp $WORK/$PROC/contour/$SONAR.nc --sadcp-source codas \\"
echo "         --sadcpfac 3 --out qa_out_codas75"
echo "=== CODAS OS75 STA RUN COMPLETE ==="

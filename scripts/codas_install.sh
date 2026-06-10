#!/bin/bash
# CODAS phase-0 install: conda env + source build + demos + topography.
set -euo pipefail
source /home/leo/anaconda3/etc/profile.d/conda.sh

echo "=== [1/6] conda env pycodas ==="
if ! conda env list | grep -q "^pycodas "; then
    conda env create -f "$(dirname "$0")/codas_processing.yml"
fi
conda activate pycodas

echo "=== [2/6] clone CODAS repos (stable) ==="
mkdir -p ~/adcpcode/programs ~/adcpcode/topog
cd ~/adcpcode/programs
for repo in codas3 pycurrents onship; do
    if [ ! -d "$repo" ]; then
        git clone -b stable "https://currents.soest.hawaii.edu/git/uh-currents-group/shipboard-adcp/${repo}.git"
    fi
done

echo "=== [3/6] build codas3 C tools ==="
cd ~/adcpcode/programs/codas3
./conda-install.sh

echo "=== [4/6] pip install pycurrents + onship ==="
cd ~/adcpcode/programs/pycurrents
pip install --no-build-isolation .
cd ../onship
pip install --no-build-isolation .

echo "=== [5/6] demos + docs ==="
cd ~/adcpcode/programs
[ -f codas_demos.zip ] || curl -sO https://currents.soest.hawaii.edu/docs/zipped/codas_demos.zip
[ -d codas_demos ] || unzip -q codas_demos.zip

echo "=== [6/6] etopo topography ==="
mkdir -p ~/adcpcode/topog/etopo
cd ~/adcpcode/topog/etopo
[ -f etopo1_for_pycurrents.zip ] || curl -sO https://currents.soest.hawaii.edu/downloads/etopo1_for_pycurrents.zip
ls *.zip >/dev/null 2>&1 && unzip -qn etopo1_for_pycurrents.zip || true

# pycurrents.data.topo.find_directory walks UP from the installed package looking
# for topog/etopo; with a pip install into site-packages it never reaches
# ~/adcpcode/topog, and single-ping averaging (Pingavg bottom editing) fails every
# segment with "'Pingavg' object has no attribute 'topo'". Link it into the env.
ln -sfn ~/adcpcode/topog "$CONDA_PREFIX/topog"

echo "=== smoke test ==="
python -c "import pycurrents; print('pycurrents OK:', pycurrents.__file__)"
quick_adcp.py --help > /dev/null && echo "quick_adcp.py OK"
which lst_conf 2>/dev/null && echo "codas3 binaries OK" || echo "codas3 binaries: check PATH"
echo "=== CODAS INSTALL COMPLETE ==="

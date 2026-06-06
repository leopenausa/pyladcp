# pyladcp

[![CI](https://github.com/leopenausa/pyladcp/actions/workflows/ci.yml/badge.svg)](https://github.com/leopenausa/pyladcp/actions/workflows/ci.yml)

Pythonic processing and **acquisition-quality assessment** for lowered-ADCP (LADCP) casts —
a clean-room re-implementation of the LDEO_IX (Visbeck) inverse-method workflow, validated
against legacy LDEO_IX outputs (MORIA 2025).

From a dual-head (down + up looker) RDI PD0 pair and a cleaned CTD cast, pyladcp produces a
QA scorecard, an ocean **velocity profile** (`u`, `v` vs depth), a bottom-track-referenced
profile, and a multi-page PDF report.

## Install

```bash
pip install pyladcp            # (PyPI release pending; for now use the source install below)
```

From source:

```bash
git clone https://github.com/leopenausa/pyladcp
cd pyladcp
pip install -e ".[dev]"
```

A conda environment is also provided (`conda env create -f environment.yml`).

## Quickstart

```bash
# auto-discover a station under <root>/LADCP and <root>/CTD
ladcp-qa 80 --root /path/to/cruise --out qa_out

# or name the files explicitly
ladcp-qa --down cast-M.000 --up cast-S.000 --ctd cast_clean.cnv --station MyCast --out qa_out
```

Each station yields, in `--out`:

- `<station>_qa.txt` / `.json` — quality-assessment scorecard
- `<station>.lad` — ocean velocity profile (`z:u:v:ev`)
- `<station>.bot` — bottom-track-referenced profile (`z:u:v:err`)
- `<station>_report.pdf` — 8-page report (scorecard + acquisition figures + velocity / shear /
  inversion-diagnostics) and the standalone PNGs

The magnetic declination defaults to IGRF-13 from the cast position; override with `--drot`.

## Library

```python
from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import compute_velocity_full

dh  = load_dualhead("cast-M.000", "cast-S.000", station="MyCast", params=moria05_params())
ctd = read_ctd_cnv("cast_clean.cnv", params=moria05_params())
res = compute_velocity_full(dh, ctd, drot=-5.4)   # res.vp (.lad), res.bp (.bot), res.shear
```

## Status

Validated end-to-end against the MORIA-80 golden: velocity `u` corr 0.998, bottom-track
profile corr 0.991. The current solver is the **shear method + reference** (the LDEO_IX
`ps.shear==1` path); a full sparse inverse is on the roadmap.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what's next (full inverse, ship-ADCP constraint,
config-robustness across instruments, Excel/ODV export, CTD-pipeline integration) and
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) for the data interfaces.

## Acknowledgments

pyladcp re-implements the algorithms of the **LDEO_IX** LADCP package by Martin Visbeck and
colleagues (Lamont-Doherty Earth Observatory). The original MATLAB code is not distributed
here. If you use pyladcp, please cite it (see [`CITATION.cff`](CITATION.cff)) and acknowledge
LDEO_IX.

## License

MIT — see [`LICENSE`](LICENSE).

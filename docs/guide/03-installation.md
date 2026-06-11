# 3 · Installation

pyladcp is pure Python (≥ 3.10) on the standard scientific stack — no MATLAB, no
compiled extensions of its own. Linux, macOS and Windows are all tested in CI on
every commit.

## Install from GitHub

```bash
git clone https://github.com/leopenausa/pyladcp
cd pyladcp
pip install -e .
```

All dependencies (numpy, scipy, pandas, xarray, netCDF4, matplotlib, gsw, ppigrf)
install automatically. Prefer conda? `conda env create -f environment.yml` builds a
ready `pyladcp` environment instead.

## Verify it

The repository ships a complete real station as a test fixture, so the test suite is
a genuine end-to-end verification:

```bash
pip install pytest
pytest -q     # expect ~246+ passed, ~21 skipped (skips = local-only cruise data)
```

Then prove it to yourself properly: run the
[chapter 4 walkthrough](04-first-station.md) and open the report PDF.

## Optional extras

| extra | enables | install |
|---|---|---|
| Excel export | `.xlsx` per-station and cruise workbooks | `pip install "pyladcp[export]"` |
| Raw-CTD ingest | `--from-hex` (build `.cnv` from `.hex`+`.XMLCON`) | clone [CTD_pipeline](https://github.com/leopenausa/CTD_pipeline) beside pyladcp, or set `LADCP_CTD_PROJECT=/path/to/it` |
| CODAS ship-ADCP route | edited/calibrated SADCP products ([chapter 8](08-ship-adcp.md)) | `bash scripts/codas_install.sh` — **Linux/macOS only** |

Everything else — including reading raw VmDAS ship-ADCP data — works on all three
platforms with the base install.

## Platform notes

- **Windows:** fully supported for the core workflow (the CI suite runs on
  `windows-latest`). The CODAS scripts are the one unix-only piece.
- **Headless servers / CI:** prefix commands with `MPLBACKEND=Agg` so matplotlib
  never looks for a display.
- **Updating:** `git pull` in the checkout is enough with an editable install
  (`pip install -e .`); re-run `pytest -q` after updating.

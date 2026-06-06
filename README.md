# LADCP Pipeline

Python re-implementation of the LDEO_IX (Visbeck inverse-method) LADCP workflow, plus an
acquisition-quality assessment layer. Self-contained; validated against the MORIA 2025
legacy outputs.

## Environment

```bash
conda env create -f environment.yml      # creates 'ladcp_pipeline'
conda activate ladcp_pipeline
pip install -e .
```

## Validate against the legacy golden output

```bash
ladcp-validate MORIA-05 --root . --out validation_out
```

Runs the implemented pipeline components on MORIA-05 and diffs against
`figures/MORIA-05.{lad,bot,log}`, writing a JSON + Markdown report.

## Status

- ✅ conda env, package scaffold, data models (`ProfileResult`, `QCMetrics`)
- ✅ readers: RDI PD0 (dual-head, beam/earth auto-detect), cleaned CTD `.cnv`, golden `.lad/.bot/.log`
- ✅ IGRF-13 magnetic declination (`ppigrf`)
- ✅ validation harness + report; config/bottom-track/CTD gates pass on MORIA-05
- ⏳ **velocity inversion core** (super-ensembles + weighted least-squares) — next milestone
- ⏳ owned preprocessing: LADCP deployment→cast cutting, sADCP VmDAS→combined
- 🟡 known finding: legacy declination is ~2° biased (IGRF-2000 + hardcoded fudge); see docs

## Docs

- `docs/DATA_CONTRACT.md` — interfaces (inputs, `ProfileResult`, `QCMetrics`, outputs, validation)
- `docs/VALIDATION_MORIA05.md` — the single-cast validation spec

## Layout

```
src/ladcp/
  models.py config.py
  io/    pd0.py ctd.py golden.py
  proc/  magdec.py        (inversion stages to come)
  validate/ compare.py harness.py cli.py
tests/   test_smoke_moria05.py
```

#!/usr/bin/env bash
# Build the printable PDF of the processing guide with pandoc + xelatex.
# Usage: bash scripts/build_guide_pdf.sh [out.pdf]    (run from the repo root)
set -euo pipefail

OUT="${1:-pyladcp_guide.pdf}"
G="docs/guide"

# Chapters in reading order (keep in sync with mkdocs.yml nav)
CHAPTERS=(
  "$G/index.md"
  "$G/01-primer.md"
  "$G/02-before-you-process.md"
  "$G/03-installation.md"
  "$G/04-first-station.md"
  "$G/05-cruise-workflow.md"
  "$G/06-qa-report.md"
  "$G/07-solvers-weights.md"
  "$G/08-ship-adcp.md"
  "$G/09-troubleshooting.md"
  "$G/10-studio.md"
  "$G/appendix-a-flags.md"
  "$G/appendix-b-legacy-map.md"
  "$G/appendix-c-formats.md"
  "$G/appendix-d-logsheets.md"
  "$G/appendix-e-glossary.md"
)

# --from markdown (not gfm): the markdown reader assigns relative column widths
# to wide pipe tables, so long cells wrap instead of overflowing the page.
pandoc "${CHAPTERS[@]}" \
  --from markdown \
  --resource-path="$G" \
  --pdf-engine=xelatex \
  --lua-filter=scripts/guide_pdf_admonitions.lua \
  -H scripts/guide_pdf_preamble.tex \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  --toc --toc-depth=2 \
  -V title="Processing LADCP data with pyladcp" \
  -V author="Leopoldo D. Pena — Universitat de Barcelona" \
  -V date="$(date +%Y-%m-%d)" \
  -V geometry:margin=2.2cm \
  -V fontsize=10pt \
  -V colorlinks=true \
  -o "$OUT"

echo "wrote $OUT"

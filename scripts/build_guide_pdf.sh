#!/usr/bin/env bash
# Build the printable PDF of the processing guide.
# Usage: bash scripts/build_guide_pdf.sh [out.pdf]    (run from the repo root)
#
# Engine selection (best look first):
#   1. pandoc + xelatex   — the canonical LaTeX build (needs a TeX install)
#   2. pandoc + weasyprint — CSS fallback (scripts/guide_pdf.css); needs pandoc >= 2.17
#      and the `weasyprint` package. Produces the same content with book-style typography
#      when no TeX engine is available.
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

TITLE="Processing LADCP data with pyladcp"
AUTHOR="Leopoldo D. Pena · Universitat de Barcelona"

if command -v xelatex >/dev/null 2>&1; then
  # --- canonical xelatex build ---
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
    -V title="$TITLE" \
    -V author="$AUTHOR" \
    -V date="$(date +%Y-%m-%d)" \
    -V geometry:margin=2.2cm \
    -V fontsize=10pt \
    -V colorlinks=true \
    -o "$OUT"
else
  # --- weasyprint CSS fallback (no TeX engine present) ---
  command -v weasyprint >/dev/null 2>&1 || {
    echo "error: neither xelatex nor weasyprint found — install a TeX engine or 'pip install weasyprint'" >&2
    exit 1
  }
  # Run pandoc from inside $G so the markdown's relative image paths (assets/*.png)
  # resolve for weasyprint; filter/css/output are made absolute first.
  ROOT="$PWD"
  case "$OUT" in /*) OUT_ABS="$OUT" ;; *) OUT_ABS="$ROOT/$OUT" ;; esac
  BASENAMES=("${CHAPTERS[@]#"$G"/}")
  ( cd "$G" && pandoc "${BASENAMES[@]}" \
      --from markdown \
      --embed-resources --standalone \
      --lua-filter="$ROOT/scripts/guide_pdf_admonitions.lua" \
      --css="$ROOT/scripts/guide_pdf.css" \
      --pdf-engine=weasyprint \
      --toc --toc-depth=2 \
      -V title="$TITLE" \
      -V author="$AUTHOR" \
      -o "$OUT_ABS" )
fi

echo "wrote $OUT"

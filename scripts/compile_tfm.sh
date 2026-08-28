#!/usr/bin/env bash
# Compila la memoria con el TeX Live portable del repo (.texlive/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/.texlive/bin/x86_64-linux:$PATH"
TEXDIR="$ROOT/project_tex/Plantilla_TFM_VIU__Universidad_Internacional_Valencia_"
cd "$TEXDIR"
pdflatex -interaction=nonstopmode -halt-on-error master.tex
bibtex master || true
pdflatex -interaction=nonstopmode -halt-on-error master.tex
pdflatex -interaction=nonstopmode -halt-on-error master.tex
echo "PDF: $TEXDIR/master.pdf"

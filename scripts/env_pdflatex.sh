# Añade pdflatex (TeX Live portable del repo) al PATH de la sesión.
# Desde la raíz del repo:
#   source scripts/env_pdflatex.sh
if [ -d "${PWD}/.texlive/bin/x86_64-linux" ]; then
  ROOT="$PWD"
elif ROOT_GIT="$(git rev-parse --show-toplevel 2>/dev/null)" && [ -d "$ROOT_GIT/.texlive/bin/x86_64-linux" ]; then
  ROOT="$ROOT_GIT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
export PATH="$ROOT/.texlive/bin/x86_64-linux:$PATH"
hash -r 2>/dev/null || true
command -v pdflatex
pdflatex -version | head -1

#!/usr/bin/env bash
# Download + extract the TexasSolver v0.2.0 Linux release into a tools dir.
#
# TexasSolver (https://github.com/bupticybee/TexasSolver) is a free, open-source
# GTO solver. Its `console_solver` reads a command file and dumps a strategy
# JSON. The project's GTO integration (src/gto/, scripts/solve_spots_gto.py)
# drives it. The binary + its resources/ dir are large and gitignored.
#
# Usage:
#   bash scripts/setup_texassolver.sh [dest_dir]      # default: tools/texassolver
#   export TEXASSOLVER_BIN=<dest_dir>/console_solver
set -euo pipefail

DEST="${1:-tools/texassolver}"
VERSION="v0.2.0"
URL="https://github.com/bupticybee/TexasSolver/releases/download/${VERSION}/TexasSolver-${VERSION}-Linux.zip"

if [ -x "${DEST}/console_solver" ] && [ -d "${DEST}/resources" ]; then
  echo "Already installed at ${DEST}/console_solver"
  exit 0
fi

mkdir -p "${DEST}"
TMP="$(mktemp -d)"
echo "Downloading ${URL} ..."
curl -fL --max-time 300 -o "${TMP}/ts.zip" "${URL}"
echo "Extracting ..."
unzip -q "${TMP}/ts.zip" -d "${TMP}"
# The zip contains a top-level TexasSolver-<ver>-Linux/ folder; flatten it so
# console_solver + resources/ sit directly under DEST.
SRC="$(find "${TMP}" -maxdepth 2 -name console_solver -printf '%h\n' | head -1)"
if [ -z "${SRC}" ]; then echo "console_solver not found in archive"; exit 1; fi
cp -r "${SRC}/console_solver" "${SRC}/resources" "${DEST}/"
chmod +x "${DEST}/console_solver"
rm -rf "${TMP}"

echo "Installed → ${DEST}/console_solver"
echo "Set: export TEXASSOLVER_BIN=${DEST}/console_solver"

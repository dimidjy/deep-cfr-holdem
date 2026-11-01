#!/usr/bin/env bash
# Download the published checkpoint + equity model from the GitHub release.
#
#   bash scripts/download_models.sh            # release v0.1.0
#   RELEASE=v0.2.0 bash scripts/download_models.sh
set -euo pipefail

REPO="dimidjy/deep-cfr-holdem"
RELEASE="${RELEASE:-v0.1.0}"
BASE="https://github.com/${REPO}/releases/download/${RELEASE}"

# file  destination  sha256
FILES=(
  "checkpoint_iter_4000.pt models/best/checkpoint_iter_4000.pt f2a586a5513d2360808a2a821eea1e7f1f7225ff22eeb915beff8dfc575de3fe"
  "equity_model_v3.pt models/equity_model_v3.pt 6c5c40d8b8544e6c7973636d6016496454b99c2dc48dcd0538f9f41b36d73cdd"
)

for entry in "${FILES[@]}"; do
  read -r name dest sha <<< "${entry}"
  if [ -f "${dest}" ] && echo "${sha}  ${dest}" | sha256sum -c --status; then
    echo "ok        ${dest}"
    continue
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "download  ${BASE}/${name}"
  curl -fL --progress-bar -o "${dest}.part" "${BASE}/${name}"
  echo "${sha}  ${dest}.part" | sha256sum -c --status \
    || { echo "checksum mismatch for ${name}"; rm -f "${dest}.part"; exit 1; }
  mv "${dest}.part" "${dest}"
done

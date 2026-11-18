#!/usr/bin/env bash
# bash scripts/data/prepare_all.sh [--limit N]
set -euo pipefail
source "$(dirname "$0")/../common.sh"
[[ -d "${MSP_ROOT}/Audios" ]] || { echo "set MSP_ROOT in configs/paths.env" >&2; exit 1; }
"${MAIN_PY}" scripts/data/utterances.py
"${MAIN_PY}" scripts/data/wavlm_labels.py
"${MAIN_PY}" scripts/data/sequence_manifests.py
"${MAIN_PY}" scripts/data/export_sequences.py "$@"
echo "data ready under ${DATA_DIR} (sequence audio: ${CONCAT_DIR})"

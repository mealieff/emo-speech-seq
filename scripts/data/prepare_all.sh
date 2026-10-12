#!/usr/bin/env bash
# Build every data file from the MSP-Podcast 1.12 release (CPU only; set MSP_ROOT first).
#
#   1. utterance lists (Train / Dev / Test1, 8 classes)       -> data/utterances/
#   2. WavLM IS2025 label table                               -> data/wavlm/processed_labels.csv
#   3. sequence manifests (runs of consecutive segments)      -> data/sequences/manifests/
#   4. concatenated sequence WAVs + sequence JSONL            -> $CONCAT_DIR, data/sequences/
#
# Step 3 reads every segment's duration once (cached in data/cache/). Step 4 writes about
# 15,700 WAVs (about 7.3 GB) to CONCAT_DIR.
#
#   bash scripts/data/prepare_all.sh            # everything
#   bash scripts/data/prepare_all.sh --limit 8  # smoke test: 8 sequences per split in step 4
set -euo pipefail
source "$(dirname "$0")/../common.sh"

[[ -d "${MSP_ROOT}/Audios" ]] || { echo "MSP-Podcast audio not found under ${MSP_ROOT}; set MSP_ROOT in configs/paths.env" >&2; exit 1; }

"${MAIN_PY}" scripts/data/utterances.py
"${MAIN_PY}" scripts/data/wavlm_labels.py
"${MAIN_PY}" scripts/data/sequence_manifests.py
"${MAIN_PY}" scripts/data/export_sequences.py "$@"
echo "data ready under ${DATA_DIR} (sequence audio: ${CONCAT_DIR})"

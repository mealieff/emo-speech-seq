#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../common.sh"
MODEL="${1:?usage: run_utterance_sft.sh qwen2_audio|audio_flamingo3|qwen3_omni}"
OUT="${OUT:-${RUNS_DIR}/sft/utterance/${MODEL}}"
"${MAIN_PY}" scripts/finetune/utterance_lora.py --model "${MODEL}" --output-dir "${OUT}"

#!/usr/bin/env bash
# SALMONN and Kimi-Audio fine-tuning, then Test1 inference, on one GPU.
# SALMONN weights come from SALMONN_DIR (vicuna/, whisper/, beats/, salmonn/), not from
# the Hugging Face hub cache in MODEL_CACHE.
#
#   1. Sequence SFT, both models: 3 epochs, length-bin-weighted CE, full-Dev selection
#   2. Utterance SFT, both models: 3 epochs, last checkpoint
#   3. Test1 for the four fine-tuned systems (scripts/inference/run_test1.sh)
#
# Finished stages are skipped. Pretrained Test1 is scripts/inference/run_test1.sh salmonn_pt kimi_audio_pt.
#   bash scripts/finetune/run_bespoke_sft.sh
set -uo pipefail
source "$(dirname "$0")/../common.sh"
require_legacy

STATUS="${LOG_DIR}/bespoke_sft.status"
UTT_TRAIN="${UTT_DIR}/msp_podcast_train.jsonl"

utt_sft() {  # model
  local model=$1 dir="${RUNS_DIR}/sft/utterance/${model}"
  if [[ -s ${dir}/final_checkpoint.json ]]; then
    stage "${model}_utt_sft" "skip ${dir}/final_checkpoint.json"
    return 0
  fi
  stage "${model}_utt_sft" "start out=${dir}"
  "${LEGACY_PY}" scripts/finetune/bespoke_lora.py --model "${model}" --task utt \
    --train-jsonl "${UTT_TRAIN}" --epochs 3 --output-dir "${dir}"
  stage "${model}_utt_sft" "exit=$?"
}

stage chain "start SALMONN_DIR=${SALMONN_DIR}"
bash scripts/finetune/run_sequence_sft.sh salmonn
bash scripts/finetune/run_sequence_sft.sh kimi_audio
utt_sft salmonn
utt_sft kimi_audio
bash scripts/inference/run_test1.sh salmonn_sft kimi_audio_sft
stage chain done

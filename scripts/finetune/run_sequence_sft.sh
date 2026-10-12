#!/usr/bin/env bash
# Sequence LoRA SFT, end to end, for one model (single GPU):
#   1. train 3 epochs, one checkpoint per epoch, length-bin-weighted CE, Dev-256 loss per epoch
#   2. unweighted CE on the full Development set for every epoch checkpoint
#   3. select the checkpoint with the lowest full-Dev loss -> selection_full_dev.json
#
#   bash scripts/finetune/run_sequence_sft.sh qwen2_audio       # main environment
#   bash scripts/finetune/run_sequence_sft.sh audio_flamingo3   # main environment, <=180 s Train/Dev
#   bash scripts/finetune/run_sequence_sft.sh qwen3_omni        # main environment, one 80 GB GPU
#   bash scripts/finetune/run_sequence_sft.sh salmonn           # legacy environment, 30 s windows
#   bash scripts/finetune/run_sequence_sft.sh kimi_audio        # legacy environment
#
# Output: $EMOSEQ_RUNS/sft/sequence/<model>/ (override with OUT=...).
# SFT_EXTRA adds sequence_lora.py options that change speed, not the recipe, e.g.
#   SFT_EXTRA="--checkpoint-above-sec 60 --dataloader-workers 4" bash scripts/finetune/run_sequence_sft.sh qwen3_omni
set -euo pipefail
source "$(dirname "$0")/../common.sh"

MODEL="${1:?usage: run_sequence_sft.sh qwen2_audio|audio_flamingo3|qwen3_omni|salmonn|kimi_audio}"
OUT="${OUT:-${RUNS_DIR}/sft/sequence/${MODEL}}"
STATUS="${LOG_DIR}/sft_sequence_${MODEL}.status"
mkdir -p "${OUT}"

case "${MODEL}" in
  qwen2_audio)     TRAIN="${SEQ_DIR}/train.jsonl";     DEV="${SEQ_DIR}/dev.jsonl" ;;
  # QWEN3_SET=af3 trains on the <=180 s Train/Dev used for Audio-Flamingo-3.
  qwen3_omni)
    if [[ "${QWEN3_SET:-full}" == af3 ]]; then TRAIN="${SEQ_DIR}/train_af3.jsonl"; DEV="${SEQ_DIR}/dev_af3.jsonl"
    else TRAIN="${SEQ_DIR}/train.jsonl"; DEV="${SEQ_DIR}/dev.jsonl"; fi ;;
  audio_flamingo3) TRAIN="${SEQ_DIR}/train_af3.jsonl"; DEV="${SEQ_DIR}/dev_af3.jsonl" ;;
  salmonn|kimi_audio) TRAIN="${SEQ_DIR}/train.jsonl";  DEV="${SEQ_DIR}/dev.jsonl"; require_legacy ;;
  *) echo "unknown model ${MODEL}" >&2; exit 1 ;;
esac
N_DEV=$(wc -l < "${DEV}")
read -ra EXTRA <<< "${SFT_EXTRA:-}"

stage train "start ${MODEL} train=${TRAIN} out=${OUT} commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
provenance "${OUT}/provenance.json" model="${MODEL}" train_jsonl="${TRAIN}" dev_jsonl="${DEV}" sft_extra="${SFT_EXTRA:-}"
case "${MODEL}" in
  qwen2_audio|audio_flamingo3|qwen3_omni)
    "${MAIN_PY}" scripts/finetune/sequence_lora.py --model "${MODEL}" --train-jsonl "${TRAIN}" --eval-jsonl "${DEV}" \
      --length-bin-weight --epochs 3 --save-total-limit 0 --logging-steps 10 --eval-limit 256 --output-dir "${OUT}" \
      "${EXTRA[@]}"
    ;;
  *)
    "${LEGACY_PY}" scripts/finetune/bespoke_lora.py --model "${MODEL}" --task seq --train-jsonl "${TRAIN}" \
      --eval-jsonl "${DEV}" --length-bin-weight --epochs 3 --eval-limit 256 --output-dir "${OUT}"
    ;;
esac
stage train "done"

mapfile -t CKPTS < <(ls -d "${OUT}"/checkpoint-* | sort -t- -k2 -n)
stage full_dev "${#CKPTS[@]} checkpoints, n_dev=${N_DEV}"
case "${MODEL}" in
  qwen2_audio|audio_flamingo3|qwen3_omni)
    for ck in "${CKPTS[@]}"; do
      "${MAIN_PY}" scripts/finetune/sequence_lora.py --model "${MODEL}" --train-jsonl "${TRAIN}" --eval-jsonl "${DEV}" \
        --length-bin-weight --eval-limit "${N_DEV}" --eval-only-checkpoint "${ck}" \
        --eval-output "${OUT}/full_dev/$(basename "${ck}").json"
    done
    ;;
  *)
    "${LEGACY_PY}" scripts/finetune/bespoke_lora.py --model "${MODEL}" --task seq --train-jsonl "${TRAIN}" \
      --eval-jsonl "${DEV}" --eval-limit 0 --output-dir "${OUT}" \
      --eval-only-checkpoint "${CKPTS[@]}" --eval-output-dir "${OUT}/full_dev"
    ;;
esac

BEST=$("${MAIN_PY}" scripts/finetune/select_checkpoint.py "${OUT}")
stage select "best=${BEST}"

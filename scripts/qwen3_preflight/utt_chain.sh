#!/usr/bin/env bash
# Qwen3-Omni utterance SFT after the batch-1 SFT sequence Test1 (sft_seq_bs1.sh):
# 1. utt_timing.sh without the loss check: memory of batch 1 x 8 without checkpointing on the 8 longest
#    clips, then 50 timed steps; 2. 3-epoch utterance LoRA (configs/finetune/utterance_lora.json, the
#    Qwen2-Audio / AF3 recipe) with checkpointing off only if it fit; 3. utterance Test1 for qwen3_omni_sft
#    with the same spot-check rule as run_test1.sh hf_utt, plus scoring on the Test1 sequences.
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1
STATUS="${LOG_DIR}/qwen3_rerun.status"
T1="${RUNS_DIR}/test1"
SPOT="${T1}/spotcheck"
UTT_BS="${UTT_BS:-16}"
UTT_JSONL="${UTT_DIR}/msp_podcast_test1.jsonl"
SEQ_JSONL="${SEQ_DIR}/test1.jsonl"
OUT="${RUNS_DIR}/sft/utterance/qwen3_omni"

if [[ -n "${SFT_ARGS+x}" ]]; then
  read -ra ARGS <<< "${SFT_ARGS}"
else
while pgrep -f "qwen3_preflight/sft_seq_bs1.sh" > /dev/null; do sleep 60; done

stage utt_timing "start"
SKIP_LOSS_CHECK=1 bash "${HERE}/utt_timing.sh" > "${LOG_DIR}/qwen3_utt_timing.log" 2>&1
P="${RUNS_DIR}/qwen3_utt_timing"
grep -q '^time_a: FAILED' "${P}/summary.txt" && { stage utt_timing "time_a failed; stopping"; exit 1; }
ARGS=()
grep -q '^memory A (no checkpointing): fits' "${P}/summary.txt" && ARGS=(--no-gradient-checkpointing)
fi
stage utt_sft "start python=${MAIN_PY} args=${ARGS[*]:-(config defaults)}"

rm -rf "${OUT}"
provenance "${OUT}/provenance.json" model=qwen3_omni train_jsonl="${UTT_DIR}/msp_podcast_train.jsonl" \
  training_config=configs/finetune/utterance_lora.json extra_args="${ARGS[*]:-}"
"${MAIN_PY}" scripts/finetune/utterance_lora.py --model qwen3_omni "${ARGS[@]}"
rc=$?
stage utt_sft "exit=${rc}"
[[ ${rc} -eq 0 && -f "${OUT}/adapter_config.json" ]] || exit 1

sys=qwen3_omni_sft; out="${T1}/utterance/${sys}"; s="${SPOT}/${sys}"
mkdir -p "${s}"
for b in 1 "${UTT_BS}"; do
  "${MAIN_PY}" scripts/inference/utterance.py --model qwen3_omni --model-dir "${OUT}" --eval-jsonl "${SPOT}/utt.jsonl" \
    --batch-size "${b}" --out-dir "${s}/utt_bs${b}" > "${s}_utt_bs${b}.log" 2>&1
done
verdict=$("${MAIN_PY}" scripts/evaluate/spotcheck.py utt "${s}/utt_bs1/predictions_utt.json" \
  "${s}/utt_bs${UTT_BS}/predictions_utt.json" "${s}/utt_bs1/eval_utt.json" "${s}/utt_bs${UTT_BS}/eval_utt.json" \
  0.01 2>>"${STATUS}")
bs=1; [[ "${verdict}" == ok ]] && bs="${UTT_BS}"
stage "${sys}_utt" "start batch=${bs} model_dir=${OUT}"
rm -rf "${out}"
provenance "${out}/provenance.json" model=qwen3_omni model_dir="${OUT}" batch_size="${bs}" \
  parse_mode=first_n eval_jsonl="${UTT_JSONL}"
"${MAIN_PY}" scripts/inference/utterance.py --model qwen3_omni --model-dir "${OUT}" --eval-jsonl "${UTT_JSONL}" \
  --batch-size "${bs}" --out-dir "${out}"
stage "${sys}_utt" "exit=$?"

o="${T1}/utt_on_sequences/${sys}"
"${MAIN_PY}" scripts/evaluate/utterance_on_sequences.py --preds "${out}/predictions_test1.json" --jsonl "${SEQ_JSONL}" \
  --out-dir "${o}" --model "${sys}" > /dev/null
provenance "${o}/provenance.json" preds="${out}/predictions_test1.json" jsonl="${SEQ_JSONL}" label=pred_for_metrics boundaries=gold
stage utt_on_sequences "${sys} exit=$?"

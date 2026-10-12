#!/usr/bin/env bash
# Qwen3-Omni utterance LoRA, with the qwen3_omni_fix audio patches (emoseq/models/qwen3_omni_fix.py):
# 1. per-example eval-mode loss, batch 1 vs batch 8, 32 Train clips (pass: max |diff| <= 1e-3),
#    also without the patches as a control; plus batch-1 loss of 8 Train sequences, fixed vs unfixed
# 2. memory, 8 longest Train clips: A = batch 1 x accum 8, no gradient checkpointing
# 3. 50 optimizer steps of A on a seed-7 sample of 400 Train clips
# 4. only if 1 passed: memory + 50 steps of B = batch 8 x accum 1, no checkpointing, grouped by length
# Output: $EMOSEQ_RUNS/qwen3_utt_timing/summary.txt
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1

P="${RUNS_DIR}/qwen3_utt_timing"
mkdir -p "${P}"
STATUS="${P}/status.log"
SUMMARY="${P}/summary.txt"
: > "${SUMMARY}"
TRAIN="${UTT_DIR}/msp_podcast_train.jsonl"
N_TRAIN=$(wc -l < "${TRAIN}")
STEPS="${TIMING_STEPS:-50}"
say() { echo "$*" | tee -a "${SUMMARY}"; }
peak() { grep -h '^\[peak\]' "$1" | tail -1; }
failed() { grep -Eq "out of memory|OutOfMemory|NVML_SUCCESS|CUDA error" "$1" || ! grep -q "^Saved ->" "$1"; }
train() {  # tag jsonl args...
  local tag=$1 jsonl=$2; shift 2
  "${MAIN_PY}" "${HERE}/peak.py" scripts/finetune/utterance_lora.py --model qwen3_omni --train-jsonl "${jsonl}" \
    --epochs 1 --output-dir "${P}/${tag}" "$@" > "${P}/${tag}.log" 2>&1
}
timing() {  # tag
  if failed "${P}/$1.log"; then say "$1: FAILED log=${P}/$1.log"; return; fi
  say "$1: $("${MAIN_PY}" "${HERE}/data_tools.py" timing_log "${P}/$1.log" "${STEPS}" "${N_TRAIN}" 8 3)  $(peak "${P}/$1.log")"
}

# ---------------------------------------------------------------- 1. batch-1 vs batch-8 per-example loss
BATCH_OK=0
if [[ -n "${SKIP_LOSS_CHECK:-}" ]]; then
  say "loss check skipped: failed already (runs/qwen3_seq_speed/batch_check.json, bf16 noise floor)"
else
stage loss_check "per-example loss, batch 1 vs 8"
"${MAIN_PY}" scripts/evaluate/qwen3_omni_batch_check.py "${TRAIN}" 32 "${SEQ_DIR}/train.jsonl" 8 "${P}/loss_check.json" \
  > "${P}/loss_check.log" 2>&1
grep -E '^\[|^PASS|^batch-1|^  (utt|seq)' "${P}/loss_check.log" | tee -a "${SUMMARY}"
grep -q '^PASS=True' "${P}/loss_check.log" && BATCH_OK=1
fi

# ---------------------------------------------------------------- 2. memory, longest clips
"${MAIN_PY}" - "${TRAIN}" "${P}/longest8.jsonl" <<'EOF'
import json, sys
import soundfile as sf
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows.sort(key=lambda r: sf.info(r["audio_path"]).duration, reverse=True)
open(sys.argv[2], "w").write("".join(json.dumps(r) + "\n" for r in rows[:8]))
print(" ".join(f"{sf.info(r['audio_path']).duration:.1f}s" for r in rows[:8]))
EOF
A_ARGS=(--no-gradient-checkpointing)
B_ARGS=(--per-device-batch-size 8 --grad-accum 1 --no-gradient-checkpointing --group-by-length)
stage mem_a "batch 1 x accum 8, no checkpointing, 8 longest clips"
train mem_a "${P}/longest8.jsonl" "${A_ARGS[@]}"
if failed "${P}/mem_a.log"; then
  say "memory A (no checkpointing): FAILED $(peak "${P}/mem_a.log"); A keeps checkpointing"
  A_ARGS=()
else
  say "memory A (no checkpointing): fits $(peak "${P}/mem_a.log")"
fi

# ---------------------------------------------------------------- 3./4. speed
"${MAIN_PY}" scripts/data/sample_jsonl.py "${TRAIN}" $((STEPS * 8)) 7 "${P}/sample.jsonl" > /dev/null
stage time_a "batch 1 x accum 8 ${A_ARGS[*]}"
train time_a "${P}/sample.jsonl" "${A_ARGS[@]}"
timing time_a
say "A args: ${A_ARGS[*]:-(config defaults: gradient checkpointing on)}"

if [[ "${BATCH_OK}" == 1 ]]; then
  stage mem_b "batch 8, 8 longest clips"
  train mem_b "${P}/longest8.jsonl" "${B_ARGS[@]}"
  if failed "${P}/mem_b.log"; then
    say "memory B: FAILED $(peak "${P}/mem_b.log"); B keeps checkpointing"
    B_ARGS=(--per-device-batch-size 8 --grad-accum 1 --group-by-length)
  else
    say "memory B: fits $(peak "${P}/mem_b.log")"
  fi
  stage time_b "${B_ARGS[*]}"
  train time_b "${P}/sample.jsonl" "${B_ARGS[@]}"
  timing time_b
  say "B args: ${B_ARGS[*]}"
else
  say "batch 8 skipped: per-example loss check did not pass"
fi
stage done "summary: ${SUMMARY}"

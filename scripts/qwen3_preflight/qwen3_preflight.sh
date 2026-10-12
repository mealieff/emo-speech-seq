#!/usr/bin/env bash
# Qwen3-Omni preflight on the real data, one 80 GB GPU. Run after the reboot and the data copy:
#
#   bash /media/volume/AudioEmotionSequenceVol/qwen3_preflight/qwen3_preflight.sh
#
#   1. sha256 of the sequence JSONLs and every referenced wav -> data/, provenance.json
#   2. one LoRA step on the longest Train sequence (memory); if it fails, the <=180 s AF3 set
#   3. eval-only loss on the longest Dev sequence with that adapter (memory)
#   4. TIMING_STEPS optimizer steps (x8 accumulation) on a seed-7 Train sample -> s/step, hours
#   5. pretrained decoding of the 4 longest Test1 sequences in one batch of QWEN3_SEQ_BS (memory)
#
# Output: $EMOSEQ_RUNS/qwen3_preflight/ (summary.txt has the verdicts).
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1

P="${RUNS_DIR}/qwen3_preflight"
mkdir -p "${P}"
STATUS="${P}/status.log"
SUMMARY="${P}/summary.txt"
: > "${SUMMARY}"
TIMING_STEPS="${TIMING_STEPS:-50}"
GA=8
BS="${QWEN3_SEQ_BS:-4}"
tools() { "${MAIN_PY}" "${HERE}/data_tools.py" "$@"; }
run() { "${MAIN_PY}" "${HERE}/peak.py" "$@"; }
say() { echo "$*" | tee -a "${SUMMARY}"; }
peak() { grep -h '^\[peak\]' "$1" | tail -1; }
failed() { grep -Eq "out of memory|OutOfMemory|NVML_SUCCESS|CUDA error" "$1"; }

# ---------------------------------------------------------------- 0. inputs
for f in train dev test1; do
  [[ -s "${SEQ_DIR}/${f}.jsonl" ]] || { echo "missing ${SEQ_DIR}/${f}.jsonl" >&2; exit 1; }
done
JSONLS=("${SEQ_DIR}/train.jsonl" "${SEQ_DIR}/dev.jsonl" "${SEQ_DIR}/test1.jsonl")
for f in train_af3 dev_af3; do [[ -s "${SEQ_DIR}/${f}.jsonl" ]] && JSONLS+=("${SEQ_DIR}/${f}.jsonl"); done
"${MAIN_PY}" - "${JSONLS[@]}" <<'EOF' || exit 1
import json, sys
from pathlib import Path
miss = [r["audio_path"] for j in sys.argv[1:] for l in open(j) if l.strip()
        for r in [json.loads(l)] if not Path(r["audio_path"]).is_file()]
if miss:
    sys.exit(f"{len(miss)} audio files missing, e.g. {miss[:3]}")
EOF

# ---------------------------------------------------------------- 1. data hashes
stage hash "sha256 of ${#JSONLS[@]} JSONLs + audio"
KV=$(tools hash "${P}/data" "${JSONLS[@]}")
provenance "${P}/provenance.json" task=qwen3_omni_preflight seq_dir="${SEQ_DIR}" ${KV}
say "data hashes: ${P}/data/data_hashes.json (per-file: ${P}/data/audio_sha256.txt)"
say "  compare with the Qwen2-Audio / AF3 copy: python data_tools.py hash <dir> <same JSONLs> there, then diff"

# ---------------------------------------------------------------- 2. longest Train sequence
mem_train() {  # tag train_jsonl
  local tag=$1 longest
  longest=$(tools longest "$2" 1 "${P}/longest_${tag}.jsonl")
  stage mem_train "${tag}: ${longest}"
  run scripts/finetune/sequence_lora.py --model qwen3_omni --train-jsonl "${P}/longest_${tag}.jsonl" \
    --no-eval --epochs 1 --save-total-limit 0 --output-dir "${P}/mem_${tag}" > "${P}/mem_${tag}.log" 2>&1
  if failed "${P}/mem_${tag}.log" || ! grep -q "saved LoRA adapter" "${P}/mem_${tag}.log"; then
    say "train memory ${tag} (${longest}): FAILED  $(peak "${P}/mem_${tag}.log")  log=${P}/mem_${tag}.log"
    return 1
  fi
  say "train memory ${tag} (${longest}): fits  $(peak "${P}/mem_${tag}.log")"
}

SET=full TRAIN="${SEQ_DIR}/train.jsonl" DEV="${SEQ_DIR}/dev.jsonl"
if ! mem_train train "${TRAIN}"; then
  say "$(tools over 180 "${JSONLS[@]}")"
  [[ -s "${SEQ_DIR}/train_af3.jsonl" ]] || { say "no train_af3.jsonl to fall back to; stop"; exit 1; }
  mem_train train_af3 "${SEQ_DIR}/train_af3.jsonl" || { say "even the <=180 s set does not fit; stop"; exit 1; }
  SET=af3 TRAIN="${SEQ_DIR}/train_af3.jsonl" DEV="${SEQ_DIR}/dev_af3.jsonl"
fi
ADAPTER="${P}/mem_$( [[ ${SET} == af3 ]] && echo train_af3 || echo train )"

# ---------------------------------------------------------------- 3. longest Dev sequence (eval loss)
LONGEST_DEV=$(tools longest "${DEV}" 1 "${P}/longest_dev.jsonl")
stage mem_dev "${LONGEST_DEV}"
run scripts/finetune/sequence_lora.py --model qwen3_omni --train-jsonl "${P}/longest_dev.jsonl" \
  --eval-jsonl "${P}/longest_dev.jsonl" --eval-limit 1 --eval-only-checkpoint "${ADAPTER}" \
  --eval-output "${P}/mem_dev.json" > "${P}/mem_dev.log" 2>&1
if failed "${P}/mem_dev.log" || [[ ! -s "${P}/mem_dev.json" ]]; then
  say "dev eval memory (${LONGEST_DEV}): FAILED  $(peak "${P}/mem_dev.log")  log=${P}/mem_dev.log"
else
  say "dev eval memory (${LONGEST_DEV}): fits  $(peak "${P}/mem_dev.log")"
fi

# ---------------------------------------------------------------- 4. speed
N_SAMPLE=$((TIMING_STEPS * GA))
"${MAIN_PY}" scripts/data/sample_jsonl.py "${TRAIN}" "${N_SAMPLE}" 7 "${P}/timing_train.jsonl" > /dev/null
stage timing "${TIMING_STEPS} steps x ${GA} on ${TRAIN}"
run scripts/finetune/sequence_lora.py --model qwen3_omni --train-jsonl "${P}/timing_train.jsonl" \
  --length-bin-weight --no-eval --epochs 1 --save-total-limit 0 --output-dir "${P}/timing" > "${P}/timing.log" 2>&1
if [[ -s "${P}/timing/trainer_state.json" ]]; then
  say "speed: $(tools timing "${P}/timing/trainer_state.json" "$(wc -l < "${TRAIN}")" "${GA}" 3)  $(peak "${P}/timing.log")"
else
  say "speed: FAILED  log=${P}/timing.log"
fi

# ---------------------------------------------------------------- 5. Test1 decoding memory at batch BS
LONGEST_T1=$(tools longest "${SEQ_DIR}/test1.jsonl" "${BS}" "${P}/longest_test1.jsonl")
stage mem_infer "batch ${BS}: ${LONGEST_T1}"
run scripts/inference/sequence.py --model qwen3_omni --jsonl "${P}/longest_test1.jsonl" --batch-size "${BS}" \
  --max-new-tokens 800 --no-resume --out-dir "${P}/mem_infer" > "${P}/mem_infer.log" 2>&1
if failed "${P}/mem_infer.log" || [[ ! -s "${P}/mem_infer/metrics.json" ]]; then
  say "test1 decoding batch ${BS} (${LONGEST_T1}): FAILED  $(peak "${P}/mem_infer.log")  -> try QWEN3_SEQ_BS=2"
else
  say "test1 decoding batch ${BS} (${LONGEST_T1}): fits  $(peak "${P}/mem_infer.log")"
fi

say ""
say "training set: QWEN3_SET=${SET}"
say "next:"
say "  QWEN3_SET=${SET} bash scripts/finetune/run_sequence_sft.sh qwen3_omni"
say "  QWEN3_SEQ_BS=${BS} bash scripts/inference/run_test1.sh qwen3_omni_pt qwen3_omni_sft"
stage done "summary: ${SUMMARY}"
